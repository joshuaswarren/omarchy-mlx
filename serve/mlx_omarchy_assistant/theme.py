"""Omarchy theme adapter: read-only, allowlist-only, accessible.

Resolves the active Omarchy theme through the same paths the ``omarchy`` CLI
uses, parses ``colors.toml`` as data, and maps the named colors into the
application's own CSS variables. Never imports shell code, never executes
anything from a theme directory, never touches the Omarchy package or the
user's selected theme. When the Omarchy data is missing or malformed it
returns a built-in fallback palette with an honest ``status`` string the
status endpoint surfaces back to the user.

Colors are picked from an allowlist (``#rgb`` / ``#rrggb``). The text-on-canvas
and focus-on-canvas pairs go through a deterministic contrast adjuster that
guarantees WCAG minimums by blending toward white or black, so a hostile or
unbalanced theme can never reduce the assistant to unreadable. Estimates and
status colors derive from ANSI slots 1 (red) and 2 (green) when present, with
the same adjustment, and fall back to the built-in tokens when missing.

No mutation: a single ``read_theme(home, fc)`` call performs N filesystem
reads + at most one ``fc-match`` subprocess per invocation. The caller can
inject a fake ``home`` root and an ``fc`` callable for tests.
"""

from __future__ import annotations

import contextlib
import dataclasses
import re
import subprocess
import tomllib
from pathlib import Path
from typing import Callable
FALLBACK_DARK = {
    "canvas": "#1a1b26",
    "panel": "#24283b",
    "text": "#c0caf5",
    "focus": "#7aa2f7",
    "ready": "#9ece6a",
    "error": "#f7768e",
    "warn": "#e0af68",
}
FALLBACK_LIGHT = {
    "canvas": "#eceef2",
    "panel": "#dee1e7",
    "text": "#2f3340",
    "focus": "#2456c4",
    "ready": "#2e6b2e",
    "error": "#a63a50",
    "warn": "#8a5a00",
}

_COLOR_RE = re.compile(r"^#[0-9a-fA-F]{6}$|^#[0-9a-fA-F]{3}$")
_ASCII_FONT_RE = re.compile(r"^[\w -]{1,64}$")
_FONT_TIMEOUT = 2.0

# Application-owned CSS variable map (kept in sync with css/app.css).
CSS_VARS = (
    "--mlx-canvas", "--mlx-panel", "--mlx-border",
    "--mlx-text", "--mlx-dim", "--mlx-focus", "--mlx-focus-ink",
    "--mlx-ready", "--mlx-error", "--mlx-warn",
    "--mlx-font-sans", "--mlx-font-mono", "--mlx-mode",
)


@dataclasses.dataclass(frozen=True)
class Theme:
    name: str              # display name ("Tokyo Night" or "Built-in dark")
    slug: str | None       # omarchy slug or None when on fallback
    source: str            # "omarchy" | "fallback"
    status: str            # human note the status endpoint surfaces
    mode: str              # "dark" | "light"
    css: dict[str, str]    # {"--mlx-canvas": "#1a1b26", ...}


def _luminance(hex_color: str) -> float:
    h = hex_color.lstrip("#")
    if len(h) == 3:
        h = "".join(ch * 2 for ch in h)
    r, g, b = (int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
    def to_linear(c: float) -> float:
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
    return 0.2126 * to_linear(r) + 0.7152 * to_linear(g) + 0.0722 * to_linear(b)


def _ratio(fg: str, bg: str) -> float:
    a, b = sorted((_luminance(fg), _luminance(bg)), reverse=True)
    return (a + 0.05) / (b + 0.05)


def _blend(fg: str, bg: str, t: float) -> str:
    h1, h2 = fg.lstrip("#"), bg.lstrip("#")
    if len(h1) == 3:
        h1 = "".join(ch * 2 for ch in h1)
    if len(h2) == 3:
        h2 = "".join(ch * 2 for ch in h2)
    out = []
    for i in (0, 2, 4):
        a, b = int(h1[i:i + 2], 16), int(h2[i:i + 2], 16)
        v = round(a * (1 - t) + b * t)
        out.append(f"{v:02x}")
    return "#" + "".join(out)


def _best_target(bg: str) -> str:
    """Return ``#000000`` or ``#ffffff``, whichever gives the larger absolute
    contrast against ``bg``. The luminance threshold the previous version used
    was the wrong axis: a mid-gray background makes "dark" win and "light"
    lose, so the only direction that *increases* the painted ratio is toward
    darker, not toward lighter. Numeric comparison of the two ratios does
    not need a threshold and is correct on every input.
    """
    black = "#000000"
    white = "#ffffff"
    return black if _ratio(black, bg) >= _ratio(white, bg) else white


def _adjust_pair(fg: str, bg: str, min_ratio: float,
                steps: tuple[float, ...] = (0.20, 0.20, 0.20, 0.20, 0.20)) -> str:
    """Push ``fg`` toward the contrast-increasing extreme until it reaches
    ``min_ratio`` against ``bg``. Returns the iteration-budget best foreground;
    if ``fg == bg`` the function replaces ``fg`` outright with the extreme —
    the painted contract requires a distinct foreground, so the caller is
    not entitled to keep ``fg`` and pretend.
    """
    if fg == bg:
        return _best_target(bg)
    if _ratio(fg, bg) >= min_ratio:
        return fg
    target = _best_target(bg)
    current = fg
    for step in steps:
        current = _blend(current, target, step)
        if _ratio(current, bg) >= min_ratio:
            return current
    return current


def _converge_panel_to_canvas(canvas: str, panel: str,
                                target: str, max_steps: int = 6,
                                step: float = 0.25) -> tuple[str, str]:
    """Pull ``panel`` toward ``canvas`` so a single foreground can satisfy
    both ``(fg, canvas)`` and ``(fg, panel)`` at the same target ratio.

    Used when even the best-contrast extreme cannot hit the foreground's
    minimum ratio against both surfaces simultaneously (e.g. midgray canvas +
    midgray foreground). Returns ``(canvas, adjusted_panel)``.
    """
    if _ratio(target, canvas) >= _ratio(target, panel):
        return canvas, panel
    current = panel
    for _ in range(max_steps):
        candidate = _blend(current, canvas, step)
        if _ratio(target, candidate) >= _ratio(target, canvas):
            return canvas, candidate
        current = candidate
    return canvas, current


def _panel_and_border(canvas: str, text: str, mode: str) -> tuple[str, str, str]:
    """Initial hues for panel/border/dim.

    Blend FROM canvas toward text so panel sits between canvas and text (so
    dim/panel is not symmetric with text/panel), and pull border toward the
    dark/light extreme so it sinks against canvas.
    """
    prefer_dark = mode == "dark"
    panel = _blend(canvas, text, 0.18)
    border = _blend(canvas, "#000000" if prefer_dark else "#ffffff", 0.12)
    dim = _blend(canvas, text, 0.45)
    return panel, border, dim


def _focus_ink(focus: str) -> str:
    # Whichever ink has the higher ratio against the focus surface wins;
    # this is independent of bg luminance and correct for blue, green, and
    # yellow accents alike.
    return _best_target(focus)


def _sanitize(value) -> str | None:
    if not isinstance(value, str) or not _COLOR_RE.match(value):
        return None
    # Normalize 3-digit hex to 6-digit for a single output shape.
    h = value.lstrip("#")
    if len(h) == 3:
        h = "".join(ch * 2 for ch in h)
    return "#" + h.lower()


def _read_slug(home: Path) -> str | None:
    name_file = home / ".local" / "state" / "omarchy" / "current" / "theme.name"
    try:
        text = name_file.read_text(encoding="utf-8").strip()
    except (FileNotFoundError, PermissionError, IsADirectoryError):
        return None
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", text):
        return None
    return text


def _theme_dir(home: Path, slug: str) -> tuple[Path, bool]:
    """Locate the theme directory, preferring the staged location."""
    candidates = [
        home / ".local" / "state" / "omarchy" / "current" / "theme",
        home / ".config" / "omarchy" / "themes" / slug,
    ]
    import os
    omarchy = os.environ.get("OMARCHY_PATH")
    if omarchy:
        candidates.append(Path(omarchy) / "themes" / slug)
    for candidate in candidates:
        colors = candidate / "colors.toml"
        if candidate.is_dir() and colors.is_file():
            return candidate, True
    # Fall back to just the first candidate existing as a directory so the
    # mode detection can still read its light.mode marker.
    for candidate in candidates:
        if candidate.is_dir():
            return candidate, False
    return candidates[0], False


def _parse_colors_toml(path: Path) -> tuple[dict[str, str], str | None]:
    try:
        with path.open("rb") as fh:
            data = tomllib.load(fh)
    except (OSError, tomllib.TOMLDecodeError):
        return {}, None
    if not isinstance(data, dict):
        return {}, None
    raw = data.get("colors", {})
    if not isinstance(raw, dict):
        return {}, None
    cleaned: dict[str, str] = {}
    for key in ("background", "foreground", "accent"):
        if key in raw:
            color = _sanitize(raw[key])
            if color is not None:
                cleaned[key] = color
    for slot in (f"color{i}" for i in range(16)):
        if slot in raw:
            color = _sanitize(raw[slot])
            if color is not None:
                cleaned[slot] = color
    mode = data.get("mode") if isinstance(data.get("mode"), str) else None
    if mode not in ("dark", "light"):
        mode = None
    return cleaned, mode


def _detect_mode(theme_dir: Path, parsed_mode: str | None,
                 canvas_hex: str | None) -> str:
    if parsed_mode in ("dark", "light"):
        return parsed_mode
    if (theme_dir / "light.mode").exists():
        return "light"
    if canvas_hex is not None:
        return "light" if _luminance(canvas_hex) > 0.55 else "dark"
    return "dark"


def _read_omarchy(home: Path) -> dict | None:
    slug = _read_slug(home)
    if slug is None:
        return None
    theme_dir, has_colors = _theme_dir(home, slug)
    colors = {}
    parsed_mode = None
    if has_colors:
        colors, parsed_mode = _parse_colors_toml(theme_dir / "colors.toml")
    if not colors:
        # Slug file exists but no usable colors.toml: report what we have and
        # let the caller fall back, naming the cause honestly.
        return {
            "slug": slug,
            "display": slug.replace("-", " ").title(),
            "mode": "dark",
            "theme_dir": str(theme_dir),
            "colors": {},
            "partial_reason": f"no readable colors.toml under {theme_dir}",
        }
    return {
        "slug": slug,
        "display": slug.replace("-", " ").title(),
        "mode": _detect_mode(theme_dir, parsed_mode, colors.get("background")),
        "theme_dir": str(theme_dir),
        "colors": colors,
    }


def _fc_default(match: str) -> str | None:
    try:
        result = subprocess.run(
            ["fc-match", match, "-f", "%{family}"],
            check=False,
            capture_output=True,
            text=True,
            timeout=_FONT_TIMEOUT,
        )
    except (subprocess.SubprocessError, OSError):
        return None
    if result.returncode != 0:
        return None
    family = result.stdout.split(",", 1)[0].strip()
    if not family or not _ASCII_FONT_RE.match(family):
        return None
    return family


def _map_colors(colors: dict[str, str], mode: str) -> dict[str, str]:
    fallback = FALLBACK_DARK if mode == "dark" else FALLBACK_LIGHT
    canvas = colors.get("background") or fallback["canvas"]
    text = colors.get("foreground") or fallback["text"]
    focus = colors.get("accent") or fallback["focus"]
    # Status colors prefer the green/red ANSI slots when present, because they
    # are designed to read well on the canvas of most Omarchy themes.
    ready = colors.get("color2") or fallback["ready"]
    error = colors.get("color1") or fallback["error"]
    warn = colors.get("color3") or fallback["warn"]
    # Initial surface hues: panel sits between canvas and text, border sinks
    # toward the dark/light extreme (so cards have an edge), dim lands halfway.
    panel, border, dim = _panel_and_border(canvas, text, mode)
    # Painted-pair coverage. Every pair involving body or status text targets
    # WCAG AA 4.5; focus and border target 3.0 (per design contract). The
    # contrast direction follows the actual larger ratio against the bg —
    # not a luminance threshold — so midgray backgrounds pick black, dark
    # backgrounds pick white, light backgrounds pick black, uniformly.
    # If both (fg, canvas) and (fg, panel) cannot reach the target with any
    # single fg value, we converge panel toward canvas so a single fg clears
    # both surfaces.
    targets = [
        ("text", "canvas", 4.5),
        ("text", "panel", 4.5),
        ("dim", "canvas", 4.5),
        ("dim", "panel", 4.5),
        ("focus", "canvas", 3.0),
        ("ready", "canvas", 4.5),
        ("error", "canvas", 4.5),
        ("warn", "canvas", 4.5),
    ]
    fg_tokens = {"text", "dim", "focus", "ready", "error", "warn"}
    pairs = {
        "text": text, "dim": dim, "focus": focus,
        "ready": ready, "error": error, "warn": warn,
    }
    backgrounds = {"canvas": canvas, "panel": panel}
    for _ in range(6):
        moved = False
        for fg_name, bg_name, min_ratio in targets:
            fg_color = pairs[fg_name]
            bg_color = backgrounds[bg_name]
            if _ratio(fg_color, bg_color) >= min_ratio:
                continue
            adjusted = _adjust_pair(fg_color, bg_color, min_ratio)
            if adjusted != fg_color:
                pairs[fg_name] = adjusted
                moved = True
        if not moved:
            break
    # If the text or dim foreground still fails because the painted sweep is
    # stuck at the input hex (equal to canvas), force the foreground onto the
    # extreme that increases the contrast against both surfaces, then converge
    # panel toward canvas if necessary to clear both painted pairs.
    for fg_name in ("text", "dim"):
        if _ratio(pairs[fg_name], canvas) >= 4.5:
            continue
        if _ratio(pairs[fg_name], panel) >= 4.5:
            continue
        # Push the foreground to the better extreme, then converge panel.
        for extreme in ("#000000", "#ffffff"):
            new_canvas, new_panel = _converge_panel_to_canvas(canvas, panel, extreme)
            if (_ratio(extreme, new_canvas) >= 4.5
                    and _ratio(extreme, new_panel) >= 4.5):
                pairs[fg_name] = extreme
                canvas = new_canvas
                panel = new_panel
                backgrounds["canvas"] = canvas
                backgrounds["panel"] = panel
                break
    # Re-run the painted sweep one more time so any change to canvas/panel
    # lands the fg back inside the contract.
    for _ in range(6):
        moved = False
        for fg_name, bg_name, min_ratio in targets:
            fg_color = pairs[fg_name]
            bg_color = backgrounds[bg_name]
            if _ratio(fg_color, bg_color) >= min_ratio:
                continue
            adjusted = _adjust_pair(fg_color, bg_color, min_ratio)
            if adjusted != fg_color:
                pairs[fg_name] = adjusted
                moved = True
        if not moved:
            break
    text = pairs["text"]
    dim = pairs["dim"]
    focus = pairs["focus"]
    ready = pairs["ready"]
    error = pairs["error"]
    warn = pairs["warn"]
    focus_ink = _focus_ink(focus)
    border_against_canvas = _ratio(border, canvas)
    if border_against_canvas < 3.0:
        border = _adjust_pair(border, canvas, 3.0)
    return {
        "--mlx-canvas": canvas,
        "--mlx-panel": panel,
        "--mlx-border": border,
        "--mlx-text": text,
        "--mlx-dim": dim,
        "--mlx-focus": focus,
        "--mlx-focus-ink": focus_ink,
        "--mlx-ready": ready,
        "--mlx-error": error,
        "--mlx-warn": warn,
        "--mlx-mode": mode,
    }


def _font_families(fc: Callable[[str], str | None]) -> tuple[str, str]:
    sans_raw = fc("sans-serif")
    mono_raw = fc("monospace")
    sans = _safe_family(sans_raw, role="sans")
    mono = _safe_family(mono_raw, role="mono")
    return sans, mono


def _safe_family(family: str | None, role: str) -> str:
    fallback = ("ui-monospace, 'Cascadia Mono', Menlo, monospace"
                if role == "mono" else "system-ui, sans-serif")
    if not family:
        return fallback
    parts = [p.strip() for p in family.split(",") if p.strip()]
    if not parts or not all(_ASCII_FONT_RE.match(p) for p in parts):
        return fallback
    # Wrap whitespace-bearing single-family values in quotes for CSS safety.
    def quote(part: str) -> str:
        if " " in part and not (part.startswith('"') or part.startswith("'")):
            return f'"{part}"'
        return part
    return ", ".join(quote(p) for p in parts)


def read_theme(home: Path | None = None,
               fc: Callable[[str], str | None] | None = None) -> Theme:
    """Resolve the active theme; return the application Theme dataclass.

    Never writes to disk. Reads at most: ``theme.name``, ``colors.toml``,
    ``light.mode``, and one ``fc-match`` call per requested family.
    """
    root = (home or Path.home()).expanduser()
    fc = fc or _fc_default
    result = _read_omarchy(root)
    sans, mono = _font_families(fc)
    if result is None:
        return Theme(
            name="Built-in dark",
            slug=None,
            source="fallback",
            status=("Omarchy theme data absent; "
                    f"using built-in fallback. Searched {root / '.local/state/omarchy/current'}"),
            mode="dark",
            css={**_map_colors({}, "dark"),
                 "--mlx-font-sans": sans, "--mlx-font-mono": mono},
        )
    if not result["colors"]:
        reason = result["partial_reason"]
        return Theme(
            name=result["display"],
            slug=result["slug"],
            source="fallback",
            status=f"{result['display']}: {reason}; using built-in fallback",
            mode=result["mode"],
            css={**_map_colors({}, result["mode"]),
                 "--mlx-font-sans": sans, "--mlx-font-mono": mono},
        )
    return Theme(
        name=result["display"],
        slug=result["slug"],
        source="omarchy",
        status=f"Omarchy theme {result['display']} ({result['mode']})",
        mode=result["mode"],
        css={**_map_colors(result["colors"], result["mode"]),
             "--mlx-font-sans": sans, "--mlx-font-mono": mono},
    )


def snapshot_tree(path: Path) -> dict[str, str]:
    """Read-only directory digest used by tests to assert no mutation."""
    digest: dict[str, str] = {}
    for entry in sorted(path.glob("**/*")) if path.exists() else []:
        if entry.is_file():
            digest[str(entry.relative_to(path))] = f"{entry.stat().st_size}"
    return digest


@contextlib.contextmanager
def temporary_home(root: Path):
    """Context manager that swaps $HOME for the duration; restores after."""
    import os
    saved = os.environ.get("HOME")
    os.environ["HOME"] = str(root)
    try:
        yield root
    finally:
        if saved is None:
            os.environ.pop("HOME", None)
        else:
            os.environ["HOME"] = saved
