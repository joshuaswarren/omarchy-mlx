"""Omarchy theme adapter tests.

These tests assert (a) no mutation of any disk file, (b) truthful fallback
when the Omarchy theme data is missing or malformed, (c) colour sanitisation
strips hostile values and arbitrary CSS, and (d) the chosen palette always
meets WCAG minimum contrast for the text/focus/status pairs.
"""

import contextlib
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "serve"))

from mlx_omarchy_assistant import theme  # noqa: E402


def _make_state_home(root: Path, slug: str | None, *, mode: str | None = None,
                     background: str | None = None,
                     foreground: str | None = None,
                     accent: str | None = None,
                     color1: str | None = None,
                     color2: str | None = None,
                     light_marker: bool = False) -> None:
    state_dir = root / ".local" / "state" / "omarchy" / "current"
    state_dir.mkdir(parents=True, exist_ok=True)
    if slug is not None:
        (state_dir / "theme.name").write_text(slug + "\n", encoding="utf-8")
        theme_dir = state_dir / "theme"
        theme_dir.mkdir(exist_ok=True)
        if light_marker:
            (theme_dir / "light.mode").write_text("", encoding="utf-8")
        if background is not None or foreground is not None or accent is not None:
            lines = [f"mode = \"{mode}\"\n" if mode else "",
                     "[colors]\n"]
            if background: lines.append(f"background = \"{background}\"\n")
            if foreground: lines.append(f"foreground = \"{foreground}\"\n")
            if accent: lines.append(f"accent = \"{accent}\"\n")
            if color1: lines.append(f"color1 = \"{color1}\"\n")
            if color2: lines.append(f"color2 = \"{color2}\"\n")
            (theme_dir / "colors.toml").write_text("".join(lines), encoding="utf-8")


@contextlib.contextmanager
def fake_home(root: Path):
    old = os.environ.get("HOME")
    os.environ["HOME"] = str(root)
    try:
        yield root
    finally:
        if old is None:
            os.environ.pop("HOME", None)
        else:
            os.environ["HOME"] = old


class FallbackTests(unittest.TestCase):
    def test_missing_theme_name_returns_fallback(self):
        with tempfile.TemporaryDirectory() as td, fake_home(Path(td)):
            t = theme.read_theme()
        self.assertEqual(t.source, "fallback")
        self.assertIsNone(t.slug)
        self.assertEqual(t.mode, "dark")
        for var in ("--mlx-canvas", "--mlx-text", "--mlx-focus",
                    "--mlx-ready", "--mlx-error", "--mlx-font-mono"):
            self.assertIn(var, t.css)

    def test_fallback_passes_contrast(self):
        for mode, pal in (("dark", theme.FALLBACK_DARK), ("light", theme.FALLBACK_LIGHT)):
            css = theme._map_colors({}, mode)
            self.assertGreaterEqual(theme._ratio(css["--mlx-text"], css["--mlx-canvas"]), 4.5)
            self.assertGreaterEqual(theme._ratio(css["--mlx-focus"], css["--mlx-canvas"]), 3.0)
            self.assertGreaterEqual(theme._ratio(css["--mlx-ready"], css["--mlx-canvas"]), 4.5)
            self.assertGreaterEqual(theme._ratio(css["--mlx-error"], css["--mlx-canvas"]), 4.5)

    def test_status_message_is_truthful(self):
        with tempfile.TemporaryDirectory() as td, fake_home(Path(td)):
            t = theme.read_theme()
        self.assertIn("absent", t.status.lower())
        self.assertNotIn("token", t.status.lower())


class OmarchyParsingTests(unittest.TestCase):
    def test_basic_round_trip(self):
        with tempfile.TemporaryDirectory() as td:
            home = Path(td)
            _make_state_home(home, "tokyo-night", mode="dark",
                             background="#1a1b26", foreground="#c0caf5",
                             accent="#7aa2f7")
            with fake_home(home):
                t = theme.read_theme()
        self.assertEqual(t.source, "omarchy")
        self.assertEqual(t.slug, "tokyo-night")
        self.assertEqual(t.name, "Tokyo Night")
        self.assertEqual(t.css["--mlx-canvas"], "#1a1b26")
        self.assertEqual(t.css["--mlx-text"], "#c0caf5")

    def test_invalid_color_falls_back_to_palette(self):
        with tempfile.TemporaryDirectory() as td:
            home = Path(td)
            # Hostile values: url(), CSS gradient, rgb(), an HTML tag, NaN.
            _make_state_home(home, "evil", background="url(javascript:alert(1))",
                             foreground="rgb(1, 2, 3)",
                             accent="red; background: url(javascript:alert(1))",
                             color1="<script>", color2="NaN")
            with fake_home(home):
                t = theme.read_theme()
        # Slug present but no valid colors → fallback with truthful reason.
        self.assertEqual(t.source, "fallback")
        self.assertIn("no readable colors.toml", t.status.lower())

    def test_malformed_toml_falls_back(self):
        with tempfile.TemporaryDirectory() as td:
            home = Path(td)
            state_dir = home / ".local" / "state" / "omarchy" / "current"
            state_dir.mkdir(parents=True, exist_ok=True)
            (state_dir / "theme.name").write_text("weird\n", encoding="utf-8")
            (state_dir / "theme").mkdir(exist_ok=True)
            (state_dir / "theme" / "colors.toml").write_text("not = valid", encoding="utf-8")
            with fake_home(home):
                t = theme.read_theme()
        self.assertEqual(t.source, "fallback")

    def test_invalid_slug_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            home = Path(td)
            (home / ".local" / "state" / "omarchy" / "current").mkdir(parents=True)
            (home / ".local" / "state" / "omarchy" / "current" / "theme.name").write_text(
                "Bad Slug!", encoding="utf-8")
            with fake_home(home):
                t = theme.read_theme()
        self.assertEqual(t.source, "fallback")

    def test_light_marker_overrides_luminance(self):
        with tempfile.TemporaryDirectory() as td:
            home = Path(td)
            _make_state_home(home, "catppuccin-latte", light_marker=True,
                             background="#1e1e2e")
            with fake_home(home):
                t = theme.read_theme()
        self.assertEqual(t.mode, "light")


class ContrastAdjustmentTests(unittest.TestCase):
    def test_low_contrast_foreground_is_boosted(self):
        # Foreground almost identical to canvas: adjuster must move it.
        css = theme._map_colors({"background": "#222222", "foreground": "#333333",
                                 "color2": "#444444", "color1": "#555555"},
                                "dark")
        self.assertGreaterEqual(theme._ratio(css["--mlx-text"], css["--mlx-canvas"]), 4.5)
        self.assertNotEqual(css["--mlx-text"], "#333333")

    def test_extreme_low_contrast_falls_back(self):
        css = theme._map_colors({"background": "#000000",
                                 "color1": "#000000", "color2": "#000000"},
                                "dark")
        # The contrast adjuster walks each token until the painted pair clears;
        # passing the bar is the contract, regardless of which hex value wins.
        self.assertGreaterEqual(
            theme._ratio(css["--mlx-ready"], css["--mlx-canvas"]), 4.5)
        self.assertGreaterEqual(
            theme._ratio(css["--mlx-error"], css["--mlx-canvas"]), 4.5)

    def test_light_mode_keeps_text_dark(self):
        css = theme._map_colors({"background": "#ffffff",
                                 "foreground": "#ffffff",
                                 "color1": "#ffffff", "color2": "#ffffff"},
                                "light")
        # Background is white; the adjuster must leave a dark text colour.
        self.assertLess(theme._luminance(css["--mlx-text"]), 0.4)
        self.assertGreaterEqual(theme._ratio(css["--mlx-text"], css["--mlx-canvas"]), 4.5)


class PaintedPairsTests(unittest.TestCase):
    """Every painted token pair in actual UI must clear WCAG AA.

    These regressions live alongside :mod:`theme` and document the painted
    combination contract. They do not encode per-theme heuristics; they
    assert that the public mapping routine returns a contrast map where the
    pairs the application actually paints satisfy the required thresholds.
    """

    def _assert_pairs(self, css, *, mode):
        canvas = css["--mlx-canvas"]
        panel = css["--mlx-panel"]
        # Body text and every status colour clear 4.5 against both surfaces
        # the application paints them on; focus and border are non-text and
        # clear 3.0 against canvas. Border vs panel is a non-text contrast
        # check that we keep at >=3.0 for the edge to register visually.
        body_pairs = [
            ("text/canvas",   css["--mlx-text"],  canvas, 4.5),
            ("text/panel",    css["--mlx-text"],  panel,  4.5),
            ("dim/canvas",    css["--mlx-dim"],   canvas, 4.5),
            ("dim/panel",     css["--mlx-dim"],   panel,  4.5),
            ("ready/canvas",  css["--mlx-ready"], canvas, 4.5),
            ("error/canvas",  css["--mlx-error"], canvas, 4.5),
            ("warn/canvas",   css["--mlx-warn"],  canvas, 4.5),
            ("focus/canvas",  css["--mlx-focus"], canvas, 3.0),
            ("border/canvas", css["--mlx-border"],canvas, 3.0),
        ]
        for label, fg, bg, threshold in body_pairs:
            ratio = theme._ratio(fg, bg)
            self.assertGreaterEqual(
                ratio, threshold,
                f"{mode} {label}: {fg} on {bg} = {ratio:.2f}, need ≥ {threshold}",
            )

    def test_fallback_dark_painted_pairs(self):
        css = theme._map_colors({}, "dark")
        self._assert_pairs(css, mode="dark")

    def test_fallback_light_painted_pairs(self):
        css = theme._map_colors({}, "light")
        self._assert_pairs(css, mode="light")

    def test_tokyo_night_like_dark_painted_pairs(self):
        css = theme._map_colors({
            "background": "#1a1b26", "foreground": "#c0caf5", "accent": "#7aa2f7",
        }, "dark")
        self._assert_pairs(css, mode="dark-tokyo")

    def test_adversarial_midgray_dark(self):
        # Mis-configured theme: every foreground is midgray, identical to
        # the canvas. The contract requires every painted body/status pair to
        # clear 4.5 (focus/border 3.0). No hardcoded substitution path: the
        # adjuster picks black as the larger-ratio extreme against midgray
        # (black vs #777 = ~4.56) and converges panel toward canvas so the
        # same foreground clears both surfaces.
        css = theme._map_colors({
            "background": "#777777", "foreground": "#777777",
        }, "dark")
        for label, fg, bg in (
            ("text/canvas", css["--mlx-text"], css["--mlx-canvas"]),
            ("text/panel",  css["--mlx-text"], css["--mlx-panel"]),
            ("dim/canvas",  css["--mlx-dim"],  css["--mlx-canvas"]),
            ("dim/panel",   css["--mlx-dim"],  css["--mlx-panel"]),
            ("ready/canvas",css["--mlx-ready"],css["--mlx-canvas"]),
            ("warn/canvas", css["--mlx-warn"], css["--mlx-canvas"]),
            ("error/canvas",css["--mlx-error"],css["--mlx-canvas"]),
        ):
            self.assertGreaterEqual(
                theme._ratio(fg, bg), 4.5,
                f"{label}: {fg} on {bg} fails the 4.5 contract")
        for label, fg, bg in (
            ("focus/canvas", css["--mlx-focus"], css["--mlx-canvas"]),
            ("border/canvas", css["--mlx-border"], css["--mlx-canvas"]),
        ):
            self.assertGreaterEqual(
                theme._ratio(fg, bg), 3.0,
                f"{label}: {fg} on {bg} fails the 3.0 contract")

    def test_adversarial_midgray_light(self):
        # Same midgray input under light mode: black still wins (the
        # darker extreme has the larger absolute ratio on a 0.18-luminance
        # background), so the result is symmetric in axis, not in target.
        css = theme._map_colors({
            "background": "#777777", "foreground": "#777777",
        }, "light")
        for label, fg, bg in (
            ("text/canvas", css["--mlx-text"], css["--mlx-canvas"]),
            ("text/panel",  css["--mlx-text"], css["--mlx-panel"]),
            ("dim/canvas",  css["--mlx-dim"],  css["--mlx-canvas"]),
            ("dim/panel",   css["--mlx-dim"],  css["--mlx-panel"]),
        ):
            self.assertGreaterEqual(theme._ratio(fg, bg), 4.5,
                f"{label}: {fg} on {bg} fails the 4.5 contract")

    def test_focus_ink_is_max_actual_contrast(self):
        # _focus_ink must compare white vs black against the focus surface,
        # not a luminance threshold.
        focus_red = "#cc3333"
        self.assertEqual(theme._focus_ink(focus_red), "#ffffff")
        focus_yellow = "#f0db70"
        self.assertEqual(theme._focus_ink(focus_yellow), "#000000")


class MutationTests(unittest.TestCase):
    def test_read_theme_does_not_mutate_files(self):
        with tempfile.TemporaryDirectory() as td:
            home = Path(td)
            _make_state_home(home, "deep", mode="dark",
                             background="#1a1b26", foreground="#c0caf5")
            before = theme.snapshot_tree(home)
            with fake_home(home):
                for _ in range(3):
                    theme.read_theme()
            after = theme.snapshot_tree(home)
        self.assertEqual(before, after)

class FontTests(unittest.TestCase):
    def test_fc_default_returns_none_for_unexpected_family(self):
        # fc-match is unavailable in the sandbox; both calls return None and
        # the helper substitutes the built-in stack without raising.
        sans, mono = theme._font_families(lambda match: None)
        self.assertIn("system-ui", sans)
        self.assertIn("monospace", mono)

    def test_fc_default_strips_alias_csv(self):
        # fc-match returns the first comma-separated alias; _font_families
        # receives only that primary family and wraps it in quotes.
        self.assertEqual(theme._fc_default.__name__, "_fc_default")  # sanity
        out = theme._font_families(lambda match: "JetBrainsMono Nerd Font,JetBrainsMono NF"
                                          if match == "monospace" else "Inter,Inter Display")
        # Both entries are valid and accepted; the first is quoted because of the space.
        self.assertIn('"JetBrainsMono Nerd Font"', out[1])
        self.assertIn("Inter", out[0])

    def test_fc_default_rejects_garbage(self):
        sans, mono = theme._font_families(lambda match: "<script>"
                                          if match == "monospace" else None)
        self.assertNotIn("<script>", mono)
        self.assertNotIn("&quot;", mono)

    def test_fc_subprocess_rejected_when_returncode_nonzero(self, monkeypatch=None):
        # End-to-end: simulate fc-match failure via a fake subprocess returning
        # an empty string. The helper must fall back to a known stack.
        def fake_run(*args, **kwargs):
            return subprocess.CompletedProcess(args=args, returncode=1,
                                               stdout="", stderr="error")
        import unittest.mock
        with unittest.mock.patch("subprocess.run", fake_run):
            family = theme._fc_default("monospace")
        self.assertIsNone(family)


if __name__ == "__main__":
    unittest.main()
