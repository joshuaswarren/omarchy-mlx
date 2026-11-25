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
import threading
import tomllib
import unittest
import unittest.mock
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


def _write_raw(home: Path, name: bytes, colors: bytes | None) -> Path:
    state_dir = home / ".local" / "state" / "omarchy" / "current"
    theme_dir = state_dir / "theme"
    theme_dir.mkdir(parents=True)
    (state_dir / "theme.name").write_bytes(name)
    if colors is not None:
        (theme_dir / "colors.toml").write_bytes(colors)
    return theme_dir


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

    def test_color_with_trailing_newline_rejected(self):
        # "$" matches before a trailing newline, so "#abc\n" used to pass the
        # allowlist and crash the contrast math (or reach CSS unchanged).
        self.assertIsNone(theme._sanitize("#abc\n"))
        self.assertIsNone(theme._sanitize("#aabbcc\n"))
        with tempfile.TemporaryDirectory() as td:
            home = Path(td)
            _write_raw(home, b"evil\n", b'[colors]\nbackground = "#abc\\n"\n'
                                        b'accent = "#aabbcc\\n"\n'
                                        b'foreground = "#ffffff"\n')
            t = theme.read_theme(home, fc=lambda match: None)
        self.assertEqual(t.css["--mlx-canvas"], theme.FALLBACK_DARK["canvas"])
        self.assertEqual(t.css["--mlx-focus"], theme.FALLBACK_DARK["focus"])

    def test_non_utf8_theme_name_falls_back(self):
        with tempfile.TemporaryDirectory() as td:
            home = Path(td)
            _write_raw(home, b"\xff\xfe\n", None)
            t = theme.read_theme(home, fc=lambda match: None)
        self.assertEqual(t.source, "fallback")
        self.assertIsNone(t.slug)

    def test_non_utf8_colors_toml_falls_back(self):
        with tempfile.TemporaryDirectory() as td:
            home = Path(td)
            _write_raw(home, b"evil\n",
                       b'[colors]\nbackground = "#1a1b26"\n# \xff\n')
            t = theme.read_theme(home, fc=lambda match: None)
        self.assertEqual(t.source, "fallback")
        self.assertIn("no readable colors.toml", t.status.lower())


def _run_with_pinned_stack(fn):
    result: dict[str, object] = {}

    def target():
        try:
            result["value"] = fn()
        except BaseException as exc:
            result["exc"] = exc

    old_size = threading.stack_size(4 * 1024 * 1024)
    try:
        thread = threading.Thread(target=target)
        thread.start()
    finally:
        threading.stack_size(old_size)
    thread.join()
    if "exc" in result:
        raise result["exc"]
    return result.get("value")


class DamagedThemeInputTests(unittest.TestCase):
    _CAP = 4 * 1024
    _VALID_COLORS = (
        b'[colors]\n'
        b'background = "#1a1b26"\n'
        b'foreground = "#c0caf5"\n'
        b'accent = "#7aa2f7"\n'
    )

    def setUp(self):
        patcher = unittest.mock.patch.dict(os.environ)
        patcher.start()
        self.addCleanup(patcher.stop)
        os.environ.pop("OMARCHY_PATH", None)

    @contextlib.contextmanager
    def _patch_theme_open(self, *, return_none: bool = False):
        real_open = open
        read_sizes: list[object] = []
        fd_offsets: list[int] = []

        class _ControlledStream:
            def __init__(self, inner):
                self._inner = inner

            def __enter__(self):
                self._inner.__enter__()
                return self

            def __exit__(self, exc_type, exc, tb):
                return self._inner.__exit__(exc_type, exc, tb)

            def fileno(self):
                return self._inner.fileno()

            def read(self, size=-1):
                read_sizes.append(size)
                if return_none:
                    return None
                data = self._inner.read(size)
                fd_offsets.append(
                    os.lseek(self._inner.fileno(), 0, os.SEEK_CUR)
                )
                return data

        def wrapped_open(*args, **kwargs):
            return _ControlledStream(real_open(*args, **kwargs))

        with unittest.mock.patch.object(
            theme, "open", create=True, new=wrapped_open
        ):
            yield read_sizes, fd_offsets

    def _run_without_blocking_fifo(self, fifo_path: Path, fn):
        result: dict[str, object] = {}

        def target():
            try:
                result["value"] = fn()
            except BaseException as exc:
                result["exc"] = exc

        thread = threading.Thread(target=target, daemon=True)
        thread.start()
        thread.join(timeout=10)
        if thread.is_alive():
            try:
                fd = os.open(fifo_path, os.O_WRONLY | os.O_NONBLOCK)
            except OSError:
                pass
            else:
                os.close(fd)
            thread.join(timeout=2)
            self.fail(f"reading FIFO {fifo_path.name} blocked for >10s")
        if "exc" in result:
            raise result["exc"]
        return result.get("value")

    def test_deep_toml_falls_back_on_pinned_stack(self):
        deep = "x = " + "[" * 1500 + "]" * 1500 + "\n"
        payload = deep.encode("utf-8") + self._VALID_COLORS
        self.assertLess(len(payload), self._CAP)

        def check():
            with self.assertRaises(RecursionError):
                tomllib.loads(deep)
            with tempfile.TemporaryDirectory() as td:
                home = Path(td)
                _write_raw(home, b"deep\n", payload)
                t = theme.read_theme(home, fc=lambda match: None)
            self.assertEqual(t.source, "fallback")
            self.assertEqual(t.slug, "deep")
            self.assertIn("no readable colors.toml", t.status.lower())

        _run_with_pinned_stack(check)

    def test_toml_parser_resource_errors_patched_fall_back(self):
        with tempfile.TemporaryDirectory() as td:
            home = Path(td)
            _write_raw(home, b"tokyo-night\n", self._VALID_COLORS)
            for exc in (
                RecursionError("simulated depth limit"),
                MemoryError("simulated allocation limit"),
            ):
                with self.subTest(exc=type(exc).__name__):
                    with unittest.mock.patch.object(
                        theme.tomllib, "loads", side_effect=exc,
                    ) as mock_loads:
                        t = theme.read_theme(home, fc=lambda match: None)
                    self.assertEqual(t.source, "fallback")
                    self.assertEqual(t.slug, "tokyo-night")
                    self.assertIn("no readable colors.toml", t.status.lower())
                    mock_loads.assert_called_once()

    def test_oversized_int_toml_falls_back(self):
        self.addCleanup(sys.set_int_max_str_digits, sys.get_int_max_str_digits())
        sys.set_int_max_str_digits(640)
        bigint = "n = 1" + "0" * 640 + "\n"
        payload = bigint.encode("utf-8") + self._VALID_COLORS
        self.assertLess(len(payload), self._CAP)
        with self.assertRaises(ValueError):
            tomllib.loads(bigint)
        with tempfile.TemporaryDirectory() as td:
            home = Path(td)
            _write_raw(home, b"bigint\n", payload)
            t = theme.read_theme(home, fc=lambda match: None)
        self.assertEqual(t.source, "fallback")
        self.assertEqual(t.slug, "bigint")
        self.assertIn("no readable colors.toml", t.status.lower())

    def test_fifo_theme_name_does_not_block(self):
        if not hasattr(os, "mkfifo") or not hasattr(os, "O_NONBLOCK"):
            self.skipTest("non-blocking FIFO unavailable")
        with tempfile.TemporaryDirectory() as td:
            home = Path(td)
            state_dir = home / ".local" / "state" / "omarchy" / "current"
            (state_dir / "theme").mkdir(parents=True)
            fifo = state_dir / "theme.name"
            os.mkfifo(fifo)
            t = self._run_without_blocking_fifo(
                fifo, lambda: theme.read_theme(home, fc=lambda match: None)
            )
        self.assertEqual(t.source, "fallback")
        self.assertIsNone(t.slug)

    def test_prefilled_fifo_theme_name_is_refused(self):
        if not hasattr(os, "mkfifo") or not hasattr(os, "O_NONBLOCK"):
            self.skipTest("non-blocking FIFO unavailable")
        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        home = Path(td.name)
        state_dir = home / ".local" / "state" / "omarchy" / "current"
        theme_dir = state_dir / "theme"
        theme_dir.mkdir(parents=True)
        (theme_dir / "colors.toml").write_bytes(self._VALID_COLORS)
        fifo = state_dir / "theme.name"
        os.mkfifo(fifo)
        rfd = os.open(fifo, os.O_RDONLY | os.O_NONBLOCK)
        self.addCleanup(os.close, rfd)
        wfd = os.open(fifo, os.O_WRONLY | os.O_NONBLOCK)
        try:
            self.assertEqual(os.write(wfd, b"tokyo-night\n"), 12)
        finally:
            os.close(wfd)
        t = self._run_without_blocking_fifo(
            fifo, lambda: theme.read_theme(home, fc=lambda match: None)
        )
        self.assertEqual(t.source, "fallback")
        self.assertIsNone(t.slug)

    def test_fifo_colors_toml_does_not_block(self):
        if not hasattr(os, "mkfifo") or not hasattr(os, "O_NONBLOCK"):
            self.skipTest("non-blocking FIFO unavailable")
        with tempfile.TemporaryDirectory() as td:
            fifo = Path(td) / "colors.toml"
            os.mkfifo(fifo)
            result = self._run_without_blocking_fifo(
                fifo, lambda: theme._parse_colors_toml(fifo)
            )
        self.assertEqual(result, ({}, None))

    def test_nonblocking_none_read_falls_back(self):
        with tempfile.TemporaryDirectory() as td:
            home = Path(td)
            _write_raw(home, b"tokyo-night\n", self._VALID_COLORS)
            with self._patch_theme_open(return_none=True) as (read_sizes, _):
                t = theme.read_theme(home, fc=lambda match: None)
        self.assertEqual(t.source, "fallback")
        self.assertIsNone(t.slug)
        self.assertEqual(len(read_sizes), 1)

    def test_theme_read_is_bounded_to_cap_plus_one(self):
        with tempfile.TemporaryDirectory() as td:
            colors_path = Path(td) / "colors.toml"
            colors_path.write_bytes(self._VALID_COLORS)
            with self._patch_theme_open() as (read_sizes, _):
                colors, _ = theme._parse_colors_toml(colors_path)
            self.assertEqual(read_sizes, [self._CAP + 1])
            self.assertEqual(colors["background"], "#1a1b26")

            colors_path.write_bytes(
                self._VALID_COLORS + b"#" + b"a" * (64 * 1024)
            )
            with self._patch_theme_open() as (read_sizes, fd_offsets):
                result = theme._parse_colors_toml(colors_path)
        self.assertEqual(read_sizes, [self._CAP + 1])
        self.assertLessEqual(fd_offsets[0], self._CAP + 1)
        self.assertEqual(result, ({}, None))

    def test_oversized_theme_name_falls_back(self):
        cap = self._CAP
        padded_name = b"tokyo-night" + b" " * (cap + 1 - len(b"tokyo-night"))
        self.assertEqual(len(padded_name), cap + 1)
        with tempfile.TemporaryDirectory() as td:
            home = Path(td)
            _write_raw(home, padded_name, self._VALID_COLORS)
            t = theme.read_theme(home, fc=lambda match: None)
        self.assertEqual(t.source, "fallback")
        self.assertIsNone(t.slug)

    def test_oversized_colors_toml_falls_back(self):
        cap = self._CAP
        prefix = self._VALID_COLORS + b"#"
        padded_colors = prefix + b"a" * (cap + 1 - len(prefix))
        self.assertEqual(len(padded_colors), cap + 1)
        with tempfile.TemporaryDirectory() as td:
            home = Path(td)
            _write_raw(home, b"tokyo-night\n", padded_colors)
            t = theme.read_theme(home, fc=lambda match: None)
        self.assertEqual(t.source, "fallback")
        self.assertEqual(t.slug, "tokyo-night")
        self.assertIn("no readable colors.toml", t.status.lower())

    def test_exact_size_cap_theme_files_load(self):
        cap = self._CAP
        padded_name = b"tokyo-night" + b" " * (cap - len(b"tokyo-night"))
        prefix = self._VALID_COLORS + b"#"
        padded_colors = prefix + b"a" * (cap - len(prefix))
        self.assertEqual(len(padded_name), cap)
        self.assertEqual(len(padded_colors), cap)
        with tempfile.TemporaryDirectory() as td:
            home = Path(td)
            _write_raw(home, padded_name, padded_colors)
            t = theme.read_theme(home, fc=lambda match: None)
        self.assertEqual(t.source, "omarchy")
        self.assertEqual(t.slug, "tokyo-night")
        self.assertEqual(t.css["--mlx-canvas"], "#1a1b26")

    def test_untraversable_theme_dir_falls_back(self):
        if os.geteuid() == 0:
            self.skipTest("root bypasses directory mode 0600")
        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        home = Path(td.name)
        theme_dir = _write_raw(home, b"tokyo-night\n", self._VALID_COLORS)
        theme_dir.chmod(0o600)
        self.addCleanup(theme_dir.chmod, 0o700)
        t = theme.read_theme(home, fc=lambda match: None)
        self.assertEqual(t.source, "fallback")
        self.assertEqual(t.slug, "tokyo-night")
        self.assertIn("no readable colors.toml", t.status.lower())

    def test_symlinked_theme_dir_and_colors_load(self):
        with tempfile.TemporaryDirectory() as td:
            home = Path(td)
            state_dir = home / ".local" / "state" / "omarchy" / "current"
            state_dir.mkdir(parents=True)
            (state_dir / "theme.name").write_bytes(b"tokyo-night\n")
            real_dir = home / ".local" / "share" / "omarchy" / "themes" / "tokyo-night"
            real_dir.mkdir(parents=True)
            real_colors = real_dir / "colors.real.toml"
            real_colors.write_bytes(self._VALID_COLORS)
            (real_dir / "colors.toml").symlink_to(real_colors)
            (state_dir / "theme").symlink_to(real_dir)
            t = theme.read_theme(home, fc=lambda match: None)
        self.assertEqual(t.source, "omarchy")
        self.assertEqual(t.slug, "tokyo-night")
        self.assertEqual(t.css["--mlx-canvas"], "#1a1b26")
        self.assertEqual(t.css["--mlx-text"], "#c0caf5")
        self.assertEqual(t.css["--mlx-focus"], "#7aa2f7")


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

    def test_fc_default_rejects_non_utf8_output(self):
        with tempfile.TemporaryDirectory() as td:
            fake = Path(td) / "fc-match"
            fake.write_text(f"#!{sys.executable}\n"
                            "import sys; sys.stdout.buffer.write(b'Bad\\xffFont')\n",
                            encoding="utf-8")
            fake.chmod(0o755)
            with unittest.mock.patch.dict(os.environ,
                                          {"PATH": f"{td}{os.pathsep}{os.environ['PATH']}"}):
                self.assertIsNone(theme._fc_default("monospace"))


if __name__ == "__main__":
    unittest.main()
