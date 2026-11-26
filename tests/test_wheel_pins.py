"""Tests for the requirements pin guard (scripts/check-wheel-pins.py).

Stdlib unittest only, no network. The 2026-09-03 trap: a requirements
freeze carried ``mlx-omarchy @ file:///...whl#sha256=...`` and an
exclusion filter anchored on ``==`` missed it, so pip reinstalled a
stale wheel generation in every fresh venv.
"""


import importlib.util
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

_spec = importlib.util.spec_from_file_location(
    "check_wheel_pins", REPO_ROOT / "scripts" / "check-wheel-pins.py")
pins = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pins)


FILE_PIN = ("mlx_omarchy @ file:///home/u/dist/"
            "mlx_omarchy-0.32.2.dev20260903-cp311-cp311-linux_x86_64.whl"
            "#sha256=6e54ab2b8e808b78d052eae0d5dc76e4f4111f2afac3e572e9eec"
            "243d8c5f42c")


class WheelPinTests(unittest.TestCase):
    def violations(self, *lines):
        return [msg for sev, msg in pins.check_lines(list(lines))
                if sev == "VIOLATION"]

    def test_clean_requirements_pass(self):
        self.assertEqual(self.violations(
            "numpy==1.26.0",
            "# a comment mentioning mlx-omarchy is not a pin",
            "mlx-lm>=0.4",
            "-r other.txt",
        ), [])

    def test_version_pin_rejected(self):
        found = self.violations("mlx-omarchy==0.32.2")
        self.assertEqual(len(found), 1)
        self.assertIn("pinned in a requirements input", found[0])

    def test_local_file_pin_rejected(self):
        found = self.violations(f"mlx-omarchy @ {FILE_PIN}")
        self.assertEqual(len(found), 1)
        self.assertIn(FILE_PIN, found[0])

    def test_direct_url_pin_rejected(self):
        found = self.violations(
            "mlx-omarchy @ https://example.com/mlx_omarchy-1.0-cp311.whl")
        self.assertEqual(len(found), 1)

    def test_bare_direct_url_line_rejected(self):
        found = self.violations(
            "https://example.com/mlx_omarchy-1.0-cp311-none-any.whl")
        self.assertEqual(len(found), 1)
        self.assertIn("direct URL", found[0])

    def test_range_pin_and_underscore_name_rejected(self):
        self.assertEqual(len(self.violations("mlx-omarchy>=0.3",
                                             "mlx_omarchy==0.1")), 2)

    def test_editable_install_is_warning_only(self):
        out = list(pins.check_lines(["-e ./src/mlx-omarchy"]))
        self.assertEqual([sev for sev, _ in out], ["WARNING"])

    def test_self_test(self):
        self.assertEqual(pins._self_test(), 0)


if __name__ == "__main__":
    unittest.main()
