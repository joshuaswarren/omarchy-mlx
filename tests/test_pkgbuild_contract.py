"""Packaging contract: the pacman package must not strip the venv.

makepkg's default ELF strip changes the shipped bytes, and then
scripts/mlx_provenance.py reports mismatch on every packaged install
(omarchy-mlx 0.7.10-1, 2026-10-02): the loaded libmlx.so could never be
quoted as evidence on a packaged host. packaging/PKGBUILD.example pins
the no-strip contract (measured cost of keeping symbols: 3.2 MB of the
25.3 MB libmlx.so, ~0.3% of the installed tree); the omarchy-pkgs
recipe tracks the example.
"""

import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PKGBUILD = REPO_ROOT / "packaging" / "PKGBUILD.example"


class PkgbuildStripContract(unittest.TestCase):
    def setUp(self):
        self.text = PKGBUILD.read_text()

    def test_example_declares_no_strip(self):
        self.assertIn("options=(!strip)", self.text)

    def test_example_never_strips_and_never_rewrites_binaries(self):
        commands = [ln for ln in self.text.splitlines()
                    if not ln.lstrip().startswith("#")
                    and not ln.strip() == "options=(!strip)"]
        self.assertFalse([ln for ln in commands
                          if "strip" in ln or "build-manifest" in ln],
                         "package() must not strip or rewrite binaries")


if __name__ == "__main__":
    unittest.main()
