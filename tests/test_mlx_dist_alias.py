"""Tests for the `mlx` distribution alias (scripts/mlx_dist_alias.py).

Stdlib unittest only, no network. Issue omarchy-mlx#566: the wheel
installs the ``mlx`` module but its dist-info is named
``mlx-omarchy``, so ``importlib.metadata.version("mlx")`` raises
PackageNotFoundError and version-probing consumers (songe's
trainability probe was the report) fail on a healthy install.
"""


import importlib.metadata
import importlib.util
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "mlx_dist_alias.py"


def make_wheel(path: Path, version: str = "0.32.4.dev202611210237") -> Path:
    dist = f"mlx_omarchy-{version}.dist-info"
    record = "\n".join([
        "mlx/__init__.py,sha256=abc,10",
        f"{dist}/METADATA,sha256=def,20",
        f"{dist}/WHEEL,sha256=ghi,30",
        f"{dist}/RECORD,,",
    ])
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("mlx/__init__.py", "__version__ = 'x'\n")
        zf.writestr(f"{dist}/METADATA",
                    f"Metadata-Version: 2.1\nName: mlx-omarchy\nVersion: {version}\n")
        zf.writestr(f"{dist}/WHEEL", "Wheel-Version: 1.0\nTag: py3-none-any\n")
        zf.writestr(f"{dist}/RECORD", record + "\n")
    return path


class WheelAliasTests(unittest.TestCase):
    def test_inject_adds_mlx_dist_info(self):
        with tempfile.TemporaryDirectory() as td:
            wheel = make_wheel(Path(td) / "mlx_omarchy-0.32.4.dev202611210237-cp314-cp314-linux_aarch64.whl")
            subprocess.run([sys.executable, str(SCRIPT), str(wheel)], check=True)
            names = zipfile.ZipFile(wheel).namelist()
            self.assertIn("mlx-0.32.4.dev202611210237.dist-info/METADATA", names)
            self.assertIn("mlx-0.32.4.dev202611210237.dist-info/RECORD", names)
            meta = zipfile.ZipFile(wheel).read(
                "mlx-0.32.4.dev202611210237.dist-info/METADATA").decode()
            self.assertIn("Name: mlx\n", meta)
            self.assertIn("Version: 0.32.4.dev202611210237\n", meta)
            # The main RECORD must cover the new files or the installer
            # flags an incomplete wheel.
            record = zipfile.ZipFile(wheel).read(
                "mlx_omarchy-0.32.4.dev202611210237.dist-info/RECORD").decode()
            self.assertIn("mlx-0.32.4.dev202611210237.dist-info/METADATA", record)

    def test_inject_is_idempotent(self):
        with tempfile.TemporaryDirectory() as td:
            wheel = make_wheel(Path(td) / "mlx_omarchy-0.32.4.dev202611210237-cp314-cp314-linux_aarch64.whl")
            for _ in range(2):
                subprocess.run([sys.executable, str(SCRIPT), str(wheel)], check=True)
            names = zipfile.ZipFile(wheel).namelist()
            self.assertEqual(
                names.count("mlx-0.32.4.dev202611210237.dist-info/METADATA"), 1)

    def test_installed_tree_resolves_version(self):
        with tempfile.TemporaryDirectory() as td:
            wheel = make_wheel(Path(td) / "w.whl")
            subprocess.run([sys.executable, str(SCRIPT), str(wheel)], check=True)
            site = Path(td) / "site"
            zipfile.ZipFile(wheel).extractall(site)
            sys.path.insert(0, str(site))
            try:
                importlib.invalidate_caches()
                self.assertEqual(
                    importlib.metadata.version("mlx"),
                    "0.32.4.dev202611210237")
            finally:
                sys.path.remove(str(site))

    def test_site_packages_mode(self):
        with tempfile.TemporaryDirectory() as td:
            site = Path(td) / "site-packages"
            site.mkdir()
            make_install = site / "mlx_omarchy-0.32.4.dist-info"
            make_install.mkdir()
            (make_install / "METADATA").write_text(
                "Metadata-Version: 2.1\nName: mlx-omarchy\nVersion: 0.32.4\n")
            subprocess.run([sys.executable, str(SCRIPT),
                            "--site-packages", str(site)], check=True)
            self.assertTrue(
                (site / "mlx-0.32.4.dist-info" / "METADATA").exists())


if __name__ == "__main__":
    unittest.main()
