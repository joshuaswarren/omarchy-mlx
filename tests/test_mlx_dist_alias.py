"""Tests for the `mlx` distribution alias (scripts/mlx_dist_alias.py).

Stdlib unittest only, no network. Issue omarchy-mlx#566: the wheel
installs the ``mlx`` module but its dist-info is ``mlx-omarchy``, so
``importlib.metadata.version("mlx")`` raises PackageNotFoundError on a
healthy install. The first fix injected a second dist-info into the
wheel itself; w8E showed pip and uv both refuse it (PEP 427: one
dist-info per wheel). The alias now lives outside the wheel: a
companion ``mlx-<ver>`` wheel for standalone pip users, and
site-packages stamping for install.sh paths. The real-pip install test
below is the regression guard for that failure class.
"""


import os
import shutil
import subprocess
import sys
import tempfile
import unittest
import venv
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "mlx_dist_alias.py"
VERSION = "0.32.4.dev202611210237"


def make_wheel(path: Path, version: str = VERSION) -> Path:
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
                    f"Metadata-Version: 2.1\nName: mlx-omarchy\n"
                    f"Version: {version}\n")
        zf.writestr(f"{dist}/WHEEL", "Wheel-Version: 1.0\nTag: py3-none-any\n")
        zf.writestr(f"{dist}/RECORD", record + "\n")
    return path


class EmitWheelTests(unittest.TestCase):
    def test_emit_builds_companion_wheel(self):
        with tempfile.TemporaryDirectory() as td:
            wheel = make_wheel(Path(td) / f"mlx_omarchy-{VERSION}-py3-none-any.whl")
            out = subprocess.run(
                [sys.executable, str(SCRIPT),
                 "--emit-wheel", str(wheel), "--out", td],
                check=True, capture_output=True, text=True).stdout.strip()
            self.assertTrue(out.endswith(f"mlx-{VERSION}-py3-none-any.whl"))
            names = zipfile.ZipFile(out).namelist()
            self.assertEqual(len(names), 4)
            meta = zipfile.ZipFile(out).read(
                f"mlx-{VERSION}.dist-info/METADATA").decode()
            self.assertIn("Name: mlx\n", meta)
            self.assertIn(f"Version: {VERSION}\n", meta)
            record = zipfile.ZipFile(out).read(
                f"mlx-{VERSION}.dist-info/RECORD").decode()
            self.assertIn(f"mlx-{VERSION}.dist-info/RECORD,,", record)

    def test_emit_from_site_packages_dir(self):
        with tempfile.TemporaryDirectory() as td:
            wheel = make_wheel(Path(td) / f"mlx_omarchy-{VERSION}-py3-none-any.whl")
            site = Path(td) / "site"
            zipfile.ZipFile(wheel).extractall(site)
            out = subprocess.run(
                [sys.executable, str(SCRIPT),
                 "--emit-wheel", str(site), "--out", td],
                check=True, capture_output=True, text=True).stdout.strip()
            self.assertTrue(Path(out).exists())


class StampTests(unittest.TestCase):
    def _install(self, site: Path, version: str = VERSION):
        dist = site / f"mlx_omarchy-{version}.dist-info"
        dist.mkdir(parents=True)
        (dist / "METADATA").write_text(
            f"Metadata-Version: 2.1\nName: mlx-omarchy\nVersion: {version}\n")

    def test_stamps_alias(self):
        with tempfile.TemporaryDirectory() as td:
            site = Path(td)
            self._install(site)
            subprocess.run([sys.executable, str(SCRIPT),
                            "--site-packages", str(site)], check=True)
            meta = (site / f"mlx-{VERSION}.dist-info" / "METADATA").read_text()
            self.assertIn("Name: mlx\n", meta)

    def test_removes_stale_alias_keeps_foreign_mlx(self):
        with tempfile.TemporaryDirectory() as td:
            site = Path(td)
            self._install(site)
            stale = site / "mlx-0.32.3.dist-info"
            stale.mkdir()
            (stale / "INSTALLER").write_text("omarchy-mlx\n")
            foreign = site / "mlx-9.9.9.dist-info"
            foreign.mkdir()
            (foreign / "INSTALLER").write_text("pip\n")
            subprocess.run([sys.executable, str(SCRIPT),
                            "--site-packages", str(site)], check=True)
            self.assertFalse(stale.exists())
            self.assertTrue(foreign.exists())
            self.assertTrue((site / f"mlx-{VERSION}.dist-info").exists())

    def test_idempotent_and_partial_recovery(self):
        with tempfile.TemporaryDirectory() as td:
            site = Path(td)
            self._install(site)
            partial = site / f"mlx-{VERSION}.dist-info"
            partial.mkdir()  # no METADATA: a torn earlier stamp
            for _ in range(2):
                subprocess.run([sys.executable, str(SCRIPT),
                                "--site-packages", str(site)], check=True)
            self.assertTrue((partial / "METADATA").exists())


class PipedInstallTests(unittest.TestCase):
    """curl|bash path: ROOT is empty, so install.sh must stamp with its
    inline fallback — no fetch of scripts/mlx_dist_alias.py from the
    release tag (which 404s on every pre-alias tag, w8E rc=22 repro)."""

    def test_no_tag_fetch_for_alias_script(self):
        text = (REPO_ROOT / "install.sh").read_text()
        self.assertNotIn("scripts/mlx_dist_alias.py\" -o", text,
                         "piped path must not curl the helper script")
        self.assertNotIn("$VERSION/scripts/mlx_dist_alias.py", text)

    def test_inline_stamper_runs_in_fresh_venv(self):
        import re
        text = (REPO_ROOT / "install.sh").read_text()
        m = re.search(r"<<'PYEOF'\n(.*?)\nPYEOF", text, re.S)
        self.assertIsNotNone(m, "stamp_mlx_alias inline block missing")
        with tempfile.TemporaryDirectory() as td:
            env = Path(td) / "venv"
            venv.create(env, with_pip=True)
            vpy = env / "bin" / "python"
            site = next((env / "lib").glob("python3*/site-packages"))
            dist = site / "mlx_omarchy-0.32.4.dist-info"
            dist.mkdir()
            (dist / "METADATA").write_text(
                "Metadata-Version: 2.1\nName: mlx-omarchy\nVersion: 0.32.4\n")
            script = Path(td) / "stamp.py"
            script.write_text(m.group(1))
            subprocess.run([str(vpy), str(script)], check=True,
                           capture_output=True, text=True)
            meta = (site / "mlx-0.32.4.dist-info" / "METADATA").read_text()
            self.assertIn("Name: mlx\n", meta)
            out = subprocess.run(
                [str(vpy), "-c",
                 "import importlib.metadata as m;print(m.version('mlx'))"],
                check=True, capture_output=True, text=True).stdout.strip()
            self.assertEqual(out, "0.32.4")


class RealInstallTests(unittest.TestCase):
    """pip and uv both install the wheel pair on a fresh venv (PEP 427)."""

    def _venv_install(self, installer):
        with tempfile.TemporaryDirectory() as td:
            wheel = make_wheel(Path(td) / f"mlx_omarchy-{VERSION}-py3-none-any.whl")
            alias = Path(subprocess.run(
                [sys.executable, str(SCRIPT),
                 "--emit-wheel", str(wheel), "--out", td],
                check=True, capture_output=True, text=True).stdout.strip())
            env = Path(td) / "venv"
            venv.create(env, with_pip=True)
            vpy = env / "bin" / "python"
            if installer == "uv":
                cmd = ["uv", "pip", "install", "--python", str(vpy),
                       "--no-deps", str(wheel), str(alias)]
            else:
                cmd = [str(env / "bin" / "pip"), "install",
                       "--no-deps", "--no-index", str(wheel), str(alias)]
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            out = subprocess.run(
                [str(vpy), "-c",
                 "import importlib.metadata as m;"
                 "print(m.version('mlx'));print(m.version('mlx-omarchy'))"],
                check=True, capture_output=True, text=True).stdout.split()
            self.assertEqual(out, [VERSION, VERSION])

    def test_pip_installs_both_wheels(self):
        self._venv_install("pip")

    def test_uv_installs_both_wheels(self):
        if shutil.which("uv") is None:
            self.skipTest("uv not on PATH")
        self._venv_install("uv")


if __name__ == "__main__":
    unittest.main()
