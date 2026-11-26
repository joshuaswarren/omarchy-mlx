"""Offline system-layout staging: vendored hashes, venv build, /usr tree.

Everything runs in a temp dir against tiny generated wheels — no network,
no hardware. The build path under test is exactly what a PKGBUILD runs:
packaging/verify-vendor.sh, packaging/build-venv.sh, then
install.sh --system.
"""

import os
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PACKAGING = REPO / "packaging"
INSTALL = REPO / "install.sh"

PYTAG = "cp%d%d" % sys.version_info[:2]


def _wheel_bytes() -> None:
    """Write the fake vendor wheels into directory `vendor`."""
    def make_wheel(path: Path, name: str, version: str, tag: str,
                   files: dict, exec_files=()) -> None:
        records = []

        def add(name_in_wheel: str, data: bytes) -> None:
            import base64
            import hashlib
            digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest())
            records.append(f"{name_in_wheel},sha256={digest.rstrip(b'=').decode()},{len(data)}")

        with zipfile.ZipFile(path, "w") as zf:
            for fname, content in files.items():
                info = zipfile.ZipInfo(fname, date_time=(2026, 9, 30, 0, 0, 0))
                info.create_system = 3  # unix: pip honors external_attr
                mode = 0o100755 if fname in exec_files else 0o100644
                info.external_attr = mode << 16
                data = content.encode()
                zf.writestr(info, data)
                add(fname, data)
            distinfo = f"{name.replace('-', '_')}-{version}.dist-info"
            for member, body in (
                ("METADATA", f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n"),
                ("WHEEL", f"Wheel-Version: 1.0\nGenerator: test\nRoot-Is-Purelib: true\nTag: {tag}\n"),
            ):
                info = zipfile.ZipInfo(f"{distinfo}/{member}", date_time=(2026, 9, 30, 0, 0, 0))
                info.create_system = 3
                info.external_attr = 0o100644 << 16
                data = body.encode()
                zf.writestr(info, data)
                add(f"{distinfo}/{member}", data)
            record = f"{distinfo}/RECORD"
            records.append(f"{record},,")
            info = zipfile.ZipInfo(record, date_time=(2026, 9, 30, 0, 0, 0))
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            zf.writestr(info, "\n".join(records) + "\n")

    def build(vendor: Path) -> None:
        vendor.mkdir(parents=True)
        make_wheel(
            vendor / f"mlx_omarchy-0.0.0.dev0+fake-{PYTAG}-{PYTAG}-linux_x86_64.whl",
            "mlx-omarchy", "0.0.0.dev0+fake", f"{PYTAG}-{PYTAG}-linux_x86_64",
            {"mlx/__init__.py": "", "mlx/bin/mlx-omarchy-info": "#!/bin/sh\necho fake-info\n",
             "mlx/bin/mlx-omarchy-parakeet": "#!/usr/bin/env python3\nprint('fake-parakeet')\n"},
            exec_files=("mlx/bin/mlx-omarchy-info", "mlx/bin/mlx-omarchy-parakeet"),
        )
        make_wheel(
            vendor / "fakelib-1.0-py3-none-any.whl",
            "fakelib", "1.0", "py3-none-any", {"fakelib/__init__.py": "X = 1\n"},
        )

    return build


make_vendor_wheels = _wheel_bytes()


class VendorToolingTest(unittest.TestCase):
    """gen-vendor-lock.sh and verify-vendor.sh against a fake vendor dir."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.vendor = self.tmp / "vendor"
        make_vendor_wheels(self.vendor)
        self.lock = self.tmp / "lock.txt"
        self.lock.write_text(
            subprocess.run(
                ["bash", str(PACKAGING / "gen-vendor-lock.sh"), str(self.vendor)],
                check=True, capture_output=True, text=True,
            ).stdout
        )

    def tearDown(self):
        self._tmp.cleanup()

    def _verify(self):
        return subprocess.run(
            ["bash", str(PACKAGING / "verify-vendor.sh"), str(self.vendor), str(self.lock)],
            capture_output=True, text=True,
        )

    def test_lock_pins_every_wheel_with_sha256(self):
        lines = self.lock.read_text().splitlines()
        self.assertEqual(len(lines), 2)
        self.assertIn("mlx-omarchy==0.0.0.dev0+fake --hash=sha256:", "\n".join(lines))
        self.assertIn("fakelib==1.0 --hash=sha256:", "\n".join(lines))
        for line in lines:
            self.assertRegex(line, r"^[\w.-]+==\S+ --hash=sha256:[0-9a-f]{64}$")

    def test_verify_accepts_intact_vendor(self):
        result = self._verify()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("verified 2", result.stdout)

    def test_verify_rejects_tampered_wheel(self):
        wheel = next(self.vendor.glob("fakelib-*.whl"))
        data = wheel.read_bytes()
        wheel.write_bytes(data.replace(b"X = 1", b"X = 2"))
        result = self._verify()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("hash mismatch", result.stderr)

    def test_verify_rejects_missing_wheel(self):
        next(self.vendor.glob("fakelib-*.whl")).unlink()
        result = self._verify()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("no wheel provides it", result.stderr)


class BuildVenvTest(unittest.TestCase):
    """build-venv.sh: offline, hash-checked, interpreter-tag-checked."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls._tmp.cleanup)
        cls.tmp = Path(cls._tmp.name)
        cls.vendor = cls.tmp / "vendor"
        make_vendor_wheels(cls.vendor)
        cls.lock = cls.tmp / "lock.txt"
        cls.lock.write_text(
            subprocess.run(
                ["bash", str(PACKAGING / "gen-vendor-lock.sh"), str(cls.vendor)],
                check=True, capture_output=True, text=True,
            ).stdout
        )
        cls.venv = cls.tmp / "venv"
        result = subprocess.run(
            ["bash", str(PACKAGING / "build-venv.sh"),
             "--vendor", str(cls.vendor), "--lock", str(cls.lock),
             "--venv", str(cls.venv), "--python", sys.executable],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise AssertionError(f"build-venv.sh failed:\n{result.stdout}\n{result.stderr}")

    def test_venv_imports_vendored_packages(self):
        result = subprocess.run(
            [str(self.venv / "bin" / "python"), "-c", "import fakelib, mlx; print('ok')"],
            capture_output=True, text=True, check=True,
        )
        self.assertEqual(result.stdout.strip(), "ok")

    def test_wrong_interpreter_tag_is_refused(self):
        vendor = self.tmp / "vendor-wrongtag"
        make_vendor_wheels(vendor)
        wheel = next(vendor.glob("mlx_omarchy-*.whl"))
        wheel.rename(wheel.with_name(wheel.name.replace(PYTAG, "cp99")))
        result = subprocess.run(
            ["bash", str(PACKAGING / "build-venv.sh"),
             "--vendor", str(vendor), "--lock", str(self.lock),
             "--venv", str(self.tmp / "venv-wrongtag"), "--python", sys.executable],
            capture_output=True, text=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("targets cp99", result.stderr)

    def test_missing_lock_refuses_to_build(self):
        result = subprocess.run(
            ["bash", str(PACKAGING / "build-venv.sh"),
             "--vendor", str(self.vendor), "--lock", str(self.tmp / "absent.txt"),
             "--venv", str(self.tmp / "venv-x"), "--python", sys.executable],
            capture_output=True, text=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse((self.tmp / "venv-x" / "bin" / "python").exists())


class SystemStageTest(unittest.TestCase):
    """install.sh --system stages the whole /usr tree offline."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls._tmp.cleanup)
        cls.tmp = Path(cls._tmp.name)
        cls.vendor = cls.tmp / "vendor"
        make_vendor_wheels(cls.vendor)
        cls.lock = cls.tmp / "lock.txt"
        cls.lock.write_text(
            subprocess.run(
                ["bash", str(PACKAGING / "gen-vendor-lock.sh"), str(cls.vendor)],
                check=True, capture_output=True, text=True,
            ).stdout
        )
        cls.stage = cls.tmp / "stage"
        result = subprocess.run(
            ["bash", str(INSTALL), "--system",
             "--dest-root", str(cls.stage),
             "--vendor", str(cls.vendor),
             "--lock", str(cls.lock),
             "--python", sys.executable],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise AssertionError(f"install.sh --system failed:\n{result.stdout}\n{result.stderr}")

    def test_venv_bin_scripts_carry_final_paths(self):
        """venv/bin scripts survive package()'s copy to /.

        build-venv.sh creates the venv at the staging absolute path, so
        pip, activate, and any entry-point scripts embed that path. A
        PKGBUILD copies the staged tree to / in package(); scripts that
        still name the staging dir would be dead on an installed system.
        """
        venv_bin = self.stage / "usr/lib/omarchy-mlx/venv/bin"
        staging_prefix = str(self.stage)
        final_venv = "/usr/lib/omarchy-mlx/venv"
        checked = 0
        for script in venv_bin.iterdir():
            if script.is_symlink() or not script.is_file():
                continue
            text = script.read_text(errors="replace")
            self.assertNotIn(staging_prefix, text,
                             f"{script.name} embeds the staging path")
            if script.name.startswith("pip"):
                self.assertTrue(text.startswith(f"#!{final_venv}/bin/python"),
                                f"{script.name} shebang: {text.splitlines()[0]}")
            checked += 1
        self.assertGreater(checked, 0, "no regular scripts under venv/bin")

    def test_full_tree_is_staged(self):
        self.assertTrue((self.stage / "usr/lib/omarchy-mlx/venv/bin/python").exists())
        for launcher in ("mlx-omarchy", "mlx-omarchy-demo", "mlx-omarchy-chat",
                         "mlx-omarchy-serve", "mlx-omarchy-info",
                         "mlx-omarchy-parakeet",
                         "mlx-omarchy-retire-legacy", "omarchy-mlx-serve"):
            self.assertTrue((self.stage / "usr/bin" / launcher).is_file(), launcher)
        self.assertTrue((self.stage / "usr/lib/systemd/user/mlx-omarchy-chat.service").is_file())
        self.assertTrue((self.stage / "usr/share/applications/mlx-omarchy-chat.desktop").is_file())
        self.assertTrue((self.stage / "usr/share/omarchy-mlx/paths.sh").is_file())
        self.assertTrue((self.stage / "usr/share/doc/mlx-omarchy/README.md").is_file())
        self.assertTrue((self.stage / "usr/lib/omarchy-mlx/chat.py").is_file())

    def test_serve_trees_live_in_the_venv(self):
        site = next((self.stage / "usr/lib/omarchy-mlx/venv").glob("lib/python3.*/site-packages"))
        for pkg in ("mlx_omarchy_serve", "mlx_omarchy_laya", "mlx_omarchy_bonsai2",
                    "mlx_omarchy_assistant"):
            self.assertTrue((site / pkg / "__init__.py").is_file(), pkg)
        self.assertTrue((site / "mlx_omarchy_paths.py").is_file())

    def test_generated_files_carry_final_system_paths(self):
        unit = (self.stage / "usr/lib/systemd/user/mlx-omarchy-chat.service").read_text()
        self.assertIn("ExecStart=/usr/bin/mlx-omarchy-chat --resume --no-browser", unit)
        serve = (self.stage / "usr/bin/mlx-omarchy-serve").read_text()
        self.assertIn('exec "/usr/lib/omarchy-mlx/venv/bin/python" -m mlx_omarchy_serve', serve)
        chat = (self.stage / "usr/bin/mlx-omarchy-chat").read_text()
        self.assertIn('exec "/usr/lib/omarchy-mlx/venv/bin/python" -m mlx_omarchy_assistant', chat)
        info = (self.stage / "usr/bin/mlx-omarchy-info").read_text()
        self.assertIn("/usr/lib/omarchy-mlx/venv/lib/python3.", info)
        parakeet = (self.stage / "usr/bin/mlx-omarchy-parakeet").read_text()
        self.assertIn('exec "/usr/lib/omarchy-mlx/venv/bin/python"', parakeet)
        self.assertIn("mlx/bin/mlx-omarchy-parakeet", parakeet)
        desktop = (self.stage / "usr/share/applications/mlx-omarchy-chat.desktop").read_text()
        self.assertIn("Exec=/usr/bin/mlx-omarchy-chat", desktop)

    def test_staged_parakeet_launcher_runs_the_wheel_cli(self):
        # The staged launcher embeds the final /usr prefix (like every
        # staged launcher here), so the sandbox exec goes through the
        # staged venv python and the staged CLI file it names.
        staged_cli = next(
            (self.stage / "usr/lib/omarchy-mlx/venv").glob(
                "lib/python3.*/site-packages/mlx/bin/mlx-omarchy-parakeet")
        )
        result = subprocess.run(
            [str(self.stage / "usr/lib/omarchy-mlx/venv/bin/python"),
             str(staged_cli)],
            capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "fake-parakeet")

    def test_staged_venv_serves_and_discovers(self):
        venv_python = self.stage / "usr/lib/omarchy-mlx/venv/bin/python"
        result = subprocess.run(
            [str(venv_python), "-m", "mlx_omarchy_serve", "--help"],
            capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("recommend", result.stdout)
        discover = subprocess.run(
            [str(venv_python), "-c",
             "import mlx_omarchy_paths as p; print(p.system_venv())"],
            capture_output=True, text=True, check=True,
        )
        self.assertEqual(discover.stdout.strip(), "/usr/lib/omarchy-mlx/venv")

    def test_staged_retire_runs_through_the_name_table(self):
        home = self.tmp / "legacy-home"
        legacy_venv = home / ".local/share/mlx-omarchy/venv"
        legacy_venv.mkdir(parents=True)
        (legacy_venv / "pyvenv.cfg").write_text("[config]\n")
        env = {
            **os.environ,
            "HOME": str(home),
            "MLX_OMARCHY_HOME": str(home / ".local/share/mlx-omarchy"),
            "SYSTEM_SHARE_PREFIX": str(self.stage / "usr/share/omarchy-mlx"),
        }
        result = subprocess.run(
            [str(self.stage / "usr/bin/mlx-omarchy-retire-legacy"), "--yes"],
            env=env, capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(legacy_venv.exists())
        self.assertTrue((home / ".local/share/mlx-omarchy").is_dir())

    def test_staged_retire_finds_name_table_without_env(self):
        """The staged retire script must locate paths.sh from its own
        staged layout (/usr/bin/../share/omarchy-mlx), with no
        SYSTEM_SHARE_PREFIX override — the same relative path works once
        package() copies the tree to /. Found broken on hardware
        (SysInstallHw, jw16 2026-09-30): the candidate list only knew
        self_dir, ../packaging, and the absolute /usr fallback."""
        home = self.tmp / "legacy-home-noenv"
        legacy_venv = home / ".local/share/mlx-omarchy/venv"
        legacy_venv.mkdir(parents=True)
        env = {k: v for k, v in os.environ.items()
               if k not in ("SYSTEM_SHARE_PREFIX", "MLX_OMARCHY_HOME")}
        env["HOME"] = str(home)
        result = subprocess.run(
            [str(self.stage / "usr/bin/mlx-omarchy-retire-legacy")],
            env=env, capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("would remove venv", result.stdout)
        self.assertTrue(legacy_venv.exists(), "dry run must delete nothing")

    def test_system_mode_requires_vendor_and_lock(self):
        result = subprocess.run(
            ["bash", str(INSTALL), "--system"], capture_output=True, text=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--vendor", result.stderr)


if __name__ == "__main__":
    unittest.main()
