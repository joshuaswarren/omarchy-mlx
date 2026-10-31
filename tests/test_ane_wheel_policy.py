"""CPU-only install-layout checks; no worker compilation, installation or execution."""
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
POLICY = "mlx/lib/tmpfiles.d/mlx-omarchy-ane.conf"
WORKER = "mlx/bin/mlx-omarchy-ane-worker"
SOURCE_POLICY = ROOT / "overlay/mlx/backend/omarchy/ane/mlx-omarchy-ane.conf"


def finished_wheel_code():
    script = (ROOT / "scripts/build-wheel.sh").read_text()
    section = script.split('echo "== verify finished wheel =="', 1)[1]
    return section.split("<<'EOF'\n", 1)[1].split("\nEOF", 1)[0]


class AneWheelPolicyTests(unittest.TestCase):
    def test_device_worker_install_carries_policy(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prefix = root / "wheel/mlx"
            (root / "CMakeLists.txt").write_text(
                'cmake_minimum_required(VERSION 3.25)\n'
                'project(policy_layout LANGUAGES CXX)\n'
                'include(GNUInstallDirs)\n'
                'add_library(mlx INTERFACE)\n'
                'set(MLX_OMARCHY_ANE_DEVICE ON)\n'
                f'add_subdirectory("{ROOT}/overlay/tools/mlx-omarchy-ane-worker" worker)\n'
            )
            subprocess.run(["cmake", "-S", str(root), "-B", str(root / "build"),
                            f"-DCMAKE_INSTALL_PREFIX={prefix}", "-DCMAKE_INSTALL_BINDIR=bin",
                            "-DCMAKE_SKIP_RPATH=ON"],
                           check=True, capture_output=True)
            # Only the install layout is under test; this fixture is never executed.
            worker = root / "build/worker/mlx-omarchy-ane-worker"
            worker.write_text("#!/bin/sh\nexit 0\n")
            result = subprocess.run(["cmake", "--install", str(root / "build")],
                                    capture_output=True, text=True, check=False)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual((root / "wheel" / POLICY).read_bytes(), SOURCE_POLICY.read_bytes())
            self.assertTrue((root / "wheel" / WORKER).is_file())
            wheel = root / "staged.whl"
            with zipfile.ZipFile(wheel, "w") as archive:
                for path in (root / "wheel").rglob("*"):
                    if path.is_file():
                        archive.write(path, path.relative_to(root / "wheel"))
            installer = (ROOT / "install.sh").read_text()
            code = installer.split("  python3 - \"$tmp/$wheel\" \"$ane_conf\" <<'PY'\n", 1)[1].split("\nPY", 1)[0]
            extracted = root / "extracted.conf"
            result = subprocess.run([sys.executable, "-c", code, str(wheel), str(extracted)],
                                    capture_output=True, text=True, check=False)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(extracted.read_bytes(), SOURCE_POLICY.read_bytes())

    def verify(self, members, suffix="aarch64"):
        with tempfile.TemporaryDirectory() as tmp:
            wheel = Path(tmp) / f"mlx_omarchy-0.0-cp314-cp314-linux_{suffix}.whl"
            with zipfile.ZipFile(wheel, "w") as archive:
                for name, data in members.items():
                    archive.writestr(name, data)
            return subprocess.run([sys.executable, "-c", finished_wheel_code(),
                                   str(wheel), str(Path(tmp) / "extracted")],
                                  capture_output=True, text=True, check=False)

    def test_missing_policy_rejected(self):
        result = self.verify({WORKER: b"worker fixture"})
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("tmpfiles policy", result.stderr)

    def test_missing_worker_rejected(self):
        self.assertNotEqual(self.verify({POLICY: SOURCE_POLICY.read_bytes()}).returncode, 0)

    def test_complete_device_wheel_accepted(self):
        result = self.verify({WORKER: b"worker fixture", POLICY: SOURCE_POLICY.read_bytes()})
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_duplicate_policy_rejected(self):
        self.assertNotEqual(self.verify({WORKER: b"worker fixture", POLICY: b"policy",
                                        "other/lib/tmpfiles.d/mlx-omarchy-ane.conf": b"policy"}).returncode, 0)

    def test_duplicate_worker_rejected(self):
        self.assertNotEqual(self.verify({WORKER: b"worker fixture", POLICY: b"policy",
                                        "other/bin/mlx-omarchy-ane-worker": b"worker fixture"}).returncode, 0)

    def test_host_wheel_needs_no_ane_policy(self):
        result = self.verify({"mlx/bin/mlx-omarchy-info": b"info fixture"}, "x86_64")
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
