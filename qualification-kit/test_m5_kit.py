import json
import os
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
SUITE = HERE / "scripts" / "geometry_suite.sh"
KIT = HERE / "scripts" / "m5_linux_mlx_kit.sh"


def run_suite(out, binary=None):
    env = dict(os.environ)
    env.pop("OMARCHY_QMM_GEOMETRY_TESTS", None)
    if binary is not None:
        env["OMARCHY_QMM_GEOMETRY_TESTS"] = str(binary)
    proc = subprocess.run(["bash", str(SUITE), str(out)], env=env, capture_output=True, text=True)
    return proc, json.loads((Path(out) / "geometry.json").read_text())


def stub(directory, code):
    path = Path(directory) / "omarchy_qmm_geometry_tests"
    path.write_text(f"#!/bin/sh\necho 'stub geometry run'\nexit {code}\n")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


class GeometrySuiteTest(unittest.TestCase):
    def test_wheel_only_install_skips_with_reason(self):
        with tempfile.TemporaryDirectory() as out:
            proc, record = run_suite(out)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertEqual(record["status"], "skip")
            self.assertIn("wheels do not ship", record["detail"]["reason"])

    def test_missing_binary_path_skips(self):
        with tempfile.TemporaryDirectory() as out:
            proc, record = run_suite(out, Path(out) / "absent")
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertEqual(record["status"], "skip")

    def test_binary_pass_and_fail_are_reported(self):
        with tempfile.TemporaryDirectory() as out:
            proc, record = run_suite(out, stub(out, 0))
            self.assertEqual((proc.returncode, record["status"]), (0, "pass"))
            self.assertEqual(record["detail"]["summary"], "stub geometry run")
        with tempfile.TemporaryDirectory() as out:
            proc, record = run_suite(out, stub(out, 1))
            self.assertEqual((proc.returncode, record["status"]), (1, "fail"))

    def test_kit_uses_the_suite_not_a_python_module(self):
        kit = KIT.read_text()
        self.assertIn('geometry_suite.sh" "$OUT"', kit)
        self.assertNotIn("--pyargs", kit)
        self.assertIn('[ "$2" = skip ]', kit)


if __name__ == "__main__":
    unittest.main()
