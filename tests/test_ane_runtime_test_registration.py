"""Configure-only checks for ANE runtime test registration in CMake/CTest."""
import json
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OMARCHY_TESTS_DIR = ROOT / "overlay/tests/omarchy"
SIBLING_TARGETS = {
    "omarchy_ane_bundle_tests",
    "omarchy_ane_worker_tests",
    "omarchy_ane_tile_layout_tests",
    "omarchy_runtime_tests",
}


@unittest.skipUnless(
    shutil.which("cmake") and shutil.which("ctest"),
    "requires cmake and ctest",
)
class AneRuntimeTestRegistrationTests(unittest.TestCase):
    @classmethod
    def _configure_omarchy_tests(cls, *, with_ane_worker):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            build = root / "build"
            query = build / ".cmake/api/v1/query/codemodel-v2"
            query.parent.mkdir(parents=True)
            query.touch()

            worker_decl = (
                "add_custom_target(mlx-omarchy-ane-worker)\n"
                if with_ane_worker
                else ""
            )
            (root / "CMakeLists.txt").write_text(
                "cmake_minimum_required(VERSION 3.25)\n"
                "project(ane_runtime_test_registration LANGUAGES CXX)\n"
                "enable_testing()\n"
                "add_library(mlx INTERFACE)\n"
                "add_library(doctest INTERFACE)\n"
                "function(doctest_discover_tests target)\n"
                "  if(NOT TARGET \"${target}\")\n"
                "    message(FATAL_ERROR \"doctest_discover_tests called on missing target: ${target}\")\n"
                "  endif()\n"
                "endfunction()\n"
                f"{worker_decl}"
                f'add_subdirectory("{OMARCHY_TESTS_DIR.as_posix()}" omarchy_tests)\n'
            )

            configure = subprocess.run(
                ["cmake", "-S", str(root), "-B", str(build)],
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
            if configure.returncode != 0:
                raise AssertionError(configure.stdout + configure.stderr)

            reply_dir = build / ".cmake/api/v1/reply"
            indices = sorted(reply_dir.glob("index-*.json"))
            if not indices:
                raise AssertionError("CMake File API index not generated")
            index = json.loads(indices[-1].read_text())
            codemodel_file = next(
                obj["jsonFile"]
                for obj in index["objects"]
                if obj["kind"] == "codemodel"
            )
            codemodel = json.loads((reply_dir / codemodel_file).read_text())
            targets = {
                target["name"]
                for config in codemodel["configurations"]
                for target in config["targets"]
            }

            ctest = subprocess.run(
                ["ctest", "--test-dir", str(build), "-N"],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            if ctest.returncode != 0:
                raise AssertionError(ctest.stdout + ctest.stderr)
            ctest_names = set(
                re.findall(
                    r"^\s*Test\s+#\d+:\s+(.+?)\s*$",
                    ctest.stdout,
                    re.MULTILINE,
                )
            )
            return targets, ctest_names

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.off_targets, cls.off_tests = cls._configure_omarchy_tests(
            with_ane_worker=False
        )
        cls.on_targets, cls.on_tests = cls._configure_omarchy_tests(
            with_ane_worker=True
        )

    def test_ane_off_omits_runtime_test_and_keeps_siblings(self):
        self.assertNotIn("omarchy_ane_runtime_tests", self.off_targets)
        self.assertNotIn("mlx-omarchy-ane-smoke", self.off_targets)
        self.assertNotIn("omarchy_ane_runtime_tests", self.off_tests)
        self.assertTrue(
            SIBLING_TARGETS.issubset(self.off_targets),
            f"missing sibling CMake targets: {sorted(SIBLING_TARGETS - self.off_targets)}",
        )
        self.assertTrue(
            SIBLING_TARGETS.issubset(self.off_tests),
            f"missing sibling CTest entries: {sorted(SIBLING_TARGETS - self.off_tests)}",
        )

    def test_ane_enabled_registers_runtime_test_and_smoke_only(self):
        self.assertEqual(
            self.on_targets,
            self.off_targets
            | {
                "omarchy_ane_runtime_tests",
                "mlx-omarchy-ane-smoke",
                "mlx-omarchy-ane-worker",
            },
        )
        self.assertEqual(
            self.on_tests,
            self.off_tests | {"omarchy_ane_runtime_tests"},
        )


if __name__ == "__main__":
    unittest.main()
