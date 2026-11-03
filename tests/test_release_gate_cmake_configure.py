#!/usr/bin/env python3
"""gate_cmake_configure contract tests (CPU-only, CT-runnable, no mlx, no real cmake).

The g13 and g15 gates reconfigure a build tree with the C++ tests on. On the
aarch64 gate hosts cmake finds no BLAS or LAPACK headers by itself, so the
configure step failed with BLAS_INCLUDE_DIRS NOTFOUND and no test ran. The
include directories now come from one place, env.sh, and both gates call it.
A fake cmake on PATH records the arguments it receives.
"""
import pathlib
import subprocess
import tempfile
import unittest

REPO = pathlib.Path(__file__).resolve().parents[1]
GATES = REPO / "scripts" / "release-gates"
ENV_SH = GATES / "env.sh"


def configure(tmp: pathlib.Path, include_dir: str, extra_env: str = "") -> list:
    """Run gate_cmake_configure with a fake cmake; return the argv it saw."""
    fake_bin = tmp / "bin"
    fake_bin.mkdir()
    out = tmp / "cmake-args"
    fake = fake_bin / "cmake"
    fake.write_text(f'#!/bin/sh\nprintf "%s\\n" "$@" > "{out}"\n')
    fake.chmod(0o755)
    script = f"""
set -uo pipefail
export TAG=test-gate-cmake ASSETS_DIR={tmp}/assets GATE_ROOT={tmp}/root LOG_DIR={tmp}/root/logs
export PATH={fake_bin}:$PATH
{extra_env}
export GATE_BLAS_INCLUDE_DIR={include_dir}
. "{ENV_SH}"
gate_cmake_configure
"""
    r = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    return out.read_text().split("\n")[:-1]


class GateCmakeConfigureTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = pathlib.Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_existing_include_dir_is_passed_for_blas_and_lapack(self):
        inc = self.dir / "openblas"
        inc.mkdir()
        args = configure(self.dir, str(inc))
        self.assertIn("-DMLX_BUILD_OMARCHY=ON", args)
        self.assertIn("-DMLX_BUILD_TESTS=ON", args)
        self.assertIn(f"-DBLAS_INCLUDE_DIRS={inc}", args)
        self.assertIn(f"-DLAPACK_INCLUDE_DIRS={inc}", args)
        self.assertEqual(args[-1], ".")

    def test_missing_include_dir_leaves_cmake_to_find_its_own(self):
        args = configure(self.dir, str(self.dir / "absent"))
        self.assertEqual(args, ["-DMLX_BUILD_OMARCHY=ON", "-DMLX_BUILD_TESTS=ON", "."])

    def test_g13_and_g15_use_the_shared_configure(self):
        for name in ("g13-gdn-maskless.sh", "g15-trig.sh"):
            src = (GATES / name).read_text()
            self.assertIn("gate_cmake_configure", src, name)
            self.assertNotIn("cmake -DMLX_BUILD_OMARCHY", src, name)


if __name__ == "__main__":
    unittest.main()
