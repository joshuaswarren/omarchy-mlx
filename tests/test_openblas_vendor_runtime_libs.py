"""Tests that the vendored OpenBLAS carries its gcc runtime libraries.

The aarch64 wheel shipped the content-hashed OpenBLAS but not the
libgfortran.so.5 / libgomp.so.1 it NEEDS, so ``import mlx.core`` failed
on containers without the gcc runtime installed. These tests fail
against that wheel shape and pass once vendor() ships the runtime
libraries beside the hashed OpenBLAS and check() verifies them.
"""

import importlib.util
import tempfile
import unittest
import zipfile
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "test_openblas_isolation",
    Path(__file__).resolve().parent / "test_openblas_isolation.py",
)
_sibling = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_sibling)
MODULE, elf64, check = _sibling.MODULE, _sibling.elf64, _sibling.check

LIBMLX_MEMBER = MODULE["LIBMLX_MEMBER"]
VENDORED = "libopenblas-deadbeef12.so.0"
GFORTRAN = "libgfortran.so.5"
GOMP = "libgomp.so.1"

RECORD_NAME = "mlx_omarchy-0.32.4.dist-info/RECORD"


class RuntimeLibTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)

    def make_wheel(self, blas_deps, members):
        """Wheel whose libmlx NEEDS the hashed OpenBLAS with $ORIGIN RUNPATH."""
        libmlx = elf64([VENDORED], runpath="$ORIGIN")
        wheel = Path(self._tmp.name) / "mlx_omarchy-0.32.4-cp314-cp314-linux_aarch64.whl"
        with zipfile.ZipFile(wheel, "w") as zf:
            zf.writestr(LIBMLX_MEMBER, libmlx)
            for name, data in members.items():
                zf.writestr(name, data)
            for text in ("COPYING3", "COPYING.RUNTIME", "COPYING.LIB"):
                zf.writestr(f"mlx/lib/licences/{text}", text.encode())
            zf.writestr(RECORD_NAME, f"{RECORD_NAME},,\n")
        return wheel

    @staticmethod
    def blas_needing(deps):
        return elf64(deps, runpath="$ORIGIN", soname=VENDORED)

    @staticmethod
    def elf_member():
        return elf64([], runpath="$ORIGIN")

    def test_check_fails_when_runtime_libs_missing(self):
        wheel = self.make_wheel(
            [GFORTRAN, GOMP], {f"mlx/lib/{VENDORED}": self.blas_needing([GFORTRAN, GOMP])}
        )
        failures = check(wheel)
        self.assertEqual(len(failures), 2, failures)
        joined = "\n".join(failures)
        self.assertIn(GFORTRAN, joined)
        self.assertIn(GOMP, joined)

    def test_check_fails_when_one_runtime_lib_missing(self):
        wheel = self.make_wheel(
            [GFORTRAN, GOMP],
            {
                f"mlx/lib/{VENDORED}": self.blas_needing([GFORTRAN, GOMP]),
                f"mlx/lib/{GOMP}": self.elf_member(),
            },
        )
        failures = check(wheel)
        self.assertEqual(len(failures), 1, failures)
        self.assertIn(GFORTRAN, failures[0])

    def test_check_passes_when_runtime_libs_present(self):
        wheel = self.make_wheel(
            [GFORTRAN, GOMP],
            {
                f"mlx/lib/{VENDORED}": self.blas_needing([GFORTRAN, GOMP]),
                f"mlx/lib/{GFORTRAN}": self.elf_member(),
                f"mlx/lib/{GOMP}": self.elf_member(),
            },
        )
        self.assertEqual(check(wheel), [])

    def test_check_walks_transitive_needed_chain_and_requires_each_runpath(self):
        wheel = self.make_wheel(
            [GFORTRAN],
            {
                f"mlx/lib/{VENDORED}": self.blas_needing([GFORTRAN]),
                f"mlx/lib/{GFORTRAN}": elf64(["libquadmath.so.0"], runpath="$ORIGIN"),
            },
        )
        failures = check(wheel)
        self.assertTrue(any("libquadmath.so.0" in failure for failure in failures), failures)

    def test_check_requires_correct_gcc_runtime_licence(self):
        wheel = self.make_wheel(
            ["libquadmath.so.0"],
            {
                f"mlx/lib/{VENDORED}": self.blas_needing(["libquadmath.so.0"]),
                "mlx/lib/libquadmath.so.0": self.elf_member(),
            },
        )
        # The quadmath requirement is LGPL (COPYING.LIB), not GPL+runtime exception.
        rewritten = wheel.with_suffix(".rewritten.whl")
        with zipfile.ZipFile(wheel) as source, zipfile.ZipFile(rewritten, "w") as target:
            for info in source.infolist():
                if info.filename != "mlx/lib/licences/COPYING.LIB":
                    target.writestr(info, source.read(info.filename))
        failures = check(rewritten)
        self.assertTrue(any("COPYING.LIB" in failure for failure in failures), failures)

    def test_check_passes_when_blas_needs_only_glibc(self):
        wheel = self.make_wheel(
            ["libm.so.6", "libc.so.6"],
            {f"mlx/lib/{VENDORED}": self.blas_needing(["libm.so.6", "libc.so.6"])},
        )
        self.assertEqual(check(wheel), [])

    def test_runtime_libraries_filters_glibc(self):
        runtime_libraries = MODULE["runtime_libraries"]
        self.assertEqual(
            runtime_libraries([GFORTRAN, GOMP, "libm.so.6", "libc.so.6",
                               "ld-linux-aarch64.so.1"]),
            [GFORTRAN, GOMP],
        )

    def test_vendor_additions_copies_from_openblas_source_dir(self):
        vendor_runtime_additions = MODULE["vendor_runtime_additions"]
        src_dir = Path(self._tmp.name)
        (src_dir / GFORTRAN).write_bytes(elf64(["libquadmath.so.0"]))
        (src_dir / "libquadmath.so.0").write_bytes(elf64([], runpath="$ORIGIN"))
        license_dir = src_dir / "licences"
        license_dir.mkdir()
        for text in ("COPYING3", "COPYING.RUNTIME", "COPYING.LIB"):
            (license_dir / text).write_text(text)
        additions = vendor_runtime_additions(
            src_dir / "libopenblas.so.0", [GFORTRAN, "libm.so.6"]
        )
        self.assertEqual(additions[f"mlx/lib/{GFORTRAN}"], elf64(["libquadmath.so.0"]))
        self.assertIn("mlx/lib/libquadmath.so.0", additions)
        self.assertIn("mlx/lib/licences/COPYING.RUNTIME", additions)

    def test_vendor_additions_errors_when_runtime_lib_missing(self):
        vendor_runtime_additions = MODULE["vendor_runtime_additions"]
        with self.assertRaises(SystemExit) as ctx:
            vendor_runtime_additions(
                Path(self._tmp.name) / "libopenblas.so.0", [GFORTRAN]
            )
        self.assertIn(GFORTRAN, str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
