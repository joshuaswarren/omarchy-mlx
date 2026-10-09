"""Tests for scripts/openblas_isolation.py.

The published Linux wheel once carried a bare libopenblas.so.0 DT_NEEDED
with no RUNPATH, so mlx and torch resolved one soname to two different
OpenBLAS builds and the import order decided which one served both
(mlx-first broke torch with ``undefined symbol: sbgemm_``). These tests
fail against that wheel shape and pass after the vendor step.
"""

import base64
import ctypes
import hashlib
import os
import runpy
import shutil
import struct
import subprocess
import tempfile
import unittest
import zipfile
from pathlib import Path

MODULE = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "scripts/openblas_isolation.py"),
    run_name="openblas_isolation_under_test",
)

parse_elf_needed = MODULE["parse_elf_needed"]
check = MODULE["check"]
vendor = MODULE["vendor"]
_rewrite_wheel = MODULE["_rewrite_wheel"]
_record_name = MODULE["_record_name"]

DT_NULL = MODULE["DT_NULL"]
DT_NEEDED = MODULE["DT_NEEDED"]
DT_STRTAB = MODULE["DT_STRTAB"]
DT_SONAME = MODULE["DT_SONAME"]
DT_RUNPATH = MODULE["DT_RUNPATH"]
BARE_SONAME = MODULE["BARE_SONAME"]
LIBMLX_MEMBER = MODULE["LIBMLX_MEMBER"]

RECORD_NAME = "mlx_omarchy-0.32.4.dist-info/RECORD"
VENDORED = "libopenblas-deadbeef12.so.0"


def elf64(needed=(), runpath=None, soname=None, machine=183):
    """Build a minimal ELF64 LE shared object with a PT_DYNAMIC table.

    String-table vaddrs equal file offsets (single PT_LOAD at 0), which is
    all the parser relies on.
    """
    strs = bytearray(b"\x00")

    def add(text):
        off = len(strs)
        strs.extend(text.encode() + b"\x00")
        return off

    needed_offs = [add(n) for n in needed]
    soname_off = add(soname) if soname else None
    runpath_off = add(runpath) if runpath else None

    dyn = []
    for off in needed_offs:
        dyn.append((DT_NEEDED, off))
    if soname_off is not None:
        dyn.append((DT_SONAME, soname_off))
    if runpath_off is not None:
        dyn.append((DT_RUNPATH, runpath_off))
    dyn.append((DT_STRTAB, 0))
    dyn.append((DT_NULL, 0))

    ehdr_size, phdr_size, dynent = 64, 56, 16
    dyn_off = ehdr_size + 2 * phdr_size
    strtab_off = dyn_off + len(dyn) * dynent
    dyn[dyn.index((DT_STRTAB, 0))] = (DT_STRTAB, strtab_off)

    buf = bytearray(struct.pack(
        "<16sHHIQQQIHHHHHH",
        b"\x7fELF" + bytes([2, 1, 1, 0]) + bytes(8),
        3, machine, 1, 0, ehdr_size, 0, 0,
        ehdr_size, phdr_size, 2, phdr_size, 0, 0,
    ))
    seg_size = dyn_off + len(dyn) * dynent + len(strs)
    buf += struct.pack("<IIQQQQQQ", 1, 5, 0, 0, 0, seg_size, seg_size, 0x1000)
    buf += struct.pack("<IIQQQQQQ", 2, 6, dyn_off, dyn_off, dyn_off, len(dyn) * dynent, len(dyn) * dynent, 8)
    for tag, val in dyn:
        buf += struct.pack("<qQ", tag, val)
    buf += strs
    return bytes(buf)


class IsolationTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)

    def make_wheel(self, libmlx_bytes, extra=None):
        wheel = Path(self._tmp.name) / "mlx_omarchy-0.32.4-cp314-cp314-linux_aarch64.whl"
        with zipfile.ZipFile(wheel, "w") as zf:
            zf.writestr(LIBMLX_MEMBER, libmlx_bytes)
            for name, data in (extra or {}).items():
                zf.writestr(name, data)
            zf.writestr(RECORD_NAME, f"{RECORD_NAME},,\n")
        return wheel


def record_is_valid(wheel):
    with zipfile.ZipFile(wheel) as zf:
        record = _record_name(zf.infolist())
        lines = zf.read(record).decode().splitlines()
        listed = {}
        for line in lines:
            name, digest, size = line.split(",")
            if not digest:
                continue
            assert digest.startswith("sha256="), line
            listed[name] = (digest[len("sha256="):], int(size))
        names = set(zf.namelist()) - {record}
        if set(listed) != names:
            return False
        for name, (digest, size) in listed.items():
            data = zf.read(name)
            want = base64.urlsafe_b64encode(
                hashlib.sha256(data).digest()
            ).rstrip(b"=").decode()
            if digest != want or size != len(data):
                return False
    return True


class ElfParseTests(unittest.TestCase):
    def test_bare_needed_without_runpath(self):
        needed, runpath, soname = parse_elf_needed(elf64([BARE_SONAME]))
        self.assertEqual(needed, [BARE_SONAME])
        self.assertIsNone(runpath)
        self.assertIsNone(soname)

    def test_needed_runpath_and_soname(self):
        buf = elf64([BARE_SONAME, VENDORED], runpath="$ORIGIN", soname="libmlx.so")
        needed, runpath, soname = parse_elf_needed(buf)
        self.assertEqual(needed, [BARE_SONAME, VENDORED])
        self.assertEqual(runpath, "$ORIGIN")
        self.assertEqual(soname, "libmlx.so")

    def test_rejects_garbage(self):
        with self.assertRaises(ValueError):
            parse_elf_needed(b"not an elf at all")


class CheckTests(IsolationTest):
    def test_bare_soname_fails(self):
        failures = check(self.make_wheel(elf64([BARE_SONAME])))
        self.assertEqual(len(failures), 1)
        self.assertIn(BARE_SONAME, failures[0])

    def test_vendored_wheel_passes(self):
        buf = elf64([VENDORED], runpath="$ORIGIN")
        failures = check(
            self.make_wheel(buf, extra={f"mlx/lib/{VENDORED}": b"\x7fELF-vendored"})
        )
        self.assertEqual(failures, [])

    def test_missing_vendored_member_fails(self):
        buf = elf64([VENDORED], runpath="$ORIGIN")
        failures = check(self.make_wheel(buf))
        self.assertEqual(len(failures), 1)
        self.assertIn(VENDORED, failures[0])

    def test_missing_runpath_fails(self):
        buf = elf64([VENDORED])
        failures = check(
            self.make_wheel(buf, extra={f"mlx/lib/{VENDORED}": b"\x7fELF-vendored"})
        )
        self.assertEqual(len(failures), 1)
        self.assertIn("$ORIGIN", failures[0])

    def test_orphan_vendored_member_fails(self):
        failures = check(
            self.make_wheel(
                elf64([BARE_SONAME]),
                extra={f"mlx/lib/{VENDORED}": b"\x7fELF-vendored"},
            )
        )
        self.assertEqual(len(failures), 2)

    def test_macho_libmlx_is_skipped(self):
        macho = b"\xcf\xfa\xed\xfe" + bytes(64)
        self.assertEqual(check(self.make_wheel(macho)), [])

    def test_wheel_without_libmlx_is_skipped(self):
        with tempfile.TemporaryDirectory() as directory:
            wheel = Path(directory) / "mlx_omarchy-0.1-cp314-none-any.whl"
            with zipfile.ZipFile(wheel, "w") as zf:
                zf.writestr(RECORD_NAME, f"{RECORD_NAME},,\n")
            self.assertEqual(check(wheel), [])


class RewriteWheelTests(IsolationTest):
    def test_rewrite_replaces_adds_and_regenerates_record(self):
        wheel = self.make_wheel(elf64([BARE_SONAME]), extra={"mlx/share/data.bin": b"123"})
        vendored = b"\x7fELF" + bytes(32)
        patched = elf64([VENDORED], runpath="$ORIGIN")
        _rewrite_wheel(
            wheel,
            replace={LIBMLX_MEMBER: patched},
            additions={f"mlx/lib/{VENDORED}": vendored},
        )
        self.assertTrue(record_is_valid(wheel))
        with zipfile.ZipFile(wheel) as zf:
            self.assertEqual(zf.read(LIBMLX_MEMBER), patched)
            self.assertEqual(zf.read(f"mlx/lib/{VENDORED}"), vendored)
            self.assertIn("mlx/share/data.bin", zf.namelist())


@unittest.skipUnless(
    shutil.which("gcc") and shutil.which("patchelf"),
    "needs gcc and patchelf on PATH",
)
class VendorEndToEndTests(unittest.TestCase):
    def compile_fixture(self, work):
        noop = work / "noop.c"
        noop.write_text("void mlxnoop(void) {}\n")
        blas = work / BARE_SONAME
        libmlx = work / "libmlx.so"
        subprocess.run(
            ["gcc", "-shared", "-fPIC",
             f"-Wl,-soname,{BARE_SONAME}", "-o", str(blas), str(noop)],
            check=True,
        )
        subprocess.run(
            ["gcc", "-shared", "-fPIC", "-Wl,--no-as-needed",
             f"-Wl,-rpath-link,{work}", "-l:libopenblas.so.0",
             "-o", str(libmlx), str(noop)],
            check=True, env=dict(os.environ, LIBRARY_PATH=str(work)),
        )
        return blas.read_bytes(), libmlx.read_bytes()

    def test_vendor_isolates_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            blas_bytes, libmlx_bytes = self.compile_fixture(work)
            wheel = work / "mlx_omarchy-0.32.4-cp314-cp314-linux_aarch64.whl"
            with zipfile.ZipFile(wheel, "w") as zf:
                zf.writestr(LIBMLX_MEMBER, libmlx_bytes)
                zf.writestr(RECORD_NAME, f"{RECORD_NAME},,\n")

            self.assertNotEqual(check(wheel), [])

            expected_hash = hashlib.sha256(blas_bytes).hexdigest()[:10]
            expected_name = f"mlx/lib/libopenblas-{expected_hash}.so.0"
            vendor(wheel, blas_bytes and str(work / BARE_SONAME), shutil.which("patchelf"))
            self.assertEqual(check(wheel), [])
            self.assertTrue(record_is_valid(wheel))
            with zipfile.ZipFile(wheel) as zf:
                self.assertIn(expected_name, zf.namelist())
                needed, runpath, _ = parse_elf_needed(zf.read(LIBMLX_MEMBER))
                self.assertNotIn(BARE_SONAME, needed)
                self.assertIn(f"libopenblas-{expected_hash}.so.0", needed)
                self.assertIn("$ORIGIN", runpath.split(":"))

            # The patched pair loads: $ORIGIN resolves the private BLAS.
            load_dir = work / "install" / "mlx" / "lib"
            load_dir.mkdir(parents=True)
            with zipfile.ZipFile(wheel) as zf:
                for name in zf.namelist():
                    if name.startswith("mlx/lib/"):
                        (load_dir / Path(name).name).write_bytes(zf.read(name))
            ctypes.CDLL(str(load_dir / "libmlx.so"), mode=ctypes.RTLD_LOCAL)

            before = wheel.read_bytes()
            vendor(wheel, str(work / BARE_SONAME), shutil.which("patchelf"))
            self.assertEqual(wheel.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
