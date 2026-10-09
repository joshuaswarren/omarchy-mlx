#!/usr/bin/env python3
"""Keep the wheel's OpenBLAS private so torch and mlx stop sharing one.

mlx's CPU backend calls cblas_sgemm/cblas_dgemm/cblas_cgemm, and on Linux
libmlx.so links the system OpenBLAS through the bare soname
``libopenblas.so.0`` (DT_NEEDED, no RUNPATH). torch CPU wheels bundle
their own ``libopenblas.so.0`` under the same soname with a different
symbol set: their libtorch_cpu.so needs ``sbgemm_``, which the Arch
system build does not define. One soname, two incompatible files: in
``import mlx.core; import torch`` the loader binds the soname to the
system copy first, and torch dies with
``libtorch_cpu.so: undefined symbol: sbgemm_``. Reverse order happens to
work because torch's copy is a superset for the three cblas symbols mlx
uses.

The vendor subcommand applies the standard auditwheel-style fix (numpy
ships its OpenBLAS the same way, as ``libscipy_openblas64_-<hash>.so``):
copy the build host's ``libopenblas.so.0`` into ``mlx/lib/`` under a
content-hashed soname, rewrite libmlx.so's NEEDED entry to that name,
and give libmlx.so a ``$ORIGIN`` RUNPATH. Nothing else in the process
uses the hashed name, so mlx and torch each keep their own BLAS and the
import order stops mattering.

The wheel's system requirements only shrink: it needed the system
openblas package before (which itself pulls the gcc runtime); it needs
only that runtime after. The PKGBUILD builds from source against system
openblas and does not use this script.

Subcommands:
  vendor WHEEL [--openblas PATH] [--patchelf BIN]
      Rewrite WHEEL in place. Needs the `patchelf` binary; the wheel is
      left untouched when libmlx.so has no bare libopenblas.so.0 NEEDED.
  check WHEEL [WHEEL ...]
      Exit nonzero if any wheel still carries the bare soname or an
      inconsistent vendored entry. Pure ELF parsing, no patchelf needed.

Both subcommands skip wheels whose libmlx.so is a Mach-O image (non-
Linux builds carry no DT_NEEDED table).
"""

import argparse
import base64
import hashlib
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

LIBMLX_MEMBER = "mlx/lib/libmlx.so"
BARE_SONAME = "libopenblas.so.0"
MACHO_MAGICS = {
    b"\xfe\xed\xfa\xce",
    b"\xfe\xed\xfa\xcf",
    b"\xce\xfa\xed\xfe",
    b"\xcf\xfa\xed\xfe",
    b"\xca\xfe\xba\xbe",
}

# ELF constants (ELF64 little-endian only; both supported wheel targets
# - aarch64 and x86_64 - use that format).
PT_LOAD = 1
PT_DYNAMIC = 2
DT_NULL = 0
DT_NEEDED = 1
DT_STRTAB = 5
DT_SONAME = 14
DT_RUNPATH = 29


def _cstr(buf, off):
    end = buf.find(b"\x00", off)
    if end < 0:
        raise ValueError("unterminated string in ELF string table")
    return buf[off:end].decode("utf-8", "replace")


def parse_elf_needed(buf):
    """Return (needed_names, runpath_or_None, soname_or_None) for ELF64 LE."""
    if buf[:4] in MACHO_MAGICS:
        return None
    if buf[:4] != b"\x7fELF":
        raise ValueError("not an ELF or Mach-O image")
    if buf[5] != 1:  # EI_DATA little-endian
        raise ValueError("only little-endian ELF is supported")
    if buf[4] != 2:  # ELFCLASS64
        raise ValueError("only ELF64 is supported")

    e_phoff = struct.unpack_from("<Q", buf, 0x20)[0]
    e_phentsize, e_phnum = struct.unpack_from("<HH", buf, 0x36)
    loads = []
    dynamic = None
    for i in range(e_phnum):
        base = e_phoff + i * e_phentsize
        p_type, _, p_offset, p_vaddr = struct.unpack_from("<IIQQ", buf, base)
        p_filesz = struct.unpack_from("<Q", buf, base + 0x20)[0]
        if p_type == PT_LOAD:
            loads.append((p_vaddr, p_offset, p_filesz))
        elif p_type == PT_DYNAMIC:
            dynamic = (p_offset, p_filesz)
    if dynamic is None:
        raise ValueError("ELF has no PT_DYNAMIC segment")

    def vaddr_to_off(vaddr):
        for seg_vaddr, seg_off, seg_filesz in loads:
            if seg_vaddr <= vaddr < seg_vaddr + seg_filesz:
                return seg_off + (vaddr - seg_vaddr)
        raise ValueError(f"ELF vaddr {vaddr:#x} not in any PT_LOAD")

    dyn_off, dyn_size = dynamic
    strtab_off = None
    entries = []
    for i in range(dyn_size // 16):
        d_tag, d_val = struct.unpack_from("<qQ", buf, dyn_off + i * 16)
        if d_tag == DT_NULL:
            break
        entries.append((d_tag, d_val))
        if d_tag == DT_STRTAB:
            strtab_off = vaddr_to_off(d_val)
    if strtab_off is None:
        raise ValueError("ELF dynamic section has no DT_STRTAB")

    needed, runpath, soname = [], None, None
    for d_tag, d_val in entries:
        if d_tag == DT_NEEDED:
            needed.append(_cstr(buf, strtab_off + d_val))
        elif d_tag == DT_RUNPATH:
            runpath = _cstr(buf, strtab_off + d_val)
        elif d_tag == DT_SONAME:
            soname = _cstr(buf, strtab_off + d_val)
    return needed, runpath, soname


def _fixed_info(info):
    """Copy a ZipInfo with a stable timestamp so rewrites are reproducible."""
    fixed = zipfile.ZipInfo(info.filename, date_time=(1980, 1, 1, 0, 0, 0))
    fixed.compress_type = zipfile.ZIP_DEFLATED
    fixed.external_attr = info.external_attr
    return fixed


def _record_name(infolist):
    records = [i.filename for i in infolist if i.filename.endswith(".dist-info/RECORD")]
    if len(records) != 1:
        raise SystemExit(f"expected exactly one dist-info/RECORD, found {records}")
    return records[0]


def _record_line(name, data):
    digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=")
    return f"{name},sha256={digest.decode('ascii')},{len(data)}\n"


def _rewrite_wheel(wheel_path, replace, additions):
    """Rewrite WHEEL with replaced members and additions; regenerate RECORD."""
    wheel_path = Path(wheel_path)
    tmp_path = wheel_path.with_name(wheel_path.name + ".tmp")
    with zipfile.ZipFile(wheel_path) as zin:
        record_name = _record_name(zin.infolist())
        lines = []
        with zipfile.ZipFile(tmp_path, "w") as zout:
            for info in zin.infolist():
                if info.filename == record_name:
                    continue
                if info.filename in replace:
                    data = replace[info.filename]
                elif info.filename in additions:
                    data = additions.pop(info.filename)
                    info = _fixed_info(info)
                else:
                    data = zin.read(info.filename)
                    info = _fixed_info(info)
                zout.writestr(info, data)
                lines.append(_record_line(info.filename, data))
            for name in sorted(additions):
                data = additions.pop(name)
                zout.writestr(_fixed_info(zipfile.ZipInfo(name)), data)
                lines.append(_record_line(name, data))
            lines.append(f"{record_name},,\n")
            zout.writestr(_fixed_info(zipfile.ZipInfo(record_name)), "".join(lines))
    os.replace(tmp_path, wheel_path)


def _run_patchelf(patchelf, args):
    result = subprocess.run(
        [patchelf, *args], capture_output=True, text=True
    )
    if result.returncode != 0:
        raise SystemExit(
            f"patchelf {' '.join(args)} failed:\n{result.stdout}{result.stderr}"
        )
    return result.stdout


def vendor(wheel_path, openblas_src, patchelf):
    wheel_path = Path(wheel_path)
    with zipfile.ZipFile(wheel_path) as zf:
        try:
            libmlx = zf.read(LIBMLX_MEMBER)
        except KeyError:
            print(f"[vendor] {wheel_path.name}: no {LIBMLX_MEMBER}, nothing to do")
            return
    parsed = parse_elf_needed(libmlx)
    if parsed is None:
        print(f"[vendor] {wheel_path.name}: libmlx.so is Mach-O, nothing to do")
        return
    needed, runpath, _ = parsed
    if BARE_SONAME not in needed:
        print(
            f"[vendor] {wheel_path.name}: libmlx.so has no bare {BARE_SONAME}"
            " NEEDED, already isolated"
        )
        return

    src = Path(openblas_src)
    if not src.is_file():
        raise SystemExit(
            f"openblas source library not found: {src}; "
            "install the distribution openblas package or pass --openblas"
        )
    src_bytes = src.read_bytes()
    short_hash = hashlib.sha256(src_bytes).hexdigest()[:10]
    vendored_name = f"libopenblas-{short_hash}.so.0"

    with tempfile.TemporaryDirectory() as work:
        work = Path(work)
        patched_mlx = work / "libmlx.so"
        patched_blas = work / vendored_name
        patched_blas.write_bytes(src_bytes)
        _run_patchelf(patchelf, ["--set-soname", vendored_name, str(patched_blas)])
        patched_mlx.write_bytes(libmlx)
        _run_patchelf(
            patchelf,
            ["--replace-needed", BARE_SONAME, vendored_name, str(patched_mlx)],
        )
        _run_patchelf(patchelf, ["--add-rpath", "$ORIGIN", str(patched_mlx)])

        # Read back through the ELF parser instead of trusting patchelf.
        blas_needed, _, blas_soname = parse_elf_needed(patched_blas.read_bytes())
        if blas_soname != vendored_name:
            raise SystemExit(f"vendored soname wrong: {blas_soname!r}")
        needed, runpath, soname = parse_elf_needed(patched_mlx.read_bytes())
        if BARE_SONAME in needed or vendored_name not in needed:
            raise SystemExit(f"patched NEEDED set wrong: {needed}")
        if not runpath or "$ORIGIN" not in runpath.split(":"):
            raise SystemExit(f"patched RUNPATH wrong: {runpath!r}")

        _rewrite_wheel(
            wheel_path,
            replace={LIBMLX_MEMBER: patched_mlx.read_bytes()},
            additions={f"mlx/lib/{vendored_name}": patched_blas.read_bytes()},
        )

    failures = check(wheel_path)
    if failures:
        raise SystemExit("vendored wheel failed its own check:\n" + "\n".join(failures))
    digest = hashlib.sha256(wheel_path.read_bytes()).hexdigest()
    print(f"[vendor] vendored {src} as mlx/lib/{vendored_name}")
    print(f"[vendor] {wheel_path}: sha256={digest} size={wheel_path.stat().st_size}")


def check(wheel_path):
    """Return a list of failure strings; empty means the wheel is isolated."""
    wheel_path = Path(wheel_path)
    with zipfile.ZipFile(wheel_path) as zf:
        names = set(zf.namelist())
        if LIBMLX_MEMBER not in names:
            return []
        try:
            parsed = parse_elf_needed(zf.read(LIBMLX_MEMBER))
        except ValueError as exc:
            return [f"{wheel_path.name}: {LIBMLX_MEMBER} unreadable: {exc}"]
        if parsed is None:
            return []
        needed, runpath, _ = parsed
        members = {n for n in names if n.startswith("mlx/lib/libopenblas-")}

    failures = []
    if BARE_SONAME in needed:
        failures.append(
            f"{wheel_path.name}: libmlx.so NEEDS bare {BARE_SONAME}; "
            "run the vendor step (scripts/build-wheel.sh) so torch and mlx "
            "do not resolve one soname to two different libraries"
        )
    vendored = [n for n in needed if n.startswith("libopenblas-")]
    for name in vendored:
        member = f"mlx/lib/{name}"
        if member not in members:
            failures.append(f"{wheel_path.name}: NEEDED {name} but {member} is not in the wheel")
    if vendored and (not runpath or "$ORIGIN" not in runpath.split(":")):
        failures.append(f"{wheel_path.name}: libmlx.so RUNPATH {runpath!r} lacks $ORIGIN")
    if not vendored and members:
        failures.append(
            f"{wheel_path.name}: carries {sorted(members)} but nothing NEEDS them"
        )
    return failures


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p_vendor = sub.add_parser("vendor", help="isolate the wheel's OpenBLAS in place")
    p_vendor.add_argument("wheel")
    p_vendor.add_argument("--openblas", default="/usr/lib/libopenblas.so.0")
    p_vendor.add_argument("--patchelf", default=shutil.which("patchelf"))
    p_check = sub.add_parser("check", help="verify wheel isolation; nonzero on failure")
    p_check.add_argument("wheels", nargs="+")
    args = parser.parse_args(argv)
    if args.command == "vendor":
        if not args.patchelf:
            raise SystemExit(
                "patchelf not found; install it (pip install patchelf works "
                "inside the build venv) or pass --patchelf"
            )
        vendor(args.wheel, args.openblas, args.patchelf)
    else:
        failures = []
        for wheel in args.wheels:
            failures.extend(check(wheel))
        if failures:
            print("\n".join(failures), file=sys.stderr)
            sys.exit(1)
        print(f"OK: {len(args.wheels)} wheel(s) isolated")


if __name__ == "__main__":
    main()
