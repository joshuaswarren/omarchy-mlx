"""Issue 537 defect 4: a wheel built on a libdir-lib64 host must not leak.

A wheel assembled from the release tarball tree on a Fedora-style host
put libmlx.so in mlx/lib64/ while core.so's RUNPATH is $ORIGIN/lib, so
import mlx.core could not find the library. The wheel build pins the
CMake install dir to lib and refuses a wheel whose libmlx.so is not in
mlx/lib/.
"""
import runpy
import subprocess
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

MODULE = runpy.run_path(
    str(ROOT / "scripts/openblas_isolation.py"),
    run_name="openblas_isolation_under_test",
)
check_layout = MODULE["check_layout"]

ELF_BYTES = b"\x7fELF" + b"\x00" * 64


def make_wheel(tmp_path, members):
    wheel = tmp_path / "probe.whl"
    with zipfile.ZipFile(wheel, "w") as zf:
        for name, payload in members.items():
            zf.writestr(name, payload)
    return wheel


def test_lib_in_lib_passes(tmp_path):
    wheel = make_wheel(tmp_path, {
        "mlx/lib/libmlx.so": ELF_BYTES,
        "mlx/lib/core.so": b"",
    })
    assert check_layout(wheel) == []


def test_lib64_member_fails(tmp_path):
    wheel = make_wheel(tmp_path, {
        "mlx/lib64/libmlx.so": ELF_BYTES,
        "mlx/lib/core.so": b"",
    })
    failures = check_layout(wheel)
    assert failures, "a lib64 libmlx.so must be refused"
    assert any("lib64" in f for f in failures)


def test_missing_libmlx_fails(tmp_path):
    wheel = make_wheel(tmp_path, {"mlx/lib/core.so": b""})
    failures = check_layout(wheel)
    assert failures, "a wheel without mlx/lib/libmlx.so must be refused"
    assert any("mlx/lib/libmlx.so" in f for f in failures)


def test_cli_check_layout_refuses(tmp_path):
    wheel = make_wheel(tmp_path, {
        "mlx/lib64/libmlx.so": ELF_BYTES,
        "mlx/lib/core.so": b"",
    })
    with pytest.raises(SystemExit) as exc:
        MODULE["main"](["check-layout", str(wheel)])
    assert exc.value.code != 0


def test_cmake_args_pin_libdir():
    script = ROOT / "scripts/wheel-cmake-args.sh"
    proc = subprocess.run(
        ["bash", "-c", f'source "{script}" && printf "%s\\n" "$CMAKE_ARGS"'],
        capture_output=True, text=True, check=True)
    assert "-DCMAKE_INSTALL_LIBDIR=lib" in proc.stdout, (
        "the wheel build must pin the CMake install dir to lib so libmlx.so "
        "lands beside core.so on every host libdir convention")
