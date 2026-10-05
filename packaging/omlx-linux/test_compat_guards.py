#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Dev-box unit tests for the oMLX-on-Linux compat layer.

Scope: hardware.py patches + AST scanner. The process_memory_enforcer
patch is exercised on the M2 inside a gpu-turn ticket, where the real
mx.device_info / sysctl behavior is testable; here we assert the patch
preserves the honest Linux message and the darwin branch.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock

OMLX_SRC = Path("/tmp/omlx-applied")
assert OMLX_SRC.is_dir(), (
    "omlx src not at /tmp/omlx-applied; from the worktree run: "
    "git clone --no-local --reference $HOME/src/omlx/.git --shared "
    "$HOME/src/omlx /tmp/omlx-applied && cd /tmp/omlx-applied && "
    "git checkout v0.7.0 && git apply $WORKTREE/packaging/omlx-linux/patches/*.patch"
)
sys.path.insert(0, str(OMLX_SRC))


def _import(name: str):
    if name in sys.modules:
        del sys.modules[name]
    return __import__(name, fromlist=["*"])


# ---------- hardware.py ----------
def test_device_total_memory_prefers_memory_size():
    h = _import("omlx.utils.hardware")
    assert h.device_total_memory({"memory_size": 5}) == 5
    assert h.device_total_memory({"total_memory": 7}) == 7
    assert h.device_total_memory({"memory_size": 0, "total_memory": 9}) == 9
    assert h.device_total_memory({}) == 0


def test_get_total_memory_linux_reads_meminfo():
    h = _import("omlx.utils.hardware")
    with mock.patch.object(sys, "platform", "linux"):
        total = h.get_total_memory_bytes()
    expected = None
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemTotal:"):
                expected = int(line.split()[1]) * 1024
                break
    assert expected is not None and total == expected and total > 0


def test_get_chip_name_linux_is_honest():
    h = _import("omlx.utils.hardware")
    with mock.patch.object(sys, "platform", "linux"):
        name = h.get_chip_name()
    # No fake Apple Silicon claim; no fake M1 parse.
    assert "Apple Silicon" not in name
    assert not name.startswith("M1")
    assert name.startswith("Unknown (") or "Apple" in name


def test_get_os_version_linux():
    h = _import("omlx.utils.hardware")
    with mock.patch.object(sys, "platform", "linux"):
        v = h.get_os_version()
    assert v.startswith("Linux ") and v != "macOS"


def test_parse_chip_info_no_fake_m1():
    h = _import("omlx.utils.hardware")
    assert h.parse_chip_info("Apple M2 Max") == ("M2", "Max")
    assert h.parse_chip_info("Apple M3") == ("M3", "")
    assert h.parse_chip_info("Apple M4 Pro") == ("M4", "Pro")
    assert h.parse_chip_info("Unknown (Linux x86_64)") == ("Unknown", "")
    assert h.parse_chip_info("") == ("Unknown", "")


def test_get_max_working_set_linux_is_three_quarters_of_total():
    h = _import("omlx.utils.hardware")
    with mock.patch.object(sys, "platform", "linux"):
        with mock.patch.object(h, "get_total_memory_bytes", return_value=100 * 1024**3):
            mws = h.get_max_working_set_bytes()
    assert mws == 75 * 1024**3


# ---------- process_memory_enforcer patch ----------
def test_enforcer_patch_logs_verbatim_honest_message():
    patch = (Path(__file__).resolve().parent / "patches"
             / "0003-linux-enforcer-no-wired-limit-log.patch").read_text()
    assert "unified system RAM" in patch
    assert "no-op" in patch
    # platform.system() — the module has no `import sys`; a sys.platform
    # check here is a NameError at server start (caught live on the M2).
    assert 'platform.system() != "Darwin"' in patch
    assert "+import platform" in patch


def test_enforcer_patch_keeps_darwin_branch():
    patch = (Path(__file__).resolve().parent / "patches"
             / "0003-linux-enforcer-no-wired-limit-log.patch").read_text()
    # The patch adds a Linux INFO block. The diff only shows +/- lines, but
    # it must preserve the surrounding darwin debug log (which still
    # mentions iogpu.wired_limit_mb) and not introduce any linux-elsewhere
    # platform hardcoding that would skip macOS.
    assert "iogpu.wired_limit_mb is" in patch  # surrounding darwin debug preserved
    assert "+import platform" in patch         # Linux branch added platform import
    assert "logger.info" in patch             # Linux adds an INFO; darwin warning untouched
    assert "logger.warning" not in patch      # the darwin warning is preserved by omission, not re-added


# ---------- scanner ----------
def test_scanner_passes_patched_tree():
    import subprocess

    r = subprocess.run(
        [sys.executable,
         str(Path(__file__).resolve().parent / "ast_import_scan.py"),
         str(OMLX_SRC / "omlx"), "--allow", "patches"],
        capture_output=True, text=True, cwd=str(Path(__file__).resolve().parent),
    )
    assert r.returncode == 0, r.stdout + r.stderr