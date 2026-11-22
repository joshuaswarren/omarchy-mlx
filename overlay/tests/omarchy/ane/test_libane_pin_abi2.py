#!/usr/bin/env python3
"""Failing gate: the pinned parakeet libane must open the ANE at dev_id 0.

Defect (jwm2, T6021, kernel 7.3.0-2-20261110.1-lab-ARCH, omarchy-mlx
0.7.49-2): the loaded ane_t6021 driver reports DRM major 2
(ANE_ABI_M2_MAJOR, omarchy-ane 27e996a, 2026-09-29) with driver name
"ane", but the pinned libane-strict.so (d06222a8, built from omarchy-ane
df5c553, 2026-09-22 — the pin has never moved) accepts only
version_major == ANE_ABI_MAJOR (1). is_ane_device() therefore rejects
/dev/accel/accel0 and the parakeet worker dies with
"failed to find device with dev_id 0" while kernel-side rows keep
succeeding. The test pins that gate only: the wheel ships H13-layout
bundles and the ABI-2 M2 path refuses H13 headers, so full-init green
needs an H14 anec passed via LIBANE_PIN_MODEL. M1-family hosts (ABI 1) stay green by design; the test goes
green on T6021 only once the pin is rebuilt from an omarchy-ane commit
with the ABI-2 backend (>= 8a4379e) and the pin json is moved in the
same change.

Run: python3 overlay/tests/omarchy/ane/test_libane_pin_abi2.py
Or against an installed tree:
  LIBANE_PIN_SHARE=/usr/lib/omarchy-mlx/venv/lib/python3.14/site-packages/\
mlx/share/mlx-omarchy/parakeet-1 python3 test_libane_pin_abi2.py
"""

import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SHARE = os.environ.get("LIBANE_PIN_SHARE")
if SHARE:
    SHARE = Path(SHARE)
else:
    _here = Path(__file__).resolve()
    _repo = _here.parents[4] if len(_here.parents) > 4 else _here.parents[-1]
    SHARE = _repo / ("overlay/tools/mlx-omarchy-parakeet/share/mlx-omarchy/"
                     "parakeet-1")

SO = SHARE / "libane/libane-strict.so"
PIN = SHARE / "parakeet-runtime-pin.json"
MODEL = SHARE / "bundles/island-pv/program-0.anec"
# island-pv is an H13 anec; ABI-2 (T6021) opens only H14 programs, so on an
# H14 host the pin's whole-encoder bundle is the model under test.
try:
    _compat = Path("/proc/device-tree/compatible").read_bytes().decode(
        "utf-8", "replace")
except OSError:
    _compat = ""
if any(f"t602{n}" in _compat for n in (0, 1, 2)) and \
        (SHARE / "bundles/parakeet-encoder-whole/program-0.anec").is_file():
    MODEL = SHARE / "bundles/parakeet-encoder-whole/program-0.anec"
ACCEL = Path("/dev/accel/accel0")

# e_machine values; the pinned libane is aarch64-only, so x86 CI skips
ELF_MACHINES = {0xB7: "aarch64", 0x3E: "x86_64"}


def elf_machine(path):
    with open(path, "rb") as f:
        return ELF_MACHINES.get(int.from_bytes(f.read(20)[18:20], "little"))


CHILD = r"""
import ctypes, sys
lib = ctypes.CDLL(sys.argv[1])
lib.__ane_init.restype = ctypes.c_void_p
lib.__ane_init.argtypes = [ctypes.c_char_p, ctypes.c_int]
print("handle=%d" % (lib.__ane_init(sys.argv[2].encode(), 0) or 0))
"""


def probe(model):
    """__ane_init(model, 0) the way the worker calls it.

    Returns (handle, stderr)."""
    env = {"PATH": "/usr/bin:/bin"}
    child = subprocess.run(
        [sys.executable, "-c", CHILD, str(SO), str(model)],
        env=env, capture_output=True, text=True, timeout=120)
    handle = 0
    match = re.search(r"handle=(\d+)", child.stdout)
    if match:
        handle = int(match.group(1))
    return handle, child.stderr


class TestLibanePinAbi2(unittest.TestCase):
    def test_pin_matches_vendored_bytes(self):
        pin = json.loads(PIN.read_text())
        self.assertEqual(
            hashlib.sha256(SO.read_bytes()).hexdigest(),
            pin["assets"]["libane"]["libane-strict.so"],
            "the .so under test is not the bytes the pin names")

    def test_pinned_libane_opens_dev0(self):
        so_mach = elf_machine(SO)
        host_mach = elf_machine("/proc/self/exe")
        if so_mach != host_mach:
            self.skipTest(
                f"pinned libane is {so_mach}; run on an {so_mach} host")
        if not ACCEL.exists():
            self.skipTest("no /dev/accel/accel0 on this host")
        model = MODEL
        override = os.environ.get("LIBANE_PIN_MODEL")
        if override:
            model = Path(override)
        handle, stderr = probe(model)
        self.assertNotIn("failed to find device with dev_id 0", stderr,
                         "pinned libane rejects the local ANE driver "
                         "(ABI/name gate); the pin predates ABI 2")
        if not override:
            # The wheel ships only H13-layout bundles (island-pv,
            # parakeet-encoder-whole) and omarchy-ane's M2 path
            # (libane/ane_m2.c) refuses H13 headers once the gate is
            # open, so full __ane_init green on T6021 needs an H14 anec.
            # This test pins the dev_id-0 ABI gate; set LIBANE_PIN_MODEL
            # to an H14 anec (w73's staged M2 h14 kit, or a fresh
            # hwxv2-to-anec compile) for the end-to-end green.
            return
        self.assertNotEqual(handle, 0,
                            "__ane_init(H14 model, 0) failed:\n" + stderr)


if __name__ == "__main__":
    unittest.main()
