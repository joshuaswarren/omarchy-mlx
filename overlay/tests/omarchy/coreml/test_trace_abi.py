# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Regression tests for the MlxOmarchyTraceSnapshot ctypes ABI.

2026-10-04: the wave-scheduler commit grew the C struct by two uint64
fields without updating the shipped ctypes mirrors; every packaged
trace_snapshot() read then overflowed 16 bytes past the mirror buffer
and the parakeet worker died at its first GC pass (PyObject_GC_Track
'object already tracked', or SIGSEGV in _PyObject_GC_New). These tests
pin the mirror to the header (always runnable) and to the loaded
libmlx via the mlx_omarchy_trace_snapshot_abi_size probe with a GC
churn pass under PYTHONMALLOC=debug (runs wherever a libmlx is
resolvable; g7c runs the same flow in the packaged venv).
"""

import os
import re
import subprocess
import sys
import unittest
from pathlib import Path

try:
    from _bootstrap import _TOOLS  # plain-script execution
except ImportError:  # package discovery (omarchy.coreml.*)
    from ._bootstrap import _TOOLS  # noqa: F401

from coreml import trace_abi

_REPO = Path(__file__).resolve().parents[4]
_HEADER = _REPO / "overlay" / "mlx" / "backend" / "omarchy" / "trace.h"
_COREML_DIR = Path(_TOOLS) / "coreml"

_FLOW = """
import ctypes, gc, os, sys
sys.path.insert(0, os.environ["TRACE_ABI_COREML"])
from trace_abi import TraceSnapshot, trace_snapshot, library_path
lib = ctypes.CDLL(library_path())
probe = lib.mlx_omarchy_trace_snapshot_abi_size
probe.argtypes = []
probe.restype = ctypes.c_uint64
size = int(probe())
mirror = ctypes.sizeof(TraceSnapshot)
if size != mirror:
    raise SystemExit(f"ABI drift: libmlx writes {size} bytes, mirror is {mirror}")
try:
    import mlx.core as mx
except ImportError:
    mx = None
before = trace_snapshot()
arrays = []
if mx is not None:
    arrays = [mx.ones((64, 64), dtype=mx.float32) for _ in range(8)]
    arrays.append(arrays[0] + arrays[1])
ballast = [[{"k": j} for j in range(8)] for _ in range(64)]
for _ in range(5):
    gc.collect()
after = trace_snapshot()
del arrays, ballast
print("OK", size, after["gpu_primitive_dispatches"] - before["gpu_primitive_dispatches"])
"""


class MirrorMatchesHeaderTest(unittest.TestCase):
    def test_mirror_field_list_matches_trace_header(self):
        text = _HEADER.read_text()
        block = re.search(
            r"struct MlxOmarchyTraceSnapshot \{(.*?)\};", text, re.S
        )
        self.assertIsNotNone(block, "MlxOmarchyTraceSnapshot struct not found")
        header_fields = tuple(re.findall(r"uint64_t (\w+);", block.group(1)))
        self.assertEqual(
            header_fields,
            tuple(trace_abi._FIELDS),
            "shipped ctypes mirror drifted from trace.h; every read would "
            "overflow the mirror buffer (2026-10-04 GC double-track crash)",
        )


class PackagedFlowGcTest(unittest.TestCase):
    def _libmlx(self):
        override = os.environ.get("MLX_OMARCHY_LIBMLX")
        if override:
            return Path(override)
        try:
            import mlx.core as mx
        except ImportError:
            return None
        return Path(mx.__file__).parent / "lib" / "libmlx.so"

    def test_snapshot_read_survives_gc_with_abi_probe(self):
        so = self._libmlx()
        if so is None or not so.is_file():
            self.skipTest(
                "no libmlx resolvable (install the wheel or set "
                "MLX_OMARCHY_LIBMLX); g7c runs this flow in the packaged venv"
            )
        env = dict(os.environ)
        env["PYTHONMALLOC"] = "debug"
        env["MLX_OMARCHY_LIBMLX"] = str(so)
        env["MLX_OMARCHY_WAVE_SCHED"] = "1"
        env["TRACE_ABI_COREML"] = str(_COREML_DIR)
        run = subprocess.run(
            [sys.executable, "-c", _FLOW],
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )
        self.assertEqual(
            run.returncode,
            0,
            f"packaged-flow snapshot read corrupted the heap or the ABI "
            f"guard fired\nstdout:\n{run.stdout}\nstderr:\n{run.stderr}",
        )
        self.assertTrue(run.stdout.startswith("OK "), run.stdout)


if __name__ == "__main__":
    unittest.main()
