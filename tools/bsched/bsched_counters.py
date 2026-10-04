#!/usr/bin/env python3
"""BarrierSched census/AB reader: backend trace counters via the C ABI.

Reads the process-wide MlxOmarchyTraceSnapshot (barriers_emitted /
barriers_skipped included) from the libmlx.so of the RUNNING python's mlx
package. Usage inside the bench driver: import, snapshot() before/after a
cell, emit deltas as JSON.
"""
import ctypes
import json
import os
import sys


class Snapshot(ctypes.Structure):
    _fields_ = [
        (name, ctypes.c_uint64)
        for name in (
            "gpu_primitive_dispatches",
            "vk_submissions",
            "vk_buffer_copies",
            "vk_buffer_fills",
            "vk_compute_dispatches",
            "omarchy_finalize_calls",
            "commit_calls_with_work",
            "commit_calls_noop",
            "barriers_emitted",
            "barriers_skipped",
        )
    ]


_LIB = None


def _lib():
    global _LIB
    if _LIB is None:
        import mlx.core as mc  # noqa: F401 - ensures the extension is loaded

        pkg = os.path.dirname(os.path.abspath(mc.__file__))
        so = os.path.join(pkg, "lib", "libmlx.so")
        if not os.path.exists(so):
            raise SystemExit(f"libmlx.so not found at {so}")
        _LIB = ctypes.CDLL(so)
        _LIB.mlx_omarchy_trace_snapshot_abi_size.argtypes = []
        _LIB.mlx_omarchy_trace_snapshot_abi_size.restype = ctypes.c_uint64
        expected = int(_LIB.mlx_omarchy_trace_snapshot_abi_size())
        if ctypes.sizeof(Snapshot) != expected:
            raise SystemExit(
                f"Snapshot mirror is {ctypes.sizeof(Snapshot)} bytes but "
                f"libmlx writes {expected}: stale mirror would corrupt the "
                "heap; update the field list to match trace.h"
            )
        _LIB.mlx_omarchy_trace_snapshot.argtypes = [
            ctypes.POINTER(Snapshot)
        ]
        _LIB.mlx_omarchy_trace_snapshot.restype = None
    return _LIB


def snapshot():
    snap = Snapshot()
    _lib().mlx_omarchy_trace_snapshot(ctypes.byref(snap))
    return {f[0]: getattr(snap, f[0]) for f in Snapshot._fields_}


def delta(before):
    after = snapshot()
    return {k: after[k] - before[k] for k in after}


if __name__ == "__main__":
    print(json.dumps(snapshot(), indent=1) if len(sys.argv) < 2 else "")
