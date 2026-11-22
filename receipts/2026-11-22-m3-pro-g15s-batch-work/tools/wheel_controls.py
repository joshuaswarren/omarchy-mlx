#!/usr/bin/env python3
"""Wheel controls for docs/parity-contract.md 2.1, one control per process.

Usage: wheel_controls.py gpu-add|cpu-negative|reopen EXPECT_WHEEL REPO_ROOT

Every mode prints the verified provenance line first and exits non-zero
on any unexpected result.
  gpu-add       adds two 1024-element float32 arrays on the default device;
                requires the exact sum and a positive vk_compute_dispatches
                delta across the add (inputs are evaluated before the first
                counter snapshot).
  cpu-negative  adds two float32 values on a new explicit CPU stream;
                requires exactly "Add has no CPU implementation.".
  reopen        a fresh process evaluates a 32x32 matmul sum on the default
                device and requires the exact value.
"""
import ctypes
import json
import os
import sys
from pathlib import Path

mode, expect_wheel, repo_root = sys.argv[1:4]
sys.path.insert(0, str(Path(repo_root) / "scripts"))
import mlx_provenance as mp

prov = mp.installed_provenance(expect_wheel=Path(expect_wheel))
if prov["verified"] != "match" or prov["version_match"] is not True:
    print("PROVENANCE_REFUSAL", json.dumps(prov), file=sys.stderr)
    sys.exit(3)
print(mp.provenance_line(prov), flush=True)

import mlx.core as mx

if mode == "gpu-add":
    lib = ctypes.CDLL(os.path.join(os.path.dirname(mx.__file__), "lib", "libmlx.so"))

    class Snap(ctypes.Structure):
        _fields_ = [(n, ctypes.c_uint64) for n in (
            "dispatches", "submissions", "copies", "fills", "compute",
            "finalize", "commit_work", "commit_noop", "barriers",
            "barriers_skipped")]

    lib.mlx_omarchy_trace_snapshot_abi_size.restype = ctypes.c_uint64
    assert lib.mlx_omarchy_trace_snapshot_abi_size() == ctypes.sizeof(Snap)

    def snap():
        s = Snap()
        lib.mlx_omarchy_trace_snapshot(ctypes.byref(s))
        return s

    x = mx.arange(1, 1025, dtype=mx.float32)
    y = mx.full((1024,), 2.0, dtype=mx.float32)
    mx.eval(x, y)
    s0 = snap()
    z = mx.add(x, y)
    mx.eval(z)
    s1 = snap()
    total = float(z.sum())
    want = float(sum(range(1, 1025)) + 2 * 1024)
    dk = s1.compute - s0.compute
    print(f"device {mx.default_device()} gpu_add_sum {total} expected {want} "
          f"vk_compute_dispatches_delta {dk} vk_submissions_delta {s1.submissions - s0.submissions}")
    sys.exit(0 if total == want and dk > 0 else 1)

if mode == "cpu-negative":
    s = mx.new_stream(mx.cpu)
    a = mx.array([1.0, 2.0], dtype=mx.float32)
    b = mx.array([3.0, 4.0], dtype=mx.float32)
    try:
        r = mx.add(a, b, stream=s)
        mx.eval(r)
    except RuntimeError as e:
        print("refusal:", str(e))
        sys.exit(0 if str(e) == "Add has no CPU implementation." else 1)
    print("UNEXPECTED_SUCCESS", r.tolist())
    sys.exit(1)

if mode == "reopen":
    a = mx.arange(1, 1025, dtype=mx.float32).reshape(32, 32)
    c = (a @ mx.eye(32)).sum()
    mx.eval(c)
    v = float(c)
    print(mx.default_device(), v, v == float(1024 * 1025 // 2))
    sys.exit(0 if v == float(1024 * 1025 // 2) else 1)

sys.exit("unknown mode: " + mode)
