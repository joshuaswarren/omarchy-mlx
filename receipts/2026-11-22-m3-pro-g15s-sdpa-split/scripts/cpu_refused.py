"""CPU-disabled wheel controls of docs/parity-contract.md § 2.1, one control per process.

usage: cpu_refused.py gpu|cpu EXPECT_WHEEL REPO_ROOT
  gpu: a nontrivial float32 GPU addition; requires the expected values and a positive Vulkan compute dispatch count (trace snapshot ABI).
  cpu: mx.add on two-element float32 inputs on a new explicit CPU stream (mx.new_stream(mx.cpu)); requires the pinned no-CPU error
       "Add has no CPU implementation."; any other exception, or a result, fails (exit 1).
Prints the scripts/mlx_provenance.py line and refuses (exit 3) unless the installed wheel matches EXPECT_WHEEL.
"""
import ctypes
import os
import sys
from pathlib import Path

mode, wheel, repo = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
sys.path.insert(0, str(repo / "scripts"))
import mlx_provenance as mp  # noqa: E402

prov = mp.installed_provenance(expect_wheel=wheel)
print(mp.provenance_line(prov), flush=True)
if prov["verified"] != "match" or prov["version_match"] is not True:
    sys.exit(3)
import mlx.core as mx  # noqa: E402

if mode == "gpu":
    lib = ctypes.CDLL(os.path.join(os.path.dirname(mx.__file__), "lib", "libmlx.so"))

    class Snap(ctypes.Structure):
        _fields_ = [(n, ctypes.c_uint64) for n in ("primitives", "submissions", "copies", "fills", "compute", "finalize",
                                                   "commit_work", "commit_noop", "barriers", "barriers_skipped")]

    def compute():
        s = Snap()
        lib.mlx_omarchy_trace_snapshot(ctypes.byref(s))
        return s.compute

    x = mx.arange(4096, dtype=mx.float32)
    y = mx.full((4096,), 0.5, dtype=mx.float32)
    d0 = compute()
    z = mx.add(x, y)
    mx.eval(z)
    d = compute() - d0
    ok = z.tolist() == [i + 0.5 for i in range(4096)] and d > 0
    print(f"gpu add: values {'expected' if ok else 'WRONG'} (first {z[:3].tolist()}), vulkan compute dispatches {d}", flush=True)
    sys.exit(0 if ok else 1)
cpu = mx.new_stream(mx.cpu)
try:
    mx.eval(mx.add(mx.array([1.0, 2.0]), mx.array([3.0, 4.0]), stream=cpu))
except Exception as e:  # noqa: BLE001
    msg = str(e)
    print(f"cpu add on mx.new_stream(mx.cpu): {type(e).__name__}: {msg}", flush=True)
    sys.exit(0 if "Add has no CPU implementation." in msg else 1)
print("cpu add on mx.new_stream(mx.cpu): RAN (control failed)", flush=True)
sys.exit(1)
