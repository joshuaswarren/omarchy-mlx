"""bf16 decode attention sweep, one process, one variant (chosen by the caller's MLX_OMARCHY_SDPA_* environment).

usage: sdpa_sweep.py REPEATS HEAD_DIM Q_HEADS KV_HEADS KV[,KV...] OUT.json EXPECT_WHEEL REPO_ROOT

Single-query grouped-query attention as in decode: q (1, Q_HEADS, 1, HEAD_DIM) against K and V (1, KV_HEADS, KV, HEAD_DIM), bf16,
scale HEAD_DIM^-0.5, no mask, for each KV. The timing method is receipts/2026-11-15-m4-pro-g16s-decode-defaults/split-k/tools/sdpa_bench_kv.py:
  independent: 36 calls on 36 distinct K and V sets in one evaluation (microseconds per call);
  chain: 36 calls where the output of one call is the query of the next (microseconds per call);
each the median of REPEATS timed evaluations after one warm-up. Added here: head width and head counts as arguments, Vulkan compute
dispatches per call (trace snapshot ABI, over one independent evaluation) so the route is visible, the largest |error| of one call against
a float64 reference on the same inputs, and the scripts/mlx_provenance.py line (refuses, exit 3, unless the installed wheel matches).
"""
import ctypes
import importlib.metadata
import json
import os
import statistics
import sys
import time
from pathlib import Path

reps, D, HQ, HKV = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
kvs = [int(a) for a in sys.argv[5].split(",")]
out_path, wheel, repo = sys.argv[6], Path(sys.argv[7]), Path(sys.argv[8])
sys.path.insert(0, str(repo / "scripts"))
import mlx_provenance as mp  # noqa: E402

prov = mp.installed_provenance(expect_wheel=wheel)
print(mp.provenance_line(prov), flush=True)
if prov["verified"] != "match" or prov["version_match"] is not True:
    sys.exit(3)
import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402

FIELDS = ("gpu_primitive_dispatches", "vk_submissions", "vk_buffer_copies", "vk_buffer_fills", "vk_compute_dispatches",
          "omarchy_finalize_calls", "commit_calls_with_work", "commit_calls_noop", "barriers_emitted", "barriers_skipped")


class Snap(ctypes.Structure):
    _fields_ = [(n, ctypes.c_uint64) for n in FIELDS]


lib = ctypes.CDLL(str(importlib.metadata.distribution("mlx-omarchy").locate_file("mlx/lib/libmlx.so")))
lib.mlx_omarchy_trace_snapshot_abi_size.restype = ctypes.c_uint64
assert int(lib.mlx_omarchy_trace_snapshot_abi_size()) == ctypes.sizeof(Snap)
lib.mlx_omarchy_trace_snapshot.argtypes = [ctypes.POINTER(Snap)]


def dispatches():
    s = Snap()
    lib.mlx_omarchy_trace_snapshot(ctypes.byref(s))
    return s.vk_compute_dispatches


N = 36
SCALE = D**-0.5
env = {k: v for k, v in sorted(os.environ.items()) if k.startswith("MLX_OMARCHY")}
print(f"SDPA mlx {mx.__version__} device {mx.default_device()} D {D} HQ {HQ} HKV {HKV} env {env}", flush=True)


def timed(fn):
    fn()
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t)
    return statistics.median(ts), min(ts)


def ref64(q, k, v):
    q, k, v = (np.array(x.astype(mx.float32)).astype(np.float64) for x in (q, k, v))
    g = HQ // HKV
    k, v = np.repeat(k, g, axis=1), np.repeat(v, g, axis=1)
    s = (q @ k.transpose(0, 1, 3, 2)) * SCALE
    p = np.exp(s - s.max(-1, keepdims=True))
    return (p / p.sum(-1, keepdims=True)) @ v


rows = []
for kv in kvs:
    mx.random.seed(kv)
    ks = [mx.random.normal((1, HKV, kv, D), dtype=mx.bfloat16) for _ in range(N)]
    vs = [mx.random.normal((1, HKV, kv, D), dtype=mx.bfloat16) for _ in range(N)]
    q0 = mx.random.normal((1, HQ, 1, D), dtype=mx.bfloat16)
    mx.eval(ks, vs, q0)

    def independent():
        mx.eval([mx.fast.scaled_dot_product_attention(q0, ks[i], vs[i], scale=SCALE) for i in range(N)])

    def chain():
        q = q0
        for i in range(N):
            q = mx.fast.scaled_dot_product_attention(q, ks[i], vs[i], scale=SCALE)
        mx.eval(q)

    independent()
    d0 = dispatches()
    independent()
    disp = (dispatches() - d0) / N
    o = mx.fast.scaled_dot_product_attention(q0, ks[0], vs[0], scale=SCALE)
    err = float(np.abs(np.array(o.astype(mx.float32)).astype(np.float64) - ref64(q0, ks[0], vs[0])).max())
    mi, bi = timed(independent)
    mc, bc = timed(chain)
    row = {"kv": kv, "independent_us": round(mi / N * 1e6, 2), "independent_best_us": round(bi / N * 1e6, 2),
           "chain_us": round(mc / N * 1e6, 2), "chain_best_us": round(bc / N * 1e6, 2),
           "dispatches_per_call": disp, "max_abs_err_vs_f64": err}
    rows.append(row)
    print(f"SDPA kv {kv}: independent {row['independent_us']} us (best {row['independent_best_us']}); chain {row['chain_us']} us "
          f"(best {row['chain_best_us']}); {disp:g} dispatches per call; max |err| vs f64 {err:.3g}", flush=True)
    del ks, vs
json.dump({"provenance": prov, "mlx": mx.__version__, "head_dim": D, "q_heads": HQ, "kv_heads": HKV, "repeats": reps, "calls": N,
           "env": env, "rows": rows, "load1_end": os.getloadavg()[0]}, open(out_path, "w"), indent=1, default=str)
print("SDPA done", flush=True)
