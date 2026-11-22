"""f16 single-query decode SDPA: route, outputs and per-call time in one process.

usage: sdpa_f16_ab.py OUT.json OUT.npz EXPECT_WHEEL REPO_ROOT LABEL

The arm under test is chosen by the caller through
MLX_OMARCHY_SDPA_F16_NO_COOPMAT ("0" = composition, "1" = fused kernel,
unset = part default). Every shape: q (1, H, 1, D), k/v (1, KV, L, D) f16,
scale D^-0.5, fixed seed, so both arms see the same inputs.

Per shape it records
  - Vulkan compute dispatches of one call (trace snapshot ABI): 3 for the
    composition, 1 (L <= 1024) or 2 (L > 1024) for the fused kernel;
  - the output bits (npz) and max |out - float64 reference|;
  - independent: host wall per call of 200 calls of SDPA plus the query
    scaling in one eval (the method of receipts/2026-11-01-m4-sdpa-decode-
    bf16-256, sdpa_bits.py), 7 repetitions, median and min;
  - chain: 36 dependent calls (the next query is the previous output) in
    one eval, the shape of a decode step's layer chain, 7 repetitions.
"""
import ctypes
import json
import os
import statistics
import sys
import time
from pathlib import Path

out_json, out_npz, expect_wheel, repo_root, label = sys.argv[1:6]
sys.path.insert(0, str(Path(repo_root) / "scripts"))
import mlx_provenance as mp  # noqa: E402

prov = mp.installed_provenance(expect_wheel=Path(expect_wheel))
pline = mp.provenance_line(prov)
print(pline, flush=True)
if prov["verified"] != "match" or prov["version_match"] is not True:
    print("PROVENANCE_REFUSAL", file=sys.stderr)
    sys.exit(3)

import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402

lib = ctypes.CDLL(os.path.join(os.path.dirname(mx.__file__), "lib", "libmlx.so"))


class Snap(ctypes.Structure):
    _fields_ = [(n, ctypes.c_uint64) for n in (
        "dispatches", "submissions", "copies", "fills", "compute", "finalize",
        "commit_work", "commit_noop", "barriers", "barriers_skipped")]


lib.mlx_omarchy_trace_snapshot_abi_size.restype = ctypes.c_uint64
assert lib.mlx_omarchy_trace_snapshot_abi_size() == ctypes.sizeof(Snap)


def dispatches():
    """Vulkan compute dispatches recorded so far (vk_compute_dispatches)."""
    s = Snap()
    lib.mlx_omarchy_trace_snapshot(ctypes.byref(s))
    return s.compute


# (query heads, kv heads, width, keys)
SHAPES = [(16, 8, 128, L) for L in (37, 263, 1100, 4096)]       # the G16S rows
SHAPES += [(32, 8, 128, L) for L in (64, 263, 1024, 1100, 2048, 4096)]  # Qwen3-4B
SHAPES += [(32, 8, 64, L) for L in (263, 1100, 4096)]           # width 64
REPS = 7
N_IND = 200
N_CHAIN = 36

rows, arrays = [], {}
for idx, (h, kv, d, L) in enumerate(SHAPES):
    mx.random.seed(1000 + idx)
    q = (mx.random.normal((1, h, 1, d)) * 0.5).astype(mx.float16)
    k = mx.random.normal((1, kv, L, d)).astype(mx.float16)
    v = mx.random.normal((1, kv, L, d)).astype(mx.float16)
    mx.eval(q, k, v)
    scale = d ** -0.5
    f = lambda x: mx.fast.scaled_dot_product_attention(x, k, v, scale=scale)

    y = f(q); mx.eval(y); mx.synchronize()
    d0 = dispatches()
    y = f(q); mx.eval(y); mx.synchronize()
    nd = dispatches() - d0
    bits = np.array(y.view(mx.uint16))
    arrays[f"c{idx}"] = bits
    # float64 reference
    qn = np.array(q.astype(mx.float32)).astype(np.float64)
    kn = np.repeat(np.array(k.astype(mx.float32)).astype(np.float64), h // kv, axis=1)
    vn = np.repeat(np.array(v.astype(mx.float32)).astype(np.float64), h // kv, axis=1)
    s = (qn @ kn.transpose(0, 1, 3, 2)) * scale
    s = np.exp(s - s.max(-1, keepdims=True))
    ref = (s / s.sum(-1, keepdims=True)) @ vn
    got = np.array(y.astype(mx.float32)).astype(np.float64)
    err = float(np.abs(got - ref).max())

    def ind():
        mx.eval([f(q * (1.0 + i * 1e-3)) for i in range(N_IND)])

    def chain():
        x = q
        for _ in range(N_CHAIN):
            x = f(x)
        mx.eval(x)

    ind(); chain(); mx.synchronize()
    t_ind, t_chain = [], []
    for _ in range(REPS):
        t0 = time.perf_counter(); ind(); t_ind.append((time.perf_counter() - t0) / N_IND * 1e6)
        t0 = time.perf_counter(); chain(); t_chain.append((time.perf_counter() - t0) / N_CHAIN * 1e6)
    row = {"heads": h, "kv_heads": kv, "width": d, "keys": L, "vk_compute_dispatches": nd,
           "max_abs_err_vs_f64": err,
           "ind_us_median": round(statistics.median(t_ind), 1), "ind_us_min": round(min(t_ind), 1),
           "chain_us_median": round(statistics.median(t_chain), 1), "chain_us_min": round(min(t_chain), 1)}
    rows.append(row)
    print(label, json.dumps(row), flush=True)

np.savez(out_npz, **arrays)
json.dump({"label": label, "provenance": pline,
           "env_MLX_OMARCHY_SDPA_F16_NO_COOPMAT": os.environ.get("MLX_OMARCHY_SDPA_F16_NO_COOPMAT"),
           "env_MLX_OMARCHY_BATCH_WORK": os.environ.get("MLX_OMARCHY_BATCH_WORK"),
           "device": str(mx.default_device()), "rows": rows}, open(out_json, "w"), indent=1)
print("DONE", label, flush=True)
