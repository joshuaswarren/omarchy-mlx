"""Op-level repro: mx.fast.rope_rms_norm vs unfused rms_norm->rope chain.

Qwen3-4B shapes: H_q=32, H_kv=8, D=128, plain (non-traditional) rope.
B in {1,2,4} x L in {1,17} x offset in {int 0, int 5, per-request array}.
Pass = bitwise equality (bf16 viewed as uint16) between fused and reference.
Expected on the shipped wheel: B>1 with array offset throws
'[broadcast_shapes] (...64) and (...128)'; everything else bit-exact.
Prints provenance first per repo test rules.
"""
import json
import sys

import mlx.core as mx
import mlx

print("PROVENANCE mlx:", mlx.__file__, mx.__version__, flush=True)

H_Q, H_KV, D, EPS = 32, 8, 128, 1e-6
BASE, SCALE = 1000000.0, 1.0


def ref(x, w, off):
    # The model's eager chain: rms_norm on row-contiguous (B,L,H,D), then
    # transpose to (B,H,L,D) and rope with the same offset.
    xn = mx.fast.rms_norm(x, w, EPS)
    return mx.fast.rope(
        xn.transpose(0, 2, 1, 3), D, traditional=False,
        base=BASE, scale=SCALE, offset=off)


def fused(x, w, off):
    return mx.fast.rope_rms_norm(
        x.transpose(0, 2, 1, 3), D, w, EPS, traditional=False,
        base=BASE, scale=SCALE, offset=off)


def bits(a):
    return a.view(mx.uint16) if hasattr(a, "view") else a


rows = []
for B in (1, 2, 4):
    for L in (1, 17):
        mx.random.seed(B * 100 + L)
        q = mx.random.normal((B, L, H_Q, D)).astype(mx.bfloat16)
        k = mx.random.normal((B, L, H_KV, D)).astype(mx.bfloat16)
        wq = mx.random.normal((D,)).astype(mx.bfloat16)
        wk = mx.random.normal((D,)).astype(mx.bfloat16)
        offs = {"int0": 0, "int5": 5}
        if B > 1:
            offs["arr"] = mx.array([3 * i + 1 for i in range(B)], dtype=mx.int32)
        else:
            offs["arr1"] = mx.array([5], dtype=mx.int32)
        for kind, off in offs.items():
            row = {"B": B, "L": L, "off": kind}
            try:
                r_q, r_k = ref(q, wq, off), ref(k, wk, off)
                f_q, f_k = fused(q, wq, off), fused(k, wk, off)
                mx.eval(r_q, r_k, f_q, f_k)
                row["q_eq"] = bool(mx.array_equal(bits(f_q), bits(r_q)))
                row["k_eq"] = bool(mx.array_equal(bits(f_k), bits(r_k)))
            except Exception as e:  # noqa: BLE001 - repro records the throw
                row["error"] = f"{type(e).__name__}: {e}"
            rows.append(row)
            print(json.dumps(row), flush=True)

crashed = [r for r in rows if "error" in r]
exact = [r for r in rows if r.get("q_eq") and r.get("k_eq")]
print(f"SUMMARY rows={len(rows)} bit_exact={len(exact)} errored={len(crashed)}", flush=True)
for r in crashed:
    print("ERR", json.dumps(r), flush=True)
sys.exit(0)
