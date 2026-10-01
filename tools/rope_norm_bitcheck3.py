#!/usr/bin/env python3
"""rope_rms_norm iso check v3 (DecodeFuse3): fused primitive vs the exact
composed model chain, WITH the eval fence contract.

v2 (DecodeGap8) demanded bit-identity on fallback legs (f16/f32/traditional/
D>256) too, because the primitive composed them in-process. v3 encodes the
DecodeFuse3 fence: the mlx/fast.cpp wrapper REFUSES any with_norm leg the
FastRopeNorm kernel would not fuse (non-bf16, traditional, D>256 or odd,
non-contiguous last dim) with invalid_argument, so those legs can never
silently skip the norm or route through an unproven in-process composition.
The model path only issues fuseable legs (the mlx-lm-rope-norm.patch python
gate), so v3 asserts:
  - fuseable legs (all bf16 model shapes, decode offsets through the KV
    window, prefill T=512/17/2): bit-identity fused vs reference chain.
  - fenced legs (f16/f32/traditional/big-D): the fused call MUST raise
    ValueError; the reference chain still runs so its bits are recorded.

usage: rope_norm_bitcheck3.py OUT.json
"""
import hashlib
import json
import sys
import zlib
import mlx.core as mx
import numpy as np

out_path = sys.argv[1]
results = {}


def bits(a):
    if a.dtype in (mx.bfloat16, mx.float16):
        return np.array(a.view(mx.uint16)).tobytes()
    return np.array(a.view(mx.uint32)).tobytes()


def ref(x_pre, weight, eps, dims, base, scale, offset, traditional=False):
    normed = mx.fast.rms_norm(x_pre, weight, eps)
    view = normed.transpose(0, 2, 1, 3)
    return mx.fast.rope(
        view, dims, traditional=traditional, base=base, scale=scale,
        offset=offset)


def fused(x_pre, weight, eps, dims, base, scale, offset, traditional=False):
    return mx.fast.rope_rms_norm(
        x_pre.transpose(0, 2, 1, 3), dims, weight, eps,
        traditional=traditional, base=base, scale=scale, offset=offset)


def case(name, B, H, T, D, dims, base, scale, offset, eps=1e-6,
         dtype=mx.bfloat16, traditional=False, scale_w=1.0, fenced=False):
    rng = np.random.default_rng(zlib.crc32(name.encode()))
    x_np = rng.standard_normal((B, T, H, D)) * scale_w
    w_np = 1.0 + 0.1 * rng.standard_normal(D)
    x_pre = mx.array(x_np.astype(np.float32)).astype(dtype)
    weight = mx.array(w_np.astype(np.float32)).astype(dtype)
    r = ref(x_pre, weight, eps, dims, base, scale, offset, traditional)
    rb = bits(r)
    row = {
        "sha_ref": hashlib.sha256(rb).hexdigest()[:16],
        "expect": "fenced" if fenced else "match",
    }
    try:
        f = fused(x_pre, weight, eps, dims, base, scale, offset, traditional)
    except ValueError as e:
        row["fenced_raised"] = True
        row["match"] = bool(fenced)
        row["note"] = str(e)[:120]
        results[name] = row
        return
    fb = bits(f)
    row["sha_fused"] = hashlib.sha256(fb).hexdigest()[:16]
    row["fenced_raised"] = False
    row["match"] = (rb == fb) and not fenced
    row["n_mismatch"] = int(sum(a != b for a, b in zip(rb, fb))) if rb != fb else 0
    results[name] = row


# The exact model shapes, decode: q (1,8,1,256) dims=64, k (1,2,1,256)
# dims=64, base 1e7, eps 1e-6, decode offsets across the KV window
# (99999 stays under the 100000 trig-accuracy gate).
for off in (0, 1, 511, 4096, 99999):
    case(f"model-q-off{off}", 1, 8, 1, 256, 64, 1e7, 1.0, off)
    case(f"model-k-off{off}", 1, 2, 1, 256, 64, 1e7, 1.0, off)
# Model-shaped PREFILL cases (the fused branch engages here too:
# head_seq_transpose, innermost stride 1).
case("model-q-prefill-t512-off0", 1, 8, 512, 256, 64, 1e7, 1.0, 0)
case("model-k-prefill-t512-off0", 1, 2, 512, 256, 64, 1e7, 1.0, 0)
case("model-q-prefill-t512-off600", 1, 8, 512, 256, 64, 1e7, 1.0, 600)
case("model-q-prefill-t17-off33", 1, 8, 17, 256, 64, 1e7, 1.0, 33)
case("model-k-prefill-t17-off33", 1, 2, 17, 256, 64, 1e7, 1.0, 33)
case("model-q-prefill-t2-off1", 1, 8, 2, 256, 64, 1e7, 1.0, 1)
# Batch/heads/time variety and dims == D (no passthrough).
case("b2-h8-t1-d256-dims256", 2, 8, 1, 256, 256, 1e7, 1.0, 7)
case("b1-h4-t3-d128-dims64", 1, 4, 3, 128, 64, 1e6, 1.0, 33)
case("b1-h2-t1-d64-dims32", 1, 2, 1, 64, 32, 5e5, 1.0, 0)
# Fenced legs: the wrapper must refuse these (v2 demanded in-process
# composition; v3 demands the fence — the model never issues them).
case("f16-b1-h4-t1-d256", 1, 4, 1, 256, 64, 1e7, 1.0, 3, dtype=mx.float16,
     fenced=True)
case("f32-b1-h4-t1-d256", 1, 4, 1, 256, 64, 1e7, 1.0, 3, dtype=mx.float32,
     fenced=True)
case("traditional-b1-h4-t1-d256", 1, 4, 1, 256, 64, 1e7, 1.0, 9,
     traditional=True, fenced=True)
case("big-d512-b1-h2-t1", 1, 2, 1, 512, 64, 1e7, 1.0, 4, fenced=True)
# Weight-free scale stress: large magnitudes and denormal-ish rows.
case("bigvals-b1-h8-t1-d256", 1, 8, 1, 256, 64, 1e7, 1.0, 2, scale_w=33.0)
case("smallvals-b1-h8-t1-d256", 1, 8, 1, 256, 64, 1e7, 1.0, 2,
     scale_w=1e-4)

fails = [k for k, v in results.items() if not v["match"]]
meta = {"mlx": mx.__version__, "n_cases": len(results), "fails": fails}
results_out = {"meta": meta, "rows": results}
with open(out_path, "w") as fh:
    json.dump(results_out, fh, indent=1, sort_keys=True)
print(f"ROPE-NORM-BITCHECK {'PASS' if not fails else 'FAIL'}"
      f" cases={len(results)} fails={fails[:8]}")
sys.exit(0 if not fails else 1)
