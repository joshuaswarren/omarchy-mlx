#!/usr/bin/env python3
"""F4 qk-norm iso check (DecodeFuse4): the fused gdn_conv_update q/k
norm epilogue vs the exact composed chain, bit-for-bit, inside ONE venv.

Rows per case name:
  qknorm.ref.q/.k/.v/.state   composed: gdn_conv_update(activate=True),
                              split at [kd, 2*kd], per-head reshape,
                              rms_norm_scaled (bf16) or the plain scalar
                              rms_norm chain (other dtypes, the model's
                              non-bf16 arm)
  qknorm.fused.q/.k/.v/.state gdn_conv_update(activate=True, qk_key_dim=kd,
                              qk_scale_q=, qk_scale_k=, qk_eps=)
  qknorm.fence.<case>         "RAISED" when the fused call refuses the
                              geometry as designed (marked fenced)
Gate: ref == fused on every row of every unfenced case; fenced cases
must raise. The bf16 geometry is the model's (B=1, kd=2048, C=6144,
K=4); the batch/kd variety cases prove the two-half reduction scales;
the f16 case proves the fallback composition exact.
"""
import hashlib
import json
import sys

import mlx.core as mx
import numpy as np

out_path = sys.argv[1]
rows = {}
gdn = mx.fast.gdn_conv_update
HEAD = 128


def h(*arrs):
    m = hashlib.sha256()
    for a in arrs:
        mx.eval(a)
        if a.dtype in (mx.float32, mx.int32, mx.uint32):
            v = np.array(a.view(mx.uint32))
        elif a.dtype in (mx.bfloat16, mx.float16, mx.int16, mx.uint16):
            v = np.array(a.view(mx.uint16))
        else:
            v = np.array(a)
        m.update(v.tobytes())
    return m.hexdigest()[:16]


def bf16(x):
    return mx.array(np.asarray(x, np.float32)).astype(mx.bfloat16)


def ref_norms(out, s, kd, sq, sk, eps):
    """The exact model composition after the conv dispatch."""
    B, S, _ = out.shape
    parts = mx.split(out, [kd, 2 * kd], -1)
    q = parts[0].reshape(B, S, kd // HEAD, HEAD)
    k = parts[1].reshape(B, S, kd // HEAD, HEAD)
    if out.dtype == mx.bfloat16:
        q = mx.fast.rms_norm_scaled(q, None, sq, eps)
        k = mx.fast.rms_norm_scaled(k, None, sk, eps)
    else:
        q = sq * mx.fast.rms_norm(q, None, eps)
        k = sk * mx.fast.rms_norm(k, None, eps)
    return q, k, parts[2], s


def case(name, st, xv, wv, kd, sq, sk, eps=1e-6, fenced=False):
    inv = 128.0**-0.5
    if sq is None:
        sq = inv * inv
    if sk is None:
        sk = inv
    out, s = gdn(st, xv, wv, True)
    rq, rk, rv, rs = ref_norms(out, s, kd, sq, sk, eps)
    rows[f"qknorm.ref.{name}"] = h(rq, rk, rv, rs)
    try:
        fout, fs = gdn(
            st, xv, wv, True,
            qk_key_dim=kd, qk_scale_q=sq, qk_scale_k=sk, qk_eps=eps,
        )
    except TypeError:
        rows[f"qknorm.fused.{name}"] = "ABSENT"
        return False
    except RuntimeError:
        rows[f"qknorm.fence.{name}"] = "RAISED"
        return not fenced
    if fenced:
        rows[f"qknorm.fence.{name}"] = "NOT-RAISED"
        return False
    B, S, _ = fout.shape
    fq = fout[:, :, :kd].reshape(B, S, kd // HEAD, HEAD)
    fk = fout[:, :, kd:2 * kd].reshape(B, S, kd // HEAD, HEAD)
    fv = fout[:, :, 2 * kd:]
    rows[f"qknorm.fused.{name}"] = h(fq, fk, fv, fs)
    return (
        h(rq) == h(fq)
        and h(rk) == h(fk)
        and h(rv) == h(fv)
        and h(rs) == h(fs)
    )


rng = np.random.default_rng(20261001)
results = {}
fails = []

# Model geometry (Qwen3.8-2B GDN decode): B=1, kd=2048, C=6144, K=4.
C, K = 6144, 4
state = bf16(rng.standard_normal((1, K - 1, C)) * 0.7)
x = bf16(rng.standard_normal((1, 1, C)) * 0.7)
w = bf16(rng.standard_normal((C, K, 1)) * 0.4)
cases = {"model-b1-kd2048": (state, x, w)}

x_np = np.array(x.astype(mx.float32))
x_np[0, 0, 0] = np.inf
x_np[0, 0, 1] = -np.inf
x_np[0, 0, 2] = np.nan
x_np[0, 0, 3] = 0.0
x_np[0, 0, 4] = -0.0
x_np[0, 0, 5] = 88.0
x_np[0, 0, 6] = -88.0
cases["model-special"] = (state, bf16(x_np), w)

st_np = np.array(state.astype(mx.float32))
st_np[0, 0, 5] = 1e30
st_np[0, 0, 6] = -1e30
cases["model-bigstate"] = (bf16(st_np), x, w)

x_big = bf16(rng.standard_normal((1, 1, C)) * 33.0)
cases["model-bigvals"] = (state, x_big, w)

# Batch 2: two batches, per-WG reduction spans one batch each (C % 256 == 0).
state2 = bf16(rng.standard_normal((2, K - 1, C)) * 0.7)
x2 = bf16(rng.standard_normal((2, 1, C)) * 0.7)
cases["model-b2-kd2048"] = (state2, x2, w)

# Geometry variety: kd 512 / 256 (v sized to 3 heads of 128).
for kd, vd in ((512, 384), (256, 128)):
    c = 2 * kd + vd
    stv = bf16(rng.standard_normal((1, K - 1, c)) * 0.7)
    xv = bf16(rng.standard_normal((1, 1, c)) * 0.7)
    wv = bf16(rng.standard_normal((c, K, 1)) * 0.4)
    cases[f"b1-kd{kd}-v{vd}"] = (stv, xv, wv)

# Scale stress: non-model scales exercise the bf16-rounded promote path.
stv = bf16(rng.standard_normal((1, K - 1, C)) * 0.7)
xv = bf16(rng.standard_normal((1, 1, C)) * 0.7)
cases["scale-33"] = (stv, xv, w)
cases["scale-tiny"] = (stv, xv, w)

# Fenced legs: the wrapper must refuse these loudly.
cases["fence-kd128"] = None  # built below: kd=128 -> key_dim % 256 != 0
cases["fence-v128"] = None  # kd=256, C=640 -> B*C % 256 != 0
cases["f16-fallback"] = None  # built below: non-bf16 -> fallback composition

# f16 fallback leg: the fused entry must compose the exact model chain.
stf = mx.array(np.asarray(rng.standard_normal((1, K - 1, C)), np.float32)).astype(mx.float16)
xf = mx.array(np.asarray(rng.standard_normal((1, 1, C)), np.float32)).astype(mx.float16)
wf = mx.array(np.asarray(rng.standard_normal((C, K, 1)), np.float32)).astype(mx.float16)

for name, spec in cases.items():
    if name == "fence-kd128":
        kd = 128
        c = 3 * kd
        stc = bf16(rng.standard_normal((1, K - 1, c)) * 0.7)
        xc = bf16(rng.standard_normal((1, 1, c)) * 0.7)
        wc = bf16(rng.standard_normal((c, K, 1)) * 0.4)
        ok = case(name, stc, xc, wc, kd, None, None, fenced=True)
    elif name == "fence-v128":
        kd = 256
        c = 2 * kd + 128
        stc = bf16(rng.standard_normal((1, K - 1, c)) * 0.7)
        xc = bf16(rng.standard_normal((1, 1, c)) * 0.7)
        wc = bf16(rng.standard_normal((c, K, 1)) * 0.4)
        ok = case(name, stc, xc, wc, kd, None, None, fenced=True)
    elif name == "scale-33":
        ok = case(name, stv, xv, w, 2048, 33.0, 0.088, 1e-6)
    elif name == "scale-tiny":
        ok = case(name, stv, xv, w, 2048, 1e-4, 1e-4, 1e-6)
    elif name == "f16-fallback":
        ok = case(name, stf, xf, wf, 2048, None, None)
    else:
        kd = int(name.split("kd")[1].split("-")[0]) if "kd" in name else 2048
        ok = case(name, *spec, kd, None, None)
    results[name] = ok
    if not ok:
        fails.append(name)

json.dump({"meta": {"mlx": mx.__version__, "fails": fails}, "rows": rows},
          open(out_path, "w"), indent=1, sort_keys=True)
print(f"QKNORM-BITCHECK {'PASS' if not fails else 'FAIL'}"
      f" cases={len(cases)} fails={fails}")
sys.exit(0 if not fails else 1)
