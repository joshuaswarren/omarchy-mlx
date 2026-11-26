#!/usr/bin/env python3
"""Value-level SDPA equivalence suite for the f16-scores rework.

Compares mx.fast.scaled_dot_product_attention (the primitive under test,
running the backend's f16-scores path) against the validated f32 composed
path it replaces, on the same build and device. The composed reference
mirrors the pre-rework ScaledDotProductAttention::eval_gpu exactly:
cast q/k/v to f32, scale as an f32 multiply, GQA via 5-D reshape, f32
scores matmul, additive mask, f32 softmax, f32 probs@v matmul, and the
5-D result flattened back to 4-D.

The f16-storage tolerance model is also checked directly: the rework's
scores are stored as f16 with f32 accumulation, so the suite emulates
that storage (f16 operands upcast to f32 for the dot, f16 store of the
scaled scores) and requires scores/probs within 2e-3 of the f32
reference and outputs within 1e-3, before comparing the real primitive.

Coverage per the census addendum 3 equivalence bar:
  - decode shapes [B, heads, 1, head_dim]
  - non-contiguous cache slices at offsets 1, 7, 41, 256
  - GQA repeat factors 1, 2, 7
  - head_dim 48 (not a power-of-two multiple)
  - q_len 1 and q_len > 1
  - causal and explicit-additive mask legs

Usage (llvmpipe dev box):
  MLX_OMARCHY_ALLOW_NON_APPLE=1 python3 scripts/sdpa_equivalence.py
Exit 0 = all cases pass; exit 1 = first failing case printed.
"""

import os
import sys

import numpy as np

import mlx.core as mx

SIGMA = 0.5


def draw(key, shape, dtype=mx.float16):
    return (mx.random.normal(shape, key=key) * SIGMA).astype(dtype)


# The bfloat16 finite maximum: the bf16 additive causal floor. bf16
# shares float32's 8-bit exponent, so every finite f32 score is
# representable and the floor stays overflow-immune where the f16 path
# caps at -65504.
BF16_FLOOR = -3.3895313892515355e38


def causal_addend(q_len, k_len, dtype):
    """The causal additive mask as a host-loaded fixture array.

    Mirrors what each C++ path writes per element: 0 / -1e30 in f32,
    0 / -65504 (the f16 finite maximum, matching the backend's
    fully-masked-row guard) in f16, 0 / BF16_FLOOR in bfloat16. Built
    in numpy: the backend refuses GPU broadcast compares, and a fixture
    needs no GPU kernel. A multiplier form would NaN the kept
    positions (0 * -inf).
    """
    rows = np.arange(q_len)[:, None] + (k_len - q_len)
    cols = np.arange(k_len)[None, :]
    blocked = rows < cols
    if dtype == np.float16:
        values = np.where(blocked, -65504.0, 0.0).astype(np.float16)
        return mx.array(values)
    if dtype == mx.bfloat16:
        return mx.array(
            np.where(blocked, BF16_FLOOR, 0.0).astype(np.float32)
        ).astype(mx.bfloat16)
    values = np.where(blocked, -1e30, 0.0).astype(np.float32)
    return mx.array(values)


def composed_reference(q, k, v, scale, mask=None, causal=False):
    """The validated f32 composition (pre-rework eval_gpu), in mx ops."""
    batch, heads, q_len, head_dim = q.shape
    kv_heads = k.shape[1]
    k_len = k.shape[2]
    v_dim = v.shape[3]
    q32 = q.astype(mx.float32)
    k32 = k.astype(mx.float32)
    v32 = v.astype(mx.float32)
    q32 = q32 * mx.array(scale, mx.float32)
    if kv_heads != heads:
        repeats = heads // kv_heads
        qs = q32.reshape(batch, kv_heads, repeats, q_len, head_dim)
        k32 = k32.reshape(batch, kv_heads, 1, k_len, head_dim)
        v32 = v32.reshape(batch, kv_heads, 1, k_len, v_dim)
    else:
        qs = q32
    scores = qs @ k32.swapaxes(-1, -2)
    if causal:
        scores = scores + causal_addend(q_len, k_len, np.float32)
    elif mask is not None:
        m = mask.astype(mx.float32)
        if kv_heads != heads:
            m = m.reshape(
                batch, kv_heads, heads // kv_heads, q_len, k_len)
        scores = scores + m
    probs = mx.softmax(scores.astype(mx.float32), axis=-1)
    out = (probs @ v32).reshape(batch, heads, q_len, v_dim)
    return out.astype(q.dtype), scores, probs


def storage_emulation(q, k, v, scale, mask=None, causal=False):
    """What the fast path computes: score and prob storage at the query
    dtype (f16 or bf16), float32 accumulation.

    matmul.comp loads the storage dtype's operands as float, accumulates
    in float, scales by alpha in float, and stores rounded to the
    storage grid; softmax_suffix.comp runs float math over that
    storage. This mirrors that round trip for both 16-bit storages.
    """
    st = q.dtype
    batch, heads, q_len, head_dim = q.shape
    kv_heads = k.shape[1]
    k_len = k.shape[2]
    v_dim = v.shape[3]
    qst = q.astype(st)
    kst = k.astype(st)
    vst = v.astype(st)
    if kv_heads != heads:
        repeats = heads // kv_heads
        qs = qst.reshape(batch, kv_heads, repeats, q_len, head_dim)
        k32 = kst.reshape(
            batch, kv_heads, 1, k_len, head_dim).astype(mx.float32)
        v32 = vst.reshape(
            batch, kv_heads, 1, k_len, v_dim).astype(mx.float32)
    else:
        qs = qst
        k32 = kst.astype(mx.float32)
        v32 = vst.astype(mx.float32)
    scores = (qs.astype(mx.float32) @ k32.swapaxes(-1, -2)) * scale
    scores = scores.astype(st)
    if causal:
        scores = scores + causal_addend(q_len, k_len, st)
    elif mask is not None:
        scores = scores + mask.astype(st)
    probs = mx.softmax(scores.astype(mx.float32), axis=-1).astype(st)
    out = (probs.astype(mx.float32) @ v32).reshape(
        batch, heads, q_len, v_dim)
    return out.astype(st), scores, probs


def derived_bf16_bars(ref_scores, v, mask, blocked):
    """bf16 bars derived from the storage grid and the reduction, not
    fitted: score storage rounds each kept score to the bf16 grid (RNE
    half-ulp 2^-8 relative); that logit error propagates through
    softmax bounded by 2x, and the PV reduction adds one prob-storage
    rounding plus one output-storage rounding, weighted by max|v|.
    Masked positions (causal-blocked or additive-floor, where the
    stored value is the exact floor) are excluded from the magnitude
    the bound is built from.
    """
    keep = np.isfinite(np.asarray(ref_scores.astype(mx.float32)))
    if blocked is not None:
        while blocked.ndim < keep.ndim:
            blocked = blocked[None, :]
        keep = keep & ~np.broadcast_to(blocked, keep.shape)
    if mask is not None:
        floor = np.asarray(mask.astype(mx.float32)) <= -1e29
        while floor.ndim < keep.ndim:
            floor = floor[None, :]
        keep = keep & ~np.broadcast_to(floor, keep.shape)
    scores_v = np.asarray(ref_scores.astype(mx.float32))[keep]
    mask_mag = 0.0
    if mask is not None:
        kept_mask = np.asarray(mask.astype(mx.float32))
        while kept_mask.ndim < keep.ndim:
            kept_mask = kept_mask[None, :]
        kept_mask = np.broadcast_to(kept_mask, keep.shape)[keep]
        mask_mag = float(np.abs(kept_mask).max())
    score_mag = float(np.abs(scores_v).max()) + mask_mag
    v_mag = float(np.abs(np.asarray(v.astype(mx.float32))).max())
    half_ulp = 2.0 ** -8
    storage_atol = half_ulp * (2.0 * score_mag + 1.0)
    atol = half_ulp * (2.0 * score_mag + 2.0) * v_mag
    return atol, storage_atol


def check(name, q, k, v, scale, mask=None, causal=False, atol=None,
          storage_atol=None):
    kwargs = {}
    if causal:
        kwargs["mask"] = "causal"
    elif mask is not None:
        kwargs["mask"] = mask
    got = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, **kwargs)
    mx.eval(got)
    ref, ref_scores, ref_probs = composed_reference(q, k, v, scale, mask, causal)
    sim, sim_scores, sim_probs = storage_emulation(q, k, v, scale, mask, causal)

    failures = []

    # Causal cases exclude masked positions from the scores/probs
    # comparison: f32 stores -1e30 there and f16 stores -65504, both by
    # design, so the raw difference is meaningless.
    causal_blocked = None
    if causal:
        q_len_m, k_len_m = q.shape[2], k.shape[2]
        rows_m = np.arange(q_len_m)[:, None] + (k_len_m - q_len_m)
        cols_m = np.arange(k_len_m)[None, :]
        causal_blocked = rows_m < cols_m
    if atol is None:
        if q.dtype == mx.bfloat16:
            atol, storage_atol = derived_bf16_bars(
                ref_scores, v, None if causal else mask, causal_blocked)
        else:
            atol, storage_atol = 1e-3, 2e-3

    def max_err(sim, ref, exclude=None):
        a = np.asarray(sim.astype(mx.float32))
        b = np.asarray(ref.astype(mx.float32))
        finite = np.isfinite(a) & np.isfinite(b)
        if exclude is not None:
            while exclude.ndim < a.ndim:
                exclude = exclude[None, :]
            finite = finite & ~np.broadcast_to(exclude, a.shape)
        if not finite.any():
            return float("nan")
        return float(np.abs(a[finite] - b[finite]).max())

    sim_scores_err = max_err(sim_scores, ref_scores, causal_blocked)
    sim_probs_err = max_err(sim_probs, ref_probs, causal_blocked)
    sim_out_err = max_err(sim, ref)
    if sim_scores_err > storage_atol:
        failures.append(
            f"emulated scores err {sim_scores_err:.3e} > {storage_atol:g}")
    if sim_probs_err > storage_atol:
        failures.append(
            f"emulated probs err {sim_probs_err:.3e} > {storage_atol:g}")
    if sim_out_err > atol:
        failures.append(f"emulated out err {sim_out_err:.3e} > {atol:g}")

    got32 = got.astype(mx.float32)
    ref32 = ref.astype(mx.float32)
    if got32.shape != ref32.shape:
        failures.append(f"shape mismatch got {got32.shape} ref {ref32.shape}")
        out_err = float("inf")
    else:
        out_err = float(mx.abs(got32 - ref32).max())
        if out_err > atol:
            failures.append(
                f"primitive out err {out_err:.3e} > {atol:g}")
    # mx bool reductions carry a known llvmpipe defect (receipts/
    # boolall-2026-09-03); finiteness goes through numpy.
    if not np.isfinite(np.asarray(got32)).all():
        failures.append("primitive output has non-finite values")

    status = "PASS" if not failures else "FAIL"
    print(f"[{status}] {name}: out_err={out_err:.3e} "
          f"(scores_err={sim_scores_err:.3e}, probs_err={sim_probs_err:.3e})")
    for failure in failures:
        print(f"    {failure}")
    return not failures


def cache_slice_case(rng, offset, heads, kv_heads, head_dim, q_len=1,
                     dtype=mx.float16):
    """k/v drawn from a longer cache array, sliced - never row-contiguous."""
    max_len = offset + 16
    k_len = offset + q_len
    key_q, key_k, key_v = mx.random.split(rng, 3)
    cache_k = draw(key_k, (1, kv_heads, max_len, head_dim), dtype)
    cache_v = draw(key_v, (1, kv_heads, max_len, head_dim), dtype)
    q = draw(key_q, (1, heads, q_len, head_dim), dtype)
    k = cache_k[:, :, :k_len, :]
    v = cache_v[:, :, :k_len, :]
    scale = 1.0 / head_dim ** 0.5
    repeats = heads // kv_heads
    tag = "bf16 " if dtype == mx.bfloat16 else ""
    name = (f"{tag}cache-slice off={offset} hd={head_dim} "
            f"heads={heads} kv={kv_heads} rep={repeats} q_len={q_len}")
    return check(name, q, k, v, scale)


def gate_enabled(name):
    value = os.environ.get(name)
    return value is not None and value != "0"


def main():
    rng = mx.random.key(0)
    ok = True

    # This suite exercises SDPA only; the RoPE gate is not consulted
    # here (the RoPE bf16 contract is C++-tested in omarchy_fast_ops).
    sdpa_gate_on = gate_enabled("MLX_OMARCHY_SDPA_BF16_FAST")
    print(f"gates: MLX_OMARCHY_SDPA_BF16_FAST={'on' if sdpa_gate_on else 'off'}")
    if "--require-gates" in sys.argv and not sdpa_gate_on:
        print(
            "FAIL: MLX_OMARCHY_SDPA_BF16_FAST is off - the bf16 legs "
            "would ride the f32 composition, so a PASS would not "
            "exercise the fast path. Set the gate and rerun."
        )
        return 1

    # (b) non-contiguous cache slices at offsets 1, 7, 41, 256.
    for offset in (1, 7, 41, 256):
        ok &= cache_slice_case(rng, offset, heads=4, kv_heads=4, head_dim=64)

    # (c) GQA repeat factors 1, 2, 7 over cache slices (the decode shape).
    for heads, kv_heads in ((4, 4), (4, 2), (7, 1)):
        ok &= cache_slice_case(rng, 41, heads, kv_heads, 64)

    # (d) head_dim 48 (not a power-of-two multiple), decode and GQA 7.
    ok &= cache_slice_case(rng, 41, heads=7, kv_heads=1, head_dim=48)
    ok &= cache_slice_case(rng, 7, heads=2, kv_heads=2, head_dim=48)

    # (e) q_len 1 and q_len > 1, dense inputs, batch > 1.
    key_q, key_k, key_v = mx.random.split(rng, 3)
    q = draw(key_q, (1, 4, 16, 48))
    k = draw(key_k, (1, 4, 16, 48))
    v = draw(key_v, (1, 4, 16, 48))
    ok &= check("q_len=16 dense hd=48 heads=4", q, k, v, 1.0 / 48 ** 0.5)
    key_q, key_k, key_v = mx.random.split(rng, 3)
    q = draw(key_q, (2, 7, 5, 64))
    k = draw(key_k, (2, 1, 9, 64))
    v = draw(key_v, (2, 1, 9, 64))
    ok &= check("q_len=5 k_len=9 GQA7 batch=2", q, k, v, 1.0 / 64 ** 0.5)

    # Prefill into an existing cache: q_len > 1 against a cache slice.
    ok &= cache_slice_case(rng, 41, heads=4, kv_heads=2, head_dim=64, q_len=5)

    # (a) decode shapes [1, heads, 1, 64], dense and causal.
    key_q, key_k, key_v = mx.random.split(rng, 3)
    q = draw(key_q, (1, 4, 1, 64))
    k = draw(key_k, (1, 4, 64, 64))
    v = draw(key_v, (1, 4, 64, 64))
    ok &= check("decode dense [1,4,1,64]", q, k, v, 0.125)
    key_q, key_k, key_v = mx.random.split(rng, 3)
    q = draw(key_q, (1, 4, 1, 64))
    k = draw(key_k, (1, 4, 64, 64))
    v = draw(key_v, (1, 4, 64, 64))
    ok &= check("decode causal [1,4,1,64]", q, k, v, 0.125, causal=True)
    key_q, key_k, key_v = mx.random.split(rng, 3)
    q = draw(key_q, (1, 4, 1, 64))
    k = draw(key_k, (1, 2, 64, 64))
    v = draw(key_v, (1, 2, 64, 64))
    ok &= check("decode causal GQA2", q, k, v, 0.125, causal=True)

    # Causal with q_len > 1 and an explicit additive f16 mask.
    key_q, key_k, key_v, key_m = mx.random.split(rng, 4)
    q = draw(key_q, (1, 4, 8, 64))
    k = draw(key_k, (1, 4, 32, 64))
    v = draw(key_v, (1, 4, 32, 64))
    mask = draw(key_m, (1, 4, 8, 32))
    ok &= check("causal q_len=8 k_len=32", q, k, v, 0.125, causal=True)
    ok &= check("additive f16 mask", q, k, v, 0.125, mask=mask)
    # Fully masked row under an additive f16 mask: padding over padded
    # positions - the degenerate case Main called out. The -65504
    # additive guard keeps the row DEFINED on the f16 path (matching
    # the f32 path's own softmax-over-masked-scores semantics), but f16
    # storage at 65504 magnitude has ulp 32, so the recovered weights
    # are quantized against the f32 reference: the honest bar here is
    # finite + bounded at 1e-1, not the normal 1e-3.
    key_q, key_k, key_v, key_m = mx.random.split(rng, 4)
    q = draw(key_q, (1, 4, 4, 64))
    k = draw(key_k, (1, 4, 16, 64))
    v = draw(key_v, (1, 4, 16, 64))
    mask = draw(key_m, (1, 4, 4, 16))
    mask[:, :, 2, :] = -65504.0
    ok &= check("fully-masked row (row 2)", q, k, v, 0.125, mask=mask,
                atol=1e-1, storage_atol=1.0)

    # Score-overflow boundary: magnitudes whose dot products overflow
    # f16 storage after scaling. The primitive must behave exactly
    # like the f16-storage emulation (same non-finite pattern), while
    # the f32 path stays finite - the documented cost of f16 score
    # storage, the same cap upstream Metal f16 sdpa has.
    key_q, key_k, key_v = mx.random.split(rng, 3)
    q = (mx.random.normal((1, 2, 1, 64), key=key_q) * 40.0).astype(mx.float16)
    k = (mx.random.normal((1, 2, 64, 64), key=key_k) * 40.0).astype(mx.float16)
    v = (mx.random.normal((1, 2, 64, 64), key=key_v) * 0.5).astype(mx.float16)
    got = mx.fast.scaled_dot_product_attention(q, k, v, scale=0.125)
    mx.eval(got)
    sim, _, _ = storage_emulation(q, k, v, 0.125)
    got_np = np.asarray(got.astype(mx.float32))
    sim_np = np.asarray(sim.astype(mx.float32))
    same_nan = np.array_equal(np.isnan(got_np), np.isnan(sim_np))
    print(f"[{'PASS' if same_nan else 'FAIL'}] overflow boundary "
          f"q,k~N(0,40): primitive and f16-storage emulation "
          f"{'agree' if same_nan else 'DISAGREE'} on NaN pattern "
          f"(f32 path stays finite here; upstream Metal f16 sdpa has "
          f"the same storage cap)")
    ok &= bool(same_nan)
    # ---- bfloat16 legs: the SDPA bf16 fast path (MLX_OMARCHY_SDPA_
    # BF16_FAST; with the gate off these ride the f32 composition and
    # pass trivially). Same coverage as the f16 legs; bars derived from
    # the bf16 grid by derived_bf16_bars.
    bf = mx.bfloat16

    # Cache slices across offsets and GQA repeats (the decode shapes).
    for offset in (1, 41, 256):
        ok &= cache_slice_case(
            rng, offset, heads=4, kv_heads=4, head_dim=64, dtype=bf)
    for heads, kv_heads in ((4, 2), (7, 1)):
        ok &= cache_slice_case(rng, 41, heads, kv_heads, 64, dtype=bf)
    ok &= cache_slice_case(rng, 41, heads=7, kv_heads=1, head_dim=48, dtype=bf)

    # q_len > 1 dense, batch 2, prefill into a cache slice.
    key_q, key_k, key_v = mx.random.split(rng, 3)
    ok &= check(
        "bf16 q_len=16 dense hd=48 heads=4",
        draw(key_q, (1, 4, 16, 48), bf),
        draw(key_k, (1, 4, 16, 48), bf),
        draw(key_v, (1, 4, 16, 48), bf),
        1.0 / 48 ** 0.5)
    key_q, key_k, key_v = mx.random.split(rng, 3)
    ok &= check(
        "bf16 q_len=5 k_len=9 GQA7 batch=2",
        draw(key_q, (2, 7, 5, 64), bf),
        draw(key_k, (2, 1, 9, 64), bf),
        draw(key_v, (2, 1, 9, 64), bf),
        1.0 / 64 ** 0.5)
    ok &= cache_slice_case(
        rng, 41, heads=4, kv_heads=2, head_dim=64, q_len=5, dtype=bf)

    # Decode shapes, dense and causal.
    key_q, key_k, key_v = mx.random.split(rng, 3)
    ok &= check(
        "bf16 decode dense [1,4,1,64]",
        draw(key_q, (1, 4, 1, 64), bf),
        draw(key_k, (1, 4, 64, 64), bf),
        draw(key_v, (1, 4, 64, 64), bf),
        0.125)
    key_q, key_k, key_v = mx.random.split(rng, 3)
    ok &= check(
        "bf16 decode causal [1,4,1,64]",
        draw(key_q, (1, 4, 1, 64), bf),
        draw(key_k, (1, 4, 64, 64), bf),
        draw(key_v, (1, 4, 64, 64), bf),
        0.125,
        causal=True)
    key_q, key_k, key_v = mx.random.split(rng, 3)
    ok &= check(
        "bf16 decode causal GQA2",
        draw(key_q, (1, 4, 1, 64), bf),
        draw(key_k, (1, 2, 64, 64), bf),
        draw(key_v, (1, 2, 64, 64), bf),
        0.125,
        causal=True)

    # Causal q_len > 1 and an explicit additive bf16 mask.
    key_q, key_k, key_v, key_m = mx.random.split(rng, 4)
    q = draw(key_q, (1, 4, 8, 64), bf)
    k = draw(key_k, (1, 4, 32, 64), bf)
    v = draw(key_v, (1, 4, 32, 64), bf)
    mask = draw(key_m, (1, 4, 8, 32), bf)
    ok &= check("bf16 causal q_len=8 k_len=32", q, k, v, 0.125, causal=True)
    ok &= check("bf16 additive mask", q, k, v, 0.125, mask=mask)

    # Fully masked row: the bf16 floor is exact in storage (the score
    # addend is absorbed identically on both paths), the row reduces to
    # a uniform softmax on both sides, and the derived bars from the
    # kept scores govern.
    key_q, key_k, key_v, key_m = mx.random.split(rng, 4)
    q = draw(key_q, (1, 4, 4, 64), bf)
    k = draw(key_k, (1, 4, 16, 64), bf)
    v = draw(key_v, (1, 4, 16, 64), bf)
    mask = draw(key_m, (1, 4, 4, 16), bf)
    mask[:, :, 2, :] = BF16_FLOOR
    ok &= check("bf16 fully-masked row (row 2)", q, k, v, 0.125, mask=mask)

    # Overflow boundary: at magnitudes that overflow f16 score storage
    # (the f16 leg above NaNs), bf16 stays finite - it shares float32's
    # exponent, so every finite f32 scaled score is representable. The
    # primitive must be finite AND within the derived bf16 bars of the
    # f32 reference.
    key_q, key_k, key_v = mx.random.split(rng, 3)
    q = (mx.random.normal((1, 2, 1, 64), key=key_q) * 40.0).astype(bf)
    k = (mx.random.normal((1, 2, 64, 64), key=key_k) * 40.0).astype(bf)
    v = (mx.random.normal((1, 2, 64, 64), key=key_v) * 0.5).astype(bf)
    ok &= check(
        "bf16 overflow boundary q,k~N(0,40) stays finite", q, k, v, 0.125)

    print("ALL PASS" if ok else "SUITE FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
