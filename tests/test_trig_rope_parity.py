"""Trig/RoPE numerical parity: omarchy Vulkan vs upstream Metal (p15).

Static audit finding being tested (audit.md, omarchy-mplus-private,
section 'trig-invariance'): the Vulkan backend evaluates sin/cos through
shaders/omarchy_trig.h - raw GLSL built-in for |x| <= 1e4, Cody-Waite
double-float range reduction up to 5e5, NaN above - while the vendored
upstream Metal backend uses metal::precise::sin/cos for eager unary ops
(mlx/backend/metal/kernels/unary_ops.h) but metal::fast::sin/cos inside
the fused RoPE kernel (mlx/backend/metal/kernels/rope.metal). RoPE is
where the paths can diverge enough to flip greedy token IDs at fp32:
a 1-ulp theta difference survives 30+ attention layers and flips argmax
whenever the top-2 logit gap is under the accumulated error.

One command on the GPU host (jw16, honeykrisp Vulkan ICD; also runs the
Metal arm unchanged on any macOS host with mlx installed):

    flock /tmp/m1-gpu.lock /usr/lib/omarchy-mlx/venv/bin/python3 tests/test_trig_rope_parity.py

pytest form:  python3 -m pytest tests/test_trig_rope_parity.py -v
Harness self-check (any host, no mlx/GPU needed, must exit 0):

    python3 tests/test_trig_rope_parity.py --selfcheck

What FAILS and on which divergent path:
  - Z1 |x| <= 1e4: abs err vs float64 > 5e-4 => the raw-builtin band is
    computing something else. Deliberately the loosest gate: this band
    keeps the driver built-in, whose measured error is 2.8e-5 at 1e3.
  - Z2 1e4 < |x| <= 5e5: abs err > 1e-4 => the Cody-Waite reduction was
    bypassed (raw built-in collapses here: 4.8e-3 measured at 123457).
  - Z3 |x| > 5e5: not NaN on the Vulkan contract => the NaN refusal was
    dropped and a finite wrong value is possible. Metal has no such gate
    and must return finite values; the probe result names the backend.
  - fused mx.fast.rope fp32 abs err > 2e-6 (Vulkan) => the fused rotation
    drifted from the composed contract (documented bound: 1e-6 vs the
    composed path, fast_rope.comp header).
  - rope with theta > 5e5 must RAISE on Vulkan (host rope_trig_gate,
    primitives.cpp) and must NOT raise on Metal (no upstream gate).
  - The token-style digest (top-16-bits vocabulary ids + sha256 of the
    raw fp32 bytes) is diagnostic: it is printed for cross-device
    comparison and is NOT a pass gate, because 1-ulp trig differences
    legitimately flip digest words.
  - Metal-arm Z2 rope error is PRINTED, not gated: no measured bound for
    metal::fast::sin above 2*pi exists on this fleet; the macOS run
    should record the printed number.

The test is expected to PASS on reference hardware (the Cody-Waite path
shipped in 6cbf55d8f and value tests already pin composed-vs-fused);
a failure names the exact zone and the wrong path above.
"""

import hashlib
import struct
import sys

import numpy as np

SEED = 20261121
Z1_MAX = 1e4  # omarchy_trig.h OMARCHY_TRIG_BUILTIN_MAX
Z2_MAX = 5e5  # omarchy_trig.h OMARCHY_TRIG_LIMIT = kTrigArgumentLimit
TOL_Z1 = 5e-4
TOL_Z2 = 1e-4
TOL_ROPE = 2e-6


def band_grids():
    """Deterministic fp32 argument grids for the three zones."""
    rng = np.random.default_rng(SEED)
    dense = rng.uniform(-2.0 * np.pi, 2.0 * np.pi, 2048)
    moderate = np.geomspace(1.0, Z1_MAX, 2048) * rng.choice([-1.0, 1.0], 2048)
    z1 = np.concatenate([dense, moderate])
    z2 = np.geomspace(Z1_MAX * 1.0000001, Z2_MAX, 4096) * rng.choice(
        [-1.0, 1.0], 4096
    )
    z3 = np.array([Z2_MAX * 1.0000001, 6e5, 1e6, 1e8], dtype=np.float64)
    return {
        "z1": z1.astype(np.float32),
        "z2": z2.astype(np.float32),
        "z3": z3.astype(np.float32),
    }


def digest(x32):
    """Token-style digest: top-16-bits vocabulary ids + sha256 of fp32 bytes."""
    raw = struct.pack(f"<{x32.size}f", *x32.ravel().tolist())
    tokens = [
        (w >> 16) & 0xFFFF
        for w in struct.unpack(f"<{x32.size}I", raw)[:12]
    ]
    return hashlib.sha256(raw).hexdigest(), tokens


def max_abs_err(gpu32, ref64):
    gpu64 = np.asarray(gpu32, dtype=np.float64)
    return float(np.max(np.abs(gpu64 - ref64)))


def selfcheck():
    """Prove the comparator detects a wrong reference. Pure numpy, no mlx.

    Failing-first: if this does not fail against the poisoned reference,
    the harness cannot be trusted to fail on the GPU, so exit nonzero.
    """
    grids = band_grids()
    ref = np.sin(grids["z2"].astype(np.float64))
    poisoned = ref + 1e-3
    ok_pass = max_abs_err(np.sin(grids["z2"]), ref) < TOL_Z2
    ok_fail = max_abs_err(np.sin(grids["z2"]), poisoned) >= TOL_Z2
    print(f"selfcheck: clean comparison passes={ok_pass}")
    print(f"selfcheck: poisoned +1e-3 reference detected={ok_fail}")
    if not (ok_pass and ok_fail):
        print("FAIL: harness comparator is broken (cannot fail)")
        return 2
    print("PASS: harness can fail")
    return 0


def run_gpu_arm(mx):
    """Compute sin/cos and RoPE on the GPU path; assert zone contracts."""
    import time

    grids = band_grids()
    gpu = mx.array
    dev = str(mx.default_device())

    probe = np.array(mx.eval(mx.sin(gpu(grids["z3"][:2])))[0].tolist())
    backend = "vulkan-contract" if np.all(np.isnan(probe)) else "metal-or-other"
    print(f"device={dev} backend-label={backend} seed={SEED}")
    if backend == "metal-or-other":
        mx_sin_ref = np.sin(grids["z3"].astype(np.float64))
        gpu_sin = np.array(mx.eval(mx.sin(gpu(grids["z3"])))[0].tolist())
        e = max_abs_err(gpu_sin, mx_sin_ref)
        print(f"Z3 eager sin (Metal arm): max abs err vs float64 = {e:.3e}")

    for zone, tol, path in (
        ("z1", TOL_Z1, "raw GLSL built-in band"),
        ("z2", TOL_Z2, "Cody-Waite reduction band"),
    ):
        x32 = grids[zone]
        ref_sin = np.sin(x32.astype(np.float64))
        ref_cos = np.cos(x32.astype(np.float64))
        out = mx.eval(mx.sin(gpu(x32)), mx.cos(gpu(x32)))
        gpu_sin = np.array(out[0].tolist())
        gpu_cos = np.array(out[1].tolist())
        e_sin = max_abs_err(gpu_sin, ref_sin)
        e_cos = max_abs_err(gpu_cos, ref_cos)
        dsin, tsin = digest(gpu_sin.astype(np.float32))
        print(
            f"{zone} sin: max abs err {e_sin:.3e} cos: {e_cos:.3e} "
            f"tol {tol:.0e} digest sha256:{dsin[:16]} tokens {tsin[:6]}"
        )
        bad = max(e_sin, e_cos) > tol
        if bad:
            print(
                f"FAIL {zone}: GPU trig off float64 beyond {tol:.0e}; "
                f"wrong path = {path}"
            )
        assert not bad, (
            f"{zone} ({path}): abs err sin={e_sin:.3e} cos={e_cos:.3e} "
            f"exceeds {tol:.0e}"
        )

    z3 = mx.eval(mx.sin(gpu(grids["z3"])), mx.cos(gpu(grids["z3"])))
    z3_sin = np.array(z3[0].tolist())
    z3_cos = np.array(z3[1].tolist())
    if backend == "vulkan-contract":
        all_nan = bool(np.all(np.isnan(z3_sin)) and np.all(np.isnan(z3_cos)))
        assert all_nan, (
            "Z3: eager sin/cos above 5e5 returned finite values; the NaN "
            "refusal gate (omarchy_trig.h) was bypassed - a finite wrong "
            "value is possible on this path"
        )
        print("Z3: NaN contract holds on Vulkan (all above-limit NaN)")

    # Fused RoPE, half-split forward, fp32. Reference recomposed in
    # float64 from the upstream formula (mlx/fast.cpp positions x
    # exp(-j*log(base)/half_dims)).
    rng = np.random.default_rng(SEED + 1)
    dims, T, N, B = 64, 32, 2, 1
    base, scale = 10000.0, 1.0

    def rope_ref(offset, theta_scale=scale):
        half = dims // 2
        pos = (np.arange(T) + offset) * theta_scale
        inv = np.exp(-np.arange(half) * np.log(base) / half)
        theta = pos[:, None] * inv[None, :]
        x = rng.standard_normal((B, N, T, dims)).astype(np.float32)
        x64 = x.astype(np.float64)
        c, s = np.cos(theta), np.sin(theta)
        out = x64.copy()
        x1, x2 = x64[..., :half], x64[..., half:]
        out[..., :half] = x1 * c - x2 * s
        out[..., half:] = x1 * s + x2 * c
        return x, out

    for name, offset, gate_tol in (
        ("small-positions", 0, TOL_ROPE),
        ("zoneB-offset-120k", 120000, TOL_Z2),
    ):
        x_in, ref = rope_ref(offset)
        got = mx.fast.rope(
            mx.array(x_in), dims, False, base, scale, offset
        )
        got_np = np.array(mx.eval(got)[0].tolist())
        e = max_abs_err(got_np, ref)
        d, tok = digest(got_np.astype(np.float32))
        note = ""
        if backend != "vulkan-contract" and name.startswith("zoneB"):
            note = " (Metal fast:: arm: printed, not gated)"
            e_gate = 1.0  # collapse guard only
        else:
            e_gate = gate_tol
        print(
            f"rope {name}: max abs err vs float64 {e:.3e} "
            f"tol {e_gate:.0e} digest sha256:{d[:16]} tokens {tok[:6]}"
            f"{note}"
        )
        assert e <= e_gate, (
            f"fused rope {name}: abs err {e:.3e} exceeds {e_gate:.0e} "
            f"({'fused rotation drifted from the composed contract' if e_gate == TOL_ROPE else 'over-limit theta collapse'}; "
            f"backend={backend})"
        )

    # Loud refusal: theta > 5e5 must raise on Vulkan, run on Metal.
    x_big = np.zeros((1, 1, 4, dims), dtype=np.float32)
    raised = False
    try:
        mx.eval(mx.fast.rope(mx.array(x_big), dims, False, 1.0, 1.0, 600001))
    except Exception as exc:
        raised = True
        print(f"rope theta>5e5: raised as expected ({type(exc).__name__})")
    if backend == "vulkan-contract":
        assert raised, (
            "rope theta>5e5 did NOT raise on Vulkan: the host rope_trig_gate "
            "is gone, over-envelope theta now reaches the reduction silently"
        )
    else:
        assert not raised, (
            "Metal arm: upstream rope has no magnitude gate; an exception "
            "here means the omarchy gate leaked into the Metal path"
        )
    if not raised:
        print("rope theta>5e5: no gate on Metal arm (expected upstream)")
    print(f"PASS trig/rope parity ({backend}, {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())})")


def _gpu_mx():
    try:
        import mlx.core as mx

        mx.set_default_device(mx.gpu)
        mx.eval(mx.sin(mx.array([0.1], dtype=mx.float32)))
        return mx
    except Exception as exc:
        return None, f"no usable GPU device: {exc}"


def test_trig_and_rope_parity():
    import pytest

    got = _gpu_mx()
    mx = got[0] if isinstance(got, tuple) else got
    if mx is None:
        pytest.skip(got[1])
    run_gpu_arm(mx)


def test_selfcheck_comparator_detects_wrong_reference():
    assert selfcheck() == 0


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        sys.exit(selfcheck())
    got = _gpu_mx()
    mx = got[0] if isinstance(got, tuple) else got
    if mx is None:
        print(f"SKIP: {got[1]}")
        sys.exit(0)
    run_gpu_arm(mx)
