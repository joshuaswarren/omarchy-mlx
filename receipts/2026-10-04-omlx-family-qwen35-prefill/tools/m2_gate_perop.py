#!/usr/bin/env python3
"""FamQwen35Prefill M2 numerics gate — per-op leg.

Runs on the T6021 M2 inside a gpu-turn ticket against a private copy of
the shared venv with this lane's wheel force-installed.

One process = one arm: the omarchy SdpaPrefillFlash256BF16 dispatch gate
is a function-local static initialized at the first SDPA eval, so the
composed arm and the flash arm must run in SEPARATE processes. The
caller runs this script twice:

    MLX_OMARCHY_SDPA_PREFILL_FLASH256=0 python3 m2_gate_perop.py --mode composed --out composed.npz
    python3 m2_gate_perop.py --mode flash --out flash.npz

then `python3 m2_gate_perop.py --compare composed.npz flash.npz` (host
numpy only) prints the gate verdict.

Shapes: Qwen3.5-2B (GQA 8/2) and 9B (GQA 16/4) full-attention prefill
at qL=kL in {512, 1024}, head_dim=256, bf16 storage. The fp64 host
reference runs on the same fp32 inputs the device bf16 cast came from
(bf16 cast is exact: bf16 is the fp32 top half), so the only deltas are
(a) bf16 input rounding, shared by both arms, and (b) summation order,
which is what the gate compares.
"""

import argparse
import os
import sys

import numpy as np


def ulp_bf16(x):
    """bf16 ULP per docs/numerics-gate.md: 2^(floor(log2|x|)-7)."""
    mag = np.maximum(np.abs(x), 1e-30)
    return np.exp2(np.floor(np.log2(mag)) - 7.0)


def bf16_round_f64(x):
    """RNE fp32 -> bf16 -> fp64, the words the device arms load."""
    bits = x.astype(np.float32).view(np.uint32)
    rne = (bits + 0x7FFF + ((bits >> 16) & 1)) >> 16
    return (rne << 16).view(np.float32).astype(np.float64)


def host_reference(q, k, v, scale):
    """fp64 causal SDPA on the bf16-rounded inputs. q [H,qL,D], k/v
    [KV,kL,D] fp32 numpy. The first lavapipe run (2026-10-04) fed the
    fp32 originals and measured a common-mode bf16 input-rounding error
    up to ~96 ULP on BOTH arms at identical elements — input-cast
    signature, not kernel error. Reference must consume the cast."""
    H, qL, D = q.shape
    KV, kL, _ = k.shape
    rep = H // KV
    q64, k64, v64 = bf16_round_f64(q), bf16_round_f64(k), bf16_round_f64(v)
    out = np.zeros((H, qL, D), dtype=np.float64)
    offset = kL - qL  # causal_offset: query i attends keys <= i + offset
    for kv in range(KV):
        for r in range(rep):
            head = kv * rep + r
            s = (q64[head] @ k64[kv].T) * np.float64(scale)
            qi = np.arange(qL)[:, None]
            ki = np.arange(kL)[None, :]
            s = np.where(ki <= qi + offset, s, -np.inf)
            m = s.max(axis=-1, keepdims=True)
            p = np.exp(s - m)
            p /= p.sum(axis=-1, keepdims=True)
            out[head] = p @ v64[kv]
    return out


def run_arm(mode, out_path):
    import mlx.core as mx

    shapes = [
        (8, 2, 512, 512, 311),
        (8, 2, 1024, 1024, 313),
        (16, 4, 512, 512, 317),
        (16, 4, 1024, 1024, 319),
        # Tail + ragged classes (2026-10-05: the model-scale divergence
        # hunt showed tails and tile boundaries need explicit coverage;
        # the first battery was exact multiples of the tile only).
        (8, 2, 16, 16, 331),
        (8, 2, 17, 17, 333),
        (8, 2, 31, 31, 337),
        (8, 2, 32, 32, 339),
        (8, 2, 33, 33, 341),
        (8, 2, 53, 53, 343),
        (8, 2, 53, 117, 347),
        (16, 4, 53, 53, 349),
    ]
    results = {}
    keys = []
    for H, KV, QL, KL, seed in shapes:
        D = 256
        rng = np.random.default_rng(seed)
        q = rng.uniform(-1.0, 1.0, (H, QL, D)).astype(np.float32)
        k = rng.uniform(-1.0, 1.0, (KV, KL, D)).astype(np.float32)
        v = rng.uniform(-1.0, 1.0, (KV, KL, D)).astype(np.float32)
        scale = np.float32(1.0 / np.sqrt(D))
        qb = mx.array(q).astype(mx.bfloat16)
        kb = mx.array(k).astype(mx.bfloat16)
        vb = mx.array(v).astype(mx.bfloat16)
        out = mx.fast.scaled_dot_product_attention(
            qb[None], kb[None], vb[None], scale=scale.item(), mask="causal"
        )
        mx.eval(out)
        o = np.array(out.astype(mx.float32)[0])
        key = f"H{H}_KV{KV}_q{QL}_k{KL}"
        keys.append(key)
        results[key] = {"q": q, "k": k, "v": v, "out": o}
        print(f"[{mode}] H={H} KV={KV} qL={QL} kL={KL} done, "
              f"out sha16={hash(o.tobytes()) & 0xFFFF:04x}", flush=True)
    np.savez_compressed(out_path, **{f"{name}_{field}": arr
                                     for name, d in results.items()
                                     for field, arr in d.items()})
    print(f"[{mode}] saved {out_path}", flush=True)


def compare(composed_path, flash_path):
    c = np.load(composed_path)
    f = np.load(flash_path)
    gate_ok = True
    report = {}
    keys = sorted({name[:-4] for name in c.files if name.endswith("_out")})
    for name in keys:
        q, k, v = c[f"{name}_q"], c[f"{name}_k"], c[f"{name}_v"]
        oc, of = c[f"{name}_out"], f[f"{name}_out"]
        scale = 1.0 / np.sqrt(q.shape[-1])
        ref = host_reference(q, k, v, scale)
        u = ulp_bf16(ref)
        abs_c = np.abs(oc.astype(np.float64) - ref)
        abs_f = np.abs(of.astype(np.float64) - ref)
        max_c, max_f = float(abs_c.max()), float(abs_f.max())
        # Gate contract (numerics-gate.md): flash's fp64 error no worse
        # than the deployed (composed) path's, with 2x summation-order
        # slack and a 1e-2 absolute sanity floor. bf16 output storage
        # puts both arms at ~1 output-ULP (~2e-3 at unit scale); a
        # per-element 16-ULP-vs-fp64 bound is not the contract (both
        # f32-accumulating arms sit ~40 ULPs off at cancellation-heavy
        # elements — symmetric, inherent). Arm-to-arm: |f-c| <= 8 ULP
        # + 1e-7 per element (mirrors the C++ gate).
        ok_noworse = max_f <= 2.0 * max(max_c, 1e-30) and max_f <= 1e-2
        arm_diff = np.abs(of.astype(np.float64) - oc.astype(np.float64))
        ok_arm_agree = bool((arm_diff <= 8.0 * u + 1e-7).all())
        # Bitwise digest between arms (informational: on llvmpipe the
        # serialized execution made the two orders coincide; on the M2
        # a ULP-scale difference is expected and is engagement
        # evidence — two different dispatch bodies ran).
        identical = bool(np.array_equal(oc, of))
        row = {
            "max_abs_composed": max_c,
            "max_abs_flash": max_f,
            "flash_no_worse_than_composed": bool(ok_noworse),
            "arms_within_8ulp_plus_1e-7": ok_arm_agree,
            "max_arm_diff": float(arm_diff.max()),
            "arms_bitwise_identical": identical,
        }
        report[name] = row
        gate_ok &= ok_noworse and ok_arm_agree
        print(name, row, flush=True)
    print("GATE_PEROP:", "PASS" if gate_ok else "FAIL", flush=True)
    return gate_ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["composed", "flash"])
    ap.add_argument("--out")
    ap.add_argument("--compare", nargs=2, metavar=("COMPOSED", "FLASH"))
    args = ap.parse_args()
    if args.compare:
        ok = compare(*args.compare)
        sys.exit(0 if ok else 1)
    if not args.mode or not args.out:
        ap.error("--mode and --out required (or --compare)")
    if args.mode == "composed":
        os.environ["MLX_OMARCHY_SDPA_PREFILL_FLASH256"] = "0"
    run_arm(args.mode, args.out)


if __name__ == "__main__":
    main()
