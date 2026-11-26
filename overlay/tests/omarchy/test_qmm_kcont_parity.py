#!/usr/bin/env python3
# Copyright 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Pin the QMM decode geometry parity rule for 2x16 vs 2x4 and 2x16kcont.

Rule:
  - Default-eligible decode geometries (2x16, etc.) MUST match 2x4 bit for bit
    on every matrix shape (0 differing bf16 elements).
  - Experimental 2x16kcont (QmmVecQ4MultiSubgroupBF16R2S16KCONT, -DQMM_KCONT=1)
    replaces the strided per-lane walk (lane*2 + it*64) with contiguous K-chunks
    (lane*(wpr/32) + it*2). It matches 2x4 at K=512 (Q4_FAST_ITERS=1) and K=768
    (Q4_KCONT_OK=0 fallback), but reassociates FP32 per-lane sums at K >= 1024
    (Q4_FAST_ITERS >= 2), producing deterministic +/-1 bf16 ULP differences on
    32-lane Vulkan GPUs. Therefore 2x16kcont is NOT bit-identical and must stay
    excluded from every default selection path.

Usage:
  python3 overlay/tests/omarchy/test_qmm_kcont_parity.py
      Verifies the rule (exits 0 when 2x16 matches 2x4 bit-for-bit, k512/k768
      match under 2x16kcont, and k1024/k4096 diverge by at most 1 bf16 ULP on
      32-lane devices).
  python3 overlay/tests/omarchy/test_qmm_kcont_parity.py --fail-on-kcont-diff
      Treats 2x16kcont diffs as failures (exits 1 deterministically on M6/any
      32-lane device, demonstrating why 2x16kcont is disqualified from defaults).
"""
import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


def run_child(k, ns, seed, geom, out_npz):
    env = dict(os.environ, MLX_OMARCHY_QMM_GEOM=geom)
    subprocess.run(
        [
            sys.executable,
            __file__,
            "--child",
            str(k),
            ",".join(map(str, ns)),
            str(seed),
            str(out_npz),
        ],
        env=env,
        check=True,
    )


def child_main(k, ns, seed, out_npz):
    import mlx.core as mx

    if not mx.metal.is_available() and str(mx.default_device()) != "Device(gpu, 0)":
        mx.set_default_device(mx.gpu)
    rng = np.random.default_rng(seed)
    x = mx.array(rng.normal(0.0, 0.6, size=(1, 1, k)).astype(np.float32)).astype(
        mx.bfloat16
    )
    mats = []
    for n in ns:
        wq, sc, bi = mx.quantize(
            mx.array(rng.normal(0.0, 0.05, size=(n, k)).astype(np.float32)),
            group_size=64,
            bits=4,
        )
        mats.append((wq, sc.astype(mx.bfloat16), bi.astype(mx.bfloat16)))
    mx.eval(x, *[t for m in mats for t in m])
    outs = [
        mx.quantized_matmul(x, *m, transpose=True, group_size=64, bits=4)
        for m in mats
    ]
    mx.eval(*outs)
    np.savez(
        out_npz,
        **{f"o{i}": np.asarray(o.view(mx.uint16)) for i, o in enumerate(outs)},
    )


def bf16_ulp_diff(a_u16, b_u16):
    def to_ord(u):
        u32 = u.astype(np.int32)
        return np.where((u32 & 0x8000) != 0, 0x8000 - (u32 & 0x7FFF), 0x8000 + u32)

    return np.abs(to_ord(a_u16) - to_ord(b_u16))


def compare(path_a, path_b):
    za, zb = np.load(path_a), np.load(path_b)
    diffs = []
    for key in sorted(za.files):
        a, b = za[key].ravel(), zb[key].ravel()
        idx = np.flatnonzero(a != b)
        if idx.size:
            ulps = bf16_ulp_diff(a[idx], b[idx])
            fa = (a[idx].astype(np.uint32) << np.uint32(16)).view(np.float32)
            fb = (b[idx].astype(np.uint32) << np.uint32(16)).view(np.float32)
            diffs.append(
                {
                    "out": key,
                    "ndiff": int(idx.size),
                    "total": int(a.size),
                    "max_ulp": int(ulps.max()),
                    "first_col": int(idx[0]),
                    "bits_2x4": f"0x{int(a[idx[0]]):04x}",
                    "bits_arm": f"0x{int(b[idx[0]]):04x}",
                    "max_abs": float(np.max(np.abs(fb - fa))),
                }
            )
    return diffs


def main():
    if len(sys.argv) == 6 and sys.argv[1] == "--child":
        child_main(
            int(sys.argv[2]),
            [int(v) for v in sys.argv[3].split(",")],
            int(sys.argv[4]),
            sys.argv[5],
        )
        return

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--fail-on-kcont-diff",
        action="store_true",
        help="Fail (exit 1) when 2x16kcont diverges from 2x4 (reproduces the parity failure).",
    )
    args = ap.parse_args()

    failures = 0
    kcont_diverged_at_k_ge_1024 = False
    cases = [
        ("k512_iters1", 512, (4096, 1024), 102, True),
        ("k768_fallback", 768, (4096, 1024), 103, True),
        ("k1024_iters2", 1024, (4096, 1024), 104, False),
        ("k4096_9b_gateup", 4096, (12288, 12288), 106, False),
    ]
    with tempfile.TemporaryDirectory(prefix="qmm-kcont-ut-") as td:
        td = Path(td)
        for label, k, ns, seed, kcont_must_match in cases:
            p_base = td / f"{label}_2x4.npz"
            run_child(k, ns, seed, "2x4", p_base)

            # Rule 1: 2x16 MUST match 2x4 bit for bit on every shape.
            p_2x16 = td / f"{label}_2x16.npz"
            run_child(k, ns, seed, "2x16", p_2x16)
            d16 = compare(p_base, p_2x16)
            if d16:
                failures += 1
                print(f"FAIL {label} 2x16 vs 2x4: {d16}", flush=True)
            else:
                print(f"PASS {label} 2x16 vs 2x4: 0 diffs (bit-identical)", flush=True)

            # Rule 2: 2x16kcont matches at K=512 (iters=1) and K=768 (fallback),
            # and reassociates by +/-1 bf16 ULP at K >= 1024 (not bit-identical).
            p_kc = td / f"{label}_2x16kcont.npz"
            run_child(k, ns, seed, "2x16kcont", p_kc)
            dk = compare(p_base, p_kc)
            if kcont_must_match:
                if dk:
                    failures += 1
                    print(f"FAIL {label} 2x16kcont vs 2x4 (expected 0 diffs): {dk}", flush=True)
                else:
                    print(f"PASS {label} 2x16kcont vs 2x4: 0 diffs (bit-identical)", flush=True)
            else:
                if dk:
                    kcont_diverged_at_k_ge_1024 = True
                    max_ulp = max(d["max_ulp"] for d in dk)
                    if args.fail_on_kcont_diff or max_ulp > 1:
                        failures += 1
                        print(f"FAIL {label} 2x16kcont vs 2x4: {dk}", flush=True)
                    else:
                        print(
                            f"EXPECTED-NON-IDENTICAL {label} 2x16kcont vs 2x4 (max_ulp={max_ulp}): {dk}",
                            flush=True,
                        )
                else:
                    print(f"PASS {label} 2x16kcont vs 2x4: 0 diffs (fallback/non-32-lane device)", flush=True)

    print(
        f"SUMMARY failures={failures} kcont_diverged_at_k_ge_1024={kcont_diverged_at_k_ge_1024}",
        flush=True,
    )
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
