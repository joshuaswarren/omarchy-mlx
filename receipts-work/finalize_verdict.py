#!/usr/bin/env python3
"""Finalize the BF16 decode attribution receipt from measured data.

Reads arms/legs.ndjson (all arms), screen/*.ndjson (span screen),
probe outputs, writes verdict.json + prints the final markdown table
block for the README. Run AFTER window B completes.
"""
import json
import statistics
import sys
from pathlib import Path

EXPECTED = {
    "short-decode-32": "f26175202f3dabe9",
    "long-decode-128": "8690dc83246b39f8",
    "longctx-1024-decode-32": "ff502900d2a179a5",
}
CENSUS = {
    "gemv": ["MatmulVecBF16", "MatmulBF16"],
    "attn": ["MatmulF32", "MatmulF32Coopmat", "SoftmaxF32"],
    "cast": ["CastBF16F32", "CastF32BF16"],
    "ewise": ["ElementwiseF32"],
    "copy": ["CopyGeneralBF16"],
    "rope": ["FastRopeF32"],
    "rms": ["FastRmsNormBF16"],
    "swiglu": ["SwigluBF16"],
    "sampler": ["LogSumExpBF16", "ArgReduceBF16"],
}
COUNTS = {
    "gemv": 169, "attn": 72, "cast": 192, "ewise": 24, "copy": 96,
    "rope": 48, "rms": 49, "swiglu": 24, "sampler": 2,
}
ORDER = ["gemv", "attn", "cast", "ewise", "copy", "rope", "rms",
         "swiglu", "sampler"]
NATIVE = {"short-decode-32": 56.43, "long-decode-128": 55.72,
          "longctx-1024-decode-32": 54.55}
LEGS = ["short-decode-32", "long-decode-128", "longctx-1024-decode-32"]


def main():
    root = Path(sys.argv[1])
    legs = [json.loads(l) for l in
            (root / "arms/legs.ndjson").read_text().splitlines() if l.strip()]
    by = {}
    for r in legs:
        by.setdefault(r["workload"], {}).setdefault(r["arm"], []).append(r)

    verdict = {"schema": "bf16-decode-attribution/1", "arms": {}, "workloads": {}}
    for wl in LEGS:
        arms = by.get(wl, {})
        wl_out = {}
        base_ms = 1000.0 / statistics.median(
            [r["decode_tok_s"] for r in arms.get("baseline", [])])
        marginals = {}
        skeleton = None
        for arm, rows in sorted(arms.items()):
            tok = [r["decode_tok_s"] for r in rows]
            ms = 1000.0 / statistics.median(tok)
            digests = sorted({r["digest"] for r in rows})
            wl_out[arm] = {
                "n": len(rows), "tok_s_median": round(statistics.median(tok), 3),
                "token_ms": round(ms, 4), "digests": digests,
            }
            if arm == "baseline":
                assert digests == [EXPECTED[wl]], (wl, digests)
            elif arm == "all":
                skeleton = ms
            elif arm != "bf16fast":
                marginals[arm] = base_ms - ms
        total = sum(marginals.values()) + (skeleton or 0.0)
        wl_out["marginals_ms_token"] = {a: round(marginals[a], 4)
                                        for a in ORDER if a in marginals}
        wl_out["skeleton_ms_token"] = round(skeleton, 4) if skeleton else None
        if skeleton:
            wl_out["sum_check_ms_token"] = round(total, 4)
            wl_out["residual_ms_token"] = round(total - base_ms, 4)
            wl_out["residual_pct"] = round(100 * (total - base_ms) / base_ms, 1)
        wl_out["baseline_fraction_of_native"] = round(
            statistics.median([r["decode_tok_s"]
                               for r in arms["baseline"]]) / NATIVE[wl], 3)
        verdict["workloads"][wl] = wl_out
        verdict["arms"][wl] = {a: len(r) for a, r in arms.items()}

    # screen summary
    screen_dir = root / "screen"
    if screen_dir.exists():
        screen = {}
        for f in sorted(screen_dir.glob("*.ndjson")):
            if f.name == "digests.ndjson":
                continue
            label = f.stem
            rounds = [json.loads(l) for l in f.read_text().splitlines()]
            by_shape = {}
            for r in rounds:
                by_shape.setdefault(r["shape"], []).append(r["wall_s"])
            screen[label] = {s: round(statistics.median(v), 3)
                             for s, v in by_shape.items()}
        verdict["screen_wall_ms_median"] = screen
        digests = [json.loads(l) for l in
                   (screen_dir / "digests.ndjson").read_text().splitlines()]
        identical = {}
        for shape in {d["shape"] for d in digests}:
            vals = {(d["digest"], tuple(d["first16"])) for d in digests
                    if d["shape"] == shape}
            identical[shape] = len(vals) == 1
        verdict["screen_outputs_bit_identical"] = identical
    (root / "verdict.json").write_text(json.dumps(verdict, indent=1) + "\n")
    print(json.dumps(verdict, indent=1))


if __name__ == "__main__":
    main()
