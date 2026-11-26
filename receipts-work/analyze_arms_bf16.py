#!/usr/bin/env python3
"""BF16 decode attribution table from arms/legs.ndjson.

Produces, per workload: baseline token ms, per-class marginal
(baseline median minus ablated median), the all-ablated skeleton, the
sum check, and the explicit unattributed residual. Cross-references the
census per-token dispatch counts per class.
"""
import json
import statistics
import sys
from pathlib import Path

CENSUS = {  # median decode-token dispatch counts per class (per-token)
    # class -> (kernels, count/token at short and ctx1024)
    "gemv": (["MatmulVecBF16", "MatmulBF16"], 97 + 72),
    "attn": (["MatmulF32", "MatmulF32Coopmat", "SoftmaxF32"], 48 + 24),
    "cast": (["CastBF16F32", "CastF32BF16"], 120 + 72),
    "ewise": (["ElementwiseF32"], 24),
    "copy": (["CopyGeneralBF16"], 96),
    "rope": (["FastRopeF32"], 48),
    "rms": (["FastRmsNormBF16"], 49),
    "swiglu": (["SwigluBF16"], 24),
    "sampler": (["LogSumExpBF16", "ArgReduceBF16"], 2),
}
TOTAL_DISPATCHES = 726
CLASS_ORDER = ["gemv", "attn", "cast", "ewise", "copy", "rope", "rms",
               "swiglu", "sampler"]


def med(xs):
    return statistics.median(xs)


def main():
    ndjson = Path(sys.argv[1])
    out_md = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    arms = {}
    digests = {}
    for line in ndjson.read_text().splitlines():
        r = json.loads(line)
        arms.setdefault(r["workload"], {}).setdefault(r["arm"], []).append(
            r["decode_tok_s"])
        digests.setdefault(r["workload"], {}).setdefault(
            r["arm"], set()).add(r["digest"])

    lines = []
    for wl, by_arm in arms.items():
        lines.append(f"## {wl}")
        lines.append("| arm | n | tok/s median | token ms | digests |")
        lines.append("|---|---|---|---|---|")
        base_ms = None
        marginals = {}
        skeleton_ms = None
        for arm in ["baseline"] + CLASS_ORDER + ["all"]:
            if arm not in by_arm:
                continue
            tok = by_arm[arm]
            ms = 1000.0 / med(tok)
            d = digests[wl][arm]
            ds = ", ".join(sorted(d)[:3]) + (".." if len(d) > 3 else "")
            lines.append(f"| {arm} | {len(tok)} | {round(med(tok), 3)} | "
                         f"{round(ms, 4)} | {ds} |")
            if arm == "baseline":
                base_ms = ms
            elif arm == "all":
                skeleton_ms = ms
            else:
                marginals[arm] = base_ms - ms
        if skeleton_ms is None:
            lines.append("(skeleton arm not run yet - partial table)")
            for arm in CLASS_ORDER:
                if arm in marginals:
                    k = CENSUS[arm]
                    lines.append(f"- {arm}: {round(marginals[arm], 4)} "
                                 f"({k[1]} disp/tok)")
            top = sorted(marginals.items(), key=lambda kv: -kv[1])[:3]
            lines.append("- top classes: " + ", ".join(
                f"{a} {round(v / base_ms * 100, 1)}%" for a, v in top))
            lines.append("")
            continue
        total = sum(marginals.values()) + skeleton_ms
        residual = total - base_ms
        lines.append("")
        lines.append("marginals (baseline - ablated), ms/token:")
        for arm in CLASS_ORDER:
            if arm in marginals:
                k = CENSUS[arm]
                lines.append(f"- {arm}: {round(marginals[arm], 4)} "
                             f"({k[1]} disp/tok)")
        lines.append(f"- skeleton (all ablated): {round(skeleton_ms, 4)}")
        lines.append(f"- sum(marginals)+skeleton: {round(total, 4)} "
                     f"vs token {round(base_ms, 4)} -> residual "
                     f"{round(residual, 4)} ms "
                     f"({round(100 * residual / base_ms, 1)}%)")
        top = sorted(marginals.items(), key=lambda kv: -kv[1])[:3]
        lines.append("- top classes: " + ", ".join(
            f"{a} {round(v / base_ms * 100, 1)}%" for a, v in top))
        lines.append("")
    report = "\n".join(lines)
    print(report)
    if out_md:
        out_md.write_text(report + "\n")


if __name__ == "__main__":
    main()
