#!/usr/bin/env python3
"""Build final summary table from per-model runs/*.json, preferring
thinking-stripped GSM/IFE scores when present.
"""
import json
import math
import sys
from pathlib import Path


def fmt(v, default="—"):
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return default
    if isinstance(v, float):
        return f"{v:.2f}"
    return str(v)


def main() -> int:
    indir = Path(sys.argv[1])
    out_md = Path(sys.argv[2])
    rows = []
    for fp in sorted(indir.glob("*.json")):
        with fp.open() as fh:
            d = json.load(fh)
        # Skip non-model files (smoke, .prev, README)
        if "model_id" not in d:
            continue
        ttft = d.get("ttft_170sys_256tok") or {}
        p512 = d.get("prefill_512") or {}
        p2048 = d.get("prefill_2048") or {}
        d512 = d.get("decode_512") or {}
        d2048 = d.get("decode_2048") or {}

        # Prefer thinking-stripped grades if present
        gsm_pass = d.get("gsm_pass_thinking_stripped")
        gsm_total = d.get("gsm_total", 0)
        if gsm_pass is None:
            gsm_pass = d.get("gsm_pass", 0)
            gsm_total = d.get("gsm_total", 0)
        ife_pass = d.get("ife_pass_thinking_stripped")
        ife_total = d.get("ife_total", 0)
        if ife_pass is None:
            ife_pass = d.get("ife_pass", 0)
            ife_total = d.get("ife_total", 0)

        rows.append({
            "label": d.get("model_label", ""),
            "model": d.get("model_id", ""),
            "cold": fmt(d.get("cold_load_s")),
            "warm": fmt(d.get("warm_load_s")),
            "peak": fmt(d.get("peak_mem_after_load_gb")),
            "ttft": fmt(ttft.get("median")),
            "p512": fmt(p512.get("median")),
            "p2048": fmt(p2048.get("median")),
            "d512": fmt(d512.get("median")),
            "d2048": fmt(d2048.get("median")),
            "card": f"{d.get('card_pass', 0)}/{d.get('card_total', 0)}",
            "gsm": f"{gsm_pass}/{gsm_total}",
            "ife": f"{ife_pass}/{ife_total}",
            "provenance": d.get("provenance", ""),
        })

    headers = ["label", "model", "cold", "warm", "peak",
               "ttft", "p512", "p2048", "d512", "d2048",
               "card", "gsm", "ife"]
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for r in rows:
        lines.append("| " + " | ".join(str(r[h]) for h in headers) + " |")
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())