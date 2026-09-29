#!/usr/bin/env python3
"""Build the final summary table from per-model JSON results.

Reads every *.json file in --indir, prints one wide row per model with all
required columns. Saves --out-csv with the same content.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path


def fmt(v, default="not measured"):
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return default
    if isinstance(v, float):
        if abs(v) >= 100:
            return f"{v:.0f}"
        return f"{v:.2f}"
    return str(v)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", required=True,
                    help="directory containing <label>.json files")
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--out-md", required=True)
    args = ap.parse_args()

    rows: list[dict] = []
    files = sorted(Path(args.indir).glob("*.json"))
    for fp in files:
        with fp.open() as fh:
            data = json.load(fh)
        rows.append(data)

    # Wide table
    headers = [
        "model", "label", "revision",
        "cold_load_s", "warm_load_s", "peak_mem_gb",
        "ttft_p50_s",
        "prefill_512_p50_s", "prefill_2048_p50_s",
        "decode_512_tps", "decode_2048_tps",
        "card_valid", "gsm8k", "ife", "provenance",
    ]
    md_lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    csv_rows = []
    for d in rows:
        ttft = d.get("ttft_170sys_256tok", {})
        p512 = d.get("prefill_512", {})
        p2048 = d.get("prefill_2048", {})
        d512 = d.get("decode_512", {})
        d2048 = d.get("decode_2048", {})
        row = {
            "model": d.get("model_id", ""),
            "label": d.get("model_label", ""),
            "revision": (d.get("revision") or "")[:12],
            "cold_load_s": fmt(d.get("cold_load_s")),
            "warm_load_s": fmt(d.get("warm_load_s")),
            "peak_mem_gb": fmt(d.get("peak_mem_after_load_gb")),
            "ttft_p50_s": fmt(ttft.get("median")),
            "prefill_512_p50_s": fmt(p512.get("median")),
            "prefill_2048_p50_s": fmt(p2048.get("median")),
            "decode_512_tps": fmt(d512.get("median")),
            "decode_2048_tps": fmt(d2048.get("median")),
            "card_valid": f"{d.get('card_pass', 0)}/{d.get('card_total', 0)}",
            "gsm8k": f"{d.get('gsm_pass', 0)}/{d.get('gsm_total', 0)}",
            "ife": f"{d.get('ife_pass', 0)}/{d.get('ife_total', 0)}",
            "provenance": d.get("provenance", ""),
        }
        md_lines.append("| " + " | ".join(row[h] for h in headers) + " |")
        csv_rows.append(row)

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=headers)
        w.writeheader()
        for r in csv_rows:
            w.writerow(r)
    Path(args.out_md).write_text("\n".join(md_lines) + "\n")

    print("\n".join(md_lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())