#!/usr/bin/env python3
"""Append one row to a per-model Markdown table from a run.json."""
import json, sys, pathlib
src = pathlib.Path(sys.argv[1])
out = pathlib.Path(sys.argv[2])
d = json.loads(src.read_text())
def f(v, p=2):
    if v is None: return "—"
    if isinstance(v, float): return f"{v:.{p}f}"
    return str(v)
ttft = d.get("ttft_170sys_256tok") or {}
p512 = d.get("prefill_512") or {}
p2048 = d.get("prefill_2048") or {}
d512 = d.get("decode_512") or {}
d2048 = d.get("decode_2048") or {}
row = (
    f"| {d.get('model_label')} | {d.get('model_id')} | "
    f"{f(d.get('cold_load_s'))} | {f(d.get('warm_load_s'))} | "
    f"{f(d.get('peak_mem_after_load_gb'))} | {f(ttft.get('median'))} | "
    f"{f(p512.get('median'))} | {f(p2048.get('median'))} | "
    f"{f(d512.get('median'))} | {f(d2048.get('median'))} | "
    f"{d.get('card_pass',0)}/{d.get('card_total',0)} | "
    f"{d.get('gsm_pass',0)}/{d.get('gsm_total',0)} | "
    f"{d.get('ife_pass',0)}/{d.get('ife_total',0)} |"
)
with out.open("a") as fh:
    fh.write(row + "\n")
print("appended row for", d.get("model_label"))