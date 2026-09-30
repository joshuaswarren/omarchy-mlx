#!/usr/bin/env python3
"""Print the chat-model bench table (Markdown) from harness-v2 run files.

Usage: aggregate.py RUN.json [RUN.json ...]
A missing value prints as the reason it is missing, never as a number.
"""
import json
import sys


def num(value, fmt="{:.2f}"):
    return fmt.format(value) if isinstance(value, (int, float)) else value


def row(path: str) -> list:
    d = json.load(open(path))
    turns = d.get("turns", [])
    load_error = next((t["load_error"] for t in turns if "load_error" in t), None)
    if load_error:
        return [d.get("label"), f"load failed: {load_error[:120]}"] + [""] * 11
    cold = next((t["load_s"] for t in turns if "load_s" in t and t["page_cache"].startswith("cold")), None)
    warm = [t["load_s"] for t in turns if "load_s" in t and t["page_cache"].startswith("warm")]
    p = d.get("perf") or {}
    ttft = p.get("ttft_card_prompt_s") or {}
    cards = d.get("cards", {})
    want_card = [c for c in cards.values() if c["expect"] == "card"]
    want_prose = [c for c in cards.values() if c["expect"] == "prose"]
    gsm, ife = d.get("gsm", {}), d.get("ife", {})
    truncated = sum(r.get("truncated", False) for part in (cards, gsm, ife) for r in part.values())
    return [
        d.get("label"),
        num(cold) if cold is not None else "not measured",
        num(min(warm)) if warm else "not measured",
        num(p.get("prefill_512_tok_s"), "{:.0f}") if p else "not measured",
        num(p.get("prefill_2048_tok_s"), "{:.0f}") if p else "not measured",
        num((p.get("decode_after_512_tok_s") or {}).get("median"), "{:.1f}") if p else "not measured",
        f"{ttft['median']:.2f} / {ttft['max']:.2f}" if ttft else "not measured",
        num(p.get("peak_mem_gb_mx")) if p else "not measured",
        (f"{sum(c['outcome'] == 'valid-card' for c in want_card)}/{len(want_card)}"
         f" (+{sum(bool(c.get('promotable')) for c in want_card if c['outcome'] != 'valid-card')} md)")
        if cards else "not run",
        f"{sum(c['outcome'] == 'prose-only' for c in want_prose)}/{len(want_prose)}" if cards else "not run",
        f"{sum(r['pass'] for r in gsm.values())}/{len(gsm)}" if gsm else "not run",
        f"{sum(r['pass'] for r in ife.values())}/{len(ife)}" if ife else "not run",
        str(truncated),
    ]


HEAD = ["model", "cold load s", "warm restart s", "prefill 512 tok/s", "prefill 2048 tok/s",
        "decode tok/s", "TTFT card prompt p50 / max s", "peak GB (mx)", "valid cards /8 (+md-promotable)",
        "prose kept prose /8", "GSM8K", "IFE", "replies hitting cap"]

if __name__ == "__main__":
    print("| " + " | ".join(HEAD) + " |")
    print("|" + "---|" * len(HEAD))
    for arg in sys.argv[1:]:
        print("| " + " | ".join(str(c) for c in row(arg)) + " |")
