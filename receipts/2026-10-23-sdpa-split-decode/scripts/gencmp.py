"""Compares greedy generations: gencmp.py DEFAULT.json SPLIT0.json REF32.json. Prints the first token where the default (split route) and split0 (fused route) generations differ,
the top-1/top-2 log-probability gap of the split0 and float32-reference runs at that step (same prefix up to it), and which token the float32 reference chose."""
import json
import sys

d, b, r = (json.load(open(p)) for p in sys.argv[1:4])
n = min(len(d["tokens"]), len(b["tokens"]))
first = next((i for i in range(n) if d["tokens"][i] != b["tokens"][i]), None)
print("prompt tokens", d["prompt_tokens"], "generated", len(d["tokens"]), len(b["tokens"]), len(r["tokens"]))
if first is None:
    print("GENCMP no divergence between default and split0")
    sys.exit(0)
same_prefix_ref = r["tokens"][:first] == b["tokens"][:first]
print("GENCMP first divergence at generated token", first, "default", d["tokens"][first], "split0", b["tokens"][first], "ref32", r["tokens"][first] if first < len(r["tokens"]) else None)
print("GENCMP top1-top2 gap at that step: default arm", round(d["top2_gap"][first], 4), "split0 arm", round(b["top2_gap"][first], 4), "ref32 arm", round(r["top2_gap"][first], 4) if first < len(r["top2_gap"]) else None, "(ref32 gap valid only if its prefix equals split0:", same_prefix_ref, ")")
rd = next((i for i in range(min(len(r["tokens"]), len(d["tokens"]))) if r["tokens"][i] != d["tokens"][i]), None)
rb = next((i for i in range(min(len(r["tokens"]), len(b["tokens"]))) if r["tokens"][i] != b["tokens"][i]), None)
print("GENCMP first divergence from ref32: default", rd, "split0", rb)
