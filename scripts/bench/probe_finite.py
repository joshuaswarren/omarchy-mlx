#!/usr/bin/env python3
"""Prefill each bench prompt once and report whether the last-position logits are finite.

Usage: probe_finite.py MODEL_SNAPSHOT_DIR
Discriminates prompt-dependent non-finite logits (replies of '!' = token 0).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import chat_model_bench as cmb  # noqa: E402
import mlx.core as mx  # noqa: E402
from mlx_lm import load  # noqa: E402

print(cmb.provenance(), flush=True)
model, tok = load(sys.argv[1])
bench = cmb.Bench(model, tok)
prompts_dir = cmb.REPO_ROOT / "scripts/bench/prompts"
cases = []
for p in cmb.load_jsonl(prompts_dir / "cards_16.jsonl"):
    full = bool(cmb.FULL_CARD_CUES.search(p["prompt"]) or cmb.user_requested_full_schema(p["prompt"]))
    cases.append((f"card{p['index']}", cmb.SYSTEM_PREFIX + (cmb.SCHEMA_PROMPT if full else cmb.SCHEMA_PROMPT_COMPACT), p["prompt"]))
for g in cmb.load_jsonl(prompts_dir / "gsm8k_20.jsonl")[:4]:
    cases.append((f"gsm{g['index']}", cmb.GSM_SYSTEM, g["q"]))
for name, system, user in cases:
    ids = bench.encode_chat(system, user)
    logits = model(mx.array(ids)[None])[0, -1].astype(mx.float32)
    finite = bool(mx.all(mx.isfinite(logits)).item())
    print(json.dumps({"case": name, "tokens": len(ids), "finite": finite,
                      "argmax": int(mx.argmax(logits).item())}), flush=True)
for n in (256, 512, 1024, 2048):
    logits = model(mx.array(cmb.synthetic_prompt(tok, n))[None])[0, -1].astype(mx.float32)
    print(json.dumps({"case": f"synthetic{n}", "tokens": n,
                      "finite": bool(mx.all(mx.isfinite(logits)).item())}), flush=True)
