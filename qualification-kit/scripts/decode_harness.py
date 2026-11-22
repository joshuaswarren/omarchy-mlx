#!/usr/bin/env python3
"""Linux/macOS reference decode harness for the M2 decode-gap work.

Contract (fixed 2026-11-21, Main):
  --max-tokens 128, rate = 127/(t127-t0) (per-token wall stamps, first to last),
  prompt "The history of France begins in antiquity. " (trailing space) tokenized then
  repeated/truncated to exactly ctx=512 ids, greedy (argmax; no sampling call),
  model loaded once, 5 rounds each with a fresh prompt cache, median of per-round
  decode tok/s, model snapshot 0867d98bfb174b042d88461c0e7c97b86b34b381 of HF repo
  models--SiddhJagani--Qwen3.8-2B-mlx-4Bit. Emits the A4 JSON schema (a4-schema.json).

  decode_harness.py run --model PATH [--ctx 512] [--max-tokens 128] [--rounds 5] \
      [--prompt "..."] --out a4.json
  decode_harness.py preflight            # HF cache snapshot check, loud abort if absent
  decode_harness.py dry-run [--ctx 512] [--max-tokens 128] [--prompt "..."]  # no mlx needed
  decode_harness.py selftest             # assert-based check of the pure logic
"""
import argparse
import hashlib
import json
import os
import statistics
import sys
import time
from pathlib import Path

SNAPSHOT = "0867d98bfb174b042d88461c0e7c97b86b34b381"
HF_REPO = "models--SiddhJagani--Qwen3.8-2B-mlx-4Bit"
DEFAULT_PROMPT = "The history of France begins in antiquity. "
DEFAULT_CTX = 512
DEFAULT_MAX_TOKENS = 128
DEFAULT_ROUNDS = 5

SCHEMA_VERSION = "A4"


def prompt_ids(prompt, ctx):
    """Tokenize once, repeat, truncate to exactly ctx ids."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(_snapshot_dir())
    base = tok.encode(prompt)
    if not base:
        raise SystemExit("prompt tokenized to zero ids")
    ids = (base * (ctx // len(base) + 1))[:ctx]
    return ids


def decode_rate(stamps):
    """tok/s over the decode window: (n-1)/(t_last - t_first); n = 128 -> 127/(t127-t0)."""
    if len(stamps) < 2:
        raise ValueError("need at least two token stamps")
    return (len(stamps) - 1) / (stamps[-1] - stamps[0])


def snapshot_candidates():
    home = Path(os.path.expanduser("~"))
    return [
        home / "Library/Caches/huggingface/hub" / HF_REPO / "snapshots" / SNAPSHOT,  # macOS
        home / ".cache/huggingface/hub" / HF_REPO / "snapshots" / SNAPSHOT,          # Linux
    ]


def preflight():
    """Loud abort unless exactly one platform's snapshot dir exists."""
    hits = [p for p in snapshot_candidates() if p.is_dir()]
    if not hits:
        print("=" * 72)
        print("ABORT: HF cache is missing the pinned model snapshot.")
        print(f"  repo:     {HF_REPO}")
        print(f"  snapshot: {SNAPSHOT}")
        print("  looked in:")
        for p in snapshot_candidates():
            print(f"    {p}")
        print("Fetch it once (from the box that has it) or fix HF_HOME before the window.")
        print("=" * 72)
        raise SystemExit(2)
    print(f"PREFLIGHT OK: {hits[0]}")
    return hits[0]


def _snapshot_dir():
    hits = [p for p in snapshot_candidates() if p.is_dir()]
    if not hits:
        preflight()  # raises with the loud banner
    return str(hits[0])


def run_round(model, ids, max_tokens):
    """One decode round on a fresh prompt cache; returns per-token wall stamps."""
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache

    cache = make_prompt_cache(model)
    stamps = []
    x = mx.array([ids])
    for _ in range(max_tokens):
        t = time.perf_counter()
        logits = model(x, cache=cache)
        tok = mx.argmax(logits[:, -1, :], axis=-1)  # greedy: no sampling call
        mx.eval(tok)
        stamps.append(time.perf_counter())
        x = tok
    return stamps


def do_run(args):
    from mlx_lm import load

    snap = preflight()
    model, _ = load(str(snap))  # load once
    ids = prompt_ids(args.prompt, args.ctx)
    rounds = []
    for i in range(args.rounds):
        stamps = run_round(model, ids, args.max_tokens)
        rounds.append({"round": i + 1, "decode_tps": decode_rate(stamps),
                       "n_tokens": len(stamps),
                       "t_first": stamps[0], "t_last": stamps[-1]})
        print(f"round {i + 1}/{args.rounds}: {rounds[-1]['decode_tps']:.2f} tok/s")
    rates = [r["decode_tps"] for r in rounds]
    out = {
        "schema_version": SCHEMA_VERSION,
        "hf_repo": HF_REPO.replace("models--", "").replace("--", "/"),
        "snapshot": SNAPSHOT,
        "snapshot_dir": str(snap),
        "ctx": args.ctx,
        "max_tokens": args.max_tokens,
        "prompt": args.prompt,
        "prompt_ids_count": len(ids),
        "prompt_ids_sha256": hashlib.sha256(
            json.dumps(ids, separators=(",", ":")).encode()).hexdigest(),
        "greedy": True,
        "rounds": rounds,
        "median_decode_tps": statistics.median(rates),
        "rate_formula": f"{args.max_tokens - 1}/(t_last - t_first)",
        "argv": sys.argv,
        "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n")
    print(f"A4 written: {args.out} median={out['median_decode_tps']:.2f} tok/s")


def do_dry_run(args):
    print("dry-run: no mlx import, no model load")
    print(f"  model repo:   {HF_REPO}")
    print(f"  snapshot:     {SNAPSHOT}")
    print(f"  ctx:          {args.ctx}")
    print(f"  max_tokens:   {args.max_tokens}")
    print(f"  rounds:       {DEFAULT_ROUNDS} (fresh prompt cache each)")
    print(f"  prompt:       {args.prompt!r}")
    print(f"  rate formula: {args.max_tokens - 1}/(t_last - t_first)")
    print(f"  sampler:      none (mx.argmax per step)")
    hit = next((p for p in snapshot_candidates() if p.is_dir()), None)
    print(f"  snapshot dir: {hit if hit else 'MISSING (preflight would abort)'}")


def do_selftest():
    # rate formula: 128 stamps 1s apart -> 127 tok/s
    stamps = [i / 127.0 for i in range(128)]  # 128 stamps spanning 1s -> 127 tok/s
    assert abs(decode_rate(stamps) - 127.0) < 1e-9
    # median of 5 rounds
    assert statistics.median([10.0, 11.0, 12.0, 13.0, 100.0]) == 12.0
    # prompt repeat/truncate logic mirrors prompt_ids without transformers
    base = [1, 2, 3]
    ids = (base * (512 // len(base) + 1))[:512]
    assert len(ids) == 512 and ids[:3] == [1, 2, 3] and ids[-1] == 2
    print("selftest OK: rate=127.0 tok/s on 128 stamps; median; ids repeat/truncate to 512")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "dry-run", "selftest", "preflight"):
        p = sub.add_parser(name)
        if name in ("run", "dry-run"):
            p.add_argument("--ctx", type=int, default=DEFAULT_CTX)
            p.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
            p.add_argument("--prompt", default=DEFAULT_PROMPT)
        if name == "run":
            p.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS)
            p.add_argument("--out", default="a4.json")
    args = ap.parse_args()
    {"run": do_run, "dry-run": do_dry_run, "selftest": lambda a: do_selftest(),
     "preflight": lambda a: preflight()}[args.cmd](args)


if __name__ == "__main__":
    main()
