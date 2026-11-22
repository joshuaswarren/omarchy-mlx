#!/usr/bin/env python3
"""Linux/macOS reference decode harness for the M2 decode-gap work.

Contract (fixed 2026-11-21, Main):
  --max-tokens 128, rate = 127/(t127-t0) (per-token wall stamps, first to last),
  prompt "The history of France begins in antiquity. " (trailing space) tokenized then
  repeated/truncated to exactly ctx=512 ids, greedy (argmax; no sampling call),
  model loaded once, 5 rounds each with a fresh prompt cache, median of per-round
  decode tok/s, model snapshot 0867d98bfb174b042d88461c0e7c97b86b34b381 of HF repo
  models--SiddhJagani--Qwen3.8-2B-mlx-4Bit. Emits the A4 JSON schema (a4-schema.json).

  decode_harness.py run [--model PATH] [--ctx 512] [--max-tokens 128] [--rounds 5] \
      [--prompt "..."] --out a4.json
  decode_harness.py prefill [--ctx 512] --out prefill.json  # one timed ctx-512 prompt pass
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


def prompt_ids(prompt, ctx, model_path=None):
    """Tokenize once, repeat, truncate to exactly ctx ids."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(model_path or _snapshot_dir()))
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


def token_digest(ids):
    """sha256 of the generated ids as compact JSON. Refuses missing, empty or non-int ids."""
    if not ids or not all(isinstance(i, int) and not isinstance(i, bool) for i in ids):
        raise ValueError(f"token ids missing or not integers: {ids!r}")
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()


def round_record(index, stamps, ids):
    return {"round": index, "decode_tps": decode_rate(stamps), "n_tokens": len(stamps),
            "t_first": stamps[0], "t_last": stamps[-1],
            "token_ids": ids, "token_digest": token_digest(ids)}


def prefill_input(ids):
    """The prompt as a (1, ctx) batch: mlx_lm models take (batch, seq)."""
    import mlx.core as mx
    return mx.array([ids])


def snapshot_candidates():
    home = Path(os.path.expanduser("~"))
    hubs = []
    hub_cache = os.environ.get("HF_HUB_CACHE", "").strip()
    if hub_cache:
        hubs.append(Path(os.path.expanduser(hub_cache)))
    hf_home = os.environ.get("HF_HOME", "").strip()
    if hf_home:
        hubs.append(Path(os.path.expanduser(hf_home)) / "hub")
    hubs.extend([
        home / "Library/Caches/huggingface/hub",  # macOS
        home / ".cache/huggingface/hub",          # Linux
    ])
    unique = {}  # resolved path -> first spelling seen, so HF_HOME=~/.cache/huggingface lists once
    for hub in hubs:
        cand = hub / HF_REPO / "snapshots" / SNAPSHOT
        unique.setdefault(cand.resolve(), cand)
    return list(unique.values())


def preflight():
    """Loud abort if no candidate snapshot dir exists; return the first hit."""
    candidates = snapshot_candidates()
    hits = [p for p in candidates if p.is_dir()]
    if not hits:
        print("=" * 72)
        print("ABORT: HF cache is missing the pinned model snapshot.")
        print(f"  repo:     {HF_REPO}")
        print(f"  snapshot: {SNAPSHOT}")
        print("  looked in:")
        for p in candidates:
            print(f"    {p}")
        print("Fetch it once (from the box that has it) or fix HF_HOME or HF_HUB_CACHE before the window.")
        print("=" * 72)
        raise SystemExit(2)
    print(f"PREFLIGHT OK: {hits[0]}")
    return hits[0]


def _snapshot_dir():
    hits = [p for p in snapshot_candidates() if p.is_dir()]
    if not hits:
        preflight()  # raises with the loud banner
    return str(hits[0])


def model_dir(value):
    path = Path(os.path.expanduser(value))
    if not value.strip() or not path.is_dir():  # Path("") is "."
        raise argparse.ArgumentTypeError(f"{value!r} is not a directory")
    return path


def run_round(model, ids, max_tokens):
    """One decode round on a fresh prompt cache; returns (per-token wall stamps, token ids)."""
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache

    cache = make_prompt_cache(model)
    stamps, out = [], []
    x = mx.array([ids])
    for _ in range(max_tokens):
        logits = model(x, cache=cache)
        tok = mx.argmax(logits[:, -1, :], axis=-1)  # greedy: no sampling call
        mx.eval(tok)
        stamps.append(time.perf_counter())
        out.append(int(tok.item()))
        x = tok[:, None]  # (batch, 1) for the next step
    return stamps, out


def do_run(args):
    from mlx_lm import load

    snap = args.model or preflight()
    model, _ = load(str(snap))  # load once
    ids = prompt_ids(args.prompt, args.ctx, snap)
    rounds = []
    for i in range(args.rounds):
        stamps, toks = run_round(model, ids, args.max_tokens)
        rounds.append(round_record(i + 1, stamps, toks))
        print(f"round {i + 1}/{args.rounds}: {rounds[-1]['decode_tps']:.2f} tok/s")
    rates = [r["decode_tps"] for r in rounds]
    digests = sorted({r["token_digest"] for r in rounds})
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
        "token_digest": digests[0] if len(digests) == 1 else None,
        "token_digests": digests,
        "rate_formula": f"{args.max_tokens - 1}/(t_last - t_first)",
        "argv": sys.argv,
        "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n")
    print(f"A4 written: {args.out} median={out['median_decode_tps']:.2f} tok/s "
          f"token_digest={out['token_digest']}")
    if len(digests) != 1:
        raise SystemExit(f"FAIL: greedy rounds produced {len(digests)} different token digests")


def do_prefill(args):
    import mlx.core as mx
    from mlx_lm import load

    snap = preflight()
    model, _ = load(str(snap))
    x = prefill_input(prompt_ids(args.prompt, args.ctx))
    mx.eval(model(x))  # warm-up
    t0 = time.perf_counter()
    mx.eval(model(x))
    dt = time.perf_counter() - t0
    out = {"ctx": args.ctx, "prompt": args.prompt, "seconds": dt, "prefill_tps": args.ctx / dt,
           "argv": sys.argv, "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n")
    print(f"prefill written: {args.out} {out['prefill_tps']:.1f} tok/s")


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
    assert token_digest([1, 2, 3]) == hashlib.sha256(b"[1,2,3]").hexdigest()
    for bad in (None, []):
        try:
            token_digest(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"token_digest accepted {bad!r}")
    print("selftest OK: rate=127.0 tok/s on 128 stamps; median; ids repeat/truncate to 512; token digest")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "prefill", "dry-run", "selftest", "preflight"):
        p = sub.add_parser(name)
        if name in ("run", "prefill", "dry-run"):
            p.add_argument("--ctx", type=int, default=DEFAULT_CTX)
            p.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
            p.add_argument("--prompt", default=DEFAULT_PROMPT)
        if name == "run":
            p.add_argument(
                "--model",
                type=model_dir,
                default=None,
                help="directory holding the pinned snapshot; default: find it in the HF cache",
            )
            p.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS)
            p.add_argument("--out", default="a4.json")
        if name == "prefill":
            p.add_argument("--out", default="prefill.json")
    args = ap.parse_args()
    {"run": do_run, "prefill": do_prefill, "dry-run": do_dry_run, "selftest": lambda a: do_selftest(),
     "preflight": lambda a: preflight()}[args.cmd](args)


if __name__ == "__main__":
    main()
