#!/usr/bin/env python3
# Timed prefill (no GPU_PROFILE) or markered prefill for the profiler.
# Usage: run_prefill.py --model REPO --targets 512,2048 --max-tokens 1 [--markers PATH]
import argparse, hashlib, os, sys, time

os.environ.setdefault("MLX_OMARCHY_NO_MLXLM_WARNING", "1")

def clock():
    return time.monotonic_ns()

def build_prompt(tok, target, seed="The quick brown fox jumps over the lazy dog near the river bank. "):
    # Grow a chat prompt until it reaches `target` tokens (or just over).
    prefix = "Summarize the following text. "
    lo, hi = 1, 8
    def n_for(rep):
        ids = tok.apply_chat_template(
            [{"role": "user", "content": prefix + seed * rep}],
            add_generation_prompt=True)
        return ids, len(ids)
    ids, n = n_for(hi)
    while n < target:
        lo, hi = hi, hi * 2
        ids, n = n_for(hi)
        if hi > 10000:
            break
    # binary search the smallest rep with n >= target
    best = ids
    while lo < hi:
        mid = (lo + hi) // 2
        ids, n = n_for(mid)
        if n >= target:
            best, hi = ids, mid
        else:
            lo = mid + 1
    ids, n = n_for(hi)
    return ids, n

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--targets", default="512,2048")
    ap.add_argument("--max-tokens", type=int, default=1)
    ap.add_argument("--markers")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    from mlx_lm import load, stream_generate
    from mlx_lm.sample_utils import make_sampler
    import mlx.core as mx

    mf = open(args.markers, "w", encoding="utf-8") if args.markers else None
    def mark(phase):
        if mf:
            mf.write(__import__("json").dumps({"t": clock(), "p": phase}) + "\n")
            mf.flush()

    mark("load_start")
    model, tok = load(args.model)
    mark("load_done")
    mx.random.seed(0)
    sampler = make_sampler(temp=0.0)

    targets = [int(x) for x in args.targets.split(",") if x]
    # Warmup on a short prompt so pipeline creation is not in the timed window.
    warm, wn = build_prompt(tok, 64)
    for _ in stream_generate(model, tok, warm, max_tokens=1, sampler=sampler):
        pass

    for t in targets:
        ids, n = build_prompt(tok, t)
        mark("prefill_start")
        t0 = clock()
        last = None
        ntok = 0
        for r in stream_generate(model, tok, ids, max_tokens=args.max_tokens, sampler=sampler):
            last = r
            ntok += 1
            if ntok == 1:
                mark("prefill_done")
                mark("decode_start")
            mark("tok")
        t1 = clock()
        mark("decode_done")
        # prompt_tps is mlx-lm's prefill throughput; also report wall around the call
        wall_ms = (t1 - t0) / 1e6
        txt = last.text if last is not None else ""
        print(
            "RESULT", args.tag or args.model.split("/")[-1],
            "target", t,
            "prompt_tokens", last.prompt_tokens if last else n,
            "prompt_tps", f"{last.prompt_tps:.2f}" if last else "?",
            "wall_ms", f"{wall_ms:.1f}",
            "gen_tokens", last.generation_tokens if last else 0,
            "generation_tps", f"{getattr(last, 'generation_tps', 0.0):.2f}" if last else "?",
            "text_sha", hashlib.sha256(txt.encode()).hexdigest()[:12],
            flush=True,
        )
    if mf:
        mf.close()

if __name__ == "__main__":
    main()
