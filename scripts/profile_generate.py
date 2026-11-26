#!/usr/bin/env python3
"""Run mlx-lm generation with host-side phase markers for GPU profiling.

Uses the same load + stream_generate path as `python -m mlx_lm generate` so
the work matches the prior receipts, while writing CLOCK_MONOTONIC markers
(same clock as the MLX_OMARCHY_GPU_PROFILE host timestamps) that let
scripts/profile_analyze.py attribute dispatches to load / prefill / decode
phases and count dispatches per decode token.

Run with MLX_OMARCHY_GPU_PROFILE=<path> set so the C++ harness records the
matching event stream. MLX_DISABLE_COMPILE=1 must be set to match the
receipt baseline (bf16 compiled tape is gated by design).

Usage:
  MLX_DISABLE_COMPILE=1 MLX_OMARCHY_GPU_PROFILE=/tmp/p.jsonl \
    python3 profile_generate.py --model PATH --prompt "Hi" \
    --max-tokens 32 --temp 0 --seed 0 --markers /tmp/m.jsonl
"""

import argparse
import json
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--markers", default="markers.jsonl")
    ap.add_argument(
        "--raw-prompt",
        action="store_true",
        help="skip the chat template (matches --ignore-chat-template)")
    args = ap.parse_args()

    mf = open(args.markers, "w", encoding="utf-8")

    def mark(phase):
        mf.write(json.dumps({"t": time.monotonic_ns(), "p": phase}) + "\n")
        mf.flush()

    from mlx_lm.utils import load
    from mlx_lm.generate import stream_generate

    mark("load_start")
    model, tokenizer = load(args.model)
    mark("load_done")

    import mlx.core as mx
    from mlx_lm.sample_utils import make_sampler
    mx.random.seed(args.seed)
    sampler = make_sampler(temp=args.temp)

    prompt = args.prompt
    if not args.raw_prompt and hasattr(tokenizer, "apply_chat_template"):
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": args.prompt}],
            add_generation_prompt=True)

    mark("prefill_start")
    n_tok = 0
    tok_times = []
    for resp in stream_generate(
        model,
        tokenizer,
        prompt,
        max_tokens=args.max_tokens,
        sampler=sampler,
    ):
        n_tok += 1
        now = time.monotonic_ns()
        tok_times.append(now)
        if n_tok == 1:
            mark("prefill_done")
            mark("decode_start")
        mark("tok")
        print(resp.text, end="", flush=True)
    mark("decode_done")
    mf.close()
    print()

    if len(tok_times) > 2:
        gaps = sorted(b - a for a, b in zip(tok_times, tok_times[1:]))
        print(f"[profile_generate] tokens={n_tok} "
              f"median_inter_token_ms={gaps[len(gaps) // 2] / 1e6:.2f} "
              f"mean_inter_token_ms="
              f"{sum(gaps) / len(gaps) / 1e6:.2f}")
    else:
        print(f"[profile_generate] tokens={n_tok} (too few for steps)")


if __name__ == "__main__":
    main()
