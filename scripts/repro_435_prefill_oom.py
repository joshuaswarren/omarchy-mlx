#!/usr/bin/env python3
"""Repro/regression harness for issue #435.

Runs the exact failure protocol from the issue: a 4-bit 4B model prefilled
in fixed-size chunks to a pinned context length, repeated N times in one
process with a fresh prompt cache each rep and the default (unbounded-by-
caller) buffer cache. A pre-#436 build fails on the stock lab kernel with
VK_ERROR_OUT_OF_DEVICE_MEMORY while MLX reports ~2-3 GB active; on a build
with the capped reuse cache the run must complete every rep.

Per rep the script prints one line with MLX's own accounting
(active/cache/quarantine-relevant counters) plus a final JSON result line,
so a receipt can quote it verbatim.

Self-test (no GPU, no mlx needed): python3 repro_435_prefill_oom.py
--self-test proves the prompt builder reaches the context length, the
chunk schedule covers it exactly and the rep loop runs the full count.

Exit codes: 0 all reps completed; 1 an allocation failed (the OOM this
issue tracks) or a rep completed fewer tokens than pinned; 2 bad
invocation or unusable environment.
"""

import argparse
import json
import sys


def build_ids(tokenizer, target_tokens, seed_text):
    """Repeat seed_text until the tokenized prompt reaches target_tokens."""
    ids = tokenizer.encode(seed_text)
    if not ids:
        raise ValueError("seed text produced no tokens")
    while len(ids) < target_tokens:
        ids += ids
    return ids[:target_tokens]


def chunk_starts(total, chunk):
    return list(range(0, total, chunk))


def run(model_id, context, chunk, reps):
    import mlx.core as mx  # noqa: PLC0415
    from mlx_lm.models.cache import make_prompt_cache  # noqa: PLC0415
    from mlx_lm.utils import load  # noqa: PLC0415

    model, tokenizer = load(model_id)
    ids = build_ids(
        tokenizer,
        context,
        "Summarize the quarterly plan for the infrastructure team. ",
    )
    starts = chunk_starts(len(ids), chunk)

    mib = 1 << 20
    for rep in range(reps):
        cache = make_prompt_cache(model)
        for start in starts:
            mx.eval(
                model(
                    mx.array(ids[start : start + chunk])[None],
                    cache=cache,
                )
            )
        print(
            f"rep {rep + 1}/{reps} done: "
            f"active={mx.get_active_memory() / mib:.0f} MiB "
            f"peak={mx.get_peak_memory() / mib:.0f} MiB "
            f"cache={mx.get_cache_memory() / mib:.0f} MiB",
            flush=True,
        )
    return {"model": model_id, "context": context, "chunk": chunk, "reps": reps}


def self_test():
    class FakeTok:
        def encode(self, text):
            return list(range(7))  # 7 tokens per seed text

    ids = build_ids(FakeTok(), 4096, "x ")
    assert len(ids) == 4096, len(ids)
    starts = chunk_starts(4096, 256)
    assert starts[0] == 0 and starts[-1] == 3840 and len(starts) == 16
    covered = sum(min(4096 - s, 256) for s in starts)
    assert covered == 4096, covered
    print("SELF-TEST-OK")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--model",
        default="mlx-community/Qwen3-4B-Instruct-2507-4bit",
        help="mlx_lm model id or local path",
    )
    ap.add_argument("--context", type=int, default=4096)
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument(
        "--self-test",
        action="store_true",
        help="verify protocol logic without mlx or a GPU",
    )
    args = ap.parse_args()

    if args.self_test:
        return self_test()
    if args.context <= 0 or args.chunk <= 0 or args.reps <= 0:
        ap.error("--context, --chunk and --reps must be positive")
    try:
        result = run(args.model, args.context, args.chunk, args.reps)
    except RuntimeError as exc:
        print(
            json.dumps(
                {"result": "oom-or-runtime-failure", "error": str(exc)}
            )
        )
        return 1
    print(json.dumps({"result": "ok", **result}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
