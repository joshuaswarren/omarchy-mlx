#!/usr/bin/env python3
"""Batch decode profile driver: B rows through an mlx_lm BatchGenerator.

Companion to tools/profile_decode_driver.py for the per-kernel batch-2
comparison (PR 303 hoisted batch qmv vs shipped, mlx_omarchy diagnostics
wheels). One measured pass of greedy BatchGenerator decode; every decode
step is one JSON window:

    {"i":<step>,"t0":<monotonic_ns>,"t1":<monotonic_ns>,"tok":[<B ids>]}

t0 is taken just before bg.next_generated() (which blocks until the step's
tokens are ready) and t1 after an explicit mx.synchronize(), so [t0,t1]
brackets exactly one decode step that produces up to B tokens (one per
row, in the uid order returned by insert()). The meta line carries the
profile_decode_driver.py keys plus "batch" (analyze_decode_profile_batch.py
reads it; the released analyzer ignores the extra key).

Env expected (set by the caller, read by the diagnostics libmlx):
  MLX_OMARCHY_GPU_PROFILE=<ndjson path>
  MLX_OMARCHY_GPU_PROFILE_LABEL=<name>
"""
import argparse
import json
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
parser.add_argument("--batch", type=int, default=2, choices=(2, 3, 4))
parser.add_argument("--warmup-tokens", type=int, default=8)
parser.add_argument("--tokens", type=int, default=32)
parser.add_argument("--rows", default="", help="comma list of prompt indices (default: the first --batch)")
parser.add_argument("--out", required=True, help="driver windows jsonl")
a = parser.parse_args()

import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import BatchGenerator
from mlx_lm.sample_utils import make_sampler

EXT = [
    "What is the capital of France? Answer in one word.",
    "Name the largest planet of the solar system in one word.",
    "Which element has the chemical symbol Fe? Answer in one word.",
    "Who wrote the play Hamlet? Answer in one short sentence.",
    "What is the capital of Italy? Answer in one word.",
    "Who painted the Mona Lisa? Answer in one short sentence.",
]
# Token lengths with the Qwen3.8 tokenizer: 12, 12, 13, 13, 12, 12. --rows picks prompts by index, so one batch size can be run unpadded
# (equal lengths: 0,1 or 0,1,4) or padded (0,2 or 0,1,2).
rows = [int(x) for x in a.rows.split(",")] if a.rows else list(range(a.batch))
assert len(rows) == a.batch, "--rows must name exactly --batch prompts"
TEXTS = [EXT[i] for i in rows]

model, tok = load(a.model)
prompts = [tok.encode(t) for t in TEXTS]


def run_bg(max_tokens):
    """One throwaway BatchGenerator pass; returns per-step token lists."""
    bg = BatchGenerator(model, max_tokens=max_tokens, sampler=make_sampler(temp=0.0))
    uids = bg.insert(prompts, max_tokens=[max_tokens] * len(prompts))
    steps = []
    while len(steps) < max_tokens:
        responses = bg.next_generated()
        if not responses:
            break
        by_uid = {r.uid: int(r.token) for r in responses}
        steps.append([by_uid.get(u, -1) for u in uids])
    bg.close()
    return steps


# Warmup: compile/warm path on a throwaway generator, then quiesce.
run_bg(a.warmup_tokens)
mx.synchronize()

# Measured pass: fresh generator, one bracketed window per decode step.
t_start = time.monotonic_ns()
measured = 0
with open(a.out, "w") as f:
    meta = {
        "prompt": " || ".join(TEXTS),
        "prompt_len": sum(len(p) for p in prompts),
        "warmup_tokens": a.warmup_tokens,
        "start_monotonic": t_start,
        "batch": a.batch,
        "rows": rows,
    }
    f.write(json.dumps(meta) + "\n")
    bg = BatchGenerator(model, max_tokens=a.tokens, sampler=make_sampler(temp=0.0))
    uids = bg.insert(prompts, max_tokens=[a.tokens] * a.batch)
    for i in range(a.tokens):
        t0 = time.monotonic_ns()
        responses = bg.next_generated()
        if not responses:
            break
        by_uid = {r.uid: int(r.token) for r in responses}
        mx.synchronize()
        t1 = time.monotonic_ns()
        f.write(json.dumps({"i": i, "t0": t0, "t1": t1,
                            "tok": [by_uid.get(u, -1) for u in uids]}) + "\n")
        f.flush()
        measured += 1
    bg.close()
mx.synchronize()
print("steps:", measured, file=sys.stderr)
