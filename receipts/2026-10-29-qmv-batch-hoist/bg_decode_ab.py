"""Batch decode (B = 2..4 rows through mlx_lm BatchGenerator), greedy, same prompts every run.

Measures the steady-state decode step (the batch kernel path: B x 1 token against 4-bit weights) and prints the generated token ids
per row as a hash, so two builds can be compared for identical output. Prefill is excluded: the clock starts after the first
response of each run (the prefill step), and the step times are medians over the remaining steps.

Args: model_dir B tokens reps
"""
import glob
import hashlib
import importlib.metadata as md
import json
import os
import statistics
import sys
import time

import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import BatchGenerator
from mlx_lm.sample_utils import make_sampler

model_dir, B, TOKENS, REPS = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
model, tok = load(model_dir)
texts = [
    "What is the capital of France? Answer in one word.",
    "Name the largest planet of the solar system in one word.",
    "Which element has the chemical symbol Fe? Answer in one word.",
    "Who wrote the play Hamlet? Answer in one short sentence.",
][:B]
prompts = [tok.encode(t) for t in texts]


def run():
    bg = BatchGenerator(model, max_tokens=TOKENS, sampler=make_sampler(temp=0.0))
    uids = bg.insert(prompts, max_tokens=[TOKENS] * B)
    ids = {u: [] for u in uids}
    stamps = []
    while True:
        responses = bg.next_generated()
        if not responses:
            break
        stamps.append(time.perf_counter())
        for r in responses:
            ids[r.uid].append(int(r.token))
    bg.close()
    steps = [b - a for a, b in zip(stamps[1:-1], stamps[2:])]
    return statistics.median(steps) * 1e3, [ids[u] for u in uids], len(steps)


def libs():
    base = os.path.dirname(mx.__file__)
    return {os.path.basename(p): hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]
            for p in sorted(glob.glob(os.path.join(base, "lib", "libmlx*"))) if os.path.isfile(p)}


try:
    wheel = md.version("mlx_omarchy")
except md.PackageNotFoundError:
    wheel = "none (mlx " + md.version("mlx") + ")"
print("PROVENANCE mlx_omarchy", wheel, libs(), "device", mx.default_device(), flush=True)
run()
meds, last = [], None
for _ in range(REPS):
    ms, last, n = run()
    meds.append(ms)
digest = hashlib.sha256(json.dumps(last).encode()).hexdigest()[:16]
print("BGDECODE " + json.dumps({"B": B, "tokens": TOKENS, "steps_timed": n, "ms_per_step_reps": [round(m, 3) for m in meds],
                                "median_ms_per_step": round(statistics.median(meds), 3), "ids_sha": digest,
                                "row_lengths": [len(x) for x in last]}), flush=True)
