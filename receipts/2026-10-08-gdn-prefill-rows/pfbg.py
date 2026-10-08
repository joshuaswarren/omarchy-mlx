"""The real mlx_lm.server prefill: four prompts of DIFFERENT lengths (left padding, a real mask) through BatchGenerator
(max_tokens=1, so the time is the prompt processing plus one token), against the same four prompts one BatchGenerator
each (the sequential case). Median of 3 each, a warm-up of each first. Also the first tokens of both (they must match).
Args: model_dir prompt_file L1 L2 L3 L4 (prompt lengths in tokens)."""
import json
import statistics
import sys
import time

import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import BatchGenerator
from mlx_lm.sample_utils import greedy_sampler

model, tok = load(sys.argv[1])
base = tok.encode(open(sys.argv[2]).read())
lengths = [int(a) for a in sys.argv[3:7]]
prompts = [(base * (n // len(base) + 1))[:n] for n in lengths]
prompts = [p[: n - 1] + [p[-1] + i + 1] for i, (p, n) in enumerate(zip(prompts, lengths))]


def run(batch):
    bg = BatchGenerator(model, max_tokens=1, sampler=greedy_sampler)
    uids = bg.insert(batch, max_tokens=[1] * len(batch))
    first = {}
    t = time.perf_counter()
    while responses := bg.next_generated():
        for r in responses:
            first[r.uid] = int(r.token)
    dt = time.perf_counter() - t
    bg.close()
    return dt, [first[u] for u in uids]


run(prompts), [run([p]) for p in prompts]
seq, bat = [], []
for _ in range(3):
    s, toks_seq = 0.0, []
    for p in prompts:
        dt, tk = run([p])
        s += dt
        toks_seq += tk
    seq.append(s)
    dt, toks_bat = run(prompts)
    bat.append(dt)
print("PFBG " + json.dumps({"mlx": mx.__version__, "lengths": lengths, "seq_s": round(statistics.median(seq), 4),
                            "batched_s": round(statistics.median(bat), 4),
                            "ratio": round(statistics.median(bat) / statistics.median(seq), 4),
                            "first_tokens_equal": toks_seq == toks_bat}), flush=True)
