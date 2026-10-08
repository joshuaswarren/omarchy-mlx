"""Does batching concurrent prefills beat running them one after another? Four identical prompts of T tokens through
the model (plain prompt caches, last-position logits only is NOT taken: the model returns what mlx-lm's own prefill
sees), per T: (a) four sequential [1, T] forward passes, (b) one [4, T] pass. Median of 3 timed repeats each, a warm-up
of each first. Also the max |row b - row 0| of the batched last-position logits (rows exact since QmmBatch cf41e9b6) and
whether row 0 equals the sequential result bit for bit. Args: model_dir prompt_file T [T ...]."""
import json
import statistics
import sys
import time

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

model, tok = load(sys.argv[1])
base = tok.encode(open(sys.argv[2]).read())
out = {"mlx": mx.__version__}


def run(ids, rows):
    cache = make_prompt_cache(model)
    x = mx.broadcast_to(ids, (rows, ids.shape[1]))
    t = time.perf_counter()
    logits = model(x, cache=cache)[:, -1, :]
    mx.eval(logits)
    return time.perf_counter() - t, logits


for T in (int(a) for a in sys.argv[3:]):
    ids = mx.array((base * (T // len(base) + 1))[:T])[None]
    run(ids, 1), run(ids, 4)
    seq, bat = [], []
    for _ in range(3):
        s, ref = 0.0, None
        for _ in range(4):
            dt, lg = run(ids, 1)
            s += dt
            ref = lg
        seq.append(s)
        dt, lg4 = run(ids, 4)
        bat.append(dt)
    row_err = float(mx.max(mx.abs(lg4 - lg4[:1])).item())
    out[f"T{T}"] = {"seq4_s": round(statistics.median(seq), 4), "batched4_s": round(statistics.median(bat), 4),
                    "ratio": round(statistics.median(bat) / statistics.median(seq), 4),
                    "row_max_abs_diff": row_err, "row0_equals_seq": bool(mx.array_equal(lg4[:1], ref).item())}
print("PFBATCH " + json.dumps(out), flush=True)
