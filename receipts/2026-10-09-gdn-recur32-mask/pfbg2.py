"""Which row of a padded batched prefill (BatchGenerator, left padding, real mask) gives a different first token than the
same prompt alone, and was that row a near-tie? pfbg.py (jwm1, lengths 120 110 90 70) printed first_tokens_equal false on
the baseline wheel AND the fixed wheel, true on the M1 Max, and does not say which row.
Pre-registered rule (written before the ticket ran). Per row: token_seq (prompt alone), token_bat (in the padded batch),
margin = top-1 minus top-2 logit of the prompt alone ([1,T] model call, last position, f32). A mismatching row is NOISE
iff margin < 0.25 (two bf16 logit steps near 20); a mismatching row with margin >= 0.25 is a BUG (the padded route changes
a confident answer). Verdict: EQUAL (no mismatch), NOISE (all mismatches are near-ties), BUG (any other).
Run twice: a row that mismatches in one run and not the other is non-deterministic and is reported as FLAKY.
Args: model_dir prompt_file L1 L2 L3 L4."""
import json
import sys

import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import BatchGenerator
from mlx_lm.sample_utils import greedy_sampler

model, tok = load(sys.argv[1])
base = tok.encode(open(sys.argv[2]).read())
lengths = [int(a) for a in sys.argv[3:7]]
prompts = [(base * (n // len(base) + 1))[:n] for n in lengths]
prompts = [p[: n - 1] + [p[-1] + i + 1] for i, (p, n) in enumerate(zip(prompts, lengths))]


def first_tokens(batch):
    bg = BatchGenerator(model, max_tokens=1, sampler=greedy_sampler)
    uids = bg.insert(batch, max_tokens=[1] * len(batch))
    first = {}
    while responses := bg.next_generated():
        for r in responses:
            first[r.uid] = int(r.token)
    bg.close()
    return [first[u] for u in uids]


def margin(prompt):
    logits = model(mx.array([prompt]))[0, -1].astype(mx.float32)
    top = mx.topk(logits, 2)
    mx.eval(top)
    return abs(float(top[0]) - float(top[1]))


seq = [first_tokens([p])[0] for p in prompts]
bat = first_tokens(prompts)
rows = []
for n, s, b, p in zip(lengths, seq, bat, prompts):
    rows.append({"len": n, "seq": s, "bat": b, "margin": round(margin(p), 4), "equal": s == b})
bad = [r for r in rows if not r["equal"]]
verdict = "EQUAL" if not bad else "NOISE" if all(r["margin"] < 0.25 for r in bad) else "BUG"
print("PFBG2 " + json.dumps({"mlx": mx.__version__, "rows": rows, "verdict": verdict}), flush=True)
