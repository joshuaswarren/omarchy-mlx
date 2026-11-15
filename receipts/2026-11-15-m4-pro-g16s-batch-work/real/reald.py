"""reald.py <model_path>: warm-up, then ~500-token prefill + 128 greedy tokens; prints CLOCK_MONOTONIC window."""
import sys, time, mlx.core as mx
from mlx_lm import load, generate, stream_generate
m, tok = load(sys.argv[1])
generate(m, tok, prompt="hello", max_tokens=4, verbose=False)
p = tok.apply_chat_template([{"role": "user", "content": "Summarize: " + "The farmer counts his sheep near the river bank. " * 50}], add_generation_prompt=True, tokenize=False)
t0 = time.clock_gettime(time.CLOCK_MONOTONIC)
tf = None
for i, r in enumerate(stream_generate(m, tok, p, max_tokens=128)):
    if i == 0: tf = time.clock_gettime(time.CLOCK_MONOTONIC)
t1 = time.clock_gettime(time.CLOCK_MONOTONIC)
print(f"WINDOW {t0:.6f} {t1:.6f}", flush=True)
print(f"FIRSTTOK {tf:.6f} gen_tps={r.generation_tps:.2f}", flush=True)
