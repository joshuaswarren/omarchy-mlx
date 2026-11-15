# the series without the speculative patch (./pa): draft-model path on Qwen3-8B
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "pa"))
import mlx_lm; from mlx_lm import load, stream_generate
assert mlx_lm.__file__.startswith(sys.path[0])
m, tok = load("mlx-community/Qwen3-8B-4bit"); d = load("mlx-community/Qwen3-0.6B-4bit")[0]
p = tok.apply_chat_template([{"role": "user", "content": "Repeat exactly: the river Danube flows through Bratislava."}], add_generation_prompt=True, enable_thinking=False)
try:
    out = [r.token for r in stream_generate(m, tok, p, max_tokens=32, draft_model=d, num_draft_tokens=3)]
    print("UPSTREAM_DRAFT ok", len(out), tok.decode(out)[:80])
except Exception as e:
    print("UPSTREAM_DRAFT", type(e).__name__, str(e)[:120])
