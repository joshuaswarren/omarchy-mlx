"""Greedy token ids of K2 for a fixed prompt (A/B across builds)."""
import hashlib, json, sys
import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import generate_step
mx.set_cache_limit(0)
model, tok = load(sys.argv[1])
msgs = [{"role": "user", "content": "Explain in detail how a bicycle derailleur works, step by step, and list common failure modes."}]
prompt = mx.array(tok.encode(tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)))
ids = []
for (t, _), _ in zip(generate_step(prompt, model, max_tokens=256), range(256)):
    ids.append(int(t))
print(json.dumps({"prompt_tokens": int(prompt.size), "n": len(ids), "sha": hashlib.sha256(json.dumps(ids).encode()).hexdigest()[:16], "ids": ids}))
