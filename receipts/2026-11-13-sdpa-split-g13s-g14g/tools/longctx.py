"""Long-context greedy decode: python longctx.py MODEL PROMPT_TOKENS GEN_TOKENS ROUNDS LABEL OUT.json
Prompt = the receipts/2026-10-23-sdpa-split-decode llmstep.py text repeated to ~PROMPT_TOKENS tokens; one warm-up; ROUNDS measured
requests (stream_generate, greedy). Reports decode tok/s per round (mlx_lm generation_tps), prefill tok/s and a token digest."""
import hashlib, json, os, statistics, sys, time
import mlx.core as mx
from mlx_lm import load, stream_generate
model_id, n_prompt, n_gen, rounds, label, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5], sys.argv[6]
model, tok = load(model_id)
text = "Apple Silicon runs local language models on Linux through Vulkan and a reverse engineered GPU driver. "
ids = tok.encode(text * max(1, round(n_prompt / 18)))
for _ in stream_generate(model, tok, tok.encode("Warm up the GPU."), max_tokens=8):
    pass
dec, pf, shas = [], [], []
for r in range(rounds):
    toks = []
    for last in stream_generate(model, tok, ids, max_tokens=n_gen):
        toks.append(int(last.token))
    dec.append(last.generation_tps); pf.append(last.prompt_tps)
    shas.append(hashlib.sha256(",".join(map(str, toks)).encode()).hexdigest()[:16])
res = {"label": label, "model": os.path.basename(model_id), "prompt_tokens": last.prompt_tokens, "gen_tokens": last.generation_tokens,
       "decode_tok_s": [round(x, 2) for x in dec], "decode_median": round(statistics.median(dec), 2),
       "prefill_tok_s": [round(x, 1) for x in pf], "tokens_sha": shas,
       "env": {k: v for k, v in os.environ.items() if k.startswith("MLX_OMARCHY")}, "mlx": mx.__version__,
       "load1_end": os.getloadavg()[0]}
json.dump(res, open(out, "w"), indent=1)
print(f"LONG {label} {res['model']} p{res['prompt_tokens']} g{res['gen_tokens']} decode {res['decode_median']} {res['decode_tok_s']} pf {res['prefill_tok_s'][0]} sha {set(shas)}", flush=True)
