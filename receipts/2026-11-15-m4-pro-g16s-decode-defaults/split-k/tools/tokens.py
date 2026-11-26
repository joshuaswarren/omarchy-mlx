"""Greedy token ids of one long-context decode: python tokens.py MODEL PROMPT_TOKENS
Same prompt as receipts/2026-11-13-sdpa-split-g13s-g14g/tools/longctx.py (the receipts/2026-10-23-sdpa-split-decode llmstep.py sentence repeated to
~PROMPT_TOKENS tokens), 128 generated. Prints one TOKS line: the MLX_OMARCHY_SDPA_DECODE_SPLIT and MLX_OMARCHY_SDPA_SPLIT_PV values, then the ids."""
import ast, json, os, sys
from mlx_lm import load, stream_generate
src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "2026-10-23-sdpa-split-decode", "scripts", "llmstep.py")
text = next(ast.literal_eval(l.split("=", 1)[1].strip()) for l in open(src) if l.startswith("text = "))
model, tok = load(sys.argv[1])
ids = tok.encode(text * max(1, round(int(sys.argv[2]) / 18)))
toks = [int(x.token) for x in stream_generate(model, tok, ids, max_tokens=128)]
env = os.environ
print("TOKS", env.get("MLX_OMARCHY_SDPA_DECODE_SPLIT", "unset"), env.get("MLX_OMARCHY_SDPA_SPLIT_PV", "unset"), json.dumps(toks), flush=True)
