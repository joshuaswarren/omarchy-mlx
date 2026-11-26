"""usage: census-tail.py MODEL N. The ledger step path (mlx-lm stream_generate, no sampler, greedy) for N generated tokens on a short prompt, after one warm-up
request. Run it with MLX_OMARCHY_TRACE_DISPATCH=1: the backend prints one DISPATCH line per kernel launch. Two runs (N = 8 and 12) give the dispatches per decode token."""
import sys

import mlx.core as mx  # noqa: F401
from mlx_lm import load, stream_generate

model_dir, n = sys.argv[1], int(sys.argv[2])
model, tok = load(model_dir)
for _ in stream_generate(model, tok, tok.encode("Warm up the GPU."), max_tokens=2):
    pass
print("CENSUS-BEGIN", flush=True)
ids = tok.encode("Apple Silicon runs local language models on Linux through Vulkan. " * 2)
count = 0
for _ in stream_generate(model, tok, ids, max_tokens=n):
    count += 1
print(f"CENSUS-END tokens {count}", flush=True)
