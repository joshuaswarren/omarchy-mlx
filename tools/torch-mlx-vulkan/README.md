# torch-mlx-vulkan

`torch.compile(backend="mlx_vulkan")` lowering PyTorch FX graphs onto the
omarchy-mlx Python bindings, plus a fused fast path for Qwen3.

## Files

- `mlx_vulkan.py` — torch dynamo backend. Per FX node: LOWERED (mlx on GPU),
  GLUE (shape bookkeeping), or FALLBACK (torch CPU). Registers itself on import.
- `qwen3_mlx.py` — `Qwen3Fast`: Qwen3 forward hand-lowered to the wheel's fused
  kernels. `mx.fast.rms_norm` for input/post/QK norms, `mx.fast.rope` for rotary,
  `mx.fast.scaled_dot_product_attention` (GQA), fused QKV projection, static
  preallocated KV cache updated by mask writes, decode step wrapped in
  `mx.compile`. Weights, activations and cache in bf16; residual stream and
  logits in fp32.
- `gen_qwen3_fast.py` — greedy-decode harness: runs the same prompt through
  torch CPU eager and `Qwen3Fast`, prints per-step tok/s and token-id match.

## Run

Requirements: Python 3.14 on aarch64 (the wheel is `cp314` only), a checkout
of this repository, and a venv with `mlx-omarchy`, `torch` (CPU wheel),
`transformers` and `safetensors`. `mlx-omarchy` is not on PyPI; install a
release wheel (see `docs/INSTALL.md` for other install paths):

```bash
python3.14 -m venv ~/.venvs/torch-mlx
~/.venvs/torch-mlx/bin/pip install "https://github.com/joshuaswarren/omarchy-mlx/releases/download/v0.7.53/mlx_omarchy-0.32.4.dev202611240009%2B197b068-cp314-cp314-linux_aarch64.whl"
~/.venvs/torch-mlx/bin/pip install torch transformers safetensors
```

Then, from the repository checkout:

```bash
~/.venvs/torch-mlx/bin/python tools/torch-mlx-vulkan/gen_qwen3_fast.py --dtype fp32
```

`--model` defaults to the HF id `Qwen/Qwen3-0.6B` (downloads ~1.5 GB on
first run); a local snapshot path also works. `--dtype fp32` and `--dtype
fp16` reproduce the 32/32 token-id match against torch CPU eager. The
`--dtype bf16` fast path matched on wheels through v0.7.51 and diverges at
the documented step-5 near-tie on v0.7.52 and later; use fp32 or fp16 for
the token-id check.

Receipt: `receipts/2026-11-23-route-d-qwen3-torch-mlx-vulkan.md`.
