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
  logits in fp32 so greedy token ids match the fp32 CPU reference exactly.
- `gen_qwen3_fast.py` — greedy-decode harness: runs the same prompt through
  torch CPU eager and `Qwen3Fast`, prints per-step tok/s and token-id match.

## Run

Inside a venv with `mlx-omarchy`, `torch` (CPU wheel), `transformers` and
`safetensors`:

```bash
python tools/torch-mlx-vulkan/gen_qwen3_fast.py --model /path/to/Qwen3-0.6B --dtype bf16
```

`--dtype fp32`/`fp16` run the same lowered graph at other precisions.
Receipt: `receipts/2026-11-23-route-d-qwen3-torch-mlx-vulkan.md`.
