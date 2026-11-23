# Route D — Qwen3-4B through the generic `torch.compile(backend="mlx_vulkan")` path (4-bit)

Companion to `2026-11-23-route-d-generic-torch-compile-qwen3.md` (0.6B generic
path). This run: the same generic backend on the real HF
`Qwen3ForCausalLM` at 4B, weights quantized to 4-bit at ingest.

## Environment

- Host: jwm1, Apple T8103 (M1, 13-inch), 15 GB RAM, Asahi Linux
- Kernel `7.3.0-2-20261110.1-lab-ARCH`, mesa `1:26.2.4-1`, Honeykrisp Vulkan
- `mlx.core` from the omarchy-mlx wheel; torch `2.15.0+cpu`, transformers `5.22.1`
- Model: `mlx-community/Qwen3-4B-Instruct-2507-4bit` (U32 packed + bf16
  scales/biases, group_size 64), dequantized to a bf16 HF snapshot
  (`qwen3-4b-4bit-dequant`, 7.6 GB on disk) so `transformers` can trace it
- Command: `SNAP=<dequant> MDT=bf16 NTOK=16 PHASE=mlx REFJ=/tmp/ref4b_ids.json
  MV_QUANT=4 python gen_qwen3.py`

## Why 4-bit (measured, not assumed)

Vulkan device-heap probe on jwm1: allocation fails at ~7.9 GB total
(`DRM_IOCTL_ASAHI_GEM_CREATE failed` after 7936 MB of 256 MB buffers).
bf16 weights for this model are ~7.5 GB — over the ceiling before cache,
activations, or the ~8 GB torch-side page cache. 4-bit packed weights are
~2.3 GB and fit with headroom.

`MV_QUANT=4` enables a new ingest path in `mlx_vulkan.py`:

- `_quantizable_phs`: a `G_*` (static param) placeholder qualifies iff every
  consumer chain — through transparent ops (`t`/`transpose`/`permute`/
  `to`/`detach`/`clone`/`contiguous`) — terminates at a matmul-family op
  (`linear`/`matmul`/`mm`/`bmm`/`addmm`). The tied `embed_tokens` fails the
  check (its `embedding` consumer needs real rows) and stays bf16; the tied
  `lm_head` matmul therefore also runs bf16.
- Qualified params convert once with `mx.quantize(w, group_size=64, bits=4)`
  and are stored packed; `mx.eval` is forced per tensor because MLX laziness
  otherwise keeps every bf16 source alive until the first compiled call
  (that bug OOM'd the first attempt).
- In `build()`, `_QW` wrappers carry a transpose flag through `t`/`transpose`/
  `permute(1,0)`; `linear` emits `mx.quantized_matmul(x, w, s, b,
  transpose=True)` and `mm`/`matmul`/`bmm` use `transpose=q.tr`. Any other
  consumer dequantizes on the fly (`mx.dequantize`) so correctness is kept
  even if the eligibility analysis missed a consumer.
- Executor stats this run: `fused_rms=145  state=108  quant=252`.

## Result

Prompt: `The capital of France is` — 16 greedy tokens.

| path | decode | ids |
|---|---|---|
| torch CPU eager bf16 (MKLDNN disabled) | 7.50 s/tok (~0.13 tok/s), prefill 98.1 s | reference |
| **generic torch.compile mlx_vulkan, MV_QUANT=4** | **0.0782 s/tok = 12.79 tok/s**, prefill (eager) 51.2 s | **16/16 identical** |
| native `mlx-lm` on the original 4-bit checkpoint | 24.245 tok/s gen, 2.364 GB peak | — |

MLX ids `[12095, 13, 3555, 374, 279, 6722, 315, 9856, 1939, 785, 6722, 315,
9856, 374, 19846, 13]` == CPU ids; `MATCH: True`.
Decoded: ` Paris. What is the capital of Germany?\n\nThe capital of Germany
is Berlin.`

- ~1.9x slower than native mlx-lm (inside the 2x bar from milestone 3's
  framing), ~96x the CPU reference.
- Memory stayed comfortable: `free -g` showed ~12 GB available during decode.

## Gotchas found

- `torch.Tensor.numpy()` rejects bf16 on this stack; `_mx` now round-trips
  bf16 through a `uint16` view + `mx.view(..., mx.bfloat16)` (bit-exact,
  no fp32 temp).
- First quantized attempt OOM'd (`vkAllocateMemory`, 48 MB request): lazy
  `mx.quantize` deferred every conversion, holding ~8 GB of bf16 sources.
  `mx.eval` after each quantize bounds peak to one tensor.
- Re-quantizing the dequantized bf16 is **not** bit-exact vs the checkpoint's
  original packing (`mx.quantize` recomputes scales; max abs err 0.035 on
  `layers.0.mlp.gate_proj.weight`). Greedy ids still matched for all 16
  tokens; longer horizons may diverge. A byte-exact path would load packed
  weights straight from the checkpoint — not possible for the generic backend
  today because placeholders carry dequantized torch values with mangled
  `G_*` names.
- `aarch64` MKLDNN bf16 GEMM is broken in torch 2.15 (SIGSEGV-class failure in
  `F.linear`); the CPU reference phase runs with
  `torch.backends.mkldnn.enabled = False`.

## Caveats

- Prefill runs torch-eager bf16 on CPU (51 s) — the compiled path covers the
  decode step only, same as the 0.6B driver.
- 12.79 tok/s is decode-only steady state over 14 timed steps (first compiled
  call is the warmup token, untimed). Compile/warmup adds tens of seconds.
- Quantized lowering applies to `linear`/`matmul`/`mm`/`bmm`; `addmm` against
  a packed weight dequantizes on the fly.
