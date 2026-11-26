# 4-bit greedy degeneracy on M1 — root cause: f16 attention-score materialization in the omarchy SDPA fallback

Status: FIXED. Root cause proven on real tensors (m1-test-host-linux, f61f3cf,
2026-09-01); the f32-score `ScaledDotProductAttention::eval_gpu` and the
`Repeat`/strided-reshape fix landed in `overlay/` the same day and all
owed verifications are green (receipts below). Nothing committed. Probe
scripts on the M1: `/tmp/qdiag1..19.py`, logs `qdiag*.log` and
`wheel13.log` in the repo root on the M1.

## Symptom (receipts/2026-08-31-m1-mlxlm-fp16-smoke.md, Attempt 10)

Qwen2.5-0.5B-Instruct-4bit greedy text is degenerate (`1.000000` repeated)
while the bf16 build "generates coherently". Both observations are the same
defect at different severities; see below.

## What was ruled out (all verified bit-exact or 1-2 ulp vs host float64 on
real M1 Honeykrisp, real model tensors)

1. Embedding path: device gather of words/scales/biases (int32 and uint32
   ids) bit-exact; `mx.dequantize` f16 kernel bit-exact vs f16-rounded host;
   f32 kernel exact (`qdiag1` stage 1).
2. QuantizedMatmul: layer-0 q_proj on the real embedding input matches host
   f64 to 3.5e-4 relative; dense f16 matmul of device-dequantized weights
   matches too (`qdiag1` stage 2, `qdiag3`).
3. `mx.repeat`-free unit suites: full `omarchy_primitive_tests` on the M1
   (Honeykrisp): 65/65 cases, 525359 assertions, 0 failed. Same suite on
   llvmpipe: 65/65, 525359, 0 failed.
4. RMSNorm output, RoPE (offsets 0-7 sweep, f16, traditional=False),
   attention output (fused sdpa vs host f64 on the same tensors: 0.0046 at
   T=8), o_proj, MLP: all within f16 rounding noise when compared correctly.
5. The 72 per-layer `q/k/v_proj.bias` tensors in the 4-bit checkpoint are
   NOT junk: they are bit-equal to the original bf16 model's trained biases
   (Qwen2.5-0.5B has attention biases; absmax 79/130 are real outlier
   features). `mlx_lm` loads them via `QuantizedLinear` with
   `attention_bias`-style bias params; the fused sdpa path handles them.

## The defect

`ScaledDotProductAttention` is `OMARCHY_USE_FALLBACK` in
`overlay/mlx/backend/omarchy/primitives.cpp`, so
`mlx::core::fast::scaled_dot_product_attention` (mlx/fast.cpp:714) runs its
composed fallback graph with the input dtype. For f16 models that
materializes attention scores as f16 after `matmul(q, k^T)`.

Measured on layer 0, 41-token chat-template prefill (`qdiag18`):

- attention scores reach absmax 647.5 (scaled) — Qwen outlier dims plus the
  trained q/k biases produce huge logits;
- f16 ulp at 647 is 0.5;
- 59.8% of causal softmax rows have a top1-top2 score gap below 1.0 (38.8%
  below 0.25);
- the fused sdpa output therefore flips softmax winners vs exact math:
  maxabs deviation 0.443 on outputs of magnitude ~0.17 (`qdiag18`), i.e.
  attention reads different v rows than exact arithmetic;
- with f16 scores, variants overflow to NaN (`qdiag18`/`qdiag19`).

The winner flips poison the residual stream from layer 0 (layer0 device-vs-
host-f64 maxabs 11.3 at `qdiag12`), amplify through the Qwen
massive-activation dims (t=20, dim=62 saturates ~±1000 from layer 3 on,
`qdiag16`), and degenerate the greedy argmax chain: device greedy gives
`'  \n唯epy repmat...'` while a host float64 reference of the SAME weights
and prompt gives argmax 59604 = `'Paris'` with logit 24.4 (top-1 by 6.9,
`qdiag12`). The bf16 model diverges from its own float64 reference the same
way and answers Chinese text on the same prompt — the defect is
dtype-independent; 4-bit is just where it was first seen.

Host float64 agreement between the 4-bit and bf16 models (logits argmax both
59604 'Paris', maxabs 5.7) proves the checkpoint and quantization are fine.

## Fix (landed in `overlay/`, not committed)
Measured layer-0, T=41 prefill (`qdiag19b`): the f32 composition matches
host float64 attention output to maxabs 0.000006, while the current fused
fallback deviates by 0.4429 on outputs of magnitude ~0.17 — a 7e4 reduction
in deviation. The validated sequence (all ops verified finite on device):

    q32s = scale * astype(q, f32)            # unflatten GQA: (1, kv, rep, L, D)
    k32   = expand_dims(astype(k, f32), 2)   # (1, kv, 1, kL, D)
    v32   = expand_dims(astype(v, f32), 2)
    scores = matmul(q32s, swapaxes(k32, -1, -2))          # 5-D f32 matmul
    vis    = greater_equal(arange offset broadcast)        # bool (qL, kL)
    scores = scores + (1 - cast(vis, f32)) * -1e30         # additive, no Select
    probs  = softmax(scores, -1)                           # f32
    out    = astype(flatten(matmul(probs, v32), 1, 2), out.dtype)

Implementation: `overlay/mlx/backend/omarchy/primitives.cpp` —
`ScaledDotProductAttention::use_fallback(...)` now returns false for
inference shapes (training logsumexp keeps the composed graph; the
`force_fused` throw stays), with this `eval_gpu` filling `out` via
`out.copy_shared_buffer(result)` for matching dtypes and a Vector cast
copy for f16/bf16 outputs. Two safe deviations from the letter of the
probe sequence, same tensors: the scale is applied with a scalar
`Multiply` after the cast exactly as validated (a host-written
host-visible scalar buffer, the Load idiom), and the causal mask is the
same float32 additive tensor (0 attended / -1e30 masked) written
directly into a fresh allocation instead of assembling it from
arange/greater_equal/(1-cast)*-1e30 — the values the softmax consumes
are identical, and arange, Select, and repeat stay out of the path.

Caveat found while validating: `mx.repeat` on omarchy returned garbage
(values ~2e27 and NaN) for f32 [1,2,41,64] -> [1,14,41,64] (`qdiag19`).
The validated composition above avoids `repeat` entirely by using the
upstream unflatten/expand_dims GQA shape, which works on omarchy in 5-D
f32.

## Repeat: root cause and fix

`mx.repeat(a, n, axis)` composes expand_dims -> broadcast_to -> reshape.
The `Broadcast` primitive output is a shared-buffer view whose stride-0
axes make `size()` (74752) exceed `data_size()` (5248) while the
span-based `flags().contiguous` stays true. Omarchy's `reshape_gpu`
trusted that flag and issued a flat `copy_buffer` of `out.nbytes()`
from a 20992-byte source allocation: rows past the source span read
foreign memory (0.0, ~2e27, NaN). Reproduced on llvmpipe: `r[0,1]`
began at element 2624 (linear source layout) and `r[0,7]` read past
the buffer. Transposed views (`data_size == size`, permuted strides)
would have taken the same wrong flat copy silently.

Fix: `reshape_gpu` in `overlay/mlx/backend/omarchy/copy.cpp` now
expresses every copy-necessary reshape as a General strided copy
through the copy engine (mirroring the Metal backend); float dtypes
plus int32/uint32 via a raw-word kernel, and the `strided reshape`
named error covers the rest (int64 and 8-bit dtypes pinned in tests).

## Landed receipts (2026-09-01, f61f3cf + WIP overlay copies)

Local llvmpipe (x86, MLX_OMARCHY_ALLOW_NON_APPLE=1):

- `omarchy_primitive_tests`: 69/69 cases, 562108 assertions, 0 failed
  (baseline 65/525359; new: f16 ~600-score, causal f16 cache-offset,
  bf16 ~600-score, repeat regression + int32 named error).
- `omarchy_kv_ops_tests` 14/14 (407), `omarchy_ane_bundle_tests`
  12/12 (131), `omarchy_copy_offset_tests` 7/7 (68),
  `omarchy_runtime_tests` 22/22 (6188).
- Python level (rebuilt cp311 wheel): `mx.repeat` exact on all 14 GQA
  heads; fused sdpa finite at scale 6.0 with f16 GQA inputs.

M1 Honeykrisp (m1-test-host-linux, `.work/venv-fp16`):

- Wheel rebuilt from the WIP `.work/mlx` copies: `wheel13.log`
  (verbatim receipt line):
  `Created wheel for mlx-omarchy: filename=mlx_omarchy-0.32.2.dev20260901+f61f3cf-cp314-cp314-linux_aarch64.whl size=2609184 sha256=3bf1754e41dcb9c49a6bc303275db7b5ddff3651dfd286e52f297c3c276f058f`
  Force-reinstalled into `.work/venv-fp16` (`Successfully uninstalled
  mlx-omarchy-0.32.2.dev20260901+f61f3cf` / `Successfully installed
  ...+f61f3cf`); `mx.__version__ 0.32.2.dev20260901+f61f3cf`, device
  `Device(gpu, 0)`.
- `omarchy_primitive_tests` on Honeykrisp: 69/69, 562108, 0 failed.
  `omarchy_kv_ops_tests` 14/14 (407); `omarchy_ane_bundle_tests`
  12/12 (131).
- `qdiag12` rerun (`qdiag12-fixed.log`): 4-bit layer0 device-vs-host
  f64 maxabs 0.0104 (was 11.3), worst layer 4.39 at the ~±1000
  massive-activation layers (f16 ulp 0.5-1 there — rounding scale),
  logits maxabs 0.040, and `argmax device/host: 59604 59604` =
  `Paris`. bf16 tracks its own f64 reference the same way (layer0
  0.187) and also answers `Paris`.
- Greedy 4-bit, prompt `What is the capital of France? Answer in one
  word.` (`MLX_DISABLE_COMPILE=1`): output `Paris` + EOS at both 8
  and 32 max tokens (prompt 41 tok ~19.2 tok/s, peak 0.292 GB; the tok/s
  figure was measured on m1-test-host-linux with 1 of 8 CPU cores online, boot
  entry `Omarchy ANE test`, so it may improve on 8-core re-measurement).
  The
  Attempt 10 `1.000000` degeneracy is gone.
- Python level on Honeykrisp: fused sdpa finite at scale 6.0 with f16
  GQA inputs; `mx.repeat` heads 5 and 12 exact vs source.


## Verification owed after the C++ lands — all green, see receipts above

1. `qdiag12` on the M1: per-layer device-vs-host f64 must drop from ~1e3 to
   f16 rounding scale, and device greedy must equal the host float64 greedy
   (`Paris`).
2. `mlx_lm generate` 4-bit greedy France prompt: sane answer.
3. Full `omarchy_primitive_tests` green on llvmpipe and Honeykrisp.

All three owed verifications above are green. Nothing committed; the
M1 tracked tree stays clean (fixes live in `.work/mlx` copies and the
dist-wip wheel).
> **Annotation 2026-09-03:** the decode tok/s figures in this document are EOS-truncated short-burst rates, not steady-state decode (generation stopped after 2-10 tokens under `--max-tokens 32`). They are not comparable across machines or wheels. See `receipts/2026-09-03-decode-metric-fix.md`; replacement protocol: `scripts/bench_decode.py` (pinned length, token-count assertion).

