# Non-transposed quantized matmul backward: composed GPU route for the bf16 small-k zone

Lands: implementation commit `4f74b4433` on `main`; the current main tip when this receipt was refreshed was `04f825676`.

## The blocker, named precisely

LoRA backward through a quantized linear computes dx as
`quantized_matmul(dy, w, scales, biases, transpose=false)` (upstream
`QuantizedMatmul::vjp`). The 2B hybrid checkpoints put one small-out
projection per block (`in_proj_ba`-class, W `[64, 2048]`, group 64,
4-bit), so that dx call has k=64, m=batch*seq, n=2048 — and hit:

> RuntimeError: [omarchy] QuantizedMatmul bf16 non-transposed tile is not
> implemented for the Omarchy Vulkan backend (dtype=bfloat16,
> shape=[1,32,2048]).

The refusal (`1474514e4`, 2026-09-11) guards exactly
`bf16 && !transpose_ && m>1 && bits!=2 && k<=128`: the bf16 non-transposed
tile's accumulation order diverges from the dense bf16 matmul reference by
one output ULP there (upstream sweep tolerance 1.5e-3 is tighter than one
ULP at those magnitudes). Every other non-transposed affine shape already
had a GPU route (tile K>=256, bits 2, f16/f32, m==1 vec). `unsupported`
prints the output array's shape, which is why the error read as a
"[1,32,2048]" matmul rather than a k=64 one.

## The change

- `dispatch_affine_dequantize`: the affine dequantize dispatch extracted
  from `fast::Quantize::eval_gpu`'s dequantize direction into one shared
  helper (no new shader; the existing `Dequant*` kernels serve bits
  2/3/4/5/6/8, group sizes with `(gs*bits)%32==0`, f16/bf16/f32).
- `QuantizedMatmul::eval_gpu`: the refused zone now composes the existing
  GPU kernels — Dequant materializes w into an encoder temporary and the
  dense matmul kernel contracts dy against it (2D weights, rank-3 x with a
  shared 2D weight, and rank-3 paired 3D weights all collapse through
  `dispatch_matmul`'s batch handling). The accumulation order IS the dense
  reference the sweep pins. `MLX_OMARCHY_NO_QMM_NT_COMPOSED=1` restores the
  named refusal for A/B.
- No CPU tensor primitive is involved: both legs are Vulkan dispatches on
  the same encoder.

## Test

`omarchy_matmul_family_tests` gains `qmm non-transposed small-k zone
matches host reference`: fp64 host reference (round-tripped operands),
family bound `(3k+2)·m_bound·2^-23 + m_bound·2^-(mantissa+1)`, covering
group 32/64/128 x bits 4/8 bf16, m=17/33, n=128 and the n=2048 k=64
blocker class, rank-3 shared and paired weights, and pinning the
untouched routes in the same zone (f16, f32, bf16 bits 2).
`artifacts-qmm-backward-probe.py` runs `mx.grad` through
`mlx_lm.tuner` `LoRALinear.from_base` on an `nn.QuantizedLinear`
(group-64 4-bit) against a dequantized dense reference.

## Results

| suite / probe | machine | result |
|---|---|---|
| new nt test case | dev box (llvmpipe, staged rehearsal ICD, `MLX_OMARCHY_ALLOW_NON_APPLE=1`) | 1/1 case, 80/80 assertions |
| `omarchy_matmul_family_tests` (full, 24 cases incl. new) | M2 Max (G14C), gpu-turn ticket | 24/24, 82,942,463 assertions (second run; first run had one order-transient failure of the jumbo f16 tile case `m=1023 n=4864 k=4864` which passed in isolation and in the rerun) |
| `omarchy_matmul_family_tests` (full) | dev box (llvmpipe) | 24/24, 21,071,998 assertions |
| `omarchy_matmul_family_tests` (full) | M1 Max (G13C) | 24/24; `omarchy_primitive_tests` 104/104, 2,743,003 assertions |
| LoRA dx grad probe | dev box (llvmpipe; wheel `0.32.4.dev202610052022+875ad7ea`) | finite; max abs diff vs dense+same-LoRA reference 1.9073486e-6, bound 0.00391412; pass |
| 2B Q4 LoRA training smoke | M2 Max (G14C), M2 GPU FIFO | 10 iterations completed; 0.873 it/s, 25.067 trained tok/s, peak memory 2.332 GB; validation loss 1.897 at iter 1 and 0.917 at iter 10 (net decrease, not monotonic). Adapter generation returned `Paris.`; trace counters reported `cpu_command_encoder_calls=0` in all 3 traces. Artifact: `artifacts/QmmBackward/20261005T1932Z/m2-qwen38-lora-smoke.txt`. |
| QMM op vs pre-dequant dense matmul | M2 Max (G14C), M2 GPU FIFO | Synthetic Q4/g64 `[64,2048]` weights, bf16 x `[32,64]`, 40 evals: composed route 0.244744 ms/eval; pre-dequant dense matmul 0.309661 ms/eval (composed 20.96% lower latency). Outputs equal; max abs diff 0.0. This is an op-level comparison, not a full-step A/B. Artifact: `artifacts/QmmBackward/20261005T1932Z/m2-qmm-vs-predequant-dense.txt`. |

## Provenance

- Implementation: main commit `4f74b4433`; no shader files changed or added. Existing Dequant and matmul kernels are reused.
- M2 smoke wheel: `mlx_omarchy-0.32.4.dev202610052115+5f77de8-cp314-cp314-linux_aarch64.whl`, built from source `5f77de8a2` which includes implementation `4f74b4433`; SHA256 `f0e3035c5e2ee5a40ccbd4471cd93eaff1fb863b9c10c2d79a6d1be8c2f7bfa2`. Provenance gate: `verified=match`, `version_match=true`, both binary RECORD hashes matched.
- LoRA gradient probe wheel: `0.32.4.dev202610052022+875ad7ea` from the probe-refined source recorded in the earlier dev-box result.
- Hosts: `M2 Max (G14C)` for the T6021 leg, `M1 Max (G13C)` for the T6001 leg (placeholder convention, no hostnames).

## Notebook

Private working record:
`apple-silicon-lab/entries/QmmBackward/20261005T1932Z-omp-studio-local-qmm-non-transposed-bf16.md`
(pre-registered before the first build; raw logs under
`artifacts/QmmBackward/`).
