# Non-transposed quantized matmul backward: composed GPU route for the bf16 small-k zone

Lands: `agent/qmm-backward` at `0ccc16b9` (base: origin/main `dd13652c6`).

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
| `omarchy_matmul_family_tests` (full, 24 cases incl. new) | M2 Max (G14C), gpu-turn ticket | 24/24, 82,942,463 assertions (second run; first run had one order-transient failure of the jumbo f16 tile case `m=1023 n=4864 k=4864` which passes in isolation and passed in the rerun) |
| `omarchy_primitive_tests`, dev-box full family sweep | in flight | recorded when landed |

## Provenance

- Source: `agent/qmm-backward` `0ccc16b9`; no shader files added or
  changed (nothing new to compile; existing `Dequant*`/matmul kernels
  reused).
- Wheel for the python probe: built with `DEV_RELEASE=1
  MLX_OMARCHY_SOURCE_COMMIT=0ccc16b9` (stamped into the version).
- Hosts: "M2 Max (G14C)" for the T6021 leg, "M1 Max (G13C)" for the T6001
  leg (placeholder convention, no hostnames).

## Notebook

Private working record:
`apple-silicon-lab/entries/QmmBackward/20261005T1932Z-omp-studio-local-qmm-non-transposed-bf16.md`
(pre-registered before the first build; raw logs under
`artifacts/QmmBackward/`).
