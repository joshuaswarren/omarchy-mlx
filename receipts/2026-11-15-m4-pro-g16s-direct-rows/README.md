# M4 Pro (G16S): direct-route rows in kDirectMatmulRows

Six rows in `overlay/mlx/backend/omarchy/matmul_direct_select.h` select
the already-shipped direct kernels on the Apple M4 Pro (G16S):

- bf16 NT (`a @ w.T`): `MatmulDirectBF16NtK4S8` from m = 512, n = 1024
- bf16 TN (`a.T @ w`): `MatmulDirectBF16TnWS8` from m = 1024, n = 1024, and
  from m = 512, n = 12288
- f16 NT: `MatmulDirectF16NtK4S8` from m = 512, n = 1024
- f16 TN: `MatmulDirectF16TnWS8` from m = 1024, n = 1024, and from m = 512,
  n = 12288

There is no f32 row. `test_matmul_family.cpp` pins every table row,
including these, at its measured (m, n) floor.

The bf16 NT path needs `VK_KHR_shader_bfloat16` with bfloat16 cooperative
matrices, which aurora-silicon/mesa#350 exposes on AGX3B. Without that
ICD the bf16 cells stay on the staged route and the dense A/B is flat.

## Host

Apple M4 Pro (J616s, T6040, G16S B1), 14 CPU cores, 48 GB, kernel 7.3.0,
stock clocks. The A/B below used the ICD from aurora-silicon/mesa#350
(Mesa 26.3.0-devel, `git-8135f91807`), Borealis, with
`VK_KHR_shader_bfloat16` (shaderBFloat16Type and
shaderBFloat16CooperativeMatrix). Full stock host facts:
`receipts/2026-11-13-m4-pro-g16s-gpu/host.txt`.

Python runs used `-I` (ignores `PYTHONPATH`). Wheels: base
`0.32.4.dev202611150046+a356ccd`, head
`0.32.4.dev202611150202+5cc26dc` (the six rows on the #439 head). This
branch carries the same six rows plus the floors test onto `main`
(`420114091`).

## Dense matmul (`mm.txt`)

Eight independent matmuls per timed evaluation, 5 timed repetitions,
median microseconds, three interleaved rounds base/head, median TFLOPS,
K = 4096. Output sha256 (first 12 hex digits of the f32 bytes) identical
on every shape and every round (48 shapes: f16/bf16 × NT/TN × M in
{512, 1024, 2048, 4096} × N in {1024, 4096, 12288}).

| dtype | layout | change vs base (12 shapes) | notes |
|---|---|---|---|
| bf16 | NT | +78.3% to +87.0% (median +85.3%) | needs mesa#350 |
| bf16 | TN | −1.2% to +7.4% (median +6.0%) | +5.2% to +7.4% where a row applies; M = 512, N < 12288 has no TN row |
| f16 | NT | +1.8% to +4.7% (median +2.5%) | |
| f16 | TN | −0.7% to +7.3% (median +6.1%) | +4.6% to +7.3% where a row applies; M = 512, N < 12288 has no TN row |

On stock Borealis 26.3.0-devel (`git-05ddb00f5c`, no
`VK_KHR_shader_bfloat16`) the same two wheels were previously flat
(−0.8% to +1.3%, sha identical).

## Model decode and prefill (`models-4bit.txt`, `models-bf16.txt`)

Same ICD and wheels, `mlx_lm` greedy, chat prompt "Explain in three
sentences why the sky is blue.", 128 tokens (granite stops at 53),
prefill at about 300 and 2000 tokens, three interleaved rounds, median.
Tokens identical in every arm.

| model | pf ~300 | pf ~2000 | decode | tokens |
|---|---|---|---|---|
| Qwen3-0.6B (bf16) | 1867 → 1897 tok/s (+1.6%) | 1999 → 2568 tok/s (+28.5%) | 113.8 → 114.3 (+0.4%) | identical |
| Qwen3-8B-4bit | 312.4 → 312.4 (0) | 326.2 → 326.0 (−0.1%) | 42.06 → 42.18 (+0.3%) | identical |
| granite-4.0-1b-4bit | 1210 → 1209 (0) | 1150 → 1151 (+0.1%) | 108.4 → 107.8 (−0.6%) | identical |

The 4-bit models do not take these dense rows; they are a no-regression
check. The Qwen3-0.6B pf2k lift is disjoint across the three rounds
(base 1999–2041, head 2567–2624).

## Tests

On this branch, stock Borealis: `omarchy_matmul_family_tests` 46/46,
`omarchy_primitive_tests` 107/107, `omarchy_runtime_tests` 60/60. The
floors test includes the six new G16S rows.

On the mesa#350 ICD the same matmul and primitive binaries report 6 and
2 failures. They are quantized-matmul pipeline creates: the ICD's AGX
shader packer refuses some qmm cooperative-matrix shaders. The same 6
`omarchy_matmul_family_tests` failures appear on an unpatched tree with
that ICD, so they are not caused by these rows.

`SHA256SUMS` covers every file in this directory.
