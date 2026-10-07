# 1007b part B: gather_qmm_sub coverage audit (dev-box read, no GPU)

## Layouts that reach the subgroup kernel today (primitives.cpp:3391-3397)

`use_sub` requires: not fp_mode, matrix_m == 1, transpose, bits == 4,
group_size == 64, out dtype bfloat16, subgroup_size 32, ARITHMETIC
subgroup op, 16-bit storage access. The shader variants exist for
f32/f16/bf16 (CMakeLists.txt:582-584) but the gate additionally pins
out dtype bfloat16 for the bf16 arm only; f32/f16 compile but are not
selected unless the gate widens.

## Layouts that fall to the scalar kernel (with file:line)

- bits 8 (any group size): primitives.cpp:3396 `bits == 4u` gate.
- group_size 32 and group_size 128 (4-bit): same line.
- f16 out / f32 out with fp_mode unset: primitives.cpp:3415-3427 select
  the scalar GatherQmm* / GatherQmmNb* arms even when use_sub could be
  true for f16 (the shader variants compile; the bf16-only gate pins
  out dtype bfloat16 at :3396).
- m > 1 (batched routed decode): primitives.cpp:3390 `matrix_m == 1`.
- non-transposed weights: same `transpose` term.
- fp_mode (codebook/fp quant): excluded by design at :3390.

## Minimal coverage proposal (in ship order)

1. bits == 8, group 64/32, bf16, transposed, m == 1: the shader's
   dequant constants are per-(bits, group) table entries; add
   -DGATHER_QMM_SUB_BITS8 (+G32/G64 spec-constant) variants and extend
   the gate. Widely shipped: Qwen2/Qwen3 MoE 8-bit exports and
   GLM-4.5/DeepSeek-V3 FP8-derivative 8-bit community quants ride
   bits8/g64+g32 in HF mlx-community conversions (evidence: the 0.32
   line's qmm tests already carry a bits8/g32 f32 scalar variant that
   failed pre-PARAM_BYTES-fix - the same layout class the scalar kernel
   is known-good on today).
2. group_size 128, 4-bit, bf16: one more group-extent constant; shipped
   by Llama-4/Qwen3-Next style 4-bit g128 conversions.
3. f16 out: mirror the bf16 arm (shader already compiled; gate term
   removal + parity runs).
Each step needs the jwm1/jw16 parity suite per layout (the scalar
kernel stays the fallback until its parity is proven - the file
documents f32-T and bits8/g32-f32 failures pre-fix, so the history says
per-layout proof, not gate widening on faith).

## Test matrix (host-reference, jw16 when GPUs return)

Per newly gated layout: (m=1, K in {256, 512, 1024, 4096}, N in {512,
1024}, affine transposed, group size per layout, bf16) vs the scalar
kernel and the CPU reference; assert bit-exact-or-within the numerics
bar; plus z-chunk coverage with count > 65535.

## Size estimate

Steps 1-2 are spec-constant + gate changes (~40 lines) plus parity
runs; step 3 is gate-only. No new shader files.
