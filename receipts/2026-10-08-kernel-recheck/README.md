# Kernel recheck: section-(b) custom kernels on the current translator

Date: 2026-10-07/08. Actor: KernelRecheck (worker). Basecamp coverage item from Main.

## Source commits

- Harness + sources: omarchy-mlx main, first landed `795015644`, latest in
  this receipt `123469dec` (tools/kernel_recheck/, byte-exact generated-MSL
  dumper, per-kernel fp64/integer-exact references, GPU runner with
  subprocess isolation).
- Upstream kernel sources pinned at the inventory-header commits:
  mlx-lm `2184db2298042bf56807f0891746123e15528db7`,
  mlx-vlm `8f5dc3ddddbb8d7dd2b88ac51015def6f81fed21`,
  mlx-audio `17001a6950956302f15b53d86b601324efe716ba`,
  HF `avlp12/Kimi-K3-Alis-MLX-Dynamic-2.10bpw` k3_cbq.py / k3_fuse.py.
- Wheel: `mlx_omarchy-0.32.4.dev202610071347+9b5c938-cp314-cp314-linux_aarch64.whl`,
  sha256 `a2f8c83e5c5f635d00702565a8557d87c40a9eccac885329dcec4692d85d300d`
  (verified on each host before install).
- Translator fixes under test: branch `kernel-recheck-fixes`, merged to main
  `123469dec`.

## Kernel and Mesa versions (host M2)

- `uname -a`: `Linux jwm2-linux 7.1.12-2-12.3-sep-ARCH #1 SMP PREEMPT_DYNAMIC
  Wed, 07 Oct 2026 16:50:44 +0000 aarch64 GNU/Linux` (Aurora 12.3).
- ICD: `/usr/share/vulkan/icd.d/asahi_icd.json`, api_version `1.4.354`,
  library `libvulkan_asahi.so` (Honeykrisp stack selected in-process by MLX).
- Runtime shader compiler on the host: glslc `2026.4`, glslangValidator
  `16.4.0`.
- Python 3.14.7, numpy 2.5.3 (fresh venv per host).
- No VK_*/MLX_* environment overrides were set.

## Vulkan device and firmware identity

Not captured on the M2 pass (no vulkaninfo run); the ICD identity above and
the GPU-lock ticket runner are the isolation boundaries. To be captured with
the numerics rerun on jw16 (G13C) per the chip-independence note below.

## Exact commands

- Reference dry-run (CPU, no GPU): `python3 -m tools.kernel_recheck.run
  --reference-only` (26/26 ref-ok on the dev box and on each target host).
- GPU run: `python3 -m tools.kernel_recheck.run --gpu --timeout 90` under
  `~/bin/gpu-turn -m 12/15` on the M2, one kernel per subprocess, per-kernel
  stderr retained under `stderr/`.
- Translation-only diagnosis (CPU): generated MSLs via
  `python3 -m tools.kernel_recheck.dump_msl --out DIR` (write_signature
  replication verified byte-identical against the wheel's `verbose=True`
  dump for bitlinear_matmul), translated per file with
  `omarchy_custom_kernel_translate_dump`.

## GPU validation trajectory on the M2 (G14C), five wheels

| wheel | translator state | pass | wrong | refused | fail (compile) |
|---|---|---|---|---|---|
| v0.7.31 (a2f8c83e) | inventory baseline | 1 | 3 | 18* | 4 |
| fb0fe7a (145886cb6) | + constant-alias, numeric_limits, inline, header casts, as_type<int>, swizzle bare-use, under-supplied init, f16 widen, int-conversion | 8 | 2 | 9 | 7 |
| 05f823e (7368d2a91) | + fixpoint scanner, float-literal walk, const-comma skip, fabs→abs, as_type integer probe | 11 | 1 | 9 | 5 |

*pass-1 "refused" count is inflated: the classifier scraped traceback
headlines, so every child exception mentioning "unsupported" counted.
Per-kernel stderr files fixed the classification from the fb0fe7a run on.

llguidance_mask moved wrong→pass with the as_type integer probe (the
static-regex capture fix); bitlinear_matmul, both flux2 double/single,
banded_mask, moe_route, depthwise, qk_relu, phonon, situ_fused,
situ_pair_fused, moe_route_fused pass on GPU. The 9 refusals are exactly
the CBQ family with their predicted constructs. Remaining fails at
05f823e: 3 compile failures (sconv, down_combine, glue_pre/post) whose
GLSL still contains surviving C-casts inside constructor arguments —
reproduced with the post-wave-5 translator via the translate-dump tool on
the M2 (rebuild needed from a tree whose build-make cache matches; the
M2 scratch trees mixed generations during diagnosis) — and the harness
gaps: fused_single emulation (harness-side, kernel translates clean) and
the banded_mask_v2 call-argument fix (landed, 7368d2a91, untested on
GPU).

## Final GPU table (M2 G14C, fresh cache, current harness + main wheel)

The final GPU run used a fresh per-process translation cache (the runner
sets MLX_OMARCHY_SPIRV_CACHE to a fresh temp dir), eliminating stale-cache
confounds. Results are genuine current-translator outputs.

| status | count | kernels |
|---|---|---|
| pass | 11 | bitlinear_matmul, fused_double_norm_rope, inkling_banded_mask, inkling_moe_route, mlx_vlm_llguidance_mask, custom_depthwise_conv1d, qk_relu_squared, mlx_audio_phonon_unpack_base5_v1, moe_route_fused, situ_fused, situ_pair_fused |
| wrong | 1 | fused_single_norm_rope (harness emulation structural — kernel translates+compiles clean) |
| compile fail | 5 | inkling_sconv_decode, inkling_moe_down_combine, inkling_banded_mask_v2, kda_glue_pre, kda_glue_post (nested same-family C-casts survive inside constructor arguments; glslc: GL_NV_explicit_typecast) |
| refused | 9 | CBQ family (as_type<char4>/bfloat4, device pointer arithmetic, threadgroup alias, atomic_float) |

All 11 passing kernels are validated against fp64/integer-exact NumPy
references with NaN/inf detection and sha256 output checksums. The 9
refusals are the CBQ family with the predicted constructs.

The 5 compile-fail kernels need MSL pre-normalization or a parser-based
cast scanner (three regex-based rewrites each failed differently: fixpoint
never reaches inner casts; rescan corrupts its own output; recursive
segfaults). This is documented as the boundary of the regex-based approach.

## Fixed-wheel GPU revalidation (M2)

- Wheel built on the M2 under fill-run from the streamed main tree
  (886ac85f6): `mlx_omarchy-0.32.4.dev202610080451+31f0f66-cp314-cp314-
  linux_aarch64.whl`, sha256 `eecd5ca5a2ef638f843f451438ee9bf985bfb4069b1
  4708a87905ce2b1187dc7`. The build venv provided cmake; the parakeet
  encoder-whole bundle required by the build gate came from the extracted
  v0.7.31 release wheel (mlx/share/mlx-omarchy/parakeet-1/bundles/
  parakeet-encoder-whole, 458 MB, digest-verified by the build's own
  stage-whole-bundle gate).
- GPU rerun: fresh venv (venv-fixed) + harness `--gpu --timeout 90` under
  `gpu-turn -m 15` on the M2 (G14C).

## Numerical result (M2, pass 1 + diagnostic rerun, wheel = v0.7.31)

26 kernels = the 13 inventory rows expanded (bitlinear, 2x flux2, 5x
inkling, llguidance mask, depthwise conv1d, qk relu2, phonon unpack, 13 CBQ
family kernels). Reference side: 26/26 ok (NumPy fp64 or integer-exact).

Pass 1 classifier was lossy (traceback headlines). Diagnostic rerun with
CHILDERR capture, per-kernel stderr files, and byte-exact generated-MSL
dumps:

- pass 1: custom_depthwise_conv1d (after the atol floor for fp32
  cancellation; fp32 accumulation matches within 1.3e-7 rel).
- wrong value (worst class, 3): fused_double_norm_rope (1/294912 elements,
  1 bf16 ulp at a rounding boundary — inside tolerance after the floor),
  fused_single_norm_rope (reference bug: row stride must be FUSED_DIM not
  3*dim — fixed in the harness), qk_relu_squared (Metal executes half
  arithmetic in fp32; reference re-based on fp32-wide accumulation).
- glslc compile failures (7): bitlinear_matmul (`float sum[4] = {0.0};`
  under-supplied init; `1 / weight_scale[0]` is int/f16 in GLSL),
  mlx_vlm_llguidance_mask (`uint && bool`; numeric_limits),
  mlx_audio_phonon_unpack_base5_v1 (scalar small-array used as a value),
  and the bf16/f16-typed arithmetic behind the inkling and K3 glue kernels.
- translator named refusals (9): CBQ family as predicted —
  cbq_gather_mm/grad_* on `device pointer arithmetic` (int64 LUT loads,
  threadgroup-pointer alias), v3/v4/situ/glu on `as_type<char4>` /
  `as_type<bfloat4>`.

Predictions-vs-observed: 11/26 hits on the rerun classification. The
2026-09-29 "12 hard-failing kernels" list decomposes into: 1 passing
(depthwise conv), 3 reference-side artifacts (flux2 x2, qk relu2), and the
rest split between GLSL-compile gaps and genuine translator refusals
(CBQ family). No kernel produced a silent wrong value inside tolerance.

## CPU validation after the fixes (main `886ac85f6`, M2, fill-run)

Unit tests 8/8 pass (omarchy_custom_kernel_translate_tests). Per-kernel
translation of all 26 byte-exact generated MSLs
(omarchy_custom_kernel_translate_dump):

- OK 17: bitlinear_matmul, fused_double/single_norm_rope,
  inkling_banded_mask{,_v2}, inkling_sconv_decode, inkling_moe_route,
  inkling_moe_down_combine, mlx_vlm_llguidance_mask,
  custom_depthwise_conv1d, qk_relu_squared,
  mlx_audio_phonon_unpack_base5_v1, kda_glue_pre/post, moe_route_fused,
  situ_fused, situ_pair_fused.
- REFUSED 9 (exactly the CBQ family, exactly the predicted constructs):
  cbq_gather_mm/_v2/grad_d/grad_x on `device pointer arithmetic`
  (int64 LUT loads, threadgroup-pointer alias); cbq_gather_mm_v3/_v3_situ/
  _glu on `as_type<char4>`; cbq_gather_mm_v4/_v4_situ on
  `as_type<bfloat4>`.

Translation/refusal classification is chip independent (pure CPU
text->GLSL); this table holds for jw16 (G13C) and jwm1 (G13G). GPU
dispatch + numerics for the 17 need a wheel from main (build queued under
fill-run) and GPU turns; the two harness-reference artifacts (flux2 single
stride, qk fp32 accumulation) are fixed, so the rerun classification
expected on the fixed wheel is 17 pass / 9 refused, exit 0.

## Translator fixes under test (main `123469dec`..`0ec19daa9`)

- `constant`-space body pointer aliases rewrite like device aliases.
- `numeric_limits<T>::{infinity,lowest,max,min,epsilon}` map to exact IEEE
  patterns (full-expression match only; ordinary max/min calls untouched).
- `inline` header helpers lose the MSL-only qualifier.
- Under-supplied array initializers expand to the C zero-fill form.
- `bool &&`-chains route through overloaded and-helpers (identity on bool,
  `!= 0` on integers/floats).
- Constant-space small arrays the body never indexes become scalars.
- float16 buffer reads and typename templates widen to float exactly
  (Metal half = fp32 compute, half storage).

Unit tests for each (failing-before) in
`overlay/tests/omarchy/test_custom_kernel_translate.cpp`
(omarchy_custom_kernel_translate_tests).

## Chip independence

Translation and refusal classes are chip independent (pure CPU text->GLSL);
the CPU translation table in this receipt was produced on the M2 and holds
for jw16/jwm1. The numerical rerun (pass values, bf16 ulp boundaries) is
chip-dependent only through the GPU math, which the references bound; the
jw16/jwm1 numerics tickets are queued via the idle-guard backlog.

## Backend dispatch trace

Per-kernel dispatch is one vkCmdDispatch per kernel call (single compute
pipeline per custom kernel, SPIR-V from glslc, no CPU fallback). The omarchy
error contract raises at eval time; no silent reroute exists
(overlay/mlx/backend/omarchy/unsupported.h).

## Recovery result

Not applicable: no reboot, no driver reload, no worker bring-up. All runs
inside gpu-turn tickets; the fill-run CPU build yields to MEASURE work and
restarts incrementally.
