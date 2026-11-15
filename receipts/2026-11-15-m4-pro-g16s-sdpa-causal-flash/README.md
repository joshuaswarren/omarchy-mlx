# M4 Pro (G16S): default the causal coopmat flash SDPA prefill route

Date 2026-11-15. Device: Apple M4 Pro (G16S B1). ICD: aurora-wip a55a1a04 + mesa #353 + #343 + #350 + #355 (Mesa 26.3.0-devel `git-f1a664ebd6`). Wheel `mlx_omarchy-0.32.4.dev202611150046+a356ccd` (venv-gsA), `python -I`, `MESA_SHADER_CACHE_DISABLE=true`, flock `/tmp/m1-gpu.lock`. Model `mlx-community/Qwen3-8B-4bit` (bf16, 4-bit group-64). Prompts were 520 and 2060 tokens; 32 greedy decode tokens after each prefill.

## Change

`SdpaCausalFlashCoopmatBF16` stays env-gated on every part except G16S. On G16S, when the device advertises `cooperative_matrix_bf16_8` (mesa #350), the route is the default for the existing contract (`q_len >= 64`, causal, bf16, hd 128, …). `MLX_OMARCHY_SDPA_CAUSAL_FLASH=0` is the kill switch; `=1` still forces the route on any eligible part. The device test now pins the composed arm with `=0` so a G16S default does not compare flash against itself.

## Why

A Qwen3-8B-4bit prefill breakdown on this ICD (unprofiled 5-run median, plus one dispatch trace per length; traces are for mix/share only):

| length | tok/s | qmm kernel | qmm share | qmm TFLOPS vs 7.30 f32 peak | SDPA family |
|---|---|---|---|---|---|
| pf520 | 330.49 (329.70–331.34) | `QmmPrefillCoopmatBF16X32FullNLdsPad` | 81.1% | 5.755 (78.8%) | 6.2% at 1.54 TFLOPS (staged QK + softmax + PV) |
| pf2060 | 318.10 (316.31–318.35) | same, plus `QmmPrefillCoopmatM16BF16X32FullN` 2.1% | 77.9% | 6.088 (83.4%) | 14.4% at 2.65 TFLOPS |

QMM is already LDS_PAD + raster G4, f32×f32 coopmat (not FMA). The second bucket is staged prefill attention, far from the roof, and it grows with T. The causal flash kernel was already in tree, default off, because the 2026-10-09 e2e +2% bar was only clearly met on G13G and G16S was not measured.

A profile with `MLX_OMARCHY_SDPA_CAUSAL_FLASH=1` at pf2060 replaced the staged QK/PV/Softmax (841.6 ms) with `SdpaCausalFlashCoopmatBF16` n=35 (475.5 ms, 0.565×).

## End-to-end A/B

Five interleaved rounds, `CAUSAL_FLASH=0` vs `=1`, same wheel and ICD, 32-token sha identical in every arm (`a8bdfde11f24` at pf520, `1dd5a67f8afc` at pf2060). Raw: `ab-timing.txt`.

| metric | composed median (range) | flash median (range) | change |
|---|---|---|---|
| pf520 tok/s | 329.78 (324.16–331.90) | 329.89 (329.43–333.60) | +0.03%, overlap |
| pf2060 tok/s | 318.55 (317.64–321.01) | 338.32 (337.39–339.24) | **+6.21%, ranges disjoint** |
| decode after pf520 tok/s | 41.40 (39.51–42.05) | 41.17 (40.77–44.42) | −0.56%, overlap |
| decode after pf2060 tok/s | 33.16 (30.37–33.89) | 33.35 (31.16–34.77) | +0.57%, overlap |

## Not claimed

- Default-on for parts other than G16S.
- A change in qmm, rmsnorm, or rope.
- Bit-identity of flash vs composed attention (the online softmax reassociates; the existing device test tolerance still applies). Generated tokens on this model and prompt were identical.
