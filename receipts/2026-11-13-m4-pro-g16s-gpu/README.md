# M4 Pro (G16S) GPU on Borealis: test battery, model runs and fixes

This receipt backs the M4 Pro row in `docs/compatibility.md`, `docs/m4-linux.md` and `README.md`.

## Host (`host.txt`)

Apple M4 Pro (J616s, T6040, G16S B1), 14 CPU cores, 48 GB. Mesa 26.3.0-devel (`git-05ddb00f5c`) Borealis, Vulkan 1.4.363, no cooperative matrix, subgroup size 32, 32 KiB shared memory, 2 GiB - 1 storage binding range.

## Fixes found by the battery

All fixes are in the omarchy overlay and apply on stock Borealis.

| commit | finding | fix |
|---|---|---|
| elementwise: keep the f16 rsqrt source from being a bare widening | the AGX packer refused `rsqrt` of a bare f16 widening, so the f16 elementwise pipelines (sqrt, exp, rsqrt, normal, erfinv) did not compile: 15 failing cases | the f16 variant ORs the operand's bits with `params.dims >> 16` (always 0), value bit-identical |
| quantized gemv group: form the token-multi route for rows > 1 | `dispatch_quantized_gemv_group` compared `node.size()` with `n`, so the multi-row route never formed; the fused_chain case compared the composed route with the GEMV bit-for-bit and flaked by one ulp | `node.size() == rows * n`; the test requires the route to form on G16S |
| sdpa prefill: flash commit every 4e9 MACs, oversized score bindings to flash | 16384x8 prefill is ~550 GMAC at ~50 GMAC/s in one ~11 s submission, past the per-job GPU timeout (`Error(110)`); the 32-head L=4096 composed route bound more than 2 GiB | the flash tile loop commits every 4e9 MACs, except while a compiled tape records; shapes whose score matrix exceeds the binding range take the flash kernel |
| sdpa decode f16 arm on G16S without the cooperative matrix | the shader never used the cooperative matrix; G16S composed three dispatches | on G16S only (`sdpa_decode_f16_without_coopmat_part`), gate on subgroup 32 + ops + 1024 lanes + shared stream; other parts keep the cooperative-matrix clause. us per call composed -> fused, f16 1100 keys 586 -> 177, 4096 keys 1872 -> 308, 263 keys 254 -> 141, 37 keys 153 -> 124; bf16 unchanged |
| qmm coopmat: widen the f16 x to an f32 A matrix | with cooperative matrices enabled (development Mesa build, `ASAHI_AGX3B_COOPERATIVE_MATRIX=1`) the packer refused the f16 A x f32 B multiply in `qmm_coopmat_f16`, so Qwen2.5-1.5B prefill failed with `VK_ERROR_UNKNOWN` | x is loaded as f16 and converted to an f32 A matrix |
| tests: distributed, gdn_bf16_state, latched statics | MPI backend loaded at run time; `test_gdn_bf16_state` did not compile with GCC 16; `MLX_OMARCHY_GDN_PF` and `MLX_OMARCHY_GDN_STATE_F32` were read once per process | test detects dlopen MPI, compile fixed, both flags read live |
| tests: stream-independence floor for Aurora Borealis | two-stream overlap ratio 0.83-0.93 on Borealis (floor 0.95), every copy byte-for-byte correct | Borealis floor 0.80 (selected by driver id, other drivers keep 0.95); correctness checks unchanged |

Review follow-ups in the same series: the capability-simulation backing check of the decode SDPA covers only the f16 arm (the bf16 arm uses no subgroup operations, so `m1-stock-no-coopmat` on lavapipe is 10 of 10 again), and the flash branch refuses a simulated profile whose hardware lacks the kernel's workgroup and shared memory.

## Standing battery (`battery.txt`)

Command per unit, on the head of this branch (code at b445c037), one process per suite (`battery-425.sh`):

```bash
flock /tmp/m1-gpu.lock timeout 1800 build-tests/tests/omarchy/<suite>
flock /tmp/m1-gpu.lock timeout 1800 build-tests/tests/omarchy/omarchy_capability_sim_tests <profile>
flock /tmp/m1-gpu.lock timeout 1800 build-tests/tests/omarchy/omarchy_sdpa_prefill_flash_tests --test-case=<case, commas escaped>
```

| group | result |
|---|---|
| 57 suites (`omarchy_ane_runtime_tests` excluded: it links against the ANE runtime sources) | 57 of 57 pass |
| capability-simulation profiles (8: m1-honeykrisp-fork, m1-stock-no-coopmat, subgroup-size-64, small-shared-memory, no-cooperative-matrix, workgroup-256, workgroup-128, m1-g13-legacy) | 8 of 8 pass, 10 of 10 cases each |
| `omarchy_sdpa_prefill_flash_tests`, one process per test case (16 cases, including 16384x8 and the tape-recording case) | 16 of 16 pass |

The earlier run of this receipt listed flash-15 as 0 passed: doctest splits `--test-case` filters at commas, and that case name has one. The runner now escapes commas; every case runs once.

Cases that skip name a missing capability (no cooperative matrix on G16S, G16G-only routes) in their log.

The capability-simulation profiles also pass 8 of 8 (10 of 10 cases each) on lavapipe (Mesa 26.3 `llvmpipe`, `MLX_OMARCHY_ALLOW_NON_APPLE=1`).

## Mesa and the opt-in routes (`battery-dual-mesa.md`, `fused-chain-token-stock-mesa.txt`)

On stock Borealis (`git-05ddb00f5c`) the backend builds and runs the opt-in routes the suites reach: the battery with `MLX_OMARCHY_QMM_VEC_TOKEN_MULTI=1`, `MLX_OMARCHY_GDN_GQA_BF16A=1`, `MLX_OMARCHY_GDN_PF_GQA=1` and `MLX_OMARCHY_GDN_PF_WIDE=1` logs no pipeline refusal. The fused_chain token-multi case forms the group and dispatches `qmm_vec_q4_multi_token4_subgroup_bf16` and `qmm_vec_q4_multi_token8_subgroup_bf16` on stock Mesa (dispatch trace and raw log in `fused-chain-token-stock-mesa.txt`).

`shaders-refused-stock-borealis.txt` comes from a standalone pipeline-creation sweep (one compute pipeline per SPIR-V file, on a device that enables 8-bit storage and int8 and uses a fixed 8-binding layout). That sweep stops 22 shaders with `Unhandled ALU op pack_64_4x16`; the backend's device and layouts do not hit that path for the shaders the suites dispatch. The earlier text of this receipt said those shaders could not run on G16S without Mesa changes; that does not hold for the shaders the suites dispatch through the backend. Shaders on that list that no suite dispatches are not checked through the backend.

The f16 elementwise refusal (first fix above) was a backend failure on stock Mesa and is fixed in the overlay.

## Models (`models.txt`)

On f4077931 (wheel `0.32.4.dev202611132133+f407793`), warm prefill / decode tok/s / peak GB: Qwen2.5-1.5B-Instruct-4bit 651.5 / 137.0 / 0.98, granite-4.0-1b-4bit 538.5 / 105.5 / 0.99, Qwen3.5-4B-MLX-4bit 189.5 / 55.0 / 2.69; all three produce coherent text. Decode varies by about 10 percent between runs. The later commits change no default route on G16S, and on b445c037 all three models produce the same text. With cooperative matrices enabled on a development Mesa build, the three models run and produce the same text (Qwen2.5-1.5B failed there before the qmm coopmat fix).

## Defaults

`decode-ab.txt` runs three interleaved rounds of the default, `MLX_OMARCHY_FUSED_GEMV=0` and `MLX_OMARCHY_FUSED_TRIO=0` on one build. The spread inside one arm is larger than the difference between arms and no arm wins on every model, so G16S keeps the existing defaults (fused GEMV on, default cooperative-matrix pick).

## f16 rounding (`rne-ab.txt`)

Shaders that used `packHalf2x16` as an f32 to f16 rounding step now use an integer round-to-nearest-even conversion (`shaders/omarchy_half.h`): GLSL leaves the rounding of `packHalf2x16` to the implementation, and a development Mesa build lowers it with round-toward-zero. The helper matches the hardware conversion over all 2^32 float32 patterns. On G16S with stock Mesa (where `packHalf2x16` rounds to nearest even) the wheel with the helper and the same tree with the seven shaders reverted to `packHalf2x16` give bit-identical outputs for f16 cast, quantized GEMV and GEMM (1, 8 and 512 rows), SiLU-gate, RMS norm and decode SDPA (1100 and 4096 keys), and the times per call agree within the run-to-run spread (two interleaved runs per arm). M1/M2 are not measured here.
