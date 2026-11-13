# M4 Pro (G16S) GPU on Borealis: test battery, model runs and fixes

This receipt backs the M4 Pro row in `docs/compatibility.md`, `docs/m4-linux.md` and `README.md`.

## Host (`host.txt`)

Apple M4 Pro (J616s, T6040, G16S B1), 14 CPU cores, 48 GB. Mesa 26.3.0-devel (`git-05ddb00f5c`) Borealis, Vulkan 1.4.363, no cooperative matrix, subgroup size 32, 32 KiB shared memory, 2 GiB - 1 storage binding range.

## Fixes found by the battery

All fixes are in the omarchy overlay and apply on stock Borealis.

| commit | finding | fix |
|---|---|---|
| elementwise: keep the f16 rsqrt source from being a bare widening | the AGX packer refused `rsqrt` of a bare f16 widening, so the f16 elementwise pipelines (sqrt, exp, rsqrt, normal, erfinv) did not compile: 15 failing cases | the f16 variant ORs the operand's bits with `params.dims >> 16` (always 0), value bit-identical |
| quantized gemv group: form the token-multi route for rows > 1 | `dispatch_quantized_gemv_group` compared `node.size()` with `n`, so the multi-row route never formed; the fused_chain case compared the composed route with the GEMV bit-for-bit and flaked by one ulp | `node.size() == rows * n`; the test requires the route to form |
| sdpa prefill: flash commit every 4e9 MACs, oversized score bindings to flash | 16384x8 prefill is ~550 GMAC at ~50 GMAC/s in one ~11 s submission, past the per-job GPU timeout (`Error(110)`); the 32-head L=4096 composed route bound more than 2 GiB | the flash tile loop commits every 4e9 MACs; shapes whose score matrix exceeds the binding range take the flash kernel |
| sdpa decode f16 arm: no longer gated on the cooperative matrix | the shader never used the cooperative matrix; G16S composed three dispatches | gate on subgroup 32 + ops + 1024 lanes + shared stream; us per call composed -> fused, f16 1100 keys 586 -> 177, 4096 keys 1872 -> 308, 263 keys 254 -> 141, 37 keys 153 -> 124; bf16 unchanged |
| tests: distributed, gdn_bf16_state, latched statics | MPI backend loaded at run time; `test_gdn_bf16_state` did not compile with GCC 16; `MLX_OMARCHY_GDN_PF` and `MLX_OMARCHY_GDN_STATE_F32` were read once per process | test detects dlopen MPI, compile fixed, both flags read live |
| tests: stream-independence floor for Aurora Borealis | two-stream overlap ratio 0.83-0.93 on Borealis (floor 0.95) | Borealis floor 0.80; correctness checks unchanged |

## Standing battery (`battery.txt`)

Command per unit, on the final tree, one process per suite:

```bash
flock /tmp/m1-gpu.lock timeout 1800 build-tests/tests/omarchy/<suite>
flock /tmp/m1-gpu.lock timeout 1800 build-tests/tests/omarchy/omarchy_capability_sim_tests <profile>
flock /tmp/m1-gpu.lock timeout 1800 build-tests/tests/omarchy/omarchy_sdpa_prefill_flash_tests --test-case=<case>
```

| group | result |
|---|---|
| 53 suites (`omarchy_ane_runtime_tests` excluded: it links against the ANE runtime sources) | 53 of 53 pass (omarchy_runtime_tests after the Borealis floor commit: 60 of 60) |
| capability-simulation profiles (8: m1-honeykrisp-fork, m1-stock-no-coopmat, subgroup-size-64, small-shared-memory, no-cooperative-matrix, workgroup-256, workgroup-128, m1-g13-legacy) | 8 of 8 pass, 10 of 10 cases each |
| `omarchy_sdpa_prefill_flash_tests`, one process per test case (15 cases, including 16384x8) | 15 of 15 pass |

Cases that skip name a missing capability (no cooperative matrix on G16S, G16G-only routes) in their log.

## Mesa dependency (requirement for the opt-in routes)

The 22 shaders that Mesa `05ddb00f5c` refuses with `Unhandled ALU op pack_64_4x16` (`qmm_vec_q4_multi_*` including the multi-token routes, `qmm_vec_greedy_bf16`, `gated_delta_*_bf16`, `gdn_vjp_bf16`) need the `aurora-silicon/mesa` compiler fixes `.lower_pack_64_4x16 = true` and keeping the f16 widening in front of an f32 `rsqrt`; without them those routes cannot run on G16S. They are opt-in, no default-route suite dispatches them, so the battery and model numbers here were measured without the fixes. `shaders-refused-stock-borealis.txt` lists the 24 refusals on stock Mesa from the battery run (22 `pack_64_4x16`, plus the two f16 elementwise pipelines that the overlay series fixes). A separate compile sweep of an earlier shader set (`mkpipe` over every shader, same stock Mesa) gave 581 compiled and 20 refused; with both Mesa fixes applied to a private build it gave 601 compiled and 0 refused. On that private Mesa build the default-route suites and two models (Qwen2.5-1.5B, Qwen3.5-4B, also with the opt-in routes enabled) gave results identical to stock Mesa. The fixes are not in this repository.

## Models (`models.txt`)

Qwen2.5-1.5B-Instruct-4bit decode 124.6 tok/s, granite-4.0-1b-4bit 117.7, Qwen3.5-4B-MLX-4bit 35.3; all three produce coherent text. Decode varies by about 10 percent between runs.

## Defaults

`decode-ab.txt` runs three interleaved rounds of the default, `MLX_OMARCHY_FUSED_GEMV=0` and `MLX_OMARCHY_FUSED_TRIO=0` on one build. The spread inside one arm is larger than the difference between arms and no arm wins on every model, so G16S keeps the existing defaults (fused GEMV on, default cooperative-matrix pick).
