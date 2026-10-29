# Overlapped weight loads in the 4-bit qmv kernel: not merged, slower

Result: two ways of issuing the next step's loads before the current step's math in the single-row 4-bit group-64 decode kernel produce byte-identical output and are SLOWER than the shipped kernel (0.73x to 0.98x per call on a 16-inch M1 Max). Nothing is merged. The preregistered bar (`PREREG.md`: at least +8% decode tok/s on the 13-inch M1 and the 16-inch M1 Max) is not met; the prereg's own stop rule (kernel-level gain under 1.10x on the large shapes: stop and report) applies.

Idea source: TensorFold unit 8a (hoist and overlap the weight loads, accumulators `precise`).

## What was changed

`qmm_vec.comp`, macro `Q4_ROWS` (used by the single-row, multi-weight and outgate 4-bit variants): the loads of the next step (weight words, scale, bias, x words) are issued before the arithmetic of the current step, and the registers rotate at the end. The arithmetic, the quad order and the `precise` accumulators are unchanged.

- Variant 1 (`variant1.patch`, branch head `9562bc74`): the next step's loads sit inside `if (more)`.
- Variant 2 (`variant2.patch`, `75e145a5`): the next step's loads are unconditional and clamped in bounds, the rotation is unconditional (no branch around the loads).

The prereg describes a run-time switch (`MLX_OMARCHY_QMV_LOADOVERLAP`). It was not built: each variant was measured as a separate build against the published v0.7.34 wheel (same release script, same boot).

## Measurements (16-inch M1 Max, Honeykrisp, one boot)

`qmv_micro.py`: one-row 4-bit matvec, 16 dependent calls per evaluation, median per call over 9 evaluations, three interleaved rounds, medians of the rounds. A is the v0.7.34 wheel (`libmlx.so` digest `7ca9b27ebb32c378`), B is the variant.

| Shape | A (us) | B1 (us) | A/B1 | A (us) | B2 (us) | A/B2 |
| --- | --- | --- | --- | --- | --- | --- |
| up/gate K 2048, N 6144 | 66.0 | 80.9 | 0.82 | 69.8 | 72.3 | 0.97 |
| down K 6144, N 2048 | 88.2 | 106.2 | 0.83 | 90.0 | 106.7 | 0.84 |
| proj K 2048, N 2048 | 46.3 | 54.3 | 0.85 | 48.2 | 49.3 | 0.98 |
| kv K 2048, N 512 | 76.4 | 79.9 | 0.96 | 84.5 | 86.0 | 0.98 |
| head K 2048, N 248320 | 1273.6 | 1750.7 | 0.73 | 1291.7 | 1463.8 | 0.88 |

Logs: `variant1-ab.log`, `variant2-micro.log` (variant 2 ran in a later window, so its A column differs).

## Exactness (all held)

- Output fingerprint of 64 fixed-seed random activation rows per shape, the head included (about 16 million outputs), sha256 of the bf16 bytes: equal between A and B for every shape, both variants.
- Variant 1, full 256-step greedy decode of Qwen3.8-2B 4-bit: generated ids equal (`e70cb52a914936c0`) and the hash of every step's full logits equal (`c0a8b9edbfa9eafe`) between A and B.
- SPIR-V, same `glslc` (2026.3) on the v0.7.34 tree and the branch: 32 of 39 `qmm_vec.comp` modules byte-identical, 7 differ (the variants that expand `Q4_ROWS`); `spirv-identity.log`.
- Tests on a software Vulkan device with variant 1: `omarchy_primitive_tests` 107 of 107, `omarchy_qmv_batch_tests` 2 of 2; `omarchy_matmul_family_tests` and `omarchy_fast_ops_tests` fail 2 and 12 cases identically with the tag's shader (limits of that device, not this change).

## Not run

The end-to-end decode pairs (the prereg's bar) were not run: the end-to-end part of the first A/B failed on a missing helper module, and the kernel-level result made a rerun pointless. The 13-inch M1 was not measured. The 13-inch M1 profile already showed the matvec at the memory roof (60 GB/s against 58 GB/s for a plain elementwise pass), which predicts no gain there.

## Why it is slower (not proven)

The extra live registers for the next step's operands and the rotation copies can lower occupancy, and the device compiler already issues part of the loads early in the original loop. The device compiler's disassembly was not read.

## Fallback from the plan

Making techmore's `MLX_OMARCHY_QMV_SMALL_T` route (#296) default-on needed at least +5% at 2 to 4 row passes on both chips AND identical outputs. The speed part is measured (1.2x to 1.8x per call at 2 to 4 rows on the 13-inch M1, #296 thread). The identity part cannot hold: that route follows the GEMV summation order, the shipped route follows the cooperative-matrix chain, so their outputs differ in the last bf16 bits. Also, one-token decode never takes a 2 to 4 row pass, so it would not change decode tok/s. Not enabled.
