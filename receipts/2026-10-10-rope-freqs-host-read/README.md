# Rope gate frequency read on the host (gemma-4 E2B / E4B board stall)

Source commit under test for the M1 Max suites: `e34d1da94230c85a6a0207476877f1460793e204`. The code files of the PR head are byte-identical to it
(the commits above it change receipts only). The M1 wheel below was built from `158fee60c4cf5599acba97bd61b2274976798699`, which has the same code.

## Defect

The gemma-4 E2B and E4B decode graphs stalled at a join named `rope_freqs_bound`. The rope trig gate (`rope_trig_gate` in
`overlay/mlx/backend/omarchy/primitives.cpp`) read the `freqs` array through a nested evaluation. `freqs` is a leaf (status
evaluated, no primitive) that still carried a pending async-eval latch, so the nested evaluation waited on a latch that nothing
would release. The fix reads `freqs` on the host instead.

## Change

The host read is the default. `MLX_OMARCHY_NO_ROPE_FREQS_HOST=1` restores the nested gate. The doctest
`rope freqs host read agrees with the nested gate on both legs` compares the two bit for bit.

## Evidence

### M1 (G13G), PR-head wheel, no flag (`h408-g13g/`)

Wheel `0.32.4.dev202610101808+158fee60` (sha256 prefix `b09851e0bbbb9241`), mlx-lm 0.32.0, Honeykrisp Vulkan, one attempt per arm.
`h408.log` is the run log with the home path replaced by `$HOME` and the download progress bars removed, so `SHA256SUMS` lists the
redacted file, not the original.

| Model | Arm | STALL lines | Result |
|---|---|---|---|
| gemma-4-E2B 4-bit | `MLX_OMARCHY_NO_ROPE_FREQS_HOST=1`, traced | 3 | timeline error (observed 18, target 19); `rope_freqs_bound` joins 12 |
| gemma-4-E2B 4-bit | default, traced | 0 | completes, digest `d9315ec0`, `rope_freqs_host` joins 6660 |
| gemma-4-E2B 4-bit | default, untraced | 0 | decode 13.265 tok/s, prefill 512 195.492 tok/s, digest `d9315ec0` |
| gemma-4-E4B 4-bit | `MLX_OMARCHY_NO_ROPE_FREQS_HOST=1`, traced | 3 | timeline error (observed 23, target 24); `rope_freqs_bound` joins 16 |
| gemma-4-E4B 4-bit | default, traced | 0 | completes, digest `118c7586`, `rope_freqs_host` joins 7498 |
| gemma-4-E4B 4-bit | default, untraced | 0 | decode 10.472 tok/s, prefill 512 113.591 tok/s, digest `118c7586` |

End-of-arm load1 for the four arms that finished: 0.30 to 0.48. The kill-switch arm never finishes, so there is no flag-off digest on this
hardware. The two default digests equal the ones measured earlier on the experiment wheel with `MLX_OMARCHY_ROPE_FREQS_HOST=1`.

### Earlier experiment wheel (`0.32.4.dev202610100859+bd0d2630`, flag `MLX_OMARCHY_ROPE_FREQS_HOST=1`)

| Chip | Model | Flag off | Flag on |
|---|---|---|---|
| M1 (G13G) | E2B, async decode | 2 of 2 runs STALL | OK, 0.58 s |
| M2 Max (G14C) | E2B | 2 of 2 runs STALL | OK, 0.46 s |
| M1 (G13G) | E4B score bench | 3 STALL (observed 23, target 24) | 0 STALL, whole bench finished, `rope_freqs_host` x7498 |

The M2 venv copy printed an older dist-info version string than the wheel under test; the loaded library was the experiment build.

### M1 Max (G13C), PR code, build and test (`g13c/`)

Linux kernel 7.1.12-2. The run exports a private Honeykrisp build through `VK_ICD_FILENAMES` (Mesa git `6543eeb7df7`, driver library sha256
prefix `3546bcafe8ed3995`, the same library as the M1 Max column of `receipts/2026-10-08-causal-flash-prefill`). The test log does not print the
device name. In the two suite logs the build-tree path prefix is replaced by `<build>`.

| Run | Result |
|---|---|
| Build + new doctest | 1 case, 3 assertions, 0 failed (`g13c/rope3-ticket-build.log`) |
| Whole `omarchy_fast_ops_tests`, default | 48 cases, 1,383,973 assertions, 8 failed (`g13c/rope3-ticket-default.log`) |
| Whole `omarchy_fast_ops_tests`, `MLX_OMARCHY_NO_ROPE_FREQS_HOST=1` | 48 cases, 1,383,973 assertions, 8 failed (`g13c/rope3-ticket-kill.log`) |

The 8 failed assertions are the same in both runs and match the freeze battery: 4 in the `may_fail` SDPA VJP case (`test_fast_ops.cpp:80`) and 4 in the
`may_fail` fused GDN VJP cases (`test_fast_ops.cpp:4066`). Both are documented in `docs/known-defects.md`. Both suite runs used one binary
(`5c4ae47c1c9d047c`). The build and both suite runs used the same exported ICD.

## Not covered

- The kill-switch arm never finishes, so equality of the host read with the nested gate rests on the doctest and the M1 Max suite, not on a flag-off digest.
- The E4B digest on Linux (`118c7586`) differs from the macOS digest (`491eb46b`). This change does not claim digest parity; it is a separate open question.
- Throughput. The untraced decode and prefill figures above are single-protocol runs, not a before/after comparison.
- Any model other than E2B and E4B.
