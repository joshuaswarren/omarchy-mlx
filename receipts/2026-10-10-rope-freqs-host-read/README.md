# Rope gate frequency read on the host (gemma-4 E2B / E4B board stall)

Source commit under test: `e34d1da94230c85a6a0207476877f1460793e204` (branch head before the rebase onto main; the rebased
commit carries the same diff).

## Defect

The gemma-4 E2B and E4B decode graphs stalled at a join named `rope_freqs_bound`. The rope trig gate (`rope_trig_gate` in
`overlay/mlx/backend/omarchy/primitives.cpp`) read the `freqs` array through a nested evaluation. `freqs` is a leaf (status
evaluated, no primitive) that still carried a pending async-eval latch, so the nested evaluation waited on a latch that nothing
would release. The fix reads `freqs` on the host instead.

## Change

The host read is the default. `MLX_OMARCHY_NO_ROPE_FREQS_HOST=1` restores the nested gate. The doctest
`rope freqs host read agrees with the nested gate on both legs` compares the two bit for bit.

## Evidence

Measured on the experiment wheel (`0.32.4.dev202610100859+bd0d2630`, flag `MLX_OMARCHY_ROPE_FREQS_HOST=1`). This is the same code path
as the default here; the diff to the PR head is the default flip, the kill switch and the doctest.

| Chip | Model | Flag off | Flag on |
|---|---|---|---|
| M1 (G13G) | E2B, async decode | 2 of 2 runs STALL | OK, 0.58 s |
| M2 Max (G14C) | E2B | 2 of 2 runs STALL | OK, 0.46 s |
| M1 (G13G) | E4B score bench | 3 STALL (observed 23, target 24) | 0 STALL, whole bench finished, `rope_freqs_host` x7498 |

The M2 venv copy printed an older dist-info version string than the wheel under test; the loaded library was the experiment build.

At the PR head, M1 Max (G13C), build and test (Linux kernel 7.1.12-2, packaged ICD `omarchy-mlx-vulkan` 0.7.28-2, driver library sha256 prefix
`ac837b1c`; the test log does not print the device name):

| Run | Result |
|---|---|
| Build + new doctest | 1 case, 3 assertions, 0 failed (`jw16-g13c/rope3-ticket-build.log`) |
| Whole `omarchy_fast_ops_tests`, default | 48 cases, 1,383,973 assertions, 8 failed (`jw16-g13c/rope3-ticket-default.log`) |
| Whole `omarchy_fast_ops_tests`, `MLX_OMARCHY_NO_ROPE_FREQS_HOST=1` | 48 cases, 1,383,973 assertions, 8 failed (`jw16-g13c/rope3-ticket-kill.log`) |

The 8 failed assertions are the same in both runs and match the freeze battery: 4 in the `may_fail` SDPA VJP case (`test_fast_ops.cpp:80`) and 4 in the
`may_fail` fused GDN VJP cases (`test_fast_ops.cpp:4066`). Both are documented in `docs/known-defects.md`. Both suite runs used one binary
(`5c4ae47c1c9d047c`). The build ran on the earlier ICD; the suites ran after the ICD update.

## Not covered

- E2B and E4B decode rows on a wheel built from the PR head, with the host read as the default. The rows above use the experiment flag.
- The E4B digest on Linux (`118c7586`) differs from the macOS digest (`491eb46b`). This change does not claim digest parity; it is a separate open question.
- Throughput. No speed number was taken.
- Any model other than E2B and E4B.
