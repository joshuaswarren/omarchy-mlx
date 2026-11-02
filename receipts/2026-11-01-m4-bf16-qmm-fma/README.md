# bf16 scalar-FMA Q4 prefill where the cooperative route is unavailable (M4 Mac mini)

Result: where the cooperative-matrix qmm route is not reachable for a call (no fp32 8x8x8 matrix, a subgroup size other than 32, less than 4 KiB of staging shared memory, or `MLX_OMARCHY_NO_COOPMAT=1`), a multi-row bf16 4-bit group-64 transposed `quantized_matmul` now runs a bf16 build of `qmm_fma.comp` instead of the 16x16 tile kernel `QmmTileBF16`. On the M4 Mac mini with the cooperative matrix hidden, prefill rises 8.45x for granite-4.0-1b-4bit at 128 tokens (21.5 -> 181.8 tok/s), 8.15x at 512 (21.1 -> 171.9) and 4.05x for Qwen3.5-2B at 512 (15.7 -> 63.6), with the same last-position argmax; the kernel runs 709 to 842 GFLOP/s at 512 to 1500 rows against 69 to 72 for the tile kernel. The outputs are the tile kernel's bits: 21/21 `qmm_fma_bits.py` cases are bit-identical to `main`'s tile route and the doctest's three shapes report 0 differing outputs between the two arms.

## What changed

- `shaders/qmm_fma.comp`: a `-DUSE_BF16=1` build. bf16 x through the same uvec4 view (two exact shifts per word), bf16 scales and biases widened exactly, output drained with round-to-nearest-even and the tile kernel's NaN rule. The f16 builds compile to byte-identical SPIR-V.
- `CMakeLists.txt`, `compute.h`, `compute.cpp`: `QmmPrefillFmaBF16` and `QmmPrefillFmaPreciseBF16`, appended before `Count`.
- `primitives.cpp` (`QuantizedMatmul::eval_gpu`): bf16 joins the f16 FMA gate (`qmm_fma_reachable` requires `!coopmat_reachable`), the precise build from 1024 rows; the route comments describe the selection as it now is. `MLX_OMARCHY_NO_QMM_FMA_BF16=1` restores the tile route; `docs/kernel-flags.md` gains its row.
- Tests: `test_matmul_family.cpp` evaluates the production predicate on the opened device and, where the cooperative route is unavailable, checks the dispatched kernel at 5, 77 and 1030 rows and both this arm and the tile arm (`MLX_OMARCHY_QMM_TILE_RB=0`) against the host reference within the tile kernel's derived bound, reporting how many outputs differ between the arms. `test_capability_simulation.cpp` runs a bf16 case under every profile: a cooperative-matrix kernel where the profile makes the route reachable, this build where it does not (including `subgroup-size-64`, a device with the matrix extension), the tile kernel with `MLX_OMARCHY_QMM_TILE_RB=0`, each within the same bound.

## Host and stack (`host.txt`)

Apple Mac mini (M4, 2024), `apple,j773g` / `apple,t8132`, 16 GB; kernel `7.1.13-1-20-ARCH` with the `asahi_m4` G16G driver (GPU ID `0x0a021100` rev `0x11`, boot firmware `mBoot-18000.161.10`), PMP not bound; Mesa 26.3.0-devel Honeykrisp `git-ea328336ef` (techmore/mesa `m4-honeykrisp-compute`), `Apple M4 (G16G B1)`; the speed and numerics runs set `HK_APPLE9_COOPMAT=0`, which hides that driver's cooperative matrix; `AGX_APPLE9_MAX_SHARED=32768`, `MLX_OMARCHY_BATCH_WORK=120000`; shaderc `glslc` 2026.4; Python 3.14.8, mlx-lm 0.31.3.

## Builds (`wheels.txt`)

A: `main` `a69d9e6`. B: every speed number belongs to the measured code commit `95a3849` (`+95a3849`), kept on [`evidence/m4-321-speed-95a3849`](https://github.com/techmore/omarchy-mlx/tree/evidence/m4-321-speed-95a3849). The head's code commit `89b54a0` differs from it only in a test file (`git diff --stat 95a3849 89b54a0`: `overlay/tests/omarchy/test_capability_simulation.cpp | 30 +++---, 1 file changed, 20 insertions(+), 10 deletions(-)`); its suites, bit compares and kernel arms were rerun on `+89b54a0`. Fresh worktree, `scripts/prepare-mlx.sh`, new CMake tree per build; every process printed the provenance line and refused unless its stamp matched the intended commit. `scripts/ab.py` stops on the first process that exits non-zero (provenance refusal 3, timeout 124, any other status) or adds a GPU driver error: it writes that record, runs no later arm and exits with the child's status. GPU error monitoring is required: if `journalctl -k` cannot run or exits non-zero, `ab.py` refuses with status 8 before any arm runs and writes no record (after an arm, it writes that record with the monitor error and stops with 8). `scripts/test_ab.py` checks the rc 3, rc 124, absent-monitor and failing-monitor cases without a GPU. The records in this receipt were written by the earlier runner, which logged such a status but kept going; every record here has `rc` 0 and no new kernel error. The monitor was live on the host: `scripts/facts.sh`, run as the same user, read the GPU driver's kernel lines through the same `journalctl -k` (`host.txt`).

## Tests (`battery.txt`)

This change (head code commit `89b54a0`):

- `omarchy_matmul_family_tests` 43/44 with the driver's cooperative matrix (the new test skips: the cooperative route serves bf16 there), 43/44 with `HK_APPLE9_COOPMAT=0` and 41/44 with `MLX_OMARCHY_NO_COOPMAT=1` (both run the new test: the FMA and tile arms dispatch as expected, stay within the bound at 5, 77 and 1030 rows, and 0 outputs differ between them).
- The failing case in every configuration, `gather_qmm sorted-expert tile route ...`, fails the same way on `main` on this host; the two extra failures under `MLX_OMARCHY_NO_COOPMAT=1` (`coopmat prefill dispatch pins M16, 32-row and TwoN builds`, `q4g32 and q8 affine formats take the word decode and coopmat prefill routes`) are route-pin tests that read the device capability and not the flag, and fail identically on `main` with the flag.
- `omarchy_capability_sim_tests` 8/8 in each of the six profiles.

This is not an all-green battery. On this host `main` fails 34 of 762 cases and the six-change integration the same 34 of 800 (`receipts/2026-11-01-m4-g16g-part1/battery.txt` names them). `omarchy_fast_ops_tests` exits 0 with 4 failed assertions, all inside `sdpa vjp dk dv match finite differences at rep=1 (path per gate)`, which upstream marks `doctest::may_fail` (an expected failure); its passing case total does not make it a clean suite.

## Numerics (`bits-p6-vs-main.log`, `qmmfma-*.log`)

`qmm_fma_bits.py` (bf16 x, 4-bit group 64, 5 to 1500 rows, N 1000 / 2050 / 3072, K 1024 / 2048; error against an f64 reference in bf16 ulps of the case's largest output): `main` (tile route) dumps, this change compares: 21/21 bit-identical, worst error 0.722 ulp on both. On one wheel of this change, the FMA arm and `MLX_OMARCHY_NO_QMM_FMA_BF16=1` (tile arm) compare identically to the same dump. `engage-qmm_fma-kernels.txt` shows the cases dispatched `QmmPrefillFmaBF16` and, from 1024 rows, `QmmPrefillFmaPreciseBF16`. `compare` exits 1 on any mismatch (`bits-selftest.log`).

## Speed (`perf-p6-summary.txt`)

`HK_APPLE9_COOPMAT=0` on both sides; two interleaved pairs; load below 0.5 and 20 s idle before each process; SMC sensors 32 to 49 C; peak GPU memory equal.

| model | prompt tokens | A main (tile) | B this change | B/A | argmax |
|---|---:|---:|---:|---:|---|
| granite-4.0-1b-4bit | 128 | 21.5 | 181.8 | 8.45 | equal |
| granite-4.0-1b-4bit | 512 | 21.1 | 171.9 | 8.15 | equal |
| Qwen3.5-2B-MLX-4bit | 512 | 15.7 | 63.6 | 4.05 | equal |

`prefill_bench.py <repo> <n> 4`: fixed synthetic prompt, fresh cache, one warm-up then 4 timed full forwards, median. Decode does not take this route: granite-4.0-1b-4bit 38.78 -> 38.80 tok/s with identical token ids (`decode_bench.py <repo> 64 4`).

Kernel (`qmmfma-arms.jsonl`, `qmm_fma_bits.py` timing, 5 calls per case): the FMA build runs 709 to 842 GFLOP/s at 512 to 1500 rows (N 1000 to 3072, K 1024 or 2048) against 69 to 72 for the tile kernel, 243 to 565 against 60 to 69 at 64 to 130 rows, and 18 to 44 against 11 to 19 at 5 rows; both arms compare bit-identical to `main`'s tile dump.

## Device reopen

`reopen.py` passed before and after each bit-compare set and at the end of the session; no new GPU driver message in the kernel log.

## Other parts

Where the cooperative route is reachable (the M1/M2 fork driver, this M4 driver by default) nothing changes. G13 and G14: not measured here. On stock-Mesa M1/M2 this becomes the bf16 Q4 prefill route; the per-output chain is the same ascending-k fused chain as the f16 FMA build, and a `docs/numerics-gate.md` run there shows whether it matches the tile kernel's bits with that compiler as it does on the M4.

## Numerics gate

**Gate: passed** (`numerics-gate/README.md`). `docs/numerics-gate.md` against stock upstream MLX 0.32.3 on the CPU in float32 with the same weights and text: granite-4.0-1b-4bit, 5,120 teacher-forced positions, five runs per arm, `main` `a69d9e6` (the tile route) against `89b54a0`, both arms with the cooperative matrix hidden (`HK_APPLE9_COOPMAT=0`), harness check passed (S=1 PPL 6.54881 against one-pass 6.56134).

- Check 1 (per-op, fp64 forward of captured operands): no worse in 21 of 21 cells.
- Check 2 (token choices): 99.648% agreement with near ties against the baseline's Wilson lower bound of 99.445%.
- Check 3 (perplexity): deviation 0.000707 against the baseline's 0.000707 plus noise 0 (PPL 7.58016 in both arms, reference 7.58552).

The passes are 64 tokens, not S=1, because this route runs only on multi-row passes without the cooperative route; an S=1 pass would take the decode GEMV in both arms. The two arms are bit-identical, and the changed kernel did run: on profiling builds, the per-op cells dispatched `QmmTileBF16` on `main` and `QmmPrefillFmaBF16` on this change. This holds for the M4 only; a stock-Mesa M1/M2 run is still needed to show whether the route keeps the tile kernel's bits there. The gate README lists every departure from the documented protocol.

## Files

`host.txt`, `wheels.txt`, `battery.txt`, `bits-p6-vs-main.log`, `bits-selftest.log`, `qmmfma-arms.jsonl`, `qmmfma-fma.log`, `qmmfma-tile.log`, `engage-qmm_fma-kernels.txt`, `perf-p6.jsonl`, `perf-p6-summary.txt`, `reopen.log`, `numerics-gate/`, `scripts/`, `SHA256SUMS`.
