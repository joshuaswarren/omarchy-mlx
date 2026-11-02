# bf16 linear layers keep the scalar-FMA kernel on G16G (M4 Mac mini)

Result: on a G16G device whose driver exposes the cooperative matrix, a bf16 matmul in the linear-layer orientation (row-major lhs, transposed rhs, alpha 1) now runs `MatmulBf16Fma` instead of `MatmulBF16Coopmat`. On the M4 Mac mini that kernel is 1.1x to 1.6x faster at Llama 3.2 1B shapes (844 to 1200 against 602 to 735 GFLOP/s), and Llama-3.2-1B-Instruct-bf16 prefill rises 1.44x at 128 prompt tokens (247.2 -> 355.1 tok/s), 1.54x at 512 (268.1 -> 413.8) and 1.50x at 2048 (253.9 -> 379.9), with the same last-position argmax. Every other part keeps today's pick.

## What changed

- `device.h`: `omarchy::matmul_bf16_fma_over_coopmat_part(device_name)`, pure, true only for `G16G`.
- `primitives.cpp` (`dispatch_matmul`): when both the FMA and the cooperative-matrix bf16 routes qualify, the FMA route wins on those parts. The decision is made per call from the executing encoder's device (no process-wide cache). Attention-shaped matmuls (alpha other than 1) and other orientations keep the cooperative matrix; `MLX_OMARCHY_NO_MATMUL_FMA=1` still turns the FMA route off.
- Tests (`test_matmul_family.cpp`): the part rule in both orders; on a G16G device with the cooperative matrix, the dispatched kernel and values of the linear orientation (`MatmulBf16Fma`) and the plain orientation (`MatmulBF16Coopmat`), with small-integer operands against the exact host reference and fractional bf16-grid operands against the dense test's existing anchor bound.

## Host and stack (`host.txt`)

Apple Mac mini (M4, 2024), `apple,j773g` / `apple,t8132`, 16 GB; kernel `7.1.13-1-20-ARCH` with the `asahi_m4` G16G driver (GPU ID `0x0a021100` rev `0x11`, boot firmware `mBoot-18000.161.10`), PMP not bound; Mesa 26.3.0-devel Honeykrisp `git-ea328336ef` (techmore/mesa `m4-honeykrisp-compute`), `Apple M4 (G16G B1)`, `VK_KHR_cooperative_matrix` exposed; `AGX_APPLE9_MAX_SHARED=32768`, `MLX_OMARCHY_BATCH_WORK=120000`; shaderc `glslc` 2026.4; Python 3.14.8, mlx-lm 0.31.3.

## Builds (`wheels.txt`)

A: `main` `a69d9e6` (`+a69d9e6`); B: this change, code commit `91d9f8a` (`+91d9f8a`). Fresh worktree, `scripts/prepare-mlx.sh`, new CMake tree per build. Every process printed the provenance line and refused unless its stamp matched the intended commit; `ab.py` refuses an A/B with equal stamps and equal environments. `scripts/ab.py` stops on the first process that exits non-zero (provenance refusal 3, timeout 124, any other status) or adds a GPU driver error: it writes that record, runs no later arm and exits with the child's status. GPU error monitoring is required: if `journalctl -k` cannot run or exits non-zero, `ab.py` refuses with status 8 before any arm runs and writes no record (after an arm, it writes that record with the monitor error and stops with 8). `scripts/test_ab.py` checks the rc 3, rc 124, absent-monitor and failing-monitor cases without a GPU. The records in this receipt were written by the earlier runner, which logged such a status but kept going; every record here has `rc` 0 and no new kernel error. The monitor was live on the host: `scripts/facts.sh`, run as the same user, read the GPU driver's kernel lines through the same `journalctl -k` (`host.txt`).

## Tests (`battery.txt`)

This change: `omarchy_matmul_family_tests` 44/45. The one failing case (`gather_qmm sorted-expert tile route matches the f32 reference and the per-row route`) fails the same way on `main` on this host; it is not a dense matmul case. The integration battery fails exactly the 34 cases `main` fails. This is not an all-green battery. On this host `main` fails 34 of 762 cases and the six-change integration the same 34 of 800 (`receipts/2026-11-01-m4-g16g-part1/battery.txt` names them). `omarchy_fast_ops_tests` exits 0 with 4 failed assertions, all inside `sdpa vjp dk dv match finite differences at rep=1 (path per gate)`, which upstream marks `doctest::may_fail` (an expected failure); its passing case total does not make it a clean suite.

## Exactness (`bits-p2-vs-main.log`)

The decode GEMV (120/120), RoPE (72/72), decode SDPA (35/35) and bf16 Q4 prefill (21/21) probes are bit-identical to `main`: none of them reaches this route. Prefill keeps the same last-position argmax on all three prompt lengths.

## Speed

Prefill (`perf-p2-summary.txt`, raw `perf-p2.jsonl`): `prefill_bench.py <repo> <n> 4` (fixed synthetic prompt, fresh cache, one warm-up then 4 timed full forwards, median), two interleaved pairs (A B, B A); load below 0.5 and 20 s idle before each process; SMC sensors 32 to 48 C; peak GPU memory equal on both sides (2.35 / 2.46 / 3.40 GiB).

| prompt tokens | A main | B this change | B/A | argmax |
|---:|---:|---:|---:|---|
| 128 | 247.2 | 355.1 | 1.436 | equal |
| 512 | 268.1 | 413.8 | 1.543 | equal |
| 2048 | 253.9 | 379.9 | 1.496 | equal |

Kernel arms on one wheel (`perf-p2-linear.jsonl`, `bf16_linear.py`, this change: default against `MLX_OMARCHY_NO_MATMUL_FMA=1`, same cooperative-matrix driver; two interleaved pairs; GFLOP/s, median of 5; error = max |y - y_ref| / (|y_ref| + 1) against an f32 CPU-stream product):

| m | arm | 2048x2048 | 512x2048 | 8192x2048 | 2048x8192 |
|---:|---|---:|---:|---:|---:|
| 128 | FMA (default) | 842 | 464 | 1122 | 847 |
| 128 | cooperative matrix | 603 | 417 | 696 | 627 |
| 512 | FMA | 1116 | 825 | 1179 | 1130 |
| 512 | cooperative matrix | 695 | 598 | 724 | 710 |
| 2048 | FMA | 1180 | 1115 | 1200 | 1149 |
| 2048 | cooperative matrix | 725 | 695 | 732 | 735 |

Both arms: error 3.8e-3 to 3.9e-3 in every cell (the bf16 output rounding).

## Dispatch trace (`linear-trace-*.txt`)

`bf16_linear.py` on the profiling build of the six-change integration (which carries this change; the trace identifies each arm's kernel, it does not attribute timing to this PR alone): the default arm dispatched `MatmulBf16Fma` for every product (572 dispatches), the `MLX_OMARCHY_NO_MATMUL_FMA=1` arm `MatmulBF16Coopmat` (580); no other matmul kernel ran.

## Device reopen

`reopen.py` passed before and after the bit-compare set and at the end of the session; no new GPU driver message in the kernel log.

## Other parts

The rule names G16G only; every other part keeps the cooperative-matrix pick (receipts/2026-09-12-prefill-fma-qualify). The route matters only on a G16G driver with the cooperative matrix; without it the FMA route was already the pick. If that driver's cooperative-matrix path gets faster, the `bf16_linear.py` arms re-measure the pick.

## Numerics gate

`numerics-gate/README.md`: docs/numerics-gate.md for this change against a stock upstream MLX CPU fp32 reference (teacher-forced agreement, S=1 or route-shaped PPL, per-op error, harness check). Its first line states whether the gate has been evaluated; until it says complete, `numerics-gate/RESUME.md` lists what remains.

## Files

`host.txt`, `wheels.txt`, `battery.txt`, `bits-p2-vs-main.log`, `perf-p2.jsonl`, `perf-p2-summary.txt`, `perf-p2-linear.jsonl`, `linear-trace-*.txt`, `reopen.log`, `scripts/`, `SHA256SUMS`.
