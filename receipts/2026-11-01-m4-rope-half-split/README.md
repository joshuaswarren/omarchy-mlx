# Half-split RoPE built separately (M4 Mac mini)

Result: the forward RoPE kernels `FastRope{,Freqs}{BF16,F16}` carry the half-split, traditional and passthrough forms behind runtime flags. Four more builds of the same source with `-DROPE_HALF_SPLIT_ONLY` (`FastRope{,Freqs}HalfSplit{BF16,F16}`) fix the traditional and passthrough flags false, so those branches fold away (SPIR-V: bf16 28.4 -> 15.7 KB, f16 11.0 -> 6.1 KB). `RoPE::eval_gpu` routes the half-split form with bf16 or f16 output to them. The output is bit-identical (72/72 `rope_bits.py` cases against `main`). On the M4 Mac mini a bf16 decode RoPE dispatch drops from 75.9 to 53.6 microseconds in the trace, a bf16 q+k RoPE pair from 136.2 to 103.3 microseconds of wall time, and granite-4.0-1b-4bit decodes 6.5 percent faster (38.84 -> 41.35 tok/s) with identical token ids. f16 models whose RoPE runs inside the fused trio on `main` (Qwen2.5-0.5B, gemma-3-1b) are unchanged by this change alone; on one integration wheel (`749902d`), `MLX_OMARCHY_NO_ROPE_HALF_SPLIT=1` against unset, Qwen2.5-0.5B-Instruct-4bit 116.13 -> 121.22 tok/s (1.044) and gemma-3-1b-it-4bit 52.96 -> 54.06 (1.021), identical token ids. `MLX_OMARCHY_NO_ROPE_HALF_SPLIT=1` keeps the general builds.

## What changed

- `shaders/fast_rope.comp`: under `ROPE_HALF_SPLIT_ONLY`, `traditional = false; passthrough = false;` after the flags are read, in both mains.
- `CMakeLists.txt`, `compute.h`, `compute.cpp`: four builds and four kernel ids, appended before `Count` so profile ids stay stable.
- `primitives.cpp` (`RoPE::eval_gpu`, non-fused path): half-split when `(flags & 0xa) == 0`; bf16 direct output and f16 output take the new builds. The fused RoPE+norm kernels and f32 are unchanged.
- `docs/kernel-flags.md`: the `MLX_OMARCHY_NO_ROPE_HALF_SPLIT` row.
- Test (`test_fast_ops.cpp`, `fused rope half-split form runs the half-split builds`): for bf16 and f16, base and `freqs`, traditional or not, full and partial rotation, the dispatched kernel id is the half-split build exactly when the form is half-split and the general build otherwise. The values are held by the existing bit-for-bit and f64 RoPE cases, which go through the same dispatch.

## Why: short kernels pay for their code size on G16G

Two synthetic kernels with the same integer work, one unrolled and one looped, dispatched 300 times in one command buffer (`gen_icache.py` and `vkchain.c`, in `receipts/2026-11-01-m4-g16g-part1/bandwidth/`, rerun there): at 4 workgroups of 256 the unrolled kernels (SPIR-V 24, 93 and 369 KB) take 10.1-10.9, 32-35 and 126-128 microseconds per dispatch against 7.3, 11.9 and 39.5 for the looped ones (1.6 KB) doing the same work; at 160 workgroups the unrolled kernels are as fast as or faster than the looped ones (27-29 vs 29.5, 93 vs 103, 373 vs 395). The cost that grows with code size appears only when few workgroups run, which is the decode RoPE's case (one to four workgroups per dispatch).

## Host and stack (`host.txt`)

Apple Mac mini (M4, 2024), `apple,j773g` / `apple,t8132`, 16 GB; kernel `7.1.13-1-20-ARCH` with the `asahi_m4` G16G driver (GPU ID `0x0a021100` rev `0x11`, boot firmware `mBoot-18000.161.10`), PMP not bound; Mesa 26.3.0-devel Honeykrisp `git-ea328336ef` (techmore/mesa `m4-honeykrisp-compute`), `Apple M4 (G16G B1)`; `AGX_APPLE9_MAX_SHARED=32768`, `MLX_OMARCHY_BATCH_WORK=120000`; shaderc `glslc` 2026.4 (all ten `fast_rope.comp` targets compiled in the wheel build); Python 3.14.8, mlx-lm 0.31.3.

## Builds (`wheels.txt`)

A: `main` `a69d9e6` (`+a69d9e6`); B: this change, code commit `5a6565a` (`+5a6565a`); profiling builds of `main` and of the six-change integration (`749902d`). Fresh worktree, `scripts/prepare-mlx.sh`, new CMake tree per build; every process printed the provenance line and refused unless its stamp matched the intended commit. `scripts/ab.py` stops on the first process that exits non-zero (provenance refusal 3, timeout 124, any other status) or adds a GPU driver error: it writes that record, runs no later arm and exits with the child's status. GPU error monitoring is required: if `journalctl -k` cannot run or exits non-zero, `ab.py` refuses with status 8 before any arm runs and writes no record (after an arm, it writes that record with the monitor error and stops with 8). `scripts/test_ab.py` checks the rc 3, rc 124, absent-monitor and failing-monitor cases without a GPU. The records here were written by the earlier runner, which logged such a status but kept going; every record here has `rc` 0 and no new kernel error. The monitor was live on the host: `scripts/facts.sh`, run as the same user, read the GPU driver's kernel lines through the same `journalctl -k` (`host.txt`).

## Tests (`battery.txt`)

`omarchy_fast_ops_tests` 50/50 test cases at code commit `b3fc834`, kept on [`evidence/m4-319-tests-b3fc834`](https://github.com/techmore/omarchy-mlx/tree/evidence/m4-319-tests-b3fc834) (the head `5a6565a` changes only the docs row: `git diff --stat b3fc834 5a6565a`: `docs/kernel-flags.md | 2 +-`) and in the integration battery; 49/49 on `main` (the new case is the 50th). The `sdpa vjp dk dv match finite differences` case records 4 failed assertions on both builds; upstream marks it `doctest::may_fail`, so the 50/50 count includes it as an expected failure. It is not a RoPE case.

This is not an all-green battery. On this host `main` fails 34 of 762 cases and the six-change integration the same 34 of 800 (`receipts/2026-11-01-m4-g16g-part1/battery.txt` names them). `omarchy_fast_ops_tests` exits 0 with 4 failed assertions, all inside `sdpa vjp dk dv match finite differences at rep=1 (path per gate)`, which upstream marks `doctest::may_fail` (an expected failure); its passing case total does not make it a clean suite.

## Exactness (`bits-p4-vs-main.log`)

`rope_bits.py`: bf16 and f16, base and `freqs`, traditional and not, decode (T = 1 with 16 and 4 heads) and 17-token shapes, offsets 0, 37 and 4093: 72/72 bit-identical; `engage-rope-kernels.txt` (a profiling run of the same cases on the integration) lists the builds they dispatched: `FastRope{,Freqs}HalfSplit{BF16,F16}` for the half-split cases, `FastRope{,Freqs}{BF16,F16}` for the traditional ones. `compare` exits 1 on any mismatch before it times anything (`bits-selftest.log`: 0 against a matching baseline, 1 against one with a flipped value). The GEMV (120/120), decode SDPA (35/35) and bf16 Q4 prefill (21/21) compares in the same log are identical too.

## Speed

Decode (`perf-p4-summary.txt`): `decode_bench.py <repo> 64 4`, two interleaved pairs (A B, B A), median of process medians; load below 0.5 and 20 s idle before each process; SMC sensors 33 to 44 C; peak GPU memory equal.

| model | A main | B this change | B/A | token ids |
|---|---:|---:|---:|---|
| granite-4.0-1b-4bit (bf16) | 38.84 | 41.35 | 1.065 | identical |
| Qwen2.5-0.5B-Instruct-4bit (f16, RoPE in the trio on `main`) | 83.64 | 83.65 | 1.000 | identical |
| gemma-3-1b-it-4bit | 42.90 | 42.89 | 1.000 | identical |

With the trio off (the integration, which carries #316), the f16 models' standalone RoPE runs the half-split build; `perf-p4b-summary.txt` is that A/B on one integration wheel, `MLX_OMARCHY_NO_ROPE_HALF_SPLIT=1` (A) against unset (B): on one integration wheel (`749902d`), `MLX_OMARCHY_NO_ROPE_HALF_SPLIT=1` against unset, Qwen2.5-0.5B-Instruct-4bit 116.13 -> 121.22 tok/s (1.044) and gemma-3-1b-it-4bit 52.96 -> 54.06 (1.021), identical token ids.

RoPE alone (`perf-p4-rope.jsonl`, `perf-p4-rope-summary.txt`; `rope_bits.py` timing: host wall time per decode q (16 heads) + k (4 heads) pair, 200 pairs in one eval, two interleaved pairs, mean of the two processes): bf16 base 136.2 -> 103.3 us, bf16 `freqs` 1397.6 -> 1334.2, f16 base 32.2 -> 33.9, f16 `freqs` 1342.4 -> 1197.4. On f16 base, one B process read 35.2 and the others 32.1 to 32.5, so these two pairs show no gain, and with two runs per arm a regression is not ruled out. Six more pairs on the merged head settle it (Follow-up, below): no measurable change on f16 base.

## Dispatch trace (`trace-*.txt`)

The traces are profiling builds of `main` and of the six-change integration; the RoPE rows compare the same build with and without `MLX_OMARCHY_NO_ROPE_HALF_SPLIT=1`, but they are not an isolated whole-model attribution for this PR.

Profiling build of the six-change integration, granite-4.0-1b-4bit decode, per step: 80 RoPE dispatches, `FastRopeHalfSplitBF16` at 53.6 microseconds each, against `FastRopeBF16` at 75.9 with `MLX_OMARCHY_NO_ROPE_HALF_SPLIT=1` (the same 771 dispatches per step otherwise). With the fused decode routes on, `main` runs `FastRopeBF16` at 72.2 and the integration `FastRopeHalfSplitBF16` at 49.4. On Qwen2.5-0.5B with the trio off, the 48 RoPE dispatches per step are `FastRopeHalfSplitF16`. Every dispatch is a Vulkan dispatch.

## Device reopen

`reopen.py` passed before and after each bit-compare set and at the end of the session; no new GPU driver message in the kernel log.

## Other parts

Bit-identical by construction everywhere. The four modules add about 44 KB of SPIR-V to the wheel. G13 and G14: not measured here. Since `1b1b71b1` the half-split builds are the default on G16G only (`omarchy::rope_half_split_default_part`), so G13 and G14 keep the general builds unless `MLX_OMARCHY_NO_ROPE_HALF_SPLIT=0` is set. To measure them: `rope_bits.py dump`/`compare` across the two wheels, and a decode A/B on one wheel with `MLX_OMARCHY_NO_ROPE_HALF_SPLIT=0` against unset.

## Follow-up (2026-11-03)

After #319 merged, three runs on the same host (`followup-2026-11-03/`, `host.txt` there). Every probe process passed its wheel-stamp check, and no run added a GPU driver error. The `main` arm is the wheel of `2e5d7ea4`, whose tree equals `main` `999afe65`.

- **Merged head `3c2160b1`** (`merged-head-3c2160b1/`): `scripts/bits.sh`, dump on `main`, compare on `3c2160b1`. `rope_bits.py` 72/72 cases bit-identical, `qmv_bits.py` 120/120, `sdpa_bits.py` 35/35, `qmm_fma_bits.py` 21/21. `omarchy_fast_ops_tests` 50/50 test cases. The 4 failed assertions are all in the `doctest::may_fail` case `sdpa vjp dk dv match finite differences at rep=1 (path per gate)`, as before. That boot ran with `nr_cpus=1`, and the run reports no timing.
- **Six more RoPE timing pairs** (`timing-3c2160b1/`): `ab.py` with 6 interleaved pairs, `main` (A) against `3c2160b1` (B), same probe and procedure as above, on a normal 10-CPU boot. `perf-3c2160b1-rope-summary.txt` lists every process.

  | form | `main`, run order | `3c2160b1`, run order | means | B/A |
  |---|---|---|---|---|
  | f16 base | 40.7, 32.6, 32.7, 32.7, 32.6, 32.8 | 33.3, 32.6, 32.8, 32.3, 32.7, 32.1 | 34.0, 32.6 | 0.959 |
  | bf16 base | 126.4, 128.8, 142.9, 137.5, 138.6, 137.7 | 98.9, 99.0, 100.3, 99.9, 100.3, 99.7 | 135.3, 99.7 | 0.737 |
  | bf16 `freqs` | 1053.7, 1239.7, 1247.0, 1290.5, 1294.6, 1290.8 | 1078.2, 1119.8, 1224.5, 1205.9, 1226.5, 1233.5 | 1236.0, 1181.4 | 0.956 |
  | f16 `freqs` | 1016.2, 1193.5, 1197.3, 1239.2, 1241.6, 1236.6 | 939.6, 986.0, 1091.0, 1069.1, 1084.5, 1099.6 | 1187.4, 1045.0 | 0.880 |

  f16 base: the B readings (32.1 to 33.3) lie on both sides of the A readings (32.6 to 32.8, apart from A's first process at 40.7). The medians are 32.7 (A) and 32.65 (B). The earlier 35.2 did not recur. So f16 base shows no measurable change either way; the 0.959 ratio of means comes from A's first process. The other three forms repeat the first measurement's direction and size. The two `freqs` forms rose over the run in both arms.
- **Half-split default keyed on G16G, `1b1b71b1`** (`g16g-default-1b1b71b1/`): `omarchy_fast_ops_tests` 51/51 test cases with `MLX_OMARCHY_NO_ROPE_HALF_SPLIT` unset, `=1` and `=0` (the new part-rule case, plus the dispatch case now checking the route each setting selects), with the same 4 may-fail assertions. `scripts/bits.sh` compare against the `main` dump: 72/72, 120/120, 35/35, 21/21 bit-identical.

`scripts/rope_summary.py` writes the RoPE summaries: `summarize.py` skips `rope_bits.py` records, which is why `perf-p4-rope-summary.txt` was empty before. The scripts that ran these steps on the host are in `followup-2026-11-03/scripts/`.

## Files

`host.txt`, `wheels.txt`, `battery.txt`, `bits-p4-vs-main.log`, `bits-selftest.log`, `engage-rope-kernels.txt`, `perf-p4*.jsonl`, `perf-p4*-summary.txt`, `trace-*.txt`, `trace-*.run.log`, `reopen.log`, `scripts/`, `followup-2026-11-03/`, `SHA256SUMS`.
