# bf16 decode attention arm on 256 lanes (M4 Mac mini)

Result: the composition-exact bf16 arm of `sdpa_decode_native.comp` now runs 256-lane workgroups instead of 1024, with the score loop striding the keys by 256. Only lanes 0..255 ever did softmax work and only lanes below the head width write outputs, so every per-key chain is unchanged: 35/35 `sdpa_bits.py` cases are bit-identical to `main`, including the rows arm, batch 2 and a 7100-key cache. On the M4 Mac mini the bf16 decode SDPA dispatch drops from 76.9 to 68.6 microseconds in the trace and from 109.3 to 94.8 (40 keys) and 146.8 to 136.2 (400 keys) microseconds of wall time per call; decode moves by +0.94 percent on granite-4.0-1b-4bit (38.85 -> 39.21 tok/s) and +0.15 percent on Qwen3.5-4B (13.30 -> 13.32) over two process pairs each, which is not a demonstrated end-to-end gain; token ids are identical. The arm's capability gate now asks for the 256-lane workgroup it declares, and two capability-simulation profiles cover that boundary.

## What changed

- `shaders/sdpa_decode_native.comp`: under `BF16_IO`, `local_size_x = 256`; the score loop strides by 256; comments say 256. The f16 arm keeps 1024 lanes.
- `primitives.cpp`: `decode_bf16_ready` asks for 256 invocations and 256 in x; the comments that described a 1024-lane bf16 workgroup say 256.
- `capability_sim.cpp`: profiles `workgroup-256` (workgroup limits 256/256) and `workgroup-128` (128/128); `tests/omarchy/CMakeLists.txt` registers both.
- `test_capability_simulation.cpp`: `bf16 decode sdpa arm routes per the workgroup and shared-memory gate` runs under every profile a one-token decode, batch 2, four causal query rows (the rows arm, `SdpaDecodeRowsBF16Hd128`) and a 7100-key cache. Each leg must dispatch the arm exactly when the workgroup limits reach 256 and shared memory holds the arm's (width + 7168 + 256)-float stream, and must store the composition's bits either way; the stamped-report case checks the two new profiles' axes, and the unit pass checks their `apply()` deltas.

## Host and stack (`host.txt`)

Apple Mac mini (M4, 2024), `apple,j773g` / `apple,t8132`, 16 GB; kernel `7.1.13-1-20-ARCH` with the `asahi_m4` G16G driver (GPU ID `0x0a021100` rev `0x11`, boot firmware `mBoot-18000.161.10`), PMP not bound; Mesa 26.3.0-devel Honeykrisp `git-ea328336ef` (techmore/mesa `m4-honeykrisp-compute`), `Apple M4 (G16G B1)`; `AGX_APPLE9_MAX_SHARED=32768` (the arm needs about 30 KB of workgroup memory), `MLX_OMARCHY_BATCH_WORK=120000`; shaderc `glslc` 2026.4 (all 11 `sdpa_decode_native.comp` targets compiled in the wheel build); Python 3.14.8, mlx-lm 0.31.3.

## Builds (`wheels.txt`)

A: `main` `a69d9e6`; B: this change, code commit `98f0911`; profiling builds of `main` and of the six-change integration (`749902d`). Fresh worktree, `scripts/prepare-mlx.sh`, new CMake tree per build; every process printed the provenance line and refused unless its stamp matched the intended commit. `scripts/ab.py` stops on the first process that exits non-zero (provenance refusal 3, timeout 124, any other status) or adds a GPU driver error: it writes that record, runs no later arm and exits with the child's status. GPU error monitoring is required: if `journalctl -k` cannot run or exits non-zero, `ab.py` refuses with status 8 before any arm runs and writes no record (after an arm, it writes that record with the monitor error and stops with 8). `scripts/test_ab.py` checks the rc 3, rc 124, absent-monitor and failing-monitor cases without a GPU. The records here were written by the earlier runner, which logged such a status but kept going; every record here has `rc` 0 and no new kernel error. The monitor was live on the host: `scripts/facts.sh`, run as the same user, read the GPU driver's kernel lines through the same `journalctl -k` (`host.txt`).

## Tests (`battery.txt`)

This change: `omarchy_capability_sim_tests` 8/8 in each of the eight profiles (the six existing ones plus `workgroup-256` and `workgroup-128`), `omarchy_sdpa_decode_fused_tests` 7/7, `omarchy_fast_ops_tests` 49/49. The integration battery fails exactly the 34 cases `main` fails.

This is not an all-green battery. On this host `main` fails 34 of 762 cases and the six-change integration the same 34 of 800 (`receipts/2026-11-01-m4-g16g-part1/battery.txt` names them). `omarchy_fast_ops_tests` exits 0 with 4 failed assertions, all inside `sdpa vjp dk dv match finite differences at rep=1 (path per gate)`, which upstream marks `doctest::may_fail` (an expected failure); its passing case total does not make it a clean suite.

## Exactness (`bits-p5-vs-main.log`)

`sdpa_bits.py` dump on `main`, compare on this change: bf16 and f16; head widths 64, 128 and 256; 1 to 600 keys; plus the bf16 rows arm (4 causal query rows), batch 2 and a 7100-key cache, with f16 controls: 35/35 bit-identical. `engage-sdpa-kernels.txt` (a profiling run of the same cases on the integration) shows the arm engaged (`SdpaDecodeNativeBF16Hd128` and the f16 arm `SdpaDecodeNativeF16Hd128`). `compare` exits 1 on any mismatch before it times anything (`bits-selftest.log`). The GEMV, RoPE and bf16 Q4 prefill compares in the same log are identical too.

## Speed

Decode (`perf-p5-summary.txt`): `decode_bench.py <repo> 64 4`, two interleaved pairs, median of process medians; load below 0.5 and 20 s idle before each process; SMC sensors 32 to 44 C.

| model | A main | B this change | B/A | token ids |
|---|---:|---:|---:|---|
| granite-4.0-1b-4bit | 38.85 | 39.21 | 1.009 | identical |
| Qwen3.5-4B-MLX-4bit | 13.30 | 13.32 | 1.002 | identical |

Decode attention alone (`perf-p5-sdpa.jsonl`, `sdpa_bits.py` timing: host wall time per call of SDPA plus the query scaling, 200 calls in one eval, 16 query heads x 128, 4 KV heads; two interleaved pairs, mean of the two processes): bf16 40 keys 109.3 -> 94.8 us, bf16 400 keys 146.8 -> 136.2 us; the unchanged f16 arm reads 31.2 -> 30.5 and 52.6 -> 49.9 (the run-to-run spread of these wall times is about 5 percent). Shader time is in the trace below.

## Dispatch trace (`trace-*.txt`)

granite-4.0-1b-4bit decode, 40 `SdpaDecodeNativeBF16Hd128` dispatches per step: 76.9 us per dispatch on `main`'s profiling build, 68.6 us on the integration's (69.7 with the fused routes on, the same dispatch list as `main`). The integration carries all six changes, so this delta is not attributed to this PR alone. Every dispatch is a Vulkan dispatch.

## Device reopen

`reopen.py` passed before and after the bit-compare set and at the end of the session; no new GPU driver message in the kernel log.

## Other parts

Bit-identical by construction. The newly admitted devices (workgroup limits 256 to 1023) are covered by the `workgroup-256` profile. G13 and G14: not measured here. Their per-width key windows were tuned with 1024 lanes; the commands for the crossover question are `sdpa_bits.py dump`/`compare` across the two wheels and a decode A/B at 40, 400 and 2000 keys.

## Files

`host.txt`, `wheels.txt`, `battery.txt`, `bits-p5-vs-main.log`, `bits-selftest.log`, `engage-sdpa-kernels.txt`, `perf-p5*.jsonl`, `perf-p5*-summary.txt`, `trace-*.txt`, `trace-*.run.log`, `reopen.log`, `scripts/`, `SHA256SUMS`.
