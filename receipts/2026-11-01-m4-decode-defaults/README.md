# Fused decode GEMV default off on G16G (M4 Mac mini)

Result: with `MLX_OMARCHY_FUSED_GEMV` unset, a G16G device (Apple M4) now runs the per-node decode GEMV instead of the fused GEMV groups. The fused trio (`MLX_OMARCHY_FUSED_TRIO`) keeps its default, on, on every part. On the M4 Mac mini this change (code commit `f333327`) decodes faster on every model measured: granite-4.0-1b-4bit 38.87 -> 46.38 tok/s (1.193x), Qwen2.5-1.5B-Instruct-4bit 44.79 -> 62.52 (1.396x), Llama-3.2-3B-Instruct-4bit 12.56 -> 14.31 (1.140x, one noisy pair; see Speed) and Qwen2.5-0.5B-Instruct-4bit 83.69 -> 114.39 (1.367x). The 64 generated token ids are identical to `main`'s on all four models, Qwen2.5-0.5B included. Numerics gate (`docs/numerics-gate.md`): **passed** on both models (Qwen2.5-0.5B's perplexity deviation is back to `main`'s; see Numerics gate). Every other part keeps the fused GEMV route on (`omarchy::fused_gemv_off_default_part` is true only for names containing `G16G`), an explicit `1` or `0` wins on every part, and the decision is made per call for the device that executes the work.

## History: why this is narrower than the first version

1. **First version, code commit `34d0153`** (both routes default off on G16G). It decoded 1.19x to 1.35x faster (measured at `9d32f27`; the first version's tables are kept below and in `perf-p1*`).
2. **First numerics gate on `34d0153`: not passed** (`numerics-gate/first-gate-34d0153/`). granite-4.0-1b-4bit passed all three checks. Qwen2.5-0.5B-Instruct-4bit passed checks 1 and 2 and **failed check 3**: S=1 PPL deviation from the reference 0.000225 against `main`'s 0.0000524, with run-to-run noise 0 (every run deterministic), so check 3 allowed no difference.
3. **Diagnosis** (`numerics-gate/diagnosis/`, run after the gate and not part of it): the same Qwen S=1 teacher-forced measurement on the `34d0153` wheel with each route forced back on. With `MLX_OMARCHY_FUSED_TRIO=1` (GEMV route off, trio on) all 5,120 chosen log-probabilities equal `main`'s exactly (PPL 7.405985). With `MLX_OMARCHY_FUSED_GEMV=1` (trio off) they equal the failing candidate's exactly (PPL 7.404706). With both forced on they equal `main`'s. The whole deviation came from defaulting the trio off; the GEMV default alone changed nothing in that measurement. Check 1 had not caught it because its per-op cells replay each op alone, not the trio's fused legs. granite was never affected: the trio legs are f16 only and granite runs bf16. An exploratory speed run (`diagnosis/speed-exploratory/`, two pairs per model, not a receipt measurement) found the GEMV default alone as fast as or faster than both routes off.
4. **This version, code commit `f333327`**, on top of `34d0153`: only the fused GEMV route defaults off on G16G. Everything in the sections below that is not labeled "first version" was measured on `f333327`: the suites, the speed A/B and a complete new numerics gate run (fresh runs of both arms).

## What changed (both commits together, against `main`)

- `device.h`: `omarchy::fused_gemv_off_default_part(device_name)`, pure, true only for `G16G`.
- `fused_chain.cpp` / `fused_chain.h`: `fused_route_enabled(variable, device_name)`: the master gate, then an explicit setting, then the default (unset `MLX_OMARCHY_FUSED_GEMV` follows the part rule; unset `MLX_OMARCHY_FUSED_TRIO` is on). Nothing is cached. `fused_gemv_enabled(device)`, `fused_trio_enabled(device)` and the two dependent gates take the device. The planner asks, per tape node, for the GPU device of the node's stream and never plans a node on a CPU stream. The GEMV gate no longer returns before the RoPE-pair plan, so the trio does not depend on it. That is what lets the trio stay on with the GEMV route off. `primitives.cpp` (RMSNorm) asks for its encoder's device.
- Scope: the GEMV route is single-row decode (the multi-token column stays behind `MLX_OMARCHY_QMM_VEC_TOKEN_MULTI`), so prefill does not change. The trio and therefore f16 prefill keep `main`'s kernels.
- Tests (`test_fused_chain.cpp`):
  - the part rule;
  - `fused_route_enabled` for G16G and G13G names alternating in both orders, with the variables unset, `1`, `0` and the master gate off: unset, the GEMV route is off on G16G only and the trio is on on both;
  - the trio stays on whether the GEMV variable is set or unset;
  - `MLX_OMARCHY_FUSED_GEMV=0 MLX_OMARCHY_FUSED_TRIO=1` dispatches one `FastTrioRopePairF16` for an f16 q/k RoPE pair, bit-identical to the standalone kernel;
  - the SwiGLU epilogue gate test at GEMV `1`, `0`, unset and master-off, with the unset dispatch count matching the part rule and all outputs bit-identical;
  - one eval mixing the GPU stream and a CPU stream, where the CPU nodes stay on the CPU and agree with the GPU.

  `test_kv_ops.cpp`: the producer-direct KV write test pins `MLX_OMARCHY_FUSED_GEMV=1`, the route that creates the merged pair it measures.
- `docs/kernel-flags.md`: one row for the two variables.

## Host and stack (`host.txt`)

- Apple Mac mini (M4, 2024), `apple,j773g` / `apple,t8132`, 16 GB, 10 CPUs. Omarchy, kernel `7.1.13-1-20-ARCH` with the `asahi_m4` G16G GPU driver (GPU ID `0x0a021100` revision `0x11`, boot firmware `mBoot-18000.161.10`). PMP not bound.
- Vulkan: Mesa 26.3.0-devel Honeykrisp (`git-ea328336ef`, techmore/mesa `m4-honeykrisp-compute` at `ea32833`), device `Apple M4 (G16G B1)`, API 1.4.359, `VK_KHR_cooperative_matrix` exposed. `AGX_APPLE9_MAX_SHARED=32768`, `MLX_OMARCHY_BATCH_WORK=120000` for every run.
- Shaders compiled on the host with shaderc `glslc` 2026.4.
- Python 3.14.8, mlx-lm 0.31.3 (no serve patches), the same runtime requirements in every venv (`scripts/mkvenv.sh`).


## Builds (`wheels.txt`)

Every wheel was built on the host from a fresh worktree with `scripts/prepare-mlx.sh` and a new CMake tree (`scripts/build.sh`), with `DEV_RELEASE=1` and `MLX_OMARCHY_LOCAL_VERSION=<short commit>`:

- This version: release `+f333327` and profiling `+diag.f333327` (`-DMLX_OMARCHY_GPU_PROFILING=ON`, for the gate's per-op kernel lists). The baseline is `main` `a69d9e6`, rebuilt on the host on 2026-11-02 (`+a69d9e6`, release and profiling); `wheels.txt` lists each wheel with its hash.
- First version: every speed number in the first-version tables belongs to the measured code commit `9d32f27`, kept on [`evidence/m4-316-speed-9d32f27`](https://github.com/techmore/omarchy-mlx/tree/evidence/m4-316-speed-9d32f27). `34d0153` differs from it only in a test file. The first round's commit `4602018` is on [`evidence/m4-316-round1-4602018`](https://github.com/techmore/omarchy-mlx/tree/evidence/m4-316-round1-4602018).

Every measured process printed the `scripts/mlx_provenance.py` line (RECORD hashes match the loaded `libmlx.so`) and refused to run unless the wheel stamp equals the commit that side was meant to be (`scripts/prov.py`). `scripts/ab.py` refuses an A/B whose two stamps and environments are equal. It stops on the first process that exits non-zero, or that adds a GPU driver error, and requires GPU error monitoring (`journalctl -k`): it refuses with status 8 if the monitor is unavailable. Every record from this version has `rc` 0 and no new kernel error.

## Tests (`battery.txt`)

The doctest suites are a separate static-library CMake build of the same prepared tree (`-DMLX_BUILD_TESTS=ON -DMLX_BUILD_PYTHON_BINDINGS=OFF`, the CI configuration), one process per suite under `/tmp/m1-gpu.lock` with a 1800 s timeout.

This version (`f333327`), all with exit status 0 and no new GPU kernel error:

- `omarchy_fused_chain_tests` 42/42;
- `omarchy_kv_ops_tests` 16/16;
- `omarchy_capability_sim_tests` 7/7 in each of the six profiles.

The first version passed the same suites. This is not an all-green battery. On this host `main` fails 34 of 762 cases and the six-change integration of the first versions fails the same 34 of 800 (`receipts/2026-11-01-m4-g16g-part1/battery.txt` names them); none is in these suites. `omarchy_fast_ops_tests` exits 0 with 4 failed assertions in the upstream `doctest::may_fail` case `sdpa vjp dk dv match finite differences at rep=1 (path per gate)`. The full battery was not rerun on `f333327`.

## Exactness

No kernel or shader changed. `bits-p1-vs-main.log` (first version, `34d0153`) holds the `qmv_bits.py` (120/120), `rope_bits.py` (72/72), `sdpa_bits.py` (35/35) and `qmm_fma_bits.py` (21/21) compares against `main`, all bit-identical. `f333327` changes only the gates' defaults, and these compares were not rerun on it.

The route change changes the accumulation order between the fused and the per-node GEMV. On all four models in the A/B, both builds generate the same 64 token ids (`perf-p1n.jsonl`, `ids_sha`), Qwen2.5-0.5B included, where the first version diverged at token 49. The numerics gate (below) measures the effect against the reference.

First version, for the record: on Qwen2.5-0.5B the first version's 64 generated ids diverged from `main`'s at token 49, a near tie (`tokens.jsonl`).

## Speed (`perf-p1n-summary.txt`, raw `perf-p1n.jsonl`)

`scripts/decode_bench.py <repo> 64 4`: a chat-templated fixed prompt, greedy, one warm-up generation, then 4 timed generations of 64 tokens in one process; the process value is the median decode tok/s of the 4. Two interleaved pairs per model (A B, B A), each process with its own provenance gate. Before every process the runner waits for the 1-minute load average to fall below 0.5 and idles 20 s, and it records the load, both SMC temperature sensors the kernel exposes and free memory before and after. Values are the median of the process values. Run 2026-11-02 between 21:51 and 22:02 UTC with nothing else on the GPU (`scripts/speed316n.sh`).

| model | A main | B this change | B/A | token ids |
|---|---:|---:|---:|---|
| granite-4.0-1b-4bit | 38.87 | 46.38 | 1.193 | identical |
| Qwen2.5-1.5B-Instruct-4bit | 44.79 | 62.52 | 1.396 | identical |
| Llama-3.2-3B-Instruct-4bit | 12.56 | 14.31 | 1.140 | identical |
| Qwen2.5-0.5B-Instruct-4bit | 83.69 | 114.39 | 1.367 | identical |

Process values (A; B): granite 38.87, 38.87; 46.37, 46.39. Qwen2.5-1.5B 44.81, 44.77; 62.55, 62.49. Llama-3.2-3B 11.87, 13.25; 14.31, 14.31. Qwen2.5-0.5B 83.70, 83.68; 114.40, 114.38. In every cell except one, the two processes of an arm are within 0.1 percent of each other. The exception is Llama-3.2-3B, where the second pair was noisy in both arms. Its `main` process generated at 13.12 to 13.58 tok/s, while the first pair's `main` process ran at 11.81 to 11.91, and the seven earlier `main` processes for this model on this host had medians of 11.76 to 11.96 (`perf-p1.jsonl`, `perf-round1-p1.jsonl`, `numerics-gate/diagnosis/speed-exploratory/`). This version's process in that pair ranged 13.45 to 14.89 within its four generations. The table reports the protocol's median, 1.140, as measured. The first pair alone gives 1.206. No pair was repeated or dropped. Temperatures 28 to 38 C; load before each process at most 0.35; peak GPU memory within 0.02 GiB of `main` on every model.

First version (`9d32f27`, both routes off), for comparison: 1.192, 1.351, 1.212 and 1.303 on the same four models (`perf-p1-summary.txt`). Its 512-token Qwen2.5-1.5B prefill check (`perf-p1-prefill*`, 385.6 -> 385.6 tok/s) measured the trio's prefill scope, which this version no longer changes. Prefill was not re-measured.

## Numerics gate (`numerics-gate/`)

**Gate: passed** (`numerics-gate/README.md`). `docs/numerics-gate.md` against stock upstream MLX 0.32.3 on the CPU in float32, with the same weights and the in-tree GPL-3 text: S=1, 5,120 teacher-forced positions per model, five fresh interleaved runs per arm, `main` `a69d9e6` against `f333327`. The harness check passed on the baseline (S=1 against one pass, relative difference 1.91e-3 for granite and 1.41e-3 for Qwen2.5-0.5B).

| model | arm | top-1 vs reference | flips (near ties) | PPL | mean abs dlogprob |
|---|---|---:|---:|---:|---:|
| granite-4.0-1b-4bit | `main` | 97.500% | 128 (106) | 7.56819 | 0.07335 |
| granite-4.0-1b-4bit | `f333327` | 97.500% | 128 (106) | 7.56819 | 0.07335 |
| granite-4.0-1b-4bit | reference | | | 7.58552 | |
| Qwen2.5-0.5B-Instruct-4bit | `main` | 99.453% | 28 (28) | 7.40599 | 0.01156 |
| Qwen2.5-0.5B-Instruct-4bit | `f333327` | 99.453% | 28 (28) | 7.40599 | 0.01156 |
| Qwen2.5-0.5B-Instruct-4bit | reference | | | 7.40637 | |

| check | granite-4.0-1b-4bit | Qwen2.5-0.5B-Instruct-4bit |
|---|---|---|
| 1, per-op (27 cells) | 27/27 no worse: pass | 27/27 no worse: pass |
| 2, token choices (with near ties, against the baseline's Wilson lower bound) | 99.570% against 99.350%: pass | 100.000% against 99.925%: pass |
| 3, PPL deviation (against the baseline's plus noise) | 0.0022871 against 0.0022871 + 0: pass | 0.0000524 against 0.0000524 + 0: pass |
| first run, `34d0153`, check 3 (for comparison) | 0.0022871 against 0.0022871 + 0: pass | **0.0002251 against 0.0000524 + 0: fail** |

The failing check from the first run is resolved under the same protocol: Qwen2.5-0.5B's deviation returns from 0.0002251 to `main`'s 0.0000524. Every run of both arms gives the same values at every position on both models, so the gate passes on equal error, not smaller error. The route did change in the gate's per-op cells. On profiling builds, `main` dispatched the fused groups (`QmmVecQ4MultiSubgroupBF16` / `F16`); `f333327` dispatched none of them, only the per-node `QmmVecQ4WordSubgroup*` GEMV, while the trio kernels (`FastTrioNormF16`, `FastTrioSwigluF16`, `FastTrioRopePairF16`) ran on both (`numerics-gate/perop/316n-*.kernels.txt`). Each profiling build's 324 outputs equal its release build's bit for bit. All 20 runs exited 0 with no new GPU driver error. The gate README lists what was reused from the first run and what was measured fresh.

Limits: M4 only, two models, and not the parity-contract model, which is not on this host.

## Dispatch trace (`trace-*.txt`, recorded with the first version's builds)

Profiling wheels, `scripts/profile_generate.py` (32 tokens, greedy), analyzed per decode step with `scripts/trace_steps.py` over the decode submissions whose records are all in the file (the harness's end record counts every dispatch with none dropped; it writes a slot's records on reuse, so some submissions have no records in the file):

| build, model, environment | dispatches per decode step | fused or per-node kernels per step |
|---|---:|---|
| main, granite | 571 | `QmmVecQ4MultiSubgroupBF16` 80, `QmmVecQ4WordSubgroupBF16` 81 |
| integration, granite, unset | 771 | `QmmVecQ4WordSubgroupBF16` 281, `CopyGeneralBF16` 80 |
| integration, granite, `FUSED_GEMV=1 FUSED_TRIO=1` | 571 | `QmmVecQ4MultiSubgroupBF16` 80 |
| main, Qwen2.5-0.5B | 225 | `QmmVecQ4MultiSubgroupF16` 96, `FastTrioNormF16` 49, `FastTrioRopePairF16` 24 |
| integration, Qwen2.5-0.5B, unset | 465 | `QmmVecQ4WordSubgroupF16` 169, `BinaryVecF16` 120, `FastRopeHalfSplitF16` 48 |
| integration, Qwen2.5-0.5B, `FUSED_GEMV=0 FUSED_TRIO=1` | 441 | `FastTrioNormF16` 49, `FastTrioRopePairF16` 24, `FastTrioSwigluF16` 24 |

The last row is the trio override with the GEMV route off, in a real model. Every dispatch is a Vulkan dispatch; no step runs on the CPU. The harness serializes each dispatch with timestamps and barriers, so its per-step GPU time (44.7 ms per-node against 41.7 ms fused for granite) is not the decode speed; the wall-clock A/B above is.

For this version: the integration row `FUSED_GEMV=0 FUSED_TRIO=1` above is the environment that `f333327` now selects by default on G16G. The per-op kernel lists from the gate (`numerics-gate/perop/316n-*-dmain.kernels.txt`, `316n-*-dp1n.kernels.txt`) come from `f333327`'s own profiling build.

## Device reopen

`reopen.py` (a fresh process opens the GPU, runs a matmul and a 4-bit `quantized_matmul` against the CPU stream) passed before and after every bit-compare set and at the end of the session (`reopen.log`). An earlier build of this change failed exactly that check: its planner looked up a command encoder for a node on the CPU stream; the per-node device lookup now skips CPU streams and the CPU-stream test covers it. The kernel log showed no new GPU driver error during any run (each record's `new_gpu_kernel_errors` is 0).

The device reopen was not repeated as a separate step for `f333327`. Every GPU process in this version's A/B, suites and gate ran under the same kernel-log monitor and added no GPU driver error.

## Not covered here

- Other parts keep today's defaults. The rule names G16G only, so M4 Pro/Max parts (G16S/G16C) keep the fused GEMV route until they are measured.
- The fused group on G16G binds 26 resources through the driver's descriptor-table path; if that path gets faster, this default is worth re-measuring with the same commands.

## Files

- `host.txt`, `wheels.txt`: host, driver, firmware and model identity, wheel names and hashes.
- `battery.txt`: suite summaries, `main`, the first version and this version.
- `perf-p1n.jsonl`, `perf-p1n-summary.txt`: this version's A/B, every process with its provenance line, environment, load, temperatures and memory.
- `numerics-gate/`: this version's gate (README, records, analysis, per-op kernel lists, scripts), `first-gate-34d0153/` (the first version's gate, unchanged records) and `diagnosis/`.
- First version: `bits-p1-vs-main.log`, `perf-p1*`, `perf-round1-p1*`, `trace-*.txt`, `trace-*.run.log`, `tokens.jsonl`, `reopen.log`.
- `scripts/`: the harness.
- `SHA256SUMS`.
