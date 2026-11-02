# Fused decode GEMV and trio default off on G16G (M4 Mac mini)

Result: with `MLX_OMARCHY_FUSED_GEMV` and `MLX_OMARCHY_FUSED_TRIO` unset, a G16G device (Apple M4) now runs the per-node GEMV and the standalone trio kernels. On the M4 Mac mini that decodes faster on every model measured: granite-4.0-1b-4bit 38.85 -> 46.32 tok/s (1.192x), Qwen2.5-1.5B-Instruct-4bit 44.80 -> 60.54 (1.351x), Llama-3.2-3B-Instruct-4bit 11.82 -> 14.32 (1.212x), Qwen2.5-0.5B-Instruct-4bit 83.64 -> 108.96 (1.303x). The 64 generated token ids are identical on the first three models; on Qwen2.5-0.5B they diverge (see Exactness). Qwen2.5-1.5B 512-token prefill is unchanged (385.6 -> 385.6 tok/s, same argmax). Every other part keeps both fused routes on (`omarchy::fused_decode_off_default_part` is true only for names containing `G16G`), an explicit `1` or `0` wins on every part, and the decision is made per call for the device that executes the work.

## What changed

- `device.h`: `omarchy::fused_decode_off_default_part(device_name)`, pure, true only for `G16G`.
- `fused_chain.cpp` / `fused_chain.h`: `fused_route_enabled(variable, device_name)` (master gate, then an explicit setting, then the part rule; nothing cached). `fused_gemv_enabled(device)`, `fused_trio_enabled(device)` and the two dependent gates take the device. The planner asks per tape node for the GPU device of the node's stream and never plans a node on a CPU stream; the GEMV gate no longer returns before the RoPE-pair plan, so the trio is independent of it. `primitives.cpp` (RMSNorm) asks for its encoder's device.
- Scope: the GEMV route is single-row. The trio's f16 RMSNorm and SwiGLU legs have no row-count guard, so on G16G the trio default moves f16 prefill norms and SwiGLU chains to their standalone kernels too; the 512-token Qwen2.5-1.5B prefill below measures that and finds no change.
- Tests (`test_fused_chain.cpp`): the part rule; `fused_route_enabled` for G16G and G13G names alternating in both orders with the variables unset, `1`, `0` and the master gate off; `MLX_OMARCHY_FUSED_GEMV=0 MLX_OMARCHY_FUSED_TRIO=1` dispatching one `FastTrioRopePairF16` for an f16 q/k RoPE pair, bit-identical to the standalone kernel; the SwiGLU epilogue gate test at GEMV `1`, `0`, unset and master-off, the unset dispatch count matching the part rule and all outputs bit-identical; one eval mixing the GPU stream and a CPU stream, the CPU nodes (single-row 4-bit and dense bf16 projections, an f16 SwiGLU chain) left to the CPU and agreeing with the GPU. `test_kv_ops.cpp`: the producer-direct KV write test pins both routes on (on G16G it failed with the new default before the pin).
- `docs/kernel-flags.md`: one row for the two variables.

## Host and stack (`host.txt`)

- Apple Mac mini (M4, 2024), `apple,j773g` / `apple,t8132`, 16 GB, 10 CPUs. Omarchy, kernel `7.1.13-1-20-ARCH` with the `asahi_m4` G16G GPU driver (GPU ID `0x0a021100` revision `0x11`, boot firmware `mBoot-18000.161.10`). PMP not bound.
- Vulkan: Mesa 26.3.0-devel Honeykrisp (`git-ea328336ef`, techmore/mesa `m4-honeykrisp-compute` at `ea32833`), device `Apple M4 (G16G B1)`, API 1.4.359, `VK_KHR_cooperative_matrix` exposed. `AGX_APPLE9_MAX_SHARED=32768`, `MLX_OMARCHY_BATCH_WORK=120000` for every run.
- Shaders compiled on the host with shaderc `glslc` 2026.4.
- Python 3.14.8, mlx-lm 0.31.3 (no serve patches), the same runtime requirements in every venv (`scripts/mkvenv.sh`).

## Builds (`wheels.txt`)

Every wheel was built on the host from a fresh worktree with `scripts/prepare-mlx.sh` and a new CMake tree (`scripts/build.sh`), `DEV_RELEASE=1` and `MLX_OMARCHY_LOCAL_VERSION=<short commit>`:

- A: upstream `main` `a69d9e6` (`+a69d9e6`).
- B: every speed number belongs to the measured code commit `9d32f27` (`+9d32f27`), kept on [`evidence/m4-316-speed-9d32f27`](https://github.com/techmore/omarchy-mlx/tree/evidence/m4-316-speed-9d32f27). The head's code commit `34d0153` differs from it only in a test file (`git diff --stat 9d32f27 34d0153`: `overlay/tests/omarchy/test_fused_chain.cpp | 58 +++---, 1 file changed, 34 insertions(+), 24 deletions(-)`); the suites and bit compares were rerun on `+34d0153`. The first round's commit `4602018` is on [`evidence/m4-316-round1-4602018`](https://github.com/techmore/omarchy-mlx/tree/evidence/m4-316-round1-4602018); it predates the review changes, which touched production code.
- Profiling builds (`-DMLX_OMARCHY_GPU_PROFILING=ON`) of `a69d9e6` and of the integration of the six M4 code changes (`749902d`) for the dispatch trace.

Every measured process printed the `scripts/mlx_provenance.py` line (RECORD hashes match the loaded `libmlx.so`) and refused to run unless the wheel stamp equals the commit that side was meant to be (`scripts/prov.py`); `scripts/ab.py` refuses an A/B whose two stamps and environments are equal. `scripts/ab.py` stops on the first process that exits non-zero (provenance refusal 3, timeout 124, any other status) or adds a GPU driver error: it writes that record, runs no later arm and exits with the child's status. GPU error monitoring is required: if `journalctl -k` cannot run or exits non-zero, `ab.py` refuses with status 8 before any arm runs and writes no record (after an arm, it writes that record with the monitor error and stops with 8). `scripts/test_ab.py` checks the rc 3, rc 124, absent-monitor and failing-monitor cases without a GPU. The records in this receipt were written by the earlier runner, which logged such a status but kept going; every record here has `rc` 0 and no new kernel error. The monitor was live on the host: `scripts/facts.sh`, run as the same user, read the GPU driver's kernel lines through the same `journalctl -k` (`host.txt`).

## Tests (`battery.txt`)

The doctest suites are a separate static-library CMake build of the same prepared tree (`-DMLX_BUILD_TESTS=ON -DMLX_BUILD_PYTHON_BINDINGS=OFF`, the CI configuration), one process per suite under `/tmp/m1-gpu.lock` with a 1800 s timeout.

This change (code commit `34d0153`): `omarchy_fused_chain_tests` 42/42, `omarchy_kv_ops_tests` 16/16, `omarchy_capability_sim_tests` 7/7 in each of the six profiles. The integration battery (`receipts/2026-11-01-m4-g16g-part1/`) fails exactly the 34 cases `main` fails on this host; none is in these suites. This is not an all-green battery. On this host `main` fails 34 of 762 cases and the six-change integration the same 34 of 800 (`receipts/2026-11-01-m4-g16g-part1/battery.txt` names them). `omarchy_fast_ops_tests` exits 0 with 4 failed assertions, all inside `sdpa vjp dk dv match finite differences at rep=1 (path per gate)`, which upstream marks `doctest::may_fail` (an expected failure); its passing case total does not make it a clean suite. Before the two route-pinning edits, the integration failed exactly those two cases on G16G (`eager q4 decode gemv swiglu epilogue env gate off keeps the chain`: 3 dispatches where the fused route gives 2; `producer-direct KV write is bit-exact against the merged pair`: no merged pair to delete).

## Exactness

The kernels are unchanged: `bits-p1-vs-main.log` holds the `qmv_bits.py` (120/120), `rope_bits.py` (72/72), `sdpa_bits.py` (35/35) and `qmm_fma_bits.py` (21/21) compares against `main`, all bit-identical, with a device reopen before and after. Each compare exits 1 on any mismatch (`bits-selftest.log`: exit 0 on a matching baseline, 1 on a baseline with one flipped value).

The route change does change the accumulation order between the fused and the per-node GEMV and trio kernels. On granite-4.0-1b, Qwen2.5-1.5B and Llama-3.2-3B the 64 generated token ids are identical. On Qwen2.5-0.5B they differ: `tokens.jsonl` holds, for both builds, the prompt token ids, the 64 generated ids and the gap between the two most likely tokens at every step. Both builds see the same 44 prompt tokens and generate the same first 48 tokens; at token 49 `main`'s two most likely tokens are tied in log-probability (gap 0.0) and this change's are 0.0156 apart, so greedy decoding takes the other one. The text stays coherent on both sides. Qwen2.5-1.5B's ids are identical over all 64 tokens in the same probe.

## Speed (`perf-p1-summary.txt`, raw `perf-p1.jsonl`)

`scripts/decode_bench.py <repo> 64 4`: chat-templated fixed prompt, greedy, one warm-up generation, then 4 timed generations of 64 tokens in one process; the process value is the median decode tok/s of the 4. Two interleaved pairs per model (A B, B A), each process its own provenance gate. Before every process the runner waits for the 1-minute load average to fall below 0.5 and idles 20 s; it records the load, both SMC temperature sensors the kernel exposes (NAND and Wi-Fi module; this kernel exposes no SoC die sensor and no thermal zones) and free memory before and after. Values are the median of the process values.

| model | A main | B this change | B/A | token ids |
|---|---:|---:|---:|---|
| granite-4.0-1b-4bit | 38.85 | 46.32 | 1.192 | identical |
| Qwen2.5-1.5B-Instruct-4bit | 44.80 | 60.54 | 1.351 | identical |
| Llama-3.2-3B-Instruct-4bit | 11.82 | 14.32 | 1.212 | identical |
| Qwen2.5-0.5B-Instruct-4bit | 83.64 | 108.96 | 1.303 | differ (Exactness) |

Prefill, f16 trio scope (`prefill_bench.py <repo> 512 4`): Qwen2.5-1.5B-Instruct-4bit 385.6 -> 385.6 tok/s (B/A 1.000), same last-position argmax. Process-to-process spread is under 0.5 percent in every cell; temperatures 33 to 45 C; peak GPU memory within 0.04 GiB.

A first round of the same A/B at code commit `4602018` (before the review changes, three pairs) gave the same picture: 1.192, 1.351 and 1.250 on the first three models (`perf-round1-p1-summary.txt`).

## Dispatch trace (`trace-*.txt`)

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

## Device reopen

`reopen.py` (a fresh process opens the GPU, runs a matmul and a 4-bit `quantized_matmul` against the CPU stream) passed before and after every bit-compare set and at the end of the session (`reopen.log`). An earlier build of this change failed exactly that check: its planner looked up a command encoder for a node on the CPU stream; the per-node device lookup now skips CPU streams and the CPU-stream test covers it. The kernel log showed no new GPU driver error during any run (each record's `new_gpu_kernel_errors` is 0).

## Not covered here

- Other parts keep today's defaults; the rule names G16G only, so M4 Pro/Max parts (G16S/G16C) keep both fused routes until they are measured.
- The fused group on G16G binds 26 resources through the driver's descriptor-table path; if that path gets faster, this default is worth re-measuring with the same two commands.

## Numerics gate

`numerics-gate/README.md`: docs/numerics-gate.md for this change against a stock upstream MLX CPU fp32 reference (teacher-forced agreement, S=1 or route-shaped PPL, per-op error, harness check). Its first line states whether the gate has been evaluated; until it says complete, `numerics-gate/RESUME.md` lists what remains.

## Files

- `host.txt`, `wheels.txt`: host, driver, firmware and model identity (model `sha256` of each `model.safetensors`), wheel names and hashes.
- `battery.txt`: suite summaries and failing case names, `main` and this change.
- `bits-p1-vs-main.log`: bit compares and reopen checks.
- `perf-p1.jsonl`, `perf-p1-summary.txt`, `perf-p1-prefill.*`, `perf-round1-p1*`: every A/B process with its provenance line, environment, load, temperatures and memory.
- `trace-*.txt`, `trace-*.run.log`: per-step dispatch tables and the traced runs.
- `tokens.jsonl`: prompt and generated token ids with the top-2 gaps.
- `scripts/`: the harness (`build.sh`, `mkvenv.sh`, `prov.py`, `ab.py`, `once.py`, `decode_bench.py`, `prefill_bench.py`, `tokens.py`, `battery.sh`, `bits.sh`, `bits_selftest.sh`, `tamper.py`, the bit probes, `trace.sh`, `trace_gen.py`, `trace_steps.py`, `summarize.py`, `facts.sh`, `reopen.py`).
- `SHA256SUMS`.
