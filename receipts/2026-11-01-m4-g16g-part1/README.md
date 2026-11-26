# M4 (G16G) support, part 1: overview, platform bandwidth and the M4 compatibility row

This receipt ties together the six M4 code changes, each with its own receipt and pull request, records what the M4 Mac mini's GPU and memory deliver on this Linux stack, and backs the M4 row in `docs/compatibility.md` and `README.md`.

## The series

Every number below is an A/B on the M4 Mac mini against upstream `main` `a69d9e6`, both sides built on the host from the named commits, with the procedure in each receipt. Each speed figure belongs to the commit it was measured on; where that differs from a PR's head code commit, the measured commit is kept on an `evidence/*` branch of the fork and the difference is test-only (each receipt gives the `git diff --stat`).

| PR | change | changed contract | how it works | measured effect | receipt |
|---|---|---|---|---|---|
| #316 | fused GEMV and trio default off on G16G | default of `MLX_OMARCHY_FUSED_GEMV` / `MLX_OMARCHY_FUSED_TRIO` when unset, per executing device | a pure device-name rule (`fused_decode_off_default_part`) picks the per-node GEMV and standalone trio kernels on G16G | decode 1.19x to 1.35x on four models (granite-4.0-1b 38.85 -> 46.32 tok/s; Qwen2.5-1.5B 44.80 -> 60.54), measured on code commit `9d32f27`; 64 token ids identical on three models, Qwen2.5-0.5B diverges at token 49 (tied top two on `main`); 512-token f16 prefill unchanged | `receipts/2026-11-01-m4-decode-defaults/` |
| #317 | bf16 linear layers keep the scalar-FMA kernel on G16G | `dispatch_matmul` pick between `MatmulBf16Fma` and `MatmulBF16Coopmat` for the linear-layer orientation | a pure device-name rule prefers the FMA kernel where both qualify | Llama-3.2-1B bf16 prefill 1.44x / 1.54x / 1.50x at 128 / 512 / 2048 tokens (268.1 -> 413.8 tok/s at 512); kernel 844-1200 vs 602-735 GFLOP/s | `receipts/2026-11-01-m4-cooperative-matrix-default/` |
| #318 | in-place 4-bit and 8-bit field decode in `qmm_vec.comp` | none observable (same floats) | the masked field is read as the mantissa of the float whose unit sits at its bit, so an `and`, an integer `add` and a subtract replace the shift and `or` | bit-identical (120/120); 4-bit GEMV 3-6 percent faster at the large shapes; decode +3.4 percent granite-4.0-1b, +6.4 percent Qwen2.5-1.5B | `receipts/2026-11-01-m4-qmm-vec-inplace-decode/` |
| #319 | half-split RoPE builds | `RoPE::eval_gpu` kernel choice for the half-split form; `MLX_OMARCHY_NO_ROPE_HALF_SPLIT` | four builds with the traditional and passthrough branches compiled out; the GPU fetches less code per dispatch | bit-identical (72/72); bf16 RoPE dispatch 75.9 -> 53.6 us; granite-4.0-1b decode +6.5 percent; Qwen2.5-0.5B and gemma-3-1b decode unchanged (1.000); f16 base RoPE dispatch 32.2 -> 33.9 us, slower within noise | `receipts/2026-11-01-m4-rope-half-split/` |
| #320 | bf16 decode attention arm on 256 lanes | the bf16 arm's workgroup (1024 -> 256) and its capability gate | only lanes 0..255 ever did work; the score loop now strides the keys by 256 | bit-identical (35/35); bf16 decode SDPA dispatch 76.9 -> 68.6 us on the integration's profiling build; decode +0.94 / +0.15 percent from two process pairs each, not a demonstrated gain | `receipts/2026-11-01-m4-sdpa-decode-bf16-256/` |
| #321 | bf16 Q4 prefill on the scalar-FMA kernel where the cooperative route is unavailable | `QuantizedMatmul::eval_gpu` route for bf16 4-bit group-64 prefill; `MLX_OMARCHY_NO_QMM_FMA_BF16` | a bf16 build of `qmm_fma.comp` replaces the 16x16 tile kernel | with the cooperative matrix hidden: prefill 4.05x to 8.45x (granite-4.0-1b 512 tokens 21.1 -> 171.9 tok/s), measured on code commit `95a3849`; outputs equal the tile kernel's bits on the M4 | `receipts/2026-11-01-m4-bf16-qmm-fma/` |

The six together (integration of the six heads, `749902d`, against `main`, same procedure):

| model | main | integration | ratio | token ids |
|---|---:|---:|---:|---|
| granite-4.0-1b-4bit | 38.83 | 51.34 | 1.322 | identical |
| Qwen2.5-1.5B-Instruct-4bit | 44.81 | 63.22 | 1.411 | identical |
| gemma-3-1b-it-4bit | 42.88 | 54.04 | 1.260 | identical |
| Qwen3.5-4B-MLX-4bit | 13.31 | 15.71 | 1.180 | identical |
| Mistral-7B-Instruct-v0.3-4bit | 16.79 | 18.64 | 1.111 | differ (fused vs per-node accumulation order, #316) |

Prefill (`prefill_bench.py <repo> 512 4`): granite-4.0-1b-4bit 374.8 -> 384.1 tok/s (1.025), Llama-3.2-1B-Instruct-bf16 268.2 -> 417.1 (1.555), same last-position argmax. Decode tok/s, `decode_bench.py <repo> 64 4`, two interleaved pairs. These figures belong to the measured integration commit `749902d`, kept on [`evidence/m4-integration-749902d`](https://github.com/techmore/omarchy-mlx/tree/evidence/m4-integration-749902d); the final integration `08e7d28` differs from it only in test files (`git diff --stat 749902d 08e7d28`: `overlay/tests/omarchy/test_capability_simulation.cpp | 30`, `overlay/tests/omarchy/test_fused_chain.cpp | 58`, 2 files changed, 54 insertions(+), 34 deletions(-)). The integration carries all six changes, so these rows and the trace below show the combined effect, not any one PR's share.

`scripts/ab.py` stops on the first process that exits non-zero (provenance refusal 3, timeout 124, any other status) or adds a GPU driver error: it writes that record, runs no later arm and exits with the child's status. GPU error monitoring is required: if `journalctl -k` cannot run or exits non-zero, `ab.py` refuses with status 8 before any arm runs and writes no record (after an arm, it writes that record with the monitor error and stops with 8). `scripts/test_ab.py` checks the rc 3, rc 124, absent-monitor and failing-monitor cases without a GPU. The records here were written by the earlier runner, which logged such a status but kept going; every record has `rc` 0 and no new kernel error. The monitor was live on the host: `scripts/facts.sh`, run as the same user, read the GPU driver's kernel lines through the same `journalctl -k` (`host.txt`).

## Host and stack (`host.txt`)

- Apple Mac mini (M4, 2024), `apple,j773g` / `apple,t8132`, 16 GB, 10 CPUs.
- Kernel `7.1.13-1-20-ARCH` with the out-of-tree `asahi_m4` G16G GPU driver (GPU ID `0x0a021100` revision `0x11`, boot firmware `mBoot-18000.161.10`); PMP not bound. The kernel work is tracked in the kernel repositories.
- Vulkan: Mesa 26.3.0-devel Honeykrisp with the Apple9 compute port (`git-ea328336ef`, techmore/mesa `m4-honeykrisp-compute` at `ea32833`), device `Apple M4 (G16G B1)`, API 1.4.359, `VK_KHR_cooperative_matrix` (fp32 8x8x8). Every run sets `AGX_APPLE9_MAX_SHARED=32768` (the port's opt-in for more than 1 KiB of workgroup memory) and `MLX_OMARCHY_BATCH_WORK=120000`.
- shaderc `glslc` 2026.4; Python 3.14.8; mlx-lm 0.31.3.

## Standing battery (`battery.txt`)

`main` and the integration, each a fresh static-library CMake build of the prepared tree, every suite of AGENTS.md "Test rules" plus `omarchy_qmv_batch_tests`, `omarchy_sdpa_decode_fused_tests` and the capability-simulation profiles, one process per suite under `/tmp/m1-gpu.lock` with a 1800 s timeout:

| build | suite runs | test cases | failing cases |
|---|---:|---:|---:|
| `main` `a69d9e6` | 27 suites + 6 profiles | 762 | 34 |
| integration `08e7d28` | 27 suites + 8 profiles | 800 | 34, the same cases |

The 34 cases that fail on `main` on this host fail identically on the integration; no case fails only on the integration. 32 of them stop at compute-pipeline creation, which the driver rejects (`VK_ERROR_UNKNOWN`): 18 complex-op cases, 8 eig cases, 3 wide-row partition cases, 2 primitive cases and 1 compiled-tape case. The other two are `gather_qmm sorted-expert tile route ...` (an evaluation error in the gather tile route) and `recycled block is not rewritten by the previous owner in flight` (runtime). Every capability-simulation profile passes on both.

This is not an all-green battery: 34 cases fail on both builds. `omarchy_fast_ops_tests` exits 0 with 4 failed assertions, all inside `sdpa vjp dk dv match finite differences at rep=1 (path per gate)`, which upstream marks `doctest::may_fail` (an expected failure); its suite status is not a clean pass.

## Dispatch trace (`trace-*.txt`)

Profiling build of the integration, granite-4.0-1b-4bit, 32 greedy tokens, per decode step (`scripts/trace_steps.py` over the submissions whose records are all in the file):

| kernel | per step | us per dispatch | share of step GPU time |
|---|---:|---:|---:|
| `QmmVecQ4WordSubgroupBF16` (4-bit GEMV) | 281 | 83.7 | 52.7% |
| `FastRopeHalfSplitBF16` | 80 | 53.6 | 9.6% |
| `CopyGeneralBF16` | 80 | 40.2 | 7.2% |
| `FastRmsNormAppleBF16` | 81 | 38.5 | 7.0% |
| `ElementwiseLiteBF16` | 83 | 37.2 | 6.9% |
| `SdpaDecodeNativeBF16Hd128` | 40 | 68.6 | 6.1% |
| `BinaryVecBF16` | 80 | 34.0 | 6.1% |
| `SwigluBF16` | 40 | 36.6 | 3.3% |

771 dispatches per step (`main`: 571 with the fused routes).

Every kernel is a Vulkan dispatch; no step runs on the CPU. The profiling harness serializes each dispatch with timestamps and barriers, so these per-dispatch times are upper bounds and the step total is not the decode speed.

## Platform status on M4: memory bandwidth

`bandwidth/run_bw.sh` builds the probes from source on the target and runs them; `bandwidth/bw.log` holds every command beside its output.

| probe (1 GiB buffer, memory types 0 and 1) | GB/s |
|---|---:|
| GPU streaming read, 32-bit loads (`bw_read.comp` VEC=1, 1280 x 256 threads) | 70.3-70.7 |
| GPU streaming read, 64-bit loads | 78.5-78.8 |
| GPU streaming read, 128-bit loads | 81.7-82.1 |
| GEMV access mix, 64-bit weight loads (`bw_gemv.comp` WIDE=0, 640 x 128 threads) | 84.3-84.4 |
| GEMV access mix, 128-bit weight loads (WIDE=1) | 86.6-86.9 |
| WIDE=0 with 32 extra live registers | 86.4-86.5 |
| WIDE=0 with 16 dependent fma pairs per row word | 86.5-87.0 |
| WIDE=0 with 64 dependent fma pairs per row word (ALU-bound) | 18.8 |
| CPU read, 1 / 4 / 10 threads (`cpubw.c`) | 35.2 / 55.7 / 70.9 |

- In the trace above the 4-bit decode GEMV is 52.7 percent of the profiled step time.
- `bw_gemv.comp` is a synthetic access mix: the GEMV's load widths, strides and per-row scale and bias reads, with every loaded component consumed (`OpLoad` counts in `bw.log`) and the dequantization, reduction and store left out. It reads faster (84-87 GB/s) than the plain streaming read.
- Hypothesis: the production GEMV's access pattern is not what limits it. The synthetic mix cannot rule the production access pattern out as the limiter, because it leaves out the GEMV's dependency chain, its dequantization and its stores.
- Hypothesis: the memory system bounds decode here. This receipt measures the GPU streaming read at about 82 GB/s (table above); it does not prove that limit is what bounds the GEMV.

Context from a separate public receipt, not measured in this one ([`receipts/2026-11-01-m4-pmp-test.md`](https://github.com/techmore/omarchy-mlx/blob/m4-bf16-qmm-fma/receipts/2026-11-01-m4-pmp-test.md) on the fork's `m4-bf16-qmm-fma` branch, with the kernel tools under `tools/m4-pmp-kernel/`):

- With the platform power manager (PMP) running and the PMC memory voting it needs, CPU reads on this machine rose from 69-73 to 83.6-86 GB/s. That is a CPU measurement.
- The GPU does not run while PMP is on yet: the G16G driver does not register the GPU in PMP's device table (the PTD ids, enable masks and the auxiliary PTD range table are empty), so the GPU firmware waits on a performance-state change that PMP never grants.
- Hypothesis: once that registration exists, the GPU read rate rises with the DRAM state as the CPU's did. This is untested on the GPU. The kernel and driver work is tracked in the kernel repositories.

`run_bw.sh` stops at the first command that fails (compile, shader generator, probe), logs its status and exits with it; it deletes earlier executables and shaders before building, so a failed rebuild cannot leave a stale binary to run. `bandwidth/test_run_bw.sh` checks this without a GPU (compile, generator, probe and OpLoad-count compiler failures). The recorded `bw.log` came from the earlier runner, which logged each status and continued; every one of its 43 logged commands has `rc=0` and the log ends with `DONE`. That runner generated and compiled the icache shaders without logging their status; the icache probes that used them all logged `rc=0`. Its OpLoad counts ran as `glslc | grep | sed` pipelines whose logged status was the last command's; every count it logged is a non-zero number (5, 5, 5, 10, 12), so the compiler produced output each time. The current runner has no pipeline step: each OpLoad count compiles to an assembly file under `run()` and then counts from that file, so a compiler failure stops the run with the compiler's status; `test_run_bw.sh` covers that with a compiler that exits 7 in the count step.

## Short kernels pay for their code size (`bandwidth/bw.log`, icache section)

at 4 workgroups of 256 the unrolled kernels (SPIR-V 24, 93 and 369 KB) take 10.1-10.9, 32-35 and 126-128 microseconds per dispatch against 7.3, 11.9 and 39.5 for the looped ones (1.6 KB) doing the same work; at 160 workgroups the unrolled kernels are as fast as or faster than the looped ones (27-29 vs 29.5, 93 vs 103, 373 vs 395). The cost that grows with code size appears only when few workgroups run, which is the decode RoPE's case (one to four workgroups per dispatch).

This is why the half-split RoPE builds (#319) help: a decode-sized RoPE dispatch fetches its whole code body each time.

## Issue rates (`op.jsonl`)

fma 1234.5, fadd 1367.8, fmul 1368.1, iadd 1288.4 G lane-op/s; `ior` 613.5; the statements `(x & k) ^ c` 778.2, `(x >> 1) + k` and `(x << 1) + k` 495; imul 495.3; u32-to-float 465.2; the former nibble decode (shift, and, or, subtract) 415.0 (`op_rate.py`, 16 independent chains).

## Compatibility row

`docs/compatibility.md` and the `README.md` hardware table move M4 from "Not yet" to "Experimental", on the evidence above. The row names what runs and what it requires. This receipt covers the numerical gate (standing battery on hardware), the dispatch gate (trace) and device recovery (reopen after every compare set and at the end); it does not cover the install gate (there is no release wheel for this stack) or a stability run, so the row is Experimental, not Tested.

## Device reopen

`reopen.py` (a fresh process opens the GPU and checks a matmul and a 4-bit `quantized_matmul` against the CPU stream) passed after each bit-compare set, after the batteries and at the end of the speed runs (`reopen.log`); no new GPU driver message appeared in the kernel log during any run.

## Files

- `host.txt`, `battery.txt`, `trace-*.txt`, `op.jsonl`, `reopen.log`, `SHA256SUMS`.
- `bandwidth/`: `vkbw.c` (enables `storageBuffer16BitAccess` and refuses a shader that needs it on a device without it), `bw_read.comp`, `bw_gemv.comp` (element and row counts derived from the buffer size vkbw pushes, every loaded component consumed), `cpubw.c` (bounded 64-byte blocks plus tail), `vkchain.c`, `gen_icache.py`, `run_bw.sh`, `test_run_bw.sh`, `bw.log`.
