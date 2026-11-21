# M3 Pro (G15S) on Linux with the Borealis driver

Date: 2026-11-21. These runs back the M3 Pro rows in `README.md`, `docs/compatibility.md`, `docs/BENCHMARKS.md`, `docs/TESTED-HARDWARE.md` and the `t6030_measured` values in `docs/chip-capability-axes.json`. One machine, the M3 Pro test host, ran everything below on one boot.

## Machine and stack

| Chip | Machine | GPU cores | Memory | OS | Kernel | Vulkan device |
|---|---|---|---|---|---|---|
| M3 Pro | MacBook Pro 14-inch, Nov 2023 (`apple,j514s` / `apple,t6030`), 11 CPU cores | 14 | 18 GB | Omarchy 4.5.0-3 on Arch Linux ARM | 7.2.11-14-1-ARCH, `linux-aurora 7.2.11.aurora14-1` | `Apple M3 Pro (G15S B1)` |

Driver: Mesa 26.3.0-devel Borealis (git-a196911147), packages `mesa-aurora` and `vulkan-aurora` `1:26.3.0_devel.20261031.231483.a1969111475-1`. It reports Vulkan 1.4.363, driver version 26.2.99, vendor 0x10005, device 0x6030 and driver ID 26 (`DRIVER_ID_MESA_HONEYKRISP`), driver name `Borealis`. The ICD is the system `asahi_icd.aarch64.json`. `omarchy-mlx-vulkan` is not installed, and `VK_DRIVER_FILES`, `VK_ICD_FILENAMES` and `MLX_OMARCHY_ALLOW_NON_APPLE` are unset for every leg except the capability-simulation profiles, which run with `MLX_OMARCHY_ALLOW_NON_APPLE=1` as their CTest registration sets it.

GPU firmware: the kernel GPU driver logs `GPU firmware build: macOS 14.7-14.8.x stub firmware (iBoot-10151.140.19.700.2)`, `version=0x7031100` and `core-mask=0x6f5bd` at boot. That mask has 14 bits set: 14 active GPU cores. The M3 Pro also ships with 18 GPU cores; only the 14-core part was measured. Governor `schedutil`, on AC power. `env/m3pro.txt` has the boot lines, the `vulkaninfo` identity and capability fields, the devicetree compatible, memory and toolchain.

Capabilities the backend reads (`env/m3pro.txt`, `env/info-before.json`): subgroup size 32 (min and max 32) with BASIC, VOTE, ARITHMETIC, BALLOT, SHUFFLE, SHUFFLE_RELATIVE, CLUSTERED, QUAD, ROTATE and ROTATE_CLUSTERED; compute shared memory 32768 bytes; workgroup 1024 invocations, size 1024 x 1024 x 64; f16, i16 and 16-bit storage; two memory types on heap 0, type 0 device-local, host-visible, coherent and cached. `VK_KHR_cooperative_matrix`, `VK_EXT_shader_atomic_float` and `VK_KHR_shader_bfloat16` are not listed, so the cooperative-matrix and bf16 direct routes stay off and the tile and scalar kernels serve those shapes.

## Standing battery

Source build of main `a893cae5c` (MLX 0.32.3, upstream `9c3d355`), prepared with `scripts/prepare-mlx.sh` and configured as in `tools/ci/run-omarchy-runtime.sh` with `CMAKE_BUILD_TYPE=Release` and `MLX_BUILD_TESTS=ON`:

```bash
cmake -S .work/mlx -B .work/mlx/build-m3pro-a893cae5c -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DMLX_BUILD_OMARCHY=ON -DMLX_BUILD_CPU=ON -DMLX_BUILD_METAL=OFF -DMLX_BUILD_CUDA=OFF \
  -DMLX_BUILD_TESTS=ON -DMLX_BUILD_EXAMPLES=OFF -DMLX_BUILD_BENCHMARKS=OFF \
  -DMLX_BUILD_PYTHON_BINDINGS=OFF -DBLAS_INCLUDE_DIRS=/usr/include/openblas
cmake --build .work/mlx/build-m3pro-a893cae5c --parallel 8 --target \
  omarchy_<suite>_tests ... omarchy_capability_sim_tests mlx-omarchy-info
```

Arch's `openblas` package installs `cblas.h` under `/usr/include/openblas`, so the configure needs `BLAS_INCLUDE_DIRS`. The targets are the 25 standing suites in `AGENTS.md` § Test rules, the capability-simulation binary and `mlx-omarchy-info`.

Each suite ran alone, in the order below, under a host GPU lock and an 1800 s timeout, with the kernel GPU log and the Vulkan identity read before and after. Logs are in `suite/`; `suite/summary.txt` has the doctest totals.

| Suite | Cases | Assertions |
|---|---|---|
| `omarchy_runtime_tests` | 74/74 | 52814/52814 |
| `omarchy_primitive_tests` | 107/107 | 2768377/2768377 |
| `omarchy_matmul_family_tests` | 58/58 | 21999845/21999845 |
| `omarchy_fast_ops_tests` | 52/52 | 1384011/1384015 |
| `omarchy_kv_ops_tests` | 16/16 | 1146/1146 |
| `omarchy_indexing_ops_tests` | 60/60 | 35938/35938 |
| `omarchy_reduce_ops_tests` | 36/36 | 7216/7216 |
| `omarchy_shape_ops_tests` | 25/25 | 898/898 |
| `omarchy_linalg_ops_tests` | 31/31 | 181183/181183 |
| `omarchy_copy_offset_tests` | 29/29 | 131711/131711 |
| `omarchy_distributed_tests` | 9/9 | 36/36 |
| `omarchy_compiled_tape_tests` | 14/14 | 46224/46224 |
| `omarchy_fft_ops_tests` | 19/19 | 1379/1379 |
| `omarchy_fft_general_tests` | 14/14 | 4245/4245 |
| `omarchy_eig_ops_tests` | 9/9 | 264/264 |
| `omarchy_take_fill_tests` | 9/9 | 9054/9054 |
| `omarchy_conv_tests` | 15/15 | 3749/3749 |
| `omarchy_complex_ops_tests` | 34/34 | 1715/1715 |
| `omarchy_select_layout_tests` | 13/13 | 4500/4500 |
| `omarchy_fast_regression_tests` | 2/2 | 16/16 |
| `omarchy_scatter_determinism_tests` | 21/21 | 127/127 |
| `omarchy_eq_math_tests` | 7/7 | 128/128 |
| `omarchy_fused_chain_tests` | 42/42 | 379399/379399 |
| `omarchy_error_contract_tests` | 3/3 | 14/14 |
| `omarchy_ane_bundle_tests` | 48/48 | 8215/8215 |

747 of 747 cases pass. The four failed assertions in `omarchy_fast_ops_tests` are inside `sdpa vjp dk dv match finite differences at rep=1 (path per gate)`, which is marked `doctest::may_fail` for the open defect in [known-defects.md](../../docs/known-defects.md#sdpa-backward-fused-vjp-value-defects-at-small-rep1-shapes-2026-10-03-open). They are that entry's signatures: `dv[104]` at shape 5x7 (twice) and `dk[28]` and `dv[28]` at shape 4x4, 0.48 to 0.59 of the finite-difference value (`suite/S04-fast_ops.txt`). The `SUBMIT-RECOVER` and `failed to advance` lines in `suite/S01-runtime.txt` come from the child processes that test the stall paths. `omarchy_ane_bundle_tests` parses and validates ANE bundle files and opens no ANE device.

Capability-simulation profiles, `omarchy_capability_sim_tests <profile>` on the real device:

| Profile | Cases | Assertions |
|---|---|---|
| `m1-honeykrisp-fork` | 11/11 | 2128/2128 |
| `m1-stock-no-coopmat` | 11/11 | 6624/6624 |
| `subgroup-size-64` | 11/11 | 6594/6594 |
| `small-shared-memory` | 11/11 | 2252/2252 |
| `no-cooperative-matrix` | 11/11 | 6596/6596 |
| `m1-g13-legacy` | 11/11 | 6625/6625 |
| `workgroup-256` | 11/11 | 6626/6626 |
| `workgroup-128` | 11/11 | 6626/6626 |

## Install

Release v0.7.49 (tag commit `bf5631dc34b2`), wheel `mlx_omarchy-0.32.4.dev202611210550+bf5631d-cp314-cp314-linux_aarch64.whl`, sha256 `d41df12441b34b4e90b27283ea52034e2d53a94db2a54119acfbbf38c4caac75`, equal to the release `SHA256SUMS` entry. The install follows [From a release wheel](../../docs/INSTALL.md#from-a-release-wheel) in a fresh Python 3.14.8 venv: the wheel, `mlx-lm==0.31.3 --no-deps`, `transformers[sentencepiece]==5.16.1 numpy protobuf pyyaml jinja2 huggingface_hub soundfile`, then `install.sh --patch-venv`. `env/venv-freeze.txt` has the resolved packages.

`mx.default_device()` is `Device(gpu, 0)`. `scripts/mlx_provenance.py --expect-wheel` reports `verified: match` and `version_match: true` for `mlx/lib/libmlx.so` and the `mlx.core` extension (`env/wheel-provenance.txt`); every timed leg below prints the same check first in its own log.

## Correctness probes

`scripts/collect_deep.py --skip profile` (exact command in `commands.txt`) from the checkout at `a893cae5c` against the v0.7.49 wheel, with nothing submitted. `collector/` has the correctness, benchmark and environment sections.

| Probe ops | Matmul fp32 median ms (TFLOPS), n = 256 / 512 / 1024 |
|---|---|
| 6/6 pass (add, matmul, softmax, argmax, cumsum, value_and_grad) | 0.157 (0.2143) / 0.332 (0.8074) / 1.836 (1.1694) |

The community dataset holds an earlier capture from the same machine and wheel, record (archive sha256) [`b5095c3b8e1a`](https://mlx-omarchy-community-data.joshua-s-warren.workers.dev/v1/results/b5095c3b8e1a9ca0131a89262694d92be3faa1016ff1bbed591d53b8c01f1008): 6/6 probe ops, matmul medians 0.157 / 0.339 / 1.828 ms.

## LLM decode and prefill

`receipts/2026-10-09-linux-vs-macos/score_bench.py` with that receipt's `prompts.jsonl` (sha256 `9299a3b2fc13...`), the procedure of [the M1 Pro, M2, M4 Pro and A18 Pro receipt](../2026-11-13-m1pro-m2-m4pro-a18pro-linux/README.md#llm-decode-and-prefill): 2 warmup generations, then 10 greedy generations of 64 tokens (decode median, time to first token excluded) and 5 prefills of 512 tokens (median). Each model ran in its own process, the 2B model first, with release defaults (submission budget 5000). Load average 0.38 before the 2B run and 0.43 before the 4B run. `bench/` has the JSON, the run log with its provenance check, and the load and temperatures before and after each run. The kernel exposes no SoC or GPU temperature sensor on this machine; the four SMC sensors it does expose are recorded.

| Model | Decode tok/s (min to max) | Prefill 512 tok/s | Time to first token | Peak memory | Digest |
|---|---:|---:|---:|---:|---|
| Qwen3.8-2B 4-bit | 76.14 (75.60 to 76.65) | 370.49 | 0.169 s | 1.49 GB | `912ebb46` |
| Qwen3-4B-Instruct-2507 4-bit | 43.94 (43.38 to 44.13) | 195.29 | 0.374 s | 2.54 GB | `67c191ee` |

Models, from Hugging Face at these revisions (hashes in `env/model-hashes.txt`):

- `SiddhJagani/Qwen3.8-2B-mlx-4Bit` `0867d98bfb174b042d88461c0e7c97b86b34b381`, 4-bit affine, group 64; `model.safetensors` sha256 `b0d5de688567bf4acd5e421027acd410dabcdc255a5bd46fdbf06c75dc2e6863`, `config.json` `6834d47a6e13a1ac135f27be825abfa0fa54b9e33980599a5575e2f298503a34`, `tokenizer.json` `87a7830d63fcf43bf241c3c5242e96e62dd3fdc29224ca26fed8ea333db72de4`.
- `mlx-community/Qwen3-4B-Instruct-2507-4bit` `50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b`, 4-bit, group 64; `model.safetensors` sha256 `2a73c6c248601ab904e035548abd8e6abb65ea27dcb5f342fb0a8910eb44173f`, `config.json` `574349e5a343236546fda55e4744a76e181f534182d7dc60ff1bad7e7a502849`, `tokenizer.json` `aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4`.

Both output digests equal the M4 Pro and A18 Pro digests in the 2026-11-13 receipt (Borealis `c208afc352`).

## Dispatch trace

`trace/trace_decode.py` loads a model from the wheel venv, generates 32 greedy tokens and prints the backend trace counter deltas around the generation. The deltas cover the prefill and the decode steps; mlx-lm queues one decode step ahead, so they include the step after token 32. Each arm ran under `receipts/2026-09-30-speech-input-gpu/harness/count_cpu.gdb.py`, which counts calls to `mlx::core::cpu::get_command_encoder(Stream)`, the entry every CPU-stream primitive takes, in the same process.

| Arm | CPU encoder calls | GPU primitive dispatches | Vulkan compute dispatches | Vulkan submissions |
|---|---:|---:|---:|---:|
| control, `mx.add(a, a, stream=mx.cpu)` | 3 | | | |
| Qwen3.8-2B 4-bit, prefill and 32 decode steps | 0 | 25383 | 10373 | 1027 |
| Qwen3-4B-Instruct-2507 4-bit, prefill and 32 decode steps | 0 | 36228 | 11701 | 697 |

The breakpoint resolved in every arm (`resolved_while_running: true`), and the control shows the trace sees a CPU dispatch. Both decodes ran on `Device(gpu, 0)` with zero CPU tensor dispatches. `trace/` has each arm's provenance line, breakpoint and counters.

## Device state and reopen

The kernel logged 36 GPU driver lines at boot and none during or after any leg: the battery, the profiles, the collector, the benchmarks and the trace (checked after each one). After the last leg (`reopen.txt`):

- `omarchy_runtime_tests --test-case='a fresh process reopens the Vulkan device'`: 1/1, the child reports `fresh process device reopen ok`.
- `mlx-omarchy-info --trace-smoke`: `buffer round trip: OK (65536 bytes, device Apple M3 Pro (G15S B1))`.
- A fresh wheel process: a 32 x 32 matmul and sum on `Device(gpu, 0)` returns 524800.0, the exact value.
- `mlx-omarchy-info --json` before the first leg and after the last (`env/info-before.json`, `env/info-after.json`) report the same device, driver, ICD and limits.

## Files

- `commands.txt`: the exact command and timeout of every GPU leg, in run order.
- `env/`: `m3pro.txt`, `info-before.json`, `info-after.json`, `wheel-provenance.txt`, `venv-freeze.txt`, `model-hashes.txt`.
- `suite/`: one log per suite and profile, each headed by its source build, start time, load, exit code and new kernel GPU lines; `summary.txt`.
- `collector/`: correctness, benchmark and environment sections.
- `bench/`: per-model JSON, run log and load file.
- `trace/`: `trace_decode.py` and one file per arm.
- `reopen.txt`, `SHA256SUMS`.
