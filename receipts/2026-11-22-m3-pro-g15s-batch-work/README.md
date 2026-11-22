# M3 Pro (G15S) submission budget: 5000 to 15000

Date: 2026-11-22. This receipt backs the `{"G15S", 15000}` row of `kBatchWorkCalibration` in `overlay/mlx/backend/omarchy/encoder.h`. Before this change, `Apple M3 Pro (G15S B1)` matched no row in `kBatchWorkCalibration` and took `kBatchWorkFallback` (5000).

## Machine and stack

| Chip | Machine | GPU cores | Memory | OS | Kernel | Vulkan device |
|---|---|---|---|---|---|---|
| M3 Pro | MacBook Pro 14-inch, Nov 2023 (`apple,j514s` / `apple,t6030`), 11 CPU cores | 14 | 18 GB | Omarchy 4.5.0-3 on Arch Linux ARM | `7.2.11-14-1-ARCH` (`linux-aurora 7.2.11.aurora14-1`) | `Apple M3 Pro (G15S B1)` |

- Host: M3 Pro test host (single boot across all legs; governor `schedutil`, AC adapter `macsmc-ac` `online=1`).
- Vulkan driver: Mesa `26.3.0-devel (git-a196911147), Borealis, derived from Honeykrisp` (`mesa-aurora` / `vulkan-aurora` `1:26.3.0_devel.20261031.231483.a1969111475-1`), Vulkan `1.4.363`, driver ID 26 (`DRIVER_ID_MESA_HONEYKRISP`), vendor `0x10005`, device `0x6030`, system ICD `/usr/share/vulkan/icd.d/asahi_icd.aarch64.json`.
- GPU firmware: `macOS 14.7-14.8.x stub firmware (iBoot-10151.140.19.700.2)`, `version=0x7031100`, `core-mask=0x6f5bd` (14 active GPU cores; the 18-core M3 Pro part is not measured).
- ANE: not applicable (GPU-only change; no ANE stage ran).
- Python stack: Python 3.14.8, `mlx-lm==0.31.3` with `install.sh --patch-venv`, `transformers==5.16.1`, offline (`HF_HUB_OFFLINE=1`).
- Builds (upstream MLX `0.32.3`, commit `9c3d35571ac450a8ecf5c17b4d0e3fac52c08bc8`, Release, `-DMLX_OMARCHY_GPU_PROFILING=OFF`, fresh CMake cache per build):
  - Base (`origin/main` `9d6f4b188`): wheel `0.32.4.dev202611220202+9d6f4b188`, `libmlx.so` sha256 `a204d1b156bbd7f788ff42be1f69e7eba79f733fd966885eacdc7253d0d67242`, `core.cpython-314-aarch64-linux-gnu.so` sha256 `615f0eb19f8f7d178f79fafee02158c3ac33d1ad165f80d76c86cfb2b94682cc`.
  - Head (code commit `bb10aab9a`): wheel `0.32.4.dev202611220234+bb10aab9a`, `libmlx.so` sha256 `1570d19a9b204e5421abcd274d3e9747f2ba53be2a80c4f79714909ac6b20b9b`, `core.cpython-314-aarch64-linux-gnu.so` sha256 `615f0eb19f8f7d178f79fafee02158c3ac33d1ad165f80d76c86cfb2b94682cc`.
  - Full `scripts/mlx_provenance.py --expect-wheel` outputs are in `env/provenance.txt`; every measurement leg verified `verified=match` and `version_match=true`.

## Models and hashes

Full SHA-256 sums are in `env/model-hashes.txt`:

- `SiddhJagani/Qwen3.8-2B-mlx-4Bit` (`0867d98bfb174b042d88461c0e7c97b86b34b381`), 4-bit affine, group 64: `model.safetensors` `b0d5de688567bf4acd5e421027acd410dabcdc255a5bd46fdbf06c75dc2e6863`, `config.json` `6834d47a6e13a1ac135f27be825abfa0fa54b9e33980599a5575e2f298503a34`, `tokenizer.json` `87a7830d63fcf43bf241c3c5242e96e62dd3fdc29224ca26fed8ea333db72de4`.
- `mlx-community/Qwen3-4B-Instruct-2507-4bit` (`50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b`), 4-bit affine, group 64: `model.safetensors` `2a73c6c248601ab904e035548abd8e6abb65ea27dcb5f342fb0a8910eb44173f`, `config.json` `574349e5a343236546fda55e4744a76e181f534182d7dc60ff1bad7e7a502849`, `tokenizer.json` `aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4`.
- `receipts/2026-10-09-linux-vs-macos/prompts.jsonl`: `9299a3b2fc13...` (exact hash in `env/model-hashes.txt`).

## Verdict

**15000.** Following `encoder.h` and the G16S (`receipts/2026-11-15-m4-pro-g16s-batch-work`) and G13G (`receipts/2026-11-21-g13g-batch-work-20000`) release-build DRM scheduler method, 15000 is the largest swept budget that:

1. Keeps Qwen3-4B-Instruct-2507 decode submission p50 inside the 2-6 ms smooth band (**3.98 ms**, p99 **4.31 ms**, max **4.62 ms**; Qwen3.8-2B decode p50 **3.09 ms**, p99 **3.84 ms**, max **3.91 ms**).
2. Keeps the all-submissions (prefill + decode) tail at the 5000 single-dispatch prefill floor on Qwen3-4B (**45.17 ms** p99 / **45.52 ms** max against 5000's **44.80 / 44.89 ms**) and strictly under 40 ms on Qwen3.8-2B (**38.42 ms** p99 / **38.51 ms** max, 0 submissions over 40 ms), with zero submissions near the ~78 ms stutter threshold of issue #19.

Higher budgets violate the tail or smoothness rules:
- **20000** merges prefill dispatches and raises Qwen3-4B all-submissions p99/max by +20 ms to **65.14 / 65.22 ms**, and pushes Qwen3.8-2B past 40 ms (**43.85 / 44.46 ms**, 23 submissions over 40 ms) for +0.7% extra decode.
- **30000** breaks both the 6 ms decode p50 band (**7.40 ms**) and the 78 ms stutter threshold (**80.08 ms** max).
- **10000** merges a neighboring dispatch into prefill (**56.61 / 57.60 ms** on 4B, **41.72 / 41.85 ms** on 2B), worse on both throughput and tail than 15000 (the same split-boundary effect observed on G16S in PR #454).

## Release-build submission GPU time (DRM scheduler ftrace)

Measured on the base Release wheel (`0.32.4.dev202611220202+9d6f4b188`, `verified=match`, `libmlx.so=sha256:a204d1b156bbd7f7`) using read-only kernel tracepoints `gpu_scheduler:drm_sched_job_{queue,run,done}` (`tools/decode_jobs.py` + `tools/parse_ftrace.py`, 8-token warmup, then ~540-token prefill + 64 greedy decode tokens per budget). `busy = done - max(run, prev_done)` in ms. Zero foreign GPU jobs appeared in any window (`all_jobs_in_window == python_jobs_in_window`). Raw ftrace files are excluded from the repository because tracefs records all host processes; parsed per-budget metrics are in `runs/submission-gpu-time.json`.

Provenance: `mlx-omarchy 0.32.4.dev202611220202+9d6f4b188 mx=0.32.4.dev202611220202+9d6f4b188 verified=match harness=9d6f4b188 core.cpython-314-aarch64-linux-gnu.so=sha256:615f0eb19f8f7d17 libmlx.so=sha256:a204d1b156bbd7f7`

### Qwen3-4B-Instruct-2507 4-bit

| Budget | Decode n | Decode p50 / p99 / max (ms) | All (prefill+decode) n | All p50 / p99 / max (ms) | Greedy ids sha256 |
|---:|---:|---|---:|---|---|
| 5000 (default) | 1260 | 1.39 / 1.81 / 4.91 | 1513 | 1.39 / 44.80 / 44.89 | `7df47e123edd716c` |
| 8000 | 882 | 2.04 / 2.10 / 2.14 | 1087 | 2.04 / 44.84 / 44.95 | `7df47e123edd716c` |
| 10000 | 693 | 2.69 / 2.82 / 6.42 | 892 | 2.69 / 56.61 / 57.60 | `7df47e123edd716c` |
| **15000** | **504** | **3.98 / 4.31 / 4.62** | **662** | **3.98 / 45.17 / 45.52** | `7df47e123edd716c` |
| 20000 | 378 | 4.93 / 5.71 / 6.20 | 496 | 4.95 / 65.14 / 65.22 | `7df47e123edd716c` |
| 30000 | 252 | 7.40 / 7.71 / 8.35 | 332 | 7.40 / 77.07 / 80.08 | `7df47e123edd716c` |
| 40000 | 189 | 9.81 / 10.65 / 11.14 | 267 | 9.81 / 77.81 / 78.48 | `7df47e123edd716c` |
| 60000 | 126 | 12.41 / 16.74 / 16.75 | 168 | 14.47 / 81.66 / 82.37 | `7df47e123edd716c` |
| 80000 | 126 | 6.21 / 20.15 / 20.76 | 168 | 19.43 / 81.59 / 81.68 | `7df47e123edd716c` |
| 0 (cap off) | 63 | 25.21 / 25.45 / 26.95 | 79 | 25.23 / 244.36 / 244.47 | `7df47e123edd716c` |

### Qwen3.8-2B 4-bit

| Budget | Decode n | Decode p50 / p99 / max (ms) | All (prefill+decode) n | All p50 / p99 / max (ms) | Greedy ids sha256 |
|---:|---:|---|---:|---|---|
| 5000 (default) | 693 | 1.05 / 2.52 / 2.70 | 855 | 1.06 / 21.99 / 22.62 | `3bd0eae30cd94543` |
| 8000 | 441 | 1.64 / 3.38 / 3.48 | 585 | 1.65 / 32.21 / 32.29 | `3bd0eae30cd94543` |
| 10000 | 378 | 2.09 / 2.94 / 3.04 | 497 | 2.09 / 41.72 / 41.85 | `3bd0eae30cd94543` |
| **15000** | **252** | **3.09 / 3.84 / 3.91** | **348** | **3.09 / 38.42 / 38.51** | `3bd0eae30cd94543` |
| 20000 | 189 | 3.91 / 5.09 / 5.56 | 260 | 4.43 / 43.85 / 44.46 | `3bd0eae30cd94543` |
| 30000 | 126 | 6.00 / 6.87 / 7.01 | 184 | 6.68 / 63.51 / 63.63 | `3bd0eae30cd94543` |
| 40000 | 126 | 5.25 / 7.53 / 7.78 | 169 | 7.46 / 63.90 / 64.01 | `3bd0eae30cd94543` |
| 60000 | 63 | 12.38 / 12.91 / 13.39 | 94 | 12.39 / 68.04 / 68.07 | `3bd0eae30cd94543` |
| 80000 | 63 | 12.35 / 12.86 / 13.40 | 88 | 12.36 / 81.06 / 81.27 | `3bd0eae30cd94543` |
| 0 (cap off) | 63 | 12.38 / 12.91 / 13.46 | 72 | 12.38 / 384.00 / 412.78 | `3bd0eae30cd94543` |

## Balanced-order decode sweep (same base Release build)

`tools/decode_sub.py` on the base Release wheel (`0.32.4.dev202611220202+9d6f4b188`, `verified=match`), 8-token warmup, then 128 greedy tokens (Qwen3.8-2B) and 96 greedy tokens (Qwen3-4B-2507), 3 balanced rounds per model (forward, reverse, forward; 30 fresh processes per model). Raw records: `runs/decode-sweep-2b.json`, `runs/decode-sweep-4b.json`.

| Budget | Qwen3.8-2B median tok/s | vs 5000 | 2B submits/tok | Qwen3-4B median tok/s | vs 5000 | 4B submits/tok |
|---:|---:|---:|---:|---:|---:|---:|
| 5000 (default) | 73.42 | 0.0% | 12.0 | 36.37 | 0.0% | 20.0 |
| 8000 | 75.83 | +3.3% | 8.0 | 37.31 | +2.6% | 14.0 |
| 10000 | 76.71 | +4.5% | 6.0 | 37.69 | +3.6% | 11.0 |
| **15000** | **78.23** | **+6.6%** | **4.0** | **38.31** | **+5.3%** | **8.0** |
| 20000 | 78.78 | +7.3% | 3.0 | 38.59 | +6.1% | 6.0 |
| 30000 | 79.73 | +8.6% | 2.0 | 38.85 | +6.8% | 4.0 |
| 40000 | 79.71 | +8.6% | 2.0 | 39.04 | +7.4% | 3.0 |
| 60000 | 80.38 | +9.5% | 1.0 | 39.25 | +7.9% | 2.0 |
| 80000 | 80.50 | +9.6% | 1.0 | 39.27 | +8.0% | 2.0 |
| 0 (cap off) | 80.03 | +9.0% | 1.0 | 39.45 | +8.5% | 1.0 |

Greedy token IDs are identical across all 30 runs of Qwen3.8-2B (`cdf6b6529b0aa39f`, 235.0 compute dispatches/token) and all 30 runs of Qwen3-4B-2507 (`31e0cd14a8057dfd`, 333.0 compute dispatches/token).

## `score_bench` A/B (5000 vs 15000, balanced rounds)

`receipts/2026-10-09-linux-vs-macos/score_bench.py` (2 warmups, 10 greedy 64-token generations, 5 prefills of 512 tokens) in balanced order (`default, 15000, 15000, default`) on the base Release wheel (`+9d6f4b188`, `verified=match`), plus one default-unset confirmation on the head Release wheel (`+bb10aab9a`, `verified=match`). Records: `runs/score-bench-ab.json`.

| Model | Arm / Wheel | Decode median tok/s | vs 5000 | Prefill 512 tok/s | Digest |
|---|---|---:|---:|---:|---|
| Qwen3.8-2B 4-bit | Base `+9d6f4b188` (default 5000, 2 rounds) | 76.21 / 76.04 (mean 76.12) | 0.0% | 370.36 / 370.83 | `912ebb461fda3c39` |
| Qwen3.8-2B 4-bit | Base `+9d6f4b188` (`BATCH_WORK=15000`, 2 rounds) | 81.19 / 81.00 (mean 81.09) | +6.5% | 372.33 / 372.08 | `912ebb461fda3c39` |
| Qwen3.8-2B 4-bit | Head `+bb10aab9a` (default unset -> 15000) | 81.16 (80.52-81.52) | +6.6% | 371.93 | `912ebb461fda3c39` |
| Qwen3-4B-2507 4-bit | Base `+9d6f4b188` (default 5000, 2 rounds) | 44.23 / 44.24 (mean 44.23) | 0.0% | 195.38 / 195.44 | `67c191eeffa36bf9` |
| Qwen3-4B-2507 4-bit | Base `+9d6f4b188` (`BATCH_WORK=15000`, 2 rounds) | 47.07 / 47.11 (mean 47.09) | +6.5% | 196.02 / 195.93 | `67c191eeffa36bf9` |
| Qwen3-4B-2507 4-bit | Head `+bb10aab9a` (default unset -> 15000) | 47.08 (47.02-47.26) | +6.4% | 196.08 | `67c191eeffa36bf9` |

## Two-build head-vs-base A/B (`MLX_OMARCHY_BATCH_WORK` unset)

Interleaved pairs (3 pairs per model, fresh process per run) comparing base Release wheel `0.32.4.dev202611220202+9d6f4b188` (fallback 5000) against head Release wheel `0.32.4.dev202611220234+bb10aab9a` (built-in `G15S` row 15000), both with `MLX_OMARCHY_BATCH_WORK` unset. Wheel stamps differ and every run verified `verified=match` (`runs/head-vs-base-ab.json`).

- **Qwen3.8-2B (128 tok)**: base `73.37 / 73.44 / 73.66` (median **73.44 tok/s**, 12.0 submits/tok) -> head `78.23 / 78.15 / 78.24` (median **78.23 tok/s**, **+6.52%**, 3/3 pairs, 4.0 submits/tok), greedy ids identical (`cdf6b6529b0aa39f`).
- **Qwen3-4B-2507 (96 tok)**: base `36.36 / 36.35 / 36.52` (median **36.36 tok/s**, 20.0 submits/tok) -> head `38.25 / 38.21 / 38.21` (median **38.21 tok/s**, **+5.10%**, 3/3 pairs, 8.0 submits/tok), greedy ids identical (`31e0cd14a8057dfd`).

## Standing battery on head (`bb10aab9a`)

Fresh Release build of `bb10aab9a` (`suite/summary.txt`):
- All **25/25** standing suites pass (**747/747** cases). In `omarchy_fast_ops_tests`, the 4 failed assertions (1,384,011/1,384,015) are inside the documented `doctest::may_fail` SDPA VJP `rep=1` case (`docs/known-defects.md`).
- All **8/8** `omarchy_capability_sim_tests` profiles pass (**88/88** cases).
- Unit tests in `overlay/tests/omarchy/test_runtime.cpp`:
  - Before the `G15S` row in `encoder.h`: `"submission-cap default resolves per calibrated board"` failed `CHECK( 0 == 15000 )` (28/29 assertions) and `"work budget resolves per device name, not per process"` failed `CHECK( 5000 == 15000 )` (8/9 assertions).
  - With the row (`bb10aab9a`): 30/30 and 9/9 assertions pass (`omarchy_runtime_tests` total: 74/74 cases, 52,821/52,821 assertions).

## Backend dispatch trace (head wheel `+bb10aab9a`, `MLX_OMARCHY_BATCH_WORK` unset)

Run under `receipts/2026-09-30-speech-input-gpu/harness/count_cpu.gdb.py` and `receipts/2026-11-21-m3-pro-g15s-linux/trace/trace_decode.py` (`trace/summary.json`):

| Arm | CPU encoder calls | GPU primitive dispatches | Vulkan compute dispatches | Vulkan submissions |
|---|---:|---:|---:|---:|
| control (`mx.cpu`) | 3 | | | |
| Qwen3.8-2B 4-bit, prefill + 32 decode steps | 0 | 25383 | 10374 | 629 |
| Qwen3-4B-2507 4-bit, prefill + 32 decode steps | 0 | 36228 | 11701 | 278 |

The breakpoint resolved in all three arms (`resolved_while_running: true`), proving zero CPU tensor dispatches.

## Timing, thermal, and device reopen record

Every GPU leg ran serialized under `flock /tmp/m1-gpu.lock` with a bounded foreground timeout (`timeout --foreground --kill-after=30s`), recording load average, memory, the four exposed SMC temperatures (`hwmon7`), governor (`schedutil`), AC online state (`1`), `vulkaninfo --summary`, and kernel GPU log counts before and after (`runs/timing-thermal.json`). Across all 148 GPU legs:
- **0 new GPU kernel log lines** (boot baseline 36 lines unchanged).
- `env/info-before.json` and `env/info-after.json` are identical.
- Final reopen checks (`reopen.txt`):
  - `omarchy_runtime_tests --test-case='a fresh process reopens the Vulkan device'`: 1/1 (`fresh process device reopen ok`).
  - `mlx-omarchy-info --trace-smoke`: `buffer round trip: OK (65536 bytes, device Apple M3 Pro (G15S B1))`.
  - Fresh head-wheel process 32x32 matmul sum on `Device(gpu, 0)`: `524800.0 True`.
