# Fused f16 decode attention on the M3 Pro (G15S)

Date: 2026-11-22. One machine, the M3 Pro test host, one boot. This receipt backs the G15S entry of `sdpa_decode_f16_without_coopmat_part` in `overlay/mlx/backend/omarchy/device.h`, its `docs/kernel-flags.md` row and the f16 line of the M3 Pro entry in `docs/compatibility.md`.

## Result

The fused f16 single-query decode attention kernel (`sdpa_decode_native*.comp`, head width 64 and 128) uses no cooperative matrix. Its gate asks for one anyway, except on the parts `sdpa_decode_f16_without_coopmat_part` names; until this change that was G16S only, so the M3 Pro composed every f16 decode attention call from three dispatches. G15S meets every other clause of the gate, and with the kernel on:

- per call, 16 query and 8 key-value heads at width 128 (the G16S receipt's shape): 545.0 to 92.8 microseconds at 1100 keys and 2016.1 to 253.2 at 4096 keys; 2.7x to 13.1x faster over all 13 shapes measured;
- end to end, Qwen3-4B-Instruct-2507 4-bit in f16, 128 greedy tokens: decode 37.24 to 44.95 tok/s at 64 context tokens (+20.7%), 16.06 to 36.36 at 1100 (+126%) and 8.03 to 30.71 at 3000 (+282%) at the release submission budget (`MLX_OMARCHY_BATCH_WORK` unset, 5000); at 15000: +20.7%, +136%, +306%;
- token ids identical in every process of both arms; every output within 4.9e-4 of the composition (the test gate is 0.01).

bf16 calls never read this gate (`primitives.cpp`: the gate only selects the f16 arm), so bf16 decode keeps its route on every part.

## Machine and stack (`env/host.txt`)

| Chip | Machine | GPU cores | Memory | Kernel | Vulkan device |
|---|---|---|---|---|---|
| M3 Pro | MacBook Pro 14-inch, Nov 2023 (`apple,j514s` / `apple,t6030`), 11 CPU cores | 14 (core mask `0x6f5bd`) | 18 GB | 7.2.11-14-1-ARCH, `linux-aurora 7.2.11.aurora14-1` | `Apple M3 Pro (G15S B1)` |

Driver: Mesa 26.3.0-devel Borealis (git-a196911147), packages `mesa-aurora` and `vulkan-aurora` `1:26.3.0_devel.20261031.231483.a1969111475-1`, Vulkan 1.4.363, driver ID 26, system ICD. GPU firmware: `macOS 14.7-14.8.x stub firmware (iBoot-10151.140.19.700.2)`, `version=0x7031100`. Governor `schedutil`, AC online. No ANE stage runs here.

The gate, as the device reports it: subgroup size 32 (min and max 32) with BASIC, ARITHMETIC and SHUFFLE among its operations; 1024 workgroup invocations and 1024 in x; 32768 bytes of shared memory against the 21,504 the kernel needs; no `VK_KHR_cooperative_matrix`.

## Builds

- Base: release wheel of `main` `52ac565fc`, `mlx_omarchy-0.32.4.dev202611220252+52ac565fc`, `libmlx.so` sha256 `107b3cd43413…`. `git diff --stat 52ac565fc 79d99c90c` lists only `tests/test_paths_module.py`, so this is the backend of `main` `79d99c90c`, the base of this change. The per-call and end-to-end A/B ran on this wheel, switching the kernel with `MLX_OMARCHY_SDPA_F16_NO_COOPMAT` (`0` composition, `1` fused; read live), one arm per process.
- Head: code commit `4e3b83d2a`, release wheel `mlx_omarchy-0.32.4.dev202611220450+4e3b83d2a` (`libmlx.so` sha256 `10345795df2c…`), and a CPU-disabled wheel of the same commit (below); C++ test tree from the same prepared source, Release, fresh build directory.

Every process printed the `scripts/mlx_provenance.py` line and refused to run unless the installed wheel's RECORD hashes matched and its version stamp matched the wheel it was meant to be. The runs are in `percall/`, `e2e/` and `head/` with that line in each JSON. The `harness=` field reads `-dirty` on the A/B runs because the source tree held this change's uncommitted edits while the base wheel ran; the wheel is the same file in every one of them (same `libmlx.so` hash).

## Per call (`percall/`)

`scripts/sdpa_f16_ab.py`: q (1, H, 1, D), k and v (1, KV, L, D) f16, fixed seeds, scale D^-0.5. Independent: host wall time per call of 200 calls of SDPA plus the query scaling in one eval (the timing of `receipts/2026-11-01-m4-sdpa-decode-bf16-256/scripts/sdpa_bits.py`). Chain: 36 dependent calls in one eval, each query the previous output. Seven repetitions each, median; three interleaved process pairs (off on on off off on), median of the process medians. Compute dispatches per call come from the backend trace counter. Budget unset. 1-minute load before each process 0.36 to 0.64, except two composition processes at 1.20 and 1.17 (`B04`, `B05`); their medians are within 1 percent of the third composition process and of the base-wheel default run `head/D01` (load 0.50).

| q heads | kv heads | width | keys | dispatches off / on | independent µs off → on | chain µs off → on | max \|on − off\| | float64 error off / on |
|---:|---:|---:|---:|---|---|---|---:|---|
| 16 | 8 | 128 | 37 | 3 / 1 | 42.3 →   15.5 (2.73) | 50.6 →   20.2 (2.50) | 4.88e-04 | 3.10e-04 / 2.43e-04 |
| 16 | 8 | 128 | 263 | 3 / 1 | 147.7 →   20.8 (7.10) | 168.3 →   33.8 (4.98) | 1.22e-04 | 1.12e-04 / 1.12e-04 |
| 16 | 8 | 128 | 1100 | 3 / 2 | 545.0 →   92.8 (5.87) | 590.8 →   97.7 (6.05) | 6.10e-05 | 4.72e-05 / 4.11e-05 |
| 16 | 8 | 128 | 4096 | 3 / 2 | 2016.1 →  253.2 (7.96) | 2098.9 →  279.7 (7.50) | 3.05e-05 | 2.55e-05 / 2.44e-05 |
| 32 | 8 | 128 | 64 | 3 / 1 | 86.2 →   17.7 (4.87) | 95.3 →   25.5 (3.74) | 2.44e-04 | 1.92e-04 / 1.68e-04 |
| 32 | 8 | 128 | 263 | 3 / 1 | 268.1 →   29.2 (9.18) | 284.1 →   44.4 (6.40) | 1.22e-04 | 1.02e-04 / 1.02e-04 |
| 32 | 8 | 128 | 1024 | 3 / 2 | 942.9 →  122.9 (7.67) | 977.4 →  129.8 (7.53) | 6.10e-05 | 5.58e-05 / 5.58e-05 |
| 32 | 8 | 128 | 1100 | 3 / 2 | 1012.5 →  149.2 (6.79) | 1047.4 →  151.9 (6.90) | 6.10e-05 | 5.41e-05 / 4.58e-05 |
| 32 | 8 | 128 | 2048 | 3 / 2 | 1877.6 →  215.9 (8.70) | 1934.2 →  222.0 (8.71) | 6.10e-05 | 3.54e-05 / 3.95e-05 |
| 32 | 8 | 128 | 4096 | 3 / 2 | 3886.1 →  376.3 (10.33) | 3879.6 →  392.1 (9.89) | 3.05e-05 | 2.44e-05 / 2.28e-05 |
| 32 | 8 | 64 | 263 | 3 / 1 | 160.7 →   17.0 (9.45) | 178.0 →   24.6 (7.24) | 1.22e-04 | 9.14e-05 / 6.91e-05 |
| 32 | 8 | 64 | 1100 | 3 / 2 | 595.4 →  101.5 (5.87) | 636.3 →  102.5 (6.21) | 6.10e-05 | 4.75e-05 / 3.89e-05 |
| 32 | 8 | 64 | 4096 | 3 / 2 | 2300.6 →  175.6 (13.10) | 2299.7 →  177.5 (12.96) | 3.05e-05 | 2.53e-05 / 2.48e-05 |

Every element of every shape: |fused - composition| at most 4.9e-4, inside the 0.01 tolerance `test_sdpa_decode_fused.cpp` holds the f16 arm to. Against a float64 reference the fused kernel's largest error is equal to or below the composition's at 12 of 13 shapes and 3.95e-5 against 3.54e-5 at the other. Both arms stored the same bits in all three of their processes.

## End to end (`e2e/`)

`scripts/e2e_f16.py`: `mlx-community/Qwen3-4B-Instruct-2507-4bit` (`50d427756c6b`, `model.safetensors` sha256 `2a73c6c248601ab904e035548abd8e6abb65ea27dcb5f342fb0a8910eb44173f`, `config.json` `574349e5…`, `tokenizer.json` `aeb13307…`, the same files as `receipts/2026-11-21-m3-pro-g15s-linux`), loaded with mlx-lm 0.31.3 and cast with `model.set_dtype(mx.float16)`, so activations, the KV cache and all 36 attention layers run in f16 at head width 128. The prompt is the first 64, 1100 or 3000 tokens of `docs/architecture.md`, `docs/roadmap.md` and `CONTRIBUTING.md` as they read at `79d99c90c`, without a chat template. Per context: one warmup and three timed greedy generations of 128 tokens (`generation_tps`), then a teacher-forced decode of the next 48 text tokens, one token per step, keeping the full-vocabulary log-softmax of every step. Three interleaved pairs per budget, budget unset first (off on on off off on), then 15000 (on off off on on off); 15 s idle before each process; 1-minute load 0.22 to 0.89 before each; SMC sensors 30 to 42 C.

| budget | context tokens | off tok/s (min-max) | on tok/s (min-max) | on / off | token ids (both arms) | compute dispatches per token off → on |
|---|---:|---|---|---:|---|---|
| unset (5000) | 64 | 37.24 (37.21-37.26) | 44.95 (44.89-45.03) | 1.207 | `93a508eefb889fa3` | 450.83 → 378.27 |
| unset (5000) | 1100 | 16.06 (16.04-16.07) | 36.36 (36.34-36.40) | 2.264 | `7f41940d1af2018e` | 450.83 → 414.55 |
| unset (5000) | 3000 | 8.03 (8.02-8.04) | 30.71 (30.70-30.71) | 3.824 | `58d209ddda254758` | 456.65 → 420.37 |
| 15000 | 64 | 39.58 (39.57-39.58) | 47.79 (47.65-47.90) | 1.207 | `93a508eefb889fa3` | 450.83 → 378.27 |
| 15000 | 1100 | 16.82 (16.82-16.84) | 39.69 (39.60-39.78) | 2.359 | `7f41940d1af2018e` | 450.83 → 414.55 |
| 15000 | 3000 | 8.16 (8.14-8.17) | 33.14 (33.04-33.21) | 4.059 | `58d209ddda254758` | 456.65 → 420.37 |

Mean of the three process medians per arm; each process's three timed runs gave the same token ids as its warmup. The 15000 rows stand whether or not a G15S budget row lands first: they are the same kernel switch under that budget.

Compute dispatches per token drop by 72 below 1024 keys (36 layers, 3 to 1) and by 36 from 1024 keys (3 to 2); every other dispatch is the same.

Teacher-forced log-probs (`e2e/logits.txt`), off against on, 48 steps by 151,936 tokens per context: mean KL 1.7e-5 to 2.3e-5, max 8.9e-5; largest |dlogprob| on the reference top-1 token 0.017; the top-1 token is the same at all 144 steps. Within an arm, every process stored the same log-probs.

## Tests (`battery/`)

Head C++ test tree, Release, built from the prepared source of `4e3b83d2a`. One process per suite, each under the GPU lock with a timeout (`scripts/run_head.sh`); `summary.txt` has every line.

| group | result |
|---|---|
| standing battery, 25 suites (`AGENTS.md` § Test rules) | 25 of 25 exit 0; 748 of 748 cases |
| capability-simulation profiles (8) | 8 of 8; 88 of 88 cases |
| `omarchy_sdpa_decode_fused_tests` | 8 of 8 cases, 108,687 assertions; the f16 case's dispatch checks (1 at 263 keys, 2 at 1100) now run here instead of printing their skip line |
| new `omarchy_runtime_tests` case `f16 decode sdpa without the cooperative matrix is a G16S and G15S default` | 9 of 9 assertions |

`omarchy_fast_ops_tests` exits 0 with 4 failed assertions, all inside `sdpa vjp dk dv match finite differences at rep=1 (path per gate)`, which is `doctest::may_fail` for the open entry in `docs/known-defects.md`. The values (`dv[104]` at 5x7 twice, `dk[28]` and `dv[28]` at 4x4) are line for line the ones `main` logs on this machine (`receipts/2026-11-21-m3-pro-g15s-linux/suite/S04-fast_ops.txt`).

Base against head, the switch and the budget unset (`head/`): the base wheel composes (3 dispatches at every shape) and stores the same bits as the forced-off arm; the head wheel takes the fused kernel (1 or 2) and stores the same bits as the forced-on arm, per call and in the end-to-end log-probs (`head/default-check.txt`). Head end to end: 44.85 / 36.25 / 30.59 tok/s at 64 / 1100 / 3000 tokens, the same three token-id hashes.

## Dispatch trace and reopen (`trace/`)

`receipts/2026-09-30-speech-input-gpu/harness/count_cpu.gdb.py` counts calls to `mlx::core::cpu::get_command_encoder(Stream)` in the same process. `scripts/trace_decode_f16.py` is `receipts/2026-11-21-m3-pro-g15s-linux/trace/trace_decode.py` with the model cast to f16 after load. Qwen3-4B 4-bit in f16, 32 greedy tokens:

| run | CPU encoder calls | Vulkan compute dispatches | token sha256 |
|---|---:|---:|---|
| `stream=mx.cpu` control, head wheel | 3 | | |
| f16 decode, base wheel | 0 | 15,370 | `4dbda57a70ea…` |
| f16 decode, head wheel | 0 | 12,994 | `4dbda57a70ea…` |

The difference, 2,376, is 33 traced decode steps times 36 layers times 2 dispatches. Every dispatch is a Vulkan dispatch.

Reopen at the end: `omarchy_runtime_tests` `a fresh process reopens the Vulkan device` passes and `mlx-omarchy-info --trace-smoke` reports a buffer round trip on `Apple M3 Pro (G15S B1)`.

## CPU-disabled wheel (`nocpu/`)

`docs/parity-contract.md` § 2.1 build of `4e3b83d2a` (`MLX_BUILD_CPU=OFF`; `no_cpu/compiled.cpp`, `no_cpu/device_info.cpp` and `no_cpu/primitives.cpp` compiled, of `cpu/` only `encoder.cpp` and `eval.cpp`; `nocpu/build.txt`), wheel `mlx_omarchy-0.32.4.dev202611220507+nocpu.4e3b83d2a`. A GPU add runs and an explicit CPU-stream add is refused. Same wheel, `MLX_OMARCHY_SDPA_F16_NO_COOPMAT=0` against unset: per call 2.8x to 13.0x over the same 13 shapes, \|on − off\| at most 4.9e-4; decode 37.25 → 44.95, 16.08 → 36.36, 8.04 → 30.73 tok/s at 64 / 1100 / 3000 tokens (two interleaved pairs; load 0.22 to 0.54 before the decode legs, 0.75 and 0.93 before the per-call legs), token ids identical (`nocpu/summary.txt`).

## Files

`env/host.txt`, `env/wheels.txt`, `env/venv-*-freeze.txt`, `percall/` (per-process JSON and summary), `e2e/` (per-process JSON, summary, `logits.txt`), `head/`, `battery/`, `trace/`, `nocpu/`, `scripts/` (the probes, the summarizers and the run scripts), `SHA256SUMS`. Per-leg device state before and after (load, memory, SMC temperatures, governor, AC, GPU kernel log line count, Vulkan identity) is in each run directory's `*.state`; no leg added a GPU kernel log line.
