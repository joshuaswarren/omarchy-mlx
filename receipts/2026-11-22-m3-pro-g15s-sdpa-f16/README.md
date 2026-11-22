# Fused f16 decode attention below 1024 keys on the M3 Pro (G15S)

Date: 2026-11-22. One machine, the M3 Pro test host. This receipt backs the G15S one-pass entry of `sdpa_decode_f16_without_coopmat_part` and `sdpa_decode_f16_two_pass_blocks` in `overlay/mlx/backend/omarchy/device.h` (code commit `95126c59f`), its `docs/kernel-flags.md` row, and the f16 sentence of the M3 Pro entry in `docs/compatibility.md`.

## Result

The fused f16 single-query decode attention kernel (`sdpa_decode_native*.comp`, head width 64 or 128) uses no cooperative matrix, but its gate requires one except where `sdpa_decode_f16_without_coopmat_part` drops it (previously G16S only, PR #439 by František Balázsy, `1a0caa96d`). On `95126c59f`, G15S runs the one-pass fused kernel below 1024 keys (1 compute dispatch instead of the 3-dispatch composition) and keeps the 3-dispatch composition from 1024 keys upward. G16S stays on at every length, every other part is unchanged, and `MLX_OMARCHY_SDPA_F16_NO_COOPMAT` overrides at every length (non-empty and not `0` drops the clause on any part; `0` or empty keeps it). bf16 decode calls do not read this gate.

Measured on the M3 Pro test host, base default (`52ac565fc`, backend of branch base `79d99c90c`) against `95126c59f` default (switch and submission budget unset):

- per call below 1024 keys: 2.8x to 16.4x faster per independent call and 2.6x to 12.1x faster chained across all 10 shapes below 1024 keys (for 16 query / 8 key-value heads at width 128: 261.0 → 29.2 µs at 512 keys and 504.2 → 43.4 µs at 1023 keys); from 1024 keys, the same 3 dispatches, equal time, and bit-identical outputs;
- end to end, `mlx-community/Qwen3-4B-Instruct-2507-4bit` cast to f16, 128 greedy tokens: decode 37.28 → 44.95 tok/s (+20.6%) at 64 context tokens and 23.91 → 41.43 tok/s (+73.2%) at 512 context tokens; at 1100 context tokens, 16.10 → 16.11 tok/s with unchanged dispatches; at `MLX_OMARCHY_BATCH_WORK=15000`, 39.62 → 47.88 tok/s (+20.8%) at 64, 25.07 → 43.97 (+75.4%) at 512, and 16.83 → 16.84 at 1100;
- numerics: token ids identical in all processes of both sides at every context; per call, |`95126c59f` − `52ac565fc`| at most 4.9e-4 below 1024 keys (the f16 test gate in `test_sdpa_decode_fused.cpp` is 0.01) and 0 from 1024 keys; teacher-forced KL at most 8.84e-5 at contexts 64 and 512 with identical top-1 at every step, and bit-identical at context 1100.

## Machine and stack (`env/host.txt`)

| Chip | Machine | GPU cores | Memory | Kernel | Vulkan device |
|---|---|---|---|---|---|
| M3 Pro | MacBook Pro 14-inch, Nov 2023 (`apple,j514s` / `apple,t6030`), 11 CPU cores | 14 (core mask `0x6f5bd`) | 18 GB | `7.2.11-14-1-ARCH`, `linux-aurora 7.2.11.aurora14-1` | `Apple M3 Pro (G15S B1)` |

Driver: Mesa 26.3.0-devel Borealis (`git-a196911147`), packages `mesa-aurora` and `vulkan-aurora` `1:26.3.0_devel.20261031.231483.a1969111475-1`, Vulkan 1.4.363, driver ID 26, system ICD (`asahi_icd.aarch64.json`). GPU firmware: `macOS 14.7-14.8.x stub firmware (iBoot-10151.140.19.700.2)`, `version=0x7031100`. Governor `schedutil`, AC online. Python 3.14.8, `mlx-lm` 0.31.3, `glslc` 2026.4. No neural-engine stage runs.

Gate capabilities reported by the device (`env/host.txt`): subgroup size 32 (min = max 32) with BASIC, ARITHMETIC and SHUFFLE; 1024 max compute workgroup invocations and 1024 in x; 32,768 bytes of shared memory against the 21,504 bytes the kernel needs; no `VK_KHR_cooperative_matrix`.

## Builds and provenance (`env/wheels.txt`, `env/venv-*-freeze.txt`)

All wheels are Release builds. Every process printed the `scripts/mlx_provenance.py` line and refused to run unless the installed wheel's RECORD hashes matched (`verified=match`) and its version stamp matched the intended commit. `harness=95126c59f` on every round-2 run (and `c15a0d4f2` / `9530b604b` on the `merged/` runs). The head and base venvs hold the same packages apart from the `mlx_omarchy` wheel (`env/venv-base-freeze.txt`, `env/venv-head-freeze.txt`).

- Base: `mlx_omarchy-0.32.4.dev202611220252+52ac565fc` (`libmlx.so` sha256 `107b3cd434130872…`, wheel sha256 `f38fb986243d…`). `git diff --stat 52ac565fc 79d99c90c` lists `tests/test_paths_module.py` only, so this wheel carries the exact backend of branch base `79d99c90c`.
- Head (`95126c59f`): `mlx_omarchy-0.32.4.dev202611220609+95126c59f` (`libmlx.so` sha256 `89386d8380fcb6fc…`, wheel sha256 `e3be8d853a33…`).
- CPU-disabled head (`95126c59f`): `mlx_omarchy-0.32.4.dev202611220614+nocpu.95126c59f` (`libmlx.so` sha256 `56643b8658f2133c…`, wheel sha256 `56f9b0d82cbd…`), built with `MLX_BUILD_CPU=OFF` (`docs/parity-contract.md` § 2.1; `nocpu/build.txt`).
- Newest green `main` (`c15a0d4f2`, gate and tests checks passing): `mlx_omarchy-0.32.4.dev202611220700+c15a0d4f2` (`libmlx.so` sha256 `d56c4b6eb3ccd1d9…`, wheel sha256 `71f9ec2d7b02…`).
- Local merge of `95126c59f` onto `c15a0d4f2` (unpushed merge commit `9530b604b`, tree `a77a5d6d656833b1520d8dd3d7f8af995c6f1aea`, equal to `git merge-tree c15a0d4f2 95126c59f`): `mlx_omarchy-0.32.4.dev202611220703+9530b604b` (`libmlx.so` sha256 `88ddb9471a58d1dc…`, wheel sha256 `62b26081338a…`). No file under `overlay/mlx/backend/omarchy` differs between `8d92e094b` and `c15a0d4f2`.

## Procedure and thermals (`scripts/`, `legs.txt`)

Every GPU command ran through `scripts/gpuleg.sh`, which acquires the machine's GPU lock, enforces a timeout, records device state before and after (`*.state`: load, memory, sensors, governor, AC, GPU kernel log line count, Vulkan identity), and records GPU kernel log lines added during the leg. No leg added a GPU kernel log line. Timed legs waited inside the lock until the 1-minute load was under 1.0 and recorded `load1_at_start`; none exceeded 1.0. Builds ran under the same lock.

Sensor readings before and after each timed process (the kernel exposes no SoC or GPU temperature sensor on this machine):

- End-to-end processes: board and audio sensors 31 to 36 C, NVMe `Composite` 38 to 42 C, NAND 39 to 42 C, battery 33 to 36 C, `Charge_Regulator_Temp` 49 to 58 C, `WiFi/BT_Module_Temp` 50 to 62 C.
- Per-call legs: `Charge_Regulator_Temp` 44 to 49 C, `WiFi/BT_Module_Temp` 44 to 50 C, all other sensors 31 to 39 C.

## Per call (`percall/`, `summary.txt`)

`scripts/sdpa_f16_ab.py` (`SDPA_AB_SHAPES=r2`): `q` of shape `(1, H, 1, D)` and `k`, `v` of shape `(1, KV, L, D)` in f16, fixed seeds, scale `D^-0.5`. Independent timing measures host wall time per call over 200 calls of SDPA plus query scaling in one eval (the timing method of `receipts/2026-11-01-m4-sdpa-decode-bf16-256/scripts/sdpa_bits.py`). Chain timing measures 36 dependent calls in one eval. Seven repetitions per process, median; Vulkan compute dispatches per call come from the backend trace counter. Base wheel default (`52ac565fc`) against head wheel default (`95126c59f`), switch and budget unset, 3 processes each, interleaved (`base head head base base head`), 1-minute load 0.28 to 0.97 before each process. Median of the 3 process medians:

| q / kv heads, width | keys | dispatches | independent µs (`52ac565fc` → `95126c59f`) | chain µs (`52ac565fc` → `95126c59f`) | max \|head − base\| |
|---|---:|---|---|---|---:|
| 16 / 8, 128 | 37 | 3 → 1 | 42.9 → 15.5 (2.77x) | 51.1 → 19.5 (2.62x) | 4.88e-04 |
| 16 / 8, 128 | 263 | 3 → 1 | 147.6 → 20.8 (7.10x) | 168.3 → 33.8 (4.98x) | 1.22e-04 |
| 16 / 8, 128 | 512 | 3 → 1 | 261.0 → 29.2 (8.94x) | 289.4 → 50.7 (5.71x) | 1.22e-04 |
| 16 / 8, 128 | 1023 | 3 → 1 | 504.2 → 43.4 (11.62x) | 553.6 → 87.1 (6.36x) | 6.10e-05 |
| 16 / 8, 128 | 1024 | 3 → 3 | 503.8 → 505.1 | 553.4 → 554.8 | 0.00e+00 |
| 16 / 8, 128 | 1100 | 3 → 3 | 543.9 → 544.6 | 590.8 → 591.9 | 0.00e+00 |
| 16 / 8, 128 | 4096 | 3 → 3 | 2016.8 → 2016.2 | 2099.8 → 2097.6 | 0.00e+00 |
| 32 / 8, 128 | 64 | 3 → 1 | 86.2 → 17.7 (4.87x) | 95.3 → 25.5 (3.74x) | 2.44e-04 |
| 32 / 8, 128 | 263 | 3 → 1 | 268.2 → 29.2 (9.18x) | 284.8 → 44.2 (6.44x) | 1.22e-04 |
| 32 / 8, 128 | 512 | 3 → 1 | 485.6 → 44.7 (10.86x) | 507.1 → 67.6 (7.50x) | 1.22e-04 |
| 32 / 8, 128 | 1023 | 3 → 1 | 941.5 → 76.0 (12.39x) | 977.7 → 118.8 (8.23x) | 6.10e-05 |
| 32 / 8, 128 | 1024 | 3 → 3 | 942.0 → 942.1 | 977.1 → 974.1 | 0.00e+00 |
| 32 / 8, 128 | 2048 | 3 → 3 | 1875.8 → 1877.0 | 1935.5 → 1934.3 | 0.00e+00 |
| 32 / 8, 64 | 263 | 3 → 1 | 160.4 → 16.8 (9.55x) | 176.7 → 24.6 (7.18x) | 1.22e-04 |
| 32 / 8, 64 | 1023 | 3 → 1 | 556.1 → 34.0 (16.36x) | 593.3 → 49.2 (12.06x) | 6.10e-05 |
| 32 / 8, 64 | 1024 | 3 → 3 | 556.0 → 556.3 | 592.3 → 593.4 | 0.00e+00 |

Below 1024 keys: 2.8x to 16.4x per independent call and 2.6x to 12.1x chained. From 1024 keys: 3 dispatches and bit-identical outputs. Across every element of every shape, |`95126c59f` − `52ac565fc`| is at most 4.9e-4 (test gate 0.01) below 1024 keys and 0 from 1024 keys. Against the float64 reference recorded in each JSON (`max_abs_err_vs_f64`), `95126c59f` is at or below `52ac565fc` at all 16 shapes. Each side stored identical bits across all three of its processes.

## End to end (`e2e/`, `summary.txt`)

`scripts/e2e_f16.py` (`E2E_CTX=64,512,1100`): `mlx-community/Qwen3-4B-Instruct-2507-4bit` (revision `50d427756c6b`, `model.safetensors` sha256 `2a73c6c248601ab904e035548abd8e6abb65ea27dcb5f342fb0a8910eb44173f`, `config.json` `574349e5a343236546fda55e4744a76e181f534182d7dc60ff1bad7e7a502849`, `tokenizer.json` `aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4`, the same files as `receipts/2026-11-21-m3-pro-g15s-linux`), cast with `model.set_dtype(mx.float16)` so activations, the KV cache, and all 36 attention layers (head width 128) run in f16. Prompt: the first 64, 512, or 1100 tokens of `docs/architecture.md`, `docs/roadmap.md`, and `CONTRIBUTING.md` at `79d99c90c`, concatenated in that order with no chat template. Contexts 64 and 512 stay below 1024 keys for all 128 generated tokens; context 1100 is above 1024 keys.

Per context: one warmup and three timed greedy generations of 128 tokens (`generation_tps`, median), then a teacher-forced decode of the next 48 text tokens one at a time keeping the full-vocabulary log-softmax (151,936 tokens). Base default (`52ac565fc`) against head default (`95126c59f`), 3 processes each per submission budget, interleaved, budget unset (5000) first, then `MLX_OMARCHY_BATCH_WORK=15000`; 15 s idle before each process; 1-minute load 0.14 to 0.53 before each; peak GPU memory 2.967 GiB in every process. Mean of the 3 process medians (min-max):

| budget | context | base `+52ac565fc` | head `+95126c59f` | head / base | dispatches per generated token |
|---|---:|---|---|---:|---|
| unset (5000) | 64 | 37.28 (37.24-37.33) | 44.95 (44.90-45.04) | 1.206 | 450.83 → 378.27 |
| unset (5000) | 512 | 23.91 (23.89-23.94) | 41.43 (41.37-41.47) | 1.732 | 450.83 → 378.27 |
| unset (5000) | 1100 | 16.10 (16.09-16.11) | 16.11 (16.11-16.11) | 1.001 | 450.83 → 450.83 |
| 15000 | 64 | 39.62 (39.62-39.62) | 47.88 (47.84-47.91) | 1.208 | 450.83 → 378.27 |
| 15000 | 512 | 25.07 (25.06-25.08) | 43.97 (43.95-44.00) | 1.754 | 450.83 → 378.27 |
| 15000 | 1100 | 16.83 (16.83-16.83) | 16.84 (16.84-16.85) | 1.001 | 450.83 → 450.83 |

Token ids are identical across all processes of both sides: ctx 64 `93a508eefb889fa3`, ctx 512 `c70f66886c70bc71`, ctx 1100 `7f41940d1af2018e`. The per-token dispatch counts surround the whole generation; per single-query decode step the 36 attention layers drop from 3 dispatches to 1 each (72 fewer). Teacher-forced log-probs (`95126c59f` against `52ac565fc`, 48 steps by 151,936 tokens): contexts 64 and 512 KL at most 8.84e-5 (8.84e-5 at ctx 64, 7.82e-5 at ctx 512), largest |dlogprob| 0.164 (0.0624 at ctx 64, 0.164 at ctx 512), top-1 token equal at all 48 steps; context 1100 bit-identical. Each side stored identical log-probs across all of its processes. The 15000-budget rows are the same comparison under a different submission budget.

## Newest `main` and local merge (`merged/`)

One process each on upstream green `main` `c15a0d4f2` and on `95126c59f` merged locally onto `c15a0d4f2` as `9530b604b` (tree `a77a5d6d656833b1520d8dd3d7f8af995c6f1aea`, clean `git merge-tree`), switch and budget unset (`scripts/run_mergedtree2.sh`), summarized by `scripts/pr_lines.py`:

```text
$ python3 pr_lines.py M3-main2.json M4-main2.json
c15a0d4f2 sdpa f16 16q/8kv w128: keys=512 disp=3 260.9us | keys=1023 disp=3 505.4us | keys=1024 disp=3 504.9us
c15a0d4f2 decode tok/s: ctx=64 37.26 | ctx=512 23.93 | ctx=1100 16.09  ids 93a508ee c70f6688 7f41940d
$ python3 pr_lines.py M3-merged2.json M4-merged2.json
9530b604b sdpa f16 16q/8kv w128: keys=512 disp=1 28.9us | keys=1023 disp=1 43.1us | keys=1024 disp=3 504.9us
9530b604b decode tok/s: ctx=64 44.93 | ctx=512 41.44 | ctx=1100 16.11  ids 93a508ee c70f6688 7f41940d
```

## Regression tests and constant-value probe (`tests/`, `overflow/`)

- `omarchy_sdpa_decode_fused_tests --test-case="G15S f16 decode stays on one-pass below 1024 keys and avoids two-pass overflow"` (`overlay/tests/omarchy/test_sdpa_decode_fused.cpp:781`): runs on G15S without cooperative matrix (prints a skip line elsewhere); tests `q = k = 0` and `v = 8192`, expecting 1 dispatch and exact 8192 at 1023 keys, and 3 dispatches and exact 8192 at 1024, 4096, and 8192 keys.
  - Built against `8304e5dd3` (the previous branch commit, with only the new test file copied in; `tests/R2-T01-newcase-before-8304e5dd3.log`): 6 of 12 assertions fail (at 1024, 4096, and 8192 keys: 2 dispatches instead of 3, and 2048 of 2048 elements are not 8192).
  - Built on `95126c59f` (`tests/R2-T02-newcase-head.log`): 12 of 12 assertions pass.
- `omarchy_runtime_tests --test-case="f16 decode sdpa without the cooperative matrix defaults on G16S and G15S one-pass"` (`overlay/tests/omarchy/test_runtime.cpp:1747`, `tests/R2-T03-runtime-case-head.log`): 24 of 24 assertions pass.
- `omarchy_sdpa_decode_fused_tests` full suite on `95126c59f` (`tests/R2-T04-sdpa-decode-fused-head.log`): 9 of 9 cases, 108,699 of 108,699 assertions pass (the existing f16 case expects 3 dispatches at 1100 keys on G15S).
- `scripts/overflow_probe.py` (`overflow/`): `q = k = 0`, `v` constant across keys 1023, 1024, 4096, 8192 and `v` in 1024, 2047, 2048, 4096, 8192 (20 cells) returns the constant exactly with 0 infinities in all 20 cells on both the `52ac565fc` base default (`R2-X1-overflow-base-default.log`) and the `95126c59f` head default (`R2-X2-overflow-head-default.log`).

## Standing battery (`battery/`)

Head C++ test tree built from the prepared source of `95126c59f` (Release, fresh build directory), one process per suite under the GPU lock (`scripts/run_r2.sh`):

- Standing M1/Omarchy battery (`AGENTS.md` § Test rules): 25 of 25 suites exit 0, 748 of 748 cases pass.
- Capability-simulation profiles (`omarchy_capability_sim_tests`): 8 of 8 profiles exit 0, 88 of 88 cases pass.

`omarchy_fast_ops_tests` exits 0 with 4 failed assertions, all in `sdpa vjp dk dv match finite differences at rep=1 (path per gate)`, which is `doctest::may_fail` for the open entry in `docs/known-defects.md` (SDPA backward fused VJP value defects at small rep=1 shapes); upstream `main` `c15a0d4f2`'s own test build logs the same four lines in the same run (`battery/fast_ops-main-vs-head.txt`).

## Dispatch trace and reopen (`trace/`)

`receipts/2026-09-30-speech-input-gpu/harness/count_cpu.gdb.py` counts calls to `mlx::core::cpu::get_command_encoder(Stream)` inside the same process. `scripts/trace_decode_f16.py` is `receipts/2026-11-21-m3-pro-g15s-linux/trace/trace_decode.py` with the f16 cast and a provenance check. Qwen3-4B-Instruct-2507 4-bit in f16, 32 greedy tokens from a short prompt:

| run | CPU encoder calls | Vulkan compute dispatches | token sha256 |
|---|---:|---:|---|
| `stream=mx.cpu` control, `95126c59f` (`R2-H30`) | 3 | | |
| f16 decode, base `52ac565fc` (`R2-H32`) | 0 | 15,370 | `4dbda57a70ea…` |
| f16 decode, head `95126c59f` (`R2-H31`) | 0 | 12,994 | `4dbda57a70ea…` |

Reopen at the end: `omarchy_runtime_tests --test-case="a fresh process reopens the Vulkan device"` passes (`R2-H40-reopen-runtime.log`), and `mlx-omarchy-info --trace-smoke` completes a 65,536-byte buffer round trip on `Apple M3 Pro (G15S B1)` (`R2-H41-reopen-smoke.log`).

## CPU-disabled wheel (`nocpu/`, `summary.txt`)

`docs/parity-contract.md` § 2.1 build of `95126c59f` (`MLX_BUILD_CPU=OFF`; `no_cpu/compiled.cpp`, `no_cpu/device_info.cpp`, and `no_cpu/primitives.cpp` compiled; of `cpu/` only `encoder.cpp` and `eval.cpp`; `nocpu/build.txt`), wheel `mlx_omarchy-0.32.4.dev202611220614+nocpu.95126c59f`.

- Control (`R2-N00-cpu-refused.log`): a GPU add runs (`[2.0, 2.0, 2.0, 2.0]`) and an explicit CPU-stream add is refused (`IndexError`).
- Same wheel, `MLX_OMARCHY_SDPA_F16_NO_COOPMAT=0` against unset: per call, 1 dispatch and 2.8x to 16.5x per independent call (2.7x to 12.1x chained) below 1024 keys, 3 dispatches and equal time from 1024 keys, |on − off| at most 4.9e-4 below 1024 keys and 0 from 1024 keys; end-to-end decode (2 processes per side, load 0.10 to 0.32) 37.29 → 44.90 tok/s at ctx 64, 23.93 → 41.47 at ctx 512, and 16.12 → 16.10 at ctx 1100, token ids identical, log-probs matching the release wheel (`KLmax` 8.84e-5 at ctx 64, 7.82e-5 at ctx 512, bit-identical at ctx 1100).

## Replay (`scripts/`)

Published scripts take four environment variables:

- `LANE`: scratch folder holding `scripts/`, `logs/`, `evidence/`, and `runs/` (copy `scripts/` from this receipt into `"$LANE/scripts/"`)
- `REPO`: path to this repository checkout
- `BASE_BUILD`: folder holding the base wheel in `dist-base/` and its venv in `venv-base/`
- `MODELS`: folder holding `qwen3-4b-2507-4bit/`

Rebuild the prompt text from commit `79d99c90c` before running `scripts/run_r2.sh` and `scripts/run_mergedtree2.sh`:

```bash
git -C "$REPO" show \
  79d99c90c:docs/architecture.md \
  79d99c90c:docs/roadmap.md \
  79d99c90c:CONTRIBUTING.md \
  > "$LANE/scripts/context-text.txt"
LANE="$LANE" REPO="$REPO" BASE_BUILD="$BASE_BUILD" MODELS="$MODELS" \
  bash "$LANE/scripts/run_r2.sh"
LANE="$LANE" MODELS="$MODELS" \
  bash "$LANE/scripts/run_mergedtree2.sh"
```

## Files

`env/host.txt`, `env/wheels.txt`, `env/venv-base-freeze.txt`, `env/venv-head-freeze.txt`, `legs.txt`, `summary.txt`, `percall/`, `e2e/`, `merged/`, `tests/`, `overflow/`, `battery/`, `trace/`, `nocpu/`, `scripts/`, `SHA256SUMS`. Per-leg device state before and after each GPU command is stored in each directory's `*.state` files.
