# Split-K bf16 decode attention defaults on the M3 Pro (G15S)

Date: 2026-11-22. One machine, the M3 Pro test host. The PR is one commit on `main` `3ae5c9ea3` (#757) holding the five G15S rows in `overlay/mlx/backend/omarchy/device.h`, the test case in `overlay/tests/omarchy/test_sdpa_split.cpp`, the three updated rows in `docs/kernel-flags.md`, the split-K bf16 sentence group of the M3 Pro entry in `docs/compatibility.md`, and this receipt. Wheel measurements were taken on code commit `60d2ae306` against `main` `69a7123e8` (not in the PR history).

## Result

In `overlay/mlx/backend/omarchy/device.h`, G15S (`Apple M3 Pro (G15S B1)`) receives calibrated defaults in all five split-K bf16 single-query decode attention resolvers:

- `sdpa_split_min_keys`: `192` (bf16 split-K decode route by default from 192 keys, head width 128)
- `sdpa_split_wide_min_keys`: `1408` (head width 256)
- `sdpa_split_pv_max_keys`: `1280` (value-phase pass 1 up to 1280 keys, head width 128)
- `sdpa_split_bk64_max_keys`: `640` (64-key blocks up to 640 keys; four query heads per key-value head only)
- `sdpa_split_pv_wide_max_workgroups`: `36` (width-256 value-phase pass 1 while key-value heads × 128-key blocks <= 36)

Before this change, G15S was unlisted and every resolver returned `0`, so bf16 single-query decode stayed on the fused kernel (1 dispatch) through 7168 keys and used the 8-dispatch composition above 7168 keys. Every other part is unchanged (including the BMG row added in `sdpa_split_min_keys` by #737); the base M3 (G15G) keeps `0` everywhere. Because #729 turns split-K verify rows on by default wherever the split route engages, G15S speculative-decode verify rows follow these same key thresholds (`overlay/tests/omarchy/test_sdpa_split.cpp` checks each row equals the one-token split result bit for bit). All environment overrides (`MLX_OMARCHY_SDPA_SPLIT_MIN_KEYS`, `MLX_OMARCHY_SDPA_SPLIT_WIDE_MIN_KEYS`, `MLX_OMARCHY_SDPA_SPLIT_PV_MAX_KEYS`, `MLX_OMARCHY_SDPA_SPLIT_BK64_MAX_KEYS`, `MLX_OMARCHY_SDPA_HD256_PV_LIMIT`) and route switches (`MLX_OMARCHY_SDPA_DECODE_SPLIT`, `MLX_OMARCHY_SDPA_SPLIT_PV`, `MLX_OMARCHY_SDPA_SPLIT_BK64`) stay unchanged. Precedents: PR #415 (G13S/G14G split default), `receipts/2026-11-15-m4-pro-g16s-decode-defaults`, `receipts/2026-11-14-g14c-sdpa-hd256-pv-limit`, and G15S PRs #624 and #630.

Summary on the M3 Pro test host, `main` (`69a7123e8`) default against measured code commit `60d2ae306` default with all split-K environment variables unset:

- per call (`default/summary.txt`, `layout/summary.txt`): at calibration shapes, width 128 (32q/8kv) speeds up chained calls from 1.67x at 192 keys to 4.38x at 4096 and 23.82x at 16384 (independent 1.04x to 26.14x); width 256 (8q/2kv) is unchanged through 1281 keys, then speeds up chained calls 2.57x at 1408 keys to 4.30x at 4096 and 17.55x at 8192; dispatches per call move 1 → 2 (8 → 2 above 7168 keys). Other width-256 layouts (`16q/4kv`, `32q/8kv`) are faster in both shapes from 1408 keys; other width-128 groups (`8q/8kv`, `16q/8kv`, `64q/8kv`) are faster chained from 192–256 keys;
- end to end (`ab/summary.txt`, 128 greedy tokens): `Qwen3-4B-Instruct-2507` 4-bit decodes +17.3% to +103.4% faster across contexts 443 to 4000; `Qwen3.8-2B` 4-bit is unchanged below 1408 keys (-0.1% / -0.2%, bit-identical forced log-probs) and decodes +12.1% / +23.7% faster at contexts 2048 / 4000;
- numerics (`gate/summary.txt`, `gate/baseline/identity.txt`, `ab/logits.txt`, `parity/summary.txt`): max |error| vs float64 per call is below 0.00185 across all routes; on the numerics gate, five identical-protocol `main` captures per model are bit-identical (run-to-run allowance 0), check 2 passes overall and in every key band on both references (every flip is a near tie), and check 3 is met on two of four rows (`Qwen3-4B` CPU 0.127% → 0.092%, `Qwen3.8-2B` GPU float32 0.088% → 0.020%) and not met on two (`Qwen3.8-2B` CPU 0.021% → 0.047%, +0.026 points; `Qwen3-4B` GPU float32 0.106% → 0.118%, +0.012 points), with every whole-sample paired block-bootstrap 95% interval containing 0; accepted under the routed-MoE precedent in `docs/numerics-gate.md` (top-1 agreement rises on both rows where deviation increases).

## Machine and stack (`env/host.txt`)

| Chip | Machine | GPU cores | Memory | Kernel | Vulkan device |
|---|---|---|---|---|---|
| M3 Pro | MacBook Pro 14-inch, Nov 2023 (`apple,j514s` / `apple,t6030`), 11 CPU cores | 14 (core mask `0x6f5bd`) | 18 GB | `7.2.11-14-1-ARCH`, `linux-aurora 7.2.11.aurora14-1` | `Apple M3 Pro (G15S B1)` |

Driver: Mesa 26.3.0-devel Borealis (`git-a196911147`), packages `mesa-aurora` and `vulkan-aurora` `1:26.3.0_devel.20261031.231483.a1969111475-1`, Vulkan 1.4.363, driver ID 26 (`MESA_HONEYKRISP`), system ICD (`asahi_icd.aarch64.json`). GPU firmware: `macOS 14.7-14.8.x stub firmware (iBoot-10151.140.19.700.2)`, `version=0x7031100`. Governor `schedutil`, AC online. Python 3.14.8, `mlx-lm` 0.31.3 patched with `scripts/apply-mlx-lm-patches.sh`, `glslc` 2026.4. No neural-engine stage runs.

Models (`env/models.txt`: sha256 of `model.safetensors`, `config.json` and `tokenizer.json`): `mlx-community/Qwen3-4B-Instruct-2507-4bit` and a Qwen3.8-2B 4-bit MLX conversion, both affine 4-bit with group size 64, bf16 activations.

Split-K route gate fields reported by the device (`env/host.txt`): subgroup size 32 (min = max 32) with `BASIC` and `ARITHMETIC` operations, and 1024 max compute workgroup invocations.

## Builds and provenance (`env/wheels.txt`, `env/venv-*-freeze.txt`)

All four wheels are Release builds. Each wheel was built in its own scratch worktree with a fresh `.work` directory (fresh CMake cache) and `scripts/wheel-cmake-args.sh` release flags. Every process printed the `scripts/mlx_provenance.py` line and refused to run unless `verified=match` and the installed version stamp matched the intended commit. The `main` and `head` virtual environments hold the same packages apart from the `mlx_omarchy` wheel (`env/venv-base-freeze.txt`, `env/venv-main-freeze.txt`, `env/venv-head-freeze.txt`).

- Base (`85e0f4480`, used for calibration sweeps, forced variants, and 32-step float32 parity): `mlx_omarchy-0.32.4.dev202611221441+85e0f4480` (`libmlx.so` sha256 `8bd56a11df3fa5c0`, wheel sha256 `ba1f220ec136…`). `85e0f4480` is `main` before the rebase onto `69a7123e8`; `git diff 85e0f4480 69a7123e8` changes no SDPA file (`overlay/` changes are `device.cpp` and `honeykrisp_identity.h` driver-identity code and one ANE worker test, plus two `mlx-lm` speculative-verify patches in both `patches/` and `patches/mlx-lm-0.32/`, 4 files), so this build carries the exact SDPA code of `69a7123e8` for calibration.
- Main (`69a7123e8`, parent of the measured code commit, PR #639; gate and tests checks green): `mlx_omarchy-0.32.4.dev202611221546+69a7123e8` (`libmlx.so` sha256 `f7fb3dd1c8afc974`, wheel sha256 `6b8099e586f1…`).
- Measured head (`60d2ae306`, pre-squash code commit): `mlx_omarchy-0.32.4.dev202611221548+60d2ae306` (`libmlx.so` sha256 `b2f669ca3ce98b44`, wheel sha256 `cdc37ff6d334…`).
- CPU-disabled head (`60d2ae306`, `MLX_BUILD_CPU=OFF`, `docs/parity-contract.md` § 2.1): `mlx_omarchy-0.32.4.dev202611221553+nocpu.60d2ae306` (`libmlx.so` sha256 `90aeca7476c0ac66`, wheel sha256 `feaaa28ec33b…`).

Single-commit PR layout: the PR is one commit on `main` `3ae5c9ea3` (#757) containing the G15S rows in `device.h`, the test case, the `docs/kernel-flags.md` rows, the `docs/compatibility.md` sentence group, and this receipt. All wheel measurements (calibration sweeps, per-call checks, head layouts, model A/B, numerics gate, and CPU-disabled checks) come from code commit `60d2ae306` against `main` `69a7123e8` (and base `85e0f4480`); `60d2ae306` and the comment-only follow-ups are not in the PR history. Relative to `60d2ae306`, the PR commit's `overlay/` differs only in comments (head-layout notes, `"Apple M3 Pro, 14-core GPU"` naming, and a split-K header sentence), the comma-free G15S test case name in `overlay/tests/omarchy/test_sdpa_split.cpp`, and `main`'s changes between `69a7123e8` and `3ae5c9ea3`: a BMG row in `sdpa_split_min_keys` (#737), split-K verify rows defaulting on wherever the split route engages (#729, so G15S speculative-decode verify rows follow these same key thresholds; `overlay/tests/omarchy/test_sdpa_split.cpp` checks each row equals the one-token split result bit for bit), a dropped per-submit fence (#658), and a push-descriptor dispatch path where the driver offers `VK_KHR_push_descriptor` (#757). On the rebased tree (the PR commit's overlay on `3ae5c9ea3`), the C++ test tree was rebuilt and rerun (`battery-rebased/`).

## Procedure and thermals (`scripts/gpuleg.sh`)

Every GPU and build step ran under the machine's GPU lock with a timeout through `scripts/gpuleg.sh`, recording device state before and after each leg (`*.state`: load, memory, sensors, governor, AC, GPU kernel log line count, Vulkan identity). No leg added a GPU kernel log line. Timed legs waited inside the lock for 1-minute load below 1.0 and recorded it: 0.07 to 0.99 at start for every kept leg (0.05 including the void legs), none at or above 1.0. Logs from the gate, layout, and rerun legs print `harness=a5693ac16-dirty` because the lane scripts ran from the checkout while edits were uncommitted; every wheel still checked `verified=match`.

Memory and sensor readings during A/B processes (`*.state`; no SoC or GPU temperature sensor is exposed on this machine): `mem_used_mb` 6,045 to 6,444 MB; board/audio 29 to 37 C, NVMe `Composite` 37 to 42 C, NAND 39 to 42 C, battery hotspot 34 to 35 C, `Charge_Regulator_Temp` 45 to 59 C, `WiFi/BT_Module_Temp` 46 to 63 C.

Failed legs are kept and labelled alongside their reruns:

- `parity/void-oom-*` and `parity/void-oom2-*`: the `Qwen3.8-2B` float32 reference ran out of GPU memory with single-pass prefill; rerun with 512-token chunked prefill (`parity/PR-base-qwen3.8-2b-4bit.*`).
- `battery/void-nofilter-H01-*`: doctest's `--test-case` filter split the original test name at its commas and matched no case; rerun with the wildcard filter `*M3 Pro (G15S) defaults*` (`battery/H01-g15s-case.*`), and after renaming the test case without commas, rerun with the full case name (`battery/H03-g15s-case-renamed.*`).
- `nocpu/void-indexerr-N00-*`: the first CPU-disabled control passed `stream=mx.cpu` and raised `IndexError`, which `docs/parity-contract.md` § 2.1 counts as a failed control; rerun using `mx.new_stream(mx.cpu)` in separate processes (`nocpu/N00-gpu-add.*`, `nocpu/N01-cpu-refused.*`).
- `gate/void-badvalidate-*`: the first bf16 numerics-gate captures' prefill check read one logit row because the packaged last-logits patch keeps one row on cached calls; rerun with `MLX_OMARCHY_FULL_LOGITS=1` for the prefill check only (`gate/NG-{main,head}-*`). The void bf16 gate `.npz` files are not committed; their logs record `ppl_teacher_forced` identical to the reruns (`27.5695` / `27.5728` on `Qwen3-4B`, `26.8196` / `26.8376` on `Qwen3.8-2B`).
- `gate/void-timelimit-NG-cpuref-qwen3-4b-2507-4bit.*`: contiguous 5,120-step `Qwen3-4B` CPU reference hit its 10,800 s timeout.
- `gate/void-stopped-NG-cpuref-qwen3-4b-2507-4bit.*`: contiguous 5,120-step `Qwen3-4B` CPU reference was stopped after step 1536 (would pass its 6 h timeout); split into two 2,560-step halves via `scripts/run_gate4.sh`, where the first half (`gate/NG-cpuref-qwen3-4b-2507-4bit-a.*`, cache 128–2,687) completed.
- `gate/void-killed-NG-cpuref-qwen3-4b-2507-4bit-b.*`: second-half `Qwen3-4B` CPU reference (`GATE_PREFIX=2688`, `GATE_STEPS=2560`, cache 2,688–5,247) was killed with its timeout wrapper after step 512/2,560 (`timeout` sends a parent-death signal to its child).

## Calibration sweeps (`sweep/`)

`scripts/sdpa_sweep.py` (using the timing method of `receipts/2026-11-15-m4-pro-g16s-decode-defaults/split-k/tools/sdpa_bench_kv.py`) ran on the base build `85e0f4480`: `q` `(1, H, 1, D)` against `K`, `V` `(1, KV, L, D)` in bf16 with scale `D^-0.5`. `independent` runs 36 calls on 36 K/V sets in one eval; `chain` runs 36 dependent calls (each output feeds the next query). Each cell reports the median of 7 timed evals after warm-up, one process per variant, 2 runs per variant with run 2 reversing variant order.

Shapes: head width 128 with 32 query / 8 key-value heads (`Qwen3-4B` attention) and head width 256 with 8 query / 2 key-value heads (`Qwen3.8-2B` full attention). Variants: `default` (env unset), `split` (`MLX_OMARCHY_SDPA_DECODE_SPLIT=1`, `PV=0`, `BK64=0`), `pv` (`PV=1`, `BK64=0`), and `bk64` (`PV=1`, `BK64=1`, width 128 only). Three sweeps ran: `SW` (128 to 16384 keys, `sweep/summary-SW.txt`), `FN` (fine, `sweep/summary-FN.txt`), and `ED` (edges, `sweep/summary-ED.txt`).

Selection rule (same as earlier parts): applied at the two calibration shapes — 32 query / 8 key-value heads (width 128) and 8 query / 2 key-value heads (width 256) — a variant becomes the default where it is faster than the alternative in both independent and chain shapes in both runs. The same key thresholds apply to other head layouts with the trade-offs in `layout/` (for example, width 128 with 64 query / 8 key-value heads at 192 keys: independent 70.9 → 97.8 µs, chain 97.8 → 98.8 µs; both shapes are faster from 320 keys). Measured edges in microseconds per call (ranges over the two runs; the 1408-key edge row spans `FN` and `ED`):

| Resolver and value | Width, heads | Below / at edge | Alternative (µs) | Selected variant (µs) |
|---|---|---|---|---|
| `sdpa_split_min_keys = 192` (`bk64` vs `default`) | 128, 32q / 8kv | 176 keys | ind 38.3–38.4, chain 66.0–66.1 | ind 38.5–38.6, chain 41.2–41.3 |
| | | 192 keys | ind 40.2–40.3, chain 69.6–70.0 | ind 38.6–38.8, chain 41.2–41.3 |
| `sdpa_split_bk64_max_keys = 640` (`bk64` vs `pv`) | 128, 32q / 8kv | 640 keys (80 wg) | `pv` ind 59.3–59.6 | `bk64` ind 49.1–50.7 |
| | | 641 keys (88 wg) | `pv` ind 59.2–60.2 | `bk64` ind 67.3–67.7 |
| `sdpa_split_pv_max_keys = 1280` (`pv` vs `split`) | 128, 32q / 8kv | 1280 keys (80 wg) | `split` ind 84.8–85.5 | `pv` ind 72.9–73.4 |
| | | 1281 keys (88 wg) | `split` ind 87.2–87.3 | `pv` ind 106.8–107.0 |
| `sdpa_split_wide_min_keys = 1408` (`pv` vs `default`) | 256, 8q / 2kv | 1344 keys | ind 88.7–89.1 | ind 88.6–89.3 (overlap) |
| | | 1408 keys | ind 90.9–91.8, chain 253.6–254.6 | ind 89.2–89.9, chain 97.6–99.6 |
| `sdpa_split_pv_wide_max_workgroups = 36` (`pv` vs `split`) | 256, 8q / 2kv | 2304 keys (18 × 2 = 36 wg) | `split` ind 132.9–138.9 | `pv` ind 94.8–103.2 |
| | | 2305 keys (19 × 2 = 38 wg) | `split` ind 132.8–139.0 | `pv` ind 156.8–160.1 |

At width 128, plain `split` first beats `default` at 448 keys. Above 1280 keys, `pv` loses to `split` from 1281 to 2048 keys and wins again at some longer lengths (2560, 3584, 4096+); the key-count rule takes the first edge (1280). Across all routes and lengths, max |error| vs float64 per call is below 0.00185 (bf16 output rounding; calibration maximum 0.00180380, layout sweep maximum 0.00184774).

## Head vs main per call (`default/`, `default/summary.txt`)

`DF1`..`DF4` ran ordered `main head head main` (2 processes per build, median, env unset). Dispatches per call change from 1 to 2 from the threshold through 7168 keys, and from 8 to 2 above 7168 keys. The `max |err| vs f64` column in both tables is set by bf16 output rounding (identical on both builds in every row).

Width 128, 32 query / 8 key-value heads (`Qwen3-4B` attention):

| keys | dispatches | independent µs (`69a7123e8` → `60d2ae306`) | chain µs (`69a7123e8` → `60d2ae306`) | max \|err\| vs f64 |
|---:|---|---:|---:|---:|
| 128 | 1 → 1 | 32.4 → 32.3 (1.00x) | 55.7 → 55.6 (1.00x) | 0.0018 → 0.0018 |
| 192 | 1 → 2 | 40.4 → 38.7 (1.04x) | 69.7 → 41.8 (1.67x) | 0.0017 → 0.0017 |
| 256 | 1 → 2 | 48.9 → 42.5 (1.15x) | 84.5 → 44.0 (1.92x) | 0.00096 → 0.00096 |
| 320 | 1 → 2 | 62.9 → 45.2 (1.39x) | 109.6 → 47.2 (2.32x) | 0.0015 → 0.0015 |
| 448 | 1 → 2 | 79.3 → 46.2 (1.72x) | 137.3 → 53.5 (2.57x) | 0.00097 → 0.00097 |
| 640 | 1 → 2 | 107.2 → 51.8 (2.07x) | 188.2 → 56.8 (3.31x) | 0.00049 → 0.00049 |
| 641 | 1 → 2 | 106.9 → 59.7 (1.79x) | 188.5 → 63.9 (2.95x) | 0.00049 → 0.00049 |
| 1024 | 1 → 2 | 163.7 → 68.8 (2.38x) | 279.6 → 71.5 (3.91x) | 0.00049 → 0.00049 |
| 1280 | 1 → 2 | 204.9 → 75.0 (2.73x) | 344.7 → 79.5 (4.34x) | 0.00048 → 0.00048 |
| 1281 | 1 → 2 | 213.2 → 87.3 (2.44x) | 357.9 → 93.8 (3.81x) | 0.00046 → 0.00046 |
| 1408 | 1 → 2 | 229.5 → 88.8 (2.58x) | 382.9 → 95.5 (4.01x) | 0.00039 → 0.00039 |
| 2048 | 1 → 2 | 324.0 → 108.2 (2.99x) | 538.6 → 117.3 (4.59x) | 0.00041 → 0.00041 |
| 2304 | 1 → 2 | 365.1 → 137.0 (2.67x) | 602.9 → 138.1 (4.37x) | 0.00042 → 0.00042 |
| 2305 | 1 → 2 | 370.1 → 157.5 (2.35x) | 615.9 → 147.0 (4.19x) | 0.00024 → 0.00024 |
| 3072 | 1 → 2 | 478.8 → 181.1 (2.64x) | 796.8 → 192.7 (4.14x) | 0.00048 → 0.00048 |
| 4096 | 1 → 2 | 633.1 → 227.6 (2.78x) | 1053.5 → 240.7 (4.38x) | 0.00024 → 0.00024 |
| 8192 | 8 → 2 | 8832.9 → 401.2 (22.02x) | 8828.5 → 450.0 (19.62x) | 0.00023 → 0.00023 |
| 16384 | 8 → 2 | 18145.3 → 694.1 (26.14x) | 18190.3 → 763.6 (23.82x) | 0.00012 → 0.00012 |

Width 256, 8 query / 2 key-value heads (`Qwen3.8-2B` full attention):

| keys | dispatches | independent µs (`69a7123e8` → `60d2ae306`) | chain µs (`69a7123e8` → `60d2ae306`) | max \|err\| vs f64 |
|---:|---|---:|---:|---:|
| 128 | 1 → 1 | 20.9 → 20.5 (1.02x) | 42.8 → 43.2 (0.99x) | 0.00098 → 0.00098 |
| 192 | 1 → 1 | 20.1 → 19.8 (1.01x) | 49.9 → 49.9 (1.00x) | 0.00094 → 0.00094 |
| 256 | 1 → 1 | 22.5 → 22.1 (1.02x) | 57.4 → 57.2 (1.00x) | 0.00095 → 0.00095 |
| 320 | 1 → 1 | 28.5 → 28.9 (0.99x) | 76.2 → 76.9 (0.99x) | 0.0009 → 0.0009 |
| 448 | 1 → 1 | 35.2 → 34.7 (1.01x) | 93.4 → 94.7 (0.99x) | 0.00063 → 0.00063 |
| 640 | 1 → 1 | 48.9 → 48.2 (1.01x) | 127.4 → 127.3 (1.00x) | 0.00049 → 0.00049 |
| 641 | 1 → 1 | 47.5 → 47.0 (1.01x) | 126.9 → 126.4 (1.00x) | 0.00049 → 0.00049 |
| 1024 | 1 → 1 | 67.9 → 67.3 (1.01x) | 188.2 → 188.2 (1.00x) | 0.00049 → 0.00049 |
| 1280 | 1 → 1 | 82.5 → 81.9 (1.01x) | 231.2 → 230.9 (1.00x) | 0.00043 → 0.00043 |
| 1281 | 1 → 1 | 85.8 → 85.1 (1.01x) | 238.1 → 238.4 (1.00x) | 0.00048 → 0.00048 |
| 1408 | 1 → 2 | 91.6 → 89.8 (1.02x) | 254.5 → 99.0 (2.57x) | 0.0003 → 0.0003 |
| 2048 | 1 → 2 | 125.0 → 97.2 (1.29x) | 358.1 → 104.8 (3.42x) | 0.00045 → 0.00045 |
| 2304 | 1 → 2 | 142.9 → 94.6 (1.51x) | 400.2 → 110.2 (3.63x) | 0.00024 → 0.00024 |
| 2305 | 1 → 2 | 140.7 → 145.2 (0.97x) | 402.9 → 147.3 (2.73x) | 0.00036 → 0.00036 |
| 3072 | 1 → 2 | 187.8 → 139.6 (1.35x) | 529.6 → 156.3 (3.39x) | 0.00024 → 0.00024 |
| 4096 | 1 → 2 | 242.7 → 142.1 (1.71x) | 699.4 → 162.7 (4.30x) | 0.00024 → 0.00024 |
| 8192 | 8 → 2 | 4606.0 → 235.0 (19.60x) | 4611.6 → 262.8 (17.55x) | 0.00023 → 0.00023 |
| 16384 | 8 → 2 | 9020.3 → 476.7 (18.92x) | 9014.9 → 525.2 (17.16x) | 0.00012 → 0.00012 |

Width 256 at 2305 keys (the first key above the 36-workgroup value-phase limit) is the only slower cell: independent 140.7 → 145.2 µs (0.97x), while chained is 402.9 → 147.3 µs (2.73x); in the calibration `ED` sweep, `split` beat `default` there in both runs (132.8–139.0 against 139.7–140.2 µs).

## Other head layouts (`layout/`, `layout/summary.txt`)

Default route of `main` (`69a7123e8`) against `head` (`60d2ae306`) per call at head layouts outside the calibration shapes, rounds `main head head main` (`LY1`..`LY4`), median of two processes per build, 1-minute load `< 1.0` at every start (`scripts/summarize_layout.py`).

| width, heads | keys | dispatches | independent µs (`main` → `head`) | chain µs (`main` → `head`) |
|---|---:|---|---:|---:|
| 256, 16q / 4kv (`Qwen3.8-9B`) | 1407 | 1 → 1 | 167.4 → 169.1 (0.99x) | 252.3 → 252.4 (1.00x) |
| 256, 16q / 4kv | 1408 | 1 → 2 | 170.2 → 136.8 (1.24x) | 254.9 → 145.7 (1.75x) |
| 256, 16q / 4kv | 4096 | 1 → 2 | 464.4 → 221.4 (2.10x) | 704.2 → 239.6 (2.94x) |
| 256, 32q / 8kv | 1407 | 1 → 1 | 323.8 → 325.4 (1.00x) | 518.3 → 518.5 (1.00x) |
| 256, 32q / 8kv | 1408 | 1 → 2 | 325.5 → 200.1 (1.63x) | 520.5 → 210.6 (2.47x) |
| 256, 32q / 8kv | 4096 | 1 → 2 | 896.1 → 436.7 (2.05x) | 1428.8 → 454.5 (3.14x) |
| 128, 8q / 8kv (group 1) | 192 | 1 → 2 | 17.2 → 24.3 (0.71x) | 41.9 → 26.7 (1.57x) |
| 128, 8q / 8kv | 448 | 1 → 2 | 29.3 → 30.8 (0.95x) | 80.1 → 32.3 (2.48x) |
| 128, 8q / 8kv | 640 | 1 → 2 | 39.1 → 33.6 (1.16x) | 104.8 → 37.2 (2.82x) |
| 128, 8q / 8kv | 4096 | 1 → 2 | 190.2 → 149.4 (1.27x) | 574.1 → 160.9 (3.57x) |
| 128, 16q / 8kv (group 2) | 192 | 1 → 2 | 24.7 → 33.0 (0.75x) | 41.6 → 35.0 (1.19x) |
| 128, 16q / 8kv | 256 | 1 → 2 | 30.0 → 34.2 (0.88x) | 50.1 → 36.3 (1.38x) |
| 128, 16q / 8kv | 320 | 1 → 2 | 37.1 → 34.5 (1.08x) | 64.1 → 36.7 (1.74x) |
| 128, 16q / 8kv | 4096 | 1 → 2 | 329.4 → 158.1 (2.08x) | 571.0 → 170.6 (3.35x) |
| 128, 64q / 8kv (group 8) | 192 | 1 → 2 | 70.9 → 97.8 (0.73x) | 97.8 → 98.8 (0.99x) |
| 128, 64q / 8kv | 256 | 1 → 2 | 87.7 → 99.0 (0.89x) | 117.7 → 101.0 (1.17x) |
| 128, 64q / 8kv | 320 | 1 → 2 | 113.9 → 101.3 (1.13x) | 153.4 → 100.5 (1.53x) |
| 128, 64q / 8kv | 4096 | 1 → 2 | 1220.8 → 410.0 (2.98x) | 1565.4 → 412.5 (3.79x) |

At width 256 with 4 or more key-value heads, the 36-workgroup value-phase limit is already passed at 1408 keys, so the plain pass 1 runs there and beats the fused default in both shapes from 1408 keys onward. At width 128, the 64-key blocks require four query heads per key-value head, so groups 1, 2, and 8 run the value-phase pass 1 from 192 keys; independent calls are slower over the short bands shown above (192–448 for group 1, 192–256 for groups 2 and 8), while chained calls are faster from 192 keys for groups 1 and 2 and from 256 keys for group 8 (0.99x at 192). A decode step is the chain shape; G15S keeps the 192-key start (the same start as G13S and G14G, calibrated at four query heads per key-value head), and `device.h` records these layout facts. Per-call max |error| vs float64 across `layout/` stays below 0.00185 (maximum 0.00184774). Full 56-row breakdown: `layout/summary.txt`.

## Model decode A/B (`ab/`, `ab/summary.txt`)

`scripts/ab_decode.py`, all split-K environment variables unset, one process per run, rounds ordered `main head head main` (`AB1`..`AB4`). Per process and context: 1 warm-up plus 2 timed greedy generations of 128 tokens (median), followed by 32 teacher-forced steps saving full-vocabulary log-probs. Prompts are the first `CTX` tokens of `$LANE/scripts/context-text.txt`, produced by `git show 79d99c90c:docs/architecture.md 79d99c90c:docs/roadmap.md 79d99c90c:CONTRIBUTING.md` (10,326 tokens with the `Qwen3-4B` tokenizer, 10,630 with `Qwen3.8-2B`). Median of the two process medians:

| model | context | decode tok/s (`69a7123e8` → `60d2ae306`) | change | prefill tok/s | dispatches / token | token sha256 (`main` / `head`) |
|---|---:|---|---:|---|---|---|
| `Qwen3-4B-Instruct-2507` 4-bit | 443 | 39.79 → 46.66 | +17.3% | 188.8 → 189.2 (+0.20%) | 343.62 → 379.91 | `d80ab845f46b8acc` / `29db0385cb42cdfb` |
| `Qwen3-4B-Instruct-2507` 4-bit | 1024 | 32.73 → 45.20 | +38.1% | 190.4 → 190.8 (+0.21%) | 343.62 → 379.91 | `9c7a5b6743aedf4f` / `9c7a5b6743aedf4f` |
| `Qwen3-4B-Instruct-2507` 4-bit | 2048 | 25.10 → 41.73 | +66.3% | 154.3 → 154.6 (+0.16%) | 343.62 → 379.91 | `cf4b70296da63ea8` / `d977632eca0ab980` |
| `Qwen3-4B-Instruct-2507` 4-bit | 4000 | 17.47 → 35.53 | +103.4% | 142.5 → 142.7 (+0.16%) | 351.09 → 387.37 | `4d04fa30f99c979e` / `a43858f19e7ebaf2` |
| `Qwen3.8-2B` 4-bit | 443 | 77.37 → 77.33 | -0.1% | 368.1 → 370.3 (+0.61%) | 239.07 → 239.07 | `6449b58716f8a770` / `6449b58716f8a770` |
| `Qwen3.8-2B` 4-bit | 1024 | 73.92 → 73.73 | -0.2% | 384.0 → 384.5 (+0.12%) | 239.07 → 239.07 | `0c054de8ef7f779d` / `0c054de8ef7f779d` |
| `Qwen3.8-2B` 4-bit | 2048 | 68.83 → 77.17 | +12.1% | 338.5 → 338.4 (-0.02%) | 239.07 → 245.12 | `8ac29360d5b1ffae` / `12294c0391deb6f0` |
| `Qwen3.8-2B` 4-bit | 4000 | 60.90 → 75.34 | +23.7% | 332.4 → 332.6 (+0.04%) | 244.96 → 251.01 | `bb33c841a417b518` / `e49b40e48367d7aa` |

On `Qwen3-4B-Instruct-2507`, all 36 attention layers add 1 dispatch per decode step (343.62 → 379.91 dispatches/token through 2048, 351.09 → 387.37 at 4000). On `Qwen3.8-2B`, the route stays off below 1408 keys at head width 256 (identical 239.07 dispatches/token and bit-identical forced log-probs at contexts 443 and 1024); from 1408 keys, its 6 full-attention layers add 1 dispatch each (239.07 → 245.12 at 2048, 244.96 → 251.01 at 4000).

## Numerics (`ab/logits.txt`, `parity/summary.txt`, `variants/summary.txt`)

Token ids are identical within each build across all of its processes. Between `main` and `head`, token ids are identical at `Qwen3-4B` context 1024 and `Qwen3.8-2B` contexts 443 and 1024, and differ at `Qwen3-4B` contexts 443, 2048, and 4000 and `Qwen3.8-2B` contexts 2048 and 4000.

Cross-build teacher-forced log-probs (32 steps, full vocabulary, `ab/logits.txt`, four sections per model):

- Same build across processes (`AB1` vs `AB4`, `AB2` vs `AB3`) and CPU-disabled head (`N2` vs `AB2`): bit-identical at every measured context on both models.
- `main` (`AB1`) vs `head` (`AB2`), `Qwen3-4B`: KL mean 3.2e-4 to 6.1e-4, KL max 2.8e-3, argmax equal 31–32/32, max |dlogprob| of the top token 0.084.
- `main` (`AB1`) vs `head` (`AB2`), `Qwen3.8-2B`: bit-identical at contexts 443 and 1024; at 2048 and 4000, KL mean 4.9e-4 / 5.3e-4, KL max 2.2e-3 / 3.2e-3, argmax equal 32/32, max |dlogprob| of the top token 0.063 / 0.114.

Float32 reference (`parity/summary.txt`, `scripts/parity_ref.py` on base build `85e0f4480`, every route in one process with variables switched live; reference is the same model cast with `set_dtype(float32)`). `KL(float32 || arm)` mean and argmax agreement over 32 forced steps, with the arm each context runs on `head` starred (`*`):

| model | context | `default` (KL mean, argmax) | head arm (`*`) (KL mean, argmax) |
|---|---:|---|---|
| `Qwen3-4B-Instruct-2507` 4-bit | 443 | 3.44e-4, 31/32 | `bk64*`: 3.27e-4, 31/32 |
| `Qwen3-4B-Instruct-2507` 4-bit | 1024 | 3.17e-4, 31/32 | `pv*`: 4.27e-4, 32/32 |
| `Qwen3-4B-Instruct-2507` 4-bit | 2048 | 2.98e-4, 32/32 | `split*`: 3.93e-4, 32/32 |
| `Qwen3-4B-Instruct-2507` 4-bit | 4000 | 3.18e-4, 31/32 | `split*`: 3.22e-4, 32/32 |
| `Qwen3.8-2B` 4-bit | 2048 | 4.68e-4, 31/32 | `pv*`: 4.44e-4, 31/32 |
| `Qwen3.8-2B` 4-bit | 4000 | 4.65e-4, 32/32 | `split*`: 4.89e-4, 32/32 |

Contexts 443 and 1024 on `Qwen3.8-2B` run the `default` arm on `head` (4.49e-4, 31/32 and 3.91e-4, 31/32 against float32). Across all contexts and variants, every bf16 arm sits 2.6e-4 to 5.2e-4 KL from the float32 model, and `default` vs split variants differ from each other by 3.2e-4 to 6.1e-4 — about the distance of each arm from float32. In forced-variant runs (`variants/summary.txt`) at 2048 tokens on `Qwen3-4B`, `split` and `pv` produce the same 128 tokens (`d977632eca0ab980`) while the fused `default` differs (`cf4b70296da63ea8`). The `composition` arm (`MLX_OMARCHY_SDPA_DECODE_NATIVE=0`) in `parity/summary.txt` is bit-identical to `default` for bf16 because that switch does not govern bf16 decode.

Width-256 `pv` covers 1408–2304 keys at 2 kv heads (`sdpa_split_pv_wide_max_workgroups = 36`; at context 4000, `Qwen3.8-2B` runs 64 workgroups, past the 36-workgroup limit, so plain `split` runs regardless). Unlike G16S (`receipts/2026-11-15-m4-pro-g16s-decode-defaults`), which kept width-256 value-phase pass 1 off because it moved tokens while plain `split` kept them, on G15S forced plain `split` moves `Qwen3.8-2B` tokens at 2048 as well (`variants/summary.txt`: context 2048 differs from token 4), so keeping `pv` off does not keep the token ids; against the float32 reference (`parity/summary.txt`), `pv` is closer to float32 than both `default` and `split` at 2048 (4.44e-4 vs 4.68e-4 and 5.19e-4). Precedent: `receipts/2026-11-14-g14c-sdpa-hd256-pv-limit` (the G14C default for the same variant) records `Qwen3.8-2B` ids that differ ("rounding order") at 2048 and 4096.

## Numerics gate (`gate/`, `gate/summary.txt`)

Measured under `docs/numerics-gate.md` on the prompt text above.

Primary reference (`scripts/ref_setup.sh`, `gate/ref_setup.log`): stock upstream MLX `0.32.3` (`mlx` and `mlx-cpu` wheels from PyPI) with stock `mlx-lm` `0.31.3` on the CPU stream, every floating parameter in float32, in a dedicated reference virtual environment containing no omarchy package. Each downloaded wheel's SHA-256 equals the hash PyPI publishes and the hash recorded in `receipts/2026-11-01-m4-decode-defaults`, and every installed file matches its `RECORD` (`mlx` 52, `mlx-cpu` 262, `mlx-lm` 193, 0 differ). This is the same reference used in `receipts/2026-11-01-m4-decode-defaults/numerics-gate`.

Stock MLX runs this CPU reference on one CPU core at about 4.5 s per token (prefill included):

- `Qwen3.8-2B`: all 5,120 steps (cache 128–5,247), 20:07Z to 23:00Z (`gate/NG-cpuref-qwen3.8-2b-4bit.*`).
- `Qwen3-4B`: steps 0–2,559 (cache 128–2,687: the fused range, 64-key blocks, value-phase range, and the first 1,407 keys of the plain range; `gate/NG-cpuref-qwen3-4b-2507-4bit-a.*`, compared against the first-2,560-step subsets `gate/NG-{main,head}-qwen3-4b-2507-4bit-first2560.npz`). Cache 2,688–5,247 did not fit in the CPU run window (`gate/void-timelimit-*`, `gate/void-stopped-*`, `gate/void-killed-*`); for cache 2,688–5,247 on `Qwen3-4B`, the receipt keeps the GPU float32 reference below.

Second reference (all 5,120 steps, cache 128–5,247 on both models): the same model with `model.set_dtype(float32)` on the omarchy GPU wheel (`gate/NG-fp32-*`). Float32 decode attention never takes the bf16-only split-K route, so this reference shares no code with the change; because `docs/numerics-gate.md` does not name it as one of its two references, it is reported beside the CPU reference rather than instead of it. The two references pick the same top-1 token at every common position (`Qwen3.8-2B` 5,120/5,120; `Qwen3-4B` first half 2,560/2,560), and their perplexities differ by 0.067% (`Qwen3.8-2B`) and 0.025% (`Qwen3-4B` first half).

Capture (`scripts/numerics_gate.py`): prefill 128 tokens, then run 5,120 teacher-forced steps one token at a time on the prompt cache (with `GATE_PREFIX` and `GATE_STEPS` selecting a segment when split). Step `i` is scored from the logits produced before token `i` is fed: step 0 from the 128-token prefill, and step `i >= 1` from the decode call attending `128 + i` keys. Scored cache lengths are therefore 128–5,247 (not 129–5,248; `numerics_gate.py` records `keys = PREFIX + 1 + arange(STEPS)`, and `scripts/gate_compare.py` subtracts 1 to derive the true cache lengths). Regimes exercised: width 128 fused below 192, 64-key blocks 192–640, value-phase 641–1280, plain 1281–5,247; width 256 fused below 1408, value-phase 1408–2304 at 2 kv heads, plain 2305–5,247. Tested arms: `main` (`69a7123e8`) bf16 and `head` (`60d2ae306`) bf16 (defaults, all split-K variables unset).

Harness validation compares S=1 teacher-forced perplexity against chunked-prefill perplexity on the same 5,120 targets (`MLX_OMARCHY_FULL_LOGITS=1` for the prefill check only, because the packaged last-logits patch keeps one row on cached calls): `Qwen3-4B` `main` 27.5695 vs 27.5731, `head` 27.5728 vs 27.5731; `Qwen3.8-2B` `main` 26.8196 vs 26.8609, `head` 26.8376 vs 26.8609. All four bf16 captures were run twice and reproduced bit for bit (argmax, top-1/top-2 logits, target log-probs). Baseline noise (`docs/parity-contract.md` § 4.2): five identical-protocol `main` captures per model (`gate/baseline/`, `NB1`–`NB5`) each equal the gate's `main` capture bit for bit (`gate/baseline/identity.txt`), so run-to-run noise and the check-3 allowance are 0.

Comparison (`scripts/gate_compare.py`, `gate/summary.txt`) computes per arm against each reference: top-1 agreement with Wilson 95% interval, flips and near ties (a flip whose reference top-2 margin is at most 2 bf16 ULP of the reference top logit, inclusive, matching `receipts/2026-11-01-m4-decode-defaults/numerics-gate`), perplexity and `|log(PPL / PPL_ref)|`, mean `|dlogprob|` of the text token, a whole-sample paired block bootstrap (128-position blocks, 10,000 resamples, seed 0) of `log(PPL_head) - log(PPL_main)`, and per-key-band metrics (per-band rows report the mean change only because bands are not whole 128-position blocks; per-band bootstrap intervals are removed).

Results (`main` `69a7123e8` default → `head` `60d2ae306` default):

| model, reference, steps | top-1 agreement `main` → `head` (raw; `main` Wilson 95% lower bound) | flips `main` → `head` (all near ties) | PPL `main` → `head` (ref) | \|log(PPL / PPL_ref)\| `main` → `head` | log-PPL change `head - main`, bootstrap 95% |
|---|---|---|---|---|---|
| `Qwen3.8-2B`, CPU, 5,120 (cache 128–5,247) | 4993 (97.520%) → 4996 (97.578%); bound 97.057% | 127 → 124 | 26.8196 → 26.8376 (26.8251) | 0.021% → 0.047% | 6.7e-4 `[-6.7e-5, 1.5e-3]` |
| `Qwen3-4B`, CPU, 2,560 (cache 128–2,687) | 2522 (98.516%) → 2520 (98.438%); bound 97.969% | 38 → 40 | 38.9166 → 38.9027 (38.8671) | 0.127% → 0.092% | -3.6e-4 `[-2.2e-3, 1.4e-3]` |
| `Qwen3.8-2B`, GPU float32, 5,120 (cache 128–5,247) | 4993 (97.520%) → 4996 (97.578%); bound 97.057% | 127 → 124 | 26.8196 → 26.8376 (26.8431) | 0.088% → 0.020% | 6.7e-4 `[-6.7e-5, 1.5e-3]` |
| `Qwen3-4B`, GPU float32, 5,120 (cache 128–5,247) | 5029 (98.223%) → 5039 (98.418%); bound 97.823% | 91 → 81 | 27.5695 → 27.5728 (27.5403) | 0.106% → 0.118% | 1.2e-4 `[-1.0e-3, 1.2e-3]` |

Old-vs-new top-1 agreement: `Qwen3.8-2B` 5,067/5,120; `Qwen3-4B` 5,046/5,120 (5,120 steps), 2,528/2,560 (first half).

Check 2 (token choices) passes in every row and every key band (`head` raw top-1 agreement is at or above `main`'s Wilson 95% lower bound):

| model, reference | key band | top-1 agreement `main` → `head` | `main` Wilson 95% lower bound |
|---|---|---|---|
| `Qwen3-4B`, CPU (cache 128–2,687) | 128–191 | 96.875% → 96.875% (bit-identical) | 89.303% |
| `Qwen3-4B`, CPU | 192–640 | 98.441% → 98.218% | 96.817% |
| `Qwen3-4B`, CPU | 641–1280 | 99.062% → 98.438% | 97.970% |
| `Qwen3-4B`, CPU | 1281–2687 | 98.365% → 98.579% | 97.559% |
| `Qwen3-4B`, GPU float32 (cache 128–5,247) | 1281–5247 | 98.084% → 98.462% | 97.609% |
| `Qwen3.8-2B`, CPU & GPU float32 | 128–1407 | 97.344% → 97.344% (bit-identical) | 96.311% |
| `Qwen3.8-2B`, CPU & GPU float32 | 1408–2304 | 96.990% → 97.213% | 95.656% |
| `Qwen3.8-2B`, CPU & GPU float32 | 2305–5247 | 97.757% → 97.791% | 97.157% |

(`Qwen3-4B` GPU float32 bands 128–191, 192–640, and 641–1280 match the CPU rows because the two references pick identical top-1 tokens across all 2,560 common positions.) Every flip of every arm is a near tie: on `Qwen3.8-2B`, the reference top-2 margin at the flips has median 0.027 logits and max 0.189, against a 2-ULP threshold near 0.25.

Check 3 (perplexity): five identical-protocol `main` captures per model (`gate/baseline/`, `NB1`–`NB5`) each equal the gate's `main` capture bit for bit (`gate/baseline/identity.txt`), so the run-to-run allowance (`docs/parity-contract.md` § 4.2) is 0 and check 3 requires `head`'s deviation `|log(PPL / PPL_ref)|` to be at most `main`'s. Row by row (`main` → `head`):

- `Qwen3-4B`, CPU reference, cache 128–2,687: `0.127% → 0.092%` (`-0.036` points): met.
- `Qwen3.8-2B`, GPU float32 reference, 5,120 steps (cache 128–5,247): `0.088% → 0.020%` (`-0.067` points): met.
- `Qwen3.8-2B`, CPU reference, 5,120 steps (cache 128–5,247): `0.021% → 0.047%` (`+0.026` points): not met.
- `Qwen3-4B`, GPU float32 reference, 5,120 steps (cache 128–5,247): `0.106% → 0.118%` (`+0.012` points): not met. On its first 2,560 positions (cache 128–2,687), the GPU reference also favours `head` (`0.102% → 0.066%`), as the CPU reference does; the increase comes from cache 2,688–5,247 (`0.109% → 0.169%`), where `head`'s top-1 agreement with the reference is higher (`2,507 → 2,519` of `2,560`).

Changes in deviation (`head` minus `main`): `+0.026`, `-0.036`, `-0.067`, and `+0.012` percentage points (`Qwen3.8-2B` CPU, `Qwen3-4B` CPU, `Qwen3.8-2B` GPU, `Qwen3-4B` GPU). The two float32 references differ from each other by `0.067%` (`Qwen3.8-2B`) and `0.025%` (`Qwen3-4B` first half) in perplexity while picking the same top-1 token at every common position (context, not the check-3 allowance). Every whole-sample paired block-bootstrap 95% interval of `log(PPL_head) - log(PPL_main)` contains 0: `Qwen3.8-2B` `6.7e-4 [-6.7e-5, 1.5e-3]`, `Qwen3-4B` GPU `1.2e-4 [-1.0e-3, 1.2e-3]`, and `Qwen3-4B` CPU `-3.6e-4 [-2.2e-3, 1.4e-3]`. Per-band rows in `gate/summary.txt` report the mean change only (bands are not whole 128-position blocks; per-band intervals are removed).

Acceptance: check 3 is not met on two of the four rows; the change is accepted under the precedent listed in `docs/numerics-gate.md` (the routed-MoE gather kernel, merged with a larger PPL deviation than the deployed kernel, `+0.416%` against `+0.245%`, because it agreed with the reference more often). Both rows match that precedent: top-1 agreement rises (`Qwen3.8-2B` CPU `97.520% → 97.578%`; `Qwen3-4B` GPU `98.223% → 98.418%`), check 2 passes in every row and band, and every flip is a near tie, while deviation rises by `0.026` and `0.012` percentage points.

## CPU-disabled wheel and dispatch trace (`nocpu/`, `trace/`)

The CPU-disabled head wheel (`nocpu/build.txt`, `mlx_omarchy-0.32.4.dev202611221553+nocpu.60d2ae306`, `MLX_BUILD_CPU=OFF`, `docs/parity-contract.md` § 2.1) compiles `no_cpu/compiled.cpp`, `no_cpu/device_info.cpp`, and `no_cpu/primitives.cpp` (and only `encoder.cpp` and `eval.cpp` from `cpu/`).

- Controls (`docs/parity-contract.md` § 2.1, `scripts/cpu_refused.py gpu|cpu`, one process each): `nocpu/N00-gpu-add.log` runs GPU float32 add on 4096 elements with expected values (first `[0.5, 1.5, 2.5]`) in 3 Vulkan compute dispatches; `nocpu/N01-cpu-refused.log` runs `mx.add` on two-element float32 inputs on `mx.new_stream(mx.cpu)` and raises `RuntimeError: Add has no CPU implementation.` (the pinned message). The earlier `nocpu/void-indexerr-N00-*` control (`stream=mx.cpu`, `IndexError`) is kept and labelled void.
- Model decode at contexts 443 and 2048 (`nocpu/N2-*.json`, `ab/summary.txt`, `ab/logits.txt`): `Qwen3-4B` decodes at 46.65 / 41.85 tok/s and `Qwen3.8-2B` at 77.39 / 77.28 tok/s, with token ids and forced log-probs bit-identical to the Release head wheel (`AB2`).
- Dispatch trace (`trace/`, `nocpu/N1-trace-*.log`, using `receipts/2026-09-30-speech-input-gpu/harness/count_cpu.gdb.py` and `scripts/trace_split.py`): the `stream=mx.cpu` control (`trace/TR0-cpu-control.log`) records 3 CPU command encoder calls; 32-token decode at contexts 443 and 2048 (`Qwen3-4B`) and context 2048 (`Qwen3.8-2B`) records 0 CPU command encoder calls on `main`, `head`, and the CPU-disabled `head`. Vulkan compute dispatches per 32 tokens move from 10,656 to 11,808 on `Qwen3-4B` (36 layers × 1 extra × 32) and from 7,520 to 7,712 on `Qwen3.8-2B` (6 full-attention layers × 1 extra × 32).

## Battery (`battery/`, `battery/summary.txt`)

Head C++ test tree built from `60d2ae306` (Release, fresh build directory), one process per suite under the GPU lock (`scripts/run_battery.sh`):

- Standing battery (`AGENTS.md` § Test rules, `battery/S01-*` to `S25-*`): 25 of 25 suites exit 0, 750 of 750 cases pass.
- `omarchy_fast_ops_tests` (`battery/S04-fast_ops.log`) exits 0 with 4 failed assertions under `doctest::may_fail` in `sdpa vjp dk dv match finite differences at rep=1 (path per gate)`, matching the open SDPA backward VJP defect in `docs/known-defects.md` ("SDPA backward fused VJP value defects at small rep=1 shapes") with the same four values `main` recorded (`0.0528708`, `0.0496869`, `0.18067`, `0.0457306`).
- Capability-simulation profiles (`omarchy_capability_sim_tests`, `battery/P01-*` to `P08-*`): 8 of 8 profiles exit 0, 88 of 88 cases pass.
- `omarchy_sdpa_split_tests` (`battery/H02-sdpa-split.log`): 16 of 16 cases, 120,312 of 120,312 assertions pass (its on-device case verifies that the unset flag follows the part rule at 300 and 1000 keys with the route's bits and dispatch count).
- New G15S case (`battery/H01-g15s-case.log`, `overlay/tests/omarchy/test_sdpa_split.cpp:486`): 36 of 36 assertions pass (thresholds; each edge one key either side: 191/192, 1407/1408 width 256, 1280/1281 pv, 640/641 bk64, 2304/2305 at 2 kv heads and 1152/1153 at 4 kv heads for width-256 pv; G15G all 0; all five variables override the part value, and the wide threshold falls back to the width-128 variable on a part without its own row).
- Renamed test rerun (pre-rebase build `scripts/build_split_test.sh`, build log `battery/build-split-test.txt`): `battery/H03-g15s-case-renamed.log` selects the comma-free full name `"split-K decode SDPA: the M3 Pro (G15S) defaults and their variable overrides; the M3 (G15G) stays uncalibrated"` directly (1 of 1 case, 36 of 36 assertions pass), and `battery/H04-sdpa-split-renamed.log` passes the whole `omarchy_sdpa_split_tests` suite (16 of 16 cases, 120,312 of 120,312 assertions).
- Device reopen: `omarchy_runtime_tests --test-case="a fresh process reopens the Vulkan device"` passes (`battery/H40-reopen-runtime.log`), and `mlx-omarchy-info --trace-smoke` completes a 65,536-byte buffer round trip on `Apple M3 Pro (G15S B1)` (`battery/H41-reopen-smoke.log`).

## Rebased tree (`battery-rebased/`)

The PR commit's overlay on main `3ae5c9ea3`, built fresh (`battery-rebased/build.txt`, `scripts/build_rebased_tests.sh`; the build commit it names, `d6a7d7e11`, has the PR commit's `overlay/`) and run one process per suite under the GPU lock (`scripts/run_battery_rebased.sh`): `omarchy_sdpa_split_tests` 16 of 16 cases (239,104 assertions) with the G15S case 36 of 36; 24 of the 25 standing suites exit 0 (770 of 771 cases), `omarchy_fast_ops_tests` with the same four `may_fail` values as above; 8 of 8 capability-simulation profiles (88 cases); device reopen and `--trace-smoke` pass; no GPU kernel log line. `omarchy_matmul_family_tests` fails one case, `single-row matvec sums 1..32 exactly through the subgroup reduction` (`REQUIRE_EQ(got.size(), 1u)`: 32 against 1), the case #681 added; plain main `3ae5c9ea3` without this change fails it the same way (`battery-rebased/M3-matmul-main-case.*`, and 62 of 63 for the suite in `M4-matmul-main-suite.*`, `scripts/build_main_matmul.sh`), as does this tree (`M5-matmul-rebased-case.*`).

## Replay (`scripts/`)

The scripts in `scripts/` take five environment variables:

- `LANE`: scratch folder holding `scripts/`, `logs/`, `evidence/`, `scratch/`, and `runs/` (copy `scripts/` from this receipt into `"$LANE/scripts/"`)
- `REPO`: path to this repository checkout
- `MODELS`: folder holding `qwen3-4b-2507-4bit/` and `qwen3.8-2b-4bit/`
- `BUILD_VENV`: Python 3.14 virtual environment with wheel build dependencies
- `BASE_VENV`: Python 3.14 virtual environment whose non-MLX package list seeds each side's venv

Run order:

1. Generate the prompt text file: `git -C "$REPO" show 79d99c90c:docs/architecture.md 79d99c90c:docs/roadmap.md 79d99c90c:CONTRIBUTING.md > "$LANE/scripts/context-text.txt"`
2. `scripts/build_wheel.sh base 85e0f4480` (calibration build), then `scripts/chain_builds.sh 69a7123e8 60d2ae306` (builds `main`, `head`, the `head` test tree, and the CPU-disabled `head` wheel); run each build under the GPU lock as `chain_builds.sh` does
3. `scripts/run_sweep.sh base` (`SW`), `scripts/run_sweep.sh base FN 144,160,176,192,608,640,672,704,736,768,800,1088,1152,1280,1408,1536,1792,2048,2560,3072,3584,4096 896,1024,1152,1280,1408,1536,1664,1792,2048,2304,2560,2816,3072,3584,4096` and `scripts/run_sweep.sh base ED 176,192,640,641,1280,1281 1280,1344,1408,1472,1536,2304,2305,2432,2433,2560`
4. `scripts/run_variants.sh base` (forced-variant decode), `scripts/run_parity.sh base qwen3-4b-2507-4bit:ref:default,composition,split,pv,bk64 qwen3.8-2b-4bit:ref:default,split,pv` (32-step float32 reference parity)
5. `scripts/run_battery.sh` (standing battery, capability profiles, `omarchy_sdpa_split_tests`, reopen), and `scripts/build_split_test.sh` followed by rerunning `omarchy_sdpa_split_tests` (`H03`, `H04`) on the comma-free test name
6. `scripts/run_ab.sh` (two-build model decode A/B, per-call `DF1`..`DF4`, and `count_cpu.gdb.py` dispatch traces)
7. `scripts/run_nocpu.sh` (CPU-disabled wheel § 2.1 controls, traces, and decode)
8. `scripts/run_gate.sh` (GPU float32 references, bf16 numerics-gate captures, and `layout/` sweep) and `scripts/run_gate2.sh` (bf16 numerics-gate captures with `MLX_OMARCHY_FULL_LOGITS=1` for prefill validation)
9. `scripts/ref_setup.sh` (builds `$LANE/venv-ref` with stock PyPI `mlx 0.32.3`, `mlx-cpu 0.32.3`, and `mlx-lm 0.31.3`, checking SHA-256 and `RECORD` hashes), `scripts/run_gate3.sh` (stock-MLX CPU float32 reference captures; `Qwen3.8-2B` completes all 5,120 steps), and `scripts/run_gate4.sh` (`Qwen3-4B` CPU reference split into 2,560-step halves via `GATE_PREFIX` / `GATE_STEPS` in `scripts/numerics_gate.py`; first half `a` completes steps 0–2,559, cache 128–2,687). `scripts/run_gate3.sh` and `scripts/run_gate4.sh` take the GPU lock before starting and fail when a capture is missing; the recorded runs used earlier versions (`run_gate4.sh` took the lock through a keeper started after the halves while the `Qwen3.8-2B` reference held it, and both logged exit codes without failing).

`scripts/summarize_sweep.py`, `scripts/summarize_df.py`, `scripts/summarize_layout.py`, `scripts/summarize_variants.py`, `scripts/summarize_ab.py`, `scripts/cmp_ab.py`, and `scripts/gate_compare.py` rebuild the text summaries from the run outputs: `gate_compare.py` reads the NPZ files in `gate/` with band edges `192,641,1281` (`qwen3-4b-2507-4bit`) and `1408,2305` (`qwen3.8-2b-4bit`), using the first-2,560-step slices `gate/NG-{main,head}-qwen3-4b-2507-4bit-first2560.npz` (the first 2,560 entries of each array from the 5,120-step `main` and `head` NPZ files) to compare against `gate/NG-cpuref-qwen3-4b-2507-4bit-a.npz`; `cmp_ab.py` reads the teacher-forced log-prob NPZ files written by `ab_decode.py` (33 to 61 MB each, not included here; `ab/logits.txt` holds their comparison).

## Files

`env/`, `sweep/`, `variants/`, `parity/`, `default/`, `layout/`, `ab/`, `gate/` (including `gate/baseline/`), `nocpu/`, `trace/`, `battery/`, `battery-rebased/`, `scripts/`, and `SHA256SUMS`. Per-leg device state before and after each GPU command is stored in each directory's `*.state` files. `SHA256SUMS` records the SHA-256 digest of every data file in this receipt folder.
