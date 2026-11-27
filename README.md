# omarchy-mlx

**No Mac? [Start here](docs/ANY-GPU.md)** for NVIDIA, Intel, AMD and software
Vulkan development paths.

Results so far: [docs/TESTED-HARDWARE.md](docs/TESTED-HARDWARE.md).

MLX on the Apple GPU under Linux, and Core ML models on the
Apple Neural Engine.

[MLX](https://github.com/ml-explore/mlx) is Apple's array framework for
machine learning on Apple silicon. Upstream runs on macOS through Metal
and on Linux through CUDA or CPU. omarchy-mlx runs the same
`import mlx.core as mx` code on an Apple silicon Mac running
[Omarchy](https://github.com/omacom/omarchy) Linux, on the Apple GPU. The
GPU driver is Mesa's Borealis Vulkan stack (Aurora's rename of the
Honeykrisp driver; both names are accepted). Tensors stay on the GPU.
There is no Metal and no CPU fallback.

It is for people who run Linux on an M1 or M2 Mac. You get local chat,
serving, and speech to text, with no macOS and no cloud account.

## The projects

Part of the Omarchy on Apple Silicon ecosystem:

- [omarchy-ane](https://github.com/joshuaswarren/omarchy-ane) — Linux driver for the Apple Neural Engine: DRM kernel modules and libane.
- [mil-hwx-compiler](https://github.com/joshuaswarren/mil-hwx-compiler) — compiles Core ML MIL into ANE programs on Linux (M1/M2 targets).
- [metal2vk](https://github.com/joshuaswarren/metal2vk) — translates Metal compute kernels to Vulkan SPIR-V with open tools.
- [mesa-1](https://github.com/joshuaswarren/mesa-1) — Mesa fork carrying the Borealis Vulkan driver work for Apple silicon (the `honeykrisp-omarchy` branch).
- [omarchy-cluster](https://github.com/joshuaswarren/omarchy-cluster) — one local model split across Macs on macOS and Omarchy Linux.
- [omarchy-apple-dev](https://github.com/joshuaswarren/omarchy-apple-dev) — build and ship iOS apps from Omarchy Linux, no Xcode or Mac.
- [omarchy-egpu](https://github.com/joshuaswarren/omarchy-egpu) — eGPU setup, diagnostics and measured local-inference results on Omarchy Linux.
- [ane-linux-experiments](https://github.com/joshuaswarren/ane-linux-experiments) — Neural Engine on Linux: verified primitives, receipt-backed macOS parity.
- [coreglass](https://github.com/joshuaswarren/coreglass) — profiler that shows where local inference loses speed on Apple Silicon Linux.
- [omarchy-inference-fleet](https://github.com/joshuaswarren/omarchy-inference-fleet) — reproducible, pinned single-node MLX inference baselines.
- [remnic](https://github.com/joshuaswarren/remnic) — local-first memory and context for AI agents.
- Platform upstream: [omarchy-mac](https://github.com/omacom/omarchy-mac) — Omarchy on Apple Silicon Macs.

## What runs on Omarchy Linux

mlx-lm, oMLX, mlx-serve, TensorFold, sushi, llama.cpp (Vulkan) and MCDMA have been tested on an Apple silicon Mac under Linux. The status of each, with the commit or measurement behind it, is in [docs/ENGINES.md](docs/ENGINES.md).

## What you get

- The `mlx` Python module with the Vulkan GPU backend. The distribution
  name is `mlx-omarchy`; the module stays `mlx`. Upstream source is
  fetched at a pinned commit, and this project's code lives in
  `overlay/` and `patches/`. The tree stays a small patch-set, not a
  fork.
- MLX Chat, a local web app for chat and comparing options, with a
  browser UI and a terminal mode. The install registers it in the
  launcher and sets up a user service that keeps the loaded pair ready
  across logins.
- `mlx-omarchy-serve`, a serving CLI with `catalog`, `plan`, and
  `serve` commands, memory admission, and approve-first model
  downloads. The same install ships the Laya typed-decision server and
  the Bonsai2 packed-runtime server.
- Core ML on the Neural Engine. `mlx-omarchy-parakeet` transcribes audio
  (`download`, `verify`, `transcribe`). `mlx-omarchy-coreml`
  inspects a Core ML package. The aarch64 wheel ships the ANE worker
  and the pinned Parakeet encoder bundles.
- An opt-in wake word. `mlx-omarchy-chat --wake-word hey_jarvis`
  listens through openWakeWord ONNX models on the CPU (SHA-256-pinned
  files, downloaded once on first use) and hands the microphone to the
  chat when it hears the phrase. It is off by default. The score
  threshold is 0.5 and lives in one place
  (`wake_word.WAKE_THRESHOLD_DEFAULT`). Measure it on your own
  microphone with
  `python -m mlx_omarchy_assistant.wake_eval --manifest clips.json
  --clips-dir clips --model <wake model> --out results.json`
  ([receipt](receipts/2026-11-21-wake-eval-v10/README.md)).
- A hands-free voice loop with no network calls.
  `python -m mlx_omarchy_assistant.voice_loop --home <data home>
  --reply-url http://127.0.0.1:8080/v1/chat/completions` waits for the
  wake word, ends your turn after a short silence, transcribes it with
  the resident Parakeet model, streams the reply from a local
  `mlx_lm.server` or oMLX, and speaks each sentence with Kokoro as soon
  as it is written. Audio plays through PipeWire's default sink. The
  reply URL must be a loopback address. Run it once with `--prepare` to
  download and verify the wake, voice activity, Parakeet and Kokoro
  models; a rerun skips verified files and fetches a half-written one
  again.

Chat models download from Hugging Face the first time you use them. You
approve each download first. The default pairs: Everyday is Qwen3.5-9B
4-bit plus the Laya model. Compact is
Qwen3-4B-Instruct-2507 4-bit plus Laya. Quality is Qwen3.8-27B 4-bit
plus Laya, and it needs a 96 GB machine.

## Supported hardware

| Chip | GPU | ANE |
|---|---|---|
| M1 | Tested on Linux: MacBook Pro 13-inch, 8-core GPU, Mesa Honeykrisp `1650fc1a33`; Qwen3-4B 4-bit decode 24.23 tok/s ([matched table](#how-close-to-macos)) | Parakeet islands |
| M1 Max | Tested on Linux: MacBook Pro 16-inch, 32-core GPU, Mesa Honeykrisp `1650fc1a33`; Qwen3-4B 4-bit decode 100.69 tok/s ([matched table](#how-close-to-macos)) | Whole encoder |
| M2 Max | [Tested on Linux](receipts/2026-11-21-m2-mesa-1650fc1-qualification/README.md): MacBook Pro 14-inch, 38-core GPU, Mesa Honeykrisp `1650fc1a33` — suite clean against the M1 Max reference, kit decode 80.75 tok/s; Qwen3.8-2B 4-bit decode 202.71 tok/s ([matched table](#how-close-to-macos)) | Whole encoder (bit-exact on the M2, see [M2 Max ANE](docs/ANE-LINUX.md#m2-max-ane-measured)) ([results](receipts/2026-10-10-m2-max-ane.md)) |
| M1 Pro | [Tested on Linux](receipts/2026-11-13-m1pro-m2-m4pro-a18pro-linux/README.md): MacBook Pro 16-inch, 16-core GPU, release wheel on Mesa Borealis `be147e3625`, Qwen3-4B 4-bit decode 31.5 tok/s | Untested |
| M2 | [Tested on Linux](receipts/2026-11-13-m1pro-m2-m4pro-a18pro-linux/README.md): MacBook Air 13-inch, 8-core GPU, release wheel on Mesa Borealis `be147e3625`, Qwen3-4B 4-bit decode 19.4 tok/s | Untested |
| M1 Ultra, M2 Pro, M2 Ultra | Untested | Untested |
| M3 | Experimental: aurora mesa-m3 graphics; compute not certified | Data-only; h15 bring-up module |
| M3 Pro | [Tested on Linux](receipts/2026-11-21-m3-pro-g15s-linux/README.md): MacBook Pro 14-inch, 14-core GPU, release wheel on Mesa Borealis `a196911147`, Qwen3-4B 4-bit decode 43.9 tok/s; standing battery 747/747 cases on source build `a893cae5c` | Untested |
| M4 | [Experimental](receipts/2026-11-01-m4-g16g-part1/README.md): Mac mini (J773g) on the out-of-tree G16G techmore/mesa `ea32833` driver (Honeykrisp `git-ea328336ef` + Apple9 compute port); no qualification row posted yet | Data-only; h16 bring-up module |
| M4 Pro | [Tested on Linux](receipts/2026-11-13-m1pro-m2-m4pro-a18pro-linux/README.md): MacBook Pro 16-inch, 20-core GPU, release wheel on Mesa Borealis `c208afc352`, Qwen3-4B 4-bit decode 49.3 tok/s; runtime 59/60 on source build `ec849019` (independent-streams ratio 0.876 against that build's 0.95 floor; main has the 0.80 Borealis floor, not re-run); [qualification-r4 row](https://github.com/joshuaswarren/omarchy-mlx/issues/537) clean bar the `test_export_unique` defect the r7 wheel fixes | Untested |
| M4 Air | Experimental: MacBook Air (J713, T8132, G16G B0) on a bring-up Borealis driver (mesa git sha not recorded); [qualification-r4 row](https://github.com/joshuaswarren/omarchy-mlx/issues/537) complete with only `test_export_unique` new over the reference (the r7 wheel fixes it); Qwen3-4B 4-bit decode 25.68 tok/s on the bring-up driver, not a final rate | Data-only; h16 bring-up module |
| A18 Pro | [Tested on Linux](receipts/2026-11-13-m1pro-m2-m4pro-a18pro-linux/README.md): MacBook Neo, 5-core GPU, release wheel on Mesa Borealis `c208afc352`, Qwen3-4B 4-bit decode 13.5 tok/s; runtime 59/60: the independent-streams timing check fails against the 0.80 Borealis floor (0.437 to 0.442 in four runs), copies are byte-exact | Untested |
| M6 | [Experimental](receipts/2026-11-22-m6-mini-g18g-linux/README.md): Mac mini (G18G, j873g) on a local Borealis ICD (mesa git sha not recorded); first Linux receipt, Qwen3-4B 4-bit decode 5.23 tok/s — early driver, not a final rate | Untested |

The supported installer targets are M1-class Macs and the M2 Max. They
share one GPU class and driver path. The Tested rows are the machines
this project measures on. The M4 row needs the out-of-tree G16G driver
stack named in its receipt. The M1 Pro, M2, M4 Pro and A18 Pro rows
run the release wheel on Fedora with aurora's Borealis Vulkan driver
(Linux 7.3); the setup is in [docs/INSTALL.md](docs/INSTALL.md#fedora-with-the-borealis-driver). The M3 Pro row runs the release wheel on Omarchy with the `linux-aurora` 7.2.11 kernel and the `mesa-aurora` and `vulkan-aurora` Borealis packages through the system ICD. Each of these rows is one machine with the GPU core count shown; the other core counts of those chips are not measured. On an M3 or M4 Mac, run the tester kit,
`bash scripts/m3m4_kit.sh` (see [scripts/m3m4_kit.md](scripts/m3m4_kit.md)),
to send us the numbers and ANE state that only real silicon gives. On an M4 that runs Linux the kit runs the collector only, and the ANE is not supported there yet. The beta path for M4 on Linux, with its steps and what is not measured, is in [docs/m4-linux.md](docs/m4-linux.md).

**M5 users: use the edge channel for now.** The M5 is not in the table above and has no receipt in this repo yet. The Omarchy rc and stable channels still carry omarchy-mlx 0.7.28, which cannot use an M5 GPU (it accepts only the stock Honeykrisp driver id; the Aurora driver id the M5 reports was added in v0.7.39). Edge carries 0.7.40-3, which ran on an M5 Max GPU in a lab run with the system driver. Switch back when rc and stable move.

The ANE is separate hardware from the GPU. It has its own driver and its
own tested state. Chip-by-chip ANE status, including what each untested
chip needs, lives in the [chip coverage
table](https://github.com/joshuaswarren/omarchy-ane#chip-coverage) of
[omarchy-ane](https://github.com/joshuaswarren/omarchy-ane).

## Install

Run one command on an Arch install for an M1-class Mac or an M2 Max. The script picks the latest release, checks the wheel against the release
`SHA256SUMS`, and requires Python 3.14 on aarch64.

```bash
curl -fsSL https://raw.githubusercontent.com/joshuaswarren/omarchy-mlx/main/install.sh | bash
```

The other ways to install (the Omarchy package, the script flags, a
release wheel) are in [docs/INSTALL.md](docs/INSTALL.md).

## Quick start

After any install path, this works:

```bash
mlx-omarchy-info                                                   # GPU and driver state
mlx-omarchy -c "import mlx.core as mx; print(mx.default_device())" # Device(gpu, 0)
mlx-omarchy-chat                                                   # chat in the browser
```

The first chat run asks you to approve a model download. Then it
streams the answer from the Apple GPU.

## Usage

See [docs/USAGE.md](docs/USAGE.md).

## torch.compile backend

The wheel ships the `mlx_vulkan` dynamo backend. After the wheel
install, two lines run a torch module on the GPU through mlx:

```python
import mlx.omarchy_torch  # registers the backend
compiled = torch.compile(model, backend="mlx_vulkan")
```

The pre-wheel flow (`PYTHONPATH=tools/torch-mlx-vulkan` plus
`import mlx_vulkan`) keeps working through a shim in
[tools/torch-mlx-vulkan](tools/torch-mlx-vulkan/README.md); the module
itself lives in the wheel.

## Other Vulkan GPUs (development)

Not a supported target. With `MLX_OMARCHY_ALLOW_NON_APPLE=1` the
backend admits a Vulkan 1.3 device that exposes timeline semaphores
and a compute queue, for development on a normal Linux box. The flag
and the llvmpipe note are in
[Development override](docs/install-omarchy.md#development-override).
Driver selection with `VK_DRIVER_FILES` is in
[Selection](docs/install-omarchy.md#selection).
The non-Apple refusal without the flag stays.

Observed, from [PR #111](https://github.com/joshuaswarren/omarchy-mlx/pull/111),
run record in [receipts/2026-10-11-non-apple-gpu-run/README.md](receipts/2026-10-11-non-apple-gpu-run/README.md):

| GPU | Driver | Result |
|---|---|---|
| Intel UHD 770 | Mesa ANV 26.2.4 | 55/55 `omarchy_runtime_tests`; op check clean |
| RTX 5060 Ti | NVIDIA 615.71.09 (suite run); 615.78.08 (latest) | 54/55; the independent-streams timing check (0.78-0.88 vs a 0.95 floor), see [Non-Apple GPUs](docs/compatibility.md#non-apple-gpus-development-drivers); Qwen3-4B 4-bit decode 98.78 tok/s on the newer driver ([receipt](receipts/2026-11-21-rtx-5060ti-batch-work/README.md)) |

AMD (RADV) has no qualifying run.

## How close to macOS

### Matched decode parity (November 2026, `MLX_OMARCHY_DEFER_TOKENS=0`)

Same MacBook Pro on both sides, same quantized model snapshot, same prompt, and identical greedy token output verified by SHA-256 hash on every run. Every Linux cell below ran on shipped Mesa Honeykrisp commit `1650fc1a33` with `MLX_OMARCHY_DEFER_TOKENS=0` (each token synchronizes on the GPU before advancing the timer, matching macOS Metal). Every macOS cell ran macOS 27.0 with stock `mlx 0.32.2` and `mlx-lm 0.31.3` on Metal. The ratio column is Linux as a share of macOS: 100% means parity, higher is faster on Linux.

| Machine | Model (snapshot) | Context / decode | Linux (Vulkan, Mesa `1650fc1a33`) | macOS (Metal, `mlx 0.32.2`) | Linux / macOS | Greedy SHA-256 (12 hex) |
|---|---|---:|---:|---:|---:|---|
| 13-inch M1 MacBook Pro (8-core GPU) | `mlx-community/Qwen3-4B-Instruct-2507-4bit` (`50d42775`) | 512 / 64 | **24.23 tok/s** (`#1067` defaults, 3 rounds / 15 runs) | 23.17 tok/s (5 pairs / 25 runs) | **104.6% (+4.6%)** | `a6fad159dfa9` |
| 13-inch M1 MacBook Pro (8-core GPU) | `mlx-community/Qwen3-4B-Instruct-2507-4bit` (`50d42775`) | 2048 / 64 | **21.06 tok/s** (5 pairs / 25 runs) | 20.18 tok/s (5 pairs / 25 runs) | **104.3% (+4.3%)** | `9115b5280c1d` |
| 13-inch M1 MacBook Pro (8-core GPU) | `SiddhJagani/Qwen3.8-2B-mlx-4Bit` (`0867d98b`) | 512 / 64 | 45.38 tok/s (`#1067` defaults, 5 pairs / 25 runs) | 49.06 tok/s (5 pairs / 25 runs) | 92.5% | `4784df039da0` |
| 13-inch M1 MacBook Pro (8-core GPU) | `SiddhJagani/Qwen3.8-2B-mlx-4Bit` (`0867d98b`) | 2048 / 64 | 44.81 tok/s (`#1067` defaults, 5 pairs / 25 runs) | 48.02 tok/s (5 pairs / 25 runs) | 93.3% | `c5fb4074d0b5` |
| 14-inch M2 Max MacBook Pro (38-core GPU) | `SiddhJagani/Qwen3.8-2B-mlx-4Bit` (`0867d98b`) | 512 / 64 | 202.71 tok/s (`819625f8`, 5 pairs) | 212.82 tok/s (5 runs) | 95.2% | `4784df039da0` |
| 16-inch M1 Max MacBook Pro (32-core GPU) | `mlx-community/Qwen3-4B-Instruct-2507-4bit` (`50d42775`) | 512 / 128 | 100.69 tok/s (`#1059` / `#1063`, 5 pairs) | 104.53 tok/s (5 runs) | 96.3% | `efb521ae04af` |


### Parity card (November 2026, Mesa `53ad9af196f5`)

One row per machine, model snapshot and context. Decode and prefill are
tokens per second on the same MacBook Pro on both sides, same snapshot,
same prompt, greedy output verified by SHA-256 on every run. Linux
cells run the unified Mesa pin `53ad9af196f5` with
`MLX_OMARCHY_DEFER_TOKENS=0`; macOS cells run stock `mlx 0.32.2` /
`mlx-lm` on Metal. The share column is Linux decode as a share of
macOS decode: 100% means parity, higher is faster on Linux.

| Machine | Model (snapshot) | Context | Linux decode | Linux prefill | macOS decode | macOS prefill | Linux / macOS | Runs | Greedy SHA-256 (12 hex) | PR | Mesa pin |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| 13-inch M1 MacBook Pro (8-core GPU) | `SiddhJagani/Qwen3.8-2B-mlx-4Bit` (`0867d98b`) | 512 | 48.46 tok/s | 517.81 tok/s | 49.06 tok/s | 347.33 tok/s | 98.8% (−1.2%) | 5 pairs / 25 runs | `4784df039da0` | `#1104` | `53ad9af196f5` |
| 13-inch M1 MacBook Pro (8-core GPU) | `SiddhJagani/Qwen3.8-2B-mlx-4Bit` (`0867d98b`) | 2048 | 47.84 tok/s | 495.89 tok/s | 48.02 tok/s | 341.46 tok/s | 99.6% (−0.4%) | 5 pairs / 25 runs | `c5fb4074d0b5` | `#1104` | `53ad9af196f5` |
| 13-inch M1 MacBook Pro (8-core GPU) | `mlx-community/Qwen3-4B-Instruct-2507-4bit` (`50d42775`) | 512 | 24.27 tok/s | 231.65 tok/s | 23.17 tok/s | 170.02 tok/s | 104.7% (+4.7%) | 5 pairs / 25 runs | `a6fad159dfa9` | `#1104` | `53ad9af196f5` |
| 13-inch M1 MacBook Pro (8-core GPU) | `mlx-community/Qwen3-4B-Instruct-2507-4bit` (`50d42775`) | 2048 | 22.20 tok/s | 195.31 tok/s | 20.18 tok/s | 161.87 tok/s | 110.0% (+10.0%) | 5 pairs / 25 runs | `9115b5280c1d` | `#1104` | `53ad9af196f5` |
| 16-inch M1 Max MacBook Pro (32-core GPU) | `mlx-community/Qwen3-4B-Instruct-2507-4bit` (`50d42775`) | 512 | **107.16 tok/s** | 894.77 tok/s | 104.53 tok/s | 660.26 tok/s | **102.5% (+2.5%)** | 5 pairs / 25 runs | `efb521ae04af` | `#1117` | `53ad9af196f5` |
| 16-inch M1 Max MacBook Pro (32-core GPU) | `mlx-community/Qwen3-4B-Instruct-2507-4bit` (`50d42775`) | 2048 | **96.69 tok/s** | 786.10 tok/s | pending | 638.54 tok/s | pending | 5 pairs / 25 runs | `542524d1f692` | `#1117` | `53ad9af196f5` |
| 16-inch M1 Max MacBook Pro (32-core GPU) | `mlx-community/Qwen3.5-9B-MLX-4bit` (`938d8919`) | 512 | **63.47 tok/s** | 470.32 tok/s | 57.77 tok/s | 339.27 tok/s | **109.9% (+9.9%)** | 5 pairs / 25 runs | `697272134f42` | `#1117` | `53ad9af196f5` |
| 16-inch M1 Max MacBook Pro (32-core GPU) | `mlx-community/Qwen3.5-9B-MLX-4bit` (`938d8919`) | 2048 | pending | pending | 57.05 tok/s | pending | pending | pending | `7dd7261b2fb8` | pending | `53ad9af196f5` |
| 14-inch M2 Max MacBook Pro (38-core GPU) | `SiddhJagani/Qwen3.8-2B-mlx-4Bit` (`0867d98b`) | 512 | **215.41 tok/s** | 2282.75 tok/s | 212.82 tok/s | 1766.07 tok/s | **101.2% (+1.2%)** | 5 pairs / 25 runs | `4784df039da0` | `#1101` | `53ad9af196f5` |
| 14-inch M2 Max MacBook Pro (38-core GPU) | `SiddhJagani/Qwen3.8-2B-mlx-4Bit` (`0867d98b`) | 2048 | **211.38 tok/s** | 2347.61 tok/s | 209.28 tok/s | 1786.03 tok/s | **101.0% (+1.0%)** | 5 pairs / 25 runs | `c5fb4074d0b5` | `#1101` | `53ad9af196f5` |

### Earlier M1 Max model sweep (October 2026)

Same MacBook Pro (M1 Max, 64 GB), same models, same prompts, MLX on both sides with the same mlx-lm (0.32.0), measured 2026-10-09 to 2026-10-11. The Linux column runs the mlx-lm patches the installer applies. Where macOS is shown, the percentage is Linux as a share of macOS: 100% means equal, higher is faster on Linux.

Decode is the median of 10 greedy generations and prefill is the median of 5 runs, except two Linux cells: gemma-4-E4B is the mean of three runs and Qwen3.6-35B-A3B the mean of two. Those two sets of runs started with a load average above 0.5, so the numbers may be low. The Qwen3.5-9B macOS cells are earlier sealed runs.

MLX 4-bit through the installer's mlx-lm patches, tokens per second. A cell shows Linux, or Linux / macOS and the percentage where macOS is shown:

| Model | Decode | Prefill 512 | Driver (Mesa git) |
|---|---|---|---|
| Qwen3.8-27B (dense 27B) | 13.5 | 87.4 | not recorded |
| gemma-4-26B-A4B (MoE, 4B active) | 12.61 | 116.50 | `6543eeb7df7` |
| gemma-4-31B (dense 31B) | 7.3 | 62.4 | not recorded |
| Qwen3.5-9B (dense 9B, gated-delta) | 37.1 / 59.4 = 62% | 299.4 / 339.0 = 88% | not recorded |
| Qwen3.6-35B-A3B (MoE, 3B active) | 13.62 | 116.37 | `6543eeb7df7` |
| gemma-4-E4B (dense 4B) | 30.5 | 366.5 | `6543eeb7df7` |
| gpt-oss-20b (MoE, attention sinks) | 2.2 | 1.9 | not recorded |

macOS shares are printed only where both sides produce identical tokens.

Driver build: the three rows naming `6543eeb7df7` ran on main at `e57cb301` with the private Honeykrisp ICD at that Mesa commit; the other rows did not record their driver. The full tables, per chip, with the wheel, the driver and the start load of each run, are in [docs/BENCHMARKS.md](docs/BENCHMARKS.md).

Decode on the 13-inch M1 changed with the CPU governor with no ANE work running. See [Set the CPU governor for local AI](docs/BENCHMARKS.md#set-the-cpu-governor-for-local-ai).

## Qwen3.5-9B fused GDN decode

See [docs/DRIVER-AND-KERNEL.md](docs/DRIVER-AND-KERNEL.md#qwen35-9b-fused-gdn-decode).

## Muse Spark Open

The mlx-community Muse Spark Open checkpoints load through the adapter in `serve/mlx_omarchy_muse`. Their 3-bit and 2-bit routed experts run on the GPU. See [docs/muse-spark.md](docs/muse-spark.md).

## How it works

See [docs/DRIVER-AND-KERNEL.md](docs/DRIVER-AND-KERNEL.md#how-it-works).

## Contribute hardware data

The fastest way to help is a capture from your machine. Owners of the
untested chips above are the unblock for those rows. Both collectors
print a fully redacted payload. Nothing is sent until you pass
`--submit`.

```bash
git clone https://github.com/joshuaswarren/omarchy-mlx.git
cd omarchy-mlx
python3 scripts/collect_quick.py    # quick capture, Linux
python3 scripts/collect_deep.py     # deep capture, Linux and macOS

# add --submit to publish. It already targets the public endpoint; the
# deep run also needs --out FILE:
python3 scripts/collect_deep.py --out mlx-omarchy-deep.tar.gz --submit
```

On a dual-boot Mac, run `collect_deep.py` under macOS and under
Omarchy, and submit both. The full guide, including what gets
collected and how redaction works, is
[docs/contribute-data.md](docs/contribute-data.md).

## Apple Neural Engine

Measured M2 Max results: [docs/ANE-LINUX.md](docs/ANE-LINUX.md#m2-max-ane-measured).
To enable the ANE on your chip: [docs/ANE-LINUX.md](docs/ANE-LINUX.md#turn-on-the-ane-for-your-chip).

## Troubleshooting

See [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md).

## Support

If omarchy-mlx is useful to you: [sponsor on GitHub](https://github.com/sponsors/joshuaswarren), a Lightning donation to `joshuaswarren@strike.me`, [star the repo](https://github.com/joshuaswarren/omarchy-mlx), or stock the lab at [joshuawarren.com/sponsors](https://joshuawarren.com/sponsors).

## Contributing

Code contributions start at
[CONTRIBUTING.md](CONTRIBUTING.md). Hardware captures start at
[docs/contribute-data.md](docs/contribute-data.md). Dev machines
without an Apple GPU can run the module under software Vulkan. GPU
kernel changes still need real hardware before a release. Any other
Vulkan GPU (NVIDIA, Intel, AMD): the contributor kit is
[docs/ANY-GPU.md](docs/ANY-GPU.md) and the results so far are in
[docs/TESTED-HARDWARE.md](docs/TESTED-HARDWARE.md).

## Releases

See [docs/RELEASING.md](docs/RELEASING.md).

## License

MIT, see [LICENSE](LICENSE). Prepared MLX source keeps Apple's MIT
license and copyright notices. Other bundled components keep their
own notices under [LICENSES](LICENSES). Not affiliated with Apple.

## Contributors

Everyone credited so far: [CONTRIBUTORS.md](CONTRIBUTORS.md).
