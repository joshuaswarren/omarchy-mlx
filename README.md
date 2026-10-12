# omarchy-mlx

[![Sponsor](https://img.shields.io/badge/Sponsor-%E2%9D%A4-pink)](https://github.com/sponsors/joshuaswarren)

MLX on the Apple GPU under Linux, and Core ML models on the
Apple Neural Engine.

[MLX](https://github.com/ml-explore/mlx) is Apple's array framework for
machine learning on Apple silicon. Upstream it speaks Metal, so it runs
on macOS only. omarchy-mlx runs the same `import mlx.core as mx` code on
an Apple silicon Mac running
[Omarchy](https://github.com/omacom/omarchy) Linux. The GPU driver is
Mesa's Honeykrisp Vulkan stack. Tensors stay on the GPU. There is no
Metal and no CPU fallback.

It is for people who run Linux on an M1 or M2 Mac. You get local chat,
serving, and speech to text, with no macOS and no cloud account.

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
- An opt-in wake word. `mlx-omarchy-assistant --wake-word hey_jarvis`
  listens through openWakeWord ONNX models on the CPU (SHA-256-pinned
  files, downloaded once on first use) and hands the microphone to the
  chat when it hears the phrase. It is off by default.

Chat models download from Hugging Face the first time you use them. You
approve each download first. The default pairs: Everyday is Qwen3.5-9B
4-bit plus the Laya model. Compact is
Qwen3-4B-Instruct-2507 4-bit plus Laya. Quality is Qwen3.8-27B 4-bit
plus Laya, and it needs a 96 GB machine.

## Supported hardware

| Chip | GPU | ANE |
|---|---|---|
| M1 | Tested | Parakeet islands |
| M1 Max | Tested | Whole encoder |
| M2 Max | [Tested on Linux](receipts/2026-10-09-linux-vs-macos/m2-max-linux/README.md) | Whole encoder (bit-exact on the M2, see [M2 Max ANE](#m2-max-ane-measured)) ([results](receipts/2026-10-10-m2-max-ane.md)) |
| M1 Pro, M1 Ultra, M2, M2 Pro, M2 Ultra | Untested | Untested overlay |
| M3 | Experimental: aurora mesa-m3 graphics; compute not certified | Data-only; h15 bring-up module |
| M4 | Not yet (no Linux GPU driver) | Data-only; h16 bring-up module |

The install accepts every chip in the M1 class and the M2 Max. They
share one GPU class and driver path. The Tested rows are the machines
this project measures on. On an M3 or M4 Mac, run the tester kit,
`bash scripts/m3m4_kit.sh` (see [scripts/m3m4_kit.md](scripts/m3m4_kit.md)),
to send us the numbers and ANE state that only real silicon gives.

The ANE is a separate lane from the GPU. It has its own driver and its
own tested state. Chip-by-chip ANE status, including what each untested
chip needs, lives in the [chip coverage
table](https://github.com/joshuaswarren/omarchy-ane#chip-coverage) of
[omarchy-ane](https://github.com/joshuaswarren/omarchy-ane).

## Other Vulkan GPUs (development)

Not a supported target. With `MLX_OMARCHY_ALLOW_NON_APPLE=1` the
backend admits any Vulkan 1.3 device for development on a normal
Linux box. The flag and the llvmpipe note are in
[Development override](docs/install-omarchy.md#development-override).
Driver selection with `VK_DRIVER_FILES` is in
[Selection](docs/install-omarchy.md#selection).
The non-Apple refusal without the flag stays.

Observed, from [PR #111](https://github.com/joshuaswarren/omarchy-mlx/pull/111):

| GPU | Driver | Result |
|---|---|---|
| Intel UHD 770 | Mesa ANV 26.2.4 | 55/55 `omarchy_runtime_tests`; op check clean |
| RTX 5060 Ti | NVIDIA 615.71.09 | 54/55; the independent-streams timing check (0.78-0.88 vs a 0.95 floor), see [Non-Apple GPUs](docs/compatibility.md#non-apple-gpus-development-drivers) |

AMD (RADV) has no qualifying run.

## Install

Run one command on an Arch install for an M1-class Mac or an M2 Max. The script picks the latest release, checks the wheel against the release
`SHA256SUMS`, and asks for Python 3.14 on aarch64.

```bash
curl -fsSL https://raw.githubusercontent.com/joshuaswarren/omarchy-mlx/main/install.sh | bash
```

The other ways to install are in [docs/INSTALL.md](docs/INSTALL.md).

### The Omarchy package

See [docs/INSTALL.md](docs/INSTALL.md#the-omarchy-package).

### Install with the script

See [docs/INSTALL.md](docs/INSTALL.md#install-with-the-script).

### From a release wheel

See [docs/INSTALL.md](docs/INSTALL.md#from-a-release-wheel).

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

## How close to macOS

Same MacBook Pro (M1 Max, 64 GB), same models, same prompts, MLX on both sides with the same mlx-lm (0.32.0), measured 2026-10-09 to 2026-10-11. The Linux column runs the mlx-lm patches the installer applies. Each cell is Linux / macOS and the percentage is Linux as a share of macOS: 100% means equal, higher is faster on Linux.

Decode is the median of 10 greedy generations and prefill is the median of 5 runs, except two Linux cells: gemma-4-E4B is the mean of three runs and Qwen3.6-35B-A3B the mean of two. Those two sets of runs started with a load average above 0.5, so the numbers may be low. The macOS cells are earlier sealed runs.

MLX 4-bit through the installer's mlx-lm patches, tokens per second, Linux / macOS:

| Model | Decode | Prefill 512 |
|---|---|---|
| Qwen3.8-27B (dense 27B) | 13.5 / 19.5 = 69% | 87.4 / 105.6 = 83% |
| gemma-4-26B-A4B (MoE, 4B active) | 12.61 / 66.0 = 19% | 116.50 / 556.1 = 21% |
| gemma-4-31B (dense 31B) | 7.3 / 15.3 = 48% | 62.4 / 90.0 = 69% |
| Qwen3.5-9B (dense 9B, gated-delta) | 37.1 / 59.4 = 62% | 299.4 / 339.0 = 88% |
| Qwen3.6-35B-A3B (MoE, 3B active) | 13.62 / 75.6 = 18% | 116.37 / 604.1 = 19% |
| gemma-4-E4B (dense 4B) | 30.5 / 73.8 = 41% | 366.5 / 568.6 = 64% |
| gpt-oss-20b (MoE, attention sinks) | 2.2 / 79.2 = 3% | 1.9 / 581.1 = 0.3% |

Driver build: the two mixture-of-experts rows (gemma-4-26B-A4B and Qwen3.6-35B-A3B) and the gemma-4-E4B row ran on main at `e57cb301` with the private Honeykrisp ICD at Mesa git `6543eeb7df7`. For the other rows the driver build was not recorded. The full tables, per chip, with the wheel, the driver and the start load of each run, are in [docs/BENCHMARKS.md](docs/BENCHMARKS.md). Differences from macOS and models that did not run are in [docs/KNOWN-ISSUES.md](docs/KNOWN-ISSUES.md).

Decode on the M1-class chips follows the CPU clock. See [Set the CPU governor for local AI](docs/BENCHMARKS.md#set-the-cpu-governor-for-local-ai).

## Qwen3.5-9B fused GDN decode

See [docs/DRIVER-AND-KERNEL.md](docs/DRIVER-AND-KERNEL.md#qwen35-9b-fused-gdn-decode).

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

## M2 Max ANE (measured)

See [docs/ANE-LINUX.md](docs/ANE-LINUX.md#m2-max-ane-measured).

## Turn on the ANE for your chip

See [docs/ANE-LINUX.md](docs/ANE-LINUX.md#turn-on-the-ane-for-your-chip).

## Troubleshooting

See [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md).

## Support

Every bit of support helps keep omarchy-mlx alive and free. If you are able, [sponsor on GitHub](https://github.com/sponsors/joshuaswarren) or send a Lightning donation to `joshuaswarren@strike.me` to directly fund continued development and new integrations.

[![Sponsor](https://img.shields.io/badge/Sponsor-%E2%9D%A4-pink?style=for-the-badge)](https://github.com/sponsors/joshuaswarren)

If financial support is not an option, you can still make a big difference: [star the repo](https://github.com/joshuaswarren/omarchy-mlx), share it, or recommend it to a colleague. Word of mouth is how most people find omarchy-mlx.

## Contributing

Code contributions start at
[CONTRIBUTING.md](CONTRIBUTING.md). Hardware captures start at
[docs/contribute-data.md](docs/contribute-data.md). Dev machines
without an Apple GPU can run the module under software Vulkan. GPU
kernel changes still need real hardware before a release.

## Releases

See [docs/RELEASING.md](docs/RELEASING.md).

## License

MIT, see [LICENSE](LICENSE). Prepared MLX source keeps Apple's MIT
license and copyright notices. Other bundled components keep their
own notices under [LICENSES](LICENSES). Not affiliated with Apple.
