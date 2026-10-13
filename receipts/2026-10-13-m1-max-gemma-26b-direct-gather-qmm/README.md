# gemma-4-26B-A4B, direct against staged `gather_qmm`, 16-inch M1 Max (main `18059c80`)

This is the source for the 16-inch M1 Max row in [docs/BENCHMARKS.md](../../docs/BENCHMARKS.md): decode 17.17 and prefill 176.03 tokens per second, P over V 1.296. P is the default direct route. V is the staged route (`MLX_OMARCHY_GATHER_QMM_DIRECT=0`) on the same wheel and the same driver. The lab's measurement notebook holds the raw logs. This directory holds the summary and the bundle hashes.

## Source and wheel

| Item | Value |
|---|---|
| Source commit | main `18059c80` (PR 124) |
| Wheel | `mlx 0.32.4.dev202610120329+18059c80` |
| `mlx/lib/libmlx.so` sha256 | `59bad2b3b4788ef77e14d682f6c5f098d18856333f097e7af547295f4dca4a8d` (matches the wheel RECORD) |
| `mlx/core.cpython-314-aarch64-linux-gnu.so` sha256 | `7542a0bcff502c00f2c48a2395367e9d49e3405b13f59b17f54abb78b9a0444d` (matches the wheel RECORD) |

## Kernel, driver and device

| Item | Value |
|---|---|
| Chip | 16-inch M1 Max (G13C), Vulkan device `Apple M1 Max (G13C C0)`, integrated GPU |
| Kernel | `7.1.12-2-20261012.16-sep-ARCH` |
| Driver | Honeykrisp, Mesa `26.3.0-devel (git-69bde15340)`, Vulkan API 1.4.362 |
| Driver library | packaged library, sha256 prefix `40282811c6e1`, package version `1:26.2.4.git69bde15340-2`, selected as the system driver (no ICD override) |
| GPU kernel driver | version 0.2.0, RTKit protocol 12 |
| Governor | `performance`, read back before each run |
| ANE | filler on: two ANE processes ran through the whole block, busy 99.3 to 99.8 percent. No ANE number is claimed here. |

## Model and quantization

Model `mlx-community/gemma-4-26b-a4b-it-4bit`, snapshot `0d77464eeb233a2da68ebf9d7dc4edaac7db956d`. Quantization: affine, group size 64, 4 bits, with 8-bit router projections on layers 0 to 29.

| File | Hash |
|---|---|
| `config.json` | git blob `9e6233a7e7174ab214c49dca84f2828a8fb504ef` |
| `model-00001-of-00003.safetensors` | sha256 `801b67693463d31f36ea860494922b09adc9db920884a59b97d0d6e12db1ad25` |
| `model-00002-of-00003.safetensors` | sha256 `5ae0d2a6bb02c4d99accbe4fdc5880280dcac9d7b0ec841bc80c40294b0ce717` |
| `model-00003-of-00003.safetensors` | sha256 `8936e717c2cc88bfc13e83eb556a4c60d5939a3b2fdc3c9b258fb9572e1daa58` |
| `model.safetensors.index.json` | git blob `f6f9e61c33077cc7e2ef701eff7056927a4a65e6` |

## Command

One ticket, run on 2026-10-13 from 17:43Z to 17:55Z:

`runprov.sh on models-v3-large.tsv <python> gemma4-26b-a4b mlx-community/gemma-4-26b-a4b-it-4bit 2 main81 main124`

The harness scripts are not in this repository. Arm V sets `MLX_OMARCHY_GATHER_QMM_DIRECT=0`. Arm P leaves it unset. Each timed run is one process that loads the model from the local cache and prints its decode and prefill rates and an output digest.

## Result

Two interleaved rounds in the order V, P, V, P. Tokens per second:

| Arm | Decode (each run, median) | Prefill (each run, median) |
|---|---|---|
| V, staged route | 13.26, 13.24 (13.25) | 176.07, 175.77 (175.92) |
| P, direct route | 17.14, 17.20 (17.17) | 176.09, 175.97 (176.03) |

Decode P over V is 1.296. Prefill P over V is 1.001. The control spread (the V arm, maximum minus minimum over the median) is 0.2 percent. The P spread is 0.35 percent. All four runs printed the same output digest, `276e9fba48abdee3`.

## Dispatch trace

A separate 4-token traced run per arm, with a debugger attached. Both arms gave the same output digest, `713907965e81609f`, with `mlx 0.32.4.dev202610120329+18059c80` and `Device(gpu, 0)`. V ran with `MLX_OMARCHY_GATHER_QMM_DIRECT=0` and P with the variable unset. The per-kernel dispatch counts are equal in the two arms except one pair: kernel id 224 (`GatherQmmSubBF16`, the staged kernel) with 537 dispatches appears only in V, and kernel id 581 (`GatherQmmSubDirectBF16`, the direct kernel) with 537 dispatches appears only in P. The ids are the positions in the `ComputeKernel` enum in `overlay/mlx/backend/omarchy/compute.h` at `18059c80`. A breakpoint on the CPU command encoder was hit 0 times in both arms.

## Timing and thermal procedure

- The load average was 2.28 at the first sample and 1.06 to 1.34 before each timed run. No settle wait ran before the block.
- A 1-second sampler ran through the block (668 rows in 674 seconds): system power, fan speeds, NAND, battery and charger temperatures, CPU frequency policy, load average and ANE busy time.
- Fan 1 ran 1496 to 2604 rpm and fan 2 ran 1626 to 2811 rpm. NAND temperature was 37.7 to 41.0 C and the charger sensor 45.6 to 59.3 C. The three CPU frequency policies held 2,064,000, 3,036,000 and 3,036,000 kHz in every row, so no run saw a frequency drop.
- No GPU timeouts, recovery lines, fault lines or out-of-memory kills before the trace, after the trace or after the timed block.

## Device reopen

After the timed block `vulkaninfo` returned 0 and a fresh MLX process opened `Device(gpu, 0)` with return code 0.

## Bundle

The raw logs, the provenance JSON, the trace output and the sampler CSV are in a bundle. Its `SHA256SUMS` file (19 entries, all verified) hashes to `484bfe565dd4d752a3e3abb853f2462e4034f0db48f86bd5b53172eaa663b39f`. The bundle archive hashes to `f332eec2c64137af4c02e92f2b00b7138833360f5c14116a2a060b39498dc4a8`.
