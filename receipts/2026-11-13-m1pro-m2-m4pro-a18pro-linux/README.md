# M1 Pro, M2, M4 Pro and A18 Pro on Linux with the Borealis driver

Date: 2026-11-13. These runs back the M1 Pro, M2, M4 Pro and A18 Pro rows in `README.md`, `docs/compatibility.md`, `docs/BENCHMARKS.md` and `docs/m4-linux.md`, and the author-run rows in `docs/TESTED-HARDWARE.md` for the machines whose runtime suite ran (see [Runtime suite](#runtime-suite)).

## Machines and stack

| Chip | Machine | GPU cores | Memory | OS | Kernel | Vulkan device |
|---|---|---|---|---|---|---|
| M1 Pro | MacBook Pro 16-inch 2021 (`apple,j316s` / `apple,t6000`) | 16 | 16 GB | Fedora 45 | 7.3.0, aurora-silicon/linux `e6611926` | `Apple M1 Pro (G13S C0)` |
| M2 | MacBook Air 13-inch 2022 (`apple,j413` / `apple,t8112`) | 8 | 8 GB | Fedora 44 | 7.3.0, `e6611926` | `Apple M2 (G14G B0)` |
| M4 Pro | MacBook Pro 16-inch (`apple,j616s` / `apple,t6040`) | 20 | 24 GB | Fedora 45 | 7.3.0, `c6c15acf` | `Apple M4 Pro (G16S B1)` |
| A18 Pro | MacBook Neo (`apple,j700` / `apple,t8140`) | 5 | 8 GB | Fedora 45 | 7.3.0, `1de767ba` | `Apple A18 Pro (G17P B1)` |

Driver: Mesa 26.3.0-devel Borealis, aurora-silicon/mesa `be147e3625` (M1 Pro, M2) and `c208afc352` (M4 Pro, A18 Pro). It reports Vulkan 1.4.363 and driver ID 35 (`VK_DRIVER_ID_AURORA_BOREALIS`). The ICD is the system `asahi_icd.aarch64.json`. `omarchy-mlx-vulkan` is not installed, and `VK_DRIVER_FILES` and `MLX_OMARCHY_ALLOW_NON_APPLE` are unset. `env/` has the `vulkaninfo` lines, the devicetree compatible and the memory of each machine. GPU cores are the active cores in the core masks the kernel GPU driver logs at boot: M1 Pro `0xff, 0xff` (two clusters of 8), M2 `0x1fd` (8 of 10), M4 Pro `0xfffff` (20), A18 Pro `0x37` (5 of 6). Each chip ships in more than one GPU core count; only these were measured.

## Install

Release v0.7.41, wheel `mlx_omarchy-0.32.4.dev202611111431+8380f002-cp314-cp314-linux_aarch64.whl`, checked against the release `SHA256SUMS`. The install follows [From a release wheel](../../docs/INSTALL.md#from-a-release-wheel): a Python 3.14 venv, the wheel, `mlx-lm==0.31.3 --no-deps`, the listed Python packages and `install.sh --patch-venv`. Fedora 45 ships Python 3.15 as `python3`, so the M1 Pro, M4 Pro and A18 Pro needed `dnf install python3.14`. The A18 Pro and M4 Pro also needed `dnf install libgfortran`. Without it, `import mlx.core` fails with `libgfortran.so.5: cannot open shared object file`.

`mx.default_device()` is `Device(gpu, 0)` on all four. The provenance check (`scripts/mlx_provenance.py`) reports `version_match: true`.

## Correctness probes

`scripts/collect_deep.py --skip profile --out ...` at tag v0.7.41, with nothing submitted. `collector/` has the correctness, benchmark and environment sections.

| Chip | Probe ops | Matmul fp32 1024, median ms (TFLOPS) |
|---|---|---|
| M1 Pro | 6/6 pass | 5.682 (0.3779) |
| M2 | 6/6 pass | 10.274 (0.209) |
| M4 Pro | 6/6 pass | 5.696 (0.377) |
| A18 Pro | 6/6 pass | 16.46 (0.1305) |

## Runtime suite

Source build of `aurora-silicon/omarchy-mlx` `ec849019` (#407: main `f3ec9599` plus the submission-budget rows), configured as in `tools/ci/run-omarchy-runtime.sh` with `MLX_BUILD_TESTS=ON` and built on each machine. Logs are in `suite/`.

| Chip | `omarchy_runtime_tests` | `omarchy_copy_offset_tests` | `mlx-omarchy-info`, `--trace-smoke` |
|---|---|---|---|
| M1 Pro | 60/60 cases, 27086/27086 assertions | 29/29, 131707/131707 | exit 0, buffer round trip OK |
| M2 | 60/60 cases, 27086/27086 assertions | 29/29, 131707/131707 | exit 0, buffer round trip OK |
| M4 Pro | 59/60 cases, 27085/27086 assertions | 29/29, 131711/131711 | exit 0, buffer round trip OK |
| A18 Pro | 59/60 cases, 27085/27086 assertions | 29/29, 131711/131711 | exit 0, buffer round trip OK |

The M4 Pro failure is `independent streams measure against the serialized sum`: a rerun of the same build (`suite/m4pro-rerun.txt`) measured ratio 0.876 (serialized 0.0498 s, interleaved 0.0568 s, 400 iterations) against the 0.95 floor, and both copies verified byte-for-byte. The A18 Pro fails the same case with ratio 0.437 in the full suite and 0.384 to 0.442 in three runs of that case alone (`suite/a18pro-full.txt`), again with both copies verified. See [KNOWN-ISSUES](../../docs/KNOWN-ISSUES.md#m4-pro-g16s-and-a18-pro-g17p-on-borealis-the-independent-streams-timing-floor-fails). The `SUBMIT-RECOVER` and `failed to advance` lines in the logs come from the child processes that test the stall paths.

## LLM decode and prefill

`receipts/2026-10-09-linux-vs-macos/score_bench.py` with that receipt's `prompts.jsonl`: 2 warmup generations, then 10 greedy generations of 64 tokens (decode median, time to first token excluded) and 5 prefills of 512 tokens (median). Each model ran in one process, the 2B model first. Governor `schedutil`, release defaults (submission budget 5000 on all four boards). `bench/` has the JSON per run and the load and SoC temperature at its start.

| Chip | Qwen3.8-2B decode | Qwen3.8-2B prefill 512 | Qwen3-4B decode | Qwen3-4B prefill 512 | Peak memory 2B / 4B |
|---|---:|---:|---:|---:|---|
| M1 Pro | 61.63 | 279.96 | 31.55 | 133.89 | 1.49 / 2.54 GB |
| M2 | 38.69 | 168.38 | 19.44 | 78.75 | 1.49 / 2.54 GB |
| M4 Pro | 105.55 | 512.57 | 49.27 | 239.57 | 1.49 / 2.54 GB |
| A18 Pro | 33.29 | 108.78 | 13.50 | 44.14 | 1.49 / 2.54 GB |

Models: `SiddhJagani/Qwen3.8-2B-mlx-4Bit` (the base M1 row in `docs/BENCHMARKS.md`) and `mlx-community/Qwen3-4B-Instruct-2507-4bit` (the Compact chat pair). Tokens per second.

Output digests over the 10 generations: Qwen3-4B gives `ee938320` on the M1 Pro and M2 and `67c191ee` on the M4 Pro and A18 Pro. Qwen3.8-2B gives `8134db97` on the M1 Pro, `aa592095` on the M2 and `912ebb46` on the M4 Pro and A18 Pro. Digests are compared within one machine, not across chips: the two driver builds above can compile the same shaders differently, and the driver compiles for each GPU generation, so the M1 Pro (G13) and M2 (G14) can get different code from the same build, so float rounding in reductions can differ, and greedy decoding turns one changed argmax into a different continuation. The grouping follows that: the 4B digests split by driver build, and the 2B digests also split between G13 and G14 on the same build. We have not traced which op diverges first.

The A18 Pro is passively cooled. Its SoC read 64 C before the 2B run and 96 C before the 4B run, and its 2B decode samples range from 30.8 to 37.8.
