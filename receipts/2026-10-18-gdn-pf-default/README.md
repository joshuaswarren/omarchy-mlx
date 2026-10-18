# The f32 A_log decode default on the 14-inch M2 Max and the 16-inch M1 Max (2026-10-18)

This receipt backs the change in this pull request. On a Vulkan device whose name contains `G14C` or `G13C`, a model with an f32 `A_log` and `Hk != Hv` now decodes with the untiled prefetch kernel with native GQA. That kernel is `GatedDeltaDecodeBF16PfComposedGqa`, id 637. Qwen3.5-9B has 16 key heads and 32 value heads, so it takes this route. Every other part keeps its previous default.

Three arms are compared on three machines. D is the new default. T is the tiled composed kernel, the old default on the two Max parts (`MLX_OMARCHY_GDN_DECODE_TILE=1`). U is the untiled composed kernel with the two expansion copies (`MLX_OMARCHY_GDN_DECODE_TILE=0`). The 13-inch M1 shows that its default is unchanged. The opt-in kernel and its first measurements are in `receipts/2026-10-18-gdn-pf-gqa/`.

## Two run sets, one change

| Set | Head | Wheel | Test binary |
|---|---|---|---|
| `first-87bd20f9` | `87bd20f991f78123a3a3c2501fc771f41b647aab` on `d8ac5fe1`, before the rebase | `mlx_omarchy-0.32.4.dev202610181116+87bd20f9`, sha256 `e9c541171ea8a471cd86365fe764ea31bb7a20d7b7336e488ddb563b82b06e5d`, libmlx prefix `4856d408883dedd2` | sha256 `dc1a37e6815cb314ad685663e23e96e5a3a89dc4cec065eb9457588c0d875dff` |
| `head-49e6b3c1` | `49e6b3c1768ff480b512e0381c3b46eade7fd898` on `b915c571` | `mlx_omarchy-0.32.4.dev202610181348+49e6b3c1`, sha256 `7654db6b519b221ab5b221e96e1bda3b28c6317b0969b78de2abf3d101f8fe59`, libmlx sha256 `960f0f358a2aba256aa7c4c282fd5fa1a14c3152dcb19d46774951ce1e1485a3` | sha256 `f22b8e1e75fe768657e36fca82e0095904cfc1a1868bd915859c7c7c073f2eb8` |

The commit was rebased once. The patch-id of the two heads is equal: `fdb0540753ab6bf6e56c641dbbce1ba99a6fed2a`. The diff of `primitives.cpp` and `device.h` between them is empty. So the backend code of the first wheel is the backend code of the second. The main below the commit differs: a bf16 test fix and two mlx-lm patches. The first set shows why the test fix was needed (see Tests). The pull request head after this receipt adds this directory and one pointer in `docs/kernel-flags.md`. It changes no code.

## Identity

| Machine | Vulkan device | Kernel | Driver library sha256 prefix | Model snapshot |
|---|---|---|---|---|
| 14-inch M2 Max | `Apple M2 Max (G14C B1)`, Mesa git `a5b1797446` | `7.1.12-2-20261013.18-sep-ARCH` | `c499962c742f399a` | `mlx-community/Qwen3.5-9B-4bit` `8b2b98c00a6b4d291155e4890773ca8f769aee53` |
| 16-inch M1 Max | `Apple M1 Max (G13C C0)`, Mesa git `69bde15340` | `7.1.12-2-12.6-sep-ARCH` | `40282811c6e14ef9` | `mlx-community/Qwen3.5-9B-MLX-4bit` `938d8919941c6e7efd3c7150eff7fe9d12afa631` |
| 13-inch M1 | `Apple M1 (G13G B1)`, Mesa git `69bde15340` | `7.1.12-2-20261013.18-sep-ARCH` | `40282811c6e14ef9` | `mlx-community/Qwen3.5-9B-4bit` `8b2b98c00a6b4d291155e4890773ca8f769aee53` |

The driver library is the installed Honeykrisp Vulkan driver file. Every `bench_decode` JSON line prints the device name. The 13-inch `run.log` has an empty `device:` line, because the `vulkaninfo` probe printed nothing there. Its device name is in all 30 timing JSON lines. The two laptops carry the same library file, with an equal sha256 prefix.

Firmware strings are in `firmware-read.txt`. They were read from the device tree after the runs, on the same hardware. The runs did not print them. The machines report OS firmware 13.5 and boot loader iBoot-8422.141.2. The system firmware is 27.0 on the M2 and the 13-inch, and 26.6.2 on the 16-inch.

Harness pins are equal on every machine. `bench_decode.py` is `8220064c5ea3aaaf`. `mlx_provenance.py` is `21337ddd29e79291`. `caps_sim_guard.py` is `9ddb9abd3d541b57`. The log-probability recorder is `4ac2745a45594687`. The venv holds mlx-lm 0.31.3 with the repo's patches. The governor is `performance` on every policy, read at the start. No GDN, MLX, HK or AGX variable is set, and the logs print 0.

The patches tree of the source checkout has digest `a003c55941c6c106` (76 files) on the laptops. After the head build it is `78de00ac2e64dd71` (77 files) on the M2. This is provenance only. The code under test is in the wheel.

The M2 and the 16-inch rebooted between the two sets. The 13-inch ran both sets on one boot. Each set is one boot per machine. Boot ids are replaced by `<id>` in the copied logs.

## What was run

`tools/gdn-s1.sh` ran the first set. `tools/gdn-s2.sh` ran the head set. Prompt P1 is "Explain in plain words how a bicycle works." Prompt P2 is "Write a short story about a lighthouse keeper."

- Part 0 runs the device test binary of the same commit. It runs the new case alone, then the whole binary, in the default environment.
- Part 1 measures speed with `bench_decode.py`. It decodes 64 tokens on both prompts, in 5 rounds. The arm order rotates: `D T U / T U D / U D T / D T U / T U D`. Each run is a fresh process, 30 in all.
- Part 2 checks numerics with the recorder. It decodes 128 greedy tokens. It stores the full log-probability vector for the first 32 steps. P1 runs with D, T, U and D again (D2). P2 runs with D and T.
- Part 3 is the census. It sets `MLX_OMARCHY_TRACE_DISPATCH=1` and runs arms D, T and U for 6 and 10 tokens of P1. Dispatches per decoded token are the 10-token histogram minus the 6-token histogram, over 4.
- Every machine ran 44 fresh processes per set: 30 speed, 6 numerics, 6 census and 2 test.

The guard's GPU-timeout counter read `before=0 after=0` for all six timed tickets. The rows are in `guard-gpu-timeouts.txt`. No separate Vulkan device re-open test was run.

## Results

The table gives median decode tokens per second over 5 rounds of 64 tokens. The spread of the 5 rounds is 1.3 percent or less in every cell. The ratios are ratios of the rounded medians in the table. `tools/gdn_recompute.py` reproduces the numbers from the raw files. Its output is `recompute.txt` in each set. That script prints ratios from the unrounded medians, so its third decimal can differ.

| Machine, head set | P1 D / T / U | P2 D / T / U | D over T | D over U | Census D / T / U |
|---|---|---|---|---|---|
| 14-inch M2 Max | 43.77 / 40.51 / 43.17 | 43.79 / 40.48 / 43.32 | 1.081, 1.082 | 1.014, 1.011 | 306 / 306 / 354 |
| 16-inch M1 Max | 42.76 / 39.31 / 42.36 | 42.76 / 39.34 / 42.30 | 1.088, 1.087 | 1.009, 1.011 | 306 / 306 / 354 |
| 13-inch M1 | 12.11 / 11.10 / 12.11 | 12.11 / 11.11 / 12.11 | 1.091, 1.090 | 1.000, 1.000 | 370 / 322 / 370 |

The first set gives the same picture.

- On the M2, D is 43.74 and 43.75, T is 40.47 and 40.46, and U is 43.23.
- On the 16-inch, D is 42.71 and 42.74, T is 39.34 and 39.33, and U is 42.24 and 42.30.
- On the 13-inch, D is 12.05 and 12.09, T is 11.05 and 11.12, and U is 12.04 and 12.08.

The D speeds of the two sets agree within 0.5 percent on every machine. The census is the same in both sets.

Kernels per decoded token, from the census histograms:

- On the M2 and the 16-inch, D has kernel 637 x24 and `CopyGeneral` (58) x8. T has 514 x24 and 8. U has 516 x24 and 56.
- On the 13-inch, D has 516 x24 and 56. This is the untiled composed kernel with the expansion copies. U is the same. T has 514 x24 and 8.

The 13-inch default is unchanged by this pull request. D equals U in kernel, dispatches (370) and speed. D over U is 1.000 in the head set and 1.001 in the first set. The new test case prints `device "Apple M1 (G13G B1)" measured 0 legacy 1: default 3 dispatches kernel 516`.

Numerics agree on every machine in both sets. The full 32-step log-probability arrays of D, T, U and D2 (P1) and of D and T (P2) are equal element for element. The maximum absolute difference is 0 and the KL is 0. Top-1 and top-5 agree in 128 of 128 steps. The generated ids are equal: P1 `089d7c229a1f72f6`, P2 `0ef184dc7060ef1a`. All 30 timing runs on each machine, 15 per prompt, carry equal ids: P1 `0a1a64828ea650e8`, P2 `fd33be8b314efb95`. The arrays are large and are not copied. `SHA256SUMS-arrays-not-copied.txt` in each run directory lists their sha256. `recompute.txt` ends with the groups of equal hashes.

## Tests

On the head set, all three machines pass. The new case alone runs 1 of 1 and 20 of 20 assertions. The whole `omarchy_gdn_gqa_bf16a_tests` binary runs 5 of 5 cases and 89 of 89 assertions, exit code 0. On the M2 and the 16-inch the new case prints `measured 1 legacy 0: default 1 dispatches kernel 637, tiled 1, untiled 3`.

On the first set, the M2 and the 16-inch pass 5 of 5 and 89 of 89. On the 13-inch the whole binary failed 7 of 89 assertions, in 4 of 5 cases passing. All 7 are in `gdn raw decode bf16 A_log: native GQA equals the expanded and the repeated routes bit for bit`. The checks at lines 206 and 211 of `test_gdn_gqa_bf16a.cpp` read native 3 dispatches against expanded 3.

That case is older than this change. The bf16 A_log native route needs the tiled kernel. The 13-inch part defaults to the untiled one, and the case never set the tile. A test-only change fixes it and is in main below this commit. The head set passes on the 13-inch. The failing output is `first-87bd20f9/13-inch/test-all.out`.

## Limits

Each set is one boot per machine. Speed is on two prompts of 64 decoded tokens. Numerics are on one prompt of 128 tokens, with two prompts for D and T. The change makes the route the default for any model with an f32 `A_log` and `Hk != Hv` on a device named G14C or G13C. Only the Qwen3.5-9B was measured, on one chip of each machine. The name rule also matches another product with the same chip code. Models with a bf16 `A_log` never take the route.

No speed claim goes beyond the table. Model-level numerics beyond the two prompts are open. Memory use and the GPU firmware version were not recorded. No thermal trace was kept. The two sets measure the same code through different wheels.

## Files

- `first-87bd20f9/` and `head-49e6b3c1/` hold one directory per machine: `m2`, `16-inch` and `13-inch`.
- Each machine directory holds `run.log`, the 30 timing outputs `t-*.out`, the recorder outputs `num-*.out` and `cen-*.out`, the census histograms `hist-*.txt` and the two test outputs.
- Each machine directory also holds the top-k files and the array hash list.
- Each set has a `recompute.txt`.
- `guard-gpu-timeouts.txt` and `firmware-read.txt` hold the guard rows and the firmware strings.
- `tools/` holds the two drivers, the recorder, and the comparison and recompute scripts. Paths and names are replaced by neutral ones. Otherwise the drivers are the scripts as run.
- `SHA256SUMS` covers every file here.
