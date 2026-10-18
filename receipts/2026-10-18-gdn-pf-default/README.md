# Default of the f32 A_log decode route on the 14-inch M2 Max and the 16-inch M1 Max (2026-10-18)

This receipt backs the change in this pull request: on a Vulkan device whose name contains `G14C` or `G13C`, the raw gated-delta decode of a model with an f32 `A_log` and `Hk != Hv` (Qwen3.5-9B: 16 key heads, 32 value heads) defaults to the untiled prefetch kernel with native GQA (`GatedDeltaDecodeBF16PfComposedGqa`, kernel id 637). Every other part keeps its previous default. It measures the new default (D) against the tiled composed kernel that was the default before on those two parts (T, `MLX_OMARCHY_GDN_DECODE_TILE=1`) and against the untiled composed kernel with the two expansion copies (U, `MLX_OMARCHY_GDN_DECODE_TILE=0`), on three machines, at two heads of the same change. It shows that the 13-inch M1 default is unchanged. The opt-in kernel and its first measurements are in `receipts/2026-10-18-gdn-pf-gqa/`.

## Two run sets, one change

| Set | Head | Wheel | Test binary |
|---|---|---|---|
| `first-87bd20f9` | `87bd20f991f78123a3a3c2501fc771f41b647aab` on `d8ac5fe1`, before the rebase | `mlx_omarchy-0.32.4.dev202610181116+87bd20f9`, sha256 `e9c541171ea8a471cd86365fe764ea31bb7a20d7b7336e488ddb563b82b06e5d`, `libmlx.so` prefix `4856d408883dedd2` | `dc1a37e6815cb314ad685663e23e96e5a3a89dc4cec065eb9457588c0d875dff` |
| `head-49e6b3c1` | `49e6b3c1768ff480b512e0381c3b46eade7fd898` on `b915c571` (the PR head) | `mlx_omarchy-0.32.4.dev202610181348+49e6b3c1`, sha256 `7654db6b519b221ab5b221e96e1bda3b28c6317b0969b78de2abf3d101f8fe59`, `libmlx.so` sha256 `960f0f358a2aba256aa7c4c282fd5fa1a14c3152dcb19d46774951ce1e1485a3` | `f22b8e1e75fe768657e36fca82e0095904cfc1a1868bd915859c7c7c073f2eb8` |

The commit was rebased once. `git patch-id --stable` of the two heads is equal (`fdb0540753ab6bf6e56c641dbbce1ba99a6fed2a`) and the diff of `primitives.cpp` and `device.h` between them is empty, so the backend code of the first wheel is the backend code of the second; what differs is the main below the commit (a bf16 test fix, two mlx-lm patches). The first set also shows why the test fix was needed (see Tests). The pull request head after this receipt is one commit on top of `49e6b3c1`; it adds this directory and one pointer in `docs/kernel-flags.md` and changes no code.

## Identity

| Machine | Vulkan device (printed in every `bench_decode` JSON line) | Kernel | Driver library `/usr/lib/libvulkan_asahi.so` sha256 prefix | Firmware (OS, system, boot loader) | Model snapshot |
|---|---|---|---|---|---|
| 14-inch M2 Max | `Apple M2 Max (G14C B1)`, Honeykrisp, Mesa git `a5b1797446` | `7.1.12-2-20261013.18-sep-ARCH` | `c499962c742f399a` | 13.5, 27.0, iBoot-8422.141.2 | `mlx-community/Qwen3.5-9B-4bit` `8b2b98c00a6b4d291155e4890773ca8f769aee53` |
| 16-inch M1 Max | `Apple M1 Max (G13C C0)`, Honeykrisp, Mesa git `69bde15340` | `7.1.12-2-12.6-sep-ARCH` | `40282811c6e14ef9` | 13.5, 26.6.2, iBoot-8422.141.2 | `mlx-community/Qwen3.5-9B-MLX-4bit` `938d8919941c6e7efd3c7150eff7fe9d12afa631` |
| 13-inch M1 | `Apple M1 (G13G B1)`, Honeykrisp, Mesa git `69bde15340` | `7.1.12-2-20261013.18-sep-ARCH` | `40282811c6e14ef9` | 13.5, 27.0, iBoot-8422.141.2 | `mlx-community/Qwen3.5-9B-4bit` `8b2b98c00a6b4d291155e4890773ca8f769aee53` |

The 13-inch `run.log` has an empty `device:` line (the `vulkaninfo` probe printed nothing there); the device name is in all of its 30 timing JSON lines, and its Mesa git `69bde15340` is read from the 16-inch (the two machines carry the same library file: equal sha256 prefix). Harness pins, equal on every machine: `bench_decode.py` `8220064c5ea3aaaf`, `mlx_provenance.py` `21337ddd29e79291`, `caps_sim_guard.py` `9ddb9abd3d541b57`, the log-probability recorder `4ac2745a45594687`; mlx-lm 0.31.3 with the repo's patches in the venv; governor `performance` on every policy (read at the start); no GDN, MLX, HK or AGX environment variable set (the logs print 0). The "patches tree" digest of the source checkout is `a003c55941c6c106` (76 files) on the two laptops and, after the head build, `78de00ac2e64dd71` (77 files) on the M2: provenance only, the code under test is in the wheel. Boots: the M2 and the 16-inch rebooted between the two sets; the 13-inch ran both sets on one boot. Each set is one boot per machine. Boot ids are replaced by `<id>` in the copied logs.

## What was run

Each run set, per machine, by `tools/gdn-s1.sh` (first) or `tools/gdn-s2.sh` (head): arms D, T, U as above; prompts P1 "Explain in plain words how a bicycle works." and P2 "Write a short story about a lighthouse keeper."

- Part 0: the device test binary `omarchy_gdn_gqa_bf16a_tests` of the same commit, the new case alone, then the whole binary, default environment.
- Part 1, speed: `bench_decode.py`, 64 tokens, both prompts, 5 rounds, arm order rotating `D T U / T U D / U D T / D T U / T U D`, one fresh process per run (30 runs).
- Part 2, numerics: the recorder, 128 greedy tokens, per-step log-probabilities (full vector for the first 32 steps): P1 with D, T, U and D again (D2); P2 with D and T.
- Part 3, census: `MLX_OMARCHY_TRACE_DISPATCH=1`, arms D, T, U, 6 and 10 tokens of P1; dispatches per decoded token = (10-token histogram minus 6-token histogram) over 4.
- Every run is a fresh process: 44 per machine per set (30 speed, 6 numerics, 6 census, 2 test). The guard's GPU-timeout counter reads `before=0 after=0` for all six tickets (M2 11:48:01Z and 14:33:57Z, 16-inch 11:47:57Z and 14:21:16Z, 13-inch 11:51:40Z and 14:24:53Z end times); no separate Vulkan device re-open test was run.

## Results

Median decode tokens per second (5 rounds, 64 tokens; spread of the 5 rounds 1.3 percent or less in every cell), and the census per decoded token. `tools/gdn_recompute.py` reproduces the table from the raw files (`recompute.txt` in each set).

| Machine, head set | P1 D / T / U | P2 D / T / U | D over T (P1, P2) | D over U (P1, P2) | Census D / T / U |
|---|---|---|---|---|---|
| 14-inch M2 Max | 43.77 / 40.51 / 43.17 | 43.79 / 40.48 / 43.32 | 1.081, 1.082 | 1.014, 1.011 | 306 / 306 / 354 |
| 16-inch M1 Max | 42.76 / 39.31 / 42.36 | 42.76 / 39.34 / 42.30 | 1.088, 1.087 | 1.009, 1.011 | 306 / 306 / 354 |
| 13-inch M1 | 12.11 / 11.10 / 12.11 | 12.11 / 11.11 / 12.11 | 1.091, 1.090 | 1.000, 1.000 | 370 / 322 / 370 |

The first set (before the rebase) gives the same picture: M2 D 43.74 and 43.75 against T 40.47 and 40.46 and U 43.23; 16-inch D 42.71 and 42.74 against T 39.34 and 39.33 and U 42.24 and 42.30; 13-inch D 12.05 and 12.09 against T 11.05 and 11.12 and U 12.04 and 12.08. The D speeds of the two sets agree within 0.5 percent on every machine. The census is the same in both sets.

- Kernels per decoded token (census histograms): M2 and 16-inch D has kernel 637 x24 and `CopyGeneral` (58) x8; T has 514 x24 and 8; U has 516 x24 and 56. 13-inch D has 516 x24 and 56 (the untiled composed kernel with the expansion copies), the same as U; T has 514 x24 and 8.
- The 13-inch default is unchanged by the pull request: D equals U in kernel, dispatches (370) and speed (D over U 1.000 and 1.001 in the two sets), and the new test case prints `device "Apple M1 (G13G B1)" measured 0 legacy 1: default 3 dispatches kernel 516`.
- Numerics: on every machine and in both sets the full 32-step log-probability arrays of D, T, U and D2 (P1) and D and T (P2) are equal element for element (maximum absolute difference 0, KL 0, top-1 and top-5 equal in 128 of 128 steps), the generated ids are equal (P1 `089d7c229a1f72f6`, P2 `0ef184dc7060ef1a`), and all 30 timing runs on each machine (15 per prompt) carry equal ids (P1 `0a1a64828ea650e8`, P2 `fd33be8b314efb95`). The arrays are not copied (they are large); `SHA256SUMS-arrays-not-copied.txt` in each run directory lists their sha256 and `recompute.txt` ends with the groups of equal hashes.

## Tests

- Head set, all three machines: the new case alone 1 of 1 and 20 of 20 assertions; the whole `omarchy_gdn_gqa_bf16a_tests` binary 5 of 5 cases and 89 of 89 assertions, exit code 0. On the M2 and the 16-inch the new case prints `measured 1 legacy 0: default 1 dispatches kernel 637, tiled 1, untiled 3`.
- First set: the M2 and the 16-inch 5 of 5 and 89 of 89. On the 13-inch the whole binary failed 7 of 89 assertions (4 of 5 cases) in `gdn raw decode bf16 A_log: native GQA equals the expanded and the repeated routes bit for bit` (`test_gdn_gqa_bf16a.cpp` lines 206 and 211: native 3 dispatches against expanded 3). That case is older than this change: the bf16 A_log native route needs the tiled kernel, the 13-inch part defaults to the untiled one, and the case never set the tile. It is fixed by a test-only change that is in main below this commit; the head set on the 13-inch passes. The failing output is `first-87bd20f9/13-inch/test-all.out`.

## Limits

One boot per machine per set, speed on two prompts of 64 decoded tokens, numerics on one prompt of 128 tokens (two prompts for D and T). The change makes the route the default for any model with an f32 `A_log` and `Hk != Hv` on a device named G14C or G13C; only the Qwen3.5-9B was measured, on one chip of each of the three machines. The name rule also matches another product with the same chip code. Models with a bf16 `A_log` never take the route. No speed claim beyond the table is made, and model-level numerics beyond the two prompts above are open. No memory use, thermal trace or GPU firmware version was recorded. The two run sets measure the same code through different wheels; the test binary of each set is the one named above.

## Files

`first-87bd20f9/` and `head-49e6b3c1/`: per machine (`m2`, `16-inch`, `13-inch`) `run.log`, the 30 timing outputs `t-*.out`, the recorder outputs `num-*.out` and `cen-*.out`, the census histograms `hist-*.txt`, the two test outputs, the top-k files, the array hash list; `recompute.txt` per set. `tools/`: the two drivers, the recorder, the comparison and recompute scripts (paths and lane names replaced by neutral ones; the drivers are the scripts as run otherwise). `SHA256SUMS` covers every file here.
