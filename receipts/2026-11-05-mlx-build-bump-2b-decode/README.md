# What an omarchy-mlx build bump is worth on the M1 Max (2026-11-05)

Two A/B tests on a 16-inch M1 Max MacBook Pro measure how much a change of the omarchy-mlx build alone moves MLX decode. The installed build is held fixed and one candidate build replaces it, with the same prompt, the same model, the same Python and mlx-lm, and the same weights file. Result: the published v0.7.38 release wheel reads +6.77 percent against the installed build, and a current-main diagnostic build reads +6.70 percent. Both are far below the 25 to 29 percent decode effect measured for the one-bit power-manager change, and that change was measured inside a single pinned build with no build swap anywhere in the comparison.

Machine: 16-inch M1 Max MacBook Pro, Linux kernel `7.1.12-2-20261103.6-lab-ARCH`. Model: Qwen3.8 2B 4-bit MLX, weights SHA-256 `b0d5de688567bf4acd5e421027acd410dabcdc255a5bd46fdbf06c75dc2e6863`.

Contract for both tests: 512-token prompt, 64 greedy decode tokens, one warmup, `MLX_DISABLE_COMPILE=1`, one fresh process per run, no profiler. Same Python 3.14.7, same mlx-lm 0.31.3 (generate.py SHA-256 `3f393062aae7435c4ae37e8914dc509d25405eabe6ddebae58e7469a7a03b1e6`), matching versions of all 36 other packages. Dependency file bytes were not compared. Tokens per second are 63 intervals over the first-to-last token window.

Every run produced the same 64 tokens: prompt digest `77d2d17d470ed5d68604f47386e2f2160c707a967b2e84489a9986d285b6de3d`, generated digest `513cf604ade0187b50075d84764e4fe495ce5adf77e1b54b7ea1380bcd8edc5d`.

The fixed side in both tests is the installed build `omarchy-mlx 0.32.4.dev202610292155+42b30a2d` (the v0.7.35 source), libmlx.so SHA-256 `2f9605565068e42d1c18d4c8d293c461fdc8dc742c2f91deeed0f08f2a02968a`. The installed runtime was never modified.

## Test A: published v0.7.38 release wheel against the installed build

Candidate: `mlx_omarchy-0.32.4.dev202611040253+a39f0251-cp314-cp314-linux_aarch64.whl`, wheel SHA-256 `b45bcbb7d000389abf7e23a79aca8b843850eede179e79352911ff8d08311156`, libmlx.so SHA-256 `6236fae222c6c966a0b096638f650671323848d9aba1f3d3fb0c595fdabae77b`, installed in an isolated virtual environment.

Ten runs, interleaved order installed, release, release, installed, installed, release, release, installed, installed, release. Five pairs in run order.

| Pairs | Installed | Release wheel | Difference |
|---|---:|---:|---:|
| 1 and 2 | 118.345 | 126.230 | +6.66% |
| 4 and 3 | 118.343 | 126.359 | +6.77% |
| 5 and 6 | 118.495 | 126.369 | +6.65% |
| 8 and 7 | 118.057 | 126.464 | +7.12% |
| 9 and 10 | 118.624 | 126.247 | +6.43% |

Median installed 118.345 tokens per second (range 118.057 to 118.624). Median release 126.359 (range 126.230 to 126.464). Median difference +6.77 percent. The release wheel was faster in all five pairs.

Per-run checks: the service unit exited 0, a drain marker followed both the generation stream and the default stream, no child processes remained, each run finished in under 20 seconds with empty stderr, the boot was unchanged, and the boot-bound kernel interval held only memory-log lines with no GPU fault.

## Test B: current-main diagnostic build against the installed build

Candidate: `mlx_omarchy-0.32.4.dev202611050329+nocpu.w91control-cp314-cp314-linux_aarch64.whl`, wheel SHA-256 `7ab420c4538e58b19ac5f150094bb526a56e719c59c8dfb6fe2b6f9600e02b1f`, libmlx.so SHA-256 `f7a00dae841e282e2b256916612d708d78917b899538a5606ae2a9f1179405ac`. A CPU-off diagnostic build of current main at the time, in an isolated environment, not a release.

Twelve runs including two pilots (pilots: installed 118.598, control 125.869 tokens per second). Five pairs, same order pattern.

| Pairs | Installed | Control build | Difference |
|---|---:|---:|---:|
| 1 and 2 | 118.404 | 126.415 | +6.77% |
| 4 and 3 | 118.633 | 125.884 | +6.11% |
| 5 and 6 | 119.075 | 126.596 | +6.32% |
| 8 and 7 | 117.843 | 126.338 | +7.21% |
| 9 and 10 | 118.367 | 126.279 | +6.68% |

Median installed 118.404 (range 117.843 to 119.075, spread 1.04 percent). Median control 126.338 (range 125.884 to 126.596, spread 0.56 percent). Median difference +6.70 percent, control faster in all five pairs. The prompt and generated digests match test A.

## The pinned build behind the one-bit comparison

For reference, the one-bit power-manager comparison quoted in the article is pinned the same way. Its protocol record states: kernel `7.1.12-2-20261103.2-lab-ARCH`, release wheel v0.7.36 build `0.32.4.dev202611030221+886ff2d1`, mlx-lm 0.31.3, 2B and 9B 4-bit snapshots with sizes and hashes pinned in every ticket, one fresh process per cell entry, 64 decode tokens, one prompt per cell, with the pattern control, treatment, control, treatment, control where control is the vote off state and treatment the vote on state. Every per-run receipt of those arms carries the same version string, `0.32.4.dev202611030221+886ff2d1`, so both arms of that comparison ran one build with the vote bit as the only difference.

## Limits

- One machine, one model, one context length, decode only.
- The tests do not isolate which change between the two builds causes the difference. The release source selects the 2x16 decode Qmm geometry by default on G13G and G13C chips.
- Says nothing about other chips.
- The control build is a diagnostic from a different commit than release v0.7.38 and is not the release result.
- The per-run JSON, unit logs and kernel logs stay in the internal measurement records for this work.
