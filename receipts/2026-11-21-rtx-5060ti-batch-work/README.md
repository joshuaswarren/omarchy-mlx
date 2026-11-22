# RTX 5060 Ti submission budget: 5000 to 40000

Date: 2026-11-21. Device: NVIDIA GeForce RTX 5060 Ti 16 GB (VBIOS 98.06.4E.00.17) on Thunderbolt, NVIDIA driver 615.78.08 (its Vulkan ICD), no display attached (`display_active=Disabled`). Host: a 16-inch M1 Max under Linux kernel `7.1.12-2-20261103.2-lab-ARCH`, one boot for every run here. Runtime env on both sides: `MLX_OMARCHY_ALLOW_NON_APPLE=1 MLX_OMARCHY_VRAM_ACTIVATIONS=1 MLX_OMARCHY_VRAM_WEIGHTS=<model dir> MLX_DISABLE_COMPILE=1`.

## Verdict

`kBatchWorkCalibration` gets the row `{"RTX 5060 Ti", 40000}`. Before the row, `NVIDIA GeForce RTX 5060 Ti` matched no table entry and took the 5000 fallback: seven command buffers per Qwen3-4B decode token. With the row a token is one command buffer.

| Qwen3-4B-Instruct-2507 4-bit, 256 greedy tokens | A: main 40ee0fd (5000) | B: this branch 3a02ec6 (40000) | B against A |
|---|---|---|---|
| Decode, whole window, median of 5 | 93.39 tok/s | 98.78 tok/s | +5.8% (5 of 5 pairs) |
| Decode, tokens 6 to 256, median of 5 | 94.71 tok/s | 102.50 tok/s | +8.2% |
| Slowest token (startup, token 1) | 46.3 to 48.0 ms | 45.0 to 48.2 ms | same |

Runs: A 94.51 / 91.58 / 92.96 / 93.39 / 93.44, B 98.91 / 98.78 / 99.25 / 97.75 / 97.47 tok/s (whole window). Every run of both sides produced the same 256 token ids. The two wheels stamp different commits: `0.32.4.dev202611220031+40ee0fd15` and `0.32.4.dev202611220038+3a02ec62a`, and the provenance check of every run reads `verified=match`.

## Submission length

Per-command-buffer GPU time from the diagnostics build of each side in buffer mode (`MLX_OMARCHY_GPU_PROFILE_DISPATCH=0`: one timestamp pair per command buffer, no per-dispatch barriers), decode buffers only:

| | 5000 | 40000 |
|---|---|---|
| Command buffers per decode token | 7.00 | 1.00 |
| Decode buffer GPU time p50 / p99 / max | 1.78 / 2.05 / 2.17 ms | 9.68 / 10.11 / 10.18 ms |
| All buffers incl. load and prefill, p99 / max | 17.46 / 17.63 ms | 17.39 / 20.86 ms |
| GPU busy per decode token | 10.09 ms | 9.70 ms |

The Apple rows hold the p50 to the 2-6 ms band because the same GPU composes the desktop and a long submission stalls the frame. This card drives no display, so the 9.68 ms decode buffer delays no frame; the load and prefill tail is unchanged (p99 17.4 ms on both sides). On a card that drives a display the budget needs its own sweep.

## Method

`bench-bw.sh` (this directory) runs each arm as a fresh process: 8 warmup tokens on a throwaway cache, then 256 greedy tokens on a fresh cache through `tools/profile_decode_driver.py`, one host sync per token. Arms alternate A, B five times, then one buffer-mode profile per side; `busy_split.py` derives busy and idle from the profile. The run held the card alone: the queue record shows `FOREIGN-GPU-USED count=0` and no NVIDIA kernel log line. Per-token windows of every release run are in `runs/`; `summary.txt` is the run log.

## Model

`mlx-community/Qwen3-4B-Instruct-2507-4bit` snapshot `50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b`: `model.safetensors` sha256 `2a73c6c248601ab904e035548abd8e6abb65ea27dcb5f342fb0a8910eb44173f`, `config.json` sha256 `574349e5a343236546fda55e4744a76e181f534182d7dc60ff1bad7e7a502849`.

## Test

`overlay/tests/omarchy/test_runtime.cpp`, test case "submission-cap default resolves per calibrated board": `calibrated_batch_work("NVIDIA GeForce RTX 5060 Ti") == 40000` and an RTX 4090 name still resolves to 0. With the test and without the row: `CHECK( 0 == 40000 ) is NOT correct!`, 1 failed. With the row: 27 of 27 assertions pass.
