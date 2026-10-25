# llama.cpp prefill batch shape on the v3 driver (2026-10-25)

`-ub 2048` prefills a 2048 token prompt 8.2 percent faster than the default `-ub 512` on Llama-2-7B Q4_0, and 11.0 percent faster on Qwen2.5-7B-Instruct Q4_K_M. Both gains beat the default in 5 of 5 rounds. `-ub 1024` gains 6.3 and 9.1 percent. Turning flash attention off costs 5.1 percent. The change is a command line flag. No code changes. The +3 percent bar of the nightly rule is met on both models.

## Setup

Apple M2 Max, Linux kernel 7.1.12-2, Honeykrisp v3 driver at `f144c7949b` (library `0838d4d73a9a0746`, run as a user-space Vulkan driver beside the system one), llama.cpp `llama-bench` build `5ec986362d0a11c8`. The command line is in `prereg.txt`. Each log line is one process.

## Results

Prefill tok/s at 2048 prompt tokens, median of five interleaved rounds.

| Model | `-ub 512` (default) | `-ub 1024` | `-ub 2048` | `-ub 512 -fa off` |
|---|---|---|---|---|
| Llama-2-7B Q4_0 | 288.68 | 306.76 (+6.3%) | 312.41 (+8.2%) | 273.82 (-5.1%) |
| Qwen2.5-7B Q4_K_M | 242.05 | 263.99 (+9.1%) | 268.68 (+11.0%) | not run |

Per round values are in `tables/`. The load average stayed below 0.5 in every sample, and every process returned 0.

## What this does not cover

- A prompt of 512 tokens or fewer does not change, because the batch is the prompt.
- Memory use. The larger batch needs a larger compute buffer, and this machine has 96 GB. The 8 GB and 16 GB sections of `docs/llamacpp-demo.md` keep their `-ub 256 -b 256` advice for the older driver.
- Other GPUs, other drivers and models above 7B.
- Decode speed. `-ub` does not touch it.
- The boot id is removed from the logs.
