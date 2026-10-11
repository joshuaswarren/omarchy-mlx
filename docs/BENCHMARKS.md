# How close to macOS

Back to the [README](../README.md). Output differences, withdrawn rows and models that could not run are in [KNOWN-ISSUES.md](KNOWN-ISSUES.md).

Same MacBook Pro (M1 Max, 64 GB), same models, same prompts, MLX on both sides with the same mlx-lm (0.32.0), measured 2026-10-09 to 2026-10-11. The Linux column runs the mlx-lm patches the installer applies. Each cell is Linux / macOS and the percentage is Linux as a share of macOS: 100% means equal, higher is faster on Linux.

- Decode: median tokens per second over 10 greedy generations of 64 tokens (prompt processing excluded).
- Prefill: median tokens per second over 5 runs of one 512-token prompt.
- llama.cpp rows: `llama-bench` pp512 and tg128 (5 repetitions), Vulkan on Linux against Metal on macOS, same llama.cpp commit.
- All models are 4-bit (MLX) or Q4_K_M (GGUF). A model that does not fit in memory is listed as skipped, not shrunk. A model that fails to run on Linux counts as 0%.

How we measure. The target is a quiet machine. A run that started with a load average above 0.5 or with a busy ANE is marked, and its numbers may be low. A cell with repeat runs lists them. No table here was measured on the newer lab driver, Mesa PR 6 at commit `69bde15340`. Rows measured on it will be added with the driver named.

The mixture-of-experts models are the largest gap on decode: gemma-4-26B-A4B reaches 19% of macOS and Qwen3.6-35B-A3B 18%. Their prefill rose after PR #81 landed on main. It is not in v0.7.32. gemma-4-26B-A4B went from 29.9 to 116.50 tokens per second (3.9 times, now 21% of macOS) and Qwen3.6-35B-A3B from 39.4 to 116.37 (2.95 times, now 19%). The dense models reach 41-69% on decode and 64-88% on prefill. On the same Linux GPU, llama.cpp is now 2.3 times faster than MLX at gemma-4-26B-A4B prefill. It was 8.9 times. gemma-4-E4B stalled a GPU submit on the v0.7.32 wheel. On main it runs, after fix #75, at 41% of macOS on decode and 64% on prefill. gpt-oss-20b runs since the attention-sinks kernel landed. It reaches only 2.2 tokens per second decode and 1.9 prefill. A 512-token prompt takes about 4.5 minutes, so that row used 3 decode samples of 16 tokens and 2 prefill runs instead of 10 of 64 and 5. The macOS runs shared the machine with background services (load 1.3-1.9). Downloads ran before the timed rows. Both can only lower the macOS numbers.

## Results on the M1 Max with 64 GB of memory

Driver: the two mixture-of-experts rows and the gemma-4-E4B row used the private Honeykrisp ICD at Mesa git `6543eeb7df7`. For the other rows the driver build was not recorded.

MLX 4-bit through the installer's mlx-lm patches, tokens per second, Linux / macOS:

| Model | Decode | Prefill 512 |
|---|---|---|
| Qwen3.8-27B (dense 27B) | 13.5 / 19.5 = **69%** | 87.4 / 105.6 = **83%** |
| gemma-4-26B-A4B (MoE, 4B active) | 12.61 / 66.0 = **19%** (was 10.0, 15%) | 116.50 / 556.1 = **21%** (was 29.9, 5%) |
| gemma-4-31B (dense 31B) | 7.3 / 15.3 = **48%** | 62.4 / 90.0 = **69%** |
| Qwen3.5-9B (dense 9B, gated-delta) | 37.1 / 59.4 = **62%** | 299.4 / 339.0 = **88%** |
| Qwen3.6-35B-A3B (MoE, 3B active) | 13.62 / 75.6 = **18%** (was 13.4, 18%) | 116.37 / 604.1 = **19%** (was 39.4, 7%) |
| gemma-4-E4B (dense 4B) | 30.5 / 73.8 = **41%** | 366.5 / 568.6 = **64%** |
| gpt-oss-20b (MoE, attention sinks) | 2.2 / 79.2 = **3%** | 1.9 / 581.1 = **0.3%** |

The two mixture-of-experts rows with "was" values were re-measured on 2026-10-10 on main at `e57cb301` (development wheel `0.32.4.dev202610102357+e57cb301`, PR #81 included), with the Honeykrisp driver from the private ICD directory at Mesa git `6543eeb7df7`, which the scoring script selects with `VK_ICD_FILENAMES`. The "was" values are the 2026-10-09 measurements; they ran on wheel `0.32.4.dev202610090923+a48b7cc5`, before PR #81, and their driver was not recorded, so the PR #81 effect is the before and after difference, not an isolated test. The macOS values were not re-run. The output digests did not change (276e9fba and 6666c20f). The gemma-4-26B-A4B run started with a load average of 0.22. The Qwen3.6-35B-A3B cells are the mean of two runs: 13.75 and 13.48 decode, 117.52 and 115.22 prefill. The second run used the Edge0 repository, which holds the same weights and gave the same output digest. Those runs started with load averages of 1.11 and 0.98, so the numbers may be low. The other rows are unchanged.

The gemma-4-E4B Linux cells ran on main at `e57cb301` (development wheel `0.32.4.dev202610102357+e57cb301`), not on v0.7.32, which still stalls on this model. They are the mean of three runs: decode 30.52, 30.45 and 30.44, prefill 366.67, 366.08 and 366.66. The driver is the private Honeykrisp ICD at Mesa git `6543eeb7df7`, selected with `VK_ICD_FILENAMES`, on the plain kernel. The runs started with load averages of 0.91, 1.37 and 1.22, above the 0.5 target, with the ANE idle. The macOS cells are the earlier sealed runs on the same machine (decode median of 10, prefill median of 5). The Linux output digest (118c7586) is the same in all three runs and differs from the macOS digest (491eb46b), so the two sides produced different tokens. The same holds on the 13-inch M1.

llama.cpp Q4_K_M, Vulkan on Linux against Metal on macOS (same llama.cpp commit), Linux / macOS:

| Model | Decode tg128 | Prefill pp512 |
|---|---|---|
| Qwen3.8-27B | 6.3 / 12.3 = 51% | 53.8 / 131.2 = 41% |
| gemma-4-26B-A4B | 26.0 / 62.7 = 41% | 265.3 / 882.8 = 30% |
| gemma-4-31B | 6.5 / 11.5 = 56% | 42.4 / 112.8 = 38% |
| Qwen3.5-9B | 19.8 / 40.5 = 49% | 171.1 / 476.3 = 36% |
| Qwen3.6-35B-A3B | 29.06 Linux, macOS not measured | 255.44 Linux, macOS not measured |
| gemma-4-E4B | 18.5 / 59.1 = 31% | 303.5 / 821.1 = 37% |

The Qwen3.6-35B-A3B llama.cpp cells are Linux only, so they have no percent. They are means of 5 repetitions, measured on 2026-10-09 (decode 29.06 with a standard deviation of 0.63, prefill 255.44 with 1.68), with the Q4_K_M unsloth GGUF on Vulkan at llama.cpp commit `65840ed`. The scoring lane reports that the kernel was the plain one. The run started with a load average of 0.99, above the 0.5 target. The raw file is `receipts/2026-10-09-linux-vs-macos/llamacpp/linux-llama-gguf-qwen36-35b-a3b.json`. The macOS run needs a macOS window on the 16-inch M1 Max.

A later change to the tiled expert kernel (PR #110) measured 1.73 to 1.74 times faster on that kernel alone (M1 Max, gemma-4-26B-A4B, 512-token prompt, two runs). Prefill for the whole model has not been re-measured.

## Models added on 2026-10-11

Models added on 2026-10-11, Linux only. No macOS run exists for them yet, so there is no percent of macOS. MLX 4-bit, the same harness as above. The driver is the private ICD at Mesa git `6543eeb7df7`, selected with `VK_ICD_FILENAMES` by the scoring script, on both machines. A run that started with a load average above 0.5 or with the ANE busy is marked, and its numbers may be low.

| Model | Chip | MLX decode | MLX prefill 512 | llama.cpp Q4_K_M decode / prefill | Wheel | Start of run |
|---|---|---|---|---|---|---|
| Llama-3.1-8B | 13-inch M1 | 11.48 | 99.57 | 8.87 / 50.10 (MLX 1.29 and 1.99 times) | `0.32.4.dev202610101808+158fee60` | MLX load 0.42; llama.cpp load 0.67, ANE busy |
| Qwen2.5-7B-Instruct | 13-inch M1 | 14.26 | 105.61 | 9.53 / 53.08 (MLX 1.50 and 1.99 times) | `0.32.4.dev202610101808+158fee60` | load 0.70 for both runs (marked) |
| Llama-3.2-3B-Instruct | 13-inch M1 | 21.52 | 227.20 | 19.47 / 121.96 (MLX 1.11 and 1.86 times) | `0.32.4.dev202610101808+158fee60` | MLX load 0.82; llama.cpp load 0.71 (marked) |
| Qwen3-4B-Instruct-2507 | 13-inch M1 | 21.98 | 185.35 | 15.49 / 89.32 (MLX 1.42 and 2.08 times) | `0.32.4.dev202610101808+158fee60` | MLX load 0.97; llama.cpp load 1.07 (marked) |
| Mellum2-12B-A2.5B (MoE) | 13-inch M1 | 4.83 | 41.21 | no result: llama.cpp could not allocate its compute buffers on 16 GB | `0.32.4.dev202610102357+e57cb301` | load 0.16 |
| Qwen3-Coder-30B-A3B (MoE) | 16-inch M1 Max | 11.69 | 102.37 | not measured | `0.32.4.dev202610102357+e57cb301` | load 0.95 (marked) |
| Ornith-1.5-35B-A3B (MoE, OptiQ 4-bit repository) | 16-inch M1 Max | 8.48 | 35.99 | not measured | `0.32.4.dev202610102357+e57cb301` | load 0.93, ANE busy (marked) |

Mellum2-12B-A2.5B on the same 13-inch M1, before and after. On the `158fee60` wheel it measured 4.84 decode and 8.05 prefill. On `e57cb301` it measured 4.83 and 41.21. Prefill is 5.1 times faster and decode is unchanged. The output digest (326cf5bd) is the same in both. This is not an isolated test of PR #81. The wheels differ by more than that PR, and each side is one run. The before run started with a load average of 0.50 and the ANE busy. The after run started at 0.16 with the ANE idle.

## Results on the M2 Max with 94 GB of memory

Driver: the private Honeykrisp ICD at Mesa git `6543eeb7df7`, with main at `e57cb301`.

MLX 4-bit on the 14-inch M2 Max (94 GB), Linux only, measured 2026-10-11. No macOS run exists for this chip in the table, so there is no percent of macOS. Tokens per second, decode / prefill 512, with llama.cpp Q4_K_M on Vulkan (tg128 / pp512) from the same machine where it ran:

| Model | MLX decode / prefill | llama.cpp decode / prefill | MLX against llama.cpp |
|---|---|---|---|
| Qwen3.8-27B (dense) | 14.72 / 105.04 | 6.94 / 66.80 | 2.12 and 1.57 times |
| gemma-4-31B (dense) | 8.79 / 77.81 | 7.38 / 56.13 | 1.19 and 1.39 times |
| Qwen3.5-9B (dense) | 38.80 / 441.43 | 22.32 / 219.05 (see the note) | 1.74 and 2.02 times |
| gemma-4-26B-A4B (MoE) | 10.57 / 147.80 | 29.43 / 321.56 | 0.36 and 0.46 times |
| Qwen3.6-35B-A3B (MoE) | 12.87 / 143.62 | 29.13 / 305.79 | 0.44 and 0.47 times |
| Qwen3-Coder-30B-A3B (MoE) | 11.68 / 132.01 | 34.84 / 299.80 | 0.34 and 0.44 times |
| Ornith-1.5-35B-A3B (MoE) | 8.43 / 60.95 | 28.14 / 289.57 | 0.30 and 0.21 times |

These rows ran on main at `e57cb301` (development wheel `0.32.4.dev202610102357+e57cb301`). The driver is the private Honeykrisp ICD at Mesa git `6543eeb7df7`, selected with `VK_ICD_FILENAMES` by the scoring script. The kernel ran with the compute-preemption option on (`asahi.debug_flags=281474976710656`), which this chip needs to avoid a GPU wedge. On the M2 the scoring lane compared the old debug flag and the new `compute_preempt=1` parameter with a stock boot on the same machine and wheel (gemma-4-26B-A4B decode, tokens per second: stock 10.47, mean of 3 runs with a 3.5% spread; debug flag 10.57, 2 runs; `compute_preempt=1` 10.34, mean of 2 runs). None of the differences is larger than the run spread. The `compute_preempt=1` parameter ran on a separate test kernel and is not merged upstream; the rows above ran with the debug flag. That is one chip, one model and a few runs, and it was not re-measured for this table. The M1 Max and 13-inch M1 rows in this section ran on the stock kernel without that option. The runs of 2026-10-11 logged 0 GPU timeouts and 0 out-of-memory kills.

Every run started with a load average between 0.72 and 1.21, above the 0.5 target, so the numbers may be low. The ANE was busy at the start of the Qwen3.8-27B, gemma-4-31B, Qwen3-Coder and Ornith MLX runs and of every llama.cpp run except Qwen3.5-9B, whose log has no ANE field. The Qwen3.5-9B, gemma-4-26B-A4B and Qwen3.6-35B-A3B MLX runs started with the ANE idle. Each cell is one run. The Qwen3.5-9B llama.cpp cell was measured on 2026-10-10 on the plain kernel, before the preemption option, with the same private ICD. The scoring lane reports the kernel. The output digests of gemma-4-26B-A4B, Qwen3.6-35B-A3B, Qwen3-Coder and Ornith equal those of the M1 Max runs above. The Qwen3.5-9B digest (80274aa7) equals the one on the 13-inch M1. The Qwen3.8-27B (70867ed5) and gemma-4-31B (57be531a) digests equal those in the 2026-10-09 M1 Max and M2 Max receipts.

## Results on a base M1 (13-inch, 16 GB)

The same measurement on a base M1 (13-inch, 16 GB, macOS 27.0), for the models that fit, Linux / macOS. The Linux cells were measured on 2026-10-10; the macOS references are earlier sealed runs:

| Model | Decode | Prefill 512 |
|---|---|---|
| Qwen3.8-2B (MLX) | 45.6 / 49.4 = 92% | 428.2 / 456.5 = 94% |
| Qwen3.5-9B (MLX) | 12.4 / 12.9 = 96% | 97.1 / 85.9 = 113% |
| gemma-4-E4B (MLX) | 10.46 / 20.52 = 51% (main at `e57cb301`; the v0.7.32 wheel still stalls on this model) | 114.31 / 143.09 = 80% (same run) |

| llama.cpp Q4_K_M (Vulkan / Metal) | Decode tg128 | Prefill pp512 |
|---|---|---|
| Qwen3.5-9B | 7.55 / 10.85 = 70% | 48.3 / 115.0 = 42% |
| gemma-4-E4B | 10.9 / 18.4 = 59% | 83.3 / 208.6 = 40% |

On the base M1 the Linux MLX prefill is faster than macOS for Qwen3.5-9B, and its decode is about 4% behind (4.2%). Qwen3.8-2B is 8% behind on decode and 6% behind on prefill. The gap on the M1 Max above is chip-dependent. The Linux numbers are from the development wheel `0.32.4.dev202610101808+158fee60`. Its default route on the base M1 produces the same greedy tokens as macOS for Qwen3.5-9B (identical output digest over the 10 test generations). The macOS values are the median of three runs for Qwen3.5-9B, a single run for gemma-4-E4B, and 8 decode and 5 prefill runs for Qwen3.8-2B. The Linux Qwen3.8-2B values are the median of 3 decode cells and 2 prefill cells.

gemma-4-E4B runs on main since the rope-frequency fix (#75). That fix is not in v0.7.32. On main at `e57cb301` it measured 10.46 / 20.52 = 51% decode and 114.31 / 143.09 = 80% prefill 512. That run used the private ICD at Mesa git `6543eeb7df7` and started with a load average of 0.90 and the ANE busy. An earlier run on the `158fee60` development wheel gave 10.47 and 113.6. Its greedy output digest (118c7586) differs from macOS (491eb46b), so the row does not count as matching.

Driver for the base M1 rows: the Honeykrisp driver from the packaged `omarchy-mlx-vulkan` 0.7.28-2 (Mesa git e7631595df), selected with no override. The system Mesa 26.2.4-1 stays installed. The newer rows on this chip (gemma-4-E4B on main and the models added on 2026-10-11) used the private ICD at Mesa git `6543eeb7df7` instead, as stated for each. For the M1 Max tables above the driver build was not recorded, except for the two MoE rows re-measured on 2026-10-10 and the gemma-4-E4B row measured on 2026-10-11. The lab is moving to a newer patched driver; the tables get a before and after column once the scoreboard has been re-run on it.

Rows are re-measured as the lab scoreboard runs; the lowest ratios are the next kernel targets. Scripts, raw JSON per model and the full table: `receipts/2026-10-09-linux-vs-macos/`.
