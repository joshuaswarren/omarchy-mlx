# SDPA split-PV A/B on the Apple M2 Max, four contexts, two models (2026-11-06)

The split-PV value-phase pass 1 in decode attention is the default up to 2048 keys on the Apple M2 Max (G14C). This receipt measures what happens when the variant is forced above that limit with `MLX_OMARCHY_SDPA_SPLIT_PV=1` against the shipped default (variable unset), using the release wheel, no code change. Outcome: one winning cell on Qwen3-4B 4-bit at a 2048-token prompt, no win on Qwen3.8-2B 4-bit, losses at every larger prompt measured, so the shipped limit stays and no pull request was opened. The gain does not generalize across models.

## Method

Every cell is six balanced A/B pairs on one machine, the 14-inch Apple M2 Max (G14C). Each run prefills the prompt, decodes 128 tokens greedily, and reports the median milliseconds per token over those 128 tokens; 8 warmup tokens precede them. The on arm sets only `MLX_OMARCHY_SDPA_SPLIT_PV=1`; the off arm leaves it unset. Pairs alternate and are separated by 45-second pauses. Every pair's 128 generated tokens are identical between arms, and the GPU timeout count is unchanged in every cell. A cell counts as winning only if the median pair gain is at least +0.5 percent and all six pairs are faster; these thresholds and the decision rules were written into pre-registration files before the runs they govern.

A prompt of P tokens decodes at cache lengths P+8 through P+136, so each cell below lists the cache-key range it actually measured.

## Results

Median pair gain of the variant over the default; positive means the variant is faster.

| Model | Prompt tokens | Cache keys | Base ms/token | PV ms/token | Median gain | Pairs faster | Tokens identical |
|---|---|---|---:|---:|---:|---:|---:|
| Qwen3-4B 4-bit | 2048 | 2056-2184 | 19.283 | 18.790 | +2.62% | 6/6 | 6/6 |
| Qwen3-4B 4-bit | 2560 | 2568-2696 | 19.92 | 21.15 | -5.91% | 0/6 | 6/6 |
| Qwen3-4B 4-bit | 3072 | 3080-3208 | 20.59 | 21.77 | -5.44% | 0/6 | 6/6 |
| Qwen3-4B 4-bit | 4096 | 4104-4232 | 21.914 | 23.025 | -4.79% | 0/6 | 6/6 |
| Qwen3.8-2B 4-bit | 2048 | 2056-2184 | 9.562 | 9.557 | +0.05% | 4/6 | 6/6 |
| Qwen3.8-2B 4-bit | 4096 | 4104-4232 | 9.902 | 9.932 | -0.34% | 1/6 | 6/6 |
| Qwen3-4B 4-bit | 2304 | 2312-2440 | 19.633 | 20.596 | -4.65% | 0/6 | 6/6 |
| Qwen3-4B 4-bit | 2432 | 2440-2568 | 19.681 | 20.649 | -4.65% | 0/6 | 6/6 |

Pair-level gains, percent: Qwen3-4B at 2560: -5.22, -6.18, -5.80, -5.91, -5.92, -5.93; at 3072: -5.40, -5.60, -5.40, -5.49, -5.48, -5.27; at 2304: -4.61, -4.73, -4.61, -4.82, -4.65, -4.64; at 2432: -4.58, -4.65, -4.78, -4.42, -4.65, -5.05. Qwen3.8-2B at 2048: +0.23, +0.07, +0.02, +0.15, -0.03, -0.08; at 4096: +0.05, -0.61, -0.09, -0.30, -0.48, -0.39.

## Decision rules and outcome

Three decision rules were frozen in the pre-registration files before the runs they govern:

1. The upstream pull request gate: open a PR only if the Qwen3.8-2B cell at a 2048-token prompt also wins, meaning a median gain of at least +0.5 percent, all six pairs faster, and identical tokens. It read +0.05 percent with 4 of 6 pairs faster, so the gate failed and no PR was opened.
2. The limit rule from the mid-context runs: the 2048-key limit may rise only if a measured cell above it wins. Both gap cells at 2304 and 2432 tokens lost 4.65 percent, so the limit stays at 2048 keys.
3. The falsifier for the whole lever: losses at the mid contexts close the lever above the winning range. It triggered.

## What the numbers say

The single winning cell is Qwen3-4B 4-bit with the cache between 2056 and 2184 keys: about 0.49 ms per token, +2.62 percent. The loss at every larger prompt is nearly flat at about 1.1 to 1.2 ms per token from 2304 through 4096 tokens. The turn from gain to loss therefore lies between 2184 and 2312 keys. Neither pre-registered model of that turn (a step or a slope) predicted the measured 2304 and 2432 values; the step shape held and the size was misjudged by about 0.4 points.

The Qwen3.8-2B result at the same prompt where the 4B gains 2.62 percent is a null (+0.05 percent, 4 of 6 pairs faster, inside noise). Why the smaller model gains nothing there is not isolated; a partition-count or launch-shape difference is a guess, not a finding.

## Not claimed

Nothing here supports raising `sdpa_split_pv_max_keys`, enabling the variant above 2048 keys by environment variable in normal use, or any effect on other chips, other models, or prefill. The variant remains the shipped default only up to 2048 keys on G14C, unchanged by this work.

## Raw data

The per-pair wall timings, pair JSON, SHA256SUMS, guard logs, and the three pre-registration files are archived with the internal measurement records; this file carries the medians and pair gains only. The runner was the reviewed arm-0 A/B runner with one changed line per context set (the allowed prompt-token list); that change is disclosed in the internal record.
