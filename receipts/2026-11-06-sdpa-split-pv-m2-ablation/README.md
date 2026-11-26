# SDPA split-PV A/B on the Apple M2 Max, six prompts, two models (2026-11-06)

The split-PV value-phase pass 1 in decode attention is the default up to 2048 keys on the Apple M2 Max (G14C). This receipt measures what happens when the variant is forced above that limit with `MLX_OMARCHY_SDPA_SPLIT_PV=1` against the shipped default (variable unset), using the release wheel, no code change. Outcome: one winning cell on Qwen3-4B 4-bit at a 2048-token prompt, no win on Qwen3.8-2B 4-bit, losses at every larger prompt measured. The recorded admission condition for an optimization pull request failed, so no optimization pull request was opened and the shipped limit is unchanged. The gain does not generalize across models.

## Method

Every cell is six balanced A/B pairs on one machine, the 14-inch Apple M2 Max (G14C). The first three pairs run the variant arm first and the default arm second; the last three pairs run the reverse order. Arms are separated by 45-second pauses. Each run executes 8 warmup generation calls and synchronizes, then builds a new prompt cache and generator for the 128 recorded outputs, so the warmup calls contribute no keys to the measured cache. The reported milliseconds per token is the median of the 126 steady inter-output intervals (outputs 2 through 127). The on arm sets only `MLX_OMARCHY_SDPA_SPLIT_PV=1`; the off arm leaves it unset. Every pair's 128 generated tokens are identical between arms, and the GPU timeout count is unchanged in every cell.

Cells are identified by prompt-token count. Each measured run decodes against its own fresh cache, so these integrated prompt-cell medians do not locate an exact cache-key crossover; the table reports prompt tokens only.

## Results

Median pair gain of the variant over the default; positive means the variant is faster.

| Model | Prompt tokens | Base ms/token | PV ms/token | Median gain | Pairs faster | Tokens identical |
|---|---|---:|---:|---:|---:|---:|
| Qwen3-4B 4-bit | 2048 | 19.283 | 18.790 | +2.62% | 6/6 | 6/6 |
| Qwen3-4B 4-bit | 2560 | 19.92 | 21.15 | -5.91% | 0/6 | 6/6 |
| Qwen3-4B 4-bit | 3072 | 20.59 | 21.77 | -5.44% | 0/6 | 6/6 |
| Qwen3-4B 4-bit | 4096 | 21.914 | 23.025 | -4.79% | 0/6 | 6/6 |
| Qwen3.8-2B 4-bit | 2048 | 9.562 | 9.557 | +0.05% | 4/6 | 6/6 |
| Qwen3.8-2B 4-bit | 4096 | 9.902 | 9.932 | -0.34% | 1/6 | 6/6 |
| Qwen3-4B 4-bit | 2304 | 19.633 | 20.596 | -4.65% | 0/6 | 6/6 |
| Qwen3-4B 4-bit | 2432 | 19.681 | 20.649 | -4.65% | 0/6 | 6/6 |

Pair-level gains, percent: Qwen3-4B at 2560: -5.22, -6.18, -5.80, -5.91, -5.92, -5.93; at 3072: -5.40, -5.60, -5.40, -5.49, -5.48, -5.27; at 2304: -4.61, -4.73, -4.61, -4.82, -4.65, -4.64; at 2432: -4.58, -4.65, -4.78, -4.42, -4.65, -5.05. Qwen3.8-2B at 2048: +0.23, +0.07, +0.02, +0.15, -0.03, -0.08; at 4096: +0.05, -0.61, -0.09, -0.30, -0.48, -0.39.

## Decision rules and outcome

The rules below were written at different times, and the text keeps them separate:

1. Before the first runs, the v4 pre-registration set the lead bar for the 4B cells: a lead required a median pair gain of at least +2.0 percent, all six pairs positive, and identical tokens. The 2048-prompt cell met it; the 4096-prompt control did not.
2. Before the mid-context runs, the v5 pre-registration defined a winning cell as a median pair gain of at least +0.5 percent, all six pairs positive, identical tokens. It froze a candidate raised limit of 2192 keys if the 2560-prompt cell failed, and a falsifier: if the 2560 cell read below +0.5 percent and the 3072 cell below zero, the useful range stays below about 2300 keys. Both cells missed, so the falsifier triggered.
3. Before the gap runs, the v6 pre-registration set the candidate limit at 2184 keys unless the 2304-prompt cell won. The 2304-prompt cell lost, so 2184 remained the candidate; that rule did not itself restore 2048.
4. Separately, after these runs, the recorded admission condition for opening an upstream optimization pull request (recorded in the internal outcome record, not in the pre-registration files) required the Qwen3.8-2B cell at a 2048-token prompt to also win under the same bar: median gain at least +0.5 percent, all six pairs faster, identical tokens. It read +0.05 percent with 4 of 6 pairs faster, so the condition failed. No optimization pull request was opened, and the shipped 2048-key default is unchanged.

## What the numbers say

The single winning cell is Qwen3-4B 4-bit at a 2048-token prompt: the variant reads about 0.49 ms per token faster (+2.62 percent). Every larger-prompt 4B cell loses, with variant-minus-default differences between 0.96 and 1.23 ms per token: 0.963 at 2304, 0.968 at 2432, 1.230 at 2560, 1.180 at 3072, 1.111 at 4096, roughly 1 ms and nearly flat across the range. The 2048-prompt cell wins and the 2304-prompt cell loses; because each cell integrates a whole prompt's decode, an exact cache-key crossover is not measured by this data. Neither pre-registered model of the turn (a step or a slope) predicted the measured 2304 and 2432 values; the step shape held and the size was misjudged by about 0.4 points.

The Qwen3.8-2B result at the same prompt where the 4B gains 2.62 percent did not meet the gate (+0.05 percent, 4 of 6 pairs faster). Why the smaller model shows no gain there is not isolated; a partition-count or launch-shape difference is a guess, not a finding.

## Not claimed

Nothing here supports raising `sdpa_split_pv_max_keys`, enabling the variant above 2048 keys by environment variable in normal use, or any effect on other chips, other models, or prefill. The variant remains the shipped default only up to 2048 keys on G14C, unchanged by this work.

## Raw data

The per-pair wall timings, pair JSON, SHA256SUMS, guard logs, and the three pre-registration files are archived with the internal measurement records; this file carries the medians and pair gains only. The runner was the reviewed arm-0 A/B runner with one changed line per context set (the allowed prompt-token list); that change is disclosed in the internal record.
