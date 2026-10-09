# Causal flash SDPA (MLX_OMARCHY_SDPA_CAUSAL_FLASH): end-to-end check on three chips

Question: is the coopmat causal flash route (flash, switch=1) worse than the composed route (switch=0) on a real model? Protocol per chip: Qwen3-4B-Instruct-2507-4bit, greedy, stock logits (full logits), 512-token prefill, 10 prompts x 3 passes, flash and composed interleaved 5 times, 20 s idle before each cell; one 1024-token pair and one 64-token decode pair. Same wheel stamp on G13C and G14C (`0.32.4.dev202610081925+3c767b30`), same private Vulkan driver build; the G13G run used the same wheel on its own venv. Output digests (ordered_records_sha256 prefix): pf512/pf1024 `eac9fe326212ffa2`, decode `42d27a8cbe93df49`, identical on all three chips and in every flash/composed pair.

| Chip | SDPA micro, L512 composed -> flash | 4B pf512 flash/composed (median of 5 pairs) | pf1024 (1 pair) | decode (flash / composed) |
|---|---|---|---|---|
| G13G (M1) | not recorded here | +3.10% (174.29 vs 169.05 tok/s; ranges do not overlap) | +4.88% | equal |
| G13C (M1 Max) | 1.988 -> 1.801 ms (-9%); L1024 6.141 -> 4.709 ms (-23%) | -0.65% median, -0.13% mean (489.84 vs 493.06 tok/s) | +2.23% | 65.31 / 65.31 |
| G14C (M2) | 1.677 -> 1.222 ms (-27%) | +0.23% median, +0.06% mean (682.08 vs 680.54 tok/s) | +3.06% | 63.54 / 63.44 |

Reading: no chip is worse. The end-to-end gain at 512 tokens shows on G13G only; on G13C and G14C the per-pair differences (-1.2% to +0.5%) are within run-to-run noise. Load before the cells was 0.1-0.9 (above the 0.3 bar on most G13C and G14C cells), so treat differences under ~0.7% as noise. pf1024 is a single pair per chip.

Raw per-cell JSON and logs are kept in the lab notebook (sha256 manifests per chip); this file is the summary.
