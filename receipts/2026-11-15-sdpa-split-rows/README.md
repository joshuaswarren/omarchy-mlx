# Split-K verify rows (`MLX_OMARCHY_SDPA_SPLIT_ROWS`): raw rows

Raw output behind every number in the PR description and in the `MLX_OMARCHY_SDPA_SPLIT_ROWS` row of `docs/kernel-flags.md`.
Date: 2026-11-15. Chips: a 16-inch M1 Max MacBook Pro (G13C) and a 14-inch M2 Max MacBook Pro (G14C).

## Files

| Path | Content |
| --- | --- |
| `results.md` | Tables of every cell: speedup over plain decode with the flag unset and set, and which arms have ids that differ from plain decode. Generated from `summary.tsv`. |
| `summary.tsv` | One line per measured cell, parsed from `raw/`: speedups, id equality with plain decode, samples per arm and per-arm sd. |
| `rows.tsv` | One line per row: the ticket text (what was measured, the prediction and the falsifier, written before the run). |
| `raw/<chip>-<row>.txt` | Run logs, one per row, with the changes under "Edits to the logs". |
| `scripts/sdpa_sr.py` | Op bench. `SR_EQ` checks each verify row against the same token decoded alone. `SR_T` times the call with the flag set and unset. Rows `sr` (16-inch M1 Max) and `srm` (14-inch M2 Max). |
| `scripts/sdpa_disp.py` | Counts dispatches per verify call from the backend's `MLX_OMARCHY_TRACE_DISPATCH` lines. Rows `sd`. |
| `scripts/e2e_m2s*.py`, `scripts/prompts*.py`, `scripts/sample_code.py` | End-to-end generators and prompts. A generator differs from the next only in the prompt file, `k` and the number of samples. |
| `doctest-g13c-m1-max.txt` | Summary of the `omarchy_sdpa_split_tests` run on the 16-inch M1 Max. |
| `provenance.txt` | Wheel, library, driver, patched mlx-lm, doctest binary and model hashes. |

## Method

- Greedy decode, 256 new tokens, patched mlx-lm 0.31.3 (`patches/mlx-lm-speculative-verify.patch`), prompt lookup with `MLX_OMARCHY_PROMPT_LOOKUP=3` unless the row says `k=2` or `k=5`.
- Per prompt: one plain and one lookup warm-up, then the order plain, lookup, lookup with the flag, lookup with the flag, lookup, plain (ABBA). That gives two samples per arm. The `e2esr*` and `e2ejsr*` rows repeat the order three times: six samples per arm, with the per-arm sd in `summary.tsv`.
- "Flag unset" is the lookup arm with `MLX_OMARCHY_SDPA_SPLIT_ROWS` unset, "flag set" the arm with it set to `1`. `MLX_OMARCHY_QMV_SMALL_T` is unset in both arms; the patch sets it per verify pass.
- Speedup is the arm's mean tokens per second over the plain arm's mean.
- "Ids equal plain" is token-id equality of every sample of the arm with the first plain sample.

## Rows

| Row | Chip | Model | About prompt tokens | Samples per arm |
| --- | --- | --- | --- | --- |
| `e2es`, `e2es2` | G14C | 4B and 9B, 2B | 400 | 2 |
| `e2esr2`, `e2esr4`, `e2esr9` | G14C | 2B, 4B, 9B | 400 | 6 |
| `e2esrm2` | G14C | 2B | 1000 | 6 |
| `e2esm`, `e2esm9`, `e2esm2` | G14C | 4B, 9B, 2B | 1000 | 2 |
| `e2esl`, `e2esl9`, `e2esl2` | G14C | 4B, 9B, 2B | 2500 | 2 |
| `e2esv`, `e2esv9`, `e2esv2` | G14C | 4B, 9B, 2B | 5000 | 2 |
| `e2esx`, `e2esx9`, `e2esx2` | G14C | 4B, 9B, 2B | 10000 | 2 |
| `e2esk` (k=5), `e2eslk` (k=2) | G14C | 4B | 400, 2500 | 2 |
| `e2eds`, `e2edsl`, `e2edsl2` | G14C | 9B target, 2B draft | 400, 2500 | 2 |
| `e2ejs` | G13C | 4B and 2B | 400 | 2 |
| `e2ejsr`, `e2ejsr4` | G13C | 2B, 4B | 400 | 6 |
| `e2ejsrm2` | G13C | 2B | 1000 | 6 |
| `e2ejd9` | G13C | 9B | 400 | 2 (prints one line per arm) |
| `e2ejdl9` | G13C | 9B | 2500 | 2 (prints one line per arm) |
| `e2ejsl`, `e2ejsl2` | G13C | 4B, 2B | 2500 | 2 |
| `e2edjs` | G13C | 9B target, 2B draft | 400 | 2 |
| `sr`, `srm`, `sd` | G13C, G14C | op level | 128 to 4096 keys | |

Every row above has a log in `raw/`. Two rows ended without output: `e2ejs9` (Qwen3.8-9B, about 400 tokens) and `e2ejsl9` (Qwen3.8-9B, about 2500 tokens), both on the 16-inch M1 Max. The kernel log shows a GPU timeout during each (00:07:13 and 00:20:43 local time); the guard ended the row and the run log stayed empty. The same model and prompts then finished in `e2ejd9` (about 400 tokens) and `e2ejdl9` (about 2500 tokens), which print one line per arm, with no GPU timeout, with the flag set and unset. The two timeouts did not reproduce; the cause is not known.

The ticket text in `rows.tsv` for `e2esv*` and `e2esx*` says "about 3000 tokens (repeated 8 times)". That text is wrong. The prompts of `e2esv*` repeat the sample 16 times (about 5000 tokens) and those of `e2esx*` repeat it 32 times (about 10000 tokens), as the prompt files in `scripts/` show. A queued ticket cannot be edited.

## Edits to the logs

1. The guard writes a `GPU-TIMEOUTS after=N delta=N` line into a run log from outside the row. It splits or truncates an `E2E` line in six logs. In `g14c-m2-max-e2es.txt` it replaced the end of the model name and the prompt name of the 9B quote line; the rest is intact and `summary.tsv` marks the cell. In `g13c-m1-max-e2ejs.txt` it broke the 2B quote line in two. In `g13c-m1-max-e2ejsr4.txt`, `g14c-m2-max-e2esr4.txt`, `g14c-m2-max-e2esr9.txt` and `g13c-m1-max-e2ejdl9.txt` it cut one line each: the speedups are intact and the id fields after the cut are lost, so `summary.tsv` shows `?` in the id columns of that cell (five arms in all).
2. The model directory name on the 16-inch M1 Max was a local alias. It is replaced by `local-model-dir`. The model of each row is in `summary.tsv`.

Host names and home paths were removed from the scripts; no log line held a host name.

## What the rows show

- Ids against plain greedy decode are not identical in every cell, with the flag set or unset. With the flag set, four cells differ: 9B code at about 1000 tokens, 9B code and quote at about 5000 tokens (all on G14C) and one draft-route chat cell (G13C). With the flag unset, twelve cells differ (three of them are among the four above); `results.md` marks each. Five arms have unknown ids (see "Edits to the logs"). The cause is not isolated.
- The 16-inch M1 Max Qwen3.8-2B loses 2.3 percent on code at about 400 tokens with the flag set (row `e2ejsr`, six samples per arm, sd under 0.8 percent), gains 3.2 percent at about 1000 tokens (`e2ejsrm2`: 0.88 to 0.91) and 10 percent at about 2500 tokens (`e2ejsl2`). The same model on the 14-inch M2 Max does not lose at about 400 tokens (`e2esr2`: code 0.86 to 0.86, quote 1.09 to 1.07 with an unset-arm sd of 4.4 percent).
- Dispatch count (rows `sd`): with the flag set, five calls (2, 3, 4, 8 and 16 rows; 130 to 3000 keys; width 128 and 256) take exactly 2 dispatches per row on both chips, with two kernel ids. With the flag unset the same calls take 1 dispatch (width 128) or 3 (width 256 at 700 keys).
- The doctest `omarchy_sdpa_split_tests` on the 16-inch M1 Max: all value comparisons pass; the dispatch-count checks read 0 in 8 of 15 cases, old cases included. See `provenance.txt` for the likely cause.
- Qwen3.8-9B on the 16-inch M1 Max with the flag set (rows `e2ejd9`, `e2ejdl9`): about 400 tokens code 0.96 to 0.96 and quote 1.51 to 1.51; about 2500 tokens code 0.91 to 0.95 and quote 1.14 to 1.25. The 14-inch M2 Max at about 400 tokens with six samples per arm (`e2esr9`): code 0.93 to 0.93, quote 1.55 to 1.72 with an unset-arm sd of 22 percent.
