# Speculative decoding with exact greedy output on the M4 Pro (G16S), 2026-11-14

## Setup

- Machine: M4 Pro (J616s, t6040), GPU `Apple M4 Pro (G16S B1)`, Mesa 26.3.0-devel
  (git-55c06da699, Borealis), Vulkan driver 26.2.99, kernel `7.3.0-m4el2.up105`.
- Wheels (provenance JSON beside this file, libmlx.so matches RECORD):
  `0.32.4.dev202611141153+602a2e0` (round 1) and `0.32.4.dev202611141457+ba57c89`
  built from this branch's tree with the G16S cap of 8 (rounds 2 and 3, `tcost`,
  `qmvT`, `e2e`).
- Models (mlx-community, `model-sha256.txt`): Qwen3-8B-4bit, Qwen3.5-4B-MLX-4bit,
  drafts Qwen3-0.6B-4bit and Qwen3.5-0.8B-MLX-4bit.
- Prompts (`prompts.py`, chat template, thinking off): `code` (rewrite a 300-token
  Python file with docstrings and type hints), `quote` (repeat a paragraph, then
  summarize it), `chat` (three sentences on why the sky is blue). Greedy, 256 tokens
  or EOS.
- Thermal procedure: each run waits for the GPU lock, the hottest zone below 72 C
  and the P-core cap at 4512 MHz. Runs started at 60-71 C. Baseline and speculative
  run back to back in one process after a warm-up of both.
- `identical` compares the full token list of the speculative run with plain greedy
  decode of the same process.

## Verify cost by pass length

`tcost.txt`: one cached forward of T tokens after a 400-token context, full logits.

| T | Qwen3-8B, short-pass GEMV | Qwen3-8B, coopmat tile | Qwen3.5-4B, short-pass GEMV | Qwen3.5-4B, tile |
|---|---|---|---|---|
| 1 | 29.4 ms | 29.5 ms | 22.5 ms | 21.7 ms |
| 2 | 36.7 | 176.8 | 28.0 | 113.5 |
| 4 | 52.0 | 186.6 | 36.6 | 107.0 |
| 5 | 76.1 | 177.3 | 55.7 | 135.9 |
| 8 | 102.2 | 193.2 | 71.4 | 135.7 |

Without the short-pass route a verify pass costs 6 times a decode step, so
speculative decoding loses on every prompt (round 1, `SMALL_T` unset: 0.28x to
0.77x) and Qwen3-8B `code` diverges from greedy at token 27. With it every run in
rounds 1 to 3 is identical to greedy. On G16S the route beats the tile at 5 to 8
rows as well, so the cap is 8 there. `qmvT.txt` has the same split per linear.

## Prompt lookup, `spec.py` (rounds 1 to 3)

k = 3 drafted tokens, n-gram 3 down to 2, drafting only while the first draft token
has been right at least 30% of the time (round 3):

| Model | Prompt | Plain greedy | Prompt lookup | Speedup | Identical |
|---|---|---|---|---|---|
| Qwen3-8B | code | 34.3 tok/s | 43.4 tok/s | 1.27x | yes |
| Qwen3-8B | quote | 34.7 | 74.1 | 2.14x | yes |
| Qwen3-8B | chat | 37.0 | 37.0 | 1.00x | yes |

Without the hit-rate gate (round 2) chat ran at 0.71x and code at 1.09x; with
more drafted tokens (k = 5 or 7) every prompt but `quote` was slower.

## Draft models, `spec.py` (round 2)

| Target + draft, k = 3 | code | quote | chat |
|---|---|---|---|
| Qwen3-8B + Qwen3-0.6B | 1.20x | 1.30x | 0.87x |
| Qwen3.5-4B + Qwen3.5-0.8B | 0.84x | 1.13x | 0.52x |

All identical to greedy. A 0.6B draft costs about a third of the 8B step on this
GPU, so draft models pay only on Qwen3-8B.

## Patched mlx-lm end to end

`e2e.py` runs `mlx_lm.stream_generate` from the series with
`mlx-lm-speculative-verify.patch` applied, greedy, 256 tokens, `MLX_OMARCHY_PROMPT_LOOKUP=3`
against the same call with the variable unset (`e2e.txt`). `MLX_OMARCHY_QMV_SMALL_T`
is unset in the environment; the patch sets it for each verify pass.

| Model | Prompt | Plain greedy | Prompt lookup | Speedup | Identical |
|---|---|---|---|---|---|
| Qwen3-8B | code | 38.9 tok/s | 45.9 tok/s | 1.18x | yes |
| Qwen3-8B | quote | 39.6 | 72.5 | 1.83x | yes |
| Qwen3-8B | chat | 42.4 | 41.0 | 0.97x | yes |
| Qwen3.5-4B | code | 52.8 | 52.8 | 1.00x | yes |
| Qwen3.5-4B | quote | 52.4 | 99.9 | 1.91x | yes |
| Qwen3.5-4B | chat | 55.8 | 54.1 | 0.97x | yes |

Qwen3.5-4B has gated-delta layers, whose cache cannot be trimmed; each rejected
draft restores their state and replays the accepted tokens.

With a draft model (`--draft`, k = 3) the patched path is identical to greedy on both
models; against the pipelined plain loop of `stream_generate` it runs 0.98x / 1.05x /
0.75x (Qwen3-8B + Qwen3-0.6B) and 0.77x / 0.93x / 0.45x (Qwen3.5-4B + Qwen3.5-0.8B).
Before the patch the same call fails on the first accepted draft token with
`IndexError: list index out of range` (`upstream_draft.py`, `ud.txt`): the
last-logits patch hands the verify pass one row of logits.
