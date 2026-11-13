# Tail-latency rework of the G13S/G14G budgets

Date: 2026-11-14. This follows the review on the parent PR: the p50-only
selection rule let the G13S (20000) and G14G (10000) rows push the submission
**tail** up near the ~78 ms stutter of issue #19, even though their p50 stayed
in the 2-6 ms band. This rework re-picks the four budgets under a tail rule and
backs the two that change with a fresh decode A/B at the new value.

## Rule

For each board, take the largest swept budget whose Qwen3-4B decode submission
keeps **both**:

- p50 at or under 6 ms (the smoothness band, unchanged), and
- p99 at or under ~40 ms — half the ~78 ms stutter threshold — with no sampled
  submission reaching 78 ms.

The 4B model sets the limit on every board (its submissions are the largest).
p99 is the primary tail statistic because the per-budget sample count falls as
the budget rises, so a single `max` outlier is noisier; every chosen budget
also keeps `max` well under 78 ms.

## Submission tail (Qwen3-4B, ms)

From `../subcap/` (original sweep) and `subcap/m2r3-*.out` (M2 re-run below).
p99 / max:

| Budget | M1 Pro (G13S) | M2 (G14G) | M4 Pro (G16S) | A18 Pro (G17P) |
|---:|---|---|---|---|
| 5000 | 14.6 / 28.7 | 18.9 / 37.4 | 10.9 / 20.0 | 24.8 / 47.2 |
| 8000 | 29.1 / 36.0 | **37.9 / 47.8** | 21.2 / 26.1 | 48.9 / 60.8 |
| 10000 | **36.2 / 43.2** | 48.3 / 56.3 | **26.6 / 31.3** | 62.6 / 73.1 |
| 15000 | 57.4 / 60.8 | 75.1 / 78.4 | 41.6 / 43.6 | 96.3 / 98.3 |
| 20000 | 72.3 / 86.0 | 94.0 / 112.4 | 53.4 / 63.2 | 123.1 / 145.2 |
| 40000 | 108.0 / 157.9 | 140.5 / 206.3 | 84.1 / 123.0 | 187.7 / 273.5 |

Bold marks the chosen budget. Chosen budgets and the change from the parent PR:

| Board | Parent PR | This rework | 4B p50 / p99 / max at the pick | First budget the rule rejects |
|---|---:|---:|---|---|
| G13S (M1 Pro) | 20000 | **10000** | 2.50 / 36.2 / 43.2 | 15000 (p99 57.4) |
| G14G (M2) | 10000 | **8000** | 3.64 / 37.9 / 47.8 | 10000 (p99 48.3) |
| G16S (M4 Pro) | 10000 | **10000** (unchanged) | 5.40 / 26.6 / 31.3 | 15000 (p99 41.6) |
| G17P (A18 Pro) | 5000 | **5000** (unchanged, = fallback) | 4.35 / 24.8 / 47.2 | 8000 (p99 48.9) |

G16S and G17P already satisfied the tail rule, so only G13S and G14G move.

## M2 re-run at cool temperature

The original M2 sweep ran hot (90-99 C, passively cooled). The M2 re-run here
(`subcap/m2r3-*.out`) started at 33 C and stayed at 36 C through the 4B sweep.
The tail is the same within noise (4B p99: 5000 18.9, 8000 37.9, 10000 48.3,
15000 75.1 ms), so the tail is a real property of the submission length, not a
thermal artifact.

## Decode A/B at the new budget

Release wheel `v0.7.41` (`0.32.4.dev202611111431+8380f002`),
`receipts/2026-10-09-linux-vs-macos/score_bench.py` (10 greedy generations of
64 tokens, 5 prefills of 512, medians). Each board ran the default (5000) first,
then `MLX_OMARCHY_BATCH_WORK=<budget>`, in one sitting (`tools/ab_bw3.sh`).
Records in `ab/`.

### M2 (G14G), new budget 8000 (`ab/ab-m2r3-*.json`)

| Model | Decode 5000 | Decode 8000 | Decode 10000 | Prefill 512 (5000 / 8000) | Tokens |
|---|---:|---:|---:|---:|---|
| Qwen3.8-2B | 38.66 | 39.71 (+2.7%) | 39.97 (+3.4%) | 168.42 / 168.67 | identical (`aa592095`) |
| Qwen3-4B | 19.58 | 20.10 (+2.7%) | 20.23 (+3.3%) | 78.72 / 78.82 | identical (`ee938320`) |

On the M2 the submission budget buys little decode (~3% at 10000); 8000 keeps
essentially all of that gain and drops the 4B p99 from 48.3 to 37.9 ms and the
max from 56.3 to 47.8 ms. Prefill is within noise.

### M1 Pro (G13S), new budget 10000 (`ab/ab-m1pror3-*.json`)

| Model | Decode 5000 | Decode 10000 | Prefill 512 (5000 / 10000) | Tokens |
|---|---:|---:|---:|---|
| Qwen3.8-2B | 61.02 | 71.66 (+17.4%) | 280.77 / 281.49 | identical (`8134db97`) |
| Qwen3-4B | 31.14 | 37.55 (+20.6%) | 133.97 / 134.32 | identical (`ee938320`) |

The M1 Pro takes far more decode from the budget than the M2 (~17-21% at
10000 against ~3%). 10000 keeps the 4B submission p99 at 36.2 ms (max 43.2)
against the 72.3 ms p99 (max 86.0, at the #19 stutter) of the parent PR's
20000 pick, so the whole gain is kept while the tail clears the stutter.
Prefill is within noise. The A/B ran in one sitting at 48-66 C
(`ab/ab-m1pror3.log`); decode digests match between arms for both models.

## Stack

Same as the parent receipt (`../README.md`): Fedora 44/45 aarch64, Linux 7.3.0
(aurora-silicon/linux integration), Mesa 26.3.0-devel Borealis, system ICD,
Python 3.14, mlx-lm 0.31.3 with `--patch-venv`, `schedutil` governor, no other
GPU work on the device.
