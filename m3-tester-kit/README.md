# m3-tester-kit r1

One-command PAR-45 GPU card runner for a base Apple M3 (Linux or macOS),
plus a read-only ANE preflight. Nothing is uploaded; every artifact stays
under `receipts/` inside this kit.

## Contents

| Path | What it is |
|---|---|
| `wheel/` | the mlx-omarchy wheel this kit qualifies |
| `run-m3-card.sh` | the card runner (below) |
| `ane-preflight.sh` | read-only ANE bring-up readiness report |
| `scripts/wheel_card_row.py` | PAR-45 card cell runner (from the repo) |
| `scripts/apply-mlx-lm-patches.sh` + `patch-mlx-lm-*.py` + `patches/` | the vendored mlx-lm patch series + check |
| `scripts/defer_tokens_guard.py` | the DEFER_TOKENS timing guard |
| `SHA256SUMS` | hash manifest — `sha256sum -c SHA256SUMS` verifies the kit |

## Run

```bash
sha256sum -c SHA256SUMS            # verify the kit first
bash ane-preflight.sh              # read-only ANE readiness report
bash run-m3-card.sh                # 2b/4b/9b x ctx 512/2048 cards
```

The runner builds `.card-venv`, installs the wheel + mlx-lm, applies the
vendored patch series, asserts `MLX_OMARCHY_DEFER_TOKENS=0`, then runs the
PAR-45 cell (1 discarded warm-up + 5 timed reps, greedy, output-digest
gated) for each model x context pair. A CPU/GPU pstate+freq snapshot is
captured around every cell under `receipts/<UTC>/power/`.

Narrow the matrix with `MODELS="4b" CTXS="512" bash run-m3-card.sh`.
`LOCAL_ONLY=1` uses the Hugging Face cache only.

When it finishes, attach `receipts/` (cards + power snapshots + the
preflight output) to your report. Review them first; nothing leaves the
machine on its own.
