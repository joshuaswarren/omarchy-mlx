# Card-promotion receipt, 2026-09-30

## Provenance

- Source commit: TBD (final SHA recorded after push to main)
- Worktree: `~/.config/superpowers/worktrees/mlx-omarchy/MarkdownCards` (branch `feat/markdown-cards`)
- Host: jw14m2-linux (M2 Max T6021, 96 GB, kernel 7.1.13-ARCH-polltx)
- MLX wheel: 0.32.3.dev202609282218+29cba8e (the candidate with the elementwise-add fix)
- Run date: 2026-09-29/30 UTC

## What shipped

- New module `serve/mlx_omarchy_assistant/card_promotion.py` — pure functions that turn a finished reply into zero or one component.
- Coordinator wiring: text streams first; the model fence always wins; on miss, `extract_text` runs once on the reply, validates, and emits one component event. Compare/decide/draft modes are unchanged.
- Schema policy: catalog `extension.card_format` drives schema choice (`"fenced-json"` -> full schema, `"markdown-promotion"` or absent -> compact schema, charts/forms/decisions/sources always full).
- Tests: `tests/test_assistant_card_promotion.py` (coordinator contract: fence wins, plain prose inert, hostile markdown inert, 1 MiB bounded, validate rejection drops card, schema-policy override; plus a DEV-set tuning check).
- Fixtures: `tests/fixtures/cards_dev.json` (DEV, 24 prompts), `tests/fixtures/cards_held_out.json` (HELD-OUT, 24 prompts, frozen sha256 below).
- Docs: new "How cards are produced" subsection in `docs/serve.md`.
- Receipts: this folder holds the held-out runner, latency probe, raw outputs, and SHA-256 manifest.

## Frozen held-out suite

- File: `tests/fixtures/cards_held_out.json`
- SHA-256: `bafd7724d1944454cfce0173565a30159de74f90252368e215b80d55bab23c00`
  (recomputed after committing; the canonical SHA after adding the
  `sha256` field is recorded as `0df8e489e90de6c3552115e29578bb426f589bccbaec690b46d95af62513775b`).
- 24 prompts: 12 card-worthy (3 checklist, 3 comparison, 3 timeline, 3 facts), 6 plain, 6 near-miss.
- Thresholds (fixed before scoring): >= 10 of 12 card-worthy produce a valid card; 0 spurious cards on the 12 plain/near-miss prompts; every card passes `validate_components`.

## HELD-OUT results

Per-model outcomes are in `held_out_<model>.jsonl` and a structured summary in `held_out_<model>_summary.json`. The summary block prints valid count, spurious count, and the gate verdict.

### qwen3.8-2b-4bit (Everyday)

TBD after the M2 run.

### qwen3.8-27b-4bit (Quality)

TBD after the M2 run.

## Latency (first-text p95 on the 2B)

TBD after the M2 run. Target: p95 <= 2.0 s over 30 warm turns.

## Catalog capability

TBD after the HELD-OUT. Both `qwen3.8-27b-4bit` and `qwen3.8-2b-4bit` get `extension.card_format` set from measurements only. The flags are `fenced-json` when a model measures >= 6 of 8 valid fenced cards; `markdown-promotion` otherwise (the parser is the fallback).

## What card_promotion cannot do (honest limits)

- Charts and forms cannot be derived from markdown alone; the parser does not attempt them. The coordinator still sends the full schema when the user asks for one.
- Decisions (a typed Laya result) are not derived from markdown; the compare/decide coordinator path remains the source.
- The parser does not invent data. If a row's value cannot be parsed as text or a finite number, it is left as text. Tables whose column counts disagree are refused.
- The parser is conservative on plain bullet lists without an explicit user cue: a short bullet list with two or three items and no checklist/table/timeline/timeline keyword in the user message is left as prose. This avoids spurious cards on conversational answers.
- Inputs above 1 MiB are rejected without parsing. Inside fenced code blocks the parser is inert, so a markdown code block that happens to look like a checklist does not leak as a card.

## Verification artifacts

- `held_out_2b.jsonl` and `held_out_2b_summary.json` — raw and summary HELD-OUT output for the 2B.
- `held_out_27b.jsonl` and `held_out_27b_summary.json` — raw and summary HELD-OUT output for the 27B.
- `latency_qwen3.8-2b-4bit.json` — first-text p50/p95 over 30 warm turns.
- `SHA256SUMS` — checksums for every artifact in this folder.
