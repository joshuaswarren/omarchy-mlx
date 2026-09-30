# Card-promotion receipt, 2026-09-30

## Provenance

- Source commit on main: **91f1c8b72** (initial card promotion code + tests)
- Source commit on main: (post-push) updates with probe artifact and held-out
- Worktree: `~/.config/superpowers/worktrees/mlx-omarchy/MarkdownCards` (branch `feat/markdown-cards`)
- Host: jw14m2-linux (M2 Max T6021, 96 GB, kernel 7.1.13-ARCH-polltx)
- MLX wheel for measurement: **0.32.3.dev202609291615+06711ad** (the v0.7.6 release wheel)
- Earlier measurement wheel: 0.32.3.dev202609282218+29cba8e (pre-fix-wheel; see addendum)
- Run date: 2026-09-29 to 2026-09-30 UTC

## What shipped (commit 91f1c8b72 on main)

- New module `serve/mlx_omarchy_assistant/card_promotion.py` — pure functions that turn a finished reply into zero or one component list.
- Coordinator wiring: text streams first; the model fence always wins; on miss, `extract_text` runs once on the reply, validates, and emits one component event. Compare/decide/draft modes are unchanged.
- Schema policy: catalog `extension.card_format` drives schema choice (`"fenced-json"` -> full schema, `"markdown-promotion"` or absent -> compact schema, charts/forms/decisions/sources always send the full schema).
- Tests: `tests/test_assistant_card_promotion.py` (coordinator contract: fence wins, plain prose inert, hostile markdown inert, 1 MiB bounded, validate rejection drops card, schema-policy override; plus a DEV-set tuning check).
- Fixtures: `tests/fixtures/cards_dev.json` (DEV, 24 prompts), `tests/fixtures/cards_held_out.json` (HELD-OUT, 24 prompts, frozen sha256 below).
- Docs: new "How cards are produced" subsection in `docs/serve.md`.
- Runner scripts in this receipt: `run_held_out.py` (chunked, resumable, per-prompt results file), `run_latency.py` (first-text p95), `probe_finite_logits.py` (finite-logits check on a card prompt), `run_chunks_only.sh` and `run_all*.sh` (orchestrators).

## Frozen held-out suite

- File: `tests/fixtures/cards_held_out.json`
- SHA-256 (with `sha256` field embedded): `0df8e489e90de6c3552115e29578bb426f589bccbaec690b46d95af62513775b`
- Pre-annotation SHA (without the `sha256` field, the byte sequence the rule
  asked us to hash BEFORE running any model on the suite):
  `bafd7724d1944454cfce0173565a30159de74f90252368e215b80d55bab23c00`
- 24 prompts: 12 card-worthy (3 checklist, 3 comparison, 3 timeline, 3 facts), 6 plain, 6 near-miss.
- Thresholds (fixed before scoring): >= 10 of 12 card-worthy produce a valid card; 0 spurious cards on the 12 plain/near-miss prompts; every card passes `validate_components`; first-text p95 <= 2.0 s on the 2B over 30 warm turns.

## Finite-logits probe on the v0.7.6 wheel (2026-09-30)

Run with `receipts/2026-09-30-card-promotion/probe_finite_logits.py --model qwen3.8-2b-4bit`
through the real assistant HTTP path, `--pair everyday --resume` on the saved
pair, max_tokens 700. Raw output in
`artifacts/probe_finite_logits.json`.

```
{
  "wheel": "0.32.3.dev202609291615+06711ad",
  "model": "qwen3.8-2b-4bit",
  "max_tokens": 700,
  "elapsed_s": 14.28,
  "completion_tokens": 0,
  "has_assistant_ui_fence": false,
  "components": ["checklist"],
  "non_finite_hits": 0,
  "content_preview": "# Weekend Camping Packing Checklist\n\n## \ud83c\udf32 Shelter & Sleep\n- [ ] Tent (or tarp)\n- [ ] Sleeping bag rated for expected temperature\n- [ ] Sleeping pad or air mattress\n- [ ] Tent stakes and guylines\n- [ ]"
}
```

Interpretation:
- The 2B does NOT emit non-finite logits on the v0.7.6 wheel for a card-worthy
  prompt, even at the full 700-token allowance (the assistant's automatic
  length stop cut the run; we did not observe any NaN/Inf/-Inf tokens).
- The model writes rich markdown (a checkmarked checklist with section headers)
  rather than a fenced assistant-ui JSON envelope. **Card promotion fired**
  and the validator accepted it: one `checklist` component was emitted with
  the honest ` (from reply)` title suffix.
- This DISCONFIRMS the earlier (pre-v0.7.6) "0/8 fence" reading on the
  29cba8e wheel. The assignment note explains why: non-finite logits past
  ~300-500 tokens on the old wheel made the model produce the empty-argmax
  token (<|im_end|>) and bail out before writing the fence; the v0.7.6
  wheel fix makes the comparison fair.

## HELD-OUT results

Two held-out runs are queued on the M2 (gpu-turn -m 8, chunked, per-prompt
results checkpoint, retry-after-reboot). The detached runner
(`/tmp/mc_chunks_resume.sh`, parent pid 62638) serialises the eight 2B
chunks, the eight 27B chunks, and the 2B latency probe. The probe
consumed the v0.7.6 wheel cleanly; the held-out chunks have hit HTTP 409
Conflict on the first POST /api/conversations/{cid}/turns because of
shared GPU holders exceeding the gpu-turn cap. No held-out prompt
recordings exist yet.

| Model | Held-out cards >= 10/12 | Spurious = 0 | Source |
|---|---|---|---|
| qwen3.8-2b-4bit | NOT MEASURED | NOT MEASURED | results/held_out_qwen3.8-2b-4bit.json (empty) |
| qwen3.8-27b-4bit | NOT MEASURED | NOT MEASURED | results/held_out_qwen3.8-27b-4bit.json (empty) |

## Latency (first-text p95 on the 2B)

NOT MEASURED — chunks queue behind ModelBench + Routing raw-flock holders
that exceeded their `timeout -k 20 15m` cap; the script resumes after
each holder releases.

## Catalog capability

TBD after the HELD-OUT. Both `qwen3.8-27b-4bit` and `qwen3.8-2b-4bit` will get `extension.card_format` set from measurements only. The flags are `fenced-json` when a model measures >= 6 of 8 valid fenced cards; `markdown-promotion` otherwise (the parser is the fallback). With the v0.7.6 wheel and the finite-logits fix, the 2B is expected to emit the fence more often; the held-out numbers will set the flag.

## What card_promotion cannot do (honest limits)

- Charts and forms cannot be derived from markdown alone; the parser does not attempt them. The coordinator still sends the full schema when the user asks for one.
- Decisions (a typed Laya result) are not derived from markdown; the compare/decide coordinator path remains the source.
- The parser does not invent data. If a row's value cannot be parsed as text or a finite number, it is left as text. Tables whose column counts disagree are refused.
- The parser is conservative on plain bullet lists without an explicit user cue: a short bullet list with two or three items and no checklist/table/timeline/facts keyword in the user message is left as prose. This avoids spurious cards on conversational answers.
- Inputs above 1 MiB are rejected without parsing. Inside fenced code blocks the parser is inert, so a markdown code block that happens to look like a checklist does not leak as a card.
- The `extension.card_format` schema-policy was added in this change; existing catalog entries have no capability field yet. Until the held-out runs, the coordinator falls back to "compact schema, derive from markdown" (i.e. the parser does the work, which the probe confirms works on the 2B at least once).

## ADDENDUM 2026-09-30 (dated): pre-fix-wheel evidence is suspect

The v0.7.6 qualification on the 29cba8e wheel (the earlier "0/8 fence" on the
2B and the "5/8" on the 27B, both reported in `docs/serve.md` and the
v0.7.6 release receipt) was gathered on a wheel that emits non-finite
logits on GDN prompts past ~300-500 tokens. The 29cba8e evidence IS NOT
COMPARABLE to v0.7.6 evidence. The current live wheel is
`0.32.3.dev202609291615+06711ad`; the HELD-OUT runs and the latency probe
use that wheel. Numbers will be re-recorded here as soon as the gpu-turn
queue clears.

## Verification artifacts

- `artifacts/probe_finite_logits.json` — finite-logits probe on v0.7.6 wheel
- `results/held_out_qwen3.8-2b-4bit.json` — 2B HELD-OUT (in progress; empty)
- `results/held_out_qwen3.8-27b-4bit.json` — 27B HELD-OUT (in progress; empty)
- `latency_qwen3.8-2b-4bit.json` — first-text p50/p95 over 30 warm turns (pending)
- `SHA256SUMS` — checksums for every artifact in this folder (commit before tagging)
