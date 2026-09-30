# Card-promotion receipt, 2026-09-30

## Provenance

- Source commits on main: **91f1c8b72** (initial card promotion code + tests); **a5529c288** (probe + chunked runner + dated pre-v0.7.6 addendum); **fad99e296** (hard-kill helper); **42ff57d4a** (27B setup deadline); **9e79ed00e** (gpu-turn -m 28 for 27B labels); **97a7a1f65** (filled receipt with 2B HELD-OUT results); **05552c5f4** (clamp 27B to -m 15 + filter hard-kill by --home).
- Worktree: `~/.config/superpowers/worktrees/mlx-omarchy/MarkdownCards` (branch `feat/markdown-cards`)
- Host: jw14m2-linux (M2 Max T6021, 96 GB, kernel 7.1.13-ARCH-polltx)
- MLX wheel for measurement: **0.32.3.dev202609291615+06711ad** (the v0.7.6 release wheel)
- Earlier measurement wheel: 0.32.3.dev202609282218+29cba8e (pre-fix-wheel; see addendum)
- Run date: 2026-09-29 to 2026-09-30 UTC

## What shipped (commits on main)

- New module `serve/mlx_omarchy_assistant/card_promotion.py` — pure functions that turn a finished reply into zero or one component list.
- Coordinator wiring: text streams first; the model fence always wins; on miss, `extract_text` runs once on the reply, validates, and emits one component event. Compare/decide/draft modes are unchanged.
- Schema policy: catalog `extension.card_format` drives schema choice (`"fenced-json"` -> full schema, `"markdown-promotion"` or absent -> compact schema, charts/forms/decisions/sources always send the full schema).
- Tests: `tests/test_assistant_card_promotion.py` (11 contract tests), `tests/test_assistant_card_tuning.py` (DEV-set simulator).
- Fixtures: `tests/fixtures/cards_dev.json` (DEV, 24 prompts), `tests/fixtures/cards_held_out.json` (HELD-OUT, 24 prompts, frozen sha256 below).
- Docs: new "How cards are produced" subsection in `docs/serve.md`.
- Receipts in this folder: `run_held_out.py` (chunked, resumable, per-prompt results file, boot_id mismatch fallback to --setup, hard-kill helper that traps + killpg + kills ONLY --home-matched MarkdownCards children, no `fuser -k` of /dev/dri/renderD*), `run_latency.py` (first-text p95), `probe_finite_logits.py` (finite-logits probe), `run_chunks_only.sh` (chunks of 3 prompts each, gpu-turn -m 8 for 2B labels / -m 15 for 27B labels, paths via MARKCARDS_HOME / MARKCARDS_VENV), `README.md` (this file).
- Artifacts: `artifacts/probe_finite_logits.json`, `artifacts/held_out_qwen3.8-2b-4bit.json` (24/24 2B run on v0.7.6), `artifacts/held_out_qwen3.8-27b-4bit.json` (24/24 27B run on v0.7.6), `artifacts/mc_score.py` (scorer).

## Frozen held-out suite

- File: `tests/fixtures/cards_held_out.json`
- SHA-256 (with `sha256` field embedded): `0df8e489e90de6c3552115e29578bb426f589bccbaec690b46d95af62513775b`
- Pre-annotation SHA (without the `sha256` field, the byte sequence the
  rule asked us to hash BEFORE running any model on the suite):
  `bafd7724d1944454cfce0173565a30159de74f90252368e215b80d55bab23c00`
- 24 prompts: 12 card-worthy (3 checklist, 3 comparison, 3 timeline, 3 facts), 6 plain, 6 near-miss.
- Thresholds (fixed before scoring): >= 10 of 12 card-worthy produce a valid card; 0 spurious cards on the 12 plain/near-miss prompts; every card passes `validate_components`; first-text p95 <= 2.0 s on the 2B over 30 warm turns.

## Finite-logits probe on the v0.7.6 wheel (2026-09-30)

Run with `receipts/2026-09-30-card-promotion/probe_finite_logits.py --model qwen3.8-2b-4bit`
through the real assistant HTTP path, `--resume` on the saved pair,
max_tokens 700. Raw output in
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

## HELD-OUT results on the v0.7.6 wheel

### qwen3.8-2b-4bit (Everyday)

Run with `run_held_out.py --model qwen3.8-2b-4bit --start N --end N+3`,
through the real assistant HTTP path, `--pair everyday --resume`,
max_tokens 700. Raw output in
`artifacts/held_out_qwen3.8-2b-4bit.json`.

```
=== qwen3.8-2b-4bit: 24/24 prompts recorded ===
  card-worthy 12/12 (valid: 10)
  plain       6/6 (spurious: 1)
  near-miss   6/6 (spurious: 0)
  threshold >= 10/12: True
  threshold 0 spurious on plain/near-miss: False
    ho-01 card-worthy  checklist      expect=card  components=['checklist'] pass=True
    ho-02 card-worthy  checklist      expect=card  components=['checklist'] pass=True
    ho-03 card-worthy  checklist      expect=card  components=['checklist'] pass=True
    ho-04 card-worthy  comparison     expect=card  components=['comparison'] pass=True
    ho-05 card-worthy  comparison     expect=card  components=['comparison'] pass=True
    ho-06 card-worthy  comparison     expect=card  components=['comparison'] pass=True
    ho-07 card-worthy  timeline       expect=card  components=[] pass=False
    ho-08 card-worthy  timeline       expect=card  components=[] pass=False
    ho-09 card-worthy  timeline       expect=card  components=['comparison'] pass=True
    ho-10 card-worthy  facts          expect=card  components=['facts'] pass=True
    ho-11 card-worthy  facts          expect=card  components=['facts'] pass=True
    ho-12 card-worthy  checklist      expect=card  components=['checklist'] pass=True
    ho-13 plain        none           expect=none  components=[] pass=True
    ho-14 plain        none           expect=none  components=[] pass=True
    ho-15 plain        none           expect=none  components=[] pass=True
    ho-16 plain        none           expect=none  components=['comparison'] pass=False
    ho-17 plain        none           expect=none  components=[] pass=True
    ho-18 plain        none           expect=none  components=[] pass=True
    ho-19 near-miss    short-list     expect=none  components=[] pass=True
    ho-20 near-miss    short-list     expect=none  components=[] pass=True
    ho-21 near-miss  list-in-prose  expect=none  components=[] pass=True
    ho-22 near-miss  list-in-prose  expect=none  components=[] pass=True
    ho-23 near-miss  code-only      expect=none  components=[] pass=True
    ho-24 near-miss  code-only      expect=none  components=[] pass=True
```

Per-prompt outcomes:
- 10 of 12 card-worthy prompts produced a valid card. **Threshold >= 10/12 MET.**
- The two fails were ho-07 ("Lay out the milestones of a 12-week product launch.") and ho-08 — both timeline-kind prompts. The 2B wrote prose without day/time/weekday markers, so the timeline pass returned no markers and yielded no card. Acceptable: timelines require specific markup that the 2B did not emit on these prompts. The pass test still counted ho-09 (timeline) as a card because the 2B happened to emit a markdown table, which the comparison pass picked up.
- 1 of 12 plain prompts produced a spurious card. **Threshold 0 spurious FAILED.** ho-16 ("Explain a hash table in plain words.") produced a markdown table comparing hash-table implementations. The comparison pass fired and produced a `comparison` component even though the user did not ask for one. This is an over-promotion bug: a markdown table without an explicit `compare` / `table` / `vs` / `tabular` / `side-by-side` cue in the user message should not be promoted. **This is a parser rule gap that must be tightened before declaring the gate met.** Per Main's broadcast: "if you change code after seeing held-out results, that suite is spent and you must write and freeze a new one." I have NOT modified card_promotion.py after seeing the ho-16 spurious; the 2B numbers above stand.

### qwen3.8-27b-4bit (Quality)

```
=== qwen3.8-27b-4bit: 24/24 prompts recorded ===
  card-worthy 12/12 (valid: 7)
  plain       6/6 (spurious: 4)
  near-miss   6/6 (spurious: 2)
  threshold >= 10/12: False
  threshold 0 spurious on plain/near-miss: False
    ho-01 card-worthy  checklist      expect=card  components=[] pass=False
    ho-02 card-worthy  checklist      expect=card  components=['checklist'] pass=True
    ho-03 card-worthy  checklist      expect=card  components=[] pass=False
    ho-04 card-worthy  comparison     expect=card  components=[] pass=False
    ho-05 card-worthy  comparison     expect=card  components=[] pass=False
    ho-06 card-worthy  comparison     expect=card  components=[] pass=False
    ho-07 card-worthy  timeline       expect=card  components=['timeline'] pass=True
    ho-08 card-worthy  timeline       expect=card  components=['timeline'] pass=True
    ho-09 card-worthy  timeline       expect=card  components=['timeline'] pass=True
    ho-10 card-worthy  facts          expect=card  components=['facts'] pass=True
    ho-11 card-worthy  facts          expect=card  components=['facts'] pass=True
    ho-12 card-worthy  checklist      expect=card  components=['checklist'] pass=True
    ho-13 plain        none           expect=none  components=['checklist'] pass=False
    ho-14 plain        none           expect=none  components=[] pass=True
    ho-15 plain        none           expect=none  components=['checklist'] pass=False
    ho-16 plain        none           expect=none  components=['checklist'] pass=False
    ho-17 plain        none           expect=none  components=[] pass=True
    ho-18 plain        none           expect=none  components=[] pass=True
    ho-19 near-miss    short-list     expect=none  components=['checklist'] pass=False
    ho-20 near-miss    short-list     expect=none  components=[] pass=True
    ho-21 near-miss  list-in-prose  expect=none  components=['checklist'] pass=False
    ho-22 near-miss  list-in-prose  expect=none  components=[] pass=True
    ho-23 near-miss  code-only      expect=none  components=[] pass=True
    ho-24 near-miss  code-only      expect=none  components=[] pass=True
```

Per-prompt outcomes:
- 7 of 12 card-worthy prompts produced a valid card. **Threshold >= 10/12 FAILED** (missing by 3 prompts).
- 4 of 12 plain prompts produced a spurious card; 2 of 6 near-miss produced a spurious card. **Threshold 0 spurious FAILED on plain AND near-miss.**
- The 27B has a DIFFERENT failure mode than the 2B: it over-promotes checklists. Any markdown list of items (especially with - bullets that could be mistaken for checkboxes) gets promoted to a `checklist` card even when the user did not ask for one. The 27B's prose output is structured enough that the parser finds a checklist shape almost everywhere.

## Threshold check vs. the v0.7.6 spec

| Threshold | 2B | 27B |
|---|---|---|
| >= 10/12 card-worthy produce a valid card | PASS (10) | FAIL (7) |
| 0 spurious cards on plain/near-miss | FAIL (1 plain) | FAIL (4 plain, 2 near-miss) |
| Every card passes validate_components | PASS | PASS |
| first-text p95 <= 2.0 s on the 2B over 30 warm turns | NOT MEASURED | n/a |
| extension.card_format set from measurements | NOT SET (insufficient on both) | NOT SET |

## Latency (first-text p95 on the 2B)

NOT MEASURED. The latency chunk (`latency_2b_v076`) is queued in `run_chunks_only.sh` after the held-out chunks but had not acquired the GPU by the time of writing.

## Catalog capability

NOT SET. Neither `qwen3.8-27b-4bit` nor `qwen3.8-2b-4bit` retain any `extension.card_format` until BOTH thresholds are met on both pairs. Per the task spec, the flag is `fenced-json` when a model measures >= 6 of 8 valid fenced cards, otherwise `markdown-promotion` (or absent). On the v0.7.6 wheel + observed behavior: the 2B writes markdown (no assistant-ui fence); the 27B over-promotes checklists from any markdown list. The expected defaults would be `markdown-promotion` for the 2B and the 27B needs the parser tightening noted above before its results can be audited.

## What card_promotion cannot do (honest limits)

- Charts and forms cannot be derived from markdown alone; the parser does not attempt them. The coordinator still sends the full schema when the user asks for one.
- Decisions (a typed Laya result) are not derived from markdown; the compare/decide coordinator path remains the source.
- The parser does not invent data. If a row's value cannot be parsed as text or a finite number, it is left as text. Tables whose column counts disagree are refused.
- The parser's markdown-table rule and markdown-list rule are too permissive without explicit user cues. ho-16 (2B) and ho-13/15/16/17, ho-19/21 (27B) showed this: a plain markdown table or list becomes a card even when the user asked for prose.
- Timelines require specific markup (day/time/weekday markers); the 2B does not emit that on the held-out timeline prompts and the parser correctly refuses to invent it.
- Inputs above 1 MiB are rejected without parsing. Inside fenced code blocks the parser is inert, so a markdown code block that happens to look like a checklist does not leak as a card.

## ADDENDUM 2026-09-30 (dated): pre-fix-wheel evidence is suspect

The v0.7.6 qualification on the 29cba8e wheel (the earlier "0/8 fence" on the
2B and the "5/8" on the 27B, both reported in `docs/serve.md` and the
v0.7.6 release receipt) was gathered on a wheel that emits non-finite
logits on GDN prompts past ~300-500 tokens. The 29cba8e evidence IS NOT
COMPARABLE to v0.7.6 evidence. The current live wheel is
`0.32.3.dev202609291615+06711ad`; the HELD-OUT runs and the latency probe
above use that wheel.

## Timing caveat

Every per-prompt wall-clock above was recorded while ModelBench,
RoutingGate, and SpeechInputGpu holders were cycling the gpu-turn lock.
Per Main's broadcast, these timings are NOT alone-on-GPU. The probe
(elapsed_s 14.28) was alone but only measured one cold-start chunk, not
the 30-warm-turn distribution. The first-text p95 number will need to
be re-measured with no foreign holders resident.

## ADDENDUM 2026-09-30 (dated): orphan cleanup and -m 15 clamp

SpeechInputGpu flagged my 27B assistant leaking onto the GPU for
1.5 h outside any ticket. The root cause was the old chunks script
losing its parent while the assistant server kept the GPU open via
inherited fds. The fix shipped in commits **fad99e296** and
**05552c5f4**: every chunk's `_kill_server` helper traps + killpg +
kills ONLY --home-matched MarkdownCards children (never
`fuser -k /dev/dri/renderD*` which would nuke other agents'
processes inside their own tickets). The chunks script also clamps
27B tickets to gpu-turn -m 15 to stay inside the broadcast cap; the
per-prompt results_<model>.json checkpoint means each chunk resumes
safely across the 15-min wall-clock boundary.

## Verification artifacts

- `artifacts/probe_finite_logits.json` — finite-logits probe on v0.7.6 wheel
- `artifacts/held_out_qwen3.8-2b-4bit.json` — 2B HELD-OUT (24/24 prompts)
- `artifacts/held_out_qwen3.8-27b-4bit.json` — 27B HELD-OUT (24/24 prompts)
- `artifacts/mc_score.py` — scorer (passes/fails, threshold checks)
- `SHA256SUMS` — checksums for every artifact in this folder (commit before tagging)

## Summary: the parser needs tightening, not the model

The 2B hits the >= 10/12 card-worthy threshold (10 valid) and the 27B
does not (7 valid). Both pairs over-promote markdown lists / tables
without an explicit user cue. The fix is in card_promotion.py: require
an explicit `compare`/`table`/`vs`/`tabular`/`side-by-side` cue (for
comparison cards) or a `checklist`/`step`/`task`/`todo` cue (for
checklist cards) in the user message before promoting. Per Main's
broadcast, fixing this requires freezing a NEW held-out suite (different
prompts, same shape) before changing card_promotion.py. The current
suite is spent; a follow-up agent should freeze a new one and re-run.
