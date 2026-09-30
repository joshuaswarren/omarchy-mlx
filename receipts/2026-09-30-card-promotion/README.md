# Card promotion receipt, 2026-09-30

Cards on the default pair without the model emitting JSON: when a chat turn
ends without a valid `assistant-ui` fence, `card_promotion.extract_text`
builds at most one card from the reply's markdown. This receipt covers three
held-out suites. v1 and v2 are spent and failed; v3 is frozen and **not yet
run**, because the project M2 went offline at about 06:55 CDT and Main then
put it off-limits to every agent (reserved for another job).

## Current status against the frozen thresholds

| Gate | 2B Everyday | 27B Quality |
|---|---|---|
| v1: >= 10/12 cards, 0 spurious on 12 | 10/12, **1 spurious: FAIL** | 7/12, **6 spurious: FAIL** |
| v2: >= 15/18 cards, 0 spurious on 18 | partial, **3 spurious in 12 recorded: FAIL** | 36/36 recorded on the M2, counts not retrieved |
| v3: >= 15/18 cards, 0 spurious on 18 | NOT RUN (M2 off-limits) | NOT RUN (M2 off-limits) |
| Every card passes `validate_components` | yes on v1 (the server only emits validated cards) | yes on v1 |
| 2B first-text p95 <= 2.0 s, 30 warm turns | NOT MEASURED | n/a |
| `extension.card_format` in the catalog | not set | not set |

No pair qualifies on this evidence. `card_format` stays unset on both entries:
neither model has a measured >= 6 of 8 valid fenced cards, and absent now
means "send no schema and promote markdown", which is the behaviour both
pairs need.

## What changed in the product (after v1 and v2 failed)

Failure classes found in the spent suites:

1. **Explanatory table promoted** (v1 2B ho-16 "Explain a hash table"): the
   v1 rules promoted any pipe table. The v2 2B spurious cards (v2-25 and
   v2-27 `comparison`, plain questions) are the same class by inference; the
   v2 runner did not record the reply text, so this is not proven.
2. **Model fence on a plain prompt** (v1 27B: 6 spurious `checklist` cards
   from its own fenced JSON, prompted by the compact schema on every turn).
3. **Timeline without dates** (v1 2B ho-07/ho-08 missed): the model wrote
   labelled phases or a Week table; v1 only accepted time/weekday/Day-N.
4. **Superlinear parser**: `extract_text("- a" + 200 KB of spaces + "b",
   "checklist, table")` did not return within 120 s (timeout exit 124) with
   the v1 module; lazy `.+?` before `\s*$` backtracks quadratically.

Fixes, each covered by `tests/test_assistant_card_promotion.py`:

- A card needs a request for that artifact in the user's message and a
  matching reply structure; explanation-led questions and prose requests stay
  prose (class 1). Rules are in the module docstring and `docs/serve.md`.
- Ordinary chat sends no card schema. The full schema goes only to Laya
  turns, messages naming charts/forms/decisions/options/facts/sources, and
  catalog entries with `card_format: "fenced-json"` (class 2; also shortens
  every 2B prompt). The compact schema and `user_requested_full_schema` are
  deleted; `scripts/bench/chat_model_bench.py` and `probe_finite.py` (the
  ModelBench scripts) now mirror this policy.
- Timelines accept labelled items, `Week 2`/`Phase 3`/`Q1` markers, headings
  and tables whose first column is the time (class 3).
- Every pattern is linear; a test feeds eight 1 MiB hostile shapes and two
  hostile user messages and requires each call under 2 s (class 4).
- Cards are titled `Checklist (from reply)` etc.; the validator rejects
  unknown keys, so the title is the honest label.

## Evidence gathered without the M2

- `PYTHONPATH=serve python3 -m unittest discover -s tests -p "test_assistant*.py"`
  ran 437 tests, OK (1 skipped), and all six `bun tests/js/*` files pass on
  the orchestrator host.
- DEV set (`tests/fixtures/cards_dev.json`, 52 prompts, mine to tune on)
  crossed with every reply shape the models were seen to write, including
  the failure classes: 67 of 67 card-worthy pairs promote, 0 cards on the
  plain and near-miss prompts. dev-52 ("Give me a 2x2 table") was relabelled
  card-worthy because the user asks for a table.
- 122 real replies recorded by the chat-model bench
  (`receipts/2026-09-30-chat-model-bench/v3/*.json`, GSM8K and IFEval turns
  from four models) with their prompts: 0 cards.

These are DEV results. They are not the gate.

## Frozen HELD-OUT v3 (not run)

- `tests/fixtures/cards_held_out_v3.json`, sha256
  `5cce5bb75b02b3c7815c5e7ecc48212e51ffc5168af17d099f95fa51985a6ebb`,
  committed in `e64cc4f50` before `extract_text` or any model saw it.
- 18 card-worthy (4 checklist, 4 comparison, 4 timeline, 3 facts, 3 mixed),
  9 plain, 9 near-miss. Thresholds are in the file.
- Run: `MARKCARDS_HOME=<home> bash run_v3.sh` on the M2 (six 2B chunks at
  `gpu-turn -m 8`, twelve 27B chunks at `-m 15`, then the latency probe).
  `run_suite.py` records per prompt the card types, titles, a `promoted`
  flag (title ends with `(from reply)`), and the reply text, so every miss
  and every spurious card can be attributed to the fence or to promotion.
  `run_latency.py` records load average, the python/mlx process list and the
  wheel provenance before and after the 30 turns.

## HELD-OUT v2 (spent, partial)

`tests/fixtures/cards_held_out_v2.json`: 36 prompts, run with the v1 rules
plus the cue-word schema prompt. The runner needed an `expect` field; it was
added from `category` after the first chunks crashed (prompt texts,
categories and kinds are byte-identical to the committed file, sha256
`04d6fd79...`; as-run file sha256 `c1cf3594...`, commit `4c272ddeb`).

- 2B: v2-25..v2-36 recorded (the earlier chunks crashed before recording).
  v2-25 `comparison`, v2-27 `comparison`, v2-35 `checklist` on non-card
  prompts: 3 spurious, so the 0-spurious threshold fails regardless of the
  missing 24. The rerun of v2-01..v2-24 was cut off by the M2 outage.
- 27B: 36/36 recorded in the M2 results file. I did not copy it before the
  M2 went away, so its counts are not in this receipt.

## HELD-OUT v1 (spent), v0.7.6 wheel 0.32.3.dev202609291615+06711ad

`tests/fixtures/cards_held_out.json`, sha256
`0df8e489e90de6c3552115e29578bb426f589bccbaec690b46d95af62513775b` (with the
embedded `sha256` field), 24 prompts, max_tokens 700, real HTTP turns.
Raw results: `artifacts/held_out_qwen3.8-2b-4bit.json`,
`artifacts/held_out_qwen3.8-27b-4bit.json`. Wall times were recorded while
other agents' jobs shared the GPU and are not latency evidence.

### qwen3.8-2b-4bit

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

### qwen3.8-27b-4bit

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

The 27B cards were its own fenced JSON (checklists on plain prompts, none on
the comparison prompts), not promotion.

## Finite-logits probe (v0.7.6 wheel)

`probe_finite_logits.py`, raw output `artifacts/probe_finite_logits.json`:

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

The 2B wrote a markdown task list without a fence; promotion built the
checklist. No non-finite logits in 700 tokens.

## What cannot be promoted

- Charts, forms, decisions and facts with sources: markdown cannot carry
  them; the full schema is sent when the user names them and only the model
  can produce them.
- A request the rules do not recognise ("what should I bring camping?")
  gets prose even if the reply is a list.
- A comparison written as bullets instead of a table gets no card.
- Timelines need three or more dated or labelled entries.
- The parser never invents data; cells are the reply's own text, truncated
  to the validator's bounds.

## Files

- `run_suite.py`: held-out runner (v2 and v3), chunked and resumable.
- `run_v3.sh`: the v3 + latency driver for the M2.
- `run_latency.py`: 2B first-text probe, 30 warm turns.
- `run_held_out.py`, `probe_finite_logits.py`: the v1 runner and probe.
- `artifacts/`: v1 results, the probe result, `mc_score.py` (v1 scorer).
