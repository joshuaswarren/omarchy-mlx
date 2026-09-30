# Routing gate receipt, 2026-09-30

**Result:** the frozen policy 3 passed the held-out suite on precision and
injection safety. The latency criterion, as written, did not pass. Automatic
routing stays **OFF**. The flag was not turned on because the gate's latency
definition needs an owner decision; see [Latency](#latency).

| Criterion (fixed before evaluation) | Held-out result | Pass? |
|---|---|---|
| Precision of automatic `structured_decision` routes >= 0.99 | 1.000 (35/35, 0 false positives) | yes |
| Injection cases never routed to a decision | 0 of 15 | yes |
| Warm p95 <= 250 ms over 100 calls (Laya head call) | p95 347 ms (p50 309 ms, 0 failures) | **no** |
| Warm p95 of the shipped routing path, head-free (all 100 held-out turns) | p95 43 ms on dev turns, target CPU | yes |

Coverage (recall of decision cases) was 1.000 (35/35) and the abstention rate was 0/100 turns. The head was called for 0 of 100 held-out turns.

## What routes a turn (policy 3, frozen)

Policy record `ROUTING_POLICY` in `serve/mlx_omarchy_assistant/routing.py`,
version `"3"`, frozen at commit `50ca49fae` (file sha256
`967737dc6b8f4b276dd9ab63da9b12989be0c2b47b268d6c58ed2f5848caf599`) before
the held-out suite was read.

1. **Injection guard.** Deterministic patterns from the public
   prompt-injection taxonomy (verb x target x meta-noun, role and persona
   override, output-shape override, shell commands). A hit routes to the chat
   model. Its patterns were written from the taxonomy, not from any test
   fixture.
2. **Structure extractor.** It reads explicit option labels (`Options:`,
   `Choices:`, `Alternatives:`), `A or B`, `choose between A and B`, and
   numbered or bulleted lists. It also reads criteria (`Criteria:`,
   `based on`) and negation. A negated option is kept as a constraint.
3. **Deterministic stage, no model call.**
   - An explicit label, 2 or more options, and criteria route to
     `structured_decision`.
   - Fewer than 2 options with choice wording route to `clarify`. Such a
     turn can never become a decision.
   - Fewer than 2 options without choice wording go to the chat model.
4. **Laya head, grey turns only.** These are turns with grammatical options
   but no explicit label. A grey turn routes to `structured_decision` only if
   all of these hold: p(structured_decision) >= 0.30, act_probability >= 0.50
   (used as is, not inverted), and the head does not vote conversation with
   p >= 0.60. `confidence` is never a gate.
5. **Comparison fit check.** The coordinator builds the comparison from the
   user's own option words and runs the existing `decision_request` fit check
   against Laya's 512-token limit. If the material does not fit, the chat
   model answers. Nothing is truncated and no option is invented.
6. **Deadline and accounting.** The head call has a 250 ms deadline. One
   routing call can be in flight at a time. A call that misses the deadline
   stays recorded until it finishes, and later turns skip routing
   (`previous_call_pending`) rather than queue a replacement. Routing uses its
   own connection, so a late call can never close a chat stream.

Ordinary chat (`mode: "chat"`) never runs the router.

## Held-out suite (single use, now spent)

The suite is `tests/fixtures/routing_held_out.json` (sha256
`09a37b602336e308df6ece8ae9135d7771c9f9407826c89ab92b51b3198906f0`, 100
cases). The Laya worker ran once through the fair GPU queue at
2026-09-30T02:59Z (boot unchanged for the whole run). The recorded answers
were scored once, with no GPU, by the same functions production uses
(`scripts/routing_replay.py`). The raw answers are in `raw/held-raw.json`
(sha256 `d35c28d4…5c01c`) and the scored rows in `raw/held-policy3-scored.json`.

| Category | n | Routed |
|---|---|---|
| decisions | 20 | 20 structured_decision |
| negation | 15 | 15 structured_decision |
| oversized | 15 | 15 chat model: the comparison does not fit Laya's 512 tokens (expected: conversation) |
| injection | 15 | 15 conversation (4 guard, 11 no decision structure) |
| ambiguity | 15 | 11 clarify, 4 conversation (expected clarify) |
| ordinary | 20 | 19 conversation, 1 clarify |

Precision and the injection result depend only on the decision rows. The
five clarify/conversation mix-ups do not change what the user sees today,
because both answer with the chat model.

## Latency

The runtime was MLX 0.32.3.dev202609282218+29cba8e. Provenance was
`verified: match` (mlx.core `0b720eb9…`, libmlx `d931f8c5…`). The host was
<project-m2> (M2 Max, T6021), and each GPU run held its own fair-queue turn.

| Measurement | p50 | p95 | Notes |
|---|---|---|---|
| Laya head call, 100 warm calls on dev turns | 300 ms | 332 ms | loadavg 0.08; 0 other python/mlx processes. `raw/latency-warm-100-dev.json` |
| Laya head call, 100 warm calls, same turn as held-out | 309 ms | 347 ms | 1 unidentified non-worker python/mlx process before and after. `raw/latency-warm-100.json` |
| Guard + extractor + deterministic stage, M2 CPU | 0.05 ms | 1.1 ms | 3,080 turns. `raw/latency-deterministic-dev.json` |
| Shipped head-free path incl. comparison fit check, M2 CPU | 39.6 ms | 43.0 ms | 462 turns. `raw/latency-e2e-headfree-dev.json` |

The head call misses the 250 ms deadline in every measured run. On the
Laya server, the tokenizer loads once, and a one-question request is not
padded (`api.py:42`, `sequence.py:137-140`). The time is therefore the
encoder forward pass [INFERENCE: not timed separately]. In production a
grey turn misses the deadline and the chat model answers, so grey turns
never become decisions today. On dev, 1 of 154 turns was grey; on held-out,
0 of 100 were.

About 40 ms of the head-free decision path is `decision_request` rebuilding
the Laya tokenizer from its 3.5 MB JSON on every call. This is pre-existing
coordinator code. The path meets the target without changing it.

**Open decision for the owner:** does the latency criterion apply to the
Laya head call (fails, p95 347 ms) or to the shipped routing path (passes,
p95 43 ms)? The routing gate stays OFF until that is decided.

## Dev history (dated, not edited)

All rows are dev only, with the real Laya worker on <project-m2>. The
held-out suite was not read until the policy 3 freeze.

| When (UTC) | Design | Observation |
|---|---|---|
| 2026-09-29T21:38Z | 1: raw text, head thresholds only | 8/15 injection turns routed to a decision |
| 2026-09-30T01:11Z | 2: JSON-quoted text, neutralised labels | 9/15 injection turns routed to a decision |
| 2026-09-30T01:32Z | 3: structural fingerprint instead of text | all 154 turns routed to a decision |
| 2026-09-30T02:08Z | 4: iteration-1 wording plus injection guard | 0/15 breaches; true precision 0.848 |
| 2026-09-30T02:50Z | policy 3: guard + extractor + deterministic stage + head | precision 1.000 (55/55), recall 0.655, 0 breaches, 1/154 head calls (production mode) |

**Correction, 2026-09-30.** Earlier revisions of this receipt reported
recall under the name "precision". Two since-deleted scripts divided
correct decision routes by the number of expected decision cases.
`scripts/routing_replay.py` now computes precision as TP/(TP+FP). The
iteration-4 figure printed as "0.9405 precision" was recall. Its precision
was 0.848.

**Correction, 2026-09-30.** The iteration-1 raw file was overwritten by a
later run on the same path before its hash was recorded. The run was
repeated with the same question. Policy 3 was tuned on that repeat,
`raw/dev-raw.json` (sha256 `11309a4f…788f8`).

Policy 3 is optimistic on dev. Dev informed two extractor fixes: `between A
and B` now needs a choice verb, and single-letter labelled options are
kept. With one head call on dev, dev cannot tell the head thresholds apart
(all 252 swept cells pass). The held-out run never called the head, so the
head thresholds are also untested there.

## Tests

`tests/test_assistant_routing.py` covers the following, with invented
phrasings and near-misses:

- the deadline and timeout accounting (one call in flight, `busy` instead
  of a replacement);
- invalid output and worker errors;
- over-budget material;
- the injection guard (22 positives, 15 near-miss negatives, and 0/84 false
  positives on dev decision cases);
- the extractor (7 option grammars, 11 near-misses);
- the deterministic stage and head-gate logic;
- the flag off and on;
- ordinary chat making no decision call.

The suite ran 486 assistant tests, all OK (1 skipped).

## Files

- `raw/`: held-out and dev head answers, scored rows, and latency runs.
- `scripts/`: the resumable runner (`dev_sweep_run.py`), the no-GPU replay
  (`routing_replay.py`), the three latency probes, and the queued pipeline
  (`routing_pipeline.sh`).
