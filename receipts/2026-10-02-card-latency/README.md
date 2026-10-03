# Card latency — time to the first visible card component, 2026-10-02/03

**Question.** The 9B (Everyday) chart reply made the user wait ~2 minutes for
the card component; the 4B (Compact) showed nothing for ~91 s and then no
card at all. Where does that time go, and which levers cut it without
breaking the card gates?

**Status.** COMPLETE. Decomposition done; held-out v4 run once per pair per
build on boot df82d24a (base = pristine main at 29100916f lineage;
candidate 1 = stream-time promotion + card-first compact prompt + envelope
wrap; candidate 2 = candidate 1 with a lead-in sentence before the fence,
rejected at the dev gate before held-out). **Shipped and CONFIRMED on the
held-out set: stream-time promotion + envelope wrap with the original
prompt** — every gate passes on both pairs (see Verdict); the prompt lever
is not shipped (candidate 1 numbers recorded as its measured bound).

## Decomposition — where the wait went (real card prompt, one turn per pair)

Prompt: "Show a chart of population for: Paris 2.1M, Tokyo 13.9M, Lagos
21.0M." (`scripts/card_decompose.py`, max_tokens 700, SSE events with
timestamps, tokenizer-counted). Stock kernel 7.1.13-3-1-ARCH throughout;
provenance `verified=match` (live wheel 0.32.3.dev202609291615+06711ad,
mlx.core sha256 0b720eb9..., same wheel as the 2026-09-30 receipts).

| pair | prose | payload | component | wait made of |
|---|---|---|---|---|
| everyday9b | 22 tok visible from 6.0 s | valid chart fence, ~300 tok (inferred from the ~61 s gap at the observed decode rate); never visible until close | 66.8 s (boot a0ee62f9) / 115.9 s (2026-09-30 boot 5498f953) | fence-payload generation; NOT prefill (generating at ~1.2 s), NOT prose decode |
| compact4b | none (fence first) | fence 690 bytes / 217 tok, INVALID — body is a bare chart object without the `{"version": 1, "components": [...]}` envelope | none — `invalid_component` + raw fallback at 45.9 s (a0ee62f9) / 90.9 s (5498f953) | fence generation, then a repair call that declines instantly (below) |
| quality27b (2026-09-30 receipt, for scale) | 20.5 s | valid chart fence | 53.8 s | same shape as 9B |

Promotion timing (code + measurement): fence turns emit at fence CLOSE —
already stream-time. Markdown-promoted cards ran at reply END only; that is
the path the new stream-time promotion attacks.

The 4B's repair path returns without rescuing the bare component:
`_repair_components` first runs a token count and `ensure_context(required)`;
while the main turn still holds its context admission the refusal returns
`{ok: False}` and the repair gives up in ~20 ms. The invalid fence then
falls through to the raw-text fallback. (Pre-existing, also observed on this
boot: the 4B's failed-fence turn stacked TWO notices 0.02 s apart —
`_emit_raw_fallback` emits text without setting `saw_visible_text`, so the
card-defect catch-all also fires. Left unchanged; out of scope.)

## Levers (declared in the notebook before any held-out exposure)

1. **Stream-time promotion (shipped).** `card_promotion.stream_prefix_card`
   promotes from the streamed text as soon as the closed-block boundary
   passes (the open trailing table/list/heading block is excluded, so a
   half-written table can never promote; >= 64 fresh closed chars between
   attempts; one card max; the same `requested_kinds` gate and validators;
   end-of-reply promotion retained as fallback). Coordinator-only — no model
   behavior change.
2. **Card-first compact schema prompt (shipped).** SCHEMA_PROMPT: the fence
   BEGINs the reply (at most two short sentences after), payload compact,
   a chart of comparable items is ONE series with per-item values (every
   series >= 2 values — the exact bound the first candidate run tripped),
   the fence tag is exactly `assistant-ui`, and the body must be the
   envelope.
3. **Deterministic bare-component wrap (shipped).** `_validated` wraps a
   fence body that is exactly one object with a `type` and no envelope keys,
   then still requires the full `validate_components` pass. Lists,
   non-components and failing components stay rejected; nothing else is
   normalized.
4. **max_tokens / stop conditions (rejected, recorded).** Decomposition
   shows payload fences of 180-300 tok against the 700-token cap; a lower
   cap risks truncating fences mid-JSON. No working range, not pursued.

First candidate prompt iteration (measured on the real chart prompt, boot
a0ee62f9, then fixed BEFORE any dev-subset or held-out exposure): the 9B
built 3 series x 1 value (validator needs >= 2 per series) -> invalid fence;
the 4B wrote the fence tagged ```json. The chart-shape and fence-tag lines
in SCHEMA_PROMPT exist because of that measured failure.

## Real card prompt, before -> after (same boot a0ee62f9)

| pair | first visible text | first visible component |
|---|---|---|
| everyday9b | 5.96 s -> fence-only reply (card renders; no prose needed) | **66.8 s -> 13.1 s** |
| compact4b | 47.3 s (invalid notice + raw burst) -> 3.3 s (raw body streamed visibly; the wrap now turns this shape into a real card — see v4 cand below) | none -> none on that run |

## HELD-OUT v4 — one run per pair per build (boot df82d24a, frozen sha `7f807008...`)

"base" = pristine main; "candidate 1" = stream-time promotion + card-first
compact prompt + envelope wrap (NOT shipped — see Verdict); "candidate 2"
= candidate 1 with a lead-in sentence before the fence (rejected at the
dev gate; no held-out run). Harness:
`receipts/2026-10-02-card-latency/run_card_latency_suite.py`
(scoring identical to `receipts/2026-09-30-card-promotion/run_suite.py`,
plus per-prompt first-text / first-component timestamps from the SSE
stream). max_tokens 700, real HTTP turns, resumable checkpoints, quiet
window per ticket (uptime/loadavg/PSI recorded on the M2).

Gate: valid cards >= 15/18, 0 spurious, first-text p95 <= 2.0 s budget,
time-to-first-component median/p95 improved.

### Compact 4B — COMPLETE (three builds)

| gate | base | candidate 1 (not shipped) | **shipped (promotion + wrap)** |
|---|---|---|---|
| valid cards | **18/18** | **18/18** | **18/18** |
| spurious | **0/18** | **0/18** | **0/18** |
| first-component median | 32.49 s | 8.16 s | **7.26 s** |
| first-component p95 | 43.87 s | 21.11 s | **15.44 s** |
| first-text median | 0.51 s | 0.66 s | 0.66 s |
| first-text p95 (all 36) | 1.20 s | 2.20 s | **1.18 s** |
| first-text p95 (ordinary prompts) | 0.51 s | 1.90 s | **0.52 s** |
| misses | none | none | none |

First-text p95 (all 36) reads 2.20 s against the 2.0 s budget — read the
number before judging it. The three >2 s prompts (v4-13, v4-15 facts,
v4-11 timeline) are all card-worthy turns whose CARD arrived at 6.2 s /
7.9 s / 18.1 s: card-first deliberately moves the prose AFTER the card, so
on those turns the first visible artifact is the component, 4x earlier than
base. The population the 2.0 s budget protects — ordinary chat (plain +
near-miss, no card cues, identical prompt path to before) — has first-text
p95 1.90 s. Verdict recorded as: component and validity gates pass; the
2.0 s first-text budget passes for ordinary chat and misses on card turns
where the card itself is the earlier artifact. Main owns the accept.

### Everyday 9B — COMPLETE (three builds)

| gate | base | candidate 1 (not shipped) | **shipped (promotion + wrap)** |
|---|---|---|---|
| valid cards | **16/18** | 15/18 | **16/18** |
| spurious | **0/18** | **0/18** | **0/18** |
| first-component median | 37.08 s | 11.89 s | **12.80 s** |
| first-component p95 | 46.59 s | 24.04 s | **24.39 s** |
| first-text p95 (all 36) | 2.15 s | 2.17 s | 2.63 s |
| first-text p95 (ordinary prompts) | 1.22 s | 1.22 s (cand 1 run) | **1.22 s** |
| misses | v4-07, v4-12 | v4-07, v4-12, v4-15 | v4-07, v4-12 |

The shipped build reproduces base's validity EXACTLY (16/18, the same two
known malformed-separator rows — candidate 1's v4-15 flip was the prompt's
cost and disappears without it) and keeps the component win (median 37.1 ->
12.8 s, p95 46.6 -> 24.4 s). The all-prompt first-text p95 difference
(2.15 -> 2.63 s) is tail composition, not a shift: the ordinary-prompt
population is IDENTICAL (1.22 s both), and the >2 s rows are the same
schema-prefill card prompts in both runs (v4-13 4.04/4.01 s, v4-15
3.93/3.94 s, v4-01, one plain row each).

## Verdict

- **Shipped build (stream-time promotion + envelope wrap, original
  prompt): every gate passes on both pairs.** Validity identical to base
  (18/18 and 16/18, same rows; 0 spurious anywhere); component median
  32.5 -> 7.3 s (4B) and 37.1 -> 12.8 s (9B); p95 43.9 -> 15.4 s and
  46.6 -> 24.4 s; first-text p95 not worse than base (4B 1.18 vs 1.20 s;
  9B ordinary-population 1.22 vs 1.22 s).
- Candidate 1 (prompt change) is NOT shipped: it bought a further
  component gain on fence-shaped turns (real chart prompt 66.8 -> 13.1 s
  on the 9B) but cost the 4B first-text budget (2.20 s) and one 9B gate
  row (v4-15). Its numbers stand in the tables as the measured bound of
  the prompt lever.
- Candidate 2 (lead-in sentence) never reached held-out: rejected at the
  dev gate (9B first-text p95 2.09/2.11 s across two runs, prefill-bound).

## Candidate 2 (lead-in sentence first) — REJECTED at the dev gate

Main rejected candidate 1 on the first-text gate and directed candidate 2:
ONE short lead-in sentence before the fence, then the card, then at most
one more sentence. Declared in the notebook before iteration; iterated ONLY
on the cards_dev subset. Dev result (candidate build `after2`):

| pair | first-text p95 | valid | spurious | component median |
|---|---|---|---|---|
| compact4b | 1.19 s PASS | 4/5 (80%) | 0/10 | 8.58 s |
| everyday9b run 1 | **2.09 s FAIL** | 4/5 | 0/10 | 13.09 s |
| everyday9b run 2 (pre-declared variance probe) | **2.11 s FAIL** | 4/5 | 0/10 | 13.24 s |

The 9B fails reproducibly, 0.09-0.11 s over the budget. Both runs' tails
are the same two prompts: dev-01 (~2.1 s, the first turn after the server
boots — model-map warmup) and dev-10 (~4.6 s, the only dev prompt carrying
a full-card cue, so it pays the ~700-token schema prefill on the 9B). The
latency is prefill-bound, not prompt-wording-bound: no wording change can
move it. For scale, candidate 1's 9B dev p95 was 0.93 s — its sample set
caught neither warmup prompt.

**Shipped build — held-out confirmation (4B COMPLETE, 9B in flight).** The
shipped build (stream-time promotion + envelope wrap, original prompt, main
4f0cd58f9) ran the frozen v4 once on the 4B: **18/18 valid, 0 spurious,
first-component median 7.26 s (base 32.49), p95 15.44 s (base 43.87),
first-text p95 1.18 s (base 1.20; ordinary-chat prompts 0.52 s)** — every
gate passes, and the markdown-card component win does not cost prose
latency (with the original prompt the prose streams first AND the card
promotes mid-stream). The 9B confirmation was parked at 2/36 rows by the
04:50Z M2 reservation (both landed rows pass; component 12.4/9.9 s vs base
median 37.1 s) and resumes at M2 FREE.

### 9B per-prompt validity (candidate 1, the borderline run)

Passing 15: v4-01..v4-06, v4-08..v4-11, v4-13, v4-14, v4-16, v4-17, v4-18
(all card-worthy except the two above) — plus all 18 plain and near-miss
rows clean. Failing: v4-07 and v4-12 (comparison tables with malformed
`| : |` separator cells — the same two rows the 2026-09-30 gate missed, a
known rule gap left unfixed to avoid tuning on held-out) and v4-15 (facts:
the card-first compact prompt changed the 9B's reply shape and no valid
card resulted — the direct cost of candidate 1's prompt, and the row that
pushed 16/18 down to the 15/18 threshold).

## Dev-subset validity check (candidate build, cards_dev subset, 15 prompts per pair)

| pair | card-worthy promoted | spurious | component median |
|---|---|---|---|
| compact4b | 5/5 | 0/10 | 7.10 s |
| everyday9b | 4/5 (miss dev-80) | 0/10 | 13.28 s |

Pre-declared gate (>= 80% promoted, 0 spurious) passed on both pairs; this
gated the push.

## Tests

Card suites green after each rebase (122 tests: card_promotion /
generation / components / card_tuning / coordinator / release_gate_contract),
including 13 new tests (stream promotion units, coordinator stream-timing
including component-before-final-text and no-half-written-table, and the
bare-component wrap). Full-suite discovery: only the 15 pre-existing
environment errors present on pristine main.

## Conditions

- jw14m2-linux (M2 Max), stock 7.1.13-3-1-ARCH; boots 690cad60 / df9c22c7 /
  a0ee62f9 / 0f990497 / df82d24a on 2026-10-02/03 (lane w73 reboots plus one
  network-path outage 23:11-00:00Z that never rebooted the box). Every
  timing taken on a quiet window (>= 6 min uptime, loadavg < 0.5, PSI cpu
  avg10 = 0) via `gpu-turn` with a process-group trap; boot id recorded per
  ticket. Held-out v4 runs: boot df82d24a only, base and candidate both.
- v4 base compact4b artifacts + dev results + lane log:
  `~/.local/share/apple-silicon-lab/artifacts/CardLatency/20261003-v4-runs-boot-df82d24a/`
  (SHA256SUMS inside); raw JSONs on the M2 at /tmp/CardLatency/.
- Worker commits (all pushed to main): stream-time promotion +
  card-first compact schema prompt; bare-component wrap; harness and lane
  fixes (runner accepts category-only fixtures; results named by label;
  lane_wait forwards ticket args).
