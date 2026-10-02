# Card latency — time to the first visible card component, 2026-10-02

**Question.** The 9B (Everyday) chart reply makes the user wait ~2 minutes
for the card component; the 4B (Compact) shows nothing for ~91 s and then no
card at all. Where does that time go, and which levers cut it without
breaking the card gates?

**Status.** Decomposition COMPLETE; levers implemented and measured on the
real card prompt; **HELD-OUT v4 NOT YET RUN** (this receipt is labeled
`held-out pending` until the v4 columns land; the gate table at the bottom
is filled only from those runs).

## Decomposition — where the wait goes (real card prompt, one turn per pair)

Prompt: "Show a chart of population for: Paris 2.1M, Tokyo 13.9M, Lagos
21.0M." (`scripts/card_decompose.py`, max_tokens 700, SSE events with
timestamps, tokenizer-counted). Stock kernel 7.1.13-3-1-ARCH; provenance
`verified=match` (live wheel, mlx.core sha256 0b720eb9..., same wheel as the
2026-09-30 receipts).

| pair | prose | payload | component | wait made of |
|---|---|---|---|---|
| everyday9b | 22 tok visible from 6.0 s | valid chart fence, ~300 tok (inferred from the ~61 s gap at the observed decode rate), never visible until close | **66.8 s** (boot a0ee62f9) / 115.9 s (2026-09-30 boot 5498f953) | fence-payload generation; NOT prefill (generating at ~1.2 s), NOT prose decode |
| compact4b | none (fence first) | fence 690 bytes / **217 tok, INVALID**: body is a bare chart object without the `{"version": 1, "components": [...]}` envelope | **none** — `invalid_component` + raw fallback at 45.9 s (a0ee62f9) / 90.9 s (5498f953) | fence generation, then a repair call that declines instantly (below) |
| quality27b (2026-09-30 receipt, for scale) | 20.5 s | valid chart fence | 53.8 s | same shape as 9B |

Promotion timing (code + measurement): fence turns emit at fence CLOSE —
already stream-time. Markdown-promoted cards ran at reply END only; that is
the path the new stream-time promotion attacks.

The 4B's repair path returns without rescuing the bare component:
`_repair_components` first runs a token count and `ensure_context(required)`
— while the main turn still holds its context admission the refusal returns
`{ok: False}` and the repair gives up in ~20 ms. The invalid fence then
falls through to the raw-text fallback.

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
   series >= 2 values — the exact bound the 4B/9B bare shapes tripped), the
   fence tag is exactly `assistant-ui`, and the body must be the envelope.
3. **Deterministic bare-component wrap (shipped, second half of the
   candidate).** `_validated` wraps a fence body that is exactly one object
   with a `type` and no envelope keys, then still requires the full
   `validate_components` pass. Lists, non-components and failing components
   stay rejected; nothing else is normalized.
4. **max_tokens / stop conditions (rejected, recorded).** Decomposition shows
   payload fences of 180-300 tok against the 700-token cap; a lower cap
   risks truncating fences mid-JSON. No numbers pursued because the lever
   has no safe working range.

## Real card prompt, before -> after (same boot a0ee62f9)

| pair | first visible text | first visible component |
|---|---|---|
| everyday9b | 5.96 s -> fence-only reply (card renders; no prose needed) | **66.8 s -> 13.1 s** |
| compact4b | 47.3 s (invalid notice + raw burst) -> 3.3 s (raw body streams visibly) | none -> none (defect unchanged in kind; the wrap now rescues this exact shape — validation on the M2 pending) |

## HELD-OUT v4 — NOT YET RUN

`tests/fixtures/cards_held_out_v4.json` (sha256 `7f807008...`, untouched),
`scripts/run_card_latency_suite.py` scoring identical to
`receipts/2026-09-30-card-promotion/run_suite.py`, plus per-prompt
first-text / first-component timing. Planned runs, one per pair per build
(each declared in the notebook before running): base (pristine 29100916f)
and candidate (levers above) — **this section is empty until they land.**

Gate to pass per pair: valid cards >= 15/18, 0 spurious, first-text p95
<= 2.0 s budget, time-to-first-component median/p95 improved on the 18
card-worthy prompts.

## Tests

Card suites green after rebase onto origin/main (122 tests across
card_promotion / generation / components / card_tuning / coordinator /
release_gate_contract), including 13 new tests (stream promotion units +
coordinator stream-timing + bare-component wrap). Full-suite discovery:
only the 15 pre-existing environment errors present on pristine main.

## Conditions

- jw14m2-linux (M2 Max), boots df9c22c7 / a0ee62f9 / 2d00cae3 on 2026-10-02
  (lane w73 reboots); every timing taken on a quiet window (>= 6 min uptime,
  loadavg < 0.5, PSI cpu avg10 = 0) via `gpu-turn` with a process-group
  trap. Boot/kernel per run recorded in /tmp/CardLatency/window*.txt on the
  M2.
- Worker commit(s): 267d5e36a + 5b810cb53 (this receipt's tree). Baseline
  served from `git archive origin/main` (pristine).
