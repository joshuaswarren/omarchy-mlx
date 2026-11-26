# 2026-09-20 — Qwen3.8 distill converter: one unrecovered test failure

## Event

During the Q4/G64 conversion session, one `pytest tools/test_qwen38_distill_quant.py`
execution (background job bg_28, chained before `verify`) reported
`1 failed, 15 passed in 8.47s`. The invocation piped pytest through
`tail -1`, so the `FAILED <test>` line and traceback were discarded at the
pipe; only the summary line survived.

## Log recovery status

**Cannot recover.** The raw log was truncated by the capturing pipeline
(`| tail -1`) before storage; no other copy exists. This was a logging
mistake in the investigation's own session: full `-v` output is now kept
for every run (see below).

## Reproduction attempts (all green, full logs kept in /tmp/q38/flake-logs/)

- 10 consecutive idle runs: 16 passed each (idle-1..idle-10.log).
- 2 runs under a deliberately recreated concurrent disk+CPU condition
  (sha256 of the 19G artifact on 4 other allowed cores): 16 passed each
  (loaded-1.log, loaded-2.log).

The suite is deterministic in design: every random draw uses pinned
`mx.random.key` seeds; no timing or load-sensitive assertions exist.

## Conditions of the failed run (what is known)

- 16-test suite, same commit content as the surrounding green runs.
- Ran inside the same shell chain as the first artifact `verify`
  (sequential: pytest, then verify), minutes after the 1h47m conversion
  published and exited.
- Total wall time 8.47s vs the suite's normal ~1s: one test consumed
  several extra seconds and failed. Which test is unknown.

## Conclusion

Root cause **undetermined**. 13 green / 1 failed across executions of the
16-test suite; the single failure is unrecovered and unreproducible in 12
targeted attempts, including a load-recreation attempt. Per Main's
direction the suite is treated as **provisionally stable**, not proven
stable; the failed run stands on the record as unresolved. If it
reproduces, capture `-v` output and bisect from the saved logs.

## Process correction

Investigation commands must never pipe test output through `tail`;
store the full stream first, summarize second.
