# Quality-pair TTFT phase decomposition — 2026-10-02

**Question.** Where do the ~3.5 s of assistant/coordinator overhead in
Quality-pair time-to-first-text go (assistant TTFT 6.31 s vs the 2.81 s
engine-level reference on a 300-token prompt), and what can be fixed without
changing answers or gates?

**Answer.** The coordinator was never the cost. With per-phase timestamps on
the real stack (5 quiet-gated runs, fresh conversation per turn, 300-token
user prompt → 325-token engine prompt, warm `qwen3.8-27b-4bit` + `laya-mlx`
pair on jw14m2-linux), the coordinator's entire per-turn path measures
**~3 ms median** of the 6.4 s. The earlier "about 3.5 s is coordinator
overhead" attribution compared two different measurements: the assistant API
turn (325 engine tokens, resident pair) against ModelBench's engine TTFT
(240-token prompt, alone on the GPU). The budget is **not met** — the floor is
the engine's own prompt path.

Instrumentation stays in the product: the coordinator appends one
`ttft-phases.jsonl` line per turn (submit, pair_start, prompt_built,
count_done, admit_done, generating_status_emitted, yield_probe_done,
chat_sent, chat_headers, first_chunk, first_text_emitted; `time.monotonic()`,
comparable across processes on one host), and the serve shim prints
`[ttft]` engine lines (`tokenized` with prompt and tokenize seconds,
`first_chunk` at the first decode step, `first_control_yield` with the
control-token buffer size) to the chat worker log when
`MLX_OMARCHY_TTFT_TRACE=1`.

## Phase table — where the time goes (medians, n=5, arm BASE)

Machine: jw14m2-linux (T6021, 96 GB), kernel 7.1.13-3-1-ARCH, boot
ef549357, quality-home resumed, release venv wheel verified
(`mlx_provenance.py` printed per session, `match`). Timing gate per turn:
load1 < 0.5 and PSI cpu avg10 = 0.00 (both recorded per run; actuals
0.16–0.44 and 0.00).

| phase (submit →) | median s |
|---|---|
| chat_sent (pair start + history + count + admission + status + yield probe) | 0.004 |
| chat_headers (HTTP + engine queue + chat template + tokenize) | 0.113 |
| first_chunk (first early-submit prefill batch + first internal chunk) | 6.441 |
| first_text_emitted | 6.441 |
| first visible text at the client (SSE) | 6.442 (runs 6.362–6.481) |
| SSE delivery lag after the coordinator emits | 0.002 |

Engine internals (worker clock, same host clock): chat-template +
tokenization of the 325-token prompt **0.001–0.017 s**; first prefill batch
(the `mlx-lm-ttft-early-submit` patch submits after the first 128-token
batch) reaches the stream **0.85 s** after tokenize; the first real text
yield arrives **~5.5 s** later — the remaining prefill at the resident-pair
rate plus the first full decode step (a 33-token prompt shows the same
~1.65 s first-decode-step cost with no remaining prefill). The control-token
stream buffers one token (`buffer: 1`), so the first visible text lands on
the second generated token.

## Fixes landed (before/after)

1. **Admission counted 2 prompt tokens for every turn** (`cd216fdc0`).
   `LocalModels.count()` took `len()` of what transformers 5 returns — a
   `BatchEncoding` (`len == 2`: input_ids + attention_mask) — so context
   admission ran on 2 prompt tokens regardless of the real prompt.
   Fixed to unwrap BatchEncoding / batched / flat shapes. Verified on the
   real tokenizer: 625 ids, identical to the encode path; on hardware the
   coordinator now reports 325 where it reported 2 (the engine's own count
   of the same prompt is 325).
2. **Setup/resume published `complete` before releasing the GPU lock**
   (`d4419e82c`). A client that polls `/api/status`, sees `complete`, and
   immediately submits could race the `finally: gpu.release()` and take a
   spurious 409 on its first turn. The outcome is now published only after
   the release.

**Before/after (same machine, same boot, same wheel, quiet-gated, n=5
each; BASE = `893878e40` line, FIX = `cd216fdc0`):**

| arm | prompt tokens counted | fvt median s | fvt min–max | text (all runs) |
|---|---|---|---|---|
| BASE | 2 (bug) | 6.397 | 6.293–6.430 | 108 chars |
| FIX | 325 | 6.427 | 6.405–6.459 | 108 chars |

First-visible-text is unchanged within noise (±45 ms), which is the point:
the coordinator was already ~3 ms. The admission fix changes what the
product counts, not what the user waits. Answers are identical across arms
(greedy decoding; identical 108-char replies on every run of every arm —
first 64 tokens included in the session JSONs).

## Verdict vs the 2.0 s budget

**Not met, and the floor is the engine.** The coordinator contributes
~3 ms of the 6.4 s; HTTP + SSE add ~0.1 s. The remaining ~6.3 s is the
worker's own request path for a 325-token prompt on the resident pair:
prefill at the resident-pair rate (the pair-gates receipt measured 42–76
tok/s resident vs 97–103 alone on GPU) plus the first decode step. Even the
best engine-level reference (ModelBench, 240-token prompt, alone on the
GPU) is 2.45–2.81 s — over budget before the coordinator exists. Meeting
2.0 s on 300-token prompts needs engine-side prefill/decode-step work (the
`mlx-lm-ttft-early-submit` land and kernel tickets), not assistant changes.
Reusing the stable prefix across turns (`--prompt-cache-size`) would help
multi-turn chats but re-opens the memory admission bound ((cache+1) ×
context worst case) — a gate decision, not a free win; deliberately not
changed here.

## Provenance and conditions

- Every session printed `scripts/mlx_provenance.py` for the loaded wheel;
  all four report verified match (recorded in each `ttft-profile.json`).
- Kernel 7.1.13-3-1-ARCH, boot ef549357-dcc3-441e-b0e1-f89e3930b3b2, M2 Max
  T6021 96 GB. All GPU work ran under `gpu-turn` tickets with the
  PairGates guard (leftover kill + watchdog); `fuser /dev/dri/renderD128`
  clean outside our own workers.
- Per-run `loadavg` and PSI in each session JSON (`gate`, `load_after`,
  `psi_after`): load1 0.16–0.49, PSI cpu avg10 0.00 on every measured run.

## Artifacts

On jw14m2-linux (private notebook, not committed):
`~/.local/share/apple-silicon-lab/artifacts/QualityTtft/` —
`profile-base/`, `profile-base2/` (BASE), `profile-fix/` (stale duplicate of
BASE, kept for the record), `profile-fix2/` (FIX): `ttft-profile.json`
(per-run fvt, phase dict, engine lines, gates, text head), `harness.log`,
`launch.log`, plus `SHA256SUMS`. Coordinator phase lines for every turn are
also in `~/mlx-assistant-runs/quality-home/assistant/logs/ttft-phases.jsonl`;
engine lines in `assistant/logs/quality-chat.log`.

## Test and gate status

`PYTHONPATH=serve python3 -m unittest discover -s tests -q` on the M2-class
stack: the assistant suites touched by this change pass
(coordinator/pairs/history/http/speech_yield, 135 tests). On hosts without
mlx the mlx-dependent modules cannot import (environment limit, not a code
failure). No gate was weakened: the context cap shim, qualification gates,
and memory admission math are untouched; card behavior untouched (no
component-code changes); zero-CPU contract untouched (no engine/C++ or
mlx-core changes in this receipt).
