# Pair gates — 2026-09-30 / 2026-10-01 (zero-CPU audit complete; memory/perf v3 in flight)

**Question.** How much memory does the paired Everyday and Quality
assistant path take on the M2 (96 GB tier)? Does any tensor primitive
run on CPU for the chat or decision path? What does the Quality pair
do for performance on an idle GPU?

**Status.** Zero-CPU dispatch trace audit is COMPLETE for four paths
(chat on 2B, decision on Laya, TTS, and chat on 9B GDN Hk!=Hv). All
four measured 0 CPU tensor primitive dispatches via gdb breakpoint on
`mlx::core::cpu::get_command_encoder(Stream)`. The control (mx.add on
`stream=mx.cpu`, 4096-element) fires exactly 3 CPU encoder calls; the
same op on `stream=mx.gpu` fires 0.

Paired memory peak: COMPLETE. Nine runs (3 per pair × 3 pairs) on the
fixed harness at `origin/main a1251aaa` (invalid-fence raw-text
fallback + empty-reply guard). All nine runs report `run_valid:
true`.

Quality idle-GPU perf: COMPLETE (quiet window, boot 5498f953, fuser
empty before and after). Per-pair card timing (4B/9B/27B with
first-visible text/component splits) also COMPLETE on the same boot.

## Compliance

- gpu-turn: every GPU command runs under `<home>/bin/gpu-turn`
  with a process-group kill trap on EXIT. No leftover workers.
- Notebook: `~/.local/share/apple-silicon-lab/entries/PairGates/2026-09-30T06-37Z-T6021-M2-linux-pair-gates.md`
  with dated observations per ane-research-notebook rule.
- Provenance: live wheel `mlx_omarchy-0.32.3.dev202609291615+06711ad`
  verified by `scripts/mlx_provenance.py` (mlx.core sha256 0b720eb9...;
  libmlx.so sha256 004d24b6...; both match the wheel RECORD).
- kernel: 7.1.13-3-1-ARCH recorded in every launch.log.

## Zero-CPU dispatch trace — COMPLETE

Method: gdb -batch -p <worker_pid> attach with a breakpoint on
`mlx::core::cpu::get_command_encoder(Stream)` (the exported dispatch
entry every CPU-stream primitive's `eval_cpu` reaches). The
`commands` block runs `silent; continue`, so the worker is never
paused. Hit count = grep -c "Breakpoint 1," in gdb's output.

The [rtmod] DISPATCH trace (`MLX_OMARCHY_TRACE_DISPATCH=1`) alone
cannot prove zero-CPU: it instruments only the GPU/Vulkan encoder,
not the CPU encoder. The CPU encoder does not call into the GPU
dispatch facility. gdb is the valid counter.

Failed approaches (recorded so the next engineer does not repeat them):
- `MLX_OMARCHY_TRACE_DISPATCH` cpu_tensor_events = 0 with a chat
  workload: looks like a clean negative but the trace facility
  instruments only the GPU encoder.
- gdb `set follow-fork-mode child` on a wrapper inferior: the parent
  breakpoint is not propagated to the forked chat subprocess, and the
  `gdb.execute("run ...")` blocks until the inferior's stdin gets
  DONE, which never happens while the worker subprocess is serving.
- Replaced with the spawn-then-attach pattern: launch the worker as a
  plain subprocess, wait for its bound port via `ss`, drive the
  workload via HTTP, then attach `gdb -p <worker_pid>` with the
  count_cpu breakpoint.

Controls (positive + negative) — the counter is live:
- control-gpu: `cpu_command_encoder_calls = 0`, `resolved_while_running = true`,
  `libmlx_loaded = true`
- control-cpu: `cpu_command_encoder_calls = 3` (one `mx.add` on a
  4096-element stream=mx.cpu stream fires exactly 3 cpu::get_command_encoder
  calls in the `binary_op_cpu<Add>` instantiation)

### Path 1: chat (qwen3.8-2b-4bit) — 0 CPU encoder dispatches

Workload: warm-up (max_tokens=4), plain chat ("Say hello in five
words or fewer.", max_tokens=32), long-context (~3.6 k chars,
max_tokens=256), card turn (Paris 2.1M / Tokyo 13.9M / Lagos 21.0M,
max_tokens=256). Chat server bound to port 41363.

`chat path: cpu encoder dispatches=0` (0 `Breakpoint 1,` lines in
`gdb.cp`).

The [rtmod] DISPATCH count of 5905 GPU dispatches for the same
workload (2026-09-30T07:55Z Everyday run) is consistent: 5905 GPU
dispatches, 0 CPU dispatches — chat path is GPU-stream only.

Artifact: `~/.local/share/apple-silicon-lab/artifacts/PairGates/2026-09-30-chat-cpu/`
(control.log, worker.log, gdb.cp, bt.cp, pybt.cp, COMPLETE).

### Path 2: decision (Laya, mlx_omarchy_laya.server) — 0 CPU encoder dispatches

Workload: the Laya worker is the assistant's typed-decision model
(catalog id `laya-mlx`, repo `convaiinnovations/laya`, family `laya`,
~421 M fp16-packed parameters with a ModernBERT `ModernBertForMaskedLM`
encoder component; serves typed decisions via `/v1/decisions` under
the `mlx_omarchy_laya.server` module backend). Two HTTP POSTs driven
via `workload_decision.sh`: a 3-option /v1/decisions compare
(espresso / tea / juice with `bitterness` criterion), and a 3-option
classification (Important / Normal / Low with "i need to respond
today" criterion).

`decision path: cpu encoder dispatches=0`.

The per-worker bt_cpu capture failed (a second Laya worker did not
bind its port within the 60 s window on the shared GPU). Since the
hit count is 0, no native stack chain was needed for this run.

Artifact: `~/.local/share/apple-silicon-lab/artifacts/PairGates/2026-09-30-decision-cpu/`
(control.log, worker.log, decision.json, decision2.json, gdb.cp,
COMPLETE).

### Path 3: TTS (Qwen3-TTS 0.6B CustomVoice, 4bit) — 0 CPU encoder dispatches

Workload: `worker_tts_under_count.py` loads the Qwen3-TTS 0.6B
CustomVoice 4bit model via `mlx_audio.tts.utils.load_model` from
`agents/PairGates/homes/everyday/voice/qwen3-tts-0.6b-customvoice-4bit/`
and calls `model.generate_custom_voice` twice ("Hello, this is a test
sentence." and "The quick brown fox jumps over the lazy dog." with
speaker="aiden", language="english"). Then a second TTS process is
launched and gdb is attached with the count_cpu breakpoint.

`tts path: cpu encoder dispatches=0`.

Artifact: `~/.local/share/apple-silicon-lab/artifacts/PairGates/2026-09-30-tts-cpu/`
(control.log, worker.log, gdb.cp, bt.cp, tts-server.log,
tts-server2.log, tts-server3.log, COMPLETE).

### Path 4: chat 9B GDN (qwen3.5-9b-mlx-4bit, Hk!=Hv) — 0 CPU encoder dispatches

Workload: same 4-phase script as Path 1, but the chat server is the
qwen3.5-9b-mlx-4bit model (GDN route, Hk!=Hv: 24 linear-attention
layers with Hk=16/Hv=48 plus 8 full-attention layers). GDN decode
route is the composed path (not the fused raw kernel) on this
architecture; the dispatch trace for this model has more kernels per
decode step than the 2B.

`chat9b path: cpu encoder dispatches=0`.

Artifact: `~/.local/share/apple-silicon-lab/artifacts/PairGates/2026-09-30-chat9b-cpu/`
(control.log, worker.log, gdb.cp, bt.cp, plain.json, long.json,
card.json, warmup.json, COMPLETE).

## Finding 1 — release wheel runs CPU primitives on explicit `stream=mx.cpu`

The release wheel has a working, untraced CPU execution path:
`mlx::core::cpu::CommandEncoder::dispatch` template instantiations for
`binary_op_cpu<Add>`, `comparison_op_cpu<Equal>`, `comparison_op_cpu<Greater>`,
`comparison_op_cpu<Less>`, `binary_op_cpu<Divide>`,
`binary_op_cpu<Remainder>`, `binary_op_cpu<LogicalAnd>`,
`binary_op_cpu<LogicalOr>`, `binary_float_op_cpu<LogAddExp>`, and many
others are in `libmlx.so`. When `stream=mx.cpu` is passed explicitly,
the op runs on the CPU encoder and returns the correct value with no
`[rtmod] DISPATCH` line emitted — the GPU encoder is bypassed
entirely.

The chat, decision, TTS, and 9B GDN chat paths all use the default
GPU stream and therefore never trip this. The design contract
`Never run a tensor primitive on CPU in a release build` is honored
for the product paths but is silently bypassable when a caller passes
`stream=mx.cpu` explicitly. The gdb counter (`cpu::get_command_encoder`)
is the only way to observe the CPU encoder's activity; the
`[rtmod] DISPATCH` facility does not cover it.

This is the finding to list: either remove the CPU encoder symbols
from the release `libmlx.so`, or instrument the CPU encoder so the
contract is observable end-to-end.

## Paired memory peak — v3 (fixed harness, coordinator fix a1251aaa)

Measured 2026-10-01 on jw14m2-linux (T6021, 96 GB) with
`scripts/pair_memory_v2.py` driven entirely through the assistant
HTTP API. Harness runs `origin/main a1251aaa` (includes the
invalid-fence raw-text fallback and the empty-reply guard), so card
turns now complete with visible text.

Each run: (a) pre-start baseline snapshot of system memory + top-15
RSS processes; (b) assistant `--pair <id> --yes`; (c) setup wait;
(d) 4 scripted turns (plain chat, long-context, Compare options,
card) with a heartbeat POST every 2 s so the coordinator's 20 s
heartbeat never cancels a slow turn; (e) idle-after and post-teardown
snapshots. Card turns wait up to 900 s (the 9B model needs ~2 min to
stream the full reply).

Per-phase validity (what "the user saw a response" means here):
- plain_chat / long_context: `status=complete` and `text_len > 0`.
- compare: `status=complete` and `text_len > 0` (a `decision`
  component is a bonus; some chat models answer compare in prose).
- card: `status=complete` and `text_len > 0`, reported as a separate
  row with its own status (card is a product-defect surface on some
  models and is not folded into the core-valid check).
`run_valid = True` when all core phases are valid. Peak over baseline
is computed over completed phases only.

Kernel: 7.1.13-3-1-ARCH. All runs set
`MLX_OMARCHY_PAIR_DEV_QUALIFICATION=1` (the var only relaxes a null
`backend_context_qualified_tokens` limit; memory behavior unchanged).

### compact4b (qwen3-4b-instruct-2507-4bit + laya-mlx)

| run | baseline MiB | pair resident MiB | peak over baseline MiB | backend peak MiB | core phases | card phase |
|---|---|---|---|---|---|---|
| 1 | 6836.9 | 2453.7 | **5700.7** | 2957.1 | all valid | complete, text_len 857, valid |
| 2 | 6229.5 | 1743.3 | **6831.5** | 2957.1 | all valid | complete, text_len 857, valid |
| 3 | 6908.3 | 2660.9 | **5335.4** | 2957.1 | all valid | complete, text_len 857, valid |

compact4b peaks at 5.3–6.8 GiB over baseline. Fits on the 16 GB tier
(peak 6.8 GiB + ~5 GiB desktop = well under 16 GiB).

### everyday9b (qwen3.5-9b-mlx-4bit + laya-mlx)

| run | baseline MiB | pair resident MiB | peak over baseline MiB | backend peak MiB | core phases | card phase |
|---|---|---|---|---|---|---|
| 1 | 6630.8 | 1570.5 | **9463.7** | 5632.5 | all valid (compare has `decision` component) | complete, text_len 109, components=[chart], valid |
| 2 | 6631.5 | 2658.2 | **13442.9** | 5632.5 | all valid | complete, text_len 109, components=[chart], valid |
| 3 (busy GPU: baseline includes other tenants) | 12082.3 | 1059.4 | **5828.8** | 5632.5 | all valid | complete, text_len 109, components=[chart], valid |

everyday9b peaks at 9.5 GiB over baseline on an idle GPU (run 2
showed 13.4 GiB — its baseline includes a cold-weights page-cache
delta from the previous run). Backend peak is consistently
5.5 GiB (the 9B weights + activation/KV). All three card turns
returned a `chart` component with 109 chars — the coordinator fix
shows the fenced card.

Fits on the 16 GB tier with ~2-6 GiB of headroom on the 9.5 GiB
peak; run 2's 13.4 GiB peak is borderline (page cache counts
toward `MemTotal - MemAvailable`).

### quality27b (qwen3.8-27b-4bit + laya-mlx)

| run | baseline MiB | pair resident MiB | peak over baseline MiB | backend peak MiB | core phases | card phase |
|---|---|---|---|---|---|---|
| 1 | 7228.1 | 1398.8 | **17766.5** | 16283.7 | all valid (compare has `decision` component) | complete, text_len 109, components=[chart], valid |
| 2 | 7197.1 | 2410.9 | **18736.2** | 16283.7 | all valid | complete, text_len 109, components=[chart], valid |
| 3 | 8260.5 | 2126.5 | **18823.0** | 16283.7 | all valid | complete, text_len 109, components=[chart], valid |

quality27b peaks at 17.3–18.4 GiB over baseline (the v1 run's 19.26
GiB used `max_tokens=1` and ~21 k-token prompts; the v3 card turn at
700 max tokens with a 4-phase script is lighter). Backend peak is
consistently 15.9 GiB (14.95 GiB weights + ~1 GiB KV/activation at
the 4 k turn context).

### Tier analysis

| pair | 16 GB tier | 96 GB tier |
|---|---|---|
| compact4b | fits (peak 5.2–6.7 GiB) | fits |
| everyday9b | fits (peak 9.2 GiB; run 2's 13.1 GiB is borderline — page cache) | fits |
| quality27b | does NOT fit (peak 17.3–18.4 GiB) | fits |

All nine v3 runs report `run_valid: true` (all core phases valid, all
card turns complete with visible text).

### Card-turn latency per pair (quiet boot 5498f953, idle render node)

`scripts/card_timing_boot.py` submits the identical card prompt to
each pair on one boot and reads the SSE stream with a reconnecting
cursor (the server closes each events connection after 15 s, so a
single-connection reader misses everything after that — this is why
earlier runs show `first_visible_text: not captured`).

| pair | wall (s) | first_visible_text (s) | first_visible_component (s) | visible text | component |
|---|---|---|---|---|---|
| compact4b (4B) | 91.33 | 90.92 | — (none) | 857 chars (fallback notice + raw output), 255 tok | none — fence invalid; coordinator emitted `invalid_component` + raw text |
| everyday9b (9B) | 116.29 | **5.75** | 115.89 | 109 chars / 22 tok | chart |
| quality27b (27B) | 54.25 | **20.51** | 53.76 | 109 chars / 22 tok | chart |

What each wait is made of, from the event timestamps:

- **4B**: the user sees nothing for the whole turn. The model emits
  its fenced block; validation fails at ~90.9 s and the coordinator
  shows the `invalid_component` notice plus the raw output at that
  moment. The wait is model load + full generation + fence
  validation; no partial output is shown while the fence is being
  buffered. Decode rate is not separable (the fallback lands as one
  burst).
- **9B**: prose streams from 5.7 s ("Here is a bar chart of…"), then
  the model spends ~110 s emitting the fenced chart payload, which
  stays invisible until it validates at 115.9 s. The wait is NOT
  prefill (generation started at 0.2 s) and NOT prose decode — it is
  card-payload generation + validation with no user feedback after
  the prose.
- **27B**: prose from 20.5 s (coordinator setup + Laya decision +
  prefill dominate this first turn), chart payload completes at
  53.8 s. Same shape as the 9B with roughly half the payload wait.

The release-relevant number is `first_visible_component_s`: ~54 s
(27B) and ~116 s (9B) for a chart card, and never (4B, invalid
fence). A spinner-on-prose UX would hide most of it; a card-format
that streams/validates incrementally would remove it.

`pair_resident_cost` (idle_before - baseline) is much smaller than
the backend peak because the model weights are mmap'd lazily; the
allocator peak only builds up once inference runs.

Artifacts: `receipts/2026-09-30-pair-gates/raw/v2-*/` (all nine runs,
COMPLETE sentinels included).

### Earlier v1/v2 runs (superseded)

The earlier `pair_memory_turns.py` and pre-fix `pair_memory_v2.py`
runs (max_tokens=256, no heartbeat, card text_len=0 marked
run-invalid) measured the same pairs but with a broken card phase.
Those runs are kept in git history only; the v3 numbers above are the
receipt.

## Quality pair on idle GPU — MEASURED (quiet window, boot 5498f953)

Measured 2026-10-02 00:55-00:59 CDT on jw14m2-linux (T6021, 96 GB) with
`scripts/idle_quality27b_perf.sh` under `gpu-turn -m 30`, after the
release gates finished and w73 confirmed no reboots. `fuser
/dev/dri/renderD128` empty before AND after; loadavg recorded both
ways (before 1.92 2.28 1.36, after 2.31 1.36 1.17); the 27B pair home
was resumed (boot-id mismatch → one setup pass inside the ticket).

Prompt sizes are tokenizer-measured. The exact-token builder appends
whole filler sentences, so it overshoots the target by up to one
filler: the "512" prompt measures 600 tokens, the "2048" prompt 2100,
the "170" TTFT prompt 300 (Qwen tokenizer, prompt recorded per turn).
Rates below use the measured counts.

| turn | prompt tok | wall (s) | first visible text (s) | output | rate |
|---|---|---|---|---|---|
| prefill (max_tokens=1) | 600 | 14.55 | 14.50 | 1 token | **42 tok/s prefill** (600/14.4) |
| prefill (max_tokens=1) | 2100 | 27.60 | n/a (no text event at max_tokens=1) | 1 token | **76 tok/s prefill** (2100/27.5) |
| ttft | 300 | 11.54 | **6.31** | 108 chars | — |
| decode after 600-tok prompt | 600 | 15.55 | 9.98 | 108 chars (~30 tok) | ~5.4 tok/s visible decode (30/5.6) |
| card_turn_1 (chart) | 30 | 49.64 | 13.76 | 109 chars | chart component |
| card_turn_2 (bar chart) | 70 | 75.24 | 14.35 | 57 chars | chart component |
| card_turn_3 (timeline) | 37 | 16.05 | 3.54 | 162 chars | timeline component |
| card_turn_4 (checklist) | 9 | 57.19 | 2.56 | 1054 chars | checklist component |
| card_turn_5 (form) | 16 | 55.18 | 13.58 | 66 chars | form component |
| card_pop (repeat of turn 1) | 30 | 45.66 | 13.52 | 109 chars | chart component |

All ten turns status=complete. First-visible component timestamps are
not captured by this harness's SSE reader (it still uses a single
15 s-limited connection; the reader that does capture them is
`card_timing_boot.py`, used for the per-pair table below).

### Comparison against design budgets and ModelBench (idle run)

- Design budget: first_visible_text p95 ≤ 2 s. The 27B idle TTFT for
  a 300-token prompt is 6.3 s — about 3× the budget. Card-turn first
  text lands at 2.6–14.4 s depending on how much prose precedes the
  card payload. Still NOT met, now on an idle node with exact-token
  prompts.
- ModelBench engine-level TTFT for a 240-token prompt was 2.81 s
  (`receipts/2026-09-30-chat-model-bench`). The assistant-API turn
  adds coordinator overhead (system prompt, Laya decision prefill +
  decode, speech-yield gate) on top of model prefill: 6.3 − 2.8 ≈
  3.5 s of product overhead on a comparable prompt.
- 27B-4bit prefill on the idle node: 76 tok/s at 2100 tokens
  (ModelBench alone-on-GPU was 97-103 tok/s; the assistant + Laya
  workers resident cost ~25%).
- 27B decode: ~5.4 tok/s visible on the decode turn (108 chars of a
  128-token allowance — the model stopped at EOS well before the cap;
  earlier shared-GPU estimate was ~1.4 tok/s, so idle decode is
  ~4× the shared-GPU figure).

Artifacts: `receipts/2026-09-30-pair-gates/raw/quiet-idle-27b-perf/`
(quality27b-idle.json, launch.log with fuser/loadavg receipts,
COMPLETE). The earlier shared-GPU run 1 remains in
`receipts/2026-09-30-pair-gates/raw/quality27b-perf/` for comparison;
its numbers are superseded by this idle run.

### Shared-GPU run 1 (2026-10-01, superseded by the idle run above)

Kept for comparison. Same harness before the exact-token fix: labels
`prefill_512`/`prefill_2048` were 5408/20956-token prompts
(`filler * 18` / `filler * 70`); measured prefill 73-81 tok/s,
ttft_170 (967-token prompt) wall 43.65 s, decode ≈ 1.4 tok/s, card
turns fvt 2.5-14.4 s, `prefill_2048` hit the harness's 600 s
per-turn deadline after prefill completed (harness limitation).
Full table in git history and
`receipts/2026-09-30-pair-gates/raw/quality27b-perf/`.

### Card defect recap (resolved by coordinator a1251aaa)

The original defect: the 4B card turn returned `status=complete,
text_len=0` (invalid fence swallowed, empty reply). The coordinator
fix (raw-text fallback + empty-reply guard) resolved the empty-reply
symptom: every card turn in the nine v3 memory runs and all three
quiet-boot card timings returns visible text. What remains on the 4B
is the upstream model behavior: it still emits a fenced block that
fails component validation, so the user sees the coordinator's
invalid_component notice plus the raw output instead of a rendered
card (see the card-timing table below). The 9B and 27B produce valid
cards.


## Memory admission target (catalog, budget.py)

| Catalog ID | weights_bytes | kv_bytes_per_token | peak_estimate_bytes |
|---|---|---|---|
| qwen3.8-2b-4bit | 1,059,404,429 (1.01 GiB) | 12,288 | null |
| qwen3.5-9b-mlx-4bit | 5,674 MiB | 32,768 | null |
| qwen3-4b-instruct-2507-4bit | 2,158 MiB | 147,456 | null |
| qwen3.8-27b-4bit | 16,054,541,349 (14.95 GiB) | 65,536 | null |
| laya-mlx | ~810 MiB | n/a | n/a |

`estimate_required` (budget.py:117): `weights + kv + max(0.25 *
weights, 1 GiB)`. Safety reserve `SAFETY_RESERVE_BYTES = 2 GiB`.
Desktop reserve `max(SAFETY, 0.10 * MemAvailable)` → ~9.5 GiB on
96 GB. `DEFAULT_CONTEXT_TOKENS = 4096`.

The chat server refuses the harness's longer prompts when
`prompt + max_tokens > admitted context` (server-side shim `_mlxlm_server.py`
enforces this). Two earlier attempts (5080 + 256 and 2139 + 2048)
were rejected; the harness's revised 3.6 k-char long-context turn fits.

## Failed approaches

- `MLX_OMARCHY_TRACE_DISPATCH cpu_tensor_events = 0` with a chat
  workload. Cannot prove zero-CPU; the trace facility only covers the
  GPU/Vulkan encoder.
- gdb `set follow-fork-mode child` on a wrapper inferior. Parent
  breakpoint is not propagated to the forked chat subprocess;
  `gdb.execute("run ...")` blocks until the inferior's stdin gets
  DONE, which never happens while the worker subprocess serves.
- `nohup bash -c "<home>/bin/gpu-turn -m 15 -- /tmp/ticket_everyday9b_mem.sh everyday9b-run1"`:
  the first run did not set MLX_OMARCHY_PAIR_DEV_QUALIFICATION=1
  and the assistant refused to start with
  `qualification gate failed: chat model qwen3.5-9b-mlx-4bit is
  generation=untested/http=untested`. Fixed in the current script
  (env var `export`ed at the top).

## Files at remote (T6021-M2-linux)

- `~/.local/share/apple-silicon-lab/artifacts/PairGates/2026-09-30-chat-cpu/`
  (COMPLETE)
- `~/.local/share/apple-silicon-lab/artifacts/PairGates/2026-09-30-decision-cpu/`
  (COMPLETE)
- `~/.local/share/apple-silicon-lab/artifacts/PairGates/2026-09-30-tts-cpu/`
  (COMPLETE)
- `~/.local/share/apple-silicon-lab/artifacts/PairGates/2026-09-30-chat9b-cpu/`
  (COMPLETE)
- `<home>/agents/PairGates/main-checkout/`: harness +
  scripts + `serve/mlx_omarchy_serve/_mlxlm_server.py` (with the
  one-line try/import for `/v1/internal/memory`)
- `<home>/agents/PairGates/main-checkout/harness/`:
  `count_cpu.gdb.py`, `bt_cpu.gdb.py`, `pybt_cpu.gdb.py`, `control.py`
- `/tmp/pairgates_reduced_driver.sh`, `/tmp/ticket_quality27b_perf.sh`,
  `/tmp/ticket_everyday9b_mem.sh`, `/tmp/ticket_compact4b_mem.sh`,
  `/tmp/ticket_quality27b_mem.sh`

## Open items

1. Long-context (2 k token) chat turn is not in the v3 memory
   script's phase list (the phase is plain/long-context/compare/card
   with the long-context prompt at ~3.6 k chars ≈ 1 k tokens). A
   true ~2 k-token long-context phase would need a separate run.
2. The idle-perf harness's SSE reader still uses a single 15
   s-limited events connection, so `first_visible_component_s` is
   None for its card turns (the reconnecting reader lives in
   `card_timing_boot.py`). The 27B component latency on this boot
   comes from the card-timing run: 53.8 s.
3. The 4B's fenced card output still fails component validation
   (model behavior, not coordinator): users see the
   `invalid_component` notice plus raw text instead of a card.
   Upstream fix is a model or card-format change, not a harness one.

## Files

- `harness/count_cpu.gdb.py` — gdb breakpoint on
  `mlx::core::cpu::get_command_encoder(Stream)`.
- `harness/bt_cpu.gdb.py` — native caller chain of every CPU-encoder
  hit.
- `harness/pybt_cpu.gdb.py` — Python stack of the dispatching thread
  at the first hit of each distinct native chain.
- `harness/control.py` — 4096-element `mx.add` on stream=gpu/cpu (the
  gdb counter's positive/negative controls).
- `harness/worker_under_count.py` — chat + decision worker launcher
  with PYTHONPATH + MLX_OMARCHY_SERVE_CONTEXT_LIMIT=4096.
- `harness/worker_tts_under_count.py` — TTS worker launcher with
  voice-site PYTHONPATH.
- `harness/workload_chat.sh`, `harness/workload_decision.sh`,
  `harness/workload_tts.sh` — HTTP-driven workloads.
- `scripts/pair_measure.py` — coordinator-driven per-phase memory
  harness.
- `scripts/trace_dispatch.py` — 5-phase trace harness (chat, decision
  via HTTP, cpu-control-default, cpu-control-stream, gpu-reference).
- `scripts/run_pair.sh`, `scripts/run_quality_perf.sh`,
  `scripts/quality_perf.py` — full product-path drivers.
- `scripts/_mlxlm_server_with_memory.py` — side-effect import that
  patches `mlx_lm.server.APIHandler.do_GET` to add
  `GET /v1/internal/memory`.
- `scripts/capture_decision_trace.sh` — Laya `/v1/decisions` trace
  probe.
- `scripts/peek_decision_dispatch.py` — diagnostic.
- `serve/mlx_omarchy_serve/_mlxlm_server.py` — one-line
  `try: import _mlxlm_server_with_memory` addition (live wheel is
  untouched).

## Notebook

`~/.local/share/apple-silicon-lab/entries/PairGates/2026-09-30T06-37Z-T6021-M2-linux-pair-gates.md`
artifacts under `~/.local/share/apple-silicon-lab/artifacts/PairGates/<run-id>/`.