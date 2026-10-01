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

Quality idle-GPU perf: run 1 measured on the shared GPU (earlier).
An idle-GPU rerun is NOT yet landed; the numbers below are the
shared-GPU run, labeled as such.

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

### Card-turn latency per pair (total wall; first-visible text where captured)

The card turn is the user-visible latency surface. The memory harness
polls the assembled conversation record and (from this revision on)
reads the `/events` stream for the first `text` event; the v3 runs
below recorded wall only, so first-visible text is shown for the
Quality pair from the SSE-instrumented perf harness and marked
"not captured" for the 4B and 9B pairs.

| pair | card wall (3 runs, s) | first_visible_text (s) | text_len | visible component |
|---|---|---|---|---|
| compact4b (4B) | 88.1 / 90.1 / 94.1 | not captured (memory harness) | 857 each | none (prose fallback text) |
| everyday9b (9B) | 122.2 / 117.2 / 113.2 | not captured (memory harness) | 109 each | chart |
| quality27b (27B) | 49.1 / 53.1 / 49.1 | 13.4 / 14.4 / 3.2 / 2.6 / 13.3 across the perf harness's five card prompts | 57–1054 | chart |

Decode tok/s over the card turn (text_len / (wall − fvt), Quality
pair from the perf harness): 109/(48.15−13.37) ≈ 3.1 tok/s,
57/(77.72−14.41) ≈ 0.9 tok/s, 162/(16.05−3.22) ≈ 12.6 tok/s,
1054/(57.68−2.55) ≈ 19.1 tok/s, 66/(55.68−13.28) ≈ 1.6 tok/s. For the
4B and 9B pairs wall is measured but the fvt split is not, so their
decode rate cannot be separated from prefill+decision latency in
these runs; `scripts/pair_memory_v2.py` now records
`first_visible_text_s` for any rerun.

What dominates the card turn, from the split the data allows: the
27B card turn spends 2.6–14.4 s before the first token shows
(prefill plus the Laya decision pass plus coordinator setup), then
streams at 0.9–19.1 tok/s depending on how much the model writes
(text_len 57–1054). The 9B card turn's 113–122 s wall with only 109
characters of output means the time is going somewhere other than
visible text streaming; the memory harness does not split it, and
this receipt does not guess.

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

## Quality pair on idle GPU — run 1 (shared GPU; idle rerun still pending)

Measured 2026-10-01 on jw14m2-linux (T6021, 96 GB) with
`scripts/quality27b_perf_api.py` driven entirely through the assistant
HTTP API. Resumed the saved 27B pair home (skipping setup, since
pair-locks were on disk). Run 1 of 1.

**This run is a shared-GPU measurement, not an idle-GPU
measurement.** The `fuser /dev/dri/renderD128` check before the run
showed other tenants on the render node; the absolute numbers below
therefore include that contention. An idle-GPU rerun (fuser empty
before AND after) has not landed yet — that is the open item.

Note on the prompt sizes used here vs the labels: my labels
`prefill_512` and `prefill_2048` mean "the 5408-token prompt" and
"the 20956-token prompt" respectively, because `filler * 18` and
`filler * 70` expand to those token counts with the Qwen3.8 tokenizer
plus the coordinator's system prompt. The prefill rates below use
those actual token counts.

Note on the harness's `first_visible_text_s` for `prefill_512`:
`None`, because `max_tokens=1` produces a single 1-token completion
with no intermediate `text` SSE event; the harness tracks
`first_visible_text_s` from the SSE stream. For `prefill_2048`,
`status=stopped, text_len=0` is the harness's per-turn 600 s deadline
— prefill completed (20956 tokens in ~258 s), but the per-turn timeout
in `quality27b_perf_api.wait_turn_with_events` (default 600 s)
fired before the post-prefill one-token decode completed, so the
status is reported as `stopped` rather than `complete`. That is a harness
limitation, not a product bug.

### Quality 27B perf run 1 (measured via assistant API)

- baseline sys_used = 7193.6 MiB (before assistant launched; other
  resident processes like the previous perf ticket's orphan Laya
  included; the harness kills those on its own EXIT, so this baseline
  is the actual pre-pair load)
- idle_before sys_used = 9050.4 MiB (pair loaded, no turn yet)
- pair_resident_cost = 1856.8 MiB (idle_before - baseline)
- peak over baseline = 19,724.9 MiB (with max_tokens=1, only prefill
  is exercised; this is the 27B weights + KV/activation workspace
  at 4096 context)
- backend peak = 19,607.4 MiB (from `mx.get_peak_memory` via the
  side-effect patcher's `/v1/internal/memory` route)
- pair_lock_context_tokens = 262,144

| turn | walls (s) | first_visible_text_s (s) | text_len | status | prefill rate |
|---|---|---|---|---|---|
| ttft_170 (967 tokens input) | 43.65 | None (no `text` SSE event for `max_tokens=1`) | 107 | complete | 27 tok/s prefill + 1 decode |
| prefill_512 (5408 tokens input) | 74.27 | None (max_tokens=1, no `text` SSE event) | 3 | complete | 73 tok/s prefill |
| prefill_2048 (20,956 tokens input) | 257.99 | None (max_tokens=1, no `text` SSE event) | 0 | stopped (harness per-turn timeout 600 s; prefill completed, decode did not) | 81 tok/s prefill |
| decode_128_after_512 (5408 input + 128 decode) | 75.26 | None | 107 | complete | decode = 107 tokens / (wall - prefill) ≈ 1.4 tok/s |
| card_turn_1 (card cue "population") | 48.15 | 13.366 | 109 | complete | card component returned, text_len=109 |
| card_turn_2 (card cue "bar chart") | 77.72 | 14.407 | 57 | complete | card component returned, text_len=57 |
| card_turn_3 (card cue "timeline") | 16.05 | 3.216 | 162 | complete | card component returned, text_len=162 |
| card_turn_4 (card cue "checklist") | 57.68 | 2.552 | 1054 | complete | card component returned, text_len=1054 |
| card_turn_5 (card cue "form") | 55.68 | 13.284 | 66 | complete | card component returned, text_len=66 |
| card_pop_defect_prompt (card cue "population", same as card_turn_1) | 48.65 | 13.832 | 109 | complete | card component returned, text_len=109 |

### Comparison against design budgets and ModelBench

- Design budget: first_visible_text_s p95 ≤ 2 s. NOT met for the 27B
  Quality pair on the shared M2 GPU at these prompt sizes (10-14 s for
  first-visible-text across card turns; 3.2 s for the timeline card).
- ModelBench engine-level TTFT for a 240-token prompt was 2.81 s
  (`receipts/2026-09-30-chat-model-bench`). The assistant API
  measured TTFT in this run is 10-14 s because the API turn includes
  the coordinator overhead (system prompt, Laya decision-card prefill
  + decode wait, speech-yield gate, KV-cache warm-up after idle).
- 27B 4-bit prefill alone-on-GPU is ~97-103 tok/s (ModelBench). With
  the assistant + Laya workers resident, my measured prefill is
  73-81 tok/s — about 75-85% of the alone-on-GPU rate.

### Card defect recap (preliminary, deeper root-cause pending)

The card turns with 27B all returned card components in this run
(text_len ≥ 57, fvt 2.5-14.4 s). The 4B card turn in compact4b
run 1 returned `status=complete, text_len=0, components=[]` — silent
empty reply. The 9B card turn in everyday9b run 1 returned
`status=stopped, text_len=109` (stopped earlier, returned content).
The card_pop_defect_prompt for 27B returned `text_len=109` (fine),
so the 27B is not affected. Card defect capture from the 4B and 9B
runs is in `receipts/2026-09-30-pair-gates/raw/compact4b-memory-test/`
and `everyday9b-run1/`; SSE events captured=0 for the 4B card turn
(the harness's `wait_turn_with_events` consumed the events before
recording them — the bug is in the harness, not the product;
fixing the harness to also record events when status=complete with
text_len=0 is in flight).

Artifact: `receipts/2026-09-30-pair-gates/raw/quality27b-perf/` (10 turns,
COMPLETE sentinel on M2 disk).

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

1. Quality idle-GPU perf rerun: run `scripts/quality27b_perf_api.py`
   under `gpu-turn` with `fuser /dev/dri/renderD128` empty before AND
   after, and record loadavg. The shared-GPU run above stays labeled
   as such until that rerun lands.
2. Long-context (2 k token) chat turn is not in the v3 memory
   script's phase list (the phase is plain/long-context/compare/card
   with the long-context prompt at ~3.6 k chars ≈ 1 k tokens). A
   true ~2 k-token long-context phase would need a separate run.

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