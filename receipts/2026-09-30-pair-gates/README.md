# Pair gates — 2026-09-30 / 2026-10-01 (interim: zero-CPU audit complete; memory/perf pending)

**Question.** How much memory does the paired Everyday and Quality
assistant path take on the M2 (96 GB tier)? Does any tensor primitive
run on CPU for the chat or decision path? What does the Quality pair
do for performance on an idle GPU?

**Status (interim).** Zero-CPU dispatch trace audit is COMPLETE for
four paths (chat on 2B, decision on Laya, TTS, and chat on 9B GDN
Hk!=Hv). All four measured 0 CPU tensor primitive dispatches via gdb
breakpoint on `mlx::core::cpu::get_command_encoder(Stream)`. The
control (mx.add on `stream=mx.cpu`, 4096-element) fires exactly 3
CPU encoder calls; the same op on `stream=mx.gpu` fires 0.

Paired memory peak and Quality idle-GPU perf are NOT MEASURED yet.
Ticket scripts are ready on the M2 host and queued for the shared
GPU; this receipt will be updated as each lands.

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

## Paired memory peak — NOT MEASURED yet

Ticket scripts are ready on the M2 host and queued for the shared
GPU. Every script:
- exports `MLX_OMARCHY_PAIR_DEV_QUALIFICATION=1` so untested pairs can
  start (memory behavior is unchanged: the var only flips
  `backend_override=True` in `admissible_context`, which relaxes a
  null limit; it does NOT bypass `estimate_required`, the safety
  reserve, the desktop reserve, or atomic admission);
- proves a real chat completion (`choices` in warmup.json) before any
  memory sample;
- records `uname -r`;
- samples system-level `MemTotal - MemAvailable` delta from
  `/proc/meminfo` (the honest whole-system peak on unified memory);
- samples per-process RSS/PSS of the assistant PID plus its
  descendants via `/proc/<pid>/task/<pid>/children` BFS plus
  `/proc/<pid>/smaps_rollup`;
- samples DRM fdinfo for each renderD128 fd (best-effort: Mesa does
  not emit `drm-*` keys on this build, reported as empty);
- samples backend allocator peak via `GET /v1/internal/memory` on the
  chat worker (added by the side-effect import of
  `scripts/_mlxlm_server_with_memory.py` in
  `serve/mlx_omarchy_serve/_mlxlm_server.py`, which patches
  `mlx_lm.server.APIHandler.do_GET` to return
  `{active, peak, cache}` from `mx.get_active_memory()`,
  `mx.get_peak_memory()`, `mx.get_cache_memory()`);
- fences on a real completion before sampling memory.

Runs planned: everyday9b run1..3 (qwen3.5-9b-mlx-4bit + laya-mlx),
compact4b run1..3 (qwen3-4b-instruct-2507-4bit + laya-mlx),
quality27b run1..3 (qwen3.8-27b-4bit + laya-mlx). Every phase: idle,
plain chat, ~2 k-token long-context, explicit Compare options
decision, card turn, idle-after.

## Quality pair on idle GPU — NOT MEASURED yet

Ticket: `/tmp/ticket_quality27b_perf.sh` (-m 30 for the 27B load).
Measures prefill tok/s at 512 and 2048 prompt lengths, TTFT for a
170-token prompt, decode tok/s (128 tokens after a 512-token prompt,
median of 5), first-visible-text latency through the assistant for a
chat turn, and 5 card turns. Also records per-phase dispatch line
counts and the GDN decode route (fused raw kernel vs composed fallback)
via the kernel histogram.

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

## Next actions when the shared GPU clears

1. Let the queued tickets run: quality27b-perf (30 min), then
   everyday9b-run1, compact4b-run1, quality27b-run1 (15 min each).
2. After each, mark COMPLETE and rsync the artifact dir into
   `receipts/2026-09-30-pair-gates/raw/`.
3. Fill in the `Paired memory peak` and `Quality pair on idle GPU`
   sections with measured numbers.
4. Push again with `git fetch origin main && git rebase origin/main`
   then push (no force).

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