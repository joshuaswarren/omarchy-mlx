# Laya typed-decisions serving — contract and limitations

Status line for docs (2026-09-20, this commit): **implemented and CPU-reference-tested against
the pinned upstream implementation; GPU hardware qualification is prepared but NOT yet run.**
Every claim below traces to a file, a test, or a command in this tree.

## What Laya is

`convaiinnovations/laya` (Apache-2.0) is a **non-autoregressive typed decision model**:
a ModernBERT-large encoder (421M params, 28 layers, hidden 1024, alternating full/sliding
attention with per-type RoPE, 8192 modeled positions) plus a from-scratch decision head
that scores one `[MASK]` marker per answer option, plus a 2-way escalate/answer action
head. One forward pass maps a `state` plus typed `questions` to calibrated typed answers.
It never generates text; `output_tokens` is always 0.

Checkpoints (upstream repo root + subfolders):

| catalog id | variant | context (max_len) | source dir |
|---|---|---|---|
| `laya` | root (English) | 512 | repo root |
| `laya-typed-decisions` | typed-decisions | 1024 | `typed-decisions/` |
| `laya-multilingual` | multilingual | — | **unsupported here** (mmBERT backbone, different architecture; the converter refuses it by `model_type` check) |

Pinned provenance used by this lane:

- source: `convaiinnovations/laya` @ `55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851`
  (upstream `NandhaKishorM/laya` now 401s; it moved to the convaiinnovations org)
- pack: `aac6fef/laya-mlx` @ `20aed815fc6acde75733882e7ec0e3f28aeb9717` (fp16, 206 tensors;
  pins source `c5d78730f3493e4fe16d61507ef4b78eef7318cf`)
- root weights sha256 `891102d372688fc2a094dac56a384bc537b87c63f21f9f3dac0be2b7cbc8d86c`
- pack weights sha256 `b9c07bf14be2fa5c78a9193a3e6d840ac80e89e62fc40f425834c3d8a6eaa3de`

## Safety properties

- **No `trust_remote_code`, no pickle.** Loading uses safetensors + JSON configs +
  `tokenizer.json` only. The decision head math is reimplemented in this package
  (`serve/mlx_omarchy_laya/model.py`), ported line-by-line from the pinned upstream
  `rl_common.py`/`rl_agent_api.py` and the transformers ModernBERT modeling code.
- **No silent CPU fallback.** The engine refuses to start unless `mx.default_device()`
  is the accelerator (`mx.gpu` = Honeykrisp Vulkan on Omarchy) unless `--allow-cpu` is
  passed explicitly for reference/testing. Backend evaluation errors surface as HTTP 500
  with the exact backend message; nothing catches-and-falls-back.
- **No automatic downloads.** The only network step is the converter, approve-first
  (`--yes`), pinned to an explicit revision, hash-verified into `manifest.json`.

## Files

| file | role |
|---|---|
| `serve/mlx_omarchy_laya/model.py` | ModernBERT + decision head in pure `mlx.core` ops |
| `serve/mlx_omarchy_laya/sequence.py` | upstream question rendering / sequence building / collate / calibration helpers |
| `serve/mlx_omarchy_laya/api.py` | `LayaEngine.system_one` — upstream envelope with exact calibration semantics |
| `serve/mlx_omarchy_laya/server.py` | stdlib HTTP server, `serve_main(argv)` for `mlx_omarchy_serve` |
| `serve/mlx_omarchy_laya/convert.py` | approve-first pinned-revision packaging (+ validation of key set/shapes) |
| `serve/mlx_omarchy_laya/qualify.py` | hardware qualification runner against the committed CPU fixture |
| `tests/test_laya_unit.py` | always-on structural tests (no download) |
| `tests/test_laya_reference.py` | env-gated upstream-torch comparison battery |
| `tests/fixtures/laya/laya_reference.json` | committed expected envelopes (upstream fp32 + port fp32) |

## Serve contract (mlx_omarchy_serve integration)

- catalog entries: `laya` and `laya-typed-decisions`, kind `decisions`
- launcher flow: `mlx_omarchy_serve` admits memory, then calls
  `mlx_omarchy_laya.server.serve_main(["--model", <ckpt>, "--host", H, "--port", P])`
  as a foreground process; chat keeps running as its own process/port.
  Laya + chat coexist as **independent model ids, ports, processes**.
- memory: co-serving admission is **dtype-aware and atomic**. At server startup -
  before any weight loads - the reserve total is computed from the run dtype:
  `parameter_bytes(dtype) + workspace(dtype, max_questions, context)` where
  parameter_bytes scales with dtype size (fp32 run = 2x fp16 parameter bytes) and
  the workspace is a conservative analytic bound (per-request batch =
  `--max-questions` 64, context = max_len, hidden 1024, 16 heads, incl. the fp32
  softmax upcast: scores B*H*T^2*(sz+4) + residual 4*B*T*D*sz + QKV/FFN
  B*T*(3D+2I+4D)*sz + fp32 logits) - not a flat constant. The total is admitted
  through the shared atomic `budget.admit_and_reserve` (serve-cli
  feat/serve-cli-catalog; fit check + registry write under one lock, per-process
  owner tokens, duplicate same-id refusal). After weights materialize the entry
  relabels `state="resident"` with `resident_floor_bytes` = the run dtype's exact
  **parameter nbytes only** (never a global allocator counter). **The admitted
  total is never grown post hoc**: a floor exceeding the admitted total is a
  mismatch and refuses under --managed (standalone degrades to full-bytes
  conservative counting). Reservations are owner-scoped: shutdown and
  failed-startup clears pass the process owner token and refuse another owner's
  live entry. Under `--managed` anything less than the atomic API is fail-closed
  with **exit 3**. Standalone runs degrade to warn-and-continue. Proven
  multiprocess (isolated registry, three OS processes): resident laya + an
  independent 9 GiB chat admission fits counting only the unmaterialized
  headroom; a 60 GiB overcommit refuses; a duplicate laya id from a second owner
  refuses; SIGTERM leaves the registry empty; managed without the atomic package
  exits exactly 3. Failing-first admission-math tests pin the formula
  (tests/test_laya_admission_math.py).

### Endpoints

- `GET /health` → `{"status":"ok","model":<catalog id>,"backend":"mlx-omarchy",
  "device":<mx device>,"dtype":<run dtype>,"checkpoint":<path>,
  "source_revision":<rev>,"reservation":<bool>}`
- `GET /v1/models` → OpenAI-style list with one entry, `kind: "decisions"`
- `POST /v1/decisions`

Request:

```json
{
  "state": {"from": "user@example.com", "subject": "...", "body": "..."},
  "questions": {
    "department": {"type": "choice", "instructions": "Which department?",
                   "criteria": {"billing": "invoices, refunds", "technical": "bugs", "other": null}},
    "urgency":    {"type": "score",  "instructions": "How urgent?",
                   "criteria": ["not urgent", "low", "medium", "high", "critical"]},
    "is_refund":  {"type": "noul",   "instructions": "Is this a refund request?",
                   "criteria": null}
  }
}
```

`state` may be a string or any JSON object (objects are serialized upstream-style with
`json.dumps(..., ensure_ascii=False)`). `instructions` may be a string or a JSON object.
`criteria`: object (choice; insertion order fixes label order), array of strings (score
levels), or object/`null` for noul. Any number of questions in one call — they share one
batched forward pass.

Response (upstream `system_one` envelope, plus the cross-server `timings` block):

```json
{
  "model": "rl-agent",
  "answers": {
    "department": {"type": "choice", "choice": "billing",
                   "probabilities": {"billing": 0.9653, "technical": 0.014, "sales": 0.0101, "other": 0.0107},
                   "confidence": 0.864, "rl_agent": {"act_probability": 1.0}},
    "urgency": {"type": "score", "score": 1.5036,
                "legend": {"0": "not urgent", "1": "low", "2": "medium", "3": "high", "4": "critical"},
                "probabilities": {"0": 0.2104, "1": 0.348, "2": 0.271, "3": 0.069, "4": 0.1016},
                "confidence": 0.0891, "rl_agent": {"act_probability": 1.0}},
    "is_refund": {"type": "noul", "noul": 0.8936, "rl_agent": {"act_probability": 1.0}}
  },
  "usage": {"input_tokens": 267, "output_tokens": 0},
  "timings": {"prompt_n": 267, "cached_n": 0, "prompt_ms": 0.9, "prompt_per_second": 295514.0,
              "predicted_n": 0, "predicted_ms": 12.3, "predicted_per_second": 0.0}
}
```

(The numbers above are the CPU-reference values from the committed fixture's email case.)

Semantic notes docs must keep:

- `"model"` is hard-coded `"rl-agent"` by upstream contract, regardless of checkpoint
  variant; the catalog identity lives in `/health`, `/v1/models`, and `manifest.json`.
- `rl_agent.act_probability` = softmax of the escalate/answer action head — the explicit
  **abstention signal**. `1.0` means "answer, do not escalate" in saturated cases.
- calibration: per-option logits are divided by a temperature fitted per
  `(question type, option-count bucket)` (`temperature_by_options`, falling back to
  per-type `temperature`) — preserved byte-for-byte from `rl_agent_config.json`.
- `confidence` = `1 − normalized entropy` of the answer distribution (Jev convention).
- `score` answers carry the expected value over ordinal levels, not an argmax.
- `timings` follows the mlx-serve cross-server contract (`prompt_n, cached_n,
  prompt_ms, prompt_per_second, predicted_n, predicted_ms, predicted_per_second`);
  `cached_n`/`predicted_n` are honestly 0 — there is no KV cache and no decode loop.
- Errors: 400 invalid JSON/schema/options-don't-fit (upstream `ValueError` text);
  404 unknown path; 500 exact backend evaluation error. There is no chat-completions
  route and none is planned.

## How to obtain a checkpoint

```bash
# path A (preferred: catalog pack; adapter normalizes its three renamed key
#          groups at load — see limitations — so the pack dir serves as-is)
#   fetch aac6fef/laya-mlx @ 20aed815fc6acde75733882e7ec0e3f28aeb9717 and verify
#   against the pack manifest; the directory layout is the serving layout.

# path B: package from pinned source (approve-first)
python -m mlx_omarchy_laya.convert --out ~/.local/share/mlx-omarchy/models/laya --yes
python -m mlx_omarchy_laya.convert --variant typed-decisions \
    --out ~/.local/share/mlx-omarchy/models/laya-typed-decisions --yes

# offline packaging from an already-fetched snapshot:
python -m mlx_omarchy_laya.convert --from-local <snapshot-dir> --out ...
```

## Evidence produced by this lane (all on the x86 dev box, CPU, mlx wheel
`0.32.3.dev202609201931+10b2f242` built from this repo's `mlx.lock` pin)

- The full upstream battery (6 cases: JSON state, string state, empty state, CJK +
  literal `[MASK]` handling, >max_len truncation, dict instructions, 12-option choice,
  multi-question padding) agrees with the pinned upstream torch implementation:
  identical token counts, identical choice argmax and label order, identical 4-decimal
  probabilities/scores/confidence within op-order noise (≤ 2e-4 gate), act_probability
  within 2e-2. See `tests/test_laya_reference.py` and the fixture.
- The catalog pack `aac6fef/laya-mlx` loads through the same engine — the loader
  normalizes its three renamed key groups (`in_proj.weight`→`in_proj_weight`,
  `scorer.layers.N`→`scorer.N`, `act_head.layers.N`→`act_head.N`) and passes
  the same comparison (`PackLoadTests`), so both source revisions (c5d78730 pack pin,
  1c5edc17 live HEAD) are covered. `mlx_omarchy_laya.convert` re-emits canonical
  source-style names and records `key_aliases_applied` in `manifest.json`.
- The committed fixture's email case: choice=billing .9653, score=1.5036, noul=.8936 —
  port and upstream byte-identical at 4dp.

## Hardware qualification — prepared, NOT yet run

**Frozen criteria (2026-09-20, before the window — failures remain failures; gates
are not relaxed after the run):**

1. Managed HTTP: the server must start under `--managed` on the GPU host with the
   shared atomic `admit_and_reserve` (dtype-aware fp16 total), reach
   `/health` `reservation:"resident"`, and answer a real `POST /v1/decisions` with
   HTTP 200 inside the numerical gates below. Startup refusals (no-fit,
   duplicate-id, missing atomic API) are recorded, never worked around.
2. Numerical gates vs the committed CPU fixture: choice argmax agreement 100%;
   probability delta ≤ 2e-2; `act_probability` delta ≤ 5e-2; `usage.input_tokens`
   identity exact.
3. Co-residency: with a chat model resident, the laya `admit_and_reserve` verdict
   and the budget table are recorded verbatim (fit or refusal — both are valid
   outcomes, only unrecorded ones are failures); a SIGTERM of laya must leave the
   registry without laya's entry.
4. Performance block: cold and warm `timings.predicted_ms` + wall latency per
   fixture case, plus the memory snapshot; no pass/fail gate on first run, but
   the numbers are receipt-mandatory.

Owner coordination required (GPU hosts are shared). Exact procedure on t6001-test-host
(host alias `t6001-test-hostmbp1-linux`) or m1-test-host, with the mlx-omarchy wheel active, SPA
queue order honored, and the exclusive lock protocol: take `/tmp/m1-gpu.lock`
only after the prior window's full release (their service stopped, lock freed),
run bounded, then unlock and restore the prior service state with health +
real completion verified:

```bash
# 1. provenance (repo rule: every measurement carries it)
python scripts/mlx_provenance.py

# 1b. on-host workspace audit: measured activation high-water vs the analytic
#     workspace bound at the frozen batch (validates the admitted workspace
#     term on real fp16 GPU hardware; python only, no server):
python tests/fixtures/laya/workspace_audit.py --model <ckpt> --max-questions 16

# 2. convert or sync the checkpoint on the host (see "How to obtain a checkpoint")

# 3. serve (foreground; keep on loopback)
python -m mlx_omarchy_laya.server --model ~/.local/share/mlx-omarchy/models/laya \
    --host 127.0.0.1 --port 8081

# 4. qualify against the committed CPU fixture (same repo, wheel venv):
python -m mlx_omarchy_laya.qualify --model ~/.local/share/mlx-omarchy/models/laya
#    or against the running server:
python -m mlx_omarchy_laya.qualify --url http://127.0.0.1:8081 \
    --model ~/.local/share/mlx-omarchy/models/laya

# 5. concurrent chat+laya admission check:
#    chat model served via mlx_omarchy_serve on 8080, laya on 8081, then
#    mlx_omarchy_serve's budget table must show the laya reservation subtracted
#    from MemAvailable before the chat admission verdict.
```

Gates (recorded either way; tighten after first empirical run): argmax agreement 100%,
probability delta ≤ 2e-2, act_probability delta ≤ 5e-2, exact token identity, plus
cold/warm `timings.predicted_ms` + wall latency from `qualify.py`. A receipt then needs
the AGENTS.md hardware fields (commit, kernel/Mesa, Vulkan device, driver identity,
model hashes, exact command, dispatch trace).

## Known limitations (docs: please carry these)

0. **Measured byte budget (x86 dev box, CPU mlx build `0.32.3.dev202609201931+10b2f242`;
   measured via `mx.get_active_memory`/`get_peak_memory`).** fp16: 842,587,660 B resident
   after load; activation peak growth 1,287,019,628 B measured on a 16-question/512-token
   forward. That measurement is why the admitted workspace uses the analytic bound
   (scores B·H·T²·(sz+4) + residual + QKV/FFN + fp32 logits) with a **2x safety
   factor**: the raw term sum (673,193,984 B) under-counted the scheduler's live
   transients by 1.91x. fp16 64x512 serving therefore admits ≈ **5.80 GiB** total
   (0.84 GiB params + 4.96 GiB workspace); the GPU qualification re-measures with
   `workspace_audit.py` on-device and any measured-exceeds-bound result is a window
   failure, not a relaxed gate. Reservation labels: `pending` before load,
   `resident` after; the floor is parameter nbytes only.

1. **GPU hardware qualification has not run yet.** Everything above is CPU-reference
   evidence; do not mark Laya GPU-served as qualified until the window in the previous
   section produces a receipt.
2. **fp16 compute.** Weights are fp16 (as upstream ships); encoder+head run fp16 with
   fp32 softmax/logits/calibration. Upstream's CUDA path uses bf16 autocast — a
   same-class approximation, not bit-identical to upstream CUDA.
3. **No `laya-multilingual`.** mmBERT backbone is a different architecture; the
   converter refuses it explicitly. Same adapter would need a separate port.
4. **Batching is per-request.** All questions of one request share one forward pass;
   concurrent requests are handled by HTTP threads but each builds its own batch
   (sequential on the GPU stream). Throughput under concurrent load is unqualified.
5. **max_position 8192 encoder, but request contexts are capped by `max_len`** (512
   root / 1024 typed-decisions) exactly as upstream `system_one` caps them; longer
   states are truncated (front-kept) by the upstream rule, not an error.
6. **The `temperature` tensor inside the safetensors is a stale training artifact**;
   calibration uses `rl_agent_config.json` values exactly as upstream does. The
   converter keeps the tensor for byte-fidelity but nothing reads it.
7. **`--allow-cpu` exists for tests/diagnostics only.** A CPU-served Laya is NOT a
   supported serving mode (and is slow: single-digit-second forwards on the dev box).
8. **ANE future path (assessment, non-blocking):** the ModernBERT body is a candidate
   ANE region (large stable matmuls, fixed shapes after padding buckets), with the
   decision head and act head left on the GPU; per-layer-type RoPE and ±64 sliding
   window masks are the two features the ANE exporter must support first, and both
   wait on the standing MIL/`libane` gates in the roadmap. Nothing in the GPU serving
   path waits on this.

## Verification environments used by the tests

- `LAYA_UPSTREAM_DIR` — pristine pinned upstream files (`rl_common.py`, `rl_agent_api.py`)
- `LAYA_REF_CKPT_DIR` — pristine HF snapshot (root variant)
- `LAYA_MLX_CKPT` — converted checkpoint
- `LAYA_PACK_DIR` — optional pack snapshot for `PackLoadTests`
