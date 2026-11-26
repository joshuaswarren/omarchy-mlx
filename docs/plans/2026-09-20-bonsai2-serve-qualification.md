# Bonsai-2-27B HTTP serving qualification plan (window QUEUED)

2026-09-20 · lane Bonsai2RuntimeEnablement · status: **QUEUED — module
code-complete on `agent/bonsai2-serving` (`3f475841`), CPU gates green;
no device window taken.** Main's conditions (2026-09-20): window opens
only after Main reviews the commit and the serve owner releases the
host, coordinated into one bounded combined session with the Laya/LN
queue; ceiling 90 minutes; **stop on first fault — no repeat loop.**

## What is being qualified

Real text HTTP serving of
`prism-ml/Ternary-Bonsai-2-27B-mlx-2bit` @
`3f926b415992eaa2ae9dd7b573706494d6bbf787` through the new
`serve/mlx_omarchy_bonsai2` module (stdlib server, frozen serve
contract), instead of the historical ad-hoc scripts. The pack:

- single `model.safetensors`, 8,595,477,990 bytes (disk = download size);
- `model_type` `prism_hadamard_qwen35` (schema 2), base `qwen3_5`,
  quantization affine 2-bit / group-128, 402 Hadamard-folded packed
  modules (block 1024), explicit sign vectors, `gdn_activation_layout`
  grouped;
- ships `runtime/` Python (remote code). **It is never imported or
  executed.** The serving module ports the packed-linearity semantics
  (Hadamard fold + `mx.quantized_matmul` / row-gathered
  `mx.dequantize`) into auditable in-repo code; the CPU suite proves
  bit-exact reload against a reference model.
- License: Apache-2.0, NOTICE requires attribution "Created using Bonsai
  by Prism ML" (built from Qwen3.8-27B, Apache-2.0). The server reports
  LICENSE/NOTICE presence in `/health` and prints the attribution at
  startup. No weights or pack files are redistributed by this repo.

## Prior evidence this plan stands on

- `receipts/2026-09-18-f7-gdn-correctness.md`: coherent greedy 96-step
  generation post-F7-fix (`da43969e`, in main), 1.87 tok/s (T6021) /
  1.45 tok/s (T6001), reference trajectory top5
  `760/6511/314/9338/369…` from step 0.
- `receipts/2026-09-18-v070-pretag-recert-t6001-test-host.md`: coherent at
  **1.44 tok/s** on the v0.7.0 candidate wheel (b283a16, T6001); that
  receipt also records the pack-internal schema-1/schema-2 entry skew.
- `receipts/2026-09-18-ablit-bonsai2-t6001-test-host-recert-wheel.md`: loader
  contract verified against the canonical pack; base revision matched.

## Attempt ledger (t6001-test-host, honest record)

| attempt | outcome |
| --- | --- |
| 1 | FAULT (my staging): mlx_omarchy_serve budget package not deployed; --managed correctly failed closed exit 3. Gates 1-2 green (identity; parity A-E). |
| 2 | FAULT (same staging family, caught by budget presence): preflight NameError — my refactor dropped the pack_footprint import; dev tests never reached that line. Fixed + full-path preflight test added. Gates 1-2 green again; parity numbers identical to attempt 1 run (deterministic). |
| 3 | HANG n=1: all gates through managed launch + health OK; first HTTP chat forward hung (server blocked 0.2% CPU 35 min). Structural suspect (unisolated): omarchy command encoders are thread_local and the server ran mx ops on per-request threads — every successful forward (CLI, probe) ran on the main thread. Concurrent 19G Distill transfer started the same minute — alternative hypothesis, untested. Killed; lock freed; service restored (health 200 + generation verified). |
| 4 (executed, stopped at Main's 10-min bound) | NO HANG — but the HTTP forward was not reached: my gate-3 fault-injection blocker (40 GiB) FIT the post-stop MemAvailable, so the "refusal" test admitted and its foreground serve became a healthy live server (health 200 JSON: device=gpu, 402 packed modules; atomic admission lifecycle PASSED: per-process owner token, pending→resident, resident_floor_bytes = exact weights 7,673,714,688). Runbook design fault (blocker not sized beyond availability; unbounded poll curl masked progress). Stopped per bound; host restored; slot released to SPA. Fixes landed: 200 GiB deterministic blocker, bounded curls, server.pid + deadline-ABRT wrapper with automatic orphan kill/reservation sweep/service restore. |
| 5 (queued) | HTTP forward validation on GPU with the hardened runbook (~10 min, transfer-free): managed lifecycle, first chat forward, stream/error/boundary gates. Slot: after SPA per queue, by owner arrangement. |

## Attempt ledger notes for reviewers

- The GPU parity numbers are REPRODUCIBLE: attempts 1/2/3 produced
  identical deltas per input set (A 1.38e-4, B 1.08e-4, C 1.10e-4,
  D 1.71e-4, E 1.16e-4) with identical greedy IDs — recorded as
  measured; the diagnostic band remains the post-measurement 1e-3/1e-2
  envelope described above, not a predeclared gate.
- The hang occurred exactly once, on the request-thread topology, only
  after the server was healthy — the transfer hypothesis and the
  thread-affinity hypothesis are both open; attempt 4 tests the
  thread-affinity fix transfer-free.

## Requested window

- Host: t6001-test-host (T6001). Backup: t6021-test-host (T6021, GPU-qualified). No ANE use.
- Duration: 90-minute ceiling, inside the combined serve-owner session
  Main is coordinating (Laya/LN queued alongside). Stop on first fault.
- Discipline: `sudo systemctl stop llm-inference.service` →
  `flock -w 900 /tmp/m1-gpu.lock` → work → release → restart service →
  `is-active` + `/health` 200 confirmed. State recorded before/after.

## Weights: already on the host, verified against upstream metadata

Main authorized a pinned download only after a free-space check and
ServePerformanceAudit coordination; no bulk fetch was needed — t6001-test-host
already cached the pack, and on 2026-09-20 its content hash was verified
against upstream AUTHORITATIVE metadata:

- snapshot `…/models--prism-ml--Ternary-Bonsai-2-27B-mlx-2bit/snapshots/3f926b415992eaa2ae9dd7b573706494d6bbf787`
  (pinned revision);
- `model.safetensors` 8,595,477,990 bytes, content sha256
  **`130de5925082c168b7866b2e91b52e44abbafc99017e3ca352b77b5b55a269ed`**
  — confirmed by three independent sources: the git-LFS pointer at the
  pinned commit, the pack's own `files.json`, and `sha256sum` of the
  cached bytes (re-fetched fresh 2026-09-20 after a false-alarm
  quarantine, see below);
- `config.json` 58,145 bytes, schema_version 2,
  model_type `prism_hadamard_qwen35` (its files.json sha256 is
  `238de7c5…`, which the loader's `config_sha256` reports at runtime);
- 57 GiB free on the host; window outputs go to disk (~/quarantine-bonsai2
  showed /tmp tmpfs at 98% — no artifacts under /tmp).

**Lesson recorded (matches ServePerformanceAudit's shard-2 finding):**
this is an Xet-backed repo — local HF cache blob FILENAMES are not
content sha256. An identity check that trusted the blob name flagged the
intact cache as corrupt; the pointer + files.json comparison resolved it
in minutes. The window's identity gate hashes the bytes and compares
against `130de592…` from the pointer, never against a cache path name.
A redundant hash-identical duplicate created during the false alarm was
verified and removed; `~/quarantine-bonsai2/QUARANTINE.txt` carries the
corrected record.

## Procedure

1. **Provenance first.** Build/install the wheel under test from this
   branch's base commit; `scripts/mlx_provenance.py` output is recorded
   in the receipt beside every number. No measurement without the
   provenance line.
2. **Weights identity gate:** re-verify snapshot revision dir, file size
   8,595,477,990, and content sha256 `130de592…` against the commit's
   LFS pointer / files.json before anything else. Any mismatch → stop
   (that is a fault, window ends). Never use the local cache blob
   filename as the expected hash (Xet-backed repo).
3. **Load gate:** `mlx_omarchy_bonsai2.server.serve_main --model
   <snapshot>` on the GPU; record startup line: `packed_modules` (expect
   402), `resident_bytes` (upper bound 8.6 GiB minus excluded vision
   tensors; the loader reports exact bytes), `excluded_bytes` prefixes,
   load wall time.
4. **HTTP gates, loopback only:**
   - `GET /health` → 200 with `config_sha256`, quantization, device=gpu,
     license files present.
   - `GET /v1/models` → id matches `--model-id`.
   - **Correctness gate — frozen prospective criteria (written before
     the window opens, per Main + MCQ).** Input sets: FIVE new fixed
     8-token sequences, `--input-set A` … `--input-set E` (derivation
     `sha256("bonsai2-parity-v1:<set>")`, published here BEFORE the
     window). Steps: 2 per input. Arms: upstream pack runtime vs this
     repo's loader, both on the omarchy GPU device, same wheel.
     - FROZEN PASS CRITERION: per input and per step, the two arms'
       greedy token IDs are IDENTICAL, and the measured max |logits|
       delta is RECORDED. A greedy-ID mismatch on any input or step is a
       fault (window ends).
     - Tolerance handling: the CPU diagnostic band is the recorded
       9.5e-05 (post-measurement band, NOT a predeclared gate — see
       receipts/2026-09-20-bonsai2-port-parity-cpu.md). The GPU deltas
       are compared against it and "tolerance movement" is REPORTED.
       The >1e-2 trip FAILS the gate — this rule stays frozen; the
       recorded per-step max|logit| magnitude column is CONTEXT for
       post-window review documentation only and does not rescue a
       trip. Any changed criterion belongs to a future experiment with
       its own prospective freeze.
     - The exact-token-877 CPU equality is a NARROW check; no blanket
       numerical-equivalence or model-quality claim is made anywhere.
   - Warm second request, different prompt; record the `timings` block.
     **No performance claim is made against the historical 1.44 tok/s
     band** (different checkpoint lineage and harness); the receipt
     records numbers with provenance and stops.
   - Strict context bounds on device: `prompt + max_tokens == cap`
     succeeds; one token over → 400 naming the cap; streaming SSE ends
     with `[DONE]`, every chunk a valid chat.completion.chunk, error
     paths return in-band errors; malformed bodies (non-object JSON,
     bad UTF-8, NaN/bool temperature, non-string content) → 400 over
     the real device.
   - Atomic resident admission on device: server launched with
     `--managed`; `reservations.json` shows the entry pending BEFORE
     weights materialize, resident after load (with
     `resident_floor_bytes` = exact weights), and absent after server
     shutdown; an `admit_and_reserve` refusal path is exercised with a
     deliberately inflated second reservation BEFORE the real launch
     and must exit 3.
5. **Memory gate:** `--managed` is the admission contract: admit +
   reserve against `mlx_omarchy_serve.budget` BEFORE any weights are
   touched (estimate_required over exact header weights + KV per token
   at the served context + workspace), transition pending → resident
   after the load, release on exit, exit 3 fail-closed on any gap.
   Non-managed runs keep the best-effort reservation. Record:
   `mx.get_peak_memory()` after generation, reservation_bytes vs actual
   resident, 16 GiB feasibility at the served `--max-context` (KV ≈ 64
   KiB/token on the real pack; 16 full-attention layers, GDN state
   constant on the other 48).
6. **Hand-back:** release flock, restart service, health 200, device
   state recorded; receipt in `receipts/` with every gate's raw output.
   First fault ends the window; findings are written up as-is, no
   retry loop inside the session.

## Not in this window

- No abliterated pack, no MTP, no vision/image serving (excluded tensors
  are reported, not loaded), no public exposure of the endpoint, no
  catalog qualification flip (that is ModelCatalogQualification's manual
  commit after this receipt exists).
