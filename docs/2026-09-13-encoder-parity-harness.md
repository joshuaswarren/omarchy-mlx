# Encoder parity harness (Phase 6 prep, plan sections 40-43)

Status: SKELETON, host-only. The parity runs execute once Phase 5
compiler closure lands; this document plus
`overlay/tools/coreml/encoder_parity.py` (tested by
`overlay/tests/omarchy/coreml/test_encoder_parity.py`) is the contract
they run under. It extends `docs/differential-harness.md` — same
first-divergence discipline, same exit codes (`0` match, `3`
divergence, `2` environment/usage error), never a tolerance where a
bit comparison is legitimate.

## Section 40 — layer-by-layer acceptance plan

The pinned encoder (3351 ops, 29 op types) splits into **26 stages**:
a prologue (163 ops: input casts, the 4-stage subsampling conv stack,
mask derivation), 24 transformer layers (`encoder_layers_00` … `_23`;
146 ops for layer 0, 132 for layers 1-22, 130 for layer 23), and an
epilogue (8 ops: final layer_norm, projector linear, output casts).
The planner derives this from the parsed MIL program by layer-prefix
boundaries; non-layer ops attach to the surrounding region.

Each stage carries:

- op range and histogram (`build_stage_plan`),
- **checkpoints**: every non-const tensor the stage produces that a
  later stage or the function outputs consume. The prologue
  checkpoints are `attention_mask_7`, `output_mask`,
  `linear_0_cast_fp16`; the epilogue checkpoints are
  `encoder_hidden`, `encoder_mask` (the function outputs), and
  `linear_217_cast_fp16`.

Pass 1 compares every checkpoint per stage in program order; on the
first diverging stage, pass 2 descends inside that stage wrapping
op groups, mirroring the differential harness's model-mode descent.
The machine-readable plan is `plan.to_dict()` (schema
`mlx-omarchy.parakeet-encoder-parity-plan.v1`).

## Section 41 — frozen tolerances per execution semantics

Two classes, fixed at plan build time by op semantics (never by
measurement of the run under test):

1. **`fp16_value_exact`** — pointwise/staging ops (`add`, `sub`,
   `mul`, `cast`, `select`, `reshape`, `transpose`, masks, …).
   Comparison is fp16 value equality with **±0 equivalence**: the
   device's `+0.0` equals a mathematical `−0.0` product — proven on
   hardware (2026-09-13 worker window; the H13 compiler models zero
   products as unsigned, mil-hwx-compiler `aa688df`), and enforced by
   `mlx-omarchy-ane-worker --expect`.
2. **`relative_l2`** — accumulating ops (`linear`, `matmul`, `conv`,
   `softmax`, `layer_norm`, …). Bounds come from the reference lock's
   frozen numerical contract and are never loosened after a failure:
   `encoder_max_abs_err = 0.3`, `encoder_mean_abs_err = 0.02`,
   `encoder_rel_l2_err = 0.1`, `nan/inf count = 0`; the end-to-end
   token IDs and transcript must match exactly.

For this encoder every stage contains accumulating ops, so stage-level
acceptance is `relative_l2`; the exact class governs the pointwise
checkpoints during descent. `frozen_tolerances(lock)` returns the
contract; `golden_anchors(lock)` binds the comparison artifacts
(`mel.npy`, `encoder_input_features.npy`, `encoder_hidden.npy`,
`encoder_mask.npy`, `token_ids.json`, `transcript.txt`, …) to their
locked sha256s.

## Section 42 — backend trace counters in the harness output

Every parity run emits the nine ANE counters from
`overlay/mlx/backend/omarchy/trace.h` (`ane_models_loaded`,
`ane_packages_compiled`, `ane_package_cache_hits`, `ane_worker_starts`,
`ane_submissions`, `ane_timeouts`, `ane_input_bytes`,
`ane_output_bytes`, `ane_exec_ns`) alongside the GPU-side counters
(`gpu_primitive_dispatches`, `vk_compute_dispatches`). The report
skeleton is `empty_counter_report()`; `ANE_COUNTER_FIELDS` names the
nine, wired through `ane_trace_snapshot()` (compile-time gated by
`MLX_OMARCHY_ANE_TRACING`, same as the runtime tests).

## Section 43 — no-CPU-tensor proof surface

The parity run asserts, per stage and in total:

1. `cpu_tensor_events == 0` — no tensor primitive executed on a CPU
   fallback path (plan section 3.3; an unsupported op must fail
   explicitly, never fall back).
2. **Byte accounting**: `accounted_input_bytes`/`accounted_output_bytes`
   (sums of the ANE and Vulkan path counters) equal the expected byte
   totals of every compared checkpoint — every compared tensor's bytes
   are attributable to a device path.
3. `ane_timeouts == 0` and worker completions = submissions − timeouts
   on a clean run.

`evaluate_invariants(report)` returns the named violation list; an
empty list is the §43 proof for that run. Violations are failures
regardless of numeric agreement.

## Procedure when Phase 5 closes

1. Build the bundle for the encoder (Phase 5 output) and load it with
   `load_bundle` (strict schema-4 validation).
2. Execute the stage plan on the target with the bounded worker;
   capture every checkpoint; fill the counter report.
3. Compare per §41; localize first divergence per the differential
   conventions; assert §43 invariants.
4. Receipt: plan JSON, counter report, per-stage verdicts, first
   divergence if any, golden-anchor hashes.
