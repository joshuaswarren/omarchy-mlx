# MLX pin bump: 59d600b5 -> 9c3d35571a (0.32.3 line, 2026-09-29 tip)

Lane: MlxBump. Branch: `agent/mlx-pin-bump-vjp`. Receipt status: results
filled after the M1 gate window; structure and upstream audit are complete.

## Why this pin

- `mlx.lock` moves from `59d600b5e64c238427d0f8d897ab7c682ef4d3d2`
  (2026-09-17) to `9c3d35571ac450a8ecf5c17b4d0e3fac52c08bc8` — upstream main
  tip at bump time, 37 commits ahead, and past both VJP merges the lane
  tracked: #4563 (SDPA VJP, merge `83b976ea43`) and #4565 (GDN VJP, merge
  `9295197533`).
- CI on the target commit: all 30 check runs `success` (checked via the
  GitHub commits/check-runs API on 2026-09-30). `9295197533` and `83b976ea43`
  are also all-green (29 each); the tip carries two more fix commits
  (#4594 cooperative-tensor build fix, #4513 slice clamp) at no extra risk.
- Latest upstream release is still v0.32.3, so `MLX_VERSION` stays 0.32.3.
- Archive: `https://github.com/ml-explore/mlx/archive/9c3d35571a….tar.gz`,
  SHA-256 `3e564bf7a0d5e5bac3a56fc26668cdcedd4a9db9fbfbf2bb6d762632ceb6f65a`
  (verified by `sha256sum` on download and by `scripts/prepare-mlx.sh`'s
  `--check` at prepare time).

## Patch series rebase

`tools/upstream-compat-check.sh probe 9c3d35571a` first reported 5/18 patches
failing; each was rebased onto the new baseline and the probe now reports
**17/17 clean**:

| patch | drift | resolution |
|---|---|---|
| `mlx-rope-settle-tape.patch` | upstream added the `InExportTracing` block above the `eval_impl` anchor (fence work #4552) | hunk re-anchored; eval-nest-depth guard unchanged |
| `mlx-python-buffer.patch` | upstream added its own try/catch around `eval()` in `getbuffer` | our PEP 3118 rewrite subsumes it (whole-body try, contiguity flags, packed copies); rebased on the new skeleton |
| `mlx-gated-delta-raw-gates.patch` | #4565 inserted `GatedDeltaUpdate::vjp` and the `GatedDeltaUpdateVJP` class, plus an `is_training_` field | `gated_delta_update_raw` re-anchored between `gated_delta_update` and the new vjp; `raw_gates_` joins `is_training_` in the private section |
| `mlx-gdn-conv-decode.patch` | same region | `gdn_conv_update` / `GdnConvUpdate` re-anchored after the VJP class |
| `mlx-distributed-reduce-scatter-assert.patch` | **absorbed upstream** (#4557 `ReduceScatter::eval_cpu` now asserts `inputs.size() == 1`) | patch deleted; `scripts/prepare-mlx.sh` entry removed |

Validation: the full `scripts/prepare-mlx.sh` run at the new pin produces a
tree byte-identical (source files, `diff -rq`) to an independently hand-built
reference tree, and `tools/upstream-compat-check.sh probe` exits 0.

## Upstream drift audit (59d600b5 -> 9c3d35571a)

`tools/upstream-compat-check.sh api` classified 37 commits. Primitive-level
diff of `mlx/primitives.h`: **no new classes**. `mlx/fast_primitives.h`:
**one new class**, `GatedDeltaUpdateVJP`. Per-item state for the Omarchy
backend:

| upstream change | commit | Omarchy backend state |
|---|---|---|
| `fast::GatedDeltaUpdateVJP` primitive (GDN VJP) | `9295197533` (#4565) | implemented (fallback): `use_fallback` returns true; GPU entry is the named compatibility rejection (`OMARCHY_UNSUPPORTED_MULTI`) — the fused backward is Metal-only upstream |
| SDPA VJP: Metal fused backward + `has_sinks` gate param | `83b976ea43` (#4563) | fallback (pre-existing): `ScaledDotProductAttentionVJP::use_fallback` = true + named rejection; overlay signature updated for `has_sinks` |
| `Fence::wait(stream, x, value)` / `Fence::update` returns value | `77e1cfb23d` (#4552) | implemented: overlay `fence.cpp` returns the published count; `wait` covers every published update (superset of the requested value) |
| `mx.matrix_transpose` | `073d2252c9` (#4402) | composed: lowers to the existing `Transpose` primitive; omarchy `Transpose` already implemented |
| fixed-size `mx.unique` | `92db340a54` (#4501) | composed: built from existing sort/slice ops; no new primitive class upstream |
| fused `fast.cross_entropy` Metal kernels | `02ce1fb6a0` (#4520) | fallback (unchanged): Metal-only; the composed path stays the reference and is tested |
| `ReduceScatter::eval_cpu` assert fix | `42987b66be` (#4557) | absorbed upstream; our patch deleted |
| SDPA D96/V64 + D256/D512 memory work, gather mm/qmm, pre-M5 memory | Metal-only | not applicable |
| shapeless scan compilation (#4510), compiled-kernel float precision (#4511) | common compile paths | covered by `omarchy_compiled_tape_tests` (battery) |
| CPU fixes (large-data binary ops #4517, negative-stride col-reduce #4529), CUDA-only items, Python-side fixes (#4488/#4492/#4555), behavior fixes (IndexError #4484, floor_divide #4515, complex64 NaN order #4519, slice clamp #4513, vmap view #4586, cumsum bool #4590, upsample #4500, split span #4540, ParallelFileReader #4532, stream leak #4576) | common/CPU/CUDA/Python | inherited automatically; no omarchy-backend entry required |

## New tests

`omarchy_fast_ops_tests` gains "sdpa and gated delta gradients hold the zero
CPU dispatch contract":

- `grad` through `fast::scaled_dot_product_attention` vs a composed
  softmax(QK^T·scale)V graph on the same device (f32, 5e-4).
- `grad` through `fast::gated_delta_update` (composed shape, T=2) vs the
  composed per-token recursion (f32, 5e-4).
- The fused decode shape (T=1, bf16, square 128 heads) runs vjp through the
  fused forward + the new `GatedDeltaUpdateVJP` fallback (2e-2, bf16
  rounding) — this is the regression guard for the new primitive.
- Dispatch counters advance; no GPU primitive moves to an explicit CPU
  stream (counters flat under a CPU-stream probe).

## Gate results

Window on the M1 host (2026-09-30T23:49-2026-10-01T00:01Z, `/tmp/m1-gpu.lock`
held throughout, kernel 7.1.13-3-2-ARCH, Mesa/Honeykrisp as installed):

- **Token-records digest parity: PASS.** The new-pin wheel on the pinned
  Qwen3.8-2B contract model (10 corpus prompts, 32 greedy tokens, 512-token
  prefill protocol, n=1 pass set) produced
  `ordered_records_sha256 = dbf704971617fdfc…` — identical to all five
  pre-bump runs on the same host (`contract-CURRENT-1/2`, `contract-NEW-1/2/3`
  on the pre-bump wheel `06711ad`). The 37-commit bump is bit-invisible on
  the contract decode path.
- **Release-equivalent wheel: built on the M1** with `glslc 2026.3` and the
  pinned whole-encoder bundle (bundle manifest/program SHA-256s match the
  runtime pin): `mlx_omarchy-0.32.4.dev202609302356+6d17edd-cp314…aarch64.whl`
  (the pinned tree is the 0.32.4 dev line — upstream's version bump
  `7916d8b0c8` lands between the old and new pin; the release tag line
  stays v0.32.3). `scripts/mlx_provenance.py --expect-wheel`:
  `version_match: true`.
- **Battery: 28/30** standing suites pass. Two failures, both test-file
  level, neither backend-runtime:
  1. `omarchy_fast_ops_tests` — the new doctest tripped the new baseline's
     strict vjp cotangent-shape rule (`value_and_grad` over a non-scalar
     output). Test bug; fixed by sum-wrapping the autograd outputs (grad of
     the sum equals grad with an all-ones cotangent). Fix committed on the
     branch; M1 rerun pending.
  2. `omarchy_indexing_ops_tests` — the bf16 small-k partition tail check
     (`test_indexing_ops.cpp` bf16 rows) mismatches the host key-map
     reference; the f32 section of the same test passes and the radix path
     still runs in exactly one dispatch. Diagnostics added to the test;
     root cause open — this suite passed at `db74f11a` yesterday, so the
     delta comes from the baseline bump and must be understood, not
     re-pinned.
- Provenance beside every number above; raw logs and the runs JSON live in
  the private lab under `artifacts/MlxBump/20260930-jwm1-pin-bump/` with
  SHA256SUMS.

## Merge status

NOT merged. The gate for merging is the full 26-suite battery green at the
final commit on the M1; the two test failures above are being fixed and the
battery will be rerun in a follow-up window before `origin/main` moves.

## Scope notes

- The Parakeet 104/104 full-ASR gate is an ANE-encoder fixture owned by the
  ANE lane; the ANE lane held the host during this lane's window, so the GPU
  battery + digest parity are the gates executed here. The
  `omarchy_ane_bundle_tests` suite (39 cases) runs in the battery.
- `omarchy_ane_runtime_tests`, the ANE worker suites, and the two-rank
  distributed harness are outside the 26-suite standing battery and require
  the private ANE checkout / a second rank; unchanged by this bump.
