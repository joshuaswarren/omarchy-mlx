# Bool MaskedScatter on the Vulkan backend — upstream #4635 (jwm1 G13G verification)

## Change
- `overlay/mlx/backend/omarchy/shaders/masked_scatter_bool.comp`: per-element
  atomicOr byte-packing kernel (no scan dependency; rank computed by the
  element's own position among prior trues via atomicAdd-free byte counting).
- Routing: `ComputeKernel::MaskedScatterBool` -> `masked_scatter_bool` shader
  target (wiring check green: 566 declared, 561 dispatch rows, 560 targets).
- Tests: `overlay/tests/omarchy/test_indexing_ops.cpp` — bool MaskedScatter
  case (33 assertions, NumPy-computed reference; 1-D through N-D, wait/no-wait,
  broadcast, trailing-write) + isolated micro-test "bool cumsum after a bool
  atomic-pair copy inside one batch" (12 assertions).

## Verification (jwm1, G13G, coreglass honeykrisp ICD)
- `git f3dc90c94` (= main after landing), prepare-mlx clean, ninja -j4.
- `omarchy_indexing_ops_tests` full: 59/59 cases, 35904/35904 assertions PASS.
- bool MaskedScatter case: 33/33 assertions PASS.
- scan micro-test: 12/12 assertions PASS on jwm1.
- Control (pure origin/main ec2cd9dc, same host/ICD): tag filter
  `[scatter],[gather],[indexing],[vmap]` = 262/264 with 6 assertion failures
  in `arg_reduce_tests.cpp:25` — IDENTICAL failure count on the branch tree
  (order-dependent shared-state flake on main, pre-existing; each arg_reduce
  case passes individually: 3/3).
- Binary not provenance-gated here (test binary, not a measurement receipt);
  the 2026-10-07 receipts remain the provenance source.

## History of the scan-zeros observation
- Observed ONLY on jw16/G13C (before the 2026-10-08 wipe) from a tree whose
  MaskedScatterBool row pointed at the DEBUG shader
  (`masked_scatter_bool_dbg`, `-DMS_BOOL_DEBUG_RANK=1`). The debug kernel is
  gone from the landed branch (wiring check rejects unwired targets); the
  landed kernel computes ranks per-element without a scan primitive.
- Does not reproduce on jwm1/G13G with the landed kernel (12/12 micro-test,
  33/33 bool case). jw16 is unavailable for re-verification (wiped).

## Landing
- `f3dc90c94` on `main` (merge of the 1007b line into main; first-parent
  lineage continues through `ec2cd9dc`).
- Leftover jwm1 trees (`~/u1007b-jwm1`, `~/main-verify-tmp`) kept on disk per
  the safe-remote-deletes rule (no remote rm; git-managed worktrees).
