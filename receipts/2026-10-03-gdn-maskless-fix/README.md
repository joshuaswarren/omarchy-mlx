# GDN maskless prefill investigation

## Current kernel-level result

The earlier claim that maskless GDN prefill produces an incorrect final state is **not reproduced by the current synthetic kernel sweep**. On jw16, the origin-main baseline wheel (`0.32.4.dev202610030818+b933d812`, SHA-256 `4f9795c63faf98499860e250c056edce8936781e2a96054fa3648d002808f51c`) matched the loaded extension and library per `scripts/mlx_provenance.py`. For T=63/64/65/96/352 and head repeats 1/2/3, maskless and all-True final-state maximum errors against the fp64 recurrence were 7.62e-8 to 1.56e-7. Y differences from fp64 were 0.00195 to 0.00388, consistent with bf16 output rounding. The probe's `bitsame` field compared Y, not state. See captured baseline output `artifact://24329`.

A route-control defect was found and corrected: GDN documented `MLX_OMARCHY_NO_COOPMAT_GDN`, but the implementation read the generic `MLX_OMARCHY_NO_COOPMAT` variable. The GDN-specific flag now controls the GDN coopmat route independently.

## Focused GPU test

The new `omarchy_gdn_maskless_correctness_tests` doctest compares Y and final f32 state against an fp64 tokenwise recurrence for T=63/64/65/96/352 and repeats 1/2/3. Its bf16 q/k/v/beta inputs are rounded before reference evaluation.

On jw16, the test binary was built from the staged current worktree source based on b933d8124. Binary SHA-256: `a35f460994cd2972fc9ff47519a8ce898c7ec91ec609bb3b29c5f9f082a62b33`. The run was announced to w72 and used `/var/tmp/appbar/gpuwin.sh`:

- Default GDN routing: 1/1 test passed; 120/120 assertions passed; 0 skipped.
- `MLX_OMARCHY_NO_COOPMAT_GDN=1`: 1/1 test passed; 120/120 assertions passed; 0 skipped.
- Combined GPU-window duration: 349.50 s. The wrapper reported `STOP svc=inactive lock_holders=[]` and `RESTORE health_ok=1 probe_finish=length active=active` for both runs.

The second invocation exercises the GDN-specific scan selector. This receipt has no dispatch-grid trace, so it does not claim the kernel grid or per-layer dispatch counts. It also does not establish 27B model-level equivalence.

## Open question

The reported 27B symptom (maskless greedy ids `[0]*8` versus nonzero masked ids) remains unverified end to end. A current kernel synthetic baseline did not reproduce it. The next investigation must compare masked and patched-maskless 27B inference on the M2, with per-layer hidden/state dumps, and identify the first divergent tensor before attributing the symptom to the backend. M2 GPU work is blocked during the owner capture window 08:30Z–11:00Z.
