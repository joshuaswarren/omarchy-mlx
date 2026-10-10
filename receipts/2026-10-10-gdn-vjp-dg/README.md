# Fused GDN backward: dg summed only one lane's partial

Source commit: `b1115c0fb7a39f2c4655aa8897b9aa40b3484934` (branch `agent/w7q-gdn-vjp-dg`).

## Defect

`gated_delta_vjp.comp`, backward pass: each of the 32 lanes owns 4 of the 128 state columns and computes its own partial of
`<s_hat, s_prev>` (the gate gradient dg for the token). Only lane 0 stored its partial to shared memory, so dg carried 4 of 128 columns.
At T = 1 the initial state is zero, so every partial is zero and dg looked exact.

## Change

Every lane stores `s_dg[lane][row]`. Lane 0 of row 0 sums all 32 lanes of the four rows before the compare-exchange add. `s_wu[4]` replaces
the second slot of the old `s_scal` array, which now holds only `w*u` per row. The two `may_fail` marks on the GDN VJP doctest cases are removed.

## Before (same doctest, same shapes)

| Chip | Source | dg relative L2 vs float32 CPU reference |
|---|---|---|
| M1 (G13G) | v0.7.32 freeze battery `f2-battery-omarchy_fast_ops_tests.log` | 0.961636 (Hk=Hv=16, T=33) |
| M2 Max (G14C) | same battery | 0.961638 (Hk=Hv=16, T=33); 1.009 (Hk=4, Hv=8, T=33); 0.946 (Hk=4, Hv=16, T=17) |
| M1 Max (G13C) | `docs/known-defects.md`, 2026-10-09 entry | 0.96 (T=33) |

## After (M1 Max, G13C)

Linux kernel 7.1.12-2. Private Honeykrisp build through `VK_ICD_FILENAMES` (Mesa git `6543eeb7df7`, driver library sha256 prefix
`3546bcafe8ed3995`). Test binary `e11e25360e075844`. The log does not print the device name. Build-tree prefix replaced by `<build>`.

| Shape | dg relative L2, fused vs composed | dg relative L2, fused vs float32 CPU reference |
|---|---|---|
| Hk=Hv=16, T=33 | 0.000000 | 0.001543 |
| Hk=4, Hv=8, T=33 (opt-in GQA flag) | 0.000000 | 0.001492 |
| Hk=4, Hv=16, T=17 (opt-in GQA flag) | 0.000000 | 0.001568 |
| Hk=4, Hv=16, T=17, host-repeated | 0.000000 | 0.001568 |

`g13c/gdnvjp-ticket.log`: the two `fused gdn vjp` cases, 30 assertions, 0 failed, with no `may_fail` mark.
`g13c/gdnvjp-suite.log`: whole `omarchy_fast_ops_tests`, 47 cases, 1,383,970 assertions, 4 failed. The 4 are the known `may_fail` SDPA VJP case
(`test_fast_ops.cpp:80`), unchanged by this change. The earlier suite runs had 8 failed assertions: 4 SDPA and 4 GDN. The 4 GDN ones are gone.

Dev-box lavapipe cannot test this kernel: its subgroup size routes the primitive to the fallback, and the numbers are identical before and after.

## Not covered

- The fix has not run on M1 (G13G) or M2 Max (G14C). The shader change is not chip-specific, but only G13C was run.
- No speed number. From the source, the kernel does the same loads and adds one 32-lane sum in shared memory per token; that is a reading, not a measurement.
- SDPA VJP: the 4 remaining failed assertions in the suite are a separate defect and are not touched.
- Training end to end through a GDN model.
