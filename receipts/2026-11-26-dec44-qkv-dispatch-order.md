# DEC-44: QMM dispatch-order probe (flag bit 19, `MLX_OMARCHY_QMM_ORDER=stride`)

Receipt for the PR branch `dec44/qmm-order-stride` (heads `a52df406` probe,
`67af96e1` fix). DEC-44 asked whether reordering the workgroup → row-block
walk inside `qmm_vec.comp` (a 64-wide transpose of the walk, flag bit 19 on
the shared params field, gated by the env var `MLX_OMARCHY_QMM_ORDER=stride`)
wins decode throughput on the DEC-44 QKV shape.

## Verdict

**Negative — the knob ships default-off and stays off.** The probe proved no
dispatch-order win on either tested part; the reordering is bit-identical to
the plain walk by construction (pure permutation of workgroup indices,
guarded `groups_total >= 64 && groups_total % 64 == 0`).

## Hardware evidence (recorded from the author's runs at `bb1d20c9`/`67af96e1`)

- Geometry suite: **GEOM-TEST-PASS 12/12 rows + 3/3 probe rows** on G13C
  (`tests/omarchy/test_qmm_geometry.cpp`), including the swiglu-2x8w2 arm
  whose grid (groups0 = 384) engages the 64-wide remap.
- 40 ABBA legs, 5 pairs, ctx512 + ctx2048, on **G13C and G14C** (boots
  `d38b156693`, `b80cae1b9d94`): every leg greedy-identical against the
  default walk; decode medians within ±0.64% — inside run-to-run noise, no
  win.

## Bug found and fixed failing-first (flag 16 paired epilogue)

Under flag 16 (paired epilogue) the dispatched grid is `groups0` alone, but
the pre-fix shader keyed `groups_total` on the full per-entry-point sum, so
the remap produced workgroup indices past `n` and the top half of the rows
was never written. The fix (commit `67af96e1`) keys `groups_total` on the
dispatched grid at both permute sites (`qmm_vec.comp` dispatch paths for the
plain and paired-epilogue routes). The regression arm gates 6144 rows so
`groups0 = 384` engages the remap, asserting byte identity.

## Shared-bit correction (2026-11-28, re-review)

The original probe shared flags bit 17 with the out-gate prologue marker;
spirv-dis of the outgate build showed the bit consumed as both the marker
and the permute enable, and every outgate dispatch with
`groups_total % 64 == 0` remapped without the env set. The probe now owns
flags bit 19 (`524288`) outright — bit 17 is solely the out-gate marker —
so the ABBA arms differ again. **The 40-leg ABBA above was potentially
vacuous on an outgate-dispatching shape** (the bit set in both arms); that
shape needs a re-run with the split bits before a "no win" claim covers it.
Non-outgate shapes were measured with neither arm carrying the bit and
stand as recorded.
