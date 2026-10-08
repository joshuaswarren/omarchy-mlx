# Remove the M1 (G13G) f32 `a @ b` direct-kernel route row

Date 2026-10-08. Follows `receipts/2026-10-08-direct-gemm-floors` (PR 38), where the same row at m = 2048 was dropped after a shape grid. This change deletes the row that was already on main (from m = 4096).

## Change
- `matmul_direct_select.h`: the row `{"G13G", float32, a @ b, from m 4096, MatmulDirectF32NnK4S8}` is gone (15 rows become 14). The M1 f32 `a @ b` product takes the generic direct f32 kernel again.
- `compute.cpp`, `CMakeLists.txt`: the k4s8 f32 `a @ b` shader target and its dispatch case are removed. The enum id `MatmulDirectF32NnK4S8` stays, marked retired, because profile ids are append-only.
- `test_matmul_family.cpp`: the floors entry is removed and the route test pins `f32 a @ b` on the M1 to the shipped route.

## Why (H36)
M = main before the f32 rows (wheel `+ef70b8cc`), head = the row present (wheel `+50e40389`). f32 `a @ b`, m 4096 and 8192, n 4096 and 9728, k 2560 and 11008, three rounds, 30 s idle before each arm run, kernel 7.1.12-2-12.6 (aurora 12.6 prerelease), driver library sha256 starting 3546bcafe8ed3995. Output hashes equal in all 8 cells. Gain of the row over M, rounds 1 / 2 / 3:

| m | n | k | gain |
|---|---|---|---|
| 4096 | 4096 | 2560 | -7.6 / -7.9 / -8.3 % |
| 4096 | 4096 | 11008 | +30.9 / +31.0 / +33.1 % |
| 4096 | 9728 | 2560 | -13.8 / -13.6 / -13.6 % |
| 4096 | 9728 | 11008 | void (M spread 80 %) |
| 8192 | 4096 | 2560 | -8.3 / -7.7 / -6.7 % |
| 8192 | 4096 | 11008 | +32.5 / +34.3 / +31.1 % |
| 8192 | 9728 | 2560 | -13.0 / -12.9 / -13.1 % |
| 8192 | 9728 | 11008 | -41.9 / -42.5 / -41.4 % |

The row loses by more than 3 % in at least two of three rounds at 5 of 8 cells, so by the registered rule it must be bounded or removed. It wins at n = 4096 with k = 11008 (and at n = k = 4096, +21 % in the earlier receipt). A bound would be fitted to two winning points, and f32 `a @ b` is not an LLM path on the base M1, so the row is removed. Recorded as given-up gain: +21 to +34 % at n = 4096, k >= 4096.

## Checks on this change (H37)
Wheel `0.32.4.dev202610081650+468db3eb`, test binary sha256 starting 62be9776be654c10, kernel 12.6 prerelease, same M1.
- `omarchy_matmul_family_tests`: 33 of 33 cases, 82942870 of 82942870 assertions (`m1-g13g/h37/family-tests.log`).
- Loaded-library provenance of both wheel environments (`scripts/mlx_provenance.py`): `verified=match`, version match true (`m1-g13g/h37/ticket.log`).
- Dispatch trace, one product per cell, compared by kernel name between M and this build (the two number their kernels differently): all 6 f32 `a @ b` cells (m 128 to 4096) dispatch the same kernel. The 23 other differing cells are the other landed rows that M predates (`m1-g13g/h37/trace-by-kernel-name.txt`).
- Timing, the 8 cells above, M against this build, three rounds: the 6 cells with M spread under 5 % are inside +-1 % in every round. Two cells (m 4096 and 8192 at n 9728, k 11008) are void by the 5 % rule (M spread 10.6 % and 87.5 %: one M round far slower) and are not reached (`m1-g13g/h37/analysis.txt`; its `row ... no-benefit` tags are the analyser's wording for a cell with no gain, which is what a deleted row should show).

## Not covered
- Other chips: the M1 Max and M2 Max have no f32 `a @ b` row; nothing was run there.
- Decode and model-level numbers: none. Loaded-library provenance was recorded in the same ticket as the other checks, first, before the family tests, trace and timing.
- Device reopen and firmware identity were not recorded.

## Files
- `m1-g13g/h36/`: raw per-round cells and the analysis of the loss. `m1-g13g/h37/`: raw cells, analysis, trace verdict, family-test log, ticket log.
- `tools/`: `gen_grid.py` (cell script, chip key `g13gnn`), `gen_an.py` (analyser; fails with both build stamps printed when an arm's recorded version does not match; `GEN_AN_STAMPS=PR=<short sha>` names the arm under test), `route_trace_names.py`.
