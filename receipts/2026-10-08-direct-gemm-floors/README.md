# Direct GEMM route rows for the M2 Max and lower m floors on the M1

Date 2026-10-08. Follows `receipts/2026-10-07-direct-gemm-variants` (f16 rows) and `receipts/2026-10-07-bf16-direct-gemm` (bf16 and f32 rows). Every measurement here is at MLX level (`a @ b` with MLX arrays, one product per `mx.eval`), n = k = 4096, with output hashes compared between arms.

## Change
`overlay/mlx/backend/omarchy/matmul_direct_select.h`, one new pipeline, one new test, one doc line. The G13G f32 `a @ b` floor (2048 in the first candidate) is not part of the change: the shape grid below showed it losing at larger n, so the row keeps its existing floor of 4096.

| chip | dtype, orientation | kernel | from m |
|---|---|---|---|
| M2 Max (G14C) | bf16 `a @ b.T` | k4s8 | 512 |
| M2 Max (G14C) | bf16 `a.T @ b` | ws8 | 4096 |
| M2 Max (G14C) | f16 `a @ b.T` | k2s8 (new `MatmulDirectF16NtK2S8`) | 1024 |
| M1 (G13G) | bf16 `a.T @ b` | ws8 | 512 (was 4096) |
| M1 (G13G) | f16 `a.T @ b` | ws8 | 1024 (was 4096) |
| M1 (G13G) | f32 `a @ b.T` | k4s8 | 512 (was 4096) |
| M1 (G13G) | f32 `a.T @ b` | ws8 | 512 (was 4096) |
| M1 (G13G) | f32 `a @ b` | k4s8 | unchanged, 4096 (a floor of 2048 lost at n 9728: see Shape generality) |

The M1 Max (G13C) rows are unchanged. The G13G f16 `a @ b.T` k4s8 row keeps its floor of 512: a lower floor was measured and fails (below). The pipeline id is appended at the end of the kernel enum.

## Method
- Arms: C2 = main before the rows (omarchy-mlx `1b77a3cfe`, wheel +27aa6738), C6 and C7 = the candidate wheels (+8235d7c6, +14ca9e39), PR head 582c1a947 (wheel +582c1a94). C7 is C6 with two M1 floors set back and the M2 Max f16 row on k2s8; the route controls below show the PR head's routes equal C7's.
- Each timed block is one process per arm and cell grid, median of 18 to 30 timed evals after 3 warmups, three rounds in the order C2 C6 / C6 C2 / C2 C6, with an idle gap before each arm run (30 s on the M1 and M1 Max, 60 s on the M2 Max: that chip lost 16 to 20 % of its throughput within 30 s of continuous load and recovered after 60 s idle in a separate probe). Shader cache off.
- Input data come from numpy with a seed derived from the cell id, so every arm sees the same bits.
- Rule for a row (written before the data): C-arm over C2 at least 5 % on every round at every measured m from the row's floor up, output hash equal, C2 time spread across rounds at most 5 %. A row below its old floor that fails at its lowest measured m moves the floor up to the smallest m from which all larger measured m pass.
- Control rule: cells where both arms take the same route must agree within 3 % on every round. This failed on the M1 Max twice and on the M2 Max twice, each time on one to three cells where one or two of three rounds were off (most with mixed signs between rounds) and C2 spreads reached 7 %. By the rule those timing controls are void, and I did not call them passed. They are replaced by a deterministic control: the dispatch trace (`MLX_OMARCHY_TRACE_DISPATCH=1`) of one product per cell, C2 against C7. Pass means exactly the cells where C7 adds a row run a different kernel, and no others.

## Results
Gain of the candidate over C2 in percent, rounds 1 / 2 / 3, output hashes equal in every cell.

### M2 Max (G14C), driver v3 6543eeb7df7, kernel 7.1.12-2-12.3-sep-ARCH
| cell | gain |
|---|---|
| bf16 `a @ b.T`, m 512 / 1024 / 2048 / 4096 | +58..+60 / +61 / +66..+67 / +68 (C6 over C2; two runs agree within 2 points) |
| bf16 `a.T @ b`, m 4096 | +7.1 / +7.1 / +7.0 and +7.1 / +7.2 / +7.0 (two runs) |
| f16 `a @ b.T` k2s8 (C7 over C2), m 1024 | +12.9 / +12.4 / +12.5 |
| f16 `a @ b.T` k2s8 (C7 over C2), m 2048 | +12.5 / +12.4 / +12.0 |
| f16 `a @ b.T` k2s8 (C7 over C2), m 4096 | +11.4 / +11.5 / +11.4 |
| f16 `a @ b.T` k2s8 (C7 over C2), m 512 (control, both arms on the shipped route) | +0.4 / -0.4 / -1.1 |

The k2s8 choice over k4s8 (C7 over C6): m 1024 +6.3 / +5.5 / +5.9, 2048 +6.0 / +5.9 / +5.4, 4096 +5.7 / +5.8 / +5.6. At the kernel level (`gemm-bench`, two passes, bits equal) k2s8 gains over k4s8: f16 +1.2 / +4.9 / +5.0 / +5.1 at m 512 / 1024 / 2048 / 4096, bf16 -0.7 / +2.1 / +2.5 / +2.7 (no bf16 k2s8 row).

### M1 (G13G), driver v3 6543eeb7df7, kernel 7.1.12-2-12.3-sep-ARCH (C6 over C2)
| row | m: gain |
|---|---|
| bf16 `a.T @ b` ws8 | 512: +11.3 / +10.9 / +11.7; 1024: +11.2 / +11.3 / +11.0; 2048: +11.2 / +11.0 / +11.2 |
| f16 `a.T @ b` ws8 | 1024: +7.6 / +7.5 / +7.7; 2048: +7.7 / +7.7 / +7.7 |
| f32 `a @ b.T` k4s8 | 512: +79.5 / +79.6 / +79.0; 1024: +91.6 / +88.3 / +90.4; 2048: +96.9 / +98.3 / +96.9 |
| f32 `a.T @ b` ws8 | 512: +31.4 / +31.6 / +28.7; 1024: +34.7 / +33.7 / +34.9; 2048: +45.9 / +45.5 / +46.0 |
| f32 `a @ b` k4s8 (floor 2048 NOT adopted) | 512: -3.7 / -4.5 / -3.9 (fails); 1024: +3.0 / +3.4 / +3.6 (fails); 2048 at n = k = 4096: +20.7 / +22.4 / +21.5, but -4.6 to -43 % at three of the four other n and k tried (Shape generality) |
| f16 `a @ b.T` k4s8 below 512 | 128: -1.7 / -8.7 / -7.2 (C2 spread 8 %, void); 256: +3.2 / +1.5 / +1.8 (fails; floor stays 512) |

All 38 M1 control cells in that run are within 3 %. The kernel-level gain at m = 128 and 256 (`gemm-bench`: +29 to +55 %) does not survive at MLX level: a product of about 0.1 ms is dominated by the 0.4 to 0.5 ms fixed cost per `mx.eval` on this chip.

### M1 Max (G13C)
No route changes. Timing control, C2 against C6: output hashes equal on all 36 cells, 34 of 36 inside 3 % in the first run (two cells at -4.0 and +4.4 %), 33 of 36 in the rerun; the excursions were one or two rounds per cell, most with mixed signs: void by rule (see Method). Dispatch-trace control: no cell differs between C2 and C7 (36 cells).

### Dispatch-trace route controls (C2 against C7)
| chip | cells | differ | verdict |
|---|---|---|---|
| M1 (G13G) | 54 | exactly the 12 cells where C7 adds or moves a row (bf16 `a.T @ b` m 512 / 1024 / 2048; f16 `a.T @ b` m 1024 / 2048; f32 `a @ b` m 2048; f32 `a @ b.T` and `a.T @ b` m 512 / 1024 / 2048). The PR head has no f32 `a @ b` row change, so its expected set is those 12 minus the f32 `a @ b` m 2048 cell; the head was not traced on this chip | pass (C7) |
| M1 Max (G13C) | 36 | none | pass |
| M2 Max (G14C) | 24 | exactly 8: bf16 `a @ b.T` at 4 m values, bf16 `a.T @ b` m 4096, f16 `a @ b.T` m 1024 / 2048 / 4096 | pass |

### Route trace of the PR wheel on the M2 Max (cells compared by kernel name)
The PR wheel (+582c1a94) and C2 number their kernels differently (the PR base inserted `MaskedScatterBool` in the middle of the kernel enum, so every later id is +1), so the id comparison printed FAIL on 16 cells that all differ by exactly one id. Mapped to names with each build's own enum, exactly the 8 expected cells differ (bf16 `a @ b.T` m 512 to 4096: `MatmulBF16Coopmat` to `MatmulDirectBF16NtK4S8`; bf16 `a.T @ b` m 4096: `MatmulDirectBF16Tn` to `MatmulDirectBF16TnWS8`; f16 `a @ b.T` m 1024, 2048, 4096: `MatmulDirectF16Nt` to `MatmulDirectF16NtK2S8`) and no other. Script `tools/route_trace_names.py`.

## Shape generality and edge tiles (MatmulGap H33, H34)
Added after review: the rows above were measured at n = k = 4096 and m on the tile grid.
- Edge tiles: 216 cells per chip with m, n, k off the tile grid, C2 against the PR wheel, output hashes: 216 of 216 equal on the M1, the M1 Max and the M2 Max.
- LLM shapes, C2 against the PR wheel, 3 rounds, n 4096 / 9728 (/ 12288 on the M2 Max), k 2560 / 11008, m 512 / 2048 (/ 8192 on the M2 Max). Hashes equal in every cell on both chips.
  - M2 Max: no regression. bf16 `a @ b.T` +64 to +71 % at 17 of 18 cells (the 18th, m 512 n 4096 k 2560, reads +34 / +39 / +65 % with C2 spread 4.4 %); f16 `a @ b.T` k2s8 +10.7 to +15.8 % in the 5 clean row cells (the cells at m 2048 gain +10.7 to +11.3 %); 13 of the 36 row cells (all 6 bf16 `a.T @ b` cells at m 8192 and 7 f16 cells) are void by the registered 5 % C2-spread rule, though each shows a gain of +7 % or more in every round. One f16 control cell (m 512) is also void.
  - M1: f32 `a @ b` at m 2048 regressed (-4.6 to -43 %, three cells), so that floor change was dropped from this PR. The other rows gain at every non-void cell: f32 `a @ b.T` +44 to +503 %, f32 `a.T @ b` +21 to +150 %, bf16 `a.T @ b` +8 to +31 %, f16 `a.T @ b` +5 to +46 %. Three f32 `a.T @ b` cells are void by the 5 % spread rule. One control cell (f32 `a @ b` m 512, n 4096, k 11008) read +3.0 % in one round with C2 spread 2.9 %, outside the 3 % control band by the letter of the rule.
- Analysis outputs: `m1-g13g/h34-analysis.txt`, `m2max-g14c/h34-analysis.txt`.

### Tests at the PR head (main c26e44df5 plus this change)
`omarchy_matmul_family_tests`: 33 of 33 cases and 82942852 of 82942852 assertions on the M1, the M1 Max and the M2 Max. The new case "every row starts at its measured m floor" checks each row at its floor, above it, one below it and at n = 4095. The 450-cell shape sweep (3 dtypes x 3 orientations x 4 m x 3 n x 4 k plus view offsets, `tools/shape_sweep.py`) is bit-identical to main on all three chips: 450 of 450.

## Not measured, not claimed
- The first and second timing controls on the M1 Max and the M2 Max did not pass; the route trace is the evidence that no row leaks across chips. A timing control with more rounds or longer products could replace it.
- Vulkan device reopen after the runs, GPU firmware identity, and the loaded-library hash check of `scripts/mlx_provenance.py` were not recorded for these runs. The wheels carry their source commit in the version stamp, and each run printed the driver library hash (`libvulkan_asahi.so` sha256 starting 3546bcafe8ed3995).
- Decode speed and model-level numbers: none. These are dense GEMM cells at 4096 columns and 4096 inner size.
- bf16 `a @ b` direct rows beyond the existing route, f32 rows on the M2 Max, and the M1 Max rows below their existing floors (f16 and bf16 `a.T @ b`, f32 `a @ b.T`, all from m = 4096): not measured here.
- Raw `gemm-bench` logs for the k2s8 kernel comparison are in the chip directories; the M1 Max k2s8 sweep shows k2s8 21 to 35 % below k4s8 in every cell, so no M1 Max k2s8 row exists.

## Files
- `m1-g13g/`, `m1max-g13c/`, `m2max-g14c/`: raw per-round cell results (`h27-*`, `rerun-*`, `h31-*`, `h32-*`, `h23-*` JSON lines: median time, TFLOP/s, output hash per cell), the analysis outputs (`analysis-*`), the route-control verdicts, and the `gemm-bench` k2s8 logs.
- `family-and-sweep-summary.txt`: doctest and sweep lines for the three chips.
- `tools/`: the cell scripts, the analysis script, the route-trace scripts and the shape-sweep scripts.
