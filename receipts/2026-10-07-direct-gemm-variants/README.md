# Direct f16 GEMM route rows: `a @ b.T` k4s8 and `a.T @ b` ws8 on the M1 and the M1 Max

Date 2026-10-07. Lane matmul-gap. Notebook: MatmulGap H8 and H9 (private).

## Change
- `overlay/mlx/backend/omarchy/matmul_direct_select.h`: a chip- and shape-keyed table picks the direct coopmat kernel variant for f16 matmuls. A row matches on a device-name substring (`G13G`, `G13C`), the operand orientation, `m >= min_m`, `n >= min_n`, and `n >= tile_n / 2` (a subgroup's edge tile shifts back by `tile_n / 2`). Anything unmatched keeps the shipped route. The selection test is in `overlay/tests/omarchy/test_matmul_family.cpp`.
- Four rows and two new pipelines, both from `shaders/matmul_coopmat_direct.comp` with the build-time knobs from main 6384448db:
  - `matmul_direct_f16_nt_k4s8` (`-DB_T=1 -DK_UNROLL=4 -DSWIZZLE_ROWS=8`, tile_n 64) for `a @ b.T`, `m >= 512`, `n >= 4096`, on G13G and G13C.
  - `matmul_direct_f16_tn_ws8` (`-DA_T=1 -DWIDE_N=1 -DSWIZZLE_ROWS=8`, tile_n 128) for `a.T @ b`, `m, n >= 4096`, on G13G and G13C.
- `tools/gemm-bench/metal_baseline.py`: two Q4 prefill cells (Qwen3.8-2B gate and down projections, m = 512) with an output hash.

## Method
- Chips: M1 (T8103, G13G) and M1 Max (T6001, G13C), Omarchy, kernel 7.1.12-2-11.38-sep-ARCH. One GPU user at a time (lock), `MESA_SHADER_CACHE_DISABLE`.
- Driver: joshuaswarren/mesa-1 6669059ac5d (`honeykrisp-omarchy-v3` 88c7241d909 plus default-off A/B switches), sha256 27e6fca16be5... on both chips.
- `gemm-bench` (`tools/gemm-bench`): one process per pass runs every side on every cell, two passes. Cells: f16 `nn` 4096^3, `nt` 4096^3, `nt` 512x4096x4096, `tn` 4096^3, `nt` 512x4096x4104. Sides: k1 (shipped), K_UNROLL 2/4, SWIZZLE_ROWS 4/8, k4s8, k4s4, WIDE_N, WIDE_N + SWIZZLE_ROWS 8.
- Acceptance (pre-registered): `out_fnv` equal to k1 on every cell; a side wins an orientation when it gains >= 5 % over k1 on a cell in both passes on both chips, with no cell of that orientation below k1 by more than 3 %.

## Results
Outputs: every side gives one `out_fnv` per cell on both chips (`nn4096` 6a7d8ec7703f6a36, `nt4096` 4ad92df0d8a37d87, `nt512` 1c4025366fdc06a4). Logs: `jwm1/run_h8.log` (M1), `jw16/run_h8.log` (M1 Max).

Gain over k1, the lower of the two passes:

| side | orientation | M1 (G13G) | M1 Max (G13C) | rule |
|---|---|---|---|---|
| k4s8 | `nt` 4096 / 512 / 512 k4104 | +18.5 / +15.7 / +3.9 % | +189.0 / +130.7 / +97.1 % | win, landed |
| ws8 | `tn` 4096 | +6.0 % | +70.3 % | win, landed |
| ws8 | `nn` 4096 | +2.1 % | +20.8 % | one chip only, not landed |
| k4 | `nt` | +9.1 / +5.6 / -3.1 % | +182.5 / +124.7 / +100.4 % | k4s8 is better on both |
| s4, s8 | all | -16 to -52 % | -5 to -23 % | lose |

k1 baselines (TFLOP/s, pass 1): M1 `nn` 1.950, `nt4096` 1.493, `tn` 1.833; M1 Max `nn` 5.772, `nt4096` 2.194, `tn` 4.238.

## Landing check (MLX level, both chips)
Wheels: base 0.32.4.dev202610070341+e034673d (empty route table, shipped routes) and land 0.32.4.dev202610070516+60388f09 (this branch at 60388f09e; 6141ae16452 adds only the test below). Same driver as above, cache off, `metal_baseline.py` alternated base / land, 2 reps, one lock holder. Logs: `jw16/verify_land.log`, `jw16/land-mlx-*.jsonl`, `jwm1/verify_land.log`, `jwm1/land-mlx-*.jsonl`.
- `omarchy_matmul_family_tests`: M1 Max 28/28 (binary at 60388f09e), M1 29/29 (binary at 6141ae16452, with the new case "the shipped rows apply only on the measured chips").
- `mlxcheck2`: 21/21 hashes equal between base and land on both chips (f16, bf16, f32 x seven shapes).
- f16 cells, TFLOP/s, base reps -> land reps:

| cell | M1 (G13G) | M1 Max (G13C) |
|---|---|---|
| `a @ b.T` 4096^3 | 1.521 / 1.526 -> 1.752 / 1.751 (+15 %) | 2.511 / 2.633 -> 6.432 / 6.357 (+141..156 %) |
| `a @ b.T` 512x4096x4096 | 1.601 / 1.594 -> 1.671 / 1.615 (+1..5 %) | 4.181 / 4.067 -> 5.360 / 5.206 (+25..32 %) |
| `a @ b.T` 512x4096x4104 | 1.518 / 1.525 -> 1.625 / 1.623 (+6..7 %) | 3.957 / 3.996 -> 5.136 / 5.070 (+27..30 %) |
| `a.T @ b` 4096^3 | 1.817 / 1.822 -> 1.949 / 1.950 (+7 %) | 4.821 / 5.114 -> 7.110 / 7.059 (+38..48 %) |
| `a @ b` 4096^3 (no row) | 1.941 / 1.941 -> 1.942 / 1.941 | 5.932 / 6.140 -> 6.068 / 5.925 |

- Cells the rows cannot reach (bf16, f32, Q4 prefill, transpose) stay within -1.8..+1.9 %, with one exception: M1 `f32` `a @ b.T` 4096^3 land rep 1 read 0.416 against base 0.521 / 0.528 (land rep 2: 0.533). The rows match f16 only, so this path is unchanged. The cell is noisy on the M1: in the H8 run, two identical configurations measured 0.566 and 0.459.
- Other chips: a row matches a device-name substring, so `Apple M2 Max (G14C B1)` keeps the shipped route (pinned by the new test case). The on-device G14C neutrality run is pre-registered (notebook MatmulGap H12) for the next M2 window.
- Merge: the tested commits (branch `agent/matmul-land` 60388f09e and 6141ae16452) were rebased onto five later main commits before the merge, one of which (a092c24ae) restores allocation guards in `dispatch_matmul`. The rebase applied without conflicts and the changed files pass an x86 syntax check. The merged tree has not run on a GPU yet; its family tests run first in the next M1 Max window (with H11).

## Not landed
- G13C-only `nn` ws8: the rule needs a win on both chips.
- `MLX_OMARCHY_MATMUL_PRETRANSPOSE_B` (copy `b` to row-major, then run the `nn` kernel): the `nt` rows beat it. On the M1 Max, k4s8 runs 6.49-6.57 TFLOP/s against 5.79-5.87 for the pre-transposed kernel without its copy. On the M1, MLX-level pre-transposition with the copy costs `nt512` 24 % and `nt512k4104` 19 %.
- `AGX_HWMAT_COLPAIR` (driver): outputs equal on 60/60 cells, but k1 `a @ b.T` gains under 5 % on the M1, and the landed k4s8 side loses 3-20 % with it on both chips.
- Causal flash prefill (`MLX_OMARCHY_SDPA_FLASH_CAUSAL`): correct (8/8 doctest cases on the M1; greedy tokens equal except one accepted near-tie at 4096), but slower. Qwen3.8-2B prefill, composed to flash: M1 Max 1453 to 1787 ms at 2048 tokens and 3019 to 4454 ms at 4096; M1 +3 % at 2048 and 4096.
