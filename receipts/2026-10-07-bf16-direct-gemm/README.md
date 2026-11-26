# Direct bf16 GEMM route and bf16/f32 route rows on the M1 and the M1 Max

Date 2026-10-07. Lane matmul-gap. Notebook: MatmulGap H13 to H16 (private; acceptance rules registered before the data).

## Change
- `MatmulDirectBF16{Nn,Nt,Tn,Tt}`: the direct cooperative-matrix kernel (`shaders/matmul_coopmat_direct.comp`) on bf16 operands with an f32 accumulator. It needs `cooperative_matrix_bf16_8` (a driver with `VK_KHR_shader_bfloat16` cooperative matrices and a shader compiler that accepts `GL_EXT_bfloat16`). Without it, bf16 keeps `MatmulBF16Coopmat`.
- bf16 `a @ b.T` takes the direct kernel only through a route row: on G13C the plain kernel lost to the staged one (-7.6 to -10.6 % at 4096^3, H13 part A).
- Route rows added to `matmul_direct_select.h` (all with min n 4096):

| dtype, orientation | G13G (M1) | G13C (M1 Max) |
|---|---|---|
| bf16 `a.T @ b` | ws8, m >= 4096 | ws8, m >= 4096 |
| bf16 `a @ b.T` | k1, m >= 4096 | k4s8, m >= 512 |
| f32 `a @ b.T` | k4s8, m >= 4096 | k4s8, m >= 4096 |
| f32 `a @ b` | k4s8, m >= 4096 | none |
| f32 `a.T @ b` | ws8, m >= 4096 | none |

## Method
- Chips: M1 (T8103, G13G) and M1 Max (T6001, G13C), Omarchy, kernel 7.1.12-2-12.1-sep-ARCH. One GPU user at a time (`gpu-turn` on the M1 Max, `/tmp/m1-gpu.lock` on the M1), `MESA_SHADER_CACHE_DISABLE=1`.
- Driver: Honeykrisp with `VK_KHR_shader_bfloat16`, the same codegen as joshuaswarren/mesa-1 `honeykrisp-omarchy-v3` 6543eeb7df7. `libvulkan_asahi.so` sha256 `ace68aa69a17a407...` (M1 Max) and `d7d034154151f965...` (M1); the 16-hex prefix is in each round log.
- Three wheels from one source each (the version stamps the commit):
  - M = main `+ef70b8cc`
  - C1 = `+fd2e3848`: the bf16 route, the transposed-rhs exclusion, and the shared rows
  - C2 = `+27aa6738`: C1 plus the per-chip bf16 `a @ b.T` rows and the G13G f32 rows. This is what landed.
- Per chip: three rounds in the order M C1 C2 / C2 C1 M / M C1 C2. Each slot runs `metal_baseline.py` (the MLX cells in `tools/gemm-bench`) and `tools/rowcheck.py` (4096^3 on every row shape: output sha and median time). A final round runs `mlxcheck2` for each arm and `omarchy_matmul_family_tests` built at C2. The runner is `tools/run_h1316.sh`.
- Acceptance, registered before the data:
  - Bits: the C1 and C2 output hashes equal main on every cell.
  - Speed: a row lands only if it gains >= 5 % on every rep pair. bf16 `a @ b.T` rows must pass on both chips. The G13G-only f32 rows are judged against C1 by rowcheck.
  - No regression: every other `metal_baseline` cell must have a candidate median >= the main median - 3 %.

## Results
Bits. Outputs equal main in every arm, rep and chip:
- rowcheck: 7 of 7 cells.
- mlxcheck2: 21 of 21 cases for C1 and for C2.
- family tests at C2: 32 of 32 cases on both chips.
- `cooperative_matrix_bf16_8`: main reports none, C1 and C2 report 1.

`metal_baseline` TFLOP/s gain of C2 over main, per rep:

| cell (4096^3) | M1 (G13G) | M1 Max (G13C) |
|---|---|---|
| bf16 `a @ b` | +156.5 / +156.7 / +156.7 % | +124.5 / +124.6 / +124.6 % |
| bf16 `a @ b.T` | +82.2 / +82.6 / +81.8 % | +77.5 / +77.8 / +78.3 % |
| f32 `a @ b.T` | +114.0 / +191.8 / +209.2 % | +231.3 / +235.4 / +233.5 % |
| f32 `a @ b` | +21.4 / +25.1 / +26.1 % | -0.5 / -0.6 / +1.4 % (no G13C row) |

The main f32 `a @ b.T` value moved between reps on the M1 (0.657 / 0.486 / 0.459 TFLOP/s). The rowcheck time for that cell is stable: C2 is +127.6 to +129.4 % over main.

rowcheck speedup over C1, per rep:

| cell (4096^3) | M1 (G13G) | M1 Max (G13C) |
|---|---|---|
| bf16 `a @ b.T` | +80.8 / +80.7 / +80.8 % | +76.2 / +70.4 / +76.2 % |
| f32 `a @ b` | +27.2 / +23.1 / +27.4 % | +0.7 / -1.7 / +0.6 % |
| f32 `a.T @ b` | +74.3 / +77.3 / +75.1 % | -0.3 / -1.5 / +2.7 % |

rowcheck speedup of C2 over main for bf16 `a.T @ b`: +217.6 to +217.8 % (G13G), +192.9 to +194.1 % (G13C).

All other `metal_baseline` cells:
- C2 medians are within -0.3 % to +1.2 % of main.
- Exception: f16 `a @ b` on the M1 Max had a C2 median 4.3 % below main.
  - Its kernel did not change: the SPIR-V of `matmul_direct_f16_nn` is identical, and the AGX shaderdb of all 6 pipelines in an f16 `a @ b` run is identical (`jw16/nnprobe-shaderdb-*.txt`).
  - 12 fresh processes alternated main and C1 (`tools/run_nnprobe.sh`). Main gave 6.026 / 6.062 / 6.012 / 6.043 / 5.953 / 6.111 TFLOP/s and C1 gave 6.014 / 5.968 / 6.035 / 6.017 / 5.995 / 5.998 TFLOP/s, a -0.5 % difference in medians. These samples came from the script's stdout, which the notebook records.
  - Main alone spans 5.93 to 6.28 TFLOP/s in the three-arm run. The dip is `metal_baseline` variance on its first cell.

M1 rep 1 note: another lane's agent restarted python processes (no GPU, no lock) at 12:16:15 to 12:16:40Z. That window overlaps the M slot of rep 1 (from 12:16:06Z) and the start of the C1 slot (12:16:37Z). Every verdict above also holds on reps 2 and 3 alone.

## Data
- `jw16/` holds the M1 Max data, and `jwm1/` holds the M1 data.
- Per arm and rep: `h1316-mb-<arm>-r<rep>.jsonl` and `h1316-rc-<arm>-r<rep>.jsonl`.
- `h1316-mlxcheck2-<arm>.json`, the round logs (`h1316-<round>.log`), and `h1316-family-c2.log`.
- `tools/`: the runner and probes. `tools/analyze.py <chip dir>` prints the tables above.
