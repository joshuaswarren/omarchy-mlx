# G412 — G16S small-M qmm prefill A/B packet (staged for J616s)

Harness verified on the x86 dev CT (2026-11-22, CPU mlx): the sweep tool
runs both arms end to end and the identity path holds — the single-shape
smoke (M=23, N=K=4096, q4-g64 transpose) produced the identical digest
`7a444d3cac5a2a42` under `MLX_OMARCHY_QMM_COOPMAT_ROWS=32` and `=16`
(tools/identity_smoke.py; CPU quantized_matmul is arm-agnostic by
construction, so this checks the mechanics, not the Vulkan dispatch — the
real gate is `ALL_MATCH` on the M4).

Card G412 (Main, 2026-11-22). The G401 audit's conclusion: the M4 Pro kit's
own workload (prompt_tokens 23) lands in the small-M arm of the G4+LdsPad
qmm prefill route — below the 64-row gate (M >= 128) and above the M <= 16
16-row arm, so it runs the 32-row tile with 9 of 32 rows dead. Which lever
wins (a 16-row small-M arm vs a dispatch-count change) is a measurement
question; this packet measures it. It follows the tile64 receipt's method
(isolated quantized_matmul per shape, bit-identity gates, interleaved e2e)
at the skinny-M shapes.

Hardware: one Apple M4 Pro (G16S B1) J616s with the shipped omarchy driver.
No two-machine setup; everything runs on the one Mac with the mlx-omarchy
release venv.

## One-paste packet

```sh
REPO=~/src/mlx-omarchy   # or any checkout of this repo
KIT=$REPO/receipts/2026-11-22-g412-small-m-ab
PY=~/.local/share/mlx-omarchy/venv/bin/python
cd "$KIT/tools"
MESA_SHADER_CACHE_DISABLE=true MLX_OMARCHY_QMM_COOPMAT_ROWS=32 $PY small_m_sweep.py | tee small-m-32.txt
MESA_SHADER_CACHE_DISABLE=true MLX_OMARCHY_QMM_COOPMAT_ROWS=16 $PY small_m_sweep.py | tee small-m-16.txt
# bit-identity gate: the sha column must match row for row between the two files
diff <(awk '{print $2,$3,$4,$6}' small-m-32.txt) <(awk '{print $2,$3,$4,$6}' small-m-16.txt) && echo ALL_MATCH
# timing comparison for the default decision
paste small-m-32.txt small-m-16.txt
```

Pass conditions:

1. `ALL_MATCH` — every arm's 15 digests identical (bit-identity gate; the
   pick must never move outputs).
2. The `us` columns are the decision input: if the 16-row arm is faster at
   M in 17..31 by a margin outside run-to-run noise, the small-M arm becomes
   the selector default for G16S; if not, the audit's dispatch-count lever
   is measured next and the 32-row default stands.
3. Record the wheel sha, driver, and host line (`system_profiler SPHardwareDataType
   | grep -E "Chip|Model"`) at the top of the receipt file.

## e2e confirmation (only if the isolated rows favor 16)

Reuse the tile64 receipt's e2e method with a 23-token prompt
(`receipts/2026-11-15-m4-pro-g16s-qmm-bf16w/tools/run_prefill.py`,
`--targets 23 --max-tokens 32`, 5 interleaved runs per arm); the greedy
32-token sha must be identical across arms.

## Why these shapes

M = 17, 20, 23, 26, 31 covers the arm boundary the audit names (17..31 is
where M is too big for the 16-row arm and too small to fill a 32-row tile);
N/K follow the kit model's qmm prefill GEMMs (mlp up/down and attention
projections at K = 4096). The M = 23 x N = 12288 row is the attention
projection shape; N = 1024 is the smallest grid (the occupancy step-down's
own regime).
