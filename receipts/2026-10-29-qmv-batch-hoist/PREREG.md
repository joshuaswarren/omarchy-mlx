# PREREG: 4-bit group-64 quantized matmul at 8 to 16 rows on Vulkan (v2, nothing run yet)

Written 2026-10-29 (about 13:45Z) before any measurement of this question. Supersedes the draft written at 11:40Z.

## Question

For a bf16 `[1, T, K]` pass, T = 8, 12 and 16, against one transposed 4-bit group-64 weight: is there a route that beats the shipped coopmat prefill tile by a margin that matters, on the 13-inch M1 (G13G), the 16-inch M1 Max (G13C) and the 14-inch M2 Max (G14C)?

## Known before the run

- The shipped tile route costs the same for 2 to 8 rows (13-inch M1: 0.62 to 0.68 ms on 2048x6144; 16-inch M1 Max: 0.52 to 0.56 ms), because it pads the rows to the tile. Not measured above 8 rows.
- Batch4 GEMV blocks (techmore PR 296, first head: ceil(T/4) dispatches): faster than the tile at 2 to 4 rows on the M1 and 2 to 7 rows on the M1 Max; slower beyond (M1 at 5 to 8, M1 Max at 8). The block cost grows with T, the tile cost does not, so the crossing is a function of the chip.
- mlx #4712 (Metal, FP quant formats, 6 to 16 rows) and trymirai/uzu `qmv_improvements` (register-MMA, M = 4 to 8) name the same weak spot; credit both in the receipt if an idea is borrowed.

## Candidates

- A: the existing Batch4 GEMV block route with the cap lifted from 4 to 16 (ceil(T/4) dispatches), built from the PR 296 head plus the cap change. No new shader.
- B (only if the stop rule below lets it start): a new kernel that pads the rows to 8, not to the coopmat tile: weights as the transposed operand, rows as the fragment dimension. Needs the device's cooperative-matrix shapes (an 8x8 or 8-row shape is NOT confirmed to exist on Honeykrisp; step 0 reads the device's shape list).

## Procedure

0. Read the cooperative-matrix property list of each chip (`vulkaninfo`-class query, no kernel).
1. One build with candidate A on each chip. Micro-bench `qmm-smallt-bench.py` extended to T = 2, 4, 6, 8, 12, 16 on the shapes 2048x6144, 2048x2048 and 2048x16384: flag unset (tile) against flag 16 (A), interleaved in one process, 3 rounds of 30 calls, quiet, boot id and `libmlx.so` digest printed, output difference printed.
2. Stop rule (fixed now, applied after step 1 on all three chips):
   - STOP, no kernel work: if at T = 8, 12 and 16 the tile route is within 1.25x of the 1-row GEMV time on the same shape on all three chips (nothing to recover), OR if A is not faster than the tile by at least 1.15x at any of T = 8, 12, 16 on any chip AND step 0 shows no 8-row cooperative-matrix shape (B has no basis).
   - CONTINUE to B only where A leaves a gap: the tile route costs more than 1.5x the 1-row GEMV time at T = 8 to 16 on some chip and A does not close it.
   - Any default change (A or B) additionally needs: output difference reported (GEMV summation order differs from the coopmat chain, so outputs are not byte-identical to the tile; decided only through the numerics gate on the parity-contract model), and an end-to-end win on a batch-8 to 16 workload (speculative or batched decode) on the chip it is enabled for.
3. Per-chip gating: a result on one chip is not carried to another; a per-part default (like the cap discussion on PR 296) is the form any win takes.

## Predictions (written before the run)

- Step 1: A beats the tile at T = 8 on the M1 Max (about 1.0x to 1.1x; the 8-row crossing was 0.94 to 0.95 with two blocks) and loses from T = 12 on both chips (three or four blocks); A never beats the tile on the M1 at T >= 8. On the M2 Max the crossing is unknown; my guess is between the two.
- Tile cost vs 1-row: I expect the tile at 8 to 16 rows to cost 2 to 3.5x the 1-row GEMV on the 2048x6144 shape (about 0.6 ms against 0.27 ms on the M1). That is above 1.5x, so the stop rule does not trigger on that clause.
- P(B reaches 1.3x over the tile at T = 8 to 16 on at least one chip, with identical-to-GEMV-order outputs): 25%. P(any default change merged this week): 10%.
- Not claimed: anything for models, prefill, rows above 16, or other chips.

## Amendment 1 (2026-10-29, about 14:55Z; written after the 16-inch M1 Max step-1 data above was seen, before any candidate C measurement)

Step-1 data on the 16-inch M1 Max (qmm16-16-inch-M1-Max-20261029T1359Z.log): the block route (A) beats the tile at 2 to 6 rows on all three shapes and loses at 12 to 16 rows on 2048x6144 and 2048x16384. The shipped batch kernel is parametrised by `QMM_VEC_BATCH` (4 today), so a wider block is one build flag, not a new kernel:

- Candidate C: the same batch-shared GEMV with `QMM_VEC_BATCH` = 8 and 16, one dispatch per 8 or 16 rows (one weight read per pass instead of one per 4 rows). Every row keeps the single-row chain (same code), so output is identical to A and to the 1-row kernel row for row.
- B (a new coopmat kernel with an 8-row tile) stays unstarted. The shipped coopmat kernel is already built on 8x8x8 matrices with `TILE_ROWS` 16 or 32, and the tile cost is flat in T, so a smaller row tile removes padding compute only, not the weight dequant and load per tile.
- Procedure change: one wheel carries blocks of 4 (A), 8 (C8) and 16 (C16), selected by a live-read env var `MLX_OMARCHY_QMV_SMALL_T_BLOCK` on top of `MLX_OMARCHY_QMV_SMALL_T`; the bench interleaves tile, A, C8, C16 in one process.
- Same stop rule and bar: a candidate must beat the tile by 1.15x or more at some T in 8..16 on a chip to continue; otherwise stop. Register pressure is the known risk (8 or 16 row accumulators per thread); I expect C8 to help at T = 8 only and C16 to lose to the tile.
- Prediction before the run: P(C8 beats the tile by 1.15x at T=8 on the 16-inch M1 Max) 40%; on the 13-inch M1 25%; P(C16 beats the tile at T=16 on any chip) 10%.

## Amendment 2 (2026-10-29, about 15:55Z; written after the 13-inch M1 data (qmm16c-13-inch-M1-20261029T1544Z.log) and before any candidate D measurement)

13-inch M1, per call median ms, tile vs the block route with 4, 8 and 16 rows per pass:
- Candidate C is dead on this chip: the 4-, 8- and 16-row passes cost the same within 2 percent at every T (b16/b4 between 0.983 and 1.021 over T = 2..16 on all three shapes). Reading the weights once instead of four times does nothing.
- The cost per added row is the same constant at every width: 0.091 ms/row (2048x6144), 0.032 (2048x2048), 0.238 (2048x16384), i.e. in proportion to the weight bytes, about one 1-row weight stream (about 69 GB/s) per extra row. The tile route is flat in T (T=16 minus T=2: -0.008, -0.006, -0.053 ms).
- The tile wins from T = 6 (mlp), 6 to 8 (proj), 4 (lm) upward on this chip; A (and C) win at T = 2 (1.6x to 1.8x) and T = 4 (1.05x to 1.24x).
- Reading of the numbers, not yet shown: the per-row cost is ALU work, not memory. `Q4_QUAD_ROWS` recomputes the nibble-to-float conversion (`Q4_NIB`: shift, and, or, subtract) for every row inside the row loop, so each row pays about four conversion ops per multiply-add. Hoisting the 8 conversions per weight word and column out of the row loop leaves the arithmetic of every row unchanged (the same values enter the same `fma` chain), so outputs stay bit-identical to the shipped batch kernel.

Candidate D: the batch kernel with the nibble conversions hoisted out of the row loop (`QMM_VEC_HOIST`), at 4, 8 and 16 rows per pass. `MLX_OMARCHY_QMV_SMALL_T_BLOCK` selects: 4 shipped Batch4, 5 Batch4 hoisted, 8 Batch8 hoisted, 16 Batch16 hoisted.
- Exactness requirement (checked before any speed is read): D output equals the shipped Batch4 kernel output bit for bit at T = 2..16 on every shape (the bench prints the difference), plus `omarchy_qmv_batch_tests` on the development machine.
- Bar: D beats the tile by 1.15x or more at some T in 8..16 on a chip, else stop; D beating the shipped Batch4 at T = 2..4 is a decode result in its own right (B = 2..4 decode batches use this kernel by default) and is reported separately with a decode A/B, not folded into the T = 8..16 claim.
- Predictions before the run: marginal cost per row falls 2x to 3x on the 13-inch M1 (P 60 percent); D beats the tile by 1.15x at T = 8 on the 13-inch M1 (P 35 percent), on the 16-inch M1 Max (P 45 percent); D beats the tile at T = 16 anywhere (P 5 percent); outputs bit-identical to Batch4 (P 95 percent).

## Amendment 3 (2026-10-29, about 21:15Z; written after the three-chip results of the hoist below and before any measurement of the per-quad form)

Result of the whole-word hoist (head `c1422078`), decode ms per step, shipped against hoisted, ids hash identical on every chip:

| chip | batch 2 | batch 3 | batch 4 |
| --- | --- | --- | --- |
| 16-inch M1 Max | 1.000x | 1.041x | 1.083x |
| 13-inch M1 | 1.081x | 1.163x | 1.203x |
| 14-inch M2 Max | 0.947x | 1.102x | 1.152x |

The one loss is batch 2 on the 14-inch M2 Max. A diagnosis with the batch route on and off on that machine shows route off equal (28.03 against 28.13 ms) and route on slower (28.02 against 29.06 ms), so the loss comes from the hoisted kernel. The kernel microbenchmark at 2 rows shows no slowdown there (1.03x, 0.99x, 1.08x), so the cause is not the ALU work per call.

Change under test: the hoist done per quad instead of per word. For each quad (four nibbles) the four conversions of each of the two columns are made once and reused by every row; the row loop moves inside the quad loop, and each row keeps its own `dot` and `input_sum` across the four quads of a step. Each (row, column) chain still adds the same products in the same order: bit-identical by the same argument.

Live values, counted from the source: whole-word hoist, 32 conversion floats live across the row loop on the 512-wide path (2 columns x 16 nibbles of two words), 16 on the narrow path, plus 3 row values; per-quad, 8 conversion floats live (2 columns x 4) on both paths, plus 12 row values (4 rows x (2 dot + 1 input sum)). Total 20 against 35 on the 512-wide path.

Prediction, written before the run:
- Cause: register pressure and the occupancy it limits on the 14-inch M2 Max, not ALU work (the hoist does fewer conversions than the shipped kernel at 2 rows). At batch 2 the kernel does little work per call, so a lower occupancy costs more than the saved conversions gain.
- 14-inch M2 Max, batch 2: the loss goes away (0.98x to 1.02x): P 55 percent. If the loss stays, register pressure is not the cause.
- Batch 4 with the per-quad form: 16-inch M1 Max stays at or above 1.08x: P 50 percent (the per-quad form reloads the activation words once per quad and keeps 12 row values live, so it may give back part of the gain); 13-inch M1 stays above 1.15x: P 80 percent; 14-inch M2 Max stays above 1.10x: P 70 percent.
- Output identical to the shipped kernel on every chip: P 95 percent.
- Merge rule, set before the run: the per-quad form merges only if it removes the M2 batch-2 loss and keeps at least 1.08x at batch 4 on both M1 parts with identical ids; otherwise the whole-word form merges with the M2 batch-2 loss stated.

### Outcome of Amendment 3 (2026-10-29, about 21:30Z; 14-inch M2 Max only, measured after the prediction above was written)

- Output identical: held (0 of 144 datasets differ from the per-row kernel in the shipped, whole-word and per-quad wheels; same combined hash).
- 14-inch M2 Max, batch 2: did not hold. Per-quad 0.970x of shipped, the same as the whole-word form (0.971x). Register pressure is not the cause.
- 14-inch M2 Max, batch 4: did not hold. Per-quad 1.076x of shipped, below the 1.10x predicted (whole-word 1.153x). Batch 3 fell to 0.987x (whole-word 1.101x).
- The per-quad form fails the merge rule on its first chip, so it was not run on the other two. The whole-word form merges with the M2 batch-2 loss stated.
