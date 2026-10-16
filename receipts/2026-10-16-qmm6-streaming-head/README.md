# Streaming decode for 6-bit affine quantized_matmul on the Muse Spark Open lm_head

The Muse Spark Open `lm_head` is 6-bit, group 64, N 202048, K 8192. The shipped route decodes it one code per lane step and sends two or more rows to the 16x16 tile. This change adds a streaming kernel for 1 to 32 rows (see the PR text for the gate). The figures below are cost and numerics on random weights. They are not model speed.

## What ran

- Chip: 14-inch M2 Max. Boot `7dcada54`, kernel `7.1.12-2-20261013.18-sep-ARCH`, performance governor, the installed omarchy Vulkan driver (library sha256 prefix `40282811c6e14ef9`).
- Product wheel: branch `w7b/qmm6-head`, commit `6f26e88ef225074c6d7efeb63ea70d8b378935c1`.
- Wheel sha256: `78ea5f72483e7b9e7a680a3136f967d77389adb33c9e7bed372fb15f2f6b58ce`.
- The loaded libmlx has sha256 prefix `61431e6d22f493f6`. The harness exits 4 if it differs.
- Harness: `m2_qmm6b.py` (sha256 prefix `c66e60757991ec26`), one process per arm.
- Three arms ran in this order in one guard ticket. The ticket ran 07:33:34Z to 07:35:14Z, exited 0, and had 0 GPU timeouts.
- `product-new1.txt` is the default dispatch. `product-old.txt` sets `MLX_OMARCHY_NO_QMM6_STREAM=1`. `product-new2.txt` is the default dispatch again.
- Shape: N 202048, K 8192, group 64, 6-bit affine, transposed.
- Weights are random normal, quantized on the device in 6314-row chunks.
- Cells are bf16 and f16 at rows 1, 2, 4, 8, 17, 32. Each cell is the median of 20 calls. One call is one `mx.eval` of the matmul.

## Rule fixed before the run

- Reference: the harness decodes the raw packed words itself. It checks that decode against `mx.dequantize` on the CPU stream, with scales and biases passed as f32. The maximum relative difference must be at most 1e-3. It then contracts in float64 on 256 sampled output rows, using the values the device holds.
- Numerics: every output (|ref| at least 0.05 sigma) within 1.5 steps of the output dtype, none beyond 2.5 steps.
- Route: the kernel ids printed per cell must contain the expected streaming id in both streaming arms. The shipped arm must show none.
- The `CastF32BF16` or `CastF32F16` id (10 or 8) before each 16-bit cell is the harness casting x on first use. It is also in the shipped arm. It is not in the timing loop.
- Timing bar, bf16: the streaming median is at most 0.50 times the shipped median at every row count, in both streaming arms.
- Where the two streaming arms differ by more than 10 percent, the larger one decides.

## Result

Numerics: all 36 cells (3 arms, 2 dtypes, 6 row counts) are finite. No output is beyond 1.5 or 2.5 steps. The largest error is 0.50 step in the streaming arms. It is 0.53 step in the shipped arm at f16 with 17 rows. The host decode matches `mx.dequantize` with a maximum relative difference of 0.

Route: bf16 1 row `QmmVec6C4R1BF16` (616), 2 to 32 rows `QmmVec6C1R8BF16` (617); f16 614 and 615. Shipped: `QmmVecSubgroup` (319 bf16, 318 f16) at 1 row, `QmmTile` (322, 321) at 2 or more.

Median ms per call:

| rows | shipped bf16 (ms) | streaming, first run (ms) | streaming, after the shipped arm (ms) | ratio first | ratio second | ids shipped / streaming |
|---|---|---|---|---|---|---|
| 1 | 40.80 | 4.11 | 4.09 | 0.101 | 0.100 | 319 / 616 |
| 2 | 89.81 | 13.77 | 13.78 | 0.153 | 0.153 | 322 / 617 |
| 4 | 89.74 | 14.16 | 14.16 | 0.158 | 0.158 | 322 / 617 |
| 8 | 89.62 | 14.53 | 14.57 | 0.162 | 0.163 | 322 / 617 |
| 17 | 180.32 | 42.37 | 46.43 | 0.235 | 0.257 | 322 / 617 |
| 32 | 196.68 | 57.51 | 67.15 | 0.292 | 0.341 | 322 / 617 |

Ratios are streaming over shipped. Lower is faster.

| rows | shipped f16 (ms) | streaming, first run (ms) | streaming, after the shipped arm (ms) | ratio first | ratio second | ids shipped / streaming |
|---|---|---|---|---|---|---|
| 1 | 39.12 | 4.22 | 4.07 | 0.108 | 0.104 | 318 / 614 |
| 2 | 89.41 | 11.20 | 11.18 | 0.125 | 0.125 | 321 / 615 |
| 4 | 92.20 | 11.62 | 11.62 | 0.126 | 0.126 | 321 / 615 |
| 8 | 96.92 | 14.14 | 14.10 | 0.146 | 0.145 | 321 / 615 |
| 17 | 200.42 | 38.25 | 38.40 | 0.191 | 0.192 | 321 / 615 |
| 32 | 213.70 | 54.87 | 54.98 | 0.257 | 0.257 | 321 / 615 |

Same columns for f16. The bar is judged on bf16 only.

The bar holds at every row count. The two streaming arms agree within 1 percent up to 8 rows. At bf16 with 17 and 32 rows the second run is 9.6 and 16.8 percent slower than the first. The 32-row cell is therefore order-affected, and its second-run ratio (0.341) decides it. The cause of the spread is not measured.

## Earlier ticket on the measurement wheel

`variants-old.txt` and `variants-new.txt` come from a wheel of commit `c4f6684`. It held six row-and-column variants behind an environment switch. The default pair there is the pair this change ships. In that ticket the same kernel (id 624 there, 617 here) measured 13.77 / 14.53 / 42.39 ms at 2 / 8 / 17 rows as the first run of its arm. As a later explicit variant it measured 20.06 / 21.02 / 62.34 ms. That gap did not recur in the product ticket. No ranking between the variants at 2 to 17 rows is claimed.

## Not measured

Real weights, model speed, bit widths other than 6, f32, other chips, other shapes.

## Reproduce

On a device build, with `TREE` the unpacked wheel:

```
PYTHONPATH=$TREE python m2_qmm6b.py --variants d --rows 1,2,4,8,17,32 --expect-libmlx 61431e6d22f493f6
MLX_OMARCHY_NO_QMM6_STREAM=1 PYTHONPATH=$TREE python m2_qmm6b.py --variants d --rows 1,2,4,8,17,32 --expect-libmlx 61431e6d22f493f6
```
