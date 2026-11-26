# mxfp4 gather_qmm: subgroup decode at the gpt-oss-20b expert shape

The routed experts of gpt-oss-20b are mxfp4: group 32, one e8m0 scale byte per group, e2m1 nibbles. Before this change they reached only the scalar `gather_qmm` kernel, one thread per output with a serial K loop. The affine subgroup route and the sorted-expert tile route are both gated on `!fp_mode`.

This change adds an `FP_MODE` branch to `gather_qmm_sub.comp`. It uses the scalar kernel's own decode helpers, so each product is the same. Only the fp32 accumulation order changes. New kernels: `GatherQmmNbSubFpF32`, `GatherQmmNbSubFpF16`, `GatherQmmNbSubFpBF16`. They serve transposed mxfp4 at one row (`m == 1`, which includes the sorted prefill rows). nvfp4 and mxfp8 stay on the scalar kernel. `MLX_OMARCHY_GATHER_QMM_SUB=0` turns the route off.

Random weights. This measures cost and numerics at one shape. It is not a model speed and not a model output.

## What ran

- Chip: 14-inch M2 Max. Boot `7dcada54`, kernel `7.1.12-2-20261013.18-sep-ARCH`, performance governor.
- Wheel: branch `agent/w6z-gather-qmm-mxfp4-sub`, commit `2f8fc4034594abf4d88a31eb4184e0f64e4267ef`, local version `0.32.4.dev202610162253+2f8fc403`.
- Wheel sha256: `1d77ffcd8f85d648406902cccb0ca999ac4c6a2e3b3a4f85d955f16e8c9ab9e8`. The loaded libmlx has sha256 `a1b8ffa54ea3688ff6aa2e8923a903cce41648e7301721ee8a03a678198bd11f`.
- Shape: 32 experts, K 2880, N 2880, top-4 per token. Cells are bf16 and f16, at `index_count` 4 (one token) and 64 (a sorted prefill block, x passed as `[B, 1, K]`).
- Each cell runs the default route and the scalar route in one process. Both are checked against an fp64 numpy reference on 128 output columns. Cost is the median of 7 calls after a warm-up.
- A second run with the dispatch trace on reads the kernel id of the default route in each cell.
- Files for the first run: `gptoss_fp_sub_gpu.py`, `m2-fpsub-gpu.sh`, `m2-run.log`.

## Rule fixed before the run

- Numerics: error at most 8e-3 of the largest reference value for bf16, and 2e-3 for f16.
- Route: the default route dispatches 620 (bf16) or 619 (f16). The scalar route dispatches 400 or 399.
- Cost bar: the default route is at least 3x faster than the scalar route at `index_count` 4.

## Result

Numerics: all eight cells pass. The default and scalar routes have the same maximum error in every cell. That error is the output rounding of the dtype. The largest error against the largest reference value is 15.25 of 5606 for bf16 and 1.92 of 6974 for f16.

Route: bf16 dispatches 620 and f16 dispatches 619 in all four cells, one dispatch per call.

| dtype | index_count | default ms (min to max) | scalar ms (min to max) | speedup |
|---|---|---|---|---|
| bf16 | 4 | 1.815 (0.992 to 1.892) | 3.893 (3.797 to 4.886) | 2.14x |
| bf16 | 64 | 8.262 (8.249 to 8.287) | 33.185 (33.157 to 33.297) | 4.02x |
| f16 | 4 | 0.755 (0.739 to 0.863) | 3.812 (3.781 to 3.852) | 5.05x |
| f16 | 64 | 8.084 (8.072 to 8.197) | 33.173 (33.163 to 33.226) | 4.10x |

The cost bar holds for f16 at 5.05x. It did not hold for bf16 in the first run, at 2.14x. The bf16 one-token cell is the first cell that run measures, and its default median sat far above its minimum.

## Repeat with the bf16 cell run twice

The same wheel, chip and boot, a second ticket, with the cell order bf16, f16, bf16 and every per-call time printed. Files: `gptoss_fp_sub_gpu_repeat.py`, `m2-fpsub-gpu2.sh`, `m2-run-repeat.log`. The weights and x are drawn again, so the reference maxima differ from the first run.

| dtype | index_count | default ms (min to max) | scalar ms (min to max) | speedup |
|---|---|---|---|---|
| bf16, first cell | 4 | 1.854 (0.993 to 1.890) | 3.810 (3.784 to 4.147) | 2.06x |
| bf16, first cell | 64 | 8.231 (8.224 to 8.293) | 33.186 (33.149 to 33.228) | 4.03x |
| f16 | 4 | 0.749 (0.738 to 0.752) | 3.785 (3.779 to 3.800) | 5.06x |
| f16 | 64 | 8.079 (8.070 to 8.102) | 33.201 (33.172 to 33.220) | 4.11x |
| bf16, second pass | 4 | 0.760 (0.758 to 0.785) | 3.763 (3.729 to 3.781) | 4.95x |
| bf16, second pass | 64 | 8.249 (8.236 to 8.259) | 33.172 (33.159 to 33.192) | 4.02x |

The seven default times of the first bf16 cell are 1.890, 1.859, 1.853, 1.854, 1.858, 1.204 and 0.993 ms. They fall during the cell and end near the second-pass value. The scalar route does not change over the same cell. The same cell run last gives 0.760 ms. So the first-run bf16 figure is a warm-up effect of the first measured cell. In steady state bf16 and f16 cost the same at one token.

The cost bar holds for both dtypes at steady state: 5.06x for f16 and 4.95x for bf16. The first measured cell of a run is slower. All numerics pass in both runs, with the same error for both routes. The traced run shows kernel 620 for every bf16 cell and 619 for every f16 cell.

Arithmetic, not a measurement of the model: 24 layers times three expert matmuls is 72 calls per token. At the scalar rate of 3.8 to 3.9 ms that is about 275 to 280 ms. At the steady default rate of 0.75 to 0.76 ms it is about 54 to 55 ms, for both dtypes.

The scalar gather costs several times more on the 16-inch M1 Max (see the PR 190 receipt), so these M2 figures say nothing about that machine's token time.

## Environment and provenance

| Item | Value |
|---|---|
| Packages | `mesa 1:26.2.4-1`, `vulkan-asahi-omarchy 1:26.2.4.git69bde15340-2`, `linux-aurora 7.1.12.aurora2-20261013.18`, `asahi-fwextract 0.8.4-1` |
| Vulkan device | Apple M2 Max (G14C B1), Honeykrisp, driver `Mesa 26.3.0-devel (git-69bde15340)`, Vulkan API 1.4.362 |
| Provenance | `scripts/mlx_provenance.py --expect-wheel` on the unpacked wheel: `verified` is `match`, and `mx.__version__` equals the wheel version `0.32.4.dev202610162253+2f8fc403` |
| Device reopen | a new process opened the device and computed a small array on it: `Device(gpu, 0)`, values `[2.0, 2.0, 2.0, 2.0]` |
| Firmware | read through the `asahi-fwextract` package version above. No separate firmware version string was captured |

The metadata run is `m2-fpsub-meta.sh` with `m2-meta.log`. It ran after the numerics runs, on the same boot and the same unpacked wheel.

## Exact commands

Each run used the wheel unpacked into a directory `TREE`:

```sh
PYTHONPATH=$TREE python gptoss_fp_sub_gpu.py
MLX_OMARCHY_TRACE_DISPATCH=1 PYTHONPATH=$TREE python gptoss_fp_sub_gpu.py
```

The first command gave the numerics and the times. The second gave the kernel ids. The repeat used `gptoss_fp_sub_gpu_repeat.py` in the same two ways. The scripts here have the working-directory paths of the run replaced by `$WORK` and `$PYENV`. Nothing else differs. The run scripts had these sha256 prefixes: first run `8c2556177c9e2310`, repeat `3cbae27b846753bd`.

## Timing procedure

The call times are medians of seven calls after one warm-up, in one process, with the performance governor and no thermal control. They are cost figures. The first cell of a run is slower than later cells, as the repeat shows.

## Kernel ids

The ids in this receipt belong to the wheel `+2f8fc403`. Other branches also append kernels at the end of the enum, so the ids can shift after a merge. Match kernels by name: `GatherQmmNbSubFpBF16` and `GatherQmmNbSubFpF16`.

## Not measured

The real model, the real weights, token output, the 16-inch M1 Max and the 13-inch M1, nvfp4 and mxfp8, and any shape other than 32 experts with K 2880 and N 2880.
