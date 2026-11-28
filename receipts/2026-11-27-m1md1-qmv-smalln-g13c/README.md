# M1MD-1 small-N QMV tile isolation (jw16, G13C M1 Max)

Row: close the 2.5x bandwidth gap on `QmmVecQ4MultiSubgroupBF16R2S8W2`
(32.7% ctx=512 / 29.4% ctx=2048 decode share, 125-133 GB/s vs 329 GB/s
lm_head, 108 disp/tok on 4B G13C).

## Change under test

`MLX_OMARCHY_QMM_SMALLN`: a decode multi dispatch whose members all have
n <= 4096 swaps the 16-column R2S8W2 tile for an 8-column tile
(`2x4w2` = 128-thread R2S4W2, `1x8w2` = 256-thread R1S8W2), doubling
workgroup count on the N=2560 o_proj/down_proj shapes (160 -> 320 WGs
on 32 cores). Per-column arithmetic unchanged; bit-identical by
construction and by the test_qmm_geometry cases.

## Per-shape isolation (run on jw16)

```
cd tools
MLX_OMARCHY_QMM_SMALLN=0     python3 gemv_smalln.py
MLX_OMARCHY_QMM_SMALLN=2x4w2 python3 gemv_smalln.py
MLX_OMARCHY_QMM_SMALLN=1x8w2 python3 gemv_smalln.py
```

Expected columns: us/member and effective weight-stream GB/s per shape
group (qkv trio, o_proj, down_proj, gate+up, lm_head reference).

## ABBA

5-pair Qwen3-4B decode ABBA at ctx=512 and ctx=2048 per
scripts/decode_bench.py convention: A = SMALLN unset, B = winning mode.
Require identical greedy token ids, matched before/after tok/s, and the
per-shape GB/s table above. If neither mode wins >= +0.5%, keep the
default off and file the negative receipt here.
