# 1008(b) API check: `global_scale` in the locked MLX 0.32.3 tree

Checked at .work/mlx after prepare-mlx.sh (pin 9c3d35571, mlx.lock of
origin/main, 2026-10-08):

- `mx.quantized_matmul`: NO `global_scale` argument (ops.h:1591-1600 —
  x, w, scales, biases, transpose, group_size, bits, mode, stream).
- `mx.gather_qmm`: HAS `global_scale` (ops.h:1642-1655, optional array
  before `sorted_indices`).
- `mx.quantize` / `mx.dequantize`: HAVE `global_scale`
  (ops.h:1604-1620).
- `mx.qqmm`: HAS `global_scale_x` / `global_scale_w` (ops.h:1629-1638).
- C++ omarchy backend: `global_scale` is handled on the Vulkan side —
  dispatch_fp_quantize binds it (primitives.cpp:2866-2898), and the
  qqmm/quantized path carries per-tensor scale bindings with a flags
  bit (:3014-3023).

Conclusion for #4458 (mixed 4/8-bit gather_qmm): the missing piece in
0.32.3 is NOT `global_scale` on gather_qmm — it is there. #4458's
substance is per-GROUP mixed widths (4-bit and 8-bit groups in ONE
weight tensor) which lands in the QuantizedMatmul/GatherQMM primitive
and the omarchy Vulkan gather_qmm kernels (dispatch at
primitives.cpp:3340-3440: the subgroup/scalar selection reads a single
`bits` and single `group_size`). A backport would touch:

- upstream ops.cpp mixed-width validation + packed-payload layout
  (patch size ~100-200 lines with tests),
- the omarchy GatherQmm Vulkan kernels for mixed-width word unpacking
  (the scalar kernel's packed-word read assumes one width; the subgroup
  kernel's per-lane dequant table is 4-bit-only) — the same per-layout
  parity-proof bar as the gather_qmm_sub widening audit
  (gather-qmm-sub-coverage.md).

Added to the backport candidate list in this receipt (queued behind
#4635; not a quick backport).
