# mlx#4641 split-K class probe — quantized_matmul on the omarchy backend
# (upstream lane 1008b; upstream commit 494c8a6c93 fixed Metal qmm_splitk
# storing partial sums in the input dtype)

## Finding

The omarchy Vulkan backend has NO split-K quantized-matmul route. The fork's
selection rule (overlay/mlx/backend/omarchy/primitives.cpp, dispatch_qmm
route):

- M == 1 -> the QmmVec* family (vec/gemv): one workgroup per N-column group
  walks the FULL K in one dispatch (subgroup variant when the device has
  subgroup_size 32 + arithmetic ops, else the plain vec kernel).
- M > 1  -> the QmmTileFp* family (qmm_tile.comp, coopmat tiles over
  (M,N) with full-K tiles inside one dispatch).
- No kernel splits K across dispatches or accumulates partial sums in the
  input dtype: the only "partial" in the qmm shaders is k-tile edge
  bound-checking; no atomicAdd partial-sum path exists in qmm_*.comp.

The upstream #4641 numeric defect class therefore cannot occur on this
backend: both routes accumulate in fp32.

## Evidence (G14C, M2 jwm2-linux, release wheel 0.32.4.dev202610050725+5c15fba)

MLX_OMARCHY_TRACE_DISPATCH=1; wheel-era enum decode (compute.h at 5c15fbaea,
505 entries — the wheel predates today's 566-entry enum, so ids are decoded
against the wheel's own header):

- kernel 9  = CastBF16F32, kernel 10 = CastF32BF16 (2 per call, the bf16 x
  cast in and the bf16 out cast back)
- kernel 398 = QmmVecQ4WordSubgroupF32 — exactly the 7 M=1 calls (3 seeds +
  4 cross-path rows): the M=1 route, subgroup vec, fp32 accumulator
- kernel 310 = QmmTileF32 — exactly the 7 M>1 calls (3x M=8, 3x M=256 seeds
  + 1 cross-path M=256): the tile route, fp32 accumulator
- kernel 315 = QuantizeF32, kernel 150 = DequantF32 (reference prep)

Shapes: K=9728, N=2560, group_size 64, bits 4, w [N,K]; x bf16; reference =
dequantized weights in fp64 @ fp64 x.

## Numbers (maxabs / rel vs fp64 reference, 3 seeds each)

| M | maxabs | rel |
|---|---|---|
| 1 | 5.58e-3 - 6.67e-3 | 1.45e-3 - 1.92e-3 |
| 8 | 6.14e-3 - 7.46e-3 | 1.47e-3 - 1.91e-3 |
| 256 | 7.97e-3 - 9.01e-3 | 1.59e-3 - 2.01e-3 |

The error scale tracks bf16 input rounding at K=9728 (uniform across M) —
NOT the K-proportional blow-up a bf16 partial-sum store would show.

Cross-path agreement: rows of the M=256 (QmmTileF32) call vs fresh M=1
(QmmVecQ4WordSubgroupF32) calls on identical inputs: max diff 8.1e-6 -
1.19e-5 over rows 0/1/127/255 — fp32-output-quantization level. The two
routes agree; neither stores partials in bf16.

## Conclusion

- Route rule: M==1 -> QmmVecQ4WordSubgroupF32 (or QmmVecFp* when subgroups
  are absent), M>1 -> QmmTileF32 (bf16 x rides fp32 accumulate).
- No split-K path exists; #4641's defect class is absent on this backend.
  Nothing to fix; no follow-up.

Probe log: M2 ~/u1008b-probe/4641-splitk.log; queued as idle-guard entry
4641-splitk (MAXMIN 10, rerun-safe).
