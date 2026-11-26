// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

#pragma once

#include <vulkan/vulkan.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <mutex>
#include <span>
#include <string>
#include <unordered_map>
namespace mlx::core::omarchy {

inline constexpr uint32_t kComputeThreadsPerGroup = 256;
inline constexpr uint32_t kMaxComputeGroupCountX = 65535;
// Vulkan's guaranteed floor for storage-buffer descriptors in one compute
// descriptor set layout. Any kernel may assume this many binding slots.
inline constexpr uint32_t kComputeBindingFloor = 4;
// Binding slots the backend wants when every shipped kernel gets its full
// workspace (multi-index scatter needs five). The live budget is fixed once
// per device at initialization: min(kComputeBindingBudget, the device's
// reported storage-buffer descriptor limits), and kernels needing more than
// that budget refuse by name instead of dispatching. The spec floor is why
// the pre-2026-09-02 four-slot constant was portable, not a device ceiling:
// real drivers report orders of magnitude more.
// Nineteen slots fit the widest kernel today: the multi-weight decode
// GEMV binds x plus, per weight, packed words, scales, biases, the
// output, an Add addend, and the Add output (kQmmVecMultiBindings).
// The triple-index scatter needs six.
inline constexpr uint32_t kComputeBindingBudget = 19;
// Bindings of the QmmVecQ4Multi kernels and their per-weight stride.
inline constexpr uint32_t kQmmVecMultiWeights = 3;
inline constexpr uint32_t kQmmVecMultiBindingsPerWeight = 6;
inline constexpr uint32_t kQmmVecMultiBindings =
    1 + kQmmVecMultiWeights * kQmmVecMultiBindingsPerWeight;

constexpr uint32_t compute_dispatch_group_count(uint32_t count) {
  if (count == 0) {
    return 0;
  }
  uint32_t groups = (count - 1) / kComputeThreadsPerGroup + 1;
  return groups < kMaxComputeGroupCountX ? groups : kMaxComputeGroupCountX;
}

constexpr bool compute_index_span_fits(uint64_t offset, uint64_t count) {
  constexpr uint64_t max_index = std::numeric_limits<uint32_t>::max();
  return count <= max_index && offset <= max_index &&
      (count == 0 || count - 1 <= max_index - offset);
}

// Clamp the final advance so a near-UINT32_MAX count cannot wrap.
inline constexpr uint32_t kLogicalChunkElements =
    kMaxComputeGroupCountX * kComputeThreadsPerGroup;

constexpr uint32_t next_logical_chunk_end(uint32_t first, uint32_t count) {
  const uint32_t remaining = count - first;
  const uint32_t chunk =
      remaining < kLogicalChunkElements ? remaining : kLogicalChunkElements;
  return first + chunk;
}

enum class ComputeKernel : uint16_t {
  ElementwiseF32,
  ElementwiseF16,
  ElementwiseBF16,
  CastF16F32,
  CastBoolF32,
  CastBoolI32,
  CastBoolF16,
  CastBoolBF16,
  CastF32F16,
  CastBF16F32,
  CastF32BF16,
  CastBF16F16,
  CastF16BF16,
  CastI32F32,
  CastU32F32,
  CastF32I32,
  CastI32F16,
  CastF16I32,
  CastI32BF16,
  CastBF16I32,
  ReduceF32,
  ReduceF16,
  ReduceBF16,
  MatmulF32,
  MatmulF32Coopmat,
  MatmulF16,
  MatmulBF16,
  // Dense single-row GEMV (decode shape); see shaders/matmul_vec.comp.
  MatmulVecF32,
  MatmulVecF16,
  MatmulVecBF16,
  FillF32,
  FillF16,
  FillBF16,
  SoftmaxF32,
  SoftmaxF16,
  SoftmaxBF16,
  LogSumExpF32,
  LogSumExpF16,
  LogSumExpBF16,
  SelectF32,
  SelectF16,
  SelectBF16,
  SelectI32,
  SelectBool,
  SelectComplex64,
  CompareF32,
  CompareF16,
  CompareBF16,
  CompareI32,
  CompareU32,
  CompareI64,
  CompareComplex,
  LogicalOrBool,
  CompareBool,
  CopyGeneralF32,
  CopyGeneralF16,
  SliceUpdatePairF16,
  CopyGeneralBF16,
  CopyGeneralU32,
  CopyGeneralBool,
  CopyGeneralU8,
  CopyGeneralU16,
  CopyGeneralU64,
  ArgReduceF32,
  ArgReduceF16,
  ArgReduceBF16,
  ArgReduceI8,
  ArgReduceU8,
  ArgReduceI16,
  ArgReduceU16,
  ArgReduceI32,
  ArgReduceU32,
  ArgReduceI64,
  ArgReduceU64,
  ArangeF32,
  ArangeF16,
  ArangeBF16,
  ArangeI32,
  ArangeU32,
  ArangeI64,
  ArangeU64,
  SortF32,
  SortF16,
  SortBF16,
  SortC64,
  ArgSortF32,
  ArgSortF16,
  ArgSortBF16,
  ArgSortC64,
  SortI32,
  SortU32,
  SortI8,
  SortU8,
  SortI16,
  SortU16,
  ArgSortI32,
  ArgSortU32,
  ArgSortI8,
  ArgSortU8,
  ArgSortI16,
  ArgSortU16,
  RandomBitsU32,
  ElementwiseI32,
  ElementwiseU32,
  ScanF32,
  ScanF16,
  ScanBF16,
  SearchSortedF32,
  SearchSortedF16,
  SearchSortedBF16,
  SearchSortedI32,
  SearchSortedU32,
  ReduceGeneralF32,
  ReduceGeneralF16,
  ReduceGeneralBF16,
  ReduceGeneralI32,
  ReduceGeneralU32,
  ReduceGeneralI8,
  ReduceGeneralU8,
  ReduceGeneralI16,
  ReduceGeneralU16,
  ReduceGeneralI64,
  ReduceGeneralU64,
  ReduceGeneralComplex,
  AnyAllF32,
  AnyAllF16,
  AnyAllBF16,
  AnyAllI32,
  AnyAllU32,
  AnyAllBool,
  ScanGeneralF32,
  ScanGeneralF16,
  ScanGeneralBF16,
  ScanGeneralI32,
  ScanGeneralU32,
  ScanGeneralBool,
  ScanGeneralI8,
  ScanGeneralU8,
  ScanGeneralI16,
  ScanGeneralU16,
  ScanGeneralI64,
  ScanGeneralU64,
  ScanGeneralComplex,
  HadamardF32,
  HadamardF16,
  HadamardBF16,
  QmmF32,
  QmmF16,
  QmmBF16,
  DequantF32,
  DequantF16,
  ConvF32,
  ConvF16,
  ConvBF16,
  // Wave 5: indexing and scatter. The U32 kernels carry bitwise word
  // storage, so float32 shares them with int32 and uint32.
  GatherAxisU32,
  GatherAxisI64,
  GatherAxisF16,
  GatherAxisBF16,
  GatherAxisComplex64,
  ScatterU32,
  ScatterF16,
  ScatterBF16,
  ScatterComplex64,
  ScatterAxisU32,
  ScatterAxisF16,
  ScatterAxisBF16,
  ScatterAxisComplex64,
  MaskedScatterU32,
  MaskedScatterF16,
  MaskedScatterBF16,
  ScatterMultiU32,
  ScatterMultiF16,
  ScatterMultiBF16,
  ScatterTripleU32,
  ScatterTripleF16,
  ScatterTripleBF16,
  ScatterBoolTriple,
  ScatterGeneralU32,
  ScatterGeneralF16,
  ScatterGeneralBF16,
  ScatterGeneralBool,
  ScatterGeneralU8,
  ScatterGeneralI8,
  ScatterGeneralU16,
  ScatterGeneralI16,
  SliceUpdateReduceF32,
  SliceUpdateReduceF16,
  SliceUpdateReduceBF16,
  SliceUpdateReduceU32,
  TakeF32,
  TakeF16,
  TakeBF16,
  TakeU32,
  TakeU16,
  TakeI64,
  TakeComplex64,
  TakeMultiF32,
  TakeMultiF16,
  TakeMultiBF16,
  TakeMultiU32,
  TakeMultiU16,
  TakeMultiI64,
  TakeMultiComplex64,
  ClearU32,
  BlockMaskF32,
  GatherMmF32,
  GatherMmF16,
  GatherMmBF16,
  SegmentedMmF32,
  SegmentedMmF16,
  SegmentedMmBF16,
  GatherQmmF32,
  GatherQmmF16,
  GatherQmmBF16,
  GatherQmmNbF32,
  GatherQmmNbF16,
  GatherQmmNbBF16,
  // Wave 8: FFT. FftF32 is the radix-2 Cooley-Tukey pass (complex64 pairs
  // in shared memory); FftRealF32 strips the real part of a complex64
  // buffer into float32 for the irfft tail; FftStageF32 runs the
  // elementwise general-length stages (Cooley-Tukey twiddle multiply,
  // Bluestein chirp multiply, b-table build, pointwise FFT multiply, and
  // the Bluestein epilogue), one thread per element.
  FftF32,
  FftRealF32,
  FftStageF32,
  // Wave 9: fused and custom kernels. Norm forward and VJP kernels run one
  // workgroup per row with float32 arithmetic; the dw kernel runs a single
  // workgroup with per-column accumulators. ConvertFP8 pairs travel as
  // little-endian uint32 word packs of four E4M3 bytes.
  FastRmsNormF32,
  FastRmsNormF16,
  FastRmsNormBF16,
  FastLayerNormF32,
  FastLayerNormF16,
  FastLayerNormBF16,
  FastRmsNormVjpDxF32,
  FastRmsNormVjpDxF16,
  FastRmsNormVjpDxBF16,
  FastLayerNormVjpDxF32,
  FastLayerNormVjpDxF16,
  FastLayerNormVjpDxBF16,
  FastRmsNormVjpDwF32,
  FastRmsNormVjpDwF16,
  FastRmsNormVjpDwBF16,
  FastLayerNormVjpDwF32,
  FastLayerNormVjpDwF16,
  FastLayerNormVjpDwBF16,
  CrossEntropyVjpF32,
  CrossEntropyVjpF16,
  CrossEntropyVjpBF16,
  CrossEntropyF32,
  CrossEntropyF16,
  CrossEntropyBF16,
  // FusedRoPE: one thread rotates one (position, frequency) pair; the
  // Freqs variants take a float32 inv-frequencies source, the base
  // variants compute exp(-i * log(base)/(dims/2)) in-shader from the
  // precomputed beta push constant. See shaders/fast_rope.comp for the
  // contract cases and the push-constant field mapping.
  FastRopeF32,
  FastRopeF16,
  FastRopeBF16,
  FastRopeFreqsF32,
  FastRopeFreqsF16,
  FastRopeFreqsBF16,
  Fp8ToF32,
  Fp8ToF16,
  Fp8ToBF16,
  Fp8FromF32,
  Fp8FromF16,
  Fp8FromBF16,
  // Wave 7: linear algebra. One workgroup per batch matrix, float32
  // only, matching the upstream CPU dtype contract; SVD runs as a
  // sweeps kernel plus a separate finalize kernel.
  LinalgCholeskyF32,
  LinalgInverseF32,
  LinalgLuF32,
  LinalgQrF32,
  LinalgEighF32,
  LinalgEigF32,
  LinalgSvdF32,
  LinalgSvdFinalizeF32,
  // FixFastSdpaAndNorm (W9): dw stage 2 - column sum of the per-row
  // partials from the VjpDw kernels. The VjpDw enums above are the
  // partial stage; this one sums rows into the gradient dtype.
  FastRmsNormVjpDwReduceF32,
  FastRmsNormVjpDwReduceF16,
  FastRmsNormVjpDwReduceBF16,
  // WideRowTopK: one workgroup per row binary-searches the monotone key
  // and serially emits the argpartition indices. One variant per input
  // dtype (f32, f16, bf16).
  // Complex64Transport: complex64 transport and elementwise. One
  // element is a vec2 (re, im) pair in std430 storage, so offsets and
  // strides are item offsets exactly like the float32 kernels; no
  // 16-bit storage features are involved. ComplexElementwise carries
  // the operation code (conjugate/add/sub/mul/div/negate) in
  // params.operation; ComplexReal and ComplexImag extract one
  // component to float32; the Cast* pairs mirror the upstream
  // static_cast rules (real source promotes to (x, 0), complex64
  // source reads real()).
  ComplexElementwise,
  ComplexReal,
  ComplexImag,
  CastF32Complex64,
  CastI32Complex64,
  CastU32Complex64,
  CastBoolComplex64,
  CastF16Complex64,
  CastBF16Complex64,
  CastComplex64F32,
  FillComplex64,
  CopyGeneralComplex64,
  // ComplexAbs: magnitude |z| of a complex64 element into float32,
  // the value-side counterpart to compare_complex and the missing
  // piece upstream allclose needs for complex differences. The shader
  // (shaders/complex_extract.comp, operation 2) implements hypot(re,
  // im) with overflow-safe scaling so large-magnitude inputs do not
  ComplexAbs,
  // ComplexAbsAsComplex: same magnitude kernel but the output buffer
  // is complex64 (vec2): the magnitude lands in .x and 0 in .y so
  // the downstream Cast(complex64 -> float32) reads real() and gets
  // the magnitude. This is the path Abs::eval_gpu takes when ops.cpp
  // builds the primary output as complex64 and applies a separate
  // astype Cast downstream.
  ComplexAbsAsComplex,
  // ScatterDeterminism: float scatter reductions ride hardware fp32
  // atomic add where VK_EXT_shader_atomic_float reports
  // shaderBufferFloat32AtomicAdd (llvmpipe does; the M1 Honeykrisp
  // does NOT advertise the extension at all - the pre-2026-09-02
  // "measured on both" note was a misread of llvmpipe's feature list
  // on a box exposing both devices). The FADD variants accumulate
  // Sum/Prod in an fp32 per-element scratch and the Bool variants
  // carry packed-word byte read-modify-write; the f16/bf16 FADD blobs
  // also serve Prod for those dtypes. The FCAS variants are the
  // no-extension twins: op 11 is replaced by the op-17 compare-
  // exchange add, Prod's op 13 CAS is unchanged.
  ScatterFAddF32,
  ScatterFAddF16,
  ScatterFAddBF16,
  ScatterFAddMultiF32,
  ScatterFAddTripleF32,
  ScatterFAddGeneralF32,
  ScatterFCasF32,
  ScatterFCasF16,
  ScatterFCasBF16,
  ScatterFCasMultiF32,
  ScatterFCasTripleF32,
  ScatterFCasGeneralF32,
  ScatterBool,
  ScatterBoolMulti,
  ScatterAxisFAddF32,
  ScatterAxisFAddF16,
  ScatterAxisFAddBF16,
  ScatterAxisFCasF32,
  ScatterAxisFCasF16,
  ScatterAxisFCasBF16,
  ScatterAxisBool,
  // DecodeGemv: matrix-vector kernel for Qmm when lhs has a single
  // row (the decode shape). One workgroup owns eight output columns;
  // lanes stride single k steps so weight reads stay coalesced. The
  // default reduction is a five-round workgroup-shared tree.
  QmmVecF32,
  QmmVecF16,
  QmmVecBF16,
  // DecodeGemvSubgroup: same shader compiled with -DUSE_SUBGROUP=1,
  // replacing the tree with one subgroupAdd per 32-lane slot. Dispatch
  // is gated in primitives.cpp on caps.subgroup_size == 32 and the
  // ARITHMETIC subgroup-feature bit; a device that lacks either falls
  // back to QmmVecF32/16/BF16. These enum values live at the end so
  // older indices stay stable for the GPU-profile NDJSON stream.
  QmmVecSubgroupF32,
  QmmVecSubgroupF16,
  QmmVecSubgroupBF16,
  QmmTileF32,
  QmmTileF16,
  QmmTileBF16,
  FusedChainF32,
  FusedChainF16,
  QuantizeF32,
  QuantizeF16,
  ReduceGeneralBool,
  // Numeric casts: one blob per source/destination storage-width pair.
  // Runtime dtype codes preserve integer signedness, floating conversion,
  // and complex real-part projection without a per-dtype kernel matrix.
  CastIntW1W1,
  CastIntW1W2,
  CastIntW1W4,
  CastIntW1W8,
  CastIntW2W1,
  CastIntW2W2,
  CastIntW2W4,
  CastIntW2W8,
  CastIntW4W1,
  CastIntW4W2,
  CastIntW4W4,
  CastIntW4W8,
  CastIntW8W1,
  CastIntW8W2,
  CastIntW8W4,
  CastIntW8W8,
  // NarrowIntTail: widened-word bitwise variants for the 8/16/64-bit
  // integer family (BitwiseBinary and BitwiseInvert only; the host
  // refuses every other operation on these dtypes by name). W2 blobs
  // need 16-bit storage, W8 blobs the shaderInt64 feature.
  CompareU64,
  ElementwiseI8,
  ElementwiseU8,
  ElementwiseI16,
  ElementwiseU16,
  ElementwiseI64,
  ElementwiseU64,
  // NarrowIntTail: narrow-int comparisons. Byte variants ride the
  // packed word transport; 16-bit variants need 16-bit storage.
  CompareI8,
  CompareU8,
  CompareI16,
  CompareU16,
  // NarrowIntTail: 8/16-bit Scatter through the packed byte-insert
  // (8-bit) and plain 16-bit store winner writes, NONE reduce only.
  ScatterU8,
  ScatterI8,
  ScatterU16,
  ScatterI16,
  ScatterMultiU8,
  ScatterMultiI8,
  ScatterMultiU16,
  ScatterMultiI16,
  ScatterTripleU8,
  ScatterTripleI8,
  ScatterTripleU16,
  ScatterTripleI16,
  // NarrowIntTail: 64-bit and 16-bit non-zero scalar fills; the U64
  // variant rides shaderInt64, the U16 variant 16-bit storage.
  FillU64,
  FillU16,
  // MultiBlockSort: one global-memory bitonic compare-exchange stage
  // (shaders/sort_merge.comp) that continues sort_suffix past 1024-wide
  // rows. The host launches one dispatch per network stage and
  // alternates a ping-pong buffer pair; ARGSORT variants carry the
  // source positions in a parallel index buffer for the stable order.
  SortMergeF32,
  SortMergeF16,
  SortMergeBF16,
  SortMergeI32,
  SortMergeU32,
  ArgSortMergeF32,
  ArgSortMergeF16,
  ArgSortMergeBF16,
  ArgSortMergeC64,
  ArgSortMergeI32,
  ArgSortMergeU32,
  // Quantize-mode kernels (mxfp4 / nvfp4 / mxfp8): the affine shaders
  // compiled with -DFP_MODE=1. Byte-packed fp4/fp8 element codes, byte
  // scales through a uint word view, no bias term. Appended at the end
  // so older indices stay stable for the GPU-profile NDJSON stream.
  QuantizeFpF32,
  QuantizeFpF16,
  QuantizeFpBF16,
  DequantFpF32,
  DequantFpF16,
  DequantFpBF16,
  QmmFpF32,
  QmmFpF16,
  QmmFpBF16,
  QmmVecFpF32,
  QmmVecFpF16,
  QmmVecFpBF16,
  QmmVecSubgroupFpF32,
  QmmVecSubgroupFpF16,
  QmmVecSubgroupFpBF16,
  QmmTileFpF32,
  QmmTileFpF16,
  QmmTileFpBF16,
  // Gathered fp-mode matmul (gather_qmm.comp with -DFP_MODE=1
  // -DNO_BIAS=1): GatherQMM and GatherQQMM in the mxfp4 / nvfp4 /
  // mxfp8 modes, no bias term. Appended after the affine fp kernels
  // so older indices stay stable for the GPU-profile NDJSON stream.
  GatherQmmNbFpF32,
  GatherQmmNbFpF16,
  GatherQmmNbFpBF16,
  // GatherQmmNbFp plus the nvfp4 output global-scale correction: a
  // fifth binding carries the float32 global_scale_w word.
  GatherQmmNbFpHgsF32,
  GatherQmmNbFpHgsF16,
  GatherQmmNbFpHgsBF16,
  MatmulComplex64,
  // DecodeQ4Word is appended for GPU-profile enum stability. The six
  // binaries share one affine transposed 4-bit/group-64 kernel shape.
  QmmVecQ4WordF32,
  QmmVecQ4WordF16,
  QmmVecQ4WordBF16,
  QmmVecQ4WordSubgroupF32,
  QmmVecQ4WordSubgroupF16,
  QmmVecQ4WordSubgroupBF16,
  // Prefill register block for the transposed affine 4-bit/group-64 f16
  // path. Appended so existing GPU-profile kernel ids stay stable.
  QmmTileRbF16,
  // Prefill on the 8x8x8 fp32 cooperative matrix, same layout as
  // QmmTileRbF16 (shaders/qmm_coopmat.comp).
  QmmPrefillCoopmatF16,
  MatmulBF16Coopmat,
  // Eager BF16 SwiGLU fusion. Appended to keep profile kernel ids stable.
  FusedChainBF16,
  // Four-wide binary add/mul/div/sub on 16-bit storage
  // (shaders/binary_vec.comp); dispatch_float_elementwise_to gates on
  // alignment. Appended to keep profile kernel ids stable.
  BinaryVecF16,
  BinaryVecBF16,
  // Register-blocked f16 matmul (shaders/matmul_rb.comp), same
  // arithmetic as MatmulF16 on a 64x64 tile; dispatch_matmul gates on
  // matrix_m >= 32 and no bias.
  MatmulRbF16,
  // Straight-line four-wide SwiGLU chain (shaders/swiglu.comp) with the
  // fused_chain.comp rounding; dispatch_chain pattern-matches it.
  SwigluF16,
  SwigluBF16,
  // DecodeFusion multi-weight Q4 GEMV (shaders/qmm_vec.comp
  // QMM_VEC_MULTI): up to three transposed 4-bit/group-64 weights per
  // dispatch with optional fused Add epilogues. Appended to keep
  // profile kernel ids stable.
  QmmVecQ4MultiF32,
  QmmVecQ4MultiF16,
  QmmVecQ4MultiBF16,
  QmmVecQ4MultiSubgroupF32,
  QmmVecQ4MultiSubgroupF16,
  QmmVecQ4MultiSubgroupBF16,
  // Non-contracted large-prefill fallback; appended to preserve profile ids.
  QmmTileRbPreciseF16,
  // Native-order single-query f16 attention; append-only profile id.
  SdpaDecodeNativeF16,
  Custom,
  Count,
};

struct ComputeBinding {
  VkBuffer buffer;
  VkDeviceSize offset;
  VkDeviceSize range;
  // Owning VulkanBuffer for batch stamping; null for placeholder slots.
  // A dispatch binds buffers that no add_temporary recorded (plain input
  // and output arrays), and an unstamped buffer whose array dies before
  // the batch drains would be recycled or destroyed while the queued
  // commands still reference it.
  const void* owner{nullptr};
};

struct ComputeParams {
  uint32_t count{0};
  uint32_t operation{0};
  uint32_t lhs_size{0};
  uint32_t rhs_size{0};
  uint32_t reduce_size{0};
  uint32_t output_size{0};
  uint32_t lhs_offset{0};
  uint32_t rhs_offset{0};
  uint32_t output_offset{0};
  uint32_t aux_size{0};
  uint32_t aux_offset{0};
  uint32_t matrix_m{0};
  uint32_t matrix_n{0};
  uint32_t matrix_k{0};
  // Elementwise-style broadcast transport selector: 0 keeps the inline
  // push-constant arrays; nonzero carries the collapsed rank and reads
  // [extents | lhs strides | rhs strides] from the axis-metadata
  // storage buffer (the reduce_general.comp binding-3 convention).
  // Matmul reuses the field as its inner-matrix K.
  // Matmul flag bits: 1 = rhs transposed, 2 = bias c used,
  // 4 = lhs transposed. Batch routing is data-driven: dims is the batch
  // axis count, shape[] the batch extents, and in_strides/out_strides[]
  // the per-operand batch strides in elements (0 = broadcast axis). The
  // shader unravels workgroup z over shape[] and offsets each operand.
  uint32_t flags{0};
  float alpha{1.0f};
  float beta{0.0f};
  // Broadcast rank for elementwise kernels; batch axis count for Matmul
  // kernels (0 for rank-2).
  uint32_t dims{0};
  uint32_t shape[4]{};
  uint32_t in_strides[4]{};
  uint32_t out_strides[4]{};
  // Matmul inner-matrix gaps in elements: the stored row stride, or the
  // stored column stride under the transposed flag. Dense operands
  // carry the natural gap; a cache slice carries its row gap and skips
  // materialization.
  uint32_t lhs_gap{0};
  uint32_t rhs_gap{0};
};

class ComputeRuntime {
 public:
  explicit ComputeRuntime(VkDevice device, uint32_t binding_limit);
  ~ComputeRuntime();

  ComputeRuntime(const ComputeRuntime&) = delete;
  ComputeRuntime& operator=(const ComputeRuntime&) = delete;

  VkPipeline pipeline(ComputeKernel kernel);
  VkPipeline pipeline(
      const std::string& cache_key,
      std::span<const uint32_t> spirv);
  VkPipelineLayout pipeline_layout() const {
    return pipeline_layout_;
  }
  // Storage-buffer binding slots available to any dispatch on this device:
  // the backend budget clamped by what the physical device reports. A kernel
  // needing more must refuse by name; kComputeBindingFloor is the minimum a
  // spec-conformant device reports, so slots up to the floor always exist.
  uint32_t binding_limit() const {
    return binding_limit_;
  }
  VkDescriptorSetLayout descriptor_layout() const {
    return descriptor_layout_;
  }

 private:
  VkPipeline create_pipeline(ComputeKernel kernel);
  VkPipeline create_pipeline(std::span<const uint32_t> spirv);

  uint32_t binding_limit_{0};

  VkDevice device_;
  VkDescriptorSetLayout descriptor_layout_{VK_NULL_HANDLE};
  VkPipelineLayout pipeline_layout_{VK_NULL_HANDLE};
  std::array<VkPipeline, static_cast<size_t>(ComputeKernel::Count)> pipelines_{};
  std::unordered_map<std::string, VkPipeline> dynamic_pipelines_;
  std::mutex mutex_;
};

} // namespace mlx::core::omarchy
