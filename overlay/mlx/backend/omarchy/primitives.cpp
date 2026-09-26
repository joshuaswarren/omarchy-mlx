// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

#include <unistd.h>
#include <sys/syscall.h>
#include "mlx/backend/omarchy/unsupported.h"
#include "mlx/transforms.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <initializer_list>
#include <optional>
#include <numeric>
#include <string>
#include <typeinfo>
#include <utility>

#include "mlx/backend/common/binary.h"
#include "mlx/backend/common/matmul.h"
#include "mlx/backend/common/slicing.h"
#include "mlx/backend/common/unary.h"
#include "mlx/backend/omarchy/allocator.h"
#include "mlx/backend/omarchy/capability_sim.h"
#include "mlx/backend/omarchy/compute.h"
#include "mlx/backend/omarchy/compiled.h"
#include "mlx/backend/omarchy/device.h"
#include "mlx/backend/omarchy/encoder.h"
#include "mlx/backend/omarchy/fused_chain.h"
#include "mlx/distributed/primitives.h"
#include "mlx/fast_primitives.h"
#include "mlx/backend/gpu/copy.h"
#include "mlx/backend/gpu/eval.h"
#include "mlx/primitives.h"
#include "mlx/ops.h"

#define OMARCHY_UNSUPPORTED(func)                                     \
  void func::eval_gpu(const std::vector<array>& inputs, array& out) { \
    omarchy::unsupported(#func, out);                                 \
  }

#define OMARCHY_UNSUPPORTED_MULTI(func)                                \
  void func::eval_gpu(                                                 \
      const std::vector<array>& inputs, std::vector<array>& outputs) { \
    omarchy::unsupported(#func, outputs.at(0));                        \
  }

#define OMARCHY_USE_FALLBACK(func)    \
  bool func::use_fallback(Stream s) { \
    return true;                      \
  }                                   \
  OMARCHY_UNSUPPORTED_MULTI(func)

// The RoPE fusion trig envelope gate; defined beside RoPE::eval_gpu.
// Declared at file scope: a qualified namespace definition inside
// mlx::core::omarchy would open mlx::core::omarchy::mlx::... and poison
// every later mlx::core::fast:: qualified lookup in this namespace.
namespace mlx::core::fast {
void rope_trig_gate(
    const std::string& name,
    const array& in,
    const array& offset,
    const array* freqs,
    int dims,
    float base,
    float scale,
    const array& out);
} // namespace mlx::core::fast

namespace mlx::core {

namespace {

// Keep in lockstep with the switch in shaders/elementwise.comp.
enum ElementwiseOperation : uint32_t {
  AddOperation,
  MultiplyOperation,
  DivideOperation,
  MaximumOperation,
  ExpOperation,
  SigmoidOperation,
  SquareOperation,
  SqrtOperation,
  RsqrtOperation,
  SubtractOperation,
  NegativeOperation,
  CosOperation,
  SinOperation,
  LogOperation,
  MinimumOperation,
  Log2Operation,
  Log10Operation,
  ArcCosOperation,
  ArcCoshOperation,
  ArcSinOperation,
  ArcSinhOperation,
  ArcTanOperation,
  ArcTan2Operation,
  ArcTanhOperation,
  CoshOperation,
  SinhOperation,
  TanOperation,
  TanhOperation,
  ErfOperation,
  ErfInvOperation,
  Expm1Operation,
  Log1pOperation,
  LogAddExpOperation,
  CeilOperation,
  FloorOperation,
  RoundOperation,
  // Wave-2 float ops. The codes continue the wave-3 block and match
  // elementwise.comp cases 36-40.
  RemainderFloatOperation,
  PowerFloatOperation,
  SignFloatOperation,
  AbsFloatOperation,
  DivQuotientFloatOperation,
};

allocator::Buffer allocate_omarchy(size_t size) {
  return omarchy::allocator().malloc(size);
}

uint32_t checked_u32(size_t value, const std::string& name, const array& out) {
  if (value > std::numeric_limits<uint32_t>::max()) {
    omarchy::unsupported(name + " with more than UINT32_MAX elements", out);
  }
  return static_cast<uint32_t>(value);
}

uint64_t integer_arange_bits(double value, const array& out) {
  if (!std::isfinite(value)) {
    omarchy::unsupported("Arange integer parameter", out);
  }

  constexpr double kWord = 4294967296.0;
  constexpr double kRange = kWord * kWord;
  double magnitude = std::fmod(std::trunc(std::abs(value)), kRange);
  auto high = static_cast<uint64_t>(std::floor(magnitude / kWord));
  auto low = static_cast<uint64_t>(magnitude - high * kWord);
  uint64_t bits = (high << 32) | low;
  return std::signbit(value) ? uint64_t{0} - bits : bits;
}

void require_float_dtype(
    const std::string& name,
    const array& input,
    const array& out,
    omarchy::CommandEncoder& encoder) {
  if ((input.dtype() != float16 && input.dtype() != float32 &&
       input.dtype() != bfloat16) ||
      input.dtype() != out.dtype()) {
    omarchy::unsupported(name + " dtype", out);
  }
  const auto& capabilities = encoder.device().capabilities();
  if (input.dtype() == float16 &&
      (!capabilities.shader_float16 ||
       !capabilities.storage_buffer_16bit_access)) {
    omarchy::unsupported(name + " float16 capability", out);
  }
  if (input.dtype() == bfloat16 &&
      (!capabilities.storage_buffer_16bit_access ||
       !capabilities.shader_int16)) {
    omarchy::unsupported(name + " bfloat16 capability", out);
  }
  if ((input.dtype() == int16 || input.dtype() == uint16) &&
      (!capabilities.storage_buffer_16bit_access ||
       !capabilities.shader_int16)) {
    omarchy::unsupported(name + " 16-bit capability", out);
  }
}

omarchy::ComputeKernel select_float_kernel(
    Dtype dtype,
    omarchy::ComputeKernel f32_kernel,
    omarchy::ComputeKernel f16_kernel,
    omarchy::ComputeKernel bf16_kernel) {
  if (dtype == float16) {
    return f16_kernel;
  }
  if (dtype == bfloat16) {
    return bf16_kernel;
  }
  return f32_kernel;
}

uint32_t checked_item_offset(
    const array& value,
    size_t count,
    const std::string& name,
    const array& out) {
  if (value.offset() % value.itemsize() != 0) {
    omarchy::unsupported(name + " byte offset", out);
  }
  uint64_t offset = value.offset() / value.itemsize();
  if (!omarchy::compute_index_span_fits(offset, count)) {
    omarchy::unsupported(name + " index span", out);
  }
  return static_cast<uint32_t>(offset);
}

omarchy::ComputeBinding binding(const array& value) {
  auto* buffer =
      static_cast<const omarchy::VulkanBuffer*>(value.buffer().ptr());
  return {buffer->buffer, 0, buffer->size, buffer};
}

// Consumer-boundary dense normalization. Returns |value| itself when
// |dense_enough| holds, so a dense operand takes the caller's existing
// path with no allocation, no header copy, and no rebind. Otherwise the
// general strided-copy engine materializes one dense same-dtype temp;
// the caller owns |temp| across its dispatches and the encoder keeps
// the buffer alive until the committed work completes.
const array& ensure_dense(
    const array& value,
    bool dense_enough,
    std::optional<array>& temp,
    omarchy::CommandEncoder& encoder,
    const Stream& s) {
  if (dense_enough) {
    return value;
  }
  temp = contiguous_copy_gpu(value, s);
  encoder.add_temporary(*temp);
  return *temp;
}

bool is_trailing_broadcast(const array& input, const array& out) {
  if (input.data_size() == 1) {
    return true;
  }
  if (input.ndim() == out.ndim() && input.shape() == out.shape()) {
    size_t expected_stride = 1;
    int axis = input.ndim() - 1;
    for (; axis >= 0 && input.strides()[axis] != 0; --axis) {
      if (input.strides()[axis] != expected_stride) {
        return false;
      }
      expected_stride *= input.shape(axis);
    }
    for (; axis >= 0; --axis) {
      if (input.strides()[axis] != 0) {
        return false;
      }
    }
    return expected_stride == input.data_size();
  }
  if (!input.flags().row_contiguous) {
    return false;
  }
  int first_axis = 0;
  while (first_axis < input.ndim() && input.shape(first_axis) == 1) {
    first_axis++;
  }
  int rank = input.ndim() - first_axis;
  if (rank > out.ndim()) {
    return false;
  }
  for (int axis = 0; axis < rank; ++axis) {
    if (input.shape(first_axis + axis) != out.shape(out.ndim() - rank + axis)) {
      return false;
    }
  }
  return true;
}

uint32_t matrix_group_count(uint32_t dimension, uint32_t tile_size = 16) {
  uint32_t groups = dimension / tile_size + (dimension % tile_size != 0);
  return std::min(groups, omarchy::kMaxComputeGroupCountX);
}

// Byte-swap scalars in place for big-endian source files. Mirrors the
// shared and CUDA Load implementations.
template <const uint8_t scalar_size>
void swap_endianness(uint8_t* data_bytes, size_t n) {
  struct Elem {
    uint8_t bytes[scalar_size];
  };

  Elem* data = reinterpret_cast<Elem*>(data_bytes);
  for (size_t i = 0; i < n; i++) {
    for (size_t j = 0; j < (scalar_size / 2); j++) {
      std::swap(data[i].bytes[j], data[i].bytes[scalar_size - j - 1]);
    }
  }
}


// Materializes a matmul operand whose inner 2-D layout fits neither the
// row-major nor the column-major gap form; concatenate results compose
// this way. The general strided-copy engine writes a standard row-major
// batch, and the encoder keeps the temp alive until the committed work
// completes. Index-span overflow keeps its named error.
array materialize_batched_matrix(
    const array& value,
    const std::string& name,
    array& out,
    const Stream& s) {
  auto& encoder = omarchy::get_command_encoder(s);
  Shape shape = value.shape();
  Strides strides(shape.size(), 1);
  for (int axis = static_cast<int>(shape.size()) - 2; axis >= 0; --axis) {
    strides[axis] = strides[axis + 1] * shape[axis + 1];
  }
  array materialized(shape, value.dtype(), nullptr, {});
  array::Flags flags;
  flags.contiguous = true;
  flags.row_contiguous = true;
  auto max_dim = std::max_element(shape.begin(), shape.end());
  flags.col_contiguous = materialized.size() <= 1 ||
      materialized.size() == *max_dim;
  materialized.set_data(
      omarchy::allocator().malloc(materialized.nbytes()),
      materialized.size(),
      strides,
      flags,
      0);
  if (getenv("MLX_OMARCHY_TRACE_MATERIALIZE")) {
    std::fprintf(
        stderr,
        "[materialize] %s shape=[", name.c_str());
    for (int i = 0; i < value.ndim(); ++i) {
      std::fprintf(stderr, "%s%d", i ? "," : "", value.shape(i));
    }
    std::fprintf(stderr, "] strides=[");
    for (int i = 0; i < value.ndim(); ++i) {
      std::fprintf(stderr, "%s%lld", i ? "," : "",
          static_cast<long long>(value.strides()[i]));
    }
    std::fprintf(stderr, "]\n");
  }
  copy_gpu_inplace(
      value,
      materialized,
      shape,
      value.strides(),
      strides,
      /*i_offset=*/0,
      /*o_offset=*/0,
      CopyType::General,
      out.primitive().stream());
  encoder.add_temporary(materialized);
  return materialized;
}

// Classifies a matmul operand's inner 2-D layout: row-major (column
// stride 1, row gap = the axis -2 stride) or column-major (row stride
// 1, column gap = the axis -1 stride). A size-1 inner axis never
// dereferences its stride. Batch axes ride their actual strides - the
// shader unravels workgroup z over them, and a stride-0 batch axis
// broadcasts - so nothing materializes for batch strides alone. Gaps
// are in elements and must fit uint32; strides outside that, or an
// inner layout fitting neither form, materialize to a dense row-major
// batch first.
void classify_matmul_operand(
    const array& value,
    bool& transposed,
    uint32_t& gap,
    std::optional<array>& materialized,
    const std::string& name,
    array& out,
    const Stream& s) {
  int rank = value.ndim();
  int64_t s_prev = value.strides()[rank - 2];
  int64_t s_last = value.strides()[rank - 1];
  constexpr int64_t kMaxGap = std::numeric_limits<uint32_t>::max();
  if ((s_last == 1 || value.shape(rank - 1) == 1) && s_prev >= 0 &&
      s_prev <= kMaxGap) {
    transposed = false;
    gap = static_cast<uint32_t>(s_prev);
  } else if ((s_prev == 1 || value.shape(rank - 2) == 1) && s_last >= 0 &&
             s_last <= kMaxGap) {
    transposed = true;
    gap = static_cast<uint32_t>(s_last);
  } else {
    materialized = materialize_batched_matrix(value, name, out, s);
    transposed = false;
    gap = static_cast<uint32_t>(value.shape(rank - 1));
  }
}

std::tuple<Shape, std::vector<Strides>> collapse_matmul_batches(
    const array& out,
    std::initializer_list<const array*> operands) {
  const int batch_rank = static_cast<int>(out.ndim()) - 2;
  Shape batch_shape(out.shape().begin(), out.shape().end() - 2);
  std::vector<Strides> batch_strides;
  batch_strides.reserve(operands.size());
  for (const array* operand : operands) {
    const int operand_batch_rank =
        std::max(static_cast<int>(operand->ndim()) - 2, 0);
    const int axis_offset = batch_rank - operand_batch_rank;
    Strides strides(batch_rank, 0);
    for (int axis = std::max(axis_offset, 0); axis < batch_rank; ++axis) {
      const int operand_axis = axis - axis_offset;
      if (operand->shape(operand_axis) != 1) {
        strides[axis] = operand->strides()[operand_axis];
      }
    }
    batch_strides.push_back(std::move(strides));
  }
  auto collapsed = collapse_contiguous_dims(batch_shape, batch_strides);
  auto& collapsed_shape = std::get<0>(collapsed);
  auto& collapsed_strides = std::get<1>(collapsed);
  if (collapsed_shape.empty()) {
    collapsed_shape.push_back(1);
    for (auto& strides : collapsed_strides) {
      strides.push_back(0);
    }
  }
  return collapsed;
}

// causal: attention shortcut for the register-blocked f16 tile
// (shaders/matmul_rb.comp flags 8 / 16), a (key length - query length,
// skip mode) pair; the other kernels ignore it. Only meaningful when
// the consumer never reads the masked output (the causal softmax).
enum class CausalSkip { None, Columns, K };
void dispatch_matmul(
    const std::string& name,
    const std::vector<array>& inputs,
    array& out,
    float alpha,
    float beta,
    bool use_c,
    const Stream& s,
    std::pair<uint32_t, CausalSkip> causal = {0u, CausalSkip::None}) {
  const array& a_in = inputs.at(0);
  const array& b_in = inputs.at(1);
  const array& c_in = use_c ? inputs.at(2) : out;
  auto& encoder = omarchy::get_command_encoder(s);
  if (out.dtype() == complex64) {
    if (a_in.dtype() != complex64 || b_in.dtype() != complex64 ||
        (use_c && c_in.dtype() != complex64)) {
      omarchy::unsupported(name + " dtype", out);
    }
  } else {
    require_float_dtype(name, a_in, out, encoder);
    require_float_dtype(name, b_in, out, encoder);
    if (use_c) {
      require_float_dtype(name, c_in, out, encoder);
    }
  }

  if (a_in.ndim() < 2 || a_in.ndim() != b_in.ndim()) {
    omarchy::unsupported("matrix rank " + name, out);
  }
  std::optional<array> a_materialized;
  std::optional<array> b_materialized;
  std::optional<array> c_materialized;
  uint32_t a_gap = 0;
  uint32_t b_gap = 0;
  bool a_transposed = false;
  bool b_transposed = false;
  classify_matmul_operand(
      a_in, a_transposed, a_gap, a_materialized, name, out, s);
  classify_matmul_operand(
      b_in, b_transposed, b_gap, b_materialized, name, out, s);
  const array* a = a_materialized ? &*a_materialized : &a_in;
  const array* b = b_materialized ? &*b_materialized : &b_in;
  if (use_c && !is_trailing_broadcast(c_in, out)) {
    c_materialized = materialize_batched_matrix(c_in, name, out, s);
  }
  const array* c = c_materialized ? &*c_materialized : &c_in;

  size_t k = a->shape(-1);
  size_t n = b->shape(-1);
  if (b->shape(-2) != k) {
    omarchy::unsupported("matrix dimensions " + name, out);
  }

  Shape batch_shape;
  Strides a_batch_strides;
  Strides b_batch_strides;
  Strides c_batch_strides;
  auto collapse = [&]() {
    auto [shape, strides] = use_c
        ? collapse_matmul_batches(out, {a, b, c})
        : collapse_matmul_batches(out, {a, b});
    batch_shape = std::move(shape);
    a_batch_strides = std::move(strides[0]);
    b_batch_strides = std::move(strides[1]);
    if (use_c) {
      c_batch_strides = std::move(strides[2]);
    }
  };
  collapse();
  if (batch_shape.size() > 4) {
    if (!a_materialized) {
      a_materialized = materialize_batched_matrix(*a, name, out, s);
    }
    if (!b_materialized) {
      b_materialized = materialize_batched_matrix(*b, name, out, s);
    }
    if (use_c && !c_materialized) {
      c_materialized = materialize_batched_matrix(*c, name, out, s);
    }
    a = &*a_materialized;
    b = &*b_materialized;
    c = use_c ? &*c_materialized : &c_in;
    a_transposed = false;
    b_transposed = false;
    a_gap = checked_u32(a->shape(-1), name, out);
    b_gap = checked_u32(b->shape(-1), name, out);
    collapse();
  }
  if (batch_shape.size() > 4) {
    omarchy::unsupported("matrix batch rank " + name, out);
  }

  out.set_data(allocate_omarchy(out.nbytes()));
  if (out.size() == 0) {
    return;
  }
  size_t m = a->shape(-2);
  size_t batch_count = out.size() / (m * n);
  if (batch_count > omarchy::kMaxComputeGroupCountX) {
    omarchy::unsupported("batch count " + name, out);
  }

  uint32_t output_size = checked_u32(out.size(), name, out);
  omarchy::ComputeParams params;
  params.count = output_size;
  params.lhs_size = checked_u32(a->size(), name, out);
  params.rhs_size = checked_u32(b->size(), name, out);
  params.reduce_size = checked_u32(k, name, out);
  params.output_size = output_size;
  params.aux_size = use_c ? checked_u32(c->data_size(), name, out) : 0;
  params.matrix_m = checked_u32(m, name, out);
  params.matrix_n = checked_u32(n, name, out);
  params.matrix_k = params.reduce_size;
  params.alpha = alpha;
  params.beta = beta;
  params.flags = (b_transposed ? 1u : 0u) | (use_c ? 2u : 0u) |
      (a_transposed ? 4u : 0u);
  params.lhs_gap = a_gap;
  params.rhs_gap = b_gap;
  params.dims = static_cast<uint32_t>(batch_shape.size());
  uint64_t a_span = 0;
  uint64_t b_span = 0;
  for (size_t axis = 0; axis < batch_shape.size(); ++axis) {
    int64_t a_stride = a_batch_strides[axis];
    int64_t b_stride = b_batch_strides[axis];
    if (a_stride < 0 || b_stride < 0 ||
        static_cast<uint64_t>(a_stride) > std::numeric_limits<uint32_t>::max() ||
        static_cast<uint64_t>(b_stride) > std::numeric_limits<uint32_t>::max()) {
      omarchy::unsupported(name + " batch stride", out);
    }
    uint32_t extent = checked_u32(batch_shape[axis], name, out);
    params.shape[axis] = extent;
    params.in_strides[axis] = static_cast<uint32_t>(a_stride);
    params.out_strides[axis] = static_cast<uint32_t>(b_stride);
    a_span += static_cast<uint64_t>(extent - 1u) * params.in_strides[axis];
    b_span += static_cast<uint64_t>(extent - 1u) * params.out_strides[axis];
  }
  const array& bound_a = a->size() == 0 ? out : *a;
  const array& bound_b = b->size() == 0 ? out : *b;
  const array& bound_c = use_c ? *c : out;
  params.lhs_offset = checked_item_offset(bound_a, bound_a.size(), name, out);
  params.rhs_offset = checked_item_offset(bound_b, bound_b.size(), name, out);
  params.aux_offset = checked_item_offset(
      bound_c, use_c ? params.aux_size : out.size(), name, out);
  params.output_offset = checked_item_offset(out, out.size(), name, out);

  std::array<omarchy::ComputeBinding, 4> bindings{
      binding(bound_a), binding(bound_b), binding(bound_c), binding(out)};
  auto kernel = out.dtype() == complex64
      ? omarchy::ComputeKernel::MatmulComplex64
      : select_float_kernel(
            out.dtype(),
            omarchy::ComputeKernel::MatmulF32,
            omarchy::ComputeKernel::MatmulF16,
            omarchy::ComputeKernel::MatmulBF16);
  static const bool coopmat_disabled =
      omarchy::env_flag("MLX_OMARCHY_NO_COOPMAT");
  const auto& caps = encoder.device().capabilities();
  const bool coopmat_base = caps.cooperative_matrix_f32_8 &&
      caps.subgroup_size == 32 && !coopmat_disabled &&
      params.matrix_m > 1u && params.matrix_k > 0u && !use_c;
  // matmul_coopmat.comp never reads alpha (MatmulF32Coopmat stays gated
  // on alpha == 1); matmul_coopmat_bf16.comp scales its f32 accumulator
  // by alpha at the drain, so MatmulBF16Coopmat may take any alpha -
  // that is the attention-scores shape (alpha = 1/sqrt(head_dim)).
  const bool coopmat_alpha = alpha == 1.0f ||
      kernel == omarchy::ComputeKernel::MatmulBF16;
  bool bf16_aligned = ((params.lhs_offset | params.rhs_offset |
      params.output_offset | a_gap | b_gap | params.matrix_n) & 1u) == 0u;
  for (uint32_t axis = 0; bf16_aligned && axis < params.dims; ++axis) {
    bf16_aligned = ((params.in_strides[axis] |
        params.out_strides[axis]) & 1u) == 0u;
  }
  // Scalar-FMA bf16 route (shaders/matmul_fma_bf16.comp): the
  // linear-layer orientation (row-major lhs, column-major rhs, alpha 1)
  // leaves the 8x8x8 matrix unit. uvec4 operand reads need 8-byte
  // offsets/strides; word-pair bf16 stores need an even output row.
  // Reserved for drivers without the cooperative matrix (the gate below);
  // MLX_OMARCHY_NO_MATMUL_FMA=1 falls back to the shipped pick for A/B.
  static const bool matmul_fma_disabled =
      omarchy::env_flag("MLX_OMARCHY_NO_MATMUL_FMA");
  bool bf16_fma_aligned = ((params.lhs_offset | params.rhs_offset |
      params.output_offset | a_gap | b_gap) % 8u) == 0u;
  for (uint32_t axis = 0; bf16_fma_aligned && axis < params.dims; ++axis) {
    bf16_fma_aligned = ((params.in_strides[axis] |
        params.out_strides[axis]) % 8u) == 0u;
  }
  bool bf16_fma = !matmul_fma_disabled &&
      kernel == omarchy::ComputeKernel::MatmulBF16 && alpha == 1.0f &&
      b_transposed && !a_transposed && !use_c &&
      params.matrix_m >= 32u && (params.matrix_k % 8u) == 0u &&
      (params.matrix_n & 1u) == 0u && bf16_fma_aligned;
  // The staged bf16 tile shares qmm_coopmat's 4 KiB staging footprint
  // (a 32x16 A patch and a 16x32 B patch); gate on the device limit the
  // same way the qmm route does instead of assuming it.
  constexpr uint32_t kMatmulCoopmatBf16SharedBytes =
      (32u * 16u + 16u * 32u) * sizeof(float);
  const bool coopmat = coopmat_base && coopmat_alpha &&
      (kernel == omarchy::ComputeKernel::MatmulF32 ||
       (kernel == omarchy::ComputeKernel::MatmulBF16 && bf16_aligned &&
        (params.matrix_k % 8u) == 0u && params.matrix_m >= 32u &&
        kMatmulCoopmatBf16SharedBytes <= caps.max_compute_shared_memory_size));
  // Driver split, measured (receipts/2026-09-12-prefill-fma-qualify):
  // where the bf16 cooperative-matrix route exists (fork) it beats the
  // scalar-FMA kernel (783.7 vs 629.9 GFLOP/s at the dominant gate_up
  // cell); where it does not (stock Mesa) the FMA kernel triples the
  // shipped staged-tile fallback (645.2 vs 195.0). The FMA pick is
  // therefore reserved for drivers without the cooperative matrix.
  bf16_fma = bf16_fma && !coopmat;
  // Bit 8 tells matmul_coopmat_bf16.comp its operands are 8-byte
  // aligned, so the two word-adjacent orientations stage with uvec2
  // pair loads (one load per two bf16 pairs) instead of scalar words.
  // Pure transport: staged values are identical either way.
  if (coopmat && kernel == omarchy::ComputeKernel::MatmulBF16 &&
      ((params.lhs_offset | params.rhs_offset | a_gap | b_gap) & 3u) == 0u) {
    bool pair_strides = true;
    for (uint32_t axis = 0; pair_strides && axis < params.dims; ++axis) {
      pair_strides = ((params.in_strides[axis] |
          params.out_strides[axis]) & 3u) == 0u;
    }
    if (pair_strides) {
      params.flags |= 8u;
    }
  }
  if (coopmat) {
    kernel = kernel == omarchy::ComputeKernel::MatmulF32
        ? omarchy::ComputeKernel::MatmulF32Coopmat
        : omarchy::ComputeKernel::MatmulBF16Coopmat;
    omarchy::capsim::require_backed(
        encoder.device(),
        caps,
        coopmat,
        "MatmulF32Coopmat/MatmulBF16Coopmat",
        "cooperative_matrix_fp32_8x8x8",
        encoder.device().hardware_capabilities().cooperative_matrix_f32_8);
  }
  // Register-blocked f16 tile for prefill-sized matrices (the
  // attention scores and probs matmuls): same per-output arithmetic as
  // matmul.comp, 64x64 tile. Decode (matrix_m == 1) keeps the 16x16
  // tile and the GEMV paths untouched.
  const bool rb = kernel == omarchy::ComputeKernel::MatmulF16 &&
      params.matrix_m >= 32u && !use_c;
  if (rb) {
    kernel = omarchy::ComputeKernel::MatmulRbF16;
    if (causal.second != CausalSkip::None) {
      params.aux_size = causal.first;
      params.flags |= causal.second == CausalSkip::Columns ? 8u : 16u;
    }
  }
  const uint32_t tile = coopmat ? 32u : (rb ? 64u : 16u);
  uint64_t a_inner = params.matrix_m == 0u || params.matrix_k == 0u
      ? 0u
      : (a_transposed
            ? static_cast<uint64_t>(params.matrix_k - 1) * a_gap +
                params.matrix_m
            : static_cast<uint64_t>(params.matrix_m - 1) * a_gap +
                params.matrix_k);
  uint64_t b_inner = params.matrix_k == 0u || params.matrix_n == 0u
      ? 0u
      : (b_transposed
            ? static_cast<uint64_t>(params.matrix_n - 1) * b_gap +
                params.matrix_k
            : static_cast<uint64_t>(params.matrix_k - 1) * b_gap +
                params.matrix_n);
  if (!omarchy::compute_index_span_fits(
          params.lhs_offset, a_span + a_inner) ||
      !omarchy::compute_index_span_fits(
          params.rhs_offset, b_span + b_inner)) {
    omarchy::unsupported(name + " index span", out);
  }
  bool bf16_vec_aligned = ((params.lhs_offset | params.rhs_offset |
      params.rhs_gap) & 3u) == 0u;
  for (uint32_t axis = 0; bf16_vec_aligned && axis < params.dims; ++axis) {
    bf16_vec_aligned = ((params.in_strides[axis] |
        params.out_strides[axis]) & 3u) == 0u;
  }
  const bool bf16_vec = out.dtype() == bfloat16 && params.matrix_m == 1u &&
      b_transposed && !a_transposed && !use_c && alpha == 1.0f &&
      (params.matrix_k % 128u) == 0u && (params.matrix_n % 4u) == 0u &&
      caps.subgroup_size == 32u &&
      (caps.subgroup_operations & VK_SUBGROUP_FEATURE_SHUFFLE_RELATIVE_BIT) != 0 &&
      bf16_vec_aligned;
  if (bf16_vec) {
    omarchy::capsim::require_backed(
        encoder.device(),
        caps,
        bf16_vec,
        "MatmulVecBF16",
        "subgroup_size==32+subgroup_ops_mask[SHUFFLE_RELATIVE]",
        (encoder.device().hardware_capabilities().subgroup_size == 32u &&
         (encoder.device().hardware_capabilities().subgroup_operations &
          VK_SUBGROUP_FEATURE_SHUFFLE_RELATIVE_BIT) != 0));
    encoder.dispatch_compute(
        omarchy::ComputeKernel::MatmulVecBF16,
        bindings,
        params,
        matrix_group_count(params.matrix_n, 4u),
        1u,
        checked_u32(batch_count, name, out));
    return;
  }
  if (bf16_fma) {
    omarchy::ComputeKernel fma_kernel = omarchy::ComputeKernel::MatmulBf16Fma;
    encoder.dispatch_compute(
        fma_kernel,
        bindings,
        params,
        matrix_group_count(params.matrix_n, 128u),
        matrix_group_count(params.matrix_m, 32u),
        checked_u32(batch_count, name, out));
    return;
  }
  encoder.dispatch_compute(
      kernel,
      bindings,
      params,
      matrix_group_count(params.matrix_n, tile),
      matrix_group_count(params.matrix_m, tile),
      checked_u32(batch_count, name, out));
}

// Fills the general broadcast transport (dims/shape/strides) shared by
// the elementwise-style kernels. Broadcast views keep the output shape
// with stride-0 axes, so the view strides index the sources directly;
// collapse_contiguous_dims merges the linear runs, and a stride of 0
// breaks every merge around a broadcast axis. Up to four collapsed axes
// ride the inline push-constant arrays. Higher ranks carry [extents | lhs
// strides | rhs strides] in an axis-metadata storage buffer - the
// reduce_general.comp binding-3 word order - and set matrix_k to the
// collapsed rank; the shader routes its unravel through the metadata
// whenever matrix_k is nonzero. Callers set the
// offsets before calling, because the span check uses them, and bind
// the returned metadata array in the kernel's metadata slot (or a
// placeholder buffer when it is empty: the slot must hold a valid
// descriptor even while the shader never reads it).
std::optional<array> fill_broadcast_transport(
    const std::string& error_name,
    omarchy::ComputeParams& params,
    const array& lhs,
    const array& rhs,
    const array& out,
    omarchy::CommandEncoder& encoder) {
  if (lhs.shape() != out.shape() || rhs.shape() != out.shape()) {
    omarchy::unsupported("broadcast " + error_name, out);
  }
  auto [collapsed_shape, collapsed_strides] = collapse_contiguous_dims(
      out.shape(), std::vector<Strides>{lhs.strides(), rhs.strides()});
  params.dims = static_cast<uint32_t>(collapsed_shape.size());
  uint64_t lhs_span = 0;
  uint64_t rhs_span = 0;
  if (collapsed_shape.size() <= 4) {
    for (size_t axis = 0; axis < collapsed_shape.size(); ++axis) {
      params.shape[axis] = static_cast<uint32_t>(collapsed_shape[axis]);
      params.in_strides[axis] =
          static_cast<uint32_t>(collapsed_strides[0][axis]);
      params.out_strides[axis] =
          static_cast<uint32_t>(collapsed_strides[1][axis]);
      uint64_t extent = params.shape[axis] - 1u;
      lhs_span += extent * params.in_strides[axis];
      rhs_span += extent * params.out_strides[axis];
    }
    if (!omarchy::compute_index_span_fits(params.lhs_offset, lhs_span + 1) ||
        !omarchy::compute_index_span_fits(params.rhs_offset, rhs_span + 1)) {
      omarchy::unsupported(error_name + " index span", out);
    }
    return std::nullopt;
  }
  size_t rank = collapsed_shape.size();
  std::vector<uint32_t> in_strides(rank);
  std::vector<uint32_t> out_strides(rank);
  for (size_t axis = 0; axis < rank; ++axis) {
    uint32_t extent = static_cast<uint32_t>(collapsed_shape[axis]);
    in_strides[axis] = static_cast<uint32_t>(collapsed_strides[0][axis]);
    out_strides[axis] = static_cast<uint32_t>(collapsed_strides[1][axis]);
    lhs_span += static_cast<uint64_t>(extent - 1u) * in_strides[axis];
    rhs_span += static_cast<uint64_t>(extent - 1u) * out_strides[axis];
  }
  if (!omarchy::compute_index_span_fits(params.lhs_offset, lhs_span + 1) ||
      !omarchy::compute_index_span_fits(params.rhs_offset, rhs_span + 1)) {
    omarchy::unsupported(error_name + " index span", out);
  }
  std::vector<uint32_t> words;
  words.reserve(3 * rank);
  for (size_t axis = 0; axis < rank; ++axis) {
    words.push_back(static_cast<uint32_t>(collapsed_shape[axis]));
  }
  words.insert(words.end(), in_strides.begin(), in_strides.end());
  words.insert(words.end(), out_strides.begin(), out_strides.end());
  array metadata(Shape{static_cast<int>(words.size())}, uint32, nullptr, {});
  array::Flags flags;
  flags.contiguous = true;
  flags.row_contiguous = true;
  flags.col_contiguous = true;
  metadata.set_data(
      allocate_omarchy(metadata.nbytes()),
      metadata.size(),
      Strides{1},
      flags,
      0);
  auto* metadata_buffer =
      static_cast<omarchy::VulkanBuffer*>(metadata.buffer().ptr());
  std::memcpy(metadata_buffer->data, words.data(), metadata.nbytes());
  encoder.add_temporary(metadata);
  params.matrix_k = static_cast<uint32_t>(rank);
  return metadata;
}

// The params fill and dispatch behind dispatch_elementwise, callable
// with a caller-allocated output so multi-output primitives (DivMod)
// can target each output in turn.
void dispatch_float_elementwise_to(
    const std::string& name,
    uint32_t operation,
    const array& lhs,
    const array& rhs,
    array& out,
    bool general_broadcast,
    omarchy::CommandEncoder& encoder) {
  uint32_t count = checked_u32(out.size(), name, out);
  if (std::getenv("MLX_OMARCHY_TRACE_DISPATCH") != nullptr) {
    fprintf(stderr, "[rtmod] FEW name=%s count=%u\n", name.c_str(), count);
    fflush(stderr);
  }
  omarchy::ComputeParams params;
  params.count = count;
  params.operation = operation;
  params.lhs_size = checked_u32(lhs.data_size(), name, out);
  params.rhs_size = checked_u32(rhs.data_size(), name, out);
  params.output_size = count;
  params.lhs_offset = checked_item_offset(lhs, params.lhs_size, name, out);
  params.rhs_offset = checked_item_offset(rhs, params.rhs_size, name, out);
  params.output_offset = checked_item_offset(out, count, name, out);
  std::optional<array> axis_metadata;
  if (general_broadcast) {
    axis_metadata = fill_broadcast_transport(name, params, lhs, rhs, out, encoder);
  }
  std::array<omarchy::ComputeBinding, 4> bindings{
      binding(lhs),
      binding(rhs),
      binding(out),
      binding(axis_metadata ? *axis_metadata : out)};
  // Four-wide fast path (shaders/binary_vec.comp) for the hot binary
  // ops on 16-bit storage: same math and modulo addressing as
  // elementwise.comp, 8-byte vector loads. Everything else, including
  // a one-element scalar operand, keeps the general kernel.
  const bool vec_op = operation == AddOperation ||
      operation == MultiplyOperation || operation == DivideOperation ||
      operation == SubtractOperation;
  if (vec_op && !general_broadcast && out.dtype() != float32 &&
      ((count | params.lhs_size | params.rhs_size | params.lhs_offset |
        params.rhs_offset | params.output_offset) & 3u) == 0u) {
    encoder.dispatch_compute(
        out.dtype() == float16 ? omarchy::ComputeKernel::BinaryVecF16
                               : omarchy::ComputeKernel::BinaryVecBF16,
        bindings,
        params,
        omarchy::compute_dispatch_group_count(count / 4u));
    return;
  }
  auto kernel = select_float_kernel(
      out.dtype(),
      omarchy::ComputeKernel::ElementwiseF32,
      omarchy::ComputeKernel::ElementwiseF16,
      omarchy::ComputeKernel::ElementwiseBF16);
  encoder.dispatch_compute(
      kernel, bindings, params, omarchy::compute_dispatch_group_count(count));
}

void dispatch_elementwise(
    const std::string& name,
    uint32_t operation,
    const std::vector<array>& inputs,
    array& out,
    const Stream& s) {
  const array& in_lhs = inputs.at(0);
  const bool binary = inputs.size() == 2;
  const array& in_rhs = binary ? inputs.at(1) : in_lhs;
  auto& encoder = omarchy::get_command_encoder(s);
  require_float_dtype(name, in_lhs, out, encoder);
  require_float_dtype(name, in_rhs, out, encoder);

  std::optional<array> lhs_temp;
  std::optional<array> rhs_temp;
  const array& lhs =
      ensure_dense(in_lhs, in_lhs.flags().contiguous, lhs_temp, encoder, s);
  const array& rhs = binary
      ? ensure_dense(in_rhs, in_rhs.flags().contiguous, rhs_temp, encoder, s)
      : lhs;


  // Scalar, suffix-aligned, and full-overlap operands keep the cheap
  // modulo indexing in the shader; anything else needs shape-aware
  // stride indexing. Gapless strided views (transposes) keep
  // flags().contiguous, so unary ops route through the stride path the
  // same way binary broadcasts do.
  bool general_broadcast = binary
      ? (!is_trailing_broadcast(lhs, out) || !is_trailing_broadcast(rhs, out))
      : !is_trailing_broadcast(lhs, out);

  if (binary) {
    auto binary_type = get_binary_op_type(lhs, rhs);
    // Broadcast views inherit contiguous=true while data_size < size, so
    // the Scalar/Vector output branches mirror the view's undersized
    // buffer or donate it. General always allocates dense output storage
    // and never donates a non-row-contiguous view.
    if (lhs.data_size() != lhs.size() || rhs.data_size() != rhs.size()) {
      binary_type = BinaryOpType::General;
    }
    set_binary_op_output_data(
        lhs, rhs, out, binary_type, allocate_omarchy);
  } else if (general_broadcast) {
    // The stride gather must not donate: it reads and writes the same
    // buffer at different indices.
    out.set_data(allocate_omarchy(out.nbytes()));
  } else if (lhs.data_size() != lhs.size()) {
    // A scalar broadcast view passes is_trailing_broadcast, but
    // set_unary_output_data would mirror its one-element buffer; the
    // output needs dense storage for all count writes.
    out.set_data(allocate_omarchy(out.nbytes()));
  } else {
    set_unary_output_data(lhs, out, allocate_omarchy);
  }

  if (out.size() == 0) {
    return;
  }
  dispatch_float_elementwise_to(
      name, operation, lhs, rhs, out, general_broadcast, encoder);
}

// The comparison family shares one shape: a bool output from two
// broadcast views of one input dtype, word-packed through the 32-bit
// bool transport. The codes match the selector in compare.comp.
// Equal serves the categorical sampler chain (isinf,
// mx.random.categorical); GreaterEqual keeps its int32 causal-mask
// contract.
enum ComparisonOperation : uint32_t {
  CompareEqual,
  CompareGreaterEqual,
  CompareGreater,
  CompareLess,
  CompareLessEqual,
  CompareNotEqual,
};

// Forward declaration: dispatch_logical is defined after this function but
// must be callable from the bool-input branch below.
void dispatch_logical(
    const std::string& name,
    uint32_t operation,
    const std::vector<array>& inputs,
    array& out);

// The params fill and dispatch behind the bool-input comparisons and
// the bool Power op: two bool inputs, a word-packed bool output. It
// uses a plain word store (no atomicOr, no zero-fill requirement) and
// a 7-op selector, avoiding both the Honeykrisp wide-selector
// store-coalescing miscompile and the shared-accumulator hazard a
// logical_or.comp extension would carry. The masked C++ cases (test
// array basics, test array types, gguf metadata, is close, random
// split, vmap comparison ops) route here, as does Power over bool.
void dispatch_compare_bool_to(
    const std::string& name,
    uint32_t bool_op,
    const array& lhs,
    const array& rhs,
    array& out,
    omarchy::CommandEncoder& encoder) {
  out.set_data(allocate_omarchy(out.nbytes()));
  if (out.size() == 0) {
    return;
  }
  uint32_t count = checked_u32(out.size(), name, out);
  uint32_t word_count = checked_u32(
      (static_cast<uint64_t>(count) + 3) / 4, name, out);
  omarchy::ComputeParams params;
  params.count = count;
  params.operation = bool_op;
  params.lhs_size = checked_u32(lhs.data_size(), name, out);
  params.rhs_size = checked_u32(rhs.data_size(), name, out);
  params.output_size = count;
  params.lhs_offset = checked_item_offset(lhs, params.lhs_size, name, out);
  params.rhs_offset = checked_item_offset(rhs, params.rhs_size, name, out);
  params.output_offset = checked_item_offset(out, count, name, out);
  bool general_broadcast =
      !is_trailing_broadcast(lhs, out) || !is_trailing_broadcast(rhs, out);
  std::optional<array> axis_metadata;
  if (general_broadcast) {
    axis_metadata = fill_broadcast_transport(name, params, lhs, rhs, out, encoder);
  }
  std::array<omarchy::ComputeBinding, 4> bindings{
      binding(lhs),
      binding(rhs),
      binding(out),
      binding(axis_metadata ? *axis_metadata : out)};
  encoder.dispatch_compute(
      omarchy::ComputeKernel::CompareBool,
      bindings,
      params,
      omarchy::compute_dispatch_group_count(word_count));
}

void dispatch_comparison(
    const std::string& name,
    uint32_t operation,
    const std::vector<array>& inputs,
    array& out,
    uint32_t flags = 0) {
  const array& in_lhs = inputs.at(0);
  const array& in_rhs = inputs.at(1);
  auto& encoder = omarchy::get_command_encoder(out.primitive().stream());
  if (out.dtype() != bool_) {
    omarchy::unsupported(name + " output dtype", out);
  }
  // Views with gaps or negative strides (flip, as_strided) materialize
  // through the strided-copy engine first: the transport strides are
  // unsigned, so a negative stride would wrap the span computation and
  // the kernel would read the wrong elements. Dense inputs, broadcast
  // views, and transposes keep the zero-copy path.
  std::optional<array> lhs_temp;
  std::optional<array> rhs_temp;
  const array& lhs = ensure_dense(
      in_lhs,
      in_lhs.flags().contiguous,
      lhs_temp,
      encoder,
      out.primitive().stream());
  const array& rhs = ensure_dense(
      in_rhs,
      in_rhs.flags().contiguous,
      rhs_temp,
      encoder,
      out.primitive().stream());
  if (lhs.dtype() == bool_) {
    // Map ComparisonOperation onto compare_bool selector codes.
    // ComparisonOperation: Equal=0, GreaterEqual=1, Greater=2, Less=3,
    // LessEqual=4, NotEqual=5. compare_bool: Equal=0, NotEqual=1,
    // Greater=2, Less=3, GreaterEqual=4, LessEqual=5, Power=6.
    uint bool_op;
    switch (operation) {
      case CompareEqual:        bool_op = 0u; break;
      case CompareNotEqual:     bool_op = 1u; break;
      case CompareGreater:      bool_op = 2u; break;
      case CompareLess:         bool_op = 3u; break;
      case CompareGreaterEqual: bool_op = 4u; break;
      case CompareLessEqual:    bool_op = 5u; break;
      default: omarchy::unsupported(name + " bool selector", out);
    }
    dispatch_compare_bool_to(name, bool_op, lhs, rhs, out, encoder);
    return;
  }
  // uint64 rides the shaderInt64-capable CompareU64 blob (value-tested
  // against the CPU stream); the narrow int family uses the word-lane
  // variants.
  if (lhs.dtype() != rhs.dtype() ||
      (lhs.dtype() != float32 && lhs.dtype() != float16 &&
       lhs.dtype() != bfloat16 && lhs.dtype() != int32 &&
       lhs.dtype() != uint32 && lhs.dtype() != int64 &&
       lhs.dtype() != uint64 &&
       lhs.dtype() != int8 && lhs.dtype() != uint8 &&
       lhs.dtype() != int16 && lhs.dtype() != uint16 &&
       lhs.dtype() != complex64)) {
    omarchy::unsupported(name + " dtype", out);
  }
  // The 8-bit variants decode packed byte lanes from uint32 words; the
  // 16-bit variants need 16-bit storage.
  if ((lhs.dtype() == int16 || lhs.dtype() == uint16) &&
      (!encoder.device().capabilities().storage_buffer_16bit_access ||
       !encoder.device().capabilities().shader_int16)) {
    omarchy::unsupported(name + " 16-bit capability", out);
  }
  const auto& capabilities = encoder.device().capabilities();
  if (lhs.dtype() == float16 &&
      (!capabilities.shader_float16 ||
       !capabilities.storage_buffer_16bit_access)) {
    omarchy::unsupported(name + " float16 capability", out);
  }
  if (lhs.dtype() == bfloat16 &&
      (!capabilities.storage_buffer_16bit_access ||
       !capabilities.shader_int16)) {
    omarchy::unsupported(name + " bfloat16 capability", out);
  }
  // 64-bit loads and compares need the device feature; the compare
  // shader uses int64_t/uint64_t storage directly.
  if ((lhs.dtype() == int64 || lhs.dtype() == uint64) &&
      !capabilities.shader_int64) {
    omarchy::unsupported(name + " int64 capability", out);
  }

  out.set_data(allocate_omarchy(out.nbytes()));
  if (out.size() == 0) {
    return;
  }
  uint32_t count = checked_u32(out.size(), name, out);
  uint32_t word_count = checked_u32(
      (static_cast<uint64_t>(count) + 3) / 4, name, out);
  omarchy::ComputeParams params;
  params.count = count;
  params.operation = operation;
  params.flags = flags;
  params.lhs_size = checked_u32(lhs.data_size(), name, out);
  params.rhs_size = checked_u32(rhs.data_size(), name, out);
  params.output_size = count;
  params.lhs_offset = checked_item_offset(lhs, params.lhs_size, name, out);
  params.rhs_offset = checked_item_offset(rhs, params.rhs_size, name, out);
  params.output_offset = checked_item_offset(out, count, name, out);
  bool general_broadcast =
      !is_trailing_broadcast(lhs, out) || !is_trailing_broadcast(rhs, out);
  std::optional<array> axis_metadata;
  if (general_broadcast) {
    axis_metadata = fill_broadcast_transport(name, params, lhs, rhs, out, encoder);
  }
  std::array<omarchy::ComputeBinding, 4> bindings{
      binding(lhs),
      binding(rhs),
      binding(out),
      binding(axis_metadata ? *axis_metadata : out)};
  // Every admitted dtype names its kernel; a dtype with no entry refuses
  // rather than reaching a kernel that reads the wrong element width
  // (int8 once fell through to CompareF32 and overread 4x).
  omarchy::ComputeKernel kernel;
  if (lhs.dtype() == float32) {
    kernel = omarchy::ComputeKernel::CompareF32;
  } else if (lhs.dtype() == float16) {
    kernel = omarchy::ComputeKernel::CompareF16;
  } else if (lhs.dtype() == bfloat16) {
    kernel = omarchy::ComputeKernel::CompareBF16;
  } else if (lhs.dtype() == int32) {
    kernel = omarchy::ComputeKernel::CompareI32;
  } else if (lhs.dtype() == uint32) {
    kernel = omarchy::ComputeKernel::CompareU32;
  } else if (lhs.dtype() == int64) {
    kernel = omarchy::ComputeKernel::CompareI64;
  } else if (lhs.dtype() == uint64) {
    kernel = omarchy::ComputeKernel::CompareU64;
  } else if (lhs.dtype() == int8) {
    kernel = omarchy::ComputeKernel::CompareI8;
  } else if (lhs.dtype() == uint8) {
    kernel = omarchy::ComputeKernel::CompareU8;
  } else if (lhs.dtype() == int16) {
    kernel = omarchy::ComputeKernel::CompareI16;
  } else if (lhs.dtype() == uint16) {
    kernel = omarchy::ComputeKernel::CompareU16;
  } else if (lhs.dtype() == complex64) {
    kernel = omarchy::ComputeKernel::CompareComplex;
  } else {
    omarchy::unsupported(name + " dtype", out);
  }
  encoder.dispatch_compute(
      kernel,
      bindings,
      params,
      omarchy::compute_dispatch_group_count(word_count));
}

// logical_or.comp runs one invocation per output element with no
// grid-stride loop (its atomicOr byte store is the Honeykrisp-safe
// form), so an output past kMaxComputeGroupCountX * 256 elements
// dispatches in back-to-back chunks with the chunk's first element in
// matrix_m; every other parameter, including count, stays global.
// Before this loop the single capped dispatch silently left every
// element from 16,776,960 on unwritten (the zero-filled destination
// read as false): mx.where over a broadcast [2048, 2048] mask against
// eight heads lost its last four heads.
void dispatch_logical_chunked(
    omarchy::CommandEncoder& encoder,
    const std::array<omarchy::ComputeBinding, 4>& bindings,
    omarchy::ComputeParams params) {
  for (uint32_t first = 0; first < params.count;) {
    const uint32_t end = omarchy::next_logical_chunk_end(first, params.count);
    params.matrix_m = first;
    encoder.dispatch_compute(
        omarchy::ComputeKernel::LogicalOrBool,
        bindings,
        params,
        omarchy::compute_dispatch_group_count(end - first));
    first = end;
  }
}

// The logical family serves the isinf composition (Or) and now And and
// Not: bool inputs, a bool output, and the same 32-bit word transport
// the comparisons use. The operation selector matches logical_or.comp;
// Not is unary and binds its single input to both operand slots.
enum LogicalOperation : uint32_t {
  LogicalOrOperation,
  LogicalAndOperation,
  LogicalNotOperation,
};

void dispatch_logical(
    const std::string& name,
    uint32_t operation,
    const std::vector<array>& inputs,
    array& out) {
  const array& lhs = inputs.at(0);
  const bool binary = inputs.size() == 2;
  const array& rhs = binary ? inputs.at(1) : lhs;
  auto& encoder = omarchy::get_command_encoder(out.primitive().stream());
  if (lhs.dtype() != bool_ || rhs.dtype() != bool_ || out.dtype() != bool_) {
    omarchy::unsupported(name + " dtype", out);
  }
  out.set_data(allocate_omarchy(out.nbytes()));
  if (out.size() == 0) {
    return;
  }
  // The logical kernel accumulates each canonical output byte with
  // atomicOr, so the destination must start zeroed - a fresh
  // allocation holds whatever the allocator recycled. Probe-proven on
  // llvmpipe: isinf([0, 1, NaN]) returned [0, 0, 128] with a stale
  // 0x80 byte, poisoning the composed isclose chain. Same pattern as
  // the scatter bool materialization path.
  {
    uint32_t zero_words = checked_u32(
        (static_cast<uint64_t>(out.size()) + 3) / 4, name, out);
    omarchy::ComputeParams clear_params;
    clear_params.count = zero_words;
    std::array<omarchy::ComputeBinding, 1> clear_bindings{binding(out)};
    encoder.dispatch_compute(
        omarchy::ComputeKernel::ClearU32,
        clear_bindings,
        clear_params,
        omarchy::compute_dispatch_group_count(zero_words));
  }
  // One invocation per output byte: the shader's index gate is
  // params.count, in elements, and its atomicOr store exists because
  // Honeykrisp coalesces adjacent word stores and drops the inner
  // three lanes' values - atomicOr on the output word bypasses that
  // path. The grid must therefore cover the element count; a
  // word-sized grid leaves every element past word_count unwritten.
  uint32_t count = checked_u32(out.size(), name, out);
  omarchy::ComputeParams params;
  params.count = count;
  params.operation = operation;
  params.lhs_size = checked_u32(lhs.data_size(), name, out);
  params.rhs_size = checked_u32(rhs.data_size(), name, out);
  params.output_size = count;
  params.lhs_offset = checked_item_offset(lhs, params.lhs_size, name, out);
  params.rhs_offset = checked_item_offset(rhs, params.rhs_size, name, out);
  params.output_offset = checked_item_offset(out, count, name, out);
  bool general_broadcast =
      !is_trailing_broadcast(lhs, out) || !is_trailing_broadcast(rhs, out);
  std::optional<array> axis_metadata;
  if (general_broadcast) {
    axis_metadata = fill_broadcast_transport(name, params, lhs, rhs, out, encoder);
  }
  std::array<omarchy::ComputeBinding, 4> bindings{
      binding(lhs),
      binding(rhs),
      binding(out),
      binding(axis_metadata ? *axis_metadata : out)};
  dispatch_logical_chunked(encoder, bindings, params);
}

// Integer twin of the binary elementwise path. The shader carries the
enum IntElementwiseOperation : uint32_t {
  IntSubtractOperation,
  IntBitwiseAndOperation,
  IntBitwiseOrOperation,
  IntBitwiseXorOperation,
  IntLeftShiftOperation,
  IntRightShiftOperation,
  IntInvertOperation,
  IntDivModQuotientOperation,
  IntModuloOperation,
  IntPowerOperation,
  IntSignOperation,
  IntAbsOperation,
  IntAddOperation,
  IntMultiplyOperation,
  IntSquareOperation,
  IntMinimumOperation,
  IntMaximumOperation,
  IntDivideOperation,
  IntNegateOperation,
};

// Every integer dtype the elementwise family serves. Bool rides the
// byte lanes too, but only for the logical ops (the call sites gate
// that subset), so it stays out of this check.
bool is_int_elementwise_dtype(Dtype dtype) {
  return dtype == int8 || dtype == uint8 || dtype == int16 ||
      dtype == uint16 || dtype == int32 || dtype == uint32 ||
      dtype == int64 || dtype == uint64;
}

// The int elementwise blob per output dtype: every variant carries the
// full op set - the 32-bit word kernels inline, the 8/16/64-bit
// widened kernels through the shared INT_ARITH_CASES block.
omarchy::ComputeKernel elementwise_kernel(Dtype dtype) {
  switch (dtype) {
    case int8:
      return omarchy::ComputeKernel::ElementwiseI8;
    case uint8:
    case bool_:
      // Bool arrays are one byte per element, so the unsigned byte
      // lanes serve them; values are 0/1 and only the call-site-gated
      // logical ops reach this kernel.
      return omarchy::ComputeKernel::ElementwiseU8;
    case int16:
      return omarchy::ComputeKernel::ElementwiseI16;
    case uint16:
      return omarchy::ComputeKernel::ElementwiseU16;
    case int64:
      return omarchy::ComputeKernel::ElementwiseI64;
    case uint64:
      return omarchy::ComputeKernel::ElementwiseU64;
    case uint32:
      return omarchy::ComputeKernel::ElementwiseU32;
    default:
      return omarchy::ComputeKernel::ElementwiseI32;
  }
}

// The params fill and dispatch behind dispatch_int_elementwise,
// callable with a caller-allocated output so the two-output DivMod can
// target quotient and remainder in turn.
void dispatch_int_elementwise_to(
    const std::string& name,
    uint32_t operation,
    const array& lhs,
    const array& rhs,
    array& out) {
  auto& encoder = omarchy::get_command_encoder(out.primitive().stream());
  uint32_t count = checked_u32(out.size(), name, out);
  omarchy::ComputeParams params;
  params.count = count;
  params.operation = operation;
  params.lhs_size = checked_u32(lhs.data_size(), name, out);
  params.rhs_size = checked_u32(rhs.data_size(), name, out);
  params.output_size = count;
  params.lhs_offset = checked_item_offset(lhs, params.lhs_size, name, out);
  params.rhs_offset = checked_item_offset(rhs, params.rhs_size, name, out);
  params.output_offset = checked_item_offset(out, count, name, out);
  std::optional<array> axis_metadata;
  if (!is_trailing_broadcast(lhs, out) || !is_trailing_broadcast(rhs, out)) {
    axis_metadata = fill_broadcast_transport(name, params, lhs, rhs, out, encoder);
  }
  std::array<omarchy::ComputeBinding, 4> bindings{
      binding(lhs),
      binding(rhs),
      binding(out),
      binding(axis_metadata ? *axis_metadata : out)};
  // Signed and unsigned run separate SPIR-V variants: `>>` arithmetic
  // versus logical, and the sign fixups compare against a signed zero.
  // The widened variants serve the 8/16/64-bit integer family with the
  // widened-word bitwise ops only.
  auto kernel = elementwise_kernel(out.dtype());
  encoder.dispatch_compute(
      kernel,
      bindings,
      params,
      omarchy::compute_dispatch_group_count(count));
}

void dispatch_int_elementwise(
    const std::string& name,
    uint32_t operation,
    const std::vector<array>& inputs,
    array& out) {
  const array& in_lhs = inputs.at(0);
  const bool binary = inputs.size() == 2;
  const array& in_rhs = binary ? inputs.at(1) : in_lhs;
  auto is_word_dtype = [](Dtype dtype) {
    return dtype == int32 || dtype == uint32;
  };
  // The 8/16/64-bit family runs the full operation set on the widened
  // variants; the inputs must already share the output dtype (upstream
  // astype's both operands to the result type), and anything mixed
  // keeps the named refusal.
  auto is_widened_dtype = [](Dtype dtype) {
    return dtype == int8 || dtype == uint8 || dtype == int16 ||
        dtype == uint16 || dtype == int64 || dtype == uint64;
  };
  // Bool rides the unsigned byte lanes (values are 0/1); only the
  // logical ops compute: BitwiseAnd/Or/Xor, Add as the logical or,
  // Maximum/Minimum, plus Abs and Sign (upstream identity and x != 0,
  // both the byte itself). Everything else keeps the named refusal.
  if (out.dtype() == bool_) {
    switch (operation) {
      case IntBitwiseAndOperation:
      case IntBitwiseOrOperation:
      case IntBitwiseXorOperation:
      case IntAddOperation:
      case IntMaximumOperation:
      case IntMinimumOperation:
      case IntAbsOperation:
      case IntSignOperation:
        break;
      default:
        omarchy::unsupported(name + " dtype", out);
    }
    if (in_lhs.dtype() != bool_ ||
        (binary && in_rhs.dtype() != bool_)) {
      omarchy::unsupported(name + " dtype", out);
    }
  } else if (is_widened_dtype(in_lhs.dtype()) ||
             is_widened_dtype(in_rhs.dtype()) ||
             is_widened_dtype(out.dtype())) {
    if (in_lhs.dtype() != out.dtype() ||
        (binary && in_rhs.dtype() != out.dtype())) {
      omarchy::unsupported(name + " dtype", out);
    }
  } else if (!is_word_dtype(in_lhs.dtype()) ||
             !is_word_dtype(in_rhs.dtype()) ||
             !is_word_dtype(out.dtype())) {
    omarchy::unsupported(name + " dtype", out);
  }
  auto& encoder = omarchy::get_command_encoder(out.primitive().stream());
  if (out.dtype() == int16 || out.dtype() == uint16) {
    const auto& capabilities = encoder.device().capabilities();
    if (!capabilities.storage_buffer_16bit_access ||
        !capabilities.shader_int16) {
      omarchy::unsupported(name + " 16-bit capability", out);
    }
  }
  if ((out.dtype() == int64 || out.dtype() == uint64) &&
      !encoder.device().capabilities().shader_int64) {
    omarchy::unsupported(name + " int64 capability", out);
  }
  std::optional<array> lhs_temp;
  std::optional<array> rhs_temp;
  const array& lhs = ensure_dense(
      in_lhs,
      in_lhs.flags().contiguous,
      lhs_temp,
      encoder,
      out.primitive().stream());
  const array& rhs = binary
      ? ensure_dense(
            in_rhs,
            in_rhs.flags().contiguous,
            rhs_temp,
            encoder,
            out.primitive().stream())
      : lhs;
  auto binary_type = get_binary_op_type(lhs, rhs);
  // Broadcast views inherit contiguous=true while data_size < size; the
  // Scalar/Vector output branches would mirror the view's undersized
  // buffer or donate it. General always allocates dense output storage.
  if (lhs.data_size() != lhs.size() || rhs.data_size() != rhs.size()) {
    binary_type = BinaryOpType::General;
  }
  set_binary_op_output_data(lhs, rhs, out, binary_type, allocate_omarchy);
  if (out.size() == 0) {
    return;
  }
  dispatch_int_elementwise_to(name, operation, lhs, rhs, out);
}

// Sort and ArgSort accept float32/float16/bfloat16/complex64 plus the
// 8/16/32-bit integer family; the 64-bit sorts keep the named refusal.
// ArgSort and ArgPartition emit uint32 indices, so the output must be uint32 and
// the dtype checks apply to the input only, the way ArgReduce checks
// its input. The value variants Sort and Partition keep the input
// dtype in the output.
void require_sort_dtype(
    const std::string& name,
    const array& input,
    const array& out,
    bool argsort,
    omarchy::CommandEncoder& encoder) {
  bool sortable_int =
      input.dtype() == int8 || input.dtype() == uint8 ||
      input.dtype() == int16 || input.dtype() == uint16 ||
      input.dtype() == int32 || input.dtype() == uint32;
  if (input.dtype() != float16 && input.dtype() != float32 &&
      input.dtype() != bfloat16 && input.dtype() != complex64 &&
      !sortable_int) {
    omarchy::unsupported(name + " dtype", out);
  }
  const auto& capabilities = encoder.device().capabilities();
  if (input.dtype() == float16 &&
      (!capabilities.shader_float16 ||
       !capabilities.storage_buffer_16bit_access)) {
    omarchy::unsupported(name + " float16 capability", out);
  }
  if (input.dtype() == bfloat16 &&
      (!capabilities.storage_buffer_16bit_access ||
       !capabilities.shader_int16)) {
    omarchy::unsupported(name + " bfloat16 capability", out);
  }
  if ((input.dtype() == int16 || input.dtype() == uint16) &&
      (!capabilities.storage_buffer_16bit_access ||
       !capabilities.shader_int16)) {
    omarchy::unsupported(name + " 16-bit capability", out);
  }
  if (argsort) {
    if (out.dtype() != uint32) {
      omarchy::unsupported(name + " output dtype", out);
    }
  } else if (input.dtype() != out.dtype()) {
    omarchy::unsupported(name + " dtype", out);
  }
}

// One workgroup bitonic-sorts one row of up to 1024 elements, and the
// wide-row stage below sorts longer rows by slicing each padded row into
// 1024-element chunks. Partition and ArgPartition route here too: a full
// sort satisfies the partition contract, the same redirect the upstream
// Metal backend makes.
constexpr size_t kSortMaxRowLength = 1024;

void dispatch_sort_wide(
    const std::string& name,
    const array& src,
    array& out,
    bool argsort,
    omarchy::CommandEncoder& encoder,
    const Stream& s);
void dispatch_sort(
    const std::string& name,
    const array& input,
    array& out,
    bool argsort,
    omarchy::CommandEncoder& encoder,
    const Stream& s) {
  // |s| comes from the caller's primitive-bearing output array. The
  // any-axis wrapper dispatches on temps built with no primitive, so
  // out.primitive().stream() is not a legal way to recover the stream
  // down here.
  std::optional<array> dense_temp;
  const array& src = ensure_dense(
      input,
      input.flags().row_contiguous,
      dense_temp,
      encoder,
      s);
  size_t row_length = src.shape(-1);
  if (row_length > kSortMaxRowLength) {
    dispatch_sort_wide(name, src, out, argsort, encoder, s);
    return;
  }
  out.set_data(allocate_omarchy(out.nbytes()));
  if (out.size() == 0) {
    return;
  }
  size_t rows = src.size() / row_length;
  uint32_t output_size = checked_u32(rows, name, out);
  omarchy::ComputeParams params;
  params.count = checked_u32(out.size(), name, out);
  params.reduce_size = checked_u32(row_length, name, out);
  params.output_size = output_size;
  params.lhs_offset = checked_item_offset(src, src.size(), name, out);
  params.output_offset = checked_item_offset(out, out.size(), name, out);
  std::array<omarchy::ComputeBinding, 3> bindings{
      binding(src), binding(src), binding(out)};
  omarchy::ComputeKernel kernel;
  switch (input.dtype()) {
    case int32:
      kernel = argsort ? omarchy::ComputeKernel::ArgSortI32
                       : omarchy::ComputeKernel::SortI32;
      break;
    case uint32:
      kernel = argsort ? omarchy::ComputeKernel::ArgSortU32
                       : omarchy::ComputeKernel::SortU32;
      break;
    case complex64:
      kernel = argsort ? omarchy::ComputeKernel::ArgSortC64
                       : omarchy::ComputeKernel::SortC64;
      break;
    case int8:
      // Byte rows ride uint32 words with the sign-bit-flip key map.
      kernel = argsort ? omarchy::ComputeKernel::ArgSortI8
                       : omarchy::ComputeKernel::SortI8;
      break;
    case uint8:
      kernel = argsort ? omarchy::ComputeKernel::ArgSortU8
                       : omarchy::ComputeKernel::SortU8;
      break;
    case int16:
      kernel = argsort ? omarchy::ComputeKernel::ArgSortI16
                       : omarchy::ComputeKernel::SortI16;
      break;
    case uint16:
      kernel = argsort ? omarchy::ComputeKernel::ArgSortU16
                       : omarchy::ComputeKernel::SortU16;
      break;
    default:
      kernel = argsort ? select_float_kernel(
                             input.dtype(),
                             omarchy::ComputeKernel::ArgSortF32,
                             omarchy::ComputeKernel::ArgSortF16,
                             omarchy::ComputeKernel::ArgSortBF16)
                       : select_float_kernel(
                             input.dtype(),
                             omarchy::ComputeKernel::SortF32,
                             omarchy::ComputeKernel::SortF16,
                             omarchy::ComputeKernel::SortBF16);
      break;
  }
  encoder.dispatch_compute(
      kernel,
      bindings,
      params,
      std::min(output_size, omarchy::kMaxComputeGroupCountX));
}

// Wide-row sort: the multi-block continuation of the suffix kernel. Each
// padded row is a power-of-two length P with the PAD sentinel in
// [row_length, P), sliced into 1024-element chunks that the suffix
// kernel sorts. Chunk mode runs the chunk-local bitonic network with the
// final k stage directed by chunk parity, which is exactly the state the
// k <= 1024 stages of the full bitonic network leave behind: even chunks
// ascending, odd chunks descending. One merge dispatch per network stage
// (block k, sub-stage j) then sorts each padded row in place over global
// memory - a compare-exchange stage touches disjoint pairs, so no
// ping-pong buffer is needed - and the real prefix is copied out. Any
// row length sorts; the hard ceilings are the backend-wide u32
// element-count and index-span limits plus the int-typed Shape dims, so
// a padded element count over INT32_MAX refuses by name instead of
// wrapping.
void dispatch_sort_wide(
    const std::string& name,
    const array& src,
    array& out,
    bool argsort,
    omarchy::CommandEncoder& encoder,
    const Stream& s) {
  size_t row_length = src.shape(-1);
  size_t rows = src.size() / row_length;
  size_t padded = 1;
  while (padded < row_length) {
    padded <<= 1;
  }
  constexpr size_t chunk = kSortMaxRowLength;
  size_t chunks_per_row = padded / chunk;
  if (rows * padded >
      static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
    omarchy::unsupported(name + " row element count", out);
  }
  uint32_t padded_elems = checked_u32(rows * padded, name, out);
  uint32_t chunk_count = checked_u32(rows * chunks_per_row, name, out);

  out.set_data(allocate_omarchy(out.nbytes()));
  if (out.size() == 0) {
    return;
  }

  // The chunk stage always sorts a value pipeline. ArgSort adds an index
  // pipeline; complex value sort also keeps it because distinct NaNs are
  // equivalent keys whose source order remains observable in the output.
  omarchy::ComputeKernel value_kernel;
  omarchy::ComputeKernel arg_kernel;
  omarchy::ComputeKernel merge_kernel;
  if (src.dtype() == int32 || src.dtype() == uint32) {
    value_kernel = src.dtype() == int32 ? omarchy::ComputeKernel::SortI32
                                        : omarchy::ComputeKernel::SortU32;
    arg_kernel = src.dtype() == int32 ? omarchy::ComputeKernel::ArgSortI32
                                      : omarchy::ComputeKernel::ArgSortU32;
    merge_kernel = argsort
        ? (src.dtype() == int32 ? omarchy::ComputeKernel::ArgSortMergeI32
                                : omarchy::ComputeKernel::ArgSortMergeU32)
        : (src.dtype() == int32 ? omarchy::ComputeKernel::SortMergeI32
                                : omarchy::ComputeKernel::SortMergeU32);
  } else if (src.dtype() == complex64) {
    value_kernel = omarchy::ComputeKernel::SortC64;
    arg_kernel = omarchy::ComputeKernel::ArgSortC64;
    merge_kernel = omarchy::ComputeKernel::ArgSortMergeC64;
  } else {
    value_kernel = select_float_kernel(
        src.dtype(),
        omarchy::ComputeKernel::SortF32,
        omarchy::ComputeKernel::SortF16,
        omarchy::ComputeKernel::SortBF16);
    arg_kernel = select_float_kernel(
        src.dtype(),
        omarchy::ComputeKernel::ArgSortF32,
        omarchy::ComputeKernel::ArgSortF16,
        omarchy::ComputeKernel::ArgSortBF16);
    merge_kernel = argsort
        ? select_float_kernel(
              src.dtype(),
              omarchy::ComputeKernel::ArgSortMergeF32,
              omarchy::ComputeKernel::ArgSortMergeF16,
              omarchy::ComputeKernel::ArgSortMergeBF16)
        : select_float_kernel(
              src.dtype(),
              omarchy::ComputeKernel::SortMergeF32,
              omarchy::ComputeKernel::SortMergeF16,
              omarchy::ComputeKernel::SortMergeBF16);
  }

  // The chunk stage reads the source copy and writes the sorted keys
  // (plus source indices when the merge must preserve them). The
  // sentinel word pads every dtype: float words become NaN keys, both
  // complex components become NaN, int32 reads 0x7fffffff whose sign-flip
  // mapping is the largest key, and unsigned words are already maximal.
  // The suffix kernel binds its input read-only and its output
  // write-only, so the chunk stage needs a second padded buffer: the
  // pad-and-copy lands in |padded_src| and the sorted chunks land in
  // |keys|, where the merge stages then run in place.
  array padded_src(
      Shape{static_cast<int>(rows), static_cast<int>(padded)},
      src.dtype(),
      nullptr,
      {});
  array keys(
      Shape{static_cast<int>(rows), static_cast<int>(padded)},
      src.dtype(),
      nullptr,
      {});
  padded_src.set_data(allocate_omarchy(padded_src.nbytes()));
  keys.set_data(allocate_omarchy(keys.nbytes()));
  encoder.add_temporary(padded_src);
  encoder.add_temporary(keys);
  uint32_t pad_word = src.dtype() == int32 ? 0x7fffffffu : 0xffffffffu;
  auto* src_storage =
      static_cast<omarchy::VulkanBuffer*>(padded_src.buffer().ptr());
  encoder.fill_buffer(src_storage->buffer, pad_word, padded_src.nbytes(), 0);
  copy_gpu_inplace(
      src,
      padded_src,
      Shape{static_cast<int>(rows), static_cast<int>(row_length)},
      Strides{static_cast<int>(row_length), 1},
      Strides{static_cast<int>(padded), 1},
      /* i_offset = */ 0,
      /* o_offset = */ 0,
      CopyType::GeneralGeneral,
      s);
  bool track_indices = argsort || src.dtype() == complex64;
  array idx = array(Shape{0}, uint32, nullptr, {});
  if (track_indices) {
    idx = array(
        Shape{static_cast<int>(rows), static_cast<int>(padded)},
        uint32,
        nullptr,
        {});
    idx.set_data(allocate_omarchy(idx.nbytes()));
    encoder.add_temporary(idx);
  }

  // In-block stage: one suffix dispatch sorts every 1024-element chunk.
  // rhs_size carries the chunks-per-row count that directs each chunk's
  // final stage and, for argsort, turns chunk-local positions into row
  // positions.
  omarchy::ComputeParams params;
  params.count = padded_elems;
  params.reduce_size = static_cast<uint32_t>(chunk);
  params.rhs_size = checked_u32(chunks_per_row, name, out);
  params.output_size = chunk_count;
  std::array<omarchy::ComputeBinding, 3> sort_bindings{
      binding(padded_src), binding(padded_src), binding(keys)};
  encoder.dispatch_compute(
      value_kernel,
      sort_bindings,
      params,
      std::min(chunk_count, omarchy::kMaxComputeGroupCountX));
  if (track_indices) {
    std::array<omarchy::ComputeBinding, 3> arg_bindings{
        binding(padded_src), binding(padded_src), binding(idx)};
    encoder.dispatch_compute(
        arg_kernel,
        arg_bindings,
        params,
        std::min(chunk_count, omarchy::kMaxComputeGroupCountX));
  }

  // Merge stages: one in-place dispatch per (k, j) pair, with the index
  // buffer in lockstep when stable source order remains observable.
  omarchy::ComputeParams merge_params;
  merge_params.count = padded_elems;
  merge_params.reduce_size = checked_u32(padded, name, out);
  for (uint64_t k = 2 * chunk; k <= padded; k <<= 1) {
    for (uint64_t j = k >> 1; j >= 1; j >>= 1) {
      merge_params.lhs_size = static_cast<uint32_t>(k);
      merge_params.rhs_size = static_cast<uint32_t>(j);
      if (track_indices) {
        std::array<omarchy::ComputeBinding, 2> merge_bindings{
            binding(keys), binding(idx)};
        encoder.dispatch_compute(
            merge_kernel,
            merge_bindings,
            merge_params,
            omarchy::compute_dispatch_group_count(padded_elems));
      } else {
        std::array<omarchy::ComputeBinding, 1> merge_bindings{binding(keys)};
        encoder.dispatch_compute(
            merge_kernel,
            merge_bindings,
            merge_params,
            omarchy::compute_dispatch_group_count(padded_elems));
      }
    }
  }

  // The first row_length positions of each padded row are the sorted
  // row: values for the value variant, source indices for argsort.
  const array& result = argsort ? idx : keys;
  copy_gpu_inplace(
      result,
      out,
      Shape{static_cast<int>(rows), static_cast<int>(row_length)},
      Strides{static_cast<int>(padded), 1},
      Strides{static_cast<int>(row_length), 1},
      /* i_offset = */ 0,
      /* o_offset = */ 0,
      CopyType::GeneralGeneral,
      s);
}
// Any-axis sort family. The suffix kernel sorts one row-contiguous row
// per workgroup, so a non-suffix axis rides the general strided-copy
// engine: move the sorted axis to the end (the same materialization a
// Transpose node takes), run the suffix kernel, and write the result
// back through the inverse permutation. Values keep the input dtype;
// argsort writes uint32 indices.
struct AxisMoveTables {
  Shape shape;
  Strides in_strides;
  Strides moved_strides;
};

// Shape tables that move `axis` of `input` to the end: the moved shape,
// the input strides read in moved order, and the moved shape's
// row-major strides for the dense temp.
AxisMoveTables axis_move_tables(const array& input, int axis) {
  AxisMoveTables tables;
  for (int i = 0; i < input.ndim(); ++i) {
    if (i == axis) {
      continue;
    }
    tables.shape.push_back(input.shape(i));
    tables.in_strides.push_back(input.strides()[i]);
  }
  tables.shape.push_back(input.shape(axis));
  tables.in_strides.push_back(input.strides()[axis]);
  // The dense temp is read as row-major rows by the suffix kernel, so
  // its strides run from the last moved dim backwards.
  tables.moved_strides = Strides(tables.shape.size(), 1);
  for (int i = static_cast<int>(tables.shape.size()) - 2; i >= 0; --i) {
    tables.moved_strides[i] =
        tables.moved_strides[i + 1] * tables.shape[i + 1];
  }
  return tables;
}

// The output strides that undo an axis move: the kept dims in original
// order, then the moved axis, matching the moved dim order.
Strides permuted_out_strides(const array& out, int axis) {
  Strides strides;
  for (int i = 0; i < out.ndim(); ++i) {
    if (i != axis) {
      strides.push_back(out.strides()[i]);
    }
  }
  strides.push_back(out.strides()[axis]);
  return strides;
}

void dispatch_sort_any_axis(
    const std::string& name,
    const array& input,
    array& out,
    int axis,
    bool argsort,
    omarchy::CommandEncoder& encoder) {
  // |out| carries the Sort/ArgSort/Partition primitive here, so its
  // stream is the authoritative one for every temp and sub-dispatch.
  auto& s = out.primitive().stream();
  if (axis == input.ndim() - 1) {
    dispatch_sort(name, input, out, argsort, encoder, s);
    return;
  }
  AxisMoveTables tables = axis_move_tables(input, axis);
  out.set_data(allocate_omarchy(out.nbytes()));
  array moved(tables.shape, input.dtype(), nullptr, {});
  moved.set_data(allocate_omarchy(moved.nbytes()));
  array sorted(tables.shape, out.dtype(), nullptr, {});
  if (out.size() != 0) {
    copy_gpu_inplace(
        input,
        moved,
        tables.shape,
        tables.in_strides,
        tables.moved_strides,
        /* i_offset = */ 0,
        /* o_offset = */ 0,
        CopyType::GeneralGeneral,
        s);
    dispatch_sort(name, moved, sorted, argsort, encoder, s);
    copy_gpu_inplace(
        sorted,
        out,
        tables.shape,
        tables.moved_strides,
        permuted_out_strides(out, axis),
        /* i_offset = */ 0,
        /* o_offset = */ 0,
        CopyType::GeneralGeneral,
        s);
  }
  encoder.add_temporary(moved);
  encoder.add_temporary(sorted);
}
// Last-axis softmax. The Softmax primitive is only constructed for a
// last-axis reduction (mlx/ops.cpp softmax), so no suffix-axis check is
// needed here. The shader accumulates in float32 for every dtype, which
// also covers the precise flag. ScaledDotProductAttention shares this
// dispatch for its float32 score normalization.
// causal=true selects the shader's causal mode; causal_offset is the key
// length minus query length and may be negative when the query outgrew
// the key (a ragged causal decode tail), so it can not double as the
// mode sentinel.
void dispatch_softmax(
    const std::string& name,
    const array& input,
    array& out,
    const Stream& s,
    const array* sinks = nullptr,
    int q_len = 0,
    bool causal = false,
    int causal_offset = 0) {
  auto& encoder = omarchy::get_command_encoder(s);
  require_float_dtype(name, input, out, encoder);
  std::optional<array> dense_temp;
  const array& src =
      ensure_dense(input, input.flags().row_contiguous, dense_temp, encoder, s);
  size_t row_length = src.shape(-1);
  size_t rows = src.size() / row_length;
  out.set_data(allocate_omarchy(out.nbytes()));
  if (out.size() == 0) {
    return;
  }
  uint32_t output_size = checked_u32(rows, name, out);
  omarchy::ComputeParams params;
  params.count = checked_u32(out.size(), name, out);
  params.reduce_size = checked_u32(row_length, name, out);
  params.output_size = output_size;
  params.lhs_offset = checked_item_offset(src, src.size(), name, out);
  params.output_offset = checked_item_offset(out, out.size(), name, out);
  std::optional<array> dense_sinks;
  if (sinks != nullptr) {
    sinks = &ensure_dense(
        *sinks, sinks->flags().row_contiguous, dense_sinks, encoder, s);
    params.operation = 1u;
    params.rhs_offset = checked_item_offset(*sinks, sinks->size(), name, out);
    params.matrix_m = checked_u32(q_len, name, out);
    params.matrix_n = checked_u32(sinks->size(), name, out);
  }
  if (causal) {
    params.flags = 1u;
    params.aux_size = checked_u32(q_len, name, out);
    // Two's complement on purpose: negative offsets ride the same u32
    // and the shader recovers the signed value per row.
    params.aux_offset = static_cast<uint32_t>(causal_offset);
  }
  std::array<omarchy::ComputeBinding, 3> bindings{
      binding(src),
      sinks != nullptr ? binding(*sinks) : binding(src),
      binding(out)};
  auto kernel = select_float_kernel(
      out.dtype(),
      omarchy::ComputeKernel::SoftmaxF32,
      omarchy::ComputeKernel::SoftmaxF16,
      omarchy::ComputeKernel::SoftmaxBF16);
  encoder.dispatch_compute(
      kernel,
      bindings,
      params,
      std::min(output_size, omarchy::kMaxComputeGroupCountX));
}

// The general-axis reduction family: integer dtypes, Any/All, and every
// non-suffix axis position. The selector matches the switch in
// shaders/reduce_general.comp.
enum ReduceOperation : uint32_t {
  ReduceSumOperation,
  ReduceProdOperation,
  ReduceMinOperation,
  ReduceMaxOperation,
  ReduceAnyOperation,
  ReduceAllOperation,
};

// Capability gate for a float input whose output is a different dtype;
// Any/All produce bool from float input, so require_float_dtype's
// input-equals-output rule does not apply.
void require_float_input(
    const std::string& name,
    const array& input,
    const array& out,
    omarchy::CommandEncoder& encoder) {
  const auto& capabilities = encoder.device().capabilities();
  if (input.dtype() == float16 &&
      (!capabilities.shader_float16 ||
       !capabilities.storage_buffer_16bit_access)) {
    omarchy::unsupported(name + " float16 capability", out);
  }
  if (input.dtype() == bfloat16 &&
      (!capabilities.storage_buffer_16bit_access ||
       !capabilities.shader_int16)) {
    omarchy::unsupported(name + " bfloat16 capability", out);
  }
  if ((input.dtype() == int16 || input.dtype() == uint16) &&
      (!capabilities.storage_buffer_16bit_access ||
       !capabilities.shader_int16)) {
    omarchy::unsupported(name + " 16-bit capability", out);
  }
}

omarchy::ComputeKernel select_anyall_kernel(Dtype input_dtype) {
  switch (input_dtype) {
    case bool_:
      return omarchy::ComputeKernel::AnyAllBool;
    case int32:
      return omarchy::ComputeKernel::AnyAllI32;
    case uint32:
      return omarchy::ComputeKernel::AnyAllU32;
    case float16:
      return omarchy::ComputeKernel::AnyAllF16;
    case bfloat16:
      return omarchy::ComputeKernel::AnyAllBF16;
    default:
      return omarchy::ComputeKernel::AnyAllF32;
  }
}

omarchy::ComputeKernel select_reduce_general_kernel(Dtype dtype) {
  switch (dtype) {
    case bool_:
      return omarchy::ComputeKernel::ReduceGeneralBool;
    case int8:
      return omarchy::ComputeKernel::ReduceGeneralI8;
    case uint8:
      return omarchy::ComputeKernel::ReduceGeneralU8;
    case int16:
      return omarchy::ComputeKernel::ReduceGeneralI16;
    case uint16:
      return omarchy::ComputeKernel::ReduceGeneralU16;
    case int32:
      return omarchy::ComputeKernel::ReduceGeneralI32;
    case uint32:
      return omarchy::ComputeKernel::ReduceGeneralU32;
    case int64:
      return omarchy::ComputeKernel::ReduceGeneralI64;
    case uint64:
      return omarchy::ComputeKernel::ReduceGeneralU64;
    case complex64:
      return omarchy::ComputeKernel::ReduceGeneralComplex;
    case float16:
      return omarchy::ComputeKernel::ReduceGeneralF16;
    case bfloat16:
      return omarchy::ComputeKernel::ReduceGeneralBF16;
    default:
      return omarchy::ComputeKernel::ReduceGeneralF32;
  }
}

struct ReductionAxis {
  uint32_t extent;
  uint32_t stride;
  bool reduced;
};

std::vector<ReductionAxis> collapse_reduction_axes(
    const std::string& operation_name,
    const array& input,
    const array& out,
    const std::vector<int>& axes) {
  std::vector<bool> reduction_mask(input.ndim(), false);
  for (int axis : axes) {
    if (axis < 0 || axis >= input.ndim()) {
      omarchy::unsupported(operation_name + " axes", out);
    }
    reduction_mask[axis] = true;
  }

  std::vector<ReductionAxis> collapsed;
  collapsed.reserve(input.ndim());
  for (int axis = 0; axis < input.ndim(); ++axis) {
    int extent = input.shape(axis);
    if (extent == 1) {
      continue;
    }
    int64_t raw_stride = extent == 0 ? 0 : input.strides()[axis];
    if (raw_stride < 0 ||
        static_cast<uint64_t>(raw_stride) >
            std::numeric_limits<uint32_t>::max()) {
      omarchy::unsupported(operation_name + " stride", out);
    }
    uint32_t axis_extent = checked_u32(
        static_cast<size_t>(extent), operation_name + " shape", out);
    uint32_t axis_stride = static_cast<uint32_t>(raw_stride);
    bool reduced = reduction_mask[axis];
    bool merge = !collapsed.empty() && collapsed.back().reduced == reduced &&
        axis_extent != 0 &&
        ((collapsed.back().stride == 0 && axis_stride == 0) ||
         static_cast<uint64_t>(axis_stride) * axis_extent ==
             collapsed.back().stride);
    if (!merge) {
      collapsed.push_back({axis_extent, axis_stride, reduced});
      continue;
    }
    uint64_t merged_extent =
        static_cast<uint64_t>(collapsed.back().extent) * axis_extent;
    if (merged_extent > std::numeric_limits<uint32_t>::max()) {
      omarchy::unsupported(operation_name + " shape", out);
    }
    collapsed.back().extent = static_cast<uint32_t>(merged_extent);
    collapsed.back().stride = axis_stride;
  }
  return collapsed;
}

void dispatch_reduce_general(
    const std::string& operation_name,
    uint32_t operation,
    const array& input,
    array& out,
    const std::vector<int>& axes,
    omarchy::CommandEncoder& encoder,
    omarchy::ComputeKernel kernel) {
  // Flip-style views carry negative strides; the kernel walk uses
  // unsigned offsets from the base pointer, so materialize a dense copy
  // first. Other strided views reduce in place as before.
  bool negative_strides = false;
  for (int axis = 0; axis < input.ndim(); ++axis) {
    negative_strides = negative_strides || input.strides()[axis] < 0;
  }
  std::optional<array> dense_temp;
  const array& src = ensure_dense(
      input,
      !negative_strides,
      dense_temp,
      encoder,
      out.primitive().stream());
  auto collapsed =
      collapse_reduction_axes(operation_name, src, out, axes);
  std::vector<ReductionAxis> kept;
  std::vector<ReductionAxis> reduced;
  kept.reserve(collapsed.size());
  reduced.reserve(collapsed.size());
  for (const auto& axis : collapsed) {
    (axis.reduced ? reduced : kept).push_back(axis);
  }

  uint64_t reduce_size = 1;
  for (const auto& axis : reduced) {
    reduce_size *= axis.extent;
    if (reduce_size > std::numeric_limits<uint32_t>::max()) {
      omarchy::unsupported(operation_name + " reduction size", out);
    }
  }
  out.set_data(allocate_omarchy(out.nbytes()));
  if (out.size() == 0) {
    return;
  }
  uint32_t output_size = checked_u32(out.size(), operation_name, out);
  uint32_t reduce_size_u32 = static_cast<uint32_t>(reduce_size);
  const array& bound_input = src.size() == 0 ? out : src;
  constexpr uint32_t kReduceGeneralChunkTrips = 4096;
  uint32_t chunks = reduce_size_u32 == 0
      ? 1u
      : (reduce_size_u32 + kReduceGeneralChunkTrips - 1u) /
          kReduceGeneralChunkTrips;
  if (chunks == 0) {
    chunks = 1;
  }
  bool bool_out = out.dtype() == bool_;
  // The scratch width follows the shader accumulator, not the output:
  // narrow integers accumulate in 32 bits, complex64 keeps two float
  Dtype scratch_dtype = float32;
  uint64_t words_per_partial = 1;
  switch (src.dtype()) {
    case bool_:
    case int8:
    case int16:
    case int32:
      scratch_dtype = int32;
      break;
    case uint8:
    case uint16:
    case uint32:
      scratch_dtype = uint32;
      break;
    case int64:
      scratch_dtype = int64;
      break;
    case uint64:
      scratch_dtype = uint64;
      break;
    case complex64:
      scratch_dtype = float32;
      words_per_partial = 2;
      break;
    default:
      scratch_dtype = float32;
      break;
  }
  if (bool_out) {
    scratch_dtype = uint32;
  }
  uint64_t scratch_elems =
      static_cast<uint64_t>(output_size) * chunks * words_per_partial;
  uint64_t partial_pairs = static_cast<uint64_t>(output_size) * chunks;
  if (partial_pairs > (1ull << 25)) {
    omarchy::unsupported(
        operation_name + " reduction split exceeds scratch budget", out);
  }
  array scratch(
      Shape{static_cast<int>(scratch_elems)}, scratch_dtype, nullptr, {});
  scratch.set_data(allocate_omarchy(scratch.nbytes()));
  encoder.add_temporary(scratch);

  omarchy::ComputeParams params;
  params.count = output_size;
  params.operation = operation;
  params.reduce_size = reduce_size_u32;
  params.output_size = output_size;
  params.lhs_offset = checked_item_offset(
      bound_input, src.size(), operation_name, out);
  params.output_offset = checked_item_offset(
      out, out.size(), operation_name, out);
  params.dims = checked_u32(kept.size(), operation_name + " rank", out);
  params.matrix_m =
      checked_u32(reduced.size(), operation_name + " rank", out);
  params.matrix_n = chunks;

  uint64_t input_span = 0;
  if (reduce_size_u32 != 0) {
    for (const auto& axis : collapsed) {
      uint64_t term =
          static_cast<uint64_t>(axis.extent - 1u) * axis.stride;
      if (term > std::numeric_limits<uint32_t>::max() - input_span) {
        omarchy::unsupported(operation_name + " index span", out);
      }
      input_span += term;
    }
    if (!omarchy::compute_index_span_fits(
            params.lhs_offset, input_span + 1)) {
      omarchy::unsupported(operation_name + " index span", out);
    }
    uint64_t input_bytes =
        (static_cast<uint64_t>(params.lhs_offset) + input_span + 1) *
        src.itemsize();
    if (input_bytes > binding(src).range) {
      omarchy::unsupported(operation_name + " input allocation", out);
    }
  }

  std::optional<array> axis_metadata;
  size_t collapsed_rank = kept.size() + reduced.size();
  if (collapsed_rank <= 4) {
    for (size_t index = 0; index < kept.size(); ++index) {
      params.shape[index] = kept[index].extent;
      params.in_strides[index] = kept[index].stride;
    }
    for (size_t index = 0; index < reduced.size(); ++index) {
      params.shape[kept.size() + index] = reduced[index].extent;
      params.out_strides[index] = reduced[index].stride;
    }
  } else {
    if (collapsed_rank > static_cast<size_t>(std::numeric_limits<int>::max()) /
            2) {
      omarchy::unsupported(operation_name + " rank", out);
    }
    std::vector<uint32_t> words;
    words.reserve(2 * collapsed_rank);
    for (const auto& axis : kept) {
      words.push_back(axis.extent);
    }
    for (const auto& axis : reduced) {
      words.push_back(axis.extent);
    }
    for (const auto& axis : kept) {
      words.push_back(axis.stride);
    }
    for (const auto& axis : reduced) {
      words.push_back(axis.stride);
    }
    axis_metadata.emplace(
        Shape{static_cast<int>(words.size())},
        uint32,
        nullptr,
        std::vector<array>{});
    array::Flags flags;
    flags.contiguous = true;
    flags.row_contiguous = true;
    flags.col_contiguous = true;
    axis_metadata->set_data(
        allocate_omarchy(axis_metadata->nbytes()),
        axis_metadata->size(),
        Strides{1},
        flags,
        0);
    auto* metadata_buffer = static_cast<omarchy::VulkanBuffer*>(
        axis_metadata->buffer().ptr());
    std::memcpy(metadata_buffer->data, words.data(), axis_metadata->nbytes());
    encoder.add_temporary(*axis_metadata);
    params.matrix_k = static_cast<uint32_t>(collapsed_rank);
  }

  auto run_phase = [&](uint32_t flags, uint32_t dispatch_count) {
    params.flags = flags;
    const array& metadata = axis_metadata ? *axis_metadata : scratch;
    std::array<omarchy::ComputeBinding, 4> bindings{
        binding(bound_input), binding(scratch), binding(out), binding(metadata)};
    encoder.dispatch_compute(
        kernel,
        bindings,
        params,
        omarchy::compute_dispatch_group_count(dispatch_count));
  };

  if (bool_out) {
    uint32_t word_count = checked_u32(
        (static_cast<uint64_t>(output_size) + 3) / 4,
        operation_name,
        out);
    if (chunks == 1) {
      run_phase(0u | 2u, word_count);
    } else {
      run_phase(0u, checked_u32(partial_pairs, operation_name, out));
      run_phase(1u, word_count);
    }
  } else if (chunks == 1) {
    run_phase(0u | 2u, output_size);
  } else {
    run_phase(0u, checked_u32(partial_pairs, operation_name, out));
    run_phase(1u, output_size);
  }
}
// ---------------------------------------------------------------------------
// Wave 5: indexing and scatter. Shared helpers live in this block; the
// eval functions sit at their alphabetical primitive sites below.
// ---------------------------------------------------------------------------

// Scatter index words: 0 = int32, 1 = uint32, 2 = int64 read as two
// little-endian words (the same encoding the gather kernels use).
uint32_t scatter_index_mode(
    const array& indices,
    const array& out,
    const std::string& name) {
  if (indices.dtype() == int32) {
    return 0;
  }
  if (indices.dtype() == uint32) {
    return 1;
  }
  if (indices.dtype() == int64) {
    return 2;
  }
  omarchy::unsupported(name + " index dtype", out);
}

uint32_t scatter_index_offset(
    const array& indices,
    const array& out,
    const std::string& name) {
  uint32_t offset = checked_item_offset(indices, indices.size(), name, out);
  if (indices.dtype() == int64) {
    if (offset > std::numeric_limits<uint32_t>::max() / 2) {
      omarchy::unsupported(name + " index span", out);
    }
    return offset * 2;
  }
  return offset;
}

// Rank/sentinel scratch must outlive the queued commands, so it rides
// the encoder temporaries.
array make_u32_scratch(size_t count, omarchy::CommandEncoder& encoder) {
  Shape shape{static_cast<int>(count)};
  array scratch(std::move(shape), uint32, nullptr, {});
  scratch.set_data(allocate_omarchy(scratch.nbytes()));
  encoder.add_temporary(scratch);
  return scratch;
}

void dispatch_clear_u32(
    array& scratch,
    uint32_t value,
    omarchy::CommandEncoder& encoder) {
  omarchy::ComputeParams params;
  params.count = checked_u32(scratch.size(), "Scatter scratch", scratch);
  params.operation = value;
  std::array<omarchy::ComputeBinding, 1> bindings{binding(scratch)};
  encoder.dispatch_compute(
      omarchy::ComputeKernel::ClearU32,
      bindings,
      params,
      omarchy::compute_dispatch_group_count(params.count));
}

// --- Wave 6: matmul family --------------------------------------------

// True when walking the strides from the last axis meets either the
// packed row-major stride or a broadcast 0 at every non-singleton axis,
// so the buffer's first data_size entries are the logical values in
// order. Bool masks need this property for the flat bool-to-float cast.
bool is_flat_readable(const array& value) {
  const auto& strides = value.strides();
  int64_t expected = 1;
  for (int axis = value.ndim() - 1; axis >= 0; --axis) {
    if (value.shape(axis) == 1) {
      continue;
    }
    if (strides[axis] != expected && strides[axis] != 0) {
      return false;
    }
    if (strides[axis] != 0) {
      expected *= value.shape(axis);
    }
  }
  return true;
}

// True when a batched matrix stack is dense: the gathered base offsets
// are index * matrix_size, which only addresses a packed stack (a
// uniform transposition inside each matrix is still packed).
bool is_dense_batched_matrix(const array& value, bool transposed) {
  int rank = value.ndim();
  if (rank < 2) {
    return false;
  }
  const auto& strides = value.strides();
  int64_t row_stride =
      transposed ? 1 : static_cast<int64_t>(value.shape(rank - 1));
  int64_t column_stride =
      transposed ? static_cast<int64_t>(value.shape(rank - 2)) : 1;
  if (strides[rank - 2] != row_stride || strides[rank - 1] != column_stride) {
    return false;
  }
  int64_t expected =
      static_cast<int64_t>(value.shape(rank - 2)) * value.shape(rank - 1);
  for (int axis = rank - 3; axis >= 0; --axis) {
    if (value.shape(axis) == 1) {
      continue;
    }
    if (strides[axis] != expected) {
      return false;
    }
    expected *= value.shape(axis);
  }
  return true;
}

array make_dense_temp(
    const array& src,
    const std::string& tag,
    array& out,
    omarchy::CommandEncoder& encoder) {
  array temp(src.shape(), src.dtype(), nullptr, {});
  array::Flags flags;
  flags.contiguous = true;
  flags.row_contiguous = true;
  auto max_dim = std::max_element(src.shape().begin(), src.shape().end());
  flags.col_contiguous = temp.size() <= 1 || temp.size() == *max_dim;
  Strides strides(src.ndim(), 1);
  for (int axis = static_cast<int>(src.ndim()) - 2; axis >= 0; --axis) {
    strides[axis] = strides[axis + 1] * src.shape(axis + 1);
  }
  temp.set_data(
      allocate_omarchy(temp.nbytes()), temp.size(), strides, flags, 0);
  encoder.add_temporary(temp);
  return temp;
}

// Casts a flat-readable bool mask to packed float32 values through the
// word-per-four-elements CastBoolF32 kernel.
array cast_bool_mask(
    const array& mask,
    const std::string& tag,
    array& out,
    omarchy::CommandEncoder& encoder) {
  array temp(
      Shape{static_cast<int>(mask.data_size())}, float32, nullptr, {});
  array::Flags flags;
  flags.contiguous = true;
  flags.row_contiguous = true;
  flags.col_contiguous = temp.size() <= 1;
  temp.set_data(
      allocate_omarchy(temp.nbytes()), temp.size(), Strides{1}, flags, 0);
  encoder.add_temporary(temp);
  uint32_t count = checked_u32(mask.data_size(), tag, out);
  if (count == 0) {
    return temp;
  }
  omarchy::ComputeParams params;
  params.count = count;
  params.lhs_size = count;
  params.rhs_size = count;
  params.output_size = count;
  params.lhs_offset =
      checked_item_offset(mask, mask.data_size(), tag, out);
  params.output_offset = 0;
  std::array<omarchy::ComputeBinding, 3> bindings{
      binding(mask), binding(mask), binding(temp)};
  encoder.dispatch_compute(
      omarchy::ComputeKernel::CastBoolF32,
      bindings,
      params,
      omarchy::compute_dispatch_group_count((count + 3) / 4));
  return temp;
}

// Applies one block mask into the dense destination stack. The data side
// reads through its own batch strides (0 = broadcast axis) and matrix
// strides, so one dispatch both materializes a broadcast operand and
// applies the mask; with data and dst sharing a buffer the kernel is a
// per-element map, so in-place masking is safe. The mask is float32
// (bool pre-cast) indexed by its own grid strides.
void dispatch_block_mask(
    const array& data,
    const array& mask,
    const array& mask_logical,
    array& dst,
    int rows,
    int cols,
    int block_size,
    const std::string& tag,
    array& out) {
  auto& encoder = omarchy::get_command_encoder(out.primitive().stream());
  int batch_axes = data.ndim() - 2;
  if (batch_axes > 4) {
    omarchy::unsupported(tag + " mask batch rank", out);
  }
  size_t batch = 1;
  for (int axis = 0; axis < batch_axes; ++axis) {
    batch *= data.shape(axis);
  }
  size_t count = batch * static_cast<size_t>(rows) * cols;
  if (count == 0 || mask.size() == 0) {
    return;
  }
  omarchy::ComputeParams params;
  params.count = checked_u32(count, tag, out);
  // Mask grid strides come from the logical mask (the bound values may
  // be a flat cast temp): the last two axes are (block rows, block
  // columns); higher axes ride out_strides, 0 = broadcast batch axis.
  params.lhs_size = checked_u32(
      mask_logical.strides()[mask_logical.ndim() - 2], tag, out);
  params.rhs_size = checked_u32(
      mask_logical.strides()[mask_logical.ndim() - 1], tag, out);
  params.lhs_offset = checked_item_offset(data, data.size(), tag, out);
  params.rhs_offset = checked_item_offset(mask, mask.size(), tag, out);
  params.output_offset = checked_item_offset(dst, dst.size(), tag, out);
  // Data matrix strides for the (row, column) decode.
  params.aux_size =
      checked_u32(data.strides()[data.ndim() - 2], tag, out);
  params.aux_offset =
      checked_u32(data.strides()[data.ndim() - 1], tag, out);
  params.matrix_m = checked_u32(rows, tag, out);
  params.matrix_n = checked_u32(cols, tag, out);
  params.matrix_k = checked_u32(block_size, tag, out);
  params.dims = static_cast<uint32_t>(batch_axes);
  uint64_t data_span = 0;
  uint64_t mask_span = 0;
  for (int axis = 0; axis < batch_axes; ++axis) {
    uint32_t extent = static_cast<uint32_t>(data.shape(axis));
    uint32_t d_stride =
        static_cast<uint32_t>(data.strides()[axis]);
    uint32_t m_stride =
        static_cast<uint32_t>(mask_logical.strides()[axis]);
    params.shape[axis] = extent;
    params.in_strides[axis] = d_stride;
    params.out_strides[axis] = m_stride;
    data_span += (extent - 1u) * d_stride;
    mask_span += (extent - 1u) * m_stride;
  }
  data_span += (rows - 1) * params.aux_size + (cols - 1) * params.aux_offset;
  mask_span += (params.lhs_size + params.rhs_size) *
      static_cast<uint64_t>(((rows + block_size - 1) / block_size) *
                            ((cols + block_size - 1) / block_size));
  if (!omarchy::compute_index_span_fits(params.lhs_offset, data_span + 1) ||
      !omarchy::compute_index_span_fits(params.rhs_offset, mask_span + 1)) {
    omarchy::unsupported(tag + " mask index span", out);
  }
  std::array<omarchy::ComputeBinding, 3> bindings{
      binding(data), binding(mask), binding(dst)};
  encoder.dispatch_compute(
      omarchy::ComputeKernel::BlockMaskF32,
      bindings,
      params,
      omarchy::compute_dispatch_group_count(params.count));
}

// Non-affine fp-mode quantize direction shared by fast::Quantize and
// the qqmm activation/float-weight paths: fp4 e2m1 or fp8 e4m3 element
// bytes packed little-endian into uint32 words plus one uint8 scale
// byte per group (round-up e8m0 at group 32, e4m3 at group 16 with the
// optional nvfp4 float32 global scale folded into the stored bytes).
// Conversions mirror the pinned Metal fp4.h / fp8.h helpers bit for
// bit. |packed| and |scales| arrive with the caller-allocated shapes:
// packed = in_w.shape with last dim * bits / 32, scales = in_w.shape
// with last dim / group_size. |out| is the error-reporting array.
void dispatch_fp_quantize(
    const std::string& tag,
    const array& in_w,
    array& packed,
    array& scales,
    const std::optional<array>& global_scale,
    int bits,
    int group_size,
    array& out,
    const Stream& s) {
  // The quantize input is a floating matrix; the 16-bit paths need
  // the matching storage capabilities.
  if (
      in_w.dtype() != float16 && in_w.dtype() != bfloat16 &&
      in_w.dtype() != float32) {
    omarchy::unsupported(tag + " input dtype", out);
  }
  auto& encoder = omarchy::get_command_encoder(s);
  const auto& fp_caps = encoder.device().capabilities();
  if (in_w.dtype() == float16 &&
      (!fp_caps.shader_float16 ||
       !fp_caps.storage_buffer_16bit_access)) {
    omarchy::unsupported(tag + " float16 capability", out);
  }
  if (in_w.dtype() == bfloat16 &&
      (!fp_caps.storage_buffer_16bit_access ||
       !fp_caps.shader_int16)) {
    omarchy::unsupported(tag + " bfloat16 capability", out);
  }
  Shape packed_shape = in_w.shape();
  packed_shape.back() = packed_shape.back() * bits / 32;
  Shape parameter_shape = in_w.shape();
  parameter_shape.back() /= group_size;
  if (
      packed.dtype() != uint32 || packed.shape() != packed_shape ||
      scales.dtype() != uint8 || scales.shape() != parameter_shape) {
    omarchy::unsupported(tag + " shape", out);
  }
  std::optional<array> w_temp;
  const array& w =
      ensure_dense(in_w, in_w.flags().row_contiguous, w_temp, encoder, s);
  packed.set_data(allocate_omarchy(packed.nbytes()));
  scales.set_data(allocate_omarchy(scales.nbytes()));
  if (w.size() == 0) {
    return;
  }
  omarchy::ComputeParams params;
  params.count = checked_u32(w.size() / group_size, tag, out);
  params.operation = static_cast<uint32_t>(bits);
  params.reduce_size = static_cast<uint32_t>(group_size);
  params.lhs_offset = checked_item_offset(w, w.size(), tag, out);
  params.rhs_offset =
      checked_item_offset(scales, scales.size(), tag, out);
  params.output_offset =
      checked_item_offset(packed, packed.size(), tag, out);
  if (global_scale) {
    params.flags = 2u; // bit 1: bound global scale
    params.aux_size = checked_item_offset(*global_scale, 1, tag, out);
  }
  std::array<omarchy::ComputeBinding, 4> bindings{
      binding(w),
      binding(packed),
      binding(scales),
      global_scale ? binding(*global_scale) : binding(scales)};
  auto kernel = select_float_kernel(
      in_w.dtype(),
      omarchy::ComputeKernel::QuantizeFpF32,
      omarchy::ComputeKernel::QuantizeFpF16,
      omarchy::ComputeKernel::QuantizeFpBF16);
  encoder.dispatch_compute(
      kernel,
      bindings,
      params,
      omarchy::compute_dispatch_group_count(params.count));
}

// Non-affine fp-mode dequantize direction: packed codes plus uint8
// scale bytes back to the floating storage dtype. fast::Quantize's
// dequantize mode and the qqmm activation fake-quantization share the
// kernel parameterization; for qqmm the output is the fake-quantized
// activation the matmul consumes. A bound global scale divides the
// decoded scale by the 2688 encode factor, mirroring dequant.comp.
void dispatch_fp_dequantize(
    const std::string& tag,
    const array& in_packed,
    const array& in_scales,
    array& out,
    const std::optional<array>& global_scale,
    int bits,
    int group_size,
    const Stream& s) {
  if (in_packed.dtype() != uint32 || in_scales.dtype() != uint8) {
    omarchy::unsupported(tag + " weight dtype", out);
  }
  if (
      out.dtype() != float16 && out.dtype() != bfloat16 &&
      out.dtype() != float32) {
    omarchy::unsupported(tag + " scales dtype", out);
  }
  auto& encoder = omarchy::get_command_encoder(s);
  {
    // The 16-bit outputs need the matching storage capabilities.
    const auto& fp_caps = encoder.device().capabilities();
    if (out.dtype() == float16 &&
        (!fp_caps.shader_float16 ||
         !fp_caps.storage_buffer_16bit_access)) {
      omarchy::unsupported(tag + " float16 capability", out);
    }
    if (out.dtype() == bfloat16 &&
        (!fp_caps.storage_buffer_16bit_access ||
         !fp_caps.shader_int16)) {
      omarchy::unsupported(tag + " bfloat16 capability", out);
    }
  }
  if (
      in_packed.shape().size() != in_scales.shape().size() ||
      in_scales.shape(-1) * static_cast<uint64_t>(group_size) !=
          static_cast<uint64_t>(in_packed.shape(-1)) * 32u / bits) {
    omarchy::unsupported(tag + " shape", out);
  }
  if (!std::equal(
          in_packed.shape().begin(),
          in_packed.shape().end() - 1,
          in_scales.shape().begin())) {
    omarchy::unsupported(tag + " shape", out);
  }
  std::optional<array> packed_temp;
  std::optional<array> scales_temp;
  const array& w = ensure_dense(
      in_packed, in_packed.flags().row_contiguous, packed_temp, encoder, s);
  const array& scales_d = ensure_dense(
      in_scales,
      in_scales.flags().row_contiguous,
      scales_temp,
      encoder,
      s);
  out.set_data(allocate_omarchy(out.nbytes()));
  if (out.size() == 0 || w.size() == 0) {
    return;
  }
  // One thread per byte pack: fp4 pairs two elements per byte, fp8
  // is one byte per element.
  uint32_t pack_factor = (bits == 8) ? 1u : 2u;
  uint64_t bytes_per_row =
      static_cast<uint64_t>(in_packed.shape(-1)) * 4u;
  omarchy::ComputeParams params;
  params.count = checked_u32(w.size() * 4u, tag, out);
  params.operation = static_cast<uint32_t>(bits);
  params.reduce_size = static_cast<uint32_t>(group_size);
  params.lhs_offset = checked_item_offset(w, w.size(), tag, out);
  params.aux_offset =
      checked_item_offset(scales_d, scales_d.size(), tag, out);
  params.rhs_size =
      global_scale ? checked_item_offset(*global_scale, 1, tag, out) : 0;
  params.output_offset = checked_item_offset(out, out.size(), tag, out);
  params.matrix_n = checked_u32(bytes_per_row, tag, out);
  params.matrix_k = checked_u32(
      static_cast<uint64_t>(in_scales.shape(-1)), tag, out);
  params.flags = global_scale ? 2u : 0u;
  std::array<omarchy::ComputeBinding, 4> bindings{
      binding(w),
      binding(scales_d),
      global_scale ? binding(*global_scale) : binding(w),
      binding(out)};
  auto kernel = select_float_kernel(
      out.dtype(),
      omarchy::ComputeKernel::DequantFpF32,
      omarchy::ComputeKernel::DequantFpF16,
      omarchy::ComputeKernel::DequantFpBF16);
  encoder.dispatch_compute(
      kernel,
      bindings,
      params,
      omarchy::compute_dispatch_group_count(params.count));
}

// Fake-quantize a floating activation for the fp-mode qqmm paths:
// quantize into packed codes plus scale bytes, then dequantize back to
// the storage dtype, yielding dequant(quant(x)) exactly. The nvfp4
// global scale folds in and back out across the two dispatches, so the
// matmul contracts x_hat against dequantized w the way the pinned
// Metal qqmm path fake-quantizes x before its qmv dispatch. The
// returned array is an encoder temporary.
array fake_quantize_fp_activation(
    const std::string& tag,
    const array& x,
    const std::optional<array>& global_scale,
    int bits,
    int group_size,
    array& out,
    const Stream& s) {
  auto& encoder = omarchy::get_command_encoder(s);
  Shape packed_shape = x.shape();
  packed_shape.back() = packed_shape.back() * bits / 32;
  Shape parameter_shape = x.shape();
  parameter_shape.back() /= group_size;
  array codes(std::move(packed_shape), uint32, nullptr, {});
  array scale_bytes(std::move(parameter_shape), uint8, nullptr, {});
  dispatch_fp_quantize(
      tag, x, codes, scale_bytes, global_scale, bits, group_size, out, s);
  encoder.add_temporary(codes);
  encoder.add_temporary(scale_bytes);
  array x_hat(x.shape(), x.dtype(), nullptr, {});
  dispatch_fp_dequantize(
      tag, codes, scale_bytes, x_hat, global_scale, bits, group_size, s);
  encoder.add_temporary(x_hat);
  return x_hat;
}

// Shared body for the gathered quantized matmuls: GatherQMM (scale +
// bias), GatherQQMM and the quantized-weight QQMatmul path (scale
// only, no_bias) in the affine modes, and GatherQMM / GatherQQMM in
// the non-affine modes (fp_mode: mxfp4 / nvfp4 / mxfp8, always
// no-bias, one uint8 scale byte per group). lhs/rhs with a single
// zero index serve the un-gathered case. The packed word buffer
// carries [scales bytes | (bias bytes) | lhs index words | rhs index
// words]; every region sits on a 4-byte boundary and padding stays
// zeroed, so 16-bit parameters decode from whole words with no
// 16-bit storage reads. transpose selects the packed weight layout
// (transposed rows [batch, N, Kp] or non-transposed columns
// [batch, Kp, N]) routed to the kernel through flags bit 0; x accepts
// any rank >= 2 and w/scales any equal batch rank, because lhs/rhs
// indices index x/w batch slices flat. In fp_mode, an optional
// out_global_scale (nvfp4 float32 global_scale_w) rides in binding 4
// through the HGS kernel variants, which multiply the finished dot by
// global_scale_w / 2688 - the packed w scale bytes keep the encode
// factor the quantize direction folded in.
void dispatch_gather_qmm(
    const std::string& tag,
    const array& x,
    const array& w,
    const array& scales,
    const std::optional<array>& biases,
    const array& lhs,
    const array& rhs,
    int group_size,
    int bits,
    bool transpose,
    bool fp_mode,
    const std::optional<array>& out_global_scale,
    array& out,
    const Stream& s) {
  auto& encoder = omarchy::get_command_encoder(s);
  require_float_dtype(tag, x, out, encoder);
  if (bits != 4 && bits != 8) {
    omarchy::unsupported(tag + " bits", out);
  }
  if (group_size != 32 && group_size != 64 && group_size != 128 &&
      !(fp_mode && group_size == 16)) {
    omarchy::unsupported(tag + " group size", out);
  }
  if (fp_mode && out_global_scale &&
      encoder.device().compute().binding_limit() < 5) {
    omarchy::unsupported(tag + " binding budget", out);
  }
  if (w.dtype() != uint32) {
    omarchy::unsupported(tag + " weight dtype", out);
  }
  if (scales.dtype() != (fp_mode ? uint8 : out.dtype()) ||
      (biases && biases->dtype() != out.dtype())) {
    // ops.cpp promotes affine scales and biases to the output dtype;
    // fp modes keep one uint8 scale byte per group.
    omarchy::unsupported(tag + " scales dtype", out);
  }
  if (x.ndim() < 2 || w.ndim() < 2 || scales.ndim() != w.ndim()) {
    omarchy::unsupported(tag + " rank", out);
  }
  int k = x.shape(-1);
  int m = x.shape(-2);
  // mx.quantize packs along the dequantized last axis: transposed w
  // is [batch, N, Kp] with scales [batch, N, K / group_size];
  // non-transposed w is [batch, K, Np] with scales
  // [batch, K, N / group_size]; Np = N * bits / 32.
  int n = transpose ? w.shape(-2) : w.shape(-1) * 32 / bits;
  int packed_inner = transpose ? w.shape(-1) * 32 / bits : w.shape(-2);
  if (packed_inner != k ||
      scales.shape(-2) != (transpose ? n : k) ||
      scales.shape(-1) * group_size != (transpose ? k : n) ||
      !std::equal(
          w.shape().begin(),
          w.shape().end() - 2,
          scales.shape().begin())) {
    omarchy::unsupported(tag + " shape", out);
  }
  if (biases && biases->shape() != scales.shape()) {
    omarchy::unsupported(tag + " scales shape", out);
  }
  if (lhs.dtype() != uint32 || rhs.dtype() != uint32) {
    omarchy::unsupported(tag + " index dtype", out);
  }
  if (lhs.shape() != rhs.shape()) {
    omarchy::unsupported(tag + " index shape", out);
  }
  std::optional<array> x_temp;
  std::optional<array> w_temp;
  std::optional<array> scales_temp;
  std::optional<array> lhs_temp;
  std::optional<array> rhs_temp;
  std::optional<array> bias_temp;
  const array& x_d =
      ensure_dense(x, x.flags().row_contiguous, x_temp, encoder, s);
  const array& w_d =
      ensure_dense(w, w.flags().row_contiguous, w_temp, encoder, s);
  const array& scales_d = ensure_dense(
      scales, scales.flags().row_contiguous, scales_temp, encoder, s);
  const array* biases_d = biases.has_value()
      ? &ensure_dense(
              *biases,
              biases->flags().row_contiguous,
              bias_temp,
              encoder,
              s)
      : nullptr;
  const array& lhs_d =
      ensure_dense(lhs, lhs.flags().row_contiguous, lhs_temp, encoder, s);
  const array& rhs_d =
      ensure_dense(rhs, rhs.flags().row_contiguous, rhs_temp, encoder, s);

  out.set_data(allocate_omarchy(out.nbytes()));
  if (out.size() == 0) {
    return;
  }
  size_t index_count = lhs_d.size();
  size_t scale_bytes = scales.nbytes();
  size_t bias_bytes = biases ? biases->nbytes() : 0;
  auto align4 = [](size_t value) { return (value + 3) & ~size_t(3); };
  size_t bias_base = align4(scale_bytes);
  size_t index_base = align4(bias_base + bias_bytes);
  size_t packed_bytes = index_base + 2 * index_count * 4;
  if (scales_d.offset() % 4 != 0 ||
      (biases_d && biases_d->offset() % 4 != 0) ||
      lhs_d.offset() % 4 != 0 || rhs_d.offset() % 4 != 0) {
    omarchy::unsupported(tag + " byte offset", out);
  }
  array packed(Shape{static_cast<int>(packed_bytes / 4)}, uint32, nullptr, {});
  array::Flags packed_flags;
  packed_flags.contiguous = true;
  packed_flags.row_contiguous = true;
  packed_flags.col_contiguous = true;
  packed.set_data(
      allocate_omarchy(packed.nbytes()),
      packed.size(),
      Strides{1},
      packed_flags,
      0);
  encoder.add_temporary(packed);
  encoder.fill_buffer(binding(packed).buffer, 0, packed_bytes, 0);
  encoder.add_temporary(scales_d);
  encoder.add_temporary(packed);
  encoder.copy_buffer(
      binding(scales_d).buffer,
      binding(packed).buffer,
      scale_bytes,
      static_cast<VkDeviceSize>(scales_d.offset()),
      0);
  if (biases_d) {
    encoder.add_temporary(*biases_d);
    encoder.add_temporary(packed);
    encoder.copy_buffer(
        binding(*biases_d).buffer,
        binding(packed).buffer,
        bias_bytes,
        static_cast<VkDeviceSize>(biases_d->offset()),
        static_cast<VkDeviceSize>(bias_base));
  }
  encoder.add_temporary(lhs_d);
  encoder.add_temporary(packed);
  encoder.copy_buffer(
      binding(lhs_d).buffer,
      binding(packed).buffer,
      index_count * 4,
      static_cast<VkDeviceSize>(lhs_d.offset()),
      static_cast<VkDeviceSize>(index_base));
  encoder.add_temporary(rhs_d);
  encoder.add_temporary(packed);
  encoder.copy_buffer(
      binding(rhs_d).buffer,
      binding(packed).buffer,
      index_count * 4,
      static_cast<VkDeviceSize>(rhs_d.offset()),
      static_cast<VkDeviceSize>(index_base + index_count * 4));

  omarchy::ComputeParams params;
  params.count = checked_u32(out.size(), tag, out);
  params.lhs_size = checked_u32(x.size(), tag, out);
  params.rhs_size = checked_u32(w.size(), tag, out);
  params.reduce_size = static_cast<uint32_t>(group_size);
  params.output_size = params.count;
  params.lhs_offset = checked_item_offset(x_d, x_d.size(), tag, out);
  params.rhs_offset = checked_item_offset(w_d, w_d.size(), tag, out);
  params.output_offset = checked_item_offset(out, out.size(), tag, out);
  params.aux_offset = 0;
  params.matrix_m = checked_u32(m, tag, out);
  params.matrix_n = checked_u32(n, tag, out);
  params.matrix_k = checked_u32(k, tag, out);
  params.operation = static_cast<uint32_t>(bits);
  // shape[0] is the per-half index count; shape[1] the bias region's
  // byte base; in_strides[0]/out_strides[0] the lhs/rhs word offsets;
  // flags bit 0 the non-transposed weight layout.
  params.shape[0] = checked_u32(index_count, tag, out);
  params.shape[1] = static_cast<uint32_t>(bias_base);
  params.flags = transpose ? 0u : 1u;
  params.in_strides[0] = static_cast<uint32_t>(index_base / 4);
  params.out_strides[0] =
      static_cast<uint32_t>(index_base / 4 + index_count);
  if (!omarchy::compute_index_span_fits(params.lhs_offset, x.size()) ||
      !omarchy::compute_index_span_fits(params.rhs_offset, w.size())) {
    omarchy::unsupported(tag + " index span", out);
  }
  // fp modes have no bias term; their kernels are the FP_MODE
  // no-bias variants. The bound global scale routes to the HGS
  // variants, whose fifth binding carries the float32 word.
  bool no_bias = !biases.has_value();
  if (fp_mode && out_global_scale) {
    std::array<omarchy::ComputeBinding, 5> hgs_bindings{
        binding(x_d),
        binding(packed),
        binding(w_d),
        binding(out),
        binding(*out_global_scale)};
    // binding() pins the whole buffer at word zero, so the scalar
    // view's own storage offset rides in aux_size - the same
    // checked_item_offset routing the quantize/dequantize global-scale
    // siblings use. Valid aligned scalar views compute; unaligned ones
    // refuse by name.
    params.aux_size = checked_item_offset(*out_global_scale, 1, tag, out);
    omarchy::ComputeKernel hgs_kernel;
    if (out.dtype() == float32) {
      hgs_kernel = omarchy::ComputeKernel::GatherQmmNbFpHgsF32;
    } else if (out.dtype() == float16) {
      hgs_kernel = omarchy::ComputeKernel::GatherQmmNbFpHgsF16;
    } else {
      hgs_kernel = omarchy::ComputeKernel::GatherQmmNbFpHgsBF16;
    }
    encoder.dispatch_compute(
        hgs_kernel,
        hgs_bindings,
        params,
        omarchy::compute_dispatch_group_count(params.count));
    return;
  }
  std::array<omarchy::ComputeBinding, 4> bindings{
      binding(x_d), binding(packed), binding(w_d), binding(out)};
  omarchy::ComputeKernel kernel;
  if (fp_mode) {
    if (out.dtype() == float32) {
      kernel = omarchy::ComputeKernel::GatherQmmNbFpF32;
    } else if (out.dtype() == float16) {
      kernel = omarchy::ComputeKernel::GatherQmmNbFpF16;
    } else {
      kernel = omarchy::ComputeKernel::GatherQmmNbFpBF16;
    }
  } else if (out.dtype() == float32) {
    kernel = no_bias ? omarchy::ComputeKernel::GatherQmmNbF32
                     : omarchy::ComputeKernel::GatherQmmF32;
  } else if (out.dtype() == float16) {
    kernel = no_bias ? omarchy::ComputeKernel::GatherQmmNbF16
                     : omarchy::ComputeKernel::GatherQmmF16;
  } else {
    kernel = no_bias ? omarchy::ComputeKernel::GatherQmmNbBF16
                     : omarchy::ComputeKernel::GatherQmmBF16;
  }
  encoder.dispatch_compute(
      kernel,
      bindings,
      params,
      omarchy::compute_dispatch_group_count(params.count));
}

// Complex64Transport. Keep in lockstep with the switch in
// shaders/complex_elementwise.comp.
enum ComplexOperation : uint32_t {
  ComplexConjugate,
  ComplexAdd,
  ComplexSubtract,
  ComplexMultiply,
  ComplexDivide,
  ComplexNegative,
  ComplexLogAddExp,
  ComplexPower,
  ComplexExp,
  ComplexSin,
  ComplexCos,
  ComplexMaximum,
  ComplexSinh,
  ComplexCosh,
  ComplexTan,
  ComplexTanh,
  ComplexLog1p,
  ComplexSign,
  ComplexArcCos,
  ComplexArcSin,
  ComplexArcTan,
  ComplexSqrt,
  ComplexRsqrt,
  ComplexLog,
  ComplexLog2,
  ComplexLog10,
  ComplexRound,
};

// The params fill and dispatch behind the complex64 elementwise
// kernel, mirroring dispatch_float_elementwise_to: a vec2 element per
// item, the same broadcast transport, and the same contiguity rule.
// Unary callers alias lhs and rhs; the shader's lhs_size/rhs_size
// handling then keeps the modulo fast path honest.
void dispatch_complex_elementwise_to(
    const std::string& name,
    uint32_t operation,
    const array& lhs,
    const array& rhs,
    array& out,
    bool general_broadcast,
    omarchy::CommandEncoder& encoder) {
  uint32_t count = checked_u32(out.size(), name, out);
  omarchy::ComputeParams params;
  params.count = count;
  params.operation = operation;
  params.lhs_size = checked_u32(lhs.data_size(), name, out);
  params.rhs_size = checked_u32(rhs.data_size(), name, out);
  params.output_size = count;
  params.lhs_offset = checked_item_offset(lhs, params.lhs_size, name, out);
  params.rhs_offset = checked_item_offset(rhs, params.rhs_size, name, out);
  params.output_offset = checked_item_offset(out, count, name, out);
  std::optional<array> axis_metadata;
  if (general_broadcast) {
    axis_metadata = fill_broadcast_transport(name, params, lhs, rhs, out, encoder);
  }
  std::array<omarchy::ComputeBinding, 4> bindings{
      binding(lhs),
      binding(rhs),
      binding(out),
      binding(axis_metadata ? *axis_metadata : out)};
  encoder.dispatch_compute(
      omarchy::ComputeKernel::ComplexElementwise,
      bindings,
      params,
      omarchy::compute_dispatch_group_count(count));
}

void dispatch_complex(
    const std::string& name,
    uint32_t operation,
    const std::vector<array>& inputs,
    array& out,
    const Stream& s) {
  const array& in_lhs = inputs.at(0);
  const bool binary = inputs.size() == 2;
  const array& in_rhs = binary ? inputs.at(1) : in_lhs;
  auto& encoder = omarchy::get_command_encoder(s);
  if (in_lhs.dtype() != complex64 || in_rhs.dtype() != complex64 ||
      out.dtype() != complex64) {
    omarchy::unsupported(name + " complex64 dtype", out);
  }
  std::optional<array> lhs_temp;
  std::optional<array> rhs_temp;
  const array& lhs =
      ensure_dense(in_lhs, in_lhs.flags().contiguous, lhs_temp, encoder, s);
  const array& rhs = binary
      ? ensure_dense(in_rhs, in_rhs.flags().contiguous, rhs_temp, encoder, s)
      : lhs;
  bool general_broadcast = binary
      ? (!is_trailing_broadcast(lhs, out) || !is_trailing_broadcast(rhs, out))
      : !is_trailing_broadcast(lhs, out);
  if (binary) {
    auto binary_type = get_binary_op_type(lhs, rhs);
    if (lhs.data_size() != lhs.size() || rhs.data_size() != rhs.size()) {
      binary_type = BinaryOpType::General;
    }
    set_binary_op_output_data(
        lhs, rhs, out, binary_type, allocate_omarchy);
  } else if (general_broadcast) {
    out.set_data(allocate_omarchy(out.nbytes()));
  } else if (lhs.data_size() != lhs.size()) {
    out.set_data(allocate_omarchy(out.nbytes()));
  } else {
    set_unary_output_data(lhs, out, allocate_omarchy);
  }
  if (out.size() == 0) {
    return;
  }
  dispatch_complex_elementwise_to(
      name, operation, lhs, rhs, out, general_broadcast, encoder);
}
// Complex64 component extraction and magnitude to float32: operation
// 0 takes the real part (ComplexReal), 1 the imaginary part
// (ComplexImag), 2 the magnitude |z| with overflow-safe hypot
// (ComplexAbs). Operations 0 and 1 mirror the upstream real()/imag()
// semantics on a complex64 array (mlx/backend/cpu/unary.cpp routes
// both through unary_complex_to_float); operation 2 mirrors cabsf /
// std::abs so upstream allclose can take the absolute value of a
// complex difference. The input offset is a complex64 item offset
// and the output offset a float32 item offset, which is what the
// per-array checked_item_offset calls already produce.
void dispatch_complex_extract(
    const std::string& name,
    uint32_t operation,
    const std::vector<array>& inputs,
    array& out,
    const Stream& s) {
  const array& in_raw = inputs.at(0);
  auto& encoder = omarchy::get_command_encoder(s);
  if (in_raw.dtype() != complex64) {
    omarchy::unsupported(name + " dtype", out);
  }
  // Operations 0 and 1 (real/imag) target float32 out; operation 2
  // (abs) can target either float32 (direct) or complex64 (the
  // intermediate path Abs::eval_gpu takes when out is the primary
  // complex64 buffer the ops layer allocated).
  if (operation < 2u && out.dtype() != float32) {
    omarchy::unsupported(name + " dtype", out);
  }
  if (operation == 2u && out.dtype() != float32 &&
      out.dtype() != complex64) {
    omarchy::unsupported(name + " dtype", out);
  }
  // The flat count/offset binding below reads the input as dense
  // row-major storage. Upstream keeps the no-gaps `contiguous` flag
  // set across broadcast (stride-0) and transposed views, so
  // row_contiguous is what separates a flat read from a re-read in a
  // different order; anything else is materialized once through the
  // same strided-copy engine dispatch_complex uses. Abs, Real, and
  // Imag all route here, so the layout transport is fixed once.
  std::optional<array> in_temp;
  const array& in = ensure_dense(
      in_raw,
      in_raw.flags().contiguous && in_raw.flags().row_contiguous,
      in_temp,
      encoder,
      s);
  uint32_t count = checked_u32(out.size(), name, out);
  omarchy::ComputeParams params;
  params.count = count;
  params.operation = operation;
  params.output_size = count;
  params.lhs_offset = checked_item_offset(in, in.data_size(), name, out);
  params.output_offset = checked_item_offset(out, count, name, out);
  if (out.size() == 0) {
    return;
  }
  out.set_data(allocate_omarchy(out.nbytes()));
  std::array<omarchy::ComputeBinding, 3> bindings{
      binding(in), binding(in), binding(out)};
  auto kernel = operation == 0
      ? omarchy::ComputeKernel::ComplexReal
      : operation == 1
      ? omarchy::ComputeKernel::ComplexImag
      : out.dtype() == complex64
      ? omarchy::ComputeKernel::ComplexAbsAsComplex
      : omarchy::ComputeKernel::ComplexAbs;
  encoder.dispatch_compute(
      kernel, bindings, params, omarchy::compute_dispatch_group_count(count));
}

} // namespace
#define OMARCHY_BINARY(func, operation)                               \
  void func::eval_gpu(const std::vector<array>& inputs, array& out) { \
    dispatch_elementwise(                                             \
        #func, operation, inputs, out, out.primitive().stream());     \
  }
#define OMARCHY_UNARY(func, operation)                                \
  void func::eval_gpu(const std::vector<array>& inputs, array& out) { \
    dispatch_elementwise(                                             \
        #func, operation, inputs, out, out.primitive().stream());     \
  }
void Abs::eval_gpu(const std::vector<array>& inputs, array& out) {
  if (inputs.at(0).dtype() == bool_ && out.dtype() == bool_) {
    // Upstream abs on bool is the identity; sign and abs of 0/1
    // bytes are the byte itself.
    dispatch_int_elementwise(name(), IntAbsOperation, inputs, out);
    return;
  }
  const array& in = inputs.at(0);
  if (in.dtype() == complex64) {
    // ops.cpp builds the primary array with the input dtype (so out
    // starts as complex64) and applies astype(complex64 -> float32)
    // as a separate Cast downstream; dispatch_complex_extract
    // detects that out is complex64 and routes to ComplexAbsAsComplex
    // (magnitude into .x, 0 into .y) so the downstream Cast reads
    // real() and gets the right value. The direct float32-out path
    // also works (one scalar per element).
    dispatch_complex_extract(name(), 2, inputs, out, out.primitive().stream());
    return;
  }
  if (is_int_elementwise_dtype(out.dtype())) {
    dispatch_int_elementwise(name(), IntAbsOperation, inputs, out);
    return;
  }
  dispatch_elementwise(
      name(), AbsFloatOperation, inputs, out, out.primitive().stream());
}
void Add::eval_gpu(const std::vector<array>& inputs, array& out) {
  if (out.dtype() == complex64) {
    dispatch_complex(
        name(), ComplexAdd, inputs, out, out.primitive().stream());
    return;
  }
  if (out.dtype() == bool_) {
    // Upstream bool add is the logical or over {0,1}: bitwise or on the
    // byte lanes is exact and cannot leave a non-0/1 byte behind.
    dispatch_int_elementwise(name(), IntBitwiseOrOperation, inputs, out);
    return;
  }
  if (is_int_elementwise_dtype(out.dtype())) {
    dispatch_int_elementwise(name(), IntAddOperation, inputs, out);
    return;
  }
  dispatch_elementwise(
      name(), AddOperation, inputs, out, out.primitive().stream());
}
void AddMM::eval_gpu(const std::vector<array>& inputs, array& out) {
  auto [alpha, beta] = state();
  dispatch_matmul(name(), inputs, out, alpha, beta, true, out.primitive().stream());
}
void Arange::eval_gpu(const std::vector<array>& inputs, array& out) {
  auto& encoder = omarchy::get_command_encoder(out.primitive().stream());
  bool is_int32 = out.dtype() == int32;
  bool is_uint32 = out.dtype() == uint32;
  bool is_int64 = out.dtype() == int64;
  bool is_uint64 = out.dtype() == uint64;
  bool is_int_arange = is_int32 || is_uint32 || is_int64 || is_uint64;
  if (!is_int_arange) {
    require_float_dtype("Arange", out, out, encoder);
  }
  if ((is_int64 || is_uint64) &&
      !encoder.device().capabilities().shader_int64) {
    omarchy::unsupported("Arange int64 capability", out);
  }
  out.set_data(allocate_omarchy(out.nbytes()));
  if (out.size() == 0) {
    return;
  }
  uint32_t count = checked_u32(out.size(), "Arange", out);
  omarchy::ComputeParams params;
  params.count = count;
  params.output_size = count;
  params.output_offset = checked_item_offset(out, count, "Arange", out);
  if (is_int_arange) {
    uint64_t start_bits = integer_arange_bits(start_, out);
    uint64_t next_bits = integer_arange_bits(start_ + step_, out);
    if (is_int32 || is_uint32) {
      uint32_t start = static_cast<uint32_t>(start_bits);
      uint32_t next = static_cast<uint32_t>(next_bits);
      params.lhs_size = start;
      params.reduce_size = next - start;
    } else {
      uint64_t step_bits = next_bits - start_bits;
      params.lhs_size = static_cast<uint32_t>(start_bits);
      params.rhs_size = static_cast<uint32_t>(start_bits >> 32);
      params.reduce_size = static_cast<uint32_t>(step_bits);
      params.aux_size = static_cast<uint32_t>(step_bits >> 32);
    }
  } else {
    params.alpha = static_cast<float>(start_);
    params.beta = static_cast<float>(step_);
  }
  std::array<omarchy::ComputeBinding, 1> bindings{binding(out)};
  auto kernel = is_int32
      ? omarchy::ComputeKernel::ArangeI32
      : is_uint32
      ? omarchy::ComputeKernel::ArangeU32
      : is_int64
      ? omarchy::ComputeKernel::ArangeI64
      : is_uint64
      ? omarchy::ComputeKernel::ArangeU64
      : select_float_kernel(
            out.dtype(),
            omarchy::ComputeKernel::ArangeF32,
            omarchy::ComputeKernel::ArangeF16,
            omarchy::ComputeKernel::ArangeBF16);
  encoder.dispatch_compute(
      kernel, bindings, params, omarchy::compute_dispatch_group_count(count));
}
void ArcCos::eval_gpu(const std::vector<array>& inputs, array& out) {
  if (out.dtype() == complex64) {
    dispatch_complex(name(), ComplexArcCos, inputs, out, stream());
    return;
  }
  dispatch_elementwise(name(), ArcCosOperation, inputs, out, stream());
}
OMARCHY_UNARY(ArcCosh, ArcCoshOperation)
void ArcSin::eval_gpu(const std::vector<array>& inputs, array& out) {
  if (out.dtype() == complex64) {
    dispatch_complex(name(), ComplexArcSin, inputs, out, stream());
    return;
  }
  dispatch_elementwise(name(), ArcSinOperation, inputs, out, stream());
}
OMARCHY_UNARY(ArcSinh, ArcSinhOperation)
void ArcTan::eval_gpu(const std::vector<array>& inputs, array& out) {
  if (out.dtype() == complex64) {
    dispatch_complex(name(), ComplexArcTan, inputs, out, stream());
    return;
  }
  dispatch_elementwise(name(), ArcTanOperation, inputs, out, stream());
}
OMARCHY_BINARY(ArcTan2, ArcTan2Operation)
OMARCHY_UNARY(ArcTanh, ArcTanhOperation)
void ArgPartition::eval_gpu(const std::vector<array>& inputs, array& out) {
  const array& input = inputs.at(0);
  auto [kth, axis] = state();
  auto& encoder = omarchy::get_command_encoder(out.primitive().stream());
  require_sort_dtype("ArgPartition", input, out, true, encoder);
  // Ops layer (mlx/ops.cpp argpartition) already validated kth in
  // [0, axis_size) so the row axis is non-empty. The full-sort redirect
  // covers any row length: the wide-row sort path sorts the row and the
  // argsort indices are the partition order.
  dispatch_sort_any_axis("ArgPartition", input, out, axis, true, encoder);
}
// One thread per output row sweeps the row-contiguous suffix row of
// `src`; the uint32 index output drops the reduced axis, so `out` is
// dense with the input's non-axis dims.
void dispatch_arg_reduce_suffix(
    const std::string& operation_name,
    bool is_max,
    const array& src,
    array& out,
    omarchy::CommandEncoder& encoder) {
  out.set_data(allocate_omarchy(out.nbytes()));
  if (out.size() == 0) {
    return;
  }
  size_t row_length = src.shape(-1);
  size_t rows = src.size() / row_length;
  uint32_t output_size = checked_u32(rows, operation_name, out);
  omarchy::ComputeParams params;
  params.count = output_size;
  params.operation = is_max ? 1u : 0u;
  params.reduce_size = checked_u32(row_length, operation_name, out);
  params.output_size = output_size;
  params.lhs_offset = checked_item_offset(
      src, src.size(), operation_name, out);
  params.output_offset = checked_item_offset(out, out.size(), operation_name, out);
  std::array<omarchy::ComputeBinding, 3> bindings{
      binding(src), binding(src), binding(out)};
  omarchy::ComputeKernel kernel;
  switch (src.dtype()) {
    case int8:
      kernel = omarchy::ComputeKernel::ArgReduceI8;
      break;
    case uint8:
      kernel = omarchy::ComputeKernel::ArgReduceU8;
      break;
    case int16:
      kernel = omarchy::ComputeKernel::ArgReduceI16;
      break;
    case uint16:
      kernel = omarchy::ComputeKernel::ArgReduceU16;
      break;
    case int32:
      kernel = omarchy::ComputeKernel::ArgReduceI32;
      break;
    case uint32:
      kernel = omarchy::ComputeKernel::ArgReduceU32;
      break;
    case int64:
      kernel = omarchy::ComputeKernel::ArgReduceI64;
      break;
    case uint64:
      kernel = omarchy::ComputeKernel::ArgReduceU64;
      break;
    case float16:
      kernel = omarchy::ComputeKernel::ArgReduceF16;
      break;
    case bfloat16:
      kernel = omarchy::ComputeKernel::ArgReduceBF16;
      break;
    default:
      kernel = omarchy::ComputeKernel::ArgReduceF32;
      break;
  }
  encoder.dispatch_compute(
      kernel,
      bindings,
      params,
      std::min(output_size, omarchy::kMaxComputeGroupCountX));
}
void ArgReduce::eval_gpu(const std::vector<array>& inputs, array& out) {
  const array& input = inputs.at(0);
  auto [reduce_type, axis] = state();
  // ArgReduce::name() is "ArgReduce"; errors name the concrete operation
  // the way the upstream Reduce primitive names Sum and Max.
  std::string operation_name =
      reduce_type == ArgReduce::ArgMax ? "ArgMax" : "ArgMin";
  auto& encoder = omarchy::get_command_encoder(out.primitive().stream());

  // The output carries indices, so the float checks apply to the input
  // only and the output must be uint32.
  switch (input.dtype()) {
    case float16:
    case bfloat16:
    case float32:
    case int8:
    case uint8:
    case int16:
    case uint16:
    case int32:
    case uint32:
    case int64:
    case uint64:
      break;
    default:
      omarchy::unsupported(operation_name + " dtype", out);
  }
  if ((input.dtype() == int64 || input.dtype() == uint64) &&
      !encoder.device().capabilities().shader_int64) {
    omarchy::unsupported(operation_name + " int64 capability", out);
  }
  const auto& capabilities = encoder.device().capabilities();
  if (input.dtype() == float16 &&
      (!capabilities.shader_float16 ||
       !capabilities.storage_buffer_16bit_access)) {
    omarchy::unsupported(operation_name + " float16 capability", out);
  }
  if (input.dtype() == bfloat16 &&
      (!capabilities.storage_buffer_16bit_access ||
       !capabilities.shader_int16)) {
    omarchy::unsupported(operation_name + " bfloat16 capability", out);
  }
  if (out.dtype() != uint32) {
    omarchy::unsupported(operation_name + " output dtype", out);
  }
  bool is_max = reduce_type == ArgReduce::ArgMax;
  if (axis == input.ndim() - 1) {
    std::optional<array> dense_temp;
    const array& src = ensure_dense(
        input,
        input.flags().row_contiguous,
        dense_temp,
        encoder,
        out.primitive().stream());
    dispatch_arg_reduce_suffix(operation_name, is_max, src, out, encoder);
    return;
  }
  // Non-suffix axis: move the reduced axis to the end through the
  // strided-copy engine. The uint32 output drops the axis, so it is
  // already the dense rows the suffix kernel writes and needs no
  // backward permutation.
  AxisMoveTables tables = axis_move_tables(input, axis);
  array moved(tables.shape, input.dtype(), nullptr, {});
  moved.set_data(allocate_omarchy(moved.nbytes()));
  if (out.size() != 0) {
    copy_gpu_inplace(
        input,
        moved,
        tables.shape,
        tables.in_strides,
        tables.moved_strides,
        /* i_offset = */ 0,
        /* o_offset = */ 0,
        CopyType::GeneralGeneral,
        out.primitive().stream());
  }
  dispatch_arg_reduce_suffix(operation_name, is_max, moved, out, encoder);
  encoder.add_temporary(moved);
}
void ArgSort::eval_gpu(const std::vector<array>& inputs, array& out) {
  const array& input = inputs.at(0);
  auto& encoder = omarchy::get_command_encoder(out.primitive().stream());
  require_sort_dtype("ArgSort", input, out, true, encoder);
  dispatch_sort_any_axis("ArgSort", input, out, state(), true, encoder);
}
// BitwiseBinary carries the upstream op enum (and/or/xor and both
// shifts). int32 and uint32 run through the integer kernel, where `>>`
// follows the operand signedness the way the upstream C++ operators
// do; other widths keep the named rejection.
void BitwiseBinary::eval_gpu(const std::vector<array>& inputs, array& out) {
  uint32_t operation;
  switch (state()) {
    case BitwiseBinary::And:
      operation = IntBitwiseAndOperation;
      break;
    case BitwiseBinary::Or:
      operation = IntBitwiseOrOperation;
      break;
    case BitwiseBinary::Xor:
      operation = IntBitwiseXorOperation;
      break;
    case BitwiseBinary::LeftShift:
      operation = IntLeftShiftOperation;
      break;
    default:
      operation = IntRightShiftOperation;
      break;
  }
  dispatch_int_elementwise(name(), operation, inputs, out);
}
void BitwiseInvert::eval_gpu(const std::vector<array>& inputs, array& out) {
  dispatch_int_elementwise(name(), IntInvertOperation, inputs, out);
}
void BlockMaskedMM::eval_gpu(const std::vector<array>& inputs, array& out) {
  const std::string tag = name();
  // Upstream restricts BlockMaskedMM to float32.
  if (out.dtype() != float32) {
    omarchy::unsupported(tag + " dtype", out);
  }
  const Stream& s = out.primitive().stream();
  auto& encoder = omarchy::get_command_encoder(s);
  const array& a_in = inputs[0];
  const array& b_in = inputs[1];
  require_float_dtype(tag, a_in, out, encoder);
  require_float_dtype(tag, b_in, out, encoder);
  size_t count = inputs.size();
  if (count < 3 || count > 5) {
    omarchy::unsupported(tag + " mask arity", out);
  }
  // Allocate the output before size checks so a too-large out surfaces
  // as the named UNSUPPORTED error before we burn cycles on temp mask
  // copies or temporary array allocations.
  out.set_data(allocate_omarchy(out.nbytes()));
  if (out.size() == 0) {
    return;
  }

  bool has_out_mask = count == 3 || count == 5;
  bool has_op_mask = count >= 4;
  const array& out_mask = inputs[2];
  const array& lhs_mask = inputs[count - 2];
  const array& rhs_mask = inputs[count - 1];

  // Bool casts read packed words in physical order. Preserve the zero-copy
  // path for dense and broadcast masks; materialize transposed layouts first
  // so the cast values and the strides used by the block-mask kernel describe
  // the same logical order.
  std::optional<array> lhs_dense;
  std::optional<array> rhs_dense;
  std::optional<array> out_dense;
  std::optional<array> lhs_values;
  std::optional<array> rhs_values;
  std::optional<array> out_values;
  const array* lhs_logical = &lhs_mask;
  const array* rhs_logical = &rhs_mask;
  const array* out_logical = &out_mask;
  auto resolve_mask = [&](
                          const array& mask,
                          std::optional<array>& dense,
                          std::optional<array>& values,
                          const array*& logical) {
    if (mask.dtype() == bool_) {
      logical = &ensure_dense(
          mask, is_flat_readable(mask), dense, encoder, s);
      values = cast_bool_mask(*logical, tag, out, encoder);
    } else if (mask.dtype() != float32) {
      omarchy::unsupported(tag + " mask dtype", out);
    }
  };
  if (has_op_mask) {
    resolve_mask(lhs_mask, lhs_dense, lhs_values, lhs_logical);
    resolve_mask(rhs_mask, rhs_dense, rhs_values, rhs_logical);
  }
  if (has_out_mask) {
    resolve_mask(out_mask, out_dense, out_values, out_logical);
  }
  const array& lhs_values_ref = lhs_values ? *lhs_values : lhs_mask;
  const array& rhs_values_ref = rhs_values ? *rhs_values : rhs_mask;
  const array& out_values_ref = out_values ? *out_values : out_mask;

  int m = a_in.shape(-2);
  int k = a_in.shape(-1);
  int n = b_in.shape(-1);

  // Masked operands: one dispatch materializes the operand into a dense
  // temp while applying the block mask; the plain tiled matmul then runs
  // on the temps. Without operand masks the matmul dispatch handles
  // layouts directly, and masked blocks contribute exactly zero to every
  // output element, so skipping their arithmetic is unnecessary for the
  // contract.
  array a_masked = a_in;
  array b_masked = b_in;
  if (has_op_mask) {
    a_masked = make_dense_temp(a_in, tag, out, encoder);
    dispatch_block_mask(
        a_in,
        lhs_values_ref,
        *lhs_logical,
        a_masked,
        m,
        k,
        block_size_,
        tag,
        out);
    b_masked = make_dense_temp(b_in, tag, out, encoder);
    dispatch_block_mask(
        b_in,
        rhs_values_ref,
        *rhs_logical,
        b_masked,
        k,
        n,
        block_size_,
        tag,
        out);
  }
  dispatch_matmul(tag, {a_masked, b_masked}, out, 1.0f, 0.0f, false, s);
  if (has_out_mask) {
    dispatch_block_mask(
        out, out_values_ref, *out_logical, out, m, n, block_size_, tag, out);
  }
}

void GatherMM::eval_gpu(const std::vector<array>& inputs, array& out) {
  const std::string tag = name();
  const Stream& s = out.primitive().stream();
  auto& encoder = omarchy::get_command_encoder(s);
  const array& a_in = inputs[0];
  const array& b_in = inputs[1];
  const array& lhs = inputs[2];
  const array& rhs = inputs[3];
  require_float_dtype(tag, a_in, out, encoder);
  require_float_dtype(tag, b_in, out, encoder);
  if (lhs.dtype() != uint32 || rhs.dtype() != uint32) {
    omarchy::unsupported(tag + " index dtype", out);
  }
  if (lhs.shape() != rhs.shape()) {
    omarchy::unsupported(tag + " index shape", out);
  }
  std::optional<array> lhs_temp;
  std::optional<array> rhs_temp;
  const array& lhs_d =
      ensure_dense(lhs, lhs.flags().row_contiguous, lhs_temp, encoder, s);
  const array& rhs_d =
      ensure_dense(rhs, rhs.flags().row_contiguous, rhs_temp, encoder, s);

  // The gathered base offsets are index * matrix_size, so both operands
  // must be dense batched stacks; a uniform in-matrix transposition keeps
  // its zero-copy path, anything else materializes first.
  bool a_transposed = is_dense_batched_matrix(a_in, true);
  std::optional<array> a_materialized;
  if (!a_transposed && !is_dense_batched_matrix(a_in, false)) {
    a_materialized = materialize_batched_matrix(a_in, tag, out, s);
  }
  const array& a = a_materialized ? *a_materialized : a_in;
  bool b_transposed = is_dense_batched_matrix(b_in, true);
  std::optional<array> b_materialized;
  if (!b_transposed && !is_dense_batched_matrix(b_in, false)) {
    b_materialized = materialize_batched_matrix(b_in, tag, out, s);
  }
  const array& b = b_materialized ? *b_materialized : b_in;
  if (a.shape(-1) != b.shape(-2)) {
    omarchy::unsupported(tag + " matrix dimensions", out);
  }

  out.set_data(allocate_omarchy(out.nbytes()));
  int m = a.shape(-2);
  int k = a.shape(-1);
  int n = b.shape(-1);
  size_t batch_count = lhs_d.size();
  if (out.size() == 0) {
    return;
  }
  if (batch_count > omarchy::kMaxComputeGroupCountX) {
    omarchy::unsupported(tag + " batch count", out);
  }

  // Binding 2 packs the index words [all lhs | all rhs]; GatherMM has no
  // C operand, so its binding slot is free.
  if (lhs_d.offset() % 4 != 0 || rhs_d.offset() % 4 != 0) {
    omarchy::unsupported(tag + " index byte offset", out);
  }
  array indices(
      Shape{static_cast<int>(2 * batch_count)}, uint32, nullptr, {});
  array::Flags index_flags;
  index_flags.contiguous = true;
  index_flags.row_contiguous = true;
  index_flags.col_contiguous = true;
  indices.set_data(
      allocate_omarchy(indices.nbytes()),
      indices.size(),
      Strides{1},
      index_flags,
      0);
  encoder.add_temporary(indices);
  encoder.add_temporary(lhs_d);
  encoder.add_temporary(rhs_d);
  encoder.copy_buffer(
      binding(lhs_d).buffer,
      binding(indices).buffer,
      lhs_d.nbytes(),
      static_cast<VkDeviceSize>(lhs_d.offset()),
      0);
  encoder.copy_buffer(
      binding(rhs_d).buffer,
      binding(indices).buffer,
      rhs_d.nbytes(),
      static_cast<VkDeviceSize>(rhs_d.offset()),
      static_cast<VkDeviceSize>(lhs_d.nbytes()));

  omarchy::ComputeParams params;
  params.count = checked_u32(out.size(), tag, out);
  params.lhs_size = checked_u32(a.size(), tag, out);
  params.rhs_size = checked_u32(b.size(), tag, out);
  params.reduce_size = checked_u32(k, tag, out);
  params.output_size = params.count;
  params.lhs_offset = checked_item_offset(a, a.size(), tag, out);
  params.rhs_offset = checked_item_offset(b, b.size(), tag, out);
  params.output_offset = checked_item_offset(out, out.size(), tag, out);
  params.aux_size = checked_u32(batch_count, tag, out);
  params.aux_offset = 0;
  params.matrix_m = checked_u32(m, tag, out);
  params.matrix_n = checked_u32(n, tag, out);
  params.matrix_k = checked_u32(k, tag, out);
  params.flags = (b_transposed ? 1u : 0u) | (a_transposed ? 4u : 0u);
  if (!omarchy::compute_index_span_fits(params.lhs_offset, a.size()) ||
      !omarchy::compute_index_span_fits(params.rhs_offset, b.size())) {
    omarchy::unsupported(tag + " index span", out);
  }
  std::array<omarchy::ComputeBinding, 4> bindings{
      binding(a), binding(b), binding(indices), binding(out)};
  auto kernel = select_float_kernel(
      out.dtype(),
      omarchy::ComputeKernel::GatherMmF32,
      omarchy::ComputeKernel::GatherMmF16,
      omarchy::ComputeKernel::GatherMmBF16);
  encoder.dispatch_compute(
      kernel,
      bindings,
      params,
      matrix_group_count(params.matrix_n),
      matrix_group_count(params.matrix_m),
      checked_u32(batch_count, tag, out));
}

void SegmentedMM::eval_gpu(const std::vector<array>& inputs, array& out) {
  const std::string tag = name();
  const Stream& s = out.primitive().stream();
  auto& encoder = omarchy::get_command_encoder(s);
  const array& a_in = inputs[0];
  const array& b_in = inputs[1];
  const array& segments = inputs[2];
  require_float_dtype(tag, a_in, out, encoder);
  require_float_dtype(tag, b_in, out, encoder);
  if (segments.dtype() != uint32) {
    omarchy::unsupported(tag + " segments dtype", out);
  }
  if (a_in.ndim() != 2 || b_in.ndim() != 2 || a_in.shape(1) != b_in.shape(0)) {
    omarchy::unsupported(tag + " matrix dimensions", out);
  }
  // The segmented shader indexes rows by matrix_k, so only the natural
  // gaps pass through; anything else materializes first.
  std::optional<array> a_materialized;
  uint32_t a_gap = 0;
  bool a_transposed = false;
  classify_matmul_operand(
      a_in, a_transposed, a_gap, a_materialized, tag, out, s);
  uint32_t a_natural = a_transposed ? a_in.shape(0) : a_in.shape(1);
  if (!a_materialized && a_gap != a_natural) {
    a_materialized = materialize_batched_matrix(a_in, tag, out, s);
    a_transposed = false;
  }
  const array& a = a_materialized ? *a_materialized : a_in;
  std::optional<array> b_materialized;
  uint32_t b_gap = 0;
  bool b_transposed = false;
  classify_matmul_operand(
      b_in, b_transposed, b_gap, b_materialized, tag, out, s);
  uint32_t b_natural = b_transposed ? b_in.shape(0) : b_in.shape(1);
  if (!b_materialized && b_gap != b_natural) {
    b_materialized = materialize_batched_matrix(b_in, tag, out, s);
    b_transposed = false;
  }
  const array& b = b_materialized ? *b_materialized : b_in;
  std::optional<array> segments_temp;
  const array& segments_d = ensure_dense(
      segments, segments.flags().row_contiguous, segments_temp, encoder, s);

  out.set_data(allocate_omarchy(out.nbytes()));
  if (out.size() == 0) {
    return;
  }
  int m = a.shape(0);
  int k = a.shape(1);
  int n = b.shape(1);
  size_t num_segments = segments_d.size() / 2;
  if (num_segments * static_cast<size_t>(m) * n != out.size()) {
    omarchy::unsupported(tag + " segment count", out);
  }
  if (num_segments > omarchy::kMaxComputeGroupCountX) {
    omarchy::unsupported(tag + " segment count", out);
  }
  omarchy::ComputeParams params;
  params.count = checked_u32(out.size(), tag, out);
  params.lhs_size = checked_u32(a.size(), tag, out);
  params.rhs_size = checked_u32(b.size(), tag, out);
  params.reduce_size = checked_u32(k, tag, out);
  params.output_size = params.count;
  params.lhs_offset = checked_item_offset(a, a.size(), tag, out);
  params.rhs_offset = checked_item_offset(b, b.size(), tag, out);
  params.output_offset = checked_item_offset(out, out.size(), tag, out);
  params.aux_offset =
      checked_item_offset(segments_d, segments_d.size(), tag, out);
  params.matrix_m = checked_u32(m, tag, out);
  params.matrix_n = checked_u32(n, tag, out);
  params.matrix_k = checked_u32(k, tag, out);
  params.flags = (b_transposed ? 1u : 0u) | (a_transposed ? 4u : 0u);
  if (!omarchy::compute_index_span_fits(params.lhs_offset, a.size()) ||
      !omarchy::compute_index_span_fits(params.rhs_offset, b.size())) {
    omarchy::unsupported(tag + " index span", out);
  }
  std::array<omarchy::ComputeBinding, 4> bindings{
      binding(a), binding(b), binding(segments_d), binding(out)};
  auto kernel = select_float_kernel(
      out.dtype(),
      omarchy::ComputeKernel::SegmentedMmF32,
      omarchy::ComputeKernel::SegmentedMmF16,
      omarchy::ComputeKernel::SegmentedMmBF16);
  encoder.dispatch_compute(
      kernel,
      bindings,
      params,
      matrix_group_count(params.matrix_n),
      matrix_group_count(params.matrix_m),
      checked_u32(num_segments, tag, out));
}

OMARCHY_UNARY(Ceil, CeilOperation)
namespace {

// Wave 7: linear algebra. Every factorization primitive runs one
// workgroup per batch matrix in float32 only, matching the upstream CPU
// dtype contract (float32/float64; float64 has no Omarchy transport).
// Iterative kernels report failure through a u32 status word that the
// host checks behind a synchronize, so a degenerate input raises a named
// error instead of returning a silently wrong factorization.

size_t linalg_batch_count(const Shape& shape) {
  size_t batch = 1;
  int batch_rank = static_cast<int>(shape.size()) - 2;
  for (int axis = 0; axis < batch_rank; ++axis) {
    batch *= static_cast<size_t>(shape[axis]);
  }
  return batch;
}

void linalg_require_f32(
    const std::string& name,
    const array& in,
    const array& out) {
  if (in.dtype() != float32) {
    omarchy::unsupported(name + " dtype", out);
  }
}

void linalg_copy_dense(
    const array& in,
    array& work,
    Stream stream) {
  copy_gpu_inplace(
      in,
      work,
      in.shape(),
      in.strides(),
      work.strides(),
      0,
      0,
      in.flags().row_contiguous ? CopyType::Vector : CopyType::General,
      stream);
}

// Join pending work and turn any kernel-pinned status word into the
// named failure for this primitive.
void linalg_check_status(
    array& scratch,
    omarchy::CommandEncoder& encoder,
    const char* message,
    const array& out) {
  encoder.synchronize();
  const uint32_t* words = scratch.data<uint32_t>();
  for (size_t i = 0; i < scratch.size(); ++i) {
    if (words[i] != 0) {
      throw std::runtime_error(message);
    }
  }
}

} // namespace

void Cholesky::eval_gpu(const std::vector<array>& inputs, array& out) {
  const array& in = inputs.at(0);
  auto& encoder = omarchy::get_command_encoder(out.primitive().stream());
  linalg_require_f32(name(), in, out);
  if (in.shape(-1) != in.shape(-2)) {
    omarchy::unsupported(std::string("non-square ") + name(), out);
  }
  const uint32_t n = checked_u32(in.shape(-1), name(), out);
  const uint32_t batch =
      checked_u32(linalg_batch_count(in.shape()), name(), out);
  out.set_data(allocate_omarchy(out.nbytes()));
  if (out.size() == 0) {
    return;
  }
  linalg_copy_dense(in, out, out.primitive().stream());
  auto scratch = make_u32_scratch(std::max<size_t>(batch, 1u), encoder);
  dispatch_clear_u32(scratch, 0, encoder);
  omarchy::ComputeParams params;
  params.matrix_n = n;
  params.output_size = batch;
  params.flags = state() ? 1u : 0u;
  std::array<omarchy::ComputeBinding, 2> bindings{
      binding(out), binding(scratch)};
  encoder.dispatch_compute(
      omarchy::ComputeKernel::LinalgCholeskyF32, bindings, params, batch);
  linalg_check_status(
      scratch,
      encoder,
      "[Cholesky::eval_gpu] Cholesky decomposition requires a positive"
      " definite matrix.",
      out);
}
void Compiled::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  omarchy::eval_compiled_tape(
      tape_, inputs_, outputs_, inputs, outputs, stream());
}
void Conjugate::eval_gpu(const std::vector<array>& inputs, array& out) {
  // Upstream conjugate() returns real input unchanged before a
  // primitive is built (mlx/ops.cpp), so eval only ever sees
  // complex64 here. The op negates the imaginary component.
  dispatch_complex(
      name(),
      ComplexConjugate,
      inputs,
      out,
      out.primitive().stream());
}
// General direct convolution for the channels-last layouts upstream
// hands this primitive: input [N, (D,) (H,) W, C] and weight
// [O, (kD,) (kH,) kW, C_per_group], both row-major. One thread owns
// each output element for ordinary kernels. Large windows split into
// bounded per-output chunks and a float32 scratch reduction so shader
// trip limits cannot truncate the result. Index guards contribute zero
// for padding. The kernel covers
// every combination the public ops build: groups including the
// depthwise case, the flip that conv_transpose uses, input dilation,
// kernel dilation, strides, and asymmetric padding, over 1D, 2D, and
// 3D spatial ranks. Lower ranks ride the same kernel with degenerate
// extent-one outer axes, matching the upstream CPU reference where
// slow_conv_1D is slow_conv_2D with an extent-one height axis.
void Convolution::eval_gpu(const std::vector<array>& inputs, array& out) {
  const array& in = inputs.at(0);
  const array& wt = inputs.at(1);
  auto& encoder = omarchy::get_command_encoder(out.primitive().stream());
  require_float_dtype("Convolution", in, out, encoder);
  require_float_dtype("Convolution", wt, out, encoder);
  const int spatial = in.ndim() - 2;
  if (spatial < 1 || spatial > 3 || in.ndim() != wt.ndim()) {
    omarchy::unsupported("Convolution shapes", out);
  }
  // Materialize operands whose strides are not the standard
  // channels-last and O(HKW) row-major layouts; cache slices and
  // transposes compose that way. The engine keeps each temp alive
  // until the submission lands.
  std::optional<array> in_materialized;
  std::optional<array> wt_materialized;
  if (!in.flags().row_contiguous) {
    in_materialized = materialize_batched_matrix(
        in, "Convolution", out, out.primitive().stream());
  }
  if (!wt.flags().row_contiguous) {
    wt_materialized = materialize_batched_matrix(
        wt, "Convolution", out, out.primitive().stream());
  }
  const array& x = in_materialized ? *in_materialized : in;
  const array& w = wt_materialized ? *wt_materialized : wt;

  auto axis_extent =
      [](const array& a, int axis) { return a.shape(1 + axis); };
  const int batch = x.shape(0);
  const int in_channels = x.shape(x.ndim() - 1);
  const int out_channels = w.shape(0);
  const int in_channels_per_group = w.shape(w.ndim() - 1);
  // Per-axis extents outermost-first with extent-one fill.
  std::vector<int> in_ext(spatial), kern_ext(spatial);
  for (int axis = 0; axis < spatial; ++axis) {
    in_ext[axis] = axis_extent(x, axis);
    kern_ext[axis] = axis_extent(w, axis);
  }
  if (groups_ <= 0 || in_channels_per_group * groups_ != in_channels ||
      out_channels % groups_ != 0 || out.shape(0) != batch ||
      out.shape(out.ndim() - 1) != out_channels) {
    omarchy::unsupported("Convolution shapes", out);
  }

  out.set_data(allocate_omarchy(out.nbytes()));
  if (out.size() == 0) {
    return;
  }
  uint32_t total = checked_u32(out.size(), "Convolution", out);
  omarchy::ComputeParams params;
  // Push-constant routing mirrors shaders/conv.comp: spatial extents,
  // kernel window, and the conv parameters ride the generic dims
  // fields because the fixed ComputeParams layout has no conv block.
  // Axes are packed outermost-first (depth, height, width); absent
  // axes read back as extent one and unit parameters. The shader
  // comment carries the full table.
  auto axis_or_one = [&](const std::vector<int>& values, int axis) {
    return static_cast<uint32_t>(axis < spatial ? values[axis] : 1);
  };
  auto pad_lo_axis = [&](int axis) {
    return axis_or_one(padding_lo_, axis);
  };
  bool unit_window = !flip_ && groups_ == 1;
  for (int axis = 0; axis < spatial; ++axis) {
    if (kern_ext[axis] != 1 || in_ext[axis] != out.shape(1 + axis) ||
        axis_or_one(kernel_strides_, axis) != 1u ||
        pad_lo_axis(axis) != 0u ||
        axis_or_one(kernel_dilation_, axis) != 1u ||
        axis_or_one(input_dilation_, axis) != 1u) {
      unit_window = false;
    }
  }
  if (unit_window) {
    const uint32_t matrix_n =
        checked_u32(out_channels, "Convolution", out);
    const uint32_t matrix_k =
        checked_u32(in_channels, "Convolution", out);
    const uint32_t matrix_m = total / matrix_n;
    omarchy::ComputeParams gemm;
    gemm.count = total;
    gemm.reduce_size = matrix_k;
    gemm.output_size = total;
    gemm.matrix_m = matrix_m;
    gemm.matrix_n = matrix_n;
    gemm.matrix_k = matrix_k;
    gemm.flags = 1u;
    gemm.lhs_gap = matrix_k;
    gemm.rhs_gap = matrix_k;
    gemm.lhs_offset = checked_item_offset(x, x.size(), "Convolution", out);
    gemm.rhs_offset = checked_item_offset(w, w.size(), "Convolution", out);
    gemm.output_offset =
        checked_item_offset(out, out.size(), "Convolution", out);
    gemm.lhs_size = checked_u32(x.size(), "Convolution", out);
    gemm.rhs_size = checked_u32(w.size(), "Convolution", out);
    std::array<omarchy::ComputeBinding, 4> bindings{
        binding(x), binding(w), binding(out), binding(out)};
    auto gemm_kernel = select_float_kernel(
        out.dtype(),
        omarchy::ComputeKernel::MatmulF32,
        omarchy::ComputeKernel::MatmulF16,
        omarchy::ComputeKernel::MatmulBF16);
    encoder.dispatch_compute(
        gemm_kernel,
        bindings,
        gemm,
        matrix_group_count(matrix_n),
        matrix_group_count(matrix_m),
        1u);
    return;
  }

  params.count = total;
  params.reduce_size = checked_u32(in_channels_per_group, "Convolution", out);
  params.lhs_offset = checked_item_offset(x, x.size(), "Convolution", out);
  params.rhs_offset = checked_item_offset(w, w.size(), "Convolution", out);
  params.output_offset =
      checked_item_offset(out, out.size(), "Convolution", out);
  params.matrix_m =
      checked_u32(out.shape(out.ndim() - 2), "Convolution", out);
  params.matrix_n = checked_u32(out_channels, "Convolution", out);
  params.rhs_gap = checked_u32(groups_, "Convolution", out);
  params.flags = flip_ ? 1u : 0u;
  params.dims = checked_u32(spatial, "Convolution", out);
  params.lhs_size = axis_or_one(input_dilation_, 0);
  params.rhs_size = axis_or_one(input_dilation_, 1);
  uint32_t input_dilation_axis_2 = axis_or_one(input_dilation_, 2);
  std::memcpy(
      &params.alpha, &input_dilation_axis_2, sizeof(input_dilation_axis_2));
  // Input extents: depth, height, width.
  params.output_size =
      checked_u32(spatial == 3 ? in_ext[0] : 1, "Convolution", out);
  params.matrix_k =
      checked_u32(spatial >= 2 ? in_ext[spatial - 2] : 1, "Convolution", out);
  params.aux_size =
      checked_u32(in_ext[spatial - 1], "Convolution", out);
  // Kernel extents: depth, height, width.
  params.lhs_gap =
      checked_u32(spatial == 3 ? kern_ext[0] : 1, "Convolution", out);
  params.aux_offset =
      checked_u32(spatial >= 2 ? kern_ext[spatial - 2] : 1, "Convolution", out);
  params.operation =
      checked_u32(kern_ext[spatial - 1], "Convolution", out);
  // Output extents outermost-first. The innermost axis rides
  // matrix_m; shape[0..1] carry depth/height when present.
  params.shape[0] =
      checked_u32(spatial >= 2 ? out.shape(1) : 1, "Convolution", out);
  params.shape[1] =
      checked_u32(spatial == 3 ? out.shape(2) : 1, "Convolution", out);
  // Kernel stride and low padding remain independent of kernel flip.
  for (int axis = 0; axis < 3; ++axis) {
    params.in_strides[axis] = axis_or_one(kernel_strides_, axis);
    params.out_strides[axis] = pad_lo_axis(axis);
  }
  // Kernel dilation: depth, height, width.
  params.shape[2] = axis_or_one(kernel_dilation_, spatial == 3 ? 0 : 3);
  params.shape[3] =
      axis_or_one(kernel_dilation_, spatial >= 2 ? spatial - 2 : 3);
  params.out_strides[3] = axis_or_one(kernel_dilation_, spatial - 1);
  auto kernel = select_float_kernel(
      out.dtype(),
      omarchy::ComputeKernel::ConvF32,
      omarchy::ComputeKernel::ConvF16,
      omarchy::ComputeKernel::ConvBF16);
  constexpr uint32_t kConvChunkProducts = 4096;
  uint32_t kernel_products = checked_u32(
      w.size() / static_cast<size_t>(out_channels), "Convolution", out);
  uint32_t chunks = kernel_products == 0
      ? 1u
      : 1u + (kernel_products - 1u) / kConvChunkProducts;
  if (chunks == 1) {
    std::array<omarchy::ComputeBinding, 4> bindings{
        binding(x), binding(w), binding(x), binding(out)};
    encoder.dispatch_compute(
        kernel,
        bindings,
        params,
        omarchy::compute_dispatch_group_count(total));
    return;
  }

  constexpr uint32_t kConvScratchElements = 1u << 20;
  uint32_t output_tile = std::min(
      total, std::max(1u, kConvScratchElements / chunks));
  uint32_t scratch_elements = output_tile * chunks;
  uint32_t second_level_chunks =
      1u + (chunks - 1u) / kConvChunkProducts;
  uint32_t secondary_elements = chunks > kConvChunkProducts
      ? output_tile * second_level_chunks
      : 0u;
  array partials(
      Shape{static_cast<int>(scratch_elements + secondary_elements)},
      float32,
      nullptr,
      {});
  partials.set_data(allocate_omarchy(partials.nbytes()));
  encoder.add_temporary(partials);
  std::array<omarchy::ComputeBinding, 4> bindings{
      binding(x), binding(w), binding(partials), binding(out)};
  const auto base_params = params;
  for (uint32_t output_base = 0; output_base < total;) {
    uint32_t tile_count = std::min(output_tile, total - output_base);
    params = base_params;
    params.flags = (flip_ ? 1u : 0u) | 2u;
    params.in_strides[3] = output_base;
    params.beta = static_cast<float>(kConvChunkProducts);
    params.count = tile_count * chunks;
    encoder.dispatch_compute(
        kernel,
        bindings,
        params,
        omarchy::compute_dispatch_group_count(params.count));

    uint32_t source_chunks = chunks;
    uint32_t source_offset = 0;
    uint32_t destination_offset = scratch_elements;
    while (source_chunks > 1u) {
      uint32_t next_chunks =
          1u + (source_chunks - 1u) / kConvChunkProducts;
      params.flags = 4u | (next_chunks == 1u ? 8u : 0u);
      params.lhs_size = source_chunks;
      params.rhs_size = source_offset;
      params.output_size = destination_offset;
      params.matrix_m = tile_count;
      params.count = tile_count * next_chunks;
      encoder.dispatch_compute(
          kernel,
          bindings,
          params,
          omarchy::compute_dispatch_group_count(params.count));
      source_chunks = next_chunks;
      std::swap(source_offset, destination_offset);
    }
    output_base += tile_count;
  }
}
// The GLSL built-in sin/cos keep upstream-grade accuracy only for
// arguments the driver's range reduction survives. Measured on the M1
// Honeykrisp (scalar sweep, 2026-09-02): built-in error 2.8e-5 at 1e3,
// 4.5e-4 at 12345, 4.8e-3 at 123457, then 1e-2 and worse toward 1e6
// and total collapse from there; llvmpipe stays accurate far higher,
// but the gate is one device-independent contract. kTrigArgumentLimit
// is 1e5, chosen for the consumer that matters: fast::RoPE is a
// fallback composition of exactly these sin/cos calls with
// inv_freq[0] = 1.0, so the limit is the positional ceiling - 1e5
// covers Qwen-class 32k contexts with 3x margin while refusing the
// band where the built-in returns garbage (error 1e-2 and collapsing).
// An in-shader Payne-Hanek fallback was probed on the same device and
// returns garbage of magnitude 1e15+ (its carry chain rides the
// dynamic-indexing shapes this driver miscompiles), so it stays dead
// code. Above the limit the op refuses by name; the compatibility
// matrix counts Sin/Cos as partial with the magnitude named error.
constexpr float kTrigArgumentLimit = 1.0e5f;

void trig_argument_gate(
    const std::string& name,
    const std::vector<array>& inputs,
    const array& out) {
  Stream stream = out.primitive().stream();
  // An empty argument has no magnitude to gate; the max over its
  // size-0 axes would throw upstream. The elementwise kernel below
  // treats the empty output as a no-op.
  if (inputs.at(0).size() == 0) {
    return;
  }
  array magnitude = astype(
      max(abs(inputs.at(0), stream), stream), float32, stream);
  const bool trace_gate =
      std::getenv("MLX_OMARCHY_TRACE_DISPATCH") != nullptr;
  if (trace_gate) {
    fprintf(stderr, "[rtmod] GATE tid=%lu %s enter\n", (unsigned long)syscall(SYS_gettid), name.c_str());
  }
  // Record the magnitude graph into the open batch without the nested
  // blocking eval: a nested eval()'s epilogue skips the signal+commit at
  // eval_nest_depth > 1 (settle semantics), so its synchronizer wait
  // could never be satisfied and parked the dispatching thread until the
  // watchdog fired - the F1 burst signature. settle() schedules the nodes
  // into the open command buffer; the synchronize() below submits that
  // batch and orders the mapped read, which is the ordering this gate
  // actually needs.
  settle({magnitude});
  if (trace_gate) {
    fprintf(stderr, "[rtmod] GATE tid=%lu %s post-eval\n", (unsigned long)syscall(SYS_gettid), name.c_str());
  }
  // The magnitude is read on the host, so the stream must be ordered
  // here: array::item() is eval() plus an immediate mapped read with no
  // completion wait, and an unordered read races this gate's own
  // submission — on hardware it returned recycled-page garbage
  // (~1e9) instead of the real magnitude, aborting compiled 4-bit
  // generation (observed 2026-09-03, first hardware run of the compiled
  // path; llvmpipe executes synchronously and masked it). Same pattern
  // as the reduce host checks: host reads behind a synchronize.
  omarchy::get_command_encoder(stream).synchronize("trig_argument_gate");
  float worst = magnitude.item<float>();
  if (worst > kTrigArgumentLimit) {
    throw std::runtime_error(
        "[omarchy] " + name + " argument magnitude " +
        std::to_string(worst) + " exceeds the built-in accuracy limit " +
        std::to_string(kTrigArgumentLimit) +
        " on this backend: the Vulkan driver's sin/cos range reduction"
        " is untrusted above it, the software Payne-Hanek fallback"
        " miscompiles on this driver, and no other accurate kernel"
        " exists. No silent wrong value and no silent CPU fallback"
        " occurs. Run it on an explicit CPU stream to use the CPU"
        " implementation.");
  }
}
void Cos::eval_gpu(const std::vector<array>& inputs, array& out) {
  if (out.dtype() == complex64) {
    // Upstream std::cos(complex) goes through glibc ccosf, which is
    // (cos a cosh b, -sin a sinh b) for normal-range inputs; the
    // float trig argument gate is irrelevant to the complex path
    // because the accuracy loss it guards against does not apply.
    dispatch_complex(
        name(), ComplexCos, inputs, out, out.primitive().stream());
    return;
  }
  trig_argument_gate(name(), inputs, out);
  dispatch_elementwise(
      name(), CosOperation, inputs, out, out.primitive().stream());
}
void Cosh::eval_gpu(const std::vector<array>& inputs, array& out) {
  if (out.dtype() == complex64) {
    dispatch_complex(
        name(), ComplexCosh, inputs, out, out.primitive().stream());
    return;
  }
  dispatch_elementwise(
      name(), CoshOperation, inputs, out, out.primitive().stream());
}
void Divide::eval_gpu(const std::vector<array>& inputs, array& out) {
  if (out.dtype() == complex64) {
    dispatch_complex(
        name(), ComplexDivide, inputs, out, out.primitive().stream());
    return;
  }
  if (is_int_elementwise_dtype(out.dtype())) {
    // Integer-output Divide is what upstream floor_divide emits for
    // promoted integer inputs; the kernel truncates like the upstream
    // C++ operator/.
    dispatch_int_elementwise(name(), IntDivideOperation, inputs, out);
    return;
  }
  dispatch_elementwise(
      name(), DivideOperation, inputs, out, out.primitive().stream());
}
// DivMod produces the Python floor-division quotient and remainder as
// two same-shaped outputs (upstream DivMod: integral_op applies the
// floor fixup to the truncating quotient, float_op pairs floor(x/y)
// with the adjusted fmod). One kernel dispatch per output; the
// per-element index mapping is 1:1, so a donated input buffer stays
// safe to read and write within one invocation.
void DivMod::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  const array& in_lhs = inputs.at(0);
  const array& in_rhs = inputs.at(1);
  array& quotient = outputs.at(0);
  array& remainder = outputs.at(1);
  auto& encoder = omarchy::get_command_encoder(quotient.primitive().stream());
  std::optional<array> lhs_temp;
  std::optional<array> rhs_temp;
  const array& lhs = ensure_dense(
      in_lhs,
      in_lhs.flags().contiguous,
      lhs_temp,
      encoder,
      quotient.primitive().stream());
  const array& rhs = ensure_dense(
      in_rhs,
      in_rhs.flags().contiguous,
      rhs_temp,
      encoder,
      quotient.primitive().stream());
  auto is_int_dtype = is_int_elementwise_dtype;
  if (is_int_dtype(lhs.dtype()) && is_int_dtype(rhs.dtype()) &&
      is_int_dtype(quotient.dtype()) && is_int_dtype(remainder.dtype())) {
    auto binary_type = get_binary_op_type(lhs, rhs);
    if (lhs.data_size() != lhs.size() || rhs.data_size() != rhs.size()) {
      binary_type = BinaryOpType::General;
    }
    set_binary_op_output_data(lhs, rhs, quotient, binary_type, allocate_omarchy);
    set_binary_op_output_data(
        lhs, rhs, remainder, binary_type, allocate_omarchy);
    if (quotient.size() == 0) {
      return;
    }
    dispatch_int_elementwise_to(
        "DivMod", IntDivModQuotientOperation, lhs, rhs, quotient);
    dispatch_int_elementwise_to(
        "DivMod", IntModuloOperation, lhs, rhs, remainder);
    return;
  }
  require_float_dtype("DivMod", lhs, quotient, encoder);
  require_float_dtype("DivMod", rhs, quotient, encoder);
  bool general_broadcast =
      !is_trailing_broadcast(lhs, quotient) ||
      !is_trailing_broadcast(rhs, quotient);
  auto binary_type = get_binary_op_type(lhs, rhs);
  if (lhs.data_size() != lhs.size() || rhs.data_size() != rhs.size()) {
    binary_type = BinaryOpType::General;
  }
  set_binary_op_output_data(lhs, rhs, quotient, binary_type, allocate_omarchy);
  set_binary_op_output_data(
      lhs, rhs, remainder, binary_type, allocate_omarchy);
  if (quotient.size() == 0) {
    return;
  }
  dispatch_float_elementwise_to(
      "DivMod",
      DivQuotientFloatOperation,
      lhs,
      rhs,
      quotient,
      general_broadcast,
      encoder);
  dispatch_float_elementwise_to(
      "DivMod",
      RemainderFloatOperation,
      lhs,
      rhs,
      remainder,
      general_broadcast,
      encoder);
}
void Equal::eval_gpu(const std::vector<array>& inputs, array& out) {
  // equal_nan_ lives on the Equal primitive; ops.cpp sets it from
  // mx.array_equal(..., equal_nan=True) and mx.allclose(..., equal_nan=True),
  // and clears it whenever the dtype is not inexact. The kernel honors
  // params.flags bit 0 in compare.comp.
  dispatch_comparison(name(), CompareEqual, inputs, out, equal_nan_ ? 1u : 0u);
}
OMARCHY_UNARY(Erf, ErfOperation)
OMARCHY_UNARY(ErfInv, ErfInvOperation)
// complex64 exp matches std::exp(complex<float>) / glibc cexpf:
// exp(a)*(cos b, sin b); the unary dispatch aliases rhs to lhs so
// the existing complex elementwise transport carries it without a
// new shader variant. The float/int/bool paths keep the standard
// elementwise dispatch the macro would have produced.
void Exp::eval_gpu(const std::vector<array>& inputs, array& out) {
  if (out.dtype() == complex64) {
    dispatch_complex(
        name(), ComplexExp, inputs, out, out.primitive().stream());
    return;
  }
  dispatch_elementwise(
      name(), ExpOperation, inputs, out, out.primitive().stream());
}
OMARCHY_UNARY(Expm1, Expm1Operation)
namespace {

// Wave 8 FFT, extended to general lengths. The FftF32 kernel runs a radix-2
// Cooley-Tukey decimation-in-time pass over shared memory: one workgroup
// computes one 1-D transform of a power-of-two length up to 2048 (16 KiB of
// shared state, the Vulkan minimum guarantee). Longer and non-power-of-two
// lengths decompose at the C++ level:
//   - a composite n factors into stages of the radix-2 pass, Cooley-Tukey
//     style (strided pass, twiddle multiply, recurse), which lifts the old
//     shared-memory cap without any larger footprint; and
//   - a length with no divisor in 2..2048 (a prime, in practice) embeds in a
//     power-of-two circular convolution, Bluestein's chirp-z transform,
//     built from the same radix-2 pass plus elementwise stage kernels.
// Pass flags mirror what the CPU primitive does per call
// (mlx/backend/cpu/fft.cpp): every inverse pass divides by its own axis
// length, so a multi-axis inverse accumulates 1/(n0*n1*...) exactly like the
// upstream scale = 1/nelem.
constexpr uint32_t kFftFlagInverse = 1u;
constexpr uint32_t kFftFlagInputReal = 2u;
constexpr uint32_t kFftFlagInputHalf = 4u;
constexpr uint32_t kFftFlagOutputHalf = 8u;
// Longest single radix-2 pass: bounded by the 16 KiB shared-memory floor,
// not by workgroup size (the kernel strides threads over any length).
constexpr uint32_t kFftMaxDirectLength = 2048;
// Longest Bluestein length: the chirp reduces k*k mod 2n in u32, which
// stays exact only while k < 65536, i.e. while the padded convolution
// length next_pow2(2n-1) stays at or under 65536.
constexpr uint32_t kFftMaxBluesteinLength = 32768;
// Longest decomposed length: Cooley-Tukey twiddle phases are products
// m0*T < n, which must stay exactly representable in float32.
constexpr uint64_t kFftMaxTotalLength = 1ull << 24;

// (sample stride, transform count) of one axis pass over a dense
// row-major buffer of the given shape.
std::pair<uint32_t, uint32_t> fft_pass_geometry(
    const Shape& shape,
    int axis,
    uint32_t length,
    const std::string& name,
    array& out) {
  uint64_t total = 1;
  uint64_t stride = 1;
  for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
    total *= shape[i];
    if (i > axis) {
      stride *= shape[i];
    }
  }
  uint64_t count = total / length;
  if (stride > 0xffffffffull || count > 0xffffffffull) {
    omarchy::unsupported(name + " transform geometry", out);
  }
  return {static_cast<uint32_t>(stride), static_cast<uint32_t>(count)};
}

// Fresh dense row-major buffer for intermediate spectra.
array make_fft_temp(Shape shape, Dtype dtype, const Stream& s) {
  auto& encoder = omarchy::get_command_encoder(s);
  Strides strides(shape.size(), 1);
  for (int axis = static_cast<int>(shape.size()) - 2; axis >= 0; --axis) {
    strides[axis] = strides[axis + 1] * shape[axis + 1];
  }
  array temp(std::move(shape), dtype, nullptr, {});
  array::Flags flags;
  flags.contiguous = true;
  flags.row_contiguous = true;
  auto max_dim = std::max_element(temp.shape().begin(), temp.shape().end());
  flags.col_contiguous = temp.size() <= 1 || temp.size() == *max_dim;
  temp.set_data(
      omarchy::allocator().malloc(temp.nbytes()),
      temp.size(),
      strides,
      flags,
      0);
  encoder.add_temporary(temp);
  return temp;
}

// Dense row-major copy of a float32 FFT input (sliced or transposed
// views). complex64 views keep their named error: the strided copy engine
// has no complex64 path.
array make_fft_input_dense(
    const array& src,
    const std::string& name,
    array& out,
    const Stream& s) {
  auto& encoder = omarchy::get_command_encoder(s);
  Shape shape = src.shape();
  Strides strides(shape.size(), 1);
  for (int axis = static_cast<int>(shape.size()) - 2; axis >= 0; --axis) {
    strides[axis] = strides[axis + 1] * shape[axis + 1];
  }
  array dense(shape, src.dtype(), nullptr, {});
  array::Flags flags;
  flags.contiguous = true;
  flags.row_contiguous = true;
  auto max_dim = std::max_element(shape.begin(), shape.end());
  flags.col_contiguous = dense.size() <= 1 || dense.size() == *max_dim;
  dense.set_data(
      omarchy::allocator().malloc(dense.nbytes()),
      dense.size(),
      strides,
      flags,
      0);
  copy_gpu_inplace(
      src,
      dense,
      shape,
      src.strides(),
      strides,
      /*i_offset=*/0,
      /*o_offset=*/0,
      CopyType::General,
      s);
  encoder.add_temporary(dense);
  return dense;
}

// One radix-2 pass: transform_count workgroups, each transforming
// `length` samples spaced `stride` elements apart inside a dense buffer.
void dispatch_fft_pass(
    const array& src,
    array& dst,
    uint32_t length,
    uint32_t stride,
    uint32_t transform_count,
    uint32_t flags,
    const std::string& name,
    array& out,
    const Stream& s) {
  auto& encoder = omarchy::get_command_encoder(s);
  omarchy::ComputeParams params;
  params.count = transform_count;
  params.operation = flags;
  params.lhs_size = length;
  params.rhs_size = stride;
  params.lhs_offset = checked_item_offset(src, 0, name, out);
  params.output_offset = checked_item_offset(dst, 0, name, out);
  uint32_t group_x =
      std::min<uint32_t>(transform_count, omarchy::kMaxComputeGroupCountX);
  uint32_t remaining = (transform_count + group_x - 1) / group_x;
  uint32_t group_y =
      std::min<uint32_t>(remaining, omarchy::kMaxComputeGroupCountX);
  uint32_t group_z = (remaining + group_y - 1) / group_y;
  if (group_z > omarchy::kMaxComputeGroupCountX) {
    omarchy::unsupported(name + " transform count", out);
  }
  std::array<omarchy::ComputeBinding, 4> bindings{
      binding(src),
      omarchy::ComputeBinding{},
      binding(dst),
      omarchy::ComputeBinding{}};
  encoder.dispatch_compute(
      omarchy::ComputeKernel::FftF32,
      bindings,
      params,
      group_x,
      group_y,
      group_z);
}

struct FftStage {
  enum class Kind : uint8_t {
    Pass,     // radix-2 pass of `length` points on samples spaced
              // axis_stride * stride_mul apart
    Twiddle,  // pointwise w_n^(m0*T) multiply for the split n = (above)*inner
    Bluestein,// whole-length chirp-z convolution for a prime-class length
    Permute   // undo the level's factor transpose: slot j1 + inner*k0 holds
              // the bin for k0 + (length/inner)*j1
  };
  Kind kind;
  uint32_t length;     // Pass/Bluestein/Permute: transform length; Twiddle: n
  uint32_t inner;      // Twiddle/Permute: inner extent M of the split
  uint32_t stride_mul; // Pass/Twiddle/Permute: multiplier on the axis stride
};

// Modes of the fft_stage_f32 elementwise kernel (reduce_size push constant).
constexpr uint32_t kFftStageTwiddle = 0;
constexpr uint32_t kFftStageBluesteinA = 1;
constexpr uint32_t kFftStageBluesteinB = 2;
constexpr uint32_t kFftStageMultiply = 3;
constexpr uint32_t kFftStageBluesteinY = 4;
constexpr uint32_t kFftStagePermute = 5;
constexpr uint32_t kFftStageRealEdge = 6;
// Append the stage list for one n-point transform whose samples sit at
// axis_stride * stride_mul apart. Refuses by name any length the two
// mechanisms cannot carry.
void plan_fft_stages(
    uint64_t n,
    uint32_t stride_mul,
    std::vector<FftStage>& stages,
    const std::string& name,
    array& out) {
  if (n <= kFftMaxDirectLength && (n & (n - 1)) == 0) {
    stages.push_back(
        {FftStage::Kind::Pass,
         static_cast<uint32_t>(n),
         0,
         stride_mul});
    return;
  }
  // Largest factor that splits n into two strictly smaller transforms.
  uint64_t factor = 0;
  uint64_t limit = std::min<uint64_t>(kFftMaxDirectLength, n / 2);
  for (uint64_t d = limit; d >= 2; --d) {
    if (n % d == 0) {
      factor = d;
      break;
    }
  }
  if (factor != 0) {
    uint64_t rest = n / factor;
    // Outer factor first: transform along it, twiddle, then the rest.
    // The twiddle addresses this level's transform-local elements, whose
    // dense spacing carries this level's stride multiplier. The level
    // leaves its output transposed (slot j1 + rest*k0 holds the bin for
    // k0 + factor*j1), so it ends with its own permute stage.
    plan_fft_stages(factor, stride_mul * rest, stages, name, out);
    stages.push_back(
        {FftStage::Kind::Twiddle,
         static_cast<uint32_t>(n),
         static_cast<uint32_t>(rest),
         stride_mul});
    plan_fft_stages(rest, stride_mul, stages, name, out);
    stages.push_back(
        {FftStage::Kind::Permute,
         static_cast<uint32_t>(n),
         static_cast<uint32_t>(rest),
         stride_mul});
    return;
  }
  if (n <= kFftMaxBluesteinLength) {
    stages.push_back(
        {FftStage::Kind::Bluestein,
         static_cast<uint32_t>(n),
         0,
         stride_mul});
    return;
  }
  omarchy::unsupported(
      name + " transform length " + std::to_string(n) +
          " (composite lengths to 2^24 decompose into radix-2 passes;"
          " prime-class lengths to 32768 embed in a Bluestein chirp-z"
          " convolution; this length has neither a divisor in 2..2048 nor"
          " Bluestein chirp headroom)",
      out);
}

std::vector<FftStage> plan_fft_axis(uint64_t n, const std::string& name, array& out) {
  std::vector<FftStage> stages;
  if (n == 0) {
    return stages;
  }
  if (n > kFftMaxTotalLength) {
    omarchy::unsupported(
        name + " transform length " + std::to_string(n) +
            " (decomposed Cooley-Tukey twiddle phases need lengths of at"
            " most 2^24 to stay exact in float32)",
        out);
  }
  plan_fft_stages(n, 1, stages, name, out);
  return stages;
}

// One elementwise fft_stage_f32 dispatch: one thread per element. The
// shader grid-strides over params.count, so one x dimension capped at
// the guaranteed group limit covers any element count (n = 2^24 needs
// 65,536 workgroups of 256, one past that limit).
void dispatch_fft_stage(
    const array& src,
    array& dst,
    uint64_t elements,
    uint32_t mode,
    uint32_t flags,
    uint32_t n,
    uint32_t aux,
    uint32_t stride,
    const std::string& name,
    array& out,
    const Stream& s) {
  auto& encoder = omarchy::get_command_encoder(s);
  omarchy::ComputeParams params;
  params.count = checked_u32(elements, name, out);
  params.operation = flags;
  params.lhs_size = n;
  params.rhs_size = stride;
  params.reduce_size = mode;
  params.output_size = aux;
  params.lhs_offset = checked_item_offset(src, 0, name, out);
  params.output_offset = checked_item_offset(dst, 0, name, out);
  std::array<omarchy::ComputeBinding, 4> bindings{
      binding(src),
      omarchy::ComputeBinding{},
      binding(dst),
      omarchy::ComputeBinding{}};
  encoder.dispatch_compute(
      omarchy::ComputeKernel::FftStageF32,
      bindings,
      params,
      omarchy::compute_dispatch_group_count(params.count),
      1u,
      1u);
}

// Bluestein chirp-z: embeds the n-point transform in a circular
// convolution of power-of-two length m = next_pow2(2n-1) built from the
// radix-2 pass. X = w * IDFT(DFT(a) * DFT(b)) with a = x*w and b the
// conjugate chirp; the epilogue applies w and the direction scale. The
// convolution buffers are packed, one m-row per transform.
void run_fft_bluestein(
    const array& src,
    array& dst,
    uint32_t n,
    uint32_t stride,
    uint32_t base_flags,
    uint32_t first_extra,
    uint32_t last_extra,
    uint64_t full_elements,
    const std::string& name,
    array& out,
    const Stream& s) {
  uint64_t transforms = full_elements / n;
  uint64_t m = 1;
  while (m < 2 * uint64_t(n) - 1) {
    m <<= 1;
  }
  uint64_t conv_elements = transforms * m;
  // Shape is int-backed, so refuse batches whose padding outgrows it.
  if (conv_elements > 0x7ffffffull) {
    omarchy::unsupported(name + " Bluestein batch", out);
  }
  Shape conv_shape{static_cast<int>(conv_elements)};
  std::vector<FftStage> conv_plan;
  plan_fft_stages(m, 1, conv_plan, name, out);

  array chirped = make_fft_temp(conv_shape, complex64, s);
  array chirp_kernel = make_fft_temp(conv_shape, complex64, s);
  array spectrum = make_fft_temp(conv_shape, complex64, s);
  array conv = make_fft_temp(conv_shape, complex64, s);

  // One full m-point pipeline (pass/twiddle stages on a packed buffer,
  // axis stride 1) from `from` into `to` under `flags`.
  auto run_conv_pipeline = [&](const array& from, array& to, uint32_t flags) {
    const array* cur = &from;
    std::optional<array> ping;
    for (size_t i = 0; i < conv_plan.size(); ++i) {
      const FftStage& stage = conv_plan[i];
      bool last = i + 1 == conv_plan.size();
      array* target;
      if (last) {
        target = &to;
      } else {
        if (!ping) {
          ping = make_fft_temp(conv_shape, complex64, s);
        }
        target = &*ping;
      }
      if (stage.kind == FftStage::Kind::Pass) {
        dispatch_fft_pass(
            *cur,
            *target,
            stage.length,
            stage.stride_mul,
            checked_u32(conv_elements / stage.length, name, out),
            flags,
            name,
            out,
            s);
      } else if (stage.kind == FftStage::Kind::Twiddle) {
        dispatch_fft_stage(
            *cur,
            *target,
            conv_elements,
            kFftStageTwiddle,
            flags,
            stage.length,
            stage.inner,
            stage.stride_mul,
            name,
            out,
            s);
      } else {
        dispatch_fft_stage(
            *cur,
            *target,
            conv_elements,
            kFftStagePermute,
            flags,
            stage.length,
            stage.inner,
            stage.stride_mul,
            name,
            out,
            s);
      }
      cur = target;
    }
  };

  dispatch_fft_stage(
      src,
      chirped,
      conv_elements,
      kFftStageBluesteinA,
      base_flags | first_extra,
      n,
      static_cast<uint32_t>(m),
      stride,
      name,
      out,
      s);
  dispatch_fft_stage(
      chirped,
      chirp_kernel,
      conv_elements,
      kFftStageBluesteinB,
      base_flags,
      n,
      static_cast<uint32_t>(m),
      stride,
      name,
      out,
      s);
  run_conv_pipeline(chirped, spectrum, base_flags);
  run_conv_pipeline(chirp_kernel, chirped, base_flags);
  // Multiply the spectra in place (per-element read-then-write): the
  // product lands in `chirped`.
  dispatch_fft_stage(
      spectrum,
      chirped,
      conv_elements,
      kFftStageMultiply,
      base_flags,
      n,
      static_cast<uint32_t>(m),
      stride,
      name,
      out,
      s);
  // Inverse of the transform direction: flip the direction/scale bit.
  // Forward pipelines need the scaled inverse pass (total 1/m); inverse
  // pipelines need the unscaled forward pass, with the m/n folding left
  // for the epilogue.
  run_conv_pipeline(chirped, conv, base_flags ^ kFftFlagInverse);
  dispatch_fft_stage(
      conv,
      dst,
      transforms * n,
      kFftStageBluesteinY,
      base_flags | last_extra,
      n,
      static_cast<uint32_t>(m),
      stride,
      name,
      out,
      s);
}

// Run one planned axis pipeline from src into dst. first_extra flags the
// pipeline's first dispatch (a real or half-spectrum input side), last_extra
// its last (a packed half-spectrum output side). full_elements is the dense
// geometry the stages address; scratch ping buffers take scratch_shape.
void run_fft_axis(
    const array& src,
    array& dst,
    const std::vector<FftStage>& stages,
    uint32_t n,
    uint32_t stride,
    uint32_t base_flags,
    uint32_t first_extra,
    uint32_t last_extra,
    uint64_t full_elements,
    const Shape& scratch_shape,
    const std::string& name,
    array& out,
    const Stream& s) {
  const array* cur = &src;
  std::optional<array> ping[2];
  for (size_t i = 0; i < stages.size(); ++i) {
    const FftStage& stage = stages[i];
    bool first = i == 0;
    bool last = i + 1 == stages.size();
    array* target;
    if (last) {
      target = &dst;
    } else {
      auto& slot = ping[i % 2];
      if (!slot) {
        slot = make_fft_temp(scratch_shape, complex64, s);
      }
      target = &*slot;
    }
    switch (stage.kind) {
      case FftStage::Kind::Pass: {
        uint32_t flags = base_flags;
        if (first) {
          flags |= first_extra;
        }
        if (last) {
          flags |= last_extra;
        }
        // Transforms tile the buffer exactly once (the pass kernel's base
        // formula interleaves them across the sample spacing), so the
        // count is elements over transform length regardless of stride.
        uint64_t count = full_elements / stage.length;
        dispatch_fft_pass(
            *cur,
            *target,
            stage.length,
            stride * stage.stride_mul,
            checked_u32(count, name, out),
            flags,
            name,
            out,
            s);
        break;
      }
      case FftStage::Kind::Twiddle:
        dispatch_fft_stage(
            *cur,
            *target,
            full_elements,
            kFftStageTwiddle,
            base_flags,
            stage.length,
            stage.inner,
            stride * stage.stride_mul,
            name,
            out,
            s);
        break;
      case FftStage::Kind::Bluestein:
        run_fft_bluestein(
            *cur,
            *target,
            stage.length,
            stride * stage.stride_mul,
            base_flags,
            first ? first_extra : 0u,
            last ? last_extra : 0u,
            full_elements,
            name,
            out,
            s);
        break;
      case FftStage::Kind::Permute: {
        uint32_t flags = base_flags;
        if (last) {
          flags |= last_extra;
        }
        // A packed half-spectrum destination gathers only bins 0..n/2
        // per transform; otherwise every element moves exactly once.
        uint64_t elements = full_elements;
        if (last && (last_extra & kFftFlagOutputHalf) != 0u) {
          elements = full_elements / stage.length * (stage.length / 2u + 1u);
        }
        dispatch_fft_stage(
            *cur,
            *target,
            elements,
            kFftStagePermute,
            flags,
            stage.length,
            stage.inner,
            stride * stage.stride_mul,
            name,
            out,
            s);
        break;
      }
    }
    cur = target;
  }
}

} // namespace

void FFT::eval_gpu(const std::vector<array>& inputs, array& out) {
  const array& in = inputs.at(0);
  const std::string tag = name();
  Stream s = out.primitive().stream();

  // The op layer (mlx/fft.cpp fft_impl) builds exactly three dtype
  // combinations; anything else keeps the named error.
  Dtype in_type = real_ && !inverse_ ? float32 : complex64;
  Dtype out_type = real_ && inverse_ ? float32 : complex64;
  if (in.dtype() != in_type || out.dtype() != out_type) {
    omarchy::unsupported(tag + " dtype combination", out);
  }

  checked_u32(in.size(), tag, out);
  checked_u32(out.size(), tag, out);

  int rank = in.ndim();
  for (size_t axis : axes_) {
    if (static_cast<int>(axis) >= rank) {
      omarchy::unsupported(tag + " axis out of range", out);
    }
  }

  // Transform length per axis. For the real variants the trailing axis of
  // the axes list is special: rfft reads n samples and writes n/2+1 bins,
  // irfft reads n/2+1 bins and writes n samples. The other axes carry the
  // same length on both sides (the op layer guarantees it; the checks
  // keep a caller bug from turning into silent garbage).
  std::vector<uint32_t> lengths;
  lengths.reserve(axes_.size());
  for (size_t i = 0; i < axes_.size(); ++i) {
    size_t axis = axes_[i];
    bool real_axis = real_ && i + 1 == axes_.size();
    if (!real_axis) {
      if (in.shape(axis) != out.shape(axis)) {
        omarchy::unsupported(tag + " shape mismatch", out);
      }
      lengths.push_back(static_cast<uint32_t>(in.shape(axis)));
    } else if (!inverse_) {
      lengths.push_back(static_cast<uint32_t>(in.shape(axis)));
      if (out.shape(axis) != lengths.back() / 2 + 1) {
        omarchy::unsupported(tag + " rfft output shape", out);
      }
    } else {
      lengths.push_back(static_cast<uint32_t>(out.shape(axis)));
      if (in.shape(axis) != lengths.back() / 2 + 1) {
        omarchy::unsupported(tag + " irfft input shape", out);
      }
    }
  }
  // Plan every axis before touching memory: a length class this backend
  // refuses must fail here, by name, before any allocation.
  std::vector<std::vector<FftStage>> plans;
  plans.reserve(lengths.size());
  for (uint32_t length : lengths) {
    plans.push_back(plan_fft_axis(length, tag, out));
  }

  out.set_data(allocate_omarchy(out.nbytes()));
  if (out.size() == 0) {
    return;
  }

  // The kernels address dense row-major buffers. Sliced or transposed
  // inputs of either dtype materialize through the strided copy
  // engine, which now carries complex64 (Complex64Transport).
  const array* src = &in;
  std::optional<array> dense_input;
  if (!(in.flags().row_contiguous && in.data_size() == in.size())) {
    dense_input = make_fft_input_dense(in, tag, out, s);
    src = &*dense_input;
  }

  if (!real_) {
    // Complex-to-complex: one planned pipeline per axis, with the last
    // pipeline landing in the output.
    const array* cur = src;
    std::optional<array> temps[2];
    for (size_t i = 0; i < axes_.size(); ++i) {
      size_t axis = axes_[i];
      auto [stride, count] =
          fft_pass_geometry(in.shape(), axis, lengths[i], tag, out);
      (void)count;
      array* dst;
      if (i + 1 == axes_.size()) {
        dst = &out;
      } else {
        auto& slot = temps[i % 2];
        if (!slot) {
          slot = make_fft_temp(Shape(in.shape()), complex64, s);
        }
        dst = &*slot;
      }
      run_fft_axis(
          *cur,
          *dst,
          plans[i],
          lengths[i],
          stride,
          inverse_ ? kFftFlagInverse : 0u,
          0,
          0,
          in.size(),
          Shape(in.shape()),
          tag,
          out,
          s);
      cur = dst;
    }
  } else if (!inverse_) {
    // Real-to-complex: promote-and-forward on the real axis (kept to
    // bins 0..n/2), then forward pipelines over the remaining axes of
    // the half-spectrum. A single radix-2 pass carries the real-promote
    // and packed-half store in-flight; a multi-stage pipeline promotes
    // up front with an edge kernel, runs plain complex, and truncates
    // with a second edge kernel, because the packed half-spectrum only
    // aligns with the pass layout for a natural-ordered single pass.
    size_t real_axis = axes_.back();
    auto [real_stride, _] = fft_pass_geometry(
        in.shape(), real_axis, lengths.back(), tag, out);
    const std::vector<FftStage>& real_plan = plans.back();
    bool single_pass_real = real_plan.size() == 1 &&
        real_plan.front().kind == FftStage::Kind::Pass;
    array real_owned = make_fft_temp(Shape(out.shape()), complex64, s);
    array& real_dst = axes_.size() == 1 ? out : real_owned;
    if (single_pass_real) {
      run_fft_axis(
          *src,
          real_dst,
          real_plan,
          lengths.back(),
          real_stride,
          0,
          kFftFlagInputReal,
          kFftFlagOutputHalf,
          in.size(),
          Shape(in.shape()),
          tag,
          out,
          s);
    } else {
      array promoted = make_fft_temp(Shape(in.shape()), complex64, s);
      dispatch_fft_stage(
          *src,
          promoted,
          in.size(),
          kFftStageRealEdge,
          kFftFlagInputReal,
          lengths.back(),
          0,
          real_stride,
          tag,
          out,
          s);
      array real_full = make_fft_temp(Shape(in.shape()), complex64, s);
      run_fft_axis(
          promoted,
          real_full,
          real_plan,
          lengths.back(),
          real_stride,
          0,
          0,
          0,
          in.size(),
          Shape(in.shape()),
          tag,
          out,
          s);
      dispatch_fft_stage(
          real_full,
          real_dst,
          out.size(),
          kFftStageRealEdge,
          0,
          lengths.back(),
          0,
          real_stride,
          tag,
          out,
          s);
    }
    const array* cur = &real_dst;
    if (axes_.size() > 1) {
      std::optional<array> temps[2];
      for (size_t i = 0; i + 1 < axes_.size(); ++i) {
        size_t axis = axes_[i];
        auto [stride, count] =
            fft_pass_geometry(out.shape(), axis, lengths[i], tag, out);
        (void)count;
        array* dst;
        bool final_axis = i + 2 == axes_.size();
        if (final_axis) {
          dst = &out;
        } else {
          auto& slot = temps[i % 2];
          if (!slot) {
            slot = make_fft_temp(Shape(out.shape()), complex64, s);
          }
          dst = &*slot;
        }
        run_fft_axis(
            *cur,
            *dst,
            plans[i],
            lengths[i],
            stride,
            0,
            0,
            0,
            out.size(),
            Shape(out.shape()),
            tag,
            out,
            s);
        cur = dst;
      }
    }
  } else {
    // Complex-to-real: inverse pipelines on the non-real axes over the
    // half-spectrum shape, then the real-axis pipeline synthesizes the
    // full Hermitian spectrum (INPUT_HALF) and inverts it; the real
    // part of the result is extracted to the float32 output.
    const array* cur = src;
    std::optional<array> temps[2];
    for (size_t i = 0; i + 1 < axes_.size(); ++i) {
      size_t axis = axes_[i];
      auto [stride, count] =
          fft_pass_geometry(in.shape(), axis, lengths[i], tag, out);
      (void)count;
      auto& slot = temps[i % 2];
      if (!slot) {
        slot = make_fft_temp(Shape(in.shape()), complex64, s);
      }
      run_fft_axis(
          *cur,
          *slot,
          plans[i],
          lengths[i],
          stride,
          kFftFlagInverse,
          0,
          0,
          in.size(),
          Shape(in.shape()),
          tag,
          out,
          s);
      cur = &*slot;
    }
    size_t real_axis = axes_.back();
    auto [real_stride, _] =
        fft_pass_geometry(out.shape(), real_axis, lengths.back(), tag, out);
    array full = make_fft_temp(Shape(out.shape()), complex64, s);
    const std::vector<FftStage>& real_plan = plans.back();
    if (real_plan.size() == 1 && real_plan.front().kind == FftStage::Kind::Pass) {
      // Single radix-2 pass: the packed half-spectrum input synthesizes
      // to the full Hermitian spectrum in-flight at load.
      run_fft_axis(
          *cur,
          full,
          real_plan,
          lengths.back(),
          real_stride,
          kFftFlagInverse,
          kFftFlagInputHalf,
          0,
          out.size(),
          Shape(out.shape()),
          tag,
          out,
          s);
    } else {
      // Multi-stage pipeline: expand the packed half spectrum to the
      // full Hermitian spectrum with an edge kernel, then run plain
      // complex. The edge kernel writes every full slot exactly once.
      array herm = make_fft_temp(Shape(out.shape()), complex64, s);
      dispatch_fft_stage(
          *cur,
          herm,
          out.size(),
          kFftStageRealEdge,
          kFftFlagInputHalf,
          lengths.back(),
          0,
          real_stride,
          tag,
          out,
          s);
      run_fft_axis(
          herm,
          full,
          real_plan,
          lengths.back(),
          real_stride,
          kFftFlagInverse,
          0,
          0,
          out.size(),
          Shape(out.shape()),
          tag,
          out,
          s);
    }
    omarchy::ComputeParams extract_params;
    extract_params.count = checked_u32(out.size(), tag, out);
    extract_params.lhs_offset = checked_item_offset(full, 0, tag, out);
    extract_params.output_offset = checked_item_offset(out, 0, tag, out);
    std::array<omarchy::ComputeBinding, 4> extract_bindings{
        binding(full),
        omarchy::ComputeBinding{},
        binding(out),
        omarchy::ComputeBinding{}};
    auto& encoder = omarchy::get_command_encoder(s);
    encoder.dispatch_compute(
        omarchy::ComputeKernel::FftRealF32,
        extract_bindings,
        extract_params,
        omarchy::compute_dispatch_group_count(extract_params.count));
  }
}
OMARCHY_UNARY(Floor, FloorOperation)
void Gather::eval_gpu(const std::vector<array>& inputs, array& out) {
  const array& table = inputs.at(0);
  auto [axes, slice_sizes] = state();
  auto& encoder = omarchy::get_command_encoder(out.primitive().stream());
  bool raw_word_table = table.dtype() == uint32 || table.dtype() == int32;
  bool raw_half_table = table.dtype() == uint16 || table.dtype() == int16;
  bool raw_i64_table = table.dtype() == int64 || table.dtype() == uint64;
  bool complex_table = table.dtype() == complex64;
  if (raw_word_table || raw_half_table || raw_i64_table || complex_table) {
    if (out.dtype() != table.dtype()) {
      omarchy::unsupported("Take dtype", out);
    }
    if (raw_half_table &&
        (!encoder.device().capabilities().storage_buffer_16bit_access ||
         !encoder.device().capabilities().shader_int16)) {
      omarchy::unsupported("Take 16-bit capability", out);
    }
  } else {
    require_float_dtype("Take", table, out, encoder);
  }
  size_t nidx = inputs.size() - 1;
  if (table.ndim() != slice_sizes.size() || table.ndim() > 4) {
    omarchy::unsupported("Take rank", out);
  }
  if (nidx == 0) {
    copy_gpu(table, out, CopyType::General, out.primitive().stream());
    return;
  }
  if (nidx != axes.size()) {
    omarchy::unsupported("Take index arity", out);
  }
  uint32_t index_mode;
  switch (inputs.at(1).dtype()) {
    case int32:
      index_mode = 0;
      break;
    case uint32:
      index_mode = 1;
      break;
    case int64:
      index_mode = 2;
      break;
    case uint8:
      index_mode = 3;
      break;
    case int8:
      index_mode = 4;
      break;
    case uint16:
      index_mode = 5;
      break;
    case int16:
      index_mode = 6;
      break;
    default:
      omarchy::unsupported("indexed Take dtype", out);
  }
  const array& idx0 = inputs.at(1);
  if (idx0.ndim() > 4) {
    omarchy::unsupported("Take index rank", out);
  }
  for (size_t i = 1; i < nidx; ++i) {
    const array& idx = inputs.at(i + 1);
    if (idx.dtype() != idx0.dtype()) {
      omarchy::unsupported("Take index dtype", out);
    }
    if (idx.shape() != idx0.shape()) {
      omarchy::unsupported("Take index shape", out);
    }
  }
  std::optional<array> table_temp;
  const array& table_d = ensure_dense(
      table, table.flags().row_contiguous, table_temp, encoder, stream());
  const array* indices = &idx0;
  std::optional<array> single_index;
  std::optional<array> packed_indices;
  std::optional<array> axis_metadata;
  if (nidx == 1) {
    if (!idx0.flags().row_contiguous || idx0.data_size() != idx0.size() ||
        idx0.offset() % 4 != 0) {
      single_index = array(idx0.shape(), idx0.dtype(), nullptr, {});
      copy_gpu(idx0, *single_index, CopyType::General, out.primitive().stream());
      encoder.add_temporary(*single_index);
      indices = &*single_index;
    }
  } else {
    size_t segment_bytes = (idx0.nbytes() + 3u) & ~size_t{3};
    size_t segment_items = segment_bytes / idx0.itemsize();
    size_t packed_items = segment_items * nidx;
    packed_indices.emplace(
        Shape{static_cast<int>(packed_items)},
        idx0.dtype(),
        nullptr,
        std::vector<array>{});
    array::Flags flags;
    flags.contiguous = true;
    flags.row_contiguous = true;
    flags.col_contiguous = true;
    packed_indices->set_data(
        allocate_omarchy(packed_indices->nbytes()),
        packed_indices->size(),
        Strides{1},
        flags,
        0);
    encoder.add_temporary(*packed_indices);
    Strides dense_strides(idx0.ndim(), 1);
    for (int axis = idx0.ndim() - 2; axis >= 0; --axis) {
      dense_strides[axis] = dense_strides[axis + 1] * idx0.shape(axis + 1);
    }
    for (size_t i = 0; i < nidx; ++i) {
      const array& idx = inputs.at(i + 1);
      copy_gpu_inplace(
          idx,
          *packed_indices,
          idx.shape(),
          idx.strides(),
          dense_strides,
          0,
          i * segment_items,
          CopyType::GeneralGeneral,
          out.primitive().stream());
    }
    indices = &*packed_indices;
    std::vector<uint32_t> words;
    words.reserve(3 * nidx);
    for (size_t i = 0; i < nidx; ++i) {
      words.push_back(checked_u32(table_d.shape(axes[i]), "Take", out));
    }
    for (size_t i = 0; i < nidx; ++i) {
      words.push_back(checked_u32(table_d.strides()[axes[i]], "Take", out));
    }
    for (size_t i = 0; i < nidx; ++i) {
      words.push_back(checked_u32(i * segment_bytes / 4, "Take", out));
    }
    axis_metadata.emplace(
        Shape{static_cast<int>(words.size())},
        uint32,
        nullptr,
        std::vector<array>{});
    axis_metadata->set_data(
        allocate_omarchy(axis_metadata->nbytes()),
        axis_metadata->size(),
        Strides{1},
        flags,
        0);
    auto* metadata_buffer = static_cast<omarchy::VulkanBuffer*>(
        axis_metadata->buffer().ptr());
    std::memcpy(metadata_buffer->data, words.data(), axis_metadata->nbytes());
    encoder.add_temporary(*axis_metadata);
  }
  size_t slice_total = 1;
  for (int dim : slice_sizes) {
    slice_total *= static_cast<size_t>(dim);
  }
  if (out.size() != idx0.size() * slice_total) {
    omarchy::unsupported("Take index shape", out);
  }
  out.set_data(allocate_omarchy(out.nbytes()));
  if (out.size() == 0) {
    return;
  }
  uint32_t count = checked_u32(out.size(), "Take", out);
  omarchy::ComputeParams params;
  params.count = count;
  params.operation = index_mode;
  params.reduce_size = checked_u32(table_d.shape(axes[0]), "Take", out);
  params.matrix_m = checked_u32(table_d.strides()[axes[0]], "Take", out);
  params.lhs_offset = checked_item_offset(table_d, table_d.size(), "Take", out);
  params.rhs_offset = nidx == 1
      ? checked_u32(
            checked_item_offset(*indices, indices->size(), "Take", out) *
                indices->itemsize() / 4,
            "Take",
            out)
      : 0u;
  params.output_offset = checked_item_offset(out, out.size(), "Take", out);
  params.flags = checked_u32(table_d.ndim(), "Take", out);
  for (int i = 0; i < table_d.ndim(); ++i) {
    params.shape[i] =
        checked_u32(static_cast<size_t>(slice_sizes[i]), "Take", out);
    params.in_strides[i] = checked_u32(table_d.strides()[i], "Take", out);
  }
  params.dims = checked_u32(idx0.ndim(), "Take", out);
  for (int i = 0; i < idx0.ndim(); ++i) {
    params.out_strides[i] = checked_u32(idx0.shape(i), "Take", out);
  }
  params.aux_size = checked_u32(nidx, "Take", out);
  std::array<omarchy::ComputeBinding, 4> bindings{
      binding(table_d), binding(*indices), binding(out), binding(out)};
  if (nidx > 1) {
    bindings[2] = binding(*axis_metadata);
  }
  auto kernel = complex_table
      ? (nidx > 1 ? omarchy::ComputeKernel::TakeMultiComplex64
                  : omarchy::ComputeKernel::TakeComplex64)
      : raw_word_table
      ? (nidx > 1 ? omarchy::ComputeKernel::TakeMultiU32
                  : omarchy::ComputeKernel::TakeU32)
      : raw_half_table
      ? (nidx > 1 ? omarchy::ComputeKernel::TakeMultiU16
                  : omarchy::ComputeKernel::TakeU16)
      : raw_i64_table
      ? (nidx > 1 ? omarchy::ComputeKernel::TakeMultiI64
                  : omarchy::ComputeKernel::TakeI64)
      : select_float_kernel(
            out.dtype(),
            nidx > 1 ? omarchy::ComputeKernel::TakeMultiF32
                     : omarchy::ComputeKernel::TakeF32,
            nidx > 1 ? omarchy::ComputeKernel::TakeMultiF16
                     : omarchy::ComputeKernel::TakeF16,
            nidx > 1 ? omarchy::ComputeKernel::TakeMultiBF16
                     : omarchy::ComputeKernel::TakeBF16);
  uint32_t bound = nidx > 1 ? 4u : 3u;
  encoder.dispatch_compute(
      kernel,
      std::span<const omarchy::ComputeBinding>(bindings.data(), bound),
      params,
      omarchy::compute_dispatch_group_count(count));
}
void GatherAxis::eval_gpu(const std::vector<array>& inputs, array& out) {
  const array& src = inputs.at(0);
  const array& indices = inputs.at(1);
  int axis = axis_;
  auto& encoder = omarchy::get_command_encoder(out.primitive().stream());
  bool raw_word = src.dtype() == uint32 || src.dtype() == int32 ||
      src.dtype() == float32;
  bool raw_i64 = src.dtype() == int64 || src.dtype() == uint64;
  bool complex = src.dtype() == complex64;
  if (raw_word || raw_i64 || complex) {
    if (out.dtype() != src.dtype()) {
      omarchy::unsupported("Take dtype", out);
    }
  } else {
    require_float_dtype("Take", src, out, encoder);
  }
  uint32_t index_mode;
  uint64_t index_words;
  switch (indices.dtype()) {
    case int32:
      index_mode = 0;
      index_words = indices.size();
      break;
    case uint32:
      index_mode = 1;
      index_words = indices.size();
      break;
    case int64:
      index_mode = 2;
      index_words = indices.size() * 2;
      break;
    default:
      omarchy::unsupported("indexed Take dtype", out);
  }
  std::optional<array> indices_temp;
  const array& indices_d = ensure_dense(
      indices,
      indices.flags().row_contiguous && indices.data_size() == indices.size(),
      indices_temp,
      encoder,
      out.primitive().stream());
  int non_axis = out.ndim() - 1;
  if (non_axis > 4) {
    // shape[] and in_strides[] cap at four slots for the non-axis walk.
    omarchy::unsupported("Take layout", out);
  }
  size_t post_size = 1;
  for (int i = axis + 1; i < out.ndim(); ++i) {
    post_size *= out.shape(i);
  }
  out.set_data(allocate_omarchy(out.nbytes()));
  if (out.size() == 0) {
    return;
  }
  uint32_t count = checked_u32(out.size(), "Take", out);
  omarchy::ComputeParams params;
  params.count = count;
  params.operation = index_mode;
  params.lhs_size = checked_u32(src.size(), "Take", out);
  params.rhs_size = checked_u32(index_words, "Take", out);
  params.reduce_size = checked_u32(src.shape(axis), "Take", out);
  params.output_size = checked_u32(post_size, "Take", out);
  params.aux_size = checked_u32(indices.shape(axis), "Take", out);
  params.lhs_offset = checked_item_offset(src, src.size(), "Take", out);
  uint32_t index_offset =
      checked_item_offset(indices_d, indices_d.size(), "Take", out);
  if (index_mode == 2) {
    if (index_offset > std::numeric_limits<uint32_t>::max() / 2) {
      omarchy::unsupported("Take index span", out);
    }
    index_offset *= 2;
  }
  params.rhs_offset = index_offset;
  params.output_offset = checked_item_offset(out, out.size(), "Take", out);
  params.matrix_m = checked_u32(axis, "Take", out);
  params.matrix_n = checked_u32(src.strides(axis), "Take", out);
  params.flags = checked_u32(non_axis, "Take", out);
  for (int i = 0, d = 0; i < out.ndim(); ++i) {
    if (i == axis) {
      continue;
    }
    params.shape[d] = checked_u32(out.shape(i), "Take", out);
    params.in_strides[d] = checked_u32(src.strides(i), "Take", out);
    ++d;
  }
  std::array<omarchy::ComputeBinding, 3> bindings{
      binding(src), binding(indices_d), binding(out)};
  auto kernel = complex ? omarchy::ComputeKernel::GatherAxisComplex64
      : raw_i64 ? omarchy::ComputeKernel::GatherAxisI64
      : src.dtype() == float16 ? omarchy::ComputeKernel::GatherAxisF16
      : src.dtype() == bfloat16 ? omarchy::ComputeKernel::GatherAxisBF16
                                : omarchy::ComputeKernel::GatherAxisU32;
  encoder.dispatch_compute(
      kernel,
      bindings,
      params,
      omarchy::compute_dispatch_group_count(count));
}


void GatherQMM::eval_gpu(const std::vector<array>& inputs, array& out) {
  const std::string tag = name();
  if (mode_ == QuantizationMode::Affine) {
    // Affine inputs: [x, w, scales, biases, lhs, rhs]. Both transpose
    // layouts are first-class.
    dispatch_gather_qmm(
        tag,
        inputs[0],
        inputs[1],
        inputs[2],
        inputs[3],
        inputs[4],
        inputs[5],
        group_size_,
        bits_,
        transpose_,
        /* fp_mode = */ false,
        /* out global scale = */ std::nullopt,
        out,
        out.primitive().stream());
    return;
  }
  // Non-affine inputs: [x, w, scales(uint8), lhs, rhs]; no bias, no
  // global scale reaches this primitive through gather_qmm.
  dispatch_gather_qmm(
      tag,
      inputs[0],
      inputs[1],
      inputs[2],
      std::nullopt,
      inputs[3],
      inputs[4],
      group_size_,
      bits_,
      transpose_,
      /* fp_mode = */ true,
      /* out global scale = */ std::nullopt,
      out,
      out.primitive().stream());
}


namespace {

// Weight side shared by the fp-mode qqmm paths: a float w is packed
// here with the same kernels that back mx.quantize (the nvfp4 global
// scale folds into the stored scale bytes); an already-quantized w
// keeps its packed words and uint8 scale bytes. The returned pair is
// what dispatch_gather_qmm consumes in fp_mode.
std::pair<array, array> fp_qqmm_weight(
    const std::string& tag,
    const std::vector<array>& inputs,
    bool w_quantized,
    size_t scales_index,
    const std::optional<array>& global_scale_w,
    int bits,
    int group_size,
    array& out,
    const Stream& s) {
  if (w_quantized) {
    return {inputs[1], inputs[scales_index]};
  }
  const array& w = inputs[1];
  auto& encoder = omarchy::get_command_encoder(s);
  Shape packed_shape = w.shape();
  packed_shape.back() = packed_shape.back() * bits / 32;
  Shape parameter_shape = w.shape();
  parameter_shape.back() /= group_size;
  array packed(std::move(packed_shape), uint32, nullptr, {});
  array scales(std::move(parameter_shape), uint8, nullptr, {});
  dispatch_fp_quantize(
      tag, w, packed, scales, global_scale_w, bits, group_size, out, s);
  encoder.add_temporary(packed);
  encoder.add_temporary(scales);
  return {packed, scales};
}

// Identity-gather index pair (S = 1, both indices zero) that reuses
// the gathered kernel for the un-gathered product.
array zero_gather_index(omarchy::CommandEncoder& encoder) {
  array zero_index(Shape{1}, uint32, nullptr, {});
  array::Flags zero_flags;
  zero_flags.contiguous = true;
  zero_flags.row_contiguous = true;
  zero_flags.col_contiguous = true;
  zero_index.set_data(
      allocate_omarchy(zero_index.nbytes()),
      zero_index.size(),
      Strides{1},
      zero_flags,
      0);
  encoder.add_temporary(zero_index);
  encoder.fill_buffer(binding(zero_index).buffer, 0, 4, 0);
  return zero_index;
}

// GPU core count for the parts this backend has measured. Vulkan
// reports no core count, Device::capabilities() carries none and
// mx.device_info() exposes none, so the width has to come from the
// device name the driver reports. Only names this project has run a
// per-shape occupancy census on are listed; anything else returns 0,
// which keeps every dispatch on the shape it has today. Match is
// exact, because the name does not distinguish a 24-core M1 Max from a
// 32-core one and guessing high would step tiles down on a part that
// never showed it needs them.
uint32_t apple_gpu_cores(const std::string& device_name) {
  if (device_name == "Apple M1 (G13G B1)") {
    return 8u;
  }
  if (device_name == "Apple M1 Max (G13C C0)") {
    return 32u;
  }
  return 0u;
}

// Occupancy floor for the coopmat prefill tile, in workgroups per GPU
// core: below it the 16-row twin is dispatched instead of the shipped
// 32-row one. Both sides of this number are measured on the M1 Max at
// m=1053, k=896, 32-rows against 16-rows per shape
// (receipts/2026-09-14-qmm-occupancy-tilem/):
//
//   n    workgroups  wg/core   32 rows   16 rows
//   128         132      4.1   213.6us   165.7us   -22.4%
//   256         264      8.2   225.7us   300.7us   +33.3%
//   384         396     12.4   249.7us   434.8us   +74.1%
//   896         924     28.9   509.2us   759.5us   +49.2%
//
// So the trade turns over between 4.1 and 8.2, and 6 is the midpoint
// that separates the two measured grids (6 x 32 = 192, strictly between
// 132 and 264). It is deliberately not larger: halving the rows halves
// how many outputs share one staged 16x32 weight tile, and past the
// crossover that reuse loss dominates by tens of percent. On the base
// M1 the same floor is 48 workgroups, which every real prefill grid
// clears, so the base M1 keeps the shipped tile untouched.
// MLX_OMARCHY_QMM_COOPMAT_WG_PER_CORE overrides it for A/B; 0 disables
// the step-down and restores the shipped pick exactly.
constexpr uint32_t kCoopmatWorkgroupsPerCore = 6u;

uint32_t coopmat_workgroups_per_core() {
  static const uint32_t value = [] {
    const char* env = std::getenv("MLX_OMARCHY_QMM_COOPMAT_WG_PER_CORE");
    if (env == nullptr) {
      return kCoopmatWorkgroupsPerCore;
    }
    char* end = nullptr;
    unsigned long parsed = std::strtoul(env, &end, 10);
    // A malformed value keeps the compiled-in floor; a huge one is
    // clamped rather than quietly reinterpreted as the default, because
    // "as small a tile as possible" is a thing an A/B run asks for.
    if (end == env || *end != '\0') {
      return kCoopmatWorkgroupsPerCore;
    }
    return static_cast<uint32_t>(
        std::min<unsigned long>(parsed, 0xfffffffful));
  }();
  return value;
}

// Output rows per coopmat workgroup: the shipped 32, or the 16-row twin
// when the shape-derived grid cannot fill the part. Halving the rows
// doubles the grid and halves weight-tile reuse; 16 is the floor
// because an 8-row, 32-lane variant was built and measured and lost
// even at 4.1 workgroups per core (58% slower than 32 rows), so there
// is nothing below 16 worth dispatching. Both row counts keep one
// output's k chain identical, so the pick does not move generated ids.
//
// A prompt of at most 16 rows fills one 16-row tile: the 32-row tile
// would pad it to 32 (a 13-token prompt runs 19 dead rows through
// every mma and A load) while the grid stays one m-group either way,
// so nothing is traded for the halved per-step work. Contract prompts
// are 11-18 tokens; on T8103 the 13-token prefill GEMMs at 32 rows
// cost 400-950 us each, 133 of them per first token.
uint32_t coopmat_tile_rows(
    uint32_t matrix_m,
    uint32_t n_groups,
    const std::string& device_name) {
  if (matrix_m <= 16u) {
    return 16u;
  }
  uint32_t cores = apple_gpu_cores(device_name);
  uint32_t per_core = coopmat_workgroups_per_core();
  if (cores == 0u || per_core == 0u) {
    return 32u;
  }
  uint64_t target = static_cast<uint64_t>(cores) * per_core;
  uint32_t m_groups_32 = (matrix_m + 31u) / 32u;
  return static_cast<uint64_t>(m_groups_32) * n_groups < target ? 16u : 32u;
}

} // namespace

void GatherQQMM::eval_gpu(const std::vector<array>& inputs, array& out) {
  const std::string tag = name();
  if (mode_ == QuantizationMode::Affine) {
    // Affine inputs: [x, w(u32), lhs, rhs, scales_w]. A float w keeps
    // its named rejection; the affine contract never binds a global
    // scale.
    if (inputs.size() > 5) {
      omarchy::unsupported(tag + " global scale", out);
    }
    if (inputs[1].dtype() != uint32) {
      omarchy::unsupported(tag + " weight dtype", out);
    }
    dispatch_gather_qmm(
        tag,
        inputs[0],
        inputs[1],
        inputs[4],
        std::nullopt,
        inputs[2],
        inputs[3],
        group_size_,
        bits_,
        true,
        /* fp_mode = */ false,
        /* out global scale = */ std::nullopt,
        out,
        out.primitive().stream());
    return;
  }
  // Non-affine inputs: [x, w(float)] or [x, w(u32), lhs, rhs,
  // scales_w(u8)], plus the nvfp4 {global_scale_x, global_scale_w}
  // pair. qqmm quantizes the x operand as well as w, so the activation
  // is fake-quantized with the mx.quantize / mx.dequantize kernels and
  // the gathered product contracts quantized-x codes against
  // quantized-w codes; raw-float x against dequantized w would return
  // silently wrong values.
  const Stream& s = out.primitive().stream();
  if (bits_ != 4 && bits_ != 8) {
    omarchy::unsupported(tag + " bits", out);
  }
  if (group_size_ != 16 && group_size_ != 32) {
    omarchy::unsupported(tag + " group size", out);
  }
  // The scale-byte encoding is mode-fixed, so a noncanonical
  // mode/group/bits combo silently misreads the stream: nvfp4 is
  // (16, 4), mxfp4 is (32, 4), mxfp8 is (32, 8).
  bool canonical_combo =
      (mode_ == QuantizationMode::Nvfp4 && group_size_ == 16 &&
       bits_ == 4) ||
      (mode_ == QuantizationMode::Mxfp4 && group_size_ == 32 && bits_ == 4) ||
      (mode_ == QuantizationMode::Mxfp8 && group_size_ == 32 && bits_ == 8);
  if (!canonical_combo) {
    omarchy::unsupported(tag + " mode", out);
  }
  bool w_quantized = inputs[1].dtype() == uint32;
  size_t base_size = w_quantized ? 5 : 4;
  bool has_global_scales =
      mode_ == QuantizationMode::Nvfp4 && inputs.size() == base_size + 2;
  if (inputs.size() != base_size && !has_global_scales) {
    omarchy::unsupported(tag + " operand count", out);
  }
  auto& encoder = omarchy::get_command_encoder(s);
  require_float_dtype(tag, inputs[0], out, encoder);
  std::optional<array> global_scale_x;
  std::optional<array> global_scale_w;
  if (has_global_scales) {
    global_scale_x = inputs[base_size];
    global_scale_w = inputs[base_size + 1];
  }
  auto [w_q, w_scales] = fp_qqmm_weight(
      tag, inputs, w_quantized, 4, global_scale_w, bits_, group_size_, out, s);
  array x_hat = fake_quantize_fp_activation(
      tag, inputs[0], global_scale_x, bits_, group_size_, out, s);
  dispatch_gather_qmm(
      tag,
      x_hat,
      w_q,
      w_scales,
      std::nullopt,
      inputs[2],
      inputs[3],
      group_size_,
      bits_,
      true,
      /* fp_mode = */ true,
      global_scale_w,
      out,
      s);
}

void QQMatmul::eval_gpu(const std::vector<array>& inputs, array& out) {
  const std::string tag = name();
  const Stream& s = out.primitive().stream();
  if (mode_ != QuantizationMode::Affine) {
    // Non-affine inputs: [x, w(float)] or [x, w(u32), scales_w(u8)],
    // plus the nvfp4 {global_scale_x, global_scale_w} pair. The
    // activation is fake-quantized (dequant(quant(x)) through the
    // mx.quantize kernels) and the product with identity gather (S = 1,
    // both indices zero) contracts those codes against the quantized-w
    // codes; raw-float x against dequantized w would return silently
    // wrong values.
    if (bits_ != 4 && bits_ != 8) {
      omarchy::unsupported(tag + " bits", out);
    }
    if (group_size_ != 16 && group_size_ != 32) {
      omarchy::unsupported(tag + " group size", out);
    }
    // The scale-byte encoding is mode-fixed, so a noncanonical
    // mode/group/bits combo silently misreads the stream: nvfp4 is
    // (16, 4), mxfp4 is (32, 4), mxfp8 is (32, 8).
    bool canonical_combo =
        (mode_ == QuantizationMode::Nvfp4 && group_size_ == 16 &&
         bits_ == 4) ||
        (mode_ == QuantizationMode::Mxfp4 && group_size_ == 32 &&
         bits_ == 4) ||
        (mode_ == QuantizationMode::Mxfp8 && group_size_ == 32 &&
         bits_ == 8);
    if (!canonical_combo) {
      omarchy::unsupported(tag + " mode", out);
    }
    bool w_quantized = inputs[1].dtype() == uint32;
    size_t base_size = w_quantized ? 3 : 2;
    bool has_global_scales =
        mode_ == QuantizationMode::Nvfp4 && inputs.size() == base_size + 2;
    if (inputs.size() != base_size && !has_global_scales) {
      omarchy::unsupported(tag + " operand count", out);
    }
    auto& encoder = omarchy::get_command_encoder(s);
    require_float_dtype(tag, inputs[0], out, encoder);
    if (inputs[0].ndim() != 2) {
      omarchy::unsupported(tag + " rank", out);
    }
    std::optional<array> global_scale_x;
    std::optional<array> global_scale_w;
    if (has_global_scales) {
      global_scale_x = inputs[base_size];
      global_scale_w = inputs[base_size + 1];
    }
    auto [w_q, w_scales] = fp_qqmm_weight(
        tag, inputs, w_quantized, 2, global_scale_w, bits_, group_size_, out, s);
    array x_hat = fake_quantize_fp_activation(
        tag, inputs[0], global_scale_x, bits_, group_size_, out, s);
    array zero_index = zero_gather_index(encoder);
    dispatch_gather_qmm(
        tag,
        x_hat,
        w_q,
        w_scales,
        std::nullopt,
        zero_index,
        zero_index,
        group_size_,
        bits_,
        true,
        /* fp_mode = */ true,
        global_scale_w,
        out,
        s);
    return;
  }
  if (inputs.size() == 2) {
    // Unquantized w needs x @ w.T; this primitive receives w as (N, K)
    // and the tiled matmul path cannot build the transposed view inside
    // an eval, so the affine float-weight form keeps a named rejection.
    // The quantized-weight form below is the contract deliverable.
    omarchy::unsupported(tag + " weight dtype", out);
    return;
  }
  // [x, w(u32), scales_w]: the scale-only dequantized product with
  // identity gather (S = 1, both indices zero).
  auto& encoder = omarchy::get_command_encoder(s);
  if (inputs[0].ndim() != 2) {
    omarchy::unsupported(tag + " rank", out);
  }
  array zero_index = zero_gather_index(encoder);
  dispatch_gather_qmm(
      tag,
      inputs[0],
      inputs[1],
      inputs[2],
      std::nullopt,
      zero_index,
      zero_index,
      group_size_,
      bits_,
      true,
      /* fp_mode = */ false,
      /* out global scale = */ std::nullopt,
      out,
      s);
}



void Greater::eval_gpu(const std::vector<array>& inputs, array& out) {
  dispatch_comparison(name(), CompareGreater, inputs, out);
}
// GreaterEqual serves the composed causal mask (two int32 index arrays
// with stride-0 axes produce a bool mask) and the rest of the
// comparison dtype table: the dispatch decides which dtypes run and
// which refuse by name. The comparison kernel packs the bytes into
// 32-bit words, so the dispatch covers words.
void GreaterEqual::eval_gpu(
    const std::vector<array>& inputs,
    array& out) {
  dispatch_comparison(name(), CompareGreaterEqual, inputs, out);
}
// Hadamard runs the fast Walsh-Hadamard transform over the last axis.
// Separate stage dispatches expose every butterfly pair to the GPU while the
// encoder barriers preserve the in-place dependency between stages.
void Hadamard::eval_gpu(const std::vector<array>& inputs, array& out) {
  const array& input = inputs.at(0);
  auto& encoder = omarchy::get_command_encoder(out.primitive().stream());
  require_float_dtype("Hadamard", input, out, encoder);
  std::optional<array> dense_temp;
  const array& src = ensure_dense(
      input,
      input.flags().row_contiguous,
      dense_temp,
      encoder,
      out.primitive().stream());
  if (out.ndim() == 0) {
    omarchy::unsupported("Hadamard rank", out);
  }
  int n = out.shape(-1);
  int m = 1;
  if ((n & (n - 1)) != 0) {
    for (int factor : {12, 20, 28}) {
      if (n % factor == 0) {
        m = factor;
        n /= factor;
        break;
      }
    }
    if (m == 1) {
      omarchy::unsupported("Hadamard size", out);
    }
  }
  if (n > (1 << 16)) {
    omarchy::unsupported("Hadamard size", out);
  }
  size_t row_length = static_cast<size_t>(n) * static_cast<size_t>(m);
  size_t rows = row_length == 0 ? 0 : out.size() / row_length;
  out.set_data(allocate_omarchy(out.nbytes()));
  if (out.size() == 0) {
    return;
  }
  encoder.add_temporary(src);
  encoder.add_temporary(out);
  encoder.copy_buffer(
      binding(src).buffer,
      binding(out).buffer,
      static_cast<VkDeviceSize>(src.nbytes()),
      static_cast<VkDeviceSize>(src.offset()),
      static_cast<VkDeviceSize>(out.offset()));

  uint32_t element_count = checked_u32(out.size(), "Hadamard", out);
  omarchy::ComputeParams params;
  params.reduce_size = static_cast<uint32_t>(n);
  params.output_size = checked_u32(rows, "Hadamard", out);
  params.matrix_m = static_cast<uint32_t>(m);
  params.alpha = scale_;
  std::array<omarchy::ComputeBinding, 1> bindings{binding(out)};
  auto kernel = select_float_kernel(
      out.dtype(),
      omarchy::ComputeKernel::HadamardF32,
      omarchy::ComputeKernel::HadamardF16,
      omarchy::ComputeKernel::HadamardBF16);
  auto dispatch = [&](size_t count, uint32_t operation, uint32_t stage) {
    params.count = checked_u32(count, "Hadamard", out);
    params.operation = operation;
    params.aux_size = stage;
    encoder.dispatch_compute(
        kernel,
        bindings,
        params,
        omarchy::compute_dispatch_group_count(params.count));
  };

  if (n > 1) {
    uint32_t pair_count = element_count / 2;
    for (uint32_t h = 1; h < static_cast<uint32_t>(n); h <<= 1) {
      dispatch(pair_count, 0, h);
    }
  }
  if (m > 1) {
    dispatch(rows * static_cast<size_t>(n), 1, 0);
  }
}
void Imag::eval_gpu(const std::vector<array>& inputs, array& out) {
  // Upstream imag() returns zeros_like for real input before a
  // primitive is built (mlx/ops.cpp), so eval only ever sees
  // complex64 here; the op extracts the imaginary component.
  dispatch_complex_extract(name(), 1, inputs, out, out.primitive().stream());
}
void Inverse::eval_gpu(const std::vector<array>& inputs, array& out) {
  const array& in = inputs.at(0);
  auto& encoder = omarchy::get_command_encoder(out.primitive().stream());
  linalg_require_f32(name(), in, out);
  if (in.shape(-1) != in.shape(-2)) {
    omarchy::unsupported(std::string("non-square ") + name(), out);
  }
  const uint32_t n = checked_u32(in.shape(-1), name(), out);
  const uint32_t batch =
      checked_u32(linalg_batch_count(in.shape()), name(), out);
  out.set_data(allocate_omarchy(out.nbytes()));
  if (out.size() == 0) {
    return;
  }
  // The kernel consumes a dense copy of A in a scratch and builds the
  // inverse in `out` starting from identity.
  array work(in.shape(), in.dtype(), nullptr, {});
  work.set_data(allocate_omarchy(work.nbytes()));
  encoder.add_temporary(work);
  linalg_copy_dense(in, work, out.primitive().stream());
  auto scratch = make_u32_scratch(std::max<size_t>(batch, 1u), encoder);
  dispatch_clear_u32(scratch, 0, encoder);
  auto [tri, upper] = state();
  omarchy::ComputeParams params;
  params.operation = tri ? 1u : 0u;
  params.matrix_n = n;
  params.output_size = batch;
  params.flags = upper ? 1u : 0u;
  std::array<omarchy::ComputeBinding, 3> bindings{
      binding(work), binding(out), binding(scratch)};
  encoder.dispatch_compute(
      omarchy::ComputeKernel::LinalgInverseF32, bindings, params, batch);
  linalg_check_status(
      scratch,
      encoder,
      tri
          ? "[Inverse::eval_gpu] triangular inverse requires a nonzero"
            " diagonal."
          : "[Inverse::eval_gpu] matrix is singular to working precision.",
      out);
}
void Less::eval_gpu(const std::vector<array>& inputs, array& out) {
  dispatch_comparison(name(), CompareLess, inputs, out);
}
void LessEqual::eval_gpu(const std::vector<array>& inputs, array& out) {
  dispatch_comparison(name(), CompareLessEqual, inputs, out);
}
void Load::eval_gpu(const std::vector<array>& inputs, array& out) {
  out.set_data(allocate_omarchy(out.nbytes()));
  // The allocator is host-visible (coherent where the memory type allows),
  // so the reader fills the output buffer directly and the data is ready
  // when this function returns. This mirrors the CUDA Load::eval_gpu, which
  // also reads synchronously on the calling thread.
  auto* out_ptr = out.data<char>();
  reader_->read(out_ptr, out.nbytes(), offset_);
  if (swap_endianness_) {
    switch (out.itemsize()) {
      case 2:
        swap_endianness<2>(reinterpret_cast<uint8_t*>(out_ptr), out.size());
        break;
      case 4:
        swap_endianness<4>(reinterpret_cast<uint8_t*>(out_ptr), out.size());
        break;
      case 8:
        swap_endianness<8>(reinterpret_cast<uint8_t*>(out_ptr), out.size());
        break;
    }
  }
}

void Log::eval_gpu(const std::vector<array>& inputs, array& out) {
  uint32_t real_operation;
  uint32_t complex_operation;
  switch (state()) {
    case Log::Base::two:
      real_operation = Log2Operation;
      complex_operation = ComplexLog2;
      break;
    case Log::Base::ten:
      real_operation = Log10Operation;
      complex_operation = ComplexLog10;
      break;
    default:
      real_operation = LogOperation;
      complex_operation = ComplexLog;
      break;
  }
  if (out.dtype() == complex64) {
    dispatch_complex(name(), complex_operation, inputs, out, stream());
    return;
  }
  dispatch_elementwise(name(), real_operation, inputs, out, stream());
}
void Log1p::eval_gpu(const std::vector<array>& inputs, array& out) {
  if (out.dtype() == complex64) {
    dispatch_complex(
        name(), ComplexLog1p, inputs, out, out.primitive().stream());
    return;
  }
  dispatch_elementwise(
      name(), Log1pOperation, inputs, out, out.primitive().stream());
}
void LogicalAnd::eval_gpu(const std::vector<array>& inputs, array& out) {
  dispatch_logical(name(), LogicalAndOperation, inputs, out);
}
void LogicalNot::eval_gpu(const std::vector<array>& inputs, array& out) {
  dispatch_logical(name(), LogicalNotOperation, inputs, out);
}
void LogicalOr::eval_gpu(const std::vector<array>& inputs, array& out) {
  dispatch_logical(name(), LogicalOrOperation, inputs, out);
}
void LogAddExp::eval_gpu(const std::vector<array>& inputs, array& out) {
  if (out.dtype() == complex64) {
    // Upstream evaluates complex logaddexp through the same binary
    // primitive; the kernel mirrors the CPU formula including the
    // -inf selects.
    dispatch_complex(
        name(), ComplexLogAddExp, inputs, out, out.primitive().stream());
    return;
  }
  dispatch_elementwise(
      name(), LogAddExpOperation, inputs, out, out.primitive().stream());
}
void LogSumExp::eval_gpu(const std::vector<array>& inputs, array& out) {
  const array& input = inputs.at(0);
  auto& encoder = omarchy::get_command_encoder(out.primitive().stream());
  require_float_dtype("LogSumExp", input, out, encoder);
  std::optional<array> dense_temp;
  const array& src = ensure_dense(
      input,
      input.flags().row_contiguous,
      dense_temp,
      encoder,
      out.primitive().stream());


  // Upstream mlx/ops.cpp logsumexp builds the LogSumExp primitive only
  // for a suffix reduce and keeps the reduced axis at size 1 (the
  // keepdims=False form squeezes on top of this output), so no
  // suffix-axis check is needed here. The shader accumulates in float32
  // for every dtype and keeps an infinite row max, matching the
  // upstream CPU rule.
  size_t row_length = src.shape(-1);
  size_t rows = src.size() / row_length;
  out.set_data(allocate_omarchy(out.nbytes()));
  if (out.size() == 0) {
    return;
  }
  uint32_t output_size = checked_u32(rows, "LogSumExp", out);
  omarchy::ComputeParams params;
  params.count = checked_u32(out.size(), "LogSumExp", out);
  params.reduce_size = checked_u32(row_length, "LogSumExp", out);
  params.output_size = output_size;
  params.lhs_offset = checked_item_offset(
      src, src.size(), "LogSumExp", out);
  params.output_offset = checked_item_offset(
      out, out.size(), "LogSumExp", out);
  std::array<omarchy::ComputeBinding, 3> bindings{
      binding(src), binding(src), binding(out)};
  auto kernel = select_float_kernel(
      out.dtype(),
      omarchy::ComputeKernel::LogSumExpF32,
      omarchy::ComputeKernel::LogSumExpF16,
      omarchy::ComputeKernel::LogSumExpBF16);
  encoder.dispatch_compute(
      kernel,
      bindings,
      params,
      std::min(output_size, omarchy::kMaxComputeGroupCountX));
}
void LUF::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  const array& in = inputs.at(0);
  auto& lu = outputs.at(0);
  auto& pivots = outputs.at(1);
  auto& row_indices = outputs.at(2);
  auto& encoder = omarchy::get_command_encoder(lu.primitive().stream());
  linalg_require_f32(name(), in, lu);
  const int m = in.shape(-2);
  const int n = in.shape(-1);
  const int k_count = std::min(m, n);
  const uint32_t batch =
      checked_u32(linalg_batch_count(in.shape()), name(), lu);
  lu.set_data(allocate_omarchy(lu.nbytes()));
  pivots.set_data(allocate_omarchy(pivots.nbytes()));
  row_indices.set_data(allocate_omarchy(row_indices.nbytes()));
  if (m == 0 || n == 0 || batch == 0) {
    // Nothing to factorize; upstream leaves the empty outputs.
    return;
  }
  // Factorize in place: the packed-LU output starts as a dense copy of
  // A, exactly like the CPU path copies A into lu before getrf.
  linalg_copy_dense(in, lu, lu.primitive().stream());
  auto scratch = make_u32_scratch(std::max<size_t>(batch, 1u), encoder);
  dispatch_clear_u32(scratch, 0, encoder);
  omarchy::ComputeParams params;
  params.matrix_m = checked_u32(m, name(), lu);
  params.matrix_n = checked_u32(n, name(), lu);
  params.matrix_k = checked_u32(k_count, name(), lu);
  params.output_size = batch;
  // Kernel binding order: 0 packed LU, 1 pivots, 2 row_indices, 3 status.
  std::array<omarchy::ComputeBinding, 4> bindings{
      binding(lu),
      binding(pivots),
      binding(row_indices),
      binding(scratch)};
  encoder.dispatch_compute(
      omarchy::ComputeKernel::LinalgLuF32, bindings, params, batch);
  linalg_check_status(
      scratch,
      encoder,
      "[LUF::eval_gpu] LU factorization encountered a zero pivot"
      " (singular matrix); refusing rather than returning a partial"
      " factorization.",
      lu);
}
void Matmul::eval_gpu(const std::vector<array>& inputs, array& out) {
  dispatch_matmul(
      name(), inputs, out, 1.0f, 0.0f, false, out.primitive().stream());
}
void Maximum::eval_gpu(const std::vector<array>& inputs, array& out) {
  if (out.dtype() == complex64) {
    // Lexicographic (real, then imaginary) maximum, exactly the
    // upstream complex64_t ordering that compare_complex encodes for
    // Equal/Greater; ties favor the left operand so max(a, b) == a.
    dispatch_complex(
        name(), ComplexMaximum, inputs, out, out.primitive().stream());
    return;
  }
  if (out.dtype() == bool_) {
    // Bool maximum is the logical or over {0,1}; min/max on the byte
    // lanes stays within 0/1.
    dispatch_int_elementwise(name(), IntMaximumOperation, inputs, out);
    return;
  }
  if (is_int_elementwise_dtype(out.dtype())) {
    dispatch_int_elementwise(name(), IntMaximumOperation, inputs, out);
    return;
  }
  dispatch_elementwise(
      name(), MaximumOperation, inputs, out, out.primitive().stream());
}
void MaskedScatter::eval_gpu(const std::vector<array>& inputs, array& out) {
  const array& dst = inputs.at(0);
  const array& mask = inputs.at(1);
  const array& src = inputs.at(2);
  auto& encoder = omarchy::get_command_encoder(out.primitive().stream());
  if (out.dtype() == float16 || out.dtype() == bfloat16) {
    require_float_dtype("MaskedScatter", dst, out, encoder);
  } else if (out.dtype() != float32 && out.dtype() != int32 &&
             out.dtype() != uint32) {
    omarchy::unsupported("MaskedScatter dtype", out);
  }
  auto kernel = out.dtype() == float16
      ? omarchy::ComputeKernel::MaskedScatterF16
      : out.dtype() == bfloat16 ? omarchy::ComputeKernel::MaskedScatterBF16
                                : omarchy::ComputeKernel::MaskedScatterU32;
  CopyType copy_type = dst.data_size() == 1 ? CopyType::Scalar
      : (dst.flags().row_contiguous && dst.data_size() == dst.size())
      ? CopyType::Vector
      : CopyType::General;
  copy_gpu(dst, out, copy_type, out.primitive().stream());
  if (mask.size() == 0) {
    return;
  }
  size_t rows = mask.shape(0);
  size_t row_length = mask.size() / rows;
  size_t src_length = src.size() / rows;
  std::optional<array> mask_temp;
  const array& mask_d = ensure_dense(
      mask,
      mask.flags().row_contiguous && mask.data_size() == mask.size(),
      mask_temp,
      encoder,
      out.primitive().stream());
  if (rows > omarchy::kMaxComputeGroupCountX) {
    omarchy::unsupported("MaskedScatter row count", out);
  }
  uint32_t mask_word_offset =
      checked_item_offset(mask_d, mask_d.size(), "MaskedScatter", out);
  if (mask_word_offset % 4 != 0) {
    omarchy::unsupported("MaskedScatter mask alignment", out);
  }
  const array* src_dense = &src;
  std::optional<array> materialized;
  if (src.data_size() != src.size() || !src.flags().row_contiguous) {
    materialized = array(src.shape(), src.dtype(), nullptr, {});
    copy_gpu(src, *materialized, CopyType::General, out.primitive().stream());
    encoder.add_temporary(*materialized);
    src_dense = &*materialized;
  }
  uint32_t count = checked_u32(mask.size(), "MaskedScatter", out);
  omarchy::ComputeParams params;
  params.count = count;
  params.reduce_size = checked_u32(row_length, "MaskedScatter", out);
  params.aux_size = checked_u32(src_length, "MaskedScatter", out);
  params.lhs_offset = checked_item_offset(
      *src_dense, src_dense->size(), "MaskedScatter", out);
  params.rhs_offset = mask_word_offset / 4;
  params.output_offset =
      checked_item_offset(out, out.size(), "MaskedScatter", out);
  std::array<omarchy::ComputeBinding, 3> bindings{
      binding(out), binding(mask_d), binding(*src_dense)};
  encoder.dispatch_compute(
      kernel, bindings, params, checked_u32(rows, "MaskedScatter", out));
}
void Minimum::eval_gpu(const std::vector<array>& inputs, array& out) {
  if (out.dtype() == bool_) {
    // Bool minimum is the logical and over {0,1}.
    dispatch_int_elementwise(name(), IntMinimumOperation, inputs, out);
    return;
  }
  if (is_int_elementwise_dtype(out.dtype())) {
    dispatch_int_elementwise(name(), IntMinimumOperation, inputs, out);
    return;
  }
  dispatch_elementwise(
      name(), MinimumOperation, inputs, out, out.primitive().stream());
}
void Multiply::eval_gpu(const std::vector<array>& inputs, array& out) {
  if (out.dtype() == complex64) {
    dispatch_complex(
        name(), ComplexMultiply, inputs, out, out.primitive().stream());
    return;
  }
  if (is_int_elementwise_dtype(out.dtype())) {
    dispatch_int_elementwise(name(), IntMultiplyOperation, inputs, out);
    return;
  }
  dispatch_elementwise(
      name(), MultiplyOperation, inputs, out, out.primitive().stream());
}
void Negative::eval_gpu(const std::vector<array>& inputs, array& out) {
  if (out.dtype() == complex64) {
    dispatch_complex(
        name(), ComplexNegative, inputs, out, out.primitive().stream());
    return;
  }
  if (is_int_elementwise_dtype(out.dtype())) {
    // Wraparound negation on the widened variants, op 18 in the
    // shader; INT_MIN stays itself the way upstream's C++ unary minus
    // does on this platform.
    dispatch_int_elementwise(name(), IntNegateOperation, inputs, out);
    return;
  }
  dispatch_elementwise(
      name(), NegativeOperation, inputs, out, out.primitive().stream());
}
void NotEqual::eval_gpu(const std::vector<array>& inputs, array& out) {
  dispatch_comparison(name(), CompareNotEqual, inputs, out);
}
void Partition::eval_gpu(const std::vector<array>& inputs, array& out) {
  const array& input = inputs.at(0);
  auto& encoder = omarchy::get_command_encoder(out.primitive().stream());
  require_sort_dtype("Partition", input, out, false, encoder);
  dispatch_sort_any_axis(
      "Partition", input, out, state().second, false, encoder);
}
// Power promotes base and exponent to one dtype upstream, so a float
// dtype runs the float pow and an integer dtype runs the upstream
// exponentiation-by-squaring (negative signed exponent yields 0).
// Float bases below zero need the sign handling GLSL's pow refuses:
// non-integer exponents produce NaN, odd integer exponents negate.
// complex64 power routes through the principal branch
// exp(y * clog(x)) matching std::pow(complex<float>), with the
// zero-base special cases the comparisons cluster deliberately
// skipped; the test power expected values are std::pow(complex<float>)
// on the host, and the tolerance is 1e-7 absolute for the small-
// magnitude test case so the formula mirrors glibc cpowf exactly
// (log(length) + atan2 for the principal log, exp * cos/sin for the
// exp).
void Power::eval_gpu(const std::vector<array>& inputs, array& out) {
  if (out.dtype() == complex64) {
    dispatch_complex(
        name(), ComplexPower, inputs, out, out.primitive().stream());
    return;
  }
  if (out.dtype() == bool_) {
    // Upstream power promotes bool to bool (promote_types of two bools)
    // and evaluates pow over {0, 1}: x^y is x || !y. The bool kernel's
    // selector code 6 carries it.
    auto& encoder = omarchy::get_command_encoder(out.primitive().stream());
    const array& lhs = inputs.at(0);
    const array& rhs = inputs.at(1);
    if (lhs.dtype() != bool_ || rhs.dtype() != bool_) {
      omarchy::unsupported(std::string(name()) + " dtype", out);
    }
    dispatch_compare_bool_to(name(), 6u, lhs, rhs, out, encoder);
    return;
  }
  if (is_int_elementwise_dtype(out.dtype())) {
    dispatch_int_elementwise(name(), IntPowerOperation, inputs, out);
    return;
  }
  dispatch_elementwise(
      name(), PowerFloatOperation, inputs, out, out.primitive().stream());
}
void QRF::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  const array& in = inputs.at(0);
  auto& q = outputs.at(0);
  // Batched inputs are gated: single-matrix factors verify against
  // Q^T Q = I and QR = A, but the batch>1 path returns garbage whose
  // root cause is not yet identified. Refuse by name instead.
  if (linalg_batch_count(in.shape()) != 1) {
    omarchy::unsupported(
        "[QRF] batched input gated pending numeric verification", q);
  }
  auto& r = outputs.at(1);
  auto& encoder = omarchy::get_command_encoder(q.primitive().stream());
  linalg_require_f32(name(), in, q);
  const int m = in.shape(-2);
  const int n = in.shape(-1);
  const int k = std::min(m, n);
  const uint32_t batch =
      checked_u32(linalg_batch_count(in.shape()), name(), q);
  q.set_data(allocate_omarchy(q.nbytes()));
  r.set_data(allocate_omarchy(r.nbytes()));
  if (m == 0 || n == 0 || batch == 0) {
    // Nothing to factorize; upstream leaves the empty outputs.
    return;
  }
  array work(in.shape(), in.dtype(), nullptr, {});
  work.set_data(allocate_omarchy(work.nbytes()));
  encoder.add_temporary(work);
  linalg_copy_dense(in, work, q.primitive().stream());
  omarchy::ComputeParams params;
  params.matrix_m = checked_u32(m, name(), q);
  params.matrix_n = checked_u32(n, name(), q);
  params.matrix_k = checked_u32(k, name(), q);
  params.output_size = batch;
  std::array<omarchy::ComputeBinding, 3> bindings{
      binding(work), binding(q), binding(r)};
  encoder.dispatch_compute(
      omarchy::ComputeKernel::LinalgQrF32, bindings, params, batch);
}
void QuantizedMatmul::eval_gpu(const std::vector<array>& inputs, array& out) {
  const std::string tag = name();
  // Non-affine modes pass three inputs, so the mode check must land
  // before the biases operand is bound.
  if (mode_ != QuantizationMode::Affine) {
    // Non-affine modes: one scale byte per group (round-up e8m0 for
    // mxfp4/mxfp8 at group 32, e4m3 for nvfp4 at group 16) and fp4
    // e2m1 or fp8 e4m3 element bytes packed little-endian into the
    // uint32 words, no bias term. Inputs {x, packed words, scale
    // bytes}; the output dtype is x's dtype (ops.cpp).
    const array& x = inputs[0];
    const array& w = inputs[1];
    const array& scales = inputs[2];
    if (bits_ != 4 && bits_ != 8) {
      omarchy::unsupported(tag + " bits", out);
    }
    if (group_size_ != 16 && group_size_ != 32) {
      omarchy::unsupported(tag + " group size", out);
    }
    auto& fp_encoder = omarchy::get_command_encoder(out.primitive().stream());
    require_float_dtype(tag, x, out, fp_encoder);
    if (
        w.dtype() != uint32 || w.ndim() < 2 || w.ndim() > 3 ||
        scales.dtype() != uint8 || scales.ndim() != w.ndim() ||
        scales.shape(0) != w.shape(0)) {
      omarchy::unsupported(tag + " weight layout", out);
    }
    std::optional<array> x_temp;
    std::optional<array> w_temp;
    std::optional<array> scales_temp;
    const array& x_d = ensure_dense(
        x, x.flags().row_contiguous, x_temp, fp_encoder, stream());
    const array& w_d = ensure_dense(
        w, w.flags().row_contiguous, w_temp, fp_encoder, stream());
    const array& scales_d = ensure_dense(
        scales,
        scales.flags().row_contiguous,
        scales_temp,
        fp_encoder,
        stream());
    int k = x_d.shape(-1);
    int n = out.shape(-1);
    size_t x_rows = x_d.size() / k;
    // mx.quantize packs along the dequantized last axis: transposed w
    // is [N, Kp] with scales [N, K / group_size]; non-transposed w is
    // [K, Np] with scales [K, N / group_size]; Np = N * bits / 32.
    // Both layouts are first-class; the shaders route on flags bit 0.
    // Slice dims read from the trailing axes so a 3D weight batch
    // checks one slice at a time.
    uint64_t packed_outer =
        static_cast<uint64_t>(w_d.shape(-1)) * 32u / bits_;
    uint64_t scale_outer =
        static_cast<uint64_t>(scales_d.shape(-1)) * group_size_;
    bool shape_ok = transpose_
        ? (w_d.shape(-2) == static_cast<uint32_t>(n) &&
              packed_outer == static_cast<uint64_t>(k) &&
              scale_outer == static_cast<uint64_t>(k))
        : (w_d.shape(-2) == static_cast<uint32_t>(k) &&
              packed_outer == static_cast<uint64_t>(n) &&
              scale_outer == static_cast<uint64_t>(n));
    if (!shape_ok) {
      omarchy::unsupported(tag + " shape", out);
    }
    // A 3D weight batch pairs one packed matrix per batch index: x
    // either carries the matching leading batch (out holds one slice
    // product per x slice) or broadcasts one shared x across every
    // slice, decided by the ops-shaped output size. A 2D weight keeps
    // the flat single-weight contract (batch = 1).
    size_t batch = 1;
    size_t m = x_rows;
    size_t x_step_rows = 0;
    if (w_d.ndim() == 3) {
      batch = w_d.shape(0);
      if (out.size() == x_rows * static_cast<size_t>(n)) {
        if (batch == 0 || x_rows % batch != 0) {
          omarchy::unsupported(tag + " shape", out);
        }
        m = x_rows / batch;
        x_step_rows = m;
      } else if (out.size() != batch * x_rows * static_cast<size_t>(n)) {
        omarchy::unsupported(tag + " shape", out);
      }
    }
    out.set_data(allocate_omarchy(out.nbytes()));
    if (out.size() == 0) {
      return;
    }
    omarchy::ComputeParams params;
    params.count = checked_u32(m * n, tag, out);
    params.operation = static_cast<uint32_t>(bits_);
    params.reduce_size = static_cast<uint32_t>(group_size_);
    params.output_size = params.count;
    params.lhs_offset = checked_item_offset(x_d, x_d.size(), tag, out);
    params.rhs_offset = checked_item_offset(w_d, w_d.size(), tag, out);
    // The scale stream is bytes; the shader reads it through the word
    // view with the constant-shift select chain, so the byte offset
    // rides in aux_offset.
    params.aux_offset =
        checked_item_offset(scales_d, scales_d.size(), tag, out);
    params.flags = transpose_ ? 0u : 1u;
    params.matrix_m = checked_u32(m, tag, out);
    params.matrix_n = checked_u32(n, tag, out);
    params.matrix_k = checked_u32(k, tag, out);
    params.output_offset = checked_item_offset(out, out.size(), tag, out);
    std::array<omarchy::ComputeBinding, 4> bindings{
        binding(x_d), binding(w_d), binding(scales_d), binding(out)};
    // Batched weights ride one dispatch per slice: the offsets advance
    // by whole slices (x only when it pairs the batch, not when it
    // broadcasts) and every slice sees the same single-batch shader
    // contract the 2D path has always served.
    uint32_t lhs_step =
        checked_u32(x_step_rows * static_cast<size_t>(k), tag, out);
    uint32_t rhs_step = checked_u32(w_d.size() / batch, tag, out);
    uint32_t aux_step = checked_u32(scales_d.size() / batch, tag, out);
    uint32_t out_step = checked_u32(
        static_cast<size_t>(m) * static_cast<size_t>(n), tag, out);
    constexpr uint32_t kFpGemvColumnsPerGroup = 8u;
    auto fp_vec_groups = (params.matrix_n + kFpGemvColumnsPerGroup - 1u) /
        kFpGemvColumnsPerGroup;
    if (params.matrix_m == 1u) {
      const auto& caps = fp_encoder.device().capabilities();
      bool subgroup_ready =
          caps.subgroup_size == 32u &&
          (caps.subgroup_operations & VK_SUBGROUP_FEATURE_ARITHMETIC_BIT) != 0;
      omarchy::capsim::require_backed(
          fp_encoder.device(),
          caps,
          subgroup_ready,
          "QmmVecSubgroupFp*",
          "subgroup_size==32+subgroup_ops_mask[ARITHMETIC]",
          (fp_encoder.device().hardware_capabilities().subgroup_size ==
                  32u &&
              (fp_encoder.device().hardware_capabilities()
                       .subgroup_operations &
                  VK_SUBGROUP_FEATURE_ARITHMETIC_BIT) != 0));
      auto vec_kernel = subgroup_ready
          ? select_float_kernel(
                out.dtype(),
                omarchy::ComputeKernel::QmmVecSubgroupFpF32,
                omarchy::ComputeKernel::QmmVecSubgroupFpF16,
                omarchy::ComputeKernel::QmmVecSubgroupFpBF16)
          : select_float_kernel(
                out.dtype(),
                omarchy::ComputeKernel::QmmVecFpF32,
                omarchy::ComputeKernel::QmmVecFpF16,
                omarchy::ComputeKernel::QmmVecFpBF16);
      for (size_t slice = 0; slice < batch; ++slice) {
        params.lhs_offset += slice == 0 ? 0u : lhs_step;
        params.rhs_offset += slice == 0 ? 0u : rhs_step;
        params.aux_offset += slice == 0 ? 0u : aux_step;
        params.output_offset += slice == 0 ? 0u : out_step;
        fp_encoder.dispatch_compute(
            vec_kernel,
            bindings,
            params,
            std::min(fp_vec_groups, omarchy::kMaxComputeGroupCountX),
            1u,
            1u);
      }
      return;
    }
    if (const char* tile_env = std::getenv("MLX_OMARCHY_QMM_TILE");
        tile_env == nullptr || std::strcmp(tile_env, "0") != 0) {
      auto tile_kernel = select_float_kernel(
          out.dtype(),
          omarchy::ComputeKernel::QmmTileFpF32,
          omarchy::ComputeKernel::QmmTileFpF16,
          omarchy::ComputeKernel::QmmTileFpBF16);
      uint32_t m_groups = (params.matrix_m + 15u) / 16u;
      uint32_t n_groups = (params.matrix_n + 15u) / 16u;
      for (size_t slice = 0; slice < batch; ++slice) {
        params.lhs_offset += slice == 0 ? 0u : lhs_step;
        params.rhs_offset += slice == 0 ? 0u : rhs_step;
        params.aux_offset += slice == 0 ? 0u : aux_step;
        params.output_offset += slice == 0 ? 0u : out_step;
        fp_encoder.dispatch_compute(
            tile_kernel,
            bindings,
            params,
            std::min(n_groups, omarchy::kMaxComputeGroupCountX),
            std::min(m_groups, omarchy::kMaxComputeGroupCountX),
            1u);
      }
      return;
    }
    auto fp_kernel = select_float_kernel(
        out.dtype(),
        omarchy::ComputeKernel::QmmFpF32,
        omarchy::ComputeKernel::QmmFpF16,
        omarchy::ComputeKernel::QmmFpBF16);
    for (size_t slice = 0; slice < batch; ++slice) {
      params.lhs_offset += slice == 0 ? 0u : lhs_step;
      params.rhs_offset += slice == 0 ? 0u : rhs_step;
      params.aux_offset += slice == 0 ? 0u : aux_step;
      params.output_offset += slice == 0 ? 0u : out_step;
      fp_encoder.dispatch_compute(
          fp_kernel,
          bindings,
          params,
          omarchy::compute_dispatch_group_count(params.count));
    }
    return;
  }
  const array& x = inputs[0];
  const array& w = inputs[1];
  const array& scales = inputs[2];
  const array& biases = inputs[3];
  // Both affine layouts are first-class: transpose=true reads the
  // packed w rows [batch, N, K * bits / 32] and computes x @ w.T;
  // transpose=false reads w columns [batch, K * bits / 32, N] and
  // computes x @ w. Batched weights (3D) arrive broadcast-materialized
  // from ops.cpp, so every batch slice is contiguous and pairs with an
  // x batch slice of matrix_m rows.
  if (bits_ != 2 && bits_ != 3 && bits_ != 4 && bits_ != 5 && bits_ != 6 &&
      bits_ != 8) {
    omarchy::unsupported(tag + " bits", out);
  }
  if (group_size_ != 32 && group_size_ != 64 && group_size_ != 128) {
    omarchy::unsupported(tag + " group size", out);
  }
  auto& encoder = omarchy::get_command_encoder(out.primitive().stream());
  require_float_dtype(tag, x, out, encoder);
  if (w.dtype() != uint32 || w.ndim() < 2 || w.ndim() > 3) {
    omarchy::unsupported(tag + " weight layout", out);
  }
  if (scales.dtype() != out.dtype() || biases.dtype() != out.dtype()) {
    // ops.cpp promotes x, scales, and biases to one affine dtype.
    omarchy::unsupported(tag + " scales dtype", out);
  }
  // mx.quantize always packs along the dequantized last axis, so the
  // two layouts derive shapes differently: transposed w is
  // [batch, N, Kp] with scales [batch, N, K / group_size];
  // non-transposed w is [batch, K, Np] with scales
  // [batch, K, N / group_size]; Np = N * bits / 32.
  int packed_cols = transpose_ ? w.shape(-1) : w.shape(-2);
  int n = transpose_ ? w.shape(-2) : w.shape(-1) * 32 / bits_;
  int scale_cols = scales.shape(-1);
  int scale_rows = scales.shape(-2);
  if (scales.ndim() != w.ndim() || scales.shape() != biases.shape()) {
    omarchy::unsupported(tag + " scales shape", out);
  }
  const char* q4_word_env = std::getenv("MLX_OMARCHY_QMM_VEC_Q4_WORD");
  bool q4_g64_transpose = transpose_ && bits_ == 4 && group_size_ == 64;
  bool use_q4_word =
      (q4_word_env == nullptr || std::strcmp(q4_word_env, "1") == 0) &&
      q4_g64_transpose;
  // Prefill coopmat eligibility, resolved before operand normalization so
  // a row-contiguous x view at an odd f16-element offset is materialized
  // into an aligned buffer instead of silently rerouting to the tile
  // kernel: the coopmat shader reads x as 32-bit word pairs
  // (x_slice = lhs_offset / 2), and the tile kernel's different
  // accumulation order changes generated tokens, so the route must not
  // depend on where the allocator placed the activation. Costs a few
  // integer ops on the aligned fast path; the copy fires only on the rare
  // unaligned view.
  static const bool coopmat_disabled =
      omarchy::env_flag("MLX_OMARCHY_NO_COOPMAT");
  // Scalar-FMA prefill route (shaders/qmm_fma.comp): no matrix unit, no
  // shared memory. Reserved for drivers whose cooperative-matrix route is
  // absent (gate at qmm_fma_reachable below); MLX_OMARCHY_NO_QMM_FMA=1
  // falls back to the shipped pick for A/B measurement.
  static const bool qmm_fma_disabled =
      omarchy::env_flag("MLX_OMARCHY_NO_QMM_FMA");
  const char* tile_env = std::getenv("MLX_OMARCHY_QMM_TILE");
  bool tile_path = tile_env == nullptr || std::strcmp(tile_env, "0") != 0;
  const char* rb_env = std::getenv("MLX_OMARCHY_QMM_TILE_RB");
  bool rb_enabled = rb_env == nullptr || std::strcmp(rb_env, "0") != 0;
  constexpr uint32_t kQmmCoopmatSharedBytes =
      (32u * 16u + 16u * 32u) * sizeof(float);
  const auto& coopmat_caps = encoder.device().capabilities();
  bool coopmat_reachable =
      tile_path && rb_enabled && q4_g64_transpose &&
      (out.dtype() == float16 || out.dtype() == bfloat16) &&
      x.ndim() >= 2 && x.shape(-2) > 1 &&
      coopmat_caps.cooperative_matrix_f32_8 &&
      coopmat_caps.subgroup_size == 32u && !coopmat_disabled &&
      kQmmCoopmatSharedBytes <= coopmat_caps.max_compute_shared_memory_size;
  bool qmm_fma_reachable = false;
  // Driver split, measured (receipts/2026-09-12-prefill-fma-qualify): the
  // scalar-FMA kernel loses to the 8x8x8 cooperative-matrix route wherever
  // that route exists (fork: 545.6 vs 1034.6 GFLOP/s at the dominant
  // gate_up cell) and doubles the shipped tile fallback where it does not
  // (stock Mesa has no cooperative-matrix extension: 545.5 vs 275.6). The
  // FMA route is therefore only the non-coopmat driver's prefill path; on
  // a coopmat-capable driver the dispatch falls through to the shipped
  // cooperative-matrix pick by name and the FMA kernels stay reserved.
  if (!qmm_fma_disabled && !coopmat_reachable && tile_path && rb_enabled &&
      q4_g64_transpose && out.dtype() == float16 && x.ndim() >= 2 &&
      x.shape(-2) > 1) {
    int fma_k = x.shape(-1);
    int fma_n = transpose_ ? w.shape(-2) : w.shape(-1) * 32 / bits_;
    qmm_fma_reachable = (fma_k % 8 == 0) && (fma_n % 2 == 0);
  }
  // The f16 Q4 shader reads eight halves as one uvec4. Materialize only the
  // rare row-contiguous view whose element offset is not 16-byte aligned;
  // the coopmat word-pair reader extends the same rule to a 2-byte
  // alignment (an even f16 element offset).
  bool packed_q4_x = use_q4_word &&
      (out.dtype() == float16 || out.dtype() == bfloat16) &&
      (x.ndim() < 2 || x.shape(-2) == 1);
  bool x_dense = x.flags().row_contiguous &&
      (!packed_q4_x || x.offset() % (8 * x.itemsize()) == 0) &&
      (!coopmat_reachable || x.offset() % (2 * x.itemsize()) == 0) &&
      (!qmm_fma_reachable || x.offset() % (8 * x.itemsize()) == 0);
  std::optional<array> x_temp;
  std::optional<array> w_temp;
  std::optional<array> scales_temp;
  std::optional<array> biases_temp;
  const array& x_d = ensure_dense(x, x_dense, x_temp, encoder, stream());

  const array& w_d =
      ensure_dense(w, w.flags().row_contiguous, w_temp, encoder, stream());
  const array& scales_d = ensure_dense(
      scales, scales.flags().row_contiguous, scales_temp, encoder, stream());
  const array& biases_d = ensure_dense(
      biases, biases.flags().row_contiguous, biases_temp, encoder, stream());
  // A rank-1 x is the 1D qvm form: one row against the whole weight,
  // and the flat [n] output writes exactly like the [1, n] dispatch.
  int k = x.shape(-1);
  int m = x.ndim() >= 2 ? x.shape(-2) : 1;
  size_t batch = x.size() / (static_cast<int64_t>(m) * k);
  // out is x-shaped with the last axis replaced by n; a 2D w (or 2D
  // scales) is shared across the x batch, 3D operands pair one slice
  // per batch index. The w words per slice are shape(-2) * shape(-1)
  // in both layouts, so only the derived inner/outer pairings and the
  // output size need checking.
  int packed_inner =
      transpose_ ? w.shape(-1) * 32 / bits_ : w.shape(-2);
  if (packed_inner != k ||
      scale_cols * group_size_ != (transpose_ ? k : n) ||
      scale_rows != (transpose_ ? n : k) ||
      out.size() != batch * static_cast<size_t>(m) *
              static_cast<size_t>(n)) {
    omarchy::unsupported(tag + " shape", out);
  }
  // The bf16 non-transposed tile routes diverge from the dense bf16
  // matmul reference by one output ULP (2^-9 at the relevant magnitude;
  // upstream tolerance 1.5e-3) on the upstream sweep's K <= 128 shapes
  // (up to two 64-wide reduction blocks; K = 256 and K >= 512 pass, as
  // do the m == 1 vec route and bits 2). Named refusal until the bf16 tile
  // accumulation matches the matmul reference - never a wrong answer.
  if (out.dtype() == bfloat16 && !transpose_ && m > 1 && bits_ != 2 &&
      k <= 128) {
    omarchy::unsupported(tag + " bf16 non-transposed tile", out);
  }

  out.set_data(allocate_omarchy(out.nbytes()));
  if (out.size() == 0) {
    return;
  }


  // Push-constant routing for the qmm shaders: operation carries bits,
  // reduce_size the group size, shape[0] the batch count, flags bit 0
  // the non-transposed weight layout (see shaders/qmm.comp).
  uint64_t total =
      static_cast<uint64_t>(batch) * static_cast<uint64_t>(m) *
      static_cast<uint64_t>(n);
  omarchy::ComputeParams params;
  params.count = checked_u32(total, tag, out);
  params.operation = static_cast<uint32_t>(bits_);
  params.reduce_size = static_cast<uint32_t>(group_size_);
  params.output_size = params.count;
  params.lhs_offset = checked_item_offset(x_d, x_d.size(), tag, out);
  params.rhs_offset = checked_item_offset(w_d, w_d.size(), tag, out);
  params.aux_offset = checked_item_offset(scales_d, scales_d.size(), tag, out);
  params.aux_size = checked_item_offset(biases_d, biases_d.size(), tag, out);
  params.matrix_m = checked_u32(m, tag, out);
  params.matrix_n = checked_u32(n, tag, out);
  params.matrix_k = checked_u32(k, tag, out);
  params.output_offset = checked_item_offset(out, out.size(), tag, out);
  params.shape[0] = checked_u32(batch, tag, out);
  // A 2D w or 2D scales is shared across the x batch (stride 0);
  // broadcast 3D operands pair one slice per batch index.
  params.shape[1] =
      checked_u32(w.ndim() == 3 ? w_d.size() / batch : 0, tag, out);
  params.shape[2] =
      checked_u32(scales.ndim() == 3 ? scales_d.size() / batch : 0, tag, out);
  params.flags = transpose_ ? 0u : 1u;
  std::array<omarchy::ComputeBinding, 5> bindings{
      binding(x_d),
      binding(w_d),
      binding(scales_d),
      binding(biases_d),
      binding(out)};
  // DecodeGemv dispatch: when lhs has a single row, the per-row GEMV
  // path replaces the 16x16 tile. The subgroup-reduction variant is
  // picked when the device reports subgroupSize == 32 AND the ARITHMETIC
  // subgroup feature bit is set, both queried at device init and held
  // on the capability report. The default fall-through path is the
  // general Qmm kernel (unaffected by this addition).
  //
  // The subgroup variant replaces the five-round workgroup-shared
  // tree with one subgroupAdd per 32-lane slot. The microbenchmark
  // tools/subgroup-bench decides whether the trade pays; if the
  // device lacks subgroup support, this gate is a no-op and the
  // general path runs unchanged. See PROTOCOL.md for the keep rule.
  //
  // Gemv group count: COLUMNS_PER_GROUP output columns per workgroup,
  // matching the lane split in shaders/qmm_vec.comp.
  constexpr uint32_t kGemvColumnsPerGroup = 8u;
  auto n_groups_qmm_vec = (params.matrix_n + kGemvColumnsPerGroup - 1u) /
      kGemvColumnsPerGroup;
  if (params.matrix_m == 1u) {
    const auto& caps = encoder.device().capabilities();
    bool subgroup_ready =
        caps.subgroup_size == 32u &&
        (caps.subgroup_operations & VK_SUBGROUP_FEATURE_ARITHMETIC_BIT) != 0;
    omarchy::capsim::require_backed(
        encoder.device(),
        caps,
        subgroup_ready,
        "QmmVec*Subgroup*",
        "subgroup_size==32+subgroup_ops_mask[ARITHMETIC]",
        (encoder.device().hardware_capabilities().subgroup_size == 32u &&
         (encoder.device().hardware_capabilities().subgroup_operations &
          VK_SUBGROUP_FEATURE_ARITHMETIC_BIT) != 0));
    // Eligibility was computed before dense normalization because the packed
    // f16 input view also requires a 16-byte-aligned x row.
    auto vec_kernel = subgroup_ready
        ? select_float_kernel(
              out.dtype(),
              use_q4_word
                  ? omarchy::ComputeKernel::QmmVecQ4WordSubgroupF32
                  : omarchy::ComputeKernel::QmmVecSubgroupF32,
              use_q4_word
                  ? omarchy::ComputeKernel::QmmVecQ4WordSubgroupF16
                  : omarchy::ComputeKernel::QmmVecSubgroupF16,
              use_q4_word
                  ? omarchy::ComputeKernel::QmmVecQ4WordSubgroupBF16
                  : omarchy::ComputeKernel::QmmVecSubgroupBF16)
        : select_float_kernel(
              out.dtype(),
              use_q4_word ? omarchy::ComputeKernel::QmmVecQ4WordF32
                          : omarchy::ComputeKernel::QmmVecF32,
              use_q4_word ? omarchy::ComputeKernel::QmmVecQ4WordF16
                          : omarchy::ComputeKernel::QmmVecF16,
              use_q4_word ? omarchy::ComputeKernel::QmmVecQ4WordBF16
                          : omarchy::ComputeKernel::QmmVecBF16);
    encoder.dispatch_compute(
        vec_kernel,
        bindings,
        params,
        std::min(n_groups_qmm_vec, omarchy::kMaxComputeGroupCountX),
        1u,
        1u);
    return;
  }
  if (tile_path) {
    // bf16-activation coopmat prefill: the pinned mlx-community Qwen3.8
    // 4-bit snapshots are bf16-hybrid, so without this route their
    // prefill falls to the scalar 16x16 QmmTileBF16 baseline. The
    // f16-only FMA and register-blocked fallbacks do not apply; a
    // driver without the fp32 cooperative matrix keeps the shipped
    // tile path (the block below declines by not dispatching, matching
    // the pre-route behavior bit for bit).
    if (rb_enabled && out.dtype() == bfloat16 && transpose_ && bits_ == 4 &&
        group_size_ == 64 && coopmat_reachable) {
      if ((params.lhs_offset | params.output_offset) & 1u) {
        omarchy::unsupported(tag + " coopmat operand alignment", out);
      }
      omarchy::capsim::require_backed(
          encoder.device(),
          coopmat_caps,
          coopmat_reachable,
          "QmmPrefillCoopmatBF16",
          "cooperative_matrix_fp32_8x8x8",
          encoder.device().hardware_capabilities()
              .cooperative_matrix_f32_8);
      uint32_t n_groups = (params.matrix_n + 31u) / 32u;
      uint32_t coopmat_rows = coopmat_tile_rows(
          params.matrix_m,
          n_groups,
          encoder.device().hardware_capabilities().device_name);
      uint32_t m_groups =
          (params.matrix_m + coopmat_rows - 1u) / coopmat_rows;
      // Direct-global-load A: widen bf16 x to f32 once (the widening is
      // exact, so the kernel k chain is bit-identical to the staged
      // path), then the shader coopMatLoads A tiles straight from the
      // f32 buffer - the x_s shared tile, its per-lane staging stores
      // and address math never run.
      array x_f32(x_d.shape(), float32, nullptr, {});
      array::Flags xf_flags;
      xf_flags.contiguous = true;
      xf_flags.row_contiguous = true;
      xf_flags.col_contiguous = x_f32.size() <= 1;
      x_f32.set_data(
          allocate_omarchy(x_f32.nbytes()),
          x_f32.size(),
          Strides{1},
          xf_flags,
          0);
      encoder.add_temporary(x_f32);
      {
        omarchy::ComputeParams cparams;
        cparams.count = checked_u32(x_d.size(), tag, out);
        cparams.lhs_offset = checked_item_offset(x_d, x_d.size(), tag, out);
        cparams.output_offset = 0;
        std::array<omarchy::ComputeBinding, 3> cbindings{
            binding(x_d), binding(x_d), binding(x_f32)};
        encoder.dispatch_compute(
            omarchy::ComputeKernel::CastBF16F32,
            cbindings,
            cparams,
            omarchy::compute_dispatch_group_count(cparams.count));
      }
      // The cast writes a fresh buffer: offset 0 in f32 elements, and
      // every x offset the shader could see is even by construction.
      params.lhs_offset = 0;
      auto qmm_bindings = bindings;
      qmm_bindings[0] = binding(x_f32);
      omarchy::ComputeKernel qmm_kernel = coopmat_rows == 16u
          ? omarchy::ComputeKernel::QmmPrefillCoopmatM16BF16X32
          : omarchy::ComputeKernel::QmmPrefillCoopmatBF16X32;
      encoder.dispatch_compute(
          qmm_kernel,
          qmm_bindings,
          params,
          std::min(n_groups, omarchy::kMaxComputeGroupCountX),
          std::min(m_groups, omarchy::kMaxComputeGroupCountX),
          1u);
      return;
    }
    if (rb_enabled && out.dtype() == float16 && transpose_ && bits_ == 4 &&
        group_size_ == 64) {
      if (qmm_fma_reachable) {
        // Scalar-FMA route: the kernel's per-output arithmetic is the
        // same ascending-k f32 chain the tile and coopmat kernels
        // produce, so the route is digest-preserving by construction
        // and verified per leg. uvec4 x reads need the 16-byte x
        // alignment materialized above; the output is a fresh
        // offset-0 allocation, so the odd-offset tripwire below
        // mirrors the coopmat one.
        if ((params.lhs_offset & 7u) != 0u ||
            (params.output_offset & 1u) != 0u) {
          omarchy::unsupported(tag + " fma operand alignment", out);
        }
        // The accumulator contraction class follows the shipped tile
        // boundary: fused below m=1024, precise (NoContraction) at and
        // above - the classes the composed tree's digests were pinned
        // against on both drivers (window1/window1c screens).
        omarchy::ComputeKernel fma_kernel =
            params.matrix_m >= 1024u
                ? omarchy::ComputeKernel::QmmPrefillFmaPreciseF16
                : omarchy::ComputeKernel::QmmPrefillFmaF16;
        encoder.dispatch_compute(
            fma_kernel,
            bindings,
            params,
            std::min((params.matrix_n + 63u) / 64u,
                     omarchy::kMaxComputeGroupCountX),
            std::min((params.matrix_m + 63u) / 64u,
                     omarchy::kMaxComputeGroupCountX),
            1u);
        return;
      }
      // The 8x8x8 fp32 cooperative matrix (shaders/qmm_coopmat.comp)
      // remains the route when the scalar-FMA kernel declines (odd n or
      // unaligned operands): a 32-column output tile per workgroup,
      // 4 KiB shared staging at the shipped 32 rows; x is
      // read as 32-bit word pairs). The materialization above stages any
      // odd-offset x view and out is a fresh offset-0 allocation, so
      // operand alignment holds by construction and coopmat_reachable
      // alone decides the route. If that contract ever broke, refusing
      // by name beats silently rerouting to the tile kernel, whose
      // different accumulation order shifts generated ids.
      // MLX_OMARCHY_NO_COOPMAT=1 forces the register-blocked tile.
      bool coopmat = coopmat_reachable;
      if (coopmat &&
          ((params.lhs_offset | params.output_offset) & 1u) != 0u) {
        omarchy::unsupported(tag + " coopmat operand alignment", out);
      }
      omarchy::capsim::require_backed(
          encoder.device(),
          coopmat_caps,
          coopmat,
          "QmmPrefillCoopmatF16",
          "cooperative_matrix_fp32_8x8x8",
          encoder.device().hardware_capabilities()
              .cooperative_matrix_f32_8);
      // Coopmat row tile: the shipped 32 unless the shape-derived grid
      // cannot fill this part, in which case the 16-row twin trades
      // weight-tile reuse for workgroups. The device width comes from
      // hardware truth, not a simulated capability profile: the pick is
      // a property of the silicon, not of a capability bit. The
      // register-blocked fallback keeps its own 32-row tile.
      uint32_t n_groups = coopmat ? (params.matrix_n + 31u) / 32u
                                  : (params.matrix_n + 15u) / 16u;
      uint32_t coopmat_rows = coopmat
          ? coopmat_tile_rows(
                params.matrix_m,
                n_groups,
                encoder.device().hardware_capabilities().device_name)
          : 32u;
      uint32_t m_groups =
          (params.matrix_m + coopmat_rows - 1u) / coopmat_rows;
      omarchy::ComputeKernel qmm_kernel = !coopmat
          ? (params.matrix_m >= 1024u
                 ? omarchy::ComputeKernel::QmmTileRbPreciseF16
                 : omarchy::ComputeKernel::QmmTileRbF16)
          : coopmat_rows == 16u
          ? omarchy::ComputeKernel::QmmPrefillCoopmatM16F16
          : omarchy::ComputeKernel::QmmPrefillCoopmatF16;
      encoder.dispatch_compute(
          qmm_kernel,
          bindings,
          params,
          std::min(n_groups, omarchy::kMaxComputeGroupCountX),
          std::min(m_groups, omarchy::kMaxComputeGroupCountX),
          1u);
      return;
    }
    auto tile_kernel = select_float_kernel(
        out.dtype(),
        omarchy::ComputeKernel::QmmTileF32,
        omarchy::ComputeKernel::QmmTileF16,
        omarchy::ComputeKernel::QmmTileBF16);
    uint32_t m_groups = (params.matrix_m + 15u) / 16u;
    uint32_t n_groups = (params.matrix_n + 15u) / 16u;
    encoder.dispatch_compute(
        tile_kernel,
        bindings,
        params,
        std::min(n_groups, omarchy::kMaxComputeGroupCountX),
        std::min(m_groups, omarchy::kMaxComputeGroupCountX),
        1u);
    return;
  }
  auto kernel = select_float_kernel(
      out.dtype(),
      omarchy::ComputeKernel::QmmF32,
      omarchy::ComputeKernel::QmmF16,
      omarchy::ComputeKernel::QmmBF16);
  encoder.dispatch_compute(
      kernel,
      bindings,
      params,
      omarchy::compute_dispatch_group_count(params.count));
}

namespace omarchy {

namespace {

// The single-row Q4 word kernel's eligibility, mirrored for the fused
// group (QuantizedMatmul::eval_gpu keeps the authoritative copy).
bool q4_word_enabled() {
  const char* env = std::getenv("MLX_OMARCHY_QMM_VEC_Q4_WORD");
  return env == nullptr || std::strcmp(env, "1") == 0;
}

bool float_dtype_supported(Dtype dtype, const CapabilityReport& caps) {
  if (dtype == float32) {
    return true;
  }
  if (dtype == float16) {
    return caps.shader_float16 && caps.storage_buffer_16bit_access;
  }
  if (dtype == bfloat16) {
    return caps.storage_buffer_16bit_access && caps.shader_int16;
  }
  return false;
}

// Whole, dense, unoffset buffer of `count` elements: the multi kernel
// addresses every per-weight stream from element 0.
bool whole_dense(const array& value, size_t count) {
  return value.data_shared_ptr() != nullptr && value.offset() == 0 &&
      value.flags().row_contiguous && value.data_size() == count;
}

// Ready to be read by a dispatch recorded now on `stream`: evaluated,
// and not waiting on another stream's unsignaled event (the eval loop
// performs that wait only ahead of the node's own turn).
bool input_ready(const array& value, const Stream& stream) {
  if (value.status() == array::Status::unscheduled) {
    return false;
  }
  if (value.event().valid() && !value.event().is_signaled() &&
      value.event().stream() != stream) {
    return false;
  }
  return true;
}

} // namespace

bool dispatch_quantized_gemv_group(
    std::vector<GemvFusionMember>& members,
    array* swiglu_out,
    const Stream& stream) {
  if (members.empty() || members.size() > kQmmVecMultiWeights ||
      !q4_word_enabled()) {
    return false;
  }
  // SwiGLU store epilogue: exactly the gate and up projections, no
  // Add epilogues, f16/bf16 (the shader rounds through the same
  // integer packing swiglu.comp uses; there is no f32 swiglu kernel).
  if (swiglu_out &&
      (members.size() != 2 || members[0].epilogue || members[1].epilogue ||
       members[0].node.dtype() == float32)) {
    return false;
  }
  auto& encoder = get_command_encoder(stream);
  const auto& caps = encoder.device().capabilities();
  if (encoder.device().compute().binding_limit() < kQmmVecMultiBindings) {
    return false;
  }
  const array& x = members[0].node.inputs().at(0);
  const Dtype dtype = members[0].node.dtype();
  if (!float_dtype_supported(dtype, caps) || x.dtype() != dtype ||
      x.ndim() < 2 || x.shape(-2) != 1 || !input_ready(x, stream) ||
      x.data_shared_ptr() == nullptr || !x.flags().row_contiguous ||
      x.offset() % x.itemsize() != 0) {
    return false;
  }
  const int k = x.shape(-1);
  if (k <= 0 || k % 64 != 0 || static_cast<size_t>(k) != x.size()) {
    return false;
  }
  ComputeParams params;
  uint32_t total_groups = 0;
  for (size_t i = 0; i < members.size(); ++i) {
    const array& node = members[i].node;
    if (node.inputs().size() != 4 || node.inputs()[0].id() != x.id() ||
        node.dtype() != dtype || node.primitive().stream() != stream ||
        typeid(node.primitive()) != typeid(QuantizedMatmul)) {
      return false;
    }
    auto [group_size, bits, mode, transpose] =
        static_cast<const QuantizedMatmul&>(node.primitive()).state();
    if (mode != QuantizationMode::Affine || !transpose || bits != 4 ||
        group_size != 64) {
      return false;
    }
    const array& w = node.inputs()[1];
    const array& scales = node.inputs()[2];
    const array& biases = node.inputs()[3];
    if (w.dtype() != uint32 || w.ndim() != 2 || scales.dtype() != dtype ||
        biases.dtype() != dtype || scales.shape() != biases.shape() ||
        scales.ndim() != 2) {
      return false;
    }
    const int n = w.shape(0);
    if (n <= 0 || w.shape(1) != k / 8 || scales.shape(0) != n ||
        scales.shape(1) != k / 64 || node.size() != static_cast<size_t>(n) ||
        !whole_dense(w, w.size()) || !whole_dense(scales, scales.size()) ||
        !whole_dense(biases, biases.size()) || !input_ready(w, stream) ||
        !input_ready(scales, stream) || !input_ready(biases, stream)) {
      return false;
    }
    if (members[i].epilogue.has_value() != members[i].addend.has_value()) {
      return false;
    }
    if (members[i].epilogue) {
      const array& add = *members[i].epilogue;
      const array& addend = *members[i].addend;
      if (add.dtype() != dtype || add.size() != node.size() ||
          add.primitive().stream() != stream || addend.dtype() != dtype ||
          !whole_dense(addend, node.size()) || !input_ready(addend, stream)) {
        return false;
      }
      params.flags |= 256u << i;
    }
    params.shape[i] = static_cast<uint32_t>(n);
    total_groups += (static_cast<uint32_t>(n) + 7u) / 8u;
  }
  if (swiglu_out &&
      (params.shape[0] != params.shape[1] || swiglu_out->dtype() != dtype ||
       swiglu_out->size() != members[0].node.size())) {
    return false;
  }
  if (swiglu_out) {
    // Paired epilogue: every workgroup computes the same column slice
    // of both weights, so the gate count is weight 0's alone.
    total_groups = (params.shape[0] + 7u) / 8u;
    params.flags |= 65536u;
  }
  if (total_groups > kMaxComputeGroupCountX) {
    return false;
  }
  params.operation = 4u;
  params.reduce_size = 64u;
  params.matrix_m = 1u;
  params.matrix_k = static_cast<uint32_t>(k);
  params.dims = static_cast<uint32_t>(members.size());
  uint64_t x_offset = x.offset() / x.itemsize();
  if (!compute_index_span_fits(x_offset, x.size())) {
    return false;
  }
  params.lhs_offset = static_cast<uint32_t>(x_offset);
  params.count = static_cast<uint32_t>(total_groups);

  // Producer-direct KV windows: a member's Add epilogue may store its
  // sum straight into an updated cache copy (see fused_chain.h), which
  // deletes the merged SliceUpdatePair dispatch for that layer. The
  // base copy of the cache is enqueued here so the rows land in order.
  bool any_kv_window = false;
  for (auto& member : members) {
    if (!member.sum_window) {
      continue;
    }
    auto& window = *member.sum_window;
    if (!member.epilogue || window.node.data_shared_ptr() != nullptr ||
        window.base.data_shared_ptr() == nullptr ||
        (window.base.dtype() != float16 &&
         window.base.dtype() != bfloat16) ||
        !window.base.flags().row_contiguous ||
        window.base.size() != window.base.data_size() ||
        window.base.offset() % window.base.itemsize() != 0 ||
        !input_ready(window.base, stream)) {
      // The direct write cannot fire: keep the sum in its own buffer
      // and unwind the pair plan to the merged pair dispatch.
      abort_kv_direct();
      member.sum_window.reset();
      continue;
    }
    any_kv_window = true;
  }
  // Contract satisfied: allocate every output, then bind. Unused
  // weight slots bind the first member's output so every binding the
  // shader declares is a valid buffer.
  if (swiglu_out) {
    // The fold stores only the product; both member outputs alias it
    // so their retained references stay valid (their only readers
    // were the swiglu dispatch this fold deletes).
    swiglu_out->set_data(allocator().malloc(swiglu_out->nbytes()));
    Strides fold_strides(members[0].node.ndim(), 1);
    for (int i = members[0].node.ndim() - 2; i >= 0; --i) {
      fold_strides[i] =
          fold_strides[i + 1] * members[0].node.shape(i + 1);
    }
    array::Flags fold_flags;
    fold_flags.contiguous = true;
    fold_flags.row_contiguous = true;
    fold_flags.col_contiguous = false;
    members[0].node.copy_shared_buffer(
        *swiglu_out,
        fold_strides,
        fold_flags,
        members[0].node.data_size(),
        0);
    members[1].node.copy_shared_buffer(
        *swiglu_out,
        fold_strides,
        fold_flags,
        members[1].node.data_size(),
        0);
  } else {
    for (auto& member : members) {
      member.node.set_data(allocator().malloc(member.node.nbytes()));
      if (member.epilogue) {
        member.epilogue->set_data(allocator().malloc(member.epilogue->nbytes()));
      }
      if (member.sum_window) {
        auto& window = *member.sum_window;
        if (window.in_place) {
          // The planner proved the cache buffer's only in-tape consumer
          // is this SliceUpdate: the epilogue writes its rows straight
          // into the live cache and no fresh copy exists.
          window.node.copy_shared_buffer(
              window.base,
              window.base.strides(),
              window.base.flags(),
              window.base.data_size());
        } else {
          window.node.set_data(allocator().malloc(window.node.nbytes()));
          copy_gpu(
              window.base,
              window.node,
              window.base.flags().contiguous ? CopyType::Vector
                                             : CopyType::General,
              stream);
        }
        encoder.add_temporary(window.node);
        commit_values_kv_write(window.node);
        encoder.add_temporary(window.base);
      }
    }
  }
  if (any_kv_window) {
    for (size_t i = 0; i < members.size(); ++i) {
      if (!members[i].sum_window) {
        continue;
      }
      const auto& window = *members[i].sum_window;
      params.in_strides[i] = window.row_gap;
      params.out_strides[i] = window.offset;
      params.flags |= 4096u << i;
      params.matrix_m = window.head_dim;
    }
  }
  std::array<ComputeBinding, kQmmVecMultiBindings> bindings{};
  bindings[0] = binding(x);
  const ComputeBinding filler = binding(members[0].node);
  for (uint32_t i = 0; i < kQmmVecMultiWeights; ++i) {
    uint32_t base = 1 + i * kQmmVecMultiBindingsPerWeight;
    if (i < members.size()) {
      const auto& member = members[i];
      bindings[base] = binding(member.node.inputs()[1]);
      bindings[base + 1] = binding(member.node.inputs()[2]);
      bindings[base + 2] = binding(member.node.inputs()[3]);
      bindings[base + 3] = binding(member.node);
      bindings[base + 4] =
          member.addend ? binding(*member.addend) : binding(member.node);
      bindings[base + 5] = member.sum_window
          ? binding(member.sum_window->node)
          : (member.epilogue ? binding(*member.epilogue)
                             : binding(member.node));
    } else {
      for (uint32_t j = 0; j < kQmmVecMultiBindingsPerWeight; ++j) {
        bindings[base + j] = filler;
      }
    }
  }
  if (swiglu_out) {
    bindings[1 + 3] = binding(*swiglu_out);
  }
  bool subgroup_ready = caps.subgroup_size == 32u &&
      (caps.subgroup_operations & VK_SUBGROUP_FEATURE_ARITHMETIC_BIT) != 0;
  auto kernel = subgroup_ready
      ? select_float_kernel(
            dtype,
            ComputeKernel::QmmVecQ4MultiSubgroupF32,
            ComputeKernel::QmmVecQ4MultiSubgroupF16,
            ComputeKernel::QmmVecQ4MultiSubgroupBF16)
      : select_float_kernel(
            dtype,
            ComputeKernel::QmmVecQ4MultiF32,
            ComputeKernel::QmmVecQ4MultiF16,
            ComputeKernel::QmmVecQ4MultiBF16);
  encoder.dispatch_compute(kernel, bindings, params, total_groups, 1u, 1u);
  return true;
}

bool dispatch_dense_gemv_group(
    std::vector<array>& nodes,
    const array& x,
    const std::optional<KvDirectWindow>* sum_windows,
    const Stream& stream) {
  if (nodes.size() < 2 || nodes.size() > kDenseVecMultiWeights) {
    return false;
  }
  auto& encoder = get_command_encoder(stream);
  const auto& caps = encoder.device().capabilities();
  if (encoder.device().compute().binding_limit() < kDenseVecMultiBindings ||
      !caps.storage_buffer_16bit_access || !caps.shader_int16 ||
      caps.subgroup_size != 32u ||
      (caps.subgroup_operations &
       VK_SUBGROUP_FEATURE_SHUFFLE_RELATIVE_BIT) == 0u) {
    return false;
  }
  const array& x_view = nodes[0].inputs().at(0);
  if (x_view.dtype() != bfloat16 || x_view.ndim() < 2 ||
      x_view.shape(-2) != 1 ||
      x_view.size() != static_cast<size_t>(x_view.shape(-1)) ||
      x.dtype() != bfloat16 || x.size() != x_view.size() ||
      x.data_shared_ptr() == nullptr || !x.flags().row_contiguous ||
      x.strides().back() != 1 || !input_ready(x, stream) ||
      x.offset() % x.itemsize() != 0) {
    return false;
  }
  const int k = x_view.shape(-1);
  const uint64_t x_offset = x.offset() / x.itemsize();
  if (k <= 0 || k % 128 != 0 || (x_offset & 3u) != 0u ||
      !compute_index_span_fits(x_offset, x.size())) {
    return false;
  }

  ComputeParams params;
  params.lhs_offset = static_cast<uint32_t>(x_offset);
  params.matrix_k = static_cast<uint32_t>(k);
  params.dims = static_cast<uint32_t>(nodes.size());
  uint32_t total_groups = 0;
  for (size_t i = 0; i < nodes.size(); ++i) {
    const array& node = nodes[i];
    if (node.inputs().size() != 2 || node.dtype() != bfloat16 ||
        node.primitive().stream() != stream ||
        typeid(node.primitive()) != typeid(Matmul)) {
      return false;
    }
    const array& node_x = node.inputs()[0];
    const bool direct_input = node_x.id() == x.id();
    const bool aliased_input = node_x.dtype() == x.dtype() &&
        node_x.size() == x.size() && node_x.data_shared_ptr() != nullptr &&
        node_x.data_shared_ptr() == x.data_shared_ptr() &&
        node_x.offset() == x.offset() && node_x.flags().row_contiguous &&
        node_x.strides().back() == 1;
    if ((!direct_input && !aliased_input) || node_x.shape() != x_view.shape()) {
      return false;
    }
    const array& weight = node.inputs()[1];
    if (weight.dtype() != bfloat16 || weight.ndim() != node_x.ndim() ||
        weight.shape(-2) != k || weight.shape(-1) <= 0 ||
        weight.strides()[weight.ndim() - 2] != 1 ||
        weight.strides().back() != k ||
        weight.data_shared_ptr() == nullptr || !input_ready(weight, stream) ||
        weight.offset() % weight.itemsize() != 0) {
      return false;
    }
    const uint32_t n = static_cast<uint32_t>(weight.shape(-1));
    const uint64_t weight_offset = weight.offset() / weight.itemsize();
    const uint64_t weight_count = static_cast<uint64_t>(n) * k;
    if ((n & 3u) != 0u || node.size() != n ||
        (weight_offset & 3u) != 0u ||
        !compute_index_span_fits(weight_offset, weight_count)) {
      return false;
    }
    params.shape[i] = n;
    params.in_strides[i] = static_cast<uint32_t>(weight_offset);
    total_groups += n / 4u;
    if (total_groups > kMaxComputeGroupCountX) {
      return false;
    }
  }

  bool any_window = false;
  for (size_t i = 0; i < nodes.size(); ++i) {
    if (sum_windows && sum_windows[i]) {
      // The caller installed the window storage (in place: the live
      // cache; otherwise a fresh copy enqueued before this dispatch).
      any_window = true;
      continue;
    }
    nodes[i].set_data(allocator().malloc(nodes[i].nbytes()));
  }
  std::array<ComputeBinding, kDenseVecMultiBindings> bindings{};
  bindings[0] = binding(x);
  const ComputeBinding filler = binding(nodes[0]);
  for (uint32_t i = 0; i < kDenseVecMultiWeights; ++i) {
    const uint32_t base = 1 + i * kDenseVecMultiBindingsPerWeight;
    if (i < nodes.size()) {
      bindings[base] = binding(nodes[i].inputs()[1]);
      bindings[base + 1] = binding(nodes[i]);
    } else {
      bindings[base] = filler;
      bindings[base + 1] = filler;
    }
  }
  capsim::require_backed(
      encoder.device(),
      caps,
      true,
      "MatmulVecMultiBF16",
      "subgroup_size==32+subgroup_ops_mask[SHUFFLE_RELATIVE]",
      encoder.device().hardware_capabilities().subgroup_size == 32u &&
          (encoder.device().hardware_capabilities().subgroup_operations &
           VK_SUBGROUP_FEATURE_SHUFFLE_RELATIVE_BIT) != 0u);
  if (any_window) {
    for (size_t i = 0; i < nodes.size(); ++i) {
      if (sum_windows && sum_windows[i]) {
        const auto& w = *sum_windows[i];
        params.flags |= 4096u << i;
        params.out_strides[i] = w.offset;
        params.matrix_n = w.row_gap;
        params.matrix_m = w.head_dim;
      }
    }
  }
  encoder.dispatch_compute(
      ComputeKernel::MatmulVecMultiBF16,
      bindings,
      params,
      total_groups,
      1u,
      1u);
  return true;
}
bool dispatch_rope_pair(
    std::array<array, 2>& nodes,
    const Stream& stream) {
  auto& encoder = get_command_encoder(stream);
  const auto& caps = encoder.device().capabilities();
  if (encoder.device().compute().binding_limit() < 8 ||
      !caps.shader_float16 || !caps.storage_buffer_16bit_access) {
    return false;
  }
  static int trio_refuse_trace = 0;
  bool trace_refusals =
      std::getenv("MLX_OMARCHY_TRIO_TRACE") != nullptr &&
      trio_refuse_trace < 8;
  auto refuse = [&](const char* why) {
    if (trace_refusals) {
      ++trio_refuse_trace;
      std::fprintf(stderr, "[trio-pair] refuse: %s\n", why);
    }
    return false;
  };
  array& q_node = nodes[0];
  array& k_node = nodes[1];
  const auto& rope =
      static_cast<const mlx::core::fast::RoPE&>(q_node.primitive());
  const auto& rope_k =
      static_cast<const mlx::core::fast::RoPE&>(k_node.primitive());
  const array& q_in = q_node.inputs()[0];
  const array& k_in = k_node.inputs()[0];
  const array& offset = q_node.inputs()[1];
  auto settled = [](const array& a) {
    return a.data_shared_ptr() != nullptr;
  };
  if (!settled(q_in) || !settled(k_in)) {
    // Pre-settle the unsettled producer chains: the pair fires at the
    // first member while bfs_max_width tape segments can leave that
    // member's own producers unscheduled. The single path tolerates
    // that only because its eval comes later; the pair must bind both
    // inputs now. The offset is a host constant (checked by the
    // shared-offset contract), so only the two input chains need
    // settling.
    //
    // The chains are linear and short (Transpose <- Reshape <- bias-Add
    // <- GEMV group), and every deeper ancestor is already evaluated by
    // the outer tape, so record only the unsettled run, producer-first,
    // with the same per-node work the tape loop performs (gpu::eval +
    // status marking; no finalize - the enclosing frame owns the open
    // command buffer). settle() schedules exactly these nodes too, but
    // its degree pass also walks the chain's evaluated ancestors - the
    // residual stream makes every earlier layer one - which is O(graph)
    // host work per pair and O(graph^2) per token: the measured ctx1053
    // bottleneck. Any chain that is not a linear same-stream run
    // (fan-in, tracer, odd status) falls back to settle(), which
    // produces the identical dispatch schedule.
    std::vector<array> chain;
    bool linear = true;
    for (const array* leaf : {&q_in, &k_in}) {
      array cur = *leaf;
      while (linear && cur.data_shared_ptr() == nullptr) {
        if (chain.size() >= 64 || cur.is_tracer() || !cur.has_primitive() ||
            cur.status() != array::Status::unscheduled ||
            cur.primitive().stream() != q_node.primitive().stream()) {
          linear = false;
          break;
        }
        chain.push_back(cur);
        std::optional<array> next;
        for (const auto& in : cur.inputs()) {
          if (in.data_shared_ptr() != nullptr) {
            continue;
          }
          if (next) {
            linear = false; // fan-in: not a linear chain
            break;
          }
          next = in;
        }
        if (!linear) {
          break;
        }
        if (!next) {
          break; // chain head: every producer is resident
        }
        cur = std::move(*next);
      }
      if (!linear) {
        break;
      }
    }
    if (linear) {
      for (auto it = chain.rbegin(); it != chain.rend(); ++it) {
        gpu::eval(*it);
        it->set_status(array::Status::evaluated);
        for (auto& sib : it->siblings()) {
          sib.set_status(array::Status::evaluated);
        }
      }
    } else {
      std::vector<array> chains;
      if (!settled(q_in)) {
        chains.push_back(q_in);
      }
      if (!settled(k_in)) {
        chains.push_back(k_in);
      }
      settle(chains);
    }
  }
  if (!settled(q_in) || !settled(k_in) || !settled(offset)) {
    auto dump_settled = [](const array& in, const char* side) {
      array cur = in;
      for (int hop = 0; hop < 5 && cur.has_primitive(); ++hop) {
        std::fprintf(
            stderr,
            "[trio-settle] %s hop%d %s settled=%d status=%d size=%zu\n",
            side,
            hop,
            cur.primitive().name(),
            static_cast<int>(cur.data_shared_ptr() != nullptr),
            static_cast<int>(cur.status()),
            cur.size());
        if (cur.inputs().empty()) {
          break;
        }
        cur = cur.inputs()[0];
      }
    };
    std::fprintf(
        stderr,
        "[trio-pair] unsettled: q=%d k=%d off=%d\n",
        static_cast<int>(settled(q_in)),
        static_cast<int>(settled(k_in)),
        static_cast<int>(settled(offset)));
    if (!settled(q_in)) {
      dump_settled(q_in, "q");
    }
    if (!settled(k_in)) {
      dump_settled(k_in, "k");
    }
    return refuse("partner input not settled");
  }
  if (q_node.dtype() != float16 || k_node.dtype() != float16 ||
      q_in.dtype() != float16 || k_in.dtype() != float16) {
    return refuse("dtype not f16");
  }
  if (rope.state() != rope_k.state() || k_node.inputs().size() != 2) {
    return refuse("rope state mismatch");
  }
  if (offset.size() != 1) {
    return refuse("offset not scalar");
  }
  if (!(offset.id() == k_node.inputs()[1].id() ||
        (offset.size() == 1 && k_node.inputs()[1].size() == 1 &&
         offset.dtype() == int32 && k_node.inputs()[1].dtype() == int32 &&
         offset.status() == array::Status::available &&
         !offset.has_primitive() &&
         k_node.inputs()[1].status() == array::Status::available &&
         !k_node.inputs()[1].has_primitive() &&
         offset.item<int>() == k_node.inputs()[1].item<int>()))) {
    return refuse("offset not shared host-constant");
  }
  // Input readiness is the evaluator tape contract, exactly as for the
  // single RoPE dispatch: data_shared_ptr() can read as null for a
  // lazily-installed producer buffer at this point while the binding is
  // still valid (measured on jw16 - the null checks refused every pair
  // while the ordinary path produced pinned digests from the same
  // arrays), so no null or input_ready probe happens here.
  if (q_in.offset() % q_in.itemsize() != 0 ||
      k_in.offset() % k_in.itemsize() != 0) {
    return refuse("input offset unaligned");
  }
  const int dims = std::get<1>(rope.state());
  const bool traditional = std::get<2>(rope.state());
  const float base = std::get<3>(rope.state());
  const float scale = std::get<4>(rope.state());
  const int D = q_in.shape(-1);
  const int T = q_in.shape(-2);
  // Per-side input layout: the two layouts the single RoPE kernel
  // serves without a copy - row contiguous, or the regular 4D
  // head/seq transposed view - resolved to (matrix, time) strides with
  // an implicit feature stride of 1. Anything else (the single path
  // makes a contiguous temporary there) refuses the pair.
  auto side_layout = [&](const array& in) -> std::optional<
      std::pair<uint32_t, uint32_t>> {
    if (in.flags().row_contiguous) {
      return std::make_pair(
          static_cast<uint32_t>(T * D), static_cast<uint32_t>(D));
    }
    if (in.ndim() == 4) {
      const int64_t n = in.shape(1);
      if (in.strides()[0] == static_cast<int64_t>(T) * n * D &&
          in.strides()[1] == D &&
          in.strides()[2] == static_cast<int64_t>(n) * D &&
          in.strides()[3] == 1) {
        return std::make_pair(
            static_cast<uint32_t>(D), static_cast<uint32_t>(n) * D);
      }
      // Transposed slice of a wider row-contiguous [B,T,W,D] producer
      // buffer (a fused GEMV output): per-head stride D, per-time
      // stride W*D, batch stride T*W*D. The kernel addresses each side
      // through (matrix, time) strides only, so any W works; the
      // feature stride must stay 1. Row-contiguous inputs never reach
      // this branch (resolved above).
      if (in.strides()[1] == D && in.strides()[3] == 1 &&
          in.strides()[2] % D == 0 &&
          (in.shape(0) == 1 ||
           in.strides()[0] ==
               static_cast<int64_t>(T) * in.strides()[2])) {
        return std::make_pair(
            static_cast<uint32_t>(D),
            static_cast<uint32_t>(in.strides()[2]));
      }
    }
    return std::nullopt;
  };
  auto q_layout = side_layout(q_in);
  auto k_layout = side_layout(k_in);
  if (!q_layout || !k_layout) {
    if (trace_refusals) {
      std::fprintf(
          stderr,
          "[trio-pair] layout: q ndim=%d rowc=%d strides=%lld,%lld,%lld,%lld"
          " k ndim=%d rowc=%d strides=%lld,%lld,%lld,%lld\n",
          static_cast<int>(q_in.ndim()),
          static_cast<int>(q_in.flags().row_contiguous),
          static_cast<long long>(q_in.strides()[0]),
          static_cast<long long>(q_in.strides()[1]),
          static_cast<long long>(q_in.strides()[2]),
          static_cast<long long>(q_in.strides()[3]),
          static_cast<int>(k_in.ndim()),
          static_cast<int>(k_in.flags().row_contiguous),
          static_cast<long long>(k_in.strides()[0]),
          static_cast<long long>(k_in.strides()[1]),
          static_cast<long long>(k_in.strides()[2]),
          static_cast<long long>(k_in.strides()[3]));
    }
    return refuse("input layout unsupported");
  }
  const bool q_transposed = !q_in.flags().row_contiguous;
  const bool k_transposed = !k_in.flags().row_contiguous;
  if (q_in.ndim() != k_in.ndim() || k_in.shape(-1) != D ||
      k_in.shape(-2) != T || D != dims || dims % 2 != 0) {
    return refuse("shape mismatch");
  }
  int64_t heads_q = 1;
  for (int i = 1; i < q_in.ndim() - 2; ++i) {
    heads_q *= q_in.shape(i);
  }
  int64_t heads_k = 1;
  for (int i = 1; i < k_in.ndim() - 2; ++i) {
    heads_k *= k_in.shape(i);
  }
  const uint32_t half = static_cast<uint32_t>(dims / 2);
  const uint64_t cnt_q =
      static_cast<uint64_t>(q_in.shape(0)) * heads_q * T * half;
  const uint64_t cnt_k =
      static_cast<uint64_t>(k_in.shape(0)) * heads_k * T * half;
  const uint64_t total = cnt_q + cnt_k;
  if (cnt_q == 0 || cnt_k == 0 ||
      q_node.size() != static_cast<size_t>(2 * cnt_q) ||
      k_node.size() != static_cast<size_t>(2 * cnt_k) ||
      total > std::numeric_limits<uint32_t>::max() ||
      (total + 255u) / 256u > kMaxComputeGroupCountX) {
    return refuse("count or size consistency");
  }
  const uint64_t q_off = q_in.offset() / q_in.itemsize();
  const uint64_t k_off = k_in.offset() / k_in.itemsize();
  const uint64_t off_off = offset.offset() / offset.itemsize();
  if (!compute_index_span_fits(q_off, q_in.size()) ||
      !compute_index_span_fits(k_off, k_in.size()) ||
      !compute_index_span_fits(off_off, offset.size())) {
    return refuse("offset span does not fit");
  }
  // The producer-direct KV window re-check mirrors RoPE::eval_gpu:
  // find_rope_kv_redirect only fires once the values side committed,
  // and a committed-but-undispatchable window refuses there; un-plan
  // here so the ordinary path raises the same contract error.
  KvDirectWindow* kv_window = find_rope_kv_redirect(k_node);
  if (kv_window) {
    auto& window = *kv_window;
    if (window.node.data_shared_ptr() != nullptr ||
        window.base.data_shared_ptr() == nullptr ||
        window.base.dtype() != float16 ||
        !window.base.flags().row_contiguous ||
        window.base.size() != window.base.data_size() ||
        window.base.offset() % window.base.itemsize() != 0 ||
        !input_ready(window.base, stream)) {
      return refuse("kv window not dispatchable");
    }
  }
  // One gate call covers both sides: the shared offset, T, dims, base,
  // and scale make the theta bound identical.
  fast::rope_trig_gate(
      "rope_pair", q_in, offset, nullptr, dims, base, scale, q_node);
  if (kv_window) {
    auto& window = *kv_window;
    window.node.set_data(allocator().malloc(window.node.nbytes()));
    copy_gpu(
        window.base,
        window.node,
        window.base.flags().contiguous ? CopyType::Vector
                                       : CopyType::General,
        stream);
    encoder.add_temporary(window.node);
    encoder.add_temporary(window.base);
  } else {
    k_node.set_data(allocator().malloc(k_node.nbytes()));
  }
  q_node.set_data(allocator().malloc(q_node.nbytes()));
  array& k_out = kv_window ? kv_window->node : k_node;
  ComputeParams params;
  params.count = static_cast<uint32_t>(total);
  params.lhs_size = static_cast<uint32_t>(cnt_q);
  params.rhs_size = static_cast<uint32_t>(cnt_k);
  params.lhs_offset = static_cast<uint32_t>(q_off);
  params.rhs_offset = static_cast<uint32_t>(k_off);
  params.output_offset = kv_window ? kv_window->offset : 0u;
  params.aux_offset = static_cast<uint32_t>(off_off);
  params.matrix_m = 0u;
  params.flags = 1u | (traditional ? 2u : 0u) |
      (q_transposed ? 4u : 0u) | (k_transposed ? 8u : 0u);
  params.alpha = scale;
  params.beta = static_cast<float>(std::log(base) / half);
  params.dims = half;
  params.shape[0] = static_cast<uint32_t>(heads_q);
  params.shape[1] = static_cast<uint32_t>(heads_k);
  params.shape[2] = static_cast<uint32_t>(T);
  params.shape[3] = static_cast<uint32_t>(D);
  params.out_strides[0] = kv_window ? kv_window->strides[0]
                                    : static_cast<uint32_t>(T * D);
  params.out_strides[1] = kv_window ? kv_window->strides[1]
                                    : static_cast<uint32_t>(D);
  params.in_strides[0] = q_layout->first;
  params.in_strides[1] = q_layout->second;
  params.in_strides[2] = k_layout->first;
  params.in_strides[3] = k_layout->second;
  params.out_strides[2] = static_cast<uint32_t>(T * D);
  params.out_strides[3] = static_cast<uint32_t>(D);
  const ComputeBinding ro_filler = binding(q_in);
  const ComputeBinding wo_filler = binding(q_node);
  std::array<ComputeBinding, 8> bindings{
      binding(q_in),
      ro_filler,
      binding(k_in),
      binding(q_node),
      binding(k_out),
      wo_filler,
      wo_filler,
      binding(offset)};
  encoder.dispatch_compute(
      ComputeKernel::FastTrioRopePairF16,
      bindings,
      params,
      compute_dispatch_group_count(params.count));
  if (kv_window) {
    commit_rope_kv_redirect(k_node);
  }
  return true;
}

SliceUpdatePairDispatch dispatch_slice_update_pair(
    std::array<array, 2>& nodes,
    const Stream& stream) {
  auto& encoder = get_command_encoder(stream);
  const auto& caps = encoder.device().capabilities();
  if (encoder.device().compute().binding_limit() < 4 ||
      !caps.shader_float16 || !caps.storage_buffer_16bit_access) {
    return SliceUpdatePairDispatch::unsupported;
  }

  const auto& first_primitive =
      static_cast<const SliceUpdate&>(nodes[0].primitive());
  const auto first_state = first_primitive.state();
  const auto& starts = std::get<1>(first_state);
  const auto& slice_strides = std::get<3>(first_state);
  if (std::get<0>(first_state) != SliceUpdate::None ||
      nodes[0].inputs().size() != 2 || nodes[1].inputs().size() != 2 ||
      static_cast<const SliceUpdate&>(nodes[1].primitive()).state() !=
          first_primitive.state()) {
    return SliceUpdatePairDispatch::unsupported;
  }

  const array& first_update = nodes[0].inputs()[1];
  if (nodes[0].dtype() != float16 || nodes[1].dtype() != float16 ||
      nodes[0].shape() != nodes[1].shape() || first_update.size() == 0 ||
      first_update.ndim() == 0 || first_update.ndim() > 4 ||
      first_update.ndim() != nodes[0].ndim() ||
      first_update.size() > std::numeric_limits<uint32_t>::max()) {
    return SliceUpdatePairDispatch::unsupported;
  }

  auto [output_offset, output_strides] =
      prepare_slice(nodes[0], starts, slice_strides);
  auto [second_output_offset, second_output_strides] =
      prepare_slice(nodes[1], starts, slice_strides);
  if (output_offset != second_output_offset ||
      output_strides != second_output_strides) {
    return SliceUpdatePairDispatch::unsupported;
  }

  ComputeParams params;
  params.dims = static_cast<uint32_t>(first_update.ndim());
  params.count = static_cast<uint32_t>(first_update.size());
  params.output_offset = static_cast<uint32_t>(output_offset);
  uint64_t input_span = 0;
  uint64_t output_span = output_offset;
  for (int axis = 0; axis < first_update.ndim(); ++axis) {
    if (first_update.shape(axis) <= 0 || first_update.strides()[axis] < 0 ||
        output_strides[axis] < 0 ||
        static_cast<uint64_t>(first_update.shape(axis)) >
            std::numeric_limits<uint32_t>::max() ||
        static_cast<uint64_t>(first_update.strides()[axis]) >
            std::numeric_limits<uint32_t>::max() ||
        static_cast<uint64_t>(output_strides[axis]) >
            std::numeric_limits<uint32_t>::max()) {
      return SliceUpdatePairDispatch::unsupported;
    }
    params.shape[axis] = static_cast<uint32_t>(first_update.shape(axis));
    params.in_strides[axis] =
        static_cast<uint32_t>(first_update.strides()[axis]);
    params.out_strides[axis] = static_cast<uint32_t>(output_strides[axis]);
    uint64_t distance = static_cast<uint64_t>(first_update.shape(axis) - 1);
    input_span += distance * params.in_strides[axis];
    output_span += distance * params.out_strides[axis];
  }
  if (output_offset > std::numeric_limits<uint32_t>::max() ||
      output_span >= nodes[0].size()) {
    return SliceUpdatePairDispatch::unsupported;
  }

  for (size_t i = 0; i < nodes.size(); ++i) {
    const array& base = nodes[i].inputs()[0];
    const array& update = nodes[i].inputs()[1];
    if (nodes[i].primitive().stream() != stream ||
        base.dtype() != float16 || update.dtype() != float16 ||
        base.shape() != nodes[i].shape() ||
        update.shape() != first_update.shape() ||
        !base.flags().row_contiguous || base.size() != base.data_size() ||
        update.offset() % update.itemsize() != 0) {
      return SliceUpdatePairDispatch::unsupported;
    }
    if (!input_ready(base, stream) || !input_ready(update, stream)) {
      return SliceUpdatePairDispatch::not_ready;
    }
    if (base.data_shared_ptr() == nullptr || update.data_shared_ptr() == nullptr) {
      return SliceUpdatePairDispatch::unsupported;
    }
    uint64_t span = input_span;
    if (i == 1) {
      span = 0;
      for (int axis = 0; axis < update.ndim(); ++axis) {
        if (update.strides()[axis] < 0 ||
            static_cast<uint64_t>(update.strides()[axis]) >
                std::numeric_limits<uint32_t>::max()) {
          return SliceUpdatePairDispatch::unsupported;
        }
        uint32_t stride = static_cast<uint32_t>(update.strides()[axis]);
        span += static_cast<uint64_t>(update.shape(axis) - 1) * stride;
        switch (axis) {
          case 0: params.operation = stride; break;
          case 1: params.lhs_size = stride; break;
          case 2: params.rhs_size = stride; break;
          case 3: params.reduce_size = stride; break;
        }
      }
    }
    uint64_t offset = update.offset() / update.itemsize();
    if (offset > std::numeric_limits<uint32_t>::max() ||
        span > std::numeric_limits<uint32_t>::max() - offset) {
      return SliceUpdatePairDispatch::unsupported;
    }
    if (i == 0) {
      params.lhs_offset = static_cast<uint32_t>(offset);
    } else {
      params.rhs_offset = static_cast<uint32_t>(offset);
    }
  }

  for (auto& node : nodes) {
    copy_gpu(node.inputs()[0], node, CopyType::Vector, stream);
  }
  std::array<ComputeBinding, 4> bindings{
      binding(nodes[0].inputs()[1]),
      binding(nodes[1].inputs()[1]),
      binding(nodes[0]),
      binding(nodes[1])};
  encoder.dispatch_compute(
      ComputeKernel::SliceUpdatePairF16,
      bindings,
      params,
      compute_dispatch_group_count(params.count));
  return SliceUpdatePairDispatch::done;
}

} // namespace omarchy

void RandomBits::eval_gpu(const std::vector<array>& inputs, array& out) {
  const array& keys = inputs.at(0);
  auto& encoder = omarchy::get_command_encoder(out.primitive().stream());
  // Upstream random::bits maps width 4/2/1 to uint32/uint16/uint8
  // (mlx/random.cpp). One threefry word stream fills the output in all
  // three cases; the shader assembles each output word from the
  // elements that own its bytes, so per-key regions that end mid-word
  // stay byte-exact with eval_cpu's copy_remaining. A word-granular
  // layout (words_per_key whole words per key) was wrong for width 2
  // and 1 with several keys: key i's masked tail word zeroed key
  // i+1's leading bytes and shifted every later element by one slot
  // (the vmap take(out, array(1), 0) mismatch, 2026-09-06).
  if (!((width_ == 4 && out.dtype() == uint32) ||
        (width_ == 2 && out.dtype() == uint16) ||
        (width_ == 1 && out.dtype() == uint8))) {
    omarchy::unsupported("RandomBits width", out);
  }
  if (keys.dtype() != uint32) {
    omarchy::unsupported("RandomBits key dtype", out);
  }
  std::optional<array> keys_temp;
  const array& keys_d = ensure_dense(
      keys, keys.flags().row_contiguous, keys_temp, encoder, stream());
  out.set_data(allocate_omarchy(out.nbytes()));
  if (out.size() == 0 || keys.size() == 0) {
    return;
  }
  // Upstream layout: keys (N1, ..., NK, 2) and out
  // (N1, ..., NK, M1, M2, ...), so every key owns an equal number of
  // output elements.
  size_t num_keys = keys.size() / 2;
  if (keys.size() % 2 != 0 || out.size() % num_keys != 0) {
    omarchy::unsupported("RandomBits shape", out);
  }
  uint32_t count =
      checked_u32((out.nbytes() + 3) / 4, "RandomBits", out);
  omarchy::ComputeParams params;
  params.count = count;
  params.operation = static_cast<uint32_t>(width_);
  params.reduce_size = checked_u32(out.size() / num_keys, "RandomBits", out);
  params.output_size = checked_u32(num_keys, "RandomBits", out);
  params.lhs_offset = checked_item_offset(keys_d, keys_d.size(), "RandomBits", out);
  params.output_offset = checked_item_offset(out, out.size(), "RandomBits", out) *
      static_cast<uint32_t>(width_) / 4u;
  std::array<omarchy::ComputeBinding, 2> bindings{
      binding(keys_d), binding(out)};
  encoder.dispatch_compute(
      omarchy::ComputeKernel::RandomBitsU32,
      bindings,
      params,
      omarchy::compute_dispatch_group_count(count));
}
void Real::eval_gpu(const std::vector<array>& inputs, array& out) {
  // Upstream real() returns its input for real dtypes before a
  // primitive is built (mlx/ops.cpp), so eval only ever sees
  // complex64 here; the op extracts the real component, matching
  // complex64_t::operator float().
  dispatch_complex_extract(name(), 0, inputs, out, out.primitive().stream());
}

void Reduce::eval_gpu(const std::vector<array>& inputs, array& out) {
  const array& input = inputs.at(0);
  auto [reduce_type, axes] = state();
  std::string operation_name = name();
  auto& encoder = omarchy::get_command_encoder(out.primitive().stream());
  if (axes.empty()) {
    omarchy::unsupported(operation_name + " axes", out);
  }

  // mx.all / mx.any: ReduceType::And / Or produce a bool output from any
  // input dtype; float inputs keep their float capability gate.
  if (reduce_type == Reduce::And || reduce_type == Reduce::Or) {
    if (out.dtype() != bool_) {
      omarchy::unsupported(operation_name + " output dtype", out);
    }
    switch (input.dtype()) {
      case bool_:
      case int32:
      case uint32:
      case float32:
        break;
      case float16:
      case bfloat16:
        require_float_input(operation_name, input, out, encoder);
        break;
      default:
        omarchy::unsupported(operation_name + " dtype", out);
    }
    uint32_t operation =
        reduce_type == Reduce::Or ? ReduceAnyOperation : ReduceAllOperation;
    dispatch_reduce_general(
        operation_name,
        operation,
        input,
        out,
        axes,
        encoder,
        select_anyall_kernel(input.dtype()));
    return;
  }

  // Upstream bool Min/Max/Prod map onto the logical reductions
  // (mlx/backend/metal/reduce.cpp remap_reduce_types): max = any,
  // min = all, prod = all, each returning bool.
  if (input.dtype() == bool_ && out.dtype() == bool_ &&
      (reduce_type == Reduce::Max || reduce_type == Reduce::Min ||
       reduce_type == Reduce::Prod)) {
    dispatch_reduce_general(
        operation_name,
        reduce_type == Reduce::Max ? ReduceAnyOperation
                                   : ReduceAllOperation,
        input,
        out,
        axes,
        encoder,
        select_anyall_kernel(bool_));
    return;
  }

  if (reduce_type != Reduce::Sum && reduce_type != Reduce::Prod &&
      reduce_type != Reduce::Min && reduce_type != Reduce::Max) {
    omarchy::unsupported(operation_name, out);
  }

  // Float Sum and Max over a row-contiguous suffix keep the dedicated
  // suffix kernel; every empty reduction routes to the general kernel,
  // whose accumulator seed supplies the upstream identity value.
  bool float_dtype = input.dtype() == float32 || input.dtype() == float16 ||
      input.dtype() == bfloat16;
  bool suffix_fast_path = false;
  size_t reduce_size = 1;
  if (float_dtype && out.dtype() == input.dtype() &&
      (reduce_type == Reduce::Sum || reduce_type == Reduce::Max) &&
      input.flags().row_contiguous) {
    int first_axis = input.ndim() - static_cast<int>(axes.size());
    suffix_fast_path = true;
    for (int index = 0; index < static_cast<int>(axes.size()); ++index) {
      if (axes[index] != first_axis + index) {
        suffix_fast_path = false;
        break;
      }
    }
    for (int axis : axes) {
      reduce_size *= input.shape(axis);
    }
    if (reduce_size == 0) {
      suffix_fast_path = false;
    }
  }

  // RTMOD-BISECT: scalar-output reduces (out.size()==1) suspected in the
  // F1 first-submit stall; route them to the general reduce kernel.
  if (suffix_fast_path && out.size() > 1) {
    out.set_data(allocate_omarchy(out.nbytes()));
    if (out.size() == 0) {
      return;
    }
    uint32_t output_size = checked_u32(out.size(), operation_name, out);
    uint32_t reduce_size_u32 =
        checked_u32(reduce_size, operation_name, out);
    const array& bound_input = input.size() == 0 ? out : input;
    // The suffix kernel runs the reduction as a single per-invocation
    // serial loop; the driver caps that loop at a few thousand trips
    // (roughly 2^16 on lavapipe for local_size_x = 256), so reduce_size
    // is split across chunks when it exceeds the chunk-trip threshold.
    constexpr uint32_t kReduceSuffixChunkTrips = 4096;
    uint32_t chunks = reduce_size_u32 == 0
        ? 1u
        : (reduce_size_u32 + kReduceSuffixChunkTrips - 1u) /
            kReduceSuffixChunkTrips;
    if (chunks == 0) {
      chunks = 1;
    }
    Dtype scratch_dtype = float32;
    array scratch(
        Shape{static_cast<int>(static_cast<uint64_t>(output_size) * chunks)},
        scratch_dtype,
        nullptr,
        {});
    scratch.set_data(allocate_omarchy(scratch.nbytes()));
    encoder.add_temporary(scratch);
    omarchy::ComputeParams params;
    params.count = output_size;
    params.operation = reduce_type == Reduce::Sum ? 0u : 3u;
    params.reduce_size = reduce_size_u32;
    params.output_size = output_size;
    params.lhs_offset = checked_item_offset(
        bound_input, input.size(), operation_name, out);
    params.output_offset =
        checked_item_offset(out, out.size(), operation_name, out);
    params.matrix_n = chunks;
    auto kernel = select_float_kernel(
        out.dtype(),
        omarchy::ComputeKernel::ReduceF32,
        omarchy::ComputeKernel::ReduceF16,
        omarchy::ComputeKernel::ReduceBF16);
    std::array<omarchy::ComputeBinding, 3> bindings{
        binding(bound_input), binding(scratch), binding(out)};
    auto run_phase = [&](uint32_t flags, uint32_t dispatch_count) {
      params.flags = flags;
      encoder.dispatch_compute(
          kernel,
          bindings,
          params,
          omarchy::compute_dispatch_group_count(dispatch_count));
    };
    if (chunks == 1) {
      run_phase(0u | 2u, output_size);
    } else {
      run_phase(
          0u,
          checked_u32(
              static_cast<uint64_t>(output_size) * chunks,
              operation_name,
              out));
      run_phase(1u, output_size);
    }
    return;
  }

  Dtype in_dtype = input.dtype();
  bool is_sum_prod =
      reduce_type == Reduce::Sum || reduce_type == Reduce::Prod;
  bool bool_numeric = in_dtype == bool_ && out.dtype() == int32 && is_sum_prod;
  // The upstream remap_reduce_types contract (mlx/backend/metal/reduce.cpp):
  // narrow integers promote Sum/Prod outputs to int32/uint32 and keep
  // Min/Max in the input width; floats, 32/64-bit integers, and complex64
  // reduce in their own width.
  bool promoted_narrow = is_sum_prod &&
      ((in_dtype == int8 || in_dtype == int16) && out.dtype() == int32 ||
       (in_dtype == uint8 || in_dtype == uint16) && out.dtype() == uint32);
  bool exact_width = out.dtype() == in_dtype &&
      (float_dtype || in_dtype == int8 || in_dtype == uint8 ||
          in_dtype == int16 || in_dtype == uint16 || in_dtype == int32 ||
          in_dtype == uint32 || in_dtype == int64 || in_dtype == uint64 ||
          in_dtype == complex64);
  if (!bool_numeric && !promoted_narrow && !exact_width) {
    omarchy::unsupported(operation_name + " dtype", out);
  }
  if ((in_dtype == int64 || in_dtype == uint64) &&
      !encoder.device().capabilities().shader_int64) {
    omarchy::unsupported(operation_name + " int64 capability", out);
  }
  if (float_dtype) {
    require_float_input(operation_name, input, out, encoder);
  }
  uint32_t operation;
  switch (reduce_type) {
    case Reduce::Sum:
      operation = ReduceSumOperation;
      break;
    case Reduce::Prod:
      operation = ReduceProdOperation;
      break;
    case Reduce::Min:
      operation = ReduceMinOperation;
      break;
    default:
      operation = ReduceMaxOperation;
      break;
  }
  dispatch_reduce_general(
      operation_name,
      operation,
      input,
      out,
      axes,
      encoder,
      select_reduce_general_kernel(input.dtype()));
}

// Remainder is the Python-style modulo: the truncating remainder
// adjusted by one divisor when its sign differs from the divisor's.
// The integer kernel and the float kernel carry the same fixup, which
// reproduces the upstream integral_op / remainder value contract
// (result takes the divisor's sign) for both operand orders.
void Remainder::eval_gpu(const std::vector<array>& inputs, array& out) {
  if (is_int_elementwise_dtype(out.dtype())) {
    dispatch_int_elementwise(name(), IntModuloOperation, inputs, out);
    return;
  }
  dispatch_elementwise(
      name(), RemainderFloatOperation, inputs, out, out.primitive().stream());
}
void Round::eval_gpu(const std::vector<array>& inputs, array& out) {
  if (out.dtype() == complex64) {
    dispatch_complex(name(), ComplexRound, inputs, out, stream());
    return;
  }
  if (!issubdtype(out.dtype(), inexact)) {
    out.copy_shared_buffer(inputs.at(0));
    return;
  }
  dispatch_elementwise(name(), RoundOperation, inputs, out, stream());
}

// Scan serves cumsum, cumprod, cummax, and cummin over any axis, direction,
// and inclusivity: one invocation owns one line along the scan axis and
// walks it serially, which keeps the shared-buffer transport simple. The
// accumulator stays float32 for every float dtype; int32 and uint32 scan in
// their own width with wrapping arithmetic. The suffix float kernel keeps
// Sum so the categorical sampler path is byte-for-byte unchanged.
void Scan::eval_gpu(const std::vector<array>& inputs, array& out) {
  auto [reduce_type, axis, reverse, inclusive] = state();
  const array& input = inputs.at(0);
  auto& encoder = omarchy::get_command_encoder(out.primitive().stream());
  // Floats and complex64 scan through require_float_dtype; integers scan
  // in their own width (narrow wraps match the upstream narrow kernels);
  // bool cumsum produces the upstream int32 output. LogAddExp runs on the
  // float family and complex64.
  bool scan_float = input.dtype() == float32 || input.dtype() == float16 ||
      input.dtype() == bfloat16;
  bool scan_complex =
      input.dtype() == complex64 && out.dtype() == complex64;
  bool scan_int = out.dtype() == input.dtype() &&
      (input.dtype() == int8 || input.dtype() == uint8 ||
          input.dtype() == int16 || input.dtype() == uint16 ||
          input.dtype() == int32 || input.dtype() == uint32 ||
          input.dtype() == int64 || input.dtype() == uint64);
  bool scan_bool = input.dtype() == bool_ &&
      ((reduce_type == Scan::Sum && out.dtype() == int32) ||
       (reduce_type != Scan::Sum && reduce_type != Scan::LogAddExp &&
           out.dtype() == bool_));
  if (scan_float) {
    require_float_dtype("Scan", input, out, encoder);
  } else if (scan_complex) {
    // complex64 scans in its own width; out == in is checked above.
  } else if (scan_int || scan_bool) {
    // Supported.
  } else {
    omarchy::unsupported("Scan dtype", out);
  }
  if ((input.dtype() == int64 || input.dtype() == uint64) &&
      !encoder.device().capabilities().shader_int64) {
    omarchy::unsupported("Scan int64 capability", out);
  }
  if (input.ndim() == 0 || input.ndim() > 4 || axis < 0 ||
      axis >= input.ndim()) {
    omarchy::unsupported("Scan rank", out);
  }

  bool suffix_float_sum = reduce_type == Scan::Sum && !reverse &&
      scan_float && input.flags().row_contiguous &&
      axis == input.ndim() - 1;
  if (suffix_float_sum) {
    size_t row_length = input.shape(-1);
    size_t rows = row_length == 0 ? 0 : input.size() / row_length;
    out.set_data(allocate_omarchy(out.nbytes()));
    if (out.size() == 0) {
      return;
    }
    uint32_t output_size = checked_u32(rows, "Scan", out);
    if (output_size > omarchy::kMaxComputeGroupCountX) {
      omarchy::unsupported("Scan row count", out);
    }
    omarchy::ComputeParams params;
    params.count = checked_u32(out.size(), "Scan", out);
    params.operation = inclusive ? 0u : 1u;
    params.reduce_size = checked_u32(row_length, "Scan", out);
    params.output_size = output_size;
    params.lhs_offset = checked_item_offset(input, input.size(), "Scan", out);
    params.output_offset = checked_item_offset(out, out.size(), "Scan", out);
    std::array<omarchy::ComputeBinding, 3> bindings{
        binding(input), binding(input), binding(out)};
    auto kernel = select_float_kernel(
        out.dtype(),
        omarchy::ComputeKernel::ScanF32,
        omarchy::ComputeKernel::ScanF16,
        omarchy::ComputeKernel::ScanBF16);
    encoder.dispatch_compute(kernel, bindings, params, output_size);
    return;
  }

  // The general stride-walking scan. Push-constant routing mirrors
  // shaders/scan_general.comp: matrix_m carries the scan axis, flags bit 0
  // the reverse flag, reduce_size the operation selector, shape[] the
  // extents, in_strides[] the input strides, and out_strides[] the output
  // strides of the fresh row-contiguous output.
  uint32_t operation_selector;
  switch (reduce_type) {
    case Scan::Sum:
      operation_selector = 0u;
      break;
    case Scan::Prod:
      operation_selector = 1u;
      break;
    case Scan::Min:
      operation_selector = 2u;
      break;
    case Scan::Max:
      operation_selector = 3u;
      break;
    default:
      operation_selector = 4u;
      break;
  }
  out.set_data(allocate_omarchy(out.nbytes()));
  if (out.size() == 0) {
    return;
  }
  size_t axis_length = input.shape(axis);
  size_t lines = out.size() / axis_length;
  uint32_t output_size = checked_u32(lines, "Scan", out);
  if (output_size > omarchy::kMaxComputeGroupCountX) {
    omarchy::unsupported("Scan row count", out);
  }
  omarchy::ComputeParams params;
  params.count = checked_u32(out.size(), "Scan", out);
  params.operation = inclusive ? 0u : 1u;
  params.reduce_size = operation_selector;
  params.output_size = output_size;
  params.lhs_offset = checked_item_offset(input, input.size(), "Scan", out);
  params.output_offset = checked_item_offset(out, out.size(), "Scan", out);
  params.matrix_m = static_cast<uint32_t>(axis);
  params.flags = reverse ? 1u : 0u;
  params.dims = static_cast<uint32_t>(input.ndim());
  uint32_t output_stride = 1;
  for (int dim = input.ndim() - 1; dim >= 0; --dim) {
    params.shape[dim] = static_cast<uint32_t>(input.shape(dim));
    params.in_strides[dim] =
        static_cast<uint32_t>(input.strides()[dim]);
    params.out_strides[dim] = output_stride;
    output_stride *= static_cast<uint32_t>(input.shape(dim));
  }
  std::array<omarchy::ComputeBinding, 3> bindings{
      binding(input), binding(input), binding(out)};
  omarchy::ComputeKernel kernel;
  switch (input.dtype()) {
    case bool_:
      kernel = omarchy::ComputeKernel::ScanGeneralBool;
      break;
    case int8:
      kernel = omarchy::ComputeKernel::ScanGeneralI8;
      break;
    case uint8:
      kernel = omarchy::ComputeKernel::ScanGeneralU8;
      break;
    case int16:
      kernel = omarchy::ComputeKernel::ScanGeneralI16;
      break;
    case uint16:
      kernel = omarchy::ComputeKernel::ScanGeneralU16;
      break;
    case int32:
      kernel = omarchy::ComputeKernel::ScanGeneralI32;
      break;
    case uint32:
      kernel = omarchy::ComputeKernel::ScanGeneralU32;
      break;
    case int64:
      kernel = omarchy::ComputeKernel::ScanGeneralI64;
      break;
    case uint64:
      kernel = omarchy::ComputeKernel::ScanGeneralU64;
      break;
    case complex64:
      kernel = omarchy::ComputeKernel::ScanGeneralComplex;
      break;
    case float16:
      kernel = omarchy::ComputeKernel::ScanGeneralF16;
      break;
    case bfloat16:
      kernel = omarchy::ComputeKernel::ScanGeneralBF16;
      break;
    default:
      kernel = omarchy::ComputeKernel::ScanGeneralF32;
      break;
  }
  encoder.dispatch_compute(kernel, bindings, params, output_size);
}
// The word-transport kernel family (raw None phases, keys, integer
// Sum/Prod) tracks the index-array count so each array gets its own
// storage-buffer slot: U32 for float32/int32/uint32 words, packed
// 16-bit and bool variants for the rest.
omarchy::ComputeKernel scatter_word_kernel(
    bool multi_index, bool triple_index, Dtype dtype) {
  if (triple_index) {
    switch (dtype) {
      case float16:
        return omarchy::ComputeKernel::ScatterTripleF16;
      case bfloat16:
        return omarchy::ComputeKernel::ScatterTripleBF16;
      case bool_:
        return omarchy::ComputeKernel::ScatterBoolTriple;
      case uint8:
        return omarchy::ComputeKernel::ScatterTripleU8;
      case int8:
        return omarchy::ComputeKernel::ScatterTripleI8;
      case uint16:
        return omarchy::ComputeKernel::ScatterTripleU16;
      case int16:
        return omarchy::ComputeKernel::ScatterTripleI16;
      default:
        return omarchy::ComputeKernel::ScatterTripleU32;
    }
  }
  if (multi_index) {
    switch (dtype) {
      case float16:
        return omarchy::ComputeKernel::ScatterMultiF16;
      case bfloat16:
        return omarchy::ComputeKernel::ScatterMultiBF16;
      case bool_:
        return omarchy::ComputeKernel::ScatterBoolMulti;
      case uint8:
        return omarchy::ComputeKernel::ScatterMultiU8;
      case int8:
        return omarchy::ComputeKernel::ScatterMultiI8;
      case uint16:
        return omarchy::ComputeKernel::ScatterMultiU16;
      case int16:
        return omarchy::ComputeKernel::ScatterMultiI16;
      default:
        return omarchy::ComputeKernel::ScatterMultiU32;
    }
  }
  switch (dtype) {
    case float16:
      return omarchy::ComputeKernel::ScatterF16;
    case bfloat16:
      return omarchy::ComputeKernel::ScatterBF16;
    case bool_:
      return omarchy::ComputeKernel::ScatterBool;
    case uint8:
      return omarchy::ComputeKernel::ScatterU8;
    case int8:
      return omarchy::ComputeKernel::ScatterI8;
    case uint16:
      return omarchy::ComputeKernel::ScatterU16;
    case int16:
      return omarchy::ComputeKernel::ScatterI16;
    default:
      return omarchy::ComputeKernel::ScatterU32;
  }
}
void Scatter::eval_gpu(const std::vector<array>& inputs, array& out) {
  auto [reduce_type, axes] = state();
  const bool no_index = axes.empty();
  const array& src = inputs.at(0);
  const array& indices = inputs.at(1);
  const array& updates = inputs.back();
  auto& encoder = omarchy::get_command_encoder(out.primitive().stream());
  const bool multi_index = axes.size() >= 2;
  const bool triple_index = axes.size() == 3;
  const bool general_index = axes.size() > 3;
  bool is_sum = reduce_type == Scatter::Sum;
  bool is_prod = reduce_type == Scatter::Prod;
  const bool float_reduce =
      out.dtype() == float32 || out.dtype() == float16 ||
      out.dtype() == bfloat16;
  const bool complex_reduce = out.dtype() == complex64;
  if (multi_index) {
    // The multi-index kernel binds out, updates, one index array per
    // axis, and - except for integer Sum - the rank/key or accumulation
    // scratch: three to six slots against the budget the device
    // reported at initialization.
    uint32_t base = triple_index ? 5u : (multi_index ? 4u : 3u);
    uint32_t needed = general_index ? 5u
        : (is_sum && !float_reduce && out.dtype() != bool_) ? base
                                                            : base + 1u;
    uint32_t allowed = encoder.device().compute().binding_limit();
    if (needed > allowed) {
      omarchy::unsupported(
          "multi-index Scatter needs " + std::to_string(needed) +
              " storage-buffer bindings; this device allows " +
              std::to_string(allowed),
          out);
    }
  }
  // Upstream grounds the float Sum/Prod path: the Metal backend
  // scatters float32 through native float atomics (is_metal_atomic
  // includes float) and float16/bfloat16 through a packed CAS loop, so
  // duplicate-index accumulation order is nondeterministic upstream
  // too and no ordering contract exists to preserve. Hardware fp32
  // atomicAdd (op 11) needs shaderBufferFloat32AtomicAdd: llvmpipe
  // reports it, the M1 Honeykrisp does not advertise the extension at
  // all, so devices without it run the FCAS kernel variants whose
  // op-17 compare-exchange add is the exact-arithmetic twin of op 11.
  // Prod's op-13 CAS multiply never needed the extension.
  const bool hw_atomic_add =
      encoder.device().capabilities().shader_atomic_float_add;
  if (multi_index && (is_sum || is_prod) &&
      (out.dtype() == float16 || out.dtype() == bfloat16)) {
    // Only fp32 has a multi-index FADD blob; fp16/bf16 accumulation
    // rides the single-index kernel.
    omarchy::unsupported(
        "multi-index Scatter float16/bfloat16 Sum/Prod dtype",
        out);
  }
  // The second and third index arrays in the multi-index case;
  // upstream's op layer broadcasts and dtype-promotes every index
  // array to one shape.
  const array& indices_b = multi_index ? inputs.at(2) : indices;
  const array& indices_c = triple_index ? inputs.at(3) : indices;
  if (complex_reduce &&
      (multi_index || (!is_sum && reduce_type != Scatter::None))) {
    omarchy::unsupported("Scatter complex reduction", out);
  }
  if (out.dtype() == float16 || out.dtype() == bfloat16) {
    require_float_dtype("Scatter", src, out, encoder);
  }
  uint32_t map_code;
  omarchy::ComputeKernel kernel;
  switch (out.dtype()) {
    case float32:
      map_code = 0;
      if (is_sum || is_prod) {
        kernel = general_index
            ? (hw_atomic_add ? omarchy::ComputeKernel::ScatterFAddGeneralF32
                             : omarchy::ComputeKernel::ScatterFCasGeneralF32)
            : triple_index
            ? (hw_atomic_add ? omarchy::ComputeKernel::ScatterFAddTripleF32
                             : omarchy::ComputeKernel::ScatterFCasTripleF32)
            : multi_index
            ? (hw_atomic_add ? omarchy::ComputeKernel::ScatterFAddMultiF32
                             : omarchy::ComputeKernel::ScatterFCasMultiF32)
            : (hw_atomic_add ? omarchy::ComputeKernel::ScatterFAddF32
                             : omarchy::ComputeKernel::ScatterFCasF32);
      } else {
        kernel = general_index ? omarchy::ComputeKernel::ScatterGeneralU32
                               : scatter_word_kernel(multi_index, triple_index, out.dtype());
      }
      break;
    case int32:
      map_code = 1;
      kernel = general_index ? omarchy::ComputeKernel::ScatterGeneralU32
                             : scatter_word_kernel(multi_index, triple_index, out.dtype());
      break;
    case uint32:
      map_code = 2;
      kernel = general_index ? omarchy::ComputeKernel::ScatterGeneralU32
                             : scatter_word_kernel(multi_index, triple_index, out.dtype());
      break;
    case float16:
      map_code = 0;
      if (is_sum || is_prod) {
        kernel = hw_atomic_add ? omarchy::ComputeKernel::ScatterFAddF16
                               : omarchy::ComputeKernel::ScatterFCasF16;
      } else {
        kernel = general_index ? omarchy::ComputeKernel::ScatterGeneralF16
                               : scatter_word_kernel(multi_index, triple_index, out.dtype());
      }
      break;
    case bfloat16:
      map_code = 0;
      if (is_sum || is_prod) {
        kernel = hw_atomic_add ? omarchy::ComputeKernel::ScatterFAddBF16
                               : omarchy::ComputeKernel::ScatterFCasBF16;
      } else {
        kernel = general_index ? omarchy::ComputeKernel::ScatterGeneralBF16
                               : scatter_word_kernel(multi_index, triple_index, out.dtype());
      }
      break;
    case bool_:
      map_code = 1;
      kernel = general_index ? omarchy::ComputeKernel::ScatterGeneralBool
                             : scatter_word_kernel(multi_index, triple_index, out.dtype());
      break;
    case complex64:
      map_code = 0;
      kernel = omarchy::ComputeKernel::ScatterComplex64;
      break;
    case uint8:
    case int8:
    case uint16:
    case int16: {
      // The narrow variants implement the rank-slot NONE path only;
      // Max/Min would need widened key scratch and Sum/Prod a widened
      // accumulation, so those reductions keep the named refusal.
      if (!no_index && reduce_type != Scatter::None) {
        omarchy::unsupported("Scatter dtype", out);
      }
      if (out.dtype() == uint16 || out.dtype() == int16) {
        const auto& caps = encoder.device().capabilities();
        if (!caps.storage_buffer_16bit_access || !caps.shader_int16) {
          omarchy::unsupported("Scatter 16-bit capability", out);
        }
      }
      map_code = (out.dtype() == int8 || out.dtype() == int16) ? 1 : 2;
      if (general_index) {
        kernel = out.dtype() == uint8 ? omarchy::ComputeKernel::ScatterGeneralU8
            : out.dtype() == int8 ? omarchy::ComputeKernel::ScatterGeneralI8
            : out.dtype() == uint16 ? omarchy::ComputeKernel::ScatterGeneralU16
                                    : omarchy::ComputeKernel::ScatterGeneralI16;
      } else {
        kernel = scatter_word_kernel(multi_index, triple_index, out.dtype());
      }
      break;
    }
    default:
      omarchy::unsupported("Scatter dtype", out);
  }
  CopyType copy_type = src.data_size() == 1 ? CopyType::Scalar
      : (src.flags().row_contiguous && src.data_size() == src.size())
      ? CopyType::Vector
      : CopyType::General;
  copy_gpu(src, out, copy_type, out.primitive().stream());
  if (updates.size() == 0 || (!no_index && indices.size() == 0)) {
    return;
  }
  if (!no_index && indices.size() > (size_t{1} << 31)) {
    omarchy::unsupported("Scatter slot count", out);
  }
  // Materialize non-dense index and update views. Test data_size
  // against size rather than trusting the flags: a broadcast view can
  // carry a contiguous flag while holding fewer elements than the
  // shape names (the lying-contiguous-flag defect family).
  const array* idx = &indices;
  std::optional<array> idx_mat;
  if (!no_index && (!indices.flags().row_contiguous ||
      indices.data_size() != indices.size())) {
    idx_mat = array(indices.shape(), indices.dtype(), nullptr, {});
    copy_gpu(indices, *idx_mat, CopyType::General, out.primitive().stream());
    encoder.add_temporary(*idx_mat);
    idx = &*idx_mat;
  }
  if (multi_index) {
    // Upstream promotes all index arrays together before the primitive
    // sees them, so the later arrays share the first's dtype and
    // shape; their layouts materialize like the first.
    if (indices_b.size() != indices.size()) {
      omarchy::unsupported("Scatter index shape", out);
    }
    if (indices_b.dtype() != indices.dtype()) {
      omarchy::unsupported("Scatter index dtype", out);
    }
  }
  if (general_index) {
    for (size_t i = 1; i < axes.size(); ++i) {
      const array& current = inputs.at(i + 1);
      if (current.size() != indices.size()) {
        omarchy::unsupported("Scatter index shape", out);
      }
      if (current.dtype() != indices.dtype()) {
        omarchy::unsupported("Scatter index dtype", out);
      }
    }
  }
  if (triple_index) {
    if (indices_c.size() != indices.size()) {
      omarchy::unsupported("Scatter index shape", out);
    }
    if (indices_c.dtype() != indices.dtype()) {
      omarchy::unsupported("Scatter index dtype", out);
    }
  }
  const array* idx_b = &indices_b;
  std::optional<array> idx_b_mat;
  if (multi_index && !general_index &&
      (!indices_b.flags().row_contiguous ||
       indices_b.data_size() != indices_b.size())) {
    idx_b_mat =
        array(indices_b.shape(), indices_b.dtype(), nullptr, {});
    copy_gpu(
        indices_b, *idx_b_mat, CopyType::General, out.primitive().stream());
    encoder.add_temporary(*idx_b_mat);
    idx_b = &*idx_b_mat;
  }
  const array* idx_c = &indices_c;
  std::optional<array> idx_c_mat;
  if (triple_index &&
      (!indices_c.flags().row_contiguous ||
       indices_c.data_size() != indices_c.size())) {
    idx_c_mat =
        array(indices_c.shape(), indices_c.dtype(), nullptr, {});
    copy_gpu(
        indices_c, *idx_c_mat, CopyType::General, out.primitive().stream());
    encoder.add_temporary(*idx_c_mat);
    idx_c = &*idx_c_mat;
  }
  const array* upd = &updates;
  std::optional<array> upd_mat;
  if (updates.data_size() != updates.size()) {
    // Broadcast update: Scalar for the single-source form, General for
    // the strided multi-source form; Vector would flat-copy past the
    // source elements.
    upd_mat = array(updates.shape(), updates.dtype(), nullptr, {});
    copy_gpu(
        updates,
        *upd_mat,
        updates.data_size() == 1 ? CopyType::Scalar : CopyType::General,
        out.primitive().stream());
    encoder.add_temporary(*upd_mat);
    upd = &*upd_mat;
  } else if (!updates.flags().row_contiguous) {
    upd_mat = array(updates.shape(), updates.dtype(), nullptr, {});
    copy_gpu(updates, *upd_mat, CopyType::Vector, out.primitive().stream());
    encoder.add_temporary(*upd_mat);
    upd = &*upd_mat;
  }
  if (no_index) {
    idx = upd;
  }
  std::optional<array> packed_indices;
  std::optional<array> axis_metadata;
  if (general_index) {
    size_t segment_bytes = (indices.nbytes() + 3u) & ~size_t{3};
    size_t segment_items = segment_bytes / indices.itemsize();
    packed_indices.emplace(
        Shape{static_cast<int>(segment_items * axes.size())},
        indices.dtype(),
        nullptr,
        std::vector<array>{});
    array::Flags flags;
    flags.contiguous = true;
    flags.row_contiguous = true;
    flags.col_contiguous = true;
    packed_indices->set_data(
        allocate_omarchy(packed_indices->nbytes()),
        packed_indices->size(),
        Strides{1},
        flags,
        0);
    encoder.add_temporary(*packed_indices);
    Strides dense_strides(indices.ndim(), 1);
    for (int axis = indices.ndim() - 2; axis >= 0; --axis) {
      dense_strides[axis] =
          dense_strides[axis + 1] * indices.shape(axis + 1);
    }
    for (size_t i = 0; i < axes.size(); ++i) {
      const array& current = inputs.at(i + 1);
      copy_gpu_inplace(
          current,
          *packed_indices,
          current.shape(),
          current.strides(),
          dense_strides,
          0,
          i * segment_items,
          CopyType::GeneralGeneral,
          out.primitive().stream());
    }
    std::vector<uint32_t> words;
    words.reserve(3 * axes.size());
    for (size_t axis : axes) {
      words.push_back(checked_u32(out.shape(axis), "Scatter", out));
    }
    for (size_t axis : axes) {
      words.push_back(checked_u32(out.strides(axis), "Scatter", out));
    }
    for (size_t i = 0; i < axes.size(); ++i) {
      words.push_back(checked_u32(i * segment_bytes / 4, "Scatter", out));
    }
    axis_metadata.emplace(
        Shape{static_cast<int>(words.size())},
        uint32,
        nullptr,
        std::vector<array>{});
    axis_metadata->set_data(
        allocate_omarchy(axis_metadata->nbytes()),
        axis_metadata->size(),
        Strides{1},
        flags,
        0);
    auto* metadata_buffer = static_cast<omarchy::VulkanBuffer*>(
        axis_metadata->buffer().ptr());
    std::memcpy(metadata_buffer->data, words.data(), axis_metadata->nbytes());
    encoder.add_temporary(*axis_metadata);
    idx = &*packed_indices;
  }
  uint32_t index_mode = no_index ? 0u : scatter_index_mode(*idx, out, "Scatter");
  size_t index_ndim = no_index ? 0u : indices.ndim();
  size_t update_ndim = updates.ndim() - index_ndim;
  if (update_ndim > 4) {
    omarchy::unsupported("Scatter update rank", out);
  }
  uint32_t count = no_index ? 1u : checked_u32(indices.size(), "Scatter", out);
  omarchy::ComputeParams params;
  // Output element bound for the shader's target guard: update blocks
  // whose trailing dims overflow the output are skipped instead of
  // writing out of range (upstream leaves that undefined).
  params.dims = checked_u32(out.size(), "Scatter", out);
  params.rhs_size = no_index ? 1u : 0u;
  params.reduce_size = no_index ? 1u : checked_u32(out.shape(axes[0]), "Scatter", out);
  params.output_size = checked_u32(updates.size() / count, "Scatter", out);
  if (multi_index && !general_index) {
    // Axis-1 addressing rides fields the single-index kernel never
    // reads: the axis dim in matrix_n, the axis stride in matrix_k,
    // and the second array's word offset in lhs_size.
    params.matrix_n = checked_u32(out.shape(axes[1]), "Scatter", out);
    params.matrix_k = checked_u32(out.strides(axes[1]), "Scatter", out);
    params.lhs_size = scatter_index_offset(*idx_b, out, "Scatter");
  }
  if (triple_index) {
    // Axis-2 rides the fields the narrower kernels never read: the
    // axis dim in lhs_gap, the axis stride in rhs_gap, and the third
    // array's word offset in out_strides[0]. rhs_size stays the
    // no-index flag the first axis checks and must not be reused.
    params.lhs_gap = checked_u32(out.shape(axes[2]), "Scatter", out);
    params.rhs_gap = checked_u32(out.strides(axes[2]), "Scatter", out);
    params.out_strides[0] = scatter_index_offset(*idx_c, out, "Scatter");
  }
  params.matrix_m = no_index ? 0u : checked_u32(out.strides(axes[0]), "Scatter", out);
  params.rhs_offset = no_index || general_index
      ? 0u
      : scatter_index_offset(*idx, out, "Scatter");
  if (general_index) {
    params.matrix_n = checked_u32(axes.size(), "Scatter", out);
  }
  params.output_offset = checked_item_offset(out, out.size(), "Scatter", out);
  params.aux_size = index_mode;
  params.aux_offset = map_code;
  params.flags = checked_u32(update_ndim, "Scatter", out);
  for (size_t i = 0; i < update_ndim; ++i) {
    params.shape[i] = checked_u32(
        updates.shape(index_ndim + i), "Scatter", out);
    params.in_strides[i] = checked_u32(out.strides(i), "Scatter", out);
  }
  // The kernel walks update elements: slot = t / output_size. The
  // update base offset matters for sliced views.
  params.lhs_offset =
      checked_item_offset(*upd, upd->size(), "Scatter", out);
  if (static_cast<uint64_t>(count) * params.output_size > 0xFFFFFFFFull) {
    omarchy::unsupported("Scatter update span", out);
  }
  uint32_t elements = checked_u32(
      static_cast<uint64_t>(count) * params.output_size, "Scatter", out);
  // The kernel loop bound is the update element count, not the slot
  // count: with multi-element update blocks, slot-count would drop
  // every element past the first block.
  params.count = elements;
  if (is_sum && (out.dtype() == int32 || out.dtype() == uint32)) {
    // Phase 6: integer atomicAdd. Integer addition is associative, so
    // the result is deterministic even under duplicate indices. The
    // multi-index kernel binds every index array; unused slots of the
    // array stay unbound for the narrower paths.
    std::array<omarchy::ComputeBinding, 5> bindings{
        binding(out), binding(*upd), binding(*idx), binding(*idx_b)};
    uint32_t bound;
    if (general_index) {
      bindings[3] = binding(out);
      bindings[4] = binding(*axis_metadata);
      bound = 5;
    } else {
      if (triple_index) {
        bindings[4] = binding(*idx_c);
      }
      bound = triple_index ? 5u : (multi_index ? 4u : 3u);
    }
    params.operation = 6;
    encoder.dispatch_compute(
        kernel,
        std::span<const omarchy::ComputeBinding>(bindings.data(), bound),
        params,
        omarchy::compute_dispatch_group_count(elements));
    return;
  }
  const bool bool_word = out.dtype() == bool_;
  if ((float_reduce && (is_sum || is_prod)) ||
      (complex_reduce && is_sum) ||
      (bool_word &&
       (is_sum || is_prod || reduce_type == Scatter::Max ||
        reduce_type == Scatter::Min))) {
    // Scratch accumulation paths: float Sum/Prod and bool Sum/Prod/
    // Max/Min. Float Sum runs op 11 (fp32 hardware atomicAdd) or the
    // FCAS op 17 (compare-exchange add) and stores at op 12; float
    // Prod runs op 13 (CAS fp32 multiply) and multiplies into the
    // copied src at op 16. Bool Sum/Max OR the canonical update bit
    // (op 6) and byte-write at op 14; Prod/Min AND (op 7) and
    // byte-write at op 15.
    bool float_sum = float_reduce && is_sum;
    bool float_prod = float_reduce && is_prod;
    bool bool_or = bool_word && (is_sum || reduce_type == Scatter::Max);
    bool bool_and = bool_word && (is_prod || reduce_type == Scatter::Min);
    uint32_t clear_value = float_prod ? 0x3F800000u
        : bool_and ? 0xFFFFFFFFu
        : 0u;
    size_t scratch_size = complex_reduce ? 2 * out.size() : out.size();
    array scratch = make_u32_scratch(scratch_size, encoder);
    dispatch_clear_u32(scratch, clear_value, encoder);
    std::array<omarchy::ComputeBinding, 6> bindings{
        binding(out),
        binding(*upd),
        binding(*idx),
        binding(*idx_b),
        binding(*idx_c),
        binding(scratch)};
    uint32_t bound;
    if (general_index) {
      bindings[3] = binding(scratch);
      bindings[4] = binding(*axis_metadata);
      bound = 5;
    } else {
      if (!multi_index) {
        bindings[3] = binding(scratch);
      } else if (!triple_index) {
        bindings[4] = binding(scratch);
      }
      bound = triple_index ? 6u : (multi_index ? 5u : 4u);
    }
    uint32_t accumulate;
    uint32_t finalize;
    if (complex_reduce) {
      accumulate = 18;
      finalize = 19;
    } else if (float_sum) {
      accumulate = hw_atomic_add ? 11u : 17u;
      finalize = 12;
    } else if (float_prod) {
      accumulate = 13;
      finalize = 16;
    } else if (bool_or) {
      accumulate = 6;
      finalize = 14;
    } else {
      accumulate = 7;
      finalize = 15;
    }
    params.operation = accumulate;
    encoder.dispatch_compute(
        kernel,
        std::span<const omarchy::ComputeBinding>(bindings.data(), bound),
        params,
        omarchy::compute_dispatch_group_count(elements));
    params.count = checked_u32(out.size(), "Scatter", out);
    params.operation = finalize;
    encoder.dispatch_compute(
        kernel,
        std::span<const omarchy::ComputeBinding>(bindings.data(), bound),
        params,
        omarchy::compute_dispatch_group_count(params.count));
    return;
  }
  uint32_t phase1 = 0;
  uint32_t phase2 = 0;
  uint32_t clear_value = 0;
  if (reduce_type == Scatter::None) {
    phase1 = 0;
    phase2 = 1;
  } else if (reduce_type == Scatter::Max) {
    phase1 = 2;
    phase2 = 4;
  } else if (reduce_type == Scatter::Prod) {
    phase1 = 7;
    phase2 = 8;
    clear_value = 1;
  } else {
    phase1 = 3;
    phase2 = 5;
    clear_value = 0xFFFFFFFFu;
  }
  array scratch = make_u32_scratch(out.size(), encoder);
  dispatch_clear_u32(scratch, clear_value, encoder);
  // The multi-index kernel moves scratch to binding 4 and binds the
  // second index array at binding 3; the span width follows the case.
  std::array<omarchy::ComputeBinding, 6> bindings{
      binding(out),
      binding(*upd),
      binding(*idx),
      binding(*idx_b),
      binding(*idx_c),
      binding(scratch)};
  uint32_t bound;
  if (general_index) {
    bindings[3] = binding(scratch);
    bindings[4] = binding(*axis_metadata);
    bound = 5;
  } else {
    if (!multi_index) {
      bindings[3] = binding(scratch);
    } else if (!triple_index) {
      bindings[4] = binding(scratch);
    }
    bound = triple_index ? 6u : (multi_index ? 5u : 4u);
  }
  params.operation = phase1;
  encoder.dispatch_compute(
      kernel,
      std::span<const omarchy::ComputeBinding>(bindings.data(), bound),
      params,
      omarchy::compute_dispatch_group_count(elements));
  // None pass 2 walks update elements; Max/Min finalize walks output
  // elements.
  if (reduce_type == Scatter::None) {
    params.count = elements;
  } else {
    params.count = checked_u32(out.size(), "Scatter", out);
  }
  params.operation = phase2;
  encoder.dispatch_compute(
      kernel,
      std::span<const omarchy::ComputeBinding>(bindings.data(), bound),
      params,
      omarchy::compute_dispatch_group_count(params.count));
}
void ScatterAxis::eval_gpu(const std::vector<array>& inputs, array& out) {
  auto [reduce_type, axis] = state();
  const array& src = inputs.at(0);
  const array& indices = inputs.at(1);
  const array& updates = inputs.at(2);
  auto& encoder = omarchy::get_command_encoder(out.primitive().stream());
  bool is_sum = reduce_type == ScatterAxis::Sum;
  if (out.dtype() == float16 || out.dtype() == bfloat16) {
    require_float_dtype("ScatterAxis", src, out, encoder);
  }
  const bool float_reduce =
      out.dtype() == float32 || out.dtype() == float16 ||
      out.dtype() == bfloat16;
  const bool complex_reduce = out.dtype() == complex64;
  // Same upstream grounding as general Scatter: Metal's float axis
  // scatter add is a native atomic fetch add, so duplicate order is
  // nondeterministic upstream and no ordering contract exists to
  // preserve. Devices without shaderBufferFloat32AtomicAdd run the
  // FCAS axis variants (op 17 compare-exchange add) instead of
  // refusing: the M1 Honeykrisp does not advertise
  // VK_EXT_shader_atomic_float at all.
  const bool hw_atomic_add =
      encoder.device().capabilities().shader_atomic_float_add;
  omarchy::ComputeKernel kernel;
  switch (out.dtype()) {
    case float32:
      kernel = !is_sum ? omarchy::ComputeKernel::ScatterAxisU32
          : hw_atomic_add ? omarchy::ComputeKernel::ScatterAxisFAddF32
                          : omarchy::ComputeKernel::ScatterAxisFCasF32;
      break;
    case int32:
    case uint32:
      kernel = omarchy::ComputeKernel::ScatterAxisU32;
      break;
    case float16:
      kernel = !is_sum ? omarchy::ComputeKernel::ScatterAxisF16
          : hw_atomic_add ? omarchy::ComputeKernel::ScatterAxisFAddF16
                          : omarchy::ComputeKernel::ScatterAxisFCasF16;
      break;
    case bfloat16:
      kernel = !is_sum ? omarchy::ComputeKernel::ScatterAxisBF16
          : hw_atomic_add ? omarchy::ComputeKernel::ScatterAxisFAddBF16
                          : omarchy::ComputeKernel::ScatterAxisFCasBF16;
      break;
    case bool_:
      kernel = omarchy::ComputeKernel::ScatterAxisBool;
      break;
    case complex64:
      kernel = omarchy::ComputeKernel::ScatterAxisComplex64;
      break;
    default:
      omarchy::unsupported("ScatterAxis dtype", out);
  }
  omarchy::capsim::require_backed(
      encoder.device(),
      encoder.device().capabilities(),
      hw_atomic_add && is_sum,
      "ScatterAxisFAdd*",
      "atomic_float_add",
      encoder.device().hardware_capabilities().shader_atomic_float_add);
  CopyType copy_type = src.data_size() == 1 ? CopyType::Scalar
      : (src.flags().row_contiguous && src.data_size() == src.size())
      ? CopyType::Vector
      : CopyType::General;
  copy_gpu(src, out, copy_type, out.primitive().stream());
  if (indices.size() == 0) {
    return;
  }
  if (indices.size() > (size_t{1} << 31)) {
    omarchy::unsupported("ScatterAxis slot count", out);
  }
  // Materialize non-dense index views; test data_size against size
  // rather than trusting the flags (lying-contiguous-flag family).
  const array* idx = &indices;
  std::optional<array> idx_mat;
  if (!indices.flags().row_contiguous ||
      indices.data_size() != indices.size()) {
    idx_mat = array(indices.shape(), indices.dtype(), nullptr, {});
    copy_gpu(indices, *idx_mat, CopyType::General, out.primitive().stream());
    encoder.add_temporary(*idx_mat);
    idx = &*idx_mat;
  }
  int non_axis = out.ndim() - 1;
  if (non_axis > 4) {
    omarchy::unsupported("ScatterAxis rank", out);
  }
  size_t post_size = 1;
  for (int i = axis + 1; i < out.ndim(); ++i) {
    post_size *= out.shape(i);
  }
  uint32_t count = checked_u32(indices.size(), "ScatterAxis", out);
  omarchy::ComputeParams params;
  params.count = count;
  params.reduce_size = checked_u32(src.shape(axis), "ScatterAxis", out);
  params.output_size = checked_u32(post_size, "ScatterAxis", out);
  params.aux_size = checked_u32(indices.shape(axis), "ScatterAxis", out);
  params.aux_offset = scatter_index_mode(*idx, out, "ScatterAxis");
  // lhs_offset carries the update base element offset; the shader
  // walks update addresses through out_strides plus matrix_n.
  params.lhs_offset =
      checked_item_offset(updates, updates.size(), "ScatterAxis", out);
  params.rhs_offset = scatter_index_offset(*idx, out, "ScatterAxis");
  params.output_offset = checked_item_offset(out, out.size(), "ScatterAxis", out);
  params.matrix_m = checked_u32(src.strides(axis), "ScatterAxis", out);
  params.matrix_n = checked_u32(updates.strides(axis), "ScatterAxis", out);
  params.matrix_k = checked_u32(axis, "ScatterAxis", out);
  params.flags = checked_u32(non_axis, "ScatterAxis", out);
  for (int i = 0, d = 0; i < out.ndim(); ++i) {
    if (i == axis) {
      continue;
    }
    params.shape[d] = checked_u32(out.shape(i), "ScatterAxis", out);
    params.in_strides[d] = checked_u32(src.strides(i), "ScatterAxis", out);
    params.out_strides[d] = checked_u32(updates.strides(i), "ScatterAxis", out);
    ++d;
  }
  // Sum: int rides the direct atomicAdd (op 6); float accumulates in
  // the fp32 scratch (op 11, or the FCAS op-17 compare-exchange add
  // without hardware float atomics) and stores once (op 12); bool
  // ORs the canonical bit (op 6) and byte-writes (op 14). Duplicate
  // order matches upstream GPU semantics.
  if (is_sum) {
    bool int_sum = out.dtype() == int32 || out.dtype() == uint32;
    std::optional<array> scratch;
    if (!int_sum) {
      scratch = make_u32_scratch(
          complex_reduce ? 2 * out.size() : out.size(), encoder);
      dispatch_clear_u32(*scratch, 0, encoder);
    }
    std::array<omarchy::ComputeBinding, 4> bindings{
        binding(out),
        binding(*idx),
        binding(updates),
        int_sum ? binding(out) : binding(*scratch)};
    uint32_t bound = int_sum ? 3u : 4u;
    params.operation = int_sum || out.dtype() == bool_ ? 6u
        : complex_reduce ? 18u
        : hw_atomic_add ? 11u
        : 17u;
    encoder.dispatch_compute(
        kernel,
        std::span<const omarchy::ComputeBinding>(bindings.data(), bound),
        params,
        omarchy::compute_dispatch_group_count(count));
    if (!int_sum) {
      params.count = checked_u32(out.size(), "ScatterAxis", out);
      params.operation = out.dtype() == bool_ ? 14u
          : complex_reduce ? 19u
          : 12u;
      encoder.dispatch_compute(
          kernel,
          std::span<const omarchy::ComputeBinding>(bindings.data(), bound),
          params,
          omarchy::compute_dispatch_group_count(params.count));
    }
    return;
  }
  // None: rank-max scratch, then the winning slot rewrites the target
  // byte/word, which reproduces the CPU's sequential last-write-wins
  // order.
  array scratch = make_u32_scratch(out.size(), encoder);
  dispatch_clear_u32(scratch, 0, encoder);
  std::array<omarchy::ComputeBinding, 4> bindings{
      binding(out), binding(*idx), binding(updates), binding(scratch)};
  params.operation = 0;
  encoder.dispatch_compute(
      kernel,
      std::span<const omarchy::ComputeBinding>(bindings.data(), 4u),
      params,
      omarchy::compute_dispatch_group_count(count));
  params.operation = 1;
  encoder.dispatch_compute(
      kernel,
      std::span<const omarchy::ComputeBinding>(bindings.data(), 4u),
      params,
      omarchy::compute_dispatch_group_count(count));
}
// SearchSorted serves the categorical sampler: one binary search over
// a sorted 1-D row per value, with both operands already promoted to
// the same dtype by the op layer. The right flag picks between the
// first index at or after the value (left) and the first index after
// it (right), matching the upstream CPU comparator.
void SearchSorted::eval_gpu(const std::vector<array>& inputs, array& out) {
  const array& sequence = inputs.at(0);
  const array& values = inputs.at(1);
  bool right = state();
  auto& encoder = omarchy::get_command_encoder(out.primitive().stream());
  if (out.dtype() != uint32 || sequence.dtype() != values.dtype()) {
    omarchy::unsupported("SearchSorted dtype", out);
  }
  if (sequence.dtype() != float32 && sequence.dtype() != float16 &&
      sequence.dtype() != bfloat16 && sequence.dtype() != int32 &&
      sequence.dtype() != uint32) {
    omarchy::unsupported("SearchSorted dtype", out);
  }
  const auto& capabilities = encoder.device().capabilities();
  if (sequence.dtype() == float16 &&
      (!capabilities.shader_float16 ||
       !capabilities.storage_buffer_16bit_access)) {
    omarchy::unsupported("SearchSorted float16 capability", out);
  }
  if (sequence.dtype() == bfloat16 &&
      (!capabilities.storage_buffer_16bit_access ||
       !capabilities.shader_int16)) {
    omarchy::unsupported("SearchSorted bfloat16 capability", out);
  }
  if (sequence.ndim() != 1) {
    omarchy::unsupported("non-1-D SearchSorted", out);
  }
  std::optional<array> sequence_temp;
  std::optional<array> values_temp;
  const array& sequence_d = ensure_dense(
      sequence,
      sequence.flags().row_contiguous,
      sequence_temp,
      encoder,
      stream());
  const array& values_d = ensure_dense(
      values, values.flags().row_contiguous, values_temp, encoder, stream());
  out.set_data(allocate_omarchy(out.nbytes()));
  if (out.size() == 0) {
    return;
  }
  omarchy::ComputeParams params;
  params.count = checked_u32(values.size(), "SearchSorted", out);
  params.operation = right ? 1u : 0u;
  params.reduce_size = checked_u32(sequence.size(), "SearchSorted", out);
  params.output_size = params.count;
  params.lhs_offset = checked_item_offset(
      sequence_d, sequence_d.size(), "SearchSorted", out);
  params.rhs_offset = checked_item_offset(
      values_d, values_d.size(), "SearchSorted", out);
  params.output_offset = checked_item_offset(out, out.size(), "SearchSorted", out);
  std::array<omarchy::ComputeBinding, 3> bindings{
      binding(sequence_d), binding(values_d), binding(out)};
  auto kernel = sequence.dtype() == float16
      ? omarchy::ComputeKernel::SearchSortedF16
      : sequence.dtype() == bfloat16
      ? omarchy::ComputeKernel::SearchSortedBF16
      : sequence.dtype() == int32 ? omarchy::ComputeKernel::SearchSortedI32
                                  : sequence.dtype() == uint32
          ? omarchy::ComputeKernel::SearchSortedU32
          : omarchy::ComputeKernel::SearchSortedF32;
  encoder.dispatch_compute(
      kernel,
      bindings,
      params,
      omarchy::compute_dispatch_group_count(params.count));
}

// Select serves tril/triu (the where() pair behind composed lu), the
// sampler chain's scalar selects, and the composed causal mask. Value
// dtypes are float32, float16, bfloat16, int32, uint32, bool, complex64,
// int64, and uint64 (the last three ride two-raw-words or word-identity
// variants), and
// every operand layout routes through one of two transports in
// select.comp. The flat transport keeps the modulo fast path for dense
// operands. The general transport unravels the output coordinate over
// the collapsed broadcast shape and indexes the true and false operands
// through their own strides, where a stride of 0 broadcasts an axis -
// the same transport the compare and elementwise kernels use. Layout
// gates consult data_size and strides directly and never the contiguous
// flags: this session found five arrays whose flags claimed dense while
// data_size was smaller than size, so a flag-trusting gate would read
// out of bounds.
void Select::eval_gpu(const std::vector<array>& inputs, array& out) {
  const array& condition = inputs.at(0);
  const array& truthy = inputs.at(1);
  const array& falsy = inputs.at(2);
  auto& encoder = omarchy::get_command_encoder(out.primitive().stream());
  if (condition.dtype() != bool_ || truthy.dtype() != out.dtype() ||
      falsy.dtype() != out.dtype()) {
    omarchy::unsupported("Select dtype", out);
  }
  omarchy::ComputeKernel kernel;
  switch (out.dtype()) {
    case float32:
      kernel = omarchy::ComputeKernel::SelectF32;
      break;
    case float16:
      kernel = omarchy::ComputeKernel::SelectF16;
      break;
    case bfloat16:
      kernel = omarchy::ComputeKernel::SelectBF16;
      break;
    case int32:
    case uint32:
      kernel = omarchy::ComputeKernel::SelectI32;
      break;
    case bool_:
      kernel = omarchy::ComputeKernel::SelectBool;
      break;
    case complex64:
      // complex64 select rides the per-element uvec2 variant: the
      // value buffers carry two raw 32-bit words per element, the
      // condition stays a packed-bool byte stream, and the word
      // count math is unchanged because the per-thread lane loop
      // already counts condition elements, not value words.
      kernel = omarchy::ComputeKernel::SelectComplex64;
      break;
    case int64:
    case uint64:
      // Same two-raw-words bit-copy variant as complex64; a select
      // does not interpret the payload, and int64 elements are exactly
      // a uvec2. Serves the gated-delta chunked path (qwen3_5 GDN).
      kernel = omarchy::ComputeKernel::SelectI64;
      break;
    default:
      omarchy::unsupported("Select dtype", out);
  }
  // The value-variant kernels statically read the axis-metadata buffer
  // at binding 4 (their rank>4 broadcast transport), so every non-bool
  // Select dispatch carries five bindings; the packed-bool variant
  // keeps its dense-only four-binding interface. The spec floor is
  // four slots, so devices at the floor cannot run the value variants
  // and must refuse by name.
  if (out.dtype() != bool_ &&
      encoder.device().compute().binding_limit() < 5) {
    omarchy::unsupported(
        "Select needs 5 storage-buffer bindings; this device allows " +
            std::to_string(encoder.device().compute().binding_limit()) +
        ".",
        out);
  }
  // Dense row-major means the flat element index is the memory index:
  // data_size equals size and the strides are the exact suffix products.
  // Broadcast views carry a stride of 0 and transposed or sliced views
  // permute the strides, so all of them fail this check and route to
  // the general transport or a materialization.
  auto dense_row_major = [](const array& value) {
    if (value.data_size() != value.size()) {
      return false;
    }
    int64_t expected = 1;
    for (int axis = value.ndim() - 1; axis >= 0; --axis) {
      if (value.strides()[axis] != expected) {
        return false;
      }
      expected *= static_cast<int64_t>(value.shape(axis));
    }
    return true;
  };
  auto is_suffix_shape = [](const Shape& small, const Shape& big) {
    if (small.size() > big.size()) {
      return false;
    }
    size_t offset = big.size() - small.size();
    for (size_t axis = 0; axis < small.size(); ++axis) {
      if (small[axis] != big[offset + axis]) {
        return false;
      }
    }
    return true;
  };
  bool condition_flat =
      dense_row_major(condition) && condition.data_size() == out.size();
  bool truthy_flat = dense_row_major(truthy) && truthy.shape() == out.shape();
  bool falsy_scalar = falsy.data_size() == 1;
  bool falsy_flat = falsy_scalar ||
      (dense_row_major(falsy) &&
       (falsy.shape() == out.shape() ||
        is_suffix_shape(falsy.shape(), out.shape())));
  // The Honeykrisp bool word read only executes correctly in the
  // straight-line select.comp form, whose condition address requires a
  // flat condition. A broadcast or strided condition view is
  // materialized through the logical_or pipeline, whose dual-buffer
  // word reads are correct on that driver; binding the same buffer as
  // both inputs makes the OR an identity copy. The packed-bool output
  // kernel keeps every bool word load in that same straight-line form,
  // so it additionally requires its value operands dense over the full
  // output count and word-aligned; anything else is materialized the
  // same way. The encoder keeps the temporaries alive until the
  // committed work completes.
  auto materialize_bool_operand = [&](const array& value) {
    array dense(value.shape(), bool_, nullptr, {});
    dense.set_data(allocate_omarchy(dense.nbytes()));
    uint32_t material_count = checked_u32(dense.size(), name(), out);
    uint32_t material_words = checked_u32(
        (static_cast<uint64_t>(material_count) + 3) / 4, name(), out);
    // The logical_or kernel accumulates each canonical output byte with
    // atomicOr, so the destination must start zeroed - a fresh
    // allocation holds whatever the allocator recycled.
    omarchy::ComputeParams clear_params;
    clear_params.count = material_words;
    std::array<omarchy::ComputeBinding, 1> clear_bindings{binding(dense)};
    encoder.dispatch_compute(
        omarchy::ComputeKernel::ClearU32,
        clear_bindings,
        clear_params,
        omarchy::compute_dispatch_group_count(material_words));
    omarchy::ComputeParams material_params;
    material_params.count = material_count;
    material_params.lhs_size = checked_u32(value.data_size(), name(), out);
    material_params.rhs_size = material_params.lhs_size;
    material_params.output_size = material_count;
    material_params.lhs_offset = checked_item_offset(
        value, material_params.lhs_size, name(), out);
    material_params.rhs_offset = material_params.lhs_offset;
    material_params.output_offset = checked_item_offset(
        dense, material_count, name(), out);
    std::optional<array> axis_metadata = fill_broadcast_transport(
        name(), material_params, value, value, dense, encoder);
    std::array<omarchy::ComputeBinding, 4> material_bindings{
        binding(value),
        binding(value),
        binding(dense),
        binding(axis_metadata ? *axis_metadata : dense)};
    dispatch_logical_chunked(encoder, material_bindings, material_params);
    encoder.add_temporary(dense);
    return dense;
  };
  auto word_aligned = [](const array& value) {
    return value.offset() % value.itemsize() == 0 &&
        (value.offset() / value.itemsize()) % 4 == 0;
  };
  auto bool_kernel_ready = [&](const array& value) {
    return dense_row_major(value) && value.data_size() == out.size() &&
        word_aligned(value);
  };
  out.set_data(allocate_omarchy(out.nbytes()));
  if (out.size() == 0) {
    return;
  }
  std::vector<array> materialized;
  // Up to three operand materializations (bool output); reserving keeps
  // the .back() pointers below valid across pushes.
  materialized.reserve(3);
  const array* condition_ptr = &condition;
  const array* truthy_ptr = &truthy;
  const array* falsy_ptr = &falsy;
  bool general = false;
  if (out.dtype() == bool_) {
    if (truthy.shape() != out.shape() || falsy.shape() != out.shape()) {
      omarchy::unsupported("Select layout", out);
    }
    if (!condition_flat || !word_aligned(condition)) {
      materialized.push_back(materialize_bool_operand(condition));
      condition_ptr = &materialized.back();
    }
    if (!bool_kernel_ready(truthy)) {
      materialized.push_back(materialize_bool_operand(truthy));
      truthy_ptr = &materialized.back();
    }
    if (!bool_kernel_ready(falsy)) {
      materialized.push_back(materialize_bool_operand(falsy));
      falsy_ptr = &materialized.back();
    }
  } else {
    if (!condition_flat) {
      materialized.push_back(materialize_bool_operand(condition));
      condition_ptr = &materialized.back();
    }
    general = !truthy_flat || !falsy_flat;
    if (general &&
        (truthy.shape() != out.shape() || falsy.shape() != out.shape())) {
      omarchy::unsupported("Select layout", out);
    }
  }
  uint32_t count = checked_u32(out.size(), name(), out);
  uint32_t lhs_offset = checked_item_offset(
      *condition_ptr, condition_ptr->data_size(), name(), out);
  // A byte-misaligned flat condition shifts the per-word element window
  // by lhs_offset % 4, so the final word covers tail elements and the
  // dispatch must grow by the same amount (matching the shader's word
  // count). The materialized or aligned flat path has mis 0.
  uint32_t condition_mis = lhs_offset & 3;
  uint32_t word_count = checked_u32(
      (static_cast<uint64_t>(count) + condition_mis + 3) / 4, name(), out);
  omarchy::ComputeParams params;
  params.count = count;
  params.output_size = count;
  params.lhs_size = checked_u32(condition_ptr->data_size(), name(), out);
  params.rhs_size = checked_u32(truthy_ptr->data_size(), name(), out);
  params.aux_size = checked_u32(falsy_ptr->data_size(), name(), out);
  params.lhs_offset = lhs_offset;
  params.rhs_offset = checked_item_offset(
      *truthy_ptr, params.rhs_size, name(), out);
  params.aux_offset = checked_item_offset(
      *falsy_ptr, params.aux_size, name(), out);
  params.output_offset = checked_item_offset(out, count, name(), out);
  std::optional<array> axis_metadata;
  if (general) {
    // Collapse contiguous runs once for both strided operands; a stride
    // of 0 breaks every merge around a broadcast axis. Up to four
    // collapsed axes ride the inline push-constant arrays exactly like
    // the elementwise kernels; ranks 5 through 8 carry
    // [extents | in_strides | out_strides] in an axis-metadata storage
    // buffer (the false operand rides in_strides, the true operand
    // out_strides, matching the shader accessors), and above 8 the
    // primitive is refused by name with the limit in the message.
    auto [collapsed_shape, collapsed_strides] = collapse_contiguous_dims(
        out.shape(),
        std::vector<Strides>{truthy_ptr->strides(), falsy_ptr->strides()});
    if (collapsed_shape.size() > 8) {
      omarchy::unsupported(
          std::string("broadcast rank ") + name() +
              " exceeds the 8-axis transport limit",
          out);
    }
    params.dims = static_cast<uint32_t>(collapsed_shape.size());
    uint64_t true_span = 0;
    uint64_t false_span = 0;
    if (collapsed_shape.size() <= 4) {
      for (size_t axis = 0; axis < collapsed_shape.size(); ++axis) {
        params.shape[axis] = static_cast<uint32_t>(collapsed_shape[axis]);
        params.out_strides[axis] =
            static_cast<uint32_t>(collapsed_strides[0][axis]);
        params.in_strides[axis] =
            static_cast<uint32_t>(collapsed_strides[1][axis]);
        uint64_t extent = params.shape[axis] - 1u;
        true_span += extent * params.out_strides[axis];
        false_span += extent * params.in_strides[axis];
      }
    } else {
      size_t rank = collapsed_shape.size();
      std::vector<uint32_t> in_strides(rank);
      std::vector<uint32_t> out_strides(rank);
      for (size_t axis = 0; axis < rank; ++axis) {
        uint32_t extent = static_cast<uint32_t>(collapsed_shape[axis]);
        out_strides[axis] = static_cast<uint32_t>(collapsed_strides[0][axis]);
        in_strides[axis] = static_cast<uint32_t>(collapsed_strides[1][axis]);
        true_span += static_cast<uint64_t>(extent - 1u) * out_strides[axis];
        false_span += static_cast<uint64_t>(extent - 1u) * in_strides[axis];
      }
      std::vector<uint32_t> words;
      words.reserve(3 * rank);
      for (size_t axis = 0; axis < rank; ++axis) {
        words.push_back(static_cast<uint32_t>(collapsed_shape[axis]));
      }
      words.insert(words.end(), in_strides.begin(), in_strides.end());
      words.insert(words.end(), out_strides.begin(), out_strides.end());
      array metadata(
          Shape{static_cast<int>(words.size())}, uint32, nullptr, {});
      array::Flags flags;
      flags.contiguous = true;
      flags.row_contiguous = true;
      flags.col_contiguous = true;
      metadata.set_data(
          allocate_omarchy(metadata.nbytes()),
          metadata.size(),
          Strides{1},
          flags,
          0);
      auto* metadata_buffer =
          static_cast<omarchy::VulkanBuffer*>(metadata.buffer().ptr());
      std::memcpy(metadata_buffer->data, words.data(), metadata.nbytes());
      encoder.add_temporary(metadata);
      axis_metadata = std::move(metadata);
      params.matrix_k = static_cast<uint32_t>(rank);
    }
    if (!omarchy::compute_index_span_fits(params.rhs_offset, true_span + 1) ||
        !omarchy::compute_index_span_fits(
            params.aux_offset, false_span + 1)) {
      omarchy::unsupported("Select index span", out);
    }
  }
  std::array<omarchy::ComputeBinding, 5> bindings{
      binding(*condition_ptr),
      binding(*truthy_ptr),
      binding(*falsy_ptr),
      binding(out),
      binding(axis_metadata ? *axis_metadata : out)};
  // The kernel processes one output word (four elements) per thread
  // with no grid-stride loop, so very large outputs dispatch in
  // back-to-back offset chunks.
  constexpr uint64_t kMaxWordsPerDispatch =
      static_cast<uint64_t>(omarchy::kMaxComputeGroupCountX) *
      omarchy::kComputeThreadsPerGroup;
  uint32_t lhs_base = params.lhs_offset;
  uint32_t rhs_base = params.rhs_offset;
  uint32_t aux_base = params.aux_offset;
  uint32_t output_base = params.output_offset;
  uint64_t words_done = 0;
  while (words_done < word_count) {
    uint32_t chunk_words = static_cast<uint32_t>(
        std::min<uint64_t>(word_count - words_done, kMaxWordsPerDispatch));
    uint64_t chunk_elements = 4ull * words_done;
    // Condition and output stay flat across chunks. In the flat
    // transport the true operand advances linearly with them and the
    // false operand does too whenever it covers the full count; a
    // scalar false operand reads one absolute element. In the general
    // transport the strided operands were span-checked once above, and
    // the shader receives the chunk's first output element in matrix_m.
    if (!omarchy::compute_index_span_fits(
            lhs_base + chunk_elements, 4ull * chunk_words) ||
        !omarchy::compute_index_span_fits(
            output_base + chunk_elements, 4ull * chunk_words) ||
        (!general &&
            (!omarchy::compute_index_span_fits(
                 rhs_base + chunk_elements, 4ull * chunk_words) ||
             (params.aux_size == count &&
              !omarchy::compute_index_span_fits(
                  aux_base + chunk_elements, 4ull * chunk_words))))) {
      omarchy::unsupported("Select index span", out);
    }
    params.lhs_offset = checked_u32(lhs_base + chunk_elements, name(), out);
    params.output_offset =
        checked_u32(output_base + chunk_elements, name(), out);
    if (general) {
      params.matrix_m = checked_u32(chunk_elements, name(), out);
    } else {
      params.rhs_offset = checked_u32(rhs_base + chunk_elements, name(), out);
      if (params.aux_size == count) {
        params.aux_offset =
            checked_u32(aux_base + chunk_elements, name(), out);
      }
    }
    encoder.dispatch_compute(
        kernel,
        bindings,
        params,
        omarchy::compute_dispatch_group_count(chunk_words));
    words_done += chunk_words;
  }
}

OMARCHY_UNARY(Sigmoid, SigmoidOperation)
// Sign keeps the upstream three-way rule (-1, 0, 1 by comparison with
// zero, NaN mapping to 0) for float dtypes, the complex64 rule z/|z|
// with the zero element mapping to itself, and the integer rule
// (unsigned 0/1) through the integer kernel. Everything else keeps the
// named float-dtype rejection from dispatch_elementwise.
void Sign::eval_gpu(const std::vector<array>& inputs, array& out) {
  if (out.dtype() == complex64) {
    dispatch_complex(
        name(), ComplexSign, inputs, out, out.primitive().stream());
    return;
  }
  if (inputs.at(0).dtype() == bool_ && out.dtype() == bool_) {
    // Upstream sign on bool is x != 0, which is the value itself for
    // the 0/1 byte lanes.
    dispatch_int_elementwise(name(), IntSignOperation, inputs, out);
    return;
  }
  if (is_int_elementwise_dtype(out.dtype())) {
    dispatch_int_elementwise(name(), IntSignOperation, inputs, out);
    return;
  }
  dispatch_elementwise(
      name(), SignFloatOperation, inputs, out, out.primitive().stream());
}
void Sin::eval_gpu(const std::vector<array>& inputs, array& out) {
  if (out.dtype() == complex64) {
    // Upstream std::sin(complex) goes through glibc csinf, which is
    // (sin a cosh b, cos a sinh b) for normal-range inputs; the float
    // trig argument gate does not apply to the complex path because
    // the accuracy loss it guards against lives in the float range
    // reduction, not in the complex extension.
    dispatch_complex(
        name(), ComplexSin, inputs, out, out.primitive().stream());
    return;
  }
  trig_argument_gate(name(), inputs, out);
  dispatch_elementwise(
      name(), SinOperation, inputs, out, out.primitive().stream());
}
void Sinh::eval_gpu(const std::vector<array>& inputs, array& out) {
  if (out.dtype() == complex64) {
    dispatch_complex(
        name(), ComplexSinh, inputs, out, out.primitive().stream());
    return;
  }
  dispatch_elementwise(
      name(), SinhOperation, inputs, out, out.primitive().stream());
}
void Softmax::eval_gpu(const std::vector<array>& inputs, array& out) {
  dispatch_softmax(name(), inputs.at(0), out, stream());
}
// Read-modify-write fold of `upd` into the slice window of `out` that
// prepare_slice named. Each window position is written at most once, so
// the fold needs no atomics and stays deterministic. The collapsed
// window rank caps at four entries; negative strides and dtypes beyond
// the float/32-bit-int set keep the named refusal, the same boundary
// the strided-copy engine draws.
void dispatch_slice_update_reduce(
    SliceUpdate::ReduceType reduce_type,
    const array& upd,
    array& out,
    int64_t data_offset,
    const Strides& window_strides) {
  if (out.dtype() != float32 && out.dtype() != float16 &&
      out.dtype() != bfloat16 && out.dtype() != int32 &&
      out.dtype() != uint32) {
    omarchy::unsupported("SliceUpdate reduce dtype", out);
  }
  auto& encoder = omarchy::get_command_encoder(out.primitive().stream());
  const auto& capabilities = encoder.device().capabilities();
  if (out.dtype() == float16 &&
      (!capabilities.shader_float16 ||
       !capabilities.storage_buffer_16bit_access)) {
    omarchy::unsupported("SliceUpdate reduce float16 capability", out);
  }
  if (out.dtype() == bfloat16 &&
      (!capabilities.storage_buffer_16bit_access ||
       !capabilities.shader_int16)) {
    omarchy::unsupported("SliceUpdate reduce bfloat16 capability", out);
  }
  uint32_t operation;
  switch (reduce_type) {
    case SliceUpdate::Sum:
      operation = 0;
      break;
    case SliceUpdate::Prod:
      operation = 1;
      break;
    case SliceUpdate::Max:
      operation = 2;
      break;
    default:
      operation = 3;
      break;
  }
  // A scalar update strides zero across the window; otherwise the
  // update walks its own layout.
  bool upd_scalar = upd.data_size() == 1;
  Strides upd_strides(upd.ndim(), 0);
  if (!upd_scalar) {
    upd_strides = upd.strides();
  }
  auto [collapsed_shape, collapsed_strides] = collapse_contiguous_dims(
      upd.shape(), std::vector<Strides>{upd_strides, window_strides});
  size_t rank = collapsed_shape.size();
  if (rank > 4) {
    omarchy::unsupported("SliceUpdate reduce rank", out);
  }
  uint32_t count = checked_u32(upd.size(), "SliceUpdate reduce", out);
  omarchy::ComputeParams params;
  params.count = count;
  params.operation = operation;
  params.dims = static_cast<uint32_t>(rank);
  params.aux_offset = out.dtype() == int32 ? 1u
      : out.dtype() == uint32              ? 2u
                                           : 0u;
  for (size_t axis = 0; axis < rank; ++axis) {
    params.shape[axis] = checked_u32(
        static_cast<size_t>(collapsed_shape[axis]),
        "SliceUpdate reduce",
        out);
    int64_t in_stride = collapsed_strides[0][axis];
    int64_t out_stride = collapsed_strides[1][axis];
    if (in_stride < 0 || out_stride < 0) {
      omarchy::unsupported("negative stride copy", out);
    }
    params.in_strides[axis] = static_cast<uint32_t>(in_stride);
    params.out_strides[axis] = static_cast<uint32_t>(out_stride);
  }
  params.lhs_offset =
      checked_item_offset(upd, upd.size(), "SliceUpdate reduce", out);
  uint32_t base =
      checked_item_offset(out, out.size(), "SliceUpdate reduce", out);
  if (data_offset < 0) {
    omarchy::unsupported("SliceUpdate reduce window", out);
  }
  uint64_t window_base =
      static_cast<uint64_t>(base) + static_cast<uint64_t>(data_offset);
  if (!omarchy::compute_index_span_fits(window_base, count)) {
    omarchy::unsupported("SliceUpdate reduce index span", out);
  }
  params.output_offset = static_cast<uint32_t>(window_base);
  auto kernel = out.dtype() == int32 || out.dtype() == uint32
      ? omarchy::ComputeKernel::SliceUpdateReduceU32
      : select_float_kernel(
            out.dtype(),
            omarchy::ComputeKernel::SliceUpdateReduceF32,
            omarchy::ComputeKernel::SliceUpdateReduceF16,
            omarchy::ComputeKernel::SliceUpdateReduceBF16);
  std::array<omarchy::ComputeBinding, 2> bindings{
      binding(upd), binding(out)};
  encoder.dispatch_compute(
      kernel,
      bindings,
      params,
      omarchy::compute_dispatch_group_count(count));
}
void SliceUpdate::eval_gpu(const std::vector<array>& inputs, array& out) {
  if (out.size() == 0) {
    return;
  }

  const auto& in = inputs[0];
  const auto& upd = inputs[1];

  if (upd.size() == 0) {
    out.copy_shared_buffer(in);
    return;
  }

  // A plain paste rides the strided-copy engine; the reduce modes run
  // the read-modify-write shader against the copied window.
  auto ctype = in.flags().contiguous && in.size() == in.data_size()
      ? CopyType::Vector
      : CopyType::General;
  copy_gpu(in, out, in.data_size() == 1 ? CopyType::Scalar : ctype, stream());

  auto [data_offset, out_strides] =
      prepare_slice(out, start_indices_, strides_);

  if (reduce_type_ != SliceUpdate::None) {
    dispatch_slice_update_reduce(
        reduce_type_, upd, out, data_offset, out_strides);
    return;
  }

  copy_gpu_inplace(
      upd,
      out,
      upd.shape(),
      upd.strides(),
      out_strides,
      /* i_offset = */ 0,
      /* o_offset = */ data_offset,
      CopyType::GeneralGeneral,
      stream());
}
void Sort::eval_gpu(const std::vector<array>& inputs, array& out) {
  const array& input = inputs.at(0);
  auto& encoder = omarchy::get_command_encoder(out.primitive().stream());
  require_sort_dtype("Sort", input, out, false, encoder);
  dispatch_sort_any_axis("Sort", input, out, state(), false, encoder);
}
void Square::eval_gpu(const std::vector<array>& inputs, array& out) {
  if (out.dtype() == complex64) {
    // z*z through the complex multiply kernel; the unary dispatch
    // binds the single input to both operand slots.
    dispatch_complex(
        name(), ComplexMultiply, inputs, out, out.primitive().stream());
    return;
  }
  if (out.dtype() == int32 || out.dtype() == uint32) {
    dispatch_int_elementwise(name(), IntSquareOperation, inputs, out);
    return;
  }
  dispatch_elementwise(
      name(), SquareOperation, inputs, out, out.primitive().stream());
}
void Sqrt::eval_gpu(const std::vector<array>& inputs, array& out) {
  if (out.dtype() == complex64) {
    dispatch_complex(
        name(), state() ? ComplexRsqrt : ComplexSqrt, inputs, out, stream());
    return;
  }
  dispatch_elementwise(
      name(), state() ? RsqrtOperation : SqrtOperation, inputs, out, stream());
}
void Subtract::eval_gpu(const std::vector<array>& inputs, array& out) {
  if (out.dtype() == complex64) {
    dispatch_complex(
        name(), ComplexSubtract, inputs, out, out.primitive().stream());
    return;
  }
  if (is_int_elementwise_dtype(out.dtype())) {
    dispatch_int_elementwise(name(), IntSubtractOperation, inputs, out);
    return;
  }
  dispatch_elementwise(
      name(), SubtractOperation, inputs, out, out.primitive().stream());
}
void SVD::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  const array& in = inputs.at(0);
  auto& encoder =
      omarchy::get_command_encoder(outputs.at(0).primitive().stream());
  linalg_require_f32(name(), in, outputs.at(0));
  const int m = in.shape(-2);
  const int n = in.shape(-1);
  const int k = std::min(m, n);
  if (k > 1024) {
    omarchy::unsupported(std::string(name()) + " matrix size", outputs.at(0));
  }
  const uint32_t batch =
      checked_u32(linalg_batch_count(in.shape()), name(), outputs.at(0));
  const bool wide = m < n;
  const bool compute_uv = state();
  // The sweeps always run on an R x C copy with R >= C: tall inputs use
  // A itself, wide inputs use a dense transposed copy.
  Shape work_shape = in.shape();
  if (wide) {
    work_shape[work_shape.size() - 2] = n;
    work_shape[work_shape.size() - 1] = m;
  }
  array work(work_shape, in.dtype(), nullptr, {});
  work.set_data(allocate_omarchy(work.nbytes()));
  encoder.add_temporary(work);
  if (wide) {
    auto strides = in.strides();
    std::swap(strides[strides.size() - 2], strides[strides.size() - 1]);
    copy_gpu_inplace(
        in,
        work,
        // The walk shape must be the transposed (R x C) work shape. With
        // in.shape() here the kernel walks the source as (m, n) over
        // transposed strides and reads n - m rows past the input buffer;
        // those recycled-page values land in work rows >= k and made
        // every wide SVD (and pinv) silently wrong, varying run to run.
        work.shape(),
        strides,
        work.strides(),
        0,
        0,
        CopyType::GeneralGeneral,
        outputs.at(0).primitive().stream());
  } else {
    linalg_copy_dense(in, work, outputs.at(0).primitive().stream());
  }
  omarchy::ComputeParams params;
  params.matrix_m = checked_u32(wide ? n : m, name(), outputs.at(0));
  params.matrix_n = checked_u32(k, name(), outputs.at(0));
  params.matrix_k = checked_u32(k, name(), outputs.at(0));
  params.output_size = batch;
  params.flags = wide ? 1u : 0u;
  if (compute_uv) {
    auto& u = outputs.at(0);
    auto& s = outputs.at(1);
    auto& vt = outputs.at(2);
    u.set_data(allocate_omarchy(u.nbytes()));
    s.set_data(allocate_omarchy(s.nbytes()));
    vt.set_data(allocate_omarchy(vt.nbytes()));
    if (batch == 0) {
      return;
    }
    auto scratch = make_u32_scratch(batch, encoder);
    dispatch_clear_u32(scratch, 0, encoder);
    params.operation = 0u;
    std::array<omarchy::ComputeBinding, 4> sweep_bindings{
        binding(work), binding(u), binding(work), binding(scratch)};
    encoder.dispatch_compute(
        omarchy::ComputeKernel::LinalgSvdF32, sweep_bindings, params, batch);
    linalg_check_status(
        scratch,
        encoder,
        "[SVD::eval_gpu] one-sided Jacobi sweep limit (60) exceeded"
        " without convergence.",
        u);
    params.operation = 2u;
    std::array<omarchy::ComputeBinding, 4> final_bindings{
        binding(work), binding(u), binding(vt), binding(s)};
    encoder.dispatch_compute(
        omarchy::ComputeKernel::LinalgSvdFinalizeF32,
        final_bindings,
        params,
        batch);
  } else {
    auto& s = outputs.at(0);
    s.set_data(allocate_omarchy(s.nbytes()));
    if (batch == 0 || s.size() == 0) {
      return;
    }
    auto scratch = make_u32_scratch(batch, encoder);
    dispatch_clear_u32(scratch, 0, encoder);
    params.operation = 1u;
    std::array<omarchy::ComputeBinding, 4> bindings{
        binding(work), binding(work), binding(s), binding(scratch)};
    encoder.dispatch_compute(
        omarchy::ComputeKernel::LinalgSvdF32, bindings, params, batch);
    linalg_check_status(
        scratch,
        encoder,
        "[SVD::eval_gpu] one-sided Jacobi sweep limit (60) exceeded"
        " without convergence.",
        s);
  }
}
void Tan::eval_gpu(const std::vector<array>& inputs, array& out) {
  if (out.dtype() == complex64) {
    dispatch_complex(
        name(), ComplexTan, inputs, out, out.primitive().stream());
    return;
  }
  dispatch_elementwise(
      name(), TanOperation, inputs, out, out.primitive().stream());
}
void Tanh::eval_gpu(const std::vector<array>& inputs, array& out) {
  if (out.dtype() == complex64) {
    dispatch_complex(
        name(), ComplexTanh, inputs, out, out.primitive().stream());
    return;
  }
  dispatch_elementwise(
      name(), TanhOperation, inputs, out, out.primitive().stream());
}
void Eig::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  const array& in = inputs.at(0);
  auto& encoder =
      omarchy::get_command_encoder(outputs.at(0).primitive().stream());
  linalg_require_f32(name(), in, outputs.at(0));
  const int n = in.shape(-1);
  if (in.shape(-2) != n) {
    omarchy::unsupported(std::string("non-square ") + name(), outputs.at(0));
  }
  if (n > 1024) {
    omarchy::unsupported(std::string(name()) + " matrix size", outputs.at(0));
  }
  const uint32_t batch =
      checked_u32(linalg_batch_count(in.shape()), name(), outputs.at(0));
  auto& values = outputs.at(0);
  values.set_data(allocate_omarchy(values.nbytes()));
  if (values.size() == 0) {
    return;
  }
  array work(in.shape(), in.dtype(), nullptr, {});
  work.set_data(allocate_omarchy(work.nbytes()));
  encoder.add_temporary(work);
  linalg_copy_dense(in, work, outputs.at(0).primitive().stream());
  const bool compute_ev = state();
  if (compute_ev) {
    auto& vectors = outputs.at(1);
    vectors.set_data(allocate_omarchy(vectors.nbytes()));
  }
  // Dense per-matrix accumulation matrix V. The complex64 vectors output
  // spans twice the floats of a dense n*n matrix, so it cannot double as
  // the accumulator; the kernel writes it once in the final interleave.
  // Values-only mode leaves it unused and binds `work` as the dummy.
  array vacc(
      Shape{static_cast<int>(compute_ev ? n * n * batch : 1)},
      float32,
      nullptr,
      {});
  vacc.set_data(allocate_omarchy(vacc.nbytes()));
  if (compute_ev) {
    encoder.add_temporary(vacc);
  }
  auto scratch = make_u32_scratch(std::max<size_t>(batch, 1u), encoder);
  dispatch_clear_u32(scratch, 0, encoder);
  omarchy::ComputeParams params;
  params.operation = compute_ev ? 1u : 0u;
  params.matrix_n = checked_u32(n, name(), values);
  params.output_size = batch;
  // Kernel binding order: 0 work H, 1 vectors output, 2 values output,
  // 3 status, 4 dense V accumulator. Values-only mode binds `work` as
  // the dummy for the unwritten slots.
  std::array<omarchy::ComputeBinding, 5> bindings{
      binding(work),
      compute_ev ? binding(outputs.at(1)) : binding(work),
      binding(values),
      binding(scratch),
      compute_ev ? binding(vacc) : binding(work)};
  encoder.dispatch_compute(
      omarchy::ComputeKernel::LinalgEigF32, bindings, params, batch);
  linalg_check_status(
      scratch,
      encoder,
      "[Eig::eval_gpu] QR iteration limit (40) exceeded without"
      " convergence.",
      values);
}

void Eigh::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  const array& in = inputs.at(0);
  auto& encoder =
      omarchy::get_command_encoder(outputs.at(0).primitive().stream());
  linalg_require_f32(name(), in, outputs.at(0));
  const int n = in.shape(-1);
  if (in.shape(-2) != n) {
    omarchy::unsupported(std::string("non-square ") + name(), outputs.at(0));
  }
  if (n > 1024) {
    omarchy::unsupported(std::string(name()) + " matrix size", outputs.at(0));
  }
  const uint32_t batch =
      checked_u32(linalg_batch_count(in.shape()), name(), outputs.at(0));
  auto [uplo, compute_ev] = state();
  const bool upper = (uplo == "U" || uplo == "upper");
  auto& values = outputs.at(0);
  values.set_data(allocate_omarchy(values.nbytes()));
  if (values.size() == 0) {
    return;
  }
  array work(in.shape(), in.dtype(), nullptr, {});
  work.set_data(allocate_omarchy(work.nbytes()));
  encoder.add_temporary(work);
  linalg_copy_dense(in, work, outputs.at(0).primitive().stream());
  if (compute_ev) {
    auto& vectors = outputs.at(1);
    vectors.set_data(allocate_omarchy(vectors.nbytes()));
  }
  auto scratch = make_u32_scratch(std::max<size_t>(batch, 1u), encoder);
  dispatch_clear_u32(scratch, 0, encoder);
  omarchy::ComputeParams params;
  params.operation = compute_ev ? 1u : 0u;
  params.matrix_n = checked_u32(n, name(), values);
  params.output_size = batch;
  params.flags = upper ? 1u : 0u;
  std::array<omarchy::ComputeBinding, 4> bindings{
      binding(work),
      compute_ev ? binding(outputs.at(1)) : binding(work),
      binding(values),
      binding(scratch)};
  encoder.dispatch_compute(
      omarchy::ComputeKernel::LinalgEighF32, bindings, params, batch);
  linalg_check_status(
      scratch,
      encoder,
      "[Eigh::eval_gpu] Jacobi sweep limit (60) exceeded without"
      " convergence.",
      values);
}

namespace fast {

bool ScaledDotProductAttention::use_fallback(
    const array& q,
    const array& k,
    const array& v,
    bool has_mask,
    bool has_arr_mask,
    bool do_causal,
    bool is_training,
    bool output_logsumexp,
    bool force_fused,
    Stream s) {
  if (force_fused) {
    throw std::invalid_argument(
        "[scaled_dot_product_attention] force_fused=True but no fused "
        "kernel is available in the Omarchy backend.");
  }
  // Training with a logsumexp output needs the VJP, which stays a
  // named rejection, so that one case keeps the composed graph.
  return output_logsumexp;
}

bool ScaledDotProductAttentionVJP::use_fallback(const array& q, Stream s) {
  return true;
}

bool ScaledDotProductAttention::supports_bool_mask() {
  return false;
}

// Shared bytes of the coopmat prefill kernel (s_state 16 KiB + tile
// staging and round trips): the dispatch gate checks the device limit
// before selecting it.
inline constexpr size_t kGdnCoopmatSharedBytes = 24832;

// Gated delta nets (upstream 0.32.3): the fused GatedDeltaDecodeBF16 kernel
// serves the decode shape (T=1, no mask, square heads, bf16 activations,
// f32 state); everything else - prefill token chunks, masks, f16/f32
// activations, ragged head layouts - keeps the composed fallback, which is
// the arithmetic reference this kernel was equivalence-checked against.
bool GatedDeltaUpdate::use_fallback(
    const int Hk,
    const int Dk,
    const int Hv,
    const int Dv,
    const bool has_mask,
    Stream s) {
  // The mask rides into the primitive (scalar per-token validity): the
  // fused kernels apply load+skip, so a mask no longer forces the
  // composed fallback on its own.
  (void)has_mask;
  return Dk != 128 || Dv != 128 || Hk != Hv;
}

void GatedDeltaUpdate::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  const std::string tag = name();
  auto s = stream();
  auto& encoder = omarchy::get_command_encoder(s);
  // Shallow copies (shared buffers): rebound to dense temporaries below
  // when a fused-path input arrives strided.
  array q = inputs.at(0);
  array k = inputs.at(1);
  array v = inputs.at(2);
  // Raw-gates mode: (q, k, v, a, b, A_log, dt_bias, state[, mask]); the
  // gate chain is computed in the decode kernel's prologue.
  // Precomputed-gates mode: (q, k, v, g, beta, state[, mask]).
  const bool raw_gates_mode = raw_gates();
  // Placeholder-initialized (never dereferenced in the unused mode).
  array a_in = array(false);
  array b_in = array(false);
  array A_log = array(false);
  array dt_bias = array(false);
  array g = array(false);
  array beta = array(false);
  array h0 = array(false);
  std::optional<array> mask_storage;
  const array* mask = nullptr;
  bool has_mask = false;
  if (raw_gates_mode) {
    a_in = inputs.at(3);
    b_in = inputs.at(4);
    A_log = inputs.at(5);
    dt_bias = inputs.at(6);
    h0 = inputs.at(7);
    has_mask = inputs.size() > 8;
    if (has_mask) {
      mask_storage = inputs.at(8);
    }
  } else {
    g = inputs.at(3);
    beta = inputs.at(4);
    h0 = inputs.at(5);
    has_mask = inputs.size() > 6;
    if (has_mask) {
      mask_storage = inputs.at(6);
    }
  }
  mask = has_mask ? &*mask_storage : nullptr;

  int B = q.shape(0);
  int T = q.shape(1);
  int Hk = q.shape(2);
  int Dk = q.shape(3);
  int Hv = v.shape(2);
  int Dv = v.shape(3);

  bool fused_ready = B == 1 && Hk == Hv && Dk == 128 && Dv == 128 &&
      q.dtype() == bfloat16 && k.dtype() == bfloat16 &&
      v.dtype() == bfloat16 && h0.dtype() == float32 &&
      outputs.at(0).dtype() == bfloat16 && outputs.at(1).dtype() == float32;
  if (raw_gates_mode) {
    // Raw-gates contract: a/b/A_log/dt_bias bf16 (the Qwen3.8 checkpoint
    // stores dt_bias bf16); decode only (the fast.cpp caller routes
    // T > 1 to the precomputed-gates scan).
    fused_ready = fused_ready && T == 1 && a_in.dtype() == bfloat16 &&
        b_in.dtype() == bfloat16 && A_log.dtype() == bfloat16 &&
        dt_bias.dtype() == bfloat16;
  } else {
    fused_ready = fused_ready && beta.dtype() == bfloat16;
  }
  bool decode_shape = fused_ready && T == 1 &&
      (raw_gates_mode || g.dtype() == bfloat16);
  // compute_g produces float32 gates (exp of f32), so the fused path
  // takes g in f32 or bf16; the shader selects by flag bit2. Raw-gates
  // mode carries no precomputed g.
  if (!raw_gates_mode) {
    bool g_ok = g.dtype() == bfloat16 || g.dtype() == float32;
    fused_ready = fused_ready && g_ok;
  }
  // Prefill scan: same contract as the decode kernel extended over the
  // token axis (one workgroup per head, sequential scan; state rides hf).
  // g is [B,T,H] (scalar decay) or [B,T,H,Dk] (per-channel decay; the
  // kernel takes a push-constant flag).
  bool prefill_shape = !raw_gates_mode && fused_ready && T > 1 &&
      (g.ndim() == 3 || g.ndim() == 4) &&
      outputs.at(0).shape() == q.shape();
  if (decode_shape || prefill_shape) {
    // Strided inputs (in-model callers pass v sliced from a fused qkv
    // projection) are materialized below into dense temporaries recorded
    // on the encoder - a plain device copy, no host sync. Requiring
    // row_contiguous here instead silently routed every in-model call to
    // the per-token composed fallback plus a full encoder synchronize.
  } else {
    fused_ready = false;
  }
  if (!fused_ready) {
    if (getenv("GDN_FALLBACK_DEBUG") != nullptr) {
      std::fprintf(
          stderr,
          "GDN-FALLBACK T=%d q=%d/%d k=%d v=%d beta=%d h0=%d "
          "gsize=%d out=%d rc=%d/%d/%d/%d/%d/%d/%d outshape_eq=%d\n",
          T,
          int(q.dtype().size()),
          Dk,
          int(k.dtype().size()),
          int(v.dtype().size()),
          int(beta.dtype().size()),
          int(h0.dtype().size()),
          int(g.ndim()),
          int(outputs.at(0).dtype().size()),
          int(q.flags().row_contiguous),
          int(k.flags().row_contiguous),
          int(v.flags().row_contiguous),
          int(g.flags().row_contiguous),
          int(beta.flags().row_contiguous),
          int(h0.flags().row_contiguous),
          int(has_mask ? mask->flags().row_contiguous : 1),
          int(outputs.at(0).shape() == q.shape()));
    }
    auto result = fallback_(inputs);
    // Record-only settle: fallback nodes join the open batch, and the
    // synchronize below submits and orders the buffer handoff (the RoPE
    // fallback pattern).
    settle(result);
    encoder.synchronize("gated_delta_fallback");
    outputs.at(0).copy_shared_buffer(result.at(0));
    outputs.at(1).copy_shared_buffer(result.at(1));
    return;
  }

  array& out = outputs.at(0);
  array& hf = outputs.at(1);

  // Materialize any strided fused-path input into a same-shape dense
  // temporary (device copy recorded on this encoder, kept alive via
  // add_temporary) and rebind the handles above.
  if (fused_ready) {
    bool any_strided = false;
    for (const auto& x : inputs) {
      // Non-row-contiguous inputs need a dense copy; so does any input
      // with a nonzero storage offset - the strided T-slice views mlx-lm
      // feeds the T=1 decode shape are still flagged row_contiguous, and
      // the decode kernel reads garbage (NaN state) for offset != 0.
      any_strided =
          any_strided || !x.flags().row_contiguous || x.offset() != 0;
    }
    if (any_strided) {
      std::vector<array> dense;
      dense.reserve(inputs.size());
      for (const auto& x : inputs) {
        if (x.flags().row_contiguous) {
          dense.push_back(x);
        } else {
          dense.push_back(contiguous_copy_gpu(x, s));
          encoder.add_temporary(dense.back());
        }
      }
      q = dense[0];
      k = dense[1];
      v = dense[2];
      if (raw_gates_mode) {
        a_in = dense[3];
        b_in = dense[4];
        A_log = dense[5];
        dt_bias = dense[6];
        h0 = dense[7];
        if (has_mask) {
          mask_storage = dense[8];
        }
      } else {
        g = dense[3];
        beta = dense[4];
        h0 = dense[5];
        if (has_mask) {
          mask_storage = dense[6];
        }
      }
    }
  }

  // Decode kernel and prefill scan: LANES=4 threads per Dv row, 128-thread
  // workgroups over 32 rows, so a head takes Dv / 32 = 4 workgroups
  // (shader constants).
  constexpr uint32_t kGdnWorkgroupsPerHead = 4;
  // Decode (T=1): the original single-token kernel, unchanged.
  if (decode_shape) {
    out.set_data(allocate_omarchy(out.nbytes()));
    hf.set_data(allocate_omarchy(hf.nbytes()));
    omarchy::ComputeParams params;
    params.count = Dv;
    params.lhs_size = checked_u32(q.data_size(), tag, out);
    params.rhs_size = checked_u32(h0.data_size(), tag, out);
    params.output_size = checked_u32(hf.data_size(), tag, out);
    params.matrix_m = checked_u32(Dk, tag, out);
    params.matrix_n = checked_u32(Dv, tag, out);
    params.matrix_k = checked_u32(Hv, tag, out);
    params.lhs_offset = checked_item_offset(q, q.size(), tag, out);
    params.rhs_offset = checked_item_offset(k, k.size(), tag, out);
    params.aux_size = checked_item_offset(v, v.size(), tag, out);
    params.output_offset = checked_item_offset(out, out.size(), tag, out);
    params.shape[1] = checked_item_offset(h0, h0.size(), tag, out);
    params.shape[2] = checked_item_offset(hf, hf.size(), tag, out);
    params.dims = static_cast<uint32_t>(T);
    std::array<omarchy::ComputeBinding, 10> bindings{
        binding(q),    // 0 QBuf
        binding(k),    // 1 KBuf
        binding(v),    // 2 VBuf
        binding(q),    // 3 GBuf/ABuf - overridden below (g | a)
        binding(k),    // 4 BBuf - overridden below (beta | b)
        binding(h0),   // 5 SIn
        binding(out),  // 6 YBuf
        binding(hf),   // 7 SOut
        binding(q),    // 8 ALogBuf - overridden in raw-gates mode
        binding(k)};   // 9 DtBuf - overridden in raw-gates mode
    if (raw_gates_mode) {
      // Shader slots: 3 = a, 4 = b, 5 = state in, 6 = out, 7 = state out,
      // 8 = A_log, 9 = dt_bias. A_log/dt_bias item offsets ride
      // in_strides[0..1]; flag bit 8 selects the raw-gates prologue.
      params.aux_offset = checked_item_offset(b_in, b_in.size(), tag, out);
      params.shape[0] = checked_item_offset(a_in, a_in.size(), tag, out);
      params.in_strides[0] = checked_item_offset(A_log, A_log.size(), tag, out);
      params.in_strides[1] =
          checked_item_offset(dt_bias, dt_bias.size(), tag, out);
      params.flags |= 8u;
      bindings[3] = binding(a_in);
      bindings[4] = binding(b_in);
      bindings[8] = binding(A_log);
      bindings[9] = binding(dt_bias);
    } else {
      params.aux_offset = checked_item_offset(beta, beta.size(), tag, out);
      params.shape[0] = checked_item_offset(g, g.size(), tag, out);
      bindings[3] = binding(g);
      bindings[4] = binding(beta);
    }
    encoder.dispatch_compute(
        omarchy::ComputeKernel::GatedDeltaDecodeBF16,
        bindings,
        params,
        static_cast<uint32_t>(Hv),
        1,
        1);
    return;
  }

  out.set_data(allocate_omarchy(out.nbytes()));
  hf.set_data(allocate_omarchy(hf.nbytes()));

  // Single-pass chunked cooperative-matrix scan (Metal
  // gated_delta_fused_chunk shape at C=8): 128-thread workgroups over
  // (head, Dv/32 slice), four simdgroups each holding an 8-row state
  // slice as sixteen 8x8 f32 coopmat tiles, so the state never reaches
  // scratch. Gated to the square bf16 maskless scalar-g shape on coopmat
  // devices with T >= kGdnCoopmatMinTokens: the chunk walk has fixed
  // per-chunk cost that loses to the scan on short prefills (ttft
  // prompts are ~12 tokens; 512-token prefill wins ~2.8x). Metal makes
  // the same trade with its GATED_DELTA_THRESH default. Everything else
  // keeps the two-pass scan below. The kill switch (and A/B lever) is
  // MLX_OMARCHY_NO_COOPMAT_GDN=1.
  constexpr uint32_t kGdnCoopmatMinTokens = 64;
  static const bool coopmat_gdn_disabled =
      omarchy::env_flag("MLX_OMARCHY_NO_COOPMAT_GDN");
  const auto& gdn_caps = encoder.device().capabilities();
  const bool gdn_coopmat = fused_ready && T >= kGdnCoopmatMinTokens &&
      !has_mask && g.ndim() == 3 && !coopmat_gdn_disabled &&
      gdn_caps.cooperative_matrix_f32_8 && gdn_caps.subgroup_size == 32u &&
      kGdnCoopmatSharedBytes <= gdn_caps.max_compute_shared_memory_size;
  if (gdn_coopmat) {
    omarchy::ComputeParams params;
    params.count = Dv;
    params.lhs_size = checked_u32(q.data_size(), tag, out);
    params.rhs_size = checked_u32(h0.data_size(), tag, out);
    params.output_size = checked_u32(hf.data_size(), tag, out);
    params.matrix_m = checked_u32(Dk, tag, out);
    params.matrix_n = checked_u32(Dv, tag, out);
    params.matrix_k = checked_u32(Hv, tag, out);
    params.lhs_offset = checked_item_offset(q, q.size(), tag, out);
    params.rhs_offset = checked_item_offset(k, k.size(), tag, out);
    params.aux_size = checked_item_offset(v, v.size(), tag, out);
    params.aux_offset = checked_item_offset(beta, beta.size(), tag, out);
    params.output_offset = checked_item_offset(out, out.size(), tag, out);
    params.shape[0] = checked_item_offset(g, g.size(), tag, out);
    params.shape[1] = checked_item_offset(h0, h0.size(), tag, out);
    params.shape[2] = checked_item_offset(hf, hf.size(), tag, out);
    params.dims = static_cast<uint32_t>(T);
    // Scalar g only: [B=1, T, Hv] (ndim gate; B==1 comes from fused_ready).
    // Bit2 selects the f32 gate load.
    params.flags = (g.dtype() == float32 ? 4u : 0u);
    std::array<omarchy::ComputeBinding, 11> bindings{
        binding(q),      // 0 QBuf
        binding(k),      // 1 KBuf
        binding(v),      // 2 VBuf
        binding(g),      // 3 GBuf
        binding(beta),   // 4 BBuf
        binding(h0),     // 5 SIn
        binding(out),    // 6 YBuf
        binding(hf),     // 7 SOut
        binding(out),    // 8 MBuf - unused (maskless gate)
        binding(g),      // 9 GBufF - unused when g is bf16
        binding(out)};   // 10 Snap - unused (single pass)
    encoder.dispatch_compute(
        omarchy::ComputeKernel::GatedDeltaPrefillCoopmatBF16,
        bindings,
        params,
        static_cast<uint32_t>(Hv),
        Dv / 32,
        1);
    return;
  }

  // Chunked two-pass scan: pass 0 (one workgroup per head) computes the
  // per-chunk initial states into a snapshot scratch; pass 1 (heads x
  // chunks) replays the tokens chunk-parallel and writes the outputs.
  // Chunk-parallel arithmetic is identical token-by-token to the single
  // workgroup scan, so outputs are bit-identical to it.
  constexpr uint32_t kGdnChunk = 64;
  uint32_t chunks = (static_cast<uint32_t>(T) + kGdnChunk - 1) / kGdnChunk;
  size_t snap_elems = chunks > 1
      ? static_cast<size_t>(Hv) * Dv * Dk * (chunks - 1)
      : 1;
  array snapshots(Shape{static_cast<int>(snap_elems)}, float32, nullptr, {});
  array::Flags snap_flags;
  snap_flags.contiguous = true;
  snap_flags.row_contiguous = true;
  snapshots.set_data(
      allocate_omarchy(snapshots.nbytes()),
      snapshots.size(),
      Strides{1},
      snap_flags,
      0);
  if (chunks > 1) {
    encoder.add_temporary(snapshots);
  }

  omarchy::ComputeParams params;
  params.count = Dv;
  params.lhs_size = checked_u32(q.data_size(), tag, out);
  params.rhs_size = checked_u32(h0.data_size(), tag, out);
  params.output_size = checked_u32(hf.data_size(), tag, out);
  params.matrix_m = checked_u32(Dk, tag, out);
  params.matrix_n = checked_u32(Dv, tag, out);
  params.matrix_k = checked_u32(Hv, tag, out);
  // Item offsets into each bound buffer (views split from wider tensors
  // keep their base offset; the shader adds these to every index).
  params.lhs_offset = checked_item_offset(q, q.size(), tag, out);
  params.rhs_offset = checked_item_offset(k, k.size(), tag, out);
  params.aux_size = checked_item_offset(v, v.size(), tag, out);
  params.aux_offset = checked_item_offset(beta, beta.size(), tag, out);
  params.output_offset = checked_item_offset(out, out.size(), tag, out);
  params.shape[0] = checked_item_offset(g, g.size(), tag, out);
  params.shape[1] = checked_item_offset(h0, h0.size(), tag, out);
  params.shape[2] = checked_item_offset(hf, hf.size(), tag, out);
  params.dims = static_cast<uint32_t>(T);
  params.lhs_gap = kGdnChunk;
  params.rhs_gap = chunks;
  uint32_t g_flags = (g.ndim() == 4 ? 1u : 0u) | (g.dtype() == float32 ? 4u : 0u);
  if (has_mask) {
    params.shape[3] =
        checked_item_offset(*mask, mask->size(), tag, out);
    g_flags |= 2u;
  }
  omarchy::ComputeBinding snapshot_binding =
      chunks > 1 ? binding(snapshots) : binding(out);
  std::array<omarchy::ComputeBinding, 11> bindings{
      binding(q),
      binding(k),
      binding(v),
      binding(g),
      binding(beta),
      binding(h0),
      binding(out),
      binding(hf),
      has_mask ? binding(*mask) : binding(out),
      binding(g),
      snapshot_binding};
  // Pass 0: prefix scan, snapshots at chunk boundaries. A single chunk
  // has no boundary to snapshot, so pass 1 restores from h0 directly and
  // pass 0 is skipped (it wrote nothing).
  if (chunks > 1) {
    params.flags = g_flags;
    encoder.dispatch_compute(
        omarchy::ComputeKernel::GatedDeltaPrefillBF16,
        bindings,
        params,
        static_cast<uint32_t>(Hv) * kGdnWorkgroupsPerHead,
        1,
        1);
  }
  // Pass 1: chunk-parallel output replay.
  params.flags = g_flags | 8u;
  encoder.dispatch_compute(
      omarchy::ComputeKernel::GatedDeltaPrefillBF16,
      bindings,
      params,
      static_cast<uint32_t>(Hv) * kGdnWorkgroupsPerHead,
      chunks,
      1);
}

bool GreedyQuantizedArgmax::use_fallback(Stream s) {
  return s.device == Device::cpu;
}

// Greedy-argmax head, shaders/qmm_vec.comp QMM_VEC_GREEDY stages 0-7.
// Its exact stage IS the QmmVecQ4WordSubgroupBF16 column, so the route
// holds only where QuantizedMatmul::eval_gpu picks that kernel for the
// single decode row; elsewhere the composed fallback answers.
void GreedyQuantizedArgmax::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  const std::string tag = name();
  auto s = stream();
  auto& encoder = omarchy::get_command_encoder(s);
  const auto& caps = encoder.device().capabilities();
  bool subgroup_ready = caps.subgroup_size == 32u &&
      (caps.subgroup_operations & VK_SUBGROUP_FEATURE_ARITHMETIC_BIT) != 0;
  omarchy::capsim::require_backed(
      encoder.device(),
      caps,
      subgroup_ready,
      "QmmVec*Subgroup*",
      "subgroup_size==32+subgroup_ops_mask[ARITHMETIC]",
      (encoder.device().hardware_capabilities().subgroup_size == 32u &&
       (encoder.device().hardware_capabilities().subgroup_operations &
        VK_SUBGROUP_FEATURE_ARITHMETIC_BIT) != 0));
  const array& w = inputs.at(1);
  uint32_t n = static_cast<uint32_t>(w.shape(0));
  if (!subgroup_ready || !omarchy::q4_word_enabled() ||
      !caps.storage_buffer_16bit_access || !caps.shader_int16) {
    auto result = fallback_(inputs);
    settle(result);
    encoder.synchronize("greedy_argmax_fallback");
    outputs.at(0).copy_shared_buffer(result.at(0));
    outputs.at(1).copy_shared_buffer(result.at(1));
    return;
  }
  // MLX_OMARCHY_GREEDY_PRUNE_TEST (qualification only): "full" forces the
  // full exact path, "keep" keeps every row as a survivor.
  static const uint32_t test_flags = [] {
    const char* env = std::getenv("MLX_OMARCHY_GREEDY_PRUNE_TEST");
    if (env == nullptr) {
      return 0u;
    }
    return std::strcmp(env, "full") == 0 ? 1u
        : std::strcmp(env, "keep") == 0  ? 2u
                                         : 0u;
  }();
  const array& x = inputs.at(0);
  std::optional<array> x_temp;
  std::optional<array> w_temp;
  std::optional<array> scales_temp;
  std::optional<array> biases_temp;
  std::optional<array> plane_temp;
  // The Q4 word column reads x as uvec4: a 16-byte-aligned dense row.
  const array& x_d = ensure_dense(
      x,
      x.flags().row_contiguous && x.offset() % (8 * x.itemsize()) == 0,
      x_temp,
      encoder,
      s);
  const array& w_d =
      ensure_dense(w, w.flags().row_contiguous, w_temp, encoder, s);
  const array& scales_d = ensure_dense(
      inputs.at(2), inputs.at(2).flags().row_contiguous, scales_temp,
      encoder, s);
  const array& biases_d = ensure_dense(
      inputs.at(3), inputs.at(3).flags().row_contiguous, biases_temp,
      encoder, s);
  const array& plane_d = ensure_dense(
      inputs.at(4), inputs.at(4).flags().row_contiguous, plane_temp,
      encoder, s);
  array& token = outputs.at(0);
  array& stats = outputs.at(1);
  token.set_data(allocate_omarchy(token.nbytes()));
  stats.set_data(allocate_omarchy(stats.nbytes()));
  auto scratch = [&](Dtype dtype, int size) {
    array a({size}, dtype, nullptr, {});
    a.set_data(allocate_omarchy(a.nbytes()));
    encoder.add_temporary(a);
    return a;
  };
  array state = scratch(uint32, 128);
  array bounds = scratch(float32, static_cast<int>(n));
  array survivors = scratch(uint32, static_cast<int>(n));
  array row_values = scratch(float32, static_cast<int>(n));
  omarchy::ComputeParams params;
  params.matrix_n = n;
  params.matrix_k = static_cast<uint32_t>(x.shape(-1));
  params.lhs_offset = checked_item_offset(x_d, x_d.size(), tag, token);
  params.rhs_offset = checked_item_offset(w_d, w_d.size(), tag, token);
  params.aux_offset =
      checked_item_offset(scales_d, scales_d.size(), tag, token);
  params.aux_size = checked_item_offset(biases_d, biases_d.size(), tag, token);
  params.rhs_size = checked_item_offset(plane_d, plane_d.size(), tag, token);
  params.flags = test_flags;
  std::array<omarchy::ComputeBinding, 11> bindings{
      binding(x_d),
      binding(w_d),
      binding(scales_d),
      binding(biases_d),
      binding(plane_d),
      binding(token),
      binding(state),
      binding(bounds),
      binding(survivors),
      binding(row_values),
      binding(stats)};
  // Stage group counts: prologue 1; bounds 128 rows per workgroup;
  // candidates and survivors 64 x 8 columns in flight; compact 256 rows
  // per workgroup; select 1; full rows 256 x 8 in flight (flagged tokens
  // only); full select 1.
  const std::array<uint32_t, 8> groups{
      1u, (n + 127u) / 128u, 64u, (n + 255u) / 256u, 64u, 1u, 256u, 1u};
  for (uint32_t stage = 0; stage < groups.size(); ++stage) {
    params.operation = stage;
    encoder.dispatch_compute(
        omarchy::ComputeKernel::QmmVecGreedyBF16,
        bindings,
        params,
        groups[stage],
        1u,
        1u);
  }
}


namespace {

// Wave 9: fused fast-op kernels. Every kernel mirrors the exact fallback
// algebra in mlx/fast.cpp; per-row statistics reduce in shared memory with
// float32 arithmetic for every storage dtype.

void require_norm_input(
    const std::string& tag,
    const array& x,
    array& out,
    omarchy::CommandEncoder& encoder) {
  require_float_dtype(tag, x, out, encoder);
  if (!x.flags().row_contiguous) {
    omarchy::unsupported("non-contiguous " + tag, out);
  }
}

void require_norm_parameter(
    const std::string& tag,
    const array& parameter,
    size_t row_length,
    array& out) {
  if (!parameter.flags().row_contiguous) {
    omarchy::unsupported("non-contiguous " + tag, out);
  }
  if (
      parameter.dtype() != out.dtype() ||
      !(parameter.size() == 1 || parameter.shape(-1) == row_length)) {
    omarchy::unsupported(tag + " parameter shape", out);
  }
}

// A weightless VJP writes a scalar zero, the upstream zeros_like(w) form.
void zero_fill(array& out) {
  out.set_data(allocate_omarchy(out.nbytes()));
  if (out.size() != 0) {
    std::memset(out.data<uint8_t>(), 0, out.nbytes());
  }
}

omarchy::ComputeParams norm_params(
    const array& x,
    size_t row_length,
    float eps,
    const std::string& tag,
    array& out) {
  omarchy::ComputeParams params;
  params.count = checked_u32(x.size(), tag, out);
  params.reduce_size = checked_u32(row_length, tag, out);
  params.output_size = checked_u32(x.size() / row_length, tag, out);
  params.lhs_offset = checked_item_offset(x, x.size(), tag, out);
  params.alpha = eps;
  return params;
}

} // namespace
bool CrossEntropy::use_fallback(Stream s) {
  return false;
}

void CrossEntropy::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  const std::string tag = name();
  array& out = outputs.at(0);
  const array& in_x = inputs.at(0);
  const array& in_y = inputs.at(1);
  auto s = stream();
  auto& encoder = omarchy::get_command_encoder(s);
  std::optional<array> x_temp;
  std::optional<array> y_temp;
  const array& x =
      ensure_dense(in_x, in_x.flags().row_contiguous, x_temp, encoder, s);
  const array& y =
      ensure_dense(in_y, in_y.flags().row_contiguous, y_temp, encoder, s);
  require_float_dtype(tag, x, x, encoder);
  if (out.dtype() != float32) {
    omarchy::unsupported(tag + " output dtype", out);
  }
  if (y.dtype() != int32) {
    omarchy::unsupported(tag + " targets dtype", out);
  }
  size_t row_length = x.shape(-1);
  out.set_data(allocate_omarchy(out.nbytes()));
  if (out.size() == 0) {
    return;
  }
  // loss = lse - score per row, one fused row kernel; the output is always
  // float32, matching the upstream astype(loss, float32).
  auto params = norm_params(x, row_length, 0.0f, tag, out);
  params.rhs_offset = checked_item_offset(y, y.size(), tag, out);
  params.output_offset = checked_item_offset(out, out.size(), tag, out);
  std::array<omarchy::ComputeBinding, 3> bindings{
      binding(x), binding(y), binding(out)};
  auto kernel = select_float_kernel(
      x.dtype(),
      omarchy::ComputeKernel::CrossEntropyF32,
      omarchy::ComputeKernel::CrossEntropyF16,
      omarchy::ComputeKernel::CrossEntropyBF16);
  encoder.dispatch_compute(
      kernel,
      bindings,
      params,
      std::min(params.output_size, omarchy::kMaxComputeGroupCountX));
}

bool RMSNorm::use_fallback(Stream s) {
  return false;
}
void RMSNorm::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  const std::string tag = name();
  array& out = outputs.at(0);
  const array& in_x = inputs.at(0);
  const array& in_w = inputs.at(1);
  auto s = stream();
  auto& encoder = omarchy::get_command_encoder(s);
  std::optional<array> x_temp;
  std::optional<array> w_temp;
  const array& x =
      ensure_dense(in_x, in_x.flags().row_contiguous, x_temp, encoder, s);
  const array& w =
      ensure_dense(in_w, in_w.flags().row_contiguous, w_temp, encoder, s);
  require_norm_input(tag, x, out, encoder);
  require_norm_parameter(tag, w, x.shape(-1), out);
  size_t row_length = x.shape(-1);
  out.set_data(allocate_omarchy(out.nbytes()));
  if (out.size() == 0) {
    return;
  }
  auto params = norm_params(x, row_length, eps_, tag, out);
  params.rhs_offset = checked_item_offset(w, w.size(), tag, out);
  params.output_offset = checked_item_offset(out, out.size(), tag, out);
  params.lhs_size = checked_u32(w.size(), tag, out);
  std::array<omarchy::ComputeBinding, 4> bindings{
      binding(x), binding(w), binding(w), binding(out)};
  omarchy::ComputeKernel kernel;
  if (out.dtype() == float16 && omarchy::fused_trio_enabled()) {
    // Trio pipeline: the near-copied RMS row body on the shared decode
    // pipeline (params.matrix_n 0 is the norm mode).
    kernel = omarchy::ComputeKernel::FastTrioNormF16;
  } else {
    kernel = select_float_kernel(
        out.dtype(),
        omarchy::ComputeKernel::FastRmsNormF32,
        omarchy::ComputeKernel::FastRmsNormF16,
        omarchy::ComputeKernel::FastRmsNormBF16);
  }
  encoder.dispatch_compute(
      kernel,
      bindings,
      params,
      std::min(params.output_size, omarchy::kMaxComputeGroupCountX));
}

bool RMSNormGated::use_fallback(Stream s) {
  return false;
}

// Fused GDN decode chain epilogue: rms_norm + silu(gate)*normed in one
// dispatch (mode 0). Bit-exact to the composed
// FastRmsNormBF16 -> CastBF16F32 x2 -> FusedChainF32(sigmoid,mul,mul)
// -> CastF32BF16 sequence because every intermediate the composed path
// rounds to bf16 is rounded identically here; see fast_norm_gated.comp.
void RMSNormGated::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  const std::string tag = name();
  array& out = outputs.at(0);
  const array& in_x = inputs.at(0);
  const array& in_gate = inputs.at(1);
  const array& in_w = inputs.at(2);
  auto s = stream();
  auto& encoder = omarchy::get_command_encoder(s);
  std::optional<array> x_temp;
  std::optional<array> gate_temp;
  std::optional<array> w_temp;
  const array& x =
      ensure_dense(in_x, in_x.flags().row_contiguous, x_temp, encoder, s);
  const array& gate = ensure_dense(
      in_gate, in_gate.flags().row_contiguous, gate_temp, encoder, s);
  const array& w =
      ensure_dense(in_w, in_w.flags().row_contiguous, w_temp, encoder, s);
  require_norm_input(tag, x, out, encoder);
  require_norm_parameter(tag, w, x.shape(-1), out);
  if (out.dtype() != bfloat16) {
    omarchy::unsupported(tag + " output dtype", out);
  }
  if (gate.shape() != x.shape() || gate.dtype() != x.dtype()) {
    omarchy::unsupported(tag + " gate shape/dtype", out);
  }
  size_t row_length = x.shape(-1);
  out.set_data(allocate_omarchy(out.nbytes()));
  if (out.size() == 0) {
    return;
  }
  auto params = norm_params(x, row_length, eps_, tag, out);
  params.operation = 0u;
  params.rhs_offset = checked_item_offset(w, w.size(), tag, out);
  params.output_offset = checked_item_offset(out, out.size(), tag, out);
  params.lhs_size = checked_u32(w.size(), tag, out);
  params.aux_offset = checked_item_offset(gate, gate.size(), tag, out);
  std::array<omarchy::ComputeBinding, 4> bindings{
      binding(x), binding(w), binding(gate), binding(out)};
  encoder.dispatch_compute(
      omarchy::ComputeKernel::FastNormGatedBF16,
      bindings,
      params,
      std::min(params.output_size, omarchy::kMaxComputeGroupCountX));
}

bool RMSNormScaled::use_fallback(Stream s) {
  return false;
}

// Fused rms_norm + scalar multiply (mode 1): replaces
// FastRmsNormBF16 + ElementwiseBF16(mul) with the bf16-rounded scalar
// the graph's promote cast materializes; the shader re-rounds
// params.beta with the same RNE, so no scalar buffer is bound.
void RMSNormScaled::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  const std::string tag = name();
  array& out = outputs.at(0);
  const array& in_x = inputs.at(0);
  const array& in_w = inputs.at(1);
  auto s = stream();
  auto& encoder = omarchy::get_command_encoder(s);
  std::optional<array> x_temp;
  std::optional<array> w_temp;
  const array& x =
      ensure_dense(in_x, in_x.flags().row_contiguous, x_temp, encoder, s);
  const array& w =
      ensure_dense(in_w, in_w.flags().row_contiguous, w_temp, encoder, s);
  require_norm_input(tag, x, out, encoder);
  require_norm_parameter(tag, w, x.shape(-1), out);
  if (out.dtype() != bfloat16) {
    omarchy::unsupported(tag + " output dtype", out);
  }
  size_t row_length = x.shape(-1);
  out.set_data(allocate_omarchy(out.nbytes()));
  if (out.size() == 0) {
    return;
  }
  auto params = norm_params(x, row_length, eps_, tag, out);
  params.operation = 1u;
  params.beta = scale_;
  params.rhs_offset = checked_item_offset(w, w.size(), tag, out);
  params.output_offset = checked_item_offset(out, out.size(), tag, out);
  params.lhs_size = checked_u32(w.size(), tag, out);
  std::array<omarchy::ComputeBinding, 4> bindings{
      binding(x), binding(w), binding(out), binding(out)};
  encoder.dispatch_compute(
      omarchy::ComputeKernel::FastNormGatedBF16,
      bindings,
      params,
      std::min(params.output_size, omarchy::kMaxComputeGroupCountX));
}

bool GdnConvUpdate::use_fallback(Stream s) {
  return false;
}

// GDN decode conv: the state++x concatenation folds into the conv read.
// Bit-exact to CopyGeneral x2 (Concatenate) -> ConvBF16 because the
// concatenation is value-transparent and the tap accumulation is the
// conv.comp body verbatim; the carry-out state moves as raw bits. See
// gdn_conv_decode.comp.
void GdnConvUpdate::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  const std::string tag = name();
  std::fprintf(stderr, "CCTRACE backend: eval_gpu entry");
  auto s = stream();
  auto& encoder = omarchy::get_command_encoder(s);
  array in_state = inputs.at(0);
  array in_x = inputs.at(1);
  array in_w = inputs.at(2);
  // Materialize any strided/offset input into a dense temporary (same
  // trap as the GDN decode kernel: nonzero storage offsets read
  // garbage through flat bindings).
  bool any_strided = false;
  for (const auto& x : inputs) {
    any_strided = any_strided || !x.flags().row_contiguous || x.offset() != 0;
  }
  if (any_strided) {
    std::fprintf(stderr, "CCTRACE backend: strided materialize");

    std::vector<array> dense;
    dense.reserve(inputs.size());
    for (const auto& x : inputs) {
      if (x.flags().row_contiguous) {
        dense.push_back(x);
      } else {
        dense.push_back(contiguous_copy_gpu(x, s));
        encoder.add_temporary(dense.back());
      }
    }
    in_state = dense[0];
    in_x = dense[1];
    in_w = dense[2];
  }
  std::fprintf(stderr, "CCTRACE backend: fused dispatch state_off=%d x_off=%d\n",
      (int)in_state.offset(), (int)in_x.offset());
  array& out = outputs.at(0);
  array& state_out = outputs.at(1);
  if (in_x.shape(1) != 1 || in_state.shape(1) != in_w.shape(1) - 1 ||
      in_state.shape(0) != in_x.shape(0) ||
      in_state.shape(2) != in_x.shape(2) || in_w.shape(0) != in_x.shape(2) ||
      in_w.shape(2) != 1 || in_state.dtype() != bfloat16 ||
      in_x.dtype() != bfloat16 || in_w.dtype() != bfloat16 ||
      out.dtype() != bfloat16 || state_out.dtype() != bfloat16) {
    omarchy::unsupported(tag + " shape/dtype", out);
  }
  const int B = in_x.shape(0);
  const int C = in_x.shape(2);
  // MLX conv weight layout: [C_out, K, C_in/groups].
  const int K = in_w.shape(1);
  out.set_data(allocate_omarchy(out.nbytes()));
  state_out.set_data(allocate_omarchy(state_out.nbytes()));
  if (out.size() == 0) {
    return;
  }
  omarchy::ComputeParams params;
  params.count = checked_u32(static_cast<size_t>(B) * C, tag, out);
  params.operation = checked_u32(K, tag, out);
  params.lhs_size = checked_u32(K - 1, tag, out);
  params.reduce_size = checked_u32(C, tag, out);
  params.lhs_offset = checked_item_offset(in_state, in_state.size(), tag, out);
  params.rhs_offset = checked_item_offset(in_x, in_x.size(), tag, out);
  params.aux_offset = checked_item_offset(in_w, in_w.size(), tag, out);
  params.output_offset = checked_item_offset(out, out.size(), tag, out);
  params.in_strides[0] =
      checked_item_offset(state_out, state_out.size(), tag, out);
  std::array<omarchy::ComputeBinding, 5> bindings{
      binding(in_state),
      binding(in_x),
      binding(in_w),
      binding(out),
      binding(state_out)};
  uint32_t groups = (params.count + 255u) / 256u;
  encoder.dispatch_compute(
      omarchy::ComputeKernel::GdnConvDecodeBF16,
      bindings,
      params,
      std::min(groups, omarchy::kMaxComputeGroupCountX),
      1,
      1);
}

bool LayerNorm::use_fallback(Stream s) {
  return false;
}

void LayerNorm::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  const std::string tag = name();
  array& out = outputs.at(0);
  const array& in_x = inputs.at(0);
  const array& in_w = inputs.at(1);
  const array& in_b = inputs.at(2);
  auto s = stream();
  auto& encoder = omarchy::get_command_encoder(s);
  std::optional<array> x_temp;
  std::optional<array> w_temp;
  std::optional<array> b_temp;
  const array& x =
      ensure_dense(in_x, in_x.flags().row_contiguous, x_temp, encoder, s);
  const array& w =
      ensure_dense(in_w, in_w.flags().row_contiguous, w_temp, encoder, s);
  const array& b =
      ensure_dense(in_b, in_b.flags().row_contiguous, b_temp, encoder, s);
  require_norm_input(tag, x, out, encoder);
  require_norm_parameter(tag, w, x.shape(-1), out);
  require_norm_parameter(tag, b, x.shape(-1), out);
  size_t row_length = x.shape(-1);
  out.set_data(allocate_omarchy(out.nbytes()));
  if (out.size() == 0) {
    return;
  }
  auto params = norm_params(x, row_length, eps_, tag, out);
  params.rhs_offset = checked_item_offset(w, w.size(), tag, out);
  params.aux_offset = checked_item_offset(b, b.size(), tag, out);
  params.output_offset = checked_item_offset(out, out.size(), tag, out);
  params.lhs_size = checked_u32(w.size(), tag, out);
  params.aux_size = checked_u32(b.size(), tag, out);
  std::array<omarchy::ComputeBinding, 4> bindings{
      binding(x), binding(w), binding(b), binding(out)};
  auto kernel = select_float_kernel(
      out.dtype(),
      omarchy::ComputeKernel::FastLayerNormF32,
      omarchy::ComputeKernel::FastLayerNormF16,
      omarchy::ComputeKernel::FastLayerNormBF16);
  encoder.dispatch_compute(
      kernel,
      bindings,
      params,
      std::min(params.output_size, omarchy::kMaxComputeGroupCountX));
}

void RMSNormVJP::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  const std::string tag = name();
  array& dx = outputs.at(0);
  array& dw = outputs.at(1);
  const array& in_x = inputs.at(0);
  const array& in_w = inputs.at(1);
  const array& in_g = inputs.at(2);
  auto s = stream();
  auto& encoder = omarchy::get_command_encoder(s);
  std::optional<array> x_temp;
  std::optional<array> w_temp;
  std::optional<array> g_temp;
  const array& x =
      ensure_dense(in_x, in_x.flags().row_contiguous, x_temp, encoder, s);
  const array& w =
      ensure_dense(in_w, in_w.flags().row_contiguous, w_temp, encoder, s);
  const array& g =
      ensure_dense(in_g, in_g.flags().row_contiguous, g_temp, encoder, s);
  require_norm_input(tag, x, dx, encoder);
  require_norm_parameter(tag, w, x.shape(-1), dx);
  size_t row_length = x.shape(-1);
  if (x.size() == 0) {
    dx.set_data(allocate_omarchy(dx.nbytes()));
    zero_fill(dw);
    return;
  }
  // dx = gw * n - x * mean(gw * x) * n^3, one fused row kernel.
  dx.set_data(allocate_omarchy(dx.nbytes()));
  auto params = norm_params(x, row_length, eps_, tag, dx);
  params.rhs_offset = checked_item_offset(w, w.size(), tag, dx);
  params.aux_offset = checked_item_offset(g, g.size(), tag, dx);
  params.output_offset = checked_item_offset(dx, dx.size(), tag, dx);
  params.lhs_size = checked_u32(w.size(), tag, dx);
  std::array<omarchy::ComputeBinding, 4> bindings{
      binding(x), binding(w), binding(g), binding(dx)};
  auto dx_kernel = select_float_kernel(
      dx.dtype(),
      omarchy::ComputeKernel::FastRmsNormVjpDxF32,
      omarchy::ComputeKernel::FastRmsNormVjpDxF16,
      omarchy::ComputeKernel::FastRmsNormVjpDxBF16);
  encoder.dispatch_compute(
      dx_kernel,
      bindings,
      params,
      std::min(params.output_size, omarchy::kMaxComputeGroupCountX));
  // dw = sum_rows(g * x * n); a scalar weight keeps the upstream
  // zeros_like(w) result.
  if (w.ndim() == 0) {
    zero_fill(dw);
    return;
  }
  dw.set_data(allocate_omarchy(dw.nbytes()));
  // Stage 1 scatters one row's per-column contributions per workgroup;
  // stage 2 sums the rows. A single workgroup used to loop all rows with
  // a barrier per tile, and llvmpipe stops honoring those barriers past
  // roughly a thousand row iterations (NaN dw at 8x100x1024 and up).
  array dw_partial(Shape{static_cast<int>(x.size())}, float32, nullptr, {});
  dw_partial.set_data(allocate_omarchy(dw_partial.nbytes()));
  encoder.add_temporary(dw_partial);
  auto dw_params = norm_params(x, row_length, eps_, tag, dw_partial);
  dw_params.rhs_offset = checked_item_offset(g, g.size(), tag, dw);
  dw_params.output_offset = 0;
  std::array<omarchy::ComputeBinding, 3> dw_bindings{
      binding(x), binding(g), binding(dw_partial)};
  auto dw_kernel = select_float_kernel(
      dw.dtype(),
      omarchy::ComputeKernel::FastRmsNormVjpDwF32,
      omarchy::ComputeKernel::FastRmsNormVjpDwF16,
      omarchy::ComputeKernel::FastRmsNormVjpDwBF16);
  encoder.dispatch_compute(
      dw_kernel,
      dw_bindings,
      dw_params,
      std::min(dw_params.output_size, omarchy::kMaxComputeGroupCountX));
  omarchy::ComputeParams reduce_params = dw_params;
  reduce_params.lhs_offset = 0;
  reduce_params.reduce_size = dw_params.output_size;
  reduce_params.output_size = checked_u32(row_length, tag, dw);
  reduce_params.output_offset = checked_item_offset(dw, dw.size(), tag, dw);
  std::array<omarchy::ComputeBinding, 2> reduce_bindings{
      binding(dw_partial), binding(dw)};
  auto reduce_kernel = select_float_kernel(
      dw.dtype(),
      omarchy::ComputeKernel::FastRmsNormVjpDwReduceF32,
      omarchy::ComputeKernel::FastRmsNormVjpDwReduceF16,
      omarchy::ComputeKernel::FastRmsNormVjpDwReduceBF16);
  encoder.dispatch_compute(
      reduce_kernel,
      reduce_bindings,
      reduce_params,
      std::min(
          (reduce_params.output_size + 255u) / 256u,
          omarchy::kMaxComputeGroupCountX));
}

void LayerNormVJP::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  const std::string tag = name();
  array& dx = outputs.at(0);
  array& dw = outputs.at(1);
  array& db = outputs.at(2);
  const array& in_x = inputs.at(0);
  const array& in_w = inputs.at(1);
  const array& in_b = inputs.at(2);
  const array& in_g = inputs.at(3);
  auto s = stream();
  auto& encoder = omarchy::get_command_encoder(s);
  std::optional<array> x_temp;
  std::optional<array> w_temp;
  std::optional<array> b_temp;
  std::optional<array> g_temp;
  const array& x =
      ensure_dense(in_x, in_x.flags().row_contiguous, x_temp, encoder, s);
  const array& w =
      ensure_dense(in_w, in_w.flags().row_contiguous, w_temp, encoder, s);
  const array& b =
      ensure_dense(in_b, in_b.flags().row_contiguous, b_temp, encoder, s);
  const array& g =
      ensure_dense(in_g, in_g.flags().row_contiguous, g_temp, encoder, s);
  require_norm_input(tag, x, dx, encoder);
  require_norm_parameter(tag, w, x.shape(-1), dx);
  require_norm_parameter(tag, b, x.shape(-1), dx);
  size_t row_length = x.shape(-1);
  if (x.size() == 0) {
    dx.set_data(allocate_omarchy(dx.nbytes()));
    zero_fill(dw);
    zero_fill(db);
    return;
  }
  // dx = (wg - mean(wg)) * n - (x - mu) * mean(wg * (x - mu)) * n^3.
  dx.set_data(allocate_omarchy(dx.nbytes()));
  auto params = norm_params(x, row_length, eps_, tag, dx);
  params.rhs_offset = checked_item_offset(w, w.size(), tag, dx);
  params.aux_offset = checked_item_offset(g, g.size(), tag, dx);
  params.output_offset = checked_item_offset(dx, dx.size(), tag, dx);
  params.lhs_size = checked_u32(w.size(), tag, dx);
  std::array<omarchy::ComputeBinding, 4> bindings{
      binding(x), binding(w), binding(g), binding(dx)};
  auto dx_kernel = select_float_kernel(
      dx.dtype(),
      omarchy::ComputeKernel::FastLayerNormVjpDxF32,
      omarchy::ComputeKernel::FastLayerNormVjpDxF16,
      omarchy::ComputeKernel::FastLayerNormVjpDxBF16);
  encoder.dispatch_compute(
      dx_kernel,
      bindings,
      params,
      std::min(params.output_size, omarchy::kMaxComputeGroupCountX));
  // dw = sum_rows(g * (x - mu) * n); db = sum_rows(g) through the general
  // reduction. Scalar parameters keep the upstream zeros_like form.
  if (w.ndim() != 0) {
    dw.set_data(allocate_omarchy(dw.nbytes()));
    // Stage 1 scatters one row's per-column contributions per workgroup;
    // stage 2 sums the rows. A single workgroup used to loop all rows with
    // a barrier per tile, and llvmpipe stops honoring those barriers past
    // roughly a thousand row iterations (NaN dw at 8x100x1024 and up).
    array dw_partial(Shape{static_cast<int>(x.size())}, float32, nullptr, {});
    dw_partial.set_data(allocate_omarchy(dw_partial.nbytes()));
    encoder.add_temporary(dw_partial);
    auto dw_params = norm_params(x, row_length, eps_, tag, dw_partial);
    dw_params.rhs_offset = checked_item_offset(g, g.size(), tag, dw);
    dw_params.output_offset = 0;
    std::array<omarchy::ComputeBinding, 3> dw_bindings{
        binding(x), binding(g), binding(dw_partial)};
    auto dw_kernel = select_float_kernel(
        dw.dtype(),
        omarchy::ComputeKernel::FastLayerNormVjpDwF32,
        omarchy::ComputeKernel::FastLayerNormVjpDwF16,
        omarchy::ComputeKernel::FastLayerNormVjpDwBF16);
    encoder.dispatch_compute(
        dw_kernel,
        dw_bindings,
        dw_params,
        std::min(dw_params.output_size, omarchy::kMaxComputeGroupCountX));
    omarchy::ComputeParams reduce_params = dw_params;
    reduce_params.lhs_offset = 0;
    reduce_params.reduce_size = dw_params.output_size;
    reduce_params.output_size = checked_u32(row_length, tag, dw);
    reduce_params.output_offset = checked_item_offset(dw, dw.size(), tag, dw);
    std::array<omarchy::ComputeBinding, 2> reduce_bindings{
        binding(dw_partial), binding(dw)};
    auto reduce_kernel = select_float_kernel(
        dw.dtype(),
        omarchy::ComputeKernel::FastRmsNormVjpDwReduceF32,
        omarchy::ComputeKernel::FastRmsNormVjpDwReduceF16,
        omarchy::ComputeKernel::FastRmsNormVjpDwReduceBF16);
    encoder.dispatch_compute(
        reduce_kernel,
        reduce_bindings,
        reduce_params,
        std::min(
            (reduce_params.output_size + 255u) / 256u,
            omarchy::kMaxComputeGroupCountX));
  } else {
    zero_fill(dw);
  }
  if (b.ndim() != 0) {
    db.set_data(allocate_omarchy(db.nbytes()));
    std::vector<int> axes(g.ndim() - 1);
    std::iota(axes.begin(), axes.end(), 0);
    dispatch_reduce_general(
        tag,
        ReduceSumOperation,
        g,
        db,
        axes,
        encoder,
        select_reduce_general_kernel(g.dtype()));
  } else {
    zero_fill(db);
  }
}

void CrossEntropyVJP::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  const std::string tag = name();
  array& out = outputs.at(0);
  const array& in_x = inputs.at(0);
  const array& in_y = inputs.at(1);
  const array& loss = inputs.at(2);
  const array& in_g = inputs.at(3);
  auto s = stream();
  auto& encoder = omarchy::get_command_encoder(s);
  std::optional<array> x_temp;
  std::optional<array> y_temp;
  std::optional<array> g_temp;
  const array& x = ensure_dense(
      in_x, in_x.flags().row_contiguous, x_temp, encoder, s);
  const array& y = ensure_dense(
      in_y, in_y.flags().row_contiguous, y_temp, encoder, s);
  const array& g = ensure_dense(
      in_g, in_g.flags().row_contiguous, g_temp, encoder, s);
  require_norm_input(tag, x, out, encoder);
  if (y.dtype() != int32) {
    omarchy::unsupported(tag + " targets dtype", out);
  }
  if (g.dtype() != float32 || loss.dtype() != float32) {
    omarchy::unsupported(tag + " cotangent dtype", out);
  }
  size_t row_length = x.shape(-1);
  out.set_data(allocate_omarchy(out.nbytes()));
  if (out.size() == 0) {
    return;
  }
  // The kernel recomputes the stable row logsumexp from the logits, so p is
  // the exact softmax probability; the loss input only rides along.
  auto params = norm_params(x, row_length, 0.0f, tag, out);
  params.rhs_offset = checked_item_offset(y, y.size(), tag, out);
  params.aux_offset = checked_item_offset(g, g.size(), tag, out);
  params.output_offset = checked_item_offset(out, out.size(), tag, out);
  std::array<omarchy::ComputeBinding, 4> bindings{
      binding(x), binding(y), binding(g), binding(out)};
  auto kernel = select_float_kernel(
      out.dtype(),
      omarchy::ComputeKernel::CrossEntropyVjpF32,
      omarchy::ComputeKernel::CrossEntropyVjpF16,
      omarchy::ComputeKernel::CrossEntropyVjpBF16);
  encoder.dispatch_compute(
      kernel,
      bindings,
      params,
      std::min(params.output_size, omarchy::kMaxComputeGroupCountX));
}

bool RoPE::use_fallback(Stream s) {
  return false;
}

// Fused RoPE absorbs the eager composition's Sin and Cos primitives, so it
// carries their kTrigArgumentLimit gate: known-defects.md, "A correction on
// the record", pins that a path which runs a primitive without going
// through its eval_gpu must carry that primitive's gates. The eager gate
// reduced the materialized theta tensor; this path never builds one, so the
// bound comes from the factors instead:
//
//   max|theta| <= (max_b |off_b| + (T - 1)) * |scale| * max_i |inv_freq_i|
//
// For the non-negative offsets every caller sends this is exact (the
// per-batch positions run off_b .. off_b + T - 1); mixed signs take the
// triangle inequality, which can only refuse marginally earlier, never
// later. max_i |inv_freq_i| is exactly 1.0 for base >= 1 (exp(0) = 1.0 in
// any IEEE exp), a host exp otherwise (an ulp caveat at the refusal
// boundary only), and the reciprocal of the smallest magnitude when freqs
// are given. Only the tiny offset and freqs arrays are read, behind the
// same stream-ordered synchronize trig_argument_gate documents: an
// unordered mapped read raced its own submission on hardware and returned
// recycled-page garbage.
void rope_trig_gate(
    const std::string& name,
    const array& in,
    const array& offset,
    const array* freqs,
    int dims,
    float base,
    float scale,
    const array& out) {
  Stream stream = out.primitive().stream();
  int T = in.shape(-2);
  float worst_offset;
  if (offset.size() == 1) {
    // A scalar constructed from a host int (what an mlx_lm decode step
    // passes) is Status::available from construction with no primitive:
    // nothing on the queue can be writing it, so it is read directly.
    // Anything else may still be in flight and takes the stream-ordered
    // synchronize. is_available() is NOT the test: it promotes an
    // in-flight evaluated array with no event. Draining the queue for
    // every offset cost two host round trips per decoder layer, 48 per
    // token (receipts/2026-09-04-rope-gate-drain.md).
    bool host_constant = offset.status() == array::Status::available &&
        !offset.has_primitive();
    if (!host_constant) {
      // Mirror the vector path below: settle schedules the offset if it
      // is still unscheduled, and synchronize proves its bytes are final
      // before the mapped read.
      settle({offset});
      omarchy::get_command_encoder(stream).synchronize("rope_offset_scalar");
      // The offset may carry an async-eval event latch from the pass that
      // dispatched it - typically THIS pass, when the model graph runs
      // inside an async_eval tape (mlx_lm BatchGenerator split/decode
      // step with a padded-batch cache offset array). That latch's only
      // signaler is the owning pass's epilogue commit, still ahead of
      // this mid-tape readback, so item()'s event wait can never be
      // satisfied: the dispatching thread deadlocked until the watchdog
      // fired and the generation thread died ('Vulkan timeline counter
      // failed to advance ... target=1', the mlx_lm.server / oMLX
      // prefill serving hangs - receipts/2026-09-19-mlxlm-server-hang-jw16.md).
      // synchronize() above proved the data is final, so drop the stale
      // latch and read.
      offset.detach_event();
    }
    worst_offset = std::abs(static_cast<float>(offset.item<int>()));
  } else {
    array offset_worst =
        astype(max(abs(offset, stream), stream), float32, stream);
    // settle, not eval(): a nested blocking eval cannot complete here
    // (its epilogue skips signal+commit at nest depth > 1) - same class
    // as the trig gate; the synchronize below orders the read.
    settle({offset_worst});
    omarchy::get_command_encoder(stream).synchronize("rope_offset_vector");
    worst_offset = offset_worst.item<float>();
  }
  float inv_freq_bound;
  if (freqs != nullptr) {
    array freqs_min = min(abs(*freqs, stream), stream);
    settle({freqs_min});
    omarchy::get_command_encoder(stream).synchronize("rope_freqs_bound");
    inv_freq_bound = 1.0f / freqs_min.item<float>();
  } else {
    float beta = static_cast<float>(std::log(base) / (dims / 2));
    inv_freq_bound =
        (beta >= 0.0f) ? 1.0f : std::exp(-static_cast<float>(dims / 2 - 1) * beta);
  }
  float bound = (worst_offset + (T - 1)) * std::abs(scale) * inv_freq_bound;
  if (bound > kTrigArgumentLimit) {
    throw std::runtime_error(
        "[omarchy] " + name + " rotational argument magnitude " +
        std::to_string(bound) + " exceeds the built-in accuracy limit " +
        std::to_string(kTrigArgumentLimit) +
        " on this backend: the fused kernel computes sin/cos with the"
        " Vulkan driver's built-in, whose range reduction is untrusted"
        " above it, and the software Payne-Hanek fallback miscompiles on"
        " this driver. No silent wrong value occurs. Run it on an explicit"
        " CPU stream to use the CPU implementation.");
  }
}

void RoPE::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  const std::string tag = name();
  array& out = outputs.at(0);
  const array& in = inputs.at(0);
  const array& offset = inputs.at(1);
  auto s = stream();
  auto& encoder = omarchy::get_command_encoder(s);

  // FENCE (2026-09-03): two variants have not passed equivalence
  // against the composed fallback - per-batch vector offsets (160 of
  // 320 element checks diverged) and the inverse/VJP path (5 of 160).
  // A fused path must never serve a leg that has not passed, so both
  // ride the composition until each passes real equivalence. Decode,
  // the case worth ~460 primitives per token, is forward with a
  // scalar offset and stays fused. Remove one conjunct per fixed
  // defect, with the equivalence test green.
  if (!forward_ || offset.size() > 1) {
    auto result = fallback_(inputs);
    // Record-only settle: fallback nodes join the open batch, and the
    // synchronize below submits and orders the buffer handoff. A nested
    // blocking eval() here deadlocks on the skipped nested epilogue.
    settle({result[0]});
    encoder.synchronize("rope_fallback");
    out.copy_shared_buffer(result[0]);
    return;
  }

  require_float_dtype(tag, in, out, encoder);
  if (out.size() == 0) {
    out.set_data(allocate_omarchy(out.nbytes()));
    return;
  }
  bool with_freqs = inputs.size() == 3;
  rope_trig_gate(
      tag,
      in,
      offset,
      with_freqs ? &inputs.at(2) : nullptr,
      dims_,
      base_,
      scale_,
      out);


  // Producer-direct KV write: when the eager planner claimed this
  // RoPE output as a SliceUpdate pair member, the rotated rows below
  // store straight into the updated cache copy's window and the merged
  // pair dispatch disappears. Aborts unwind to that dispatch.
  omarchy::KvDirectWindow* kv_window =
      (out.size() > 0 &&
       (out.dtype() == float16 || out.dtype() == bfloat16) &&
       inputs.size() == 2)
      ? omarchy::find_rope_kv_redirect(out)
      : nullptr;
  if (kv_window) {
    omarchy::KvDirectWindow& window = *kv_window;
    // In place, the values side has already installed the shared
    // storage; otherwise the node must still be unallocated. Either
    // way the rope writes its rows into the buffer the member array
    // exposes.
    if (
        window.node.data_shared_ptr() !=
            (window.in_place ? window.base.data_shared_ptr() : nullptr) ||
        window.base.data_shared_ptr() == nullptr ||
        window.base.dtype() != out.dtype() ||
        (out.dtype() == bfloat16 && (kv_window->offset & 1u) != 0u) ||
        !window.base.flags().row_contiguous ||
        window.base.size() != window.base.data_size() ||
        window.base.offset() % window.base.itemsize() != 0 ||
        !omarchy::input_ready(window.base, s) ||
        !omarchy::input_ready(in, s) ||
        !omarchy::input_ready(offset, s)) {
      // The values side already stored its rows (find_rope_kv_redirect
      // only fires after that commit), so unwinding to the merged pair
      // dispatch is impossible: the redirected sum buffer never
      // materializes. Fail loudly instead of corrupting the cache.
      throw std::runtime_error(
          "[mlx-omarchy] direct KV write committed values but the keys "
          "window is not dispatchable");
    }
    if (!window.in_place) {
      window.node.set_data(allocate_omarchy(window.node.nbytes()));
      copy_gpu(
          window.base,
          window.node,
          window.base.flags().contiguous ? CopyType::Vector
                                         : CopyType::General,
          s);
    }
    encoder.add_temporary(window.node);
    encoder.add_temporary(window.base);
  }
  bool kv_direct = kv_window != nullptr;
  // The kernel never rotates in place: the input binding is readonly, the
  // output binding writeonly, and they never alias (an aliased
  // readonly/writeonly pair drops stores on llvmpipe - observed
  // 2026-09-03 - so every case below reads the input buffer directly or
  // through a temporary and writes a fresh output). The stride cases
  // mirror the upstream Metal decision tree (mlx/backend/metal/rope.cpp).
  int ndim = in.ndim();
  int B = in.shape(0);
  int T = in.shape(-2);
  int D = in.shape(-1);
  int half_dims = dims_ / 2;
  bool direct_bf16 = out.dtype() == bfloat16;
  bool passthrough = dims_ < D;
  // The pair-granular packed bf16 word store covers a word only when
  // both lanes are written: the passthrough row extent and the
  // rotation half must both be even (rotation pairs straddle the
  // half boundary otherwise).
  if (direct_bf16 &&
      (((passthrough && (D % 2) != 0)) || (half_dims % 2) != 0)) {
    omarchy::unsupported(tag + " bf16 odd rotation pair", out);
  }
  size_t mat_size = static_cast<size_t>(T) * D;
  std::optional<array> normalized;
  const array* src = &in;
  bool row_contiguous = src->flags().row_contiguous;
  bool head_seq_transpose = false;
  int64_t strides[3];

  int dispatch_ndim = ndim;
  while (in.shape(-dispatch_ndim) == 1 && dispatch_ndim > 3) {
    dispatch_ndim--;
  }
  int N = 1;
  for (int i = 1; i < (ndim - 2); ++i) {
    N *= in.shape(i);
  }
  if (row_contiguous) {
    strides[0] = mat_size;
    strides[1] = src->strides()[ndim - 2];
    strides[2] = src->strides()[ndim - 1];
  } else if (dispatch_ndim == 3) {
    // Handle non-contiguous 3D inputs
    strides[0] = src->strides()[ndim - 3];
    strides[1] = src->strides()[ndim - 2];
    strides[2] = src->strides()[ndim - 1];
  } else if (
      ndim == 4 &&
      // batch dim is regularly strided
      src->strides()[0] == static_cast<int64_t>(T) * N * D &&
      // sequence and head dimensions are transposed
      src->strides()[1] == D &&
      src->strides()[2] == static_cast<int64_t>(N) * D) {
    head_seq_transpose = true;
    strides[0] = src->strides()[1];
    strides[1] = src->strides()[2];
    strides[2] = src->strides()[3];
  } else {
    // Copy non-contiguous > 3D inputs into a contiguous temporary and
    // rotate from there.
    normalized = array(in.shape(), in.dtype(), nullptr, {});
    copy_gpu(in, *normalized, CopyType::General, s);
    encoder.add_temporary(*normalized);
    src = &*normalized;
    strides[0] = mat_size;
    strides[1] = (*normalized).strides()[ndim - 2];
    strides[2] = (*normalized).strides()[ndim - 1];
  }

  // A size-1 offset is a scalar no matter its rank: the composition
  // broadcasts it to every batch. A (1,)-shaped array has stride 1, so
  // borrowing it verbatim would read past the buffer for b > 0 (found
  // 2026-09-03 by the fence's scalar-offset equivalence test, at the
  // first element of batch 1).
  int64_t offset_stride =
      (offset.size() == 1 || offset.ndim() == 0) ? 0 : offset.strides()[0];
  omarchy::ComputeParams params;
  params.count = checked_u32(
      static_cast<size_t>(B) * N * T * (passthrough ? D : half_dims),
      tag,
      out);
  if (direct_bf16 && passthrough) {
    // The pair-granular bf16 main enumerates output words: D/2 per row.
    params.count = checked_u32(params.count / 2, tag, out);
  }
  params.dims = checked_u32(half_dims, tag, out);
  params.flags = (forward_ ? 1u : 0u) | (traditional_ ? 2u : 0u) |
      (head_seq_transpose ? 4u : 0u) | (passthrough ? 8u : 0u);
  params.alpha = scale_;
  params.beta = static_cast<float>(std::log(base_) / half_dims);
  params.lhs_offset = checked_item_offset(*src, in.size(), tag, out);
  params.rhs_offset = checked_item_offset(offset, offset.size(), tag, out);
  params.output_offset = checked_item_offset(out, out.size(), tag, out);
  params.matrix_m = checked_u32(offset_stride, tag, out);
  params.shape[0] = checked_u32(N, tag, out);
  params.shape[1] = checked_u32(T, tag, out);
  params.shape[2] = checked_u32(D, tag, out);
  params.in_strides[0] = checked_u32(strides[0], tag, out);
  params.in_strides[1] = checked_u32(strides[1], tag, out);
  params.in_strides[2] = checked_u32(strides[2], tag, out);
  // The output is freshly allocated here, so it is row contiguous.
  params.out_strides[0] = checked_u32(mat_size, tag, out);
  params.out_strides[1] = checked_u32(D, tag, out);
  params.out_strides[2] = 1u;
  if (kv_direct) {
    params.output_offset = kv_window->offset;
    params.out_strides[0] = kv_window->strides[0];
    params.out_strides[1] = kv_window->strides[1];
    params.out_strides[2] = kv_window->strides[2];
  }
  if (with_freqs) {
    params.aux_offset =
        checked_item_offset(inputs.at(2), inputs.at(2).size(), tag, out);
  }
  // The no-freqs shader variants declare three bindings; the freqs slot
  // doubles the offset binding like the scalar-weight norm kernels do.
  // The bfloat16 shader legs read and store packed bf16 words with the
  // repo's constant-shift form (commit cf68e7d): a uint16_t-typed block
  // compiled from that source returned recycled memory on llvmpipe.
  array rope_output = out;
  if (kv_direct) {
    rope_output = kv_window->node;
  } else {
    out.set_data(allocate_omarchy(out.nbytes()));
  }
  std::array<omarchy::ComputeBinding, 4> bindings{
      binding(*src),
      binding(rope_output),
      binding(offset),
      binding(with_freqs ? inputs.at(2) : offset)};
  omarchy::ComputeKernel kernel;
  if (direct_bf16) {
    kernel = with_freqs ? omarchy::ComputeKernel::FastRopeFreqsBF16
                        : omarchy::ComputeKernel::FastRopeBF16;
  } else {
    kernel = select_float_kernel(
        rope_output.dtype(),
        with_freqs ? omarchy::ComputeKernel::FastRopeFreqsF32
                   : omarchy::ComputeKernel::FastRopeF32,
        with_freqs ? omarchy::ComputeKernel::FastRopeFreqsF16
                   : omarchy::ComputeKernel::FastRopeF16,
        with_freqs ? omarchy::ComputeKernel::FastRopeFreqsBF16
                   : omarchy::ComputeKernel::FastRopeBF16);
  }
  encoder.dispatch_compute(
      kernel,
      bindings,
      params,
      omarchy::compute_dispatch_group_count(params.count));
  if (kv_direct) {
    omarchy::commit_rope_kv_redirect(out);
  }
}
void ScaledDotProductAttention::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  const std::string tag = name();
  array& out = outputs.at(0);
  const array& q = inputs.at(0);
  const array& k = inputs.at(1);
  const array& v = inputs.at(2);
  auto s = stream();
  auto& encoder = omarchy::get_command_encoder(s);
  require_float_dtype(tag, q, out, encoder);
  require_float_dtype(tag, k, out, encoder);
  require_float_dtype(tag, v, out, encoder);
  if (q.ndim() != 4 || k.ndim() != 4 || v.ndim() != 4) {
    omarchy::unsupported("attention rank " + tag, out);
  }
  int batch = q.shape(0);
  int heads = q.shape(1);
  int q_len = q.shape(2);
  int head_dim = q.shape(3);
  int kv_heads = k.shape(1);
  int k_len = k.shape(2);
  int v_dim = v.shape(3);
  if (
      k.shape(0) != batch || v.shape(0) != batch || k.shape(3) != head_dim ||
      v.shape(1) != kv_heads || v.shape(2) != k_len ||
      out.shape() != Shape{batch, heads, q_len, v_dim}) {
    omarchy::unsupported("attention shapes " + tag, out);
  }
  if (kv_heads == 0 || heads % kv_heads != 0) {
    omarchy::unsupported("attention head split " + tag, out);
  }
  int repeats = heads / kv_heads;
  if (out.size() == 0) {
    out.set_data(allocate_omarchy(out.nbytes()));
    return;
  }

  // Shared commit of the attention result into the output: with GQA the
  // result is 5-D (kv, repeat) while out is 4-D (heads). The flat
  // row-major layouts match, so share the buffer under out's own
  // strides: copy_shared_buffer(result) alone would install 5-D strides
  // and a 5-element stride vector on a 4-D array, and every later
  // reader would walk wrong addresses (the upstream fallback flattens
  // axes 1-2 here).
  auto commit_result = [&](const array& result) {
    if (repeats > 1) {
      Strides out_strides(out.ndim(), 1);
      for (int axis = out.ndim() - 2; axis >= 0; --axis) {
        out_strides[axis] = out_strides[axis + 1] * out.shape(axis + 1);
      }
      array::Flags flags;
      flags.contiguous = true;
      flags.row_contiguous = true;
      auto max_dim = std::max_element(out.shape().begin(), out.shape().end());
      flags.col_contiguous = out.size() <= 1 || out.size() == *max_dim;
      out.copy_shared_buffer(result, out_strides, flags, result.data_size());
    } else {
      out.copy_shared_buffer(result);
    }
  };

  auto make_mask_view = [&](const array& base, Shape shape, Strides strides) {
    array view(std::move(shape), base.dtype(), nullptr, {});
    array::Flags flags;
    flags.contiguous = base.flags().contiguous;
    flags.row_contiguous = false;
    flags.col_contiguous = false;
    view.copy_shared_buffer(base, strides, flags, base.data_size());
    encoder.add_temporary(view);
    return view;
  };
  auto prepare_mask = [&](const array& input, const Shape& score_shape) {
    array mask = input;
    if (repeats > 1 && mask.ndim() >= 3) {
      int head_axis = mask.ndim() - 3;
      Shape shape = mask.shape();
      Strides strides = mask.strides();
      int64_t head_stride = strides[head_axis];
      if (shape[head_axis] == 1) {
        shape.insert(shape.begin() + head_axis + 1, 1);
        strides.insert(strides.begin() + head_axis + 1, 0);
      } else {
        if (shape[head_axis] != heads) {
          omarchy::unsupported("attention mask shape " + tag, out);
        }
        shape[head_axis] = kv_heads;
        shape.insert(shape.begin() + head_axis + 1, repeats);
        strides[head_axis] = head_stride * repeats;
        strides.insert(strides.begin() + head_axis + 1, head_stride);
      }
      mask = make_mask_view(mask, std::move(shape), std::move(strides));
    }
    if (mask.ndim() > score_shape.size()) {
      omarchy::unsupported("attention mask shape " + tag, out);
    }
    Strides strides(score_shape.size(), 0);
    int offset = static_cast<int>(score_shape.size()) - mask.ndim();
    for (int axis = 0; axis < mask.ndim(); ++axis) {
      int target_axis = offset + axis;
      if (mask.shape(axis) == score_shape[target_axis]) {
        strides[target_axis] = mask.strides()[axis];
      } else if (mask.shape(axis) != 1) {
        omarchy::unsupported("attention mask shape " + tag, out);
      }
    }
    if (mask.shape() != score_shape || mask.strides() != strides) {
      mask = make_mask_view(mask, score_shape, std::move(strides));
    }
    return mask;
  };

  // f16 runs at f16 storage end to end: the scores and probs matmuls
  // keep f16 operands and accumulate in float inside the shader
  // (matmul.comp), the scale rides the scores matmul's alpha, and the
  // softmax runs float math over f16 storage (softmax_suffix.comp). The
  // only new rounding against the f32 composition below is f16 storage
  // of the score and prob intermediates; the qmm decode path already
  // runs this pattern. Deletes the three q/k/v upcasts, the scale
  // broadcast and multiply, and the output downcast per call.
  //
  // bfloat16 rides the same shape under MLX_OMARCHY_SDPA_BF16_FAST
  // (default off until the M1 equivalence leg passes; any value other
  // than "0" enables): MatmulBF16, SoftmaxBF16, and ElementwiseBF16 are
  // the proven uint16_t-typed USE_BF16 legs the projections, the
  // residual adds, and the norm already run - float accumulation inside
  // the shader, bf16 storage with round-to-nearest-even stores on both
  // drivers. Scores store bf16 (2^-8 relative rounding per stored
  // score), and the output stays bf16 end to end. Deletes the three
  // q/k/v upcasts, the f32 scale multiply, the f32 softmax, and the
  // output downcast per call. The causal mask is not materialized on
  // either storage dtype (see the scores matmul below). Single-query
  // decode never reaches this route: the fused kernel above returns
  // first, so this arm is prefill-shaped traffic only.
  bool bf16_fast = false;
  if (q.dtype() == bfloat16) {
    if (const char* env = std::getenv("MLX_OMARCHY_SDPA_BF16_FAST");
        env != nullptr && std::strcmp(env, "0") != 0) {
      bf16_fast = true;
    }
  }

  // Decode uses native Metal's 32-key online-softmax order directly over
  // Q and the strided KV cache. This replaces both attention matmuls, both
  // layout copies, softmax, and the q/k/v upcasts with one dispatch per
  // layer. float16 rides the proven SdpaDecodeNativeF16 arm. bfloat16
  // rides the same structure through the -DBF16_IO blob: exact widening
  // in, RNE stores, every intermediate in float32 (upstream Metal keeps
  // every attention intermediate in float32, sdpa_vector.h:50), and the
  // one-pass 32-stream merge at every context length - the f32 partial
  // array the two-pass needs cannot fit the workgroup store at the
  // 128-block crossover, and the bf16 parity contract is the f32-score
  // composition below, not native's two-pass partial format.
  const char* decode_env = std::getenv("MLX_OMARCHY_SDPA_DECODE_NATIVE");
  const auto& decode_caps = encoder.device().capabilities();
  constexpr VkSubgroupFeatureFlags kDecodeSubgroupFeatures =
      VK_SUBGROUP_FEATURE_BASIC_BIT | VK_SUBGROUP_FEATURE_ARITHMETIC_BIT |
      VK_SUBGROUP_FEATURE_SHUFFLE_BIT;
  constexpr uint32_t kDecodeSharedBytes =
      (32u * 32u + 2u * 128u + 128u * 32u) * sizeof(float);
  const bool decode_subgroup_ready = decode_caps.cooperative_matrix_f32_8 &&
      decode_caps.subgroup_size == 32u &&
      decode_caps.max_compute_work_group_invocations >= 1024u &&
      decode_caps.max_compute_work_group_size[0] >= 1024u &&
      decode_caps.max_compute_shared_memory_size >= kDecodeSharedBytes &&
      (decode_caps.subgroup_operations & kDecodeSubgroupFeatures) ==
          kDecodeSubgroupFeatures;
  // The composition-exact bf16 arm uses no subgroup operations at all - its
  // per-thread work and barriers need only the 1024-thread workgroup and the
  // static shared the arm declares (9,472 bytes at query width 64, 10,240 at
  // the Qwen3.8 full-attention width 256) - so it gates on those alone and
  // engages on any device that meets them, including software drivers, where
  // its bit-identity against the composition is testable.
  // Perf-only shape gate: below 256 keys the arm's serial accumulation
  // loses to the composition (34.1 vs 31.2 ms/token at the short leg); at
  // and above it the arm wins (34.4 vs 34.7 at 262, 39.9 vs 43.3 at 1K).
  // The gate cannot move a digest: both routes store identical words for
  // every input, so either side of the boundary the token stream is the
  // base stream. 256 sits between the measured losing (<=61-key) and
  // winning (>=262-key) regimes on the 16-wide tile boundary.
  constexpr uint32_t kDecodeBf16SharedBytes =
      (64u + 2048u + 256u) * sizeof(float);
  constexpr uint32_t kDecodeBf16SharedBytesHd256 =
      (256u + 2048u + 256u) * sizeof(float);
  const uint32_t decode_bf16_shared_required = head_dim == 256
      ? kDecodeBf16SharedBytesHd256
      : kDecodeBf16SharedBytes;
  // Perf-only k window: bitwise identity holds for every k (both routes
  // store identical words), so the boundary cannot move a token - only the
  // wall. Width 64 keeps the measured 256..2048 window from the original
  // qualification. Width 256 re-measured on t6001-host (Qwen3.8 decode
  // shapes, one-call wall, 300 reps): the arm is at or above the
  // composition from 24 keys through the 128-key tie point and loses
  // beyond it (the single-workgroup-per-head walk loses to the
  // composition's parallel GEMV tiles once the key stream dominates), so
  // the arm engages from 12 keys - just under the 13-key first decode step
  // of a 12-token prompt - through 128; larger contexts keep the
  // composition. The 128 cap is also the measured-safe side of both
  // 2026-09-24 sweeps (receipts-work/receipt-sdpa-k512-jw16.md): the arm
  // wins through 192 keys, ties near 256-300, and loses 33-46% in the
  // 448..544 contract regime (502.9 vs 339.8 us at 512), where the shipped
  // <=2048 bound had been routing decode to the slower arm.
  const bool decode_bf16_ready =
      decode_caps.max_compute_work_group_invocations >= 1024u &&
      decode_caps.max_compute_work_group_size[0] >= 1024u &&
      decode_caps.max_compute_shared_memory_size >=
          decode_bf16_shared_required;
  const bool decode_bf16_probe = q.dtype() == bfloat16;
  const bool decode_route_ready =
      decode_bf16_probe ? decode_bf16_ready : decode_subgroup_ready;
  if ((decode_env == nullptr || std::strcmp(decode_env, "0") != 0) &&
      decode_route_ready && inputs.size() == 3 && !has_sinks_ &&
      !output_logsumexp_ && batch == 1 && q_len == 1 &&
      (q.dtype() == float16 || q.dtype() == bfloat16) &&
      ((head_dim == 64 && v_dim == 64) ||
          (decode_bf16_probe && head_dim == 256 && v_dim == 256)) &&
      k_len > 0 &&
      (q.dtype() != bfloat16 ||
          (k_len >= (head_dim == 256 ? uint32_t{12} : uint32_t{256}) &&
           k_len <=
               (head_dim == 256 ? uint32_t{128} : uint32_t{2048}))) &&
      q.strides()[3] == 1 && k.strides()[3] == 1 && v.strides()[3] == 1) {
    const bool decode_bf16 = decode_bf16_probe;
    out.set_data(allocate_omarchy(out.nbytes()));
    omarchy::ComputeParams params;
    params.count = checked_u32(out.size(), tag, out);
    params.matrix_m = checked_u32(heads, tag, out);
    params.matrix_n = checked_u32(kv_heads, tag, out);
    params.matrix_k = checked_u32(k_len, tag, out);
    params.alpha = scale_;
    // Native M1 Max switches to 64 two-pass blocks at 1024 keys and 128
    // above it; zero selects the one-pass 32-SIMD decode kernel. bf16
    // always runs one-pass (see above).
    params.flags = decode_bf16
        ? 0u
        : (k_len > 1024 ? 128u : (k_len == 1024 ? 64u : 0u));
    params.lhs_offset = checked_item_offset(q, q.size(), tag, out);
    params.rhs_offset = checked_item_offset(k, k.size(), tag, out);
    params.aux_offset = checked_item_offset(v, v.size(), tag, out);
    params.output_offset = checked_item_offset(out, out.size(), tag, out);
    params.shape[0] = checked_u32(q.strides()[1], tag, out);
    params.shape[1] = checked_u32(k.strides()[1], tag, out);
    params.shape[2] = checked_u32(k.strides()[2], tag, out);
    params.shape[3] = checked_u32(q.strides()[3], tag, out);
    params.in_strides[0] = checked_u32(k.strides()[3], tag, out);
    params.in_strides[1] = checked_u32(v.strides()[1], tag, out);
    params.in_strides[2] = checked_u32(v.strides()[2], tag, out);
    params.in_strides[3] = checked_u32(v.strides()[3], tag, out);
    std::array<omarchy::ComputeBinding, 4> bindings{
        binding(q), binding(k), binding(v), binding(out)};
      omarchy::capsim::require_backed(
          encoder.device(),
          decode_caps,
          decode_subgroup_ready,
          decode_bf16
              ? (head_dim == 256 ? "SdpaDecodeNativeBF16Hd256"
                                 : "SdpaDecodeNativeBF16")
              : "SdpaDecodeNativeF16",
          "cooperative_matrix_fp32_8x8x8+workgroup_limits+"
          "shared_memory_limit_bytes",
          encoder.device().hardware_capabilities().cooperative_matrix_f32_8);
    if (params.flags != 0u) {
      // Native-shape per-block two-pass (f16 only; bf16 always one-pass).
      // The fused kernel keeps one workgroup per head, so its walk runs
      // k/32 sequential key-steps per subgroup no matter how the blocks
      // split; native Metal instead grids heads x blocks with one 32-thread
      // simdgroup per block, dropping the serial depth to ~k/blocks with
      // heads*blocks concurrent chains. Pass 1 reproduces the fused
      // kernel's per-block loop exactly and stores the same f16/f32 block
      // partials; pass 2 folds them with the fused kernel's pass-2 code
      // unchanged, so the pair is bit-identical to the fused two-pass.
      const uint32_t blocks = params.flags;
      // Scratch words per head: blocks max + blocks sum + blocks*32 packed
      // f16 output-pair words (uint32 each), matching the shaders'
      // SCRATCH_STRIDE_FACTOR layout.
      const uint64_t scratch_words =
          static_cast<uint64_t>(heads) * blocks * 34u;
      array scratch(
          Shape{static_cast<int>(scratch_words)}, uint32, nullptr, {});
      scratch.set_data(allocate_omarchy(scratch.nbytes()));
      encoder.add_temporary(scratch);
      std::array<omarchy::ComputeBinding, 4> pass1_bindings{
          binding(q), binding(k), binding(v), binding(scratch)};
      omarchy::ComputeParams pass1_params = params;
      pass1_params.count = checked_u32(scratch_words, tag, out);
      encoder.dispatch_compute(
          omarchy::ComputeKernel::SdpaDecodeNativeTwoPassP1F16,
          pass1_bindings,
          pass1_params,
          params.matrix_m,
          blocks);
      std::array<omarchy::ComputeBinding, 2> pass2_bindings{
          binding(scratch), binding(out)};
      encoder.dispatch_compute(
          omarchy::ComputeKernel::SdpaDecodeNativeTwoPassP2F16,
          pass2_bindings,
          params,
          params.matrix_m);
      return;
    }
    encoder.dispatch_compute(
        decode_bf16
            ? (head_dim == 256
                  ? omarchy::ComputeKernel::SdpaDecodeNativeBF16Hd256
                  : omarchy::ComputeKernel::SdpaDecodeNativeBF16)
            : omarchy::ComputeKernel::SdpaDecodeNativeF16,
        bindings,
        params,
        params.matrix_m);
    return;
  }
  if (q.dtype() == float16 || bf16_fast) {
    const bool bf16 = q.dtype() == bfloat16;
    const Dtype storage_dtype = bf16 ? bfloat16 : float16;
    // GQA regroup as pure stride views, so a non-contiguous cache
    // slice rides its own strides straight into the matmul;
    // reshape_in_eval would copy - it only views row-contiguous
    // inputs, and a cache slice never is one. q splits the heads axis
    // into (kv, repeat); k and v keep their kv heads and insert a
    // size-1 repeat axis the matmul broadcasts (stride pinned 0).
    auto regroup_view = [&](const array& base) {
      bool splits = base.shape(1) != kv_heads;
      int rep = splits ? repeats : 1;
      Shape shape = {
          base.shape(0), kv_heads, rep, base.shape(2), base.shape(3)};
      Strides strides(5);
      strides[0] = base.strides()[0];
      strides[1] = splits ? base.strides()[1] * rep : base.strides()[1];
      strides[2] = splits ? base.strides()[1] : 0;
      strides[3] = base.strides()[2];
      strides[4] = base.strides()[3];
      array view(std::move(shape), base.dtype(), nullptr, {});
      view.copy_shared_buffer(
          base, strides, {false, false, false}, base.size());
      encoder.add_temporary(view);
      return view;
    };
    array qs = repeats > 1 ? regroup_view(q) : q;
    array ks = repeats > 1 ? regroup_view(k) : k;
    array vs = repeats > 1 ? regroup_view(v) : v;
    array keys_t = swapaxes_in_eval(ks, -1, -2);
    encoder.add_temporary(keys_t);
    Shape score_shape = qs.shape();
    score_shape.back() = k_len;
    // The causal case never materializes a mask: the softmax runs in
    // its causal mode (only keys <= k_len - q_len + position take
    // part, the rest store exact zeros, the same values the additive
    // storage-floor mask produced), so the scores matmul may skip the
    // fully masked column tiles and the probs matmul the k tiles past
    // every row's last key. That deletes a q_len x k_len host mask
    // fill and one full read-modify-write of the scores per call.
    const uint32_t causal_offset =
        do_causal_ ? static_cast<uint32_t>(k_len - q_len) : 0u;
    array scores(score_shape, storage_dtype, nullptr, {});
    dispatch_matmul(
        tag,
        {qs, keys_t},
        scores,
        scale_,
        0.0f,
        false,
        s,
        {causal_offset,
         do_causal_ ? CausalSkip::Columns : CausalSkip::None});
    encoder.add_temporary(scores);

    std::optional<array> masked;
    const array* sinks =
        has_sinks_ ? &inputs.at(inputs.size() - 1) : nullptr;
    if (sinks != nullptr && (*sinks).dtype() != storage_dtype) {
      omarchy::unsupported("attention sinks dtype " + tag, out);
    }
    const bool has_arr_mask =
        (inputs.size() == 5) || (inputs.size() == 4 && !has_sinks_);
    if (!do_causal_ && has_arr_mask) {
      const array& mask = inputs.at(3);
      if (mask.dtype() != storage_dtype) {
        omarchy::unsupported("attention mask dtype " + tag, out);
      }
      masked = prepare_mask(mask, scores.shape());
      array added(scores.shape(), storage_dtype, nullptr, {});
      dispatch_elementwise(tag, AddOperation, {scores, *masked}, added, s);
      masked = added;
      encoder.add_temporary(*masked);
    }
    const array& logits = masked ? *masked : scores;
    encoder.add_temporary(logits);

    array probs(logits.shape(), storage_dtype, nullptr, {});
    dispatch_softmax(
        tag,
        logits,
        probs,
        s,
        sinks,
        q_len,
        do_causal_,
        do_causal_ ? static_cast<int>(causal_offset) : 0);
    encoder.add_temporary(probs);

    Shape result_shape = probs.shape();
    result_shape.back() = v_dim;
    array result(result_shape, storage_dtype, nullptr, {});
    dispatch_matmul(
        tag,
        {probs, vs},
        result,
        1.0f,
        0.0f,
        false,
        s,
        {causal_offset, do_causal_ ? CausalSkip::K : CausalSkip::None});
    encoder.add_temporary(result);
    commit_result(result);
    return;
  }

  // The validated f32-score composition (the M1 4-bit degeneracy fix,
  // docs/2026-09-01-m1-4bit-greedy-sdpa-f16-scores.md): cast to
  // float32, scale, express GQA through the unflatten/expand_dims
  // shapes instead of mx.repeat, add the mask as a float32 additive
  // term, and keep every intermediate in float32 so score magnitudes
  // far beyond float16 stay finite. Only the result narrows back to
  // the output dtype.

  // The backend allocator is host-visible, so constants and the causal
  // mask are written straight into fresh allocations (the Load idiom)
  // before any command references them.
  auto to_f32 = [&](const array& x) {
    array wide(x.shape(), float32, nullptr, {});
    if (x.flags().row_contiguous) {
      copy_gpu(x, wide, CopyType::Vector, s);
    } else {
      array dense = contiguous_copy_gpu(x, s);
      encoder.add_temporary(dense);
      copy_gpu(dense, wide, CopyType::Vector, s);
    }
    encoder.add_temporary(wide);
    return wide;
  };
  auto broadcast_view = [&](const array& base, Shape shape) {
    array view(std::move(shape), base.dtype(), nullptr, {});
    view.copy_shared_buffer(base, Strides(view.ndim(), 0), {true, false, false}, base.size());
    encoder.add_temporary(view);
    return view;
  };

  array q32 = to_f32(q);
  array scale_arr(scale_);
  scale_arr.set_data(allocate_omarchy(scale_arr.nbytes()));
  scale_arr.data<float>()[0] = scale_;
  encoder.add_temporary(scale_arr);
  array qs(q32.shape(), float32, nullptr, {});
  dispatch_elementwise(
      tag,
      MultiplyOperation,
      {q32, broadcast_view(scale_arr, q32.shape())},
      qs,
      s);
  encoder.add_temporary(qs);

  array k32 = to_f32(k);
  array v32 = to_f32(v);
  if (repeats > 1) {
    qs = reshape_in_eval(
        qs, Shape{batch, kv_heads, repeats, q_len, head_dim}, s);
    k32 = reshape_in_eval(k32, Shape{batch, kv_heads, 1, k_len, head_dim}, s);
    v32 = reshape_in_eval(v32, Shape{batch, kv_heads, 1, k_len, v_dim}, s);
    encoder.add_temporary(qs);
    encoder.add_temporary(k32);
    encoder.add_temporary(v32);
  }

  Shape score_shape = qs.shape();
  score_shape.back() = k_len;
  array scores(score_shape, float32, nullptr, {});
  array keys_t = swapaxes_in_eval(k32, -1, -2);
  encoder.add_temporary(keys_t);
  dispatch_matmul(tag, {qs, keys_t}, scores, 1.0f, 0.0f, false, s);
  encoder.add_temporary(scores);

  std::optional<array> masked;
  const array* sinks = has_sinks_ ? &inputs.at(inputs.size() - 1) : nullptr;
  if (sinks != nullptr && (*sinks).dtype() != float32) {
    omarchy::unsupported("attention sinks dtype " + tag, out);
  }
  const bool has_arr_mask =
      (inputs.size() == 5) || (inputs.size() == 4 && !has_sinks_);
  const bool causal_fast = do_causal_ && k_len >= q_len;
  if (!causal_fast && do_causal_) {
    // The additive causal mask holds 0 for attended positions and
    // -1e30 elsewhere: the same float32 tensor the validated
    // composition built from arange/greater_equal and
    // (1 - cast) * -1e30, without Select or repeat. Kept only for
    // k_len < q_len: a row with zero admissible keys normalizes to
    // 1/k_len under the additive mask but stores an all-zero row in
    // the softmax causal mode, so the two agree only from offset >= 0.
    array mask(Shape{q_len, k_len}, float32, nullptr, {});
    mask.set_data(allocate_omarchy(mask.nbytes()));
    float* values = mask.data<float>();
    int offset = k_len - q_len;
    for (int row = 0; row < q_len; ++row) {
      for (int col = 0; col < k_len; ++col) {
        values[row * k_len + col] = offset + row >= col ? 0.0f : -1e30f;
      }
    }
    encoder.add_temporary(mask);
    masked = array(scores.shape(), float32, nullptr, {});
    dispatch_elementwise(tag, AddOperation, {scores, mask}, *masked, s);
  } else if (has_arr_mask) {
    array mask = inputs.at(3).dtype() == float32
        ? inputs.at(3)
        : to_f32(inputs.at(3));
    mask = prepare_mask(mask, scores.shape());
    masked = array(scores.shape(), float32, nullptr, {});
    dispatch_elementwise(tag, AddOperation, {scores, mask}, *masked, s);
  }
  const array& logits = masked ? *masked : scores;
  encoder.add_temporary(logits);

  array probs(logits.shape(), float32, nullptr, {});
  dispatch_softmax(
      tag,
      logits,
      probs,
      s,
      sinks,
      q_len,
      causal_fast,
      causal_fast ? k_len - q_len : 0);
  encoder.add_temporary(probs);

  Shape result_shape = probs.shape();
  result_shape.back() = v_dim;
  array result(result_shape, float32, nullptr, {});
  dispatch_matmul(tag, {probs, v32}, result, 1.0f, 0.0f, false, s);
  encoder.add_temporary(result);
  if (result.dtype() == out.dtype()) {
    commit_result(result);
  } else {
    copy_gpu(result, out, CopyType::Vector, s);
  }
}

OMARCHY_UNSUPPORTED_MULTI(ScaledDotProductAttentionVJP)
void ConvertFP8::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  const std::string tag = name();
  array& out = outputs.at(0);
  auto s = stream();
  auto& encoder = omarchy::get_command_encoder(s);
  std::optional<array> in_temp;
  const array& in =
      ensure_dense(inputs.at(0), inputs.at(0).flags().row_contiguous, in_temp, encoder, s);
  // E4M3 payloads travel as little-endian uint32 word packs of four bytes,
  // so the byte-side array offset must stay 4-byte aligned.
  auto word_aligned = [&](const array& value) {
    return checked_item_offset(value, value.size(), tag, out) % 4 == 0;
  };
  auto kernel = omarchy::ComputeKernel::Fp8ToF32;
  if (to_fp8_) {
    if (out.dtype() != uint8 ||
        (in.dtype() != float32 && in.dtype() != float16 &&
         in.dtype() != bfloat16)) {
      omarchy::unsupported(tag + " dtype", out);
    }
    if (!word_aligned(out)) {
      omarchy::unsupported(tag + " byte alignment", out);
    }
    kernel = select_float_kernel(
        in.dtype(),
        omarchy::ComputeKernel::Fp8FromF32,
        omarchy::ComputeKernel::Fp8FromF16,
        omarchy::ComputeKernel::Fp8FromBF16);
  } else {
    if (in.dtype() != uint8 ||
        (out.dtype() != float32 && out.dtype() != float16 &&
         out.dtype() != bfloat16)) {
      omarchy::unsupported(tag + " dtype", out);
    }
    if (!word_aligned(in)) {
      omarchy::unsupported(tag + " byte alignment", out);
    }
    kernel = select_float_kernel(
        out.dtype(),
        omarchy::ComputeKernel::Fp8ToF32,
        omarchy::ComputeKernel::Fp8ToF16,
        omarchy::ComputeKernel::Fp8ToBF16);
  }
  out.set_data(allocate_omarchy(out.nbytes()));
  if (out.size() == 0) {
    return;
  }
  omarchy::ComputeParams params;
  params.count = checked_u32(in.size(), tag, out);
  params.lhs_offset = checked_item_offset(in, in.size(), tag, out);
  params.output_offset = checked_item_offset(out, out.size(), tag, out);
  std::array<omarchy::ComputeBinding, 4> bindings{
      binding(in), binding(in), binding(in), binding(out)};
  uint32_t words = checked_u32((in.size() + 3) / 4, tag, out);
  encoder.dispatch_compute(
      kernel, bindings, params, omarchy::compute_dispatch_group_count(words));
}

void Quantize::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  const std::string tag = name();
  array& out = outputs.at(0);
  auto& encoder = omarchy::get_command_encoder(out.primitive().stream());
  if (mode_ != QuantizationMode::Affine) {
    // Non-affine modes: mxfp4 (round-up e8m0 scale, fp4 e2m1 elements,
    // group 32), nvfp4 (e4m3 scale, fp4 elements, group 16, optional
    // float32 global scale), mxfp8 (round-up e8m0 scale, fp8 e4m3
    // elements, group 32). Conversions mirror the pinned Metal
    // fp4.h / fp8.h helpers bit for bit.
    const array& in_w = inputs.at(0);
    if (bits_ != 4 && bits_ != 8) {
      omarchy::unsupported(tag + " bits", out);
    }
    if (group_size_ != 16 && group_size_ != 32) {
      omarchy::unsupported(tag + " group size", out);
    }
    // ops.cpp binds the global scale to nvfp4 only. The operand count
    // that signals a bound global scale differs by direction: quantize
    // receives {w} plus the optional global scale, dequantize
    // {w, scales} plus it.
    if (!dequantize_ && inputs.size() > 1 && group_size_ != 16) {
      omarchy::unsupported(tag + " global scale", out);
    }
    if (dequantize_ && inputs.size() > 2 && group_size_ != 16) {
      omarchy::unsupported(tag + " global scale", out);
    }
    if (!dequantize_) {
      // Quantize direction: outputs {packed words, one scale byte per
      // group}. Scale bytes insert through the word view with lane
      // atomics, so rhs_offset carries the byte offset.
      array& scales = outputs.at(1);
      std::optional<array> global_scale;
      if (inputs.size() > 1) {
        global_scale = inputs.at(1);
      }
      dispatch_fp_quantize(
          tag,
          in_w,
          out,
          scales,
          global_scale,
          bits_,
          group_size_,
          out,
          stream());
      return;
    }

    // Dequantize direction: inputs {packed words, scale bytes} plus the
    // optional global scale; the output dtype is the ops-promoted
    // floating type (float16, bfloat16, or float32).
    std::optional<array> global_scale;
    if (inputs.size() > 2) {
      global_scale = inputs.at(2);
    }
    dispatch_fp_dequantize(
        tag,
        in_w,
        inputs.at(1),
        out,
        global_scale,
        bits_,
        group_size_,
        stream());
    return;
  }
  // Aligned affine packing: bits 2/4/8 pack uint32 words LSB first with
  // 32/bits values each; bits 3/5/6 pack byte-oriented - 8 values span
  // 3 bytes (bits 3), 8 values span 5 bytes (bits 5), 4 values span 3
  // bytes (bits 6) - into the same uint32 output typed words (upstream
  // CPU quantized.cpp / Metal affine_quantize). The byte stream is
  // exact whenever (group_size * bits) % 32 == 0, which the op-level
  // validation guarantees together with k % group_size == 0.
  if (bits_ != 2 && bits_ != 3 && bits_ != 4 && bits_ != 5 && bits_ != 6 &&
      bits_ != 8) {
    omarchy::unsupported(tag + " bits", out);
  }
  if (group_size_ <= 0 || (group_size_ * bits_) % 32 != 0) {
    omarchy::unsupported(tag + " group size", out);
  }
  if (!dequantize_) {
    const array& in_w = inputs.at(0);
    array& scales = outputs.at(1);
    array& biases = outputs.at(2);
    if (
        (in_w.dtype() != float16 && in_w.dtype() != bfloat16 &&
         in_w.dtype() != float32) ||
        scales.dtype() != in_w.dtype() || biases.dtype() != in_w.dtype()) {
      omarchy::unsupported(tag + " input dtype", out);
    }
    if (out.dtype() != uint32) {
      omarchy::unsupported(tag + " output dtype", out);
    }
    require_float_dtype(tag, in_w, scales, encoder);
    Shape packed_shape = in_w.shape();
    Shape parameter_shape = in_w.shape();
    packed_shape.back() = packed_shape.back() * bits_ / 32;
    parameter_shape.back() /= group_size_;
    if (
        out.shape() != packed_shape || scales.shape() != parameter_shape ||
        biases.shape() != parameter_shape) {
      omarchy::unsupported(tag + " shape", out);
    }
    std::optional<array> w_temp;
    const array& w = ensure_dense(
        in_w, in_w.flags().row_contiguous, w_temp, encoder, stream());
    out.set_data(allocate_omarchy(out.nbytes()));
    scales.set_data(allocate_omarchy(scales.nbytes()));
    biases.set_data(allocate_omarchy(biases.nbytes()));
    if (w.size() == 0) {
      return;
    }
    omarchy::ComputeParams params;
    params.count = checked_u32(w.size() / group_size_, tag, out);
    params.operation = static_cast<uint32_t>(bits_);
    params.lhs_size = checked_u32(w.size(), tag, out);
    params.rhs_size = checked_u32(scales.size(), tag, out);
    params.reduce_size = static_cast<uint32_t>(group_size_);
    params.output_size = checked_u32(out.size(), tag, out);
    params.lhs_offset = checked_item_offset(w, w.size(), tag, out);
    params.rhs_offset =
        checked_item_offset(scales, scales.size(), tag, out);
    params.output_offset = checked_item_offset(out, out.size(), tag, out);
    params.aux_size = checked_u32(biases.size(), tag, out);
    params.aux_offset = checked_item_offset(biases, biases.size(), tag, out);
    std::array<omarchy::ComputeBinding, 4> bindings{
        binding(w), binding(out), binding(scales), binding(biases)};
    auto kernel = select_float_kernel(
        in_w.dtype(),
        omarchy::ComputeKernel::QuantizeF32,
        omarchy::ComputeKernel::QuantizeF16,
        omarchy::ComputeKernel::QuantizeBF16);
    encoder.dispatch_compute(
        kernel,
        bindings,
        params,
        omarchy::compute_dispatch_group_count(params.count));
    return;
  }
  const array& in_w = inputs.at(0);
  const array& in_scales = inputs.at(1);
  const array& in_biases = inputs.at(2);
  // The output dtype is the promoted scales dtype (ops.cpp), and the
  // kernel reads the group parameters in that dtype, so mixed dtypes
  // stay a named rejection. Float64 parameters keep the named error.
  if (
      in_scales.dtype() != out.dtype() ||
      in_biases.dtype() != out.dtype() ||
      (in_scales.dtype() != float16 && in_scales.dtype() != bfloat16 &&
       in_scales.dtype() != float32)) {
    omarchy::unsupported(tag + " scales dtype", out);
  }
  require_float_dtype(tag, in_scales, out, encoder);
  if (in_w.dtype() != uint32) {
    omarchy::unsupported(tag + " weight dtype", out);
  }
  if (
      in_w.shape().size() != in_scales.shape().size() ||
      in_scales.shape() != in_biases.shape()) {
    omarchy::unsupported(tag + " scales shape", out);
  }
  size_t words_per_row = in_w.shape(-1);
  uint64_t out_columns =
      static_cast<uint64_t>(words_per_row) * 32u / bits_;
  uint64_t groups_per_row =
      static_cast<uint64_t>(in_scales.shape(-1));
  if (
      groups_per_row * group_size_ != out_columns ||
      !std::equal(
          in_w.shape().begin(),
          in_w.shape().end() - 1,
          in_scales.shape().begin())) {
    omarchy::unsupported(tag + " shape", out);
  }
  std::optional<array> w_temp;
  std::optional<array> scales_temp;
  std::optional<array> biases_temp;
  const array& w =
      ensure_dense(in_w, in_w.flags().row_contiguous, w_temp, encoder, stream());
  const array& scales = ensure_dense(
      in_scales, in_scales.flags().row_contiguous, scales_temp, encoder, stream());
  const array& biases = ensure_dense(
      in_biases, in_biases.flags().row_contiguous, biases_temp, encoder, stream());
  out.set_data(allocate_omarchy(out.nbytes()));
  if (out.size() == 0 || w.size() == 0) {
    return;
  }
  // One thread owns one packed unit: a uint32 word for bits 2/4/8, or
  // one byte pack (3 bytes for bits 3/6, 5 bytes for bits 5) inside the
  // little-endian byte stream. Every unit in a row-contiguous weight
  // maps to a unique output span, so the unit load is linear in the
  // thread index and the group parameters reuse one address for the
  // whole unit.
  bool byte_bits = (bits_ == 3 || bits_ == 5 || bits_ == 6);
  uint32_t bytes_per_pack = byte_bits ? ((bits_ == 5) ? 5u : 3u) : 4u;
  omarchy::ComputeParams params;
  params.count =
      checked_u32(w.size() * 4u / bytes_per_pack, tag, out);
  params.operation = static_cast<uint32_t>(bits_);
  params.lhs_size = params.count;
  params.rhs_size = checked_u32(scales.size(), tag, out);
  params.reduce_size = static_cast<uint32_t>(group_size_);
  params.output_size = checked_u32(out.size(), tag, out);
  params.lhs_offset = checked_item_offset(w, w.size(), tag, out);
  params.rhs_offset = checked_item_offset(scales, scales.size(), tag, out);
  params.aux_size = checked_u32(biases.size(), tag, out);
  params.aux_offset = checked_item_offset(biases, biases.size(), tag, out);
  params.output_offset = checked_item_offset(out, out.size(), tag, out);
  params.matrix_n =
      checked_u32(static_cast<uint64_t>(words_per_row) * 4u / bytes_per_pack,
          tag,
          out);
  params.matrix_k = checked_u32(groups_per_row, tag, out);
  std::array<omarchy::ComputeBinding, 4> bindings{
      binding(w), binding(scales), binding(biases), binding(out)};
  auto kernel = select_float_kernel(
      out.dtype(),
      omarchy::ComputeKernel::DequantF32,
      omarchy::ComputeKernel::DequantF16,
      omarchy::ComputeKernel::DequantBF16);
  encoder.dispatch_compute(
      kernel,
      bindings,
      params,
      omarchy::compute_dispatch_group_count(params.count));
}


} // namespace fast

namespace distributed {
OMARCHY_UNSUPPORTED_MULTI(AllReduce)
OMARCHY_UNSUPPORTED_MULTI(AllGather)
OMARCHY_UNSUPPORTED_MULTI(Send)
OMARCHY_UNSUPPORTED_MULTI(Recv)
OMARCHY_UNSUPPORTED_MULTI(ReduceScatter)
} // namespace distributed

} // namespace mlx::core
