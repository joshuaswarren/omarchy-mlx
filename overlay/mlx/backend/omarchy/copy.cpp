// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

// Buffer-level copy path. Transfer commands handle contiguous same-dtype
// ranges. Vulkan compute handles scalar, contiguous, and strided numeric
// conversions plus same-dtype copies. Packed byte destinations use atomic
// lane insertion so adjacent outputs cannot lose each other's writes.

#include "mlx/backend/omarchy/unsupported.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <execinfo.h>
#include <limits>
#include <optional>
#include <string>

#include "mlx/backend/common/copy.h"
#include "mlx/backend/common/utils.h"
#include "mlx/primitives.h"
#include "mlx/backend/gpu/copy.h"
#include "mlx/backend/omarchy/allocator.h"
#include "mlx/backend/omarchy/trace.h"
#include "mlx/backend/omarchy/compute.h"
#include "mlx/backend/omarchy/encoder.h"
#include "mlx/backend/omarchy/vulkan.h"

namespace mlx::core {

namespace {

VkBuffer buffer_handle(const array& arr) {
  auto* vbuf = static_cast<const omarchy::VulkanBuffer*>(arr.buffer().ptr());
  return vbuf->buffer;
}

bool scalar_is_zero(const array& in, int64_t item_offset) {
  const auto* bytes = in.data<char>() + item_offset * in.itemsize();
  size_t itemsize = in.itemsize();
  for (size_t i = 0; i < itemsize; ++i) {
    if (bytes[i] != 0) {
      return false;
    }
  }
  return true;
}

// Host-side read of a scalar fill value. Callers must synchronize the
// stream first so a GPU-produced scalar is visible.
float scalar_fill_value(const array& in, int64_t item_offset) {
  const auto* bytes = in.data<char>() + item_offset * in.itemsize();
  if (in.dtype() == float32) {
    float value = 0.0f;
    std::memcpy(&value, bytes, sizeof(value));
    return value;
  }
  uint16_t bits = 0;
  std::memcpy(&bits, bytes, sizeof(bits));
  if (in.dtype() == bfloat16) {
    uint32_t wide = static_cast<uint32_t>(bits) << 16;
    float value = 0.0f;
    std::memcpy(&value, &wide, sizeof(value));
    return value;
  }
  // Widen the float16 bit pattern to float32.
  uint32_t sign = static_cast<uint32_t>(bits & 0x8000u) << 16;
  uint32_t exponent = (bits >> 10) & 0x1fu;
  uint32_t mantissa = bits & 0x3ffu;
  uint32_t wide = sign;
  if (exponent == 0 && mantissa != 0) {
    // Renormalize a subnormal into the float32 exponent field.
    exponent = 113;
    while ((mantissa & 0x400u) == 0) {
      mantissa <<= 1;
      exponent--;
    }
    mantissa &= 0x3ffu;
    wide |= (exponent << 23) | (mantissa << 13);
  } else if (exponent == 0x1fu) {
    wide |= 0x7f800000u | (mantissa << 13);
  } else if (exponent != 0) {
    wide |= ((exponent + 112u) << 23) | (mantissa << 13);
  }
  float value = 0.0f;
  std::memcpy(&value, &wide, sizeof(value));
  return value;
}

// Byte offset of a copy start in the backing VkBuffer: the array's own
// buffer offset plus an explicit item offset.
VkDeviceSize byte_offset(const array& arr, int64_t item_offset) {
  return static_cast<VkDeviceSize>(arr.offset() + item_offset * arr.itemsize());
}

uint32_t checked_u32(
    size_t value,
    const std::string& name,
    const array& out) {
  if (value > std::numeric_limits<uint32_t>::max()) {
    omarchy::unsupported(name + " with more than UINT32_MAX elements", out);
  }
  return static_cast<uint32_t>(value);
}

uint32_t compute_item_offset(
    const array& value,
    int64_t item_offset,
    const std::string& name,
    const array& out) {
  if (item_offset < 0 || value.offset() % value.itemsize() != 0) {
    omarchy::unsupported(name + " byte offset", out);
  }
  uint64_t base = value.offset() / value.itemsize();
  uint64_t delta = static_cast<uint64_t>(item_offset);
  constexpr uint64_t max_index = std::numeric_limits<uint32_t>::max();
  if (base > max_index || delta > max_index - base) {
    omarchy::unsupported(name + " index span", out);
  }
  return static_cast<uint32_t>(base + delta);
}

using omarchy::binding;

array make_copy_axis_metadata(
    const Shape& shape,
    const std::vector<Strides>& strides,
    omarchy::CommandEncoder& encoder) {
  size_t rank = shape.size();
  std::vector<uint32_t> words;
  words.reserve(3 * rank);
  for (int extent : shape) {
    words.push_back(static_cast<uint32_t>(extent));
  }
  for (const auto& operand_strides : strides) {
    for (int64_t stride : operand_strides) {
      words.push_back(static_cast<uint32_t>(stride));
    }
  }
  array metadata(Shape{static_cast<int>(words.size())}, uint32, nullptr, {});
  array::Flags flags;
  flags.contiguous = true;
  flags.row_contiguous = true;
  flags.col_contiguous = true;
  metadata.set_data(
      omarchy::allocator().malloc(metadata.nbytes()),
      metadata.size(),
      Strides{1},
      flags,
      0);
  std::memcpy(metadata.data<uint32_t>(), words.data(), metadata.nbytes());
  encoder.add_temporary(metadata);
  return metadata;
}

// Fill an allocated output region at its destination offset with a
// repeated byte: whole 4-byte words on the device with vkCmdFillBuffer,
// misaligned lead and trailing bytes on the host. When edges exist,
// prior device work is drained first and the host memset happens
// immediately - never in a completion handler, or a later same-stream
// GPU submission could read stale bytes. The next submission's flush
// publishes the host writes.
void fill_pattern(
    const Stream& s,
    array& out,
    int64_t o_offset,
    uint8_t byte) {
  auto& encoder = omarchy::get_command_encoder(s);
  auto* vulkan_buffer =
      static_cast<omarchy::VulkanBuffer*>(out.buffer().ptr());
  // A fill into freshly recycled storage is the one case where the write
  // provably loses to the recycled block's in-flight previous writes
  // (convolve/pool boundary garbage, 2026-09-11-wrong-value-sweep): the
  // barrier before the fill does not re-establish what a drain provides,
  // so recycled targets drain first. Fills into fresh allocations - and
  // any fill after this buffer's first - skip the drain.
  const bool needs_drain = vulkan_buffer->recycled;
  size_t start = byte_offset(out, o_offset);
  size_t nbytes = out.nbytes();
  // vkCmdFillBuffer requires a 4-byte-aligned offset and size.
  size_t lead = (4 - start % 4) % 4;
  if (lead > nbytes) {
    lead = nbytes;
  }
  size_t words_bytes = (nbytes - lead) & ~size_t(3);
  size_t tail = nbytes - lead - words_bytes;
  if (lead + tail > 0) {
    encoder.synchronize();
    auto* base = static_cast<uint8_t*>(out.buffer().raw_ptr());
    std::memset(base + start, byte, lead);
    std::memset(base + start + lead + words_bytes, byte, tail);
  }
  if (words_bytes > 0) {
    if (needs_drain) {
      // The recycled block's previous occupant may still have writes in
      // flight on another stream; order this fill's submission after all
      // submitted work through the device completion timeline. Unlike the
      // host drain this replaces, the wait rides the fill's own
      // submission: no host stall, and back-to-back fills land in one
      // submission. The host-written edge bytes above still need the
      // full drain and keep it.
      encoder.wait_outstanding_submissions();
    }
    encoder.add_temporary(out);
    encoder.fill_buffer(
        buffer_handle(out), 0x01010101u * byte, words_bytes, start + lead);
  }
  vulkan_buffer->recycled = false;
}

omarchy::ComputeKernel fill_kernel(Dtype dtype) {
  if (dtype == float16) {
    return omarchy::ComputeKernel::FillF16;
  }
  if (dtype == bfloat16) {
    return omarchy::ComputeKernel::FillBF16;
  }
  if (dtype == complex64) {
    // Complex64Transport: vec2 element, the scalar's real part rides
    // in alpha and the imaginary part in beta.
    return omarchy::ComputeKernel::FillComplex64;
  }
  return omarchy::ComputeKernel::FillF32;
}

omarchy::ComputeKernel copy_general_kernel(Dtype dtype, const array& out) {
  if (dtype == bool_) {
    // Packed word transport: 4 bools per word, byte-lane atomics in
    // the shader (scatter.comp shape).
    return omarchy::ComputeKernel::CopyGeneralBool;
  }
  if (dtype == int8 || dtype == uint8) {
    // Packed word transport: 4 raw bytes per word, the bool transport
    // minus canonicalization.
    return omarchy::ComputeKernel::CopyGeneralU8;
  }
  if (dtype == float16) {
    return omarchy::ComputeKernel::CopyGeneralF16;
  }
  if (dtype == bfloat16) {
    return omarchy::ComputeKernel::CopyGeneralBF16;
  }
  if (dtype == int16 || dtype == uint16) {
    // uint16_t storage, the bfloat16 shape.
    return omarchy::ComputeKernel::CopyGeneralU16;
  }
  if (dtype == int32 || dtype == uint32) {
    // Raw-word copies: the same 4-byte stride math as float32 with no
    // float conversion, so packed words stay bit-exact.
    return omarchy::ComputeKernel::CopyGeneralU32;
  }
  if (dtype == complex64) {
    // Complex64Transport: a vec2 element copies bit-exact with no
    // conversion, and the 8-byte stride math is the item math the
    // params already carry.
    return omarchy::ComputeKernel::CopyGeneralComplex64;
  }
  if (dtype == int64 || dtype == uint64) {
    // One 8-byte item per element, the complex64 stride shape.
    return omarchy::ComputeKernel::CopyGeneralU64;
  }
  if (dtype == float32) {
    return omarchy::ComputeKernel::CopyGeneralF32;
  }
  omarchy::unsupported("strided copy dtype", out);
}

// Element-width class of the integer family: 1-byte (bool, int8,
// uint8), 2-byte (int16, uint16), 4-byte (int32, uint32), 8-byte
// (int64, uint64). 0 for every other dtype.
int int_width(Dtype dtype) {
  switch (dtype) {
    case bool_:
    case int8:
    case uint8:
      return 1;
    case int16:
    case uint16:
      return 2;
    case int32:
    case uint32:
      return 4;
    case int64:
    case uint64:
      return 8;
    default:
      return 0;
  }
}

int cast_width(Dtype dtype) {
  int width = int_width(dtype);
  if (width != 0) {
    return width;
  }
  if (dtype == float16 || dtype == bfloat16) {
    return 2;
  }
  if (dtype == float32) {
    return 4;
  }
  if (dtype == complex64) {
    return 8;
  }
  return 0;
}

// Dtype codes cast_int.comp understands. The pair rides in
// params.operation as source | (destination << 16).
uint32_t cast_code(Dtype dtype) {
  switch (dtype) {
    case bool_:
      return 0;
    case uint8:
      return 1;
    case int8:
      return 2;
    case uint16:
      return 3;
    case int16:
      return 4;
    case uint32:
      return 5;
    case int32:
      return 6;
    case uint64:
      return 7;
    case int64:
      return 8;
    case float32:
      return 9;
    case float16:
      return 10;
    case bfloat16:
      return 11;
    case complex64:
      return 12;
    default:
      return 0xFFFFFFFFu;
  }
}

// One numeric cast kernel per source/destination storage-width pair.
// Runtime dtype codes preserve signedness and float/complex semantics.
// Eight-byte blobs need shaderInt64; two-byte blobs need 16-bit storage.
std::optional<omarchy::ComputeKernel> cast_numeric_kernel(
    Dtype in_dtype,
    Dtype out_dtype,
    const omarchy::CapabilityReport& capabilities) {
  int sw = cast_width(in_dtype);
  int dw = cast_width(out_dtype);
  if (sw == 0 || dw == 0) {
    return std::nullopt;
  }
  if ((sw == 8 || dw == 8) && !capabilities.shader_int64) {
    return std::nullopt;
  }
  if ((sw == 2 || dw == 2) &&
      (!capabilities.storage_buffer_16bit_access ||
       !capabilities.shader_int16)) {
    return std::nullopt;
  }
  static constexpr omarchy::ComputeKernel kTable[4][4] = {
      {omarchy::ComputeKernel::CastIntW1W1,
       omarchy::ComputeKernel::CastIntW1W2,
       omarchy::ComputeKernel::CastIntW1W4,
       omarchy::ComputeKernel::CastIntW1W8},
      {omarchy::ComputeKernel::CastIntW2W1,
       omarchy::ComputeKernel::CastIntW2W2,
       omarchy::ComputeKernel::CastIntW2W4,
       omarchy::ComputeKernel::CastIntW2W8},
      {omarchy::ComputeKernel::CastIntW4W1,
       omarchy::ComputeKernel::CastIntW4W2,
       omarchy::ComputeKernel::CastIntW4W4,
       omarchy::ComputeKernel::CastIntW4W8},
      {omarchy::ComputeKernel::CastIntW8W1,
       omarchy::ComputeKernel::CastIntW8W2,
       omarchy::ComputeKernel::CastIntW8W4,
       omarchy::ComputeKernel::CastIntW8W8},
  };
  auto col = [](int width) {
    return width == 1 ? 0 : width == 2 ? 1 : width == 4 ? 2 : 3;
  };
  return kTable[col(sw)][col(dw)];
}

// Storage-buffer access requirements for a 16-bit float dtype, shared by
// the fill and strided-copy kernels.
void require_float_storage(
    const std::string& name,
    Dtype dtype,
    const array& out,
    omarchy::CommandEncoder& encoder) {
  const auto& capabilities = encoder.device().capabilities();
  if ((dtype == float16 &&
       (!capabilities.shader_float16 ||
        !capabilities.storage_buffer_16bit_access)) ||
      (dtype == bfloat16 &&
       (!capabilities.storage_buffer_16bit_access ||
        !capabilities.shader_int16))) {
    omarchy::unsupported(name + " capability", out);
  }
}

} // namespace

void copy_gpu_inplace(
    const array& in,
    array& out,
    const Shape& data_shape,
    const Strides& i_strides,
    const Strides& o_strides,
    int64_t i_offset,
    int64_t o_offset,
    CopyType ctype,
    const Stream& s,
    std::optional<array> dynamic_i_offset,
    std::optional<array> dynamic_o_offset) {
  // The only producer of these optionals is the shared GPU
  // DynamicSlice/DynamicSliceUpdate eval, which builds them through
  // compute_dynamic_offset: that helper synchronizes the stream and
  // writes one int64 into host-visible storage before this copy is
  // scheduled. Reading the scalar here is a synchronized host read, so
  // the offset folds into the item offset and the copy takes the same
  // strided path as any other slice.
  if (dynamic_i_offset) {
    i_offset += *dynamic_i_offset->data<int64_t>();
  }
  if (dynamic_o_offset) {
    o_offset += *dynamic_o_offset->data<int64_t>();
  }

  if (out.nbytes() == 0) {
    return;
  }

  auto& encoder = omarchy::get_command_encoder(s);
  if (ctype == CopyType::Scalar) {
    if (in.has_primitive()) {
      omarchy::unsupported("GPU-in-flight scalar fill", out);
    }
    // The scalar's bytes are read on the host by everything below (the
    // zero check, the value reads, the dtype-converting scratch copy).
    // When those bytes are GPU-produced and possibly unpublished - a
    // pad's astype(pad_value) cast, or any scheduled scalar that has not
    // drained - a host read races the producer (3f7e549a regression:
    // pad and concat filled whole regions with recycled garbage read as
    // the fill value) and a synchronize to close the race costs the
    // prefill path its pipelining. So for an unpublished device-produced
    // scalar the value is never read on the host at all: it is
    // broadcast over the output with a zero-stride strided copy, which
    // consumes the value on the device behind the per-dispatch barrier
    // like any other dependent dispatch. Host-written scalars
    // (completion == 0) take the fast paths below with no extra work.
    const auto* in_buffer =
        static_cast<const omarchy::VulkanBuffer*>(in.buffer().ptr());
    if (in_buffer->completion != 0 &&
        in_buffer->completion >
            encoder.device().completions().drained_value()) {
      array broadcast_view(
          Shape(out.shape()), in.dtype(), nullptr, {});
      broadcast_view.copy_shared_buffer(
          in, Strides(broadcast_view.ndim(), 0), {true, false, false},
          in.size());
      encoder.add_temporary(broadcast_view);
      copy_gpu_inplace(
          broadcast_view,
          out,
          broadcast_view.shape(),
          broadcast_view.strides(),
          out.strides(),
          /*i_offset=*/0,
          o_offset,
          CopyType::GeneralGeneral,
          s);
      return;
    }
    if (scalar_is_zero(in, i_offset)) {
      fill_pattern(s, out, o_offset, 0);
      return;
    }
    if (in.dtype() != out.dtype()) {
      const auto& capabilities = encoder.device().capabilities();
      auto kernel = cast_numeric_kernel(in.dtype(), out.dtype(), capabilities);
      if (!kernel) {
        omarchy::unsupported("dtype converting copy", out);
      }
      array scalar({1}, in.dtype(), nullptr, {});
      scalar.set_data(omarchy::allocator().malloc(scalar.nbytes()));
      auto* scalar_buffer =
          static_cast<omarchy::VulkanBuffer*>(scalar.buffer().ptr());
      if (scalar_buffer->recycled) {
        // Host write into freshly recycled storage: only a host drain
        // orders it against the block's in-flight previous writes. The
        // malloc hands recycled blocks out, so this stays reachable; the
        // common fresh-allocation case skips the stall entirely.
        encoder.synchronize();
      }
      std::memcpy(
          scalar.data<char>(),
          in.data<char>() + i_offset * in.itemsize(),
          in.itemsize());
      encoder.add_temporary(scalar);
      uint32_t count = checked_u32(out.data_size(), "dtype converting copy", out);
      omarchy::ComputeParams params;
      params.count = count;
      params.operation = cast_code(in.dtype()) |
          (cast_code(out.dtype()) << 16);
      params.lhs_offset = 0;
      params.output_offset =
          compute_item_offset(out, o_offset, "dtype converting copy", out);
      params.output_size = count;
      if (!omarchy::compute_index_span_fits(params.output_offset, count) ||
          (out.itemsize() == 8 &&
           !omarchy::compute_index_span_fits(
               2ull * params.output_offset, 2ull * count))) {
        omarchy::unsupported("dtype converting copy index span", out);
      }
      params.flags = 1;
      std::array<omarchy::ComputeBinding, 4> bindings{
          binding(scalar),
          binding(scalar),
          binding(out),
          binding(out)};
      encoder.dispatch_compute(
          *kernel,
          bindings,
          params,
          omarchy::compute_dispatch_group_count(count));
      return;
    }
    if (in.itemsize() == 1) {
      // Byte-packed fills: bool stores canonical 0/1 bytes and the
      // 8-bit ints store their raw byte, so a repeated-byte word
      // pattern covers the whole region. The scalar is non-zero here.
      if (in.dtype() != out.dtype()) {
        omarchy::unsupported("non-zero scalar fill dtype", out);
      }
      fill_pattern(s, out, o_offset, in.data<char>()[i_offset]);
      return;
    }
    if (in.dtype() == int32 || in.dtype() == uint32) {
      if (in.dtype() != out.dtype()) {
        omarchy::unsupported("non-zero scalar fill dtype", out);
      }
      uint32_t count = checked_u32(out.data_size(), "scalar fill", out);
      uint32_t word = 0;
      std::memcpy(
          &word, in.data<char>() + i_offset * in.itemsize(), sizeof(word));
      VkDeviceSize start = static_cast<VkDeviceSize>(
          compute_item_offset(out, o_offset, "scalar fill", out)) *
          sizeof(uint32_t);
      encoder.add_temporary(out);
      encoder.fill_buffer(
          buffer_handle(out), word, count * sizeof(uint32_t), start);
      return;
    }
    if (in.dtype() == int16 || in.dtype() == uint16) {
      if (in.dtype() != out.dtype()) {
        omarchy::unsupported("non-zero scalar fill dtype", out);
      }
      if (!encoder.device().capabilities().storage_buffer_16bit_access ||
          !encoder.device().capabilities().shader_int16) {
        omarchy::unsupported("non-zero scalar fill 16-bit capability", out);
      }
      uint32_t count = checked_u32(out.data_size(), "scalar fill", out);
      omarchy::ComputeParams params;
      params.count = count;
      params.output_size = count;
      params.output_offset =
          compute_item_offset(out, o_offset, "scalar fill", out);
      if (!omarchy::compute_index_span_fits(params.output_offset, count)) {
        omarchy::unsupported("scalar fill index span", out);
      }
      // The raw halfword rides the alpha slot's low 16 bits.
      uint16_t half = 0;
      std::memcpy(
          &half, in.data<char>() + i_offset * in.itemsize(), sizeof(half));
      uint32_t word = half;
      std::memcpy(&params.alpha, &word, sizeof(float));
      std::array<omarchy::ComputeBinding, 1> bindings{binding(out)};
      encoder.dispatch_compute(
          omarchy::ComputeKernel::FillU16,
          bindings,
          params,
          omarchy::compute_dispatch_group_count(count));
      return;
    }
    if (in.dtype() == int64 || in.dtype() == uint64) {
      if (in.dtype() != out.dtype()) {
        omarchy::unsupported("non-zero scalar fill dtype", out);
      }
      if (!encoder.device().capabilities().shader_int64) {
        omarchy::unsupported("non-zero scalar fill int64 capability", out);
      }
      uint32_t count = checked_u32(out.data_size(), "scalar fill", out);
      omarchy::ComputeParams params;
      params.count = count;
      params.output_size = count;
      params.output_offset =
          compute_item_offset(out, o_offset, "scalar fill", out);
      if (!omarchy::compute_index_span_fits(
              2ull * params.output_offset, 2ull * count)) {
        omarchy::unsupported("scalar fill index span", out);
      }
      // The two little-endian words ride the float push-constant slots
      // bit-exactly: every uint32 is a valid float bit pattern.
      uint32_t words[2] = {0, 0};
      std::memcpy(
          words, in.data<char>() + i_offset * in.itemsize(), sizeof(words));
      std::memcpy(&params.alpha, &words[0], sizeof(float));
      std::memcpy(&params.beta, &words[1], sizeof(float));
      std::array<omarchy::ComputeBinding, 1> bindings{binding(out)};
      encoder.dispatch_compute(
          omarchy::ComputeKernel::FillU64,
          bindings,
          params,
          omarchy::compute_dispatch_group_count(count));
      return;
    }
    if (in.dtype() != float32 && in.dtype() != float16 &&
        in.dtype() != bfloat16 && in.dtype() != complex64) {
      omarchy::unsupported("non-zero scalar fill", out);
    }
    require_float_storage("non-zero scalar fill", in.dtype(), out, encoder);
    uint32_t count = checked_u32(out.data_size(), "scalar fill", out);
    omarchy::ComputeParams params;
    params.count = count;
    params.output_size = count;
    params.output_offset =
        compute_item_offset(out, o_offset, "scalar fill", out);
    if (in.dtype() == complex64) {
      // Complex64Transport: the fill kernel writes vec2(alpha, beta),
      // so the scalar's two float32 words ride in alpha and beta.
      // scalar_fill_value would decode the real part as float16 bits,
      // so both components come from the raw 8 bytes instead.
      float parts[2] = {0.0f, 0.0f};
      std::memcpy(
          parts,
          in.data<char>() + i_offset * in.itemsize(),
          sizeof(parts));
      params.alpha = parts[0];
      params.beta = parts[1];
    } else {
      params.alpha = scalar_fill_value(in, i_offset);
    }
    std::array<omarchy::ComputeBinding, 1> bindings{binding(out)};
    encoder.dispatch_compute(
        fill_kernel(in.dtype()),
        bindings,
        params,
        omarchy::compute_dispatch_group_count(count));
    return;
  }

  if (ctype == CopyType::General || ctype == CopyType::GeneralGeneral) {
    if (cast_width(in.dtype()) == 0 || cast_width(out.dtype()) == 0) {
      omarchy::unsupported("strided copy", out);
    }
    if (in.dtype() == out.dtype()) {
      require_float_storage("strided copy", in.dtype(), out, encoder);
    }
    const auto& capabilities = encoder.device().capabilities();
    if ((in.dtype() == int16 || in.dtype() == uint16) &&
        (!capabilities.storage_buffer_16bit_access ||
         !capabilities.shader_int16)) {
      omarchy::unsupported("strided copy 16-bit capability", out);
    }
    if ((in.dtype() == int64 || in.dtype() == uint64) &&
        !capabilities.shader_int64) {
      omarchy::unsupported("strided copy int64 capability", out);
    }
    auto [collapsed_shape, collapsed_strides] = collapse_contiguous_dims(
        data_shape, std::vector<Strides>{i_strides, o_strides});
    size_t rank = collapsed_shape.size();
    size_t total = 1;
    for (size_t axis = 0; axis < rank; ++axis) {
      total *= static_cast<size_t>(collapsed_shape[axis]);
    }
    uint32_t count = checked_u32(total, "strided copy", out);
    omarchy::ComputeParams params;
    params.count = count;
    params.dims = static_cast<uint32_t>(rank);
    // Signed span: strides may be negative (a flip carries its base at
    // the far end), and the shader indexes with int math, so every
    // intermediate index must land in [0, INT32_MAX] per buffer.
    int64_t in_lo = 0;
    int64_t in_hi = 0;
    int64_t out_lo = 0;
    int64_t out_hi = 0;
    for (size_t axis = 0; axis < rank; ++axis) {
      uint32_t extent = static_cast<uint32_t>(collapsed_shape[axis]);
      uint32_t in_stride =
          static_cast<uint32_t>(collapsed_strides[0][axis]);
      uint32_t out_stride =
          static_cast<uint32_t>(collapsed_strides[1][axis]);
      if (rank <= 4) {
        params.shape[axis] = extent;
        params.in_strides[axis] = in_stride;
        params.out_strides[axis] = out_stride;
      }
      int64_t distance = static_cast<int64_t>(extent) - 1;
      int64_t ist = static_cast<int64_t>(static_cast<int32_t>(in_stride));
      int64_t ost = static_cast<int64_t>(static_cast<int32_t>(out_stride));
      in_lo += std::min<int64_t>(0, ist * distance);
      in_hi += std::max<int64_t>(0, ist * distance);
      out_lo += std::min<int64_t>(0, ost * distance);
      out_hi += std::max<int64_t>(0, ost * distance);
    }
    params.lhs_offset =
        compute_item_offset(in, i_offset, "strided copy", out);
    params.output_offset =
        compute_item_offset(out, o_offset, "strided copy", out);
    if (in_lo < -static_cast<int64_t>(params.lhs_offset) ||
        in_hi >
            std::numeric_limits<int32_t>::max() -
                static_cast<int64_t>(params.lhs_offset) ||
        out_lo < -static_cast<int64_t>(params.output_offset) ||
        out_hi >
            std::numeric_limits<int32_t>::max() -
                static_cast<int64_t>(params.output_offset)) {
      omarchy::unsupported("strided copy index span", out);
    }
    std::optional<array> axis_metadata;
    if (rank > 4) {
      axis_metadata =
          make_copy_axis_metadata(collapsed_shape, collapsed_strides, encoder);
      params.matrix_k = static_cast<uint32_t>(rank);
    }
    // A > 2 GiB flat contiguous copy (collapsed to one axis, unit
    // strides; includes dtype-converting casts) splits by flat index:
    // each window binds the source and destination bytes [c0, c1).
    // Strided copies keep today's single dispatch.
    {
      const auto& copy_caps = encoder.device().capabilities();
      const VkDeviceSize copy_limit = copy_caps.max_storage_buffer_range;
      const uint32_t copy_alignment =
          copy_caps.min_storage_buffer_offset_alignment;
      const bool flat_copy = rank == 1 && params.shape[0] == count &&
          params.in_strides[0] == 1u && params.out_strides[0] == 1u;
      const uint64_t copy_bytes =
          static_cast<uint64_t>(count) * out.itemsize();
      if (flat_copy && copy_bytes > copy_limit) {
        const uint64_t copy_chunk = std::max<uint64_t>(
            (copy_limit / std::max(in.itemsize(), out.itemsize()) / 16u) *
                16u,
            16u);
        omarchy::ComputeKernel window_kernel;
        if (in.dtype() == out.dtype()) {
          window_kernel = copy_general_kernel(in.dtype(), out);
        } else {
          auto cast_kernel =
              cast_numeric_kernel(in.dtype(), out.dtype(), capabilities);
          if (!cast_kernel) {
            omarchy::unsupported("dtype converting copy", out);
          }
          window_kernel = *cast_kernel;
        }
        const uint32_t in_base = params.lhs_offset;
        const uint32_t out_base = params.output_offset;
        for (uint64_t c0 = 0; c0 < count; c0 += copy_chunk) {
          const uint64_t c1 =
              std::min(c0 + copy_chunk, static_cast<uint64_t>(count));
          omarchy::ComputeParams wparams = params;
          wparams.count = checked_u32(c1 - c0, "strided copy", out);
          wparams.shape[0] = wparams.count;
          if (in.dtype() != out.dtype()) {
            wparams.operation = cast_code(in.dtype()) |
                (cast_code(out.dtype()) << 16);
            wparams.flags = 2;
          }
          const uint64_t in_first =
              (static_cast<uint64_t>(in_base) + c0) * in.itemsize();
          const uint64_t in_last =
              (static_cast<uint64_t>(in_base) + c1) * in.itemsize();
          const uint64_t dst_first =
              (static_cast<uint64_t>(out_base) + c0) * out.itemsize();
          const uint64_t dst_last =
              (static_cast<uint64_t>(out_base) + c1) * out.itemsize();
          wparams.lhs_offset = in_base + static_cast<uint32_t>(c0) -
              omarchy::window_item_correction(
                  in_first, copy_alignment, in.itemsize());
          wparams.output_offset = out_base + static_cast<uint32_t>(c0) -
              omarchy::window_item_correction(
                  dst_first, copy_alignment, out.itemsize());
          const omarchy::ComputeBinding in_window = omarchy::window_binding(
              in, in_first, in_last, copy_alignment);
          std::array<omarchy::ComputeBinding, 4> wbindings{
              in_window,
              in_window,
              omarchy::window_binding(
                  out, dst_first, dst_last, copy_alignment),
              in_window};
          encoder.dispatch_compute(
              window_kernel,
              wbindings,
              wparams,
              omarchy::compute_dispatch_group_count(wparams.count));
        }
        return;
      }
    }
    std::array<omarchy::ComputeBinding, 4> bindings{
        binding(in),
        binding(in),
        binding(out),
        binding(axis_metadata ? *axis_metadata : out)};
    omarchy::ComputeKernel kernel;
    if (in.dtype() == out.dtype()) {
      kernel = copy_general_kernel(in.dtype(), out);
      static int copy_trace = 0;
      if (in.dtype() == bfloat16 &&
          std::getenv("MLX_OMARCHY_COPY_TRACE") != nullptr &&
          copy_trace < 4000) {
        ++copy_trace;
        std::fprintf(
            stderr,
            "[copyBF16] bytes=%zu out_shape=", out.nbytes());
        for (auto d : out.shape()) {
          std::fprintf(stderr, "%d,", static_cast<int>(d));
        }
        std::fprintf(stderr, " in_prim=%s consumer=%s in_shape=",
                     in.has_primitive() ? in.primitive().name() : "<none>",
                     std::string(omarchy::trace::current_prim()).c_str());
        for (auto d : in.shape()) {
          std::fprintf(stderr, "%d,", static_cast<int>(d));
        }
        std::fprintf(
            stderr,
            " in_rowc=%d in_strides=",
            static_cast<int>(in.flags().row_contiguous));
        for (auto s : in.strides()) {
          std::fprintf(stderr, "%lld,", static_cast<long long>(s));
        }
        std::fprintf(stderr, "\n");
        if (copy_trace < 40) {
          void* frames[12];
          int n = backtrace(frames, 12);
          backtrace_symbols_fd(frames, n, 2);
        }
      }
    } else {
      auto cast_kernel = cast_numeric_kernel(in.dtype(), out.dtype(), capabilities);
      if (!cast_kernel) {
        omarchy::unsupported("dtype converting copy", out);
      }
      kernel = *cast_kernel;
      params.operation = cast_code(in.dtype()) |
          (cast_code(out.dtype()) << 16);
      params.flags = 2;
    }
    encoder.dispatch_compute(
        kernel,
        bindings,
        params,
        omarchy::compute_dispatch_group_count(count));
    return;
  }

  if (in.dtype() != out.dtype()) {
    const auto& capabilities = encoder.device().capabilities();
    omarchy::ComputeKernel kernel;
    bool numeric_transport = false;
    if (in.dtype() == bool_ && out.dtype() == float32) {
      kernel = omarchy::ComputeKernel::CastBoolF32;
    } else if (in.dtype() == bool_ && out.dtype() == int32) {
      kernel = omarchy::ComputeKernel::CastBoolI32;
    } else if (in.dtype() == bool_ && out.dtype() == float16) {
      kernel = omarchy::ComputeKernel::CastBoolF16;
    } else if (in.dtype() == bool_ && out.dtype() == bfloat16) {
      kernel = omarchy::ComputeKernel::CastBoolBF16;
    } else if (in.dtype() == float16 && out.dtype() == float32) {
      kernel = omarchy::ComputeKernel::CastF16F32;
    } else if (in.dtype() == float32 && out.dtype() == float16) {
      kernel = omarchy::ComputeKernel::CastF32F16;
    } else if (in.dtype() == bfloat16 && out.dtype() == float32) {
      kernel = omarchy::ComputeKernel::CastBF16F32;
    } else if (in.dtype() == float32 && out.dtype() == bfloat16) {
      kernel = omarchy::ComputeKernel::CastF32BF16;
    } else if (in.dtype() == bfloat16 && out.dtype() == float16) {
      kernel = omarchy::ComputeKernel::CastBF16F16;
    } else if (in.dtype() == float16 && out.dtype() == bfloat16) {
      kernel = omarchy::ComputeKernel::CastF16BF16;
    } else if (in.dtype() == int32 && out.dtype() == float32) {
      kernel = omarchy::ComputeKernel::CastI32F32;
    } else if (in.dtype() == float32 && out.dtype() == int32) {
      kernel = omarchy::ComputeKernel::CastF32I32;
    } else if (in.dtype() == int32 && out.dtype() == float16) {
      kernel = omarchy::ComputeKernel::CastI32F16;
    } else if (in.dtype() == float16 && out.dtype() == int32) {
      kernel = omarchy::ComputeKernel::CastF16I32;
    } else if (in.dtype() == int32 && out.dtype() == bfloat16) {
      kernel = omarchy::ComputeKernel::CastI32BF16;
    } else if (in.dtype() == bfloat16 && out.dtype() == int32) {
      kernel = omarchy::ComputeKernel::CastBF16I32;
    } else if (in.dtype() == uint32 && out.dtype() == float32) {
      kernel = omarchy::ComputeKernel::CastU32F32;
    } else if (in.dtype() == float32 && out.dtype() == complex64) {
      kernel = omarchy::ComputeKernel::CastF32Complex64;
    } else if (in.dtype() == int32 && out.dtype() == complex64) {
      kernel = omarchy::ComputeKernel::CastI32Complex64;
    } else if (in.dtype() == uint32 && out.dtype() == complex64) {
      kernel = omarchy::ComputeKernel::CastU32Complex64;
    } else if (in.dtype() == bool_ && out.dtype() == complex64) {
      kernel = omarchy::ComputeKernel::CastBoolComplex64;
    } else if (in.dtype() == float16 && out.dtype() == complex64) {
      kernel = omarchy::ComputeKernel::CastF16Complex64;
    } else if (in.dtype() == bfloat16 && out.dtype() == complex64) {
      kernel = omarchy::ComputeKernel::CastBF16Complex64;
    } else if (in.dtype() == complex64 && out.dtype() == float32) {
      kernel = omarchy::ComputeKernel::CastComplex64F32;
    } else if (
        auto int_kernel = cast_numeric_kernel(in.dtype(), out.dtype(),
                                          capabilities)) {
      kernel = *int_kernel;
      numeric_transport = true;
    } else {
      omarchy::unsupported("dtype converting copy", out);
    }

    if ((in.dtype() == float16 || out.dtype() == float16) &&
        (!capabilities.shader_float16 ||
         !capabilities.storage_buffer_16bit_access)) {
      omarchy::unsupported("dtype converting copy float16 capability", out);
    }
    if ((in.dtype() == bfloat16 || out.dtype() == bfloat16) &&
        (!capabilities.storage_buffer_16bit_access ||
         !capabilities.shader_int16)) {
      omarchy::unsupported("dtype converting copy bfloat16 capability", out);
    }

    uint32_t count =
        checked_u32(out.data_size(), "dtype converting copy", out);
    omarchy::ComputeParams params;
    params.count = count;
    params.lhs_size =
        checked_u32(in.data_size(), "dtype converting copy", out);
    params.rhs_size = params.lhs_size;
    params.output_size = count;
    params.operation =
        cast_code(in.dtype()) | (cast_code(out.dtype()) << 16);
    params.lhs_offset =
        compute_item_offset(in, i_offset, "dtype converting copy", out);
    params.output_offset =
        compute_item_offset(out, o_offset, "dtype converting copy", out);
    if (!omarchy::compute_index_span_fits(params.lhs_offset, count) ||
        !omarchy::compute_index_span_fits(params.output_offset, count)) {
      omarchy::unsupported("dtype converting copy index span", out);
    }
    if (in.itemsize() == 8 || out.itemsize() == 8) {
      // The shader addresses 64-bit items as little-endian word pairs,
      // so the word span 2*(offset+count) must stay in uint32 range.
      if (!omarchy::compute_index_span_fits(
              2ull * params.lhs_offset, 2ull * count) ||
          !omarchy::compute_index_span_fits(
              2ull * params.output_offset, 2ull * count)) {
        omarchy::unsupported("dtype converting copy index span", out);
      }
    }
    // The legacy bool-source kernels dispatch one thread per word
    // window (cast.comp SOURCE_BOOL shape); the cast_int blobs run one
    // thread per item.
    const bool source_bool_kernel =
        kernel == omarchy::ComputeKernel::CastBoolF32 ||
        kernel == omarchy::ComputeKernel::CastBoolI32 ||
        kernel == omarchy::ComputeKernel::CastBoolF16 ||
        kernel == omarchy::ComputeKernel::CastBoolBF16 ||
        kernel == omarchy::ComputeKernel::CastBoolComplex64;
    uint32_t dispatch_count = source_bool_kernel
        ? checked_u32(
              (static_cast<uint64_t>(count) +
               static_cast<uint64_t>(params.lhs_offset & 3u) + 3u) /
                  4u,
              "dtype converting copy",
              out)
        : count;
    if (numeric_transport) {
      std::array<omarchy::ComputeBinding, 4> bindings{
          binding(in),
          binding(in),
          binding(out),
          binding(out)};
      encoder.dispatch_compute(
          kernel,
          bindings,
          params,
          omarchy::compute_dispatch_group_count(dispatch_count));
    } else {
      std::array<omarchy::ComputeBinding, 3> bindings{
          binding(in), binding(in), binding(out)};
      // Flat casts read and write 1:1 by item, so when the whole buffers
      // exceed the storage range the same per-window scheme as
      // dispatch_elementwise_windows applies — but this kernel family
      // exposes only three descriptor slots, so the loop lives here
      // instead of in that helper. In-limit sizes keep the single
      // dispatch below unchanged.
      const auto& caps = encoder.device().capabilities();
      const VkDeviceSize range_limit = caps.max_storage_buffer_range;
      const uint32_t alignment = caps.min_storage_buffer_offset_alignment;
      if (static_cast<uint64_t>(out.nbytes()) > range_limit &&
          !source_bool_kernel) {
        const size_t in_item = in.itemsize();
        const size_t out_item = out.itemsize();
        // Chunk in OUT items, but the input window spans the same items at
        // in.itemsize() bytes — divide by the LARGER itemsize or the input
        // window alone exceeds the storage range on widening casts.
        const uint64_t chunk = std::max<uint64_t>(
            (range_limit / std::max(in_item, out_item) / 16u) * 16u, 16u);
        for (uint64_t c0 = 0; c0 < count; c0 += chunk) {
          const uint64_t c1 = std::min(c0 + chunk, static_cast<uint64_t>(count));
          const uint64_t in_first =
              (static_cast<uint64_t>(params.lhs_offset) + c0) * in_item;
          const uint64_t in_last =
              (static_cast<uint64_t>(params.lhs_offset) + c1) * in_item;
          const uint64_t out_first =
              (static_cast<uint64_t>(params.output_offset) + c0) * out_item;
          const uint64_t out_last =
              (static_cast<uint64_t>(params.output_offset) + c1) * out_item;
          omarchy::ComputeParams wparams = params;
          wparams.count = checked_u32(c1 - c0, "dtype converting copy", out);
          wparams.lhs_offset =
              params.lhs_offset + static_cast<uint32_t>(c0) -
              omarchy::window_item_correction(in_first, alignment, in_item);
          wparams.output_offset =
              params.output_offset + static_cast<uint32_t>(c0) -
              omarchy::window_item_correction(out_first, alignment, out_item);
          std::array<omarchy::ComputeBinding, 3> wbindings{
              omarchy::window_binding(in, in_first, in_last, alignment),
              omarchy::window_binding(in, in_first, in_last, alignment),
              omarchy::window_binding(out, out_first, out_last, alignment)};
          encoder.dispatch_compute(
              kernel,
              wbindings,
              wparams,
              omarchy::compute_dispatch_group_count(wparams.count));
        }
        return;
      }
      encoder.dispatch_compute(
          kernel,
          bindings,
          params,
          omarchy::compute_dispatch_group_count(dispatch_count));
    }
    return;
  }

  encoder.add_temporary(in);
  encoder.add_temporary(out);
  encoder.copy_buffer(
      buffer_handle(in),
      buffer_handle(out),
      out.nbytes(),
      byte_offset(in, i_offset),
      byte_offset(out, o_offset));
}

void copy_gpu(const array& input, array& out, CopyType ctype, const Stream& s) {
  // Vector and dtype-converting copies read the source flat, in storage
  // order. That is only the logical order when the flags honestly
  // describe a dense layout. Two contiguous-flagged views violate it:
  // a transposed view of a contiguous array (data_size == size but
  // storage order differs from logical order, so astype returned the
  // source order), and a stride-0 broadcast view (data_size smaller
  // than size, so full_like with an array fill and a dtype cast read
  // past the one-element buffer and zero-filled everything after the
  // first element). Both report contiguous=true under the span-based
  // definition and row_contiguous=false, so row_contiguous alone
  // decides; the data_size relation differs between the two and must
  // not be part of the test. Materialize through a same-dtype strided
  // copy first; the flat op then reads a dense buffer whose storage
  // order is the logical order and whose allocation covers the whole
  // shape.
  std::optional<array> dense;
  const array* in = &input;
  bool flat_only =
      (ctype == CopyType::Vector || input.dtype() != out.dtype()) &&
      !input.flags().row_contiguous;
  if (flat_only) {
    dense = array(input.shape(), input.dtype(), nullptr, {});
    dense->set_data(omarchy::allocator().malloc(dense->nbytes()));
    // Pin it: this local dies at return while both dispatches that touch
    // it are still queued. Unpinned, the allocator recycled its bytes into
    // the next token's RoPE offset scalar and the strided copy wrote an
    // f32 over it (receipts/2026-09-04-rope-gate-drain.md). The per-call
    // queue drains that used to sit in rope_trig_gate masked it.
    omarchy::get_command_encoder(s).add_temporary(*dense);
    copy_gpu_inplace(
        input,
        *dense,
        input.shape(),
        input.strides(),
        make_contiguous_strides(input.shape()),
        /*i_offset=*/0,
        /*o_offset=*/0,
        CopyType::GeneralGeneral,
        s);
    in = &*dense;
    // The gather above leaves |dense| row-contiguous, so the follow-up
    // copy reads flat storage. A dtype-converting AsType picks General
    // for a non-contiguous input; kept here it hits the dtype-converting
    // strided-copy refusal even though nothing strided remains (the
    // db10f53 slice views). Vector reaches the flat cast path.
    ctype = CopyType::Vector;
  }
  // Upstream's set_copy_output_data always gives the output a buffer, even
  // for zero-size outputs (malloc(0) yields a valid empty VulkanBuffer).
  // Skipping set_data left array_desc_->data null, and any later
  // buffer_size()/data() access on the eval'd array segfaulted.
  out.set_data(omarchy::allocator().malloc(out.nbytes()));
  copy_gpu_inplace(
      *in, out, in->shape(), in->strides(), out.strides(), 0, 0, ctype, s);
}

void fill_gpu(const array& val, array& out, const Stream& s) {
  copy_gpu(val, out, CopyType::Scalar, s);
}


void reshape_gpu(const array& in, array& out, Stream s) {
  auto [copy_necessary, out_strides] = prepare_reshape(in, out);
  if (!copy_necessary) {
    shared_buffer_reshape(in, out_strides, out);
    return;
  }
  // A strided reshape is a general gather. Broadcast views from
  // mx.repeat and mx.tile carry stride-0 axes, and transposed views
  // permute strides; both report flags().contiguous under the
  // span-based definition while size() exceeds data_size(), so the
  // old flat buffer copy here read past the source allocation. The
  // strided-copy engine expresses both shapes for every supported
  // numeric storage width.
  if (out.nbytes() > 0) {
    out.set_data(omarchy::allocator().malloc(out.nbytes()));
  }
  copy_gpu_inplace(
      in,
      out,
      in.shape(),
      in.strides(),
      make_contiguous_strides(in.shape()),
      /*i_offset=*/0,
      /*o_offset=*/0,
      CopyType::General,
      s);
}
} // namespace mlx::core
