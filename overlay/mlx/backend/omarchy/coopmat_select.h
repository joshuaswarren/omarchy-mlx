// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

// Cooperative-matrix shape bookkeeping and the pure predicates kernel
// selection uses to decide whether a coopmat route is legal on the running
// device. A coopmat kernel may be chosen only when the exact shape it emits
// (dimensions, A/B/C/result element types, scope, saturation) is one the
// device advertised through vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR;
// otherwise the caller falls back to the non-coopmat route.
//
// The types here are deliberately free of <vulkan/vulkan.h> and the MLX
// headers so the selector and its host unit test compile without a Vulkan
// SDK or a GPU. device.cpp maps the advertised VkCooperativeMatrixPropertiesKHR
// entries onto these when it probes the device.

#pragma once

#include <cstdint>
#include <vector>

namespace mlx::core::omarchy {

// Element type of a cooperative-matrix operand. "Other" covers component
// types the backend has no kernel for; such shapes are still recorded so an
// exact match never aliases onto a type the selector understands.
enum class CoopmatType : uint8_t {
  Float16,
  BFloat16,
  Float32,
  Other,
};

// Scope the cooperative matrix spans. The backend's kernels are all
// subgroup-scoped; a workgroup or unknown scope is recorded but never
// matches a kernel's required shape.
enum class CoopmatScope : uint8_t {
  Subgroup,
  Workgroup,
  Other,
};

// One shape from vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR. Recorded
// verbatim, including mixed-operand shapes where a != b: a kernel that loads
// A and B as different types needs that exact mixed shape advertised, so it
// cannot be dropped the way an A==B summary flag would.
struct CoopmatShape {
  uint32_t m{0};
  uint32_t n{0};
  uint32_t k{0};
  CoopmatType a{CoopmatType::Other};
  CoopmatType b{CoopmatType::Other};
  CoopmatType c{CoopmatType::Other};
  CoopmatType result{CoopmatType::Other};
  CoopmatScope scope{CoopmatScope::Other};
  bool saturating{false};
};

inline bool operator==(const CoopmatShape& lhs, const CoopmatShape& rhs) {
  return lhs.m == rhs.m && lhs.n == rhs.n && lhs.k == rhs.k &&
      lhs.a == rhs.a && lhs.b == rhs.b && lhs.c == rhs.c &&
      lhs.result == rhs.result && lhs.scope == rhs.scope &&
      lhs.saturating == rhs.saturating;
}

inline bool operator!=(const CoopmatShape& lhs, const CoopmatShape& rhs) {
  return !(lhs == rhs);
}

// True when |want| is one of the shapes the device advertised. Kernel
// selection calls this with the exact shape a candidate coopmat kernel emits;
// a false result means the route is unavailable and the caller must fall back.
bool cooperative_matrix_advertises(
    const std::vector<CoopmatShape>& advertised, const CoopmatShape& want);

// The exact 8x8x8 subgroup shape the quantized-matmul prefill coopmat shader
// (shaders/qmm_coopmat.comp) emits for a given output dtype. The weight matrix
// B is always dequantized to fp32 and the accumulator and result are fp32. The
// activation matrix A is read directly as fp16 for an fp16 output — the default
// shader build, a MIXED fp16-A/fp32-B multiply — and is widened to fp32 for a
// bf16 output (the X_F32 / X_BF16 builds cast activations before the matmad, so
// A is fp32). Non-saturating. A device that lists only same-operand 8x8x8
// shapes (fp32xfp32 and fp16xfp16, as Apple AGX3B does) advertises the bf16
// route's shape but not the fp16 route's mixed shape, so the fp16 route falls
// back there while the bf16 route keeps the cooperative matrix.
CoopmatShape qmm_prefill_coopmat_shape(bool out_is_f16);

} // namespace mlx::core::omarchy
