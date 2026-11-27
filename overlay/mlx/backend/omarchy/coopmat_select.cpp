// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

#include "mlx/backend/omarchy/coopmat_select.h"

#include <algorithm>

namespace mlx::core::omarchy {

bool cooperative_matrix_advertises(
    const std::vector<CoopmatShape>& advertised, const CoopmatShape& want) {
  return std::find(advertised.begin(), advertised.end(), want) !=
      advertised.end();
}

CoopmatShape qmm_prefill_coopmat_shape() {
  CoopmatShape shape;
  shape.m = 8u;
  shape.n = 8u;
  shape.k = 8u;
  // Both output dtypes widen the activations to an fp32 A matrix before the
  // matmad (fp16 x is loaded as fp16 and converted), so the shape is fp32xfp32.
  shape.a = CoopmatType::Float32;
  shape.b = CoopmatType::Float32;
  shape.c = CoopmatType::Float32;
  shape.result = CoopmatType::Float32;
  shape.scope = CoopmatScope::Subgroup;
  shape.saturating = false;
  return shape;
}

CoopmatShape qmm_prefill_coopmat16_bf16_shape() {
  CoopmatShape shape;
  shape.m = 16u;
  shape.n = 16u;
  shape.k = 16u;
  shape.a = CoopmatType::BFloat16;
  shape.b = CoopmatType::BFloat16;
  shape.c = CoopmatType::Float32;
  shape.result = CoopmatType::Float32;
  shape.scope = CoopmatScope::Subgroup;
  shape.saturating = false;
  return shape;
}

CoopmatShape qmm_prefill_coopmat_shape_f16x16() {
  CoopmatShape shape;
  shape.m = 16u;
  shape.n = 16u;
  shape.k = 16u;
  // The CMAT16 build keeps fp16 operands (bf16 x is staged as fp16, the
  // dequantized weights store as fp16) with the fp32 accumulator.
  shape.a = CoopmatType::Float16;
  shape.b = CoopmatType::Float16;
  shape.c = CoopmatType::Float32;
  shape.result = CoopmatType::Float32;
  shape.scope = CoopmatScope::Subgroup;
  shape.saturating = false;
  return shape;
}

CoopmatShape matmul_direct_coopmat_shape_c16(CoopmatType operand) {
  CoopmatShape shape;
  shape.m = 16u;
  shape.n = 16u;
  shape.k = 16u;
  // The dense direct kernel loads A and B as the storage dtype and keeps
  // an fp32 accumulator; the result drains converted to ELEM in the same
  // shader, so the advertised C/result stay fp32.
  shape.a = operand;
  shape.b = operand;
  shape.c = CoopmatType::Float32;
  shape.result = CoopmatType::Float32;
  shape.scope = CoopmatScope::Subgroup;
  shape.saturating = false;
  return shape;
}

CoopmatShape matmul_xe16_bf16_shape() {
  CoopmatShape shape;
  shape.m = 8u;
  shape.n = 16u;
  shape.k = 16u;
  shape.a = CoopmatType::BFloat16;
  shape.b = CoopmatType::BFloat16;
  shape.c = CoopmatType::Float32;
  shape.result = CoopmatType::Float32;
  shape.scope = CoopmatScope::Subgroup;
  shape.saturating = false;
  return shape;
}

CoopmatShape qmm_prefill_coopmat16_xe_shape() {
  CoopmatShape shape;
  shape.m = 8u;
  shape.n = 16u;
  shape.k = 16u;
  shape.a = CoopmatType::BFloat16;
  shape.b = CoopmatType::BFloat16;
  shape.c = CoopmatType::Float32;
  shape.result = CoopmatType::Float32;
  shape.scope = CoopmatScope::Subgroup;
  shape.saturating = false;
  return shape;
}

} // namespace mlx::core::omarchy
