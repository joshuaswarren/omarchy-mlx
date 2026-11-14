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

CoopmatShape qmm_prefill_coopmat_shape(bool out_is_f16) {
  CoopmatShape shape;
  shape.m = 8u;
  shape.n = 8u;
  shape.k = 8u;
  // fp16 output reads activations as fp16 (mixed A=fp16, B=fp32); bf16 output
  // widens activations to fp32 before the matmad (A=fp32, B=fp32).
  shape.a = out_is_f16 ? CoopmatType::Float16 : CoopmatType::Float32;
  shape.b = CoopmatType::Float32;
  shape.c = CoopmatType::Float32;
  shape.result = CoopmatType::Float32;
  shape.scope = CoopmatScope::Subgroup;
  shape.saturating = false;
  return shape;
}

} // namespace mlx::core::omarchy
