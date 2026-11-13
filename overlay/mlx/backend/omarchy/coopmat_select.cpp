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
  // Both output dtypes widen the activations to an fp32 A matrix before the
  // matmad (fp16 x is loaded as fp16 and converted), so the shape is fp32xfp32.
  (void)out_is_f16;
  shape.a = CoopmatType::Float32;
  shape.b = CoopmatType::Float32;
  shape.c = CoopmatType::Float32;
  shape.result = CoopmatType::Float32;
  shape.scope = CoopmatScope::Subgroup;
  shape.saturating = false;
  return shape;
}

} // namespace mlx::core::omarchy
