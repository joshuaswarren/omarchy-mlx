// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

// Selection for the bf16 composed-scores prefill route in
// ScaledDotProductAttention::eval_gpu. The route stores the score matrix
// in bf16 instead of f32 (no q/k/v upcasts, bf16 softmax, bf16 matmuls).
// MLX_OMARCHY_SDPA_BF16_FAST pins the choice either way; unset it now
// follows the coopmat composed capability: a device without the 8x8x8
// f32 cooperative-matrix shape has no faster composed route anyway, so
// the bf16 scores path is strictly better than the f32 fallback there.

#pragma once

#include <cstring>

namespace mlx::core::omarchy {

// |coopmat_composed_capable| is the device-capability half of
// composed_ready in eval_gpu: cooperative_matrix_f32_8 with a subgroup
// of 32 and MLX_OMARCHY_NO_COOPMAT unset. |prefill_shaped| distinguishes
// multi-query calls from single-query decode: the automatic default is
// prefill-only because a decode call with bf16_fast on vetoes the fused
// masked decode route, which is the faster arm there. Shape and dtype
// terms stay at the call site; this predicate sees only what the flag
// needs.
inline bool sdpa_bf16_fast_enabled(
    const char* env,
    bool coopmat_composed_capable,
    bool prefill_shaped) {
  if (env != nullptr) {
    return std::strcmp(env, "0") != 0;
  }
  return prefill_shaped && !coopmat_composed_capable;
}

} // namespace mlx::core::omarchy
