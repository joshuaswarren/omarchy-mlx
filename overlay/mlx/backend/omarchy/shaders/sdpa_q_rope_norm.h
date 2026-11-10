// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT
//
// Query prologue of the split-K decode SDPA pass 1 (sdpa_decode_split_p1.comp, sdpa_decode_split_p1v.comp, -DQ_ROPE_NORM, head width 128):
// the q RMSNorm + rotary embedding that shaders/fast_rope_norm.comp runs as its own dispatch, computed in the pass-1 workgroup instead.
// The pass-1 binding 0 then holds the raw q (before the norm), binding 4 the norm weight and binding 5 the position offset.
//
// The arithmetic is shaders/fast_rope_norm.comp (default, non-Apple reduction; head width 128 never selects the Apple reduction,
// see apple_norm_shape_selected in primitives.cpp) for one row of 128 elements, dims == 128, forward, scalar offset, time index 0:
// the squared sum over a stride-64..1 shared tree (the 256-thread reference starts at stride 128 and adds 0.0 there, an exact no-op),
// mean = sum / 128, norm = inversesqrt(mean + eps), (weight * norm) * value rounded to bf16, then the half-split rotation with
// rope_round intermediates. The returned value is the bf16-rounded float the standalone kernel stores; the caller applies the q scale.
//
// Push-constant mapping (ComputeParams fields not read by pass 1):
//   operation      eps bits
//   reduce_size    row length (128)
//   dims           half of the rotated width (64)
//   beta           log(base) / half_dims
//   output_size    rope scale bits (position = scale * (0 + offset))
//   lhs_size       norm weight item offset
//   rhs_size       position offset item offset
//   out_strides[0] raw q head stride, out_strides[2] raw q element stride (in elements)
//   flags          bit 0 set: use the fold (informational)
//
// Do not copy this logic into another shader; include this header.

#include "omarchy_trig.h"

layout(set = 0, binding = 4, std430) readonly buffer QNormWeight {
  uint values[];
} qnorm_weight_data;
layout(set = 0, binding = 5, std430) readonly buffer QOffsets {
  int values[];
} qoffset_data;

shared float q_block_values[128];
shared float q_rounded[128];

float q_rope_round(float value) {
  uint bits = floatBitsToUint(value);
  uint rounded = isnan(value)
      ? ((bits >> 16u) | 0x40u)
      : ((bits + 0x7fffu + ((bits >> 16u) & 1u)) >> 16u);
  return uintBitsToFloat(rounded << 16u);
}

float q_raw_load(uint head, uint d) {
  uint index = params.lhs_offset + head * params.out_strides[0] + d * params.out_strides[2];
  return uintBitsToFloat(uint(query_data.values[index]) << 16u);
}

float q_weight_load(uint d) {
  uint index = params.lhs_size + d;
  uint word = qnorm_weight_data.values[index >> 1u];
  uint bits = ((index & 1u) == 0u) ? (word & 0xFFFFu) : (word >> 16u);
  return uintBitsToFloat(bits << 16u);
}

// Thread t of 128 returns element t of the rotated, bf16-rounded row of query head `head`. Every thread must call it with the same head
// (the reduction barriers are workgroup-uniform).
float q_rope_norm_element(uint head, uint t) {
  uint row_length = params.reduce_size;
  uint half_dims = params.dims;
  float eps = uintBitsToFloat(params.operation);
  float squared = q_raw_load(head, t);
  squared = squared * squared;
  q_block_values[t] = squared;
  barrier();
  for (uint stride = 64u; stride > 0u; stride >>= 1u) {
    if (t < stride) {
      q_block_values[t] += q_block_values[t + stride];
    }
    barrier();
  }
  float mean_square = q_block_values[0] / float(row_length);
  barrier();
  precise float norm = inversesqrt(mean_square + eps);
  precise float value = q_raw_load(head, t);
  precise float weight = q_weight_load(t);
  precise float weight_norm = weight * norm;
  q_rounded[t] = q_rope_round(value * weight_norm);
  barrier();

  float offset = float(qoffset_data.values[params.rhs_size]);
  float position = uintBitsToFloat(params.output_size) * offset;
  bool low = t < half_dims;
  uint i = low ? t : (t - half_dims);
  float inv_freq = exp(-float(i) * params.beta);
  float theta = position * inv_freq;
  float cos_t = q_rope_round(omarchy_trig_cos(theta));
  float sin_t = q_rope_round(omarchy_trig_sin(theta));
  float x1 = q_rounded[i];
  float x2 = q_rounded[i + half_dims];
  float result = low ? (q_rope_round(x1 * cos_t) - q_rope_round(x2 * sin_t))
                     : (q_rope_round(x1 * sin_t) + q_rope_round(x2 * cos_t));
  barrier();
  return q_rope_round(result);
}
