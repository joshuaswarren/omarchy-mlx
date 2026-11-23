// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

#include "mlx/backend/omarchy/device_exts.h"

namespace mlx::core::omarchy {

std::vector<const char*> device_extension_names(
    const DeviceExtInput& in,
    bool low_queue,
    bool trace_timestamps) {
  std::vector<const char*> names;
  if (in.shader_atomic_float_add) {
    names.push_back("VK_EXT_shader_atomic_float");
  }
  if (in.cooperative_matrix_f32_8) {
    names.push_back("VK_KHR_cooperative_matrix");
  }
  // cooperative_matrix_bf16_8 reaches the input only when the caller
  // compiled with OMARCHY_BF16_CMAT and the device advertises it.
  if (in.cooperative_matrix_bf16_8) {
    names.push_back("VK_KHR_shader_bfloat16");
  }
  if (in.memory_priority && in.pageable_device_local_memory) {
    names.push_back("VK_EXT_memory_priority");
    names.push_back("VK_EXT_pageable_device_local_memory");
  }
  if (low_queue && in.queue_global_priority) {
    names.push_back("VK_EXT_global_priority");
  }
  if (trace_timestamps && in.calibrated_timestamps) {
    names.push_back("VK_EXT_calibrated_timestamps");
  }
  if (in.push_descriptor) {
    names.push_back("VK_KHR_push_descriptor");
  }
  return names;
}

}  // namespace mlx::core::omarchy
