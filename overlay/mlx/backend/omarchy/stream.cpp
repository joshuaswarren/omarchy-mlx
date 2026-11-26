// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

// Stream lifecycle for the Omarchy backend: one command encoder per stream,
// thread-local like the CUDA backend, plus the thread-unsafe global map.

#include "mlx/backend/gpu/eval.h"

#include <cassert>

#include "mlx/backend/omarchy/device.h"
#include "mlx/backend/omarchy/encoder.h"
#include "mlx/utils.h"

namespace mlx::core::gpu {

void new_stream(Stream s) {
  assert(s.device == Device::gpu);
  auto& encoders = omarchy::get_command_encoders();
  auto& d = omarchy::device(s.device.index);
  encoders.try_emplace(s.index, d);
}

void new_thread_unsafe_stream(Stream s) {
  assert(s.device == Device::gpu);
  auto& encoders = omarchy::get_global_command_encoders();
  auto& d = omarchy::device(s.device.index);
  encoders.try_emplace(s.index, d);
}

void clear_streams() {
  omarchy::get_command_encoders().clear();
  if (is_main_thread()) {
    omarchy::get_global_command_encoders().clear();
  }
}

} // namespace mlx::core::gpu
