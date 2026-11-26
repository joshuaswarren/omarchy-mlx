// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

#pragma once

#include <stdexcept>
#include <string>

namespace mlx::core::omarchy {

// Default-device policy. The GPU backend owns every tensor primitive, so a
// release build must refuse CPU evaluation instead of silently degrading
// when Vulkan device creation failed. Non-release builds keep the CPU
// fallback so development hosts without Apple GPUs can run unit logic.
// Returns true when the default device is the GPU, false when a CPU
// fallback is permitted; refusal throws.
inline bool gpu_default_or_refuse(
    bool gpu_available,
    bool release_build,
    const std::string& init_error) {
  if (gpu_available) {
    return true;
  }
  if (release_build) {
    throw std::runtime_error(
        "[omarchy] refusing CPU tensor fallback in a release build;"
        " the GPU backend is unavailable: " +
        (init_error.empty() ? std::string("no reason recorded") : init_error));
  }
  return false;
}

} // namespace mlx::core::omarchy
