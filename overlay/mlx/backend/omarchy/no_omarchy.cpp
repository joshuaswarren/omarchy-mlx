// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

// Compiled without the Omarchy backend (MLX_BUILD_OMARCHY=OFF): report the
// backend as unavailable, mirroring no_metal.cpp / no_cuda.cpp.

#include "mlx/backend/omarchy/omarchy.h"

namespace mlx::core::omarchy {

bool is_available() {
  return false;
}

const std::string& init_error() {
  static const std::string reason =
      "[omarchy] built without MLX_BUILD_OMARCHY.";
  return reason;
}

const std::unordered_map<std::string, std::variant<std::string, size_t>>&
device_info(int /* device_index */) {
  static const std::
      unordered_map<std::string, std::variant<std::string, size_t>>
          empty;
  return empty;
}

} // namespace mlx::core::omarchy
