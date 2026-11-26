// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

#pragma once

#include <sstream>
#include <stdexcept>
#include <string>

#include "mlx/array.h"
#include "mlx/dtype_utils.h"

// The exact compatibility error contract for the Omarchy backend (plan R7,
// AE2): a primitive the accelerators cannot run fails with its name, dtype,
// and shape. Nothing reroutes that GPU work to CPU silently; a CPU
// implementation runs only on an explicit CPU stream.
namespace mlx::core::omarchy {

[[noreturn]] inline void unsupported(
    const std::string& name,
    Dtype dtype,
    const Shape& shape) {
  std::ostringstream msg;
  msg << "[omarchy] " << name
      << " is not implemented for the Omarchy Vulkan backend (dtype="
      << dtype_to_string(dtype) << ", shape=[";
  for (int i = 0; i < static_cast<int>(shape.size()); ++i) {
    if (i > 0) {
      msg << ",";
    }
    msg << shape[i];
  }
  msg << "]). No GPU kernel exists for it; no silent CPU fallback occurs."
      << " Run it on an explicit CPU stream to use the CPU implementation.";
  throw std::runtime_error(msg.str());
}

[[noreturn]] inline void unsupported(
    const std::string& name,
    const array& out) {
  unsupported(name, out.dtype(), out.shape());
}

} // namespace mlx::core::omarchy
