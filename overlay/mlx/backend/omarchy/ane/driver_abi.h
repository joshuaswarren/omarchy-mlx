// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

#pragma once

#include <stdexcept>
#include <string>

namespace mlx::core::omarchy::ane::detail {

inline void require_driver_abi_major(int expected, int actual) {
  if (actual != expected) {
    throw std::runtime_error(
        "ANE driver ABI mismatch: runtime requires major " +
        std::to_string(expected) + ", driver reports major " +
        std::to_string(actual));
  }
}

} // namespace mlx::core::omarchy::ane::detail
