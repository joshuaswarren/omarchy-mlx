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

// Runtime PM acceptance for the ANE platform device. The packaged ane
// driver manages runtime PM itself (power/control 'auto'); a dev setup
// may pin the device 'on'. Either control value is acceptable as long
// as the device reports 'active': the gate opens /dev/accel/accel0
// before this check, which resumes a runtime-PM-managed device, so a
// device that cannot power up still fails here.
inline bool runtime_pm_acceptable(
    const std::string& control,
    const std::string& status) {
  return (control == "on" || control == "auto") && status == "active";
}

} // namespace mlx::core::omarchy::ane::detail
