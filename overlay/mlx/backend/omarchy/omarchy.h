// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

#pragma once

#include <string>
#include <unordered_map>
#include <variant>

#include "mlx/api.h"

// Public Omarchy backend surface. Mirrors mlx/backend/metal/metal.h so callers
// can query the backend without depending on Vulkan headers.
namespace mlx::core::omarchy {

MLX_API bool is_available();

// The recorded reason the backend is unavailable. Empty when available.
MLX_API const std::string& init_error();

MLX_API const
    std::unordered_map<std::string, std::variant<std::string, size_t>>&
    device_info(int device_index = 0);

} // namespace mlx::core::omarchy
