// SPDX-License-Identifier: MIT
#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <optional>

namespace mlx::core::omarchy {

struct SdpaDecodeMaskLayout {
  uint32_t offset;
  uint32_t batch;
  uint32_t head;
  uint32_t key;
};

inline std::optional<SdpaDecodeMaskLayout> sdpa_decode_mask_layout(
    uint64_t offset,
    uint64_t accessible,
    int64_t batch_stride,
    int64_t head_stride,
    int64_t key_stride,
    uint32_t batches,
    uint32_t heads,
    uint32_t keys) {
  if (accessible == 0 || offset > UINT32_MAX || batches == 0 || heads == 0 || keys == 0) {
    return std::nullopt;
  }
  uint64_t remaining = std::min(accessible - 1, uint64_t{UINT32_MAX} - offset);
  const std::array strides{batch_stride, head_stride, key_stride};
  const std::array extents{batches - 1, heads - 1, keys - 1};
  for (std::size_t i = 0; i < strides.size(); ++i) {
    const int64_t stride = strides[i];
    if (stride < 0 || stride > UINT32_MAX ||
        (extents[i] != 0 && static_cast<uint64_t>(stride) > remaining / extents[i])) {
      return std::nullopt;
    }
    remaining -= static_cast<uint64_t>(stride) * extents[i];
  }
  return SdpaDecodeMaskLayout{static_cast<uint32_t>(offset),
                              static_cast<uint32_t>(batch_stride),
                              static_cast<uint32_t>(head_stride),
                              static_cast<uint32_t>(key_stride)};
}

// MLX_OMARCHY_SDPA_DECODE_BOOL_MASK: the fused masked batch-2 route is on
// by default. A literal "0" selects the composed route; unset, empty and
// every other value keep the fused route.
inline bool sdpa_decode_mask_fused_enabled(const char* value) {
  return value == nullptr || value[0] != '0' || value[1] != '\0';
}

} // namespace mlx::core::omarchy
