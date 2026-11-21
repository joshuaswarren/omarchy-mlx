// SPDX-License-Identifier: MIT
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vulkan/vulkan.h>

namespace mlx::core::omarchy {

inline uint32_t staging_memory_type(
    const VkPhysicalDeviceMemoryProperties& memory,
    uint32_t types) {
  uint32_t fallback = UINT32_MAX;
  for (uint32_t i = 0; i < memory.memoryTypeCount; ++i) {
    const auto& type = memory.memoryTypes[i];
    if (!(types & (1u << i)) ||
        !(type.propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) ||
        (memory.memoryHeaps[type.heapIndex].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)) {
      continue;
    }
    if (type.propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) {
      return i;
    }
    fallback = i;
  }
  return fallback;
}

inline uint32_t weight_memory_type(
    VkPhysicalDeviceType device_type,
    const VkPhysicalDeviceMemoryProperties& memory,
    uint32_t types) {
  if (device_type != VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU ||
      staging_memory_type(memory, UINT32_MAX) == UINT32_MAX) {
    return UINT32_MAX;
  }
  for (uint32_t i = 0; i < memory.memoryTypeCount; ++i) {
    const auto& type = memory.memoryTypes[i];
    if ((types & (1u << i)) &&
        (type.propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) &&
        !(type.propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) &&
        (memory.memoryHeaps[type.heapIndex].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)) {
      return i;
    }
  }
  return UINT32_MAX;
}

inline constexpr size_t kWeightStagingBytes = 4 * 1024 * 1024;

// The card-memory upload path moves bytes with DMA copies and cannot run the
// host-side endianness swap that the mapped path applies after reading. A
// big-endian source combined with VRAM weights must fail loudly, not produce
// byte-swapped weights in card memory.
inline void require_card_memory_upload(bool inference_weight, bool swap_endianness) {
  if (inference_weight && swap_endianness) {
    throw std::runtime_error(
        "[omarchy] card-memory inference weights cannot apply the endianness"
        " swap; load big-endian sources without MLX_OMARCHY_VRAM_WEIGHTS");
  }
}

// Sets the flag for its whole scope and clears it on every exit path,
// including an exception mid-upload, so a failed upload cannot poison every
// later one with "previous weight upload did not complete".
class InFlightFlag {
 public:
  explicit InFlightFlag(bool& flag) : flag_(flag) {
    flag_ = true;
  }
  ~InFlightFlag() {
    flag_ = false;
  }
  InFlightFlag(const InFlightFlag&) = delete;
  InFlightFlag& operator=(const InFlightFlag&) = delete;

 private:
  bool& flag_;
};

struct WeightUploadChunk {
  size_t read_bytes;
  size_t copy_bytes;
};

inline WeightUploadChunk weight_upload_chunk(size_t bytes, size_t offset) {
  const size_t read = std::min(bytes - offset, kWeightStagingBytes);
  return {read, (read + 3) & ~size_t{3}};
}

} // namespace mlx::core::omarchy
