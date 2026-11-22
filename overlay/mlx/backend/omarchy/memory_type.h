// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <vulkan/vulkan.h>

namespace mlx::core::omarchy {

// Memory type with the required property flags, chosen per-heap by
// capacity: among the types allowed by |type_bits| that carry |required|,
// a DEVICE_LOCAL heap wins only when its capacity is at least the best
// matching non-device-local heap; otherwise the largest matching heap
// wins. Largest wins within each class; first match breaks ties; returns
// UINT32_MAX when nothing matches.
//
// The previous chooser returned the first DEVICE_LOCAL-heap match, so on a
// driver whose device-local heap is a small window (the B60 over USB4
// shows ~256 MiB) every mapped mlx buffer was aimed at that heap and
// failed at vkAllocateMemory while a larger heap matched the same flags.
// ba64b28c6a fixed the identical first-match flaw for weight placement;
// this chooser applies the same capacity awareness to the allocator's
// mapped-buffer path (Apple-UMA layouts keep the unified device-local
// heap, which is already the largest).
inline uint32_t capacity_aware_memory_type(
    const VkPhysicalDeviceMemoryProperties& memory,
    uint32_t type_bits,
    VkMemoryPropertyFlags required) {
  uint32_t best_local = UINT32_MAX;
  uint32_t best_other = UINT32_MAX;
  VkDeviceSize best_local_size = 0;
  VkDeviceSize best_other_size = 0;
  for (uint32_t i = 0; i < memory.memoryTypeCount; ++i) {
    const auto& type = memory.memoryTypes[i];
    if (!(type_bits & (1u << i)) ||
        (type.propertyFlags & required) != required) {
      continue;
    }
    const auto& heap = memory.memoryHeaps[type.heapIndex];
    if (heap.flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
      if (best_local == UINT32_MAX || heap.size > best_local_size) {
        best_local = i;
        best_local_size = heap.size;
      }
    } else if (best_other == UINT32_MAX || heap.size > best_other_size) {
      best_other = i;
      best_other_size = heap.size;
    }
  }
  if (best_local != UINT32_MAX && best_local_size >= best_other_size) {
    return best_local;
  }
  return best_other;
}

} // namespace mlx::core::omarchy
