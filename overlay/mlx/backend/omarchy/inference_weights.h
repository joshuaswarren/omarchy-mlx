// SPDX-License-Identifier: MIT
#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <cstddef>
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

// The largest pure DEVICE_LOCAL memory type (true VRAM: device local and
// not host visible). UINT32_MAX when the device has none. Ties keep the
// first match.
inline uint32_t local_device_memory_type(
    const VkPhysicalDeviceMemoryProperties& memory,
    uint32_t types) {
  uint32_t best = UINT32_MAX;
  VkDeviceSize best_size = 0;
  for (uint32_t i = 0; i < memory.memoryTypeCount; ++i) {
    const auto& type = memory.memoryTypes[i];
    if (!(types & (1u << i)) ||
        !(type.propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) ||
        (type.propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) ||
        !(memory.memoryHeaps[type.heapIndex].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)) {
      continue;
    }
    const auto& heap = memory.memoryHeaps[type.heapIndex];
    if (best == UINT32_MAX || heap.size > best_size) {
      best = i;
      best_size = heap.size;
    }
  }
  return best;
}

// True when the device exposes any device-local memory type the host can
// map: a resizable-bar aperture or unified memory. A card behind a link
// without one (Thunderbolt dGPU) splits memory into pure VRAM and host-only
// GTT, so mapped weight buffers land in GTT and fail allocation far below
// real VRAM; such a card must take the DMA-upload weight path.
inline bool mapped_device_local_memory_type(
    const VkPhysicalDeviceMemoryProperties& memory) {
  for (uint32_t i = 0; i < memory.memoryTypeCount; ++i) {
    const auto& type = memory.memoryTypes[i];
    if ((type.propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) &&
        (type.propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) &&
        (memory.memoryHeaps[type.heapIndex].flags &
         VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)) {
      return true;
    }
  }
  return false;
}

// Where inference weights live on a discrete GPU.
//
// The card memory a driver exposes is not always the card memory a model
// needs: a discrete card behind a link without a large bar aperture shows a
// small device-local heap (a few hundred MiB) while its real memory stays
// unreachable, and a bar-mapped card shows one large host-visible
// device-local heap. Placing a model in the first device-local type that
// matches then fills the small heap and fails the allocation.
//
// So the choice is budget aware: this process's live weight bytes plus the
// request plus a headroom margin must fit in the heap. The largest pure
// device-local heap is preferred when the budget fits it, then the largest
// host-visible device-local heap, and UINT32_MAX when neither can - the
// caller then keeps the weights in host memory and the card still runs the
// compute.
//
// Integrated GPUs never take this path: they keep the mapped weights.
inline constexpr size_t kWeightHeapHeadroomBytes = 64ull * 1024 * 1024;

inline size_t saturating_add(size_t a, size_t b) {
  return a > std::numeric_limits<size_t>::max() - b
      ? std::numeric_limits<size_t>::max()
      : a + b;
}

inline uint32_t weight_memory_type_for_size(
    VkPhysicalDeviceType device_type,
    const VkPhysicalDeviceMemoryProperties& memory,
    uint32_t types,
    size_t bytes,
    size_t live_weight_bytes = 0) {
  if (device_type != VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU ||
      staging_memory_type(memory, UINT32_MAX) == UINT32_MAX) {
    return UINT32_MAX;
  }
  // Saturating: a budget this large cannot fit any heap anyway.
  const size_t needed = saturating_add(
      saturating_add(bytes, live_weight_bytes), kWeightHeapHeadroomBytes);
  uint32_t best_local = UINT32_MAX;
  uint32_t best_mapped = UINT32_MAX;
  VkDeviceSize best_local_size = 0;
  VkDeviceSize best_mapped_size = 0;
  for (uint32_t i = 0; i < memory.memoryTypeCount; ++i) {
    const auto& type = memory.memoryTypes[i];
    if (!(types & (1u << i)) ||
        !(type.propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
      continue;
    }
    const auto& heap = memory.memoryHeaps[type.heapIndex];
    if (!(heap.flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)) {
      continue;
    }
    if (type.propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) {
      if (heap.size > best_mapped_size) {
        best_mapped_size = heap.size;
        best_mapped = i;
      }
    } else if (heap.size > best_local_size) {
      best_local_size = heap.size;
      best_local = i;
    }
  }
  if (best_local != UINT32_MAX && best_local_size >= needed) {
    return best_local;
  }
  if (best_mapped != UINT32_MAX && best_mapped_size >= needed) {
    return best_mapped;
  }
  return UINT32_MAX;
}

inline uint32_t weight_memory_type(
    VkPhysicalDeviceType device_type,
    const VkPhysicalDeviceMemoryProperties& memory,
    uint32_t types) {
  if (device_type != VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU ||
      staging_memory_type(memory, UINT32_MAX) == UINT32_MAX) {
    return UINT32_MAX;
  }
  return local_device_memory_type(memory, types);
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
