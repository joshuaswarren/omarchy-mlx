// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string_view>
#include <vulkan/vulkan.h>

#include "mlx/backend/omarchy/inference_weights.h"

namespace mlx::core::omarchy {

// Memory type for opt-in device-local activation/general buffers on discrete
// GPUs. Returns UINT32_MAX when disabled, on integrated/UMA/software GPUs, or
// when no pure DEVICE_LOCAL + non-DEVICE_LOCAL HOST_VISIBLE staging pair
// exists.
inline uint32_t vram_activation_memory_type(
    bool opt_in_enabled,
    VkPhysicalDeviceType device_type,
    const VkPhysicalDeviceMemoryProperties& memory,
    uint32_t memory_type_bits) {
  if (!opt_in_enabled) {
    return UINT32_MAX;
  }
  return weight_memory_type(device_type, memory, memory_type_bits);
}

// Largest DEVICE_LOCAL Vulkan memory heap in bytes (0 if none).
inline size_t max_device_local_heap_bytes(
    const VkPhysicalDeviceMemoryProperties& memory) {
  size_t best = 0;
  for (uint32_t i = 0; i < memory.memoryHeapCount; ++i) {
    if ((memory.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0) {
      best = std::max(best, static_cast<size_t>(memory.memoryHeaps[i].size));
    }
  }
  return best;
}

// One chunk of a staging upload or download (bounded to kWeightStagingBytes =
// 4 MiB so host-visible DART/IOMMU mappings stay tiny).
struct StagingTransferChunk {
  size_t byte_offset;
  size_t byte_count;
  size_t copy_bytes; // 4-byte aligned Vulkan transfer length
};

inline StagingTransferChunk vram_staging_chunk(
    size_t total_bytes,
    size_t offset,
    size_t staging_cap = kWeightStagingBytes) {
  if (offset >= total_bytes || staging_cap == 0) {
    return {offset, 0, 0};
  }
  const size_t n = std::min(total_bytes - offset, staging_cap);
  const size_t aligned = (n + 3) & ~size_t{3};
  return {offset, n, aligned};
}

// Pure capability check for the 2x16 Qmm decode workgroup geometry
// (SLOTS_PER_GROUP=16, 32-lane subgroups -> 512 threads along local_size_x).
struct QmmWideCaps {
  uint32_t subgroup_size{0};
  uint32_t subgroup_operations{0};
  uint32_t max_workgroup_invocations{0};
  uint32_t max_workgroup_size_x{0};
};

inline bool supports_qmm_2x16(const QmmWideCaps& caps) {
  return caps.subgroup_size == 32u &&
      (caps.subgroup_operations & VK_SUBGROUP_FEATURE_ARITHMETIC_BIT) != 0u &&
      caps.max_workgroup_invocations >= 512u &&
      caps.max_workgroup_size_x >= 512u;
}

// Default Qmm decode geometry selection when MLX_OMARCHY_QMM_GEOM is unset.
// Preserves Apple defaults (G13G/G13C -> 2x16; other Apple chips -> 2x4 until
// calibrated) and enables 2x16 on discrete GPUs with opt-in VRAM activations
// whenever the device advertises support for 512-thread 32-wide subgroup WGs.
inline bool select_qmm_wide_2x16(
    std::string_view device_name,
    VkPhysicalDeviceType physical_type,
    const QmmWideCaps& caps,
    bool vram_activations_enabled) {
  if (device_name.find("G13G") != std::string_view::npos ||
      device_name.find("G13C") != std::string_view::npos) {
    return true;
  }
  if (vram_activations_enabled &&
      physical_type == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU &&
      supports_qmm_2x16(caps)) {
    return true;
  }
  return false;
}

// Pure host/device coherence state machine for one device-local buffer.
// Models the exact transitions enforced by Buffer::raw_ptr() and
// CommandEncoder::{dispatch_compute_pipeline, copy_buffer, fill_buffer}.
struct DeviceLocalSyncState {
  bool host_dirty{false};
  bool device_dirty{false};

  // Host requests a raw pointer via Buffer::raw_ptr().
  // Returns true if the caller must download device -> host_shadow first.
  bool on_host_raw_ptr() {
    const bool need_download = device_dirty;
    device_dirty = false;
    host_dirty = true;
    return need_download;
  }

  // GPU command references this buffer for read and/or write over [offset, offset+range).
  // Returns true if the caller must upload host_shadow -> device first.
  bool on_gpu_access(
      bool read,
      bool write,
      size_t offset,
      size_t range,
      size_t buffer_size) {
    const bool full_overwrite =
        write && !read && offset == 0 && range >= buffer_size;
    const bool need_upload = host_dirty && !full_overwrite;
    if (need_upload) {
      host_dirty = false;
    }
    if (write) {
      device_dirty = true;
      host_dirty = false;
    }
    return need_upload;
  }

  void on_recycle() {
    host_dirty = false;
    device_dirty = false;
  }
};

} // namespace mlx::core::omarchy
