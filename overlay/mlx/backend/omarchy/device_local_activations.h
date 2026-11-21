// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <chrono>
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
  // Pure query: whether the caller must download device -> host_shadow first.
  bool needs_download() const {
    return device_dirty;
  }

  // Commit state after any required download completes.
  bool on_host_raw_ptr() {
    const bool need_download = device_dirty;
    device_dirty = false;
    host_dirty = true;
    return need_download;
  }

  // Pure query: whether GPU access over [offset, offset+range) must upload
  // host_shadow -> device first. Does NOT mutate state, so a thrown upload
  // leaves host_dirty / device_dirty intact and never lets a later raw_ptr()
  // overwrite the authoritative host shadow with stale or partial device data.
  bool needs_upload(
      bool read,
      bool write,
      size_t offset,
      size_t range,
      size_t buffer_size) const {
    const bool full_overwrite =
        write && !read && offset == 0 && range >= buffer_size;
    return host_dirty && !full_overwrite;
  }

  // Commit after host_shadow -> device upload has succeeded.
  void commit_gpu_upload() {
    host_dirty = false;
  }

  // Commit after all uploads for a GPU command have succeeded and the command
  // will write this buffer.
  void commit_gpu_write() {
    device_dirty = true;
    host_dirty = false;
  }

  // Single-step transition when transfer and record cannot fail independently.
  bool on_gpu_access(
      bool read,
      bool write,
      size_t offset,
      size_t range,
      size_t buffer_size) {
    const bool need_upload =
        needs_upload(read, write, offset, range, buffer_size);
    if (need_upload) {
      commit_gpu_upload();
    }
    if (write) {
      commit_gpu_write();
    }
    return need_upload;
  }

  void on_recycle() {
    host_dirty = false;
    device_dirty = false;
  }
};

// Outcome of a bounded wait on an activation staging transfer fence.
enum class StagingTransferWaitResult : uint8_t {
  Completed,
  TimedOut,
  DeviceError,
};

// Tracks whether the Device's private staging resources (transfer_staging_,
// transfer_staging_memory_, transfer_pool_, transfer_cmd_, transfer_fence_)
// are still referenced by an unretired GPU submission.
//
// While in_flight is true, callers MUST NOT reset/reuse transfer_cmd_, reset
// transfer_fence_, overwrite transfer_staging_, or destroy any of those
// handles. Subsequent calls poll GetFenceStatus; only VK_SUCCESS clears
// in_flight and allows safe reuse/recovery.
struct StagingTransferState {
  bool in_flight{false};

  template <typename GetStatusFn>
  VkResult check_prior_in_flight(GetStatusFn&& get_fence_status) {
    if (!in_flight) {
      return VK_SUCCESS;
    }
    const VkResult st = get_fence_status();
    if (st == VK_SUCCESS) {
      in_flight = false;
    }
    return st;
  }

  void mark_submitted() {
    in_flight = true;
  }

  // Poll wait_slice(slice_ns) until VK_SUCCESS, non-timeout error, or
  // now_ns() - start_ns >= timeout_ns. Preserves in_flight=true on timeout
  // or device error so in-flight resources are never reused or destroyed.
  template <typename WaitSliceFn, typename NowNsFn>
  StagingTransferWaitResult wait_bounded_with_clock(
      uint64_t timeout_ns,
      uint64_t poll_slice_ns,
      WaitSliceFn&& wait_slice,
      NowNsFn&& now_ns,
      VkResult* out_error = nullptr) {
    const uint64_t start_ns = now_ns();
    const uint64_t slice_cap =
        poll_slice_ns > 0 ? poll_slice_ns : (timeout_ns > 0 ? timeout_ns : 1);
    for (;;) {
      const uint64_t cur_ns = now_ns();
      const uint64_t elapsed = cur_ns >= start_ns ? (cur_ns - start_ns) : 0;
      const uint64_t remaining =
          elapsed < timeout_ns ? (timeout_ns - elapsed) : 0;
      const uint64_t step =
          remaining == 0 ? 0 : std::min(remaining, slice_cap);
      const VkResult res = wait_slice(step);
      if (res == VK_SUCCESS) {
        in_flight = false;
        if (out_error) {
          *out_error = VK_SUCCESS;
        }
        return StagingTransferWaitResult::Completed;
      }
      if (res != VK_TIMEOUT) {
        if (out_error) {
          *out_error = res;
        }
        return StagingTransferWaitResult::DeviceError;
      }
      const uint64_t after_ns = now_ns();
      const uint64_t after_elapsed =
          after_ns >= start_ns ? (after_ns - start_ns) : 0;
      if (remaining == 0 || after_elapsed >= timeout_ns) {
        if (out_error) {
          *out_error = VK_TIMEOUT;
        }
        return StagingTransferWaitResult::TimedOut;
      }
    }
  }

  template <typename WaitSliceFn>
  StagingTransferWaitResult wait_bounded(
      uint64_t timeout_ns,
      uint64_t poll_slice_ns,
      WaitSliceFn&& wait_slice,
      VkResult* out_error = nullptr) {
    return wait_bounded_with_clock(
        timeout_ns,
        poll_slice_ns,
        std::forward<WaitSliceFn>(wait_slice),
        []() -> uint64_t {
          return static_cast<uint64_t>(
              std::chrono::duration_cast<std::chrono::nanoseconds>(
                  std::chrono::steady_clock::now().time_since_epoch())
                  .count());
        },
        out_error);
  }
};

// Coordinated Vulkan teardown plan for one Device and its parent VkInstance.
// Vulkan requires child handles (VkDevice, pipelines, semaphores, buffers,
// fences, command pools) to be idle before destruction and every VkDevice to
// be destroyed before its parent VkInstance. Either the device is idle and
// every dependent handle plus VkDevice/VkInstance retires together, or an
// unretired transfer keeps VkDevice, its dependent handles, and the parent
// VkInstance alive together until process exit.
struct DeviceTeardownPlan {
  bool wait_device_idle{true};
  bool destroy_completion_semaphore{true};
  bool destroy_weight_staging{true};
  bool destroy_transfer_staging{true};
  bool destroy_compute_pipelines{true};
  bool destroy_device{true};
  bool allow_destroy_instance{true};
};

inline DeviceTeardownPlan plan_device_teardown(
    bool transfer_in_flight,
    VkResult transfer_fence_status,
    bool weight_staging_in_flight) {
  const bool transfer_idle =
      !transfer_in_flight || transfer_fence_status == VK_SUCCESS;
  const bool device_idle = transfer_idle && !weight_staging_in_flight;
  return DeviceTeardownPlan{
      device_idle,
      device_idle,
      device_idle,
      device_idle,
      device_idle,
      device_idle,
      device_idle,
  };
}

} // namespace mlx::core::omarchy
