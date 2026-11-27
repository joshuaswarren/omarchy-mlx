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

// Minimum *unrounded* allocation size placed in pure DEVICE_LOCAL VRAM on
// the discrete-GPU activation route. Tiny scalar buffers (<= 16 B, e.g. 4-byte
// sampled token IDs, RoPE offsets, scalar reductions) stay
// HOST_VISIBLE|HOST_COHERENT so pipelined host readback (y.item()) never
// stalls behind the next token's queued compute work on a shared VkQueue.
inline constexpr size_t kMinDeviceLocalActivationBytes = 16;

inline bool should_use_device_local_activation(
    bool opt_in_enabled,
    size_t requested_size) {
  return opt_in_enabled && requested_size > kMinDeviceLocalActivationBytes;
}

// Whether the device-local activation route applies for a device type and
// env setting (-1 unset, 0 off, 1 on). Discrete GPUs default ON: a foreign
// card behind a narrow BAR window exposes a small device-local+host-visible
// heap (the B60 over USB4 shows ~256 MiB) that captured every mapped
// buffer until vkAllocateMemory failed far below real VRAM; routing the
// large allocations to the pure device-local heap with staged upload is the
// working path. Non-discrete parts keep the strict opt-in.
inline bool device_local_activations_default(
    VkPhysicalDeviceType device_type,
    int env_setting) {
  if (env_setting >= 0) {
    return env_setting > 0;
  }
  return device_type == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU;
}

// Memory type for device-local activation/general buffers on discrete
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

// Vulkan 1.0 core limit for inline vkCmdUpdateBuffer: dataSize <= 65536 and
// multiple of 4 (dstOffset multiple of 4). Recording small host->device
// uploads inline inside the active compute command buffer avoids per-buffer
// vkQueueSubmit + vkWaitForFences round-trips over PCIe/Thunderbolt.
inline constexpr size_t kInlineUpdateMaxBytes = 65536;

inline bool can_inline_update_device_local(size_t buffer_size) {
  return buffer_size > 0 && buffer_size <= kInlineUpdateMaxBytes &&
      (buffer_size % 4u) == 0;
}

// Whether an inline vkCmdUpdateBuffer recorded on this encoder must add a
// timeline semaphore wait for a buffer's prior completion stamp. Same-stream
// submitted work (<= last_completion) and already-retired work (<= drained)
// require no extra wait; pending-sentinel work is handled by the caller
// (program order on the same encoder, or flushing other pending encoders).
inline bool needs_inline_update_semaphore_wait(
    uint64_t comp,
    uint64_t pending_sentinel,
    uint64_t last_completion,
    uint64_t drained_value) {
  return comp != 0 && comp != pending_sentinel && comp > last_completion &&
      comp > drained_value;
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
// Apple parts route through select_qmm_apple_2x8w2 (G13G/G13C/G14C -> 2x8w2;
// other Apple chips -> 2x4 until calibrated), so this probe only ever sees
// non-Apple names; it enables 2x16 on discrete GPUs with VRAM activations
// whenever the device advertises support for 512-thread 32-wide subgroup WGs.
inline bool select_qmm_wide_2x16(
    std::string_view device_name,
    VkPhysicalDeviceType physical_type,
    const QmmWideCaps& caps,
    bool vram_activations_enabled) {
  // No-gating default (2026-11-25): the 2x16 geometry selects by the
  // 512-thread 32-wide-subgroup probe on every part. G13G/G13C/G14C keep
  // their measured 2x8w2 route; a discrete part still needs VRAM
  // activations. Any other part that meets the probe gets 2x16.
  if (device_name.find("G13G") != std::string_view::npos ||
      device_name.find("G13C") != std::string_view::npos ||
      device_name.find("G14C") != std::string_view::npos) {
    return false;
  }
  if (physical_type == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
    return vram_activations_enabled && supports_qmm_2x16(caps);
  }
  return supports_qmm_2x16(caps);
}

// Pure capability check for the 2x8w2 Qmm decode workgroup geometry
// (SLOTS_PER_GROUP=8, 32-lane subgroups -> 256 threads along local_size_x).
// The kernels assume subgroupSize == 32 (see shaders/qmm_vec.comp), so a
// device reporting subgroup_size == 64 can never take this path today: it
// would need a kernel variant that pins requiredSubgroupSize = 32, which
// does not exist yet. The probe keeps such parts safely on the wider/2x4
// checks instead of compiling a wrong-sum kernel.
inline bool supports_qmm_2x8w2(const QmmWideCaps& caps) {
  return caps.subgroup_size == 32u &&
      (caps.subgroup_operations & VK_SUBGROUP_FEATURE_ARITHMETIC_BIT) != 0u &&
      caps.max_workgroup_invocations >= 256u &&
      caps.max_workgroup_size_x >= 256u;
}

// G13G (13-inch M1), G13C (16-inch M1 Max), and G14C (14-inch M2 Max) default
// Qmm decode geometry: 2x8w2 (2 rows/slot x 8 slots = 256 threads/WG,
// 16 cols/WG, hoisted uvec2 loads across Word, Multi, 1-pass interleaved
// SwiGLU, ResNorm, and OutgateResNorm).
inline bool select_qmm_apple_2x8w2(std::string_view device_name) {
  return device_name.find("G13G") != std::string_view::npos ||
      device_name.find("G13C") != std::string_view::npos ||
      device_name.find("G14C") != std::string_view::npos;
}

// Default 2x8w2 selection when MLX_OMARCHY_QMM_GEOM is unset. The capability
// probe gates every part; the Apple list stays as the measured exception
// table for integrated parts (any other integrated part waits for its own
// A/B). Discrete GPUs qualify by capability: VRAM activations plus the
// 256-thread 32-wide-subgroup probe. BMG keeps its measured bit-identical
// 2x16w2 route in the next check instead.
inline bool select_qmm_2x8w2(
    std::string_view device_name,
    VkPhysicalDeviceType physical_type,
    const QmmWideCaps& caps,
    bool vram_activations_enabled) {
  if (!supports_qmm_2x8w2(caps)) {
    return false;
  }
  if (select_qmm_apple_2x8w2(device_name)) {
    return true;
  }
  if (device_name.find("BMG") != std::string_view::npos) {
    return false;
  }
  return physical_type == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU &&
      vram_activations_enabled;
}

// Default decode QMM geometry name when MLX_OMARCHY_QMM_GEOM is unset.
// Every default-eligible geometry keeps the strided per-lane summation order
// so outputs are bit-identical to 2x4. 2x16kcont and 2x16kcs16 change
// per-lane FP32 reduction order at K >= 1024 (+/-1 bf16 ULP, flipping exact
// greedy ties on Qwen3.5-9B) and are excluded from every default path.
inline const char* select_default_qmm_geometry_name(
    std::string_view device_name,
    VkPhysicalDeviceType physical_type,
    const QmmWideCaps& caps,
    bool vram_activations_enabled) {
  if (select_qmm_2x8w2(
          device_name, physical_type, caps, vram_activations_enabled)) {
    return "2x8w2";
  }
  if (select_qmm_wide_2x16(
          device_name, physical_type, caps, vram_activations_enabled)) {
    if (device_name.find("BMG") != std::string_view::npos) {
      return "2x16w2";
    }
    return "2x16";
  }
  return "2x4";
}

// Dispatch-only hazards take a COMPUTE_SHADER to COMPUTE_SHADER barrier
// instead of the full ALL_COMMANDS one (encoder.cpp record_dispatch_barrier).
// Default on for discrete GPUs: on the RTX 5060 Ti Qwen3-4B decode went from
// 98.76 to 124.22 tok/s with identical ids. Apple chips keep the full barrier
// until their own A/B. setting: MLX_OMARCHY_COMPUTE_BARRIER 0/1, or -1 unset.
inline bool select_compute_barrier(VkPhysicalDeviceType physical_type, int setting) {
  if (setting >= 0) {
    return setting == 1;
  }
  return physical_type == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU;
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
  //
  // |exact_write_coverage| may only be true when the GPU command is proven to
  // overwrite every byte in [offset, offset+range) (e.g. vkCmdCopyBuffer /
  // vkCmdFillBuffer). Compute shader descriptors often span the full
  // allocation even when the shader only writes a slice (output_offset +
  // count), so compute dispatches pass exact_write_coverage=false and
  // conservatively upload any host-dirty buffer.
  bool needs_upload(
      bool read,
      bool write,
      size_t offset,
      size_t range,
      size_t buffer_size,
      bool exact_write_coverage = false) const {
    const bool full_overwrite =
        exact_write_coverage && write && !read && offset == 0 &&
        range >= buffer_size;
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
      size_t buffer_size,
      bool exact_write_coverage = false) {
    const bool need_upload = needs_upload(
        read, write, offset, range, buffer_size, exact_write_coverage);
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
