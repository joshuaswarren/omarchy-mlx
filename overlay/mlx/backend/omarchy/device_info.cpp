// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

// mlx::core::gpu device_info backed by Omarchy discovery.

#include <mutex>
#include <unordered_map>
#include <variant>

#include "mlx/backend/gpu/device_info.h"
#include "mlx/backend/omarchy/device.h"

namespace mlx::core::gpu {

bool is_available() {
  return omarchy::device_count() > 0;
}

int device_count() {
  return omarchy::device_count();
}

const std::unordered_map<std::string, std::variant<std::string, size_t>>&
device_info(int device_index) {
  using InfoMap =
      std::unordered_map<std::string, std::variant<std::string, size_t>>;
  // One mutex guards cache lookup and insertion. Entries are filled once
  // under the lock and never mutated afterwards, so a returned reference is
  // stable and safe to read without a lock: std::unordered_map never moves
  // existing nodes on insertion.
  static std::mutex mutex;
  static std::unordered_map<int, InfoMap> cache;
  std::lock_guard<std::mutex> lk(mutex);
  auto [entry, inserted] = cache.try_emplace(device_index);
  if (!inserted) {
    return entry->second;
  }
  auto& info = entry->second;

  try {
    const auto& caps = omarchy::capability_report(device_index);
    info["device_name"] = caps.device_name;
    info["driver"] = caps.driver_name;
    info["driver_info"] = caps.driver_info;
    info["driver_sha"] = caps.driver_sha;
    info["icd_path"] = caps.icd_path;
    info["icd_source"] = caps.icd_source;
    info["expected_sha"] = caps.expected_sha;
    info["expected_sha_source"] = caps.expected_sha_source;
    info["api_version"] =
        std::to_string(VK_API_VERSION_MAJOR(caps.api_version)) + "." +
        std::to_string(VK_API_VERSION_MINOR(caps.api_version)) + "." +
        std::to_string(VK_API_VERSION_PATCH(caps.api_version));
    info["driver_version"] = static_cast<size_t>(caps.driver_version);
    info["vendor_id"] = static_cast<size_t>(caps.vendor_id);
    info["device_id"] = static_cast<size_t>(caps.device_id);
    info["architecture"] =
        caps.driver_name.find("Honeykrisp") != std::string::npos
        ? std::string("honeykrisp")
        : std::string("vulkan");
    info["total_memory"] = caps.total_memory;
    info["unified_memory"] = static_cast<size_t>(caps.unified_memory ? 1 : 0);
    info["host_visible_coherent"] =
        static_cast<size_t>(caps.host_visible_coherent ? 1 : 0);
    info["shader_float16"] = static_cast<size_t>(caps.shader_float16 ? 1 : 0);
    info["shader_int16"] = static_cast<size_t>(caps.shader_int16 ? 1 : 0);
    info["shader_int64"] = static_cast<size_t>(caps.shader_int64 ? 1 : 0);
    info["storage_buffer_16bit_access"] =
        static_cast<size_t>(caps.storage_buffer_16bit_access ? 1 : 0);
    info["cooperative_matrix_f32_8"] =
        static_cast<size_t>(caps.cooperative_matrix_f32_8 ? 1 : 0);
    info["queue_global_priority"] =
        static_cast<size_t>(caps.queue_global_priority ? 1 : 0);
    info["max_compute_shared_memory_size"] =
        static_cast<size_t>(caps.max_compute_shared_memory_size);
    info["max_compute_work_group_invocations"] =
        static_cast<size_t>(caps.max_compute_work_group_invocations);
    info["max_compute_work_group_size_x"] =
        static_cast<size_t>(caps.max_compute_work_group_size[0]);
    info["max_storage_buffer_range"] =
        static_cast<size_t>(caps.max_storage_buffer_range);
    info["timestamp_period_ns"] = static_cast<size_t>(caps.timestamp_period);
    if (caps.simulated) {
      info["simulated"] = size_t{1};
      info["simulation_profile"] = caps.simulation_profile;
      info["simulated_driver_variant"] = caps.simulated_driver_variant;
    }
  } catch (const std::exception&) {
    // Leave the entry empty; upstream documents that keys vary and the
    // unavailable case reports an empty map (no_gpu behavior).
  }
  return info;
}

} // namespace mlx::core::gpu
