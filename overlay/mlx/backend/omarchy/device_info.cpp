// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

// mlx::core::gpu device_info backed by Omarchy discovery.
//
// Public keys surfaced here must be the ones oMLX (dflash, oq, memory_monitor)
// and TensorFold (kernels/device.py, threads.py, prefill_mm.py) read at the
// pinned commits. See receipts/2026-10-04-hwprobe-device-info for the call-site
// list and the receipt whose probes exercise them on T6021.
//
// `max_recommended_working_set_size` and `gpu_cores` are NOT invented. The
// former is computed each first call from /proc/meminfo MemTotal and the
// Vulkan-reported unified heap (see working_set_size below); the latter
// is omitted entirely unless the device-tree compatible string matches a
// chip-id in the static gpu_cores_for_ map.

#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include <variant>

#include "mlx/backend/gpu/device_info.h"
#include "mlx/backend/omarchy/device.h"

namespace mlx::core::gpu {

namespace {

// Map an Apple chip-id device-tree "compatible" leaf (the trailing token,
// e.g. "apple,t6021") to the public marketing name. Anything not in this
// table is omitted from device_info rather than guessed.
struct ChipIdEntry {
  const char* chip_id;
  const char* marketing_name;
  size_t gpu_cores;
};

constexpr ChipIdEntry kChipIdTable[] = {
    {"apple,t8103", "Apple M1", 8},
    {"apple,t8112", "Apple M2", 8},
    {"apple,t6000", "Apple M1 Pro", 16},
    {"apple,t6001", "Apple M1 Max", 32},
    {"apple,t6002", "Apple M1 Ultra", 64},
    {"apple,t6020", "Apple M2 Pro", 19},
    {"apple,t6021", "Apple M2 Max", 38},
    {"apple,t6022", "Apple M2 Ultra", 76},
};

const ChipIdEntry* find_chip_entry(const std::string& compatible) {
  // Compatible is space-separated, generic last; the most specific Apple
  // chip-id appears earlier. Match on substring so partial compat strings
  // resolve cleanly.
  for (const auto& entry : kChipIdTable) {
    if (compatible.find(entry.chip_id) != std::string::npos) {
      return &entry;
    }
  }
  return nullptr;
}

// Read /proc/device-tree/compatible once (one fd open, one read). Empty
// string on any error or absence — the caller omits marketing/gpu keys
// rather than reporting an empty stub.
std::string read_dt_compatible() {
  int fd = ::open("/proc/device-tree/compatible", O_RDONLY);
  if (fd < 0) {
    return {};
  }
  std::string out;
  char buf[256];
  while (true) {
    ssize_t n = ::read(fd, buf, sizeof(buf));
    if (n <= 0) {
      break;
    }
    out.append(buf, static_cast<size_t>(n));
  }
  ::close(fd);
  // Strip trailing NULs the device tree appends.
  while (!out.empty() && out.back() == '\0') {
    out.pop_back();
  }
  return out;
}

// Read /proc/meminfo MemTotal in kibibytes; return 0 on failure. Called
// lazily on first device_info() per process; subsequent calls reuse the
// cached value. /proc/meminfo is plain ASCII, so a 256-byte buffer is
// plenty for one line.
size_t read_mem_total_bytes() {
  static std::once_flag once;
  static size_t cached = 0;
  std::call_once(once, []() {
    int fd = ::open("/proc/meminfo", O_RDONLY);
    if (fd < 0) {
      return;
    }
    char buf[4096];
    ssize_t total = ::read(fd, buf, sizeof(buf) - 1);
    ::close(fd);
    if (total <= 0) {
      return;
    }
    buf[total] = '\0';
    const char* line = std::strstr(buf, "MemTotal:");
    if (line == nullptr) {
      return;
    }
    unsigned long long kbytes = 0;
    if (std::sscanf(line, "MemTotal: %llu kB", &kbytes) != 1) {
      return;
    }
    cached = static_cast<size_t>(kbytes) * 1024ull;
  });
  return cached;
}

// max_recommended_working_set_size — oMLX reads this in dflash.py:526
// (BatchGenerator wired-limit acquire) and oq.py:3629 (calibration
// memory budget). The honest formula:
//
//   working_set = min(vulkan_heap_bytes, 80% of MemTotal)
//
// Why min(): vulkan_heap is the largest HOST_VISIBLE memory heap reported
// by the ICD (omarchy allocator.h: the only type we can allocate from).
// MemTotal is the kernel-reported total RAM. The 80% reserve prevents the
// kernel from starving: contiguous-page pressure, slab, page tables, ANE
// driver buffers, and the Vulkan ICD's own bookkeeping all live in
// MemTotal but not in the heap budget. Capped at vulkan_heap because
// going higher would request wired pages the GPU cannot actually use.
//
// Returns 0 when either input is unknown — oMLX's `.get(..., 0)` fallback
// treats 0 as "no wired limit requested".
size_t working_set_size(size_t vulkan_heap_bytes) {
  const size_t mem_total = read_mem_total_bytes();
  if (vulkan_heap_bytes == 0 || mem_total == 0) {
    return 0;
  }
  const size_t kernel_reserve = mem_total - (mem_total / 5); // 80% of MemTotal
  return std::min(vulkan_heap_bytes, kernel_reserve);
}

} // namespace

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
    // Alias for oMLX/TensorFold readers that key on memory_size. Same
    // bytes as total_memory, the actual Vulkan-reported unified heap.
    info["memory_size"] = caps.total_memory;
    info["max_recommended_working_set_size"] =
        working_set_size(caps.total_memory);
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
    // Marketing name and GPU core count come from the device tree, NOT
    // from Honeykrisp (which does not expose GPU cores). The chip-id
    // table is the only source: an unknown chip leaves both keys absent
    // rather than reporting a guessed number. Compatible tokens are
    // NUL-separated in the DT blob; interior NULs become single spaces
    // so the reported string is a normal printable compatible list.
    static const std::string dt_compatible_raw = read_dt_compatible();
    if (!dt_compatible_raw.empty()) {
      std::string dt_compatible = dt_compatible_raw;
      std::replace(dt_compatible.begin(), dt_compatible.end(), '\0', ' ');
      info["chip_compatible"] = dt_compatible;
      if (const ChipIdEntry* chip = find_chip_entry(dt_compatible_raw)) {
        info["marketing_name"] = chip->marketing_name;
        info["gpu_cores"] = chip->gpu_cores;
      }
    }
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
