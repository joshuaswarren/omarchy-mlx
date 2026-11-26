// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

#pragma once

#include "mlx/backend/omarchy/ane/manifest.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>
#include <unistd.h>

namespace mlx::core::omarchy::ane::detail {

constexpr int kWorkerControlFd = 3;
constexpr int kWorkerStagingFd = 4;
constexpr int kWorkerManifestFd = 5;
constexpr int kWorkerHardwareLockFd = 6;
constexpr const char* kRuntimeOwnershipDirectory =
    "/run/lock/mlx-omarchy-ane";
constexpr const char* kRuntimeOwnershipLockPath =
    "/run/lock/mlx-omarchy-ane/device.lock";
constexpr const char* kRuntimeQuarantinePath =
    "/run/lock/mlx-omarchy-ane/quarantine";
constexpr int kWorkerPayloadFdBase = 16;
constexpr size_t kWorkerDetailBytes = 1024;
constexpr uint32_t kWorkerProtocolVersion = 1;

inline std::runtime_error runtime_error(const std::string& reason) {
  return std::runtime_error("[omarchy-ane] runtime: " + reason + ".");
}

inline std::filesystem::path installed_worker_path(
    const std::filesystem::path& library_path) {
  const auto prefix = library_path.parent_path().parent_path();
  if (prefix.empty()) {
    throw runtime_error("loaded libmlx path has no installation prefix");
  }
  return prefix / "bin" / "mlx-omarchy-ane-worker";
}
inline std::filesystem::path worker_executable_path(
    const std::filesystem::path& loaded_library,
    const std::filesystem::path& build_library,
    const std::filesystem::path& build_worker) {
  std::error_code error;
  const auto loaded_image = std::filesystem::canonical(loaded_library, error);
  if (error) {
    throw runtime_error("private ANE worker executable not found");
  }

  error.clear();
  const auto build_image = std::filesystem::canonical(build_library, error);
  if (!error && loaded_image == build_image) {
    error.clear();
    const auto worker = std::filesystem::canonical(build_worker, error);
    if (!error && ::access(worker.c_str(), X_OK) == 0) {
      return worker;
    }
    throw runtime_error("private ANE worker executable not found");
  }

  error.clear();
  const auto worker = std::filesystem::canonical(
      installed_worker_path(loaded_image), error);
  if (!error && ::access(worker.c_str(), X_OK) == 0) {
    return worker;
  }
  throw runtime_error("private ANE worker executable not found");
}
inline int worker_source_fd_floor(
    size_t payload_count,
    uint64_t descriptor_limit) {
  const uint64_t payloads = payload_count;
  if (payloads > std::numeric_limits<uint64_t>::max() - kWorkerPayloadFdBase) {
    throw runtime_error("bundle payload count exceeds worker descriptor limit");
  }
  const uint64_t highest_destination = payloads == 0
      ? uint64_t{kWorkerHardwareLockFd}
      : uint64_t{kWorkerPayloadFdBase} + payloads - 1;
  const uint64_t source_floor = highest_destination + 1;
  const uint64_t source_count = payloads + 5;
  const uint64_t integer_limit = std::numeric_limits<int>::max();
  const uint64_t effective_limit = std::min(descriptor_limit, integer_limit);
  if (source_floor >= effective_limit ||
      source_count > effective_limit - source_floor) {
    throw runtime_error("bundle payload count exceeds worker descriptor limit");
  }
  return static_cast<int>(source_floor);
}

using RuntimeClock = std::chrono::steady_clock;

struct CheckedDeadline {
  RuntimeClock::time_point time;
  int64_t monotonic_nanoseconds;
};

inline CheckedDeadline checked_deadline(std::chrono::milliseconds duration) {
  if (duration.count() <= 0) {
    throw std::invalid_argument("[omarchy-ane] runtime: deadline must be positive.");
  }
  const auto now = RuntimeClock::now();
  const auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(
      RuntimeClock::time_point::max() - now);
  if (duration > remaining) {
    throw std::invalid_argument("[omarchy-ane] runtime: deadline exceeds the monotonic clock range.");
  }
  const auto deadline = now + std::chrono::duration_cast<RuntimeClock::duration>(duration);
  const auto maximum_nanoseconds = std::chrono::duration_cast<RuntimeClock::duration>(
      std::chrono::nanoseconds::max());
  if (deadline.time_since_epoch() > maximum_nanoseconds) {
    throw std::invalid_argument("[omarchy-ane] runtime: deadline exceeds the worker protocol range.");
  }
  const int64_t nanoseconds = std::chrono::duration_cast<std::chrono::nanoseconds>(
      deadline.time_since_epoch()).count();
  if (nanoseconds <= 0) {
    throw std::invalid_argument("[omarchy-ane] runtime: deadline is not representable.");
  }
  return {deadline, nanoseconds};
}

inline void ensure_worker_before(
    RuntimeClock::time_point deadline,
    const std::string& phase) {
  if (RuntimeClock::now() >= deadline) {
    throw runtime_error("deadline expired before " + phase);
  }
}

inline size_t checked_size(uint64_t value, const std::string& label) {
  if (value > std::numeric_limits<size_t>::max()) {
    throw runtime_error(label + " exceeds host size_t");
  }
  return static_cast<size_t>(value);
}

inline uint64_t checked_product(uint64_t lhs, uint64_t rhs, const std::string& label) {
  if (lhs != 0 && rhs > std::numeric_limits<uint64_t>::max() / lhs) {
    throw runtime_error(label + " overflows uint64");
  }
  return lhs * rhs;
}

inline uint64_t checked_sum(uint64_t lhs, uint64_t rhs, const std::string& label) {
  if (rhs > std::numeric_limits<uint64_t>::max() - lhs) {
    throw runtime_error(label + " overflows uint64");
  }
  return lhs + rhs;
}

inline size_t packed_element_offset(
    const AneProgramBinding& binding,
    uint64_t element) {
  const uint64_t batch = binding.nchw[0];
  const uint64_t channels = binding.nchw[1];
  const uint64_t height = binding.nchw[2];
  const uint64_t width = binding.nchw[3];
  const uint64_t plane_stride = binding.nchw[4];
  const uint64_t row_stride = binding.nchw[5];
  if (batch == 0 || channels == 0 || height == 0 || width == 0) {
    throw runtime_error("tensor '" + binding.tensor + "' packed NCHW is empty");
  }
  const uint64_t plane_elements =
      checked_product(height, width, "packed plane element count");
  const uint64_t plane_count =
      checked_product(batch, channels, "packed plane count");
  if (element >= checked_product(
                     plane_count, plane_elements, "packed tensor element count")) {
    throw runtime_error("tensor '" + binding.tensor + "' packed element exceeds NCHW");
  }
  const uint64_t plane = element / plane_elements;
  const uint64_t within_plane = element % plane_elements;
  const uint64_t row = within_plane / width;
  const uint64_t column = within_plane % width;
  uint64_t offset = checked_product(plane, plane_stride, "packed plane offset");
  offset = checked_sum(
      offset, checked_product(row, row_stride, "packed row offset"), "packed offset");
  offset = checked_sum(
      offset, checked_product(column, uint64_t{2}, "packed column offset"), "packed offset");
  return checked_size(offset, "packed byte offset");
}

inline void pack_binding(
    const AneProgramBinding& binding,
    const std::vector<uint8_t>& dense,
    std::vector<uint8_t>& packed) {
  const size_t logical = checked_size(binding.logical_bytes, "logical byte count");
  const size_t allocation =
      checked_size(binding.allocation_bytes, "allocation byte count");
  const size_t begin = checked_size(
      checked_product(binding.element_offset, uint64_t{2}, "element byte offset"),
      "element byte offset");
  if (begin > dense.size() || logical > dense.size() - begin) {
    throw runtime_error(
        "tensor '" + binding.tensor + "' dense staging byte count is " +
        std::to_string(dense.size()) + ", expected at least " +
        std::to_string(begin + logical));
  }
  if (packed.size() != allocation) {
    throw runtime_error(
        "tensor '" + binding.tensor + "' packed byte count is " +
        std::to_string(packed.size()) + ", expected " +
        std::to_string(allocation));
  }

  std::fill(packed.begin(), packed.end(), 0);
  for (size_t element = 0; element < logical / 2; ++element) {
    const size_t destination = packed_element_offset(binding, element);
    if (destination > packed.size() || packed.size() - destination < 2) {
      throw runtime_error("tensor '" + binding.tensor + "' packed layout exceeds allocation");
    }
    std::memcpy(packed.data() + destination, dense.data() + begin + element * 2, 2);
  }
}

inline std::vector<uint8_t> pack_binding(
    const AneProgramBinding& binding,
    const std::vector<uint8_t>& dense) {
  std::vector<uint8_t> packed(
      checked_size(binding.allocation_bytes, "allocation byte count"));
  pack_binding(binding, dense, packed);
  return packed;
}

inline void unpack_binding(
    const AneProgramBinding& binding,
    const std::vector<uint8_t>& packed,
    std::vector<uint8_t>& dense) {
  const size_t logical = checked_size(binding.logical_bytes, "logical byte count");
  const size_t allocation =
      checked_size(binding.allocation_bytes, "allocation byte count");
  const size_t begin = checked_size(
      checked_product(binding.element_offset, uint64_t{2}, "element byte offset"),
      "element byte offset");
  if (packed.size() != allocation) {
    throw runtime_error(
        "tensor '" + binding.tensor + "' packed byte count is " +
        std::to_string(packed.size()) + ", expected " + std::to_string(allocation));
  }
  if (begin > dense.size() || logical > dense.size() - begin) {
    throw runtime_error(
        "tensor '" + binding.tensor + "' dense staging byte count is " +
        std::to_string(dense.size()) + ", expected at least " +
        std::to_string(begin + logical));
  }

  for (size_t element = 0; element < logical / 2; ++element) {
    const size_t source = packed_element_offset(binding, element);
    if (source > packed.size() || packed.size() - source < 2) {
      throw runtime_error("tensor '" + binding.tensor + "' packed layout exceeds allocation");
    }
    std::memcpy(dense.data() + begin + element * 2, packed.data() + source, 2);
  }
}

enum class WorkerOperation : uint32_t {
  execute = 1,
  shutdown = 2,
};

enum class WorkerReplyKind : uint32_t {
  ready = 1,
  executed = 2,
  stopped = 3,
  failed = 4,
  uncertain = 5,
};

struct WorkerCommand {
  uint32_t version{kWorkerProtocolVersion};
  WorkerOperation operation{WorkerOperation::execute};
  uint64_t serial{0};
  int64_t deadline_monotonic_nanoseconds{0};
};

struct WorkerReply {
  uint32_t version{kWorkerProtocolVersion};
  WorkerReplyKind kind{WorkerReplyKind::failed};
  uint64_t serial{0};
  uint64_t released_programs{0};
  char detail[kWorkerDetailBytes]{};
};

int run_worker(
    int control_fd,
    int staging_fd,
    size_t staging_size,
    const std::filesystem::path& manifest_path,
    const std::map<std::string, std::filesystem::path>& payload_paths);

} // namespace mlx::core::omarchy::ane::detail
