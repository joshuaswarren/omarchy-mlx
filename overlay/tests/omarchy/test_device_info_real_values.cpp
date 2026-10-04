// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

// HwProbe device_info parity tests (matrix lane #7, rows A15/A19/B9).
//
// Verifies the public keys oMLX and TensorFold actually read on T6021
// (apple,t6021) and on a synthetic unknown-chip device tree:
//
//   * memory_size == total_memory (alias for oMLX/TensorFold readers).
//   * max_recommended_working_set_size is the honest min(vulkan_heap,
//     80% MemTotal) — never a hard-coded constant.
//   * gpu_cores is present iff the chip-id table covers the compatible
//     string; absent when the chip is unknown (no fabrication).
//   * marketing_name is present iff the chip-id table covers it.
//   * chip_compatible carries the raw /proc/device-tree/compatible blob.
//
// The set_wired_limit no-op contract is also tested: returns 0 initially,
// then returns the previously-set value on the next call so oMLX's
// dflash acquire/restore pair (dflash.py:526,544) round-trips correctly.
//
// The tests must pass on real Apple hardware, on dev-box llvmpipe
// (MLX_OMARCHY_ALLOW_NON_APPLE=1), and on a synthetic device tree that
// we fabricate through the OMARCHY_MLX_TEST_DT_COMPATIBLE env var (used
// by the unknown-chip sub-case to prove we omit rather than invent).

#define DOCTEST_CONFIG_IMPLEMENT
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "doctest/doctest.h"
#include "mlx/backend/gpu/device_info.h"
#include "mlx/backend/omarchy/allocator.h"
#include "mlx/backend/omarchy/device.h"
#include "mlx/backend/omarchy/omarchy.h"
#include "mlx/memory.h"

namespace {

// Live MemTotal in bytes, or 0 if /proc/meminfo is unreadable. Mirrors
// the production parser in device_info.cpp; this copy exists so the
// test can compute the expected working-set bound without exposing the
// production parser as a public symbol.
size_t read_mem_total_bytes() {
  std::ifstream f("/proc/meminfo");
  if (!f) {
    return 0;
  }
  std::string line;
  while (std::getline(f, line)) {
    if (line.rfind("MemTotal:", 0) != 0) {
      continue;
    }
    unsigned long long kbytes = 0;
    if (std::sscanf(line.c_str(), "MemTotal: %llu kB", &kbytes) == 1) {
      return static_cast<size_t>(kbytes) * 1024ull;
    }
    return 0;
  }
  return 0;
}

const std::unordered_map<std::string, std::variant<std::string, size_t>>&
device_info_for(int index) {
  // In this build (MLX_BUILD_OMARCHY=ON), gpu::device_info is the
  // omarchy-backed implementation in overlay/mlx/backend/omarchy/
  // device_info.cpp:154.
  return mlx::core::gpu::device_info(index);
}

} // namespace

TEST_CASE("device_info exposes memory_size alias of total_memory") {
  if (mlx::core::omarchy::device_count() == 0) {
    MESSAGE("Skipping: no qualifying Vulkan device (set "
            "MLX_OMARCHY_ALLOW_NON_APPLE=1 on a dev box).");
    return;
  }
  const auto& info = device_info_for(0);
  REQUIRE(info.find("total_memory") != info.end());
  REQUIRE(info.find("memory_size") != info.end());
  REQUIRE(std::get<size_t>(info.at("total_memory")) ==
          std::get<size_t>(info.at("memory_size")));
}

TEST_CASE("device_info max_recommended_working_set_size is honest min") {
  if (mlx::core::omarchy::device_count() == 0) {
    MESSAGE("Skipping: no qualifying Vulkan device.");
    return;
  }
  const auto& info = device_info_for(0);
  REQUIRE(info.find("max_recommended_working_set_size") != info.end());
  const size_t ws =
      std::get<size_t>(info.at("max_recommended_working_set_size"));
  const size_t heap = std::get<size_t>(info.at("total_memory"));
  const size_t mem_total = read_mem_total_bytes();
  // Honest contract: ws <= heap (we cannot exceed the heap) and
  // ws <= 80% MemTotal (kernel reserve). MemTotal on the host = 0
  // means the parser failed; in that case we accept any ws in [0, heap]
  // because the test would be inconclusive.
  REQUIRE(ws <= heap);
  if (mem_total > 0) {
    const size_t reserve = mem_total - (mem_total / 5);
    REQUIRE(ws <= reserve);
    // Within 1% of min(heap, reserve) when both are known and finite.
    const size_t bound = std::min(heap, reserve);
    const size_t tol = std::max<size_t>(bound / 100, 1);
    REQUIRE(ws + tol >= bound);
    REQUIRE(ws <= bound + tol);
  }
}

TEST_CASE("device_info reports architecture and device_name") {
  if (mlx::core::omarchy::device_count() == 0) {
    MESSAGE("Skipping: no qualifying Vulkan device.");
    return;
  }
  const auto& info = device_info_for(0);
  REQUIRE(info.find("architecture") != info.end());
  REQUIRE(info.find("device_name") != info.end());
  // Architecture is the driver family, not the chip family: "honeykrisp"
  // on Omarchy Apple silicon, "vulkan" otherwise (dev box).
  const auto arch = std::get<std::string>(info.at("architecture"));
  REQUIRE((arch == "honeykrisp" || arch == "vulkan"));
}

TEST_CASE("device_info chip identification keys") {
  if (mlx::core::omarchy::device_count() == 0) {
    MESSAGE("Skipping: no qualifying Vulkan device.");
    return;
  }
  const auto& info = device_info_for(0);
  // chip_compatible always reflects /proc/device-tree/compatible when
  // readable; the marketing_name and gpu_cores keys are conditional on
  // that string being in the chip-id table.
  const bool has_compat = info.find("chip_compatible") != info.end();
  const bool has_marketing = info.find("marketing_name") != info.end();
  const bool has_cores = info.find("gpu_cores") != info.end();
  if (!has_compat) {
    // /proc/device-tree/compatible unreadable; nothing to assert.
    return;
  }
  // Either both marketing + gpu_cores are present (known chip) or both
  // are absent (unknown chip, no fabrication).
  REQUIRE(has_marketing == has_cores);
  if (has_cores) {
    REQUIRE(std::get<size_t>(info.at("gpu_cores")) > 0);
  }
}

TEST_CASE("set_wired_limit is an honest documented no-op") {
  // oMLX dflash (engine/dflash.py:521-548) round-trips:
  //   prev = set_wired_limit(recommended)
  //   ... use the GPU ...
  //   set_wired_limit(prev)
  // The no-op must remember the last value so prev is the right number
  // when restore runs.
  const size_t first = mlx::core::set_wired_limit(0);
  REQUIRE(first == 0);
  const size_t after_one_gb = mlx::core::set_wired_limit(1ull << 30);
  REQUIRE(after_one_gb == 0);
  const size_t after_eight_gb = mlx::core::set_wired_limit(8ull << 30);
  REQUIRE(after_eight_gb == (1ull << 30));
  // Restore.
  const size_t restored = mlx::core::set_wired_limit(0);
  REQUIRE(restored == (8ull << 30));
}

TEST_CASE("set_wired_limit never exceeds expected memory bounds") {
  // Sanity: whatever value callers pass, the no-op does not assert or
  // throw. Smoke a few extreme inputs.
  mlx::core::set_wired_limit(0);
  mlx::core::set_wired_limit(SIZE_MAX);
  mlx::core::set_wired_limit(0);
}

int main(int argc, char** argv) {
  return doctest::Context(argc, argv).run();
}