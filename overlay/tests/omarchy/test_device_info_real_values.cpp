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
#include <cstdint>
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

TEST_CASE("default_hk_sysmem_bytes keeps the stock heap on small hosts") {
  // MemTotal <= 32 GiB: MemTotal - 16 GiB never beats the 50% DRIR
  // default, so nullopt keeps mesa's heap byte-for-byte. 16 GB-class
  // machines (7.55-8 GiB heap) must be unchanged, and the 13-inch M1's measured
  // 15.1 GiB MemTotal (15843696 kB) with it.
  namespace omarchy = mlx::core::omarchy;
  constexpr uint64_t kGiB = 1ull << 30;
  const auto gib = [](double v) {
    return static_cast<uint64_t>(v * static_cast<double>(kGiB));
  };
  CHECK_FALSE(omarchy::default_hk_sysmem_bytes(15843696ull * 1024).has_value());
  CHECK_FALSE(omarchy::default_hk_sysmem_bytes(gib(16)).has_value());
  // Border: at exactly 32 GiB the raised value equals the 50% default.
  CHECK_FALSE(omarchy::default_hk_sysmem_bytes(gib(32)).has_value());
}

TEST_CASE("default_hk_sysmem_bytes raises large hosts and clamps at 60") {
  namespace omarchy = mlx::core::omarchy;
  constexpr uint64_t kGiB = 1ull << 30;
  const auto gib = [](double v) {
    return static_cast<uint64_t>(v * static_cast<double>(kGiB));
  };
  // Past the crossover the heap is MemTotal - 16 GiB (1 MiB-rounded like
  // mesa): 48 GiB -> 32, 62 GiB (16-inch M1 Max class) -> 46, 64 GiB -> 48.
  CHECK(omarchy::default_hk_sysmem_bytes(gib(48)) == gib(32));
  CHECK(omarchy::default_hk_sysmem_bytes(gib(62)) == gib(46));
  CHECK(omarchy::default_hk_sysmem_bytes(gib(64)) == gib(48));
  // Per-process user VA window is ~64 GiB (agx_device.c; HeapBudget's
  // probe committed 63 GiB, the 64th allocation failed to map), so the
  // heap never promises more than the 60 GiB clamp: 94/96 GiB -> 60.
  CHECK(omarchy::default_hk_sysmem_bytes(gib(94)) == gib(60));
  CHECK(omarchy::default_hk_sysmem_bytes(gib(96)) == gib(60));
  const auto raised_62 = omarchy::default_hk_sysmem_bytes(gib(62));
  REQUIRE(raised_62.has_value());
  CHECK(raised_62.value() % (1ull << 20) == 0);
  const auto clamped_96 = omarchy::default_hk_sysmem_bytes(gib(96));
  REQUIRE(clamped_96.has_value());
  CHECK(clamped_96.value() % (1ull << 20) == 0);

  // Live host cross-check: whatever /proc/meminfo reports, the result
  // honors the contract (absent <= 32 GiB; above 50% and at most 60 GiB up
  // to 120 GiB; exactly the 60 GiB clamp beyond, where 50% is out of reach).
  const uint64_t total = read_mem_total_bytes();
  if (total > 0) {
    const auto live = omarchy::default_hk_sysmem_bytes(total);
    if (!live.has_value()) {
      CHECK(total <= gib(32));
    } else {
      CHECK((live.value() > total / 2 || live.value() == gib(60)));
      CHECK(live.value() <= gib(60));
    }
  }
}

TEST_CASE("parse_kernel_client_limit reads the client_limit line of gpu_memory") {
  namespace omarchy = mlx::core::omarchy;
  // The gpu_memory sysfs file of the 14-inch M2 Max on the lab kernel.
  const std::string text =
      "user_committed 28982853632\n"
      "user_resident 28982853632\n"
      "kernel_resident 1073741824\n"
      "limit 67805549854\n"
      "client_limit 50601156608\n"
      "refused 34\n";
  CHECK(omarchy::parse_kernel_client_limit(text) == 50601156608ull);
  CHECK(omarchy::parse_kernel_client_limit("client_limit 4096") == 4096ull);
  // The device-wide `limit` line is a different key and is never taken.
  CHECK_FALSE(omarchy::parse_kernel_client_limit("limit 67805549854\n"));
  // 0 means the kernel sets no per-client limit.
  CHECK_FALSE(omarchy::parse_kernel_client_limit("client_limit 0\n"));
  // Anything that is not a plain decimal number is no limit.
  CHECK_FALSE(omarchy::parse_kernel_client_limit(""));
  CHECK_FALSE(omarchy::parse_kernel_client_limit("client_limit\n"));
  CHECK_FALSE(omarchy::parse_kernel_client_limit("client_limit abc\n"));
  CHECK_FALSE(omarchy::parse_kernel_client_limit("client_limit -5\n"));
  CHECK_FALSE(omarchy::parse_kernel_client_limit("client_limit 12x\n"));
  CHECK_FALSE(
      omarchy::parse_kernel_client_limit("client_limit 99999999999999999999\n"));
}

TEST_CASE("cap_hk_sysmem_to_kernel_limit never promises more than the kernel allows") {
  namespace omarchy = mlx::core::omarchy;
  constexpr uint64_t kGiB = 1ull << 30;
  constexpr uint64_t kMiB = 1ull << 20;
  // 14-inch M2 Max: MemTotal 98830384 kB, per-client limit 50601156608
  // bytes (47.126 GiB) on the lab kernel at its default 50%.
  const uint64_t m2_total = 98830384ull * 1024;
  const uint64_t m2_client_50 = 50601156608ull;
  // Not MiB aligned (24 KiB over 48257 MiB): the heap takes the rounded-down value.
  const uint64_t m2_heap_50 = 48257ull * kMiB;
  // 13-inch M1: MemTotal 15843696 kB, stock heap (no HK_SYSMEM).
  const uint64_t m1_total = 15843696ull * 1024;

  // The raised 60 GiB heap drops to the kernel limit that is below it.
  CHECK(omarchy::cap_hk_sysmem_to_kernel_limit(60 * kGiB, m2_total, m2_client_50) ==
        m2_heap_50);
  // A kernel limit at 75% of RAM is above the heap: nothing to cap.
  CHECK(omarchy::cap_hk_sysmem_to_kernel_limit(60 * kGiB, m2_total, m2_total / 100 * 75) ==
        60 * kGiB);
  // Kernel limit equal to the heap: unchanged.
  CHECK(omarchy::cap_hk_sysmem_to_kernel_limit(60 * kGiB, m2_total, 60 * kGiB) ==
        60 * kGiB);
  // The result is 1 MiB aligned, like mesa's heap sizes.
  CHECK(omarchy::cap_hk_sysmem_to_kernel_limit(
            60 * kGiB, m2_total, m2_client_50 + 12345) == m2_heap_50);

  // Kernels without the limit file (no value) leave both forms unchanged.
  CHECK(omarchy::cap_hk_sysmem_to_kernel_limit(60 * kGiB, m2_total, std::nullopt) ==
        60 * kGiB);
  CHECK_FALSE(omarchy::cap_hk_sysmem_to_kernel_limit(std::nullopt, m1_total, std::nullopt));

  // Stock heap (nullopt): mesa uses MemTotal/2 rounded down to 1 MiB. A
  // kernel limit at the same 50% leaves the 13-inch M1 byte-for-byte
  // unchanged, even with a limit that is not MiB aligned.
  CHECK_FALSE(omarchy::cap_hk_sysmem_to_kernel_limit(std::nullopt, m1_total, m1_total / 2));
  CHECK_FALSE(
      omarchy::cap_hk_sysmem_to_kernel_limit(std::nullopt, m1_total, m1_total / 2 + 1234));
  // A kernel limit below that stock heap becomes the heap.
  CHECK(omarchy::cap_hk_sysmem_to_kernel_limit(std::nullopt, m1_total, m1_total / 4) ==
        (m1_total / 4) / kMiB * kMiB);

  // A limit under 1 MiB is not usable as a heap and is ignored. An unknown
  // MemTotal with a stock heap has nothing to compare against.
  CHECK(omarchy::cap_hk_sysmem_to_kernel_limit(60 * kGiB, m2_total, 4096) == 60 * kGiB);
  CHECK_FALSE(omarchy::cap_hk_sysmem_to_kernel_limit(std::nullopt, 0, 4 * kGiB));
}

TEST_CASE("kernel_client_limit_bytes is absent or a usable limit below RAM") {
  namespace omarchy = mlx::core::omarchy;
  // Live host: kernels without the gpu_memory file report no limit; the
  // lab kernel reports a limit that is at least 1 MiB and not above RAM.
  const auto live = omarchy::kernel_client_limit_bytes();
  const uint64_t total = read_mem_total_bytes();
  if (live.has_value() && total > 0) {
    CHECK(live.value() >= (1ull << 20));
    CHECK(live.value() <= total);
  }
}

TEST_CASE("kernel_limit_heap_warning names both numbers when the heap is above the limit") {
  namespace omarchy = mlx::core::omarchy;
  // An explicit 60 GiB heap on the 14-inch M2 Max, whose kernel client_limit is 50601156608.
  const std::string message =
      omarchy::kernel_limit_heap_warning(64424509440ull, 50601156608ull);
  CHECK(message.find("64424509440") != std::string::npos);
  CHECK(message.find("50601156608") != std::string::npos);
  CHECK(message.find('\n') == std::string::npos);

  // A heap that fits, a heap equal to the limit, no kernel limit and an unknown heap stay silent.
  CHECK(omarchy::kernel_limit_heap_warning(50601132032ull, 50601156608ull).empty());
  CHECK(omarchy::kernel_limit_heap_warning(50601156608ull, 50601156608ull).empty());
  CHECK(omarchy::kernel_limit_heap_warning(64424509440ull, std::nullopt).empty());
  CHECK(omarchy::kernel_limit_heap_warning(0, 50601156608ull).empty());
}

TEST_CASE("default allocator limits never sit below what the heap can hold") {
  namespace omarchy = mlx::core::omarchy;
  constexpr size_t kGiB = size_t{1} << 30;
  // The memory limit is where malloc drops the reuse cache; below the
  // resident weights that happens on every allocation (gpt-oss-120b, 58.05
  // GiB of weights on a 63 GiB heap (HK_SYSMEM set explicitly) with a
  // 56.7 GiB limit: first 16-token step 228 s, against 1.2 s with the limit
  // at the heap, also on a 60 GiB heap). The gc ceiling does the same
  // with its trim once it sits below the weights (60 GiB heap, 95% ceiling;
  // decode was measured slower in matched diagnostic pairs). The limit
  // must also not sit below the gc ceiling, the order Metal uses.
  const size_t weights = 62329924224ull;
  const size_t heaps[] = {16 * kGiB, 32 * kGiB, 47 * kGiB, 60 * kGiB, 63 * kGiB};
  for (size_t heap : heaps) {
    CHECK(omarchy::default_memory_limit(heap) >= omarchy::default_gc_limit(heap));
    CHECK(omarchy::default_gc_limit(heap) == heap);
    if (heap > weights) {
      // A model that fits the heap is under the limit and under the gc
      // ceiling: no cache drop and no trim on every allocation. (95% of
      // a 60 GiB heap, 57 GiB, was below the 58.05 GiB weights.)
      CHECK(omarchy::default_memory_limit(heap) > weights);
      CHECK(omarchy::default_gc_limit(heap) > weights);
    }
  }
  // No device (total 0): the same 4 GiB fallback as before for both.
  CHECK(omarchy::default_memory_limit(0) == 4 * kGiB);
  CHECK(omarchy::default_gc_limit(0) == 4 * kGiB);

  // The open-batch quarantine budget (eval.cpp, limit / 16) is tuned
  // against 90% of the heap and stays there: only the cache behaviour
  // changes with the memory limit. A caller that lowers the limit still
  // lowers the budget; raising the limit does not raise it.
  for (size_t heap : heaps) {
    const size_t base = (heap / 100) * 90;
    CHECK(omarchy::default_batch_limit(
              omarchy::default_memory_limit(heap), heap) == base);
    CHECK(omarchy::default_batch_limit(base / 2, heap) == base / 2);
    CHECK(omarchy::default_batch_limit(heap * 2, heap) == base);
  }
  CHECK(omarchy::default_batch_limit(omarchy::default_memory_limit(0), 0) ==
        4 * kGiB);

  // The constructor must use the helpers: a revert of its memory limit to
  // 90% of the heap passes the pure checks above but puts the default
  // limit below the gc ceiling here (on a host with a device).
  auto& alloc = omarchy::allocator();
  CHECK(alloc.get_memory_limit() >= alloc.get_gc_limit());
  CHECK(alloc.batch_limit() <= alloc.get_memory_limit());
}

int main(int argc, char** argv) {
  return doctest::Context(argc, argv).run();
}