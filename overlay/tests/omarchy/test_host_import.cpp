// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

// Host-pointer import tests (VK_EXT_external_memory_host): the zero-copy
// path that the array(void*, ...) ctor takes when the device supports it,
// the aliasing assertion sushi's data_ptr == slab.ptr check needs, and
// the copy fallback when the path is unavailable (env kill or unaligned
// pointer). Four subcases:
//   1. aligned: host buffer aligned to max(sysconf(_SC_PAGESIZE), 4096)
//      so a 16 KiB-page device is exercised at its real alignment.
//      When the import aliases, arr.data == the host pointer and a host
//      rewrite of the slab bytes is visible to a subsequent GPU sum.
//   2. subpage-offset: base aligned to the same page size, pointer at
//      base + 4096. On a 4 KiB-page device this is still aligned (the
//      import aliases); on a 16 KiB-page device it declines. Either way
//      the GPU op must be numerically correct.
//   3. misaligned-100: pointer at base + 100. This declines on every
//      driver; the copy path must run without error and stay correct.
//   4. env-kill: aligned buffer, MLX_OMARCHY_HOST_IMPORT=0 set before
//      the array ctor: copy path regardless of device capability.
// Each case prints ONE stable marker line:
//   HOST_IMPORT_ALIASED=<0|1> reason=<aliased|copy-fallback>
//       case=<aligned|subpage-offset|misaligned-100|env-kill>
//       align=<N> ptr_mod=<N>
// A device that legitimately declines passes its case (correctness is
// asserted on both paths); the marker plus the
// MLX_OMARCHY_HOST_IMPORT_DEBUG stderr lines are what a device receipt
// reads to see which path ran and why the other one declined.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

#include <unistd.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "mlx/backend/gpu/device_info.h"
#include "mlx/device.h"
#include "mlx/ops.h"
#include "mlx/stream.h"
#include "mlx/transforms.h"

using namespace mlx::core;

namespace {

size_t host_align() {
  long page = sysconf(_SC_PAGESIZE);
  if (page < 4096) {
    page = 4096;
  }
  return static_cast<size_t>(page);
}

void marker(
    bool aliased,
    const char* reason,
    const char* kase,
    size_t align,
    uintptr_t ptr_mod) {
  std::cout << "HOST_IMPORT_ALIASED=" << (aliased ? 1 : 0)
            << " reason=" << reason
            << " case=" << kase
            << " align=" << align
            << " ptr_mod=" << ptr_mod << "\n";
}

bool omarchy_gpu_available() {
  return gpu::is_available();
}

} // namespace

TEST_CASE("host pointer import: page-aligned buffer aliases and shows rewrites") {
  const size_t align = host_align();
  if (!omarchy_gpu_available()) {
    std::cout << "Skipping: no omarchy GPU device\n";
    return;
  }
  // Force the import path to the enabled state for this case.
  unsetenv("MLX_OMARCHY_HOST_IMPORT");
  constexpr int kN = 1024;
  void* raw = nullptr;
  REQUIRE(posix_memalign(&raw, align, sizeof(float) * kN) == 0);
  REQUIRE(raw != nullptr);
  auto* host = static_cast<float*>(raw);
  std::mt19937 rng(0xC0FFEE);
  std::uniform_real_distribution<float> dist(-1000.0f, 1000.0f);
  for (int i = 0; i < kN; ++i) {
    host[i] = dist(rng);
  }
  array arr(
      static_cast<void*>(host),
      Shape{kN},
      float32,
      [](void* p) { std::free(p); });
  bool aliased = (arr.data<float>() == host);
  array sum1 = sum(arr);
  eval(sum1);
  float s1 = sum1.item<float>();
  if (aliased) {
    CHECK(arr.data<float>() == host);
    // Rewrite the host slab in place, simulating sushi's
    // pread-into-slab expert-cache step, then re-issue a fresh sum:
    // under aliasing the rewritten bytes are what the GPU reads.
    for (int i = 0; i < kN; ++i) {
      host[i] = static_cast<float>(i + 1);
    }
    array sum2 = sum(arr);
    eval(sum2);
    CHECK(sum2.item<float>() == doctest::Approx(524800.0f)); // 1+..+1024
  } else {
    CHECK(arr.data<float>() != host);
    CHECK(std::isfinite(s1));
  }
  marker(
      aliased,
      aliased ? "aliased" : "copy-fallback",
      "aligned",
      align,
      reinterpret_cast<uintptr_t>(host) % align);
}

TEST_CASE("host pointer import: 4 KiB offset inside a page-aligned block") {
  const size_t align = host_align();
  if (!omarchy_gpu_available()) {
    std::cout << "Skipping: no omarchy GPU device\n";
    return;
  }
  unsetenv("MLX_OMARCHY_HOST_IMPORT");
  constexpr int kN = 256;
  void* raw = nullptr;
  REQUIRE(posix_memalign(&raw, align, sizeof(float) * kN + 4096) == 0);
  REQUIRE(raw != nullptr);
  auto* host = reinterpret_cast<float*>(static_cast<char*>(raw) + 4096);
  for (int i = 0; i < kN; ++i) {
    host[i] = static_cast<float>(i);
  }
  array arr(
      static_cast<void*>(host),
      Shape{kN},
      float32,
      [raw](void* p) {
        (void)p;
        std::free(raw);
      });
  // On a 4 KiB-page host base+4096 is still page-aligned and the
  // import aliases; on a 16 KiB-page host it declines. Both paths must
  // produce the right numbers.
  bool aliased = (arr.data<float>() == host);
  array s = sum(arr);
  eval(s);
  CHECK(s.item<float>() == doctest::Approx(static_cast<float>(kN * (kN - 1) / 2)));
  if (aliased) {
    CHECK(arr.data<float>() == host);
  } else {
    CHECK(arr.data<float>() != host);
  }
  marker(
      aliased,
      aliased ? "aliased" : "copy-fallback",
      "subpage-offset",
      align,
      reinterpret_cast<uintptr_t>(host) % align);
}

TEST_CASE("host pointer import: 100-byte offset declines on every driver") {
  const size_t align = host_align();
  if (!omarchy_gpu_available()) {
    std::cout << "Skipping: no omarchy GPU device\n";
    return;
  }
  unsetenv("MLX_OMARCHY_HOST_IMPORT");
  constexpr int kN = 256;
  void* raw = nullptr;
  REQUIRE(posix_memalign(&raw, align, sizeof(float) * kN + 4096) == 0);
  REQUIRE(raw != nullptr);
  auto* host = reinterpret_cast<float*>(static_cast<char*>(raw) + 100);
  for (int i = 0; i < kN; ++i) {
    host[i] = static_cast<float>(i);
  }
  array arr(
      static_cast<void*>(host),
      Shape{kN},
      float32,
      [raw](void* p) {
        (void)p;
        std::free(raw);
      });
  // No plausible minImportedHostPointerAlignment divides 100, so this
  // is the always-decline case: copy path, no error, correct numbers.
  CHECK(arr.data<float>() != host);
  array s = sum(arr);
  eval(s);
  CHECK(s.item<float>() == doctest::Approx(static_cast<float>(kN * (kN - 1) / 2)));
  marker(
      false,
      "copy-fallback",
      "misaligned-100",
      align,
      reinterpret_cast<uintptr_t>(host) % align);
}

TEST_CASE("host pointer import: env kill switch forces copy path") {
  const size_t align = host_align();
  if (!omarchy_gpu_available()) {
    std::cout << "Skipping: no omarchy GPU device\n";
    return;
  }
  REQUIRE(setenv("MLX_OMARCHY_HOST_IMPORT", "0", 1) == 0);
  constexpr int kN = 256;
  void* raw = nullptr;
  REQUIRE(posix_memalign(&raw, align, sizeof(float) * kN) == 0);
  REQUIRE(raw != nullptr);
  auto* host = static_cast<float*>(raw);
  for (int i = 0; i < kN; ++i) {
    host[i] = static_cast<float>(i);
  }
  array arr(
      static_cast<void*>(host),
      Shape{kN},
      float32,
      [](void* p) { std::free(p); });
  // Env kill -> copy path regardless of device capability.
  CHECK(arr.data<float>() != host);
  array s = sum(arr);
  eval(s);
  CHECK(s.item<float>() == doctest::Approx(static_cast<float>(kN * (kN - 1) / 2)));
  marker(
      false,
      "copy-fallback",
      "env-kill",
      align,
      reinterpret_cast<uintptr_t>(host) % align);
  unsetenv("MLX_OMARCHY_HOST_IMPORT");
}
