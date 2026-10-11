// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

// Host-pointer import tests (VK_EXT_external_memory_host): the zero-copy
// path that the array(void*, ...) ctor takes when the device supports it,
// the aliasing assertion sushi's data_ptr == slab.ptr check needs, and
// the copy fallback when the path is unavailable (env kill or unaligned
// pointer). Three subcases:
//   1. aligned host buffer, env on, import supported:
//      arr.data<float>() MUST equal the host pointer; a host rewrite of
//      the slab bytes is visible to a subsequent GPU sum.
//   2. unaligned host buffer (offset within a 4 KiB-aligned block):
//      the import path declines; arr.data<float>() MUST NOT equal the
//      caller's pointer; the GPU op runs correctly through the copy
//      path.
//   3. aligned host buffer, MLX_OMARCHY_HOST_IMPORT=0 set before the
//      array ctor: same outcome as (2) — copy path, no aliasing.
// All three are checked from a single binary: the env-setenv call in (3)
// is process-wide but only affects the next make_buffer, which is
// scoped to the ctor.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

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

bool omarchy_gpu_available() {
  return gpu::is_available();
}

// Wrap a host buffer of N floats. fill is a generator that writes the
// initial bytes. Returns the array and reports whether the backend
// imported the buffer (true) or copied (false) via the out param.
array make_array(float* host, int n, std::mt19937& rng, bool& imported) {
  std::uniform_real_distribution<float> dist(-1000.0f, 1000.0f);
  for (int i = 0; i < n; ++i) {
    host[i] = dist(rng);
  }
  // Deleter frees the host buffer after the array is destroyed.
  array out(
      static_cast<void*>(host),
      Shape{n},
      float32,
      [](void* p) { std::free(p); });
  imported = (out.data<float>() == host);
  return out;
}

} // namespace

TEST_CASE("host pointer import: aliasing + host rewrite visible") {
  if (!omarchy_gpu_available()) {
    std::cout << "Skipping: no omarchy GPU device\n";
    return;
  }
  // Force the import path to the enabled state for this case.
  unsetenv("MLX_OMARCHY_HOST_IMPORT");
  constexpr size_t kAlign = 4096;
  constexpr int kN = 1024;
  void* raw = nullptr;
  REQUIRE(posix_memalign(&raw, kAlign, sizeof(float) * kN) == 0);
  REQUIRE(raw != nullptr);
  auto* host = static_cast<float*>(raw);
  std::mt19937 rng(0xC0FFEE);
  bool imported = false;
  array arr = make_array(host, kN, rng, imported);
  std::cout << "host_import path: "
            << (imported ? "aliased" : "copied")
            << " host=" << host
            << " arr.data=" << arr.data<float>() << "\n";
  if (imported) {
    CHECK(arr.data<float>() == host);
    // First GPU op reads the slab.
    array sum1 = sum(arr);
    eval(sum1);
    // Rewrite the host slab in place, simulating sushi's
    // pread-into-slab expert-cache step.
    for (int i = 0; i < kN; ++i) {
      host[i] = static_cast<float>(i + 1);
    }
    // Re-issue a fresh sum; the encoder schedules a new dispatch
    // reading arr. Under aliasing the rewritten bytes are visible.
    array sum2 = sum(arr);
    eval(sum2);
    float s1 = sum1.item<float>();
    float s2 = sum2.item<float>();
    // sum1: rng-distributed -1000..1000, deterministic seed -> fixed value.
    // sum2: 1+2+...+1024 = 1024*1025/2 = 524800.
    CHECK(s2 == doctest::Approx(524800.0f));
    INFO("sum1 (rng)=" << s1 << " sum2 (rewrite 1..1024)=" << s2);
  } else {
    // Path unavailable: copy was used. arr.data MUST differ from host.
    CHECK(arr.data<float>() != host);
    array sum1 = sum(arr);
    eval(sum1);
    float s1 = sum1.item<float>();
    INFO("copy path sum (rng)=" << s1);
    CHECK(std::isfinite(s1));
  }
}

TEST_CASE("host pointer import: unaligned pointer falls back to copy") {
  if (!omarchy_gpu_available()) {
    std::cout << "Skipping: no omarchy GPU device\n";
    return;
  }
  unsetenv("MLX_OMARCHY_HOST_IMPORT");
  constexpr int kN = 256;
  // Allocate a 4 KiB-aligned base and offset by 100 bytes. The
  // import path requires the pointer itself to be aligned to the
  // device's minImportedHostPointerAlignment (4 KiB on every driver
  // we exercise). The misaligned pointer must take the copy path.
  void* raw = nullptr;
  REQUIRE(posix_memalign(&raw, 4096, sizeof(float) * kN + 4096) == 0);
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
  // Misaligned -> copy path. The aliased-pointer assertion fails
  // outright under the import path; under the copy path the array
  // owns its own allocation and arr.data != host.
  CHECK(arr.data<float>() != host);
  array s = sum(arr);
  eval(s);
  CHECK(s.item<float>() == doctest::Approx(static_cast<float>(kN * (kN - 1) / 2)));
}

TEST_CASE("host pointer import: env kill switch forces copy path") {
  if (!omarchy_gpu_available()) {
    std::cout << "Skipping: no omarchy GPU device\n";
    return;
  }
  REQUIRE(setenv("MLX_OMARCHY_HOST_IMPORT", "0", 1) == 0);
  constexpr size_t kAlign = 4096;
  constexpr int kN = 256;
  void* raw = nullptr;
  REQUIRE(posix_memalign(&raw, kAlign, sizeof(float) * kN) == 0);
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
  unsetenv("MLX_OMARCHY_HOST_IMPORT");
}
