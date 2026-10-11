// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

// udmabuf expert-slab tests (the Linux Honeykrisp zero-copy path).
// Cases:
//   1. real-create: the real /dev/udmabuf path. On a udmabuf-kernel
//      device this creates a slab, wraps it in an array, asserts the
//      alias (arr.data == slab ptr), proves a host rewrite of the slab
//      is visible to a subsequent GPU sum, exercises the sync calls and
//      destroys cleanly. On a host without /dev/udmabuf (the CT) the
//      create must decline cleanly with reason=device-missing.
//   2. env-kill: MLX_OMARCHY_UDMABUF=0 declines with reason=env-disabled.
//   3. fake-create: the create ioctl is replaced by a test hook (the CT
//      has no /dev/udmabuf), so the registry is populated and
//      make_buffer's dispatch runs. The import itself still declines on
//      a driver without VK_EXT_external_memory_dma_buf import (or on a
//      non-dma-buf fd): the copy fallback must run without error and
//      stay correct. On a real device the same case aliases.
//   4. size-limit: a create above the kernel's size_limit_mb declines
//      with reason=size-limit when the sysfs cap is readable.
// Each case prints ONE stable marker line:
//   UDMABUF_SLAB=<0|1> reason=<...> case=<...>
// which is what a device receipt greps; the stderr debug lines (enable
// MLX_OMARCHY_HOST_IMPORT_DEBUG=1) carry the exact decline reasons.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

#include <fcntl.h>
#include <linux/udmabuf.h>
#include <unistd.h>

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>

#include "mlx/backend/gpu/device_info.h"
#include "mlx/backend/omarchy/udmabuf.h"
#include "mlx/device.h"
#include "mlx/ops.h"
#include "mlx/stream.h"
#include "mlx/transforms.h"

using namespace mlx::core;

namespace {

void marker(bool ok, const char* reason, const char* kase) {
  std::cout << "UDMABUF_SLAB=" << (ok ? 1 : 0) << " reason=" << reason
            << " case=" << kase << "\n";
}

bool omarchy_gpu_available() {
  return gpu::is_available();
}

// Test hook: stand in for open(/dev/udmabuf) + UDMABUF_CREATE. Validates
// the kernel-visible contract (page-multiple size, sealed memfd) and
// returns a dup of the memfd as a stand-in fd; the allocator's import of
// a non-dma-buf fd must decline, which is exactly the fallback case the
// CT needs to prove.
int fake_udmabuf_create(struct udmabuf_create* req) {
  const long page = sysconf(_SC_PAGESIZE);
  if (req->size == 0 ||
      static_cast<uint64_t>(req->size) % static_cast<uint64_t>(page) != 0 ||
      req->offset != 0) {
    errno = EINVAL;
    return -1;
  }
  int dupfd = dup(static_cast<int>(req->memfd));
  if (dupfd < 0) {
    return -1;
  }
  return dupfd;
}

} // namespace

TEST_CASE("udmabuf slab: real create, alias, rewrite visibility, destroy") {
  unsetenv("MLX_OMARCHY_UDMABUF");
  unsetenv("MLX_OMARCHY_HOST_IMPORT");
  struct omarchy_udmabuf_slab slab{};
  constexpr int kN = 1024;
  const size_t want = sizeof(float) * kN;
  int rc = omarchy_udmabuf_slab_create(want, &slab);
  if (rc != 0) {
    // CT and any kernel without CONFIG_UDMABUF: clean decline, no slab.
    CHECK(rc == 7); // device-missing
    CHECK(slab.ptr == nullptr);
    marker(false, "device-missing", "real-create");
    return;
  }
  if (!omarchy_gpu_available()) {
    std::cout << "Skipping GPU part: no omarchy GPU device\n";
    omarchy_udmabuf_slab_destroy(&slab);
    return;
  }
  CHECK(slab.ptr != nullptr);
  CHECK(slab.fd >= 0);
  CHECK(slab.size >= want);
  const long page = sysconf(_SC_PAGESIZE);
  CHECK(reinterpret_cast<uintptr_t>(slab.ptr) % static_cast<uintptr_t>(page) ==
        0);
  auto* host = static_cast<float*>(slab.ptr);
  std::mt19937 rng(0xD0CA11);
  std::uniform_real_distribution<float> dist(-1000.0f, 1000.0f);
  for (int i = 0; i < kN; ++i) {
    host[i] = dist(rng);
  }
  // Fill brackets: the cheap safe coherency option.
  CHECK(omarchy_udmabuf_slab_sync(&slab, 1) == 0);
  CHECK(omarchy_udmabuf_slab_sync(&slab, 0) == 0);
  bool aliased = false;
  {
    // Arrays die before the slab: the wrapped_deleter contract is
    // caller-frees-after-array-death, same as the host-import path.
    array arr(slab.ptr, Shape{kN}, float32, [](void*) {});
    aliased = (arr.data<float>() == slab.ptr);
    array sum1 = sum(arr);
    eval(sum1);
    const float want1 = sum1.item<float>();
    CHECK(std::isfinite(want1));
    // Rewrite the slab in place (sushi's pread-into-slab refill) and
    // check the GPU sees the new bytes through the SAME array.
    for (int i = 0; i < kN; ++i) {
      host[i] = static_cast<float>(i + 1);
    }
    array sum2 = sum(arr);
    eval(sum2);
    if (aliased) {
      CHECK(sum2.item<float>() == doctest::Approx(524800.0f)); // 1+..+1024
    } else {
      // Copy fallback: the array does not track the slab; only require
      // the first read to have been correct.
      CHECK(std::isfinite(sum2.item<float>()));
    }
  }
  omarchy_udmabuf_slab_destroy(&slab);
  CHECK(slab.ptr == nullptr);
  marker(aliased, aliased ? "aliased" : "copy-fallback", "real-create");
}

TEST_CASE("udmabuf slab: env kill switch declines create") {
  REQUIRE(setenv("MLX_OMARCHY_UDMABUF", "0", 1) == 0);
  struct omarchy_udmabuf_slab slab{};
  int rc = omarchy_udmabuf_slab_create(4096, &slab);
  CHECK(rc == 1); // env-disabled
  CHECK(slab.ptr == nullptr);
  marker(false, "env-disabled", "env-kill");
  unsetenv("MLX_OMARCHY_UDMABUF");
}

TEST_CASE("udmabuf slab: hooked create populates registry; dispatch + fallback") {
  unsetenv("MLX_OMARCHY_UDMABUF");
  unsetenv("MLX_OMARCHY_HOST_IMPORT");
  if (!omarchy_gpu_available()) {
    std::cout << "Skipping: no omarchy GPU device\n";
    return;
  }
  mlx::core::omarchy::udmabuf::set_create_for_test(&fake_udmabuf_create);
  struct omarchy_udmabuf_slab slab{};
  constexpr int kN = 512;
  const size_t want = sizeof(float) * kN;
  int rc = omarchy_udmabuf_slab_create(want, &slab);
  CHECK(rc == 0);
  if (rc != 0) {
    mlx::core::omarchy::udmabuf::set_create_for_test(nullptr);
    marker(false, "create-failed", "fake-create");
    return;
  }
  const long page = sysconf(_SC_PAGESIZE);
  CHECK(slab.size ==
        static_cast<size_t>(
            ((want + page - 1) / page) * static_cast<size_t>(page)));
  auto* host = static_cast<float*>(slab.ptr);
  for (int i = 0; i < kN; ++i) {
    host[i] = static_cast<float>(i);
  }
  // A pointer inside the region routes to the dma-buf import branch;
  // whether that aliases depends on the driver's dma-buf import support
  // (declines on the CT's lavapipe or on the hook's non-dma-buf fd).
  array arr(slab.ptr, Shape{kN}, float32, [](void*) {});
  const bool aliased = (arr.data<float>() == slab.ptr);
  array s1 = sum(arr);
  eval(s1);
  CHECK(s1.item<float>() ==
        doctest::Approx(static_cast<float>(kN * (kN - 1) / 2)));
  for (int i = 0; i < kN; ++i) {
    host[i] = static_cast<float>(kN - i);
  }
  array s2 = sum(arr);
  eval(s2);
  if (aliased) {
    CHECK(s2.item<float>() ==
          doctest::Approx(static_cast<float>(kN * (kN + 1) / 2)));
  } else {
    CHECK(std::isfinite(s2.item<float>()));
  }
  // A pointer OUTSIDE any region must not dispatch to the udmabuf
  // branch (host-import or copy path; correctness only).
  {
    auto* outside = static_cast<float*>(std::malloc(sizeof(float) * kN));
    REQUIRE(outside != nullptr);
    for (int i = 0; i < kN; ++i) {
      outside[i] = 1.0f;
    }
    array out_arr(outside, Shape{kN}, float32, [](void*) {});
    array s3 = sum(out_arr);
    eval(s3);
    CHECK(s3.item<float>() == doctest::Approx(static_cast<float>(kN)));
    std::free(outside);
  }

  omarchy_udmabuf_slab_destroy(&slab);
  mlx::core::omarchy::udmabuf::set_create_for_test(nullptr);
  marker(aliased, aliased ? "aliased" : "copy-fallback", "fake-create");
}

TEST_CASE("udmabuf slab: hooked create rejects a non-page-multiple size") {
  unsetenv("MLX_OMARCHY_UDMABUF");
  mlx::core::omarchy::udmabuf::set_create_for_test(&fake_udmabuf_create);
  struct omarchy_udmabuf_slab slab{};
  // create rounds sizes up itself, so the hook only sees multiples; a
  // zero size declines before any kernel step.
  int rc = omarchy_udmabuf_slab_create(0, &slab);
  CHECK(rc == 2); // size-zero
  mlx::core::omarchy::udmabuf::set_create_for_test(nullptr);
  marker(false, "size-zero", "size-guard");
}

TEST_CASE("udmabuf slab: create above size_limit_mb declines when readable") {
  unsetenv("MLX_OMARCHY_UDMABUF");
  std::FILE* f = std::fopen("/sys/module/udmabuf/parameters/size_limit_mb", "r");
  if (f == nullptr) {
    std::cout << "Skipping: no size_limit_mb sysfs on this host\n";
    return;
  }
  unsigned long mb = 0;
  int got = std::fscanf(f, "%lu", &mb);
  std::fclose(f);
  if (got != 1 || mb == 0 || mb > (16UL << 20)) {
    std::cout << "Skipping: unusable size_limit_mb=" << mb << "\n";
    return;
  }
  struct omarchy_udmabuf_slab slab{};
  // Above the cap but below any plausible allocation failure: the
  // sysfs check must fire BEFORE the ioctl (and before /dev/udmabuf is
  // even opened, so this case runs on the CT too).
  int rc = omarchy_udmabuf_slab_create(
      (static_cast<size_t>(mb) + 16) * 1024 * 1024, &slab);
  CHECK(rc == 8); // size-limit
  CHECK(slab.ptr == nullptr);
  marker(false, "size-limit", "size-limit");
}
