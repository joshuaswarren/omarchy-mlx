// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"
#include <cstdlib>
#include <stdexcept>
#include <vector>
#include "mlx/device.h"
#include "mlx/backend/gpu/device_info.h"
#include "mlx/ops.h"
#include "mlx/stream.h"

using namespace mlx::core;

TEST_CASE("oversize report skips a dispatch; default still throws") {
  if (!gpu::is_available()) return;
  setenv("MLX_OMARCHY_MAX_STORAGE_RANGE", "1024", 1);
  Stream gs = new_stream(Device::gpu);
  array a = ones({32, 32}, float32, gs);
  array b = ones({32, 32}, float32, gs);

  setenv("MLX_OMARCHY_OVERSIZE_REPORT", "1", 1);
  setenv("MLX_OMARCHY_ALLOC_REPORT", "1", 1);
  array reported = matmul(a, b, gs);
  CHECK_NOTHROW(reported.eval());
  CHECK(reported.shape() == Shape({32, 32}));
  array repeated = matmul(a, b, gs);
  CHECK_NOTHROW(repeated.eval());
  array reported_sum = sum(a, std::vector<int>{0}, false, gs);
  CHECK_NOTHROW(reported_sum.eval());
  array reported_max = max(a, std::vector<int>{0}, false, gs);
  CHECK_NOTHROW(reported_max.eval());

  unsetenv("MLX_OMARCHY_OVERSIZE_REPORT");
  array strict = matmul(a, b, gs);
  CHECK_THROWS_WITH(strict.eval(), "[omarchy] Matmul compute dispatch binds 4096 bytes at storage-buffer binding 0; this device's maxStorageBufferRange is 1024 bytes. One binding cannot address that much memory.");

  // Keep reporting enabled through process exit so its summary is emitted.
  setenv("MLX_OMARCHY_OVERSIZE_REPORT", "1", 1);
}
