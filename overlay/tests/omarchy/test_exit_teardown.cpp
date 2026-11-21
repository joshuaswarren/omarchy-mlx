// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

// PR 519: one small op on the Vulkan device, then the process must exit 0.
// The crash was in the static Runtime destructor after main returned, when
// Device::~Device called into an already-unloaded NVIDIA ICD. A failure is
// a signal at exit, so the standalone process is the assertion.

#include <cstdio>

#include "mlx/backend/gpu/device_info.h"
#include "mlx/device.h"
#include "mlx/ops.h"
#include "mlx/transforms.h"
#include "mlx/stream.h"

using namespace mlx::core;

int main() {
  if (!gpu::is_available()) {
    std::printf("omarchy exit-teardown: no qualifying Vulkan device, skip\n");
    return 0;
  }
  set_default_device(Device::gpu);
  array values({1.0f, 2.0f, 3.0f});
  array scaled = exp(values);
  scaled.eval();
  const float* host = scaled.data<float>();
  if (host[0] < 2.7f || host[1] < 7.3f || host[2] < 20.0f) {
    std::printf(
        "omarchy exit-teardown: wrong op result %f %f %f\n",
        host[0],
        host[1],
        host[2]);
    return 1;
  }
  std::printf("omarchy exit-teardown: op ok, expecting exit 0\n");
  return 0;
}
