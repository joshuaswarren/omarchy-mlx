// Test stub for mlx/backend/omarchy/device.h: only what gpu_profiler.h reads from a Device. The real header needs a Vulkan device; this test runs the
// profiler against a fake device table instead.
#pragma once

#include <vulkan/vulkan.h>

#include <string>

namespace mlx::core::omarchy {

struct DeviceCapabilitiesStub {
  std::string device_name{"fake"};
  float timestamp_period{1.0f};
  uint32_t queue_timestamp_valid_bits{64};
};

class Device {
 public:
  const DeviceCapabilitiesStub& capabilities() const {
    return caps_;
  }
  VkDevice handle() const {
    return reinterpret_cast<VkDevice>(0x1);
  }

 private:
  DeviceCapabilitiesStub caps_;
};

} // namespace mlx::core::omarchy
