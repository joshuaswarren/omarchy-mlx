// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

#include "mlx/backend/omarchy/vulkan.h"

#include <dlfcn.h>

namespace mlx::core::omarchy::vk {

namespace {

constexpr const char* kLoaderNames[] = {
    "libvulkan.so.1",
    "libvulkan.so",
};

void* g_loader{nullptr};

} // namespace

PFN_vkGetInstanceProcAddr GetInstanceProcAddr{nullptr};

bool load_loader() {
  if (GetInstanceProcAddr) {
    return true;
  }
  for (const char* name : kLoaderNames) {
    if (void* handle = dlopen(name, RTLD_NOW | RTLD_LOCAL)) {
      auto proc = reinterpret_cast<PFN_vkGetInstanceProcAddr>(
          dlsym(handle, "vkGetInstanceProcAddr"));
      if (proc) {
        g_loader = handle;
        GetInstanceProcAddr = proc;
        return true;
      }
      dlclose(handle);
    }
  }
  return false;
}

InstanceTable& instance_table() {
  static InstanceTable table;
  return table;
}

DeviceTable& device_table() {
  static DeviceTable table;
  return table;
}

const char* result_string(VkResult result) {
  switch (result) {
    case VK_SUCCESS:
      return "VK_SUCCESS";
    case VK_NOT_READY:
      return "VK_NOT_READY";
    case VK_TIMEOUT:
      return "VK_TIMEOUT";
    case VK_EVENT_SET:
      return "VK_EVENT_SET";
    case VK_EVENT_RESET:
      return "VK_EVENT_RESET";
    case VK_INCOMPLETE:
      return "VK_INCOMPLETE";
    case VK_ERROR_OUT_OF_HOST_MEMORY:
      return "VK_ERROR_OUT_OF_HOST_MEMORY";
    case VK_ERROR_OUT_OF_DEVICE_MEMORY:
      return "VK_ERROR_OUT_OF_DEVICE_MEMORY";
    case VK_ERROR_DEVICE_LOST:
      return "VK_ERROR_DEVICE_LOST";
    case VK_ERROR_OUT_OF_POOL_MEMORY:
      return "VK_ERROR_OUT_OF_POOL_MEMORY";
    case VK_ERROR_INVALID_EXTERNAL_HANDLE:
      return "VK_ERROR_INVALID_EXTERNAL_HANDLE";
    case VK_ERROR_FRAGMENTATION:
      return "VK_ERROR_FRAGMENTATION";
    case VK_ERROR_NOT_PERMITTED_KHR:
      return "VK_ERROR_NOT_PERMITTED";
    case VK_ERROR_INVALID_OPAQUE_CAPTURE_ADDRESS:
      return "VK_ERROR_INVALID_OPAQUE_CAPTURE_ADDRESS";
    case VK_PIPELINE_COMPILE_REQUIRED:
      return "VK_PIPELINE_COMPILE_REQUIRED";
    default:
      return "VK_ERROR_UNKNOWN";
  }
}

} // namespace mlx::core::omarchy::vk
