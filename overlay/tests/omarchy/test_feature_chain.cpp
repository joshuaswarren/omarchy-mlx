// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT
//
// VkDeviceCreateInfo feature-chain contract test (PR-LILYVFT).
//
// aurora-lily measured per-feature costs on Honeykrisp-M5 and stripped
// everything its encoder does not need: sparse residency, robust
// buffer/image access, and buffer-device-address capture/replay each
// carried a real driver-side cost. The omarchy device chain was audited
// the same way: it requests only what the backend and its shaders
// consume. This test pins that contract by intercepting the
// vkCreateDevice call and walking the pNext chain the backend actually
// hands the driver.
//
// The hook is installed after runtime init (which populates the
// instance table) and before the lazy device creation inside
// omarchy::device(0), so the test captures the real VkDeviceCreateInfo
// the production path builds.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

#include <cstring>
#include <string>
#include <vector>

#include "mlx/backend/omarchy/device.h"
#include "mlx/backend/omarchy/vulkan.h"

namespace {

PFN_vkCreateDevice real_create_device = nullptr;
bool captured = false;
bool captured_enabled_features = false;
std::vector<VkStructureType> unexpected_stypes;
std::vector<std::string> enabled_exts;
bool seen_f2 = false;
VkPhysicalDeviceFeatures2 cap_f2{
    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
bool seen_f12 = false;
VkPhysicalDeviceVulkan12Features cap_f12{
    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
bool seen_f13 = false;
VkPhysicalDeviceVulkan13Features cap_f13{
    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
bool seen_f16 = false;
VkPhysicalDevice16BitStorageFeatures cap_f16{
    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES};

// Feature-struct layout is {sType; pNext; VkBool32 fields...}. Counting
// nonzero VkBool32 slots after the two leading members catches every
// set feature, including ones a named check would have to list one by
// one.
uint32_t true_fields(const void* s, size_t size) {
  constexpr size_t kHeader = sizeof(VkStructureType) + sizeof(void*);
  const auto* fields =
      reinterpret_cast<const uint32_t*>(reinterpret_cast<const char*>(s) +
                                        kHeader);
  uint32_t n = 0;
  for (size_t off = 0; off + sizeof(uint32_t) <= size - kHeader;
       off += sizeof(uint32_t)) {
    if (fields[off / sizeof(uint32_t)] != 0) {
      ++n;
    }
  }
  return n;
}

VkResult VKAPI_CALL capture_create_device(
    VkPhysicalDevice pd,
    const VkDeviceCreateInfo* ci,
    const VkAllocationCallbacks* cb,
    VkDevice* out) {
  captured = true;
  captured_enabled_features = ci->pEnabledFeatures != nullptr;
  for (const auto* s =
           reinterpret_cast<const VkBaseInStructure*>(ci->pNext);
       s != nullptr;
       s = s->pNext) {
    switch (s->sType) {
      case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2:
        cap_f2 = *reinterpret_cast<const VkPhysicalDeviceFeatures2*>(s);
        cap_f2.pNext = nullptr;
        seen_f2 = true;
        break;
      case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES:
        cap_f12 =
            *reinterpret_cast<const VkPhysicalDeviceVulkan12Features*>(s);
        cap_f12.pNext = nullptr;
        seen_f12 = true;
        break;
      case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES:
        cap_f13 =
            *reinterpret_cast<const VkPhysicalDeviceVulkan13Features*>(s);
        cap_f13.pNext = nullptr;
        seen_f13 = true;
        break;
      case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES:
        cap_f16 =
            *reinterpret_cast<const VkPhysicalDevice16BitStorageFeatures*>(s);
        cap_f16.pNext = nullptr;
        seen_f16 = true;
        break;
      case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT:
      case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR:
#ifdef VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_BFLOAT16_FEATURES_KHR
      case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_BFLOAT16_FEATURES_KHR:
#endif
        break;
      default:
        unexpected_stypes.push_back(s->sType);
        break;
    }
  }
  for (uint32_t i = 0; i < ci->enabledExtensionCount; ++i) {
    enabled_exts.emplace_back(ci->ppEnabledExtensionNames[i]);
  }
  return real_create_device(pd, ci, cb, out);
}

bool allowed_extension(const std::string& name) {
  static const char* allowed[] = {
      "VK_EXT_shader_atomic_float",
      "VK_KHR_cooperative_matrix",
      "VK_KHR_shader_bfloat16",
      "VK_EXT_global_priority",
  };
  for (const char* a : allowed) {
    if (name == a) {
      return true;
    }
  }
  return false;
}

} // namespace

TEST_CASE("VkDeviceCreateInfo requests only the audited feature set") {
  using namespace mlx::core::omarchy;
  if (device_count() == 0) {
    return; // no Vulkan device on this host; nothing to capture
  }
  auto& it = vk::instance_table();
  REQUIRE(it.CreateDevice != nullptr);
  real_create_device = it.CreateDevice;
  it.CreateDevice = capture_create_device;
  (void)device(0);
  it.CreateDevice = real_create_device;
  REQUIRE(captured);

  CHECK_MESSAGE(
      !captured_enabled_features,
      "pEnabledFeatures must stay null; every feature rides the pNext chain");
  CHECK_MESSAGE(
      unexpected_stypes.empty(),
      "unexpected feature struct(s) in the pNext chain; extend the audit "
      "before enabling anything new");
  CHECK_MESSAGE(seen_f2, "VkPhysicalDeviceFeatures2 heads the chain");
  CHECK_MESSAGE(seen_f12, "Vulkan 1.2 feature struct is chained");
  CHECK_MESSAGE(seen_f13, "Vulkan 1.3 feature struct is chained");
  CHECK_MESSAGE(seen_f16, "16-bit-storage feature struct is chained");

  // The lily-measured costly features must never be requested.
  CHECK(!cap_f2.features.robustBufferAccess);
  CHECK(!cap_f2.features.sparseBinding);
  CHECK(!cap_f2.features.sparseResidencyBuffer);
  CHECK(!cap_f2.features.sparseResidencyImage2D);
  CHECK(!cap_f2.features.sparseResidencyImage3D);
  CHECK(!cap_f2.features.sparseResidency2Samples);
  CHECK(!cap_f2.features.sparseResidency4Samples);
  CHECK(!cap_f2.features.sparseResidency8Samples);
  CHECK(!cap_f2.features.sparseResidency16Samples);
  CHECK(!cap_f2.features.sparseResidencyAliased);
  CHECK(!cap_f12.bufferDeviceAddress);
  CHECK(!cap_f12.bufferDeviceAddressCaptureReplay);
  CHECK(!cap_f12.bufferDeviceAddressMultiDevice);
  CHECK(!cap_f12.scalarBlockLayout);
  CHECK(!cap_f12.descriptorIndexing);
  CHECK(!cap_f12.runtimeDescriptorArray);
  CHECK(!cap_f13.robustImageAccess);
  CHECK(!cap_f13.dynamicRendering);
  CHECK(!cap_f13.maintenance4);

  // Whitelists: every set feature bit is one the backend or its shaders
  // consume. The count catches additions a named check would miss.
  CHECK(
      true_fields(&cap_f2, sizeof(cap_f2)) ==
      (cap_f2.features.shaderInt16 ? 1 : 0) +
          (cap_f2.features.shaderInt64 ? 1 : 0));
  CHECK(
      true_fields(&cap_f12, sizeof(cap_f12)) ==
      (cap_f12.timelineSemaphore ? 1 : 0) + (cap_f12.shaderFloat16 ? 1 : 0));
  CHECK(
      true_fields(&cap_f13, sizeof(cap_f13)) ==
      (cap_f13.subgroupSizeControl ? 1 : 0));
  CHECK(
      true_fields(&cap_f16, sizeof(cap_f16)) ==
      (cap_f16.storageBuffer16BitAccess ? 1 : 0));

  for (const auto& ext : enabled_exts) {
    CHECK_MESSAGE(
        allowed_extension(ext),
        "unexpected device extension requested: " << ext);
  }
}
