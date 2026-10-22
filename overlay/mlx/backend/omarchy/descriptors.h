// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

#pragma once

#include <vulkan/vulkan.h>

#include <span>

namespace mlx::core::omarchy {

// ComputeRuntime uses consecutive, identical storage-buffer bindings. Vulkan
// permits one write to span them; each buffer retains its own offset and range.
// The dispatch caller validates that buffers contains 1..binding_limit entries.
inline void update_storage_descriptors(
    PFN_vkUpdateDescriptorSets update,
    VkDevice device,
    VkDescriptorSet set,
    std::span<const VkDescriptorBufferInfo> buffers) {
  VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
  write.dstSet = set;
  write.descriptorCount = static_cast<uint32_t>(buffers.size());
  write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  write.pBufferInfo = buffers.data();
  update(device, 1, &write, 0, nullptr);
}

} // namespace mlx::core::omarchy
