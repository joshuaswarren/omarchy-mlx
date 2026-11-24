#include <stdexcept>

#include "inference_weights.h"

using namespace mlx::core::omarchy;

int main() {
  VkPhysicalDeviceMemoryProperties memory{};
  memory.memoryHeapCount = 2;
  memory.memoryHeaps[0].flags = VK_MEMORY_HEAP_DEVICE_LOCAL_BIT;
  memory.memoryTypeCount = 3;
  memory.memoryTypes[0] = {VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0};
  memory.memoryTypes[1] = {VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, 1};
  memory.memoryTypes[2] = {VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
      VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 1};
  if (weight_memory_type(VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU, memory, 7) != 0 ||
      weight_memory_type(VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU, memory, 6) != UINT32_MAX ||
      weight_memory_type(VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU, memory, 7) != UINT32_MAX ||
      staging_memory_type(memory, 7) != 2 || staging_memory_type(memory, 3) != 1 ||
      staging_memory_type(memory, 1) != UINT32_MAX) {
    return 1;
  }
  memory.memoryTypes[1].heapIndex = 0;
  memory.memoryTypes[2].heapIndex = 0;
  if (weight_memory_type(VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU, memory, 7) != UINT32_MAX) {
    return 2;
  }
  const auto tail = weight_upload_chunk(kWeightStagingBytes + 3, kWeightStagingBytes);
  const auto first = weight_upload_chunk(kWeightStagingBytes + 3, 0);
  if (first.read_bytes != kWeightStagingBytes || first.copy_bytes != kWeightStagingBytes ||
      tail.read_bytes != 3 || tail.copy_bytes != 4 || weight_upload_chunk(0, 0).copy_bytes != 0) {
    return 3;
  }
  bool threw = false;
  try {
    require_card_memory_upload(true, true);
  } catch (const std::exception&) {
    threw = true;
  }
  if (!threw) {
    return 4;
  }
  require_card_memory_upload(true, false);
  require_card_memory_upload(false, true);
  require_card_memory_upload(false, false);
  bool in_flight = false;
  {
    InFlightFlag guard(in_flight);
    if (!in_flight) {
      return 5;
    }
  }
  if (in_flight) {
    return 6;
  }
  try {
    {
      InFlightFlag guard(in_flight);
      throw std::runtime_error("upload failed mid-copy");
    }
  } catch (const std::exception&) {
  }
  if (in_flight) {
    return 7;
  }
  // A Thunderbolt dGPU splits memory into a pure DEVICE_LOCAL heap and a
  // host-only GTT heap: no type carries DEVICE_LOCAL and HOST_VISIBLE, so
  // checkpoint weights must take the upload path without the env opt-in.
  VkPhysicalDeviceMemoryProperties split{};
  split.memoryHeapCount = 2;
  split.memoryHeaps[0].flags = VK_MEMORY_HEAP_DEVICE_LOCAL_BIT;
  split.memoryHeaps[1].flags = 0;
  split.memoryTypeCount = 3;
  split.memoryTypes[0] = {VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0};
  split.memoryTypes[1] = {VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
      VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 1};
  split.memoryTypes[2] = {VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
      VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
      VK_MEMORY_PROPERTY_HOST_CACHED_BIT, 1};
  if (mapped_device_local_memory_type(split)) {
    return 8;
  }
  // A resizable-bar card adds a mapped device-local type: weights keep the
  // mapped path and the env stays an opt-in.
  split.memoryTypes[0].propertyFlags |= VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
  if (!mapped_device_local_memory_type(split)) {
    return 9;
  }
  // Unified memory reports every type device local and host visible.
  split.memoryTypes[1].heapIndex = 0;
  split.memoryTypes[1].propertyFlags |= VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
  if (!mapped_device_local_memory_type(split)) {
    return 10;
  }
}
