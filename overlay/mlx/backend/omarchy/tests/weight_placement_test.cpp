// SPDX-License-Identifier: MIT
// Weight placement on a discrete GPU: the card memory a driver exposes is
// not always the card memory a model needs.
//
// A discrete card behind a link without a large bar aperture shows one
// small device-local heap (the B60 over USB4: ~256 MiB) and no other
// device-local heap, so a 2.3 GiB model cannot live there. A bar-mapped
// card shows one large host-visible device-local heap instead. The
// placement must therefore be size aware: a model that does not fit in the
// card's device-local memory stays in host memory and the card still runs
// the compute.
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "../inference_weights.h"

namespace {

int failures = 0;

void expect(bool condition, const char* what) {
  if (condition) {
    std::printf("ok %s\n", what);
  } else {
    std::printf("FAIL %s\n", what);
    failures++;
  }
}

constexpr VkDeviceSize kMiB = 1024ull * 1024ull;
constexpr VkDeviceSize kGiB = 1024ull * 1024ull * 1024ull;

// A memory layout with one heap per entry, and one memory type per heap.
struct Layout {
  std::vector<VkMemoryHeap> heaps;
  std::vector<VkMemoryType> types;
};

void add_heap(
    Layout& layout,
    VkDeviceSize size,
    VkMemoryHeapFlags flags,
    VkMemoryPropertyFlags properties) {
  VkMemoryHeap heap{};
  heap.size = size;
  heap.flags = flags;
  layout.heaps.push_back(heap);
  VkMemoryType type{};
  type.propertyFlags = properties;
  type.heapIndex = static_cast<uint32_t>(layout.heaps.size() - 1);
  layout.types.push_back(type);
}

VkPhysicalDeviceMemoryProperties properties_of(const Layout& layout) {
  VkPhysicalDeviceMemoryProperties props{};
  props.memoryTypeCount = static_cast<uint32_t>(layout.types.size());
  props.memoryHeapCount = static_cast<uint32_t>(layout.heaps.size());
  for (uint32_t i = 0; i < props.memoryTypeCount; ++i) {
    props.memoryTypes[i] = layout.types[i];
  }
  for (uint32_t i = 0; i < props.memoryHeapCount; ++i) {
    props.memoryHeaps[i] = layout.heaps[i];
  }
  return props;
}

}  // namespace

int main() {
  using mlx::core::omarchy::weight_memory_type_for_size;

  // A normal discrete card: one large pure device-local heap.
  {
    Layout layout;
    add_heap(layout, 24 * kGiB, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT,
             VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    add_heap(layout, 32 * kGiB, 0,
             VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                 VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    const auto props = properties_of(layout);
    const uint32_t type = weight_memory_type_for_size(
        VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU, props, UINT32_MAX, 2 * kGiB);
    expect(type != UINT32_MAX, "large-card-weights-fit");
    expect(type < props.memoryTypeCount &&
               (props.memoryTypes[type].propertyFlags &
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0,
           "large-card-weights-are-device-local");
  }

  // The B60 over USB4: one small pure device-local heap, no other
  // device-local heap. A model does not fit, so the weights stay in host
  // memory instead of failing the allocation.
  {
    Layout layout;
    add_heap(layout, 256 * kMiB, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT,
             VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    add_heap(layout, 48 * kGiB, 0,
             VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                 VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    const auto props = properties_of(layout);
    expect(
        weight_memory_type_for_size(
            VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU,
            props,
            UINT32_MAX,
            2 * kGiB) == UINT32_MAX,
        "small-card-heap-refuses-a-model-too-big");
    expect(
        weight_memory_type_for_size(
            VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU,
            props,
            UINT32_MAX,
            64 * kMiB) != UINT32_MAX,
        "small-card-heap-still-takes-a-small-buffer");
  }

  // A bar-mapped card: one large host-visible device-local heap.
  {
    Layout layout;
    add_heap(layout, 24 * kGiB, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT,
             VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT |
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                 VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    add_heap(layout, 32 * kGiB, 0,
             VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                 VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    const auto props = properties_of(layout);
    const uint32_t type = weight_memory_type_for_size(
        VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU, props, UINT32_MAX, 2 * kGiB);
    expect(type != UINT32_MAX, "bar-mapped-card-takes-the-model");
    expect(type < props.memoryTypeCount &&
               (props.memoryTypes[type].propertyFlags &
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0,
           "bar-mapped-card-weights-are-device-local");
  }

  // An integrated GPU never takes the card path.
  {
    Layout layout;
    add_heap(layout, 64 * kGiB, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT,
             VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    const auto props = properties_of(layout);
    expect(
        weight_memory_type_for_size(
            VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU,
            props,
            UINT32_MAX,
            2 * kGiB) == UINT32_MAX,
        "integrated-gpu-keeps-mapped-weights");
  }

  // A card with no device-local heap at all.
  {
    Layout layout;
    add_heap(layout, 32 * kGiB, 0,
             VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                 VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    const auto props = properties_of(layout);
    expect(
        weight_memory_type_for_size(
            VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU,
            props,
            UINT32_MAX,
            2 * kGiB) == UINT32_MAX,
        "no-device-local-heap-refuses-card-weights");
  }

  // The largest pure device-local heap wins, not the first one: the
  // comment promises the largest and the selector must deliver it.
  {
    Layout layout;
    add_heap(layout, 256 * kMiB, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT,
             VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    add_heap(layout, 24 * kGiB, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT,
             VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    const auto props = properties_of(layout);
    expect(mlx::core::omarchy::local_device_memory_type(props, UINT32_MAX) == 1,
           "largest-pure-local-heap-wins");
  }

  if (failures) {
    std::printf("%d failure(s)\n", failures);
    return 1;
  }
  std::printf("weight-placement tests pass\n");
  return 0;
}
