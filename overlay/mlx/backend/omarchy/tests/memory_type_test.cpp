// SPDX-License-Identifier: MIT
// Per-heap capacity-aware memory type selection for the allocator's mapped
// buffers.
//
// The first-match chooser returned the first DEVICE_LOCAL-heap type whose
// flags matched, regardless of heap size. On a driver whose device-local
// heap is a small window (the B60 over USB4 shows ~256 MiB) that heap
// captured every mapped mlx allocation and failed at vkAllocateMemory
// while a larger heap matched the same flags; ba64b28c6a fixed the
// identical flaw for weight placement. This suite pins the allocator's own
// chooser to heap capacity: a DEVICE_LOCAL heap wins only when it is at
// least as large as the best matching non-device-local heap, and the
// largest heap wins within each class.
//
// All layouts here are synthetic VkPhysicalDeviceMemoryProperties structs;
// no Vulkan device is created.
#include <cstdint>
#include <cstdio>
#include <vector>

#include "../memory_type.h"

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
  using mlx::core::omarchy::capacity_aware_memory_type;
  constexpr auto kLocal = VK_MEMORY_HEAP_DEVICE_LOCAL_BIT;
  constexpr auto kHV = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
  constexpr auto kHC = VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

  // The failing shape behind ba64b28c6a, for the mapped-buffer path: a
  // small device-local BAR window next to a larger system heap, both
  // host-visible coherent. The large system heap must win; the old
  // first-match chooser returned the BAR type.
  {
    Layout l;
    add_heap(l, 256 * kMiB, kLocal, kHV | kHC);
    add_heap(l, 8 * kGiB, 0, kHV | kHC);
    expect(
        capacity_aware_memory_type(properties_of(l), UINT32_MAX, kHV | kHC) ==
            1,
        "small device-local heap loses to the larger matching system heap");
  }

  // Apple-style unified memory: the device-local heap is the largest
  // matching heap and still wins, as before.
  {
    Layout l;
    add_heap(l, 16 * kGiB, kLocal, kHV | kHC);
    expect(
        capacity_aware_memory_type(properties_of(l), UINT32_MAX, kHV | kHC) ==
            0,
        "unified device-local heap wins when it is the largest match");
  }

  // Equal capacity keeps the device-local preference.
  {
    Layout l;
    add_heap(l, 4 * kGiB, kLocal, kHV | kHC);
    add_heap(l, 4 * kGiB, 0, kHV | kHC);
    expect(
        capacity_aware_memory_type(properties_of(l), UINT32_MAX, kHV | kHC) ==
            0,
        "tie on heap capacity keeps the device-local type");
  }

  // Flag filtering: the largest heap is skipped when its type lacks the
  // required properties.
  {
    Layout l;
    add_heap(l, 16 * kGiB, kLocal, 0);
    add_heap(l, 8 * kGiB, 0, kHV | kHC);
    expect(
        capacity_aware_memory_type(properties_of(l), UINT32_MAX, kHV | kHC) ==
            1,
        "a largest-but-flag-mismatched heap does not win");
  }

  // The largest device-local candidate wins within its class, not the
  // first one.
  {
    Layout l;
    add_heap(l, 512 * kMiB, kLocal, kHV | kHC);
    add_heap(l, 16 * kGiB, kLocal, kHV | kHC);
    add_heap(l, 8 * kGiB, 0, kHV | kHC);
    expect(
        capacity_aware_memory_type(properties_of(l), UINT32_MAX, kHV | kHC) ==
            1,
        "largest device-local heap beats a smaller device-local heap");
  }

  // The caller's type_bits mask excludes a type even when its heap is the
  // largest and device-local.
  {
    Layout l;
    add_heap(l, 16 * kGiB, kLocal, kHV | kHC);
    add_heap(l, 8 * kGiB, 0, kHV | kHC);
    expect(
        capacity_aware_memory_type(properties_of(l), 1u << 1, kHV | kHC) == 1,
        "type_bits mask excludes the larger device-local heap");
  }

  // The non-coherent fallback path (HOST_VISIBLE without HOST_COHERENT)
  // follows the same capacity rule.
  {
    Layout l;
    add_heap(l, 256 * kMiB, kLocal, kHV);
    add_heap(l, 8 * kGiB, 0, kHV);
    expect(capacity_aware_memory_type(properties_of(l), UINT32_MAX, kHV) == 1,
           "non-coherent fallback also prefers the larger heap");
  }

  // Only a device-local type matches: it is returned (existing behaviour).
  {
    Layout l;
    add_heap(l, 16 * kGiB, kLocal, kHV | kHC);
    expect(
        capacity_aware_memory_type(properties_of(l), UINT32_MAX, kHV | kHC) ==
            0,
        "sole matching device-local type is returned");
  }

  // Nothing matches: UINT32_MAX so the caller fails closed.
  {
    Layout l;
    add_heap(l, 16 * kGiB, kLocal, 0);
    expect(
        capacity_aware_memory_type(properties_of(l), UINT32_MAX, kHV | kHC) ==
            UINT32_MAX,
        "no matching type yields UINT32_MAX");
  }

  if (failures) {
    std::printf("memory-type tests FAILED (%d)\n", failures);
    return 1;
  }
  std::printf("memory-type tests pass\n");
  return 0;
}
