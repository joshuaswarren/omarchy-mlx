// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

#include "mlx/backend/omarchy/allocator.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <mutex>
#include <stdexcept>
#include <string>

#include "mlx/backend/omarchy/device.h"
#include "mlx/backend/omarchy/vulkan.h"
#include "mlx/memory.h"

namespace mlx::core {

namespace omarchy {

constexpr size_t kPageSize = 4096;
constexpr size_t kPowerOfTwoBinThreshold = 1u << 20;

size_t round_size(size_t size) {
  if (size <= kPageSize) {
    return kPageSize;
  }
  if (size > kPowerOfTwoBinThreshold) {
    size_t bin = kPowerOfTwoBinThreshold;
    while (bin < size && bin <= SIZE_MAX / 2) {
      bin *= 2;
    }
    if (bin >= size) {
      return bin;
    }
  }
  return kPageSize * ((size + kPageSize - 1) / kPageSize);
}

// MLX_OMARCHY_POISON_FREED (diagnostic, docs/install-omarchy.md): fill
// recycled storage with the poison word so any stale read of a recycled
// buffer announces itself. The word is float32 123456789.0, chosen so a
// stale read that reaches the Cos trigonometric gate aborts the run with
// exactly this magnitude in the message instead of returning silent
// wrong values, and so no legitimate f16 model tensor can contain it
// (f16 max finite is 65504). The hardware regression check runs the
// France prompt with this armed: correct "Paris" output proves no
// recycled-storage read served the run.
constexpr uint32_t kPoisonFreedWord = 0x4CD6D231u;

bool poison_freed() {
  static const bool enabled = env_flag("MLX_OMARCHY_POISON_FREED");
  return enabled;
}

// MLX_OMARCHY_REUSE_LAG (diagnostic, docs/install-omarchy.md): extra
// completion generations a freed buffer waits before it may be reused.
// 0 (default) keeps the one-generation rule - a buffer stamped with
// completion V may be recycled once generation V+1 is observed, which
// assumes the driver's submit-final cleanup for V has run by then. If
// cleanup for V can lag V+1's observation on some driver/timing, the
// reuse cache hands out a block whose previous occupant is still being
// written: the nondeterministic row-corruption NaN class. K widens the
// quarantine by K generations; NaN gone at K>=1 pins the class and is
// also the shape of the fix (lag reuse, not disable it).
int reuse_lag() {
  static const int lag = [] {
    const char* v = std::getenv("MLX_OMARCHY_REUSE_LAG");
    int n = v ? atoi(v) : 0;
    return n < 0 ? 0 : (n > 16 ? 16 : n);
  }();
  return lag;
}

void poison_freed_buffer(void* data, size_t size) {
  auto* words = static_cast<uint32_t*>(data);
  size_t count = size / sizeof(uint32_t);
  for (size_t i = 0; i < count; ++i) {
    words[i] = kPoisonFreedWord;
  }
}

// Test-only hook: simulate one or more vkAllocateMemory OOM results.
bool consume_test_oom() {
  const char* env = std::getenv("MLX_OMARCHY_TEST_OOM_REMAINING");
  if (env == nullptr || *env == '\0') {
    return false;
  }
  char* end = nullptr;
  long remaining = std::strtol(env, &end, 10);
  if (end == env || remaining <= 0) {
    return false;
  }
  if (remaining == 1) {
    unsetenv("MLX_OMARCHY_TEST_OOM_REMAINING");
  } else {
    std::string next = std::to_string(remaining - 1);
    setenv("MLX_OMARCHY_TEST_OOM_REMAINING", next.c_str(), 1);
  }
  return true;
}

uint32_t VulkanAllocator::find_memory_type(
    uint32_t type_bits,
    VkMemoryPropertyFlags required) const {
  const auto& mem = device().memory_properties();
  uint32_t fallback = UINT32_MAX;
  for (uint32_t i = 0; i < mem.memoryTypeCount; ++i) {
    const auto& type = mem.memoryTypes[i];
    if (!(type_bits & (1u << i))) {
      continue;
    }
    if ((type.propertyFlags & required) != required) {
      continue;
    }
    bool device_local =
        mem.memoryHeaps[type.heapIndex].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT;
    if (device_local) {
      return i;
    }
    if (fallback == UINT32_MAX) {
      fallback = i;
    }
  }
  return fallback;
}

VulkanAllocator::VulkanAllocator()
    : buffer_cache_(
          kPageSize,
          [](VulkanBuffer* buf) { return buf->size; },
          [this](VulkanBuffer* buf) { destroy_buffer(buf); }) {
  size_t total = is_available() ? capability_report(0).total_memory : 0;
  memory_limit_ = total > 0 ? (total / 100) * 90 : (1ull << 32);
  // Metal's `gc_limit_` is `min(0.95 * max_rec, 0.95 * memsize)` —
  // Honeykrisp has no separate max-recommended, so the ceiling is
  // 0.95 of the host-visible heap. cache_limit_ matches the same
  // fraction; a single number drives both the malloc gate and the
  // free-side pool cap, so changing one or the other never gets them
  // out of sync.
  size_t ceiling = total > 0 ? (total / 100) * 95 : (1ull << 32);
  gc_limit_ = ceiling;
  cache_limit_ = ceiling;
}

Buffer VulkanAllocator::malloc(size_t size) {
  // The table is empty until the first device exists. A core flow can reach
  // malloc before anything else touches the device, so initialize here; the
  // table field would otherwise be read before the lazy init fills it.
  device();
  auto& dt = vk::device_table();
  if (size == 0) {
    return Buffer{new VulkanBuffer{}};
  }
  size = round_size(size);

  std::unique_lock lk(mutex_);
  // MLX_OMARCHY_TAPE_NO_REUSE (diagnostic, docs/install-omarchy.md):
  // skip the cache entirely so every allocation lands in fresh device
  // memory and nothing recycled can alias a tape dispatch.
  // MLX_OMARCHY_NO_BUFFER_CACHE (diagnostic): the same, but process-
  // wide - it also covers frees and re-hands outside the tape window,
  // which TAPE_NO_REUSE deliberately does not.
  if (!tape_no_reuse() && !buffer_cache_disabled()) {
    if (void* cached = buffer_cache_.reuse_from_cache(size)) {
      auto* buf = static_cast<VulkanBuffer*>(cached);
      buf->recycled = true;
      active_memory_ += buf->size;
      peak_memory_ = std::max(active_memory_, peak_memory_);
      lk.unlock();
      return Buffer{buf};
    }
  }
  // Honor caller-set limits the way the Metal allocator does: an
  // allocation that would push active memory past the effective limit
  // (max(memory limit, wired limit)) first releases the reuse cache.
  // The vkAllocateMemory below stays the final arbiter — the limit
  // triggers cache release; it never fabricates a failure the driver
  // did not report.
  size_t effective_limit = std::max(memory_limit_, wired_limit_);
  if (active_memory_ + size > effective_limit) {
    buffer_cache_.clear();
  }
  // Working-set ceiling (Metal: `gc_limit_`): keep active + cache + this
  // request under the ceiling by releasing the reuse cache before we ask
  // the driver. This is the A20 fix — without it, KV-cache growth across
  // decode steps adds a fresh size class per step and the cache keeps
  // every prior class around, climbing into the Honeykrisp heap until
  // vkAllocateMemory returns OOM. The ceiling is set to 95% of
  // total_memory in the constructor and is shared with `cache_limit_`,
  // so the free-side cap and the malloc-side gate can never disagree.
  if (active_memory_ + buffer_cache_.cache_size() + size > gc_limit_) {
    buffer_cache_.release_cached_buffers(
        active_memory_ + buffer_cache_.cache_size() + size - gc_limit_);
  }
  lk.unlock();

  auto* buf = new VulkanBuffer{};
  buf->size = size;

  VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  bci.size = size;
  bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
      VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  VKX_CHECK(dt.CreateBuffer(device().handle(), &bci, nullptr, &buf->buffer));

  VkMemoryRequirements reqs{};
  dt.GetBufferMemoryRequirements(device().handle(), buf->buffer, &reqs);

  // Prefer host-visible coherent memory (unified memory on Apple GPUs). Fall
  // back to plain host-visible memory with explicit cache maintenance; fail
  // closed when even that is unavailable.
  uint32_t type_index = find_memory_type(
      reqs.memoryTypeBits,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
          VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  if (type_index != UINT32_MAX) {
    buf->coherent = true;
  } else {
    type_index = find_memory_type(
        reqs.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
    buf->coherent = false;
  }
  if (type_index == UINT32_MAX) {
    dt.DestroyBuffer(device().handle(), buf->buffer, nullptr);
    delete buf;
    throw std::runtime_error(
        "[omarchy] no host-visible memory type available;"
        " the backend cannot stage CPU data without one.");
  }

  VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  mai.allocationSize = reqs.size;
  mai.memoryTypeIndex = type_index;
  VkResult alloc_result = VK_SUCCESS;
  if (consume_test_oom()) {
    alloc_result = VK_ERROR_OUT_OF_DEVICE_MEMORY;
  } else {
    alloc_result =
        dt.AllocateMemory(device().handle(), &mai, nullptr, &buf->memory);
  }
  if (alloc_result == VK_ERROR_OUT_OF_DEVICE_MEMORY ||
      alloc_result == VK_ERROR_OUT_OF_HOST_MEMORY) {
    {
      std::unique_lock rl(mutex_);
      buffer_cache_.clear();
    }
    alloc_result =
        dt.AllocateMemory(device().handle(), &mai, nullptr, &buf->memory);
  }
  if (alloc_result != VK_SUCCESS) {
    dt.DestroyBuffer(device().handle(), buf->buffer, nullptr);
    delete buf;
    throw std::runtime_error(
        "[omarchy] vkAllocateMemory failed: " +
        std::string(vk::result_string(alloc_result)) +
        " (request=" + std::to_string(size) + " bytes; cache released)");
  }
  VKX_CHECK(
      dt.BindBufferMemory(device().handle(), buf->buffer, buf->memory, 0));
  VKX_CHECK(dt.MapMemory(
      device().handle(), buf->memory, 0, VK_WHOLE_SIZE, 0, &buf->data));

  lk.lock();
  active_memory_ += buf->size;
  peak_memory_ = std::max(active_memory_, peak_memory_);
  if (!buf->coherent) {
    noncoherent_.push_back(buf);
  }
  lk.unlock();
  return Buffer{buf};
}

void VulkanAllocator::destroy_buffer(VulkanBuffer* buf) {
  // Cache clear/release funnels through here while callers already hold
  // mutex_, so the non-coherent registry drops the buffer before its
  // memory dies; later flush/invalidate must never see it.
  std::erase(noncoherent_, buf);
  auto& dt = vk::device_table();
  if (buf->memory != VK_NULL_HANDLE) {
    dt.UnmapMemory(device().handle(), buf->memory);
    dt.FreeMemory(device().handle(), buf->memory, nullptr);
  }
  if (buf->buffer != VK_NULL_HANDLE) {
    dt.DestroyBuffer(device().handle(), buf->buffer, nullptr);
  }
  delete buf;
}

void VulkanAllocator::free(Buffer buffer) {
  auto* buf = static_cast<VulkanBuffer*>(buffer.ptr());
  if (!buf) {
    return;
  }
  size_t sz = buf->size;
  std::unique_lock lk(mutex_);
  active_memory_ -= sz;
  // A buffer stamped with a completion value has recorded GPU work that
  // may still touch it (kPendingCompletion = an open batch). Freeing the
  // array drops the accounting here — upstream Metal semantics — but the
  // raw buffer skips the reuse cache and waits in the quarantine until
  // release_quarantine proves the driver finished its submission. A
  // buffer whose generation drained at least one generation ago needs
  // no wait; without this an idle process parks every freed buffer
  // until the next submission.
  if (buf->completion != 0) {
    if (buf->completion == kPendingCompletion || !runtime_alive() ||
        buf->completion + 1 + reuse_lag() >
            device().completions().drained_value()) {
      if (buf->completion == kPendingCompletion) {
        pending_quarantine_bytes_ += sz;
      }
      quarantine_.push_back(buf);
      return;
    }
    buf->completion = 0;
  }
  if (sz > 0 && !tape_no_reuse() && !buffer_cache_disabled() &&
      buffer_cache_.cache_size() + sz <= cache_limit_) {
    // Buffers stay mapped for their whole lifetime (malloc maps at
    // creation and only destroy_buffer unmaps), so the poison is a
    // plain host memset. Non-coherent buffers are flushed at the next
    // submit like any other host write.
    if (poison_freed()) {
      poison_freed_buffer(buf->data, sz);
    }
    buffer_cache_.recycle_to_cache(buf);
    return;
  }
  // The cache is full, so release this block instead of retaining it.
  destroy_buffer(buf);
}

void VulkanAllocator::release_quarantine(uint64_t cleanup_done_through) {
  // Called from the completion drain holding drain_mutex_, so calls are
  // serialized. Drain order proves that submit-final cleanup for values
  // <= |cleanup_done_through| has run (Mesa signals a submission's
  // semaphores before its cleanup retires the timeline points, so a
  // buffer is released only one generation after its own completion).
  std::unique_lock lk(mutex_);
  if (quarantine_.empty()) {
    return;
  }
  std::vector<VulkanBuffer*> still_quarantined;
  for (auto* buf : quarantine_) {
    if (buf->completion == kPendingCompletion ||
        buf->completion + reuse_lag() > cleanup_done_through) {
      still_quarantined.push_back(buf);
      continue;
    }
    buf->completion = 0;
    size_t sz = buf->size;
    if (sz > 0 && !tape_no_reuse() && !buffer_cache_disabled() &&
        buffer_cache_.cache_size() + sz <= cache_limit_) {
      if (poison_freed()) {
        poison_freed_buffer(buf->data, sz);
      }
      buffer_cache_.recycle_to_cache(buf);
    } else {
      // The cache is full, so release this block instead of retaining it.
      destroy_buffer(buf);
    }
  }
  quarantine_ = std::move(still_quarantined);
}

void VulkanAllocator::stamp_batch(
    const std::vector<VulkanBuffer*>& bufs,
    uint64_t completion) {
  std::unique_lock lk(mutex_);
  for (auto* buf : bufs) {
    if (buf) {
      buf->completion = completion;
    }
  }
  // Recount instead of subtracting per buffer: a buffer can sit in
  // several open batches (one per stream), and only a scan knows which
  // quarantined buffers still carry the pending stamp.
  size_t pending = 0;
  for (auto* buf : quarantine_) {
    if (buf->completion == kPendingCompletion) {
      pending += buf->size;
    }
  }
  pending_quarantine_bytes_ = pending;
}

size_t VulkanAllocator::size(Buffer buffer) const {
  auto* buf = static_cast<VulkanBuffer*>(buffer.ptr());
  return buf ? buf->size : 0;
}

size_t VulkanAllocator::set_cache_limit(size_t limit) {
  std::unique_lock lk(mutex_);
  std::swap(cache_limit_, limit);
  if (buffer_cache_.cache_size() > cache_limit_) {
    buffer_cache_.release_cached_buffers(
        buffer_cache_.cache_size() - cache_limit_);
  }
  return limit;
}

void VulkanAllocator::clear_cache() {
  std::unique_lock lk(mutex_);
  buffer_cache_.clear();
}

void VulkanAllocator::flush_noncoherent(VkDevice device) {
  std::vector<VkMappedMemoryRange> ranges;
  {
    std::lock_guard lk(mutex_);
    ranges.reserve(noncoherent_.size());
    for (auto* buf : noncoherent_) {
      if (!buf || !buf->memory) {
        continue;
      }
      VkMappedMemoryRange range{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
      range.memory = buf->memory;
      range.offset = 0;
      range.size = VK_WHOLE_SIZE;
      ranges.push_back(range);
    }
  }
  if (!ranges.empty()) {
    VKX_CHECK(
        vk::device_table().FlushMappedMemoryRanges(
            device, static_cast<uint32_t>(ranges.size()), ranges.data()));
  }
}

void VulkanAllocator::invalidate_noncoherent(VkDevice device) {
  std::vector<VkMappedMemoryRange> ranges;
  {
    std::lock_guard lk(mutex_);
    ranges.reserve(noncoherent_.size());
    for (auto* buf : noncoherent_) {
      if (!buf || !buf->memory) {
        continue;
      }
      VkMappedMemoryRange range{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
      range.memory = buf->memory;
      range.offset = 0;
      range.size = VK_WHOLE_SIZE;
      ranges.push_back(range);
    }
  }
  if (!ranges.empty()) {
    VKX_CHECK(
        vk::device_table().InvalidateMappedMemoryRanges(
            device, static_cast<uint32_t>(ranges.size()), ranges.data()));
  }
}

VulkanAllocator& allocator() {
  // Deliberately never destroyed. The completion dispatcher drains
  // whenever a submission retires - including during static destruction,
  // when a watchdog-thrown submit finishes after the teardown has begun
  // - and the drained temporaries free arrays through this object's
  // virtuals. Destroying it under such a late drain aborts with "pure
  // virtual method called" (observed on the qmm teardown crash). The
  // device-side resources behind cached buffers are children of the
  // VkDevice and die with vkDestroyDevice; only the small host structs
  // leak, matching the scheduler-leak and teardown-temporaries-leak
  // precedents.
  static auto* allocator_ = new VulkanAllocator();
  return *allocator_;
}

} // namespace omarchy

namespace allocator {

Allocator& allocator() {
  return omarchy::allocator();
}

void* Buffer::raw_ptr() {
  if (!ptr_) {
    return nullptr;
  }
  return static_cast<omarchy::VulkanBuffer*>(ptr_)->data;
}

bool can_reuse_alien_buffer(void*) {
  return true;
}

} // namespace allocator

size_t get_active_memory() {
  return omarchy::allocator().get_active_memory();
}
size_t get_peak_memory() {
  return omarchy::allocator().get_peak_memory();
}
void reset_peak_memory() {
  omarchy::allocator().reset_peak_memory();
}
size_t set_memory_limit(size_t limit) {
  return omarchy::allocator().set_memory_limit(limit);
}
size_t get_memory_limit() {
  return omarchy::allocator().get_memory_limit();
}
size_t get_cache_memory() {
  return omarchy::allocator().get_cache_memory();
}
size_t set_cache_limit(size_t limit) {
  return omarchy::allocator().set_cache_limit(limit);
}
void clear_cache() {
  omarchy::allocator().clear_cache();
}

// Omarchy is unified memory: there is no GPU-private RAM to wire and
// mlock(2) would only pin CPU pages the Vulkan allocator does not need
// pinned. The wired limit is therefore a second allocator ceiling: while
// it exceeds the memory limit (oMLX dflash acquire/restore raises it to
// max_recommended_working_set_size during a load phase), the effective
// limit becomes max(memory_limit, wired_limit) in malloc. The value is
// remembered in the allocator and restore round-trips the previous
// setting, so an "unset" state stays distinguishable from "set to 0".
// vkAllocateMemory remains the final arbiter of what actually fits.
size_t set_wired_limit(size_t limit) {
  return omarchy::allocator().set_wired_limit(limit);
}

} // namespace mlx::core
