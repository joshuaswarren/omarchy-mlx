// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

#include "mlx/backend/omarchy/allocator.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <mutex>
#include <unordered_set>
#include <stdexcept>
#include <string>

#include "mlx/backend/omarchy/device.h"
#include "mlx/backend/omarchy/vulkan.h"
#include "mlx/memory.h"

namespace mlx::core {

namespace omarchy {

constexpr size_t kPageSize = 4096;
// Requests above kBinThreshold round up to one of kSubBinsPerOctave sizes
// per power of two, so a buffer wastes at most 1/kSubBinsPerOctave of its
// size (12.5%) and still lands in a reusable bin. The previous rule rounded
// to the next power of two (up to 100% waste): a 24 MiB offload slot array
// became 32 MiB, 33% more device memory per array, and on a 16 GB host the
// offloaded Qwen3-30B-A3B ran out of the Honeykrisp heap (50% of RAM).
constexpr size_t kBinThreshold = 1u << 20;
constexpr size_t kSubBinsPerOctave = 8;

size_t round_size(size_t size) {
  if (size <= kPageSize) {
    return kPageSize;
  }
  if (size > kBinThreshold && size <= SIZE_MAX / 2) {
    const size_t octave = size_t{1} << (63 - __builtin_clzll(size));
    const size_t step = octave / kSubBinsPerOctave;
    return step * ((size + step - 1) / step);
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

void poison_freed_buffer(void* data, size_t size) {
  auto* words = static_cast<uint32_t*>(data);
  size_t count = size / sizeof(uint32_t);
  for (size_t i = 0; i < count; ++i) {
    words[i] = kPoisonFreedWord;
  }
}

// Host-pointer import toggle. Default on (the import path is the
// sushi zero-copy expert-slab design's contract). The kill switch
// forces the old copy path: MLX_OMARCHY_HOST_IMPORT=0|off|false|no
// returns a null Buffer from make_buffer, identical to a backend
// without the extension. Used to bisect a regression to the new
// import path without rebuilding the test.
bool host_import_disabled() {
  const char* v = std::getenv("MLX_OMARCHY_HOST_IMPORT");
  if (v == nullptr || *v == '\0') {
    return false;
  }
  std::string s = v;
  std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return s == "0" || s == "off" || s == "false" || s == "no";
}

// Compact 0x.. formatter for the memory-type bit masks the debug
// lines report.
std::string hex32(uint32_t bits) {
  static const char* digits = "0123456789abcdef";
  std::string out = "0x";
  if (bits == 0) {
    return out + "0";
  }
  bool started = false;
  for (int shift = 28; shift >= 0; shift -= 4) {
    uint32_t nibble = (bits >> shift) & 0xf;
    if (nibble != 0 || started) {
      out.push_back(digits[nibble]);
      started = true;
    }
  }
  return out;
}

// MLX_OMARCHY_HOST_IMPORT_DEBUG=1: one stderr line per process per
// distinct decline reason (and one on first successful alias), so a
// device receipt shows exactly which gate declined and with what
// driver-reported state. Line shape:
//   [omarchy] host-import: decline=<reason> ext=<0|1> align=<N>
//       props_types=<0x..> reqs_types=<0x..> <detail>
//   [omarchy] host-import: aliased align=<N>
// The per-process-once rule keeps a serve loop from spamming stderr:
// the same slab re-imported every layer would print the same line
// every call otherwise.
void host_import_note(const std::string& key, const std::string& detail) {
  static const bool enabled = env_flag("MLX_OMARCHY_HOST_IMPORT_DEBUG");
  if (!enabled) {
    return;
  }
  static std::mutex log_mu;
  static std::unordered_set<std::string> logged;
  std::unique_lock lk(log_mu);
  if (!logged.insert(key).second) {
    return;
  }
  std::fprintf(
      stderr, "[omarchy] host-import: %s%s\n", key.c_str(), detail.c_str());
  std::fflush(stderr);
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
      live_by_handle_[buf->buffer] = buf;
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
    // Freed buffers of submitted batches wait in the quarantine; they hold
    // heap bytes that neither active_memory_ nor the cache count, so the
    // retry would fail while that memory is about to come back. Wait for
    // every submitted batch (the drain recycles its quarantined buffers),
    // then drop the cache they landed in. No allocator lock is held here.
    if (const uint64_t submitted = device().completions().last_reserved();
        submitted != 0) {
      device().completions().wait(submitted);
    }
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
  live_by_handle_[buf->buffer] = buf;
  active_memory_ += buf->size;
  peak_memory_ = std::max(active_memory_, peak_memory_);
  if (!buf->coherent) {
    noncoherent_.push_back(buf);
  }
  lk.unlock();
  return Buffer{buf};
}

Buffer VulkanAllocator::make_buffer(void* ptr, size_t size) {
  // Five early outs to the copy path. Every one of them is a contract
  // the array(void*, ...) ctor accepts (Buffer{nullptr} -> copy): no
  // device, env-killed, ptr/size too small, extension unadvertised, or
  // the function pointer the backend's loader resolved to nothing.
  if (ptr == nullptr || size == 0) {
    return Buffer{nullptr};
  }
  if (!is_available()) {
    host_import_note(
        "decline=no-device ", "");
    return Buffer{nullptr};
  }
  if (host_import_disabled()) {
    host_import_note(
        "decline=env-disabled ", "MLX_OMARCHY_HOST_IMPORT=0");
    return Buffer{nullptr};
  }
  device();
  const auto& hw = device().hardware_capabilities();
  const std::string caps = "ext=" +
      std::to_string(hw.host_pointer_import ? 1 : 0) +
      " align=" + std::to_string(hw.host_pointer_import_alignment);
  if (!hw.host_pointer_import) {
    // Either the driver does not list the extension, or it lists it
    // with a zero alignment. The two values above say which; a listed
    // extension with align=0 would also cover a driver that only
    // imports dma-buf-backed handles.
    host_import_note("decline=ext-not-advertised ", caps);
    return Buffer{nullptr};
  }
  auto& dt = vk::device_table();
  if (dt.GetMemoryHostPointerPropertiesEXT == nullptr) {
    host_import_note("decline=fn-missing ", caps);
    return Buffer{nullptr};
  }
  const uint32_t align = hw.host_pointer_import_alignment;
  if (align == 0) {
    host_import_note("decline=align-zero ", caps);
    return Buffer{nullptr};
  }
  if ((reinterpret_cast<uintptr_t>(ptr) % align) != 0) {
    host_import_note(
        "decline=unaligned-pointer ",
        caps + " ptr_mod_align=" +
            std::to_string(reinterpret_cast<uintptr_t>(ptr) % align));
    return Buffer{nullptr};
  }
  // Per-pointer validation: the host pointer must be importable and
  // must land in at least one memory type.  The driver returns the
  // memoryTypeBits we must intersect with the buffer's reqs; if that
  // intersection is empty the import is impossible for this shape.
  VkMemoryHostPointerPropertiesEXT hpp{
      VK_STRUCTURE_TYPE_MEMORY_HOST_POINTER_PROPERTIES_EXT};
  VkResult props_res = dt.GetMemoryHostPointerPropertiesEXT(
      device().handle(),
      VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT,
      ptr,
      &hpp);
  if (props_res != VK_SUCCESS || hpp.memoryTypeBits == 0) {
    // A failure here on a driver that lists the extension means the
    // HOST_ALLOCATION handle type is not actually supported for this
    // pointer (some drivers only wire dma-buf imports through).
    host_import_note(
        "decline=props-failed ",
        caps + " vkresult=" + vk::result_string(props_res) +
            " props_types=" + hex32(hpp.memoryTypeBits));
    return Buffer{nullptr};
  }
  // The allocation is taken to cover the requested bytes rounded UP to
  // the import alignment (the spec requires the host range to be
  // aligned; rounding beyond the requested size is harmless on every
  // driver we exercise as long as the caller's slab is at least that
  // big — the GPU never reads past nbytes). The buffer's memory
  // requirements gate `reqs.size > import_size` is treated as
  // impossible: a buffer cannot be backed by a memory allocation
  // smaller than the driver itself requires, and clamping would
  // silently corrupt a future rebind.
  const size_t import_size = ((size + align - 1) / align) * align;

  auto* buf = new VulkanBuffer{};
  buf->imported = true;
  buf->data = ptr;
  buf->size = size;

  VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  bci.size = size;
  bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
      VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  VkResult res = dt.CreateBuffer(device().handle(), &bci, nullptr, &buf->buffer);
  if (res != VK_SUCCESS) {
    host_import_note(
        "decline=create-buffer-failed ",
        caps + " vkresult=" + vk::result_string(res));
    delete buf;
    return Buffer{nullptr};
  }

  VkMemoryRequirements reqs{};
  dt.GetBufferMemoryRequirements(device().handle(), buf->buffer, &reqs);
  if (reqs.size > import_size) {
    host_import_note(
        "decline=reqs-too-big ",
        caps + " reqs_size=" + std::to_string(reqs.size) +
            " import_size=" + std::to_string(import_size));
    dt.DestroyBuffer(device().handle(), buf->buffer, nullptr);
    delete buf;
    return Buffer{nullptr};
  }
  // The memory type must satisfy both the buffer's reqs and the
  // host-pointer props, and must be HOST_VISIBLE | HOST_COHERENT.
  // Host memory is coherent with the CPU by definition; a non-coherent
  // memory type for an imported host range is driver error, not our
  // problem to paper over.
  uint32_t type_bits =
      reqs.memoryTypeBits & hpp.memoryTypeBits;
  uint32_t type_index = find_memory_type(
      type_bits,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
          VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  if (type_index == UINT32_MAX) {
    // Distinguish "no common type at all" from "common types exist but
    // none is HOST_VISIBLE|HOST_COHERENT", and dump the candidate
    // types' property flags: a driver whose imported-host type lacks
    // HOST_COHERENT lands here and the receipt shows it.
    std::string detail = caps + " props_types=" + hex32(hpp.memoryTypeBits) +
        " reqs_types=" + hex32(reqs.memoryTypeBits) + " common=" +
        hex32(type_bits);
    if (type_bits != 0) {
      const auto& mem = device().memory_properties();
      for (uint32_t t = 0; t < mem.memoryTypeCount && t < 32; ++t) {
        if (type_bits & (1u << t)) {
          detail += " t" + std::to_string(t) + "=0x" +
              hex32(mem.memoryTypes[t].propertyFlags).substr(2);
        }
      }
      host_import_note("decline=no-coherent-type ", detail);
    } else {
      host_import_note("decline=no-common-type ", detail);
    }
    dt.DestroyBuffer(device().handle(), buf->buffer, nullptr);
    delete buf;
    return Buffer{nullptr};
  }

  VkImportMemoryHostPointerInfoEXT imp{
      VK_STRUCTURE_TYPE_IMPORT_MEMORY_HOST_POINTER_INFO_EXT};
  imp.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT;
  imp.pHostPointer = ptr;
  VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  mai.pNext = &imp;
  mai.allocationSize = import_size;
  mai.memoryTypeIndex = type_index;
  res = dt.AllocateMemory(device().handle(), &mai, nullptr, &buf->memory);
  if (res != VK_SUCCESS) {
    host_import_note(
        "decline=alloc-failed ",
        caps + " type=" + std::to_string(type_index) + " import_size=" +
            std::to_string(import_size) +
            " vkresult=" + vk::result_string(res));
    dt.DestroyBuffer(device().handle(), buf->buffer, nullptr);
    delete buf;
    return Buffer{nullptr};
  }
  res = dt.BindBufferMemory(device().handle(), buf->buffer, buf->memory, 0);
  if (res != VK_SUCCESS) {
    host_import_note(
        "decline=bind-failed ",
        caps + " vkresult=" + vk::result_string(res));
    // FreeMemory on an import has no effect on the host range; safe.
    dt.FreeMemory(device().handle(), buf->memory, nullptr);
    dt.DestroyBuffer(device().handle(), buf->buffer, nullptr);
    delete buf;
    return Buffer{nullptr};
  }
  // No MapMemory: the imported range IS host memory; CPU writes go
  // straight to |ptr|. buf->data = ptr is the aliasing contract that
  // makes sushi's `data_ptr == slab.ptr` check pass.
  {
    std::unique_lock lk(mutex_);
    live_by_handle_[buf->buffer] = buf;
    active_memory_ += buf->size;
    peak_memory_ = std::max(active_memory_, peak_memory_);
  }
  host_import_note(
      "aliased ", "align=" + std::to_string(align) +
          " import_size=" + std::to_string(import_size));
  return Buffer{buf};
}

void VulkanAllocator::release(Buffer buffer) {
  // The make_buffer mirror: only imported buffers arrive here. The
  // quarantine mirrors free() so the Vk objects survive any in-flight
  // GPU work before destroy_buffer() runs. Host bytes are the
  // caller's; the wrapped_deleter in array.cpp invokes our deleter
  // last, after we return.
  auto* buf = static_cast<VulkanBuffer*>(buffer.ptr());
  if (buf == nullptr) {
    return;
  }
  std::unique_lock lk(mutex_);
  active_memory_ -= buf->size;
  if (buf->completion == 0) {
    lk.unlock();
    destroy_buffer(buf);
    return;
  }
  if (buf->completion == kPendingCompletion) {
    pending_quarantine_bytes_ += buf->size;
  }
  quarantine_.push_back(buf);
}

void VulkanAllocator::destroy_buffer(VulkanBuffer* buf) {
  // Cache clear/release funnels through here while callers already hold
  // mutex_, so the non-coherent registry drops the buffer before its
  // memory dies; later flush/invalidate must never see it.
  std::erase(noncoherent_, buf);
  auto& dt = vk::device_table();
  live_by_handle_.erase(buf->buffer);
  if (buf->memory != VK_NULL_HANDLE) {
    // Imported host memory is never mapped (it IS host memory);
    // UnmapMemory on a never-mapped import is an error per spec.
    if (!buf->imported) {
      dt.UnmapMemory(device().handle(), buf->memory);
    }
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
  // Imported buffers are caller-owned bytes; nothing the reuse cache
  // does is ever safe. Quarantine until the driver drains, then
  // destroy the Vk objects (release_quarantine sees the flag).
  if (buf->imported) {
    quarantine_.push_back(buf);
    return;
  }
  // A buffer stamped with a completion value has recorded GPU work that
  // may still touch it (kPendingCompletion = an open batch). Freeing the
  // array drops the accounting here — upstream Metal semantics — but the
  // raw buffer skips the reuse cache and waits in the quarantine until
  // release_quarantine proves the driver finished its submission. A
  // buffer whose generation drained at least one generation ago (a later
  // generation is observable) needs no wait; without this an idle
  // process parks every freed buffer until the next submission.
  if (buf->completion != 0) {
    if (buf->completion == kPendingCompletion || !runtime_alive() ||
        buf->completion >= device().completions().drained_value() ||
        !device().completions().execution_complete(buf->completion)) {
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
  // Callable from any thread: the allocator mutex serializes passes, and
  // each entry's own stamps decide recyclability. A block recycles only
  // when its completion generation is strictly behind |cleanup_done_
  // through| (the design's one-generation rule: observing generation N
  // does not prove the driver's submit-final cleanup for N has run) and
  // its execution fence has signalled. A fence that signals after this
  // pass is picked up by a later one - the dispatcher keeps ticking
  // release passes while fences are outstanding or blocks are parked.
  std::unique_lock lk(mutex_);
  if (quarantine_.empty()) {
    return;
  }
  std::vector<VulkanBuffer*> still_quarantined;
  for (auto* buf : quarantine_) {
    if (buf->completion > cleanup_done_through ||
        buf->completion == kPendingCompletion ||
        !device().completions().execution_complete(buf->completion)) {
      still_quarantined.push_back(buf);
      continue;
    }
    // Imported buffers reach this pass via free() (defensive; the
    // normal path goes through release). Their Vk objects are safe to
    // destroy; the host bytes belong to the caller and are untouched
    // by destroy_buffer's UnmapMemory-skipping branch.
    if (buf->imported) {
      destroy_buffer(buf);
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

bool VulkanAllocator::has_quarantined() const {
  std::unique_lock lk(mutex_);
  return !quarantine_.empty();
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


VulkanBuffer* VulkanAllocator::note_buffer_handle(VkBuffer handle) {
  std::unique_lock lk(mutex_);
  auto it = live_by_handle_.find(handle);
  if (it == live_by_handle_.end()) {
    return nullptr;
  }
  VulkanBuffer* buf = it->second;
  buf->completion = kPendingCompletion;
  return buf;
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
