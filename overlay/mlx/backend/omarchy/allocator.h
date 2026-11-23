// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

#pragma once

#include <vulkan/vulkan.h>

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <span>
#include <unordered_map>
#include <utility>
#include <vector>
#include "mlx/allocator.h"
#include "mlx/backend/common/buffer_cache.h"
#include "mlx/backend/omarchy/device_local_activations.h"
#include "mlx/backend/omarchy/memory_type.h"

namespace mlx::core::omarchy {

using allocator::Buffer;

// Host-visible buffer. Apple GPUs are UMA: Honeykrisp exposes a device-local
// heap that is host visible and host coherent, so the mapped pointer is both
// the CPU access path and the GPU buffer backing. When a driver offers no
// coherent type the allocator falls back to HOST_VISIBLE memory and the
// runtime performs explicit vkFlushMappedMemoryRanges /
// vkInvalidateMappedMemoryRanges maintenance. Each buffer owns one
// VkDeviceMemory allocation, which keeps mapped-range lifetime exact.
struct VulkanBuffer {
  VkBuffer buffer{VK_NULL_HANDLE};
  VkDeviceMemory memory{VK_NULL_HANDLE};
  void* data{nullptr};
  size_t size{0};
  bool inference_weight{false};
  // True when backed by non-HOST_VISIBLE DEVICE_LOCAL VRAM under the opt-in
  // discrete-GPU activation path (MLX_OMARCHY_VRAM_ACTIVATIONS=1).
  bool device_local{false};
  // Lazily allocated CPU shadow buffer, created only if host code requests
  // Buffer::raw_ptr() on a device_local buffer.
  void* host_shadow{nullptr};
  DeviceLocalSyncState sync_state{};
  // False when the memory type lacks HOST_COHERENT.
  bool coherent{true};
  // Completion-timeline stamp of the submission that references this
  // buffer. 0 = no in-flight window; kPendingCompletion = recorded by an
  // open (not yet submitted) batch. Written by the encoder (add_temporary
  // and submit), consumed by the allocator quarantine (see free()).
  uint64_t completion{0};
  // True while a private Device staging transfer (upload_host_to_device /
  // download_device_to_host) has a command buffer referencing this VkBuffer
  // in flight. Protected by VulkanAllocator::mutex_. Keeps free() and
  // release_quarantine() from recycling or destroying the buffer on timeout.
  bool transfer_in_flight{false};
  // True from the moment the reuse cache hands this block out until a
  // scalar fill drains it: a fill whose destination is freshly recycled
  // storage must drain prior work first, because its write provably
  // loses to the in-flight writes of the block's previous occupant
  // (convolve/pool boundary garbage, receipts/2026-09-11-wrong-value-
  // sweep). Fresh device-memory allocations are never recycled.
  bool recycled{false};
};

// Stamp written by add_temporary for buffers recorded into an open
// batch; submit() replaces it with the real completion value.
inline constexpr uint64_t kPendingCompletion = UINT64_MAX;

// Default allocator limits from the Vulkan host-visible heap size
// (0 = no device: 4 GiB). The memory limit is where malloc drops the
// whole reuse cache; it must not sit below what the heap can hold, or a
// model whose weights fit the heap but exceed the limit loses the cache
// on every allocation. The gc ceiling is the heap as well: a ceiling below
// the weights (95% of a 60 GiB heap is 57 GiB, under a 58 GiB model) makes
// every malloc release cached buffers, and that was measured to slow decode
// (matched diagnostic pairs, gpt-oss-120b weights on a 60 GiB heap). The
// ceiling still trims the cache by its
// overshoot (A20). The limit is not below the gc ceiling,
// the order Metal uses, so by default both fire only past the whole heap:
// tuning either below the heap is what changes cache behaviour.
// The open-batch quarantine budget (eval.cpp) keeps its old base, 90% of
// the heap, so only the cache behaviour moves with the default limit.
// The BEGIN/END markers delimit the text a standalone test harness extracts;
// keep them.
// BEGIN default limits
inline size_t default_memory_limit(size_t heap_bytes) {
  return heap_bytes > 0 ? heap_bytes : (size_t{1} << 32);
}

inline size_t default_gc_limit(size_t heap_bytes) {
  return heap_bytes > 0 ? heap_bytes : (size_t{1} << 32);
}

// The reuse cache holds a quarter of the heap. Freed device memory leaves
// the GEM total late (#435: 4.4 GB of a released cache still counted one
// 256-token chunk later), so a cache allowed to grow to the whole heap
// left no room for the next allocation: on the 13-inch M1, a 4096-token
// prefill grew the cache to 5 GB next to 2.4 GB active and failed
// GEM_CREATE at 7.7 GB. The cap bounds only freed buffers, never active
// ones, so a model that fits the heap still never trims on every malloc.
inline size_t default_cache_limit(size_t heap_bytes) {
  return default_gc_limit(heap_bytes) / 4;
}

inline size_t default_batch_limit(size_t memory_limit, size_t heap_bytes) {
  const size_t base =
      heap_bytes > 0 ? (heap_bytes / 100) * 90 : (size_t{1} << 32);
  return memory_limit < base ? memory_limit : base;
}
// END default limits

class VulkanAllocator : public allocator::Allocator {
 public:
  Buffer malloc(size_t size) override;
  Buffer malloc_weight(size_t size);
  // Bytes currently live in card-memory inference weights. The placement
  // gate reads this per load, so it takes the mutex like the other
  // accounting getters.
  size_t weight_memory_bytes() const {
    std::unique_lock lk(mutex_);
    return weight_memory_;
  }
  void free(Buffer buffer) override;
  size_t size(Buffer buffer) const override;

  size_t get_active_memory() const {
    std::unique_lock lk(mutex_);
    return active_memory_ + weight_memory_;
  }
  size_t get_peak_memory() const {
    std::unique_lock lk(mutex_);
    return peak_memory_;
  }
  void reset_peak_memory() {
    std::unique_lock lk(mutex_);
    peak_memory_ = 0;
  }
  size_t get_memory_limit() const {
    std::unique_lock lk(mutex_);
    return memory_limit_;
  }
  size_t set_memory_limit(size_t limit) {
    std::unique_lock lk(mutex_);
    std::swap(memory_limit_, limit);
    return limit;
  }
  // Bytes the evaluator divides by kBatchByteBudgetDivisor for the open-
  // batch quarantine budget: the memory limit, capped at the default 90%
  // of the heap (see default_batch_limit).
  size_t batch_limit() const {
    std::unique_lock lk(mutex_);
    return default_batch_limit(memory_limit_, heap_bytes_);
  }
  size_t get_wired_limit() const {
    std::unique_lock lk(mutex_);
    return wired_limit_;
  }
  size_t set_wired_limit(size_t limit) {
    std::unique_lock lk(mutex_);
    std::swap(wired_limit_, limit);
    return limit;
  }
  size_t get_cache_memory() const {
    std::unique_lock lk(mutex_);
    return buffer_cache_.cache_size() + device_local_cache_.cache_size();
  }
  size_t get_cache_limit() const {
    std::unique_lock lk(mutex_);
    return cache_limit_;
  }
  // Working-set ceiling (Metal: `gc_limit_`). When `active + cache + size`
  // would exceed it, malloc releases cached buffers before allocating.
  // Defaults to `total_memory` (Honeykrisp reports a single
  // unified host-visible heap on Apple-UMA, so there is no separate
  // "max_recommended_working_set_size" like Metal exposes).
  size_t get_gc_limit() const {
    std::unique_lock lk(mutex_);
    return gc_limit_;
  }
  size_t set_gc_limit(size_t limit) {
    std::unique_lock lk(mutex_);
    std::swap(gc_limit_, limit);
    return limit;
  }
  size_t set_cache_limit(size_t limit);
  void clear_cache();

  // Cache maintenance for non-coherent mapped memory (Khronos guidance:
  // HOST_VISIBLE alone does not guarantee coherence). The encoder flushes
  // before submission and invalidates after the fence wait. Buffers on
  // coherent memory types are skipped.
  void flush_noncoherent(VkDevice device);
  void invalidate_noncoherent(VkDevice device);

  // Quarantine for buffers whose last reference died while recorded GPU
  // work may still touch them. free() routes such buffers here instead of
  // the reuse cache; release_quarantine recycles entries whose generation
  // has drained one completion earlier (|cleanup_done_through|) AND whose
  // execution fence has signalled. Accounting (active_memory_) drops at
  // free() time, matching upstream Metal.
  void release_quarantine(uint64_t cleanup_done_through);

  // True while any block waits in the quarantine. The dispatcher's idle
  // tick uses this to keep release passes running after the last
  // submission drained (a generation-gated block recycles on a later
  // pass even once its fence retired).
  bool has_quarantined() const;

  // Encoder-level transfers (copy_buffer/fill_buffer) take raw VkBuffer
  // handles: resolve + stamp the owning live block so a transfer marks its
  // target in flight; returns nullptr for unknown handles.
  VulkanBuffer* note_buffer_handle(VkBuffer handle);
  // Non-mutating lookup of a live block by VkBuffer handle.
  VulkanBuffer* lookup_buffer_handle(VkBuffer handle);
  // Thread-safe read of a live buffer's completion stamp.
  uint64_t buffer_completion(const VulkanBuffer* buf) const;
  // Pin or unpin a buffer referenced by an in-flight private staging transfer
  // so timeout/unwind preserves its VkBuffer/VkDeviceMemory lifetime.
  void mark_transfer_in_flight(VkBuffer handle, bool in_flight);

  // Record a buffer referenced by an open batch (encoder add_temporary
  // and dispatch bindings). The stamp is unconditional: a buffer whose
  // stamp still names an older in-flight generation must not recycle
  // under that older generation's rule while THIS batch holds it
  // recorded, and a never-submitted buffer (completion 0) must not
  // recycle at all before its batch submits. submit() overwrites the
  // stamp with the real completion value.
  void note_batch_buffer(VulkanBuffer* buf) {
    std::unique_lock lk(mutex_);
    if (buf) {
      buf->completion = kPendingCompletion;
    }
  }

  // note_batch_buffer for a whole binding set under one lock.
  void note_batch_buffers(std::span<VulkanBuffer* const> bufs) {
    std::unique_lock lk(mutex_);
    for (auto* buf : bufs) {
      if (buf) {
        buf->completion = kPendingCompletion;
      }
    }
  }

  // Stamp a submitted batch's buffers with their real completion value
  // and refresh pending_quarantine_bytes(): freed buffers of the batch
  // stop counting against the open-batch budget once they can drain.
  void stamp_batch(const std::vector<VulkanBuffer*>& bufs, uint64_t completion);

  // Bytes of freed buffers pinned by a batch that has not yet been
  // submitted (quarantined at kPendingCompletion). Nothing can recycle
  // them until the batch reaches the queue, so the evaluator flushes the
  // batch once this passes get_memory_limit() / kBatchByteBudgetDivisor
  // (encoder.h). Weights and live intermediates are not counted: their
  // memory is owed to the graph, not to batching.
  size_t pending_quarantine_bytes() const {
    std::unique_lock lk(mutex_);
    return pending_quarantine_bytes_;
  }

 private:
  VulkanAllocator();
  friend VulkanAllocator& allocator();

  void destroy_buffer(VulkanBuffer* buf);
  // Resolve a memory type with the required flags, per-heap capacity-aware:
  // prefer a device-local heap (unified memory on Apple GPUs) only while it
  // is at least as large as the best matching non-device-local heap, else
  // the largest matching heap (see capacity_aware_memory_type).
  uint32_t find_memory_type(uint32_t type_bits, VkMemoryPropertyFlags required)
      const;

  mutable std::mutex mutex_;
  size_t memory_limit_;
  size_t heap_bytes_{0};
  size_t wired_limit_{0};
  size_t cache_limit_;
  // Metal's `gc_limit_`: working-set ceiling above which malloc
  // proactively releases cached buffers. Defaults to the device
  // host-visible heap (Honeykrisp's reported `total_memory`); oMLX can
  // raise it with `set_memory_limit` when the dflash acquire raises the
  // working set.
  size_t gc_limit_;
  size_t active_memory_{0};
  size_t weight_memory_{0};
  size_t peak_memory_{0};
  mutable BufferCache<VulkanBuffer> buffer_cache_;
  mutable BufferCache<VulkanBuffer> device_local_cache_;
  size_t device_local_cache_limit_{0};
  size_t device_local_gc_limit_{0};
  // Live blocks by VkBuffer handle (encoder-level transfer stamping).
  std::unordered_map<VkBuffer, VulkanBuffer*> live_by_handle_;
  std::vector<VulkanBuffer*> noncoherent_;
  // Freed buffers whose recorded GPU work may still reference them (see
  // release_quarantine). Held outside the reuse cache until released.
  std::vector<VulkanBuffer*> quarantine_;
  size_t pending_quarantine_bytes_{0};
  // Bytes of every buffer in quarantine_: freed, but still device memory.
  size_t quarantine_bytes_{0};
};

MLX_API VulkanAllocator& allocator();

// True when MLX_OMARCHY_VRAM_ACTIVATIONS=1 is set and the active device is a
// discrete GPU with both a pure DEVICE_LOCAL memory type and a host-visible
// staging memory type.
bool device_local_activations_enabled();

// Round a request to the allocator page so cached buffers can be reused for
// any smaller request.
size_t round_size(size_t size);

} // namespace mlx::core::omarchy
