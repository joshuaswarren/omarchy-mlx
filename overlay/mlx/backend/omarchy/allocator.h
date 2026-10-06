// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

#pragma once

#include <vulkan/vulkan.h>

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>
#include "mlx/allocator.h"
#include "mlx/backend/common/buffer_cache.h"

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
  // False when the memory type lacks HOST_COHERENT.
  bool coherent{true};
  // Completion-timeline stamp of the submission that references this
  // buffer. 0 = no in-flight window; kPendingCompletion = recorded by an
  // open (not yet submitted) batch. Written by the encoder (add_temporary
  // and submit), consumed by the allocator quarantine (see free()).
  uint64_t completion{0};
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

class VulkanAllocator : public allocator::Allocator {
 public:
  Buffer malloc(size_t size) override;
  void free(Buffer buffer) override;
  size_t size(Buffer buffer) const override;

  size_t get_active_memory() const {
    std::unique_lock lk(mutex_);
    return active_memory_;
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
    return buffer_cache_.cache_size();
  }
  size_t get_cache_limit() const {
    std::unique_lock lk(mutex_);
    return cache_limit_;
  }
  // Working-set ceiling (Metal: `gc_limit_`). When `active + cache + size`
  // would exceed it, malloc releases cached buffers before allocating.
  // Defaults to 95% of `total_memory` (Honeykrisp reports a single
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
  // the reuse cache; release_quarantine recycles them one completion
  // generation later, once the driver's submit-final cleanup for their
  // submission has provably run. Accounting (active_memory_) drops at
  // free() time, matching upstream Metal.
  void release_quarantine(uint64_t cleanup_done_through);

  // Encoder-level transfers (copy_buffer/fill_buffer) take raw VkBuffer
  // handles: resolve + stamp the owning live block so a transfer marks its
  // target in flight; returns nullptr for unknown handles.
  VulkanBuffer* note_buffer_handle(VkBuffer handle);

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
  // Resolve a memory type with the required flags, preferring types backed by
  // a device-local heap (unified memory on Apple GPUs).
  uint32_t find_memory_type(uint32_t type_bits, VkMemoryPropertyFlags required)
      const;

  mutable std::mutex mutex_;
  size_t memory_limit_;
  size_t wired_limit_{0};
  size_t cache_limit_;
  // Metal's `gc_limit_`: working-set ceiling above which malloc
  // proactively releases cached buffers. Defaults to 95% of the device
  // host-visible heap (Honeykrisp's reported `total_memory`); oMLX can
  // raise it with `set_memory_limit` when the dflash acquire raises the
  // working set.
  size_t gc_limit_;
  size_t active_memory_{0};
  size_t peak_memory_{0};
  mutable BufferCache<VulkanBuffer> buffer_cache_;
  // Live blocks by VkBuffer handle (encoder-level transfer stamping).
  std::unordered_map<VkBuffer, VulkanBuffer*> live_by_handle_;
  std::vector<VulkanBuffer*> noncoherent_;
  // Freed buffers whose recorded GPU work may still reference them (see
  // release_quarantine). Held outside the reuse cache until released.
  std::vector<VulkanBuffer*> quarantine_;
  size_t pending_quarantine_bytes_{0};
};

MLX_API VulkanAllocator& allocator();

// Round a request to the allocator page so cached buffers can be reused for
// any smaller request.
size_t round_size(size_t size);

} // namespace mlx::core::omarchy
