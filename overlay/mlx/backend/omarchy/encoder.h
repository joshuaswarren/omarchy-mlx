// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

#pragma once

#include <vulkan/vulkan.h>

#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include "mlx/array.h"
#include "mlx/backend/omarchy/allocator.h"
#include "mlx/backend/omarchy/compute.h"
#include "mlx/backend/omarchy/device.h"
#include "mlx/stream.h"

namespace mlx::core::omarchy {

// Per-stream command recorder over the device's single VkQueue. Recording
// is BATCHED: primitive evals append to an open command buffer and the
// buffer is submitted when a node/work budget is reached, a flush is
// demanded (semaphore operation, host read), or the in-flight ring is
// exhausted. Batching is order-safe because every dispatch is separated
// from its neighbors by a full dependency (MLX_OMARCHY_GATED_BARRIERS=0,
// the default: unconditional pre+post memory barriers per dispatch;
// =1: a barrier only when the node's ranges overlap work recorded since
// the last barrier, tracked per open batch), every submission waits on
// the completion-timeline value of this stream's previous submission
// (Vulkan defines no cross-submission dependency without a wait) and
// cross-submission waits use ALL_COMMANDS stage masks, so merging
// submissions weakens no dependency; it removes the per-eval host join
// between them. Temporaries and completion handlers released per
// submission still release exactly when that submission's GPU work
// finishes, via the device completion timeline.
//
// Command buffers come from a small ring so the device can execute one
// batch while the host records the next; the host blocks only when every
// ring slot is in flight (then it joins just the oldest) or when a host
// read demands it (synchronize()).
//
// Every submission signals a strictly increasing value on the device
// completion timeline; the device's CompletionDispatcher runs this
// submission's handlers and releases its temporaries and queued-semaphore
// ownership when the GPU work finishes. Cross-stream order comes from
// timeline semaphore waits and signals carried in the submissions;
// handler-only submissions (no recorded commands) still signal the
// timeline, so ordering with prior queue work is preserved.
// Caps recorded work and pinned buffers per submission. Larger batches reduce
// submit overhead but extend buffer lifetimes and watchdog exposure.
// Decode graphs (~585 nodes/token on jwm1) split into three submits per token
// at 256 (jwm1 decode profile, 2026-09-23, DecodeGap receipt); 4096 keeps a
// whole decode token on one submit. The byte budget below remains the real
// cap for large-tensor graphs, so prefill flush behavior is unchanged
// (2026-09-08 8.11 GB incident stays covered by the byte budget).
inline constexpr int kBatchNodeBudget = 4096;
// Byte budget for the same batch: freed intermediates stay pinned in the
// allocator quarantine until their batch submits and drains, so the open
// batch may hold at most 1/16 of the allocator memory limit in such bytes
// before the evaluator flushes it. Up to five generations can be pinned at
// once (four ring slots in flight plus the one-generation-late quarantine
// release), so the bound keeps batching-owed memory under a third of the
// limit. A 2,048-token Qwen2.5-0.5B forward held 8.11 GB in one 257-node
// batch against Honeykrisp's 7.56 GiB heap without this (2026-09-08).
inline constexpr size_t kBatchByteBudgetDivisor = 16;

class MLX_API CommandEncoder {
 public:
  explicit CommandEncoder(Device& device);
  ~CommandEncoder();

  CommandEncoder(const CommandEncoder&) = delete;
  CommandEncoder& operator=(const CommandEncoder&) = delete;

  // Record the array's buffer as referenced by the open batch. The raw
  // buffer is stamped kPendingCompletion now and with the submission's
  // completion value at submit(); the allocator quarantine keeps a
  // buffer whose array died mid-flight out of the reuse cache until the
  // driver provably finished that submission. No shared_ptr is held:
  // eager donation (is_donatable) must see the input's refcount at 1.
  void add_temporary(const array& arr) {
    auto data = arr.data_shared_ptr();
    if (!data) {
      return;
    }
    auto* buf = static_cast<VulkanBuffer*>(data->buffer.ptr());
    if (!buf) {
      return;
    }
    allocator().note_batch_buffer(buf);
    batch_buffers_.push_back(buf);
  }

  // Record one dispatch binding's owning buffer exactly like
  // add_temporary records an array: ComputeBinding::owner covers the
  // buffers a dispatch binds that no add_temporary call registered
  // (plain input and output arrays), which otherwise could be freed
  // with completion == 0 while the queued commands still reference
  // them, and then be recycled or destroyed mid-flight.
  void note_binding_owner(const void* owner) {
    auto* buf = static_cast<VulkanBuffer*>(const_cast<void*>(owner));
    if (!buf) {
      return;
    }
    allocator().note_batch_buffer(buf);
    batch_buffers_.push_back(buf);
  }

  // Handlers run on the device completion thread when this submission's
  // GPU work finishes.
  void add_completed_handler(std::function<void()> task) {
    completed_handlers_.push_back(std::move(task));
  }

  bool needs_commit() const {
    return node_count_ > 0;
  }

  // True while eval_compiled_tape is recording a tape node's dispatches.
  // Set around each node's primitive.eval_gpu call; the GPU profiler
  // tags per-dispatch events with it so dispatch counts attribute
  // between the tape and eager paths. Recording is single-threaded per
  // encoder, so a plain bool is safe.
  bool in_tape_recording{false};

  // Nodes recorded in the open batch. The evaluator flushes the batch at
  // kBatchNodeBudget so a long graph cannot pin unbounded temporaries
  // behind one open command buffer.
  int nodes() const {
    return node_count_;
  }

  // True when nothing is recorded, nothing is queued for submission, and
  // no batch buffers are pending stamping: the encoder owns no GPU work
  // and no lifetime obligations. An Event::signal issued on this state
  // moves the target timeline counter from the host instead of
  // submitting a signal-only command buffer; the batch-buffer check is
  // what makes the flush contract explicit — a signal on an encoder that
  // still owes a batch flush takes the queued path, which submits and
  // releases.
  bool idle() const {
    return !recording_ && wait_semaphores_.empty() &&
        signal_semaphores_.empty() && completed_handlers_.empty() &&
        batch_buffers_.empty();
  }

  // Record a device-to-device buffer copy. Both buffers must have
  // VK_BUFFER_USAGE_TRANSFER_* usage bits (all backend buffers do).
  void copy_buffer(
      VkBuffer src,
      VkBuffer dst,
      VkDeviceSize size,
      VkDeviceSize src_offset = 0,
      VkDeviceSize dst_offset = 0);

  // Record one compute dispatch. The binding count must not exceed the
  // device's runtime budget (ComputeRuntime::binding_limit()); kernels
  // needing more refuse by name at the primitive that builds them.
  void dispatch_compute(
      ComputeKernel kernel,
      std::span<const ComputeBinding> bindings,
      const ComputeParams& params,
      uint32_t group_count_x,
      uint32_t group_count_y = 1,
      uint32_t group_count_z = 1);
  void dispatch_compute(
      const std::string& cache_key,
      std::span<const uint32_t> spirv,
      std::span<const ComputeBinding> bindings,
      const ComputeParams& params,
      uint32_t group_count_x,
      uint32_t group_count_y = 1,
      uint32_t group_count_z = 1);

  // Record a four-byte-word fill. Size and offset must be multiples of 4.
  void fill_buffer(
      VkBuffer dst,
      uint32_t value,
      VkDeviceSize size,
      VkDeviceSize offset = 0);

  // Timeline semaphore operations for Event support. Waits apply to this
  // stream's next submission; signals are flushed by commit(). The keepalive
  // token (the owning Event's shared state) must stay held until the
  // submission completes: a queued wait or signal never stores a bare
  // VkSemaphore whose owner could be destroyed first.
  void add_semaphore_wait(
      VkSemaphore semaphore,
      uint64_t value,
      std::shared_ptr<void> keepalive) {
    wait_semaphores_.push_back({semaphore, value, std::move(keepalive)});
  }

  void add_semaphore_signal(
      VkSemaphore semaphore,
      uint64_t value,
      std::shared_ptr<void> keepalive) {
    signal_semaphores_.push_back({semaphore, value, std::move(keepalive)});
  }

  // Submit recorded work, semaphore operations, and completion handlers.
  // Eager: every finalize flushes, one op-granular submission. Deeper
  // batching was measured (2026-09-03) and made generation 4.5x SLOWER:
  // the evaluator throttles at MAX_ACTIVE_TASKS with finalize plus
  // wait_for_one, so any flush granularity coarser than one op converts
  // its pipelined wait into a stop-and-wait proportional to batch
  // execution time — worse with deeper batches and longer models, never
  // better. A future attempt must flush when the scheduler would block,
  // not on a node budget (upstream-shape work). Safe to call repeatedly;
  // a no-op when nothing is pending.
  void commit();

  // Submit pending work and block (bounded) until it completes.
  void synchronize(const char* reason = "explicit");

  // Order this stream's next submission after every submission reserved so
  // far on any stream, without a host join: the ordering rides the next
  // submission as a timeline wait on the device completion semaphore. This
  // is the device-visible equivalent of a drain for GPU work. It cannot
  // order host writes to mapped memory, which still need synchronize().
  void wait_outstanding_submissions();

  // True when every submission queued through this encoder has completed
  // and its handlers have run, and no batch is open with un-submitted
  // work. Host reads of input bytes are then sound.
  bool synchronized() const {
    return node_count_ == 0 &&
        (last_completion_ == 0 ||
         device_.completions().drained_value() >= last_completion_);
  }

  Device& device() {
    return device_;
  }

  uint64_t last_submitted_completion() const {
    return last_completion_;
  }

 private:
  // A semaphore operation pending in this encoder, plus the ownership token
  // that keeps the semaphore's owner alive until GPU completion.
  struct PendingSemaphore {
    VkSemaphore semaphore;
    uint64_t value;
    std::shared_ptr<void> keepalive;
  };

  // One command buffer and one reusable execution-start marker. The event
  // is reset from the host only after this slot's prior completion drained,
  // then set by the device at the head of the next command buffer. It is not
  // destroyed with the encoder because a watchdog error can leave it pending;
  // vkDestroyDevice releases it with the command pool.
  static constexpr int kInFlightCommandBuffers = 4;
  struct Slot {
    VkCommandBuffer cmd{VK_NULL_HANDLE};
    VkEvent started{VK_NULL_HANDLE};
    uint64_t in_flight{0};
  };

  // Descriptor-set pool cache: sets are allocated from a large pool that
  // is created once per kDescriptorSetsPerPool dispatches instead of a
  // pool create/destroy per dispatch. A retired pool is kept alive until
  // the submission recording at retirement time completes, which is after
  // every submission whose batches allocated from it (timeline order).
  static constexpr uint32_t kDescriptorSetsPerPool = 2048;
  VkDescriptorSet acquire_descriptor_set(ComputeRuntime& compute);
  void dispatch_compute_pipeline(
      VkPipeline pipeline,
      ComputeKernel profile_kernel,
      std::span<const ComputeBinding> bindings,
      const ComputeParams& params,
      uint32_t group_count_x,
      uint32_t group_count_y,
      uint32_t group_count_z);

  // Pick a completed ring slot (joining the oldest only when all are in
  // flight) and begin recording into it.
  void ensure_recording();

  // Join this encoder's newest in-flight submission (if any) through the
  // CompletionDispatcher, clear last_completion_, and invalidate
  // noncoherent host mappings. Guarantees the newest command buffer has
  // left the pending state.
  void join_last_completion(const char* reason);
  void submit();

  // Dependency-gated barrier state (MLX_OMARCHY_GATED_BARRIERS, default
  // off). Buffer ranges recorded by the open batch since the last
  // barrier: a node skips its barrier only when neither its reads nor
  // its writes overlap an unsynced range. Dispatch bindings carry no
  // read/write split, so they are tracked as both. head_synced_ is the
  // batch-head dependency: the first node of a freshly begun command
  // buffer always records a barrier so host writes and the allocator's
  // noncoherent flush keep exactly the visibility the unconditional
  // path provides.
  struct TrackedRange {
    VkBuffer buffer;
    VkDeviceSize offset;
    VkDeviceSize end;
  };
  static bool gated_barriers();
  /* Phase-1 dependency-export (design 22b395d): per-dispatch records
   * computed from the GATED_BARRIERS tracker, stored host-side, not
   * consumed yet (the driver learns nothing until phase 2). */
  struct DepRecord {
    uint32_t disjoint;
    uint32_t usc_changed;
    uint64_t seq;
  };
  static bool export_dep_masks();
  static bool dep_dump();
  static int dep_dump_n();
  std::vector<DepRecord> dep_records_;
  uint32_t dep_prev_signature_{0};
  uint32_t dep_have_prev_{0};
  uint64_t dep_seq_{0};
  bool batch_needs_barrier(
      std::span<const TrackedRange> reads,
      std::span<const TrackedRange> writes) const;
  void record_dependency_barrier();
  void reset_dependency_tracking();
  std::vector<TrackedRange> tracked_reads_;
  std::vector<TrackedRange> tracked_writes_;
  bool head_synced_{false};

  Device& device_;
  VkCommandPool pool_{VK_NULL_HANDLE};
  std::array<Slot, kInFlightCommandBuffers> slots_{};
  int current_slot_{0};
  VkCommandBuffer cmd_{VK_NULL_HANDLE};
  bool recording_{false};
  int node_count_{0};
  uint64_t last_completion_{0};
  VkDescriptorPool desc_pool_{VK_NULL_HANDLE};
  uint32_t desc_pool_remaining_{0};
  std::vector<PendingSemaphore> wait_semaphores_;
  std::vector<PendingSemaphore> signal_semaphores_;
  // Raw buffers referenced by the open batch. Owned by their arrays or,
  // after free(), by the allocator quarantine; this list only carries
  // the completion stamp at submit time.
  std::vector<VulkanBuffer*> batch_buffers_;
  // Descriptor pools retired during recording. They ride the dispatcher
  // payload (one-generation retire) so a set allocated from them cannot
  // outlive its pool.
  std::vector<std::shared_ptr<void>> retired_pools_;
  std::vector<std::function<void()>> completed_handlers_;
};

MLX_API CommandEncoder& get_command_encoder(Stream s);
std::unordered_map<int, CommandEncoder>& get_command_encoders();
std::unordered_map<int, CommandEncoder>& get_global_command_encoders();

} // namespace mlx::core::omarchy
