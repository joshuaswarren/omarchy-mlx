// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

#pragma once
#include <cstdlib>

#include <vulkan/vulkan.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>
#include "mlx/array.h"
#include "mlx/backend/omarchy/allocator.h"
#include "mlx/backend/omarchy/compute.h"
#include "mlx/backend/omarchy/device.h"
#include "mlx/stream.h"

namespace mlx::core::omarchy {

// Storage-buffer binding for |value|: its buffer from byte 0 (kernels
// add the array's element offset themselves) through the last byte its
// data reaches, rounded up to 32-bit word granularity. Not the
// allocation size: allocator round_size bins every request above 1 MiB
// to one of 8 sizes per power of two (up to 12.5% above the request),
// so binding the whole allocation could reach past the device's
// maxStorageBufferRange for an array just under it
// (dispatch_compute_pipeline refuses those). The round exists because
// the byte-lane kernels address whole words: a take over a 5-byte uint8
// table reads input word (in_elem >> 2) and a 3-byte output merges its
// last byte through atomicOr into word (out_elem >> 2), so a logical
// 3-byte range would have the shader touch a byte outside the bound
// range. BYTE_AT extracts only lanes below the array's element count
// and unclaimed output lanes stay at the host pre-zero fill, so the
// widened window keeps results bit-exact. Allocations are page-rounded
// (round_size), so the rounded range stays inside the backing buffer;
// the clamp below keeps that true regardless.
inline ComputeBinding binding(const array& value) {
  auto* buffer = static_cast<const VulkanBuffer*>(value.buffer().ptr());
  const VkDeviceSize end = static_cast<VkDeviceSize>(value.offset()) +
      static_cast<VkDeviceSize>(value.data_size()) * value.itemsize();
  const VkDeviceSize word_end = (end + 3) & ~static_cast<VkDeviceSize>(3);
  // A zero range is invalid Vulkan; an empty array keeps one byte.
  return {
      buffer->buffer,
      0,
      std::min<VkDeviceSize>(std::max<VkDeviceSize>(word_end, 1),
                             buffer->size),
      buffer};
}

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
// MLX_OMARCHY_BATCH_NODES=<n> overrides the node budget (default
// kBatchNodeBudget). A smaller budget submits the first nodes of a long graph
// while the host is still recording the rest, overlapping host record time
// with GPU execution; scheduling only, results are unchanged.
// MLX_OMARCHY_BATCH_FIRST=<n>: the FIRST batch of every graph evaluation is
// submitted after n nodes (default 0 = off, uses batch_node_budget()), the
// rest of the graph then rides the normal budget. The GPU starts on the head
// of a long graph while the host still records the tail (host record is
// ~45 us per dispatch on Linux and is otherwise fully exposed in synchronous
// paths), at the cost of one extra submit per graph. Scheduling only.
inline int batch_first_budget() {
  // Read on every call (once per graph start): mlx-lm's generate_step sets
  // the variable around prompt processing and clears it after the first token.
  const char* e = std::getenv("MLX_OMARCHY_BATCH_FIRST");
  int n = e ? std::atoi(e) : 0;
  return n > 0 ? n : 0;
}
inline int batch_node_budget() {
  static const int v = []() {
    const char* e = std::getenv("MLX_OMARCHY_BATCH_NODES");
    int n = e ? std::atoi(e) : 0;
    return n > 0 ? n : kBatchNodeBudget;
  }();
  return v;
}
// Estimated GPU time per submission, in dispatch work-groups. A queue
// submission monopolizes the GPU until it finishes: the compositor gets
// the queue only between submissions, so one long submission makes the
// desktop hitch for its whole length (issue #19: ~78 ms submissions
// stutter, 2-6 ms stay smooth, same GPU busy fraction). Dispatch
// durations are not cheaply measurable at record time, so the evaluator
// flushes the open batch once its summed work-group counts reach this
// budget. The default is calibrated so a 4B decode step splits into
// ~2-6 ms submissions on the calibration host (measured groups per token
// and per-group cost in receipts/2026-10-02-submission-cap-19). A single
// dispatch whose group count alone exceeds the budget still runs whole:
// splitting happens between dispatches. Copies and fills are not
// counted; the node and byte budgets keep those batches bounded.
// MLX_OMARCHY_BATCH_WORK=<groups> overrides; =0 turns the work cap off
// (node and byte budgets still apply). Scheduling only: every submission
// already waits on this stream's previous completion and cross-submission
// waits use ALL_COMMANDS stage masks (batching note above), so splitting
// at any dispatch boundary preserves every dependency and results stay
// bit-identical.
inline constexpr uint64_t kBatchWorkBudget = 40000;
inline uint64_t batch_work_budget() {
  static const uint64_t v = []() {
    const char* e = std::getenv("MLX_OMARCHY_BATCH_WORK");
    if (!e) {
      return kBatchWorkBudget;
    }
    long long n = std::atoll(e);
    return n > 0 ? static_cast<uint64_t>(n) : 0;
  }();
  return v;
}
// The one flush predicate for the open batch, shared by the eager
// evaluator (eval.cpp) and the compiled-tape recorder (compiled.cpp) so
// both close batches at the same points. omarchy_runtime_tests drives it
// deterministically with synthetic dispatch sequences and asserts the
// number of submissions and their boundaries; no GPU involved.
inline bool batch_over_budget(
    int nodes,
    uint64_t work,
    int node_budget,
    uint64_t work_budget) {
  return nodes >= node_budget || (work_budget > 0 && work >= work_budget);
}
// Byte budget for the same batch: freed intermediates stay pinned in the
// allocator quarantine until their batch submits and drains, so the open
// batch may hold at most 1/16 of the allocator memory limit in such bytes
// before the evaluator flushes it. Up to five generations can be pinned at
// once (four ring slots in flight plus the one-generation-late quarantine
// release), so the bound keeps batching-owed memory under a third of the
// limit. A 2,048-token Qwen2.5-0.5B forward held 8.11 GB in one 257-node
// batch against Honeykrisp's 7.56 GiB heap without this (2026-09-08).
inline constexpr size_t kBatchByteBudgetDivisor = 16;

// One tracked buffer byte-range of the dependency tracker (start
// saturates at the address space; see tracked_range_end).
struct TrackedRange {
  VkBuffer buffer;
  VkDeviceSize offset;
  VkDeviceSize end;
};

// One buffered node's access sets for wave scheduling: the byte ranges a
// node reads and writes (SPIR-V reflected readonly/writeonly split).
struct WaveNode {
  std::span<const TrackedRange> reads;
  std::span<const TrackedRange> writes;
};

// Greedy earliest-wave schedule for one batch: level[i] is the smallest
// wave index such that every earlier tape node that node i hazards sits
// in a strictly smaller wave. A hazard is RAW (i reads a range an earlier
// node writes), WAW, or WAR (i writes a range an earlier node reads) over
// the exact tracked byte ranges. Nodes of equal level are pairwise
// hazard-free, so a stream that emits waves in tape order with one full
// dependency barrier between waves preserves every dependency of the tape
// order while letting hazard-free nodes run concurrently. Public so tests
// can assert the schedule contract without a device.
MLX_API std::vector<uint32_t> wave_levels(std::span<const WaveNode> nodes);

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

  // Summed dispatch work-group counts of the open batch: the bounded
  // GPU-time proxy the work budget flushes on (batch_work_budget).
  uint64_t batch_work() const {
    return batch_work_;
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
    return !recording_ && pending_.empty() && wait_semaphores_.empty() &&
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
  // ON since 2026-09-29; set 0 to restore the unconditional pre+post
  // barriers). Buffer ranges recorded by the open batch since the last
  // barrier: a node skips its barrier only when neither its reads nor
  // its writes overlap an unsynced range. Dispatch bindings carry the
  // SPIR-V reflected read/write split, so read-read pairs skip.
  // head_synced_ is the batch-head dependency: the first node of a
  // freshly begun command buffer always records a barrier so host writes
  // and the allocator's noncoherent flush keep exactly the visibility
  // the unconditional path provides.
  static bool gated_barriers();

  // Wave scheduling (MLX_OMARCHY_WAVE_SCHED, default off; requires gated
  // barriers). Arrival-ordered nodes buffered by the open batch instead
  // of recording straight into the command buffer; submit() records them
  // in wave_levels() order with one full dependency barrier between
  // waves. A hazard-free node therefore runs concurrently with its wave
  // instead of serializing behind a barrier its tape neighbors induced.
  struct PendingNode {
    enum class Kind : uint8_t { Dispatch, Copy, Fill };
    Kind kind{Kind::Dispatch};
    // Profiler attribution, captured at arrival (emit happens later, at
    // submit): the evaluating primitive's name and the tape flag.
    std::string_view prim{};
    uint32_t tape{0};
    uint64_t host_t0{0};
    // Dispatch payload:
    VkPipeline pipeline{VK_NULL_HANDLE};
    ComputeKernel profile_kernel{};
    std::array<ComputeBinding, kComputeBindingBudget> bindings{};
    uint32_t binding_count{0};
    ComputeParams params{};
    uint32_t group_count_x{1};
    uint32_t group_count_y{1};
    uint32_t group_count_z{1};
    VkDescriptorSet descriptor_set{VK_NULL_HANDLE};
    // Copy/fill payload:
    VkBuffer src{VK_NULL_HANDLE};
    VkBuffer dst{VK_NULL_HANDLE};
    VkDeviceSize size{0};
    VkDeviceSize src_offset{0};
    VkDeviceSize dst_offset{0};
    VkDeviceSize write_offset{0};
    uint32_t value{0};
    // Hazard access sets (exact tracked ranges):
    std::array<TrackedRange, kComputeBindingBudget> rd{};
    std::array<TrackedRange, kComputeBindingBudget> wr{};
    uint32_t nr{0};
    uint32_t nw{0};
  };
  std::vector<PendingNode> pending_;
  static bool wave_sched();
  static bool wave_diag();
  // Record the buffered nodes in wave order (wave_levels), one full
  // dependency barrier between waves plus the batch-head barrier, and
  // update the tracker exactly like the tape-order path. Called from
  // submit() before the command buffer is closed.
  void emit_pending();

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
  uint64_t batch_work_{0};
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
