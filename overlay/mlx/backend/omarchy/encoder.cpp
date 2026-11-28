// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

#include "mlx/backend/omarchy/encoder.h"
#include <algorithm>
#include <map>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string_view>
#include <unistd.h>
#include <sys/syscall.h>

#include "mlx/backend/omarchy/allocator.h"
#include "mlx/backend/omarchy/cpu_pd_hold.h"
#include "mlx/backend/omarchy/cpu_uclamp.h"
#include "mlx/backend/omarchy/device.h"
#include "mlx/backend/omarchy/step_timeline.h"
#include "mlx/backend/omarchy/trace.h"
#include "mlx/backend/omarchy/gpu_profiler.h"
#include "kernel_names.h"
#include "mlx/backend/omarchy/vulkan.h"
#include "mlx/scheduler.h"
#include "mlx/utils.h"

namespace mlx::core::omarchy {

namespace {

// Saturating byte-range end for dependency tracking. VK_WHOLE_SIZE and
// overflowing sizes clamp to the address space so a tracked range never
// under-covers the accesses it stands for.
inline VkDeviceSize tracked_range_end(VkDeviceSize offset, VkDeviceSize size) {
  if (size == VK_WHOLE_SIZE) {
    return UINT64_MAX;
  }
  VkDeviceSize end = offset + size;
  return end < offset ? UINT64_MAX : end;
}

// The byte window a binding contributes to the dependency tracker: the
// view's own span when binding(array) narrowed it, else the bound
// descriptor range.
inline TrackedRange binding_tracked_range(const ComputeBinding& b) {
  if (b.tracked_end != 0) {
    return {b.buffer, b.tracked_offset, b.tracked_end};
  }
  return {b.buffer, b.offset, tracked_range_end(b.offset, b.range)};
}

} // namespace

void wave_levels_into(
    std::span<const WaveNode> nodes,
    WaveIndex& index,
    std::vector<uint32_t>& levels) {
  // Earlier reads and writes of each buffer, one entry per exact byte range
  // with the highest level that touched it. A node takes 1 + the highest
  // level among overlapping earlier writes (RAW, WAW) and, for its writes,
  // overlapping earlier reads (WAR). Read-read pairs never hazard. Many
  // nodes that read one range, or write disjoint slices of one buffer, keep
  // each query short, so the pass stays near linear in the batch length.
  index.reset();
  levels.clear();
  levels.resize(nodes.size(), 0);
  auto highest = [](const std::vector<WaveRangeLevels>& v,
                    const BufferSlots& slots,
                    const TrackedRange& q) -> int64_t {
    // One slot table serves both indexes: a slot another index assigned
    // names a buffer this index has no entries for.
    const uint32_t s = slots.find(q.buffer);
    return s == UINT32_MAX || s >= v.size() ? -1 : v[s].max_overlap(q);
  };
  auto entry = [](std::vector<WaveRangeLevels>& v,
                  uint32_t s) -> WaveRangeLevels& {
    if (s >= v.size()) {
      v.resize(s + 1);
    }
    return v[s];
  };
  for (size_t i = 0; i < nodes.size(); ++i) {
    int64_t source = -1;
    for (const auto& r : nodes[i].reads) {
      source = std::max(source, highest(index.writers, index.slots, r));
    }
    for (const auto& w : nodes[i].writes) {
      source = std::max(
          {source,
           highest(index.writers, index.slots, w),
           highest(index.readers, index.slots, w)});
    }
    levels[i] = static_cast<uint32_t>(source + 1);
    for (const auto& r : nodes[i].reads) {
      entry(index.readers, index.slots.slot_of(r.buffer)).add(r, levels[i]);
    }
    for (const auto& w : nodes[i].writes) {
      entry(index.writers, index.slots.slot_of(w.buffer)).add(w, levels[i]);
    }
  }
}

std::vector<uint32_t> wave_levels(std::span<const WaveNode> nodes) {
  WaveIndex index;
  std::vector<uint32_t> levels;
  wave_levels_into(nodes, index, levels);
  return levels;
}

bool CommandEncoder::wave_sched() {
  // MLX_OMARCHY_WAVE_SCHED (docs/install-omarchy.md): unset means the
  // per-part default (wave_sched_default_part, device.h), 1 forces it on and
  // 0 off. When on (and only with gated barriers — wave mode rides the same
  // SPIR-V access reflection and tracker), dispatch/copy/fill nodes buffer at arrival
  // and submit() records the batch in greedy earliest-wave order with one
  // full dependency barrier between waves. Reordering is numerics-inert:
  // a node's inputs are final once its hazard sources complete, and every
  // hazard source of a node sits in a strictly earlier wave.
  static const bool on = []() {
    const char* e = std::getenv("MLX_OMARCHY_WAVE_SCHED");
    const bool want = e != nullptr
        ? env_flag("MLX_OMARCHY_WAVE_SCHED")
        : wave_sched_default_part(
              omarchy::device().capabilities().device_name);
    return want && gated_barriers();
  }();
  return on;
}

bool CommandEncoder::wave_diag() {
  // MLX_OMARCHY_WAVE_DIAG (census diagnostic): per flush, classify the
  // edge that determined each node's wave (RAW/WAW/WAR against the
  // max-level hazard source) and print a compact histogram. Counts only;
  // inert for recording.
  static const bool on = []() {
    const char* e = std::getenv("MLX_OMARCHY_WAVE_DIAG");
    return e != nullptr && env_flag("MLX_OMARCHY_WAVE_DIAG");
  }();
  return on;
}

bool CommandEncoder::export_dep_masks() {
  // MLX_OMARCHY_EXPORT_DEP_MASKS (design 22b395d phase 1): default ON
  // for the phase-1 branch — the computation reuses tracker state and
  // is provably inert (records are stored, never consumed).
  static const bool on = !env_flag("MLX_OMARCHY_NO_EXPORT_DEP_MASKS");
  return on;
}
bool CommandEncoder::dep_dump() {
  static const int n = []() {
    const char *v = getenv("MLX_OMARCHY_DEP_DUMP");
    return v ? atoi(v) : 0;
  }();
  return n > 0;
}
int CommandEncoder::dep_dump_n() {
  static const int n = []() {
    const char *v = getenv("MLX_OMARCHY_DEP_DUMP");
    return v ? atoi(v) : 0;
  }();
  return n;
}
bool CommandEncoder::gated_barriers() {
  // MLX_OMARCHY_GATED_BARRIERS (docs/install-omarchy.md): default ON since
  // 2026-09-29 (set 0 to restore the unconditional pre+post barriers).
  // Dispatch/copy/fill nodes record a barrier only when their buffer
  // ranges overlap unsynced work of the open batch. Read once: the gate
  // shapes recorded commands, so flipping it mid-batch would desync the
  // tracker from the command buffer.
  static const bool on = []() {
    return std::getenv("MLX_OMARCHY_GATED_BARRIERS") == nullptr ||
        env_flag("MLX_OMARCHY_GATED_BARRIERS");
  }();
  return on;
}

void CommandEncoder::record_dependency_barrier() {
  // The heaviest correct dependency: all commands, all memory access,
  // both directions. The tracker restarts after it because the barrier
  // orders everything recorded before it.
  VkMemoryBarrier full{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
  full.srcAccessMask =
      VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
  full.dstAccessMask =
      VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
  vk::device_table().CmdPipelineBarrier(
      cmd_,
      VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
      VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
      0,
      1,
      &full,
      0,
      nullptr,
      0,
      nullptr);
  tracker_.clear();
  head_synced_ = true;
  transfer_since_barrier_ = false;
}

bool CommandEncoder::compute_barrier() {
  // A dispatch whose hazard is only earlier dispatches gets a compute-to-
  // compute barrier instead of the full one (select_compute_barrier: default
  // on for discrete GPUs, MLX_OMARCHY_COMPUTE_BARRIER=0/1 overrides). The
  // batch head or a copy, fill or inline update since the last full barrier
  // keeps the full barrier.
  static const bool on = []() {
    const int setting = std::getenv("MLX_OMARCHY_COMPUTE_BARRIER") == nullptr
        ? -1
        : (env_flag("MLX_OMARCHY_COMPUTE_BARRIER") ? 1 : 0);
    return select_compute_barrier(omarchy::device().physical_type(), setting);
  }();
  return on;
}

void CommandEncoder::record_dispatch_barrier() {
  if (!compute_barrier() || !head_synced_ || transfer_since_barrier_) {
    record_dependency_barrier();
    return;
  }
  VkMemoryBarrier compute{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
  compute.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
  compute.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
  vk::device_table().CmdPipelineBarrier(
      cmd_,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      0,
      1,
      &compute,
      0,
      nullptr,
      0,
      nullptr);
  tracker_.clear();
  head_synced_ = true;
}

void CommandEncoder::reset_dependency_tracking() {
  tracker_.clear();
  head_synced_ = false;
  boundary_forced_ = false;
}

CommandEncoder::CommandEncoder(Device& device) : device_(device) {
  static std::atomic<uint16_t> next_encoder_id{1};
  encoder_id_ = next_encoder_id.fetch_add(1, std::memory_order_relaxed);
  auto& dt = vk::device_table();
  VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
  pci.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT |
      VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  pci.queueFamilyIndex = device_.queue_family();
  VKX_CHECK(dt.CreateCommandPool(device_.handle(), &pci, nullptr, &pool_));

  std::array<VkCommandBuffer, kInFlightCommandBuffers> buffers{};
  VkCommandBufferAllocateInfo ai{
      VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  ai.commandPool = pool_;
  ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  ai.commandBufferCount = kInFlightCommandBuffers;
  VKX_CHECK(dt.AllocateCommandBuffers(device_.handle(), &ai, buffers.data()));
  VkEventCreateInfo event_info{VK_STRUCTURE_TYPE_EVENT_CREATE_INFO};
  for (int i = 0; i < kInFlightCommandBuffers; ++i) {
    slots_[i].cmd = buffers[i];
    VKX_CHECK(dt.CreateEvent(
        device_.handle(), &event_info, nullptr, &slots_[i].started));
  }
  prof::get().attach(this, device_);
}

// Encoders that owe a deferred event signal, process-wide. get_command_
// encoders() is thread-local, so a wait on another thread cannot reach
// the producer's encoder through it; this map is the cross-thread route.
// Weak ownership: the flush may outlive the producer thread, and a raw
// pointer dereferenced after the registry lock drops is a UAF — the
// weak_ptr expires with the encoder instead.
namespace {
std::mutex& deferred_signal_registry_mutex() {
  static std::mutex m;
  return m;
}
std::unordered_map<CommandEncoder*, std::weak_ptr<CommandEncoder>>&
deferred_signal_registry() {
  static std::unordered_map<CommandEncoder*, std::weak_ptr<CommandEncoder>>
      encoders;
  return encoders;
}
} // namespace

void CommandEncoder::note_deferred_signals() {
  if (!deferred_signals_pending_.exchange(
          true, std::memory_order_acq_rel)) {
    std::lock_guard<std::mutex> lk(deferred_signal_registry_mutex());
    deferred_signal_registry()[this] = weak_from_this();
  }
}

CommandEncoder::~CommandEncoder() {
  // Drain pending work (and its temporaries) while the device is alive.
  // A wedged queue throws here; swallow so destruction can continue, the
  // bounded error already surfaced through synchronize().
  //
  // The command and descriptor pools are deliberately NOT destroyed
  // here. After a watchdog throw the newest submission is still in
  // flight, and even after a successful join Mesa signals a submission's
  // semaphores before its submit-final cleanup retires the command
  // buffer (see drain_through), so a pool destroy in this destructor can
  // free state the driver's queue thread still walks - observed as
  // teardown SIGSEGV, pure-virtual aborts, and Khronos-validation-layer
  // crashes after a watchdog throw. The pools are children of the
  // VkDevice; vkDestroyDevice releases them with it.
  if (!runtime_alive()) {
    return;
  }
  try {
    synchronize();
    if (resnorm_ticket_buf_.ptr()) {
      omarchy::allocator().free(resnorm_ticket_buf_);
      resnorm_ticket_buf_ = mlx::core::allocator::Buffer{nullptr};
    }
  } catch (const std::exception&) {
  }
}

// Join this encoder's newest in-flight submission, if any, and refresh
// host mappings. After this, every command buffer this encoder submitted
// has left the pending state (the completion timeline is strictly
// ordered), so they can legally be begun again, and host reads see the
// submissions' final bytes.
void CommandEncoder::join_last_completion(const char* reason) {
  // Read the join target under op_mutex_: a foreign deferred-signal
  // commit can publish a newer completion while this thread sleeps in
  // the wait below, and the zero afterwards must not clobber it.
  uint64_t value;
  {
    std::lock_guard<std::recursive_mutex> op_lk(op_mutex_);
    value = last_completion_;
  }
  if (env_set("MLX_OMARCHY_TRACE_DISPATCH")) {
    fprintf(stderr, "[rtmod] JOIN tid=%lu reason=%s last=%lu\n", (unsigned long)syscall(SYS_gettid), reason,
            (unsigned long)value);
  }
  if (value == 0) {
    return;
  }
  // A queued deferred signal may be the only producer left that can
  // satisfy submissions this join (or a wait inside one of them) blocks
  // on; flush before blocking so the host wait always terminates.
  flush_deferred_signal_encoders();
  timeline::ScopedEvent tl_join(
      timeline::Kind::Readback,
      timeline::SubKind::SyncJoin,
      encoder_id_,
      value);
  uint64_t join_t0 = prof::get().profiling() ? prof::host_ns() : 0;
  device_.completions().wait(value);
  uint64_t wait_t1 = prof::get().profiling() ? prof::host_ns() : 0;
  {
    std::lock_guard<std::recursive_mutex> op_lk(op_mutex_);
    if (last_completion_ == value) {
      last_completion_ = 0;
    }
  }
  {
    timeline::ScopedEvent tl_inv(
        timeline::Kind::Readback,
        timeline::SubKind::InvalidateNoncoherent,
        encoder_id_,
        value);
    omarchy::allocator().invalidate_noncoherent(device_.handle());
  }
  uint64_t inval_t2 = prof::get().profiling() ? prof::host_ns() : 0;
  prof::get().on_join(this, value, join_t0, wait_t1, inval_t2, reason);
}

void CommandEncoder::ensure_recording() {
  if (recording_) {
    return;
  }
  uint64_t begin_t0 = prof::get().profiling() ? prof::host_ns() : 0;
  // Acquire a ring slot whose submission has completed. Newer submissions
  // keep executing on the device while this batch records; only when all
  // slots are in flight does the host join the oldest. The profiler's
  // begin cost includes this wait: it is the residual host-side stall the
  // 2026-09-02 profile attributed to per-record joins.
  auto& completions = device_.completions();
  int chosen = -1;
  uint64_t oldest_value = UINT64_MAX;
  int oldest = -1;
  for (int i = 0; i < kInFlightCommandBuffers; ++i) {
    uint64_t value = slots_[i].in_flight;
    if (value == 0 || completions.drained_value() >= value) {
      slots_[i].in_flight = 0;
      chosen = i;
      break;
    }
    if (value < oldest_value) {
      oldest_value = value;
      oldest = i;
    }
  }
  if (chosen < 0) {
    {
      timeline::ScopedEvent tl_slot(
          timeline::Kind::CmdBegin,
          timeline::SubKind::SlotWait,
          encoder_id_,
          oldest_value);
      completions.wait(oldest_value);
    }
    slots_[oldest].in_flight = 0;
    chosen = oldest;
  }
  current_slot_ = chosen;
  cmd_ = slots_[chosen].cmd;
  bound_pipeline_ = VK_NULL_HANDLE;
  {
    timeline::ScopedEvent tl_begin(
        timeline::Kind::CmdBegin,
        timeline::SubKind::Begin,
        encoder_id_,
        static_cast<uint64_t>(chosen));
    completions.reset_progress_event(slots_[chosen].started);
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VKX_CHECK(vk::device_table().BeginCommandBuffer(cmd_, &bi));
    vk::device_table().CmdSetEvent(
        cmd_, slots_[chosen].started, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT);
  }
  prof::get().on_begin(
      this, chosen, cmd_, begin_t0 != 0 ? prof::host_ns() - begin_t0 : 0);
  recording_ = true;
}

void CommandEncoder::upload_device_local_if_needed(
    VulkanBuffer* vb,
    bool read,
    bool write,
    VkDeviceSize offset,
    VkDeviceSize range,
    bool exact_write_coverage) {
    std::lock_guard<std::recursive_mutex> op_lk(op_mutex_);

  if (!vb || !vb->device_local) {
    return;
  }
  const size_t eff_range = range == VK_WHOLE_SIZE
      ? (vb->size > static_cast<size_t>(offset)
             ? vb->size - static_cast<size_t>(offset)
             : 0)
      : static_cast<size_t>(range);
  if (!vb->sync_state.needs_upload(
          read,
          write,
          static_cast<size_t>(offset),
          eff_range,
          vb->size,
          exact_write_coverage)) {
    return;
  }
  if (vb->host_shadow != nullptr) {
    auto& dt = vk::device_table();
    if (can_inline_update_device_local(vb->size) &&
        dt.CmdUpdateBuffer != nullptr) {
      uint64_t comp = omarchy::allocator().buffer_completion(vb);
      if (comp == kPendingCompletion) {
        // Same-stream pending work is already recorded in this encoder, so
        // program order plus the barrier tracker/wave scheduler orders the
        // inline vkCmdUpdateBuffer after it without flushing. Only flush if
        // another encoder holds the uncommitted use of vb.
        const bool on_this_encoder = batch_buffers_.count(vb) != 0;
        if (!on_this_encoder) {
          synchronize_pending_encoders();
          comp = omarchy::allocator().buffer_completion(vb);
        }
      }
      if (needs_inline_update_semaphore_wait(
              comp,
              kPendingCompletion,
              last_completion_,
              device_.completions().drained_value())) {
        add_semaphore_wait(device_.completions().semaphore(), comp, nullptr);
      }
      if (auto* noted = omarchy::allocator().note_buffer_handle(vb->buffer)) {
        batch_buffers_.insert(noted);
      }
      if (wave_sched()) {
        PendingNode& node = pending_.emplace_back();
        node.kind = PendingNode::Kind::Update;
        node.prim = trace::current_prim();
        node.dst = vb->buffer;
        node.size = vb->size;
        node.write_offset = 0;
        const auto* src = static_cast<const uint8_t*>(vb->host_shadow);
        node.update_data.assign(src, src + vb->size);
        node.wr[node.nw++] = {vb->buffer, 0, vb->size};
      } else {
        ensure_recording();
        if (tape_full_barriers()) {
          record_dependency_barrier();
        }
        if (gated_barriers()) {
          TrackedRange wr{vb->buffer, 0, vb->size};
          if (boundary_decision(
                  tracker_, {}, {&wr, 1},
                  !head_synced_ && boundary_forced_) ==
              BoundaryDecision::Barrier) {
            record_dependency_barrier();
            trace::counters().barriers_emitted++;
            prof::get().on_barrier(true);
          } else {
            trace::counters().barriers_skipped++;
            prof::get().on_barrier(false);
          }
          head_synced_ = true;
          boundary_forced_ = false;
          tracker_.add_write(wr);
        }
        dt.CmdUpdateBuffer(cmd_, vb->buffer, 0, vb->size, vb->host_shadow);
        transfer_since_barrier_ = true;
        if (!gated_barriers()) {
          record_dependency_barrier();
        }
      }
    } else {
      uint64_t comp = omarchy::allocator().buffer_completion(vb);
      if (comp == kPendingCompletion) {
        synchronize("vram host upload");
        comp = omarchy::allocator().buffer_completion(vb);
      }
      if (comp != 0 && comp != kPendingCompletion) {
        device_.completions().wait(comp);
      }
      device_.upload_host_to_device(vb->buffer, vb->host_shadow, vb->size);
    }
  }
  vb->sync_state.commit_gpu_upload();
}

void CommandEncoder::commit_device_local_write(VulkanBuffer* vb) {
  if (vb && vb->device_local) {
    vb->sync_state.commit_gpu_write();
  }
}

void CommandEncoder::copy_buffer(
    VkBuffer src,
    VkBuffer dst,
    VkDeviceSize size,
    VkDeviceSize src_offset,
    VkDeviceSize dst_offset) {
    std::lock_guard<std::recursive_mutex> op_lk(op_mutex_);

  auto* src_vb = omarchy::allocator().lookup_buffer_handle(src);
  auto* dst_vb = omarchy::allocator().lookup_buffer_handle(dst);
  if (src_vb && src_vb == dst_vb) {
    upload_device_local_if_needed(src_vb, true, true, dst_offset, size, true);
    commit_device_local_write(src_vb);
  } else {
    upload_device_local_if_needed(src_vb, true, false, src_offset, size, false);
    upload_device_local_if_needed(dst_vb, false, true, dst_offset, size, true);
    commit_device_local_write(dst_vb);
  }
  if (auto* vb = omarchy::allocator().note_buffer_handle(src)) {
    batch_buffers_.insert(vb);
  }
  if (auto* vb = omarchy::allocator().note_buffer_handle(dst)) {
    batch_buffers_.insert(vb);
  }
  if (wave_sched()) {
    PendingNode& node = pending_.emplace_back();
    node.kind = PendingNode::Kind::Copy;
    node.prim = trace::current_prim();
    node.src = src;
    node.dst = dst;
    node.size = size;
    node.src_offset = src_offset;
    node.dst_offset = dst_offset;
    node.rd[node.nr++] = {src, src_offset, tracked_range_end(src_offset, size)};
    node.wr[node.nw++] = {dst, dst_offset, tracked_range_end(dst_offset, size)};
    node_count_++;
    trace::counters().vk_buffer_copies++;
    return;
  }
  ensure_recording();
  // The tape-full diagnostic forces the heaviest dependency around the
  // copy and restarts tracking either way.
  if (tape_full_barriers()) {
    record_dependency_barrier();
  }
  if (gated_barriers()) {
    TrackedRange read{src, src_offset, tracked_range_end(src_offset, size)};
    TrackedRange write{dst, dst_offset, tracked_range_end(dst_offset, size)};
    if (boundary_decision(
            tracker_, {&read, 1}, {&write, 1},
            !head_synced_ && boundary_forced_) ==
        BoundaryDecision::Barrier) {
      record_dependency_barrier();
      trace::counters().barriers_emitted++;
      prof::get().on_barrier(true);
    } else {
      trace::counters().barriers_skipped++;
      prof::get().on_barrier(false);
    }
    head_synced_ = true;
    boundary_forced_ = false;
    tracker_.add_read(read);
    tracker_.add_write(write);
  }
  VkBufferCopy region{};
  region.srcOffset = src_offset;
  region.dstOffset = dst_offset;
  region.size = size;
  vk::device_table().CmdCopyBuffer(cmd_, src, dst, 1, &region);
  transfer_since_barrier_ = true;
  node_count_++;
  trace::counters().vk_buffer_copies++;
}

void CommandEncoder::fill_buffer(
    VkBuffer dst,
    uint32_t value,
    VkDeviceSize size,
    VkDeviceSize offset) {
    std::lock_guard<std::recursive_mutex> op_lk(op_mutex_);

  auto* dst_vb = omarchy::allocator().lookup_buffer_handle(dst);
  upload_device_local_if_needed(dst_vb, false, true, offset, size, true);
  commit_device_local_write(dst_vb);
  if (auto* vb = omarchy::allocator().note_buffer_handle(dst)) {
    batch_buffers_.insert(vb);
  }
  if (wave_sched()) {
    PendingNode& node = pending_.emplace_back();
    node.kind = PendingNode::Kind::Fill;
    node.prim = trace::current_prim();
    node.dst = dst;
    node.size = size;
    node.write_offset = offset;
    node.value = value;
    node.wr[node.nw++] = {dst, offset, tracked_range_end(offset, size)};
    node_count_++;
    trace::counters().vk_buffer_fills++;
    return;
  }
  ensure_recording();
  if (tape_full_barriers()) {
    record_dependency_barrier();
  }
  if (gated_barriers()) {
    TrackedRange write{dst, offset, tracked_range_end(offset, size)};
    if (boundary_decision(
            tracker_, {}, {&write, 1},
            !head_synced_ && boundary_forced_) ==
        BoundaryDecision::Barrier) {
      record_dependency_barrier();
      trace::counters().barriers_emitted++;
      prof::get().on_barrier(true);
    } else {
      trace::counters().barriers_skipped++;
      prof::get().on_barrier(false);
    }
    head_synced_ = true;
    boundary_forced_ = false;
    tracker_.add_write(write);
  }
  vk::device_table().CmdFillBuffer(cmd_, dst, offset, size, value);
  transfer_since_barrier_ = true;
  node_count_++;
  trace::counters().vk_buffer_fills++;
}

ComputeBinding CommandEncoder::resnorm_ticket_binding() {
    std::lock_guard<std::recursive_mutex> op_lk(op_mutex_);

  if (!resnorm_ticket_buf_.ptr()) {
    resnorm_ticket_buf_ = omarchy::allocator().malloc(
        kInFlightCommandBuffers * kResNormTicketStride);
    auto* vb = static_cast<VulkanBuffer*>(resnorm_ticket_buf_.ptr());
    fill_buffer(
        vb->buffer, 0u, kInFlightCommandBuffers * kResNormTicketStride, 0u);
  }
  auto* vb = static_cast<VulkanBuffer*>(resnorm_ticket_buf_.ptr());
  if (resnorm_ticket_slot_ < 0) {
    auto& completions = device_.completions();
    int chosen = -1;
    uint64_t oldest_value = UINT64_MAX;
    int oldest = 0;
    for (int i = 0; i < kInFlightCommandBuffers; ++i) {
      uint64_t value = resnorm_ticket_in_flight_[i];
      if (value == 0 || completions.drained_value() >= value) {
        resnorm_ticket_in_flight_[i] = 0;
        chosen = i;
        break;
      }
      if (value < oldest_value) {
        oldest_value = value;
        oldest = i;
      }
    }
    if (chosen < 0) {
      completions.wait(oldest_value);
      resnorm_ticket_in_flight_[oldest] = 0;
      chosen = oldest;
    }
    resnorm_ticket_slot_ = chosen;
  }
  return ComputeBinding{
      vb->buffer,
      static_cast<VkDeviceSize>(resnorm_ticket_slot_) * kResNormTicketStride,
      kResNormTicketStride,
      vb};
}

// Allocate one descriptor set from the cached pool. A pool serves up to
// kDescriptorSetsPerPool dispatches; on exhaustion it is retired into the
// currently-recording submission's temporaries, so it is destroyed only
// after that submission completes — which is strictly after every earlier
// submission whose batches allocated sets from it (completion timeline
// values increase along the queue).
VkDescriptorSet CommandEncoder::acquire_descriptor_set(
    ComputeRuntime& compute) {
    std::lock_guard<std::recursive_mutex> op_lk(op_mutex_);

  timeline::ScopedEvent tl_desc(
      timeline::Kind::DescUpdate, timeline::SubKind::DescAlloc, encoder_id_);
  auto& dt = vk::device_table();
  // MLX_OMARCHY_TAPE_NO_REUSE (diagnostic, docs/install-omarchy.md):
  // every dispatch gets its own descriptor pool with exactly one set, so
  // no pool - and therefore no set - is shared with any other dispatch.
  // The pool retires into the current submission's temporaries with the
  // same lifetime rule as a retired cached pool below.
  if (tape_no_reuse()) {
    tl_desc.set_subkind(timeline::SubKind::DescPoolCreate);
    VkDescriptorPoolSize pool_size{};
    pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    pool_size.descriptorCount = compute.binding_limit();
    VkDescriptorPoolCreateInfo pool_info{
        VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pool_info.maxSets = 1;
    pool_info.poolSizeCount = 1;
    pool_info.pPoolSizes = &pool_size;
    VkDescriptorPool pool{VK_NULL_HANDLE};
    VKX_CHECK(dt.CreateDescriptorPool(
        device_.handle(), &pool_info, nullptr, &pool));
    retired_pools_.push_back(std::shared_ptr<VkDescriptorPool>(
        new VkDescriptorPool(pool),
        [device = device_.handle()](VkDescriptorPool* owned) {
          vk::device_table().DestroyDescriptorPool(device, *owned, nullptr);
          delete owned;
        }));
    VkDescriptorSetLayout descriptor_layout = compute.descriptor_layout();
    VkDescriptorSetAllocateInfo allocate_info{
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    allocate_info.descriptorPool = pool;
    allocate_info.descriptorSetCount = 1;
    allocate_info.pSetLayouts = &descriptor_layout;
    VkDescriptorSet descriptor_set{VK_NULL_HANDLE};
    VKX_CHECK(dt.AllocateDescriptorSets(
        device_.handle(), &allocate_info, &descriptor_set));
    return descriptor_set;
  }
  if (desc_pool_ == VK_NULL_HANDLE || desc_pool_remaining_ == 0) {
    tl_desc.set_subkind(timeline::SubKind::DescPoolCreate);
    if (desc_pool_ != VK_NULL_HANDLE) {
      retired_pools_.push_back(std::shared_ptr<VkDescriptorPool>(
          new VkDescriptorPool(desc_pool_),
          [device = device_.handle()](VkDescriptorPool* owned) {
            vk::device_table().DestroyDescriptorPool(device, *owned, nullptr);
            delete owned;
          }));
      desc_pool_ = VK_NULL_HANDLE;
    }
    VkDescriptorPoolSize pool_size{};
    pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    pool_size.descriptorCount =
        kDescriptorSetsPerPool * compute.binding_limit();
    VkDescriptorPoolCreateInfo pool_info{
        VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pool_info.maxSets = kDescriptorSetsPerPool;
    pool_info.poolSizeCount = 1;
    pool_info.pPoolSizes = &pool_size;
    VKX_CHECK(dt.CreateDescriptorPool(
        device_.handle(), &pool_info, nullptr, &desc_pool_));
    desc_pool_remaining_ = kDescriptorSetsPerPool;
  }
  VkDescriptorSetLayout descriptor_layout = compute.descriptor_layout();
  VkDescriptorSetAllocateInfo allocate_info{
      VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
  allocate_info.descriptorPool = desc_pool_;
  allocate_info.descriptorSetCount = 1;
  allocate_info.pSetLayouts = &descriptor_layout;
  VkDescriptorSet descriptor_set{VK_NULL_HANDLE};
  VKX_CHECK(dt.AllocateDescriptorSets(
      device_.handle(), &allocate_info, &descriptor_set));
  desc_pool_remaining_--;
  return descriptor_set;
}

void CommandEncoder::dispatch_compute(
    ComputeKernel kernel,
    std::span<const ComputeBinding> bindings,
    const ComputeParams& params,
    uint32_t group_count_x,
    uint32_t group_count_y,
    uint32_t group_count_z) {
    std::lock_guard<std::recursive_mutex> op_lk(op_mutex_);

  if (group_count_x == 0 || group_count_y == 0 || group_count_z == 0) {
    return;
  }
  auto& compute = device_.compute();
  dispatch_compute_pipeline(
      compute.pipeline(kernel),
      kernel,
      bindings,
      params,
      group_count_x,
      group_count_y,
      group_count_z);
}

void CommandEncoder::dispatch_compute(
    const std::string& cache_key,
    std::span<const uint32_t> spirv,
    std::span<const ComputeBinding> bindings,
    const ComputeParams& params,
    uint32_t group_count_x,
    uint32_t group_count_y,
    uint32_t group_count_z) {
    std::lock_guard<std::recursive_mutex> op_lk(op_mutex_);

  if (group_count_x == 0 || group_count_y == 0 || group_count_z == 0) {
    return;
  }
  auto& compute = device_.compute();
  dispatch_compute_pipeline(
      compute.pipeline(cache_key, spirv),
      ComputeKernel::Custom,
      bindings,
      params,
      group_count_x,
      group_count_y,
      group_count_z);
}

void CommandEncoder::dispatch_compute(
    const std::string& cache_key,
    std::span<const uint32_t> spirv,
    const std::string& entry_name,
    const VkSpecializationInfo* spec,
    std::span<const ComputeBinding> bindings,
    const ComputeParams& params,
    uint32_t group_count_x,
    uint32_t group_count_y,
    uint32_t group_count_z) {
    std::lock_guard<std::recursive_mutex> op_lk(op_mutex_);

  if (group_count_x == 0 || group_count_y == 0 || group_count_z == 0) {
    return;
  }
  auto& compute = device_.compute();
  dispatch_compute_pipeline(
      compute.pipeline(cache_key, spirv, entry_name, spec),
      ComputeKernel::Custom,
      bindings,
      params,
      group_count_x,
      group_count_y,
      group_count_z);
}

void CommandEncoder::dispatch_compute_pipeline(
    VkPipeline pipeline,
    ComputeKernel profile_kernel,
    std::span<const ComputeBinding> bindings,
    const ComputeParams& params,
    uint32_t group_count_x,
    uint32_t group_count_y,
    uint32_t group_count_z) {
    std::lock_guard<std::recursive_mutex> op_lk(op_mutex_);

  // The stuck-submit diagnostic names this kernel (trace.h); one relaxed
  // store per dispatch, unconditional so a hang report needs no env.
  trace::counters().last_dispatched_kernel.store(
      static_cast<int64_t>(profile_kernel), std::memory_order_relaxed);
  if (env_set("MLX_OMARCHY_TRACE_DISPATCH")) {
    fprintf(stderr,
            "[rtmod] DISPATCH kernel=%d count=%u gx=%u gy=%u gz=%u\n",
            static_cast<int>(profile_kernel), params.count, group_count_x,
            group_count_y, group_count_z);
    for (size_t b = 0; b < bindings.size(); ++b) {
      const auto* owned =
          static_cast<const VulkanBuffer*>(bindings[b].owner);
      fprintf(stderr,
              "[rtmod]   bind[%zu] off=%llu range=%llu alloc=%llu\n",
              b,
              static_cast<unsigned long long>(bindings[b].offset),
              static_cast<unsigned long long>(bindings[b].range),
              static_cast<unsigned long long>(owned ? owned->size : 0llu));
    }
    fflush(stderr);
  }
  auto& compute = device_.compute();
  uint32_t binding_limit = compute.binding_limit();
  if (bindings.empty() || bindings.size() > binding_limit) {
    throw std::invalid_argument(
        "[omarchy] compute dispatch needs " +
        std::to_string(bindings.size()) +
        " storage-buffer bindings; this device allows " +
        std::to_string(binding_limit) + ".");
  }
  // A range past maxStorageBufferRange is invalid Vulkan. Honeykrisp
  // reports INT32_MAX because its storage-buffer addressing uses 32-bit
  // offsets: a release build gives no guarantee past it, a debug build
  // asserts once the range passes UINT32_MAX. Refuse here instead.
  const VkDeviceSize range_limit =
      device_.capabilities().max_storage_buffer_range;
  for (size_t i = 0; i < bindings.size(); ++i) {
    if (bindings[i].range > range_limit) {
      throw std::invalid_argument(
          "[omarchy] " + std::string(trace::current_prim()) +
          " compute dispatch binds " + std::to_string(bindings[i].range) +
          " bytes at storage-buffer binding " + std::to_string(i) +
          "; this device's maxStorageBufferRange is " +
          std::to_string(range_limit) +
          " bytes. One binding cannot address that much memory.");
    }
  }
  bool any_device_local = false;
  for (const auto& item : bindings) {
    auto* vb = const_cast<VulkanBuffer*>(
        static_cast<const VulkanBuffer*>(item.owner));
    if (!vb) {
      vb = omarchy::allocator().lookup_buffer_handle(item.buffer);
    }
    if (vb && vb->device_local) {
      any_device_local = true;
      break;
    }
  }
  if (any_device_local) {
    const auto access = compute.binding_access(pipeline);
    struct AccessAgg {
      VulkanBuffer* vb{nullptr};
      bool read{false};
      bool write{false};
      VkDeviceSize min_off{UINT64_MAX};
      VkDeviceSize max_end{0};
    };
    std::array<AccessAgg, kComputeBindingBudget> aggs{};
    size_t agg_count = 0;
    for (size_t i = 0; i < bindings.size(); ++i) {
      auto* vb = const_cast<VulkanBuffer*>(
          static_cast<const VulkanBuffer*>(bindings[i].owner));
      if (!vb) {
        vb = omarchy::allocator().lookup_buffer_handle(bindings[i].buffer);
      }
      if (!vb || !vb->device_local) {
        continue;
      }
      const bool r = ((access.read_mask >> i) & 1u) != 0u;
      const bool w = ((access.write_mask >> i) & 1u) != 0u;
      const VkDeviceSize off = bindings[i].offset;
      const VkDeviceSize end = bindings[i].range == VK_WHOLE_SIZE
          ? static_cast<VkDeviceSize>(vb->size)
          : off + bindings[i].range;
      AccessAgg* slot = nullptr;
      for (size_t j = 0; j < agg_count; ++j) {
        if (aggs[j].vb == vb) {
          slot = &aggs[j];
          break;
        }
      }
      if (!slot) {
        slot = &aggs[agg_count++];
        slot->vb = vb;
      }
      slot->read = slot->read || r;
      slot->write = slot->write || w;
      slot->min_off = std::min(slot->min_off, off);
      slot->max_end = std::max(slot->max_end, end);
    }
    // Phase 1: complete every required host->device upload before marking any
    // binding device_dirty, so an exception mid-upload leaves both the failing
    // buffer's host_dirty and any sibling write target's device_dirty intact.
    for (size_t j = 0; j < agg_count; ++j) {
      const VkDeviceSize rng = aggs[j].max_end > aggs[j].min_off
          ? aggs[j].max_end - aggs[j].min_off
          : 0;
      // Compute shader descriptor bounds do not guarantee full-allocation
      // overwrite (e.g. Fill/slice kernels writing from output_offset), so
      // conservatively upload any host-dirty compute binding.
      upload_device_local_if_needed(
          aggs[j].vb,
          aggs[j].read,
          aggs[j].write,
          aggs[j].min_off,
          rng,
          /*exact_write_coverage=*/false);
    }
    // Phase 2: all uploads succeeded; mark write targets device_dirty.
    for (size_t j = 0; j < agg_count; ++j) {
      if (aggs[j].write) {
        commit_device_local_write(aggs[j].vb);
      }
    }
  }
  {
    // One allocator lock for the whole binding set (note_batch_buffers):
    // a per-binding lock + handle probe was ~10% of host encode on the
    // M1/M2 Max decode profile.
    std::array<VulkanBuffer*, kComputeBindingBudget> owners{};
    size_t owner_count = 0;
    for (const auto& item : bindings) {
      auto* buf =
          static_cast<VulkanBuffer*>(const_cast<void*>(item.owner));
      if (buf) {
        owners[owner_count++] = buf;
        batch_buffers_.insert(buf);
      }
    }
    omarchy::allocator().note_batch_buffers({owners.data(), owner_count});
  }
  group_count_x = std::min(group_count_x, kMaxComputeGroupCountX);
  group_count_y = std::min(group_count_y, kMaxComputeGroupCountX);
  group_count_z = std::min(group_count_z, kMaxComputeGroupCountX);
  // Bounded GPU-time proxy for the work budget (encoder.h): the open
  // batch's summed work-group counts. Each factor is already clamped to
  // kMaxComputeGroupCountX, so the product fits uint64_t; accumulate
  // saturating so a pathologic sequence cannot wrap.
  const uint64_t groups = static_cast<uint64_t>(group_count_x) *
      group_count_y * group_count_z;
  batch_work_ =
      UINT64_MAX - batch_work_ > groups ? batch_work_ + groups : UINT64_MAX;
  if (!batch_has_prefill_) {
    static const auto kPrefillKernel = []() {
      std::array<bool, static_cast<size_t>(ComputeKernel::Count)> table{};
      for (size_t i = 0; i < table.size(); ++i) {
        table[i] = is_prefill_work_kernel_name(
            compute_kernel_name(static_cast<ComputeKernel>(i)));
      }
      return table;
    }();
    const auto idx = static_cast<size_t>(profile_kernel);
    if (idx < kPrefillKernel.size() && kPrefillKernel[idx]) {
      batch_has_prefill_ = true;
    }
  }

  auto& dt = vk::device_table();
  // Push mode allocates no set: the layout carries PUSH_DESCRIPTOR_BIT,
  // so every write goes straight into the command buffer.
  const bool push_desc = compute.push_descriptors();
  VkDescriptorSet descriptor_set =
      push_desc ? VK_NULL_HANDLE : acquire_descriptor_set(compute);

  std::array<VkDescriptorBufferInfo, kComputeBindingBudget> buffer_info;
  std::array<VkWriteDescriptorSet, kComputeBindingBudget> writes;
  if (!push_desc || !wave_sched()) {
    for (uint32_t index = 0; index < bindings.size(); ++index) {
      buffer_info[index] = {
          bindings[index].buffer, bindings[index].offset, bindings[index].range};
      writes[index] = VkWriteDescriptorSet{
          VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
          nullptr,
          descriptor_set,
          index,
          0,
          1,
          VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
          nullptr,
          &buffer_info[index],
          nullptr};
    }
    if (!push_desc) {
      timeline::ScopedEvent tl_write(
          timeline::Kind::DescUpdate,
          timeline::SubKind::DescWrite,
          encoder_id_,
          static_cast<uint64_t>(bindings.size()));
      dt.UpdateDescriptorSets(
          device_.handle(),
          static_cast<uint32_t>(bindings.size()),
          writes.data(),
          0,
          nullptr);
      trace::counters().vk_descriptor_update_writes += bindings.size();
    }
  }

  if (wave_sched()) {
    // Buffer the node; submit() records it in wave order (emit_pending).
    // Everything order-independent happens here: binding-owner pinning,
    // work accounting, descriptor allocation and writing.
    PendingNode& node = pending_.emplace_back();
    node.kind = PendingNode::Kind::Dispatch;
    node.prim = trace::current_prim();
    node.tape = in_tape_recording ? 1u : 0u;
    node.host_t0 = prof::get().profiling() ? prof::host_ns() : 0;
    node.pipeline = pipeline;
    node.profile_kernel = profile_kernel;
    for (size_t i = 0; i < bindings.size(); ++i) {
      node.bindings[i] = bindings[i];
    }
    node.binding_count = static_cast<uint32_t>(bindings.size());
    node.params = params;
    node.group_count_x = group_count_x;
    node.group_count_y = group_count_y;
    node.group_count_z = group_count_z;
    node.descriptor_set = descriptor_set;
    const auto access = compute.binding_access(pipeline);
    for (size_t i = 0; i < bindings.size(); ++i) {
      TrackedRange range = binding_tracked_range(bindings[i]);
      if (((access.read_mask >> i) & 1u) != 0u) {
        node.rd[node.nr++] = range;
      }
      if (((access.write_mask >> i) & 1u) != 0u) {
        node.wr[node.nw++] = range;
      }
    }
    node_count_++;
    trace::counters().vk_compute_dispatches++;
    return;
  }

  ensure_recording();
  timeline::ScopedEvent tl_dispatch(
      timeline::Kind::CmdRecord,
      timeline::SubKind::Dispatch,
      encoder_id_,
      static_cast<uint64_t>(profile_kernel));
  uint64_t host_t0 = prof::get().profiling() ? prof::host_ns() : 0;
  // MLX_OMARCHY_TAPE_FULL_BARRIERS (diagnostic, docs/install-omarchy.md):
  // the heaviest correct dependency - all commands, all memory access,
  // both directions - ahead of every dispatch, on top of the regular
  // dependency below. Probes whether the driver drops an in-buffer
  // dependency the regular dependency already expresses. It also
  // restarts the gated tracker: the full barrier orders everything
  // recorded before it.
  if (tape_full_barriers()) {
    record_dependency_barrier();
  }
  // MLX_OMARCHY_GATED_BARRIERS (default off): one full barrier before
  // the dispatch only when a binding overlaps unsynced work of the open
  // batch (RAW/WAW against tracked writes, WAR against tracked reads)
  // or when this is the first node of a fresh command buffer. Off: the
  // historic unconditional pre-dispatch barrier. Bindings carry no
  // read/write split, so each binding is tracked as both read and
  // write - the tracker may barrier a read-read pair, never skip a
  // real hazard.
  bool barrier_recorded = false;
  if (gated_barriers()) {
    std::array<TrackedRange, kComputeBindingBudget> ranges{};
    for (size_t i = 0; i < bindings.size(); ++i) {
      ranges[i] = binding_tracked_range(bindings[i]);
    }
    // Per-binding access reflected from the pipeline's SPIR-V (readonly /
    // writeonly block qualifiers): read-only bindings no longer count as
    // writes, so read-read pairs (two consumers of one activation) skip the
    // barrier. Unknown bindings stay read+write. MLX_OMARCHY_DEP_RW=0
    // restores the all-read+write tracking.
    const auto access = compute.binding_access(pipeline);
    std::array<TrackedRange, kComputeBindingBudget> rd{};
    std::array<TrackedRange, kComputeBindingBudget> wr{};
    size_t nr = 0;
    size_t nw = 0;
    for (size_t i = 0; i < bindings.size(); ++i) {
      if (((access.read_mask >> i) & 1u) != 0u) {
        rd[nr++] = ranges[i];
      }
      if (((access.write_mask >> i) & 1u) != 0u) {
        wr[nw++] = ranges[i];
      }
    }
    if (boundary_decision(
            tracker_, {rd.data(), nr}, {wr.data(), nw},
            !head_synced_ && boundary_forced_) ==
        BoundaryDecision::Barrier) {
      record_dispatch_barrier();
      trace::counters().barriers_emitted++;
      prof::get().on_barrier(true);
      barrier_recorded = true;
    } else {
      trace::counters().barriers_skipped++;
      prof::get().on_barrier(false);
    }
    head_synced_ = true;
    boundary_forced_ = false;
    for (size_t i = 0; i < nr; ++i) {
      tracker_.add_read(rd[i]);
    }
    for (size_t i = 0; i < nw; ++i) {
      tracker_.add_write(wr[i]);
    }
    if (export_dep_masks()) {
      // Compute the two exported signals (design 22b395d): (a) whether
      // this dispatch's ranges are provably disjoint from everything
      // since the last explicit dependency (same proof
      // tracker_.needs_barrier just ran, inverted); (b) whether the launch
      // reuses the previous dispatch's binding-layout signature (the
      // mlx-level analogue of the driver's USC register configuration).
      uint32_t usc_signature = 0;
      for (size_t i = 0; i < bindings.size(); ++i) {
        usc_signature = usc_signature * 31 +
                        uint32_t((uintptr_t)ranges[i].buffer ^ ranges[i].offset ^
                                 ranges[i].end);
      }
      uint32_t usc_changed =
          (dep_have_prev_ && usc_signature != dep_prev_signature_) ? 1 : 0;
      dep_records_.push_back({static_cast<uint32_t>(barrier_recorded ? 0 : 1),
                              usc_changed, ++dep_seq_});
      dep_prev_signature_ = usc_signature;
      dep_have_prev_ = 1;
    }
  } else {
    VkMemoryBarrier before{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    before.srcAccessMask =
        VK_ACCESS_HOST_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT |
        VK_ACCESS_SHADER_WRITE_BIT;
    before.dstAccessMask =
        VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    dt.CmdPipelineBarrier(
        cmd_,
        VK_PIPELINE_STAGE_HOST_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT |
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0,
        1,
        &before,
        0,
        nullptr,
        0,
        nullptr);
    head_synced_ = true;
    trace::counters().barriers_emitted++;
    prof::get().on_barrier(true);
    barrier_recorded = true;
  }
  prof::get().before_dispatch(this, current_slot_, cmd_, barrier_recorded);

  VkPipelineLayout pipeline_layout = compute.pipeline_layout();
  if (pipeline != bound_pipeline_) {
    dt.CmdBindPipeline(cmd_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    bound_pipeline_ = pipeline;
  }
  if (push_desc) {
    timeline::ScopedEvent tl_push_write(
        timeline::Kind::DescUpdate,
        timeline::SubKind::DescWrite,
        encoder_id_,
        static_cast<uint64_t>(bindings.size()));
    dt.CmdPushDescriptorSetKHR(
        cmd_,
        VK_PIPELINE_BIND_POINT_COMPUTE,
        pipeline_layout,
        0,
        static_cast<uint32_t>(bindings.size()),
        writes.data());
  } else {
    dt.CmdBindDescriptorSets(
        cmd_,
        VK_PIPELINE_BIND_POINT_COMPUTE,
        pipeline_layout,
        0,
        1,
        &descriptor_set,
        0,
        nullptr);
  }
  dt.CmdPushConstants(
      cmd_,
      pipeline_layout,
      VK_SHADER_STAGE_COMPUTE_BIT,
      0,
      sizeof(params),
      &params);
  dt.CmdDispatch(cmd_, group_count_x, group_count_y, group_count_z);
  prof::get().after_dispatch(
      this,
      current_slot_,
      cmd_,
      profile_kernel,
      params,
      bindings,
      group_count_x,
      group_count_y,
      group_count_z,
      host_t0 != 0 ? prof::host_ns() - host_t0 : 0,
      in_tape_recording ? 1u : 0u);

  // Gated mode tracks this dispatch's writes instead of recording a
  // post barrier; the next node's overlap test consumes the tracking.
  if (!gated_barriers()) {
    VkMemoryBarrier after{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    after.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    after.dstAccessMask =
        VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT |
        VK_ACCESS_HOST_READ_BIT;
    dt.CmdPipelineBarrier(
        cmd_,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT |
            VK_PIPELINE_STAGE_HOST_BIT,
        0,
        1,
        &after,
        0,
        nullptr,
        0,
        nullptr);
    trace::counters().barriers_emitted++;
    prof::get().on_barrier(true);
  }
  if (tape_full_barriers()) {
    // Diagnostic: matching full barrier out of this dispatch, so every
    // dependency between two dispatches is the heaviest form (see the
    // pre-dispatch barrier above).
    record_dependency_barrier();
  }

  node_count_++;
  trace::counters().vk_compute_dispatches++;
}

void CommandEncoder::emit_pending() {
  if (pending_.empty()) {
    return;
  }
  auto& dt = vk::device_table();
  auto& compute = device_.compute();
  ensure_recording();

  wave_nodes_.clear();
  wave_nodes_.reserve(pending_.size());
  for (const auto& p : pending_) {
    wave_nodes_.push_back({{p.rd.data(), p.nr}, {p.wr.data(), p.nw}});
  }
  auto& levels = wave_levels_buf_;
  {
    timeline::ScopedEvent tl_sched(
        timeline::Kind::CmdRecord,
        timeline::SubKind::WaveSchedule,
        encoder_id_,
        static_cast<uint64_t>(pending_.size()));
    wave_levels_into(wave_nodes_, wave_index_, levels);
  }
  uint32_t waves = 0;
  for (uint32_t l : levels) {
    waves = std::max(waves, l);
  }

  if (wave_diag()) {
    // Classify the level-determining edge of every node: which access
    // class (RAW/WAW/WAR) against the deepest hazard source forces this
    // node into its wave. Nodes at level 0 are unforced.
    auto overlaps = [](const TrackedRange& a, const TrackedRange& b) {
      return a.buffer == b.buffer && a.offset < b.end && b.offset < a.end;
    };
    uint64_t hist[4] = {0, 0, 0, 0}; // none, raw, waw, war
    uint64_t depth_sum = 0;
    for (size_t i = 0; i < pending_.size(); ++i) {
      depth_sum += levels[i];
      if (levels[i] == 0) {
        hist[0]++;
        continue;
      }
      uint32_t best = 0;
      for (size_t j = 0; j < i; ++j) {
        if (levels[j] + 1 == levels[i]) {
          best = static_cast<uint32_t>(j);
          // keep the LAST node one wave below; any is representative
        }
      }
      const WaveNode& src = wave_nodes_[best];
      bool raw = false, waw = false, war = false;
      for (const auto& r : wave_nodes_[i].reads) {
        for (const auto& w : src.writes) {
          raw = raw || overlaps(r, w);
        }
      }
      for (const auto& w : wave_nodes_[i].writes) {
        for (const auto& tw : src.writes) {
          waw = waw || overlaps(w, tw);
        }
        for (const auto& tr : src.reads) {
          war = war || overlaps(w, tr);
        }
      }
      hist[raw ? 1 : (waw ? 2 : (war ? 3 : 0))]++;
    }
    std::fprintf(
        stderr,
        "[wave-diag] nodes=%zu waves=%u levels_sum=%llu class "
        "none=%llu raw=%llu waw=%llu war=%llu\n",
        pending_.size(),
        waves + 1,
        static_cast<unsigned long long>(depth_sum),
        static_cast<unsigned long long>(hist[0]),
        static_cast<unsigned long long>(hist[1]),
        static_cast<unsigned long long>(hist[2]),
        static_cast<unsigned long long>(hist[3]));
    std::fflush(stderr);
  }

  // The batch-head dependency on the carried tracker (M2D-6): only a
  // wave-0 node has no barrier before it, so it alone decides the
  // boundary barrier against the carried set (waves 1+ are preceded by
  // their wave boundary barrier, which orders them after carried work
  // either way). Disjoint heads ride the submission's in-order wait.
  bool head_needs_barrier = boundary_forced_;
  if (!head_synced_ && !head_needs_barrier) {
    for (size_t i = 0; i < pending_.size(); ++i) {
      if (levels[i] != 0) {
        continue;
      }
      const auto& node = pending_[i];
      if (tracker_.needs_barrier(
              {node.rd.data(), node.nr}, {node.wr.data(), node.nw})) {
        head_needs_barrier = true;
        break;
      }
    }
  }
  head_synced_ = true;
  boundary_forced_ = false;
  const std::string_view submit_prim = trace::current_prim();

  {
    timeline::ScopedEvent tl_emit(
        timeline::Kind::CmdRecord,
        timeline::SubKind::WaveEmit,
        encoder_id_,
        static_cast<uint64_t>(waves + 1));
    const size_t n_nodes = pending_.size();
    const size_t n_waves = static_cast<size_t>(waves) + 1;
    wave_counts_.assign(n_waves, 0);
    wave_has_transfer_.assign(n_waves, 0);
    for (size_t i = 0; i < n_nodes; ++i) {
      const uint32_t lv = levels[i];
      wave_counts_[lv]++;
      if (pending_[i].kind != PendingNode::Kind::Dispatch) {
        wave_has_transfer_[lv] = 1;
      }
    }
    wave_offsets_.resize(n_waves);
    uint32_t running = 0;
    for (size_t w = 0; w < n_waves; ++w) {
      wave_offsets_[w] = running;
      running += wave_counts_[w];
    }
    wave_order_.resize(n_nodes);
    for (size_t i = 0; i < n_nodes; ++i) {
      const uint32_t lv = levels[i];
      wave_order_[wave_offsets_[lv]++] = static_cast<uint32_t>(i);
    }
    std::array<VkDescriptorBufferInfo, kComputeBindingBudget> buffer_info;
    std::array<VkWriteDescriptorSet, kComputeBindingBudget> writes;
    size_t cursor = 0;
    for (uint32_t w = 0; w <= waves; ++w) {
    const uint32_t cnt = wave_counts_[w];
    const bool wave_pure_dispatch = (wave_has_transfer_[w] == 0);
    bool barrier_owed = (w > 0) || head_needs_barrier;
    for (uint32_t k = 0; k < cnt; ++k) {
      const size_t i = wave_order_[cursor++];
      // One dependency barrier per wave, on the wave head; every later node
      // of the wave rides it hazard-free. Pure compute waves use
      // record_dispatch_barrier() when compute_barrier() is enabled.
      bool preceded_by_barrier = barrier_owed;
      if (barrier_owed) {
        if (wave_pure_dispatch) {
          record_dispatch_barrier();
        } else {
          record_dependency_barrier();
        }
        trace::counters().barriers_emitted++;
        prof::get().on_barrier(true);
        barrier_owed = false;
      } else {
        trace::counters().barriers_skipped++;
        prof::get().on_barrier(false);
      }
      const PendingNode& node = pending_[i];
      if (node.kind == PendingNode::Kind::Dispatch) {
        if (tape_full_barriers()) {
          // Diagnostic: the heaviest dependency before every dispatch
          // (degenerates the wave schedule; not counted as a decision).
          record_dependency_barrier();
          preceded_by_barrier = true;
        }
        prof::get().before_dispatch(
            this, current_slot_, cmd_, preceded_by_barrier);
        VkPipelineLayout pipeline_layout = compute.pipeline_layout();
        if (node.pipeline != bound_pipeline_) {
          dt.CmdBindPipeline(
              cmd_, VK_PIPELINE_BIND_POINT_COMPUTE, node.pipeline);
          bound_pipeline_ = node.pipeline;
        }
        if (compute.push_descriptors()) {
          for (uint32_t b = 0; b < node.binding_count; ++b) {
            buffer_info[b] = {node.bindings[b].buffer,
                              node.bindings[b].offset,
                              node.bindings[b].range};
            writes[b] = VkWriteDescriptorSet{
                VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                nullptr,
                VK_NULL_HANDLE,
                b,
                0,
                1,
                VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                nullptr,
                &buffer_info[b],
                nullptr};
          }
          timeline::ScopedEvent tl_push_write(
              timeline::Kind::DescUpdate,
              timeline::SubKind::DescWrite,
              encoder_id_,
              static_cast<uint64_t>(node.binding_count));
          dt.CmdPushDescriptorSetKHR(
              cmd_,
              VK_PIPELINE_BIND_POINT_COMPUTE,
              pipeline_layout,
              0,
              node.binding_count,
              writes.data());
        } else {
          dt.CmdBindDescriptorSets(
              cmd_,
              VK_PIPELINE_BIND_POINT_COMPUTE,
              pipeline_layout,
              0,
              1,
              &node.descriptor_set,
              0,
              nullptr);
        }
        dt.CmdPushConstants(
            cmd_,
            pipeline_layout,
            VK_SHADER_STAGE_COMPUTE_BIT,
            0,
            sizeof(node.params),
            &node.params);
        dt.CmdDispatch(
            cmd_,
            node.group_count_x,
            node.group_count_y,
            node.group_count_z);
        trace::current_prim() = node.prim;
        prof::get().after_dispatch(
            this,
            current_slot_,
            cmd_,
            node.profile_kernel,
            node.params,
            std::span<const ComputeBinding>(
                node.bindings.data(), node.binding_count),
            node.group_count_x,
            node.group_count_y,
            node.group_count_z,
            node.host_t0 != 0 ? prof::host_ns() - node.host_t0 : 0,
            node.tape);
        trace::current_prim() = submit_prim;
        if (export_dep_masks()) {
          // A node that rode its wave's barrier is provably disjoint from
          // everything since that barrier (the level assignment proved no
          // hazard against any earlier node); a wave-head node recorded
          // the barrier itself.
          uint32_t usc_signature = 0;
          for (uint32_t b = 0; b < node.nr + node.nw; ++b) {
            const TrackedRange& r =
                b < node.nr ? node.rd[b] : node.wr[b - node.nr];
            usc_signature = usc_signature * 31 +
                uint32_t((uintptr_t)r.buffer ^ r.offset ^ r.end);
          }
          dep_records_.push_back(
              {static_cast<uint32_t>(preceded_by_barrier ? 0 : 1),
               static_cast<uint32_t>(
                   (dep_have_prev_ && usc_signature != dep_prev_signature_)
                       ? 1
                       : 0),
               ++dep_seq_});
          dep_prev_signature_ = usc_signature;
          dep_have_prev_ = 1;
        }
      } else if (node.kind == PendingNode::Kind::Copy) {
        VkBufferCopy region{};
        region.srcOffset = node.src_offset;
        region.dstOffset = node.dst_offset;
        region.size = node.size;
        dt.CmdCopyBuffer(cmd_, node.src, node.dst, 1, &region);
        transfer_since_barrier_ = true;
      } else if (node.kind == PendingNode::Kind::Fill) {
        dt.CmdFillBuffer(
            cmd_, node.dst, node.write_offset, node.size, node.value);
        transfer_since_barrier_ = true;
      } else {
        dt.CmdUpdateBuffer(
            cmd_,
            node.dst,
            node.write_offset,
            node.update_data.size(),
            node.update_data.data());
        transfer_since_barrier_ = true;
      }
      for (uint32_t r = 0; r < node.nr; ++r) {
        tracker_.add_read(node.rd[r]);
      }
      for (uint32_t r = 0; r < node.nw; ++r) {
        tracker_.add_write(node.wr[r]);
      }
    }
    }
  }
  pending_.clear();
}

void CommandEncoder::commit() {
    std::lock_guard<std::recursive_mutex> op_lk(op_mutex_);

  if (!recording_ && pending_.empty() && wait_semaphores_.empty() &&
      signal_semaphores_.empty() && completed_handlers_.empty()) {
    trace::counters().commit_calls_noop++;
    if (env_set("MLX_OMARCHY_TRACE_DISPATCH")) {
      fprintf(stderr, "[rtmod] COMMIT-NOOP\n");
    }
    return;
  }
  trace::counters().commit_calls_with_work++;
  submit();
}

void CommandEncoder::synchronize(const char* reason) {
  timeline::ScopedEvent tl_sync(
      timeline::Kind::StepEnd,
      timeline::SubKind::Synchronize,
      encoder_id_,
      static_cast<uint64_t>(node_count_));
  commit();
  join_last_completion(reason);
}

void CommandEncoder::wait_outstanding_submissions() {
    std::lock_guard<std::recursive_mutex> op_lk(op_mutex_);

  // Timeline wait on the device completion semaphore at the newest
  // reserved value: the next submission this encoder commits starts only
  // after every submission reserved so far, on any stream, has completed.
  // The wait rides that next submission, so no extra submission is
  // created and the host never blocks. Reading the value here is safe
  // against self-deadlock: this stream's own next completion value is
  // reserved strictly later (at commit), so the waited value always
  // belongs to a submission queued ahead of ours.
  uint64_t value = device_.completions().last_reserved();
  if (value == 0) {
    return;
  }
  add_semaphore_wait(device_.completions().semaphore(), value, nullptr);
}

ComputeBinding CommandEncoder::sdpa_merge_counters(uint32_t heads) {
  // Two words per head: an arrival count and a publish count for the
  // single-dispatch split-K merge (sdpa_decode_split_p1m.comp).
  const uint32_t words = std::max<uint32_t>(2u * heads, 128u);
  if (sdpa_counter_words_ < words) {
    sdpa_counter_buf_ = omarchy::allocator().malloc(
        static_cast<size_t>(words) * sizeof(uint32_t));
    auto* vb = static_cast<VulkanBuffer*>(sdpa_counter_buf_.ptr());
    // In-stream fill (the resnorm ticket's pattern): the pool's one-time
    // zeroing is a recorded vkCmdFillBuffer ordered before the first merge
    // dispatch, so it can never land on a buffer the device never sees.
    fill_buffer(
        vb->buffer, 0u, static_cast<VkDeviceSize>(words) * sizeof(uint32_t), 0u);
    sdpa_counter_words_ = words;
  }
  auto* vb = static_cast<VulkanBuffer*>(sdpa_counter_buf_.ptr());
  return ComputeBinding{
      vb->buffer, 0u, static_cast<VkDeviceSize>(sdpa_counter_words_) * sizeof(uint32_t), vb};
}

void CommandEncoder::submit() {
  // The submitting thread may predate Runtime::init (the MLX stream thread).
  raise_cpu_clamp_this_thread();
  // Deferred signals let another thread queue (or this flush commit) wait/
  // signal ops against this encoder: serialize the snapshot below against
  // add_semaphore_wait/add_semaphore_signal.
  std::lock_guard<std::mutex> pending_lk(pending_sem_mutex_);
  auto& dt = vk::device_table();
  if (env_set("MLX_OMARCHY_TRACE_DISPATCH")) {
    fprintf(stderr, "[rtmod] SUBMIT-ENTER tid=%lu\n", (unsigned long)syscall(SYS_gettid));
  }
  // Wave scheduling: record the buffered nodes into the command buffer
  // now, in wave order, before the readback barrier and EndCommandBuffer.
  emit_pending();
  bool was_recording = recording_;
  uint64_t submit_t0 = prof::get().profiling() ? prof::host_ns() : 0;
  uint64_t close_t = 0;
  uint64_t queue_t0 = 0;
  uint64_t queue_t1 = 0;
  uint64_t submitted = 0;

  if (recording_) {
    timeline::ScopedEvent tl_end(
        timeline::Kind::CmdRecord,
        timeline::SubKind::CmdEnd,
        encoder_id_,
        static_cast<uint64_t>(node_count_));
    if (gated_barriers()) {
      VkMemoryBarrier readback{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
      readback.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
      readback.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
      dt.CmdPipelineBarrier(
          cmd_, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
          VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &readback,
          0, nullptr, 0, nullptr);
      trace::counters().barriers_emitted++;
      prof::get().on_barrier(true);
    }
    prof::get().on_buffer_end(this, current_slot_, cmd_);
    VKX_CHECK(dt.EndCommandBuffer(cmd_));
    close_t = prof::get().profiling() ? prof::host_ns() : 0;
  }

  auto& wait_sems = submit_wait_sems_;
  auto& wait_values = submit_wait_values_;
  auto& wait_stages = submit_wait_stages_;
  wait_sems.clear();
  wait_values.clear();
  wait_sems.reserve(wait_semaphores_.size() + 1);
  wait_values.reserve(wait_semaphores_.size() + 1);
  // In-order stream: this submission waits for the stream's previous
  // submission. Vulkan defines no execution or memory dependency between
  // submissions without a semaphore wait, and the per-dispatch barriers
  // cover hazards inside one command buffer only. Without this wait, a
  // submission queued behind long work can read a prior tiny submission's
  // output before that write lands (Honeykrisp: compiled 4-bit decode
  // read an eager one-element f32 as recycled page garbage, 20/20).
  // Waiting on an already-signaled timeline value is a driver
  // pass-through, so the shallow-queue case pays nothing.
  if (last_completion_ != 0) {
    wait_sems.push_back(device_.completions().semaphore());
    wait_values.push_back(last_completion_);
  }
  for (auto& pending : wait_semaphores_) {
    wait_sems.push_back(pending.semaphore);
    wait_values.push_back(pending.value);
  }
  wait_stages.assign(wait_sems.size(), VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);

  VkTimelineSemaphoreSubmitInfo timeline{
      VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO};
  timeline.waitSemaphoreValueCount = static_cast<uint32_t>(wait_values.size());
  timeline.pWaitSemaphoreValues = wait_values.data();

  VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
  si.pNext = &timeline;
  si.waitSemaphoreCount = static_cast<uint32_t>(wait_sems.size());
  si.pWaitSemaphores = wait_sems.data();
  si.pWaitDstStageMask = wait_stages.data();
  si.commandBufferCount = recording_ ? 1u : 0u;
  si.pCommandBuffers = recording_ ? &cmd_ : nullptr;

  // Every submission also signals the device completion timeline; user
  // (event) signals ride along in the same submission, in order.
  auto& signal_sems = submit_signal_sems_;
  auto& signal_values = submit_signal_values_;
  signal_sems.clear();
  signal_values.clear();
  signal_sems.reserve(signal_semaphores_.size() + 1);
  signal_values.reserve(signal_semaphores_.size() + 1);
  for (auto& pending : signal_semaphores_) {
    // Deferred signals let several Event::signal calls on the same Event
    // coalesce into one submission; pSignalSemaphores must not repeat a
    // semaphore, so merge to the highest value per handle. Every pending
    // on_submit still fires below regardless of the merge.
    bool merged = false;
    for (size_t i = 0; i < signal_sems.size(); ++i) {
      if (signal_sems[i] == pending.semaphore) {
        if (pending.value > signal_values[i]) {
          signal_values[i] = pending.value;
        }
        merged = true;
        break;
      }
    }
    if (!merged) {
      signal_sems.push_back(pending.semaphore);
      signal_values.push_back(pending.value);
    }
  }

  // Ownership moves to the dispatcher entry: each pending-semaphore
  // keepalive keeps its Event alive, and handlers run on the completion
  // thread. The batch's buffers need no shared_ptr: they were stamped
  // with the completion value below, and the allocator quarantine keeps
  // any freed-but-in-flight buffer out of the reuse cache until its
  // generation drains.
  std::vector<std::shared_ptr<void>> keepalive;
  for (auto& pool : retired_pools_) {
    keepalive.push_back(std::move(pool));
  }
  retired_pools_.clear();
  for (auto& pending : wait_semaphores_) {
    keepalive.push_back(std::move(pending.keepalive));
  }
  for (auto& pending : signal_semaphores_) {
    keepalive.push_back(std::move(pending.keepalive));
  }

  {
    // VkQueue is externally synchronized: the lock covers completion-value
    // assignment (timeline signals must increase along the queue) through
    // QueueSubmit and the dispatcher enqueue. It is never held across a
    // host wait.
    std::unique_lock<std::mutex> lk(device_.queue_mutex(), std::defer_lock);
    {
      timeline::ScopedEvent tl_qlock(
          timeline::Kind::QueueSubmit,
          timeline::SubKind::QueueLockWait,
          encoder_id_);
      lk.lock();
    }
    // Khronos guidance: HOST_VISIBLE memory without HOST_COHERENT needs an
    // explicit flush before submission.
    {
      timeline::ScopedEvent tl_flush(
          timeline::Kind::Readback,
          timeline::SubKind::FlushNoncoherent,
          encoder_id_);
      omarchy::allocator().flush_noncoherent(device_.handle());
    }
    uint64_t completion_value = device_.completions().reserve();
    // GPU work is now in flight: keep the CPU complex out of the deep idle
    // state while it runs (T8103 GPU performance state; cpu_pd_hold.h).
    cpu_pd_hold_note_submit(device_, completion_value);
    // Stamp every buffer referenced by this batch with the completion
    // value just reserved. Free-before-drain sends such buffers to the
    // allocator quarantine; release_quarantine recycles them one
    // generation later. Semaphore keepalives still move into the
    // dispatcher payload below.
    {
      timeline::ScopedEvent tl_stamp(
          timeline::Kind::QueueSubmit,
          timeline::SubKind::StampBatch,
          encoder_id_,
          static_cast<uint64_t>(batch_buffers_.size()));
      omarchy::allocator().stamp_batch(batch_buffers_, completion_value);
    }
    batch_buffers_.clear();
    VkSemaphore completion_sem = device_.completions().semaphore();
    signal_sems.push_back(completion_sem);
    signal_values.push_back(completion_value);
    timeline.signalSemaphoreValueCount =
        static_cast<uint32_t>(signal_values.size());
    timeline.pSignalSemaphoreValues = signal_values.data();
    si.signalSemaphoreCount = static_cast<uint32_t>(signal_sems.size());
    si.pSignalSemaphores = signal_sems.data();
    try {
      queue_t0 = prof::get().profiling() ? prof::host_ns() : 0;
      // Deterministic swallow simulation for the recovery ladder, in the
      // two field-observed classes:
      //
      // MLX_OMARCHY_TEST_DROP_SUBMIT drops the Nth (1-based) QueueSubmit
      // entirely, the way Honeykrisp drops submissions in burst windows:
      // the batch never begins, so its started event never sets and
      // recovery rung 1 (kick + resubmit at fresh values) must fire.
      //
      // MLX_OMARCHY_TEST_DROP_SIGNAL submits the Nth batch but strips
      // every timeline signal from it (ride-along user events AND the
      // device completion): the kernels execute and the started event
      // sets, but no signal ever publishes - the executed-but-unsignaled
      // class - so recovery rung 2 (host vkSignalSemaphore) must fire.
      //
      // Both share one submit sequence; the host still publishes the
      // completion entry, so the watchdog waits on a signal the GPU will
      // never (submit-drop) / did not (signal-strip) deliver. Comma-
      // separated 1-based ordinals: "1" hits the first submit, "1,2" the
      // first two (consecutive-swallow recovery proof).
      static std::atomic<uint64_t> submit_sequence{0};
      uint64_t ordinal = submit_sequence.fetch_add(1) + 1;
      auto env_hits_ordinal = [ordinal](const char* name) {
        const char* e = env_str(name);
        if (!e) {
          return false;
        }
        char* p = const_cast<char*>(e);
        while (*p) {
          if (strtoull(p, &p, 10) == ordinal) {
            return true;
          }
          if (*p == ',') {
            ++p;
          } else {
            break;
          }
        }
        return false;
      };
      bool simulate_drop = env_hits_ordinal("MLX_OMARCHY_TEST_DROP_SUBMIT");
      bool simulate_strip =
          !simulate_drop && env_hits_ordinal("MLX_OMARCHY_TEST_DROP_SIGNAL");
      if (simulate_strip) {
        si.signalSemaphoreCount = 0;
        si.pSignalSemaphores = nullptr;
        timeline.signalSemaphoreValueCount = 0;
        timeline.pSignalSemaphoreValues = nullptr;
      }
      if (simulate_drop) {
        fprintf(stderr, "[rtmod] TEST-DROP tid=%lu cv=%lu\n",
                (unsigned long)syscall(SYS_gettid), (unsigned long)completion_value);
      } else if (simulate_strip) {
        fprintf(stderr, "[rtmod] TEST-STRIP tid=%lu cv=%lu\n",
                (unsigned long)syscall(SYS_gettid), (unsigned long)completion_value);
      } else {
        timeline::ScopedEvent tl_qsub(
            timeline::Kind::QueueSubmit,
            timeline::SubKind::VkQueueSubmit,
            encoder_id_,
            completion_value);
        // No submission fence: recycle safety comes from the completion
        // timeline's one-generation rule (observing a later drain proves
        // the driver's submit-final cleanup), so a submit carries no
        // fence and the host issues no per-submit syncobj poll.
        const VkResult submit_result =
            dt.QueueSubmit(device_.queue(), 1, &si, VK_NULL_HANDLE);
        if (submit_result != VK_SUCCESS) {
          // A synchronous QueueSubmit failure is a rejection of this
          // submission, not a GPU hang: on Borealis the kernel's -EINVAL
          // surfaces through vk_device_set_lost, so it reports as
          // VK_ERROR_DEVICE_LOST even though the device is fine (issue
          // #526). A genuinely lost device is discovered at a wait or
          // poll, never inside the submit call. Name the failing
          // parameters so the kernel's rejection log can be matched
          // field-for-field.
          std::string waits;
          for (size_t i = 0; i < wait_sems.size(); ++i) {
            waits += (i ? "," : "") + std::to_string(wait_values[i]);
          }
          std::string signals;
          for (size_t i = 0; i < signal_sems.size(); ++i) {
            signals += (i ? "," : "") + std::to_string(signal_values[i]);
          }
          throw std::runtime_error(
              std::string("[omarchy] invalid submission: vkQueueSubmit "
                          "rejected with ") +
              vk::result_string(submit_result) +
              " (ordinal=" + std::to_string(ordinal) +
              " encoder=" + std::to_string(encoder_id_) +
              " cv=" + std::to_string(completion_value) +
              " nodes=" + std::to_string(node_count_) +
              " work=" + std::to_string(batch_work_) +
              " command_buffer=" + std::to_string(si.commandBufferCount) +
              " wait_values=[" + waits + "]" +
              " signal_values=[" + signals + "]" +
              ")");
        }
      }
      queue_t1 = prof::get().profiling() ? prof::host_ns() : 0;
    } catch (...) {
      // The submission never reached the driver: the ended command buffer
      // and the pending semaphore lists are dead (their keepalives have
      // already moved into the local payload and die with this frame).
      // Reset the encoder so it can be reused or destroyed cleanly; the
      // typed error propagates to the stream's error handling. The
      // batch's buffer stamps are harmless: those buffers stay alive via
      // their arrays or the quarantine.
      batch_buffers_.clear();
      pending_.clear();
      dep_records_.clear();
      recording_ = false;
      resnorm_ticket_slot_ = -1;
      node_count_ = 0;
      batch_work_ = 0;
  batch_has_prefill_ = false;
      wait_semaphores_.clear();
      signal_semaphores_.clear();
      completed_handlers_.clear();
      deferred_signals_pending_ = false;
      reset_dependency_tracking();
      // The reserved completion value can never signal GPU-side: without
      // a pending entry, later joins would block on drained_value_
      // forever. Publish the empty entry - its waiters get the typed
      // watchdog error instead of an unbounded join (reserved values must
      // always have a completion entry).
      device_.completions().enqueue(completion_value, {}, {});
      throw;
    }
    // Publish only after the submit: the dispatcher must never wait on a
    // value whose submission has not been handed to the driver.
    // Retain everything the batch put on the queue so the watchdog's
    // recovery ladder can resubmit it if Honeykrisp drops the submission
    // (erased when the completion drains).
    {
      CompletionDispatcher::ResubmitBatch batch;
      batch.value = completion_value;
      batch.cmd = recording_ ? cmd_ : VK_NULL_HANDLE;
      batch.wait_sems = wait_sems;
      batch.wait_values = wait_values;
      batch.signal_sems = signal_sems;
      batch.signal_values = signal_values;
      device_.completions().retain_for_resubmit(
          completion_value, std::move(batch));
    }
    // Publish the carrying completion value to every queued signal's
    // owner: deferred signals (defer_commits_enabled) could not name it at
    // queue time. Runs before the keepalive list moves into the
    // dispatcher — on a fire-and-forget Event the queued keepalive can
    // be the last reference, and a completion thread draining it first
    // would free the publish target under the hook.
    for (auto& pending : signal_semaphores_) {
      if (pending.on_submit) {
        pending.on_submit(completion_value);
      }
    }
    device_.completions().enqueue(
        completion_value,
        std::move(keepalive),
        std::move(completed_handlers_),
        was_recording ? slots_[current_slot_].started : VK_NULL_HANDLE);
    last_completion_ = completion_value;
    submitted = completion_value;
    if (resnorm_ticket_slot_ >= 0) {
      resnorm_ticket_in_flight_[resnorm_ticket_slot_] = completion_value;
      resnorm_ticket_slot_ = -1;
    }
    if (was_recording) {
      slots_[current_slot_].in_flight = completion_value;
    }
    trace::counters().vk_submissions++;
    if (env_set("MLX_OMARCHY_TRACE_DISPATCH")) {
      fprintf(stderr,
              "[rtmod] SUBMIT tid=%lu cv=%lu waits=%lu sigs=%lu cmds=%u\n",
              (unsigned long)syscall(SYS_gettid),
              (unsigned long)completion_value,
              (unsigned long)wait_sems.size(),
              (unsigned long)signal_values.size(),
              (unsigned)si.commandBufferCount);
    }
  }

  recording_ = false;
  node_count_ = 0;
  batch_work_ = 0;
  batch_has_prefill_ = false;
  pending_.clear();
  // Phase-1 records have no consumer yet; bound the vector at the batch
  // boundary like every other per-batch list.
  dep_records_.clear();
  wait_semaphores_.clear();
  signal_semaphores_.clear();
  completed_handlers_.clear();
  deferred_signals_pending_ = false;
  // Carry the tracker across the submission boundary (M2D-6, phase 2 of
  // docs/research/m2d6-boundary-barrier-design.md): the submission's
  // in-order wait drains the closed batch, so the next head's own ranges
  // decide the boundary barrier on evidence instead of position. The
  // access budget bounds the carried table: crossing it clears now and
  // forces a real barrier at the next head.
  if (tracker_.tracked_accesses() >= kBoundaryTrackerAccessBudget) {
    tracker_.clear();
    boundary_forced_ = true;
  }
  head_synced_ = false;
  transfer_since_barrier_ = false;
  prof::get().on_submit_boundary(submitted, close_t, queue_t0, queue_t1);
  prof::get().on_submit_end(
      this,
      submitted,
      submit_t0 != 0 ? prof::host_ns() - submit_t0 : 0,
      was_recording ? current_slot_ : -1);

  // No vkResetCommandBuffer: the buffer may still be executing. The ring
  // slot is marked in flight above; ensure_recording() only reuses a slot
  // whose submission completed (or joins the oldest), and BeginCommandBuffer
  // then resets the buffer implicitly (the pool was created with
  // RESET_COMMAND_BUFFER_BIT).
}

CommandEncoder& get_command_encoder(Stream s) {
  // Mirrors the CUDA backend: the per-thread table misses for a stream
  // created on another thread, so fall back to the global thread-unsafe
  // table and finally raise the upstream std::runtime_error contract.
  // unordered_map::at would throw std::out_of_range, which escapes the
  // caller's catch(std::runtime_error) and terminates the process.
  auto& encoders = get_command_encoders();
  auto it = encoders.find(s.index);
  if (it == encoders.end()) {
    auto& global_encoders = get_global_command_encoders();
    it = global_encoders.find(s.index);
    if (it == global_encoders.end()) {
      throw std::runtime_error(
          "There is no Stream(gpu, " + std::to_string(s.index) +
          ") in current thread.");
    }
  }
  return *it->second;
}

std::unordered_map<int, std::shared_ptr<CommandEncoder>>&
get_command_encoders() {
  static thread_local std::unordered_map<int, std::shared_ptr<CommandEncoder>>
      encoders;
  return encoders;
}

std::unordered_map<int, std::shared_ptr<CommandEncoder>>&
get_global_command_encoders() {
  static std::unordered_map<int, std::shared_ptr<CommandEncoder>>
      global_encoders;
  return global_encoders;
}

void synchronize_pending_encoders() {
  for (auto& [idx, enc] : get_command_encoders()) {
    if (enc->needs_commit()) {
      enc->synchronize("vram readback");
    }
  }
  for (auto& [idx, enc] : get_global_command_encoders()) {
    if (enc->needs_commit()) {
      enc->synchronize("vram readback");
    }
  }
}

void flush_deferred_signal_encoders() {
  // Snapshot the registry, then commit outside the lock: commits re-enter
  // note_deferred_signals() paths and must not hold it. Entries whose flag
  // cleared in the meantime (already submitted) are skipped, not pruned;
  // the destructor or the next flush drops them.
  std::vector<std::weak_ptr<CommandEncoder>> pending;
  {
    std::lock_guard<std::mutex> lk(deferred_signal_registry_mutex());
    pending.reserve(deferred_signal_registry().size());
    for (auto& [ptr, weak] : deferred_signal_registry()) {
      pending.push_back(weak);
    }
    deferred_signal_registry().clear();
  }
  // Commits run outside the registry lock and under each encoder's
  // op_mutex_, so a commit from a foreign thread serializes against the
  // owner's recording. lock() keeps the encoder alive through its own
  // destructor's wait: an expired entry means the producer already died,
  // and its destructor drained the batch.
  for (auto& weak : pending) {
    std::shared_ptr<CommandEncoder> enc = weak.lock();
    if (enc && enc->deferred_signals_pending()) {
      enc->commit();
    }
  }
}

} // namespace mlx::core::omarchy
