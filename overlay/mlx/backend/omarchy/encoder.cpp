// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

#include "mlx/backend/omarchy/encoder.h"
#include <stdexcept>
#include <unistd.h>
#include <sys/syscall.h>

#include "mlx/backend/omarchy/allocator.h"
#include "mlx/backend/omarchy/device.h"
#include "mlx/backend/omarchy/trace.h"
#include "mlx/backend/omarchy/gpu_profiler.h"
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

} // namespace

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
  // MLX_OMARCHY_GATED_BARRIERS (docs/install-omarchy.md): default off.
  // On, dispatch/copy/fill nodes record a barrier only when their buffer
  // ranges overlap unsynced work of the open batch; off, the historic
  // unconditional pre+post dispatch barriers apply. Read once: the gate
  // shapes recorded commands, so flipping it mid-batch would desync the
  // tracker from the command buffer.
  static const bool on = env_flag("MLX_OMARCHY_GATED_BARRIERS");
  return on;
}

bool CommandEncoder::batch_needs_barrier(
    std::span<const TrackedRange> reads,
    std::span<const TrackedRange> writes) const {
  auto overlaps = [](const TrackedRange& a, const TrackedRange& b) {
    return a.buffer == b.buffer && a.offset < b.end && b.offset < a.end;
  };
  // Read after write and write after write.
  for (const auto& r : reads) {
    for (const auto& w : tracked_writes_) {
      if (overlaps(r, w)) {
        return true;
      }
    }
  }
  // Write after write and write after read: compute-to-compute in one
  // queue has no execution dependency without a barrier, so a node
  // writing a range any earlier node read must also wait.
  for (const auto& w : writes) {
    for (const auto& tw : tracked_writes_) {
      if (overlaps(w, tw)) {
        return true;
      }
    }
    for (const auto& tr : tracked_reads_) {
      if (overlaps(w, tr)) {
        return true;
      }
    }
  }
  return false;
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
  tracked_reads_.clear();
  tracked_writes_.clear();
  head_synced_ = true;
}

void CommandEncoder::reset_dependency_tracking() {
  tracked_reads_.clear();
  tracked_writes_.clear();
  head_synced_ = false;
}

CommandEncoder::CommandEncoder(Device& device) : device_(device) {
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
  try {
    synchronize();
  } catch (const std::exception&) {
  }
}

// Join this encoder's newest in-flight submission, if any, and refresh
// host mappings. After this, every command buffer this encoder submitted
// has left the pending state (the completion timeline is strictly
// ordered), so they can legally be begun again, and host reads see the
// submissions' final bytes.
void CommandEncoder::join_last_completion(const char* reason) {
  if (std::getenv("MLX_OMARCHY_TRACE_DISPATCH") != nullptr) {
    fprintf(stderr, "[rtmod] JOIN tid=%lu reason=%s last=%lu\n", (unsigned long)syscall(SYS_gettid), reason,
            (unsigned long)last_completion_);
  }
  if (last_completion_ == 0) {
    return;
  }
  uint64_t value = last_completion_;
  uint64_t join_t0 = prof::get().profiling() ? prof::host_ns() : 0;
  device_.completions().wait(value);
  uint64_t wait_t1 = prof::get().profiling() ? prof::host_ns() : 0;
  last_completion_ = 0;
  omarchy::allocator().invalidate_noncoherent(device_.handle());
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
    completions.wait(oldest_value);
    slots_[oldest].in_flight = 0;
    chosen = oldest;
  }
  current_slot_ = chosen;
  cmd_ = slots_[chosen].cmd;
  completions.reset_progress_event(slots_[chosen].started);
  VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  VKX_CHECK(vk::device_table().BeginCommandBuffer(cmd_, &bi));
  vk::device_table().CmdSetEvent(
      cmd_, slots_[chosen].started, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT);
  prof::get().on_begin(
      this, chosen, cmd_, begin_t0 != 0 ? prof::host_ns() - begin_t0 : 0);
  recording_ = true;
}

void CommandEncoder::copy_buffer(
    VkBuffer src,
    VkBuffer dst,
    VkDeviceSize size,
    VkDeviceSize src_offset,
    VkDeviceSize dst_offset) {
  ensure_recording();
  // The tape-full diagnostic forces the heaviest dependency around the
  // copy and restarts tracking either way.
  if (tape_full_barriers()) {
    record_dependency_barrier();
  }
  if (gated_barriers()) {
    TrackedRange read{src, src_offset, tracked_range_end(src_offset, size)};
    TrackedRange write{dst, dst_offset, tracked_range_end(dst_offset, size)};
    if (!head_synced_ || batch_needs_barrier({&read, 1}, {&write, 1})) {
      record_dependency_barrier();
      trace::counters().barriers_emitted++;
      prof::get().on_barrier(true);
    } else {
      trace::counters().barriers_skipped++;
      prof::get().on_barrier(false);
    }
    tracked_reads_.push_back(read);
    tracked_writes_.push_back(write);
  }
  VkBufferCopy region{};
  region.srcOffset = src_offset;
  region.dstOffset = dst_offset;
  region.size = size;
  vk::device_table().CmdCopyBuffer(cmd_, src, dst, 1, &region);
  node_count_++;
  trace::counters().vk_buffer_copies++;
}

void CommandEncoder::fill_buffer(
    VkBuffer dst,
    uint32_t value,
    VkDeviceSize size,
    VkDeviceSize offset) {
  ensure_recording();
  if (tape_full_barriers()) {
    record_dependency_barrier();
  }
  if (gated_barriers()) {
    TrackedRange write{dst, offset, tracked_range_end(offset, size)};
    if (!head_synced_ || batch_needs_barrier({}, {&write, 1})) {
      record_dependency_barrier();
      trace::counters().barriers_emitted++;
      prof::get().on_barrier(true);
    } else {
      trace::counters().barriers_skipped++;
      prof::get().on_barrier(false);
    }
    tracked_writes_.push_back(write);
  }
  vk::device_table().CmdFillBuffer(cmd_, dst, offset, size, value);
  node_count_++;
  trace::counters().vk_buffer_fills++;
}

// Allocate one descriptor set from the cached pool. A pool serves up to
// kDescriptorSetsPerPool dispatches; on exhaustion it is retired into the
// currently-recording submission's temporaries, so it is destroyed only
// after that submission completes — which is strictly after every earlier
// submission whose batches allocated sets from it (completion timeline
// values increase along the queue).
VkDescriptorSet CommandEncoder::acquire_descriptor_set(
    ComputeRuntime& compute) {
  auto& dt = vk::device_table();
  // MLX_OMARCHY_TAPE_NO_REUSE (diagnostic, docs/install-omarchy.md):
  // every dispatch gets its own descriptor pool with exactly one set, so
  // no pool - and therefore no set - is shared with any other dispatch.
  // The pool retires into the current submission's temporaries with the
  // same lifetime rule as a retired cached pool below.
  if (tape_no_reuse()) {
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

void CommandEncoder::dispatch_compute_pipeline(
    VkPipeline pipeline,
    ComputeKernel profile_kernel,
    std::span<const ComputeBinding> bindings,
    const ComputeParams& params,
    uint32_t group_count_x,
    uint32_t group_count_y,
    uint32_t group_count_z) {
  if (std::getenv("MLX_OMARCHY_TRACE_DISPATCH") != nullptr) {
    fprintf(stderr,
            "[rtmod] DISPATCH kernel=%d count=%u gx=%u gy=%u gz=%u\n",
            static_cast<int>(profile_kernel), params.count, group_count_x,
            group_count_y, group_count_z);
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
  for (const auto& item : bindings) {
    note_binding_owner(item.owner);
  }
  group_count_x = std::min(group_count_x, kMaxComputeGroupCountX);
  group_count_y = std::min(group_count_y, kMaxComputeGroupCountX);
  group_count_z = std::min(group_count_z, kMaxComputeGroupCountX);

  auto& dt = vk::device_table();
  VkDescriptorSet descriptor_set = acquire_descriptor_set(compute);

  std::array<VkDescriptorBufferInfo, kComputeBindingBudget> buffer_info{};
  std::array<VkWriteDescriptorSet, kComputeBindingBudget> writes{};
  for (uint32_t index = 0; index < bindings.size(); ++index) {
    buffer_info[index] = {
        bindings[index].buffer, bindings[index].offset, bindings[index].range};
    writes[index].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[index].dstSet = descriptor_set;
    writes[index].dstBinding = index;
    writes[index].descriptorCount = 1;
    writes[index].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[index].pBufferInfo = &buffer_info[index];
  }
  dt.UpdateDescriptorSets(
      device_.handle(),
      static_cast<uint32_t>(bindings.size()),
      writes.data(),
      0,
      nullptr);
  trace::counters().vk_descriptor_update_writes += bindings.size();

  ensure_recording();
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
      ranges[i] = {bindings[i].buffer,
                   bindings[i].offset,
                   tracked_range_end(bindings[i].offset, bindings[i].range)};
    }
    std::span<const TrackedRange> view{ranges.data(), bindings.size()};
    if (!head_synced_ || batch_needs_barrier(view, view)) {
      record_dependency_barrier();
      trace::counters().barriers_emitted++;
      prof::get().on_barrier(true);
      barrier_recorded = true;
    } else {
      trace::counters().barriers_skipped++;
      prof::get().on_barrier(false);
    }
    for (size_t i = 0; i < bindings.size(); ++i) {
      tracked_reads_.push_back(ranges[i]);
      tracked_writes_.push_back(ranges[i]);
    }
    if (export_dep_masks()) {
      // Compute the two exported signals (design 22b395d): (a) whether
      // this dispatch's ranges are provably disjoint from everything
      // since the last explicit dependency (same proof
      // batch_needs_barrier just ran, inverted); (b) whether the launch
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
  dt.CmdBindPipeline(cmd_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
  dt.CmdBindDescriptorSets(
      cmd_,
      VK_PIPELINE_BIND_POINT_COMPUTE,
      pipeline_layout,
      0,
      1,
      &descriptor_set,
      0,
      nullptr);
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

void CommandEncoder::commit() {
  if (!recording_ && wait_semaphores_.empty() && signal_semaphores_.empty() &&
      completed_handlers_.empty()) {
    trace::counters().commit_calls_noop++;
    fprintf(stderr, "[rtmod] COMMIT-NOOP\n");
    return;
  }
  trace::counters().commit_calls_with_work++;
  submit();
}

void CommandEncoder::synchronize(const char* reason) {
  commit();
  join_last_completion(reason);
}

void CommandEncoder::wait_outstanding_submissions() {
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

void CommandEncoder::submit() {
  auto& dt = vk::device_table();
  if (std::getenv("MLX_OMARCHY_TRACE_DISPATCH")) {
    fprintf(stderr, "[rtmod] SUBMIT-ENTER tid=%lu\n", (unsigned long)syscall(SYS_gettid));
  }
  bool was_recording = recording_;
  uint64_t submit_t0 = prof::get().profiling() ? prof::host_ns() : 0;
  uint64_t close_t = 0;
  uint64_t queue_t0 = 0;
  uint64_t queue_t1 = 0;
  uint64_t submitted = 0;

  if (recording_) {
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
    VKX_CHECK(dt.EndCommandBuffer(cmd_));
    close_t = prof::get().profiling() ? prof::host_ns() : 0;
  }

  std::vector<VkSemaphore> wait_sems;
  std::vector<uint64_t> wait_values;
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
  std::vector<VkPipelineStageFlags> wait_stages(
      wait_sems.size(), VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);

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
  std::vector<VkSemaphore> signal_sems;
  std::vector<uint64_t> signal_values;
  signal_sems.reserve(signal_semaphores_.size() + 1);
  signal_values.reserve(signal_semaphores_.size() + 1);
  for (auto& pending : signal_semaphores_) {
    signal_sems.push_back(pending.semaphore);
    signal_values.push_back(pending.value);
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
    std::lock_guard<std::mutex> lk(device_.queue_mutex());
    // Khronos guidance: HOST_VISIBLE memory without HOST_COHERENT needs an
    // explicit flush before submission.
    omarchy::allocator().flush_noncoherent(device_.handle());
    uint64_t completion_value = device_.completions().reserve();
    // Stamp every buffer referenced by this batch with the completion
    // value just reserved. Free-before-drain sends such buffers to the
    // allocator quarantine; release_quarantine recycles them one
    // generation later. Semaphore keepalives still move into the
    // dispatcher payload below.
    omarchy::allocator().stamp_batch(batch_buffers_, completion_value);
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
        const char* e = std::getenv(name);
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
        VKX_CHECK(dt.QueueSubmit(device_.queue(), 1, &si, VK_NULL_HANDLE));
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
      recording_ = false;
      node_count_ = 0;
      wait_semaphores_.clear();
      signal_semaphores_.clear();
      completed_handlers_.clear();
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
    device_.completions().enqueue(
        completion_value,
        std::move(keepalive),
        std::move(completed_handlers_),
        was_recording ? slots_[current_slot_].started : VK_NULL_HANDLE);
    last_completion_ = completion_value;
    submitted = completion_value;
    if (was_recording) {
      slots_[current_slot_].in_flight = completion_value;
    }
    trace::counters().vk_submissions++;
    fprintf(stderr, "[rtmod] SUBMIT tid=%lu cv=%lu waits=%lu sigs=%lu cmds=%u\n",
            (unsigned long)syscall(SYS_gettid), (unsigned long)completion_value, (unsigned long)wait_sems.size(),
            (unsigned long)signal_values.size(),
            (unsigned)si.commandBufferCount);
  }

  recording_ = false;
  node_count_ = 0;
  wait_semaphores_.clear();
  signal_semaphores_.clear();
  completed_handlers_.clear();
  // The submission's in-order wait (last_completion_) provides the
  // cross-submission dependency, so the open batch's unsynced ranges
  // die here either way.
  reset_dependency_tracking();
  prof::get().on_submit_boundary(submitted, close_t, queue_t0, queue_t1);
  prof::get().on_submit_end(
      this,
      submitted,
      submit_t0 != 0 ? prof::host_ns() - submit_t0 : 0,
      current_slot_);

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
  return it->second;
}

std::unordered_map<int, CommandEncoder>& get_command_encoders() {
  static thread_local std::unordered_map<int, CommandEncoder> encoders;
  return encoders;
}

std::unordered_map<int, CommandEncoder>& get_global_command_encoders() {
  static std::unordered_map<int, CommandEncoder> global_encoders;
  return global_encoders;
}

} // namespace mlx::core::omarchy
