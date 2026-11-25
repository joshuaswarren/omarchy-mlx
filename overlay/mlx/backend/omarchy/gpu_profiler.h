// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

#pragma once

// Env-gated GPU profiling harness. COMPILE-TIME gated: without
// MLX_OMARCHY_GPU_PROFILING defined at library build time this header
// compiles to no-op inline stubs and release builds carry zero profiling
// code (no branches in the dispatch path, no getenv). With it defined,
// set MLX_OMARCHY_GPU_PROFILE=<path> (and optionally
// MLX_OMARCHY_GPU_PROFILE_LABEL=<name>) to append one NDJSON event stream
// to <path> for the life of the process. Set
// MLX_OMARCHY_GPU_PROFILE_DISPATCH=0 for buffer-level mode (default is
// per-dispatch mode):
//
//   {"k":"meta",...}  once: device name, timestamp period and valid bits,
//                     pool capacity, label, host clock start, wheel version
//                     and local tag of the loaded library
//   {"k":"kn",...}    once, after meta: ordered kernel-name table from the
//                     loaded library ("d".e indexes into it); kills
//                     compute.h enum drift between profile and analyzer
//   {"k":"b",...}     per command buffer begin: host cost of the ring
//                     slot wait (when all slots are executing) plus
//                     BeginCommandBuffer, and the host clock at begin.
//                     Buffer mode emits this record at flush instead and
//                     adds "bt0"/"bt1" (query-pool cursor indices) and
//                     "t0"/"t1" (the buffer's GPU ticks, omitted when
//                     unavailable)
//   {"k":"d",...}     per compute dispatch (per-dispatch mode only):
//                     kernel enum, groups,
//                     params.count, host record cost, raw GPU ticks (t0 at
//                     TOP_OF_PIPE, parked on pending_start_ts so it stamps
//                     the dispatch's own stream start; t1 after the dispatch
//                     at BOTTOM_OF_PIPE on the same stream's tail), the binding
//                     list (buffer, offset, range) used as the dependency
//                     proxy between consecutive dispatches, and "bar" (1
//                     when the dispatch recorded a dependency barrier,
//                     0 when gated tracking skipped it)
//   {"k":"s",...}     per submission: host cost of submit() and the host
//                     clock at submit end
//   {"k":"q",...}     per submission: command-buffer close and queue-submit
//                     entry/return host clocks
//   {"k":"j",...}     per join: host cost of the completion-timeline wait
//                     and of the noncoherent invalidate, plus the caller reason
//   {"k":"end",...}   at exit: totals incl. barrier decisions ("barriers",
//                     "barriers_skipped"; transfer/fill decisions included);
//                     buffer mode adds "dispatches_total" (per-dispatch
//                     mode accounts dispatches through the k=d records)
//   {"k":"cal",...}   once, after meta: the calibrated submission floor —
//                     median host cost of an empty command buffer submitted
//                     alone and waited ("floor_ns"), and its sample count
//
// The k=cal record adopts aurora-lily's timestamp+calibration model
// (src/vulkan.rs ProfileStamps + calibrate_profile): lily measures the
// ~150 us fixed cost of one lone submit+wait and subtracts it from every
// host-timed kernel sample. Our per-dispatch GPU ticks already bracket
// the dispatch inside the command buffer, so job cost is not in k=d
// spans; floor_ns exists so analysis can subtract the same fixed cost
// from the host-side submit/join spans (k=s, k=q, k=j) and so the
// figure itself is on record per driver.
//
// Calibration submits are strictly command-free: a command buffer that
// carries only a timestamp write submits empty on compute-only chips
// (#526), so no GPU-tick epoch is anchored here.
//
// GPU ticks convert to nanoseconds with meta.period_ns; tick wraparound
// wraps at 2^valid_bits. Kernel enum values map to names by their
// declaration order in overlay/mlx/backend/omarchy/compute.h (the
// analysis script parses that header; no name table lives in C++).
//
// Per-dispatch mode (the default) adds two timestamps and an execution
// barrier per dispatch; buffer mode (MLX_OMARCHY_GPU_PROFILE_DISPATCH=0)
// adds one BOTTOM_OF_PIPE timestamp pair per command buffer (t0 after
// BeginCommandBuffer, t1 before EndCommandBuffer) and no per-dispatch
// commands. Both add a query-pool reset per command buffer. In
// per-dispatch mode t0 uses TOP_OF_PIPE: a BOTTOM_OF_PIPE write after the
// previous dispatch's t1 cannot ride a stream that already carries an end
// timestamp, so it would need a command-free timestamp-only stream, which
// submits empty on compute-only chips (#526). The barrier isolates the
// dispatch from later compute work in Honeykrisp's timestamp-bearing
// stream. These commands
// perturb scheduling; profiled timings are not release performance. Results
// are read during join_last_completion (which completes every submission
// of the encoder at once) or when the encoder reuses a completed ring
// slot, and only then. Every hook is main-thread only (recording and
// joining both happen on the encoder's thread; the completion thread
// never calls in), so there is no lock. All Vulkan entry points go
// through the dlopened device table; nothing links libvulkan directly.

#ifdef MLX_OMARCHY_GPU_PROFILING

#include <vulkan/vulkan.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cinttypes>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string_view>
#include <memory>
#include <span>
#include <unordered_map>
#include <vector>

#include "mlx/backend/omarchy/compute.h"
#include "mlx/backend/omarchy/trace.h"
#include "mlx/backend/omarchy/device.h"
#include "kernel_names.h"

#ifndef MLX_OMARCHY_PROFILE_VERSION
#define MLX_OMARCHY_PROFILE_VERSION "unknown"
#endif
#ifndef MLX_OMARCHY_PROFILE_TAG
#define MLX_OMARCHY_PROFILE_TAG ""
#endif
#include "mlx/backend/omarchy/vulkan.h"

namespace mlx::core::omarchy::prof {

inline uint64_t host_ns() {
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now().time_since_epoch())
          .count());
}

class GpuProfiler {
 public:
  static GpuProfiler& get() {
    static GpuProfiler instance;
    return instance;
  }

  bool profiling() const {
    return out_ != nullptr;
  }

  // Called once per CommandEncoder construction. Keyed by the encoder
  // pointer; a destroyed encoder leaves its context (and query pools) for
  // process exit rather than touching Vulkan after teardown.
  void attach(const void* owner, const Device& device) {
    if (!out_) {
      open_output(device);
    }
    if (!out_ || valid_bits_ == 0) {
      return;
    }
    auto ctx = std::make_unique<Ctx>();
    bool all_created = true;
    for (int i = 0; i < kProfilerSlots; ++i) {
      VkQueryPoolCreateInfo info{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
      info.queryType = VK_QUERY_TYPE_TIMESTAMP;
      info.queryCount = kPoolQueries;
      if (vk::device_table().CreateQueryPool(
              device.handle(), &info, nullptr, &ctx->slots[i].pool) !=
          VK_SUCCESS) {
        all_created = false;
        break;
      }
    }
    if (all_created) {
      contexts_[owner] = std::move(ctx);
    } else {
      std::fprintf(stderr, "[prof] vkCreateQueryPool failed; GPU ticks off\n");
    }
  }

  // Buffer mode defers the k=b record to flush (it gains the buffer's
  // GPU tick pair there) and records t0 right after BeginCommandBuffer.
  // Per-dispatch mode flushes the previous batch's unread timestamps here
  // (slot reuse implies completion) and resets the pool for the new batch.
  void on_begin(
      const void* owner,
      int slot,
      VkCommandBuffer cmd,
      uint64_t begin_cost) {
    if (out_ == nullptr) {
      return;
    }
    Ctx* ctx = find(owner);
    if (!dispatch_enabled_) {
      if (ctx == nullptr) {
        emitf("{\"k\":\"b\",\"o\":%" PRIu64 ",\"dur\":%" PRIu64
              ",\"t\":%" PRIu64 "}\n",
              owner_id(owner),
              begin_cost,
              host_ns());
        return;
      }
      flush_slot(*ctx, slot, ctx->slots[slot].last_sub);
      SlotCtx& s = ctx->slots[slot];
      s.cursor = 0;
      s.pending.clear();
      s.buffers.clear();
      vk::device_table().CmdResetQueryPool(cmd, s.pool, 0, kPoolQueries);
      PendingBuffer p{};
      p.owner = owner_id(owner);
      p.begin_cost = begin_cost;
      p.begin_t = host_ns();
      p.bt0 = s.cursor;
      p.bt1 = p.bt0 + 1;
      vk::device_table().CmdWriteTimestamp(
          cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, s.pool, s.cursor);
      s.cursor = 1; // bt1 is written by on_buffer_end before EndCommandBuffer
      s.buffers.push_back(p);
      return;
    }
    emitf("{\"k\":\"b\",\"o\":%" PRIu64 ",\"dur\":%" PRIu64
          ",\"t\":%" PRIu64 "}\n",
          owner_id(owner),
          begin_cost,
          host_ns());
    if (ctx == nullptr) {
      return;
    }
    flush_slot(*ctx, slot, ctx->slots[slot].last_sub);
    SlotCtx& s = ctx->slots[slot];
    s.cursor = 0;
    s.pending.clear();
    vk::device_table().CmdResetQueryPool(cmd, s.pool, 0, kPoolQueries);
  }

  // Called after the pre-dispatch barrier (when one was recorded) and
  // before the dispatch itself. t0 is TOP_OF_PIPE so the driver attaches
  // it to the start of the next command-bearing stream; it lands when
  // prior GPU work (including the barrier) completes and the dispatch
  // stream begins. barrier reports whether this dispatch recorded a
  // dependency barrier (gated mode may skip it).
  void before_dispatch(
      const void* owner,
      int slot,
      VkCommandBuffer cmd,
      bool barrier) {
    if (out_ == nullptr) {
      return;
    }
    if (!dispatch_enabled_) {
      dispatches_total_++;
      return;
    }
    Ctx* ctx = find(owner);
    if (ctx == nullptr) {
      return;
    }
    SlotCtx& s = ctx->slots[slot];
    PendingDispatch p{};
    p.skipped = s.cursor + 2 > kPoolQueries;
    p.bar = barrier ? 1u : 0u;
    if (p.skipped) {
      dropped_++;
    } else {
      p.tick_index = s.cursor;
      vk::device_table().CmdWriteTimestamp(
          cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, s.pool, s.cursor);
      vk::device_table().CmdPipelineBarrier(
          cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
          VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr, 0, nullptr);
    }
    s.pending.push_back(p);
  }

  // Dependency-barrier decision counter (dispatch pre/post and
  // transfer/fill decisions; excludes TAPE_FULL_BARRIERS diagnostics).
  void on_barrier(bool emitted) {
    if (out_ == nullptr) {
      return;
    }
    if (emitted) {
      barriers_emitted_++;
    } else {
      barriers_skipped_++;
    }
  }

  // Called right after the dispatch (before the post-dispatch barrier), so
  // t1 lands when the dispatch completes.
  void after_dispatch(
      const void* owner,
      int slot,
      VkCommandBuffer cmd,
      ComputeKernel kernel,
      const ComputeParams& params,
      std::span<const ComputeBinding> bindings,
      uint32_t gx,
      uint32_t gy,
      uint32_t gz,
      uint64_t host_cost,
      uint32_t tape) {
    if (out_ == nullptr || !dispatch_enabled_) {
      return;
    }
    Ctx* ctx = find(owner);
    if (ctx == nullptr) {
      return;
    }
    SlotCtx& s = ctx->slots[slot];
    PendingDispatch& p = s.pending.back();
    p.kernel = static_cast<uint32_t>(kernel);
    p.operation = params.operation;
    p.count = params.count;
    p.gx = gx;
    p.gy = gy;
    p.gz = gz;
    p.host_cost = host_cost;
    p.tape = tape;
    {
      std::string_view prim = trace::current_prim();
      std::memcpy(p.prim.data(), prim.data(), std::min(prim.size(), p.prim.size() - 1));
      p.prim[std::min(prim.size(), p.prim.size() - 1)] = 0;
    }
    p.nb = std::min(bindings.size(), p.bind.size());
    for (size_t i = 0; i < p.nb; ++i) {
      p.bind[i] = {reinterpret_cast<uintptr_t>(bindings[i].buffer),
                   bindings[i].offset,
                   bindings[i].range};
    }
    if (!p.skipped) {
      vk::device_table().CmdWriteTimestamp(
          cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, s.pool, p.tick_index + 1);
      s.cursor = p.tick_index + 2;
    }
    dispatches_++;
  }

  // Buffer mode only: writes the buffer's closing timestamp; called by the
  // encoder right before EndCommandBuffer.
  void on_buffer_end(const void* owner, int slot, VkCommandBuffer cmd) {
    if (out_ == nullptr || dispatch_enabled_) {
      return;
    }
    Ctx* ctx = find(owner);
    if (ctx == nullptr) {
      return;
    }
    SlotCtx& s = ctx->slots[slot];
    if (s.buffers.empty()) {
      return;
    }
    PendingBuffer& p = s.buffers.back();
    vk::device_table().CmdWriteTimestamp(
        cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, s.pool, p.bt1);
    s.cursor = p.bt1 + 1;
  }

  // Called at the end of a successful submit(); sub is 0 for submissions
  // that carried no completion signal. slot is -1 when the submit carried
  // no command buffer: that slot's dispatches belong to its own submission.
  void on_submit_end(
      const void* owner,
      uint64_t sub,
      uint64_t submit_cost,
      int slot) {
    if (out_ == nullptr || sub == 0) {
      return;
    }
    submissions_++;
    Ctx* ctx = find(owner);
    if (ctx != nullptr && slot >= 0) {
      ctx->slots[slot].last_sub = sub;
    }
    emitf("{\"k\":\"s\",\"s\":%" PRIu64 ",\"dur\":%" PRIu64
          ",\"t\":%" PRIu64 "}\n",
          sub,
          submit_cost,
          host_ns());
  }

  void on_submit_boundary(
      uint64_t sub,
      uint64_t close_t,
      uint64_t queue_t0,
      uint64_t queue_t1) {
    if (out_ == nullptr || sub == 0) {
      return;
    }
    emitf("{\"k\":\"q\",\"s\":%" PRIu64
          ",\"close\":%" PRIu64 ",\"queue_t0\":%" PRIu64
          ",\"queue_t1\":%" PRIu64 "}\n",
          sub,
          close_t,
          queue_t0,
          queue_t1);
  }

  // Called from join_last_completion after the wait and the noncoherent
  // invalidate. The join completed the newest submission, which is every
  // submission this encoder has outstanding, so all ring slots' unread
  // timestamps are now available and get flushed here.
  void on_join(
      const void* owner,
      uint64_t sub,
      uint64_t join_t0,
      uint64_t wait_t1,
      uint64_t inval_t2,
      const char* reason) {
    if (out_ == nullptr) {
      return;
    }
    joins_++;
    emitf("{\"k\":\"j\",\"s\":%" PRIu64 ",\"wait\":%" PRIu64
          ",\"inval\":%" PRIu64 ",\"t\":%" PRIu64
          ",\"reason\":\"%s\"}\n",
          sub,
          wait_t1 - join_t0,
          inval_t2 - wait_t1,
          join_t0,
          reason);
    Ctx* ctx = find(owner);
    if (ctx == nullptr) {
      return;
    }
    for (int i = 0; i < kProfilerSlots; ++i) {
      flush_slot(*ctx, i, ctx->slots[i].last_sub);
    }
  }

 private:
  static constexpr int kProfilerSlots = 4; // matches the encoder ring
  static constexpr size_t kMaxRecordedBindings = 5;

  struct PendingDispatch {
    uint32_t kernel{0};
    uint32_t operation{0};
    uint32_t count{0};
    uint32_t gx{0};
    uint32_t gy{0};
    uint32_t gz{0};
    uint64_t host_cost{0};
    uint32_t tick_index{0};
    // buffer, offset, range triples for the dependency proxy between
    // consecutive dispatches: overlapping buffers suggest the pair may be
    // data dependent; disjoint buffers prove the barrier between them
    // unnecessary.
    std::array<std::array<uint64_t, 3>, kMaxRecordedBindings> bind{};
    size_t nb{0};
    bool skipped{false};
    uint32_t tape{0};
    // 1 when this dispatch recorded a dependency barrier, 0 when the
    // gated tracker skipped it.
    uint32_t bar{0};
    // Consuming primitive name (trace::current_prim at record time).
    std::array<char, 48> prim{};
  };

  // Buffer mode: one per command buffer; the k=b record emitted at flush.
  struct PendingBuffer {
    uint64_t owner{0};
    uint64_t begin_cost{0};
    uint64_t begin_t{0};
    uint32_t bt0{0};
    uint32_t bt1{0};
  };

  // Per-ring-slot recording state: one query pool per slot so an in-flight
  // batch's timestamps survive while a newer batch records.
  struct SlotCtx {
    VkQueryPool pool{VK_NULL_HANDLE};
    uint32_t cursor{0};
    uint64_t last_sub{0};
    std::vector<PendingDispatch> pending;
    std::vector<PendingBuffer> buffers;
  };

  struct Ctx {
    std::array<SlotCtx, kProfilerSlots> slots;
  };

  // Generous per-command-buffer budget: 32768 dispatches. Overflow drops
  // timestamps (counted) rather than changing recording behavior.
  static constexpr uint32_t kPoolQueries = 1u << 16;

  GpuProfiler() = default;
  ~GpuProfiler() {
    if (out_ == nullptr) {
      return;
    }
    emitf("{\"k\":\"end\",\"t\":%" PRIu64 ",\"dispatches\":%" PRIu64
          ",\"dropped\":%u,\"submissions\":%" PRIu64 ",\"joins\":%" PRIu64
          ",\"barriers\":%" PRIu64 ",\"barriers_skipped\":%" PRIu64,
          host_ns(),
          dispatches_,
          dropped_,
          submissions_,
          joins_,
          barriers_emitted_,
          barriers_skipped_);
    if (!dispatch_enabled_) {
      emitf(",\"dispatches_total\":%" PRIu64, dispatches_total_);
    }
    emitf("}\n");
    std::fclose(out_);
    out_ = nullptr;
  }

  GpuProfiler(const GpuProfiler&) = delete;
  GpuProfiler& operator=(const GpuProfiler&) = delete;

  // Read back and emit one slot's pending buffer spans and dispatches.
  // Caller guarantees the slot's submission has completed (slot reuse or a
  // full join).
  void flush_slot(Ctx& ctx, int slot, uint64_t sub) {
    SlotCtx& s = ctx.slots[slot];
    if (s.pending.empty() && s.buffers.empty()) {
      s.cursor = 0;
      return;
    }
    if (valid_bits_ == 0 || s.last_sub == 0) {
      s.pending.clear();
      s.buffers.clear();
      s.cursor = 0;
      return;
    }
    uint32_t queries = s.cursor;
    std::vector<uint64_t> ticks;
    if (queries > 0) {
      ticks.resize(queries);
      VkResult result = vk::device_table().GetQueryPoolResults(
          device_,
          s.pool,
          0,
          queries,
          queries * sizeof(uint64_t),
          ticks.data(),
          sizeof(uint64_t),
          VK_QUERY_RESULT_64_BIT);
      if (result != VK_SUCCESS) {
        s.pending.clear();
        s.buffers.clear();
        s.cursor = 0;
        return;
      }
    }
    for (size_t i = 0; i < s.buffers.size(); ++i) {
      const PendingBuffer& p = s.buffers[i];
      emitf("{\"k\":\"b\",\"o\":%" PRIu64 ",\"dur\":%" PRIu64
            ",\"t\":%" PRIu64 ",\"bt0\":%" PRIu32 ",\"bt1\":%" PRIu32,
            p.owner,
            p.begin_cost,
            p.begin_t,
            p.bt0,
            p.bt1);
      if (p.bt1 < ticks.size() && ticks[p.bt1] >= ticks[p.bt0]) {
        emitf(",\"t0\":%" PRIu64 ",\"t1\":%" PRIu64,
              ticks[p.bt0],
              ticks[p.bt1]);
      }
      emitf("}\n");
    }
    for (size_t i = 0; i < s.pending.size(); ++i) {
      const PendingDispatch& p = s.pending[i];
      if (p.skipped) {
        continue;
      }
      emitf("{\"k\":\"d\",\"s\":%" PRIu64 ",\"e\":%u,\"op\":%u,\"n\":%u"
            ",\"gx\":%u,\"gy\":%u,\"gz\":%u,\"h\":%" PRIu64 ",\"tp\":%u"
            ",\"bar\":%u,\"p\":\"%s\"",
            sub,
            p.kernel,
            p.operation,
            p.count,
            p.gx,
            p.gy,
            p.gz,
            p.host_cost,
            p.tape,
            p.bar,
            p.prim.data());
      if (p.tick_index + 1 < queries &&
          ticks[p.tick_index + 1] >= ticks[p.tick_index]) {
        emitf(",\"t0\":%" PRIu64 ",\"t1\":%" PRIu64,
              ticks[p.tick_index],
              ticks[p.tick_index + 1]);
      }
      emitf(",\"b\":[");
      for (size_t j = 0; j < p.nb; ++j) {
        emitf("%s[%" PRIu64 ",%u,%u]",
              j > 0 ? "," : "",
              p.bind[j][0],
              p.bind[j][1],
              p.bind[j][2]);
      }
      emitf("]}\n");
    }
    s.pending.clear();
    s.buffers.clear();
    s.cursor = 0;
  }

  void open_output(const Device& device) {
    const char* path = std::getenv("MLX_OMARCHY_GPU_PROFILE");
    if (path == nullptr || path[0] == '\0') {
      return;
    }
    out_ = std::fopen(path, "w");
    if (out_ == nullptr) {
      std::fprintf(
          stderr, "[prof] cannot open MLX_OMARCHY_GPU_PROFILE path %s\n", path);
      return;
    }
    const char* dispatch_env = std::getenv("MLX_OMARCHY_GPU_PROFILE_DISPATCH");
    dispatch_enabled_ =
        dispatch_env == nullptr || std::strcmp(dispatch_env, "0") != 0;
    valid_bits_ = device.capabilities().queue_timestamp_valid_bits;
    period_ns_ = device.capabilities().timestamp_period;
    board_blocks_ticks_ = !device.capabilities().submission_timestamps;
    if (board_blocks_ticks_) {
      // The board's compute admission check refuses a timestamp-bearing
      // submit (#526: -EINVAL surfaced as VK_ERROR_DEVICE_LOST). No
      // timestamp means no GPU ticks: host costs only, and no query pool
      // is created, so no submit carries a timestamp command.
      valid_bits_ = 0;
    }
    device_ = device.handle();
    host_t0_ = host_ns();
    const char* label = std::getenv("MLX_OMARCHY_GPU_PROFILE_LABEL");
    emitf("{\"k\":\"meta\",\"device\":\"%s\",\"period_ns\":%.6f"
          ",\"valid_bits\":%u,\"pool\":%u,\"label\":\"%s\",\"host_t0\":%"
          PRIu64 ",\"wheel\":\"%s\",\"tag\":\"%s\"}\n",
          device.capabilities().device_name.c_str(),
          static_cast<double>(period_ns_),
          valid_bits_,
          kPoolQueries,
          label != nullptr ? label : "",
          host_t0_,
          MLX_OMARCHY_PROFILE_VERSION,
          MLX_OMARCHY_PROFILE_TAG);
    // Kernel names resolved from the loaded library at record time; a
    // compute.h from another commit cannot rename them.
    emitf("{\"k\":\"kn\",\"names\":[");
    for (size_t i = 0;
         i < sizeof(kComputeKernelNames) / sizeof(kComputeKernelNames[0]);
         ++i) {
      emitf("%s\"%s\"", i ? "," : "", kComputeKernelNames[i]);
    }
    emitf("]}\n");
    if (board_blocks_ticks_) {
      std::fprintf(
          stderr,
          "[prof] board cannot carry per-submission timestamps; host costs only\n");
    } else if (valid_bits_ == 0) {
      std::fprintf(
          stderr,
          "[prof] queue reports no timestamp support; host costs only\n");
    }
    calibrate(device);
  }

  // aurora-lily's calibrate_profile (src/vulkan.rs): measure the fixed
  // cost of one lone submit+wait of an empty command buffer and record
  // it as the profiling floor. Buffers stay strictly command-free — a
  // timestamp-only stream submits empty on compute-only chips (#526), so
  // no stamped epoch is anchored here.
  void calibrate(const Device& device) {
    if (calibrated_) {
      return;
    }
    calibrated_ = true;
    auto& dt = vk::device_table();
    if (!dt.CreateCommandPool || !dt.AllocateCommandBuffers ||
        !dt.BeginCommandBuffer || !dt.EndCommandBuffer ||
        !dt.ResetCommandBuffer || !dt.QueueSubmit || !dt.QueueWaitIdle ||
        !dt.FreeCommandBuffers || !dt.DestroyCommandPool) {
      return;
    }
    VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pci.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT |
        VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pci.queueFamilyIndex = device.queue_family();
    VkCommandPool pool = VK_NULL_HANDLE;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    bool ok = dt.CreateCommandPool(device.handle(), &pci, nullptr, &pool) ==
        VK_SUCCESS;
    if (ok) {
      VkCommandBufferAllocateInfo ai{
          VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
      ai.commandPool = pool;
      ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
      ai.commandBufferCount = 1;
      ok = dt.AllocateCommandBuffers(device.handle(), &ai, &cmd) ==
          VK_SUCCESS;
    }
    constexpr int kWarmup = 2;
    constexpr int kSamples = 32;
    std::vector<uint64_t> spans;
    if (ok) {
      // No MLX work exists at attach time, so calibration submits are the
      // only traffic; take the queue mutex anyway to match every other
      // submission site.
      std::lock_guard<std::mutex> queue_lock(
          const_cast<Device&>(device).queue_mutex());
      for (int i = 0; ok && i < kWarmup + kSamples; ++i) {
        VkCommandBufferBeginInfo bi{
            VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        ok = dt.ResetCommandBuffer(cmd, 0) == VK_SUCCESS &&
            dt.BeginCommandBuffer(cmd, &bi) == VK_SUCCESS &&
            dt.EndCommandBuffer(cmd) == VK_SUCCESS;
        if (!ok) {
          break;
        }
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.commandBufferCount = 1;
        si.pCommandBuffers = &cmd;
        uint64_t t0 = host_ns();
        ok = dt.QueueSubmit(device.queue(), 1, &si, VK_NULL_HANDLE) ==
                VK_SUCCESS &&
            dt.QueueWaitIdle(device.queue()) == VK_SUCCESS;
        if (ok && i >= kWarmup) {
          spans.push_back(host_ns() - t0);
        }
      }
    }
    if (cmd != VK_NULL_HANDLE) {
      dt.FreeCommandBuffers(device.handle(), pool, 1, &cmd);
    }
    if (pool != VK_NULL_HANDLE) {
      dt.DestroyCommandPool(device.handle(), pool, nullptr);
    }
    if (spans.empty()) {
      return;
    }
    std::sort(spans.begin(), spans.end());
    // Even sample count: average the two middle samples; a partial run
    // that left an odd count takes the middle one.
    size_t n = spans.size();
    floor_ns_ = n % 2 == 0 ? (spans[n / 2 - 1] + spans[n / 2]) / 2
                           : spans[n / 2];
    emitf("{\"k\":\"cal\",\"floor_ns\":%" PRIu64 ",\"samples\":%zu"
          ",\"warmup\":%d}\n",
          floor_ns_,
          spans.size(),
          kWarmup);
  }

  Ctx* find(const void* owner) {
    auto it = contexts_.find(owner);
    return it != contexts_.end() ? it->second.get() : nullptr;
  }

  uint64_t owner_id(const void* owner) {
    return static_cast<uint64_t>(reinterpret_cast<uintptr_t>(owner));
  }

  void emitf(const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    std::vfprintf(out_, fmt, args);
    va_end(args);
  }

  std::unordered_map<const void*, std::unique_ptr<Ctx>> contexts_;
  VkDevice device_{VK_NULL_HANDLE};
  FILE* out_{nullptr};
  uint32_t valid_bits_{0};
  float period_ns_{1.0f};
  uint64_t host_t0_{0};
  bool dispatch_enabled_{true};
  bool board_blocks_ticks_{false};
  uint64_t floor_ns_{0};
  bool calibrated_{false};
  uint64_t dispatches_{0};
  uint64_t dispatches_total_{0};
  uint64_t submissions_{0};
  uint64_t joins_{0};
  uint32_t dropped_{0};
  uint64_t barriers_emitted_{0};
  uint64_t barriers_skipped_{0};
};

// Namespace-level accessor used by encoder.cpp call sites.
inline GpuProfiler& get() {
  return GpuProfiler::get();
}

} // namespace mlx::core::omarchy::prof

#else

// Compiled-out mirror: identical call signatures, zero cost. mlx builds
// without MLX_OMARCHY_GPU_PROFILING never touch Vulkan profiling state.

#include <cstdint>

namespace mlx::core::omarchy::prof {

class GpuProfiler {
 public:
  static GpuProfiler& get() {
    static GpuProfiler instance;
    return instance;
  }

  bool profiling() const {
    return false;
  }

  void attach(const void*, const Device&) {}
  void on_begin(const void*, int, VkCommandBuffer, uint64_t) {}
  void before_dispatch(const void*, int, VkCommandBuffer, bool) {}
  void on_buffer_end(const void*, int, VkCommandBuffer) {}
  void on_barrier(bool) {}
  void after_dispatch(
      const void*,
      int,
      VkCommandBuffer,
      ComputeKernel,
      const ComputeParams&,
      std::span<const ComputeBinding>,
      uint32_t,
      uint32_t,
      uint32_t,
      uint64_t,
      uint32_t) {}
  void on_submit_end(const void*, uint64_t, uint64_t, int) {}
  void on_submit_boundary(uint64_t, uint64_t, uint64_t, uint64_t) {}
  void on_join(
      const void*, uint64_t, uint64_t, uint64_t, uint64_t, const char*) {}
};

// Namespace-level accessor used by encoder.cpp call sites.
inline GpuProfiler& get() {
  return GpuProfiler::get();
}

inline uint64_t host_ns() {
  return 0;
}

} // namespace mlx::core::omarchy::prof

#endif
