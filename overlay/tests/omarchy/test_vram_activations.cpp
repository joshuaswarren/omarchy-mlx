// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

#include "device_local_activations.h"

#include <cstdio>

using namespace mlx::core::omarchy;

namespace {

int check(bool cond, const char* msg) {
  if (!cond) {
    std::fprintf(stderr, "FAIL: %s\n", msg);
    return 1;
  }
  return 0;
}

} // namespace

int main() {
  int fails = 0;

  // 1. Memory type selection: discrete + opt-in + staging required.
  VkPhysicalDeviceMemoryProperties discrete{};
  discrete.memoryHeapCount = 2;
  discrete.memoryHeaps[0] = {16ull << 30, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT};
  discrete.memoryHeaps[1] = {56ull << 30, 0};
  discrete.memoryTypeCount = 3;
  discrete.memoryTypes[0] = {VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0};
  discrete.memoryTypes[1] = {VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, 1};
  discrete.memoryTypes[2] = {
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
      1};

  fails += check(
      vram_activation_memory_type(true, VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU, discrete, 7) == 0,
      "discrete opt-in selects pure DEVICE_LOCAL type 0");
  fails += check(
      vram_activation_memory_type(false, VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU, discrete, 7) == UINT32_MAX,
      "discrete with opt-in OFF refuses DEVICE_LOCAL activation path");
  fails += check(
      vram_activation_memory_type(true, VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU, discrete, 7) == UINT32_MAX,
      "integrated/UMA GPU never uses discrete staging path");
  fails += check(
      vram_activation_memory_type(true, VK_PHYSICAL_DEVICE_TYPE_CPU, discrete, 7) == UINT32_MAX,
      "software CPU Vulkan never uses discrete staging path");
  fails += check(
      max_device_local_heap_bytes(discrete) == (16ull << 30),
      "max_device_local_heap_bytes finds 16 GiB VRAM heap");

  // 2. Chunking within 4 MiB staging cap.
  const auto c0 = vram_staging_chunk(kWeightStagingBytes + 5, 0);
  const auto c1 = vram_staging_chunk(kWeightStagingBytes + 5, c0.byte_count);
  const auto c2 = vram_staging_chunk(kWeightStagingBytes + 5, c0.byte_count + c1.byte_count);
  fails += check(
      c0.byte_offset == 0 && c0.byte_count == kWeightStagingBytes &&
          c0.copy_bytes == kWeightStagingBytes,
      "first chunk fills 4 MiB staging cap");
  fails += check(
      c1.byte_offset == kWeightStagingBytes && c1.byte_count == 5 &&
          c1.copy_bytes == 8,
      "tail chunk rounds copy_bytes to 4 bytes");
  fails += check(c2.byte_count == 0 && c2.copy_bytes == 0, "end of buffer returns empty chunk");

  // 2b. Inline vkCmdUpdateBuffer eligibility (<= 65536 B, non-zero, 4-byte aligned)
  // and tiny scalar HOST_VISIBLE fast path (checked against unrounded request size).
  fails += check(!should_use_device_local_activation(true, 4),
                 "4-byte scalar stays HOST_VISIBLE for zero-stall pipelined readback");
  fails += check(!should_use_device_local_activation(true, 16),
                 "16-byte scalar stays HOST_VISIBLE for zero-stall pipelined readback");
  fails += check(should_use_device_local_activation(true, 80),
                 "80-byte activation uses DEVICE_LOCAL VRAM");
  fails += check(!should_use_device_local_activation(false, 4096),
                 "disabled opt-in never uses DEVICE_LOCAL VRAM");
  fails += check(!can_inline_update_device_local(0), "0 bytes cannot inline update");
  fails += check(can_inline_update_device_local(4), "4-byte scalar can inline update");
  fails += check(can_inline_update_device_local(256), "256-byte buffer can inline update");
  fails += check(can_inline_update_device_local(kInlineUpdateMaxBytes), "65536 B exact cap can inline update");
  fails += check(!can_inline_update_device_local(kInlineUpdateMaxBytes + 4), "above 65536 B falls back to staging");
  fails += check(!can_inline_update_device_local(6), "non-4-byte-aligned size falls back to staging");

  // 2c. Inline update timeline semaphore wait predicate.
  constexpr uint64_t kPending = ~uint64_t{0};
  fails += check(!needs_inline_update_semaphore_wait(0, kPending, 5, 3),
                 "zero completion needs no semaphore wait");
  fails += check(!needs_inline_update_semaphore_wait(kPending, kPending, 5, 3),
                 "pending sentinel is handled via same-encoder order or cross-encoder flush");
  fails += check(!needs_inline_update_semaphore_wait(5, kPending, 5, 3),
                 "completion <= last_completion on this stream needs no extra wait");
  fails += check(!needs_inline_update_semaphore_wait(7, kPending, 5, 7),
                 "already-drained completion needs no semaphore wait");
  fails += check(needs_inline_update_semaphore_wait(9, kPending, 5, 7),
                 "unretired cross-stream completion requires timeline semaphore wait");

  // 3. Wider 2x16 Qmm workgroup selection.
  QmmWideCaps rtx_caps{32u, VK_SUBGROUP_FEATURE_ARITHMETIC_BIT, 1024u, 1024u};
  QmmWideCaps narrow_wg{32u, VK_SUBGROUP_FEATURE_ARITHMETIC_BIT, 256u, 1024u};
  QmmWideCaps narrow_x{32u, VK_SUBGROUP_FEATURE_ARITHMETIC_BIT, 1024u, 256u};
  QmmWideCaps sg16{16u, VK_SUBGROUP_FEATURE_ARITHMETIC_BIT, 1024u, 1024u};
  QmmWideCaps no_arith{32u, VK_SUBGROUP_FEATURE_BASIC_BIT, 1024u, 1024u};

  fails += check(
      select_qmm_wide_2x16("NVIDIA GeForce RTX 5060 Ti", VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU, rtx_caps, true),
      "discrete GPU with opt-in and 512/sg32 caps selects 2x16");
  fails += check(
      !select_qmm_wide_2x16("NVIDIA GeForce RTX 5060 Ti", VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU, rtx_caps, false),
      "discrete GPU with opt-in OFF keeps existing 2x4 default");
  fails += check(
      !select_qmm_wide_2x16("NVIDIA GeForce RTX 5060 Ti", VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU, narrow_wg, true),
      "refuses 2x16 when max_workgroup_invocations < 512");
  fails += check(
      !select_qmm_wide_2x16("NVIDIA GeForce RTX 5060 Ti", VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU, narrow_x, true),
      "refuses 2x16 when max_workgroup_size_x < 512");
  fails += check(
      !select_qmm_wide_2x16("NVIDIA GeForce RTX 5060 Ti", VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU, sg16, true),
      "refuses 2x16 when subgroup_size != 32");
  fails += check(
      !select_qmm_wide_2x16("NVIDIA GeForce RTX 5060 Ti", VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU, no_arith, true),
      "refuses 2x16 without subgroup arithmetic");
  fails += check(
      !select_qmm_wide_2x16("Apple M1 Max (G13C C0)", VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU, narrow_wg, false),
      "G13C Apple default moves from 2x16 to 2x8w2");
  fails += check(
      !select_qmm_wide_2x16("Apple M1 (G13G B1)", VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU, rtx_caps, true),
      "G13G Apple default moves from 2x16 to 2x8w2");
  QmmWideCaps wg256{32u, VK_SUBGROUP_FEATURE_ARITHMETIC_BIT, 256u, 256u};
  QmmWideCaps sg64{64u, VK_SUBGROUP_FEATURE_ARITHMETIC_BIT, 1024u, 1024u};

  fails += check(
      select_qmm_2x8w2("Apple M1 Max (G13C C0)", VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU, rtx_caps, false),
      "Apple G13C selects 2x8w2");
  fails += check(
      select_qmm_2x8w2("Apple M1 (G13G B1)", VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU, rtx_caps, false),
      "Apple G13G selects 2x8w2");
  fails += check(
      select_qmm_2x8w2("Apple M2 Max (G14C B1)", VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU, rtx_caps, false),
      "Apple G14C B1 selects 2x8w2");
  fails += check(
      select_qmm_2x8w2("Apple M2 Max (G14C C0)", VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU, rtx_caps, false),
      "Apple G14C C0 selects 2x8w2");
  fails += check(
      !select_qmm_2x8w2("Apple M2 (G14G B0)", VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU, rtx_caps, false),
      "Apple G14G does not select 2x8w2");
  fails += check(
      !select_qmm_2x8w2("Apple M2 Pro (G14S B1)", VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU, rtx_caps, false),
      "Apple G14S does not select 2x8w2");
  fails += check(
      !select_qmm_2x8w2("llvmpipe (LLVM 19.1.7, 256 bits)", VK_PHYSICAL_DEVICE_TYPE_CPU, rtx_caps, false),
      "llvmpipe does not select 2x8w2");
  fails += check(
      select_qmm_2x8w2("NVIDIA GeForce RTX 5060 Ti", VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU, rtx_caps, true),
      "discrete GPU with VRAM activations and 256/sg32 caps selects 2x8w2");
  fails += check(
      select_qmm_2x8w2("AMD Radeon RX 9070", VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU, wg256, true),
      "discrete GPU limited to 256-thread workgroups still selects 2x8w2");
  fails += check(
      !select_qmm_2x8w2("NVIDIA GeForce RTX 5060 Ti", VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU, rtx_caps, false),
      "discrete GPU without VRAM activations does not select 2x8w2");
  fails += check(
      !select_qmm_2x8w2("AMD Radeon RX 9070", VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU, sg64, true),
      "subgroup-64 device does not select the 32-lane 2x8w2 kernels");
  fails += check(
      !select_qmm_2x8w2("NVIDIA GeForce RTX 5060 Ti", VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU, no_arith, true),
      "refuses 2x8w2 without subgroup arithmetic");
  fails += check(
      !select_qmm_2x8w2("Intel Arc B60 (BMG)", VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU, rtx_caps, true),
      "BMG keeps its measured 2x16kcont/kcs16 route");
  fails += check(
      !select_qmm_wide_2x16("Apple M2 Max (G14C C0)", VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU, rtx_caps, true),
      "Apple G14C does not select 2x16 even when opt-in env is set");
  fails += check(
      select_qmm_apple_r4s4w2_single_matrix("Apple M1 Max (G13C C0)"),
      "Apple G13C selects per-shape 4x4w2 single-matrix QMV");
  fails += check(
      !select_qmm_apple_r4s4w2_single_matrix("Apple M1 (G13G B1)"),
      "Apple G13G stays on 2x8w2");
  fails += check(
      !select_qmm_apple_r4s4w2_single_matrix("Apple M2 Max (G14C C0)"),
      "Apple G14C stays on 2x8w2 until its own paired A/B");
  fails += check(
      !select_qmm_apple_r4s4w2_single_matrix("NVIDIA GeForce RTX 5060 Ti"),
      "discrete GPU stays on 2x8w2");

  // 4. Host/device coherence state machine.
  {
    // Pure GPU intermediate: never touches host_raw_ptr -> 0 uploads, 0 downloads.
    DeviceLocalSyncState st{};
    fails += check(!st.on_gpu_access(false, true, 0, 4096, 4096), "fresh GPU write needs no upload");
    fails += check(st.device_dirty && !st.host_dirty, "GPU write marks device_dirty");
    fails += check(!st.on_gpu_access(true, false, 0, 4096, 4096), "GPU read of GPU-written buffer needs no upload");
    // Final host readback (e.g. int(tok)): downloads once, marks host_dirty.
    fails += check(st.on_host_raw_ptr(), "host raw_ptr after GPU write triggers download");
    fails += check(!st.device_dirty && st.host_dirty, "post-download clears device_dirty and sets host_dirty");
    // Second host raw_ptr without intervening GPU write: no redundant download.
    fails += check(!st.on_host_raw_ptr(), "second host raw_ptr skips redundant download");
    // Passing token back as GPU input uploads once.
    fails += check(st.on_gpu_access(true, false, 0, 4096, 4096), "GPU read after host raw_ptr triggers upload");
    fails += check(!st.host_dirty && !st.device_dirty, "post-upload clears host_dirty");
  }
  {
    // Fresh host-initialized tensor (mx.array(prompt_ids)): no download on raw_ptr, uploads on first GPU read.
    DeviceLocalSyncState st{};
    fails += check(!st.on_host_raw_ptr(), "fresh host write needs no download");
    fails += check(st.on_gpu_access(true, false, 0, 4096, 4096), "first GPU read uploads host writes");
    fails += check(!st.on_gpu_access(true, false, 0, 4096, 4096), "second GPU read skips upload");
  }
  {
    // Partial GPU write over host-dirty buffer must upload first so unwritten bytes survive.
    DeviceLocalSyncState st{};
    st.on_host_raw_ptr();
    fails += check(st.on_gpu_access(false, true, 128, 64, 4096), "partial GPU write uploads host_dirty buffer first");
    fails += check(st.device_dirty && !st.host_dirty, "after partial GPU write, device_dirty=true host_dirty=false");
  }
  {
    // Compute write-only descriptor spanning the whole allocation does NOT prove
    // full-buffer overwrite (e.g. offset Fill/slice shaders), so it must still
    // upload host_dirty bytes before clearing host_dirty.
    DeviceLocalSyncState st{};
    st.on_host_raw_ptr();
    fails += check(st.on_gpu_access(false, true, 0, 4096, 4096, false),
                   "compute write-only descriptor over full allocation still uploads host_dirty buffer");
    fails += check(st.device_dirty && !st.host_dirty,
                   "after compute write upload, device_dirty=true host_dirty=false");
  }
  {
    // Exact full-buffer transfer overwrite (copy_buffer/fill_buffer) skips
    // redundant upload; partial transfer still uploads.
    DeviceLocalSyncState partial{};
    partial.on_host_raw_ptr();
    fails += check(partial.on_gpu_access(false, true, 128, 64, 4096, true),
                   "partial exact transfer still uploads host_dirty buffer");
    DeviceLocalSyncState st{};
    st.on_host_raw_ptr();
    fails += check(!st.on_gpu_access(false, true, 0, 4096, 4096, true),
                   "full exact transfer overwrite skips redundant upload");
    fails += check(st.device_dirty && !st.host_dirty, "full overwrite marks device_dirty");
    st.on_recycle();
    fails += check(!st.device_dirty && !st.host_dirty, "recycle resets dirty flags");
  }
  {
    // Exception mid-upload must preserve host_dirty and never set device_dirty
    // (on read-only, partial-write, or sibling-write multi-binding dispatches).
    DeviceLocalSyncState read_st{};
    DeviceLocalSyncState partial_write_st{};
    DeviceLocalSyncState sibling_out_st{};
    read_st.on_host_raw_ptr();
    partial_write_st.on_host_raw_ptr();

    // Simulate upload throwing before commit_gpu_upload / commit_gpu_write.
    fails += check(read_st.needs_upload(true, false, 0, 4096, 4096),
                   "read-only query requests upload without mutating state");
    fails += check(read_st.host_dirty && !read_st.device_dirty,
                   "failed read-only upload preserves host_dirty=true device_dirty=false");
    fails += check(partial_write_st.needs_upload(false, true, 128, 64, 4096),
                   "partial-write query requests upload without mutating state");
    fails += check(partial_write_st.host_dirty && !partial_write_st.device_dirty,
                   "failed partial-write upload preserves host_dirty=true device_dirty=false");
    fails += check(!partial_write_st.needs_download(),
                   "next raw_ptr after failed partial-write upload will not download stale VRAM over host_shadow");
    fails += check(!sibling_out_st.device_dirty,
                   "sibling write target stays device_dirty=false when another binding upload throws");

    // Subsequent retry succeeds and commits cleanly.
    read_st.commit_gpu_upload();
    partial_write_st.commit_gpu_upload();
    partial_write_st.commit_gpu_write();
    sibling_out_st.commit_gpu_write();
    fails += check(!read_st.host_dirty && !read_st.device_dirty,
                   "committed read-only upload clears host_dirty");
    fails += check(!partial_write_st.host_dirty && partial_write_st.device_dirty,
                   "committed partial write sets device_dirty and clears host_dirty");
    fails += check(sibling_out_st.device_dirty,
                   "committed sibling write sets device_dirty");
  }

  // 5. Bounded staging-transfer fence state machine (timeout, in-flight preservation, recovery).
  {
    StagingTransferState ts{};
    uint64_t now = 1000;
    VkResult err = VK_SUCCESS;
    int slices = 0;

    // 5a. Normal completion on first slice clears in_flight.
    ts.mark_submitted();
    auto r = ts.wait_bounded_with_clock(
        500, 100,
        [&](uint64_t step) {
          ++slices;
          now += step;
          return VK_SUCCESS;
        },
        [&]() { return now; },
        &err);
    fails += check(r == StagingTransferWaitResult::Completed && err == VK_SUCCESS &&
                       !ts.in_flight && slices == 1,
                   "normal staging wait clears in_flight on first slice");

    // 5b. Multiple VK_TIMEOUT slices then VK_SUCCESS within budget completes cleanly.
    slices = 0;
    ts.mark_submitted();
    r = ts.wait_bounded_with_clock(
        500, 100,
        [&](uint64_t step) {
          ++slices;
          now += step;
          return slices < 3 ? VK_TIMEOUT : VK_SUCCESS;
        },
        [&]() { return now; },
        &err);
    fails += check(r == StagingTransferWaitResult::Completed && !ts.in_flight && slices == 3,
                   "staging wait tolerates transient timeouts within wall budget");

    // 5c. Exhausted wall budget returns TimedOut and preserves in_flight=true.
    slices = 0;
    ts.mark_submitted();
    r = ts.wait_bounded_with_clock(
        350, 100,
        [&](uint64_t step) {
          ++slices;
          now += step;
          return VK_TIMEOUT;
        },
        [&]() { return now; },
        &err);
    fails += check(r == StagingTransferWaitResult::TimedOut && err == VK_TIMEOUT &&
                       ts.in_flight && slices == 4,
                   "staging timeout preserves in_flight=true after bounded slices");

    // 5d. Prior in-flight poll with VK_NOT_READY keeps in_flight=true (refuses reuse).
    fails += check(ts.check_prior_in_flight([]() { return VK_NOT_READY; }) == VK_NOT_READY &&
                       ts.in_flight,
                   "VK_NOT_READY preserves in_flight so staging resources cannot be reused");

    // 5e. Late completion poll with VK_SUCCESS clears in_flight (safe recovery).
    fails += check(ts.check_prior_in_flight([]() { return VK_SUCCESS; }) == VK_SUCCESS &&
                       !ts.in_flight,
                   "VK_SUCCESS clears in_flight for safe recovery");

    // 5f. Non-timeout device error preserves in_flight=true and surfaces VkResult.
    ts.mark_submitted();
    r = ts.wait_bounded_with_clock(
        500, 100,
        [&](uint64_t) { return VK_ERROR_DEVICE_LOST; },
        [&]() { return now; },
        &err);
    fails += check(r == StagingTransferWaitResult::DeviceError &&
                       err == VK_ERROR_DEVICE_LOST && ts.in_flight,
                   "VK_ERROR_DEVICE_LOST preserves in_flight=true and returns exact error");
  }

  // 6. Coordinated device/instance teardown plan.
  {
    auto all_true = [](const DeviceTeardownPlan& p) {
      return p.wait_device_idle && p.destroy_completion_semaphore &&
          p.destroy_weight_staging && p.destroy_transfer_staging &&
          p.destroy_compute_pipelines && p.destroy_device &&
          p.allow_destroy_instance;
    };
    auto all_false = [](const DeviceTeardownPlan& p) {
      return !p.wait_device_idle && !p.destroy_completion_semaphore &&
          !p.destroy_weight_staging && !p.destroy_transfer_staging &&
          !p.destroy_compute_pipelines && !p.destroy_device &&
          !p.allow_destroy_instance;
    };

    fails += check(
        all_true(plan_device_teardown(false, VK_SUCCESS, false)),
        "idle device retires all dependent handles, VkDevice, and VkInstance together");
    fails += check(
        all_true(plan_device_teardown(true, VK_SUCCESS, false)),
        "late-signaled transfer fence retires all handles together");
    fails += check(
        all_false(plan_device_teardown(true, VK_NOT_READY, false)),
        "still-pending transfer fence preserves all dependent handles, VkDevice, and VkInstance");
    fails += check(
        all_false(plan_device_teardown(true, VK_ERROR_DEVICE_LOST, false)),
        "device-lost transfer preserves all handles together");
    fails += check(
        all_false(plan_device_teardown(false, VK_SUCCESS, true)),
        "pending weight staging preserves all handles together");
    fails += check(
        all_false(plan_device_teardown(true, VK_NOT_READY, true)),
        "both staging transfers pending preserves all handles together");
  }

  // Compute-to-compute dispatch barrier: on by default on discrete GPUs only;
  // MLX_OMARCHY_COMPUTE_BARRIER=0/1 (setting 0/1, unset -1) overrides.
  fails += check(
      select_compute_barrier(VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU, -1),
      "discrete GPU takes the compute barrier by default");
  fails += check(
      !select_compute_barrier(VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU, -1),
      "integrated (Apple) GPUs keep the full barrier by default");
  fails += check(
      !select_compute_barrier(VK_PHYSICAL_DEVICE_TYPE_CPU, -1),
      "software Vulkan keeps the full barrier by default");
  fails += check(
      !select_compute_barrier(VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU, 0),
      "MLX_OMARCHY_COMPUTE_BARRIER=0 forces the full barrier");
  fails += check(
      select_compute_barrier(VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU, 1),
      "MLX_OMARCHY_COMPUTE_BARRIER=1 forces the compute barrier");

  // 6. Activation route default: discrete GPUs take the device-local route
  // with the env unset; everything else keeps the strict opt-in.
  fails += check(
      device_local_activations_default(VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU, -1),
      "discrete GPU defaults the device-local activation route ON");
  fails += check(
      !device_local_activations_default(VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU, -1),
      "integrated (Apple) GPUs keep the opt-in: env unset stays off");
  fails += check(
      !device_local_activations_default(VK_PHYSICAL_DEVICE_TYPE_CPU, -1),
      "software Vulkan keeps the opt-in: env unset stays off");
  fails += check(
      !device_local_activations_default(VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU, 0),
      "MLX_OMARCHY_VRAM_ACTIVATIONS=0 opts a discrete GPU out");
  fails += check(
      device_local_activations_default(VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU, 1),
      "MLX_OMARCHY_VRAM_ACTIVATIONS=1 still enables an integrated GPU");

  // 7. The B60-over-USB4 memory shape that motivated the default: a large
  // pure device-local VRAM heap, a host system-memory heap, and a ~256 MiB
  // device-local+host-visible BAR window. The activation route must land on
  // the large VRAM heap's type, never the window's.
  VkPhysicalDeviceMemoryProperties b60{};
  b60.memoryHeapCount = 3;
  b60.memoryHeaps[0] = {23ull << 30, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT};
  b60.memoryHeaps[1] = {46ull << 30, 0};
  b60.memoryHeaps[2] = {256ull << 20, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT};
  b60.memoryTypeCount = 4;
  b60.memoryTypes[0] = {VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0};
  b60.memoryTypes[1] = {
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
          VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
          VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
      1};
  b60.memoryTypes[2] = {
      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT |
          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
          VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
      2};
  b60.memoryTypes[3] = {VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0};
  const uint32_t b60_type = vram_activation_memory_type(
      device_local_activations_default(VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU, -1),
      VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU, b60, 0xF);
  fails += check(
      b60_type != UINT32_MAX && b60.memoryTypes[b60_type].heapIndex == 0,
      "default-on B60 route selects the large VRAM heap, not the BAR window");
  fails += check(
      vram_activation_memory_type(
          device_local_activations_default(
              VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU, 0),
          VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU, b60, 0xF) == UINT32_MAX,
      "opted-out B60 route selects nothing");

  if (fails != 0) {
    return 1;
  }
  std::printf("omarchy_vram_activations_tests: all assertions passed\n");
  return 0;
}
