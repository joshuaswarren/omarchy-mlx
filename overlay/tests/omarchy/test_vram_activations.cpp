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
      select_qmm_wide_2x16("Apple M1 Max (G13C C0)", VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU, narrow_wg, false),
      "G13C Apple default stays 2x16");
  fails += check(
      !select_qmm_wide_2x16("Apple M2 Max (G14C C0)", VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU, rtx_caps, true),
      "uncalibrated Apple chips stay 2x4 even when opt-in env is set");

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
    // Full GPU overwrite of a host-dirty buffer skips redundant upload.
    DeviceLocalSyncState st{};
    st.on_host_raw_ptr();
    fails += check(!st.on_gpu_access(false, true, 0, 4096, 4096), "full GPU overwrite skips redundant upload");
    fails += check(st.device_dirty && !st.host_dirty, "full overwrite marks device_dirty");
    st.on_recycle();
    fails += check(!st.device_dirty && !st.host_dirty, "recycle resets dirty flags");
  }

  if (fails != 0) {
    return 1;
  }
  std::printf("omarchy_vram_activations_tests: all assertions passed\n");
  return 0;
}
