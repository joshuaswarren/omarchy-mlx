// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

#include "mlx/backend/omarchy/allocator.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <mutex>
#include <new>
#include <stdexcept>
#include <string>
#include <thread>

#include "mlx/backend/omarchy/device.h"
#include "mlx/backend/omarchy/device_local_activations.h"
#include "mlx/backend/omarchy/encoder.h"
#include "mlx/backend/omarchy/inference_weights.h"
#include "mlx/backend/omarchy/vulkan.h"
#include "mlx/io/load.h"
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

// Retries after the quarantine drain and cache clear when vkAllocateMemory
// reports OOM: spaced attempts give a driver that uncharges closed GEM
// objects asynchronously time to catch up (#435).
constexpr int kOomRetryAttempts = 3;
constexpr auto kOomRetrySettle = std::chrono::seconds(3);

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

// Test-only hook: simulate one or more bounded staging-transfer fence timeouts
// so runtime tests can verify timeout handling, in-flight lifetime
// preservation, and late-completion recovery on real Vulkan drivers.
bool consume_test_staging_timeout() {
  const char* env = std::getenv("MLX_OMARCHY_TEST_STAGING_TIMEOUT");
  if (env == nullptr || *env == '\0') {
    return false;
  }
  char* end = nullptr;
  long remaining = std::strtol(env, &end, 10);
  if (end == env || remaining <= 0) {
    return false;
  }
  if (remaining == 1) {
    unsetenv("MLX_OMARCHY_TEST_STAGING_TIMEOUT");
  } else {
    std::string next = std::to_string(remaining - 1);
    setenv("MLX_OMARCHY_TEST_STAGING_TIMEOUT", next.c_str(), 1);
  }
  return true;
}

namespace {

VulkanBuffer make_weight_buffer(
    Device& dev,
    size_t size,
    bool staging,
    size_t live_weight_bytes) {
  auto& dt = vk::device_table();
  const auto handle = dev.handle();
  VulkanBuffer buf{};
  buf.size = size;
  buf.inference_weight = !staging;
  VkBufferCreateInfo create{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  create.size = size;
  create.usage = staging ? VK_BUFFER_USAGE_TRANSFER_SRC_BIT
                         : VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
          VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  create.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  VKX_CHECK(dt.CreateBuffer(handle, &create, nullptr, &buf.buffer));
  try {
    VkMemoryRequirements requirements{};
    dt.GetBufferMemoryRequirements(handle, buf.buffer, &requirements);
    const auto& memory = dev.memory_properties();
    const uint32_t type = staging
        ? staging_memory_type(memory, requirements.memoryTypeBits)
        : weight_memory_type_for_size(
              dev.physical_type(),
              memory,
              requirements.memoryTypeBits,
              requirements.size,
              live_weight_bytes);
    if (type == UINT32_MAX) {
      throw std::runtime_error(staging
          ? "[omarchy] no compatible host memory for weight staging"
          : "[omarchy] no compatible device-local memory for inference weights");
    }
    VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    allocation.allocationSize = requirements.size;
    allocation.memoryTypeIndex = type;
    if (staging && allocation.allocationSize > kWeightStagingBytes) {
      throw std::runtime_error("[omarchy] weight staging allocation exceeds 4 MiB");
    }
    const VkResult result = dt.AllocateMemory(handle, &allocation, nullptr, &buf.memory);
    if (result != VK_SUCCESS) {
      throw std::runtime_error(std::string("[omarchy] ") +
          (staging ? "weight staging" : "device-local inference weight") +
          " allocation failed (" + std::to_string(requirements.size) + " bytes): " +
          vk::result_string(result));
    }
    VKX_CHECK(dt.BindBufferMemory(handle, buf.buffer, buf.memory, 0));
    if (staging) {
      buf.coherent = memory.memoryTypes[type].propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
      void* data = nullptr;
      VKX_CHECK(dt.MapMemory(handle, buf.memory, 0, VK_WHOLE_SIZE, 0, &data));
      buf.data = data;
    }
    if (env_flag("MLX_OMARCHY_VRAM_WEIGHTS_LOG")) {
      std::fprintf(stderr, "[omarchy-weights] placement=%s heap=%u bytes=%llu mapped=%d\n",
          staging ? "staging" : "device-local",
          memory.memoryTypes[type].heapIndex,
          static_cast<unsigned long long>(requirements.size),
          staging);
    }
  } catch (...) {
    dt.DestroyBuffer(handle, buf.buffer, nullptr);
    if (buf.memory != VK_NULL_HANDLE) {
      dt.FreeMemory(handle, buf.memory, nullptr);
    }
    throw;
  }
  return buf;
}

} // namespace

void Device::upload_weights(
    CommandEncoder& encoder,
    VkBuffer destination,
    size_t bytes,
    io::Reader& reader,
    size_t file_offset) {
  std::lock_guard lk(weight_upload_mutex_);
  if (weight_staging_in_flight_) {
    throw std::runtime_error("[omarchy] previous weight upload did not complete");
  }
  if (weight_staging_ == VK_NULL_HANDLE) {
    const auto staging = make_weight_buffer(*this, kWeightStagingBytes, true, 0);
    weight_staging_ = staging.buffer;
    weight_staging_memory_ = staging.memory;
    weight_staging_data_ = staging.data;
    weight_staging_coherent_ = staging.coherent;
  }
  auto* data = static_cast<char*>(weight_staging_data_);
  size_t copies = 0;
  for (size_t offset = 0; offset < bytes;) {
    const auto chunk = weight_upload_chunk(bytes, offset);
    reader.read(data, chunk.read_bytes, file_offset + offset);
    std::memset(data + chunk.read_bytes, 0, chunk.copy_bytes - chunk.read_bytes);
    if (!weight_staging_coherent_) {
      VkMappedMemoryRange range{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
      range.memory = weight_staging_memory_;
      range.size = VK_WHOLE_SIZE;
      VKX_CHECK(vk::device_table().FlushMappedMemoryRanges(device_, 1, &range));
    }
    InFlightFlag in_flight(weight_staging_in_flight_);
    try {
      encoder.copy_buffer(weight_staging_, destination, chunk.copy_bytes, 0, offset);
      encoder.synchronize("inference weight upload");
    } catch (...) {
      // The submitted copy may still read the staging buffer. Leave it
      // unreferenced so the next upload allocates fresh storage instead of
      // overwriting mapped memory the GPU is reading; DestroyDevice reclaims
      // the abandoned allocation at teardown.
      weight_staging_ = VK_NULL_HANDLE;
      weight_staging_memory_ = VK_NULL_HANDLE;
      weight_staging_data_ = nullptr;
      weight_staging_coherent_ = false;
      throw;
    }
    offset += chunk.read_bytes;
    ++copies;
  }
  if (env_flag("MLX_OMARCHY_VRAM_WEIGHTS_LOG")) {
    std::fprintf(stderr, "[omarchy-weights] uploaded_bytes=%zu copies=%zu\n", bytes, copies);
  }
}

size_t Device::device_local_heap_bytes() const {
  return max_device_local_heap_bytes(mem_props_);
}

void Device::ensure_transfer_resources_locked() {
  if (transfer_staging_ != VK_NULL_HANDLE) {
    return;
  }
  auto& dt = vk::device_table();
  VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  bci.size = kWeightStagingBytes;
  bci.usage =
      VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  VKX_CHECK(dt.CreateBuffer(device_, &bci, nullptr, &transfer_staging_));
  try {
    VkMemoryRequirements reqs{};
    dt.GetBufferMemoryRequirements(device_, transfer_staging_, &reqs);
    uint32_t type = staging_memory_type(mem_props_, reqs.memoryTypeBits);
    if (type == UINT32_MAX) {
      // Fallback for UMA/software-Vulkan tests that exercise staging transfers.
      for (uint32_t i = 0; i < mem_props_.memoryTypeCount; ++i) {
        if ((reqs.memoryTypeBits & (1u << i)) &&
            (mem_props_.memoryTypes[i].propertyFlags &
             VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)) {
          type = i;
          if (mem_props_.memoryTypes[i].propertyFlags &
              VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) {
            break;
          }
        }
      }
    }
    if (type == UINT32_MAX) {
      throw std::runtime_error(
          "[omarchy] no compatible host-visible memory for activation staging");
    }
    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    mai.allocationSize = reqs.size;
    mai.memoryTypeIndex = type;
    VKX_CHECK(
        dt.AllocateMemory(device_, &mai, nullptr, &transfer_staging_memory_));
    VKX_CHECK(dt.BindBufferMemory(
        device_, transfer_staging_, transfer_staging_memory_, 0));
    transfer_staging_coherent_ =
        (mem_props_.memoryTypes[type].propertyFlags &
         VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0;
    VKX_CHECK(dt.MapMemory(
        device_,
        transfer_staging_memory_,
        0,
        VK_WHOLE_SIZE,
        0,
        &transfer_staging_data_));

    VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pci.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT |
        VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pci.queueFamilyIndex = caps_.queue_family_index;
    VKX_CHECK(dt.CreateCommandPool(device_, &pci, nullptr, &transfer_pool_));

    VkCommandBufferAllocateInfo cbai{
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cbai.commandPool = transfer_pool_;
    cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbai.commandBufferCount = 1;
    VKX_CHECK(dt.AllocateCommandBuffers(device_, &cbai, &transfer_cmd_));

    VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VKX_CHECK(dt.CreateFence(device_, &fci, nullptr, &transfer_fence_));
  } catch (...) {
    if (transfer_fence_ != VK_NULL_HANDLE) {
      dt.DestroyFence(device_, transfer_fence_, nullptr);
      transfer_fence_ = VK_NULL_HANDLE;
    }
    if (transfer_pool_ != VK_NULL_HANDLE) {
      dt.DestroyCommandPool(device_, transfer_pool_, nullptr);
      transfer_pool_ = VK_NULL_HANDLE;
      transfer_cmd_ = VK_NULL_HANDLE;
    }
    if (transfer_staging_data_ != nullptr) {
      dt.UnmapMemory(device_, transfer_staging_memory_);
      transfer_staging_data_ = nullptr;
    }
    if (transfer_staging_ != VK_NULL_HANDLE) {
      dt.DestroyBuffer(device_, transfer_staging_, nullptr);
      transfer_staging_ = VK_NULL_HANDLE;
    }
    if (transfer_staging_memory_ != VK_NULL_HANDLE) {
      dt.FreeMemory(device_, transfer_staging_memory_, nullptr);
      transfer_staging_memory_ = VK_NULL_HANDLE;
    }
    throw;
  }
}

void Device::check_prior_transfer_locked(const char* direction) {
  if (!transfer_state_.in_flight) {
    return;
  }
  auto& dt = vk::device_table();
  const VkResult st = transfer_state_.check_prior_in_flight([&]() {
    if (env_flag("MLX_OMARCHY_TEST_STAGING_FENCE_BUSY")) {
      return VK_NOT_READY;
    }
    return dt.GetFenceStatus(device_, transfer_fence_);
  });
  if (st == VK_SUCCESS) {
    if (transfer_in_flight_buffer_ != VK_NULL_HANDLE) {
      omarchy::allocator().mark_transfer_in_flight(
          transfer_in_flight_buffer_, false);
      transfer_in_flight_buffer_ = VK_NULL_HANDLE;
    }
    return;
  }
  if (st == VK_NOT_READY) {
    throw std::runtime_error(
        std::string("[omarchy] previous activation staging transfer is still "
                    "in flight; cannot start ") +
        direction + " (Timeout). In-flight staging resources preserved.");
  }
  throw std::runtime_error(
      std::string("[omarchy] activation staging fence status failed before ") +
      direction + " (" + vk::result_string(st) + ").");
}

void Device::wait_transfer_fence_locked(
    VkBuffer target_buffer,
    const char* direction) {
  auto& dt = vk::device_table();
  const uint64_t timeout_ns =
      std::min(submit_hang_no_progress_ns(), submit_max_wall_ns());
  VkResult err = VK_SUCCESS;
  StagingTransferWaitResult wait_res = StagingTransferWaitResult::Completed;
  if (consume_test_staging_timeout()) {
    err = VK_TIMEOUT;
    wait_res = StagingTransferWaitResult::TimedOut;
  } else {
    wait_res = transfer_state_.wait_bounded(
        timeout_ns,
        kHangWatchPollNs,
        [&](uint64_t slice_ns) {
          return dt.WaitForFences(
              device_, 1, &transfer_fence_, VK_TRUE, slice_ns);
        },
        &err);
  }
  if (wait_res == StagingTransferWaitResult::Completed) {
    omarchy::allocator().mark_transfer_in_flight(target_buffer, false);
    transfer_in_flight_buffer_ = VK_NULL_HANDLE;
    return;
  }
  if (wait_res == StagingTransferWaitResult::TimedOut) {
    throw std::runtime_error(
        std::string("[omarchy] Vulkan activation staging ") + direction +
        " timed out after " + std::to_string(timeout_ns / 1000000ull) +
        " ms (Timeout). The device may be hung; in-flight staging resources"
        " preserved.");
  }
  throw std::runtime_error(
      std::string("[omarchy] Vulkan activation staging ") + direction +
      " fence wait failed (" + vk::result_string(err) +
      "). In-flight staging resources preserved.");
}

void Device::upload_host_to_device(
    VkBuffer destination,
    const void* src,
    size_t bytes) {
  if (bytes == 0) {
    return;
  }
  std::lock_guard lk(transfer_mutex_);
  ensure_transfer_resources_locked();
  check_prior_transfer_locked("upload");
  auto& dt = vk::device_table();
  const auto* src_bytes = static_cast<const uint8_t*>(src);
  auto* stage = static_cast<uint8_t*>(transfer_staging_data_);

  for (size_t offset = 0; offset < bytes;) {
    const auto chunk = vram_staging_chunk(bytes, offset);
    std::memcpy(stage, src_bytes + chunk.byte_offset, chunk.byte_count);
    if (chunk.copy_bytes > chunk.byte_count) {
      std::memset(
          stage + chunk.byte_count, 0, chunk.copy_bytes - chunk.byte_count);
    }
    if (!transfer_staging_coherent_) {
      VkMappedMemoryRange range{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
      range.memory = transfer_staging_memory_;
      range.size = VK_WHOLE_SIZE;
      VKX_CHECK(dt.FlushMappedMemoryRanges(device_, 1, &range));
    }
    VKX_CHECK(dt.ResetCommandBuffer(transfer_cmd_, 0));
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VKX_CHECK(dt.BeginCommandBuffer(transfer_cmd_, &bi));
    VkBufferCopy region{};
    region.srcOffset = 0;
    region.dstOffset = chunk.byte_offset;
    region.size = chunk.copy_bytes;
    dt.CmdCopyBuffer(transfer_cmd_, transfer_staging_, destination, 1, &region);
    // Make transfer writes visible to subsequent compute and transfer
    // consumers on the device before signaling transfer_fence_.
    VkMemoryBarrier post_mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    post_mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    post_mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT |
        VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT |
        VK_ACCESS_TRANSFER_WRITE_BIT;
    dt.CmdPipelineBarrier(
        transfer_cmd_,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
        0,
        1,
        &post_mb,
        0,
        nullptr,
        0,
        nullptr);
    VKX_CHECK(dt.EndCommandBuffer(transfer_cmd_));

    VKX_CHECK(dt.ResetFences(device_, 1, &transfer_fence_));
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &transfer_cmd_;
    omarchy::allocator().mark_transfer_in_flight(destination, true);
    transfer_in_flight_buffer_ = destination;
    try {
      std::lock_guard qlk(queue_mutex_);
      VKX_CHECK(dt.QueueSubmit(queue_, 1, &si, transfer_fence_));
    } catch (...) {
      omarchy::allocator().mark_transfer_in_flight(destination, false);
      transfer_in_flight_buffer_ = VK_NULL_HANDLE;
      throw;
    }
    transfer_state_.mark_submitted();
    wait_transfer_fence_locked(destination, "upload");
    offset += chunk.byte_count;
  }
}

void Device::download_device_to_host(
    VkBuffer source,
    void* dst,
    size_t bytes) {
  if (bytes == 0) {
    return;
  }
  std::lock_guard lk(transfer_mutex_);
  ensure_transfer_resources_locked();
  check_prior_transfer_locked("download");
  auto& dt = vk::device_table();
  auto* dst_bytes = static_cast<uint8_t*>(dst);
  const auto* stage = static_cast<const uint8_t*>(transfer_staging_data_);

  for (size_t offset = 0; offset < bytes;) {
    const auto chunk = vram_staging_chunk(bytes, offset);
    VKX_CHECK(dt.ResetCommandBuffer(transfer_cmd_, 0));
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VKX_CHECK(dt.BeginCommandBuffer(transfer_cmd_, &bi));
    // Establish device-memory visibility from prior GPU compute/transfer
    // writes to transfer reads before copying into staging (with gated
    // barriers, producer submit only emits a write->HOST_READ barrier).
    VkMemoryBarrier pre_mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    pre_mb.srcAccessMask =
        VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    pre_mb.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    dt.CmdPipelineBarrier(
        transfer_cmd_,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        0,
        1,
        &pre_mb,
        0,
        nullptr,
        0,
        nullptr);
    VkBufferCopy region{};
    region.srcOffset = chunk.byte_offset;
    region.dstOffset = 0;
    region.size = chunk.copy_bytes;
    dt.CmdCopyBuffer(transfer_cmd_, source, transfer_staging_, 1, &region);
    VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    dt.CmdPipelineBarrier(
        transfer_cmd_,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_HOST_BIT,
        0,
        1,
        &mb,
        0,
        nullptr,
        0,
        nullptr);
    VKX_CHECK(dt.EndCommandBuffer(transfer_cmd_));

    VKX_CHECK(dt.ResetFences(device_, 1, &transfer_fence_));
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &transfer_cmd_;
    omarchy::allocator().mark_transfer_in_flight(source, true);
    transfer_in_flight_buffer_ = source;
    try {
      std::lock_guard qlk(queue_mutex_);
      VKX_CHECK(dt.QueueSubmit(queue_, 1, &si, transfer_fence_));
    } catch (...) {
      omarchy::allocator().mark_transfer_in_flight(source, false);
      transfer_in_flight_buffer_ = VK_NULL_HANDLE;
      throw;
    }
    transfer_state_.mark_submitted();
    wait_transfer_fence_locked(source, "download");
    if (!transfer_staging_coherent_) {
      VkMappedMemoryRange range{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
      range.memory = transfer_staging_memory_;
      range.size = VK_WHOLE_SIZE;
      VKX_CHECK(dt.InvalidateMappedMemoryRanges(device_, 1, &range));
    }
    std::memcpy(dst_bytes + chunk.byte_offset, stage, chunk.byte_count);
    offset += chunk.byte_count;
  }
}

bool device_local_activations_enabled() {
  if (!is_available()) {
    return false;
  }
  auto& dev = device();
  const int env_setting = std::getenv("MLX_OMARCHY_VRAM_ACTIVATIONS") == nullptr
      ? -1
      : (env_flag("MLX_OMARCHY_VRAM_ACTIVATIONS") ? 1 : 0);
  if (!device_local_activations_default(dev.physical_type(), env_setting)) {
    return false;
  }
  return vram_activation_memory_type(
             true, dev.physical_type(), dev.memory_properties(), UINT32_MAX) !=
      UINT32_MAX;
}

uint32_t VulkanAllocator::find_memory_type(
    uint32_t type_bits,
    VkMemoryPropertyFlags required) const {
  return capacity_aware_memory_type(
      device().memory_properties(), type_bits, required);
}

VulkanAllocator::VulkanAllocator()
    : buffer_cache_(
          kPageSize,
          [](VulkanBuffer* buf) { return buf->size; },
          [this](VulkanBuffer* buf) { destroy_buffer(buf); }),
      device_local_cache_(
          kPageSize,
          [](VulkanBuffer* buf) { return buf->size; },
          [this](VulkanBuffer* buf) { destroy_buffer(buf); }) {
  size_t total = is_available() ? capability_report(0).total_memory : 0;
  heap_bytes_ = total;
  memory_limit_ = default_memory_limit(total);
  // Metal's `gc_limit_` is `min(0.95 * max_rec, 0.95 * memsize)`.
  // Honeykrisp has no separate max-recommended and the default heap is
  // already kept below the host memory (an explicit HK_SYSMEM is not), so
  // the ceiling is the whole host-visible heap: a ceiling below the
  // resident weights of a model that fits the heap makes every malloc
  // release cached buffers.
  gc_limit_ = default_gc_limit(total);
  cache_limit_ = default_cache_limit(total);
  size_t vram_total = is_available() ? device().device_local_heap_bytes() : 0;
  device_local_gc_limit_ = default_gc_limit(vram_total > 0 ? vram_total : total);
  device_local_cache_limit_ =
      default_cache_limit(vram_total > 0 ? vram_total : total);
}

Buffer VulkanAllocator::malloc(size_t size) {
  // The table is empty until the first device exists. A core flow can reach
  // malloc before anything else touches the device, so initialize here; the
  // table field would otherwise be read before the lazy init fills it.
  auto& dev = device();
  auto& dt = vk::device_table();
  if (size == 0) {
    return Buffer{new VulkanBuffer{}};
  }
  const size_t requested_size = size;
  size = round_size(size);
  const bool want_device_local = should_use_device_local_activation(
      device_local_activations_enabled(), requested_size);

  std::unique_lock lk(mutex_);
  auto& cache = want_device_local ? device_local_cache_ : buffer_cache_;
  const size_t effective_gc_limit =
      want_device_local ? device_local_gc_limit_ : gc_limit_;
  // MLX_OMARCHY_TAPE_NO_REUSE (diagnostic, docs/install-omarchy.md):
  // skip the cache entirely so every allocation lands in fresh device
  // memory and nothing recycled can alias a tape dispatch.
  // MLX_OMARCHY_NO_BUFFER_CACHE (diagnostic): the same, but process-
  // wide - it also covers frees and re-hands outside the tape window,
  // which TAPE_NO_REUSE deliberately does not.
  if (!tape_no_reuse() && !buffer_cache_disabled()) {
    if (void* cached = cache.reuse_from_cache(size)) {
      auto* buf = static_cast<VulkanBuffer*>(cached);
      buf->recycled = true;
      buf->sync_state.on_recycle();
      live_by_handle_[buf->buffer] = buf;
      active_memory_ += buf->size;
      peak_memory_ = std::max(active_memory_ + weight_memory_, peak_memory_);
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
    cache.clear();
  }
  // Working-set ceiling (Metal: `gc_limit_`): keep active + cache + this
  // request under the ceiling by releasing the reuse cache before we ask
  // the driver. This is the A20 fix — without it, KV-cache growth across
  // decode steps adds a fresh size class per step and the cache keeps
  // every prior class around, climbing into the Honeykrisp heap until
  // vkAllocateMemory returns OOM. The ceiling is the whole total_memory;
  // the free-side cap (`cache_limit_`) sits below it.
  // Quarantined buffers still hold their device memory, so they count
  // against the ceiling like active and cached bytes do.
  if (const size_t held = active_memory_ + quarantine_bytes_ +
          cache.cache_size() + size;
      held > effective_gc_limit) {
    cache.release_cached_buffers(held - effective_gc_limit);
  }
  lk.unlock();

  auto* buf = new VulkanBuffer{};
  buf->size = size;

  VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  bci.size = size;
  bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
      VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  VKX_CHECK(dt.CreateBuffer(dev.handle(), &bci, nullptr, &buf->buffer));

  VkMemoryRequirements reqs{};
  dt.GetBufferMemoryRequirements(dev.handle(), buf->buffer, &reqs);

  uint32_t type_index = UINT32_MAX;
  if (want_device_local) {
    type_index = vram_activation_memory_type(
        true, dev.physical_type(), dev.memory_properties(), reqs.memoryTypeBits);
    if (type_index != UINT32_MAX) {
      buf->device_local = true;
      buf->coherent = true;
    }
  }
  if (!buf->device_local) {
    // Prefer host-visible coherent memory (unified memory on Apple GPUs). Fall
    // back to plain host-visible memory with explicit cache maintenance; fail
    // closed when even that is unavailable.
    type_index = find_memory_type(
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
      dt.DestroyBuffer(dev.handle(), buf->buffer, nullptr);
      delete buf;
      throw std::runtime_error(
          "[omarchy] no host-visible memory type available;"
          " the backend cannot stage CPU data without one.");
    }
  }

  VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  mai.allocationSize = reqs.size;
  mai.memoryTypeIndex = type_index;
  VkResult alloc_result = VK_SUCCESS;
  if (consume_test_oom()) {
    alloc_result = VK_ERROR_OUT_OF_DEVICE_MEMORY;
  } else {
    alloc_result =
        dt.AllocateMemory(dev.handle(), &mai, nullptr, &buf->memory);
  }
  if (alloc_result == VK_ERROR_OUT_OF_DEVICE_MEMORY ||
      alloc_result == VK_ERROR_OUT_OF_HOST_MEMORY) {
    if (const uint64_t submitted = dev.completions().last_reserved();
        submitted != 0) {
      dev.completions().wait(submitted);
      if (dev.completions().wait_settled(
              submitted,
              std::chrono::nanoseconds(submit_hang_no_progress_ns()))) {
        release_quarantine(submitted);
      }
    }
    {
      std::unique_lock rl(mutex_);
      buffer_cache_.clear();
      device_local_cache_.clear();
    }
    for (int attempt = 0; attempt < kOomRetryAttempts; ++attempt) {
      alloc_result = consume_test_oom()
          ? VK_ERROR_OUT_OF_DEVICE_MEMORY
          : dt.AllocateMemory(
                dev.handle(), &mai, nullptr, &buf->memory);
      if (alloc_result != VK_ERROR_OUT_OF_DEVICE_MEMORY &&
          alloc_result != VK_ERROR_OUT_OF_HOST_MEMORY) {
        break;
      }
      std::this_thread::sleep_for(kOomRetrySettle);
    }
  }
  if (alloc_result != VK_SUCCESS) {
    dt.DestroyBuffer(dev.handle(), buf->buffer, nullptr);
    delete buf;
    lk.lock();
    const std::string held = "active=" + std::to_string(active_memory_) +
        " quarantined=" + std::to_string(quarantine_bytes_);
    lk.unlock();
    throw std::runtime_error(
        "[omarchy] vkAllocateMemory failed: " +
        std::string(vk::result_string(alloc_result)) + " (request=" +
        std::to_string(size) + " bytes; " + held +
        " bytes; cache released)");
  }
  VKX_CHECK(
      dt.BindBufferMemory(dev.handle(), buf->buffer, buf->memory, 0));
  if (!buf->device_local) {
    VKX_CHECK(dt.MapMemory(
        dev.handle(), buf->memory, 0, VK_WHOLE_SIZE, 0, &buf->data));
  }

  lk.lock();
  live_by_handle_[buf->buffer] = buf;
  active_memory_ += buf->size;
  peak_memory_ = std::max(active_memory_ + weight_memory_, peak_memory_);
  if (!buf->device_local && !buf->coherent) {
    noncoherent_.push_back(buf);
  }
  lk.unlock();
  return Buffer{buf};
}

Buffer VulkanAllocator::malloc_weight(size_t size) {
  auto buf = std::make_unique<VulkanBuffer>();
  // Snapshot the live total before building the buffer: reading it inside
  // make_weight_buffer would race concurrent weight frees. The scoped
  // guard releases the mutex before the Vulkan calls, and the relock
  // below is exception-safe if make_weight_buffer throws.
  size_t live_weight_bytes = 0;
  {
    std::lock_guard<std::mutex> guard(mutex_);
    live_weight_bytes = weight_memory_;
  }
  *buf = make_weight_buffer(
      device(), (size + 3) & ~size_t{3}, false, live_weight_bytes);
  std::unique_lock lk(mutex_);
  try {
    live_by_handle_.emplace(buf->buffer, buf.get());
  } catch (...) {
    destroy_buffer(buf.release());
    throw;
  }
  weight_memory_ += buf->size;
  peak_memory_ = std::max(active_memory_ + weight_memory_, peak_memory_);
  return Buffer{buf.release()};
}

void VulkanAllocator::destroy_buffer(VulkanBuffer* buf) {
  // Cache clear/release funnels through here while callers already hold
  // mutex_, so the non-coherent registry drops the buffer before its
  // memory dies; later flush/invalidate must never see it.
  std::erase(noncoherent_, buf);
  auto& dt = vk::device_table();
  live_by_handle_.erase(buf->buffer);
  if (buf->host_shadow) {
    std::free(buf->host_shadow);
    buf->host_shadow = nullptr;
  }
  if (buf->data) {
    dt.UnmapMemory(device().handle(), buf->memory);
  }
  if (buf->buffer != VK_NULL_HANDLE) {
    dt.DestroyBuffer(device().handle(), buf->buffer, nullptr);
  }
  if (buf->memory != VK_NULL_HANDLE) {
    dt.FreeMemory(device().handle(), buf->memory, nullptr);
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
  (buf->inference_weight ? weight_memory_ : active_memory_) -= sz;
  // A buffer stamped with a completion value has recorded GPU work that
  // may still touch it (kPendingCompletion = an open batch). Freeing the
  // array drops the accounting here — upstream Metal semantics — but the
  // raw buffer skips the reuse cache and waits in the quarantine until
  // release_quarantine proves the driver finished its submission. A
  // buffer whose generation drained at least one generation ago (a later
  // generation is observable) needs no wait; without this an idle
  // process parks every freed buffer until the next submission.
  if (buf->transfer_in_flight || buf->completion != 0) {
    if (buf->transfer_in_flight ||
        buf->completion == kPendingCompletion || !runtime_alive() ||
        buf->completion >= device().completions().drained_value()) {
      if (buf->completion == kPendingCompletion) {
        pending_quarantine_bytes_ += sz;
      }
      if (!buf->inference_weight) {
        quarantine_bytes_ += sz;
      }
      quarantine_.push_back(buf);
      return;
    }
    buf->completion = 0;
  }
  auto& cache = buf->device_local ? device_local_cache_ : buffer_cache_;
  const size_t limit =
      buf->device_local ? device_local_cache_limit_ : cache_limit_;
  if (!buf->inference_weight && sz > 0 && !tape_no_reuse() &&
      !buffer_cache_disabled() && cache.cache_size() + sz <= limit) {
    if (buf->host_shadow) {
      std::free(buf->host_shadow);
      buf->host_shadow = nullptr;
    }
    buf->sync_state.on_recycle();
    // Buffers stay mapped for their whole lifetime (malloc maps at
    // creation and only destroy_buffer unmaps), so the poison is a
    // plain host memset. Non-coherent buffers are flushed at the next
    // submit like any other host write.
    if (poison_freed() && buf->data != nullptr) {
      poison_freed_buffer(buf->data, sz);
    }
    cache.recycle_to_cache(buf);
    return;
  }
  // The cache is full, so release this block instead of retaining it.
  destroy_buffer(buf);
}

void VulkanAllocator::release_quarantine(uint64_t cleanup_done_through) {
  // Callable from any thread: the allocator mutex serializes passes, and
  // each entry's own stamps decide recyclability. A block recycles only
  // when its completion generation is at or below |cleanup_done_through|.
  // Callers pass a generation whose cleanup is proven. The drain pass passes
  // one behind the newest drained generation. The dispatcher's idle tick
  // passes the newest drained generation once the drain has been quiet for
  // a poll interval, and one behind it otherwise. The OOM retry passes the
  // requested generation after a later generation drained or the queue
  // went idle (the design's one-generation rule: observing generation N
  // does not prove the driver's submit-final cleanup for N has run).
  std::unique_lock lk(mutex_);
  if (quarantine_.empty()) {
    return;
  }
  std::vector<VulkanBuffer*> still_quarantined;
  for (auto* buf : quarantine_) {
    if (buf->transfer_in_flight ||
        buf->completion > cleanup_done_through ||
        buf->completion == kPendingCompletion) {
      still_quarantined.push_back(buf);
      continue;
    }
    buf->completion = 0;
    size_t sz = buf->size;
    if (!buf->inference_weight) {
      quarantine_bytes_ -= sz;
    }
    auto& cache = buf->device_local ? device_local_cache_ : buffer_cache_;
    const size_t limit =
        buf->device_local ? device_local_cache_limit_ : cache_limit_;
    if (!buf->inference_weight && sz > 0 && !tape_no_reuse() &&
        !buffer_cache_disabled() && cache.cache_size() + sz <= limit) {
      if (buf->host_shadow) {
        std::free(buf->host_shadow);
        buf->host_shadow = nullptr;
      }
      buf->sync_state.on_recycle();
      if (poison_freed() && buf->data != nullptr) {
        poison_freed_buffer(buf->data, sz);
      }
      cache.recycle_to_cache(buf);
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
    const std::unordered_set<VulkanBuffer*>& bufs,
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

VulkanBuffer* VulkanAllocator::lookup_buffer_handle(VkBuffer handle) {
  std::unique_lock lk(mutex_);
  auto it = live_by_handle_.find(handle);
  return it == live_by_handle_.end() ? nullptr : it->second;
}

uint64_t VulkanAllocator::buffer_completion(const VulkanBuffer* buf) const {
  std::unique_lock lk(mutex_);
  return buf ? buf->completion : 0;
}

void VulkanAllocator::mark_transfer_in_flight(
    VkBuffer handle,
    bool in_flight) {
  if (handle == VK_NULL_HANDLE) {
    return;
  }
  std::unique_lock lk(mutex_);
  auto it = live_by_handle_.find(handle);
  if (it != live_by_handle_.end() && it->second != nullptr) {
    it->second->transfer_in_flight = in_flight;
    return;
  }
  for (auto* buf : quarantine_) {
    if (buf && buf->buffer == handle) {
      buf->transfer_in_flight = in_flight;
      return;
    }
  }
}

size_t VulkanAllocator::size(Buffer buffer) const {
  auto* buf = static_cast<VulkanBuffer*>(buffer.ptr());
  return buf ? buf->size : 0;
}

size_t VulkanAllocator::set_cache_limit(size_t limit) {
  std::unique_lock lk(mutex_);
  std::swap(cache_limit_, limit);
  device_local_cache_limit_ = cache_limit_;
  if (buffer_cache_.cache_size() > cache_limit_) {
    buffer_cache_.release_cached_buffers(
        buffer_cache_.cache_size() - cache_limit_);
  }
  if (device_local_cache_.cache_size() > device_local_cache_limit_) {
    device_local_cache_.release_cached_buffers(
        device_local_cache_.cache_size() - device_local_cache_limit_);
  }
  return limit;
}

void VulkanAllocator::clear_cache() {
  std::unique_lock lk(mutex_);
  buffer_cache_.clear();
  device_local_cache_.clear();
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
  auto* buffer = static_cast<omarchy::VulkanBuffer*>(ptr_);
  if (buffer->inference_weight) {
    throw std::runtime_error("[omarchy] device-local inference weights are not host-accessible");
  }
  if (buffer->device_local) {
    if (buffer->sync_state.device_dirty) {
      uint64_t comp = omarchy::allocator().buffer_completion(buffer);
      if (comp == omarchy::kPendingCompletion) {
        omarchy::synchronize_pending_encoders();
        comp = omarchy::allocator().buffer_completion(buffer);
      }
      if (comp != 0 && comp != omarchy::kPendingCompletion &&
          omarchy::runtime_alive()) {
        omarchy::device().completions().wait(comp);
      }
      if (buffer->host_shadow == nullptr) {
        buffer->host_shadow = std::calloc(1, buffer->size);
        if (buffer->host_shadow == nullptr) {
          throw std::bad_alloc();
        }
      }
      omarchy::device().download_device_to_host(
          buffer->buffer, buffer->host_shadow, buffer->size);
    } else if (buffer->host_shadow == nullptr) {
      buffer->host_shadow = std::calloc(1, buffer->size);
      if (buffer->host_shadow == nullptr) {
        throw std::bad_alloc();
      }
    }
    (void)buffer->sync_state.on_host_raw_ptr();
    return buffer->host_shadow;
  }
  return buffer->data;
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
