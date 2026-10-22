// Standalone shader check; run with tests/test_gather_qmm_tile.py. No MLX library is loaded.
#include "mlx/backend/omarchy/compute.h"

#include <bit>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>

using mlx::core::omarchy::ComputeParams;
constexpr uint32_t kBindings = 7;
constexpr uint32_t kPushBytes = offsetof(ComputeParams, lhs_gap);

void check(VkResult result) {
  if (result != VK_SUCCESS) {
    std::cerr << "Vulkan failure: " << result << '\n';
    std::exit(1);
  }
}

struct Buffer {
  VkBuffer buffer;
  VkDeviceMemory memory;
  void* data;
  size_t bytes;
};

struct Context {
  VkInstance instance{};
  VkPhysicalDevice physical{};
  VkDevice device{};
  VkQueue queue{};
  uint32_t family = 0;
  VkPhysicalDeviceMemoryProperties memory{};
  VkDescriptorSetLayout descriptor_layout{};
  VkPipelineLayout pipeline_layout{};
  VkPipeline pipeline{};
  VkShaderModule module{};
  VkDescriptorPool descriptor_pool{};
  VkCommandPool command_pool{};
  VkCommandBuffer command{};
  VkFence fence{};

  explicit Context(const char* path) {
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.apiVersion = VK_API_VERSION_1_2;
    VkInstanceCreateInfo info{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    info.pApplicationInfo = &app;
    check(vkCreateInstance(&info, nullptr, &instance));
    uint32_t count = 1;
    check(vkEnumeratePhysicalDevices(instance, &count, &physical));
    VkPhysicalDeviceProperties properties;
    vkGetPhysicalDeviceProperties(physical, &properties);
    if (properties.deviceType != VK_PHYSICAL_DEVICE_TYPE_CPU) {
      std::cerr << "This check requires a CPU Vulkan device\n";
      std::exit(1);
    }
    std::cout << "device=" << properties.deviceName << '\n';
    vkGetPhysicalDeviceMemoryProperties(physical, &memory);
    vkGetPhysicalDeviceQueueFamilyProperties(physical, &count, nullptr);
    std::vector<VkQueueFamilyProperties> families(count);
    vkGetPhysicalDeviceQueueFamilyProperties(physical, &count, families.data());
    while (family < count && !(families[family].queueFlags & VK_QUEUE_COMPUTE_BIT)) {
      ++family;
    }
    if (family == count) {
      std::exit(1);
    }
    float priority = 1;
    VkDeviceQueueCreateInfo queue_info{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    queue_info.queueFamilyIndex = family;
    queue_info.queueCount = 1;
    queue_info.pQueuePriorities = &priority;
    VkPhysicalDeviceVulkan11Features f11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};
    f11.storageBuffer16BitAccess = true;
    VkPhysicalDeviceVulkan12Features f12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    f12.shaderFloat16 = true;
    f11.pNext = &f12;
    VkPhysicalDeviceFeatures features{};
    features.shaderInt16 = true;
    VkDeviceCreateInfo device_info{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    device_info.pNext = &f11;
    device_info.pEnabledFeatures = &features;
    device_info.queueCreateInfoCount = 1;
    device_info.pQueueCreateInfos = &queue_info;
    check(vkCreateDevice(physical, &device_info, nullptr, &device));
    vkGetDeviceQueue(device, family, 0, &queue);
    VkDescriptorSetLayoutBinding bindings[kBindings]{};
    for (uint32_t i = 0; i < kBindings; ++i) {
      bindings[i] = {i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    }
    VkDescriptorSetLayoutCreateInfo layout_info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    layout_info.bindingCount = kBindings;
    layout_info.pBindings = bindings;
    check(vkCreateDescriptorSetLayout(device, &layout_info, nullptr, &descriptor_layout));
    VkPushConstantRange push{VK_SHADER_STAGE_COMPUTE_BIT, 0, kPushBytes};
    VkPipelineLayoutCreateInfo pipeline_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pipeline_info.setLayoutCount = 1;
    pipeline_info.pSetLayouts = &descriptor_layout;
    pipeline_info.pushConstantRangeCount = 1;
    pipeline_info.pPushConstantRanges = &push;
    check(vkCreatePipelineLayout(device, &pipeline_info, nullptr, &pipeline_layout));
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    size_t bytes = static_cast<size_t>(input.tellg());
    if (!input || bytes % 4 != 0) {
      std::exit(1);
    }
    std::vector<uint32_t> code(bytes / 4);
    input.seekg(0);
    input.read(reinterpret_cast<char*>(code.data()), bytes);
    VkShaderModuleCreateInfo shader_info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    shader_info.codeSize = bytes;
    shader_info.pCode = code.data();
    check(vkCreateShaderModule(device, &shader_info, nullptr, &module));
    VkComputePipelineCreateInfo compute_info{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    compute_info.layout = pipeline_layout;
    compute_info.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
                          VK_SHADER_STAGE_COMPUTE_BIT, module, "main", nullptr};
    check(vkCreateComputePipelines(device, {}, 1, &compute_info, nullptr, &pipeline));
    VkDescriptorPoolSize pool_size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kBindings};
    VkDescriptorPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pool_info.maxSets = 1;
    pool_info.poolSizeCount = 1;
    pool_info.pPoolSizes = &pool_size;
    check(vkCreateDescriptorPool(device, &pool_info, nullptr, &descriptor_pool));
    VkCommandPoolCreateInfo command_info{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    command_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    command_info.queueFamilyIndex = family;
    check(vkCreateCommandPool(device, &command_info, nullptr, &command_pool));
    VkCommandBufferAllocateInfo allocation{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    allocation.commandPool = command_pool;
    allocation.commandBufferCount = 1;
    check(vkAllocateCommandBuffers(device, &allocation, &command));
    VkFenceCreateInfo fence_info{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    check(vkCreateFence(device, &fence_info, nullptr, &fence));
  }

  Buffer buffer(size_t bytes) {
    Buffer result{};
    result.bytes = bytes;
    VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    info.size = bytes;
    info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    check(vkCreateBuffer(device, &info, nullptr, &result.buffer));
    VkMemoryRequirements required;
    vkGetBufferMemoryRequirements(device, result.buffer, &required);
    uint32_t type = 0;
    constexpr auto flags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    while (type < memory.memoryTypeCount &&
           (!(required.memoryTypeBits & (1u << type)) ||
            (memory.memoryTypes[type].propertyFlags & flags) != flags)) {
      ++type;
    }
    if (type == memory.memoryTypeCount) {
      std::exit(1);
    }
    VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    allocation.allocationSize = required.size;
    allocation.memoryTypeIndex = type;
    check(vkAllocateMemory(device, &allocation, nullptr, &result.memory));
    check(vkBindBufferMemory(device, result.buffer, result.memory, 0));
    check(vkMapMemory(device, result.memory, 0, VK_WHOLE_SIZE, 0, &result.data));
    std::memset(result.data, 0xa5, bytes);
    return result;
  }

  void run(Buffer (&buffers)[kBindings], const ComputeParams& params, uint32_t n, uint32_t rows) {
    check(vkResetDescriptorPool(device, descriptor_pool, 0));
    VkDescriptorSetAllocateInfo allocation{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    allocation.descriptorPool = descriptor_pool;
    allocation.descriptorSetCount = 1;
    allocation.pSetLayouts = &descriptor_layout;
    VkDescriptorSet set;
    check(vkAllocateDescriptorSets(device, &allocation, &set));
    VkDescriptorBufferInfo infos[kBindings];
    VkWriteDescriptorSet writes[kBindings]{};
    for (uint32_t i = 0; i < kBindings; ++i) {
      infos[i] = {buffers[i].buffer, 0, buffers[i].bytes};
      writes[i] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, set, i, 0, 1,
                   VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &infos[i], nullptr};
    }
    vkUpdateDescriptorSets(device, kBindings, writes, 0, nullptr);
    check(vkResetCommandBuffer(command, 0));
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    check(vkBeginCommandBuffer(command, &begin));
    vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout, 0, 1, &set, 0, nullptr);
    vkCmdPushConstants(command, pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, kPushBytes, &params);
    vkCmdDispatch(command, n / 64, (rows + 31) / 32, 1);
    VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT,
                         0, 1, &barrier, 0, nullptr, 0, nullptr);
    check(vkEndCommandBuffer(command));
    check(vkResetFences(device, 1, &fence));
    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &command;
    check(vkQueueSubmit(queue, 1, &submit, fence));
    check(vkWaitForFences(device, 1, &fence, true, 10000000000ull));
  }

  void free(Buffer& buffer) {
    vkUnmapMemory(device, buffer.memory);
    vkDestroyBuffer(device, buffer.buffer, nullptr);
    vkFreeMemory(device, buffer.memory, nullptr);
  }

  ~Context() {
    vkDestroyFence(device, fence, nullptr);
    vkDestroyCommandPool(device, command_pool, nullptr);
    vkDestroyDescriptorPool(device, descriptor_pool, nullptr);
    vkDestroyPipeline(device, pipeline, nullptr);
    vkDestroyShaderModule(device, module, nullptr);
    vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
    vkDestroyDescriptorSetLayout(device, descriptor_layout, nullptr);
    vkDestroyDevice(device, nullptr);
    vkDestroyInstance(instance, nullptr);
  }
};

uint16_t store(float value, bool bf16) {
  if (!bf16) {
    return std::bit_cast<uint16_t>(static_cast<_Float16>(value));
  }
  uint32_t bits = std::bit_cast<uint32_t>(value);
  return static_cast<uint16_t>((bits + 0x7fffu + ((bits >> 16u) & 1u)) >> 16u);
}

void run_case(Context& context, bool bf16, uint32_t rows, uint32_t n, uint32_t k, bool shuffled) {
  constexpr uint32_t experts = 3;
  constexpr uint32_t x_offset = 8, w_offset = 3, scale_offset = 4, bias_offset = 6;
  constexpr uint32_t lhs_offset = 2, rhs_offset = 3, output_offset = 4;
  const uint32_t words = experts * n * k / 8;
  const uint32_t groups = k / 64;
  const uint32_t parameters = experts * n * groups;
  Buffer buffers[kBindings] = {
      context.buffer((rows * k + x_offset + 8) * 2),
      context.buffer((parameters + scale_offset + 8) * 2),
      context.buffer((words + w_offset + 4) * 4),
      context.buffer((rows * n + output_offset + 8) * 2),
      context.buffer((parameters + bias_offset + 8) * 2),
      context.buffer((rows + lhs_offset + 4) * 4),
      context.buffer((rows + rhs_offset + 4) * 4)};
  auto* x = static_cast<uint16_t*>(buffers[0].data) + x_offset;
  auto* scale = static_cast<uint16_t*>(buffers[1].data) + scale_offset;
  auto* weight = static_cast<uint32_t*>(buffers[2].data) + w_offset;
  auto* output = static_cast<uint16_t*>(buffers[3].data);
  auto* bias = static_cast<uint16_t*>(buffers[4].data) + bias_offset;
  auto* lhs = static_cast<uint32_t*>(buffers[5].data) + lhs_offset;
  auto* rhs = static_cast<uint32_t*>(buffers[6].data) + rhs_offset;
  auto input = [](uint32_t row, uint32_t column) { return int((row * 3 + column * 7) % 5) - 2; };
  auto code = [](uint32_t expert, uint32_t column, uint32_t inner) {
    return (expert * 11 + column * 3 + inner * 7) % 16;
  };
  for (uint32_t row = 0; row < rows; ++row) {
    lhs[row] = rows - row - 1;
    rhs[row] = shuffled ? (row * 7) % experts : (row < 17 ? 0 : 2);
    for (uint32_t inner = 0; inner < k; ++inner) {
      x[row * k + inner] = store(input(row, inner), bf16);
    }
  }
  for (uint32_t e = 0; e < experts; ++e) {
    for (uint32_t column = 0; column < n; ++column) {
      for (uint32_t group = 0; group < groups; ++group) {
        uint32_t index = (e * n + column) * groups + group;
        scale[index] = store(float(e + group + 1) / 8, bf16);
        bias[index] = store(float(int((column + group) % 3) - 1) / 4, bf16);
      }
      for (uint32_t inner = 0; inner < k; inner += 8) {
        uint32_t word = 0;
        for (uint32_t q = 0; q < 8; ++q) {
          word |= code(e, column, inner + q) << (q * 4);
        }
        weight[(e * n + column) * (k / 8) + inner / 8] = word;
      }
    }
  }
  ComputeParams params;
  params.count = rows * n;
  params.operation = 4;
  params.reduce_size = 64;
  params.lhs_offset = x_offset;
  params.rhs_offset = w_offset;
  params.output_offset = output_offset;
  params.matrix_m = 1;
  params.matrix_n = n;
  params.matrix_k = k;
  params.shape[0] = rows;
  params.shape[2] = scale_offset / 2;
  params.shape[3] = bias_offset / 2;
  params.in_strides[0] = lhs_offset;
  params.out_strides[0] = rhs_offset;
  context.run(buffers, params, n, rows);
  for (uint32_t row = 0; row < rows; ++row) {
    for (uint32_t column = 0; column < n; ++column) {
      float expected = 0;
      for (uint32_t inner = 0; inner < k; ++inner) {
        float value = float(code(rhs[row], column, inner)) * float(rhs[row] + inner / 64 + 1) / 8 +
            float(int((column + inner / 64) % 3) - 1) / 4;
        expected += input(lhs[row], inner) * value;
      }
      uint32_t index = output_offset + row * n + column;
      if (output[index] != store(expected, bf16)) {
        std::cerr << "mismatch row=" << row << " column=" << column << " actual=" << output[index]
                  << " expected=" << store(expected, bf16) << '\n';
        std::exit(1);
      }
    }
  }
  for (uint32_t i = 0; i < output_offset; ++i) {
    if (output[i] != 0xa5a5) std::exit(1);
  }
  for (uint32_t i = output_offset + rows * n; i < buffers[3].bytes / 2; ++i) {
    if (output[i] != 0xa5a5) std::exit(1);
  }
  for (auto& buffer : buffers) context.free(buffer);
  std::cout << "PASS rows=" << rows << " N=" << n << " K=" << k << " shuffled=" << shuffled << '\n';
}

int main(int argc, char** argv) {
  if (argc != 3) return 2;
  Context context(argv[1]);
  bool bf16 = std::strcmp(argv[2], "bf16") == 0;
  run_case(context, bf16, 65, 128, 128, false);
  run_case(context, bf16, 67, 64, 192, true);
}
