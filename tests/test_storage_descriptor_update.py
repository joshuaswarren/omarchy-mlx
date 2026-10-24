"""Exercise real descriptor updates on software Vulkan, without MLX or an Apple GPU."""

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = r'''
#include <vulkan/vulkan.h>
#include <algorithm>
#include <array>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <vector>
#include "mlx/backend/omarchy/descriptors.h"

#define VK_CHECK(expr) do { if (auto rc = (expr); rc != VK_SUCCESS) { \
  std::cerr << #expr << ": " << rc << '\n'; return 1; } } while (false)

int main(int argc, char** argv) {
  if (argc != 3) return 2;
  const uint32_t count = std::stoul(argv[2]);
  VkInstance instance;
  VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
  VK_CHECK(vkCreateInstance(&ici, nullptr, &instance));
  uint32_t devices = 0;
  VK_CHECK(vkEnumeratePhysicalDevices(instance, &devices, nullptr));
  std::vector<VkPhysicalDevice> physical(devices);
  VK_CHECK(vkEnumeratePhysicalDevices(instance, &devices, physical.data()));
  if (physical.empty()) return 77;
  auto gpu = physical.front();
  VkPhysicalDeviceProperties props;
  vkGetPhysicalDeviceProperties(gpu, &props);
  const uint32_t binding_limit = std::min({28u, props.limits.maxPerStageDescriptorStorageBuffers,
                                          props.limits.maxDescriptorSetStorageBuffers});
  if (count > binding_limit) return 77;
  uint32_t families = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(gpu, &families, nullptr);
  std::vector<VkQueueFamilyProperties> queues(families);
  vkGetPhysicalDeviceQueueFamilyProperties(gpu, &families, queues.data());
  uint32_t family = 0;
  while (family < families && !(queues[family].queueFlags & VK_QUEUE_COMPUTE_BIT)) ++family;
  if (family == families) return 77;
  float priority = 1;
  VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
  qci.queueFamilyIndex = family;
  qci.queueCount = 1;
  qci.pQueuePriorities = &priority;
  VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
  dci.queueCreateInfoCount = 1;
  dci.pQueueCreateInfos = &qci;
  VkDevice device;
  VK_CHECK(vkCreateDevice(gpu, &dci, nullptr, &device));
  VkQueue queue;
  vkGetDeviceQueue(device, family, 0, &queue);
  const VkDeviceSize stride = std::max<VkDeviceSize>(64, props.limits.minStorageBufferOffsetAlignment);
  std::array<VkBuffer, 2> buffers;
  std::array<VkDeviceMemory, 2> memory;
  std::array<uint32_t*, 2> mapped;
  VkPhysicalDeviceMemoryProperties mp;
  vkGetPhysicalDeviceMemoryProperties(gpu, &mp);
  for (size_t b = 0; b < buffers.size(); ++b) {
    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.size = (count + 2) * stride;
    bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    VK_CHECK(vkCreateBuffer(device, &bci, nullptr, &buffers[b]));
    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(device, buffers[b], &req);
    uint32_t type = 0;
    const auto flags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    while (type < mp.memoryTypeCount &&
           (!(req.memoryTypeBits & (1u << type)) || (mp.memoryTypes[type].propertyFlags & flags) != flags)) ++type;
    if (type == mp.memoryTypeCount) return 77;
    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    mai.allocationSize = req.size;
    mai.memoryTypeIndex = type;
    VK_CHECK(vkAllocateMemory(device, &mai, nullptr, &memory[b]));
    VK_CHECK(vkBindBufferMemory(device, buffers[b], memory[b], 0));
    VK_CHECK(vkMapMemory(device, memory[b], 0, VK_WHOLE_SIZE, 0, reinterpret_cast<void**>(&mapped[b])));
    std::fill_n(mapped[b], bci.size / sizeof(uint32_t), 0xdeadbeefu);
  }
  std::vector<VkDescriptorSetLayoutBinding> layout(binding_limit);
  std::vector<VkDescriptorBufferInfo> infos(count);
  for (uint32_t i = 0; i < binding_limit; ++i) {
    layout[i] = {i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
  }
  for (uint32_t i = 0; i < count; ++i) {
    infos[i] = {buffers[i % 2], (i + 1) * stride, 2 * sizeof(uint32_t)};
    auto* data = mapped[i % 2] + infos[i].offset / sizeof(uint32_t);
    data[0] = i + 7;
    data[1] = i * 3 + 11;
  }
  VkDescriptorSetLayoutCreateInfo slci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
  slci.bindingCount = binding_limit;
  slci.pBindings = layout.data();
  VkDescriptorSetLayout set_layout;
  VK_CHECK(vkCreateDescriptorSetLayout(device, &slci, nullptr, &set_layout));
  VkDescriptorPoolSize size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, binding_limit};
  VkDescriptorPoolCreateInfo dpci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
  dpci.maxSets = 1;
  dpci.poolSizeCount = 1;
  dpci.pPoolSizes = &size;
  VkDescriptorPool pool;
  VK_CHECK(vkCreateDescriptorPool(device, &dpci, nullptr, &pool));
  VkDescriptorSetAllocateInfo sai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
  sai.descriptorPool = pool;
  sai.descriptorSetCount = 1;
  sai.pSetLayouts = &set_layout;
  VkDescriptorSet set;
  VK_CHECK(vkAllocateDescriptorSets(device, &sai, &set));
  mlx::core::omarchy::update_storage_descriptors(vkUpdateDescriptorSets, device, set, infos);
  VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  plci.setLayoutCount = 1;
  plci.pSetLayouts = &set_layout;
  VkPipelineLayout pipeline_layout;
  VK_CHECK(vkCreatePipelineLayout(device, &plci, nullptr, &pipeline_layout));
  std::ifstream file(argv[1], std::ios::binary | std::ios::ate);
  const auto bytes = file.tellg();
  if (bytes <= 0 || bytes % 4) return 2;
  std::vector<uint32_t> code(static_cast<size_t>(bytes) / 4);
  file.seekg(0);
  file.read(reinterpret_cast<char*>(code.data()), bytes);
  VkShaderModuleCreateInfo smci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
  smci.codeSize = static_cast<size_t>(bytes);
  smci.pCode = code.data();
  VkShaderModule module;
  VK_CHECK(vkCreateShaderModule(device, &smci, nullptr, &module));
  VkComputePipelineCreateInfo cpci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
  cpci.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
                VK_SHADER_STAGE_COMPUTE_BIT, module, "main", nullptr};
  cpci.layout = pipeline_layout;
  VkPipeline pipeline;
  VK_CHECK(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cpci, nullptr, &pipeline));
  VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
  pci.queueFamilyIndex = family;
  VkCommandPool command_pool;
  VK_CHECK(vkCreateCommandPool(device, &pci, nullptr, &command_pool));
  VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  cai.commandPool = command_pool;
  cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  cai.commandBufferCount = 1;
  VkCommandBuffer cmd;
  VK_CHECK(vkAllocateCommandBuffers(device, &cai, &cmd));
  VkCommandBufferBeginInfo cbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  VK_CHECK(vkBeginCommandBuffer(cmd, &cbi));
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout, 0, 1, &set, 0, nullptr);
  vkCmdDispatch(cmd, 1, 1, 1);
  VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
  barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
  barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
  vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT,
                       0, 1, &barrier, 0, nullptr, 0, nullptr);
  VK_CHECK(vkEndCommandBuffer(cmd));
  VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
  VkFence fence;
  VK_CHECK(vkCreateFence(device, &fci, nullptr, &fence));
  VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
  submit.commandBufferCount = 1;
  submit.pCommandBuffers = &cmd;
  VK_CHECK(vkQueueSubmit(queue, 1, &submit, fence));
  VK_CHECK(vkWaitForFences(device, 1, &fence, VK_TRUE, 5000000000ULL));
  uint32_t expected = 23;
  for (uint32_t i = 0; i + 1 < count; ++i) expected += (i + 1) * ((i + 7) + (i * 3 + 11));
  auto* result = mapped[(count - 1) % 2] + infos.back().offset / sizeof(uint32_t);
  bool passed = result[0] == expected && result[1] == (expected ^ 0x12345678u);
  passed = passed && result[-1] == 0xdeadbeefu && result[2] == 0xdeadbeefu;
  std::cout << props.deviceName << " bindings=" << count << " result=" << result[0]
            << " expected=" << expected << " guard=" << passed << '\n';
  vkDestroyFence(device, fence, nullptr);
  vkDestroyCommandPool(device, command_pool, nullptr);
  vkDestroyPipeline(device, pipeline, nullptr);
  vkDestroyShaderModule(device, module, nullptr);
  vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
  vkDestroyDescriptorPool(device, pool, nullptr);
  vkDestroyDescriptorSetLayout(device, set_layout, nullptr);
  for (size_t b = 0; b < buffers.size(); ++b) {
    vkUnmapMemory(device, memory[b]);
    vkDestroyBuffer(device, buffers[b], nullptr);
    vkFreeMemory(device, memory[b], nullptr);
  }
  vkDestroyDevice(device, nullptr);
  vkDestroyInstance(instance, nullptr);
  return passed ? 0 : 1;
}
'''


class StorageDescriptorUpdateTests(unittest.TestCase):
    def test_consecutive_bindings_preserve_buffers_offsets_and_ranges(self):
        compiler = shutil.which(os.environ.get("CXX", "c++"))
        glslc = shutil.which("glslc")
        icds = sorted(Path("/usr/share/vulkan/icd.d").glob("lvp_icd*.json"))
        if not compiler or not glslc or not icds:
            self.skipTest("Requires c++, glslc and the software Vulkan ICD")
        env = dict(os.environ, VK_DRIVER_FILES=str(icds[0]), VK_ICD_FILENAMES=str(icds[0]), LP_NUM_THREADS="2")
        if Path("/usr/share/vulkan/explicit_layer.d/VkLayer_khronos_validation.json").exists():
            env["VK_INSTANCE_LAYERS"] = "VK_LAYER_KHRONOS_validation"
        with tempfile.TemporaryDirectory(prefix="storage-descriptors-") as directory:
            work = Path(directory)
            source = work / "test.cpp"
            source.write_text(SOURCE)
            binary = work / "test"
            subprocess.run([compiler, "-std=c++20", "-O0", "-I", str(ROOT / "overlay"),
                            str(source), "-lvulkan", "-o", str(binary)], check=True, timeout=60)
            for count in (1, 5, 28):
                with self.subTest(bindings=count):
                    declarations = [f"layout(set=0,binding={i},std430) buffer B{i} {{ uint v[]; }} b{i};"
                                    for i in range(count)]
                    terms = [f"s += {i + 1}u * (b{i}.v[0] + b{i}.v[1]);" for i in range(count - 1)]
                    shader = "\n".join(["#version 450", "layout(local_size_x=1) in;", *declarations,
                                          "void main() { uint s=23u;", *terms,
                                          f"b{count - 1}.v[0]=s; b{count - 1}.v[1]=s ^ 0x12345678u; }}"])
                    comp, spv = work / "test.comp", work / "test.spv"
                    comp.write_text(shader)
                    subprocess.run([glslc, str(comp), "-o", str(spv)],
                                   check=True, timeout=15)
                    result = subprocess.run([str(binary), str(spv), str(count)], env=env,
                                            text=True, capture_output=True, check=False, timeout=60)
                    print(result.stdout, end="")
                    if result.returncode == 77:
                        self.skipTest(f"Software Vulkan does not support {count} storage buffers")
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    self.assertNotIn("Validation Error", result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
