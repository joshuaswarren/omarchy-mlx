// Dispatch-chain probe: N dispatches of one kernel in one command buffer,
// with or without a compute->compute barrier between them.
// usage: vkchain <shader.spv> <vec> <groups> <mem_type> <MiB> <reps> <N> <barrier 0|1>
//
// usage: vkbw <shader.spv> <vec 1|4> <groups> [mem_type|-1] [MiB] [reps]
//   shader: read-only kernel, binding 0 = source, binding 1 = one uint per
//           thread, push constant = element count (in units of the vector).
//   groups: workgroups of 256 threads; the kernel grid-strides.
//   mem_type: Vulkan memory type index for the source (-1 = every type the
//           buffer allows, one line each).
// Prints GB/s from the best wall-clock submit+wait of <reps> runs.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <vulkan/vulkan.h>

#define CHECK(x) do { VkResult r_ = (x); if (r_ != VK_SUCCESS) { printf("FAIL %s = %d\n", #x, r_); exit(1); } } while (0)

static double now(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

static uint32_t *read_spv(const char *path, size_t *size) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END); *size = ftell(f); fseek(f, 0, SEEK_SET);
    uint32_t *b = malloc(*size);
    if (fread(b, 1, *size, f) != *size) { fclose(f); return NULL; }
    fclose(f);
    return b;
}

static VkInstance inst; static VkPhysicalDevice pd; static VkDevice dev; static VkQueue q;
static VkPhysicalDeviceMemoryProperties mp;

static VkDeviceMemory alloc_for(VkBuffer b, int type, int *chosen) {
    VkMemoryRequirements r; vkGetBufferMemoryRequirements(dev, b, &r);
    if (type < 0) {
        for (uint32_t i = 0; i < mp.memoryTypeCount; ++i)
            if (r.memoryTypeBits & (1u << i)) { type = i; break; }
    }
    if (!(r.memoryTypeBits & (1u << type))) return VK_NULL_HANDLE;
    VkMemoryAllocateInfo ai = { .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                                .allocationSize = r.size, .memoryTypeIndex = type };
    VkDeviceMemory m;
    if (vkAllocateMemory(dev, &ai, NULL, &m) != VK_SUCCESS) return VK_NULL_HANDLE;
    CHECK(vkBindBufferMemory(dev, b, m, 0));
    *chosen = type;
    return m;
}

static VkBuffer make_buffer(VkDeviceSize size) {
    VkBufferCreateInfo bci = { .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = size,
                               .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT };
    VkBuffer b; CHECK(vkCreateBuffer(dev, &bci, NULL, &b));
    return b;
}

int main(int argc, char **argv) {
    if (argc < 4) { printf("usage: %s shader.spv vec groups [mem_type] [MiB] [reps]\n", argv[0]); return 2; }
    const char *spv = argv[1];
    unsigned vec = atoi(argv[2]), groups = atoi(argv[3]);
    int want_type = argc > 4 ? atoi(argv[4]) : -1;
    VkDeviceSize mib = argc > 5 ? atoi(argv[5]) : 512;
    int reps = argc > 6 ? atoi(argv[6]) : 5;
    int chain = argc > 7 ? atoi(argv[7]) : 100, use_barrier = argc > 8 ? atoi(argv[8]) : 1;
    VkDeviceSize bytes = mib << 20;
    uint32_t elems = (uint32_t)(bytes / (4 * vec));

    VkApplicationInfo app = { .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO, .apiVersion = VK_API_VERSION_1_3 };
    VkInstanceCreateInfo ici = { .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pApplicationInfo = &app };
    CHECK(vkCreateInstance(&ici, NULL, &inst));
    uint32_t n = 1; CHECK(vkEnumeratePhysicalDevices(inst, &n, &pd));
    vkGetPhysicalDeviceMemoryProperties(pd, &mp);
    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci = { .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                                     .queueFamilyIndex = 0, .queueCount = 1, .pQueuePriorities = &prio };
    VkDeviceCreateInfo dci = { .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
                               .queueCreateInfoCount = 1, .pQueueCreateInfos = &qci };
    CHECK(vkCreateDevice(pd, &dci, NULL, &dev));
    vkGetDeviceQueue(dev, 0, 0, &q);

    size_t sz; uint32_t *code = read_spv(spv, &sz);
    if (!code) { printf("FAIL no spv\n"); return 1; }
    VkShaderModuleCreateInfo smci = { .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize = sz, .pCode = code };
    VkShaderModule sm; CHECK(vkCreateShaderModule(dev, &smci, NULL, &sm));
    VkDescriptorSetLayoutBinding lb[2] = {
        { .binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT },
        { .binding = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT } };
    VkDescriptorSetLayoutCreateInfo dlci = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, .bindingCount = 2, .pBindings = lb };
    VkDescriptorSetLayout dsl; CHECK(vkCreateDescriptorSetLayout(dev, &dlci, NULL, &dsl));
    VkPushConstantRange pcr = { .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT, .size = 4 };
    VkPipelineLayoutCreateInfo plci = { .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .setLayoutCount = 1,
                                        .pSetLayouts = &dsl, .pushConstantRangeCount = 1, .pPushConstantRanges = &pcr };
    VkPipelineLayout pl; CHECK(vkCreatePipelineLayout(dev, &plci, NULL, &pl));
    VkComputePipelineCreateInfo cpci = { .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = sm, .pName = "main" },
        .layout = pl };
    VkPipeline pipe; CHECK(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpci, NULL, &pipe));

    VkBuffer out = make_buffer((VkDeviceSize)groups * 256 * 4);
    int out_type; if (!alloc_for(out, -1, &out_type)) { printf("FAIL out alloc\n"); return 1; }

    VkCommandPoolCreateInfo cpi = { .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, .queueFamilyIndex = 0 };
    VkCommandPool pool; CHECK(vkCreateCommandPool(dev, &cpi, NULL, &pool));
    VkFenceCreateInfo fci = { .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
    VkFence fence; CHECK(vkCreateFence(dev, &fci, NULL, &fence));

    int first = want_type < 0 ? 0 : want_type, last = want_type < 0 ? (int)mp.memoryTypeCount - 1 : want_type;
    for (int type = first; type <= last; ++type) {
        VkBuffer src = make_buffer(bytes);
        int chosen;
        VkDeviceMemory sm_mem = alloc_for(src, type, &chosen);
        if (!sm_mem) { vkDestroyBuffer(dev, src, NULL); continue; }
        VkMemoryPropertyFlags fl = mp.memoryTypes[chosen].propertyFlags;
        if (fl & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) {
            void *ptr; CHECK(vkMapMemory(dev, sm_mem, 0, bytes, 0, &ptr));
            memset(ptr, 0x5a, bytes);
            vkUnmapMemory(dev, sm_mem);
        }
        VkDescriptorPoolSize ps = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2 };
        VkDescriptorPoolCreateInfo dpci = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, .maxSets = 1, .poolSizeCount = 1, .pPoolSizes = &ps };
        VkDescriptorPool dp; CHECK(vkCreateDescriptorPool(dev, &dpci, NULL, &dp));
        VkDescriptorSetAllocateInfo dsai = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, .descriptorPool = dp, .descriptorSetCount = 1, .pSetLayouts = &dsl };
        VkDescriptorSet ds; CHECK(vkAllocateDescriptorSets(dev, &dsai, &ds));
        VkDescriptorBufferInfo bi[2] = { { src, 0, VK_WHOLE_SIZE }, { out, 0, VK_WHOLE_SIZE } };
        VkWriteDescriptorSet w[2] = {
            { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = ds, .dstBinding = 0, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &bi[0] },
            { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = ds, .dstBinding = 1, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &bi[1] } };
        vkUpdateDescriptorSets(dev, 2, w, 0, NULL);

        VkCommandBufferAllocateInfo cbai = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, .commandPool = pool, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1 };
        VkCommandBuffer cb; CHECK(vkAllocateCommandBuffers(dev, &cbai, &cb));
        VkCommandBufferBeginInfo cbbi = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
        CHECK(vkBeginCommandBuffer(cb, &cbbi));
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &ds, 0, NULL);
        vkCmdPushConstants(cb, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, 4, &elems);
        for (int c = 0; c < chain; ++c) {
            vkCmdDispatch(cb, groups, 1, 1);
            if (use_barrier) {
                VkMemoryBarrier mb = { .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                    .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT, .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT };
                vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
            }
        }
        CHECK(vkEndCommandBuffer(cb));

        double best = 1e9;
        for (int r = 0; r < reps + 1; ++r) {
            VkSubmitInfo si = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &cb };
            double t = now();
            CHECK(vkQueueSubmit(q, 1, &si, fence));
            CHECK(vkWaitForFences(dev, 1, &fence, VK_TRUE, UINT64_MAX));
            t = now() - t;
            CHECK(vkResetFences(dev, 1, &fence));
            if (r && t < best) best = t;  /* first run warms */
        }
        printf("chain=%d barrier=%d groups=%u MiB=%llu  %8.2f us/dispatch  %6.1f GB/s  ",
               chain, use_barrier, groups, (unsigned long long)mib, best * 1e6 / chain, chain * bytes / best / 1e9);
        printf("type=%d flags=%s%s%s%s vec=%u groups=%u threads=%u MiB=%llu  %7.2f ms  %6.1f GB/s\n",
               chosen,
               fl & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT ? "D" : "-",
               fl & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT ? "V" : "-",
               fl & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT ? "C" : "-",
               fl & VK_MEMORY_PROPERTY_HOST_CACHED_BIT ? "$" : "-",
               vec, groups, groups * 256, (unsigned long long)mib, best * 1e3, bytes / best / 1e9);
        fflush(stdout);
        vkFreeCommandBuffers(dev, pool, 1, &cb);
        vkDestroyDescriptorPool(dev, dp, NULL);
        vkDestroyBuffer(dev, src, NULL);
        vkFreeMemory(dev, sm_mem, NULL);
    }
    return 0;
}
