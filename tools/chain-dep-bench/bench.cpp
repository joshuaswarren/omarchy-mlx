// Three-dispatch dependent-chain microbench for Honeykrisp/AGX (Apple M1).
//
// Discriminates where the per-dependent-dispatch turnaround lives by timing
// the SAME chain A -> B -> C (grid-1 trivial writes, the TDT fold/fold_proj/
// control shape) under every dependency mechanism Vulkan offers:
//
//   cs1_none          one CB, three dispatches, no dependency records (floor)
//   cs1_barrier       one CB, full MEMORY/ALL_COMMANDS barrier between
//   cs1_event         one CB, CmdSetEvent/CmdWaitEvents pair between
//   cs3_1submit_sema  three CBs + timeline semaphores, one vkQueueSubmit call
//                     (three SubmitInfo entries: three CSes, one ioctl)
//   cs3_3submit_sema  same three CBs, three back-to-back vkQueueSubmit calls
//   cs3_fencejoin     three submits, vkQueueWaitIdle between each (host join)
//
// Wall = host CLOCK_MONOTONIC around the first queue touch .. QueueWaitIdle.
// GPU hop = per-dispatch TOP/BOTTOM timestamp pairs (same-run ratio scale;
// honeykrisp timestamp periods are not trustworthy in absolute terms,
// receipts/2026-09-10-q4-gemv-bandwidth/verdict.json).
//
// NDJSON on stdout. Run under the GPU lock on the M1:
//   flock -w 2400 /tmp/m1-gpu.lock tools/chain-dep-bench/run-m1.sh

#include <dlfcn.h>
#include <vulkan/vulkan.h>
#include <inttypes.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>

#include <algorithm>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

// ---- Vulkan entry points via dlopen (same shape as dispatch-floor-bench) --

static void* vk_lib = nullptr;
static PFN_vkGetInstanceProcAddr g_gipa = nullptr;

#define LOAD_G(name) g_vk.name = (PFN_vk##name)g_gipa(nullptr, "vk" #name)
#define LOAD(name) g_vk.name = (PFN_vk##name)g_gipa(g_vk.inst, "vk" #name)

struct Vk;
struct Vk {
  VkInstance inst{VK_NULL_HANDLE};
  VkDevice dev{VK_NULL_HANDLE};
  VkQueue queue{VK_NULL_HANDLE};
  VkInstance inst_dummy{VK_NULL_HANDLE};  // unused; device procs via dev gpa
  PFN_vkCreateInstance CreateInstance{nullptr};
  PFN_vkDestroyInstance DestroyInstance{nullptr};
  PFN_vkEnumeratePhysicalDevices EnumeratePhysicalDevices{nullptr};
  PFN_vkGetPhysicalDeviceProperties GetPhysicalDeviceProperties{nullptr};
  PFN_vkGetPhysicalDeviceMemoryProperties GetPhysicalDeviceMemoryProperties{
      nullptr};
  PFN_vkGetPhysicalDeviceQueueFamilyProperties
      GetPhysicalDeviceQueueFamilyProperties{nullptr};
  PFN_vkCreateDevice CreateDevice{nullptr};
  PFN_vkGetDeviceProcAddr GetDeviceProcAddr{nullptr};
  PFN_vkGetDeviceQueue GetDeviceQueue{nullptr};
  PFN_vkCreateBuffer CreateBuffer{nullptr};
  PFN_vkGetBufferMemoryRequirements GetBufferMemoryRequirements{nullptr};
  PFN_vkBindBufferMemory BindBufferMemory{nullptr};
  PFN_vkAllocateMemory AllocateMemory{nullptr};
  PFN_vkFreeMemory FreeMemory{nullptr};
  PFN_vkCreateShaderModule CreateShaderModule{nullptr};
  PFN_vkCreateComputePipelines CreateComputePipelines{nullptr};
  PFN_vkCreatePipelineLayout CreatePipelineLayout{nullptr};
  PFN_vkCreateDescriptorSetLayout CreateDescriptorSetLayout{nullptr};
  PFN_vkCreateDescriptorPool CreateDescriptorPool{nullptr};
  PFN_vkAllocateDescriptorSets AllocateDescriptorSets{nullptr};
  PFN_vkUpdateDescriptorSets UpdateDescriptorSets{nullptr};
  PFN_vkCreateCommandPool CreateCommandPool{nullptr};
  PFN_vkAllocateCommandBuffers AllocateCommandBuffers{nullptr};
  PFN_vkBeginCommandBuffer BeginCommandBuffer{nullptr};
  PFN_vkEndCommandBuffer EndCommandBuffer{nullptr};
  PFN_vkCmdBindPipeline CmdBindPipeline{nullptr};
  PFN_vkCmdBindDescriptorSets CmdBindDescriptorSets{nullptr};
  PFN_vkCmdDispatch CmdDispatch{nullptr};
  PFN_vkCmdPushConstants CmdPushConstants{nullptr};
  PFN_vkCmdPipelineBarrier CmdPipelineBarrier{nullptr};
  PFN_vkCmdSetEvent CmdSetEvent{nullptr};
  PFN_vkCmdWaitEvents CmdWaitEvents{nullptr};
  PFN_vkCreateEvent CreateEvent{nullptr};
  PFN_vkCreateQueryPool CreateQueryPool{nullptr};
  PFN_vkCmdResetQueryPool CmdResetQueryPool{nullptr};
  PFN_vkCmdWriteTimestamp CmdWriteTimestamp{nullptr};
  PFN_vkGetQueryPoolResults GetQueryPoolResults{nullptr};
  PFN_vkQueueSubmit QueueSubmit{nullptr};
  PFN_vkQueueWaitIdle QueueWaitIdle{nullptr};
  PFN_vkResetQueryPool ResetQueryPool{nullptr};
  PFN_vkCreateSemaphore CreateSemaphore{nullptr};
  PFN_vkDestroySemaphore DestroySemaphore{nullptr};
  PFN_vkDestroyShaderModule DestroyShaderModule{nullptr};
  PFN_vkDestroyPipeline DestroyPipeline{nullptr};
  PFN_vkDestroyPipelineLayout DestroyPipelineLayout{nullptr};
  PFN_vkDestroyDescriptorSetLayout DestroyDescriptorSetLayout{nullptr};
  PFN_vkDestroyDescriptorPool DestroyDescriptorPool{nullptr};
  PFN_vkDestroyBuffer DestroyBuffer{nullptr};
  PFN_vkDestroyCommandPool DestroyCommandPool{nullptr};
  PFN_vkDestroyDevice DestroyDevice{nullptr};
};
static Vk g_vk;

struct Ctx {
  uint32_t qfi{0};
  float ts_period_ns{1.0f};
  uint32_t ts_valid_bits{24};
  bool timeline_semaphore{false};
  const char* name{""};
  VkPhysicalDeviceMemoryProperties mp{};
};
static Ctx ctx;

[[noreturn]] static void die(const char* fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  fprintf(stderr, "chain-dep-bench: ");
  vfprintf(stderr, fmt, ap);
  fprintf(stderr, "\n");
  va_end(ap);
  exit(1);
}

static double now_us() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}

static void setup_device() {
  vk_lib = dlopen("libvulkan.so.1", RTLD_NOW | RTLD_LOCAL);
  if (!vk_lib) die("dlopen libvulkan.so.1: %s", dlerror());
  g_gipa = (PFN_vkGetInstanceProcAddr)dlsym(vk_lib, "vkGetInstanceProcAddr");
  if (!g_gipa) die("no vkGetInstanceProcAddr");
  LOAD_G(CreateInstance);

  VkApplicationInfo ai{};
  ai.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  ai.pApplicationName = "chain-dep-bench";
  ai.apiVersion = VK_API_VERSION_1_2;
  VkInstanceCreateInfo ici{};
  ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  ici.pApplicationInfo = &ai;
  if (g_vk.CreateInstance(&ici, nullptr, &g_vk.inst) != VK_SUCCESS)
    die("CreateInstance");
  LOAD(GetDeviceProcAddr);
  LOAD(EnumeratePhysicalDevices);
  LOAD(GetPhysicalDeviceProperties);
  LOAD(GetPhysicalDeviceMemoryProperties);
  LOAD(GetPhysicalDeviceQueueFamilyProperties);
  LOAD(CreateDevice);

  uint32_t n = 0;
  g_vk.EnumeratePhysicalDevices(g_vk.inst, &n, nullptr);
  if (n == 0) die("no physical devices");
  std::vector<VkPhysicalDevice> pds(n);
  g_vk.EnumeratePhysicalDevices(g_vk.inst, &n, pds.data());
  for (uint32_t i = 0; i < n; ++i) {
    VkPhysicalDeviceProperties props{};
    PFN_vkGetPhysicalDeviceProperties gpp =
        (PFN_vkGetPhysicalDeviceProperties)g_gipa(
            g_vk.inst, "vkGetPhysicalDeviceProperties");
    gpp(pds[i], &props);
    if (props.apiVersion < VK_API_VERSION_1_2) continue;
    if (!strstr(props.deviceName, "Apple")) continue;
    uint32_t qfn = 0;
    PFN_vkGetPhysicalDeviceQueueFamilyProperties gqfp =
        (PFN_vkGetPhysicalDeviceQueueFamilyProperties)g_gipa(
            g_vk.inst, "vkGetPhysicalDeviceQueueFamilyProperties");
    gqfp(pds[i], &qfn, nullptr);
    std::vector<VkQueueFamilyProperties> qfpv(qfn);
    gqfp(pds[i], &qfn, qfpv.data());
    for (uint32_t q = 0; q < qfn; ++q) {
      if ((qfpv[q].queueFlags & VK_QUEUE_COMPUTE_BIT) == 0) continue;
      ctx.qfi = q;
      ctx.ts_period_ns = props.limits.timestampPeriod;
      ctx.ts_valid_bits = qfpv[q].timestampValidBits;
      ctx.name = props.deviceName;
      {
        PFN_vkGetPhysicalDeviceFeatures2 gpf2 =
            (PFN_vkGetPhysicalDeviceFeatures2)g_gipa(
                g_vk.inst, "vkGetPhysicalDeviceFeatures2");
        if (gpf2) {
          VkPhysicalDeviceVulkan12Features f{
              VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
          VkPhysicalDeviceFeatures2 f2{
              VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
          f2.pNext = &f;
          gpf2(pds[i], &f2);
          ctx.timeline_semaphore = f.timelineSemaphore != 0;
        }
      }
      PFN_vkGetPhysicalDeviceMemoryProperties gmp =
          (PFN_vkGetPhysicalDeviceMemoryProperties)g_gipa(
              g_vk.inst, "vkGetPhysicalDeviceMemoryProperties");
      gmp(pds[i], &ctx.mp);
      float prio = 1.0f;
      VkDeviceQueueCreateInfo qci{};
      qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
      qci.queueFamilyIndex = q;
      qci.queueCount = 1;
      qci.pQueuePriorities = &prio;
      VkDeviceCreateInfo dci{};
      dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
      dci.queueCreateInfoCount = 1;
      dci.pQueueCreateInfos = &qci;
      PFN_vkCreateDevice cd =
          (PFN_vkCreateDevice)g_gipa(g_vk.inst, "vkCreateDevice");
      if (cd(pds[i], &dci, nullptr, &g_vk.dev) != VK_SUCCESS)
        die("CreateDevice");
      break;
    }
    if (g_vk.dev != VK_NULL_HANDLE) break;
  }
  if (g_vk.dev == VK_NULL_HANDLE) die("no Apple Vulkan 1.2 compute device");

#define DEV(fn) g_vk.fn = (PFN_vk##fn)g_vk.GetDeviceProcAddr(g_vk.dev, "vk" #fn)
  DEV(GetDeviceQueue);
  DEV(CreateBuffer);
  DEV(GetBufferMemoryRequirements);
  DEV(BindBufferMemory);
  DEV(AllocateMemory);
  DEV(FreeMemory);
  DEV(CreateShaderModule);
  DEV(CreateComputePipelines);
  DEV(CreatePipelineLayout);
  DEV(CreateDescriptorSetLayout);
  DEV(CreateDescriptorPool);
  DEV(AllocateDescriptorSets);
  DEV(UpdateDescriptorSets);
  DEV(CreateCommandPool);
  DEV(AllocateCommandBuffers);
  DEV(BeginCommandBuffer);
  DEV(EndCommandBuffer);
  DEV(CmdBindPipeline);
  DEV(CmdBindDescriptorSets);
  DEV(CmdDispatch);
  DEV(CmdPushConstants);
  DEV(CmdPipelineBarrier);
  DEV(CmdSetEvent);
  DEV(CmdWaitEvents);
  DEV(CreateEvent);
  DEV(CreateQueryPool);
  DEV(CmdResetQueryPool);
  DEV(CmdWriteTimestamp);
  DEV(GetQueryPoolResults);
  DEV(QueueSubmit);
  DEV(QueueWaitIdle);
  DEV(ResetQueryPool);
  DEV(CreateSemaphore);
  DEV(DestroySemaphore);
  DEV(DestroyShaderModule);
  DEV(DestroyPipeline);
  DEV(DestroyPipelineLayout);
  DEV(DestroyDescriptorSetLayout);
  DEV(DestroyDescriptorPool);
  DEV(DestroyBuffer);
  DEV(DestroyCommandPool);
  DEV(DestroyDevice);
#undef DEV
  { const char* core[] = {"GetDeviceQueue", "QueueSubmit", "QueueWaitIdle",
      "CreateSemaphore", "CreateQueryPool", "CmdWriteTimestamp",
      "GetQueryPoolResults", "CmdPipelineBarrier", "CmdDispatch", nullptr};
    for (int i = 0; core[i]; ++i) {
      void* p = nullptr;
      // resolved already; re-resolving by name is easiest and cheap
      p = (void*)g_vk.GetDeviceProcAddr(g_vk.dev,
          (std::string("vk") + core[i]).c_str());
      if (!p) die("driver does not export %s", core[i]);
    } }
  g_vk.GetDeviceQueue(g_vk.dev, ctx.qfi, 0, &g_vk.queue);
  printf("{\"k\":\"meta\",\"dev\":\"%s\",\"ts_period_ns\":%.3f,"
         "\"ts_valid_bits\":%u,\"timeline_semaphore\":%d,"
         "\"vk_driver_files\":\"%s\"}\n",
         ctx.name, ctx.ts_period_ns, ctx.ts_valid_bits,
         ctx.timeline_semaphore ? 1 : 0,
         getenv("VK_DRIVER_FILES") ? getenv("VK_DRIVER_FILES") : "");
  fflush(stdout);
}

// ---- buffers / shaders / pipelines ----------------------------------------

struct Buf {
  VkBuffer buf{VK_NULL_HANDLE};
  VkDeviceMemory mem{VK_NULL_HANDLE};
};
static uint32_t find_memtype(uint32_t bits, VkMemoryPropertyFlags want) {
  for (uint32_t i = 0; i < ctx.mp.memoryTypeCount; ++i) {
    if ((bits & (1u << i)) &&
        (ctx.mp.memoryTypes[i].propertyFlags & want) == want)
      return i;
  }
  return UINT32_MAX;
}
static Buf make_buf(VkDeviceSize size) {
  Buf b;
  VkBufferCreateInfo bi{};
  bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  bi.size = size;
  bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
  if (g_vk.CreateBuffer(g_vk.dev, &bi, nullptr, &b.buf) != VK_SUCCESS)
    die("CreateBuffer");
  VkMemoryRequirements req;
  g_vk.GetBufferMemoryRequirements(g_vk.dev, b.buf, &req);
  uint32_t mt = find_memtype(req.memoryTypeBits,
      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  if (mt == UINT32_MAX)
    mt = find_memtype(req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
  if (mt == UINT32_MAX) die("no memtype");
  VkMemoryAllocateInfo mai{};
  mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  mai.allocationSize = req.size;
  mai.memoryTypeIndex = mt;
  if (g_vk.AllocateMemory(g_vk.dev, &mai, nullptr, &b.mem) != VK_SUCCESS)
    die("AllocateMemory");
  if (g_vk.BindBufferMemory(g_vk.dev, b.buf, b.mem, 0) != VK_SUCCESS)
    die("BindBufferMemory");
  return b;
}

static const char* kShaderHead =
    "#version 450\n"
    "layout(local_size_x = 32, local_size_y = 1, local_size_z = 1) in;\n"
    "layout(set = 0, binding = 0) buffer Out { uint out_buf[]; };\n"
    "layout(push_constant) uniform PC { uint pc_val; } pc;\n"
    "void main() { out_buf[gl_WorkGroupID.x] = pc.pc_val; }\n";

static std::string compile_shader() {
  char path_src[] = "/tmp/cdb-src-XXXXXX.comp";
  char path_spv[] = "/tmp/cdb-out-XXXXXX.spv";
  int fd = mkstemps(path_src, 5);
  if (fd < 0) die("mkstemps src");
  if (write(fd, kShaderHead, strlen(kShaderHead)) != (ssize_t)strlen(kShaderHead))
    die("write src");
  close(fd);
  fd = mkstemps(path_spv, 4);
  if (fd < 0) die("mkstemps spv");
  close(fd);
  std::string cmd = std::string("glslc -fshader-stage=compute "
      "--target-env=vulkan1.3 ") + path_src + " -o " + path_spv + " 2>&1";
  if (system(cmd.c_str()) != 0) {
    cmd = std::string("glslangValidator -V --target-env vulkan1.3 ") +
          path_src + " -o " + path_spv + " 2>&1";
    if (system(cmd.c_str()) != 0) die("shader compile failed");
  }
  std::ifstream f(path_spv, std::ios::binary);
  std::ostringstream ss;
  ss << f.rdbuf();
  unlink(path_src);
  unlink(path_spv);
  return ss.str();
}

struct Bench {
  VkDescriptorSetLayout dsl{VK_NULL_HANDLE};
  VkPipelineLayout layout{VK_NULL_HANDLE};
  VkDescriptorPool pool{VK_NULL_HANDLE};
  VkDescriptorSet set{VK_NULL_HANDLE};
  Buf out;
  VkCommandPool cpool{VK_NULL_HANDLE};
  VkCommandBuffer cmd[3]{VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE};
  VkQueryPool qpool{VK_NULL_HANDLE};
  VkShaderModule mod{VK_NULL_HANDLE};
  VkPipeline pipe{VK_NULL_HANDLE};
};

static void setup_bench(Bench& b) {
  VkDescriptorSetLayoutBinding bind{};
  bind.binding = 0;
  bind.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  bind.descriptorCount = 1;
  bind.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  VkDescriptorSetLayoutCreateInfo dlci{};
  dlci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  dlci.bindingCount = 1;
  dlci.pBindings = &bind;
  if (g_vk.CreateDescriptorSetLayout(g_vk.dev, &dlci, nullptr, &b.dsl) !=
      VK_SUCCESS) die("CreateDescriptorSetLayout");
  VkPushConstantRange pcr{};
  pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  pcr.offset = 0;
  pcr.size = 128;
  VkPipelineLayoutCreateInfo plci{};
  plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  plci.setLayoutCount = 1;
  plci.pSetLayouts = &b.dsl;
  plci.pushConstantRangeCount = 1;
  plci.pPushConstantRanges = &pcr;
  if (g_vk.CreatePipelineLayout(g_vk.dev, &plci, nullptr, &b.layout) !=
      VK_SUCCESS) die("CreatePipelineLayout");

  VkDescriptorPoolSize ps{};
  ps.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  ps.descriptorCount = 1;
  VkDescriptorPoolCreateInfo pci{};
  pci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  pci.maxSets = 1;
  pci.poolSizeCount = 1;
  pci.pPoolSizes = &ps;
  if (g_vk.CreateDescriptorPool(g_vk.dev, &pci, nullptr, &b.pool) !=
      VK_SUCCESS) die("CreateDescriptorPool");
  VkDescriptorSetAllocateInfo dsai{};
  dsai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  dsai.descriptorPool = b.pool;
  dsai.descriptorSetCount = 1;
  dsai.pSetLayouts = &b.dsl;
  if (g_vk.AllocateDescriptorSets(g_vk.dev, &dsai, &b.set) != VK_SUCCESS)
    die("AllocateDescriptorSets");
  b.out = make_buf(1 << 12);
  VkDescriptorBufferInfo dbi{};
  dbi.buffer = b.out.buf;
  dbi.offset = 0;
  dbi.range = VK_WHOLE_SIZE;
  VkWriteDescriptorSet wr{};
  wr.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  wr.dstSet = b.set;
  wr.dstBinding = 0;
  wr.descriptorCount = 1;
  wr.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  wr.pBufferInfo = &dbi;
  g_vk.UpdateDescriptorSets(g_vk.dev, 1, &wr, 0, nullptr);

  std::string spv = compile_shader();
  VkShaderModuleCreateInfo mi{};
  mi.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  mi.codeSize = spv.size();
  mi.pCode = (const uint32_t*)spv.data();
  if (g_vk.CreateShaderModule(g_vk.dev, &mi, nullptr, &b.mod) != VK_SUCCESS)
    die("CreateShaderModule");
  VkComputePipelineCreateInfo cpi{};
  cpi.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
  cpi.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  cpi.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  cpi.stage.module = b.mod;
  cpi.stage.pName = "main";
  cpi.layout = b.layout;
  if (g_vk.CreateComputePipelines(g_vk.dev, VK_NULL_HANDLE, 1, &cpi, nullptr,
      &b.pipe) != VK_SUCCESS) die("CreateComputePipelines");

  VkCommandPoolCreateInfo cpci{};
  cpci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  cpci.queueFamilyIndex = ctx.qfi;
  if (g_vk.CreateCommandPool(g_vk.dev, &cpci, nullptr, &b.cpool) != VK_SUCCESS)
    die("CreateCommandPool");
  VkCommandBufferAllocateInfo cbai{};
  cbai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  cbai.commandPool = b.cpool;
  cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  cbai.commandBufferCount = 3;
  if (g_vk.AllocateCommandBuffers(g_vk.dev, &cbai, b.cmd) != VK_SUCCESS)
    die("AllocateCommandBuffers");

  VkQueryPoolCreateInfo qpci{};
  qpci.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
  qpci.queryType = VK_QUERY_TYPE_TIMESTAMP;
  qpci.queryCount = 8;
  if (g_vk.CreateQueryPool(g_vk.dev, &qpci, nullptr, &b.qpool) != VK_SUCCESS)
    die("CreateQueryPool");
}

static void begin_cmd(VkCommandBuffer c) {
  VkCommandBufferBeginInfo bi{};
  bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  g_vk.BeginCommandBuffer(c, &bi);
}

// Record one dispatch (with TOP/BOT timestamps at queries 2i/2i+1) into c.
static void record_dispatch(Bench& b, VkCommandBuffer c, uint32_t grid,
    uint32_t pc_val) {
  g_vk.CmdBindPipeline(c, VK_PIPELINE_BIND_POINT_COMPUTE, b.pipe);
  g_vk.CmdBindDescriptorSets(c, VK_PIPELINE_BIND_POINT_COMPUTE, b.layout, 0,
      1, &b.set, 0, nullptr);
  g_vk.CmdPushConstants(c, b.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 4,
      &pc_val);
  g_vk.CmdDispatch(c, grid, 1, 1);
}
// One timestamp pair bracketing the whole CS (queries 2*slot, 2*slot+1).
static void cs_span(Bench& b, VkCommandBuffer c, uint32_t slot) {
  g_vk.CmdWriteTimestamp(c, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, b.qpool,
      2 * slot);
  g_vk.CmdWriteTimestamp(c, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, b.qpool,
      2 * slot + 1);
}

static void full_barrier(VkCommandBuffer c) {
  VkMemoryBarrier mb{};
  mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
  mb.srcAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
  mb.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
  g_vk.CmdPipelineBarrier(c, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
      VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
}

static double ticks_to_us(uint64_t t) {
  return (double)t * ctx.ts_period_ns / 1000.0;
}
static uint64_t tick_delta(uint64_t a, uint64_t b) {  // b - a, wrap-safe
  uint64_t mask = ctx.ts_valid_bits >= 64
                      ? ~0ULL
                      : ((1ULL << ctx.ts_valid_bits) - 1);
  uint64_t d = (b - a) & mask;
  return d;
}
static double ticks_between(uint64_t a, uint64_t b) {
  return (double)tick_delta(a, b) * ctx.ts_period_ns / 1000.0;
}

struct RunRes {
  double wall_us;
  double hop01_us;  // GPU gap: end of A .. start of B
  double hop12_us;  // GPU gap: end of B .. start of C
  double span_us;   // start of A .. end of C (GPU)
};

// One measurement rep of a case. `mode` picks the dependency encoding.
static RunRes run_case(Bench& b, const char* mode, uint32_t grid,
    VkSemaphore sem, VkEvent ev, bool& ev_supported, uint64_t vbase) {
  uint32_t pcv = 0x3f;
  if (strcmp(mode, "cs1_none") == 0 || strcmp(mode, "cs1_barrier") == 0 ||
      strcmp(mode, "cs1_event") == 0) {
    g_vk.ResetQueryPool(g_vk.dev, b.qpool, 0, 8);
    begin_cmd(b.cmd[0]);
    record_dispatch(b, b.cmd[0], grid, pcv);
    if (strcmp(mode, "cs1_barrier") == 0) {
      full_barrier(b.cmd[0]);
    } else if (strcmp(mode, "cs1_event") == 0) {
      g_vk.CmdSetEvent(b.cmd[0], ev, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
      g_vk.CmdWaitEvents(b.cmd[0], 1, &ev, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, nullptr, 0, nullptr, 0,
          nullptr);
    }
    record_dispatch(b, b.cmd[0], grid, pcv + 1);
    if (strcmp(mode, "cs1_barrier") == 0) {
      full_barrier(b.cmd[0]);
    } else if (strcmp(mode, "cs1_event") == 0) {
      g_vk.CmdSetEvent(b.cmd[0], ev, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
      g_vk.CmdWaitEvents(b.cmd[0], 1, &ev, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, nullptr, 0, nullptr, 0,
          nullptr);
    }
    record_dispatch(b, b.cmd[0], grid, pcv + 2);
    g_vk.EndCommandBuffer(b.cmd[0]);
    VkSubmitInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &b.cmd[0];
    double t0 = now_us();
    VkResult src = g_vk.QueueSubmit(g_vk.queue, 1, &si, VK_NULL_HANDLE);
    if (src != VK_SUCCESS) {
      // Is the device gone, or just this CB rejected? Empty-CB probe.
      VkCommandBufferBeginInfo bi2{};
      bi2.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
      g_vk.BeginCommandBuffer(b.cmd[1], &bi2);
      g_vk.EndCommandBuffer(b.cmd[1]);
      VkSubmitInfo s2{};
      s2.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
      s2.commandBufferCount = 1;
      s2.pCommandBuffers = &b.cmd[1];
      VkResult rc2 = g_vk.QueueSubmit(g_vk.queue, 1, &s2, VK_NULL_HANDLE);
      VkResult w2 = rc2 == VK_SUCCESS ? g_vk.QueueWaitIdle(g_vk.queue) : rc2;
      die("QueueSubmit cs1 vkrc=%d empty-probe submit=%d wait=%d",
          (int)src, (int)rc2, (int)w2);
    }
    VkResult wrc = g_vk.QueueWaitIdle(g_vk.queue);
    if (wrc != VK_SUCCESS) die("WaitIdle cs1 vkrc=%d", (int)wrc);
    double t1 = now_us();
    return {t1 - t0, 0.0, 0.0, 0.0};
  }

  // Three-CS variants. CB i waits sem value v_i, signals v_i + 1.
  for (int i = 0; i < 3; ++i) {
    begin_cmd(b.cmd[i]);
    g_vk.CmdResetQueryPool(b.cmd[i], b.qpool, 0, 8);
    record_dispatch(b, b.cmd[i], grid, pcv + i);
    g_vk.EndCommandBuffer(b.cmd[i]);
  }
  VkTimelineSemaphoreSubmitInfo tsinfo[3]{};
  VkSemaphore wait_sems[2] = {sem, sem};
  VkSemaphore sig_sems[3] = {sem, sem, sem};
  uint64_t wait_vals[2] = {vbase + 1, vbase + 2};
  uint64_t sig_vals[3] = {vbase + 1, vbase + 2, vbase + 3};
  VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
  VkSubmitInfo si[3]{};
  for (int i = 0; i < 3; ++i) {
    si[i].sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si[i].commandBufferCount = 1;
    si[i].pCommandBuffers = &b.cmd[i];
    si[i].pSignalSemaphores = &sig_sems[i];
    si[i].signalSemaphoreCount = 1;
    si[i].pWaitDstStageMask = &wait_stage;
    si[i].pNext = &tsinfo[i];
    tsinfo[i].sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
  }
  si[1].waitSemaphoreCount = 1;
  si[1].pWaitSemaphores = &wait_sems[0];
  si[2].waitSemaphoreCount = 1;
  si[2].pWaitSemaphores = &wait_sems[1];
  tsinfo[0].signalSemaphoreValueCount = 1;
  tsinfo[0].pSignalSemaphoreValues = &sig_vals[0];
  tsinfo[1].waitSemaphoreValueCount = 1;
  tsinfo[1].pWaitSemaphoreValues = &wait_vals[0];
  tsinfo[1].signalSemaphoreValueCount = 1;
  tsinfo[1].pSignalSemaphoreValues = &sig_vals[1];
  tsinfo[2].waitSemaphoreValueCount = 1;
  tsinfo[2].pWaitSemaphoreValues = &wait_vals[1];
  tsinfo[2].signalSemaphoreValueCount = 1;
  tsinfo[2].pSignalSemaphoreValues = &sig_vals[2];

  double t0 = now_us();
  VkResult qrc = VK_ERROR_UNKNOWN;
  if (strcmp(mode, "cs3_1submit_sema") == 0) {
    qrc = g_vk.QueueSubmit(g_vk.queue, 3, si, VK_NULL_HANDLE);
    if (qrc == VK_SUCCESS) qrc = g_vk.QueueWaitIdle(g_vk.queue);
  } else if (strcmp(mode, "cs3_3submit_sema") == 0) {
    for (int i = 0; i < 3; ++i) {
      qrc = g_vk.QueueSubmit(g_vk.queue, 1, &si[i], VK_NULL_HANDLE);
      if (qrc != VK_SUCCESS) break;
    }
    if (qrc == VK_SUCCESS) qrc = g_vk.QueueWaitIdle(g_vk.queue);
  } else {  // cs3_fencejoin
    for (int i = 0; i < 3; ++i) {
      qrc = g_vk.QueueSubmit(g_vk.queue, 1, &si[i], VK_NULL_HANDLE);
      if (qrc == VK_SUCCESS) qrc = g_vk.QueueWaitIdle(g_vk.queue);
      if (qrc != VK_SUCCESS) break;
    }
  }
  double t1 = now_us();
  if (qrc != VK_SUCCESS) {
    printf("{\"k\":\"fail\",\"mode\":\"%s\",\"vkrc\":%d}\n", mode,
           (int)qrc);
    fflush(stdout);
    RunRes bad{0, 0, 0, 0};
    return bad;
  }
  return {t1 - t0, 0.0, 0.0, 0.0};
}

int main() {
  setup_device();
  Bench b;
  setup_bench(b);

  VkSemaphoreTypeCreateInfo sti{};
  sti.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
  sti.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
  sti.initialValue = 0;
  VkSemaphoreCreateInfo sci{};
  sci.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
  sci.pNext = &sti;
  VkSemaphore sem;
  if (g_vk.CreateSemaphore(g_vk.dev, &sci, nullptr, &sem) != VK_SUCCESS)
    die("CreateSemaphore (timeline)");
  VkEventCreateInfo eci{};
  eci.sType = VK_STRUCTURE_TYPE_EVENT_CREATE_INFO;
  VkEvent ev{VK_NULL_HANDLE};
  bool ev_ok = g_vk.CreateEvent != nullptr &&
      g_vk.CreateEvent(g_vk.dev, &eci, nullptr, &ev) == VK_SUCCESS;
  printf("{\"k\":\"meta2\",\"events\":%d}\n", ev_ok ? 1 : 0);

  // Bisect: what does honeykrisp accept in a submit?
  {
    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    struct T { const char* n; bool q; bool pipe; int ndisp; };
    T ts[] = {{"empty", false, false, 0}, {"ts", true, false, 0},
              {"ts+pipe", true, true, 1},
              {"3pipe-nots", false, true, 3},
              {"3pipe-ts", true, true, 3}};
    for (auto& t : ts) {
      g_vk.BeginCommandBuffer(b.cmd[0], &bi);
      if (t.q) {
        g_vk.CmdResetQueryPool(b.cmd[0], b.qpool, 0, 8);
        g_vk.CmdWriteTimestamp(b.cmd[0], VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
            b.qpool, 0);
      }
      for (int dd = 0; dd < t.ndisp; ++dd) {
        if (t.q)
          g_vk.CmdWriteTimestamp(b.cmd[0], VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
              b.qpool, 2 * dd);
        g_vk.CmdBindPipeline(b.cmd[0], VK_PIPELINE_BIND_POINT_COMPUTE,
            b.pipe);
        g_vk.CmdBindDescriptorSets(b.cmd[0], VK_PIPELINE_BIND_POINT_COMPUTE,
            b.layout, 0, 1, &b.set, 0, nullptr);
        uint32_t v = 7;
        g_vk.CmdPushConstants(b.cmd[0], b.layout,
            VK_SHADER_STAGE_COMPUTE_BIT, 0, 4, &v);
        g_vk.CmdDispatch(b.cmd[0], 1, 1, 1);
        if (t.q)
          g_vk.CmdWriteTimestamp(b.cmd[0],
              VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, b.qpool, 2 * dd + 1);
      }
      g_vk.EndCommandBuffer(b.cmd[0]);
      VkSubmitInfo si{};
      si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
      si.commandBufferCount = 1;
      si.pCommandBuffers = &b.cmd[0];
      VkResult rc = g_vk.QueueSubmit(g_vk.queue, 1, &si, VK_NULL_HANDLE);
      VkResult w = rc == VK_SUCCESS ? g_vk.QueueWaitIdle(g_vk.queue) : rc;
      printf("{\"k\":\"bisect\",\"what\":\"%s\",\"submit\":%d,"
             "\"wait\":%d}\n", t.n, (int)rc, (int)w);
      fflush(stdout);
    }
  }

  const char* modes[] = {"cs1_none", "cs1_barrier", "cs1_event",
      "cs3_1submit_sema", "cs3_3submit_sema", "cs3_fencejoin"};
  const uint32_t REPS = 31, WARM = 7;
  for (const char* mode : modes) {
    std::vector<double> walls, h0, h1, spans;
    bool ev_supported = ev_ok;
    for (uint32_t r = 0; r < WARM + REPS; ++r) {
      RunRes rr = run_case(b, mode, 1, sem, ev, ev_supported, r + 1);
      if (r >= WARM) {
        walls.push_back(rr.wall_us);
        h0.push_back(rr.hop01_us);
        h1.push_back(rr.hop12_us);
        spans.push_back(rr.span_us);
      }
    }
    auto med = [](std::vector<double> v) {
      std::sort(v.begin(), v.end());
      return v[v.size() / 2];
    };
    printf(
        "{\"k\":\"case\",\"name\":\"%s\",\"n\":%u,"
        "\"wall_med_us\":%.1f,\"wall_min_us\":%.1f,"
        "\"hop01_med_us\":%.2f,\"hop12_med_us\":%.2f,\"span_med_us\":%.2f}\n",
        mode, REPS, med(walls),
        *std::min_element(walls.begin(), walls.end()), med(h0), med(h1),
        med(spans));
    fflush(stdout);
  }
  printf("{\"k\":\"end\"}\n");
  return 0;
}
