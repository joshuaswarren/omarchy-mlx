// Per-dispatch floor microbenchmark for Honeykrisp/AGX (Apple M1).
//
// Attributes the ~28-30 us per-dispatch floor to a specific driver operation
// by sweeping, inside one command buffer unless stated otherwise:
//   empty vs trivial kernel, workgroup count, local size, Vulkan barrier,
//   descriptor rebinding, push-constant updates, pipeline switching,
//   per-dispatch timestamp brackets, and one-dispatch-per-submit round trips.
//
// Timing: host CLOCK_MONOTONIC around QueueSubmit..QueueWaitIdle is the
// absolute scale (honeykrisp timestamp periods are not trustworthy in
// absolute terms, see receipts/2026-09-10-q4-gemv-bandwidth/verdict.json);
// device timestamps give same-run ratios and per-dispatch brackets.
//
// NDJSON on stdout. Run under the GPU lock on the M1:
//   flock -w 2400 /tmp/m1-gpu.lock timeout 900 bash tools/dispatch-floor-bench/run-m1.sh

#include <vulkan/vulkan.h>
#include <dlfcn.h>
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

#define LIBVK "libvulkan.so.1"

#define LOAD(name) g_vk.name = (PFN_vk##name)vk_load(g_vk.inst, "vk" #name)
#define LOAD_DEV(name) \
  g_vk.name = (PFN_vk##name)vk_dev_load(g_vk.dev, "vk" #name)

struct VkTable;
struct VkTable {
  void* handle{nullptr};
  VkInstance inst{VK_NULL_HANDLE};
  VkDevice dev{VK_NULL_HANDLE};
  VkQueue queue{VK_NULL_HANDLE};
  PFN_vkGetInstanceProcAddr GetInstanceProcAddr{nullptr};
  PFN_vkGetDeviceProcAddr GetDeviceProcAddr{nullptr};
#define VT(fn) PFN_vk##fn fn{nullptr};
  VT(CreateInstance) VT(EnumeratePhysicalDevices) VT(DestroyInstance)
  VT(GetPhysicalDeviceProperties) VT(GetPhysicalDeviceMemoryProperties)
  VT(GetPhysicalDeviceQueueFamilyProperties) VT(CreateDevice)
  VT(GetDeviceQueue) VT(DestroyDevice)
  VT(CreateBuffer) VT(GetBufferMemoryRequirements) VT(BindBufferMemory)
  VT(AllocateMemory) VT(FreeMemory) VT(MapMemory) VT(UnmapMemory)
  VT(CreateShaderModule) VT(CreateComputePipelines) VT(CreatePipelineLayout)
  VT(CreateDescriptorSetLayout) VT(CreateDescriptorPool)
  VT(AllocateDescriptorSets) VT(UpdateDescriptorSets)
  VT(CreateCommandPool) VT(AllocateCommandBuffers) VT(DestroyCommandPool)
  VT(BeginCommandBuffer) VT(EndCommandBuffer)
  VT(CmdBindPipeline) VT(CmdBindDescriptorSets) VT(CmdDispatch)
  VT(CmdPushConstants) VT(CmdPipelineBarrier)
  VT(CreateQueryPool) VT(CmdResetQueryPool) VT(CmdWriteTimestamp)
  VT(GetQueryPoolResults)
  VT(QueueSubmit) VT(QueueWaitIdle)
  VT(DestroyShaderModule) VT(DestroyPipeline) VT(DestroyPipelineLayout)
  VT(DestroyDescriptorSetLayout) VT(DestroyDescriptorPool)
  VT(DestroyBuffer)
#undef VT
};
static VkTable g_vk;

static PFN_vkVoidFunction vk_load(VkInstance h, const char* name) {
  return g_vk.GetInstanceProcAddr(h, name);
}
static PFN_vkVoidFunction vk_dev_load(VkDevice d, const char* name) {
  return g_vk.GetDeviceProcAddr(d, name);
}

static double now_us() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}

[[noreturn]] static void die(const char* fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  std::vfprintf(stderr, fmt, ap);
  va_end(ap);
  fputc('\n', stderr);
  exit(1);
}

static int vk_init() {
  g_vk.handle = dlopen(LIBVK, RTLD_NOW | RTLD_LOCAL);
  if (!g_vk.handle) die("dlopen %s: %s", LIBVK, dlerror());
  g_vk.GetInstanceProcAddr =
      (PFN_vkGetInstanceProcAddr)dlsym(g_vk.handle, "vkGetInstanceProcAddr");
  g_vk.GetDeviceProcAddr =
      (PFN_vkGetDeviceProcAddr)dlsym(g_vk.handle, "vkGetDeviceProcAddr");
  if (!g_vk.GetInstanceProcAddr || !g_vk.GetDeviceProcAddr)
    die("vk proc symbols missing");
  g_vk.CreateInstance = (PFN_vkCreateInstance)vk_load(nullptr, "vkCreateInstance");
  if (!g_vk.CreateInstance) die("vkCreateInstance missing");
  return 0;
}

struct DeviceCtx {
  uint32_t qfi{0};
  float timestamp_period_ns{1.0f};
  std::string name;
  VkPhysicalDeviceMemoryProperties mp{};
};
static DeviceCtx ctx;

static void setup_device() {
  VkApplicationInfo ai{};
  ai.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  ai.pApplicationName = "dispatch-floor-bench";
  ai.apiVersion = VK_API_VERSION_1_2;
  VkInstanceCreateInfo ici{};
  ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  ici.pApplicationInfo = &ai;
  if (g_vk.CreateInstance(&ici, nullptr, &g_vk.inst) != VK_SUCCESS)
    die("CreateInstance");
  LOAD(DestroyInstance);
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
    g_vk.GetPhysicalDeviceProperties(pds[i], &props);
    if (props.apiVersion < VK_API_VERSION_1_2) continue;
    if (!strstr(props.deviceName, "Apple")) continue;
    uint32_t qfn = 0;
    g_vk.GetPhysicalDeviceQueueFamilyProperties(pds[i], &qfn, nullptr);
    std::vector<VkQueueFamilyProperties> qfpv(qfn);
    g_vk.GetPhysicalDeviceQueueFamilyProperties(pds[i], &qfn, qfpv.data());
    for (uint32_t q = 0; q < qfn; ++q) {
      if ((qfpv[q].queueFlags & VK_QUEUE_COMPUTE_BIT) == 0) continue;
      ctx.qfi = q;
      ctx.timestamp_period_ns = props.limits.timestampPeriod;
      ctx.name = props.deviceName;
      g_vk.GetPhysicalDeviceMemoryProperties(pds[i], &ctx.mp);
      float prio = 1.0f;
      VkDeviceQueueCreateInfo qci{};
      qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
      qci.queueFamilyIndex = q;
      qci.queueCount = 1;
      qci.pQueuePriorities = &prio;
      VkPhysicalDeviceVulkan12Features f12{
          VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
      f12.shaderFloat16 = VK_TRUE;
      VkDeviceCreateInfo dci{};
      dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
      dci.pNext = &f12;
      dci.queueCreateInfoCount = 1;
      dci.pQueueCreateInfos = &qci;
      if (g_vk.CreateDevice(pds[i], &dci, nullptr, &g_vk.dev) != VK_SUCCESS)
        die("CreateDevice");
      LOAD_DEV(GetDeviceQueue);
      g_vk.GetDeviceQueue(g_vk.dev, q, 0, &g_vk.queue);
      LOAD_DEV(DestroyDevice);
      LOAD_DEV(CreateBuffer); LOAD_DEV(GetBufferMemoryRequirements);
      LOAD_DEV(BindBufferMemory); LOAD_DEV(AllocateMemory);
      LOAD_DEV(FreeMemory); LOAD_DEV(MapMemory); LOAD_DEV(UnmapMemory);
      LOAD_DEV(CreateShaderModule); LOAD_DEV(CreateComputePipelines);
      LOAD_DEV(CreatePipelineLayout); LOAD_DEV(CreateDescriptorSetLayout);
      LOAD_DEV(CreateDescriptorPool); LOAD_DEV(AllocateDescriptorSets);
      LOAD_DEV(UpdateDescriptorSets);
      LOAD_DEV(CreateCommandPool); LOAD_DEV(AllocateCommandBuffers);
      LOAD_DEV(DestroyCommandPool);
      LOAD_DEV(BeginCommandBuffer); LOAD_DEV(EndCommandBuffer);
      LOAD_DEV(CmdBindPipeline); LOAD_DEV(CmdBindDescriptorSets);
      LOAD_DEV(CmdDispatch); LOAD_DEV(CmdPushConstants);
      LOAD_DEV(CmdPipelineBarrier);
      LOAD_DEV(CreateQueryPool); LOAD_DEV(CmdResetQueryPool);
      LOAD_DEV(CmdWriteTimestamp); LOAD_DEV(GetQueryPoolResults);
      LOAD_DEV(QueueSubmit); LOAD_DEV(QueueWaitIdle);
      LOAD_DEV(DestroyShaderModule); LOAD_DEV(DestroyPipeline);
      LOAD_DEV(DestroyPipelineLayout); LOAD_DEV(DestroyDescriptorSetLayout);
      LOAD_DEV(DestroyDescriptorPool); LOAD_DEV(DestroyBuffer);
      const char* missing = nullptr;
      auto chk = [&](void* p, const char* n) {
        if (!p && !missing) missing = n;
      };
      chk((void*)g_vk.GetDeviceQueue, "GetDeviceQueue");
      chk((void*)g_vk.CreateBuffer, "CreateBuffer");
      chk((void*)g_vk.BindBufferMemory, "BindBufferMemory");
      chk((void*)g_vk.AllocateMemory, "AllocateMemory");
      chk((void*)g_vk.CreateShaderModule, "CreateShaderModule");
      chk((void*)g_vk.CreateComputePipelines, "CreateComputePipelines");
      chk((void*)g_vk.CreatePipelineLayout, "CreatePipelineLayout");
      chk((void*)g_vk.CreateDescriptorSetLayout, "CreateDescriptorSetLayout");
      chk((void*)g_vk.CreateDescriptorPool, "CreateDescriptorPool");
      chk((void*)g_vk.AllocateDescriptorSets, "AllocateDescriptorSets");
      chk((void*)g_vk.UpdateDescriptorSets, "UpdateDescriptorSets");
      chk((void*)g_vk.CreateCommandPool, "CreateCommandPool");
      chk((void*)g_vk.AllocateCommandBuffers, "AllocateCommandBuffers");
      chk((void*)g_vk.BeginCommandBuffer, "BeginCommandBuffer");
      chk((void*)g_vk.EndCommandBuffer, "EndCommandBuffer");
      chk((void*)g_vk.CmdBindPipeline, "CmdBindPipeline");
      chk((void*)g_vk.CmdBindDescriptorSets, "CmdBindDescriptorSets");
      chk((void*)g_vk.CmdDispatch, "CmdDispatch");
      chk((void*)g_vk.CmdPushConstants, "CmdPushConstants");
      chk((void*)g_vk.CmdPipelineBarrier, "CmdPipelineBarrier");
      chk((void*)g_vk.CreateQueryPool, "CreateQueryPool");
      chk((void*)g_vk.CmdResetQueryPool, "CmdResetQueryPool");
      chk((void*)g_vk.CmdWriteTimestamp, "CmdWriteTimestamp");
      chk((void*)g_vk.GetQueryPoolResults, "GetQueryPoolResults");
      chk((void*)g_vk.QueueSubmit, "QueueSubmit");
      chk((void*)g_vk.QueueWaitIdle, "QueueWaitIdle");
      if (missing) die("unresolved device function: %s", missing);
      printf("{\"k\":\"meta\",\"dev\":\"%s\",\"ts_period_ns\":%.3f,"
             "\"vk_driver_files\":\"%s\",\"hk_perftest\":\"%s\"}\n",
             props.deviceName, ctx.timestamp_period_ns,
             getenv("VK_DRIVER_FILES") ? getenv("VK_DRIVER_FILES") : "",
             getenv("HK_PERFTEST") ? getenv("HK_PERFTEST") : "");
      fflush(stdout);
      return;
    }
  }
  die("no Apple Vulkan 1.2 compute device");
}

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

static VkShaderModule make_module(const std::string& spv) {
  VkShaderModuleCreateInfo mi{};
  mi.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  mi.codeSize = spv.size();
  mi.pCode = (const uint32_t*)spv.data();
  VkShaderModule m;
  if (g_vk.CreateShaderModule(g_vk.dev, &mi, nullptr, &m) != VK_SUCCESS)
    die("CreateShaderModule");
  return m;
}

// One compute pipeline set: shared layout (1 SSBO + 128B push constants).
struct Pipe {
  VkShaderModule mod{VK_NULL_HANDLE};
  VkPipeline pipe{VK_NULL_HANDLE};
};
struct Bench {
  VkDescriptorSetLayout dsl{VK_NULL_HANDLE};
  VkPipelineLayout layout{VK_NULL_HANDLE};
  VkDescriptorPool pool{VK_NULL_HANDLE};
  VkDescriptorSet set_a{VK_NULL_HANDLE}, set_b{VK_NULL_HANDLE};
  Buf out;
  VkCommandPool cpool{VK_NULL_HANDLE};
  VkCommandBuffer cmd{VK_NULL_HANDLE};
  VkQueryPool qpool{VK_NULL_HANDLE};
  uint32_t qpool_n{0};
  Pipe empty, trivial, tpc, nop, trivial_l32, trivial_l128, trivial_l256,
       trivial_l512, trivial_b;  // trivial_b: second identical pipeline
};


// BODY variants
static const char* kBodyEmpty = R"GLSL(
void main() {}
)GLSL";
static const char* kBodyTrivial = R"GLSL(
void main() { out_buf[gl_WorkGroupID.x] = gl_WorkGroupID.x; }
)GLSL";
static const char* kBodyTpc = R"GLSL(
void main() { out_buf[gl_WorkGroupID.x] = pc.pc_val; }
)GLSL";
static const char* kBodyNop = R"GLSL(
void main() { if (pc.pc_val == 0x7fffffffu) out_buf[0] = 1u; }
)GLSL";
static const char* kShaderHead =
    "#version 450\n"
    "layout(local_size_x = ";
static const char* kShaderMid =
    ", local_size_y = 1, local_size_z = 1) in;\n"
    "layout(set = 0, binding = 0) buffer Out { uint out_buf[]; };\n"
    "layout(push_constant) uniform PC { uint pc_val; } pc;\n";
static std::string compile_shader(const char* body, const char* local_x) {
  char path_src[] = "/tmp/dfb-src-XXXXXX.comp";
  char path_spv[] = "/tmp/dfb-out-XXXXXX.spv";
  int fd;
  fd = mkstemps(path_src, 5);
  if (fd < 0) die("mkstemps src");
  std::string src =
      std::string(kShaderHead) + local_x + kShaderMid + body;
  if (write(fd, src.data(), src.size()) != (ssize_t)src.size())
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
    if (system(cmd.c_str()) != 0) die("shader compile failed:\n%s", src.c_str());
  }
  std::ifstream f(path_spv, std::ios::binary);
  std::ostringstream ss;
  ss << f.rdbuf();
  unlink(path_src);
  unlink(path_spv);
  return ss.str();
}

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
  ps.descriptorCount = 2;
  VkDescriptorPoolCreateInfo pci{};
  pci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  pci.maxSets = 2;
  pci.poolSizeCount = 1;
  pci.pPoolSizes = &ps;
  if (g_vk.CreateDescriptorPool(g_vk.dev, &pci, nullptr, &b.pool) !=
      VK_SUCCESS) die("CreateDescriptorPool");
  VkDescriptorSetAllocateInfo dsai{};
  dsai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  dsai.descriptorPool = b.pool;
  dsai.descriptorSetCount = 1;
  dsai.pSetLayouts = &b.dsl;
  VkDescriptorBufferInfo dbi[2];
  b.out = make_buf(1 << 20);
  for (int i = 0; i < 2; ++i) {
    if (g_vk.AllocateDescriptorSets(g_vk.dev, &dsai,
        i == 0 ? &b.set_a : &b.set_b) != VK_SUCCESS) die("AllocateDescriptorSets");
    dbi[i].buffer = b.out.buf;
    dbi[i].offset = 0;
    dbi[i].range = VK_WHOLE_SIZE;
  }
  VkWriteDescriptorSet wr[2]{};
  for (int i = 0; i < 2; ++i) {
    wr[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    wr[i].dstSet = (i == 0) ? b.set_a : b.set_b;
    wr[i].dstBinding = 0;
    wr[i].descriptorCount = 1;
    wr[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    wr[i].pBufferInfo = &dbi[i];
  }
  g_vk.UpdateDescriptorSets(g_vk.dev, 2, wr, 0, nullptr);

  VkCommandPoolCreateInfo cpci{};
  cpci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  cpci.queueFamilyIndex = ctx.qfi;
  if (g_vk.CreateCommandPool(g_vk.dev, &cpci, nullptr, &b.cpool) != VK_SUCCESS)
    die("CreateCommandPool");
  VkCommandBufferAllocateInfo cbai{};
  cbai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  cbai.commandPool = b.cpool;
  cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  cbai.commandBufferCount = 1;
  if (g_vk.AllocateCommandBuffers(g_vk.dev, &cbai, &b.cmd) != VK_SUCCESS)
    die("AllocateCommandBuffers");

  b.qpool_n = 2 * 4096 + 8;
  VkQueryPoolCreateInfo qpci{};
  qpci.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
  qpci.queryType = VK_QUERY_TYPE_TIMESTAMP;
  qpci.queryCount = b.qpool_n;
  if (g_vk.CreateQueryPool(g_vk.dev, &qpci, nullptr, &b.qpool) != VK_SUCCESS)
    die("CreateQueryPool");
}

static VkPipeline make_pipe(Bench& b, VkShaderModule mod) {
  VkComputePipelineCreateInfo cpi{};
  cpi.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
  cpi.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  cpi.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  cpi.stage.module = mod;
  cpi.stage.pName = "main";
  cpi.layout = b.layout;
  VkPipeline p;
  if (g_vk.CreateComputePipelines(g_vk.dev, VK_NULL_HANDLE, 1, &cpi, nullptr,
      &p) != VK_SUCCESS) die("CreateComputePipelines");
  return p;
}

struct RecCfg {
  const Pipe* pipe{nullptr};
  const Pipe* pipe_alt{nullptr};
  uint32_t grid{1};
  uint32_t n{512};
  bool barrier{false};
  // 0 = none, 1 = post c2c barrier, 2 = exact MLX pre+post production pair,
  // 3 = full ALL_COMMANDS/MEMORY both-ways barrier before each dispatch
  // (encoder.cpp record_dependency_barrier shape).
  int barrier_mode{0};
  bool rebind{false};
  bool pushconst{false};
  bool pipeswitch{false};
  bool ts_per_dispatch{false};
  uint32_t push_val{0x3f};
};

// Records the batch and submits it; returns host wall us and gpu ticks.
struct RecRes {
  double wall_us;
  double record_us;
  uint64_t t0, t1;
  std::vector<uint64_t> per_disp;  // only when ts_per_dispatch
};

static RecRes run_recorded(Bench& b, const RecCfg& c) {
  VkCommandBuffer cmd = b.cmd;
  double rec0 = now_us();
  VkCommandBufferBeginInfo bi{};
  bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  g_vk.BeginCommandBuffer(cmd, &bi);
  g_vk.CmdResetQueryPool(cmd, b.qpool, 0, c.ts_per_dispatch ? 2 * c.n + 2 : 2);
  g_vk.CmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, b.qpool, 0);
  g_vk.CmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, c.pipe->pipe);
  g_vk.CmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, b.layout, 0,
      1, &b.set_a, 0, nullptr);
  uint32_t q = 2;
  for (uint32_t i = 0; i < c.n; ++i) {
    if (c.pipeswitch && (i & 1))
      g_vk.CmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
          c.pipe_alt->pipe);
    if (c.rebind) {
      VkDescriptorSet s = (i & 1) ? b.set_b : b.set_a;
      g_vk.CmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, b.layout,
          0, 1, &s, 0, nullptr);
    }
    if (c.pushconst) {
      uint32_t v = c.push_val + i;
      g_vk.CmdPushConstants(cmd, b.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 4,
          &v);
    }
    if (c.ts_per_dispatch)
      g_vk.CmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, b.qpool,
          q++);
    if (c.barrier_mode == 2) {
      // Exact MLX production pre-dispatch barrier (encoder.cpp, gated off):
      // src HOST|TRANSFER|COMPUTE writes, dst COMPUTE read/write.
      VkMemoryBarrier before{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
      before.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT |
          VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT;
      before.dstAccessMask =
          VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
      g_vk.CmdPipelineBarrier(cmd,
          VK_PIPELINE_STAGE_HOST_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT |
              VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &before, 0, nullptr,
          0, nullptr);
    } else if (c.barrier_mode == 3) {
      // Full dependency barrier (encoder.cpp record_dependency_barrier).
      VkMemoryBarrier full{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
      full.srcAccessMask =
          VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
      full.dstAccessMask =
          VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
      g_vk.CmdPipelineBarrier(cmd,
          VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
          VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &full, 0, nullptr, 0,
          nullptr);
    }
    g_vk.CmdDispatch(cmd, c.grid, 1, 1);
    if (c.ts_per_dispatch)
      g_vk.CmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, b.qpool,
          q++);
    if (c.barrier_mode == 1) {
      VkMemoryBarrier mb{};
      mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
      mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
      mb.dstAccessMask =
          VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
      VkPipelineStageFlags src = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
      VkPipelineStageFlags dst = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
      g_vk.CmdPipelineBarrier(cmd, src, dst, 0, 1, &mb, 0, nullptr, 0, nullptr);
    } else if (c.barrier_mode == 2) {
      // Exact MLX production post-dispatch barrier (encoder.cpp, gated off).
      VkMemoryBarrier after{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
      after.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
      after.dstAccessMask = VK_ACCESS_SHADER_READ_BIT |
          VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_HOST_READ_BIT;
      g_vk.CmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT |
              VK_PIPELINE_STAGE_HOST_BIT,
          0, 1, &after, 0, nullptr, 0, nullptr);
    }
  }
  g_vk.CmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, b.qpool, 1);
  g_vk.EndCommandBuffer(cmd);
  double rec1 = now_us();

  VkSubmitInfo si{};
  si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  si.commandBufferCount = 1;
  si.pCommandBuffers = &cmd;
  double t0 = now_us();
  if (g_vk.QueueSubmit(g_vk.queue, 1, &si, VK_NULL_HANDLE) != VK_SUCCESS)
    die("QueueSubmit");
  if (g_vk.QueueWaitIdle(g_vk.queue) != VK_SUCCESS) die("QueueWaitIdle");
  double t1 = now_us();

  uint32_t qn = c.ts_per_dispatch ? 2 * c.n + 2 : 2;
  std::vector<uint64_t> ticks(qn);
  if (g_vk.GetQueryPoolResults(g_vk.dev, b.qpool, 0, qn, qn * sizeof(uint64_t),
      ticks.data(), sizeof(uint64_t),
      VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT) != VK_SUCCESS)
    die("GetQueryPoolResults");

  RecRes r;
  r.record_us = rec1 - rec0;
  r.wall_us = t1 - t0;
  r.t0 = ticks[0];
  r.t1 = ticks[1];
  if (c.ts_per_dispatch) {
    r.per_disp.resize(c.n);
    for (uint32_t i = 0; i < c.n; ++i)
      r.per_disp[i] = ticks[2 * i + 3] - ticks[2 * i + 2];
  }
  return r;
}

// One dispatch per submit + QueueWaitIdle round trip.
static double submit_loop(Bench& b, const Pipe* pipe, uint32_t grid,
    uint32_t iters) {
  VkCommandBuffer cmd = b.cmd;
  double total = 0;
  for (uint32_t i = 0; i < iters; ++i) {
    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    g_vk.BeginCommandBuffer(cmd, &bi);
    g_vk.CmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe->pipe);
    g_vk.CmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, b.layout,
        0, 1, &b.set_a, 0, nullptr);
    g_vk.CmdDispatch(cmd, grid, 1, 1);
    g_vk.EndCommandBuffer(cmd);
    VkSubmitInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;
    double t0 = now_us();
    g_vk.QueueSubmit(g_vk.queue, 1, &si, VK_NULL_HANDLE);
    g_vk.QueueWaitIdle(g_vk.queue);
    total += now_us() - t0;
  }
  return total / iters;
}

// Empty command buffer submit+wait: pure submission floor.
static double null_submit_loop(Bench& b, uint32_t iters) {
  VkCommandBuffer cmd = b.cmd;
  double total = 0;
  for (uint32_t i = 0; i < iters; ++i) {
    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    g_vk.BeginCommandBuffer(cmd, &bi);
    g_vk.EndCommandBuffer(cmd);
    VkSubmitInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;
    double t0 = now_us();
    g_vk.QueueSubmit(g_vk.queue, 1, &si, VK_NULL_HANDLE);
    g_vk.QueueWaitIdle(g_vk.queue);
    total += now_us() - t0;
  }
  return total / iters;
}

static double ticks_to_us(uint64_t ticks) {
  return (double)ticks * ctx.timestamp_period_ns / 1000.0;
}

struct CaseDef {
  const char* name;
  Pipe* pipe;
  Pipe* pipe_alt;
  uint32_t grid;
  uint32_t n;
  bool barrier, rebind, pushconst, pipeswitch, ts_per;
  int barrier_mode{0};
};

int main() {
  vk_init();
  setup_device();
  Bench b;
  setup_bench(b);

  b.empty.mod = make_module(compile_shader(kBodyEmpty, "64"));
  b.trivial.mod = make_module(compile_shader(kBodyTrivial, "64"));
  b.tpc.mod = make_module(compile_shader(kBodyTpc, "64"));
  b.trivial_l32.mod = make_module(compile_shader(kBodyTrivial, "32"));
  b.trivial_l128.mod = make_module(compile_shader(kBodyTrivial, "128"));
  b.trivial_l256.mod = make_module(compile_shader(kBodyTrivial, "256"));
  b.trivial_l512.mod = make_module(compile_shader(kBodyTrivial, "512"));
  b.trivial_b.mod = make_module(compile_shader(kBodyTrivial, "64"));
  b.nop.mod = make_module(compile_shader(kBodyNop, "64"));
  b.empty.pipe = make_pipe(b, b.empty.mod);
  b.trivial.pipe = make_pipe(b, b.trivial.mod);
  b.tpc.pipe = make_pipe(b, b.tpc.mod);
  b.trivial_l32.pipe = make_pipe(b, b.trivial_l32.mod);
  b.trivial_l128.pipe = make_pipe(b, b.trivial_l128.mod);
  b.trivial_l256.pipe = make_pipe(b, b.trivial_l256.mod);
  b.trivial_l512.pipe = make_pipe(b, b.trivial_l512.mod);
  b.trivial_b.pipe = make_pipe(b, b.trivial_b.mod);
  b.nop.pipe = make_pipe(b, b.nop.mod);

  std::vector<CaseDef> cases = {
      {"empty_1wg", &b.empty, nullptr, 1, 512, 0, 0, 0, 0, 0},
      {"empty_8wg", &b.empty, nullptr, 8, 512, 0, 0, 0, 0, 0},
      {"empty_64wg", &b.empty, nullptr, 64, 512, 0, 0, 0, 0, 0},
      {"empty_144wg", &b.empty, nullptr, 144, 512, 0, 0, 0, 0, 0},
      {"empty_1024wg", &b.empty, nullptr, 1024, 256, 0, 0, 0, 0, 0},
      {"empty_4096wg", &b.empty, nullptr, 4096, 64, 0, 0, 0, 0, 0},
      {"trivial_1wg", &b.trivial, nullptr, 1, 512, 0, 0, 0, 0, 0},
      {"trivialpc_1wg", &b.tpc, nullptr, 1, 512, 0, 0, 0, 0, 0},
      {"local32_1wg", &b.trivial_l32, nullptr, 1, 512, 0, 0, 0, 0, 0},
      {"local128_1wg", &b.trivial_l128, nullptr, 1, 512, 0, 0, 0, 0, 0},
      {"local256_1wg", &b.trivial_l256, nullptr, 1, 512, 0, 0, 0, 0, 0},
      {"local512_1wg", &b.trivial_l512, nullptr, 1, 512, 0, 0, 0, 0, 0},
      {"barrier_trivial_1wg", &b.trivial, nullptr, 1, 512, 1, 0, 0, 0, 0},
      {"barrier_empty_1wg", &b.empty, nullptr, 1, 512, 1, 0, 0, 0, 0},
      {"barrierfull_trivial_1wg", &b.trivial, nullptr, 1, 512, 0, 0, 0, 0,
          0, 3},
      {"mlxpair_trivial_1wg", &b.trivial, nullptr, 1, 512, 0, 0, 0, 0, 0, 2},
      {"rebind_trivial_1wg", &b.trivial, nullptr, 1, 512, 0, 1, 0, 0, 0},
      {"pushconst_trivial_1wg", &b.tpc, nullptr, 1, 512, 0, 0, 1, 0, 0},
      {"pipeswitch_trivial_1wg", &b.trivial, &b.trivial_b, 1, 512, 0, 0, 0, 1,
          0},
      {"nop_1wg", &b.nop, nullptr, 1, 512, 0, 0, 0, 0, 0},
      {"nop_4096wg", &b.nop, nullptr, 4096, 64, 0, 0, 0, 0, 0},
      {"tsper_trivial_1wg", &b.trivial, nullptr, 1, 512, 0, 0, 0, 0, 1},
  };

  const int kReps = 5;
  for (const auto& c : cases) {
    RecCfg rc{};
    rc.pipe = c.pipe;
    rc.pipe_alt = c.pipe_alt;
    rc.grid = c.grid;
    rc.n = c.n;
    rc.barrier = c.barrier;
    rc.barrier_mode = c.barrier_mode;
    rc.rebind = c.rebind;
    rc.pushconst = c.pushconst;
    rc.pipeswitch = c.pipeswitch;
    rc.ts_per_dispatch = c.ts_per;
    for (int rep = 0; rep < kReps; ++rep) {
      RecRes r = run_recorded(b, rc);
      double span_us = (r.t1 > r.t0) ? ticks_to_us(r.t1 - r.t0) : 0.0;
      printf(
          "{\"k\":\"case\",\"name\":\"%s\",\"grid\":%u,\"n\":%u,\"rep\":%d,"
          "\"wall_us\":%.1f,\"record_us\":%.1f,\"span_ticks_us\":%.1f,"
          "\"per_dispatch_wall_us\":%.3f,\"per_dispatch_record_us\":%.3f,"
          "\"per_dispatch_span_us\":%.3f}\n",
          c.name, c.grid, c.n, rep, r.wall_us, r.record_us, span_us,
          r.wall_us / c.n, r.record_us / c.n, span_us / c.n);
      if (c.ts_per) {
        std::vector<uint64_t> sorted = r.per_disp;
        std::sort(sorted.begin(), sorted.end());
        uint64_t med = sorted[sorted.size() / 2];
        printf("{\"k\":\"tsper\",\"name\":\"%s\",\"median_bracket_us\":%.3f,"
               "\"min_bracket_us\":%.3f}\n",
            c.name, ticks_to_us(med),
            ticks_to_us(sorted.front()));
      }
      fflush(stdout);
    }
  }

  double null_us = null_submit_loop(b, 200);
  printf("{\"k\":\"hostcase\",\"name\":\"nullsubmit\",\"iters\":200,"
      "\"mean_us\":%.3f}\n", null_us);
  double sw_us = submit_loop(b, &b.trivial, 1, 200);
  printf("{\"k\":\"hostcase\",\"name\":\"submitwait_trivial_1wg\","
      "\"iters\":200,\"mean_us\":%.3f}\n", sw_us);
  double swg_us = submit_loop(b, &b.trivial, 144, 200);
  printf("{\"k\":\"hostcase\",\"name\":\"submitwait_trivial_144wg\","
      "\"iters\":200,\"mean_us\":%.3f}\n", swg_us);
  fflush(stdout);

  printf("DISPATCH_FLOOR_BENCH_DONE\n");
  return 0;
}
