// Software grid-barrier bench for Honeykrisp/AGX: occupancy probe + cost
// sweep (H136/H137a method) and a producer-consumer correctness stress
// (extends H137b's bit-exact pair to >=1e7 barrier crossings with random
// data and per-value validation).
//
// Continues: jwm1-parity H136/H137a/H137b (grid barrier 0.72-1.9 us at
// co-resident G; persistent Q4 pair bit-exact but residency-capped).
//
// Modes:
//   probe <local_size>            occupancy ceiling + us/barrier vs G
//   stress <local_size> <G> <rounds> [plain|coherent]
//
// Env: GB_DEV=device-name substring (default "Apple"; "llvmpipe" on the dev
// box), GB_GS=comma G list for probe (default depends on local size).
//
// Safety: bounded spin (2^17 polls, ~40 ms) turns non-residency into a
// timeout flag instead of a GPU hang (H137b incident: 2^22 crossed the
// driver watchdog). Host wrapper must run under the GPU lock + timeout.
//
// Timing: host CLOCK_MONOTONIC wall around submit+wait; probe reports the
// (min wall R=2001 - min wall R=1) / 2000 slope, which removes the submit
// constant (device timestamps are not trusted in absolute terms,
// receipts/2026-09-10-q4-gemv-bandwidth).

#include <vulkan/vulkan.h>
#include <dlfcn.h>
#include <inttypes.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>

#include <string>
#include <vector>

#define LIBVK "libvulkan.so.1"

static double now_us() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec * 1e6 + (double)ts.tv_nsec / 1e3;
}

[[noreturn]] static void die(const char* fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  fprintf(stderr, "gridbarrier-bench: ");
  vfprintf(stderr, fmt, ap);
  fprintf(stderr, "\n");
  va_end(ap);
  exit(1);
}

struct VkTable {
  void* handle{nullptr};
  VkInstance inst{VK_NULL_HANDLE};
  VkDevice dev{VK_NULL_HANDLE};
  VkQueue queue{VK_NULL_HANDLE};
  PFN_vkGetInstanceProcAddr GetInstanceProcAddr{nullptr};
  PFN_vkGetDeviceProcAddr GetDeviceProcAddr{nullptr};
#define VT(fn) PFN_vk##fn fn{nullptr};
  VT(CreateInstance) VT(DestroyInstance) VT(EnumeratePhysicalDevices)
  VT(GetPhysicalDeviceProperties) VT(GetPhysicalDeviceFeatures2)
  VT(GetPhysicalDeviceMemoryProperties)
  VT(GetPhysicalDeviceQueueFamilyProperties) VT(CreateDevice)
  VT(GetDeviceQueue) VT(DestroyDevice)
  VT(CreateBuffer) VT(DestroyBuffer) VT(GetBufferMemoryRequirements)
  VT(BindBufferMemory) VT(AllocateMemory) VT(FreeMemory) VT(MapMemory)
  VT(UnmapMemory)
  VT(CreateShaderModule) VT(DestroyShaderModule)
  VT(CreateComputePipelines) VT(DestroyPipeline)
  VT(CreatePipelineLayout) VT(DestroyPipelineLayout)
  VT(CreateDescriptorSetLayout) VT(DestroyDescriptorSetLayout)
  VT(CreateDescriptorPool) VT(DestroyDescriptorPool)
  VT(AllocateDescriptorSets) VT(UpdateDescriptorSets)
  VT(CreateCommandPool) VT(DestroyCommandPool) VT(AllocateCommandBuffers)
  VT(BeginCommandBuffer) VT(EndCommandBuffer)
  VT(CmdBindPipeline) VT(CmdBindDescriptorSets) VT(CmdDispatch)
  VT(CmdPushConstants) VT(CmdFillBuffer) VT(CmdPipelineBarrier)
  VT(CmdCopyBuffer)
  VT(QueueSubmit) VT(QueueWaitIdle)
#undef VT
};
static VkTable g_vk;

static int vk_init() {
  g_vk.handle = dlopen(LIBVK, RTLD_NOW | RTLD_LOCAL);
  if (!g_vk.handle) die("dlopen %s: %s", LIBVK, dlerror());
  g_vk.GetInstanceProcAddr =
      (PFN_vkGetInstanceProcAddr)dlsym(g_vk.handle, "vkGetInstanceProcAddr");
  g_vk.GetDeviceProcAddr =
      (PFN_vkGetDeviceProcAddr)dlsym(g_vk.handle, "vkGetDeviceProcAddr");
  if (!g_vk.GetInstanceProcAddr || !g_vk.GetDeviceProcAddr)
    die("vk proc symbols missing");
  g_vk.CreateInstance =
      (PFN_vkCreateInstance)g_vk.GetInstanceProcAddr(nullptr, "vkCreateInstance");
  if (!g_vk.CreateInstance) die("vkCreateInstance missing");
  return 0;
}

struct DeviceCtx {
  uint32_t qfi{0};
  float ts_ns{1.0f};
  std::string name;
  VkPhysicalDeviceMemoryProperties mp{};
  VkPhysicalDeviceVulkan12Features f12{};
};
static DeviceCtx ctx;

static void setup_device() {
  VkApplicationInfo ai{};
  ai.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  ai.pApplicationName = "gridbarrier-bench";
  ai.apiVersion = VK_API_VERSION_1_2;
  VkInstanceCreateInfo ici{};
  ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  ici.pApplicationInfo = &ai;
  if (g_vk.CreateInstance(&ici, nullptr, &g_vk.inst) != VK_SUCCESS)
    die("CreateInstance");
#define LOAD(name) \
  g_vk.name = (PFN_vk##name)g_vk.GetInstanceProcAddr(g_vk.inst, "vk" #name)
  LOAD(DestroyInstance);
  LOAD(EnumeratePhysicalDevices);
  LOAD(GetPhysicalDeviceProperties);
  LOAD(GetPhysicalDeviceFeatures2);
  LOAD(GetPhysicalDeviceMemoryProperties);
  LOAD(GetPhysicalDeviceQueueFamilyProperties);
  LOAD(CreateDevice);
#undef LOAD
  const char* want = getenv("GB_DEV");
  if (!want || !want[0]) want = "Apple";
  uint32_t n = 0;
  g_vk.EnumeratePhysicalDevices(g_vk.inst, &n, nullptr);
  if (n == 0) die("no physical devices");
  std::vector<VkPhysicalDevice> pds(n);
  g_vk.EnumeratePhysicalDevices(g_vk.inst, &n, pds.data());
  for (uint32_t i = 0; i < n; ++i) {
    VkPhysicalDeviceProperties props{};
    g_vk.GetPhysicalDeviceProperties(pds[i], &props);
    printf("{\"k\":\"device\",\"i\":%u,\"name\":\"%s\",\"api\":\"%u.%u\"}\n",
        i, props.deviceName, VK_API_VERSION_MAJOR(props.apiVersion),
        VK_API_VERSION_MINOR(props.apiVersion));
    if (props.apiVersion < VK_API_VERSION_1_2) continue;
    if (!strstr(props.deviceName, want)) continue;
    uint32_t qfn = 0;
    g_vk.GetPhysicalDeviceQueueFamilyProperties(pds[i], &qfn, nullptr);
    std::vector<VkQueueFamilyProperties> qfpv(qfn);
    g_vk.GetPhysicalDeviceQueueFamilyProperties(pds[i], &qfn, qfpv.data());
    for (uint32_t q = 0; q < qfn; ++q) {
      if ((qfpv[q].queueFlags & VK_QUEUE_COMPUTE_BIT) == 0) continue;
      ctx.qfi = q;
      ctx.ts_ns = props.limits.timestampPeriod;
      ctx.name = props.deviceName;
      g_vk.GetPhysicalDeviceMemoryProperties(pds[i], &ctx.mp);
      ctx.f12.sType =
          VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
      g_vk.GetPhysicalDeviceFeatures2(pds[i],
          (VkPhysicalDeviceFeatures2*)&ctx.f12);
      float prio = 1.0f;
      VkDeviceQueueCreateInfo qci{};
      qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
      qci.queueFamilyIndex = q;
      qci.queueCount = 1;
      qci.pQueuePriorities = &prio;
      static const bool nofeat = getenv("GB_NOFEAT") != nullptr;
      VkPhysicalDeviceVulkan12Features en{
          VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
      if (nofeat) { en.sType = (VkStructureType)0; }
      // 32-bit storage-buffer atomics are core since 1.1 (no feature bit);
      // the memory model is opt-in, so enable it when the driver has it.
      if (ctx.f12.vulkanMemoryModel) en.vulkanMemoryModel = VK_TRUE;
      if (ctx.f12.vulkanMemoryModelDeviceScope)
        en.vulkanMemoryModelDeviceScope = VK_TRUE;
      if (ctx.f12.shaderBufferInt64Atomics)
        en.shaderBufferInt64Atomics = VK_TRUE;
      VkDeviceCreateInfo dci{};
      dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
      dci.pNext = &en;
      dci.queueCreateInfoCount = 1;
      dci.pQueueCreateInfos = &qci;
      if (g_vk.CreateDevice(pds[i], &dci, nullptr, &g_vk.dev) != VK_SUCCESS)
        die("CreateDevice");
#define LOAD_D(fn) \
  g_vk.fn = (PFN_vk##fn)g_vk.GetDeviceProcAddr(g_vk.dev, "vk" #fn)
      LOAD_D(GetDeviceQueue); LOAD_D(DestroyDevice);
      LOAD_D(CreateBuffer); LOAD_D(DestroyBuffer);
      LOAD_D(GetBufferMemoryRequirements); LOAD_D(BindBufferMemory);
      LOAD_D(AllocateMemory); LOAD_D(FreeMemory); LOAD_D(MapMemory);
      LOAD_D(UnmapMemory);
      LOAD_D(CreateShaderModule); LOAD_D(DestroyShaderModule);
      LOAD_D(CreateComputePipelines); LOAD_D(DestroyPipeline);
      LOAD_D(CreatePipelineLayout); LOAD_D(DestroyPipelineLayout);
      LOAD_D(CreateDescriptorSetLayout); LOAD_D(DestroyDescriptorSetLayout);
      LOAD_D(CreateDescriptorPool); LOAD_D(DestroyDescriptorPool);
      LOAD_D(AllocateDescriptorSets); LOAD_D(UpdateDescriptorSets);
      LOAD_D(CreateCommandPool); LOAD_D(DestroyCommandPool);
      LOAD_D(AllocateCommandBuffers);
      LOAD_D(BeginCommandBuffer); LOAD_D(EndCommandBuffer);
      LOAD_D(CmdBindPipeline); LOAD_D(CmdBindDescriptorSets);
      LOAD_D(CmdDispatch); LOAD_D(CmdPushConstants); LOAD_D(CmdFillBuffer);
      LOAD_D(CmdPipelineBarrier); LOAD_D(CmdCopyBuffer);
      LOAD_D(QueueSubmit); LOAD_D(QueueWaitIdle);
      g_vk.GetDeviceQueue(g_vk.dev, q, 0, &g_vk.queue);
#undef LOAD_D
      printf(
          "{\"k\":\"meta\",\"dev\":\"%s\",\"ts_period_ns\":%.3f,"
          "\"feat_vulkanMemoryModel\":%d,"
          "\"feat_vulkanMemoryModelDeviceScope\":%d,"
          "\"feat_shaderBufferInt64Atomics\":%d,"
          "\"maxComputeWorkGroupCount\":[%u,%u,%u],"
          "\"maxComputeWorkGroupInvocations\":%u}\n",
          ctx.name.c_str(), ctx.ts_ns,
          (int)ctx.f12.vulkanMemoryModel,
          (int)ctx.f12.vulkanMemoryModelDeviceScope,
          (int)ctx.f12.shaderBufferInt64Atomics,
          props.limits.maxComputeWorkGroupCount[0],
          props.limits.maxComputeWorkGroupCount[1],
          props.limits.maxComputeWorkGroupCount[2],
          props.limits.maxComputeWorkGroupInvocations);
      fflush(stdout);
      return;
    }
  }
  die("no Vulkan 1.2 compute device matching '%s'", want);
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
  bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
             VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  if (g_vk.CreateBuffer(g_vk.dev, &bi, nullptr, &b.buf) != VK_SUCCESS)
    die("CreateBuffer");
  VkMemoryRequirements req;
  g_vk.GetBufferMemoryRequirements(g_vk.dev, b.buf, &req);
  uint32_t mt = find_memtype(req.memoryTypeBits,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  if (mt == UINT32_MAX)
    mt = find_memtype(req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
  if (mt == UINT32_MAX) die("no host-visible memtype");
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

struct Pipe {
  VkShaderModule mod{VK_NULL_HANDLE};
  VkPipeline pipe{VK_NULL_HANDLE};
};

struct Bench {
  VkDescriptorSetLayout dsl{VK_NULL_HANDLE};
  VkPipelineLayout layout{VK_NULL_HANDLE};
  VkDescriptorPool pool{VK_NULL_HANDLE};
  VkDescriptorSet set{VK_NULL_HANDLE};
  Buf out;
  void* map{nullptr};
  VkDeviceSize out_size{0};
  VkCommandPool cpool{VK_NULL_HANDLE};
  VkCommandBuffer cmd{VK_NULL_HANDLE};
};

static std::string compile_shader(const std::string& src) {
  char path_src[] = "/tmp/gbb-src-XXXXXX.comp";
  char path_spv[] = "/tmp/gbb-out-XXXXXX.spv";
  int fd = mkstemps(path_src, 5);
  if (fd < 0) die("mkstemps src");
  if (write(fd, src.data(), src.size()) != (ssize_t)src.size())
    die("write src");
  close(fd);
  fd = mkstemps(path_spv, 4);
  if (fd < 0) die("mkstemps spv");
  close(fd);
  std::string cmd = std::string("glslc -fshader-stage=compute "
      "--target-env=vulkan1.3 ") + path_src + " -o " + path_spv + " 2>&1";
  if (system(cmd.c_str()) != 0) {
    // Dev box: /usr/local/bin/glslc is not the Mac glslc; fall back to
    // glslangValidator. Hardware runs use glslc (same as the Macs).
    cmd = std::string("glslangValidator -V --target-env vulkan1.3 ")
        + path_src + " -o " + path_spv + " 2>&1";
    if (system(cmd.c_str()) != 0)
      die("glslc and glslangValidator failed for shader:\n%s", src.c_str());
  }
  FILE* f = fopen(path_spv, "rb");
  if (!f) die("open spv");
  std::string spv;
  char tmp[65536];
  size_t r;
  while ((r = fread(tmp, 1, sizeof(tmp), f)) > 0) spv.append(tmp, r);
  fclose(f);
  unlink(path_src);
  unlink(path_spv);
  return spv;
}

static void bench_setup(Bench& b, VkDeviceSize out_bytes) {
  VkDescriptorSetLayoutBinding lb{};
  lb.binding = 0;
  lb.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  lb.descriptorCount = 1;
  lb.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  VkDescriptorSetLayoutCreateInfo dli{};
  dli.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  dli.bindingCount = 1;
  dli.pBindings = &lb;
  if (g_vk.CreateDescriptorSetLayout(g_vk.dev, &dli, nullptr, &b.dsl) !=
      VK_SUCCESS)
    die("CreateDescriptorSetLayout");
  VkPushConstantRange pr{};
  pr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  pr.size = 4;
  VkPipelineLayoutCreateInfo pli{};
  pli.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  pli.setLayoutCount = 1;
  pli.pSetLayouts = &b.dsl;
  pli.pushConstantRangeCount = 1;
  pli.pPushConstantRanges = &pr;
  if (g_vk.CreatePipelineLayout(g_vk.dev, &pli, nullptr, &b.layout) !=
      VK_SUCCESS)
    die("CreatePipelineLayout");
  VkDescriptorPoolSize ps{};
  ps.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  ps.descriptorCount = 2;
  VkDescriptorPoolCreateInfo pi{};
  pi.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  pi.maxSets = 2;
  pi.poolSizeCount = 1;
  pi.pPoolSizes = &ps;
  if (g_vk.CreateDescriptorPool(g_vk.dev, &pi, nullptr, &b.pool) != VK_SUCCESS)
    die("CreateDescriptorPool");
  VkDescriptorSetAllocateInfo ai{};
  ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  ai.descriptorPool = b.pool;
  ai.descriptorSetCount = 1;
  ai.pSetLayouts = &b.dsl;
  if (g_vk.AllocateDescriptorSets(g_vk.dev, &ai, &b.set) != VK_SUCCESS)
    die("AllocateDescriptorSets");
  b.out = make_buf(out_bytes);
  b.out_size = out_bytes;
  VkDescriptorBufferInfo bi{};
  bi.buffer = b.out.buf;
  bi.offset = 0;
  bi.range = out_bytes;
  VkWriteDescriptorSet w{};
  w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  w.dstSet = b.set;
  w.dstBinding = 0;
  w.descriptorCount = 1;
  w.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  w.pBufferInfo = &bi;
  g_vk.UpdateDescriptorSets(g_vk.dev, 1, &w, 0, nullptr);
  if (g_vk.MapMemory(g_vk.dev, b.out.mem, 0, VK_WHOLE_SIZE, 0, &b.map) !=
      VK_SUCCESS)
    die("MapMemory");
  VkCommandPoolCreateInfo cpi{};
  cpi.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  cpi.queueFamilyIndex = ctx.qfi;
  if (g_vk.CreateCommandPool(g_vk.dev, &cpi, nullptr, &b.cpool) != VK_SUCCESS)
    die("CreateCommandPool");
  VkCommandBufferAllocateInfo cbi{};
  cbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  cbi.commandPool = b.cpool;
  cbi.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  cbi.commandBufferCount = 1;
  if (g_vk.AllocateCommandBuffers(g_vk.dev, &cbi, &b.cmd) != VK_SUCCESS)
    die("AllocateCommandBuffers");
}

static Pipe make_pipe(Bench& b, const std::string& src) {
  Pipe p;
  p.mod = make_module(compile_shader(src));
  VkComputePipelineCreateInfo cpi{};
  cpi.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
  cpi.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  cpi.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  cpi.stage.module = p.mod;
  cpi.stage.pName = "main";
  cpi.layout = b.layout;
  if (g_vk.CreateComputePipelines(g_vk.dev, VK_NULL_HANDLE, 1, &cpi, nullptr,
      &p.pipe) != VK_SUCCESS)
    die("CreateComputePipelines");
  return p;
}

// Submit: fill whole out buffer to zero, barrier, dispatch(grid), wait.
static double run_pass(Bench& b, VkPipeline pipe, uint32_t grid,
    uint32_t push_val) {
  double t0 = now_us();
  const bool dbg = getenv("GB_TRACE") != nullptr;
  static const bool empty_cb = getenv("GB_EMPTY") != nullptr;
  if (empty_cb) { dbg ? (void)0 : (void)0; }
  VkCommandBufferBeginInfo bi{};
  bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  if (dbg) fprintf(stderr, "trace: begin\n");
  if (empty_cb) {
    g_vk.BeginCommandBuffer(b.cmd, &bi);
    g_vk.EndCommandBuffer(b.cmd);
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &b.cmd;
    double t0 = now_us();
    g_vk.QueueSubmit(g_vk.queue, 1, &si, VK_NULL_HANDLE);
    g_vk.QueueWaitIdle(g_vk.queue);
    printf("{\"k\":\"empty_cb_us\":%.1f}\n", now_us() - t0);
    fflush(stdout);
    return 0.0;
  }
  g_vk.BeginCommandBuffer(b.cmd, &bi);
  if (dbg) fprintf(stderr, "trace: fill\n");
  static const bool nofill = getenv("GB_NOFILL") != nullptr;
  if (!nofill) g_vk.CmdFillBuffer(b.cmd, b.out.buf, 0, b.out_size, 0);
  VkMemoryBarrier mb{};
  mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
  mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
  static const bool nobar = getenv("GB_NOBAR") != nullptr;
  if (dbg) fprintf(stderr, "trace: pbar\n");
  if (!nofill && !nobar) g_vk.CmdPipelineBarrier(b.cmd,
      VK_PIPELINE_STAGE_TRANSFER_BIT,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, nullptr, 0,
      nullptr);
  if (dbg) fprintf(stderr, "trace: bindpipe\n");
  g_vk.CmdBindPipeline(b.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
  g_vk.CmdBindDescriptorSets(b.cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
      b.layout, 0, 1, &b.set, 0, nullptr);
  if (dbg) fprintf(stderr, "trace: pushconst\n");
  g_vk.CmdPushConstants(b.cmd, b.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 4,
      &push_val);
  if (dbg) fprintf(stderr, "trace: dispatch G=%u\n", grid);
  static const bool nodisp = getenv("GB_NODISP") != nullptr;
  if (!nodisp) g_vk.CmdDispatch(b.cmd, grid, 1, 1);
  if (dbg) fprintf(stderr, "trace: end\n");
  g_vk.EndCommandBuffer(b.cmd);
  VkSubmitInfo si{};
  si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  si.commandBufferCount = 1;
  si.pCommandBuffers = &b.cmd;
  if (dbg) fprintf(stderr, "trace: submit fn=%p\n", (void*)g_vk.QueueSubmit);
  if (g_vk.QueueSubmit(g_vk.queue, 1, &si, VK_NULL_HANDLE) != VK_SUCCESS)
    die("QueueSubmit");
  if (dbg) fprintf(stderr, "trace: waitidle\n");
  if (g_vk.QueueWaitIdle(g_vk.queue) != VK_SUCCESS) die("QueueWaitIdle");
  return now_us() - t0;
}

static std::vector<uint32_t> parse_gs(const char* env,
    const std::vector<uint32_t>& dflt) {
  const char* s = getenv(env);
  if (!s || !s[0]) return dflt;
  std::vector<uint32_t> out;
  char* dup = strdup(s);
  for (char* t = strtok(dup, ","); t; t = strtok(nullptr, ","))
    out.push_back((uint32_t)atoi(t));
  free(dup);
  if (out.empty()) return dflt;
  return out;
}

// ---------------------------------------------------------------- probe
// H136/H137a: G workgroups, R grid barriers per dispatch, lead thread of
// each workgroup arrives (atomic counter + generation), bounded spin.
// Ordering check: every 64th round, WG 0 verifies each workgroup's round
// marker caught up (counter[3] counts violations).
static const char* kProbeHead =
    "#version 450\n"
    "layout(local_size_x = ";
static const char* kProbeMid =
    ", local_size_y = 1, local_size_z = 1) in;\n"
    "layout(set = 0, binding = 0, std430) buffer Out { uint out_buf[]; };\n"
    "layout(push_constant) uniform PC { uint pc_val; } pc;\n";
static const char* kProbeBody = R"GLSL(
void main() {
  uint G = gl_NumWorkGroups.x;
  uint w = gl_WorkGroupID.x;
  bool lead = gl_LocalInvocationIndex == 0u;
  for (uint r = 0u; r < pc.pc_val; ++r) {
    if (lead) {
      atomicExchange(out_buf[16u + w], r + 1u);
      uint gen = atomicAdd(out_buf[1], 0u);
      uint old = atomicAdd(out_buf[0], 1u);
      if (old == G - 1u) {
        atomicExchange(out_buf[0], 0u);
        atomicAdd(out_buf[1], 1u);
      } else {
        uint spins = 0u;
        while (atomicAdd(out_buf[1], 0u) == gen) {
          if (++spins > 32768u) { atomicOr(out_buf[2], 1u); break; }
        }
      }
      if ((r & 63u) == 63u && w == 0u) {
        for (uint j = 0u; j < G; ++j)
          if (atomicAdd(out_buf[16u + j], 0u) < r + 1u)
            atomicAdd(out_buf[3], 1u);
      }
    }
    barrier();
  }
}
)GLSL";

static int probe_mode(Bench& b, uint32_t local) {
  Pipe reset, bar;
  reset = make_pipe(b, std::string(kProbeHead) + "1" + kProbeMid +
      "void main() { if (gl_LocalInvocationIndex == 0u && "
      "gl_WorkGroupID.x == 0u) { for (uint i = 0u; i < 128u; ++i) "
      "out_buf[i] = 0u; } }\n");
  bar = make_pipe(b, std::string(kProbeHead) + std::to_string(local) +
      kProbeMid + kProbeBody);
  std::vector<uint32_t> dflt;
  if (local <= 32) {
    for (uint32_t g : {1u, 2u, 4u, 8u, 16u, 32u, 64u, 96u, 128u, 192u, 256u})
      dflt.push_back(g);
  } else {
    for (uint32_t g : {8u, 16u, 32u, 48u, 64u, 96u, 128u, 160u, 192u, 224u,
         256u, 288u, 320u, 384u, 448u, 512u})
      dflt.push_back(g);
  }
  std::vector<uint32_t> Gs = parse_gs("GB_GS", dflt);
  const uint32_t Rs[2] = {1, 2001};
  for (uint32_t G : Gs) {
    double best[2] = {1e18, 1e18};
    uint32_t timeouts = 0, violations = 0;
    for (int ri = 0; ri < 2; ++ri) {
      for (int rep = 0; rep < 5; ++rep) {
        run_pass(b, reset.pipe, 1, 1);
        double wall = run_pass(b, bar.pipe, G, Rs[ri]);
        volatile uint32_t* h = (volatile uint32_t*)b.map;
        if (h[2]) ++timeouts;
        violations += h[3];
        if (!h[2] && wall < best[ri]) best[ri] = wall;
        printf("{\"k\":\"probe_rep\",\"local\":%u,\"G\":%u,\"R\":%u,"
            "\"rep\":%d,\"wall_us\":%.1f,\"timeout\":%u,\"violations\":%u}\n",
            local, G, Rs[ri], rep, wall, h[2], h[3]);
        fflush(stdout);
        if (h[2]) break;
      }
    }
    double slope = (best[0] < 1e17 && best[1] < 1e17)
        ? (best[1] - best[0]) / 2000.0 : -1.0;
    printf("{\"k\":\"probe\",\"local\":%u,\"G\":%u,\"slope_us_per_barrier\":"
        "%.3f,\"min_wall_R1\":%.1f,\"min_wall_R2001\":%.1f,"
        "\"timeouts\":%u,\"violations\":%u}\n",
        local, G, slope, best[0], best[1], timeouts, violations);
    fflush(stdout);
    if (timeouts) break;  // above residency; escalating further only spins
  }
  printf("PROBE_DONE\n");
  return 0;
}

// ---------------------------------------------------------------- stress
// Producer-consumer across the grid barrier with random data:
//   round r, WG w writes slot[w][r&1][lane] = hash(seed(r, w, lane))
//   grid barrier
//   WG w validates every other WG's slot[r&1][lane] against the same hash
//   any mismatch -> violations counter + first-bad record
// Parity double-buffering: round r+1 writes touch the other slot, so a
// straggler's round-r validation can never race a producer's next write.
// Buffer: [0] arrive, [1] gen, [2] timeout, [3] violations, [4] done_wgs,
// [5] bad_round+1, [6] bad_producer, [7] bad_consumer, [8] bad_got,
// [9] bad_want; slots at 1024 + (w*2 + (r&1))*128 + lane.
static const char* kStressBody = R"GLSL(
uint gbb_hash(uint x) {
  x ^= x >> 16; x *= 0x7feb352du;
  x ^= x >> 15; x *= 0x846ca68bu;
  x ^= x >> 16; return x;
}
void gbb_barrier(uint G) {
  memoryBarrierBuffer();
  barrier();
  if (gl_LocalInvocationIndex == 0u) {
    uint gen = atomicAdd(out_buf[1], 0u);
    uint old = atomicAdd(out_buf[0], 1u);
    if (old == G - 1u) {
      atomicExchange(out_buf[0], 0u);
      atomicAdd(out_buf[1], 1u);
    } else {
      uint spins = 0u;
      while (atomicAdd(out_buf[1], 0u) == gen) {
        if (++spins > 32768u) { atomicOr(out_buf[2], 1u); break; }
      }
    }
  }
  barrier();
  memoryBarrierBuffer();
}
void main() {
  uint G = gl_NumWorkGroups.x;
  uint w = gl_WorkGroupID.x;
  uint lane = gl_LocalInvocationIndex;
  bool lead = lane == 0u;
  uint timed_out = 0u;
  for (uint r = 0u; r < pc.pc_val && timed_out == 0u; ++r) {
    uint par = r & 1u;
    out_buf[1024u + (w * 2u + par) * 128u + lane] =
        gbb_hash(((r * G + w) * 128u + lane) ^ 0x9e3779b9u);
    gbb_barrier(G);
    if (atomicAdd(out_buf[2], 0u) != 0u) { timed_out = 1u; break; }
    for (uint v = 0u; v < G; ++v) {
      if (v == w) continue;
      uint got = out_buf[1024u + (v * 2u + par) * 128u + lane];
      uint want = gbb_hash(((r * G + v) * 128u + lane) ^ 0x9e3779b9u);
      if (got != want) {
        atomicAdd(out_buf[3], 1u);
        if (lead) {
          atomicExchange(out_buf[5], r + 1u);
          atomicExchange(out_buf[6], v);
          atomicExchange(out_buf[7], w);
          atomicExchange(out_buf[8], got);
          atomicExchange(out_buf[9], want);
        }
      }
    }
  }
  if (lead && timed_out == 0u) atomicAdd(out_buf[4], 1u);
}
)GLSL";

static int stress_mode(Bench& b, uint32_t local, uint32_t G, uint32_t rounds,
    bool coherent) {
  std::string qual = coherent ? "coherent " : "";
  std::string src = std::string(kProbeHead) + std::to_string(local) +
      kProbeMid;
  src.replace(src.find("buffer Out"), 10, qual + "buffer Out");
  src += kStressBody;
  Pipe st = make_pipe(b, src);
  if (G > 512) die("G>512 unsupported (buffer layout)");
  run_pass(b, st.pipe, G, 1);
  double wall = run_pass(b, st.pipe, G, rounds);
  volatile uint32_t* h = (volatile uint32_t*)b.map;
  printf("{\"k\":\"stress\",\"local\":%u,\"G\":%u,\"rounds\":%u,"
      "\"crossings\":%llu,\"qual\":\"%s\",\"wall_us\":%.1f,"
      "\"us_per_crossing\":%.4f,\"violations\":%u,\"timeout\":%u,"
      "\"done_wgs\":%u,\"bad_round\":%u,\"bad_producer\":%u,"
      "\"bad_consumer\":%u,\"bad_got\":%u,\"bad_want\":%u}\n",
      local, G, rounds, (unsigned long long)G * rounds,
      coherent ? "coherent" : "plain", wall, wall / (double)rounds,
      h[3], h[2], h[4], h[5], h[6], h[7], h[8], h[9]);
  fflush(stdout);
  printf("STRESS_DONE\n");
  return 0;
}

int main(int argc, char** argv) {
  if (argc < 2) {
    fprintf(stderr,
        "usage: %s probe <local> | stress <local> <G> <rounds> "
        "[plain|coherent]\n",
        argv[0]);
    return 2;
  }
  vk_init();
  setup_device();
  Bench b;
  bench_setup(b, 1024ull * 1024ull);  // 1 MiB: ctrl + 512 WG parity slots
  if (strcmp(argv[1], "probe") == 0) {
    uint32_t local = argc > 2 ? (uint32_t)atoi(argv[2]) : 32;
    return probe_mode(b, local);
  }
  if (strcmp(argv[1], "stress") == 0) {
    uint32_t local = argc > 2 ? (uint32_t)atoi(argv[2]) : 128;
    uint32_t G = argc > 3 ? (uint32_t)atoi(argv[3]) : 64;
    uint32_t rounds = argc > 4 ? (uint32_t)atoi(argv[4]) : 100000;
    bool coherent = argc > 5 ? strcmp(argv[5], "coherent") == 0 : false;
    return stress_mode(b, local, G, rounds, coherent);
  }
  die("unknown mode %s", argv[1]);
}
