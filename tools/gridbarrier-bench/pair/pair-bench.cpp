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


// ---- H137b pair bench (everything below is new; boilerplate above is
// copied from tools/dispatch-floor-bench/bench.cpp lines 1-284) ----
#include <cmath>
#include <cstdint>

struct Params {
  uint32_t count, operation, lhs_size, rhs_size, reduce_size, output_size,
      lhs_offset, rhs_offset, output_offset, aux_size, aux_offset, matrix_m,
      matrix_n, matrix_k, flags;
  float alpha, beta;
  uint32_t dims, shape[4], in_strides[4], out_strides[4];
};
static_assert(sizeof(Params) == 120, "push block");

#ifndef KBIND
#define KBIND 26
#endif
static const uint32_t kBind = KBIND;
static const uint32_t kN = 2048, kK = 2048; static uint32_t kL = 16;
static const uint64_t kOff = 8192;  // mid region byte offset inside X
struct MB { Buf b; uint8_t* p; uint64_t size; };
static MB mk(uint64_t size) {
  MB m;
  m.size = size;
  m.b = make_buf(size);
  void* p = nullptr;
  if (g_vk.MapMemory(g_vk.dev, m.b.mem, 0, VK_WHOLE_SIZE, 0, &p) != VK_SUCCESS)
    die("MapMemory");
  m.p = (uint8_t*)p;
  memset(m.p, 0, size);
  return m;
}
static uint64_t g_rng = 0x9e3779b97f4a7c15ull;
static uint32_t rnd() {
  g_rng ^= g_rng << 13; g_rng ^= g_rng >> 7; g_rng ^= g_rng << 17;
  return (uint32_t)(g_rng >> 16);
}
static uint16_t bf16_of(float f) {
  uint32_t b; memcpy(&b, &f, 4);
  return (uint16_t)((b + 0x7fffu + ((b >> 16) & 1u)) >> 16);
}
static float frnd() { return (float)(rnd() & 0xffff) / 65535.0f * 2.0f - 1.0f; }

static VkShaderModule build(const char* src, const char* defs) {
  std::string spv = "/tmp/h137b-" + std::to_string(getpid()) + ".spv";
  std::string cmd = std::string("glslc -fshader-stage=compute --target-env=vulkan1.3 ") +
      defs + " " + src + " -o " + spv + " 2>&1";
  if (system(cmd.c_str()) != 0) die("compile failed: %s", cmd.c_str());
  std::ifstream f(spv, std::ios::binary);
  std::ostringstream ss;
  ss << f.rdbuf();
  unlink(spv.c_str());
  return make_module(ss.str());
}

int main(int argc, char** argv) {
  const char* fused_src = argc > 1 ? argv[1] : "/tmp/pair_fused.comp";
  const char* base_src = argc > 2 ? argv[2] : "tools/q4-bw-bench/shaders/qmm_vec_base.comp";
  if (getenv("PAIR_L")) kL = (uint32_t)atoi(getenv("PAIR_L"));
  vk_init();
  setup_device();
  const char* defs =
      "-DUSE_BF16=1 -DUSE_SUBGROUP=1 -DQMM_VEC_Q4_WORD=1 -DQMM_VEC_MULTI=1 "
      "-DROWS_PER_SLOT=2 -DSLOTS_PER_GROUP=4";
  VkShaderModule mod_f = build(fused_src, defs);
  VkShaderModule mod_r = build(base_src, defs);
  printf("{\"k\":\"stage\",\"s\":\"compiled\"}\n"); fflush(stdout);

  VkDescriptorSetLayoutBinding lb[kBind]{};
  for (uint32_t i = 0; i < kBind; ++i) {
    lb[i].binding = i;
    lb[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    lb[i].descriptorCount = 1;
    lb[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  }
  VkDescriptorSetLayoutCreateInfo dlci{};
  dlci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  dlci.bindingCount = kBind;
  dlci.pBindings = lb;
  VkDescriptorSetLayout dsl;
  if (g_vk.CreateDescriptorSetLayout(g_vk.dev, &dlci, nullptr, &dsl) != VK_SUCCESS)
    die("dsl");
  VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, 128};
  VkPipelineLayoutCreateInfo plci{};
  plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  plci.setLayoutCount = 1;
  plci.pSetLayouts = &dsl;
  plci.pushConstantRangeCount = 1;
  plci.pPushConstantRanges = &pcr;
  VkPipelineLayout layout;
  if (g_vk.CreatePipelineLayout(g_vk.dev, &plci, nullptr, &layout) != VK_SUCCESS)
    die("layout");
  auto mkpipe = [&](VkShaderModule m) {
    VkComputePipelineCreateInfo cpi{};
    cpi.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cpi.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpi.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpi.stage.module = m;
    cpi.stage.pName = "main";
    cpi.layout = layout;
    VkPipeline p;
    if (g_vk.CreateComputePipelines(g_vk.dev, VK_NULL_HANDLE, 1, &cpi, nullptr, &p) != VK_SUCCESS)
      die("pipeline");
    return p;
  };
  VkPipeline pipe_f = mkpipe(mod_f), pipe_r = mkpipe(mod_r);

  // Data.
  MB dummy = mk(1 << 20), sync = mk(4096);
  struct Set { MB w[2], s[2], bi[2], x, midr, outr, outf; };
  std::vector<Set> sets(kL);
  const uint64_t wbytes = (uint64_t)kN * (kK / 8) * 4, sbytes = (uint64_t)kN * (kK / 64) * 2;
  for (auto& st : sets) {
    for (int j = 0; j < 2; ++j) {
      st.w[j] = mk(wbytes); st.s[j] = mk(sbytes); st.bi[j] = mk(sbytes);
      uint32_t* w = (uint32_t*)st.w[j].p;
      for (uint64_t i = 0; i < wbytes / 4; ++i) w[i] = rnd() | (rnd() << 16);
      uint16_t* s = (uint16_t*)st.s[j].p; uint16_t* bb = (uint16_t*)st.bi[j].p;
      for (uint64_t i = 0; i < sbytes / 2; ++i) {
        s[i] = bf16_of(0.004f + 0.004f * fabsf(frnd()));
        bb[i] = bf16_of(0.02f * frnd());
      }
    }
    st.x = mk(kOff + 4096);
    uint16_t* x = (uint16_t*)st.x.p;
    for (uint32_t i = 0; i < kK; ++i) x[i] = bf16_of(frnd());
    st.midr = mk(4096); st.outr = mk(4096); st.outf = mk(4096);
  }

  printf("{\"k\":\"stage\",\"s\":\"init_done\"}\n"); fflush(stdout);
  // Descriptor sets: fused F, reference R1, R2 per weight set.
  VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kBind * 3 * kL};
  VkDescriptorPoolCreateInfo pci{};
  pci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  pci.maxSets = 3 * kL;
  pci.poolSizeCount = 1;
  pci.pPoolSizes = &ps;
  VkDescriptorPool pool;
  if (g_vk.CreateDescriptorPool(g_vk.dev, &pci, nullptr, &pool) != VK_SUCCESS) die("pool");
  struct Bnd { VkBuffer b; uint64_t off; };
  auto make_set = [&](const Bnd* bn) {
    VkDescriptorSetAllocateInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    ai.descriptorPool = pool;
    ai.descriptorSetCount = 1;
    ai.pSetLayouts = &dsl;
    VkDescriptorSet d;
    if (g_vk.AllocateDescriptorSets(g_vk.dev, &ai, &d) != VK_SUCCESS) die("alloc set");
    VkDescriptorBufferInfo bi[kBind];
    VkWriteDescriptorSet wr[kBind]{};
    for (uint32_t i = 0; i < kBind; ++i) {
      bi[i] = {bn[i].b, bn[i].off, VK_WHOLE_SIZE};
      wr[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      wr[i].dstSet = d;
      wr[i].dstBinding = i;
      wr[i].descriptorCount = 1;
      wr[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      wr[i].pBufferInfo = &bi[i];
    }
    g_vk.UpdateDescriptorSets(g_vk.dev, kBind, wr, 0, nullptr);
    return d;
  };
  std::vector<VkDescriptorSet> dF(kL), dR1(kL), dR2(kL);
  for (uint32_t l = 0; l < kL; ++l) {
    Set& st = sets[l];
    Bnd b[kBind];
    for (auto& e : b) e = {dummy.b.buf, 0};
    b[0] = {st.x.b.buf, 0};
    b[1] = {st.w[0].b.buf, 0}; b[2] = {st.s[0].b.buf, 0}; b[3] = {st.bi[0].b.buf, 0};
    b[4] = {st.x.b.buf, kOff};
    b[7] = {st.w[1].b.buf, 0}; b[8] = {st.s[1].b.buf, 0}; b[9] = {st.bi[1].b.buf, 0};
    b[10] = {st.outf.b.buf, 0};
    if (kBind > 25) b[25] = {sync.b.buf, 0};
    dF[l] = make_set(b);
    for (auto& e : b) e = {dummy.b.buf, 0};
    b[0] = {st.x.b.buf, 0};
    b[1] = {st.w[0].b.buf, 0}; b[2] = {st.s[0].b.buf, 0}; b[3] = {st.bi[0].b.buf, 0};
    b[4] = {st.midr.b.buf, 0};
    dR1[l] = make_set(b);
    for (auto& e : b) e = {dummy.b.buf, 0};
    b[0] = {st.midr.b.buf, 0};
    b[1] = {st.w[1].b.buf, 0}; b[2] = {st.s[1].b.buf, 0}; b[3] = {st.bi[1].b.buf, 0};
    b[4] = {st.outr.b.buf, 0};
    dR2[l] = make_set(b);
  }

  VkCommandPoolCreateInfo cpci{};
  cpci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  cpci.queueFamilyIndex = ctx.qfi;
  VkCommandPool cp;
  g_vk.CreateCommandPool(g_vk.dev, &cpci, nullptr, &cp);
  VkCommandBufferAllocateInfo cbai{};
  cbai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  cbai.commandPool = cp;
  cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  cbai.commandBufferCount = 1;
  VkCommandBuffer cmd;
  g_vk.AllocateCommandBuffers(g_vk.dev, &cbai, &cmd);

  const uint32_t tiles = kN / 8;
  Params pf{}, p1{};
  pf.operation = 4; pf.reduce_size = 64; pf.matrix_m = 1; pf.matrix_k = kK;
  pf.dims = 2; pf.shape[0] = kN; pf.shape[1] = kN; pf.aux_offset = (uint32_t)(kOff / 2);
  p1 = pf;
  p1.dims = 1; p1.shape[1] = 0; p1.aux_offset = 0; p1.count = tiles;

  auto full_barrier = [&]() {
    VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mb.srcAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    mb.dstAccessMask = mb.srcAccessMask;
    g_vk.CmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
        VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
  };
  auto run = [&](bool fused, uint32_t G, uint32_t pairs, uint32_t base = 0) {
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    g_vk.BeginCommandBuffer(cmd, &bi);
    for (uint32_t p = 0; p < pairs; ++p) {
      uint32_t l = (base + p) % kL;
      if (fused) {
        Params q = pf; q.count = G;
        g_vk.CmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_f);
        g_vk.CmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &dF[l], 0, nullptr);
        g_vk.CmdPushConstants(cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(q), &q);
        g_vk.CmdDispatch(cmd, G, 1, 1);
        full_barrier();
      } else {
        g_vk.CmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_r);
        g_vk.CmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &dR1[l], 0, nullptr);
        g_vk.CmdPushConstants(cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(p1), &p1);
        g_vk.CmdDispatch(cmd, tiles, 1, 1);
        full_barrier();
        g_vk.CmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &dR2[l], 0, nullptr);
        g_vk.CmdDispatch(cmd, tiles, 1, 1);
        full_barrier();
      }
    }
    g_vk.EndCommandBuffer(cmd);
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;
    double t0 = now_us();
    g_vk.QueueSubmit(g_vk.queue, 1, &si, VK_NULL_HANDLE);
    g_vk.QueueWaitIdle(g_vk.queue);
    return now_us() - t0;
  };

  // Exactness: one pass over every set, fused vs reference.
  volatile uint32_t* sy = (volatile uint32_t*)sync.p;
  std::vector<uint32_t> Gs = {8, 16, 32, 64, 128};
  if (getenv("PAIR_GS")) { Gs.clear(); for (char* t = strtok(strdup(getenv("PAIR_GS")), ","); t; t = strtok(nullptr, ",")) Gs.push_back((uint32_t)atoi(t)); }
  printf("{\"k\":\"stage\",\"s\":\"ref_run\"}\n"); fflush(stdout);
  run(false, 0, kL);
  printf("{\"k\":\"stage\",\"s\":\"ref_done\"}\n"); fflush(stdout);
  if (getenv("PAIR_REFONLY")) return 0;
  std::vector<std::vector<uint8_t>> refmid(kL), refout(kL);
  for (uint32_t l = 0; l < kL; ++l) {
    refmid[l].assign(sets[l].midr.p, sets[l].midr.p + 4096);
    refout[l].assign(sets[l].outr.p, sets[l].outr.p + 4096);
  }
  std::vector<uint32_t> okG;
  for (uint32_t G : Gs) {
    for (auto& st : sets) { memset(st.outf.p, 0, 4096); memset(st.x.p + kOff, 0, 4096); }
    memset((void*)sync.p, 0, 4096);
    // One short submission per set: a spin-bound timeout must stay far
    // below the driver's GPU watchdog.
    for (uint32_t l = 0; l < kL && !sy[2]; ++l) run(true, G, 1, l);
    uint32_t bad_mid = 0, bad_out = 0;
    for (uint32_t l = 0; l < kL; ++l) {
      bad_mid += memcmp(sets[l].x.p + kOff, refmid[l].data(), 4096) != 0;
      bad_out += memcmp(sets[l].outf.p, refout[l].data(), 4096) != 0;
    }
    printf("{\"k\":\"pair_exact\",\"G\":%u,\"sets\":%u,\"bad_mid\":%u,\"bad_out\":%u,"
           "\"timeout\":%u}\n", G, kL, bad_mid, bad_out, sy[2]);
    fflush(stdout);
    if (sy[2]) break;
    if (!bad_mid && !bad_out) okG.push_back(G);
  }
  // Non-degenerate check: the reference output must not be all zero.
  if (okG.empty()) { printf("{\"k\":\"pair_no_passing_G\"}\nPAIR_DONE\n"); return 0; }
  uint32_t nz = 0;
  for (uint32_t i = 0; i < 4096; ++i) nz += refout[0][i] != 0;
  printf("{\"k\":\"pair_ref_nonzero_bytes\",\"n\":%u}\n", nz);

  // Timing, interleaved.
  const uint32_t kPairs = 64;
  for (int round = 0; round < 7; ++round) {
    double tr = run(false, 0, kPairs) / kPairs;
    printf("{\"k\":\"pair_time\",\"arm\":\"ref\",\"round\":%d,\"us_per_pair\":%.2f}\n", round, tr);
    for (uint32_t G : okG) {
      double tf = run(true, G, kPairs) / kPairs;
      printf("{\"k\":\"pair_time\",\"arm\":\"fused_G%u\",\"round\":%d,\"us_per_pair\":%.2f,\"timeout\":%u}\n",
          G, round, tf, sy[2]);
    }
    fflush(stdout);
  }
  printf("PAIR_DONE\n");
  return 0;
}
