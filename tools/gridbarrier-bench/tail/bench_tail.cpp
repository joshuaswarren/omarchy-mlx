// Op-level fused MLP-tail test (H290): the 2B decode tail
//   rmsnorm(2048) -> qmm_vec multi {gate,up} + swiglu fold (6144) ->
//   qmm_vec multi {down} + add epilogue (2048) + residual
// as THREE production dispatches (unmodified shaders, shipped geometry)
// versus ONE persistent dispatch of the generated fused_tail.comp
// (gen_tail.py) with 2 internal software grid barriers.
//
// Bit-exact contract: identical per-row chains, quad order, scale/bias,
// swiglu and add rounding. Compares norm output, swiglu output, and final
// sum separately so a divergence names its stage.
//
// Env: GB_DEV=device substring (default Apple), GB_G=fused grid (default 4;
// lavapipe must stay tiny), GB_SETS=exactness data refills (default 8),
// GB_ROUNDS=timing rounds per submit (default 256).
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

#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
static PFN_vkGetPhysicalDeviceFeatures2 g_get_features2{nullptr};

#define LIBVK "libvulkan.so.1"

static double now_us() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec * 1e6 + (double)ts.tv_nsec / 1e3;
}
[[noreturn]] static void die(const char* fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  fprintf(stderr, "bench-tail: ");
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
  VT(GetPhysicalDeviceProperties) VT(GetPhysicalDeviceMemoryProperties)
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
  VT(QueueSubmit) VT(QueueWaitIdle)
#undef VT
};
static VkTable g_vk;

static void vk_init() {
  g_vk.handle = dlopen(LIBVK, RTLD_NOW | RTLD_LOCAL);
  if (!g_vk.handle) die("dlopen: %s", dlerror());
  g_vk.GetInstanceProcAddr =
      (PFN_vkGetInstanceProcAddr)dlsym(g_vk.handle, "vkGetInstanceProcAddr");
  g_vk.GetDeviceProcAddr =
      (PFN_vkGetDeviceProcAddr)dlsym(g_vk.handle, "vkGetDeviceProcAddr");
  g_vk.CreateInstance = (PFN_vkCreateInstance)g_vk.GetInstanceProcAddr(
      nullptr, "vkCreateInstance");
  if (!g_vk.CreateInstance) die("vkCreateInstance");
}

struct DeviceCtx {
  uint32_t qfi{0};
  std::string name;
  VkPhysicalDeviceMemoryProperties mp{};
};
static DeviceCtx ctx;

static void setup_device() {
  VkApplicationInfo ai{};
  ai.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  ai.pApplicationName = "bench-tail";
  ai.apiVersion = VK_API_VERSION_1_2;
  VkInstanceCreateInfo ici{};
  ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  ici.pApplicationInfo = &ai;
  if (g_vk.CreateInstance(&ici, nullptr, &g_vk.inst) != VK_SUCCESS)
    die("CreateInstance");
#define LOAD(name) \
  g_vk.name = (PFN_vk##name)g_vk.GetInstanceProcAddr(g_vk.inst, "vk" #name)
  LOAD(DestroyInstance); LOAD(EnumeratePhysicalDevices);
  LOAD(GetPhysicalDeviceProperties); LOAD(GetPhysicalDeviceMemoryProperties);
  LOAD(GetPhysicalDeviceQueueFamilyProperties); LOAD(CreateDevice);
#undef LOAD
  const char* want = getenv("GB_DEV");
  if (!want || !want[0]) want = "Apple";
  uint32_t n = 0;
  g_vk.EnumeratePhysicalDevices(g_vk.inst, &n, nullptr);
  std::vector<VkPhysicalDevice> pds(n ? n : 0);
  if (n) g_vk.EnumeratePhysicalDevices(g_vk.inst, &n, pds.data());
  for (uint32_t i = 0; i < n; ++i) {
    VkPhysicalDeviceProperties props{};
    g_vk.GetPhysicalDeviceProperties(pds[i], &props);
    if (props.apiVersion < VK_API_VERSION_1_2) continue;
    if (!strstr(props.deviceName, want)) continue;
    uint32_t qfn = 0;
    g_vk.GetPhysicalDeviceQueueFamilyProperties(pds[i], &qfn, nullptr);
    std::vector<VkQueueFamilyProperties> qfpv(qfn);
    g_vk.GetPhysicalDeviceQueueFamilyProperties(pds[i], &qfn, qfpv.data());
    for (uint32_t q = 0; q < qfn; ++q) {
      if ((qfpv[q].queueFlags & VK_QUEUE_COMPUTE_BIT) == 0) continue;
      ctx.qfi = q;
      ctx.name = props.deviceName;
      g_vk.GetPhysicalDeviceMemoryProperties(pds[i], &ctx.mp);
      float prio = 1.0f;
      VkDeviceQueueCreateInfo qci{};
      qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
      qci.queueFamilyIndex = q;
      qci.queueCount = 1;
      qci.pQueuePriorities = &prio;
      VkPhysicalDevice16BitStorageFeatures f16{
          VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES};
      VkPhysicalDeviceSubgroupSizeControlFeatures fssc{
          VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES};
      VkPhysicalDeviceFeatures2 f2{
          VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
      f2.pNext = &f16;
      f16.pNext = &fssc;
      g_get_features2 =
          (PFN_vkGetPhysicalDeviceFeatures2)g_vk.GetInstanceProcAddr(
              g_vk.inst, "vkGetPhysicalDeviceFeatures2");
      if (g_get_features2) g_get_features2(pds[i], &f2);
      if (f16.storageBuffer16BitAccess) f16.storageBuffer16BitAccess = VK_TRUE;
      if (fssc.subgroupSizeControl) fssc.subgroupSizeControl = VK_TRUE;
      VkDeviceCreateInfo dci{};
      dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
      dci.pNext = &f16;
      dci.queueCreateInfoCount = 1;
      dci.pQueueCreateInfos = &qci;
      if (g_vk.CreateDevice(pds[i], &dci, nullptr, &g_vk.dev) != VK_SUCCESS)
        die("CreateDevice");
#define LOAD_D(fn) \
  g_vk.fn = (PFN_vk##fn)g_vk.GetDeviceProcAddr(g_vk.dev, "vk" #fn)
      LOAD_D(GetDeviceQueue); LOAD_D(DestroyDevice); LOAD_D(CreateBuffer); LOAD_D(DestroyBuffer);
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
      LOAD_D(CmdPipelineBarrier); LOAD_D(QueueSubmit); LOAD_D(QueueWaitIdle);
      g_vk.GetDeviceQueue(g_vk.dev, q, 0, &g_vk.queue);
#undef LOAD_D
      printf("{\"k\":\"meta\",\"dev\":\"%s\"}\n",
          ctx.name.c_str());
      fflush(stdout);
      return;
    }
  }
  die("no device matching '%s'", want);
}

struct MB {
  VkBuffer buf{VK_NULL_HANDLE};
  VkDeviceMemory mem{VK_NULL_HANDLE};
  uint8_t* p{nullptr};
};
static uint32_t find_memtype(uint32_t bits, VkMemoryPropertyFlags want) {
  for (uint32_t i = 0; i < ctx.mp.memoryTypeCount; ++i) {
    if ((bits & (1u << i)) &&
        (ctx.mp.memoryTypes[i].propertyFlags & want) == want)
      return i;
  }
  return UINT32_MAX;
}
static MB mk(VkDeviceSize size) {
  MB m;
  VkBufferCreateInfo bi{};
  bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  bi.size = size;
  bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
             VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  if (g_vk.CreateBuffer(g_vk.dev, &bi, nullptr, &m.buf) != VK_SUCCESS)
    die("CreateBuffer");
  VkMemoryRequirements req;
  g_vk.GetBufferMemoryRequirements(g_vk.dev, m.buf, &req);
  uint32_t mt = find_memtype(req.memoryTypeBits,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
      VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  if (mt == UINT32_MAX) die("no host-visible memtype");
  VkMemoryAllocateInfo mai{};
  mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  mai.allocationSize = req.size;
  mai.memoryTypeIndex = mt;
  if (g_vk.AllocateMemory(g_vk.dev, &mai, nullptr, &m.mem) != VK_SUCCESS)
    die("AllocateMemory");
  if (g_vk.BindBufferMemory(g_vk.dev, m.buf, m.mem, 0) != VK_SUCCESS)
    die("BindBufferMemory");
  void* p = nullptr;
  if (g_vk.MapMemory(g_vk.dev, m.mem, 0, VK_WHOLE_SIZE, 0, &p) != VK_SUCCESS)
    die("MapMemory");
  m.p = (uint8_t*)p;
  memset(m.p, 0, (size_t)size);
  return m;
}

static uint64_t g_rng = 0x9e3779b97f4a7c15ull;
static uint32_t rnd() {
  g_rng ^= g_rng << 13; g_rng ^= g_rng >> 7; g_rng ^= g_rng << 17;
  return (uint32_t)(g_rng >> 16);
}
static uint16_t bf16_of(float f) {
  uint32_t b;
  memcpy(&b, &f, 4);
  return (uint16_t)((b + 0x7fffu + ((b >> 16) & 1u)) >> 16);
}
static float frnd() {
  return (float)(rnd() & 0xffff) / 65535.0f * 2.0f - 1.0f;
}

static VkShaderModule build(const char* src, const char* defs) {
  std::string spv = "/tmp/h290-" + std::to_string(getpid()) + ".spv";
  std::string cmd = std::string("glslc -fshader-stage=compute "
      "--target-env=vulkan1.3 ") + defs + " " + src + " -o " + spv + " 2>&1";
  int rc = system(cmd.c_str());
  if (rc != 0) {
    cmd = std::string("glslangValidator -V --target-env vulkan1.3 ") + defs +
        " " + src + " -o " + spv + " 2>&1";
    if (system(cmd.c_str()) != 0) die("compile failed:\n%s", cmd.c_str());
  }
  std::ifstream f(spv, std::ios::binary);
  std::ostringstream ss;
  ss << f.rdbuf();
  unlink(spv.c_str());
  VkShaderModuleCreateInfo mi{};
  mi.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  mi.codeSize = ss.str().size();
  mi.pCode = (const uint32_t*)ss.str().data();
  VkShaderModule m;
  if (g_vk.CreateShaderModule(g_vk.dev, &mi, nullptr, &m) != VK_SUCCESS)
    die("CreateShaderModule");
  return m;
}

struct Params {
  uint32_t count, operation, lhs_size, rhs_size, reduce_size, output_size,
      lhs_offset, rhs_offset, output_offset, aux_size, aux_offset,
      matrix_m, matrix_n, matrix_k, flags;
  float alpha, beta;
  uint32_t dims, shape[4], in_strides[4], out_strides[4];
};
static_assert(sizeof(Params) == 120, "push block");

// 2B decode MLP tail shapes (H276 chain: rms 2048, silu*u 6144).
static const uint32_t kHid = 2048;   // hidden / K of gate+up
static const uint32_t kMid = 6144;   // intermediate
static const uint32_t kOut = 2048;   // down output = hidden

static void fill_weights(MB& w, MB& s, MB& b, uint32_t N, uint32_t K) {
  uint64_t wbytes = (uint64_t)N * (K / 8) * 4;
  uint64_t gbytes = (uint64_t)N * (K / 64) * 2;
  uint32_t* wp = (uint32_t*)w.p;
  for (uint64_t i = 0; i < wbytes / 4; ++i) wp[i] = rnd() | (rnd() << 16);
  uint16_t* sp = (uint16_t*)s.p;
  uint16_t* bp = (uint16_t*)b.p;
  for (uint64_t i = 0; i < gbytes / 2; ++i) {
    sp[i] = bf16_of(0.004f + 0.004f * fabsf(frnd()));
    bp[i] = bf16_of(0.02f * frnd());
  }
}

int main() {
  vk_init();
  setup_device();
  uint32_t G = getenv("GB_G") ? (uint32_t)atoi(getenv("GB_G")) : 4;
  uint32_t sets_n = getenv("GB_SETS") ? (uint32_t)atoi(getenv("GB_SETS")) : 8;
  uint32_t rounds = getenv("GB_ROUNDS") ? (uint32_t)atoi(getenv("GB_ROUNDS"))
                                        : 256;
  const char* root = getenv("GB_ROOT");
  if (!root || !root[0]) root = ".";
  std::string norm_src = std::string(root) +
      "/overlay/mlx/backend/omarchy/shaders/fast_norm.comp";
  std::string qmm_src = std::string(root) +
      "/overlay/mlx/backend/omarchy/shaders/qmm_vec.comp";
  std::string fused_src = std::string(root) +
      "/tools/gridbarrier-bench/tail/fused_tail.comp";

  const char* defs_qmm4 =
      "-DUSE_BF16=1 -DUSE_SUBGROUP=1 -DQMM_VEC_Q4_WORD=1 -DQMM_VEC_MULTI=1 "
      "-DROWS_PER_SLOT=2 -DSLOTS_PER_GROUP=4";
  const char* defs_qmm8 =
      "-DUSE_BF16=1 -DUSE_SUBGROUP=1 -DQMM_VEC_Q4_WORD=1 -DQMM_VEC_MULTI=1 "
      "-DROWS_PER_SLOT=2 -DSLOTS_PER_GROUP=8";
  VkShaderModule mod_norm = build(norm_src.c_str(), "-DUSE_BF16=1");
  VkShaderModule mod_gu = build(qmm_src.c_str(), defs_qmm4);
  VkShaderModule mod_dn = build(qmm_src.c_str(), defs_qmm4);
  VkShaderModule mod_fu = build(fused_src.c_str(), defs_qmm8);
  printf("{\"k\":\"stage\",\"s\":\"compiled\"}\n");
  fflush(stdout);

  // Per-pipeline layouts: lavapipe caps maxPerStageDescriptorStorageBuffers
  // at 32; each layout only carries the bindings its shader statically uses.
  auto mklayout = [&](const std::vector<uint32_t>& bs) {
    std::vector<VkDescriptorSetLayoutBinding> lb(bs.size());
    for (size_t i = 0; i < bs.size(); ++i) {
      lb[i].binding = bs[i];
      lb[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      lb[i].descriptorCount = 1;
      lb[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dli{};
    dli.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dli.bindingCount = (uint32_t)bs.size();
    dli.pBindings = lb.data();
    VkDescriptorSetLayout d;
    if (g_vk.CreateDescriptorSetLayout(g_vk.dev, &dli, nullptr, &d) !=
        VK_SUCCESS)
      die("dsl");
    return d;
  };
  std::vector<uint32_t> bs_norm = {0, 1, 3};
  std::vector<uint32_t> bs_qmm;
  for (uint32_t b = 0; b <= 24; ++b) bs_qmm.push_back(b);
  std::vector<uint32_t> bs_fused = bs_qmm;
  for (uint32_t b : {30u, 31u, 34u, 35u, 36u, 37u}) bs_fused.push_back(b);
  VkDescriptorSetLayout dsl_norm = mklayout(bs_norm);
  VkDescriptorSetLayout dsl_qmm = mklayout(bs_qmm);
  VkDescriptorSetLayout dsl_fused = mklayout(bs_fused);
  VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(Params)};
  auto mkplayout = [&](VkDescriptorSetLayout d) {
    VkPipelineLayoutCreateInfo pli{};
    pli.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pli.setLayoutCount = 1;
    pli.pSetLayouts = &d;
    pli.pushConstantRangeCount = 1;
    pli.pPushConstantRanges = &pcr;
    VkPipelineLayout l;
    if (g_vk.CreatePipelineLayout(g_vk.dev, &pli, nullptr, &l) != VK_SUCCESS)
      die("layout");
    return l;
  };
  VkPipelineLayout layout_norm = mkplayout(dsl_norm);
  VkPipelineLayout layout_qmm = mkplayout(dsl_qmm);
  VkPipelineLayout layout_fused = mkplayout(dsl_fused);
  int pipe_idx = 0;
  auto mkpipe = [&](VkShaderModule m, VkPipelineLayout lay) {
    ++pipe_idx;
    VkComputePipelineCreateInfo cpi{};
    cpi.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cpi.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpi.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpi.stage.module = m;
    cpi.stage.pName = "main";
    cpi.layout = lay;
    VkPipeline p;
    VkResult r = g_vk.CreateComputePipelines(g_vk.dev, VK_NULL_HANDLE, 1,
        &cpi, nullptr, &p);
    if (r != VK_SUCCESS) die("pipeline %d rc=%d", pipe_idx, (int)r);
    return p;
  };
  const char* only = getenv("GB_ONLY");
  VkPipeline pipe_norm = VK_NULL_HANDLE, pipe_gu, pipe_dn, pipe_fu;
  if (!only || !strstr(only, "nonorm")) pipe_norm = mkpipe(mod_norm, layout_norm);
  if (!only || strstr(only, "gu") || !strstr(only, "fu"))
    pipe_gu = mkpipe(mod_gu, layout_qmm);
  pipe_dn = mkpipe(mod_dn, layout_qmm);
  pipe_fu = mkpipe(mod_fu, layout_fused);

  VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 200};
  VkDescriptorPoolCreateInfo pi{};
  pi.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  pi.maxSets = 4;
  pi.poolSizeCount = 1;
  pi.pPoolSizes = &ps;
  VkDescriptorPool pool;
  if (g_vk.CreateDescriptorPool(g_vk.dev, &pi, nullptr, &pool) != VK_SUCCESS)
    die("pool");
  enum { S_NORM = 0, S_GU = 1, S_DN = 2, S_FU = 3 };
  VkDescriptorSet sets[4];
  auto alloc1 = [&](VkDescriptorSetLayout d, VkDescriptorSet* out) {
    VkDescriptorSetAllocateInfo dai{};
    dai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dai.descriptorPool = pool;
    dai.descriptorSetCount = 1;
    dai.pSetLayouts = &d;
    if (g_vk.AllocateDescriptorSets(g_vk.dev, &dai, out) != VK_SUCCESS)
      die("sets");
  };
  alloc1(dsl_norm, &sets[S_NORM]);
  alloc1(dsl_qmm, &sets[S_GU]);
  alloc1(dsl_qmm, &sets[S_DN]);
  alloc1(dsl_fused, &sets[S_FU]);

  MB dummy = mk(1 << 20);
  MB syncb = mk(4096);
  MB hid = mk(8192), normw = mk(8192);
  MB x_norm = mk(8192), mid = mk(32768), resid = mk(8192);
  MB out_ref = mk(8192), out_fu = mk(8192), mid_ref = mk(32768);
  MB gw[2], gs[2], gb[2];  // gate(0), up(1)
  MB dw, ds, db;
  for (int j = 0; j < 2; ++j) {
    gw[j] = mk((uint64_t)kMid * (kHid / 8) * 4);
    gs[j] = mk((uint64_t)kMid * (kHid / 64) * 2);
    gb[j] = mk((uint64_t)kMid * (kHid / 64) * 2);
  }
  dw = mk((uint64_t)kOut * (kMid / 8) * 4);
  ds = mk((uint64_t)kOut * (kMid / 64) * 2);
  db = mk((uint64_t)kOut * (kMid / 64) * 2);

  auto bind = [&](VkDescriptorSet set, uint32_t b, MB& m) {
    VkDescriptorBufferInfo bi{m.buf, 0, VK_WHOLE_SIZE};
    VkWriteDescriptorSet w{};
    w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w.dstSet = set;
    w.dstBinding = b;
    w.descriptorCount = 1;
    w.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    w.pBufferInfo = &bi;
    g_vk.UpdateDescriptorSets(g_vk.dev, 1, &w, 0, nullptr);
  };
  // norm set
  bind(sets[S_NORM], 0, hid);
  bind(sets[S_NORM], 1, normw);
  bind(sets[S_NORM], 3, x_norm);
  // gate/up fold set: x = x_norm, block0 output = mid (fold store)
  bind(sets[S_GU], 0, x_norm);
  for (int j = 0; j < 2; ++j) {
    uint32_t base = 1 + j * 6;
    bind(sets[S_GU], base, gw[j]);
    bind(sets[S_GU], base + 1, gs[j]);
    bind(sets[S_GU], base + 2, gb[j]);
    bind(sets[S_GU], base + 3, j == 0 ? mid : dummy);
    bind(sets[S_GU], base + 4, dummy);
    bind(sets[S_GU], base + 5, dummy);
  }
  // down+add set: x = mid, output0 = dummy scratch, addend = resid,
  // sum = out_ref
  bind(sets[S_DN], 0, mid);
  bind(sets[S_DN], 1, dw);
  bind(sets[S_DN], 2, ds);
  bind(sets[S_DN], 3, db);
  bind(sets[S_DN], 4, dummy);
  bind(sets[S_DN], 5, resid);
  bind(sets[S_DN], 6, out_ref);
  // fused set
  bind(sets[S_FU], 1, gw[0]);
  bind(sets[S_FU], 2, gs[0]);
  bind(sets[S_FU], 3, gb[0]);
  bind(sets[S_FU], 4, mid);          // output0: fold store
  bind(sets[S_FU], 5, dummy);
  bind(sets[S_FU], 6, dummy);
  bind(sets[S_FU], 7, gw[1]);
  bind(sets[S_FU], 8, gs[1]);
  bind(sets[S_FU], 9, gb[1]);
  for (uint32_t b : {10u, 11u, 12u}) bind(sets[S_FU], b, dummy);
  bind(sets[S_FU], 13, dw);
  bind(sets[S_FU], 14, ds);
  bind(sets[S_FU], 15, db);
  bind(sets[S_FU], 16, dummy);       // output2 (unused: sum2 has the value)
  bind(sets[S_FU], 17, resid);       // addend2
  bind(sets[S_FU], 18, out_fu);      // sum2
  for (uint32_t b = 19u; b <= 29u; ++b) bind(sets[S_FU], b, dummy);
  bind(sets[S_FU], 30, hid);
  bind(sets[S_FU], 31, normw);
  bind(sets[S_FU], 34, x_norm);      // norm output
  bind(sets[S_FU], 35, mid);         // x_mid (stage 2 x view)
  bind(sets[S_FU], 36, x_norm);      // x_norm (stage 1 x view)
  bind(sets[S_FU], 37, syncb);
  for (uint32_t b : {0u, 32u, 33u}) bind(sets[S_FU], b, dummy);
  for (uint32_t b = 38u; b <= 40u; ++b) bind(sets[S_FU], b, dummy);

  auto fill_data = [&]() {
    uint16_t* h = (uint16_t*)hid.p;
    for (uint32_t i = 0; i < kHid; ++i) h[i] = bf16_of(frnd());
    uint16_t* nw = (uint16_t*)normw.p;
    for (uint32_t i = 0; i < kHid; ++i)
      nw[i] = bf16_of(0.5f + 0.5f * fabsf(frnd()));
    uint16_t* r = (uint16_t*)resid.p;
    for (uint32_t i = 0; i < kOut; ++i) r[i] = bf16_of(frnd());
    fill_weights(gw[0], gs[0], gb[0], kMid, kHid);
    fill_weights(gw[1], gs[1], gb[1], kMid, kHid);
    fill_weights(dw, ds, db, kOut, kMid);
  };
  fill_data();

  VkCommandPoolCreateInfo cpi{};
  cpi.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  cpi.queueFamilyIndex = ctx.qfi;
  VkCommandPool cpool;
  if (g_vk.CreateCommandPool(g_vk.dev, &cpi, nullptr, &cpool) != VK_SUCCESS)
    die("cpool");
  VkCommandBufferAllocateInfo cbi{};
  cbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  cbi.commandPool = cpool;
  cbi.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  cbi.commandBufferCount = 1;
  VkCommandBuffer cmd;
  if (g_vk.AllocateCommandBuffers(g_vk.dev, &cbi, &cmd) != VK_SUCCESS)
    die("cmd");

  auto full_barrier = [&]() {
    VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mb.srcAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    mb.dstAccessMask = mb.srcAccessMask;
    g_vk.CmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
        VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &mb, 0, nullptr, 0,
        nullptr);
  };
  auto zero_sync = [&]() {
    g_vk.CmdFillBuffer(cmd, syncb.buf, 0, 4096, 0);
    VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    mb.dstAccessMask =
        VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    g_vk.CmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, nullptr, 0,
        nullptr);
  };

  Params zn{};
  zn.lhs_size = kHid;         // per-index norm weight
  zn.reduce_size = kHid;      // row length
  zn.output_size = 1;         // one decode row
  zn.alpha = 1e-6f;           // rms eps
  Params pg{};
  pg.operation = 4;
  pg.lhs_size = kHid;
  pg.reduce_size = 64;        // q4 group size
  pg.matrix_m = 1;
  pg.matrix_k = kHid;
  pg.dims = 2;
  pg.shape[0] = kMid;
  pg.shape[1] = kMid;
  pg.flags = 65536u;          // swiglu fold
  pg.count = kMid / 8;        // shipped-geometry workgroups
  Params pd{};
  pd.operation = 4;
  pd.reduce_size = 64;
  pd.matrix_m = 1;
  pd.matrix_k = kMid;
  pd.dims = 1;
  pd.shape[0] = kOut;
  pd.flags = 256u;            // add epilogue weight 0
  pd.count = kOut / 8;
  Params pf{};
  pf.operation = 4;
  pf.lhs_size = kHid;
  pf.reduce_size = kHid;      // norm row length (shared field)
  pf.output_size = 1;
  pf.alpha = 1e-6f;
  pf.matrix_m = 1;
  pf.matrix_k = kHid;         // gu stage K
  pf.dims = 2;
  pf.shape[0] = kMid;
  pf.shape[1] = kMid;
  pf.flags = 65536u;          // gu fold
  pf.out_strides[0] = kOut;   // dn_n
  pf.out_strides[1] = 1024u;  // dn_flags: add epilogue for remapped weight 2 (bit 8+2)
  pf.out_strides[2] = kMid;   // dn_k

  auto submit = [&]() {
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;
    double t0 = now_us();
    if (g_vk.QueueSubmit(g_vk.queue, 1, &si, VK_NULL_HANDLE) != VK_SUCCESS)
      die("QueueSubmit");
    if (g_vk.QueueWaitIdle(g_vk.queue) != VK_SUCCESS) die("QueueWaitIdle");
    return now_us() - t0;
  };

  auto run_ref = [&]() {
    VkCommandBufferBeginInfo bi{
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    g_vk.BeginCommandBuffer(cmd, &bi);
    g_vk.CmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_norm);
    g_vk.CmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
        layout_norm, 0, 1, &sets[S_NORM], 0, nullptr);
    g_vk.CmdPushConstants(cmd, layout_norm, VK_SHADER_STAGE_COMPUTE_BIT, 0,
        sizeof(Params), &zn);
    g_vk.CmdDispatch(cmd, 1, 1, 1);
    full_barrier();
    g_vk.CmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_gu);
    g_vk.CmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
        layout_qmm, 0, 1, &sets[S_GU], 0, nullptr);
    g_vk.CmdPushConstants(cmd, layout_qmm, VK_SHADER_STAGE_COMPUTE_BIT, 0,
        sizeof(Params), &pg);
    g_vk.CmdDispatch(cmd, pg.count, 1, 1);
    full_barrier();
    g_vk.CmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_dn);
    g_vk.CmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
        layout_qmm, 0, 1, &sets[S_DN], 0, nullptr);
    g_vk.CmdPushConstants(cmd, layout_qmm, VK_SHADER_STAGE_COMPUTE_BIT, 0,
        sizeof(Params), &pd);
    g_vk.CmdDispatch(cmd, pd.count, 1, 1);
    full_barrier();
    g_vk.EndCommandBuffer(cmd);
    return submit();
  };

  auto run_fused = [&](bool timed) {
    VkCommandBufferBeginInfo bi{
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    g_vk.BeginCommandBuffer(cmd, &bi);
    zero_sync();
    g_vk.CmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_fu);
    g_vk.CmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
        layout_fused, 0, 1, &sets[S_FU], 0, nullptr);
    g_vk.CmdPushConstants(cmd, layout_fused, VK_SHADER_STAGE_COMPUTE_BIT, 0,
        sizeof(Params), &pf);
    for (uint32_t r = 0; r < (timed ? rounds : 1u); ++r)
      g_vk.CmdDispatch(cmd, G, 1, 1);
    g_vk.EndCommandBuffer(cmd);
    return submit();
  };

  // ---- exactness over randomized data refills -----------------------------
  uint32_t bad_sets = 0, timeouts = 0;
  int64_t first_bad = -1;
  char bad_stage = 0;
  static uint8_t mid_save[32768], xn_save[8192], out_save[8192];
  for (uint32_t s = 0; s < sets_n; ++s) {
    fill_data();
    run_ref();
    memcpy(xn_save, x_norm.p, 8192);
    memcpy(mid_save, mid.p, 32768);
    memcpy(out_save, out_ref.p, 8192);
    memset(x_norm.p, 0, 8192);
    memset(mid.p, 0, 32768);
    memset(out_fu.p, 0, 8192);
    run_fused(false);
    volatile uint32_t* sy = (volatile uint32_t*)syncb.p;
    timeouts += sy[2] != 0;
    struct Cmp {
      const char* name;
      const uint8_t* a;
      const uint8_t* b;
      size_t n;
    };
    const Cmp chk[3] = {
        {"norm", xn_save, (uint8_t*)x_norm.p, 4096},
        {"mid", mid_save, (uint8_t*)mid.p, 12288},
        {"sum", out_save, (uint8_t*)out_fu.p, 4096}};
    for (int c = 0; c < 3; ++c) {
      if (memcmp(chk[c].a, chk[c].b, chk[c].n) != 0) {
        ++bad_sets;
        if (first_bad < 0) {
          bad_stage = chk[c].name[0];
          const uint16_t* A = (const uint16_t*)chk[c].a;
          const uint16_t* B = (const uint16_t*)chk[c].b;
          for (size_t i = 0; i < chk[c].n / 2; ++i) {
            if (A[i] != B[i]) {
              first_bad = (int64_t)i;
              printf("{\"k\":\"first_bad\",\"stage\":\"%s\",\"i\":%zu,"
                  "\"ref\":%04x,\"fused\":%04x}\n",
                  chk[c].name, i, A[i], B[i]);
              break;
            }
          }
        }
      }
    }
    printf("{\"k\":\"exact\",\"set\":%u,\"timeout\":%u}\n", s, sy[2]);
    fflush(stdout);
    if (sy[2]) break;
  }

  // ---- timing --------------------------------------------------------------
  double best_ref = 1e18, best_fu = 1e18;
  if (bad_sets == 0 && timeouts == 0) {
    for (int r = 0; r < 7; ++r) {
      double a = run_ref();
      double b = run_fused(true);
      if (a < best_ref) best_ref = a;
      if (b < best_fu) best_fu = b;
    }
  }

  printf("{\"k\":\"tail\",\"dev\":\"%s\",\"G\":%u,\"sets\":%u,\"bad_sets\":%u,"
      "\"timeouts\":%u,\"first_bad_stage\":\"%c\",\"first_bad_i\":%lld,"
      "\"ref_us\":%.1f,\"fused_us\":%.1f,"
      "\"ref_us_per_tail\":%.4f,\"fused_us_per_tail\":%.4f}\n",
      ctx.name.c_str(), G, sets_n, bad_sets, timeouts,
      bad_sets ? bad_stage : '-', (long long)first_bad,
      best_ref, best_fu,
      best_ref < 1e17 ? best_ref / rounds : -1.0,
      best_fu < 1e17 ? best_fu / rounds : -1.0);
  printf(bad_sets == 0 && timeouts == 0 ? "TAIL_EXACT_OK\n" : "TAIL_FAIL\n");
  return bad_sets == 0 && timeouts == 0 ? 0 : 1;
}
