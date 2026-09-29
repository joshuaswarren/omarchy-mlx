// Dense BF16 prefill matmul bench for the M1 (Honeykrisp fork) and
// correctness screening elsewhere. It times the REAL production shader
// (shaders/mm_bf16_base.comp, a frozen copy of
// overlay/mlx/backend/omarchy/shaders/matmul_coopmat_bf16.comp), a
// candidate restructure (shaders/mm_bf16_cand.comp), and the 16x16
// scalar tile the m < 32 gate falls back to (shaders/mm_tile_base.comp,
// a frozen copy of shaders/matmul.comp compiled with -DUSE_BF16=1) on
// the per-layer projection shapes of the Qwen2.5-0.5B BF16 prefill:
//   q/o    k=896  n=896
//   k/v    k=896  n=128
//   gate/up k=896  n=4864
//   down   k=4864 n=896
// at matrix_m in {30, 262, 1053}, flags=1 (transposed weights), the
// push-constant and binding contract of dispatch_matmul
// (primitives.cpp): four bindings a/b/c/out, alpha=1, beta=0, no bias.
//
// Measurements per (m, shape):
//   * bit-exactness: identical integer-valued bf16 inputs (exact in
//     every accumulation order) through every pair of kernels into
//     distinct outputs, compared as raw storage bits - must be 0
//     mismatches.
//   * the same with fractional bf16 inputs, plus a float64 host
//     reference at m=30: both kernels must sit within the documented
//     anchor bound of truth.
//   * wall-clock timing of R alternating base/cand/tile dispatches in
//     one command buffer, three rounds, per-kernel medians. Wall clock
//     over R=50 dispatches of 50-500 us each keeps submission overhead
//     in the noise and works on devices without timestamps.
//
// Devices without VK_KHR_cooperative_matrix (llvmpipe, stock Mesa) run
// the tile legs and the capability probe, then stop: the coopmat
// kernels cannot dispatch there, which is exactly why their validation
// is a hardware leg.
//
// Build: g++ -std=c++17 -O2 -o /tmp/qmm-prefill-bench tools/qmm-prefill-bench/bench.cpp
// Run (repo root): /tmp/qmm-prefill-bench [--quick]

#include <vulkan/vulkan.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>
#include <vector>

#define LIBVK "libvulkan.so.1"
// KHR cooperative matrix landed in Vulkan headers after 1.3.239; the
// struct is layout-stable and reuses the NV extension's type value, so
// declare it when the installed header lacks it.
#ifndef VK_KHR_cooperative_matrix
#define VK_KHR_cooperative_matrix 1
#define VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR \
  ((VkStructureType)1000249000)
typedef struct VkPhysicalDeviceCooperativeMatrixFeaturesKHR {
  VkStructureType sType;
  void* pNext;
  VkBool32 cooperativeMatrix;
} VkPhysicalDeviceCooperativeMatrixFeaturesKHR;
#endif


struct VkTable {
  void* handle{nullptr};
  VkInstance inst{VK_NULL_HANDLE};
  VkDevice dev{VK_NULL_HANDLE};
#define VK_FN(name) PFN_vk##name name{nullptr}
  VK_FN(GetInstanceProcAddr);
  VK_FN(GetDeviceProcAddr);
  VK_FN(CreateInstance);
  VK_FN(EnumerateDeviceExtensionProperties);
  VK_FN(EnumeratePhysicalDevices);
  VK_FN(GetPhysicalDeviceProperties);
  VK_FN(GetPhysicalDeviceProperties2);
  VK_FN(GetPhysicalDeviceFeatures2);
  VK_FN(GetPhysicalDeviceMemoryProperties);
  VK_FN(GetPhysicalDeviceQueueFamilyProperties);
  VK_FN(CreateDevice);
  VK_FN(GetDeviceQueue);
  VK_FN(CreateBuffer);
  VK_FN(GetBufferMemoryRequirements);
  VK_FN(BindBufferMemory);
  VK_FN(MapMemory);
  VK_FN(UnmapMemory);
  VK_FN(AllocateMemory);
  VK_FN(FreeMemory);
  VK_FN(CreateShaderModule);
  VK_FN(CreateComputePipelines);
  VK_FN(CreatePipelineLayout);
  VK_FN(CreateDescriptorSetLayout);
  VK_FN(CreateDescriptorPool);
  VK_FN(AllocateDescriptorSets);
  VK_FN(UpdateDescriptorSets);
  VK_FN(CreateCommandPool);
  VK_FN(AllocateCommandBuffers);
  VK_FN(BeginCommandBuffer);
  VK_FN(EndCommandBuffer);
  VK_FN(CmdBindPipeline);
  VK_FN(CmdBindDescriptorSets);
  VK_FN(CmdDispatch);
  VK_FN(CmdPushConstants);
  VK_FN(CreateFence);
  VK_FN(WaitForFences);
  VK_FN(ResetFences);
  VK_FN(QueueSubmit);
  VK_FN(DestroyShaderModule);
  VK_FN(DestroyPipeline);
  VK_FN(DestroyPipelineLayout);
  VK_FN(DestroyDescriptorSetLayout);
  VK_FN(DestroyDescriptorPool);
  VK_FN(DestroyCommandPool);
  VK_FN(DestroyBuffer);
  VK_FN(DestroyDevice);
  VK_FN(DestroyInstance);
#undef VK_FN
};

static VkTable g_vk;

static PFN_vkVoidFunction vk_load(VkInstance h, const char* name) {
  return ((PFN_vkGetInstanceProcAddr)g_vk.GetInstanceProcAddr)(h, name);
}
static PFN_vkVoidFunction vk_dev_load(VkDevice d, const char* name) {
  return ((PFN_vkGetDeviceProcAddr)g_vk.GetDeviceProcAddr)(d, name);
}

#define LOAD(name) g_vk.name = (decltype(g_vk.name))vk_load(g_vk.inst, "vk" #name)
#define LOAD_DEV(name) \
  g_vk.name = (decltype(g_vk.name))vk_dev_load(g_vk.dev, "vk" #name)

[[noreturn]] static void die(const char* fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  std::vfprintf(stderr, fmt, ap);
  va_end(ap);
  std::fputc('\n', stderr);
  std::exit(1);
}

static std::string read_file(const char* path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) die("open %s", path);
  std::ostringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

static int compile_shader(const char* src, const char* defines,
    const char* out_spv) {
  const char* fronts[] = {
      "glslangValidator -V --target-env vulkan1.3 ",
      "glslc -fshader-stage=compute --target-env=vulkan1.3 ",
  };
  for (const char* front : fronts) {
    std::string cmd = front;
    cmd += defines;
    cmd += " ";
    cmd += src;
    cmd += " -o ";
    cmd += out_spv;
    cmd += " 2>&1";
    if (system(cmd.c_str()) == 0) {
      struct stat st;
      if (stat(out_spv, &st) == 0 && st.st_size != 0) return 0;
    }
    std::fprintf(stderr, "shader compile failed: %s\n", cmd.c_str());
  }
  return -1;
}

struct Buf {
  VkBuffer buf{VK_NULL_HANDLE};
  VkDeviceMemory mem{VK_NULL_HANDLE};
  VkDeviceSize size{0};
  void* mapped{nullptr};
};

static uint32_t find_memtype(uint32_t bits, VkMemoryPropertyFlags want,
    const VkPhysicalDeviceMemoryProperties& mp) {
  for (uint32_t i = 0; i < mp.memoryTypeCount; ++i) {
    if ((bits & (1u << i)) &&
        (mp.memoryTypes[i].propertyFlags & want) == want) {
      return i;
    }
  }
  return UINT32_MAX;
}

static Buf make_buf(VkDevice dev, const VkPhysicalDeviceMemoryProperties& mp,
    VkDeviceSize size) {
  Buf b;
  b.size = size;
  VkBufferCreateInfo bi{};
  bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  bi.size = size;
  bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
  bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  if (g_vk.CreateBuffer(dev, &bi, nullptr, &b.buf) != VK_SUCCESS)
    die("CreateBuffer");
  VkMemoryRequirements req;
  g_vk.GetBufferMemoryRequirements(dev, b.buf, &req);
  uint32_t mt = find_memtype(req.memoryTypeBits,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
          VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
      mp);
  if (mt == UINT32_MAX) die("no host-visible memtype");
  VkMemoryAllocateInfo ai{};
  ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  ai.allocationSize = req.size;
  ai.memoryTypeIndex = mt;
  if (g_vk.AllocateMemory(dev, &ai, nullptr, &b.mem) != VK_SUCCESS)
    die("AllocateMemory");
  if (g_vk.BindBufferMemory(dev, b.buf, b.mem, 0) != VK_SUCCESS)
    die("BindBufferMemory");
  if (g_vk.MapMemory(dev, b.mem, 0, size, 0, &b.mapped) != VK_SUCCESS)
    die("MapMemory");
  return b;
}

// Push constants: must byte-match the matmul shader Params block
// (32 x 4-byte words, scalar alignment throughout).
struct Params {
  uint32_t count;
  uint32_t operation;
  uint32_t lhs_size;
  uint32_t rhs_size;
  uint32_t reduce_size;
  uint32_t output_size;
  uint32_t lhs_offset;
  uint32_t rhs_offset;
  uint32_t output_offset;
  uint32_t aux_size;
  uint32_t aux_offset;
  uint32_t matrix_m;
  uint32_t matrix_n;
  uint32_t matrix_k;
  uint32_t flags;
  float alpha;
  float beta;
  uint32_t dims;
  uint32_t shape[4];
  uint32_t in_strides[4];
  uint32_t out_strides[4];
};
static_assert(sizeof(Params) == 120, "push constant block must be 120 bytes");

static constexpr uint32_t kBindings = 5;

struct Shape {
  const char* name;
  uint32_t k;
  uint32_t n;
};

static const Shape kShapes[] = {
    {"q_o_896x896", 896, 896},
    {"kv_896x128", 896, 128},
    {"gate_up_896x4864", 896, 4864},
    {"down_4864x896", 4864, 896},
};
static const uint32_t kMs[] = {30, 262, 1053};

struct DeviceCtx {
  VkQueue queue{VK_NULL_HANDLE};
  uint32_t qfi{0};
  VkPhysicalDeviceMemoryProperties mp{};
  std::string name;
  uint32_t subgroupSize{0};
  bool coopmat{false};
};

static DeviceCtx setup_device() {
  VkApplicationInfo ai{};
  ai.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  ai.pApplicationName = "qmm-prefill-bench";
  ai.apiVersion = VK_API_VERSION_1_2;
  VkInstanceCreateInfo ici{};
  ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  ici.pApplicationInfo = &ai;
  if (g_vk.CreateInstance(&ici, nullptr, &g_vk.inst) != VK_SUCCESS)
    die("CreateInstance");
  LOAD(DestroyInstance);
  LOAD(EnumeratePhysicalDevices);
  LOAD(EnumerateDeviceExtensionProperties);
  LOAD(GetPhysicalDeviceFeatures2);
  LOAD(GetPhysicalDeviceProperties);
  LOAD(GetPhysicalDeviceProperties2);
  LOAD(GetPhysicalDeviceMemoryProperties);
  LOAD(GetPhysicalDeviceQueueFamilyProperties);
  LOAD(CreateDevice);

  uint32_t n = 0;
  g_vk.EnumeratePhysicalDevices(g_vk.inst, &n, nullptr);
  if (n == 0) die("no Vulkan physical devices");
  std::vector<VkPhysicalDevice> pds(n);
  g_vk.EnumeratePhysicalDevices(g_vk.inst, &n, pds.data());
  DeviceCtx c;
  // Pre-pass: when any device exposes cooperative matrix (the fork
  // driver), prefer it over a software device that enumerates first.
  bool any_cm = false;
  for (uint32_t i = 0; i < n && !any_cm; ++i) {
    uint32_t extn = 0;
    g_vk.EnumerateDeviceExtensionProperties(pds[i], nullptr, &extn, nullptr);
    std::vector<VkExtensionProperties> exts(extn);
    if (extn) {
      g_vk.EnumerateDeviceExtensionProperties(
          pds[i], nullptr, &extn, exts.data());
    }
    for (const auto& e : exts) {
      if (std::strcmp(e.extensionName, "VK_KHR_cooperative_matrix") == 0) {
        any_cm = true;
        break;
      }
    }
  }
  for (uint32_t i = 0; i < n; ++i) {
    VkPhysicalDeviceProperties props{};
    g_vk.GetPhysicalDeviceProperties(pds[i], &props);
    if (props.apiVersion < VK_API_VERSION_1_1) continue;
    std::fprintf(stderr, "probing %s api=%u.%u\n", props.deviceName,
        VK_API_VERSION_MAJOR(props.apiVersion),
        VK_API_VERSION_MINOR(props.apiVersion));
    uint32_t extn = 0;
    g_vk.EnumerateDeviceExtensionProperties(pds[i], nullptr, &extn, nullptr);
    std::vector<VkExtensionProperties> exts(extn);
    if (extn) {
      g_vk.EnumerateDeviceExtensionProperties(
          pds[i], nullptr, &extn, exts.data());
    }
    bool has_cm = false;
    for (const auto& e : exts) {
      if (std::strcmp(e.extensionName, "VK_KHR_cooperative_matrix") == 0) {
        has_cm = true;
      }
    }

    VkPhysicalDeviceSubgroupProperties sub{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};
    VkPhysicalDeviceCooperativeMatrixFeaturesKHR cmf{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR};
    std::memset(&cmf, 0, sizeof(cmf));
    cmf.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR;
    VkPhysicalDeviceProperties2 props2{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    props2.pNext = &sub;
    // Only chain the cooperative-matrix struct when the driver knows
    // the extension: old llvmpipe crashed on unknown property types.
    sub.pNext = nullptr;
    g_vk.GetPhysicalDeviceProperties2(pds[i], &props2);
    // Feature flags are queried through Features2, not Properties2:
    // the driver leaves unknown-to-properties structs untouched.
    if (has_cm) {
      VkPhysicalDeviceFeatures2 f2{
          VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
      f2.pNext = &cmf;
      g_vk.GetPhysicalDeviceFeatures2(pds[i], &f2);
    }
    uint32_t qfn = 0;
    g_vk.GetPhysicalDeviceQueueFamilyProperties(pds[i], &qfn, nullptr);
    std::vector<VkQueueFamilyProperties> qfpv(qfn);
    g_vk.GetPhysicalDeviceQueueFamilyProperties(pds[i], &qfn, qfpv.data());
    for (uint32_t q = 0; q < qfn; ++q) {
      if ((qfpv[q].queueFlags & VK_QUEUE_COMPUTE_BIT) == 0) continue;
      c.qfi = q;
      c.subgroupSize = sub.subgroupSize;
      c.coopmat = has_cm && cmf.cooperativeMatrix == VK_TRUE;
      c.name = props.deviceName;
      g_vk.GetPhysicalDeviceMemoryProperties(pds[i], &c.mp);
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
      const char* exts_on[1] = {"VK_KHR_cooperative_matrix"};
      if (c.coopmat) {
        dci.enabledExtensionCount = 1;
        dci.ppEnabledExtensionNames = exts_on;
        dci.pNext = &cmf;
      }
      if (g_vk.CreateDevice(pds[i], &dci, nullptr, &g_vk.dev) != VK_SUCCESS)
        die("CreateDevice");
      LOAD_DEV(GetDeviceQueue);
      LOAD_DEV(CreateBuffer);
      LOAD_DEV(GetBufferMemoryRequirements);
      LOAD_DEV(BindBufferMemory);
      LOAD_DEV(MapMemory);
      LOAD_DEV(UnmapMemory);
      LOAD_DEV(AllocateMemory);
      LOAD_DEV(FreeMemory);
      LOAD_DEV(CreateShaderModule);
      LOAD_DEV(CreateComputePipelines);
      LOAD_DEV(CreatePipelineLayout);
      LOAD_DEV(CreateDescriptorSetLayout);
      LOAD_DEV(CreateDescriptorPool);
      LOAD_DEV(AllocateDescriptorSets);
      LOAD_DEV(UpdateDescriptorSets);
      LOAD_DEV(CreateCommandPool);
      LOAD_DEV(AllocateCommandBuffers);
      LOAD_DEV(BeginCommandBuffer);
      LOAD_DEV(EndCommandBuffer);
      LOAD_DEV(CmdBindPipeline);
      LOAD_DEV(CmdBindDescriptorSets);
      LOAD_DEV(CmdDispatch);
      LOAD_DEV(CmdPushConstants);
      LOAD_DEV(CreateFence);
      LOAD_DEV(WaitForFences);
      LOAD_DEV(ResetFences);
      LOAD_DEV(QueueSubmit);
      LOAD_DEV(DestroyShaderModule);
      LOAD_DEV(DestroyPipeline);
      LOAD_DEV(DestroyPipelineLayout);
      LOAD_DEV(DestroyDescriptorSetLayout);
      LOAD_DEV(DestroyDescriptorPool);
      LOAD_DEV(DestroyCommandPool);
      LOAD_DEV(DestroyBuffer);
      LOAD_DEV(FreeMemory);
      LOAD_DEV(DestroyDevice);
      g_vk.GetDeviceQueue(g_vk.dev, q, 0, &c.queue);
      std::printf(
          "{\"k\":\"dev\",\"name\":\"%s\",\"subgroupSize\":%u,"
          "\"coopmat\":%s}\n",
          c.name.c_str(), c.subgroupSize, c.coopmat ? "true" : "false");
      return c;
    }
  }
  die("no Vulkan 1.2 compute device");
}

struct Side {
  const char* tag;
  VkShaderModule mod{VK_NULL_HANDLE};
  VkDescriptorSetLayout dsl{VK_NULL_HANDLE};
  VkPipelineLayout layout{VK_NULL_HANDLE};
  VkPipeline pipe{VK_NULL_HANDLE};
};

static void make_pipeline(DeviceCtx& c, Side& side) {
  VkDescriptorSetLayoutBinding b[kBindings]{};
  for (uint32_t i = 0; i < kBindings; ++i) {
    b[i].binding = i;
    b[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    b[i].descriptorCount = 1;
    b[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  }
  VkDescriptorSetLayoutCreateInfo dslci{};
  dslci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  dslci.bindingCount = kBindings;
  dslci.pBindings = b;
  if (g_vk.CreateDescriptorSetLayout(g_vk.dev, &dslci, nullptr, &side.dsl) !=
      VK_SUCCESS)
    die("CreateDescriptorSetLayout");

  VkPipelineLayoutCreateInfo plci{};
  plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  plci.setLayoutCount = 1;
  plci.pSetLayouts = &side.dsl;
  VkPushConstantRange pc{};
  pc.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  pc.offset = 0;
  pc.size = sizeof(Params);
  plci.pushConstantRangeCount = 1;
  plci.pPushConstantRanges = &pc;
  if (g_vk.CreatePipelineLayout(g_vk.dev, &plci, nullptr, &side.layout) !=
      VK_SUCCESS)
    die("CreatePipelineLayout");

  VkComputePipelineCreateInfo cpci{};
  cpci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
  cpci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  cpci.stage.module = side.mod;
  cpci.stage.pName = "main";
  cpci.layout = side.layout;
  if (g_vk.CreateComputePipelines(g_vk.dev, VK_NULL_HANDLE, 1, &cpci, nullptr,
          &side.pipe) != VK_SUCCESS)
    die("CreateComputePipelines");
}

struct SetBufs {
  Buf x;
  Buf w;
  Buf s;
  Buf bias;
  Buf out;
};

static VkDescriptorSet make_set(DeviceCtx& c, VkDescriptorPool pool,
    VkDescriptorSetLayout dsl, const SetBufs& bufs) {
  VkDescriptorSetAllocateInfo dsai{};
  dsai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  dsai.descriptorPool = pool;
  dsai.descriptorSetCount = 1;
  dsai.pSetLayouts = &dsl;
  VkDescriptorSet set;
  if (g_vk.AllocateDescriptorSets(g_vk.dev, &dsai, &set) != VK_SUCCESS)
    die("AllocateDescriptorSets");
  VkDescriptorBufferInfo dbi[kBindings]{};
  VkWriteDescriptorSet w[kBindings]{};
  Buf const* bufs_arr[kBindings] = {&bufs.x, &bufs.w, &bufs.s, &bufs.bias, &bufs.out};
  for (uint32_t i = 0; i < kBindings; ++i) {
    dbi[i].buffer = bufs_arr[i]->buf;
    dbi[i].offset = 0;
    dbi[i].range = VK_WHOLE_SIZE;
    w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w[i].dstSet = set;
    w[i].dstBinding = i;
    w[i].descriptorCount = 1;
    w[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    w[i].pBufferInfo = &dbi[i];
  }
  g_vk.UpdateDescriptorSets(g_vk.dev, kBindings, w, 0, nullptr);
  return set;
}

static uint16_t bf16_of(float v) {
  uint32_t bits;
  std::memcpy(&bits, &v, 4);
  return (uint16_t)((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
}
static uint32_t xorshift(uint32_t& s) {
  s ^= s << 13;
  s ^= s >> 17;
  s ^= s << 5;
  return s;
}

static std::string arg_str(int argc, char** argv, const char* name, const char* dflt) {
  for (int i = 1; i + 1 < argc; ++i)
    if (std::strcmp(argv[i], name) == 0) return argv[i + 1];
  return dflt;
}

int main(int argc, char** argv) {
  g_vk.handle = dlopen(LIBVK, RTLD_NOW | RTLD_LOCAL);
  if (!g_vk.handle) die("dlopen %s: %s", LIBVK, dlerror());
  g_vk.GetInstanceProcAddr =
      (PFN_vkGetInstanceProcAddr)dlsym(g_vk.handle, "vkGetInstanceProcAddr");
  g_vk.GetDeviceProcAddr =
      (PFN_vkGetDeviceProcAddr)dlsym(g_vk.handle, "vkGetDeviceProcAddr");
  g_vk.CreateInstance =
      (PFN_vkCreateInstance)g_vk.GetInstanceProcAddr(nullptr, "vkCreateInstance");
  if (!g_vk.CreateInstance) die("vkCreateInstance missing");
  DeviceCtx c = setup_device();
  if (!c.coopmat) die("no coopmat device");

  const uint32_t M = (uint32_t)std::atoi(arg_str(argc, argv, "--m", "512").c_str());
  const std::string shape = arg_str(argc, argv, "--shape", "gate");
  uint32_t N = 6144, K = 2048;
  if (shape == "down") { N = 2048; K = 6144; }
  else if (shape == "qkvz") { N = 8192; K = 2048; }
  else if (shape == "zout") { N = 2048; K = 2048; }
  else if (shape == "q4096") { N = 4096; K = 2048; }
  const std::string base_src = arg_str(argc, argv, "--base",
      "overlay/mlx/backend/omarchy/shaders/qmm_coopmat.comp");
  const std::string cand_src = arg_str(argc, argv, "--cand", base_src.c_str());
  const std::string base_def = arg_str(argc, argv, "--base-def", "-DX_F32=1 -DOUT_BF16=1");
  const std::string cand_def = arg_str(argc, argv, "--cand-def", base_def.c_str());
  const uint32_t base_rows = (uint32_t)std::atoi(arg_str(argc, argv, "--base-rows", "32").c_str());
  const uint32_t cand_rows = (uint32_t)std::atoi(arg_str(argc, argv, "--cand-rows", "32").c_str());
  const int reps = std::atoi(arg_str(argc, argv, "--reps", "20").c_str());
  const int rounds = std::atoi(arg_str(argc, argv, "--rounds", "5").c_str());
  const uint32_t base_cols = (uint32_t)std::atoi(arg_str(argc, argv, "--base-cols", "32").c_str());
  const uint32_t cand_cols = (uint32_t)std::atoi(arg_str(argc, argv, "--cand-cols", "32").c_str());
  const uint32_t cand_flags = (uint32_t)std::atoi(arg_str(argc, argv, "--cand-flags", "0").c_str());

  auto make_module = [&](const std::string& spv) {
    VkShaderModuleCreateInfo mi{};
    mi.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    mi.codeSize = spv.size();
    mi.pCode = (const uint32_t*)spv.data();
    VkShaderModule m;
    if (g_vk.CreateShaderModule(g_vk.dev, &mi, nullptr, &m) != VK_SUCCESS)
      die("CreateShaderModule");
    return m;
  };
  std::string bp = "/tmp/qpb_base." + std::to_string(getpid()) + ".spv";
  std::string cp = "/tmp/qpb_cand." + std::to_string(getpid()) + ".spv";
  if (compile_shader(base_src.c_str(), base_def.c_str(), bp.c_str()) != 0) die("base compile");
  if (compile_shader(cand_src.c_str(), cand_def.c_str(), cp.c_str()) != 0) die("cand compile");
  Side base{"base"}, cand{"cand"};
  base.mod = make_module(read_file(bp.c_str()));
  cand.mod = make_module(read_file(cp.c_str()));
  make_pipeline(c, base);
  make_pipeline(c, cand);

  VkCommandPoolCreateInfo cpi{};
  cpi.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  cpi.queueFamilyIndex = c.qfi;
  VkCommandPool pool;
  if (g_vk.CreateCommandPool(g_vk.dev, &cpi, nullptr, &pool) != VK_SUCCESS) die("pool");
  VkCommandBufferAllocateInfo cbai{};
  cbai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  cbai.commandPool = pool;
  cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  cbai.commandBufferCount = 1;
  VkCommandBuffer cmd;
  if (g_vk.AllocateCommandBuffers(g_vk.dev, &cbai, &cmd) != VK_SUCCESS) die("cmdbuf");
  VkFenceCreateInfo fci{};
  fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  VkFence fence;
  if (g_vk.CreateFence(g_vk.dev, &fci, nullptr, &fence) != VK_SUCCESS) die("fence");
  VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kBindings * 4};
  VkDescriptorPoolCreateInfo dpci{};
  dpci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  dpci.maxSets = 4;
  dpci.poolSizeCount = 1;
  dpci.pPoolSizes = &ps;
  VkDescriptorPool dpool;
  if (g_vk.CreateDescriptorPool(g_vk.dev, &dpci, nullptr, &dpool) != VK_SUCCESS) die("dpool");

  // Buffers: x f32 (bf16-exact values), packed 4-bit weights, bf16 scales/biases, bf16 out.
  // --pad-w / --pad-s / --pad-o KB: dummy allocations before W / scales / out to shift relative placement.
  auto padkb = [&](const char* name) { return (size_t)std::atoll(arg_str(argc, argv, name, "0").c_str()) * 1024; };
  Buf bx = make_buf(g_vk.dev, c.mp, (size_t)M * K * 4);
  if (padkb("--pad-w")) make_buf(g_vk.dev, c.mp, padkb("--pad-w"));
  Buf bw = make_buf(g_vk.dev, c.mp, (size_t)N * (K / 8) * 4);
  if (padkb("--pad-s")) make_buf(g_vk.dev, c.mp, padkb("--pad-s"));
  Buf bs = make_buf(g_vk.dev, c.mp, (size_t)N * (K / 64) * 2);
  Buf bb = make_buf(g_vk.dev, c.mp, (size_t)N * (K / 64) * 2);
  if (padkb("--pad-o")) make_buf(g_vk.dev, c.mp, padkb("--pad-o"));
  Buf bo = make_buf(g_vk.dev, c.mp, (size_t)M * N * 2);
  SetBufs bufs{bx, bw, bs, bb, bo};
  uint32_t seed = 0x1234567u;
  float* x = (float*)bufs.x.mapped;
  for (size_t i = 0; i < (size_t)M * K; ++i) {
    float v = ((int32_t)(xorshift(seed) & 0xffff) - 32768) / 65536.0f;
    uint32_t bits;
    std::memcpy(&bits, &v, 4);
    bits &= 0xffff0000u;
    std::memcpy(&x[i], &bits, 4);
  }
  uint32_t* w = (uint32_t*)bufs.w.mapped;
  for (size_t i = 0; i < (size_t)N * (K / 8); ++i) w[i] = xorshift(seed);
  uint16_t* sc = (uint16_t*)bufs.s.mapped;
  uint16_t* bi = (uint16_t*)bufs.bias.mapped;
  for (size_t i = 0; i < (size_t)N * (K / 64); ++i) {
    sc[i] = bf16_of(0.01f + (xorshift(seed) & 0xff) / 25600.0f);
    bi[i] = bf16_of(-0.05f + (xorshift(seed) & 0xff) / 2560.0f);
  }
  std::memset(bufs.out.mapped, 0, (size_t)M * N * 2);
  VkDescriptorSet base_set = make_set(c, dpool, base.dsl, bufs);
  SetBufs cbufs = bufs;
  if (std::atoi(arg_str(argc, argv, "--cand-xtile", "0").c_str())) {
    // 8x8-tile-major copy of x: tile (rb, cb) is 64 contiguous floats, row-major inside.
    uint32_t mpad = (M + 7) / 8 * 8;
    cbufs.x = make_buf(g_vk.dev, c.mp, (size_t)mpad * K * 4);
    float* xt = (float*)cbufs.x.mapped;
    std::memset(xt, 0, (size_t)mpad * K * 4);
    for (uint32_t r = 0; r < M; ++r)
      for (uint32_t k = 0; k < K; ++k)
        xt[(((size_t)(r / 8) * (K / 8) + k / 8) * 64) + (r % 8) * 8 + (k % 8)] = x[(size_t)r * K + k];
  }
  VkDescriptorSet cand_set = make_set(c, dpool, cand.dsl, cbufs);

  Params p{};
  p.count = M * N;
  p.output_size = M * N;
  p.matrix_m = M;
  p.matrix_n = N;
  p.matrix_k = K;
  p.shape[0] = 1;
  auto run = [&](Side& side, VkDescriptorSet set, uint32_t rows, uint32_t flags, int nreps, uint32_t cols) {
    Params q = p;
    q.flags = flags << 16;
    VkCommandBufferBeginInfo bi2{};
    bi2.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi2.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    g_vk.BeginCommandBuffer(cmd, &bi2);
    g_vk.CmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, side.pipe);
    g_vk.CmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, side.layout, 0, 1, &set, 0, nullptr);
    g_vk.CmdPushConstants(cmd, side.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(Params), &q);
    for (int i = 0; i < nreps; ++i) g_vk.CmdDispatch(cmd, (N + cols - 1) / cols, (M + rows - 1) / rows, 1);
    g_vk.EndCommandBuffer(cmd);
    g_vk.ResetFences(g_vk.dev, 1, &fence);
    VkSubmitInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;
    auto t0 = std::chrono::steady_clock::now();
    g_vk.QueueSubmit(c.queue, 1, &si, fence);
    g_vk.WaitForFences(g_vk.dev, 1, &fence, VK_TRUE, 60'000'000'000ull);
    auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::micro>(t1 - t0).count() / nreps;
  };
  // Correctness: one dispatch each, compare bit for bit.
  std::vector<uint16_t> base_out((size_t)M * N), cand_out((size_t)M * N);
  run(base, base_set, base_rows, 0, 1, base_cols);
  std::memcpy(base_out.data(), bufs.out.mapped, base_out.size() * 2);
  std::memset(bufs.out.mapped, 0, (size_t)M * N * 2);
  run(cand, cand_set, cand_rows, cand_flags, 1, cand_cols);
  std::memcpy(cand_out.data(), bufs.out.mapped, cand_out.size() * 2);
  size_t mism = 0, first = (size_t)-1;
  for (size_t i = 0; i < base_out.size(); ++i)
    if (base_out[i] != cand_out[i]) { if (first == (size_t)-1) first = i; ++mism; }
  // --realloc N: rebuild and refill all buffers N times inside this process,
  // timing the base pipeline after each rebuild. If the 9.6/13.0 ms mode is a
  // property of the allocation, it re-rolls per rebuild.
  if (int n = std::atoi(arg_str(argc, argv, "--realloc", "0").c_str()); n > 0) {
    for (int i = 0; i < n; ++i) {
      g_vk.FreeMemory(g_vk.dev, bufs.x.mem, nullptr);
      g_vk.DestroyBuffer(g_vk.dev, bufs.x.buf, nullptr);
      g_vk.FreeMemory(g_vk.dev, bufs.w.mem, nullptr);
      g_vk.DestroyBuffer(g_vk.dev, bufs.w.buf, nullptr);
      g_vk.FreeMemory(g_vk.dev, bufs.out.mem, nullptr);
      g_vk.DestroyBuffer(g_vk.dev, bufs.out.buf, nullptr);
      bufs.x = make_buf(g_vk.dev, c.mp, (size_t)M * K * 4);
      bufs.w = make_buf(g_vk.dev, c.mp, (size_t)N * (K / 8) * 4);
      bufs.out = make_buf(g_vk.dev, c.mp, (size_t)M * N * 2);
      VkDescriptorSet ns = make_set(c, dpool, base.dsl, bufs);
      run(base, ns, base_rows, 0, 3, base_cols);
      double us = run(base, ns, base_rows, 0, 20, base_cols);
      std::printf("{\"k\":\"realloc\",\"i\":%d,\"base_us\":%.1f}\n", i, us);
    }
    return 0;
  }
  // Timing: alternate arms, warmup then rounds.
  run(base, base_set, base_rows, 0, 3, base_cols);
  run(cand, cand_set, cand_rows, cand_flags, 3, cand_cols);
  std::vector<double> bt, ct;
  for (int r = 0; r < rounds; ++r) {
    bt.push_back(run(base, base_set, base_rows, 0, reps, base_cols));
    ct.push_back(run(cand, cand_set, cand_rows, cand_flags, reps, cand_cols));
  }
  std::sort(bt.begin(), bt.end());
  std::sort(ct.begin(), ct.end());
  double flops = 2.0 * M * N * K;
  std::printf(
      "{\"k\":\"qmm\",\"shape\":\"%s\",\"m\":%u,\"n\":%u,\"kk\":%u,\"mismatch\":%zu,\"first_bad\":%lld,"
      "\"base_us\":%.1f,\"cand_us\":%.1f,\"base_tflops\":%.3f,\"cand_tflops\":%.3f,\"cand_vs_base\":%.4f}\n",
      shape.c_str(), M, N, K, mism, first == (size_t)-1 ? -1LL : (long long)first, bt[bt.size() / 2], ct[ct.size() / 2],
      flops / (bt[bt.size() / 2] * 1e6), flops / (ct[ct.size() / 2] * 1e6), bt[bt.size() / 2] / ct[ct.size() / 2]);
  return 0;
}
