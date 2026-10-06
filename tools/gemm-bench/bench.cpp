// Dense f16/bf16 GEMM bench for the omarchy matmul shaders. It dispatches
// any number of shader "sides" on one M x N x K problem through the
// dispatch_matmul push-constant and binding contract (primitives.cpp:
// four bindings a/b/c/out, alpha 1, beta 0, no bias), then reports:
//   * accuracy against a float64 host reference on sampled outputs:
//     the fraction of outputs equal to the correctly rounded truth and
//     the mean/max error in output ULPs (a reordering passes when both
//     are equal or better than the shipped kernel);
//   * storage-bit mismatches of every side against the first side;
//   * wall-clock time per dispatch (R dispatches per command buffer,
//     sides interleaved round by round so they share one power state)
//     and TFLOP/s from the median round.
//
// Side spec: name=path:tile_m:tile_n:flags[:-DX=1,-DY=2]. flags carries
// the transposes (1 = rhs n-major, 4 = lhs column-major) and any
// kernel bits the host dispatch would set. The grid is
// (ceil(n / tile_n), ceil(m / tile_m), 1).
//
// Build: g++ -std=c++17 -O2 -pthread -o /tmp/gemm-bench tools/gemm-bench/bench.cpp
// Run (repo root): /tmp/gemm-bench --dtype f16 --mnk 4096,4096,4096
//   --side rb=overlay/mlx/backend/omarchy/shaders/matmul_rb.comp:64:64:0

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
  uint32_t lhs_gap;
  uint32_t rhs_gap;
};
static_assert(sizeof(Params) == 128, "push constant block must be 128 bytes");

static constexpr uint32_t kBindings = 4;

enum class Dtype { F16, BF16 };

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
  ai.pApplicationName = "gemm-bench";
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
      if (c.coopmat) {
        auto props_fn = (PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR)
            vk_load(g_vk.inst,
                "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR");
        uint32_t count = 0;
        if (props_fn && props_fn(pds[i], &count, nullptr) == VK_SUCCESS) {
          std::vector<VkCooperativeMatrixPropertiesKHR> shapes(count,
              {VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR});
          props_fn(pds[i], &count, shapes.data());
          for (const auto& s : shapes) {
            std::printf("{\"k\":\"coopmat_shape\",\"mnk\":[%u,%u,%u],"
                "\"abcr\":[%d,%d,%d,%d],\"scope\":%d}\n",
                s.MSize, s.NSize, s.KSize, s.AType, s.BType, s.CType,
                s.ResultType, s.scope);
          }
        }
      }
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
  Buf a;
  Buf b;
  Buf c;
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
  Buf const* bufs_arr[kBindings] = {&bufs.a, &bufs.b, &bufs.c, &bufs.out};
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

static uint32_t group_count(uint32_t extent, uint32_t tile) {
  return (extent + tile - 1u) / tile;
}

static float f16_to_f32(uint16_t h) {
  uint32_t exp = (h >> 10) & 0x1fu;
  uint32_t man = h & 0x3ffu;
  float v;
  if (exp == 0) {
    v = std::ldexp((float)man, -24);
  } else if (exp == 31) {
    v = man ? NAN : INFINITY;
  } else {
    v = std::ldexp((float)(man | 0x400u), (int)exp - 25);
  }
  return (h & 0x8000u) ? -v : v;
}

static double to_f64(Dtype dt, uint16_t h) {
  if (dt == Dtype::F16) return f16_to_f32(h);
  uint32_t bits = (uint32_t)h << 16;
  float v;
  std::memcpy(&v, &bits, 4);
  return v;
}

// Round-to-nearest-even onto the storage grid through f32 first: the
// kernels round an f32 accumulator once.
static uint16_t round_store(Dtype dt, double value) {
  float f = (float)value;
  uint32_t bits;
  std::memcpy(&bits, &f, 4);
  if (dt == Dtype::BF16) {
    if (std::isnan(f)) return (uint16_t)((bits >> 16) | 0x40u);
    return (uint16_t)((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
  }
  uint16_t sign = (uint16_t)((bits >> 16) & 0x8000u);
  double a = std::fabs((double)f);
  if (a >= 65520.0) return (uint16_t)(sign | 0x7c00u);
  int e;
  std::frexp(a, &e);
  // Spacing 2^(e-11) for normals, 2^-24 for subnormals.
  int shift = std::max(e - 11, -24);
  double q = std::nearbyint(std::ldexp(a, -shift));  // integer multiple
  if (q == 0.0) return sign;
  // q * 2^shift; renormalize when rounding carried into the next binade.
  if (e <= -14) {
    uint32_t mant = (uint32_t)q;  // <= 1024: subnormal or smallest normal
    return (uint16_t)(sign | mant);
  }
  uint32_t qi = (uint32_t)q;  // in [1024, 2048]
  int exp = shift + 10;       // value = qi * 2^shift = (qi/1024) * 2^exp
  if (qi == 2048u) {
    qi = 1024u;
    ++exp;
  }
  if (exp > 15) return (uint16_t)(sign | 0x7c00u);
  return (uint16_t)(sign | ((uint32_t)(exp + 15) << 10) | (qi - 1024u));
}

static double ulp_at(Dtype dt, double v) {
  int e;
  std::frexp(std::fabs(v), &e);
  return dt == Dtype::F16 ? std::ldexp(1.0, std::max(e - 11, -24))
                          : std::ldexp(1.0, e - 8);
}

struct SideSpec {
  std::string name;
  std::string path;
  uint32_t tile_m{32};
  uint32_t tile_n{32};
  uint32_t flags{0};
  std::string defines;
  Side side{"side"};
  VkDescriptorSet set{VK_NULL_HANDLE};
  Buf out;
  std::vector<uint16_t> result;
  std::vector<double> us;
};

static SideSpec parse_side(const char* arg) {
  SideSpec s;
  std::string a = arg;
  size_t eq = a.find('=');
  if (eq == std::string::npos) die("side spec needs name=: %s", arg);
  s.name = a.substr(0, eq);
  std::vector<std::string> parts;
  std::stringstream ss(a.substr(eq + 1));
  std::string part;
  while (std::getline(ss, part, ':')) parts.push_back(part);
  if (parts.size() < 4) die("side spec path:tm:tn:flags[:defs]: %s", arg);
  s.path = parts[0];
  s.tile_m = (uint32_t)std::stoul(parts[1]);
  s.tile_n = (uint32_t)std::stoul(parts[2]);
  s.flags = (uint32_t)std::stoul(parts[3], nullptr, 0);
  if (parts.size() > 4) {
    s.defines = parts[4];
    for (char& ch : s.defines) {
      if (ch == ',') ch = ' ';
    }
  }
  return s;
}

int main(int argc, char** argv) {
  Dtype dt = Dtype::F16;
  uint32_t m = 4096, n = 4096, k = 4096;
  int reps = 10, rounds = 5, samples = 4096;
  std::vector<SideSpec> sides;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&]() -> const char* {
      if (i + 1 >= argc) die("%s needs a value", a.c_str());
      return argv[++i];
    };
    if (a == "--dtype") {
      dt = std::string(next()) == "bf16" ? Dtype::BF16 : Dtype::F16;
    } else if (a == "--mnk") {
      if (std::sscanf(next(), "%u,%u,%u", &m, &n, &k) != 3) die("--mnk m,n,k");
    } else if (a == "--reps") {
      reps = std::atoi(next());
    } else if (a == "--rounds") {
      rounds = std::atoi(next());
    } else if (a == "--samples") {
      samples = std::atoi(next());
    } else if (a == "--side") {
      sides.push_back(parse_side(next()));
    } else {
      die("unknown argument %s", a.c_str());
    }
  }
  if (sides.empty()) die("no --side given");
  // All sides read the same buffers, so they must agree on transposes.
  const uint32_t layout = sides[0].flags & 5u;
  for (const auto& s : sides) {
    if ((s.flags & 5u) != layout) die("sides disagree on transposes");
  }
  const bool a_t = (layout & 4u) != 0u;
  const bool b_t = (layout & 1u) != 0u;

  g_vk.handle = dlopen(LIBVK, RTLD_NOW | RTLD_LOCAL);
  if (!g_vk.handle) die("dlopen %s: %s", LIBVK, dlerror());
  g_vk.GetInstanceProcAddr =
      (PFN_vkGetInstanceProcAddr)dlsym(g_vk.handle, "vkGetInstanceProcAddr");
  g_vk.GetDeviceProcAddr =
      (PFN_vkGetDeviceProcAddr)dlsym(g_vk.handle, "vkGetDeviceProcAddr");
  if (!g_vk.GetInstanceProcAddr || !g_vk.GetDeviceProcAddr)
    die("vulkan entry points missing");
  g_vk.CreateInstance = (PFN_vkCreateInstance)g_vk.GetInstanceProcAddr(
      nullptr, "vkCreateInstance");
  if (!g_vk.CreateInstance) die("vkCreateInstance missing");
  DeviceCtx c = setup_device();

  for (auto& s : sides) {
    std::string spv = "/tmp/gemm-bench." + std::to_string(getpid()) + "." +
        s.name + ".spv";
    if (compile_shader(s.path.c_str(), s.defines.c_str(), spv.c_str()) != 0)
      die("compile %s", s.path.c_str());
    std::string code = read_file(spv.c_str());
    unlink(spv.c_str());
    VkShaderModuleCreateInfo mi{};
    mi.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    mi.codeSize = code.size();
    mi.pCode = (const uint32_t*)code.data();
    if (g_vk.CreateShaderModule(g_vk.dev, &mi, nullptr, &s.side.mod) !=
        VK_SUCCESS)
      die("CreateShaderModule %s", s.name.c_str());
    make_pipeline(c, s.side);
  }

  VkCommandPoolCreateInfo cpi{};
  cpi.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  cpi.queueFamilyIndex = c.qfi;
  VkCommandPool pool;
  if (g_vk.CreateCommandPool(g_vk.dev, &cpi, nullptr, &pool) != VK_SUCCESS)
    die("CreateCommandPool");
  VkCommandBufferAllocateInfo cbai{};
  cbai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  cbai.commandPool = pool;
  cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  cbai.commandBufferCount = 1;
  VkCommandBuffer cmd;
  if (g_vk.AllocateCommandBuffers(g_vk.dev, &cbai, &cmd) != VK_SUCCESS)
    die("AllocateCommandBuffers");
  VkFenceCreateInfo fci{};
  fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  VkFence fence;
  if (g_vk.CreateFence(g_vk.dev, &fci, nullptr, &fence) != VK_SUCCESS)
    die("CreateFence");
  VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      kBindings * (uint32_t)sides.size()};
  VkDescriptorPoolCreateInfo dpci{};
  dpci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  dpci.maxSets = (uint32_t)sides.size();
  dpci.poolSizeCount = 1;
  dpci.pPoolSizes = &ps;
  VkDescriptorPool dpool;
  if (g_vk.CreateDescriptorPool(g_vk.dev, &dpci, nullptr, &dpool) !=
      VK_SUCCESS)
    die("CreateDescriptorPool");

  // Operands: N(0, 1) rounded onto the storage grid.
  const size_t a_elems = (size_t)m * k, b_elems = (size_t)k * n;
  const size_t out_elems = (size_t)m * n;
  Buf a_buf = make_buf(g_vk.dev, c.mp, a_elems * 2);
  Buf b_buf = make_buf(g_vk.dev, c.mp, b_elems * 2);
  Buf c_buf = make_buf(g_vk.dev, c.mp, 16);
  std::mt19937 rng(0xC0FFEEu);
  std::normal_distribution<double> nd(0.0, 1.0);
  uint16_t* ah = (uint16_t*)a_buf.mapped;
  uint16_t* bh = (uint16_t*)b_buf.mapped;
  for (size_t i = 0; i < a_elems; ++i) ah[i] = round_store(dt, nd(rng));
  for (size_t i = 0; i < b_elems; ++i) bh[i] = round_store(dt, nd(rng));
  // The same operands widened exactly to f32, bound for sides whose
  // defines carry -DOPERAND_F32 (kernels that read pre-widened inputs).
  Buf a32_buf = make_buf(g_vk.dev, c.mp, a_elems * 4);
  Buf b32_buf = make_buf(g_vk.dev, c.mp, b_elems * 4);
  for (size_t i = 0; i < a_elems; ++i)
    ((float*)a32_buf.mapped)[i] = (float)to_f64(dt, ah[i]);
  for (size_t i = 0; i < b_elems; ++i)
    ((float*)b32_buf.mapped)[i] = (float)to_f64(dt, bh[i]);

  Params p{};
  p.count = m * n;
  p.output_size = m * n;
  p.lhs_size = m * k;
  p.rhs_size = k * n;
  p.reduce_size = k;
  p.matrix_m = m;
  p.matrix_n = n;
  p.matrix_k = k;
  p.alpha = 1.0f;
  p.lhs_gap = a_t ? m : k;
  p.rhs_gap = b_t ? k : n;

  auto begin = [&]() {
    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    g_vk.BeginCommandBuffer(cmd, &bi);
  };
  auto record = [&](SideSpec& s, int count) {
    Params sp = p;
    sp.flags = s.flags;
    g_vk.CmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, s.side.pipe);
    g_vk.CmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
        s.side.layout, 0, 1, &s.set, 0, nullptr);
    g_vk.CmdPushConstants(cmd, s.side.layout, VK_SHADER_STAGE_COMPUTE_BIT,
        0, sizeof(Params), &sp);
    for (int i = 0; i < count; ++i) {
      g_vk.CmdDispatch(cmd, group_count(n, s.tile_n),
          group_count(m, s.tile_m), 1);
    }
  };
  auto submit = [&]() {
    g_vk.ResetFences(g_vk.dev, 1, &fence);
    VkSubmitInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;
    auto t0 = std::chrono::steady_clock::now();
    if (g_vk.QueueSubmit(c.queue, 1, &si, fence) != VK_SUCCESS)
      die("QueueSubmit");
    if (g_vk.WaitForFences(g_vk.dev, 1, &fence, VK_TRUE, 60'000'000'000ull) !=
        VK_SUCCESS)
      die("fence timeout");
    auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::micro>(t1 - t0).count();
  };

  for (auto& s : sides) {
    s.out = make_buf(g_vk.dev, c.mp, out_elems * 2);
    std::memset(s.out.mapped, 0xff, out_elems * 2);
    bool wide = s.defines.find("-DOPERAND_F32") != std::string::npos;
    s.set = make_set(c, dpool, s.side.dsl,
        SetBufs{wide ? a32_buf : a_buf, wide ? b32_buf : b_buf, c_buf, s.out});
    begin();
    record(s, 1);
    g_vk.EndCommandBuffer(cmd);
    submit();
    s.result.assign((const uint16_t*)s.out.mapped,
        (const uint16_t*)s.out.mapped + out_elems);
  }

  // float64 truth on sampled outputs (every output when samples covers
  // the problem).
  std::vector<size_t> picks;
  if ((size_t)samples >= out_elems) {
    for (size_t i = 0; i < out_elems; ++i) picks.push_back(i);
  } else {
    std::mt19937_64 prng(0x5EEDu);
    for (int i = 0; i < samples; ++i) picks.push_back(prng() % out_elems);
  }
  std::vector<double> truth(picks.size());
  {
    unsigned threads = std::max(1u, std::thread::hardware_concurrency());
    std::vector<std::thread> workers;
    for (unsigned t = 0; t < threads; ++t) {
      workers.emplace_back([&, t]() {
        for (size_t s = t; s < picks.size(); s += threads) {
          size_t r = picks[s] / n, col = picks[s] % n;
          double acc = 0.0;
          for (uint32_t kk = 0; kk < k; ++kk) {
            double av = to_f64(dt, a_t ? ah[(size_t)kk * m + r]
                                       : ah[r * k + kk]);
            double bv = to_f64(dt, b_t ? bh[col * k + kk]
                                       : bh[(size_t)kk * n + col]);
            acc += av * bv;
          }
          truth[s] = acc;
        }
      });
    }
    for (auto& th : workers) th.join();
  }
  std::printf("{\"k\":\"problem\",\"dtype\":\"%s\",\"m\":%u,\"n\":%u,"
      "\"kdim\":%u,\"a_t\":%d,\"b_t\":%d,\"samples\":%zu}\n",
      dt == Dtype::F16 ? "f16" : "bf16", m, n, k, a_t, b_t, picks.size());
  for (auto& s : sides) {
    size_t exact = 0, mismatch_first = 0, nonfinite = 0;
    double sum_ulp = 0.0, max_ulp = 0.0;
    for (size_t i = 0; i < picks.size(); ++i) {
      uint16_t got = s.result[picks[i]];
      double gv = to_f64(dt, got);
      if (!std::isfinite(gv)) {
        ++nonfinite;
        continue;
      }
      if (got == round_store(dt, truth[i])) ++exact;
      double e = std::fabs(gv - truth[i]) / ulp_at(dt, truth[i]);
      sum_ulp += e;
      max_ulp = std::max(max_ulp, e);
    }
    for (size_t i = 0; i < out_elems; ++i) {
      if (s.result[i] != sides[0].result[i]) ++mismatch_first;
    }
    std::printf("{\"k\":\"accuracy\",\"side\":\"%s\",\"exact_rounded\":%.5f,"
        "\"mean_ulp\":%.4f,\"max_ulp\":%.3f,\"nonfinite\":%zu,"
        "\"bits_vs_%s\":%zu}\n",
        s.name.c_str(), (double)exact / picks.size(),
        sum_ulp / picks.size(), max_ulp, nonfinite, sides[0].name.c_str(),
        mismatch_first);
  }

  // Timing: one warm-up submission per side, then the sides interleaved
  // round by round.
  for (auto& s : sides) {
    begin();
    record(s, 2);
    g_vk.EndCommandBuffer(cmd);
    submit();
  }
  for (int round = 0; round < rounds; ++round) {
    for (auto& s : sides) {
      begin();
      record(s, reps);
      g_vk.EndCommandBuffer(cmd);
      s.us.push_back(submit() / reps);
    }
  }
  const double flops = 2.0 * m * n * k;
  for (auto& s : sides) {
    std::vector<double> v = s.us;
    std::sort(v.begin(), v.end());
    double med = v[v.size() / 2];
    std::printf("{\"k\":\"time\",\"side\":\"%s\",\"median_us\":%.1f,"
        "\"min_us\":%.1f,\"max_us\":%.1f,\"tflops\":%.3f,"
        "\"tflops_best\":%.3f}\n",
        s.name.c_str(), med, v.front(), v.back(), flops / (med * 1e6),
        flops / (v.front() * 1e6));
  }
  return 0;
}
