// Q4 decode GEMV bandwidth bench for the M1 (Honeykrisp) and lavapipe
// correctness screening. It times the REAL production shader
// (shaders/qmm_vec_base.comp, a frozen copy of
// overlay/mlx/backend/omarchy/shaders/qmm_vec.comp) and a candidate
// edit (shaders/qmm_vec_cand.comp) on the four per-layer dispatch
// shapes the Qwen2.5-0.5B decode chain issues through
// QmmVecQ4MultiSubgroupF16:
//   qkv     dims=3  n=(896,128,128)  k=896  grid=144
//   o       dims=1  n=(896,)         k=896  grid=112
//   gate_up dims=2  n=(4864,4864)    k=896  grid=1216
//   down    dims=1  n=(896,)         k=4864 grid=112
// with the push-constant and binding contract of
// dispatch_quantized_gemv_group (primitives.cpp): operation=4,
// reduce_size=64, matrix_m=1, per-weight shape[i], grid =
// sum(ceil(n_i/8)), bindings 0 = x then 6 per weight slot.
//
// Measurements:
//   * streaming copy and read probes over 256 MB: the device's
//     achievable bandwidth ceiling.
//   * isolated per-shape dispatch, timestamped per submit (base and
//     candidate alternate within each repetition to cancel drift).
//   * the four-dispatch layer chain in ONE command buffer with a
//     timestamp after each dispatch (what one decode token issues per
//     layer), two interleaved rounds.
//   * bit-exactness: identical input buffers through base and
//     candidate into distinct outputs, compared as raw storage bits -
//     must be 0 mismatches per shape.
//
// Modes: default compiles the production subgroup variant
// (-DUSE_SUBGROUP=1, requires subgroupSize == 32 - the M1); "--tree"
// compiles the tree-reduction variants for lavapipe screening
// (lavapipe subgroupSize != 32 would make the subgroup variant wrong
// by design; llvmpipe dispatches the tree variant).
//
// Build: g++ -std=c++17 -O2 -o /tmp/q4-bw-bench tools/q4-bw-bench/bench.cpp
// Run (repo root): /tmp/q4-bw-bench [--tree] [--quick]
// Output: NDJSON on stdout.

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
#include <unistd.h>
#include <vector>

#define LIBVK "libvulkan.so.1"

struct VkTable {
  void* handle{nullptr};
  VkInstance inst{VK_NULL_HANDLE};
  VkDevice dev{VK_NULL_HANDLE};
#define VK_FN(name) PFN_vk##name name{nullptr}
  VK_FN(GetInstanceProcAddr);
  VK_FN(GetDeviceProcAddr);
  VK_FN(CreateInstance);
  VK_FN(EnumerateInstanceExtensionProperties);
  VK_FN(EnumerateInstanceLayerProperties);
  VK_FN(DestroyInstance);
  VK_FN(EnumeratePhysicalDevices);
  VK_FN(GetPhysicalDeviceProperties);
  VK_FN(GetPhysicalDeviceProperties2);
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
  VK_FN(DestroyFence);
  VK_FN(WaitForFences);
  VK_FN(CreateQueryPool);
  VK_FN(CmdResetQueryPool);
  VK_FN(CmdWriteTimestamp);
  VK_FN(GetQueryPoolResults);
  VK_FN(QueueSubmit);
  VK_FN(DestroyShaderModule);
  VK_FN(DestroyPipeline);
  VK_FN(DestroyPipelineLayout);
  VK_FN(DestroyDescriptorSetLayout);
  VK_FN(DestroyDescriptorPool);
  VK_FN(DestroyCommandPool);
  VK_FN(DestroyBuffer);
  VK_FN(DestroyDevice);
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

static int vk_init() {
  g_vk.handle = dlopen(LIBVK, RTLD_NOW | RTLD_LOCAL);
  if (!g_vk.handle) {
    std::fprintf(stderr, "dlopen %s: %s\n", LIBVK, dlerror());
    return -1;
  }
  g_vk.GetInstanceProcAddr =
      (PFN_vkGetInstanceProcAddr)dlsym(g_vk.handle, "vkGetInstanceProcAddr");
  if (!g_vk.GetInstanceProcAddr) {
    std::fprintf(stderr, "vkGetInstanceProcAddr missing\n");
    return -1;
  }
  g_vk.CreateInstance =
      (PFN_vkCreateInstance)vk_load(nullptr, "vkCreateInstance");
  g_vk.EnumerateInstanceExtensionProperties =
      (PFN_vkEnumerateInstanceExtensionProperties)vk_load(
          nullptr, "vkEnumerateInstanceExtensionProperties");
  if (!g_vk.CreateInstance) {
    std::fprintf(stderr, "vkCreateInstance missing\n");
    return -1;
  }
  g_vk.GetDeviceProcAddr = (PFN_vkGetDeviceProcAddr)dlsym(
      g_vk.handle, "vkGetDeviceProcAddr");
  if (!g_vk.GetDeviceProcAddr) {
    std::fprintf(stderr, "vkGetDeviceProcAddr missing\n");
    return -1;
  }
  return 0;
}

static void vk_load_instance() {
  LOAD(DestroyInstance);
  LOAD(EnumeratePhysicalDevices);
  LOAD(GetPhysicalDeviceProperties);
  LOAD(GetPhysicalDeviceProperties2);
  LOAD(GetPhysicalDeviceMemoryProperties);
  LOAD(GetPhysicalDeviceQueueFamilyProperties);
  LOAD(CreateDevice);
}

static void vk_load_device() {
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
  LOAD_DEV(DestroyFence);
  LOAD_DEV(WaitForFences);
  LOAD_DEV(CreateQueryPool);
  LOAD_DEV(CmdResetQueryPool);
  LOAD_DEV(CmdWriteTimestamp);
  LOAD_DEV(GetQueryPoolResults);
  LOAD_DEV(QueueSubmit);
  LOAD_DEV(DestroyShaderModule);
  LOAD_DEV(DestroyPipeline);
  LOAD_DEV(DestroyPipelineLayout);
  LOAD_DEV(DestroyDescriptorSetLayout);
  LOAD_DEV(DestroyDescriptorPool);
  LOAD_DEV(DestroyCommandPool);
  LOAD_DEV(DestroyBuffer);
  LOAD_DEV(DestroyDevice);
}

#undef LOAD
#undef LOAD_DEV

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

static bool have_tool(const char* tool) {
  std::string cmd = std::string("which ") + tool + " >/dev/null 2>&1";
  return system(cmd.c_str()) == 0;
}

// Compile src with the given defines into out_spv, matching the repo's
// omarchy_shader (glslc preferred; vulkan1.3 target required for the
// subgroup extensions).
static int compile_shader(const char* src, const char* defines,
    const char* out_spv) {
  // glslc preferred (repo rule where present), but a broken glslc shim
  // must not kill the bench: fall back to glslangValidator.
  const char* front = have_tool("glslc")
      ? "glslc -fshader-stage=compute --target-env=vulkan1.3 "
      : "glslangValidator -V --target-env vulkan1.3 ";
  for (const char* front_try : {front,
           "glslangValidator -V --target-env vulkan1.3 "}) {
    std::string cmd = front_try;
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
    VkDeviceSize size, bool zero) {
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
  if (zero) return b;
  // Deterministic fill. As f16 halves the pattern lands in
  // [2^-4, 2^-1): never NaN/Inf, so every dot chain stays finite; as
  // packed uint32 weight words the nibbles are plain 0..15 values.
  void* p;
  g_vk.MapMemory(dev, b.mem, 0, size, 0, &p);
  std::mt19937 rng(0xC0FFEEu ^ (uint32_t)(size * 2654435761ull));
  uint16_t* h = (uint16_t*)p;
  for (VkDeviceSize i = 0; i < size / sizeof(uint16_t); ++i) {
    h[i] = (uint16_t)(0x3000u | (rng() & 0x3FFu));
  }
  g_vk.UnmapMemory(dev, b.mem);
  return b;
}

// Push constants: must byte-match the shader Params block (30 x
// 4-byte words, scalar alignment throughout).
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

static constexpr uint32_t kBindings = 19; // x + 3 weight slots * 6
static constexpr uint32_t kMaxDims = 3;

struct Shape {
  const char* name;
  uint32_t k;
  uint32_t dims;
  uint32_t n[kMaxDims];
  // Per-token shapes (lm_head) run once per token, not once per layer:
  // the gap arms cycle per-layer sets x24, which cannot afford 24 copies
  // of a 286 MB weight, so they skip these (iso mode still benches them;
  // their streaming rate is honest at any rep count because the weight
  // never fits cache).
  bool per_token{false};
};

// Shape tables, selected by --2b. The 0.5B table is the historical
// Qwen2.5-0.5B decode chain. The 2B table is the Qwen3.8-2B decode
// chain per full-attention / GDN layer (hidden 2048, inter 6144,
// head_dim 256, 8 q heads / 2 kv heads, GDN qkvz 8192), cross-checked
// against the checkpoint tensor shapes (qwen3_5: q_proj carries the
// attn_output_gate half, 4096 rows) and the decode profile families:
//   qkv      dims=3  n=(4096,512,512)  k=2048  (full attn, multi)
//   gate_up  dims=2  n=(6144,6144)     k=2048  (in-model gate/up also
//            dispatch solo as the 6144-row shape below)
//   down     dims=1  n=(2048,)         k=6144
//   qkvz     dims=1  n=(8192,)         k=2048  (GDN in_proj; in-model
//            the 32-wide ba projection groups with it, same k)
//   gate6144 dims=1  n=(6144,)         k=2048  (in_proj_qkv, gate, up
//            dispatch solo per the decode profile: n=6144 gx=768)
//   zout     dims=1  n=(2048,)         k=2048  (in_proj_z, out_proj, o)
//   q4096    dims=1  n=(4096,)         k=2048  (attn q incl. gate half)
//   kv512    dims=1  n=(512,)          k=2048  (attn k and v)
//   ab       dims=2  n=(16,16)         k=2048  (GDN in_proj_a/b)
//   lm_head  dims=1  n=(248320,)       k=2048  (26% of token bytes)
static const Shape kShapes[] = {
    {"qkv", 896, 3, {896, 128, 128}},
    {"o", 896, 1, {896, 0, 0}},
    {"gate_up", 896, 2, {4864, 4864, 0}},
    {"down", 4864, 1, {896, 0, 0}},
};
static const Shape kShapes2B[] = {
    {"qkv", 2048, 3, {4096, 512, 512}},
    {"gate_up", 2048, 2, {6144, 6144, 0}},
    {"down", 6144, 1, {2048, 0, 0}},
    {"qkvz", 2048, 1, {8192, 0, 0}},
    {"gate6144", 2048, 1, {6144, 0, 0}},
    {"zout", 2048, 1, {2048, 0, 0}},
    {"q4096", 2048, 1, {4096, 0, 0}},
    {"kv512", 2048, 1, {512, 0, 0}},
    {"ab", 2048, 2, {16, 16, 0}},
    {"lm_head", 2048, 1, {248320, 0, 0}, true},
};
static const Shape* g_shapes = kShapes;
// --cand <path>[:columns]: screen ONE candidate shader against base (default
// mode: iso timing + bit compare; gap mode: iso4/widedep arms) instead of the
// historical fixed candidate list. columns = rows per workgroup (grid = sum
// ceil(n_i / columns)); default 8 = the production COLUMNS_PER_GROUP.
static std::string g_cand_src;
static uint32_t g_cand_columns = 8u;
// --cand-def "-DX=1 ...": extra compile defines for the candidate only.
static std::string g_cand_defs;
// --grid-cap N: dispatch at most N workgroups per candidate GEMV (gap mode);
// params.count keeps the logical group count for grid-stride shaders.
static uint32_t g_grid_cap = 0u;
// --2b also switches the gap-mode compile to the production bf16 x-load
// mix (the 2B decode chain runs QmmVecQ4*BF16, not FP16).
static bool g_gap_bf16 = false;
// Sized for the 2B table (10 entries); the 0.5B table uses the first 4.
static constexpr uint32_t kNumShapes = 10;
static constexpr uint32_t kNumShapes05B = 4;
// Active table size: loops must never read past the selected table.
static uint32_t g_num_shapes = kNumShapes05B;

// Bytes one dispatch touches: weight words + scales + biases + x row +
// outputs (flags are 0 in the bench, so no addend reads / sum writes).
static uint64_t shape_bytes(const Shape& s) {
  uint64_t words = 0, params = 0, outs = 0;
  for (uint32_t i = 0; i < s.dims; ++i) {
    words += (uint64_t)s.n[i] * (s.k / 8u) * 4u;
    params += 2ull * s.n[i] * (s.k / 64u) * 2u;
    outs += s.n[i] * 2u;
  }
  return words + params + (uint64_t)s.k * 2u + outs;
}

static uint32_t shape_groups(const Shape& s, uint32_t columns) {
  uint32_t total = 0;
  for (uint32_t i = 0; i < s.dims; ++i) {
    total += (s.n[i] + columns - 1u) / columns;
  }
  return total;
}

struct DeviceCtx {
  VkQueue queue{VK_NULL_HANDLE};
  uint32_t qfi{0};
  VkPhysicalDeviceMemoryProperties mp{};
  uint32_t timestampPeriod{0};
  uint32_t subgroupSize{0};
  bool subgroupArith{false};
  std::string name;
};

static DeviceCtx setup_device() {
  VkApplicationInfo ai{};
  ai.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  ai.pApplicationName = "q4-bw-bench";
  ai.apiVersion = VK_API_VERSION_1_2;
  VkInstanceCreateInfo ici{};
  ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  ici.pApplicationInfo = &ai;
  if (g_vk.CreateInstance(&ici, nullptr, &g_vk.inst) != VK_SUCCESS)
    die("CreateInstance");
  vk_load_instance();

  uint32_t n = 0;
  g_vk.EnumeratePhysicalDevices(g_vk.inst, &n, nullptr);
  if (n == 0) die("no Vulkan physical devices");
  std::vector<VkPhysicalDevice> pds(n);
  g_vk.EnumeratePhysicalDevices(g_vk.inst, &n, pds.data());
  DeviceCtx c;
  for (uint32_t i = 0; i < n; ++i) {
    VkPhysicalDeviceProperties props{};
    g_vk.GetPhysicalDeviceProperties(pds[i], &props);
    if (props.apiVersion < VK_API_VERSION_1_2) continue;
    VkPhysicalDeviceSubgroupProperties sub{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};
    VkPhysicalDeviceProperties2 props2{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    props2.pNext = &sub;
    g_vk.GetPhysicalDeviceProperties2(pds[i], &props2);
    uint32_t qfn = 0;
    g_vk.GetPhysicalDeviceQueueFamilyProperties(pds[i], &qfn, nullptr);
    std::vector<VkQueueFamilyProperties> qfpv(qfn);
    g_vk.GetPhysicalDeviceQueueFamilyProperties(pds[i], &qfn, qfpv.data());
    for (uint32_t q = 0; q < qfn; ++q) {
      if ((qfpv[q].queueFlags & VK_QUEUE_COMPUTE_BIT) == 0) continue;
      c.qfi = q;
      c.subgroupSize = sub.subgroupSize;
      c.subgroupArith =
          (sub.supportedOperations & VK_SUBGROUP_FEATURE_ARITHMETIC_BIT) != 0;
      c.timestampPeriod = props.limits.timestampPeriod;
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
      VkPhysicalDeviceVulkan12Features f12{
          VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
      f12.shaderFloat16 = VK_TRUE;
      dci.pNext = &f12;
      if (g_vk.CreateDevice(pds[i], &dci, nullptr, &g_vk.dev) != VK_SUCCESS)
        die("CreateDevice");
      vk_load_device();
      g_vk.GetDeviceQueue(g_vk.dev, q, 0, &c.queue);
      std::printf(
          "{\"k\":\"dev\",\"name\":\"%s\",\"subgroupSize\":%u,"
          "\"arith\":%s,\"ts_period_ns\":%.3f}\n",
          props.deviceName, c.subgroupSize, c.subgroupArith ? "true" : "false",
          c.timestampPeriod / 1.0);
      return c;
    }
  }
  die("no Vulkan 1.2 compute device");
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

struct Side {
  std::string tag;
  VkShaderModule mod{VK_NULL_HANDLE};
  VkDescriptorSetLayout dsl{VK_NULL_HANDLE};
  VkPipelineLayout layout{VK_NULL_HANDLE};
  VkPipeline pipe{VK_NULL_HANDLE};
};

static void make_pipeline(Side& side) {
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
  if (g_vk.CreateComputePipelines(g_vk.dev, VK_NULL_HANDLE, 1, &cpci,
          nullptr, &side.pipe) != VK_SUCCESS)
    die("CreateComputePipelines");
}

struct SetBufs {
  Buf x;
  Buf w[kMaxDims];
  Buf scales[kMaxDims];
  Buf biases[kMaxDims];
  Buf out[kMaxDims];
};

static VkDescriptorSet make_set(VkDescriptorPool pool, VkDescriptorSetLayout dsl,
    const SetBufs& bufs, uint32_t dims) {
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
  auto info = [&](uint32_t i, Buf& buf) {
    dbi[i].buffer = buf.buf;
    dbi[i].offset = 0;
    dbi[i].range = VK_WHOLE_SIZE;
  };
  info(0, const_cast<SetBufs&>(bufs).x);
  for (uint32_t i = 0; i < kMaxDims; ++i) {
    uint32_t base = 1 + i * 6;
    if (i < dims) {
      info(base, const_cast<SetBufs&>(bufs).w[i]);
      info(base + 1, const_cast<SetBufs&>(bufs).scales[i]);
      info(base + 2, const_cast<SetBufs&>(bufs).biases[i]);
      info(base + 3, const_cast<SetBufs&>(bufs).out[i]);
      info(base + 4, const_cast<SetBufs&>(bufs).out[i]); // addend filler
      info(base + 5, const_cast<SetBufs&>(bufs).out[i]); // sum filler
    } else {
      for (uint32_t j = 0; j < 6; ++j) {
        info(base + j, const_cast<SetBufs&>(bufs).out[0]);
      }
    }
  }
  for (uint32_t i = 0; i < kBindings; ++i) {
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

static void fill_params(Params& p, const Shape& s, uint32_t groups) {
  std::memset(&p, 0, sizeof(p));
  p.count = groups;
  p.operation = 4u;
  p.reduce_size = 64u;
  p.matrix_m = 1u;
  p.matrix_k = s.k;
  p.dims = s.dims;
  for (uint32_t i = 0; i < s.dims; ++i) p.shape[i] = s.n[i];
}

struct CmdRes {
  VkCommandPool pool{VK_NULL_HANDLE};
  VkCommandBuffer cmd{VK_NULL_HANDLE};
  VkQueryPool qpool{VK_NULL_HANDLE};
  uint32_t queries{0};
};

static CmdRes make_cmd(const DeviceCtx& ctx, uint32_t queries) {
  CmdRes r;
  r.queries = queries;
  VkCommandPoolCreateInfo cpci{};
  cpci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  cpci.queueFamilyIndex = ctx.qfi;
  if (g_vk.CreateCommandPool(g_vk.dev, &cpci, nullptr, &r.pool) !=
      VK_SUCCESS)
    die("CreateCommandPool");
  VkCommandBufferAllocateInfo cbai{};
  cbai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  cbai.commandPool = r.pool;
  cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  cbai.commandBufferCount = 1;
  if (g_vk.AllocateCommandBuffers(g_vk.dev, &cbai, &r.cmd) != VK_SUCCESS)
    die("AllocateCommandBuffers");
  VkQueryPoolCreateInfo qpci{};
  qpci.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
  qpci.queryType = VK_QUERY_TYPE_TIMESTAMP;
  qpci.queryCount = queries;
  if (g_vk.CreateQueryPool(g_vk.dev, &qpci, nullptr, &r.qpool) !=
      VK_SUCCESS)
    die("CreateQueryPool");
  return r;
}

static uint64_t g_wall_ns = 0;

// Host CLOCK_MONOTONIC bracket around the whole submit (queue submit +
// fence wait). This is the only trustworthy timing instrument on this
// driver (receipts/2026-09-10-dispatch-floor,
// receipts/2026-09-11-decode-gap): every wall number this bench prints
// comes from here.
static VkFence submit_and_wait(const DeviceCtx& ctx, VkCommandBuffer cmd) {
  VkSubmitInfo si{};
  si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  si.commandBufferCount = 1;
  si.pCommandBuffers = &cmd;
  VkFence fence;
  VkFenceCreateInfo fci{};
  fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  if (g_vk.CreateFence(g_vk.dev, &fci, nullptr, &fence) != VK_SUCCESS)
    die("CreateFence");
  auto wall_t0 = std::chrono::steady_clock::now();
  if (g_vk.QueueSubmit(ctx.queue, 1, &si, fence) != VK_SUCCESS)
    die("QueueSubmit");
  if (g_vk.WaitForFences(g_vk.dev, 1, &fence, VK_TRUE, UINT64_MAX) !=
      VK_SUCCESS)
    die("WaitForFences");
  auto wall_t1 = std::chrono::steady_clock::now();
  g_wall_ns = (uint64_t)std::chrono::duration_cast<
      std::chrono::nanoseconds>(wall_t1 - wall_t0)
                  .count();
  g_vk.DestroyFence(g_vk.dev, fence, nullptr);
  return VK_NULL_HANDLE;
}

static void begin(CmdRes& r) {
  VkCommandBufferBeginInfo bi{};
  bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  g_vk.BeginCommandBuffer(r.cmd, &bi);
  g_vk.CmdResetQueryPool(r.cmd, r.qpool, 0, r.queries);
}

static uint64_t read_ticks(const DeviceCtx& ctx, CmdRes& r, uint32_t count,
    uint64_t* ticks) {
  if (g_vk.GetQueryPoolResults(g_vk.dev, r.qpool, 0, count,
          count * sizeof(uint64_t), ticks, sizeof(uint64_t),
          VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT) != VK_SUCCESS)
    die("GetQueryPoolResults");
  return ticks[0];
}

// One isolated dispatch per submit; returns gpu ns across the dispatch.
static uint64_t dispatch_isolated(const DeviceCtx& ctx, const Side& side,
    CmdRes& r, VkDescriptorSet set, const Params& params, uint32_t groups) {
  begin(r);
  g_vk.CmdWriteTimestamp(r.cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, r.qpool, 0);
  g_vk.CmdBindPipeline(r.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, side.pipe);
  g_vk.CmdBindDescriptorSets(r.cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
      side.layout, 0, 1, &set, 0, nullptr);
  g_vk.CmdPushConstants(r.cmd, side.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
      sizeof(Params), &params);
  g_vk.CmdDispatch(r.cmd, groups, 1, 1);
  g_vk.CmdWriteTimestamp(r.cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, r.qpool, 1);
  g_vk.EndCommandBuffer(r.cmd);
  submit_and_wait(ctx, r.cmd);
  uint64_t ticks[2] = {0, 0};
  read_ticks(ctx, r, 2, ticks);
  return (uint64_t)((double)(ticks[1] - ticks[0]) * ctx.timestampPeriod);
}


static uint64_t median(std::vector<uint64_t> v) {
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

static uint64_t min_of(const std::vector<uint64_t>& v) {
  return *std::min_element(v.begin(), v.end());
}

struct PeakPipe {
  VkShaderModule mod{VK_NULL_HANDLE};
  VkDescriptorSetLayout dsl{VK_NULL_HANDLE};
  VkPipelineLayout layout{VK_NULL_HANDLE};
  VkPipeline pipe{VK_NULL_HANDLE};
};

// Streaming probe: 2 bindings, a push-constant-free pipeline, one
// dispatch over uvec4_count elements.
static void run_peak(const DeviceCtx& ctx, const char* src_path,
    const char* spv, const char* tag, uint32_t uvec4_count, bool skip) {
  if (compile_shader(src_path, "", spv) != 0) die("compile %s", tag);
  PeakPipe p;
  p.mod = make_module(read_file(spv));
  VkDescriptorSetLayoutBinding b[2]{};
  for (uint32_t i = 0; i < 2; ++i) {
    b[i].binding = i;
    b[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    b[i].descriptorCount = 1;
    b[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  }
  VkDescriptorSetLayoutCreateInfo dslci{};
  dslci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  dslci.bindingCount = 2;
  dslci.pBindings = b;
  if (g_vk.CreateDescriptorSetLayout(g_vk.dev, &dslci, nullptr, &p.dsl) !=
      VK_SUCCESS)
    die("CreateDescriptorSetLayout peak");
  VkPipelineLayoutCreateInfo plci{};
  plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  plci.setLayoutCount = 1;
  plci.pSetLayouts = &p.dsl;
  if (g_vk.CreatePipelineLayout(g_vk.dev, &plci, nullptr, &p.layout) !=
      VK_SUCCESS)
    die("CreatePipelineLayout peak");
  VkComputePipelineCreateInfo cpci{};
  cpci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
  cpci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  cpci.stage.module = p.mod;
  cpci.stage.pName = "main";
  cpci.layout = p.layout;
  if (g_vk.CreateComputePipelines(g_vk.dev, VK_NULL_HANDLE, 1, &cpci,
          nullptr, &p.pipe) != VK_SUCCESS)
    die("CreateComputePipelines peak");

  Buf a = make_buf(g_vk.dev, ctx.mp, (uint64_t)uvec4_count * 16u, false);
  Buf dst = make_buf(g_vk.dev, ctx.mp, (uint64_t)uvec4_count * 16u, true);
  VkDescriptorPoolSize ps{};
  ps.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  ps.descriptorCount = 2;
  VkDescriptorPoolCreateInfo dpci{};
  dpci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  dpci.maxSets = 1;
  dpci.poolSizeCount = 1;
  dpci.pPoolSizes = &ps;
  VkDescriptorPool pool;
  if (g_vk.CreateDescriptorPool(g_vk.dev, &dpci, nullptr, &pool) !=
      VK_SUCCESS)
    die("CreateDescriptorPool peak");
  VkDescriptorSetAllocateInfo dsai{};
  dsai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  dsai.descriptorPool = pool;
  dsai.descriptorSetCount = 1;
  dsai.pSetLayouts = &p.dsl;
  VkDescriptorSet set;
  if (g_vk.AllocateDescriptorSets(g_vk.dev, &dsai, &set) != VK_SUCCESS)
    die("AllocateDescriptorSets peak");
  VkDescriptorBufferInfo dbi[2]{};
  dbi[0].buffer = a.buf;
  dbi[0].range = VK_WHOLE_SIZE;
  dbi[1].buffer = dst.buf;
  dbi[1].range = VK_WHOLE_SIZE;
  VkWriteDescriptorSet w[2]{};
  for (int i = 0; i < 2; ++i) {
    w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w[i].dstSet = set;
    w[i].dstBinding = i;
    w[i].descriptorCount = 1;
    w[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    w[i].pBufferInfo = &dbi[i];
  }
  g_vk.UpdateDescriptorSets(g_vk.dev, 2, w, 0, nullptr);

  CmdRes cmd = make_cmd(ctx, 2);
  // The M1 maxWorkGroupCount is 65535: a 1<<24 uvec4 probe over 256-wide
  // workgroups needs 65536, which is an invalid dispatch (honeykrisp
  // silently mis-executes it; that is where the historic 9-13 TB/s
  // "copy peak" came from). Clamp and account the real coverage.
  uint32_t groups = uvec4_count / 256u;
  if (groups > 65535u) groups = 65535u;
  std::vector<uint64_t> samples;
  std::vector<uint64_t> wall_samples;
  if (!skip) {
    for (int rep = 0; rep < 3; ++rep) {
      begin(cmd);
      g_vk.CmdWriteTimestamp(
          cmd.cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, cmd.qpool, 0);
      g_vk.CmdBindPipeline(
          cmd.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, p.pipe);
      g_vk.CmdBindDescriptorSets(cmd.cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
          p.layout, 0, 1, &set, 0, nullptr);
      g_vk.CmdDispatch(cmd.cmd, groups, 1, 1);
      g_vk.CmdWriteTimestamp(
          cmd.cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, cmd.qpool, 1);
      g_vk.EndCommandBuffer(cmd.cmd);
      submit_and_wait(ctx, cmd.cmd);
      uint64_t ticks[2] = {0, 0};
      read_ticks(ctx, cmd, 2, ticks);
      samples.push_back(
          (uint64_t)((double)(ticks[1] - ticks[0]) * ctx.timestampPeriod));
      // Wall-anchored repeat of the same dispatch: the copy roof the
      // kernel numbers are judged against, measured with the same
      // submit bracket (so per-submit overhead inflates both equally).
      begin(cmd);
      g_vk.CmdBindPipeline(
          cmd.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, p.pipe);
      g_vk.CmdBindDescriptorSets(cmd.cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
          p.layout, 0, 1, &set, 0, nullptr);
      g_vk.CmdDispatch(cmd.cmd, groups, 1, 1);
      g_vk.EndCommandBuffer(cmd.cmd);
      submit_and_wait(ctx, cmd.cmd);
      wall_samples.push_back(g_wall_ns);
    }
  }
  double bytes = (double)groups * 256.0 * 16.0 *
      (std::string(tag) == "copy" ? 2.0 : 1.0);
  if (samples.empty()) {
    std::printf("{\"k\":\"peak\",\"tag\":\"%s\",\"skipped\":true}\n", tag);
  } else {
    uint64_t med = median(samples);
    uint64_t wmed = median(wall_samples);
    std::printf(
        "{\"k\":\"peak\",\"tag\":\"%s\",\"bytes\":%llu,\"med_gpu_ns\":%llu,"
        "\"min_gpu_ns\":%llu,\"med_gb_s\":%.2f,\"min_gb_s\":%.2f,"
        "\"wall_med_ns\":%llu,\"wall_med_gb_s\":%.2f}\n",
        tag, (unsigned long long)bytes, (unsigned long long)med,
        (unsigned long long)min_of(samples), bytes / (double)med,
        bytes / (double)min_of(samples), (unsigned long long)wmed,
        bytes / (double)wmed);
  }
  g_vk.DestroyDescriptorPool(g_vk.dev, pool, nullptr);
  g_vk.DestroyBuffer(g_vk.dev, a.buf, nullptr);
  g_vk.FreeMemory(g_vk.dev, a.mem, nullptr);
  g_vk.DestroyBuffer(g_vk.dev, dst.buf, nullptr);
  g_vk.FreeMemory(g_vk.dev, dst.mem, nullptr);
}

// ---------------------------------------------------------------------------
// Decode gap mode (--gap): reproduce the in-model decode interleaving in
// isolation and move one between-dispatch factor at a time. All timing is
// host CLOCK_MONOTONIC around whole submits (tokens x sets x 4 dispatches
// per bracket) - per-dispatch timestamp brackets are forbidden as an
// instrument (receipts/2026-09-10-dispatch-floor: 21-23 us per bracket
// pair, untrustworthy device timestamp period).
//
// Arms (one factor each):
//   iso4      one weight set, the four layer dispatches, no chaining -
//             calibrates against the prior isolated instrument.
//   iso4dep   same but o/gate_up/down/next-token-qkv read the previous
//             dispatch's output buffer (RAW chain like the model).
//   fill2 / fill24   iso4 with the peak_read streaming probe pushed
//             between the GEMV dispatches: 2 MB/layer (real interstitial
//             traffic scale) vs 24 MB/layer (forced eviction).
//   churn     iso4 with fresh x/out allocations per round (address churn;
//             allocations happen between timed brackets like the host
//             allocator acts between tokens).
//   wideind   24 weight sets cycled per token, no chaining: the ~200 MB
//             working set cannot stay cache-resident.
//   widedep   24 sets cycled AND chained: the in-model analog.
//   sub2async / sub2sync   widedep as two submissions per token (sets
//             0-11, 12-23), queued without / with an inter-submit wait.
//   sweep     wideind at sets = 1,2,4,6,8,12,16,24: the residency
//             crossover curve.
struct GapShapeBufs {
  Buf x;
  Buf out[kMaxDims], w[kMaxDims], scales[kMaxDims], biases[kMaxDims];
};
struct GapSet {
  GapShapeBufs sh[kNumShapes];
};

static Buf gap_alloc(const DeviceCtx& ctx, uint64_t bytes) {
  return make_buf(g_vk.dev, ctx.mp, bytes, true);
}
static void gap_free(Buf& b) {
  if (b.buf != VK_NULL_HANDLE)
    g_vk.DestroyBuffer(g_vk.dev, b.buf, nullptr);
  if (b.mem != VK_NULL_HANDLE)
    g_vk.FreeMemory(g_vk.dev, b.mem, nullptr);
  b.buf = VK_NULL_HANDLE;
  b.mem = VK_NULL_HANDLE;
}
static void gap_free_shape(GapShapeBufs& g) {
  gap_free(g.x);
  for (uint32_t d = 0; d < kMaxDims; ++d) {
    gap_free(g.out[d]);
    gap_free(g.w[d]);
    gap_free(g.scales[d]);
    gap_free(g.biases[d]);
  }
}

// One descriptor set for shape sh_idx. x/out come from g; the weight
// slots come from wsrc (allows churn to pair fresh x/out with frozen
// weights). x_override replaces the x binding (RAW chaining).
static VkDescriptorSet gap_make_set(VkDescriptorPool pool,
    VkDescriptorSetLayout dsl, const GapShapeBufs& g,
    const GapShapeBufs& wsrc, uint32_t sh_idx, const Buf* x_override) {
  const Shape& sh = g_shapes[sh_idx];
  VkDescriptorSetAllocateInfo dsai{};
  dsai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  dsai.descriptorPool = pool;
  dsai.descriptorSetCount = 1;
  dsai.pSetLayouts = &dsl;
  VkDescriptorSet set;
  if (g_vk.AllocateDescriptorSets(g_vk.dev, &dsai, &set) != VK_SUCCESS)
    die("gap AllocateDescriptorSets");
  VkDescriptorBufferInfo dbi[kBindings]{};
  auto info = [&](uint32_t i, const Buf& b) {
    dbi[i].buffer = b.buf;
    dbi[i].offset = 0;
    dbi[i].range = VK_WHOLE_SIZE;
  };
  info(0, x_override ? *x_override : g.x);
  for (uint32_t d = 0; d < kMaxDims; ++d) {
    uint32_t base = 1 + d * 6;
    if (d < sh.dims) {
      info(base, wsrc.w[d]);
      info(base + 1, wsrc.scales[d]);
      info(base + 2, wsrc.biases[d]);
      info(base + 3, g.out[d]);
      info(base + 4, g.out[d]);
      info(base + 5, g.out[d]);
    } else {
      for (uint32_t j = 0; j < 6; ++j) info(base + j, g.out[0]);
    }
  }
  VkWriteDescriptorSet wr[kBindings]{};
  for (uint32_t i = 0; i < kBindings; ++i) {
    wr[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    wr[i].dstSet = set;
    wr[i].dstBinding = i;
    wr[i].descriptorCount = 1;
    wr[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    wr[i].pBufferInfo = &dbi[i];
  }
  g_vk.UpdateDescriptorSets(g_vk.dev, kBindings, wr, 0, nullptr);
  return set;
}

static void run_gap_mode(const DeviceCtx& ctx, bool quick) {
  const int rounds = quick ? 3 : 7;
  const uint32_t tokens_iso = quick ? 32 : 64;
  const uint32_t tokens_wide = quick ? 8 : 16;
  const uint32_t Lmax = 24;

  const char* q4_defines = g_gap_bf16
      ? "-DUSE_BF16=1 -DUSE_SUBGROUP=1 -DQMM_VEC_Q4_WORD=1 -DQMM_VEC_MULTI=1"
      : "-DUSE_FP16=1 -DUSE_SUBGROUP=1 -DQMM_VEC_Q4_WORD=1 -DQMM_VEC_MULTI=1";
  if (compile_shader("tools/q4-bw-bench/shaders/qmm_vec_base.comp",
          q4_defines, "/tmp/q4gap_base.spv") != 0)
    die("compile gap base");
  Side side;
  side.tag = "base";
  side.mod = make_module(read_file("/tmp/q4gap_base.spv"));
  make_pipeline(side);

  Params sparams[kNumShapes];
  uint32_t sgroups[kNumShapes];
  uint64_t layer_bytes = 0;
  for (uint32_t s = 0; s < g_num_shapes; ++s) {
    if (g_shapes[s].per_token) continue;
    sgroups[s] = shape_groups(g_shapes[s], 8u);
    fill_params(sparams[s], g_shapes[s], sgroups[s]);
    layer_bytes += shape_bytes(g_shapes[s]);
  }
  std::printf(
      "{\"k\":\"gap_cfg\",\"layer_bytes\":%llu,\"sets\":%u,\"bytes_all\":%llu,"
      "\"tokens_iso\":%u,\"tokens_wide\":%u,\"rounds\":%d}\n",
      (unsigned long long)layer_bytes, Lmax,
      (unsigned long long)layer_bytes * Lmax, tokens_iso, tokens_wide,
      rounds);

  // Weight sets: Lmax independent per-layer weight footprints.
  std::vector<GapSet> sets(Lmax);
  for (uint32_t s = 0; s < g_num_shapes; ++s) {
    if (g_shapes[s].per_token) continue;
    const Shape& sh = g_shapes[s];
    for (uint32_t l = 0; l < Lmax; ++l) {
      GapShapeBufs& g = sets[l].sh[s];
      g.x = gap_alloc(ctx, (uint64_t)sh.k * 2u);
      for (uint32_t d = 0; d < sh.dims; ++d) {
        g.w[d] = gap_alloc(ctx, (uint64_t)sh.n[d] * (sh.k / 8u) * 4u);
        g.scales[d] = gap_alloc(ctx, (uint64_t)sh.n[d] * (sh.k / 64u) * 2u);
        g.biases[d] = gap_alloc(ctx, (uint64_t)sh.n[d] * (sh.k / 64u) * 2u);
        g.out[d] = gap_alloc(ctx, (uint64_t)sh.n[d] * 2u);
      }
    }
  }

  VkDescriptorPoolSize ps{};
  ps.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  ps.descriptorCount = kBindings * 512 * (g_num_shapes / 4u + 1u);
  VkDescriptorPoolCreateInfo dpci{};
  dpci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  dpci.maxSets = 512 * (g_num_shapes / 4u + 1u);
  dpci.poolSizeCount = 1;
  dpci.pPoolSizes = &ps;
  dpci.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
  VkDescriptorPool pool;
  if (g_vk.CreateDescriptorPool(g_vk.dev, &dpci, nullptr, &pool) !=
      VK_SUCCESS)
    die("gap descriptor pool");

  using D4 = std::array<VkDescriptorSet, kNumShapes>;
  // Chained descriptor variants: shape s reads the producer's output.
  auto chain_x = [&](uint32_t l, uint32_t s, bool ring_b) -> const Buf* {
    if (s == 0 || s >= 4) {
      // qkv consumes the previous layer's down output; the added probe
      // shapes (gate6144/zout/q4096/kv512/ab/lm_head) run on their own x.
      if (s != 0) return nullptr;
    }
    if (s == 1) return &sets[l].sh[0].out[0]; // o.x = qkv.out0
    if (s == 2) return &sets[l].sh[1].out[0]; // gate_up.x = o.out0
    if (s == 3) return &sets[l].sh[2].out[0]; // down.x = gate_up.out0
    // qkv.x = previous set's down.out0; set 0 closes the ring across tokens.
    if (l == 0) return ring_b ? &sets[Lmax - 1].sh[3].out[0] : nullptr;
    return &sets[l - 1].sh[3].out[0];
  };
  std::vector<D4> desc_ind(Lmax), dep_a(Lmax), dep_b(Lmax);
  for (uint32_t l = 0; l < Lmax; ++l) {
    for (uint32_t s = 0; s < g_num_shapes; ++s) {
      if (g_shapes[s].per_token) continue;
      desc_ind[l][s] = gap_make_set(pool, side.dsl, sets[l].sh[s],
          sets[l].sh[s], s, nullptr);
      dep_a[l][s] = gap_make_set(pool, side.dsl, sets[l].sh[s],
          sets[l].sh[s], s, chain_x(l, s, false));
      dep_b[l][s] = gap_make_set(pool, side.dsl, sets[l].sh[s],
          sets[l].sh[s], s, chain_x(l, s, true));
    }
  }
  // One-set RAW ring (iso4dep): odd tokens close the ring on its own down.
  D4 iso_dep_a, iso_dep_b;
  for (uint32_t s = 0; s < g_num_shapes; ++s) {
    if (g_shapes[s].per_token) continue;
    const Buf* xa = chain_x(0, s, false);
    const Buf* xb = chain_x(0, s, true);
    iso_dep_a[s] = gap_make_set(pool, side.dsl, sets[0].sh[s],
        sets[0].sh[s], s, xa);
    iso_dep_b[s] = gap_make_set(pool, side.dsl, sets[0].sh[s],
        sets[0].sh[s], s, xb);
  }

  // Filler pipeline: the peak_read streaming probe (same access shape as
  // the weight stream), pushed between GEMV dispatches.
  if (compile_shader("tools/q4-bw-bench/shaders/peak_read.comp", "",
          "/tmp/q4gap_fill.spv") != 0)
    die("compile gap filler");
  PeakPipe fill;
  fill.mod = make_module(read_file("/tmp/q4gap_fill.spv"));
  {
    VkDescriptorSetLayoutBinding b[2]{};
    for (uint32_t i = 0; i < 2; ++i) {
      b[i].binding = i;
      b[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      b[i].descriptorCount = 1;
      b[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dslci{};
    dslci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dslci.bindingCount = 2;
    dslci.pBindings = b;
    if (g_vk.CreateDescriptorSetLayout(
            g_vk.dev, &dslci, nullptr, &fill.dsl) != VK_SUCCESS)
      die("gap filler dsl");
    VkPipelineLayoutCreateInfo plci{};
    plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plci.setLayoutCount = 1;
    plci.pSetLayouts = &fill.dsl;
    if (g_vk.CreatePipelineLayout(
            g_vk.dev, &plci, nullptr, &fill.layout) != VK_SUCCESS)
      die("gap filler layout");
    VkComputePipelineCreateInfo cpci{};
    cpci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cpci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = fill.mod;
    cpci.stage.pName = "main";
    cpci.layout = fill.layout;
    if (g_vk.CreateComputePipelines(
            g_vk.dev, VK_NULL_HANDLE, 1, &cpci, nullptr, &fill.pipe) !=
        VK_SUCCESS)
      die("gap filler pipe");
  }
  Buf fill_src = gap_alloc(ctx, 64ull << 20);
  VkDescriptorSet fill_set{};
  Buf fill_sink = gap_alloc(ctx, 65536);
  {
    VkDescriptorPoolSize fps{};
    fps.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    fps.descriptorCount = 2;
    VkDescriptorPoolCreateInfo fdpci{};
    fdpci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    fdpci.maxSets = 1;
    fdpci.poolSizeCount = 1;
    fdpci.pPoolSizes = &fps;
    VkDescriptorPool fpool;
    if (g_vk.CreateDescriptorPool(g_vk.dev, &fdpci, nullptr, &fpool) !=
        VK_SUCCESS)
      die("gap filler pool");
    VkDescriptorSetAllocateInfo dsai{};
    dsai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dsai.descriptorPool = fpool;
    dsai.descriptorSetCount = 1;
    dsai.pSetLayouts = &fill.dsl;
    if (g_vk.AllocateDescriptorSets(g_vk.dev, &dsai, &fill_set) != VK_SUCCESS)
      die("gap filler set");
    VkDescriptorBufferInfo dbi[2]{};
    dbi[0].buffer = fill_src.buf;
    dbi[0].range = VK_WHOLE_SIZE;
    dbi[1].buffer = fill_sink.buf;
    dbi[1].range = VK_WHOLE_SIZE;
    VkWriteDescriptorSet w[2]{};
    for (int i = 0; i < 2; ++i) {
      w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      w[i].dstSet = fill_set;
      w[i].dstBinding = i;
      w[i].descriptorCount = 1;
      w[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      w[i].pBufferInfo = &dbi[i];
    }
    g_vk.UpdateDescriptorSets(g_vk.dev, 2, w, 0, nullptr);
  }

  // The arm recorders are parameterized over the producing side so the
  // candidate screen at the bottom can reuse the exact base descriptor
  // sets and arm machinery; defaults keep every pre-existing arm
  // byte-identical in behavior.
  Side* arm_side = &side;
  uint32_t* arm_groups = sgroups;
  Params* arm_params = sparams;
  CmdRes cmd = make_cmd(ctx, 2);

  auto record_filler = [&](VkCommandBuffer c, uint32_t groups) {
    g_vk.CmdBindPipeline(c, VK_PIPELINE_BIND_POINT_COMPUTE, fill.pipe);
    g_vk.CmdBindDescriptorSets(c, VK_PIPELINE_BIND_POINT_COMPUTE,
        fill.layout, 0, 1, &fill_set, 0, nullptr);
    g_vk.CmdDispatch(c, groups, 1, 1);
    g_vk.CmdBindPipeline(c, VK_PIPELINE_BIND_POINT_COMPUTE, arm_side->pipe);
  };
  auto record_layer = [&](VkCommandBuffer c, const VkDescriptorSet* d4,
                            bool with_filler, uint32_t fill_per) {
    for (uint32_t s = 0; s < g_num_shapes; ++s) {
      if (g_shapes[s].per_token) continue;
      if (with_filler) record_filler(c, fill_per);
      g_vk.CmdBindDescriptorSets(c, VK_PIPELINE_BIND_POINT_COMPUTE,
          arm_side->layout, 0, 1, &d4[s], 0, nullptr);
      g_vk.CmdPushConstants(c, arm_side->layout,
          VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(Params), &arm_params[s]);
      g_vk.CmdDispatch(c, arm_groups[s], 1, 1);
    }
    if (with_filler) record_filler(c, fill_per);
  };

  auto line = [&](const char* arm, uint32_t nsets, uint32_t tokens,
                    const std::vector<uint64_t>& per_layer_ns, int pass) {
    uint64_t med = median(per_layer_ns), mn = min_of(per_layer_ns);
    std::printf(
        "{\"k\":\"gap\",\"arm\":\"%s\",\"sets\":%u,\"tokens\":%u,"
        "\"rounds\":%d,\"layer_bytes\":%llu,\"layer_ns_med\":%llu,"
        "\"layer_ns_min\":%llu,\"token_ns_med\":%llu,\"weight_gb_s\":%.2f,"
        "\"pass\":%d}\n",
        arm, nsets, tokens, rounds, (unsigned long long)layer_bytes,
        (unsigned long long)med, (unsigned long long)mn,
        (unsigned long long)med * nsets,
        (double)layer_bytes / ((double)med * 1e-9) / 1e9, pass);
  };

  // Standard arm: tokens_per_submit tokens recorded into one submit;
  // per-round wall clock divided by tokens*nsets gives per-layer ns.
  auto run_arm = [&](const char* arm, uint32_t nsets, uint32_t tokens,
      bool chained, bool with_filler, uint32_t filler_mb, int pass) {
    uint32_t fill_per = filler_mb * 64u; // filler dispatch = mb/4 MB
    std::vector<uint64_t> samples;
    for (int r = -1; r < rounds; ++r) {
      begin(cmd);
      g_vk.CmdBindPipeline(
          cmd.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, arm_side->pipe);
      for (uint32_t t = 0; t < tokens; ++t) {
        uint32_t parity = t & 1u;
        for (uint32_t j = 0; j < nsets; ++j) {
          const VkDescriptorSet* d4 =
              chained ? (parity ? (nsets == 1 ? iso_dep_b.data()
                                                : dep_b[j].data())
                                : (nsets == 1 ? iso_dep_a.data()
                                                : dep_a[j].data()))
                      : desc_ind[j].data();
          record_layer(cmd.cmd, d4, with_filler, fill_per);
        }
      }
      g_vk.EndCommandBuffer(cmd.cmd);
      auto t0 = std::chrono::steady_clock::now();
      submit_and_wait(ctx, cmd.cmd);
      auto t1 = std::chrono::steady_clock::now();
      if (r >= 0)
        samples.push_back(
            std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0)
                    .count() /
                (tokens * nsets));
    }
    line(arm, nsets, tokens, samples, pass);
  };

  // churn: fresh x/out allocations per round (outside the timed bracket,
  // as the host allocator acts between tokens).
  auto run_churn = [&](uint32_t tokens, int pass) {
    std::vector<uint64_t> samples;
    for (int r = -1; r < rounds; ++r) {
      GapSet fresh{};
      for (uint32_t s = 0; s < g_num_shapes; ++s) {
        if (g_shapes[s].per_token) continue;
        const Shape& sh = g_shapes[s];
        GapShapeBufs& g = fresh.sh[s];
        g.x = gap_alloc(ctx, (uint64_t)sh.k * 2u);
        for (uint32_t d = 0; d < sh.dims; ++d)
          g.out[d] = gap_alloc(ctx, (uint64_t)sh.n[d] * 2u);
      }
      D4 d4;
      for (uint32_t s = 0; s < g_num_shapes; ++s) {
        if (g_shapes[s].per_token) continue;
        d4[s] = gap_make_set(pool, side.dsl, fresh.sh[s], sets[0].sh[s],
            s, nullptr);
      }
      begin(cmd);
      g_vk.CmdBindPipeline(
          cmd.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, side.pipe);
      for (uint32_t t = 0; t < tokens; ++t)
        record_layer(cmd.cmd, d4.data(), false, 0);
      g_vk.EndCommandBuffer(cmd.cmd);
      auto t0 = std::chrono::steady_clock::now();
      submit_and_wait(ctx, cmd.cmd);
      auto t1 = std::chrono::steady_clock::now();
      if (r >= 0)
        samples.push_back(
            std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0)
                    .count() /
                tokens);
      for (auto& gf : fresh.sh) gap_free_shape(gf);
      Buf shift = gap_alloc(ctx, (1ull << 20) + (r & 1) * 4096ull);
      gap_free(shift);
    }
    line("churn", 1, tokens, samples, pass);
  };

  // sub2: widedep as two submissions per token (half the sets each).
  auto run_sub2 = [&](bool sync, uint32_t tokens, int pass) {
    std::vector<uint64_t> samples;
    uint32_t half = Lmax / 2;
    for (int r = -1; r < rounds; ++r) {
      auto t0 = std::chrono::steady_clock::now();
      for (uint32_t t = 0; t < tokens; ++t) {
        uint32_t parity = t & 1u;
        for (int h = 0; h < 2; ++h) {
          begin(cmd);
          g_vk.CmdBindPipeline(cmd.cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
              side.pipe);
          for (uint32_t j = h * half; j < (h + 1) * half; ++j) {
            const VkDescriptorSet* d4 =
                parity ? dep_b[j].data() : dep_a[j].data();
            record_layer(cmd.cmd, d4, false, 0);
          }
          g_vk.EndCommandBuffer(cmd.cmd);
          VkSubmitInfo si{};
          si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
          si.commandBufferCount = 1;
          si.pCommandBuffers = &cmd.cmd;
          VkFence f;
          VkFenceCreateInfo fci{};
          fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
          g_vk.CreateFence(g_vk.dev, &fci, nullptr, &f);
          g_vk.QueueSubmit(ctx.queue, 1, &si, f);
          if (sync || h == 1)
            g_vk.WaitForFences(g_vk.dev, 1, &f, VK_TRUE, UINT64_MAX);
          g_vk.DestroyFence(g_vk.dev, f, nullptr);
        }
      }
      auto t1 = std::chrono::steady_clock::now();
      if (r >= 0)
        samples.push_back(
            std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0)
                    .count() /
                (tokens * Lmax));
    }
    line(sync ? "sub2sync" : "sub2async", Lmax, tokens, samples, pass);
  };

  // Anchor probes: the device DRAM ceiling through the same instrument.
  run_peak(ctx, "tools/q4-bw-bench/shaders/peak_copy.comp",
      "/tmp/peak_copy.spv", "copy", 1u << 24u, false);
  run_peak(ctx, "tools/q4-bw-bench/shaders/peak_read.comp",
      "/tmp/peak_read.spv", "read", 1u << 24u, false);

  for (int pass = 1; pass <= 2; ++pass) {
    run_arm("iso4", 1, tokens_iso, false, false, 0, pass);
    run_arm("iso4dep", 1, tokens_iso, true, false, 0, pass);
    run_arm("fill2", 1, tokens_iso / 2, false, true, 2, pass);
    run_arm("fill24", 1, tokens_iso / 4, false, true, 24, pass);
    run_churn(tokens_iso, pass);
    const uint32_t sweep_sizes[] = {1, 2, 4, 6, 8, 12, 16, 24};
    for (uint32_t n : sweep_sizes) {
      uint32_t tok = n >= 12 ? tokens_wide : tokens_iso / (n / 2 + 1);
      run_arm("sweep", n, tok, false, false, 0, pass);
    }
    run_arm("wideind", Lmax, tokens_wide, false, false, 0, pass);
    run_arm("widedep", Lmax, tokens_wide, true, false, 0, pass);
    run_sub2(false, tokens_wide / 2, pass);
    run_sub2(true, tokens_wide / 2, pass);
  }

  // Candidate screen, wall-anchored, in the in-model-style widedep
  // environment (24 independent sets, RAW chain) plus the iso4
  // calibration arm. The candidates keep the production lane->word map
  // and arithmetic order (receipts/2026-09-10-q4-gemv-bandwidth); they
  // vary instruction scheduling and workgroup mapping only. Descriptor
  // sets stay the base ones: every side's layout is identically defined,
  // which is exactly Vulkan layout compatibility.
  {
    struct CandSpec {
      const char* tag;
      const char* src;
      const char* spv;
      uint32_t columns;
    };
    std::vector<CandSpec> cands = {
        {"unroll", "tools/q4-bw-bench/shaders/qmm_vec_cand_unroll.comp",
            "/tmp/q4gap_cand_u.spv", 8u},
        {"loadfirst",
            "tools/q4-bw-bench/shaders/qmm_vec_cand_loadfirst.comp",
            "/tmp/q4gap_cand_l.spv", 8u},
        {"wg128", "tools/q4-bw-bench/shaders/qmm_vec_cand_wg128.comp",
            "/tmp/q4gap_cand_w.spv", 4u},
        {"xpack", "tools/q4-bw-bench/shaders/qmm_vec_cand_xpack.comp",
            "/tmp/q4gap_cand_x.spv", 8u},
    };
    if (!g_cand_src.empty())
      cands = {{"cand", g_cand_src.c_str(), "/tmp/q4gap_cand_c.spv",
          g_cand_columns}};
    for (const CandSpec& c : cands) {
      // loadfirst's explicit x-quad addressing is measurement-only for
      // the f16 uvec4 view; it does not compile under the --2b bf16
      // defines, so it is skipped there rather than killing the screen.
      if (g_gap_bf16 && std::string(c.tag) == "loadfirst") continue;
      std::string cdefs = std::string(q4_defines) +
          (std::string(c.tag) == "cand" ? g_cand_defs : std::string());
      if (compile_shader(c.src, cdefs.c_str(), c.spv) != 0)
        die("compile gap cand %s", c.tag);
      Side cs;
      cs.tag = c.tag;
      cs.mod = make_module(read_file(c.spv));
      make_pipeline(cs);
      uint32_t cgroups[kNumShapes];
      Params cparams[kNumShapes];
      for (uint32_t s = 0; s < g_num_shapes; ++s) {
        if (g_shapes[s].per_token) continue;
        cgroups[s] = shape_groups(g_shapes[s], c.columns);
        fill_params(cparams[s], g_shapes[s], cgroups[s]);
        if (g_grid_cap && cgroups[s] > g_grid_cap) cgroups[s] = g_grid_cap;
      }
      arm_side = &cs;
      arm_groups = cgroups;
      arm_params = cparams;
      for (int pass = 1; pass <= 2; ++pass) {
        std::string i4 = std::string("iso4_") + c.tag;
        std::string wd = std::string("widedep_") + c.tag;
        run_arm(i4.c_str(), 1, tokens_iso, false, false, 0, pass);
        run_arm(wd.c_str(), Lmax, tokens_wide, true, false, 0, pass);
      }
    }
    arm_side = &side;
    arm_groups = sgroups;
    arm_params = sparams;
  }

  // Cleanup (process exits anyway; keep the driver happy on the way out).
  gap_free(fill_src);
  gap_free(fill_sink);
  for (auto& gs : sets)
    for (uint32_t s = 0; s < g_num_shapes; ++s)
      gap_free_shape(gs.sh[s]);
}
// ---------------------------------------------------------------------------
// Access-pattern roof mode (--roof): what GB/s the memory system delivers
// for the exact Q4 decode GEMV byte layout, wall-anchored only. Arms:
//   * streaming copy/read peaks (same submit bracket as everything else),
//   * the parametrized pattern probe (shaders/q4_pattern.comp): the
//     packed-word + per-group-scale/bias + x stream with a near-empty
//     body, sweeping load width, rows per workgroup and row pitch,
//   * the production base kernel wall-timed per dispatch shape (8
//     back-to-back dispatches per submit, divided by 8) and as the
//     four-dispatch layer chain in one submit.
struct PatArm {
  const char* name;
  const char* defines;
  uint32_t rows_per_wg;
  uint32_t pitch;
};

static void run_roof_mode(const DeviceCtx& ctx, bool quick,
    const char* variant_defines) {
  const uint32_t rows = 524288u; // 524288 x 448 B words = 235 MB
  const uint32_t words_per_row = 112u; // k = 896
  const uint32_t groups_per_row = 14u;
  const uint32_t max_pitch = 128u;
  const int rounds = quick ? 3 : 7;

  // Streaming peaks, same instrument as every arm below.
  run_peak(ctx, "tools/q4-bw-bench/shaders/peak_copy.comp",
      "/tmp/roof_copy.spv", "copy", 1u << 24u, false);
  run_peak(ctx, "tools/q4-bw-bench/shaders/peak_read.comp",
      "/tmp/roof_read.spv", "read", 1u << 24u, false);

  // Pattern probe buffers, sized once for the largest pitch.
  Buf x_b = make_buf(g_vk.dev, ctx.mp, (uint64_t)words_per_row * 16u, false);
  Buf w_b = make_buf(g_vk.dev, ctx.mp,
      (uint64_t)rows * max_pitch * 4u, false);
  Buf s_b = make_buf(g_vk.dev, ctx.mp,
      (uint64_t)rows * groups_per_row * 2u, false);
  Buf b_b = make_buf(g_vk.dev, ctx.mp,
      (uint64_t)rows * groups_per_row * 2u, false);
  Buf o_b = make_buf(g_vk.dev, ctx.mp, (uint64_t)rows * 4u, true);

  struct PatPipe {
    VkShaderModule mod{VK_NULL_HANDLE};
    VkDescriptorSetLayout dsl{VK_NULL_HANDLE};
    VkPipelineLayout layout{VK_NULL_HANDLE};
    VkPipeline pipe{VK_NULL_HANDLE};
    VkDescriptorPool pool{VK_NULL_HANDLE};
    VkDescriptorSet set{VK_NULL_HANDLE};
    CmdRes cmd;
  };
  auto make_pat = [&](PatPipe& pp, const char* defines) {
    if (compile_shader("tools/q4-bw-bench/shaders/q4_pattern.comp",
            defines, "/tmp/roof_pat.spv") != 0)
      die("compile pattern arm %s", defines);
    pp.mod = make_module(read_file("/tmp/roof_pat.spv"));
    VkDescriptorSetLayoutBinding b[7]{};
    for (uint32_t i = 0; i < 7; ++i) {
      b[i].binding = i;
      b[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      b[i].descriptorCount = 1;
      b[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dslci{};
    dslci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dslci.bindingCount = 7;
    dslci.pBindings = b;
    if (g_vk.CreateDescriptorSetLayout(g_vk.dev, &dslci, nullptr,
            &pp.dsl) != VK_SUCCESS)
      die("pattern dsl");
    VkPushConstantRange pc{};
    pc.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pc.offset = 0;
    pc.size = 16u;
    VkPipelineLayoutCreateInfo plci{};
    plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plci.setLayoutCount = 1;
    plci.pSetLayouts = &pp.dsl;
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges = &pc;
    if (g_vk.CreatePipelineLayout(g_vk.dev, &plci, nullptr, &pp.layout) !=
        VK_SUCCESS)
      die("pattern layout");
    VkComputePipelineCreateInfo cpci{};
    cpci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cpci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = pp.mod;
    cpci.stage.pName = "main";
    cpci.layout = pp.layout;
    if (g_vk.CreateComputePipelines(g_vk.dev, VK_NULL_HANDLE, 1, &cpci,
            nullptr, &pp.pipe) != VK_SUCCESS)
      die("pattern pipe");
    VkDescriptorPoolSize ps{};
    ps.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    ps.descriptorCount = 7;
    VkDescriptorPoolCreateInfo dpci{};
    dpci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dpci.maxSets = 1;
    dpci.poolSizeCount = 1;
    dpci.pPoolSizes = &ps;
    if (g_vk.CreateDescriptorPool(g_vk.dev, &dpci, nullptr, &pp.pool) !=
        VK_SUCCESS)
      die("pattern pool");
    VkDescriptorSetAllocateInfo dsai{};
    dsai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dsai.descriptorPool = pp.pool;
    dsai.descriptorSetCount = 1;
    dsai.pSetLayouts = &pp.dsl;
    if (g_vk.AllocateDescriptorSets(g_vk.dev, &dsai, &pp.set) !=
        VK_SUCCESS)
      die("pattern set");
    VkDescriptorBufferInfo dbi[7]{};
    Buf* bufs[7] = {&x_b, &w_b, &s_b, &b_b, &o_b, &w_b, &w_b};
    VkWriteDescriptorSet wr[7]{};
    for (uint32_t i = 0; i < 7; ++i) {
      dbi[i].buffer = bufs[i]->buf;
      dbi[i].range = VK_WHOLE_SIZE;
      wr[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      wr[i].dstSet = pp.set;
      wr[i].dstBinding = i;
      wr[i].descriptorCount = 1;
      wr[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      wr[i].pBufferInfo = &dbi[i];
    }
    g_vk.UpdateDescriptorSets(g_vk.dev, 7, wr, 0, nullptr);
    pp.cmd = make_cmd(ctx, 2);
  };

  const PatArm arms[] = {
      // Production access pattern: scalar words, one per lane per step,
      // 8 rows per workgroup, contiguous 448 B rows.
      {"pat_w1_r8", "-DROWS_PER_WG=8 -DWORDS_PER_LANE=1 -DLOAD_WIDTH=1",
          8u, 112u},
      // Pitch/alignment: line-aligned 512 B rows and odd-phase rows.
      {"pat_w1_r8_l128",
          "-DROWS_PER_WG=8 -DWORDS_PER_LANE=1 -DLOAD_WIDTH=1", 8u, 128u},
      {"pat_w1_r8_p114",
          "-DROWS_PER_WG=8 -DWORDS_PER_LANE=1 -DLOAD_WIDTH=1", 8u, 114u},
      // Load width at the production row count.
      {"pat_w2_r8", "-DROWS_PER_WG=8 -DWORDS_PER_LANE=2 -DLOAD_WIDTH=1",
          8u, 112u},
      {"pat_v2_r8", "-DROWS_PER_WG=8 -DWORDS_PER_LANE=2 -DLOAD_WIDTH=2",
          8u, 112u},
      {"pat_v4_r8", "-DROWS_PER_WG=8 -DWORDS_PER_LANE=4 -DLOAD_WIDTH=4",
          8u, 112u},
      // Concurrent rows per workgroup at the production load width.
      {"pat_w1_r1", "-DROWS_PER_WG=1 -DWORDS_PER_LANE=1 -DLOAD_WIDTH=1",
          1u, 112u},
      {"pat_w1_r2", "-DROWS_PER_WG=2 -DWORDS_PER_LANE=1 -DLOAD_WIDTH=1",
          2u, 112u},
      {"pat_w1_r4", "-DROWS_PER_WG=4 -DWORDS_PER_LANE=1 -DLOAD_WIDTH=1",
          4u, 112u},
      {"pat_w1_r16", "-DROWS_PER_WG=16 -DWORDS_PER_LANE=1 -DLOAD_WIDTH=1",
          16u, 112u},
      // Best-guess wide x deep combinations.
      {"pat_v4_r16", "-DROWS_PER_WG=16 -DWORDS_PER_LANE=4 -DLOAD_WIDTH=4",
          16u, 112u},
      {"pat_v2_r16", "-DROWS_PER_WG=16 -DWORDS_PER_LANE=2 -DLOAD_WIDTH=2",
          16u, 112u},
  };
  for (const PatArm& arm : arms) {
    PatPipe pp;
    make_pat(pp, arm.defines);
    uint32_t xg = (rows + arm.rows_per_wg - 1u) / arm.rows_per_wg;
    uint32_t gx = xg < 32768u ? xg : 32768u;
    uint32_t gy = (xg + gx - 1u) / gx;
    uint32_t pcvals[4] = {rows, words_per_row, arm.pitch, groups_per_row};
    double bytes = (double)rows * ((double)words_per_row * 4.0 +
        (double)groups_per_row * 4.0 + 4.0) + (double)words_per_row * 16.0;
    std::vector<uint64_t> samples;
    for (int rep = 0; rep < rounds + 1; ++rep) {
      begin(pp.cmd);
      g_vk.CmdBindPipeline(
          pp.cmd.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pp.pipe);
      g_vk.CmdBindDescriptorSets(pp.cmd.cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
          pp.layout, 0, 1, &pp.set, 0, nullptr);
      g_vk.CmdPushConstants(pp.cmd.cmd, pp.layout,
          VK_SHADER_STAGE_COMPUTE_BIT, 0, 16u, pcvals);
      g_vk.CmdDispatch(pp.cmd.cmd, gx, gy, 1);
      g_vk.EndCommandBuffer(pp.cmd.cmd);
      submit_and_wait(ctx, pp.cmd.cmd);
      if (rep > 0) samples.push_back(g_wall_ns);
    }
    uint64_t med = median(samples);
    std::printf(
        "{\"k\":\"pat\",\"arm\":\"%s\",\"rows\":%u,\"pitch\":%u,"
        "\"rows_per_wg\":%u,\"bytes\":%.0f,\"med_wall_ns\":%llu,"
        "\"med_gb_s\":%.2f,\"min_gb_s\":%.2f}\n",
        arm.name, rows, arm.pitch, arm.rows_per_wg, bytes,
        (unsigned long long)med, bytes / (double)med,
        bytes / (double)min_of(samples));
    fflush(stdout);
    g_vk.DestroyCommandPool(g_vk.dev, pp.cmd.pool, nullptr);
    g_vk.DestroyDescriptorPool(g_vk.dev, pp.pool, nullptr);
    g_vk.DestroyPipeline(g_vk.dev, pp.pipe, nullptr);
    g_vk.DestroyPipelineLayout(g_vk.dev, pp.layout, nullptr);
    g_vk.DestroyDescriptorSetLayout(g_vk.dev, pp.dsl, nullptr);
    g_vk.DestroyShaderModule(g_vk.dev, pp.mod, nullptr);
  }

  // Production base kernel, wall-anchored per shape and as the layer
  // chain. Bit-identity is not at stake here (no candidate side); this
  // anchors the pattern arms to the real kernel's achieved rate.
  Side base;
  base.tag = "base";
  if (compile_shader("tools/q4-bw-bench/shaders/qmm_vec_base.comp",
          variant_defines, "/tmp/roof_base.spv") != 0)
    die("compile base");
  base.mod = make_module(read_file("/tmp/roof_base.spv"));
  make_pipeline(base);
  SetBufs inputs[kNumShapes];
  SetBufs outs[kNumShapes];
  Params shape_params[kNumShapes];
  uint32_t groups_v[kNumShapes];
  for (uint32_t s = 0; s < g_num_shapes; ++s) {
    const Shape& sh = g_shapes[s];
    inputs[s].x = make_buf(g_vk.dev, ctx.mp, (uint64_t)sh.k * 2u, false);
    for (uint32_t d = 0; d < sh.dims; ++d) {
      inputs[s].w[d] = make_buf(g_vk.dev, ctx.mp,
          (uint64_t)sh.n[d] * (sh.k / 8u) * 4u, false);
      inputs[s].scales[d] = make_buf(g_vk.dev, ctx.mp,
          (uint64_t)sh.n[d] * (sh.k / 64u) * 2u, false);
      inputs[s].biases[d] = make_buf(g_vk.dev, ctx.mp,
          (uint64_t)sh.n[d] * (sh.k / 64u) * 2u, false);
      outs[s].out[d] = make_buf(g_vk.dev, ctx.mp,
          (uint64_t)sh.n[d] * 2u, true);
    }
    for (uint32_t d = sh.dims; d < kMaxDims; ++d)
      outs[s].out[d] = outs[s].out[0];
    groups_v[s] = shape_groups(sh, 8u);
    fill_params(shape_params[s], sh, groups_v[s]);
  }
  VkDescriptorPoolSize ps{};
  ps.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  ps.descriptorCount = kBindings * kNumShapes;
  VkDescriptorPoolCreateInfo dpci{};
  dpci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  dpci.maxSets = kNumShapes;
  dpci.poolSizeCount = 1;
  dpci.pPoolSizes = &ps;
  VkDescriptorPool pool;
  if (g_vk.CreateDescriptorPool(g_vk.dev, &dpci, nullptr, &pool) !=
      VK_SUCCESS)
    die("roof base pool");
  VkDescriptorSet sets[kNumShapes];
  for (uint32_t s = 0; s < g_num_shapes; ++s) {
    SetBufs combined = inputs[s];
    for (uint32_t d = 0; d < kMaxDims; ++d)
      combined.out[d] = outs[s].out[d];
    sets[s] = make_set(pool, base.dsl, combined, g_shapes[s].dims);
  }
  CmdRes cmd = make_cmd(ctx, 2);
  const uint32_t k_repeat = 8u;
  for (uint32_t s = 0; s < g_num_shapes; ++s) {
    const Shape& sh = g_shapes[s];
    std::vector<uint64_t> samples;
    for (int rep = 0; rep < rounds + 1; ++rep) {
      begin(cmd);
      g_vk.CmdBindPipeline(
          cmd.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, base.pipe);
      g_vk.CmdBindDescriptorSets(cmd.cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
          base.layout, 0, 1, &sets[s], 0, nullptr);
      for (uint32_t k = 0; k < k_repeat; ++k) {
        g_vk.CmdPushConstants(cmd.cmd, base.layout,
            VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(Params),
            &shape_params[s]);
        g_vk.CmdDispatch(cmd.cmd, groups_v[s], 1, 1);
      }
      g_vk.EndCommandBuffer(cmd.cmd);
      submit_and_wait(ctx, cmd.cmd);
      if (rep > 0) samples.push_back(g_wall_ns / k_repeat);
    }
    uint64_t med = median(samples);
    double bytes = (double)shape_bytes(sh);
    std::printf(
        "{\"k\":\"q4wall\",\"shape\":\"%s\",\"grid\":%u,\"k_repeat\":%u,"
        "\"bytes\":%.0f,\"med_wall_ns\":%llu,\"med_gb_s\":%.2f,"
        "\"min_gb_s\":%.2f}\n",
        sh.name, groups_v[s], k_repeat, bytes, (unsigned long long)med,
        bytes / (double)med, bytes / (double)min_of(samples));
    fflush(stdout);
  }
  // One layer chain: qkv, o, gate_up, down in one submit, like one
  // decode-token layer issues them.
  {
    std::vector<uint64_t> samples;
    for (int rep = 0; rep < rounds + 1; ++rep) {
      begin(cmd);
      g_vk.CmdBindPipeline(
          cmd.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, base.pipe);
      for (uint32_t s = 0; s < g_num_shapes; ++s) {
        g_vk.CmdBindDescriptorSets(cmd.cmd,
            VK_PIPELINE_BIND_POINT_COMPUTE, base.layout, 0, 1, &sets[s],
            0, nullptr);
        g_vk.CmdPushConstants(cmd.cmd, base.layout,
            VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(Params),
            &shape_params[s]);
        g_vk.CmdDispatch(cmd.cmd, groups_v[s], 1, 1);
      }
      g_vk.EndCommandBuffer(cmd.cmd);
      submit_and_wait(ctx, cmd.cmd);
      if (rep > 0) samples.push_back(g_wall_ns);
    }
    uint64_t med = median(samples);
    double bytes = 0.0;
    for (uint32_t s = 0; s < g_num_shapes; ++s)
      bytes += (double)shape_bytes(g_shapes[s]);
    std::printf(
        "{\"k\":\"q4layer\",\"bytes\":%.0f,\"med_wall_ns\":%llu,"
        "\"med_gb_s\":%.2f,\"us_per_layer\":%.1f,\"ms_per_token_24\":%.2f,"
        "\"tok_s_q4only\":%.1f}\n",
        bytes, (unsigned long long)med, bytes / (double)med,
        (double)med / 1000.0, 24.0 * (double)med / 1e6,
        1000.0 / (24.0 * (double)med / 1e9));
    fflush(stdout);
  }
}

static bool got_e_dump(const std::vector<uint16_t>&, uint16_t*, size_t) { return false; }
int main(int argc, char** argv) {
  bool tree_mode = false;
  bool quick = false;
  bool gap_mode = false;
  bool roof_mode = false;
  bool bf16_eq = false;
  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]) == "--tree") tree_mode = true;
    if (std::string(argv[i]) == "--quick") quick = true;
    if (std::string(argv[i]) == "--bf16eq") bf16_eq = true;
    if (std::string(argv[i]) == "--gap") gap_mode = true;
    if (std::string(argv[i]) == "--roof") roof_mode = true;
    if (std::string(argv[i]) == "--2b") {
      g_shapes = kShapes2B;
      g_num_shapes = std::uint32_t{sizeof(kShapes2B) / sizeof(kShapes2B[0])};
      g_gap_bf16 = true;
    }
    if (std::string(argv[i]) == "--grid-cap" && i + 1 < argc)
      g_grid_cap = (uint32_t)std::atoi(argv[++i]);
    if (std::string(argv[i]) == "--cand-def" && i + 1 < argc)
      g_cand_defs = std::string(" ") + argv[++i];
    if (std::string(argv[i]) == "--cand" && i + 1 < argc) {
      g_cand_src = argv[++i];
      size_t colon = g_cand_src.rfind(':');
      if (colon != std::string::npos) {
        g_cand_columns = (uint32_t)std::atoi(g_cand_src.c_str() + colon + 1);
        g_cand_src.resize(colon);
      }
    }
  }
  const int reps = quick ? 7 : 21;

  const char* q4_defines =
      "-DUSE_FP16=1 -DQMM_VEC_Q4_WORD=1 -DQMM_VEC_MULTI=1";
  const char* variant_defines = tree_mode
      ? q4_defines
      : bf16_eq ? "-DUSE_BF16=1 -DUSE_SUBGROUP=1 -DQMM_VEC_Q4_WORD=1 "
                "-DQMM_VEC_MULTI=1"
                : "-DUSE_FP16=1 -DUSE_SUBGROUP=1 -DQMM_VEC_Q4_WORD=1 "
                "-DQMM_VEC_MULTI=1";
  struct SideSpec {
    const char* tag;
    const char* src;
    const char* spv;
    uint32_t columns;
  };
  std::vector<SideSpec> specs = {
      {"base", "tools/q4-bw-bench/shaders/qmm_vec_base.comp",
          "/tmp/q4base.spv", 8u},
      {"unroll", "tools/q4-bw-bench/shaders/qmm_vec_cand_unroll.comp",
          "/tmp/q4cand_u.spv", 8u},
      {bf16_eq ? "xpack" : "loadfirst",
          bf16_eq ? "tools/q4-bw-bench/shaders/qmm_vec_cand_xpack.comp"
                  : "tools/q4-bw-bench/shaders/qmm_vec_cand_loadfirst.comp",
          bf16_eq ? "/tmp/q4cand_x.spv" : "/tmp/q4cand_l.spv", 8u},
      {"wg128", "tools/q4-bw-bench/shaders/qmm_vec_cand_wg128.comp",
          "/tmp/q4cand_w.spv", 4u},
      {bf16_eq ? "xpack2" : "xpack",
          bf16_eq ? "tools/q4-bw-bench/shaders/qmm_vec_cand_xpack2.comp"
                  : "tools/q4-bw-bench/shaders/qmm_vec_cand_xpack.comp",
          bf16_eq ? "/tmp/q4cand_x2.spv" : "/tmp/q4cand_x.spv", 8u},
  };
  if (!g_cand_src.empty()) {
    specs.resize(1);
    specs.push_back({"cand", g_cand_src.c_str(), "/tmp/q4cand_c.spv",
        g_cand_columns});
  }
  constexpr int kMaxSides = 5;
  const int num_sides = (int)specs.size();
  if (num_sides > kMaxSides) die("too many sides");
  for (int i = 0; i < num_sides; ++i) {
    // loadfirst cannot compile under bf16 defines (f16-only form).
    if (bf16_eq && std::string(specs[i].tag) == "loadfirst") continue;
    std::string sdefs = std::string(variant_defines) +
        (std::string(specs[i].tag) == "cand" ? g_cand_defs : std::string());
    if (compile_shader(specs[i].src, sdefs.c_str(), specs[i].spv) != 0)
      die("compile %s", specs[i].tag);
  }

  if (vk_init() != 0) return 1;
  DeviceCtx ctx = setup_device();
  std::printf(
      "{\"k\":\"mode\",\"variant\":\"%s\",\"reps\":%d}\n",
      tree_mode ? "tree" : "subgroup", reps);
  if (gap_mode) {
    run_gap_mode(ctx, quick);
    std::printf("{\"k\":\"done\"}\n");
    return 0;
  }
  if (roof_mode) {
    run_roof_mode(ctx, quick, variant_defines);
    std::printf("{\"k\":\"done\"}\n");
    return 0;
  }

  std::vector<Side> sides(num_sides);
  for (int i = 0; i < num_sides; ++i) {
    sides[i].tag = specs[i].tag;
    sides[i].mod = make_module(read_file(specs[i].spv));
    make_pipeline(sides[i]);
  }

  // Input buffers shared by all sides; output buffers distinct per side
  // so the bit-exactness compare sees each side's own writes.
  SetBufs inputs[kNumShapes];
  SetBufs outs[kNumShapes][kMaxSides];
  Params shape_params[kNumShapes];
  uint32_t groups_v[kNumShapes][kMaxSides];
  for (uint32_t s = 0; s < g_num_shapes; ++s) {
    const Shape& sh = g_shapes[s];
    inputs[s].x = make_buf(g_vk.dev, ctx.mp, (uint64_t)sh.k * 2u, false);
    for (uint32_t d = 0; d < sh.dims; ++d) {
      inputs[s].w[d] = make_buf(g_vk.dev, ctx.mp,
          (uint64_t)sh.n[d] * (sh.k / 8u) * 4u, false);
      inputs[s].scales[d] = make_buf(g_vk.dev, ctx.mp,
          (uint64_t)sh.n[d] * (sh.k / 64u) * 2u, false);
      inputs[s].biases[d] = make_buf(g_vk.dev, ctx.mp,
          (uint64_t)sh.n[d] * (sh.k / 64u) * 2u, false);
    }
    for (int i = 0; i < num_sides; ++i) {
      for (uint32_t d = 0; d < kMaxDims; ++d) {
        uint32_t n = d < sh.dims ? sh.n[d] : sh.n[0];
        outs[s][i].out[d] =
            make_buf(g_vk.dev, ctx.mp, (uint64_t)n * 2u, true);
      }
      groups_v[s][i] = shape_groups(sh, specs[i].columns);
    }
    fill_params(shape_params[s], sh, groups_v[s][0]);
  }

  VkDescriptorPoolSize ps{};
  ps.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  ps.descriptorCount = kBindings * kNumShapes * num_sides;
  VkDescriptorPoolCreateInfo dpci{};
  dpci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  dpci.maxSets = kNumShapes * num_sides;
  dpci.poolSizeCount = 1;
  dpci.pPoolSizes = &ps;
  VkDescriptorPool pool;
  if (g_vk.CreateDescriptorPool(g_vk.dev, &dpci, nullptr, &pool) !=
      VK_SUCCESS)
    die("CreateDescriptorPool");
  VkDescriptorSet sets[kNumShapes][kMaxSides];
  for (uint32_t s = 0; s < g_num_shapes; ++s) {
    for (int i = 0; i < num_sides; ++i) {
      SetBufs combined = inputs[s];
      for (uint32_t d = 0; d < kMaxDims; ++d) {
        combined.out[d] = outs[s][i].out[d];
      }
      sets[s][i] =
          make_set(pool, sides[i].dsl, combined, g_shapes[s].dims);
    }
  }

  CmdRes iso_cmd = make_cmd(ctx, 2);

  run_peak(ctx, "tools/q4-bw-bench/shaders/peak_copy.comp",
      "/tmp/peak_copy.spv", "copy", 1u << 24u, quick);
  run_peak(ctx, "tools/q4-bw-bench/shaders/peak_read.comp",
      "/tmp/peak_read.spv", "read", 1u << 24u, quick);

  // ---- Bit-exactness: identical inputs through base and each side ----
  if (true) {
    uint32_t s = 0; const Shape& sh = g_shapes[s];
    std::mt19937 rng(0xABCDu);
    {
      void* p; g_vk.MapMemory(g_vk.dev, inputs[s].x.mem, 0, inputs[s].x.size, 0, &p);
      uint16_t* h = (uint16_t*)p;
      for (size_t e = 0; e < inputs[s].x.size / 2; ++e)
        h[e] = (uint16_t)(0x3F00u | (rng() & 0xFFu));
      g_vk.UnmapMemory(g_vk.dev, inputs[s].x.mem);
    }
    for (uint32_t d = 0; d < sh.dims; ++d) {
      void* p;
      g_vk.MapMemory(g_vk.dev, inputs[s].scales[d].mem, 0, inputs[s].scales[d].size, 0, &p);
      std::memset(p, 0, inputs[s].scales[d].size);
      for (size_t e = 0; e < inputs[s].scales[d].size / 2; ++e) ((uint16_t*)p)[e] = 0x3F80;
      g_vk.UnmapMemory(g_vk.dev, inputs[s].scales[d].mem);
      g_vk.MapMemory(g_vk.dev, inputs[s].biases[d].mem, 0, inputs[s].biases[d].size, 0, &p);
      std::memset(p, 0, inputs[s].biases[d].size);
      g_vk.UnmapMemory(g_vk.dev, inputs[s].biases[d].mem);
    }
  }
  for (uint32_t s = 0; s < g_num_shapes; ++s) {
    const Shape& sh = g_shapes[s];
    for (int i = 0; i < num_sides; ++i) {
      dispatch_isolated(ctx, sides[i], iso_cmd, sets[s][i],
          shape_params[s], groups_v[s][i]);
    }
    // Side 0 is the base; every other side must reproduce its bits.
    for (uint32_t d = 0; d < sh.dims; ++d) {
      void* p;
      g_vk.MapMemory(g_vk.dev, outs[s][0].out[d].mem, 0,
          outs[s][0].out[d].size, 0, &p);
      std::vector<uint16_t> snap(outs[s][0].out[d].size / 2);
      std::memcpy(snap.data(), p, outs[s][0].out[d].size);
      g_vk.UnmapMemory(g_vk.dev, outs[s][0].out[d].mem);
      for (int i = 1; i < num_sides; ++i) {
        g_vk.MapMemory(g_vk.dev, outs[s][i].out[d].mem, 0,
            outs[s][i].out[d].size, 0, &p);
        uint32_t mismatches = 0;
        uint16_t* got = (uint16_t*)p;
        for (size_t e = 0; e < snap.size(); ++e) {
          if (got[e] != snap[e]) ++mismatches;
        }
        g_vk.UnmapMemory(g_vk.dev, outs[s][i].out[d].mem);
        std::printf(
            "{\"k\":\"eq\",\"shape\":\"%s\",\"dim\":%u,\"side\":\"%s\","
            "\"elements\":%zu,\"bit_mismatches\":%u}\n",
            sh.name, d, specs[i].tag, snap.size(), mismatches);
        if (mismatches > 0) {
          std::printf("MM %s %s dim%u:", sh.name, specs[i].tag, d);
          int shown = 0;
          for (size_t e = 0; e < snap.size() && shown < 6; ++e) {
            if (got_e_dump(snap, (uint16_t*)nullptr, e)) continue;
          }
          // simple dump of first 6 mismatching indices
          g_vk.MapMemory(g_vk.dev, outs[s][i].out[d].mem, 0,
              outs[s][i].out[d].size, 0, &p);
          uint16_t* g2 = (uint16_t*)p;
          for (size_t e = 0; e < snap.size() && shown < 6; ++e) {
            if (g2[e] != snap[e]) {
              std::printf(" [%zu] %04x vs %04x", e, g2[e], snap[e]);
              ++shown;
            }
          }
          g_vk.UnmapMemory(g_vk.dev, outs[s][i].out[d].mem);
          std::printf("\n");
        }
      }
    }
  }

  // ---- DUMP for reference check (qkv only) ----
  {
    uint32_t s = 0; const Shape& sh = g_shapes[s];
    auto dumpbuf = [&](const Buf& b, const char* nm) {
      FILE* f = fopen(nm, "wb");
      void* p;
      g_vk.MapMemory(g_vk.dev, b.mem, 0, b.size, 0, &p);
      fwrite(p, 1, b.size, f);
      g_vk.UnmapMemory(g_vk.dev, b.mem);
      fclose(f);
    };
    dumpbuf(inputs[s].x, "/tmp/q4dump_x.bin");
    for (uint32_t d = 0; d < sh.dims; ++d) {
      char nm[64];
      snprintf(nm, 64, "/tmp/q4dump_w%u.bin", d);
      dumpbuf(inputs[s].w[d], nm);
      snprintf(nm, 64, "/tmp/q4dump_s%u.bin", d);
      dumpbuf(inputs[s].scales[d], nm);
      snprintf(nm, 64, "/tmp/q4dump_b%u.bin", d);
      dumpbuf(inputs[s].biases[d], nm);
    }
    for (int i = 0; i < num_sides; ++i)
      for (uint32_t d = 0; d < sh.dims; ++d) {
        char nm[64];
        snprintf(nm, 64, "/tmp/q4dump_o_%s_%u.bin", specs[i].tag, d);
        dumpbuf(outs[s][i].out[d], nm);
      }
    std::printf("{\"k\":\"dumped\"}\n");
  }
  // ---- Isolated per-shape timings, sides interleaved per rep ----
  for (uint32_t s = 0; s < g_num_shapes; ++s) {
    const Shape& sh = g_shapes[s];
    std::vector<uint64_t> med(num_sides), minv(num_sides);
    std::vector<std::vector<uint64_t>> samples(num_sides);
    std::vector<std::vector<uint64_t>> wall(num_sides);
    for (int rep = 0; rep < reps; ++rep) {
      for (int i = 0; i < num_sides; ++i) {
        samples[i].push_back(dispatch_isolated(ctx, sides[i], iso_cmd,
            sets[s][i], shape_params[s], groups_v[s][i]));
        wall[i].push_back(g_wall_ns);
      }
    }
    uint64_t bytes = shape_bytes(sh);
    for (int i = 0; i < num_sides; ++i) {
      med[i] = median(samples[i]);
      minv[i] = min_of(samples[i]);
    }
    std::vector<uint64_t> wmed(num_sides);
    for (int i = 0; i < num_sides; ++i) {
      wmed[i] = median(wall[i]);
    }
    std::printf("{\"k\":\"shape\",\"name\":\"%s\",\"bytes\":%llu",
        sh.name, (unsigned long long)bytes);
    for (int i = 0; i < num_sides; ++i) {
      std::printf(
          ",\"%s\":{\"grid\":%u,\"med_ns\":%llu,\"min_ns\":%llu,"
          "\"ratio_vs_base\":%.4f,\"wall_med_ns\":%llu,"
          "\"wall_gb_s\":%.2f,\"wall_ratio_vs_base\":%.4f}",
          specs[i].tag, groups_v[s][i], (unsigned long long)med[i],
          (unsigned long long)minv[i], (double)med[i] / (double)med[0],
          (unsigned long long)wmed[i], (double)bytes / (double)wmed[i],
          (double)wmed[i] / (double)wmed[0]);
    }
    std::printf("}\n");
  }

  std::printf("{\"k\":\"done\"}\n");
  return 0;
}
