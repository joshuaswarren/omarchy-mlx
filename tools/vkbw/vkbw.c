// vkbw: a standalone Vulkan memory read and copy bandwidth microbenchmark for any Vulkan GPU.
//
// Build: make.   Run: ./vkbw --list, ./vkbw [--device N] [--csv FILE].  See README.md.
//
// One compute dispatch walks a storage buffer once. The shader sums every element (the host checks the sum, so a skipped or
// wrong read cannot pass) and, in copy mode, writes each element to a second buffer. Cases vary the load width (2, 4, 8, 16
// bytes: bf16 / f32 / vec2 / vec4), the distance between neighbouring lanes (coalesced to 1 KiB apart), the buffer size, the
// workgroup size and, where the driver allows it, the subgroup size. GB/s is useful bytes (read, plus written for copy) over the
// wall time of a batch of dispatches in one submission (GPU timestamp queries are not used: they do not measure the work on
// every driver); 1 GB is 1e9 bytes. A pass slower than 2 s is skipped and a submission over 10 s stops the program.
#define _POSIX_C_SOURCE 200809L
#include <time.h>
#include <inttypes.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vulkan/vulkan.h>

#include "shaders.h"

#define MAX_SLOTS 256
#define TARGET_MS 60.0
#define REPEATS 5
#define MAX_SUBMIT_S 10
#define MAX_PASS_S 2.0

static void die(const char* fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  fprintf(stderr, "vkbw: ");
  vfprintf(stderr, fmt, ap);
  fprintf(stderr, "\n");
  va_end(ap);
  exit(1);
}
#define VK(call) do { VkResult r_ = (call); if (r_ != VK_SUCCESS) die("%s failed: %d (%s:%d)", #call, (int)r_, __FILE__, __LINE__); } while (0)

// ---- device state ----
static VkInstance inst;
static VkPhysicalDevice phys;
static VkDevice dev;
static VkQueue queue;
static uint32_t qfam;
static VkPhysicalDeviceMemoryProperties memprops;
static bool has_16bit, has_sgsize;
static uint32_t sg_default, sg_min, sg_max, max_wg_invocations;
static uint64_t max_alloc, max_storage_range;
static VkDescriptorSetLayout dsl;
static VkPipelineLayout pl;
static VkDescriptorPool dpool;
static VkCommandPool cpool;
static VkCommandBuffer cb;
static VkFence fence;

typedef struct { VkBuffer buf; VkDeviceMemory mem; VkDeviceSize size; void* map; } Buf;

static uint32_t find_mem(uint32_t bits, VkMemoryPropertyFlags want, VkMemoryPropertyFlags avoid) {
  for (uint32_t i = 0; i < memprops.memoryTypeCount; i++)
    if ((bits & (1u << i)) && (memprops.memoryTypes[i].propertyFlags & want) == want && !(memprops.memoryTypes[i].propertyFlags & avoid)) return i;
  return UINT32_MAX;
}

static Buf make_buf(VkDeviceSize size, VkBufferUsageFlags usage, bool host) {
  Buf b = {0};
  b.size = size;
  VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = size, .usage = usage, .sharingMode = VK_SHARING_MODE_EXCLUSIVE};
  VK(vkCreateBuffer(dev, &bi, NULL, &b.buf));
  VkMemoryRequirements mr;
  vkGetBufferMemoryRequirements(dev, b.buf, &mr);
  uint32_t t = host ? find_mem(mr.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 0)
                    : find_mem(mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
  if (t == UINT32_MAX && !host) t = find_mem(mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0);  // unified memory: every type is host visible
  if (t == UINT32_MAX) die("no suitable memory type for a %" PRIu64 " byte buffer", (uint64_t)size);
  VkMemoryAllocateInfo ai = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = mr.size, .memoryTypeIndex = t};
  VK(vkAllocateMemory(dev, &ai, NULL, &b.mem));
  VK(vkBindBufferMemory(dev, b.buf, b.mem, 0));
  if (host) VK(vkMapMemory(dev, b.mem, 0, VK_WHOLE_SIZE, 0, &b.map));
  return b;
}

static void free_buf(Buf* b) {
  if (b->map) vkUnmapMemory(dev, b->mem);
  vkDestroyBuffer(dev, b->buf, NULL);
  vkFreeMemory(dev, b->mem, NULL);
  memset(b, 0, sizeof *b);
}

static VkShaderModule make_module(const uint32_t* code, size_t bytes) {
  VkShaderModuleCreateInfo si = {.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize = bytes, .pCode = code};
  VkShaderModule m;
  VK(vkCreateShaderModule(dev, &si, NULL, &m));
  return m;
}

// ---- pipelines ----
typedef struct { VkPipeline p; int w; uint32_t wg, mode, sg; } Pipe;
static Pipe pipes[256];
static int npipes;
static VkShaderModule mod_init, mod_w[17];

static VkPipeline make_pipeline(VkShaderModule m, uint32_t wg, uint32_t mode, uint32_t sg) {
  uint32_t spec[2] = {wg, mode};
  VkSpecializationMapEntry me[2] = {{0, 0, 4}, {1, 4, 4}};
  VkSpecializationInfo sp = {2, me, sizeof spec, spec};
  VkPipelineShaderStageRequiredSubgroupSizeCreateInfo rs = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO, .requiredSubgroupSize = sg};
  VkComputePipelineCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, .layout = pl};
  ci.stage = (VkPipelineShaderStageCreateInfo){.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .pNext = sg ? &rs : NULL, .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = m, .pName = "main", .pSpecializationInfo = &sp};
  VkPipeline p;
  VK(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &ci, NULL, &p));
  return p;
}

static VkPipeline get_pipe(int w, uint32_t wg, uint32_t mode, uint32_t sg) {
  for (int i = 0; i < npipes; i++)
    if (pipes[i].w == w && pipes[i].wg == wg && pipes[i].mode == mode && pipes[i].sg == sg) return pipes[i].p;
  if (npipes == 256) die("pipeline cache full");
  pipes[npipes] = (Pipe){make_pipeline(mod_w[w], wg, mode, sg), w, wg, mode, sg};
  return pipes[npipes++].p;
}

// ---- command helpers ----
static void begin(void) {
  VK(vkResetCommandBuffer(cb, 0));
  VkCommandBufferBeginInfo bi = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
  VK(vkBeginCommandBuffer(cb, &bi));
}
static double last_submit_ns;  // wall time of the last submit_wait, ns
static void submit_wait(void) {
  VK(vkEndCommandBuffer(cb));
  VkSubmitInfo si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &cb};
  VK(vkResetFences(dev, 1, &fence));
  struct timespec t0, t1;
  clock_gettime(CLOCK_MONOTONIC, &t0);
  VK(vkQueueSubmit(queue, 1, &si, fence));
  VkResult r = vkWaitForFences(dev, 1, &fence, VK_TRUE, MAX_SUBMIT_S * 1000000000ull);
  clock_gettime(CLOCK_MONOTONIC, &t1);
  if (r == VK_TIMEOUT) die("a submission ran longer than %d s; stopping before the GPU watchdog does", MAX_SUBMIT_S);
  if (r != VK_SUCCESS) die("vkWaitForFences failed: %d", (int)r);
  last_submit_ns = (double)(t1.tv_sec - t0.tv_sec) * 1e9 + (double)(t1.tv_nsec - t0.tv_nsec);
}
static void barrier_compute(void) {
  VkMemoryBarrier mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER, .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT, .dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT};
  vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
}

// ---- buffers shared by every case ----
static Buf src, dst, res, rb;
static VkDescriptorSet set_fwd, set_rev;  // fwd: src -> dst, rev: dst -> src (to verify a copy)
static uint64_t cur_bytes;

static void write_set(VkDescriptorSet s, VkBuffer a, VkBuffer b, VkBuffer r) {
  VkDescriptorBufferInfo bi[3] = {{a, 0, VK_WHOLE_SIZE}, {b, 0, VK_WHOLE_SIZE}, {r, 0, VK_WHOLE_SIZE}};
  VkWriteDescriptorSet w[3];
  for (int i = 0; i < 3; i++)
    w[i] = (VkWriteDescriptorSet){.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = s, .dstBinding = (uint32_t)i, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &bi[i]};
  vkUpdateDescriptorSets(dev, 3, w, 0, NULL);
}

static void alloc_working_set(uint64_t bytes) {
  if (cur_bytes == bytes) return;
  vkDeviceWaitIdle(dev);
  if (src.buf) { free_buf(&src); free_buf(&dst); }
  src = make_buf(bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, false);
  dst = make_buf(bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, false);
  write_set(set_fwd, src.buf, dst.buf, res.buf);
  write_set(set_rev, dst.buf, src.buf, res.buf);
  cur_bytes = bytes;
}

// ---- reference sums ----
static uint32_t hash32(uint32_t i) {
  uint32_t x = i * 2654435761u;
  x ^= x >> 15;
  x *= 2246822519u;
  x ^= x >> 13;
  return x;
}

typedef struct { uint64_t bytes; uint32_t sum32, sum16; } RefSum;
static RefSum refs[16];
static int nrefs;
static RefSum ref_for(uint64_t bytes) {
  for (int i = 0; i < nrefs; i++) if (refs[i].bytes == bytes) return refs[i];
  uint32_t s32 = 0, s16 = 0;
  for (uint64_t i = 0; i < bytes / 4; i++) {
    uint32_t h = hash32((uint32_t)i);
    s32 += h;
    s16 += (h & 0xffffu) + (h >> 16);
  }
  refs[nrefs] = (RefSum){bytes, s32, s16};
  return refs[nrefs++];
}

// Fill src with hash(i). Set fwd binds src at binding 0, which is where the init shader writes.
static void init_src(uint64_t bytes) {
  VkPipeline pi = make_pipeline(mod_init, 64, 0, 0);
  begin();
  vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pi);
  vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &set_fwd, 0, NULL);
  uint32_t words = (uint32_t)(bytes / 4);
  vkCmdPushConstants(cb, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, 4, &words);
  vkCmdDispatch(cb, 4096, 1, 1);
  submit_wait();
  vkDestroyPipeline(dev, pi, NULL);
}

// Zero dst so a copy that skips or misplaces data cannot pass on leftovers from an earlier case.
static void clear_dst(void) {
  begin();
  vkCmdFillBuffer(cb, dst.buf, 0, VK_WHOLE_SIZE, 0);
  submit_wait();
}

// ---- one case ----
typedef struct { double med, mn, mx, ms; bool ok; } Result;

static int cmp_d(const void* a, const void* b) { double x = *(const double*)a, y = *(const double*)b; return (x > y) - (x < y); }

static uint32_t read_res(uint32_t n, uint32_t* out) {
  memcpy(out, rb.map, n * 4);
  return n;
}

// Time K dispatches of one pipeline in one submission; returns GPU nanoseconds and leaves the K checksums in rb.
static double run_batch(VkPipeline p, VkDescriptorSet set, uint32_t groups, const uint32_t pc_base[4], uint32_t K) {
  begin();
  vkCmdFillBuffer(cb, res.buf, 0, MAX_SLOTS * 4, 0);
  VkMemoryBarrier mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER, .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT, .dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT};
  vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
  vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, p);
  vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &set, 0, NULL);
  for (uint32_t i = 0; i < K; i++) {
    uint32_t pc[5] = {pc_base[0], pc_base[1], pc_base[2], pc_base[3], i};
    vkCmdPushConstants(cb, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof pc, pc);
    vkCmdDispatch(cb, groups, 1, 1);
    barrier_compute();
  }
  VkBufferCopy bc = {0, 0, MAX_SLOTS * 4};
  VkMemoryBarrier mb2 = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER, .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT, .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT};
  vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &mb2, 0, NULL, 0, NULL);
  vkCmdCopyBuffer(cb, res.buf, rb.buf, 1, &bc);
  VkMemoryBarrier mb3 = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER, .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT, .dstAccessMask = VK_ACCESS_HOST_READ_BIT};
  vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &mb3, 0, NULL, 0, NULL);
  submit_wait();
  return last_submit_ns;
}

static bool all_equal(uint32_t n, uint32_t want) {
  uint32_t v[MAX_SLOTS];
  read_res(n, v);
  for (uint32_t i = 0; i < n; i++) if (v[i] != want) return false;
  return true;
}

// op 0 = read, 1 = copy. stride_bytes = distance between neighbouring lanes (0 or w = coalesced).
static bool run_case(int op, int w, uint32_t wg, uint32_t sg, uint64_t bytes, uint32_t stride_bytes, Result* out) {
  uint64_t N = bytes / w;  // elements
  uint32_t s = stride_bytes <= (uint32_t)w ? 1 : stride_bytes / (uint32_t)w;
  uint64_t grid = N / 4 < (1u << 20) ? N / 4 : (1u << 20);
  if (grid < wg || N / s < 1 || N > (1ull << 31)) return false;
  uint32_t iters = (uint32_t)(N / grid);
  uint32_t log2_rows = 0;
  while ((1ull << (log2_rows + 1)) <= N / s) log2_rows++;
  uint32_t pc[4] = {log2_rows, s, iters, (uint32_t)grid};
  VkPipeline p = get_pipe(w, wg, (uint32_t)op, sg);
  RefSum ref = ref_for(bytes);
  uint32_t want = w == 2 ? ref.sum16 : ref.sum32;
  uint32_t groups = (uint32_t)(grid / wg);

  if (op == 1) clear_dst();
  double ns1 = run_batch(p, set_fwd, groups, pc, 1);  // warm up
  if (!all_equal(1, want)) { out->ok = false; return true; }
  if (ns1 > MAX_PASS_S * 1e9) { fprintf(stderr, "vkbw: skipped a case: one pass took %.1f s\n", ns1 / 1e9); return false; }
  double ns5 = run_batch(p, set_fwd, groups, pc, 5);
  double per_pass = ns5 > ns1 ? (ns5 - ns1) / 4 : ns5 / 5;  // the difference removes the fixed submit cost
  uint32_t K = (uint32_t)(TARGET_MS * 1e6 / (per_pass > 1 ? per_pass : 1));
  K = K < 1 ? 1 : (K > MAX_SLOTS ? MAX_SLOTS : K);
  run_batch(p, set_fwd, groups, pc, K);  // clocks and caches settle, not timed
  double gbs[REPEATS];
  double ns_last = 0;
  bool ok = true;
  for (int r = 0; r < REPEATS; r++) {
    double ns = run_batch(p, set_fwd, groups, pc, K);
    ok = ok && all_equal(K, want);
    gbs[r] = (double)(op == 1 ? 2 * bytes : bytes) * K / ns;  // bytes per ns is GB/s
    ns_last = ns / K;
  }
  if (op == 1) {  // the copy must have reproduced the source: read dst back with the 4-byte kernel
    uint32_t vpc[4] = {0, 1, 0, 0};
    uint64_t n4 = bytes / 4, g4 = n4 / 4 < (1u << 20) ? n4 / 4 : (1u << 20);
    while ((1ull << (vpc[0] + 1)) <= n4) vpc[0]++;
    vpc[2] = (uint32_t)(n4 / g4);
    vpc[3] = (uint32_t)g4;
    run_batch(get_pipe(4, 128, 0, 0), set_rev, (uint32_t)(g4 / 128), vpc, 1);
    ok = ok && all_equal(1, ref.sum32);
  }
  qsort(gbs, REPEATS, sizeof gbs[0], cmp_d);
  out->med = gbs[REPEATS / 2];
  out->mn = gbs[0];
  out->mx = gbs[REPEATS - 1];
  out->ms = ns_last / 1e6;
  out->ok = ok;
  return true;
}

// ---- setup ----
static const char* dev_type(VkPhysicalDeviceType t) {
  switch (t) {
    case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: return "integrated";
    case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU: return "discrete";
    case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU: return "virtual";
    case VK_PHYSICAL_DEVICE_TYPE_CPU: return "cpu";
    default: return "other";
  }
}

static bool has_ext(VkPhysicalDevice d, const char* name) {
  uint32_t n = 0;
  vkEnumerateDeviceExtensionProperties(d, NULL, &n, NULL);
  VkExtensionProperties* e = calloc(n, sizeof *e);
  vkEnumerateDeviceExtensionProperties(d, NULL, &n, e);
  bool f = false;
  for (uint32_t i = 0; i < n; i++) if (!strcmp(e[i].extensionName, name)) f = true;
  free(e);
  return f;
}

static void parse_list(const char* s, uint32_t* out, int* n, int max) {
  *n = 0;
  char* c = strdup(s);
  for (char* t = strtok(c, ","); t && *n < max; t = strtok(NULL, ",")) out[(*n)++] = (uint32_t)strtoul(t, NULL, 10);
  free(c);
}

int main(int argc, char** argv) {
  int want_dev = -1;
  bool list = false, quick = false;
  const char* csv_path = NULL;
  uint64_t max_mib = 512;
  uint32_t sizes_mib[8] = {4, 64, 512};
  int nsizes = 3;
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "--list")) list = true;
    else if (!strcmp(argv[i], "--quick")) quick = true;
    else if (!strcmp(argv[i], "--device") && i + 1 < argc) want_dev = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--csv") && i + 1 < argc) csv_path = argv[++i];
    else if (!strcmp(argv[i], "--max-mib") && i + 1 < argc) max_mib = strtoull(argv[++i], NULL, 10);
    else if (!strcmp(argv[i], "--sizes-mib") && i + 1 < argc) parse_list(argv[++i], sizes_mib, &nsizes, 8);
    else { fprintf(stderr, "usage: vkbw [--list] [--device N] [--csv FILE] [--quick] [--max-mib N] [--sizes-mib a,b,c]\n"); return 2; }
  }
  for (int i = 0; i < nsizes; i++) if (!sizes_mib[i] || (sizes_mib[i] & (sizes_mib[i] - 1))) die("--sizes-mib values must be powers of two");
  VkApplicationInfo ai = {.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO, .pApplicationName = "vkbw", .apiVersion = VK_API_VERSION_1_1};
  uint32_t nie = 0;
  vkEnumerateInstanceExtensionProperties(NULL, &nie, NULL);
  VkExtensionProperties* ie = calloc(nie, sizeof *ie);
  vkEnumerateInstanceExtensionProperties(NULL, &nie, ie);
  const char* iexts[1] = {"VK_KHR_portability_enumeration"};  // MoltenVK and other layered drivers list their device only with this
  bool portability = false;
  for (uint32_t i = 0; i < nie; i++) if (!strcmp(ie[i].extensionName, iexts[0])) portability = true;
  free(ie);
  VkInstanceCreateInfo ici = {.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pApplicationInfo = &ai, .flags = portability ? VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR : 0,
                              .enabledExtensionCount = portability ? 1u : 0u, .ppEnabledExtensionNames = iexts};
  VK(vkCreateInstance(&ici, NULL, &inst));
  uint32_t nd = 0;
  VK(vkEnumeratePhysicalDevices(inst, &nd, NULL));
  if (!nd) die("no Vulkan device");
  VkPhysicalDevice* pd = calloc(nd, sizeof *pd);
  VK(vkEnumeratePhysicalDevices(inst, &nd, pd));
  int pick = -1;
  for (uint32_t i = 0; i < nd; i++) {
    VkPhysicalDeviceProperties pp;
    vkGetPhysicalDeviceProperties(pd[i], &pp);
    if (list) printf("%u: %s (%s) api %u.%u.%u\n", i, pp.deviceName, dev_type(pp.deviceType), VK_VERSION_MAJOR(pp.apiVersion), VK_VERSION_MINOR(pp.apiVersion), VK_VERSION_PATCH(pp.apiVersion));
    if (pick < 0 && pp.deviceType != VK_PHYSICAL_DEVICE_TYPE_CPU) pick = (int)i;
  }
  if (list) return 0;
  if (want_dev >= 0) { if ((uint32_t)want_dev >= nd) die("no device %d", want_dev); pick = want_dev; }
  if (pick < 0) pick = 0;
  phys = pd[pick];

  VkPhysicalDeviceSubgroupSizeControlProperties sgcp = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES};
  VkPhysicalDeviceSubgroupProperties sgp = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES, .pNext = &sgcp};
  VkPhysicalDeviceMaintenance3Properties m3 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_3_PROPERTIES, .pNext = &sgp};
  VkPhysicalDeviceDriverProperties drv = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES, .pNext = &m3};
  VkPhysicalDeviceProperties2 p2 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, .pNext = &drv};
  bool ext_sg = has_ext(phys, "VK_EXT_subgroup_size_control");
  if (!ext_sg) sgp.pNext = NULL;
  vkGetPhysicalDeviceProperties2(phys, &p2);
  vkGetPhysicalDeviceMemoryProperties(phys, &memprops);
  VkPhysicalDevice16BitStorageFeatures f16 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES};
  VkPhysicalDeviceSubgroupSizeControlFeatures fsg = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES, .pNext = &f16};
  VkPhysicalDeviceFeatures2 f2 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, .pNext = ext_sg ? (void*)&fsg : (void*)&f16};
  vkGetPhysicalDeviceFeatures2(phys, &f2);
  has_16bit = f16.storageBuffer16BitAccess == VK_TRUE && f2.features.shaderInt16 == VK_TRUE;
  has_sgsize = ext_sg && fsg.subgroupSizeControl == VK_TRUE;
  sg_default = sgp.subgroupSize;
  sg_min = has_sgsize ? sgcp.minSubgroupSize : sg_default;
  sg_max = has_sgsize ? sgcp.maxSubgroupSize : sg_default;
  max_wg_invocations = p2.properties.limits.maxComputeWorkGroupInvocations;
  if (p2.properties.limits.maxComputeWorkGroupSize[0] < max_wg_invocations) max_wg_invocations = p2.properties.limits.maxComputeWorkGroupSize[0];  // one dimension is used
  max_alloc = m3.maxMemoryAllocationSize;
  max_storage_range = p2.properties.limits.maxStorageBufferRange;

  uint32_t nq = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(phys, &nq, NULL);
  VkQueueFamilyProperties* qf = calloc(nq, sizeof *qf);
  vkGetPhysicalDeviceQueueFamilyProperties(phys, &nq, qf);
  qfam = UINT32_MAX;
  for (uint32_t i = 0; i < nq; i++) if (qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { qfam = i; break; }
  if (qfam == UINT32_MAX) die("no compute queue");

  float prio = 1.0f;
  VkDeviceQueueCreateInfo dq = {.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, .queueFamilyIndex = qfam, .queueCount = 1, .pQueuePriorities = &prio};
  f16.pNext = NULL;
  fsg.pNext = &f16;
  const char* exts[2];
  uint32_t next = 0;
  if (has_sgsize) exts[next++] = "VK_EXT_subgroup_size_control";
  if (has_ext(phys, "VK_KHR_portability_subset")) exts[next++] = "VK_KHR_portability_subset";
  VkPhysicalDeviceFeatures enabled = {.shaderInt16 = has_16bit ? VK_TRUE : VK_FALSE};
  VkDeviceCreateInfo dci = {.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, .pNext = has_sgsize ? (void*)&fsg : (void*)&f16, .queueCreateInfoCount = 1, .pQueueCreateInfos = &dq,
                            .enabledExtensionCount = next, .ppEnabledExtensionNames = exts, .pEnabledFeatures = &enabled};
  f16.storageBuffer16BitAccess = has_16bit ? VK_TRUE : VK_FALSE;
  fsg.subgroupSizeControl = has_sgsize ? VK_TRUE : VK_FALSE;
  VK(vkCreateDevice(phys, &dci, NULL, &dev));
  vkGetDeviceQueue(dev, qfam, 0, &queue);

  // layouts, pools, buffers
  VkDescriptorSetLayoutBinding lb[3];
  for (int i = 0; i < 3; i++) lb[i] = (VkDescriptorSetLayoutBinding){.binding = (uint32_t)i, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT};
  VkDescriptorSetLayoutCreateInfo dli = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, .bindingCount = 3, .pBindings = lb};
  VK(vkCreateDescriptorSetLayout(dev, &dli, NULL, &dsl));
  VkPushConstantRange pr = {VK_SHADER_STAGE_COMPUTE_BIT, 0, 20};
  VkPipelineLayoutCreateInfo pli = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .setLayoutCount = 1, .pSetLayouts = &dsl, .pushConstantRangeCount = 1, .pPushConstantRanges = &pr};
  VK(vkCreatePipelineLayout(dev, &pli, NULL, &pl));
  VkDescriptorPoolSize ps = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 6};
  VkDescriptorPoolCreateInfo dpi = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, .maxSets = 2, .poolSizeCount = 1, .pPoolSizes = &ps};
  VK(vkCreateDescriptorPool(dev, &dpi, NULL, &dpool));
  VkDescriptorSetLayout both[2] = {dsl, dsl};
  VkDescriptorSet sets[2];
  VkDescriptorSetAllocateInfo dai = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, .descriptorPool = dpool, .descriptorSetCount = 2, .pSetLayouts = both};
  VK(vkAllocateDescriptorSets(dev, &dai, sets));
  set_fwd = sets[0];
  set_rev = sets[1];
  VkCommandPoolCreateInfo cpi = {.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, .queueFamilyIndex = qfam};
  VK(vkCreateCommandPool(dev, &cpi, NULL, &cpool));
  VkCommandBufferAllocateInfo cai = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, .commandPool = cpool, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1};
  VK(vkAllocateCommandBuffers(dev, &cai, &cb));
  VkFenceCreateInfo fi = {.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
  VK(vkCreateFence(dev, &fi, NULL, &fence));
  res = make_buf(MAX_SLOTS * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, false);
  rb = make_buf(MAX_SLOTS * 4, VK_BUFFER_USAGE_TRANSFER_DST_BIT, true);
  mod_init = make_module(spv_init, sizeof spv_init);
  mod_w[2] = make_module(spv_w2, sizeof spv_w2);
  mod_w[4] = make_module(spv_w4, sizeof spv_w4);
  mod_w[8] = make_module(spv_w8, sizeof spv_w8);
  mod_w[16] = make_module(spv_w16, sizeof spv_w16);

  // largest working set: one buffer, capped by the device, by an eighth of the largest device-local heap, and by --max-mib
  uint64_t heap = 0;
  for (uint32_t i = 0; i < memprops.memoryHeapCount; i++)
    if ((memprops.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) && memprops.memoryHeaps[i].size > heap) heap = memprops.memoryHeaps[i].size;
  uint64_t cap = max_mib << 20;
  if (cap > max_storage_range) cap = max_storage_range;
  if (cap > max_alloc) cap = max_alloc;
  if (heap && cap > heap / 8) cap = heap / 8;

  FILE* csv = csv_path ? fopen(csv_path, "w") : NULL;
  if (csv_path && !csv) die("cannot write %s", csv_path);
#define BOTH(...) do { printf(__VA_ARGS__); if (csv) { fprintf(csv, __VA_ARGS__); fflush(csv); } } while (0)
  BOTH("# vkbw device: %s (%s)\n", p2.properties.deviceName, dev_type(p2.properties.deviceType));
  BOTH("# driver: %s %s | api %u.%u.%u\n", drv.driverName, drv.driverInfo, VK_VERSION_MAJOR(p2.properties.apiVersion), VK_VERSION_MINOR(p2.properties.apiVersion), VK_VERSION_PATCH(p2.properties.apiVersion));
  BOTH("# vendor 0x%04x device 0x%04x | subgroup default %u, controllable %s (%u..%u) | max workgroup invocations %u\n", p2.properties.vendorID, p2.properties.deviceID, sg_default, has_sgsize ? "yes" : "no", sg_min, sg_max, max_wg_invocations);
  BOTH("# storage buffer 16-bit access: %s | largest buffer used %" PRIu64 " MiB\n", has_16bit ? "yes" : "no (bf16 width skipped)", cap >> 20);
  BOTH("# GB/s is 1e9 bytes per second of useful bytes (read, plus written for copy), from wall time; stride_bytes is the distance between neighbouring lanes\n");
  BOTH("suite,op,width_bytes,wg,subgroup,size_mib,stride_bytes,gbs_median,gbs_min,gbs_max,ms_per_pass,check\n");

  uint64_t biggest = 0;
  for (int i = 0; i < nsizes; i++) { uint64_t b = (uint64_t)sizes_mib[i] << 20; if (b <= cap && b > biggest) biggest = b; }
  if (!biggest) die("no requested size fits under the %" PRIu64 " MiB cap", cap >> 20);

  int widths[4] = {2, 4, 8, 16};
  uint32_t strides[6] = {0, 32, 64, 128, 256, 1024};
  int nstr = quick ? 2 : 6;
  for (int si = 0; si < nsizes; si++) {
    uint64_t bytes = (uint64_t)sizes_mib[si] << 20;
    if (bytes > cap) continue;
    alloc_working_set(bytes);
    init_src(bytes);
    for (int op = 0; op < 2; op++)
      for (int wi = 0; wi < 4; wi++) {
        int w = widths[wi];
        if (w == 2 && !has_16bit) continue;
        for (int sti = 0; sti < nstr; sti++) {
          Result r = {0};
          if (!run_case(op, w, 128, 0, bytes, strides[sti] ? strides[sti] : (uint32_t)w, &r)) continue;
          BOTH("A,%s,%d,128,default,%" PRIu64 ",%u,%.2f,%.2f,%.2f,%.3f,%s\n", op ? "copy" : "read", w, bytes >> 20, strides[sti] ? strides[sti] : (uint32_t)w, r.med, r.mn, r.mx, r.ms, r.ok ? "ok" : "BAD");
          fflush(stdout);
        }
      }
  }
  // workgroup and subgroup size sweep: coalesced reads at the largest size
  alloc_working_set(biggest);
  init_src(biggest);
  uint32_t wgs[5] = {32, 64, 128, 256, 512};
  for (int wi = 0; wi < 3; wi++) {
    int w = (int[]){2, 4, 16}[wi];
    if (w == 2 && !has_16bit) continue;
    for (int gi = 0; gi < 5; gi++) {
      uint32_t wg = wgs[gi];
      if (wg > max_wg_invocations) continue;
      uint32_t sgs[8];
      int nsg = 0;
      sgs[nsg++] = 0;
      if (has_sgsize) for (uint32_t s = sg_min; s <= sg_max && nsg < 8; s *= 2) if (s <= wg && wg % s == 0) sgs[nsg++] = s;
      for (int k = 0; k < nsg; k++) {
        Result r = {0};
        if (!run_case(0, w, wg, sgs[k], biggest, (uint32_t)w, &r)) continue;
        char sgname[16];
        if (sgs[k]) snprintf(sgname, sizeof sgname, "%u", sgs[k]); else snprintf(sgname, sizeof sgname, "default");
        BOTH("B,read,%d,%u,%s,%" PRIu64 ",%d,%.2f,%.2f,%.2f,%.3f,%s\n", w, wg, sgname, biggest >> 20, w, r.med, r.mn, r.mx, r.ms, r.ok ? "ok" : "BAD");
        fflush(stdout);
      }
    }
  }
  if (csv) fclose(csv);
  vkDeviceWaitIdle(dev);
  return 0;
}
