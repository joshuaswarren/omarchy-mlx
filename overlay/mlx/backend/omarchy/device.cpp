// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

#include "mlx/backend/omarchy/device.h"

#include "mlx/backend/omarchy/compute.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <atomic>
#include <unistd.h>
#include <sys/syscall.h>
#include <memory>
#include <mutex>
#include <stdexcept>

#include "mlx/backend/omarchy/vulkan.h"

#include "mlx/backend/omarchy/capability_sim.h"
#include "mlx/backend/omarchy/allocator.h"

namespace mlx::core::omarchy {

namespace {
constexpr uint32_t kVulkan13 = VK_API_VERSION_1_3;

// MLX_OMARCHY_NO_BUFFER_CACHE (diagnostic, docs/install-omarchy.md).
// Declared here because Runtime::init_impl sets it; the exported
// accessor lives at omarchy scope below.
std::atomic<bool> g_no_buffer_cache{false};

int32_t to_lower_ascii(char c) {
  return static_cast<int32_t>(std::tolower(static_cast<unsigned char>(c)));
}

bool contains_case_insensitive(const char* haystack, const char* needle) {
  const size_t nlen = std::strlen(needle);
  const size_t hlen = std::strlen(haystack);
  if (nlen == 0 || nlen > hlen) {
    return false;
  }
  for (size_t i = 0; i + nlen <= hlen; ++i) {
    size_t j = 0;
    while (j < nlen &&
           to_lower_ascii(haystack[i + j]) == to_lower_ascii(needle[j])) {
      ++j;
    }
    if (j == nlen) {
      return true;
    }
  }
  return false;
}

const char* driver_id_name(int32_t id) {
  // Driver-id constants are plain enum values. kMesaHoneykrispDriverId is
  // spelled out in device.h so Vulkan 1.3 headers (which predate the enum
  // entry) compile; unknown ids fall through to the default.
  switch (id) {
    case kMesaHoneykrispDriverId:
      return "Mesa Honeykrisp";
    case VK_DRIVER_ID_MESA_RADV:
      return "Mesa RADV";
    case VK_DRIVER_ID_MESA_LLVMPIPE:
      return "Mesa llvmpipe";
    case VK_DRIVER_ID_NVIDIA_PROPRIETARY:
      return "NVIDIA proprietary";
    case VK_DRIVER_ID_AMD_OPEN_SOURCE:
      return "AMD open source";
    default:
      return nullptr;
  }
}

int env_index(const char* name) {
  const char* v = std::getenv(name);
  if (!v) {
    return -1;
  }
  char* end = nullptr;
  long parsed = std::strtol(v, &end, 10);
  if (end == v || parsed < 0 || parsed > 1024) {
    return -1;
  }
  return static_cast<int>(parsed);
}

struct PhysicalDeviceInfo {
  VkPhysicalDevice handle{VK_NULL_HANDLE};
  DeviceSupport support;
  // Dispatch/provenance view; a named simulation profile when
  // MLX_OMARCHY_CAPS_SIM is active, otherwise identical to hardware.
  CapabilityReport caps;
  // Hardware truth from discovery, never overridden by simulation.
  CapabilityReport hardware;
};

// Process-wide Vulkan state. The VkInstance lives as long as the process;
// VkDevices are created lazily per used index.
std::atomic<bool> g_runtime_destroyed{false};

struct Runtime {
  std::mutex mutex;
  VkInstance instance{VK_NULL_HANDLE};
  std::vector<PhysicalDeviceInfo> supported;
  std::vector<std::unique_ptr<Device>> devices;
  std::string error{"backend not initialized"};
  bool ready{false};
  bool probed{false};
  bool allow_non_apple{false};
  int preferred_device_index{-1};

  // Discover devices once per process. Never throws; failures are recorded
  // in |error| so callers can surface exact reasons.
  bool init() {
    std::lock_guard<std::mutex> lk(mutex);
    if (probed) {
      return ready;
    }
    probed = true;
    try {
      return init_impl();
    } catch (const std::exception& ex) {
      error = ex.what();
      if (instance != VK_NULL_HANDLE) {
        if (auto& it = vk::instance_table(); it.DestroyInstance) {
          it.DestroyInstance(instance, nullptr);
        }
        instance = VK_NULL_HANDLE;
      }
      return false;
    }
  }

  bool init_impl();

  ~Runtime() {
    g_runtime_destroyed.store(true, std::memory_order_release);
    devices.clear();
    if (instance != VK_NULL_HANDLE) {
      if (auto& it = vk::instance_table(); it.DestroyInstance) {
        it.DestroyInstance(instance, nullptr);
      }
    }
  }
};

Runtime& runtime() {
  static Runtime rt;
  return rt;
}


CapabilityReport collect_capabilities(
    vk::InstanceTable& it,
    VkPhysicalDevice pd,
    const DeviceSupport& support,
    const VkPhysicalDeviceProperties2& props2,
    const VkPhysicalDeviceMemoryProperties2& mem2,
    const VkPhysicalDeviceFeatures2& feats2,
    const VkPhysicalDeviceVulkan12Features& f12,
    const VkPhysicalDeviceVulkan13Features& f13,
    const VkPhysicalDeviceShaderAtomicFloatFeaturesEXT& fa,
    const VkPhysicalDeviceCooperativeMatrixFeaturesKHR& cm,
    const VkPhysicalDevice16BitStorageFeatures& f16,
    const VkPhysicalDeviceMaintenance3Properties& m3,
    const VkPhysicalDeviceMaintenance4Properties& m4,
    const VkPhysicalDeviceSubgroupProperties& subgroup) {
  CapabilityReport caps;
  caps.subgroup_size = subgroup.subgroupSize;
  caps.subgroup_operations = subgroup.supportedOperations;
  const auto& props = props2.properties;
  const auto& limits = props.limits;
  caps.device_name = props.deviceName;
  caps.vendor_id = props.vendorID;
  caps.device_id = props.deviceID;
  caps.driver_version = props.driverVersion;
  caps.api_version = props.apiVersion;
  caps.driver_id = support.driver_id;
  if (const char* name = driver_id_name(support.driver_id)) {
    caps.driver_name = name;
  } else {
    caps.driver_name = "driver id " + std::to_string(support.driver_id);
  }
  std::memcpy(
      caps.pipeline_cache_uuid.data(), props.pipelineCacheUUID, VK_UUID_SIZE);

  caps.unified_memory = false;
  size_t device_local = 0;
  size_t host_visible_heap = 0;
  for (uint32_t i = 0; i < mem2.memoryProperties.memoryHeapCount; ++i) {
    const auto& heap = mem2.memoryProperties.memoryHeaps[i];
    if (heap.flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
      device_local = std::max(device_local, static_cast<size_t>(heap.size));
    }
  }
  for (uint32_t i = 0; i < mem2.memoryProperties.memoryTypeCount; ++i) {
    const auto& type = mem2.memoryProperties.memoryTypes[i];
    bool host_visible_coherent =
        (type.propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) &&
        (type.propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (host_visible_coherent) {
      caps.host_visible_coherent = true;
      if (mem2.memoryProperties.memoryHeaps[type.heapIndex].flags &
          VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
        caps.unified_memory = true;
        break;
      }
    }
  }
  // total_memory must be the memory the allocator can actually use: every
  // buffer is allocated from a HOST_VISIBLE type, so the usable size is the
  // largest heap backing any HOST_VISIBLE type. The largest DEVICE_LOCAL
  // heap is the wrong measure on Apple-UMA drivers (Honeykrisp on T8103
  // reports a ~16 MB device-local heap alongside the multi-GB unified
  // heap), which collapsed memory_limit_ and the batch byte budget
  // (receipt 2026-09-23-m1host-decode-gap-batchbudget, addendum 2).
  for (uint32_t i = 0; i < mem2.memoryProperties.memoryTypeCount; ++i) {
    const auto& type = mem2.memoryProperties.memoryTypes[i];
    if (type.propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) {
      host_visible_heap =
          std::max(host_visible_heap, static_cast<size_t>(
              mem2.memoryProperties.memoryHeaps[type.heapIndex].size));
    }
  }
  caps.total_memory = host_visible_heap > 0 ? host_visible_heap : device_local;

  caps.timeline_semaphore = f12.timelineSemaphore == VK_TRUE;
  caps.shader_float16 = f12.shaderFloat16 == VK_TRUE;
  caps.shader_int16 = feats2.features.shaderInt16 == VK_TRUE;
  caps.shader_int64 = feats2.features.shaderInt64 == VK_TRUE;
  caps.storage_buffer_16bit_access = f16.storageBuffer16BitAccess == VK_TRUE;

  caps.max_allocation_size = m3.maxMemoryAllocationSize;
  caps.max_buffer_size = m4.maxBufferSize;
  caps.max_storage_buffer_range = limits.maxStorageBufferRange;
  caps.max_per_stage_descriptor_storage_buffers =
      limits.maxPerStageDescriptorStorageBuffers;
  caps.max_descriptor_set_storage_buffers =
      limits.maxDescriptorSetStorageBuffers;
  caps.max_compute_work_group_invocations =
      limits.maxComputeWorkGroupInvocations;
  caps.max_compute_shared_memory_size = limits.maxComputeSharedMemorySize;
  for (int i = 0; i < 3; ++i) {
    caps.max_compute_work_group_size[i] = limits.maxComputeWorkGroupSize[i];
  }
  caps.timestamp_period = limits.timestampPeriod;

  uint32_t family_count = 0;
  it.GetPhysicalDeviceQueueFamilyProperties(pd, &family_count, nullptr);
  std::vector<VkQueueFamilyProperties> families(family_count);
  it.GetPhysicalDeviceQueueFamilyProperties(pd, &family_count, families.data());
  for (uint32_t f = 0; f < family_count; ++f) {
    if (families[f].queueFlags & VK_QUEUE_COMPUTE_BIT) {
      caps.queue_family_index = f;
      caps.queue_count = families[f].queueCount;
      caps.queue_timestamp_valid_bits = families[f].timestampValidBits;
      caps.queue_timestamp_valid_bits = families[f].timestampValidBits;
      break;
    }
  }
  // Extension-gated features: the extension name must be listed AND the
  // feature bit must be on. llvmpipe advertises VK_EXT_shader_atomic_float
  // and clears the float32 buffer-add bit only when it truly lacks it;
  // Honeykrisp lists VK_KHR_cooperative_matrix only on the
  // honeykrisp-coopmat branch behind AGX_SIMDMAT, and stock Mesa 26.1.7
  // does not list it at all.
  caps.shader_atomic_float_add = false;
  caps.cooperative_matrix_f32_8 = false;
  bool has_coopmat_ext = false;
  uint32_t ext_count = 0;
  if (it.EnumerateDeviceExtensionProperties &&
      it.EnumerateDeviceExtensionProperties(pd, nullptr, &ext_count, nullptr) ==
          VK_SUCCESS &&
      ext_count > 0) {
    std::vector<VkExtensionProperties> exts(ext_count);
    if (it.EnumerateDeviceExtensionProperties(
            pd, nullptr, &ext_count, exts.data()) == VK_SUCCESS) {
      for (const auto& e : exts) {
        if (std::strcmp(e.extensionName, "VK_EXT_shader_atomic_float") == 0) {
          caps.shader_atomic_float_add =
              fa.shaderBufferFloat32AtomicAdd == VK_TRUE;
        } else if (
            std::strcmp(
                e.extensionName, VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME) ==
            0) {
          has_coopmat_ext = cm.cooperativeMatrix == VK_TRUE;
        }
      }
    }
  }
  // The matmul kernel is written for exactly one shape: 8x8x8, all fp32,
  // subgroup scope, non-saturating.
  if (has_coopmat_ext && it.GetPhysicalDeviceCooperativeMatrixPropertiesKHR) {
    uint32_t n = 0;
    if (it.GetPhysicalDeviceCooperativeMatrixPropertiesKHR(pd, &n, nullptr) ==
            VK_SUCCESS &&
        n > 0) {
      std::vector<VkCooperativeMatrixPropertiesKHR> shapes(
          n, {VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR});
      if (it.GetPhysicalDeviceCooperativeMatrixPropertiesKHR(
              pd, &n, shapes.data()) == VK_SUCCESS) {
        for (const auto& p : shapes) {
          if (p.MSize == 8 && p.NSize == 8 && p.KSize == 8 &&
              p.AType == VK_COMPONENT_TYPE_FLOAT32_KHR &&
              p.BType == VK_COMPONENT_TYPE_FLOAT32_KHR &&
              p.CType == VK_COMPONENT_TYPE_FLOAT32_KHR &&
              p.ResultType == VK_COMPONENT_TYPE_FLOAT32_KHR &&
              p.saturatingAccumulation == VK_FALSE &&
              p.scope == VK_SCOPE_SUBGROUP_KHR) {
            caps.cooperative_matrix_f32_8 = true;
            break;
          }
        }
      }
    }
  }
  return caps;
}

bool Runtime::init_impl() {
  // Capability simulation resolves once per process. An unknown
  // MLX_OMARCHY_CAPS_SIM name throws here and lands in |error|: the
  // backend refuses to start rather than silently running real caps.
  const std::string sim_profile = capsim::requested_profile();
  const capsim::SimulationProfile* sim =
      sim_profile.empty() ? nullptr : capsim::find(sim_profile);
  if (sim) {
    std::fprintf(
        stderr,
        "[omarchy] CAPABILITY SIMULATION ACTIVE: profile '%s'"
        " (stands in for driver_variant %s)\n",
        sim->name,
        sim->represents_driver_variant);
    std::fprintf(
        stderr,
        "[omarchy] simulated runs are not hardware results: no"
        " benchmark or generated-id-digest evidence may be recorded"
        " from this process.\n");
  }
  allow_non_apple = env_flag("MLX_OMARCHY_ALLOW_NON_APPLE");
  preferred_device_index = env_index("MLX_OMARCHY_DEVICE_INDEX");
  // MLX_OMARCHY_NO_BUFFER_CACHE (diagnostic, docs/install-omarchy.md):
  // read once at init, not per allocation - the allocator polls the
  // flag on every malloc and free.
  g_no_buffer_cache.store(
      env_flag("MLX_OMARCHY_NO_BUFFER_CACHE"), std::memory_order_relaxed);
  if (g_no_buffer_cache.load(std::memory_order_relaxed)) {
    std::fprintf(
        stderr,
        "[omarchy] MLX_OMARCHY_NO_BUFFER_CACHE active (diagnostics only,"
        " not product configuration; docs/install-omarchy.md): the"
        " buffer cache is off for the whole process.\n");
  }

  if (!vk::load_loader()) {
    error =
        "[omarchy] Vulkan loader not found (libvulkan.so.1)."
        " Install vulkan-icd-loader or the distribution equivalent.";
    return false;
  }
  auto& it = vk::instance_table();
  it.CreateInstance = reinterpret_cast<PFN_vkCreateInstance>(
      vk::GetInstanceProcAddr(nullptr, "vkCreateInstance"));
  it.EnumerateInstanceVersion =
      reinterpret_cast<PFN_vkEnumerateInstanceVersion>(
          vk::GetInstanceProcAddr(nullptr, "vkEnumerateInstanceVersion"));
  if (!it.CreateInstance) {
    error = "[omarchy] Vulkan loader has no vkCreateInstance.";
    return false;
  }
  if (!it.EnumerateInstanceVersion) {
    error = "[omarchy] Vulkan loader predates Vulkan 1.1.";
    return false;
  }

  VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
  app.pApplicationName = "mlx-omarchy";
  app.pEngineName = "mlx-omarchy";
  app.apiVersion = kVulkan13;
  VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
  ici.pApplicationInfo = &app;
  VKX_CHECK(it.CreateInstance(&ici, nullptr, &instance));

  it.EnumeratePhysicalDevices =
      reinterpret_cast<PFN_vkEnumeratePhysicalDevices>(
          vk::GetInstanceProcAddr(instance, "vkEnumeratePhysicalDevices"));
  it.GetPhysicalDeviceProperties2 =
      reinterpret_cast<PFN_vkGetPhysicalDeviceProperties2>(
          vk::GetInstanceProcAddr(instance, "vkGetPhysicalDeviceProperties2"));
  it.EnumerateDeviceExtensionProperties =
      reinterpret_cast<PFN_vkEnumerateDeviceExtensionProperties>(
          vk::GetInstanceProcAddr(
              instance, "vkEnumerateDeviceExtensionProperties"));
  it.GetPhysicalDeviceFeatures2 =
      reinterpret_cast<PFN_vkGetPhysicalDeviceFeatures2>(
          vk::GetInstanceProcAddr(instance, "vkGetPhysicalDeviceFeatures2"));
  it.GetPhysicalDeviceMemoryProperties2 =
      reinterpret_cast<PFN_vkGetPhysicalDeviceMemoryProperties2>(
          vk::GetInstanceProcAddr(
              instance, "vkGetPhysicalDeviceMemoryProperties2"));
  it.GetPhysicalDeviceQueueFamilyProperties =
      reinterpret_cast<PFN_vkGetPhysicalDeviceQueueFamilyProperties>(
          vk::GetInstanceProcAddr(
              instance, "vkGetPhysicalDeviceQueueFamilyProperties"));
  it.CreateDevice = reinterpret_cast<PFN_vkCreateDevice>(
      vk::GetInstanceProcAddr(instance, "vkCreateDevice"));
  it.GetPhysicalDeviceCooperativeMatrixPropertiesKHR =
      reinterpret_cast<PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR>(
          vk::GetInstanceProcAddr(
              instance, "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR"));
  it.DestroyInstance = reinterpret_cast<PFN_vkDestroyInstance>(
      vk::GetInstanceProcAddr(instance, "vkDestroyInstance"));
  if (!it.EnumeratePhysicalDevices || !it.GetPhysicalDeviceProperties2 ||
      !it.GetPhysicalDeviceFeatures2 ||
      !it.GetPhysicalDeviceMemoryProperties2 ||
      !it.GetPhysicalDeviceQueueFamilyProperties || !it.CreateDevice ||
      !it.DestroyInstance || !it.EnumerateDeviceExtensionProperties) {
    error = "[omarchy] Vulkan instance does not expose required 1.3 functions.";
    return false;
  }

  uint32_t count = 0;
  if (it.EnumeratePhysicalDevices(instance, &count, nullptr) != VK_SUCCESS ||
      count == 0) {
    error =
        "[omarchy] no Vulkan physical devices found."
        " Is the Mesa Honeykrisp driver installed?";
    return false;
  }
  std::vector<VkPhysicalDevice> pds(count);
  VKX_CHECK(it.EnumeratePhysicalDevices(instance, &count, pds.data()));

  std::string first_refusal;

  for (VkPhysicalDevice pd : pds) {
    VkPhysicalDeviceDriverProperties driver{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES};
    VkPhysicalDeviceMaintenance3Properties m3{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_3_PROPERTIES};
    VkPhysicalDeviceMaintenance4Properties m4{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_4_PROPERTIES};
    driver.pNext = &m3;
    m3.pNext = &m4;
    VkPhysicalDeviceSubgroupProperties subgroup{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};
    VkPhysicalDeviceProperties2 props2{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    props2.pNext = &driver;
    driver.pNext = &subgroup;
    it.GetPhysicalDeviceProperties2(pd, &props2);

    DeviceSupport support = classify_physical_device(
        props2.properties,
        static_cast<int32_t>(driver.driverID),
        allow_non_apple);

    VkPhysicalDeviceMemoryProperties2 mem2{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2};
    it.GetPhysicalDeviceMemoryProperties2(pd, &mem2);

    VkPhysicalDevice16BitStorageFeatures f16{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES};
    VkPhysicalDeviceVulkan12Features f12{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    VkPhysicalDeviceVulkan13Features f13{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    VkPhysicalDeviceShaderAtomicFloatFeaturesEXT fa{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT};
    VkPhysicalDeviceCooperativeMatrixFeaturesKHR cm{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR};
    VkPhysicalDeviceFeatures2 feats2{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    f16.pNext = &f12;
    f12.pNext = &f13;
    f13.pNext = &fa;
    fa.pNext = &cm;
    feats2.pNext = &f16;
    it.GetPhysicalDeviceFeatures2(pd, &feats2);

    if (support.supported && f12.timelineSemaphore != VK_TRUE) {
      support.supported = false;
      support.reason = "[omarchy] device '" +
          std::string(props2.properties.deviceName) +
          "' does not expose timeline semaphores;"
          " the backend requires Vulkan 1.2+ synchronization.";
    }

    if (!support.supported) {
      if (first_refusal.empty()) {
        first_refusal = support.reason;
      }
      continue;
    }

    PhysicalDeviceInfo info;
    info.handle = pd;
    info.support = std::move(support);
    info.caps = collect_capabilities(
        it,
        pd,
        info.support,
        props2,
        mem2,
        feats2,
        f12,
        f13,
        fa,
        cm,
        f16,
        m3,
        m4,
        subgroup);
    info.hardware = info.caps;
    if (sim) {
      info.caps = capsim::apply(info.caps, *sim);
      std::fprintf(
          stderr,
          "[omarchy] simulated device %u: '%s' (%s) as"
          " subgroup_size=%u subgroup_ops_mask=0x%x"
          " cooperative_matrix_fp32_8x8x8=%d"
          " shared_memory_limit_bytes=%zu"
          " workgroup_invocations=%u workgroup_size_x=%u"
          " atomic_float_add=%d\n",
          static_cast<unsigned>(supported.size()),
          info.caps.device_name.c_str(),
          info.caps.driver_name.c_str(),
          info.caps.subgroup_size,
          info.caps.subgroup_operations,
          info.caps.cooperative_matrix_f32_8 ? 1 : 0,
          info.caps.max_compute_shared_memory_size,
          info.caps.max_compute_work_group_invocations,
          info.caps.max_compute_work_group_size[0],
          info.caps.shader_atomic_float_add ? 1 : 0);
    }
    if (info.caps.queue_count == 0) {
      if (first_refusal.empty()) {
        first_refusal = "[omarchy] device '" + info.caps.device_name +
            "' exposes no compute queue family.";
      }
      continue;
    }
    supported.push_back(std::move(info));
  }

  if (supported.empty()) {
    error = first_refusal.empty()
        ? "[omarchy] no qualifying Vulkan 1.3 device found."
        : first_refusal;
    it.DestroyInstance(instance, nullptr);
    instance = VK_NULL_HANDLE;
    return false;
  }

  error.clear();
  // One lazily-filled slot per supported device; device() indexes this
  // vector directly, so it must match |supported| before ready flips.
  devices.resize(supported.size());
  ready = true;
  return true;
}

} // namespace

bool runtime_alive() {
  return !g_runtime_destroyed.load(std::memory_order_acquire);
}

// Compiled-tape debug switch plumbing (device.h). At omarchy scope, not
// in the anonymous namespace above: the encoder and allocator call these
// through the header declarations.
bool env_flag(const char* name) {
  const char* v = std::getenv(name);
  if (!v) {
    return false;
  }
  std::string s = v;
  std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return s == "1" || s == "on" || s == "true" || s == "yes";
}

// Scoped compiled-tape diagnostic state (device.h). Plain atomics: the
// encoder and the allocator poll these per dispatch or per allocation,
// so the unset path must stay cheaper than an environment lookup.
// g_no_buffer_cache is declared in the anonymous namespace above because
// Runtime::init_impl sets it before this point of the file.
namespace {
std::atomic<bool> g_tape_full_barriers{false};
std::atomic<bool> g_tape_no_reuse{false};
} // namespace

bool buffer_cache_disabled() {
  return g_no_buffer_cache.load(std::memory_order_relaxed);
}

TapeDebugScope::TapeDebugScope(bool full_barriers, bool no_reuse)
    : full_barriers_(full_barriers), no_reuse_(no_reuse) {
  g_tape_full_barriers.store(full_barriers, std::memory_order_relaxed);
  g_tape_no_reuse.store(no_reuse, std::memory_order_relaxed);
}

TapeDebugScope::~TapeDebugScope() {
  g_tape_full_barriers.store(false, std::memory_order_relaxed);
  g_tape_no_reuse.store(false, std::memory_order_relaxed);
}

bool tape_full_barriers() {
  return g_tape_full_barriers.load(std::memory_order_relaxed);
}

bool tape_no_reuse() {
  return g_tape_no_reuse.load(std::memory_order_relaxed);
}

DeviceSupport classify_physical_device(
    const VkPhysicalDeviceProperties& props,
    int32_t driver_id,
    bool allow_non_apple) {
  DeviceSupport s;
  s.device_name = props.deviceName;
  s.vendor_id = props.vendorID;
  s.device_id = props.deviceID;
  s.api_version = props.apiVersion;
  s.driver_id = driver_id;

  if (props.apiVersion < kVulkan13) {
    s.reason = "[omarchy] device '" + std::string(props.deviceName) +
        "' reports Vulkan " +
        std::to_string(VK_API_VERSION_MAJOR(props.apiVersion)) + "." +
        std::to_string(VK_API_VERSION_MINOR(props.apiVersion)) +
        "; the backend requires Vulkan 1.3.";
    return s;
  }

  // Driver identity is authoritative: the M1 target reports vendor 0x10005
  // and deviceName "Apple M1" under Mesa Honeykrisp, so the driver id is
  // the signal that accepts it. Driver-id constants are C enum values, not
  // macros; never guard them with #ifdef. Apple vendor 0x106b stays as an
  // alternate signal for Apple GPUs on other driver builds.
  bool apple = props.vendorID == kAppleVendorId;
  bool honeykrisp_name =
      contains_case_insensitive(props.deviceName, "honeykrisp");
  bool honeykrisp_driver = driver_id == kMesaHoneykrispDriverId;

  if (apple || honeykrisp_name || honeykrisp_driver) {
    s.supported = true;
    return s;
  }

  if (allow_non_apple) {
    s.supported = true;
    s.non_apple_dev = true;
    return s;
  }

  s.reason = "[omarchy] device '" + std::string(props.deviceName) +
      "' is not an Apple GPU running Omarchy Honeykrisp."
      " Set MLX_OMARCHY_ALLOW_NON_APPLE=1 to use it for development only.";
  return s;
}

// --- Process-wide API -----------------------------------------------------

bool init() {
  return runtime().init();
}

bool is_available() {
  return runtime().init();
}

const std::string& init_error() {
  runtime().init();
  return runtime().error;
}

int device_count() {
  if (!runtime().init()) {
    return 0;
  }
  return static_cast<int>(runtime().supported.size());
}

Device& device(uint32_t index) {
  auto& rt = runtime();
  if (!rt.init()) {
    throw std::runtime_error(rt.error);
  }
  uint32_t effective = index;
  if (effective == 0 && rt.preferred_device_index >= 0) {
    effective = static_cast<uint32_t>(rt.preferred_device_index);
  }
  if (effective >= rt.supported.size()) {
    throw std::invalid_argument(
        "[omarchy] device index " + std::to_string(effective) +
        " is out of range; " + std::to_string(rt.supported.size()) +
        " supported device(s) found.");
  }
  std::lock_guard<std::mutex> lk(rt.mutex);
  auto& slot = rt.devices[effective];
  if (!slot) {
    slot = std::make_unique<Device>(effective);
  }
  return *slot;
}

const CapabilityReport& capability_report(uint32_t index) {
  auto& rt = runtime();
  if (!rt.init()) {
    throw std::runtime_error(rt.error);
  }
  if (index >= rt.supported.size()) {
    throw std::invalid_argument(
        "[omarchy] device index " + std::to_string(index) +
        " is out of range.");
  }
  return rt.supported[index].caps;
}

// Hardware-truth report for a supported index regardless of any active
// capability simulation. mlx-omarchy-info --hardware and the
// simulation tests read this.
const CapabilityReport& hardware_capability_report(uint32_t index) {
  auto& rt = runtime();
  if (!rt.init()) {
    throw std::runtime_error(rt.error);
  }
  if (index >= rt.supported.size()) {
    throw std::invalid_argument(
        "[omarchy] device index " + std::to_string(index) +
        " is out of range.");
  }
  return rt.supported[index].hardware;
}

// --- Device ---------------------------------------------------------------

Device::Device(uint32_t physical_device_index) {
  auto& rt = runtime();
  auto& info = rt.supported.at(physical_device_index);
  // Dispatch and provenance see the (possibly simulated) report; the
  // VkDevice is created from hardware truth, so a simulated capability
  // can never enable an extension or feature the driver does not
  // actually have.
  caps_ = info.caps;
  hardware_caps_ = info.hardware;
  const CapabilityReport& hw = hardware_caps_;
  VkPhysicalDevice pd = info.handle;
  auto& it = vk::instance_table();

  VkPhysicalDeviceShaderAtomicFloatFeaturesEXT enabled_fa{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT};
  enabled_fa.shaderBufferFloat32AtomicAdd =
      hw.shader_atomic_float_add ? VK_TRUE : VK_FALSE;
  VkPhysicalDeviceCooperativeMatrixFeaturesKHR enabled_cm{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR};
  enabled_cm.cooperativeMatrix =
      hw.cooperative_matrix_f32_8 ? VK_TRUE : VK_FALSE;
  VkPhysicalDevice16BitStorageFeatures enabled16{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES};
  enabled16.storageBuffer16BitAccess =
      hw.storage_buffer_16bit_access ? VK_TRUE : VK_FALSE;
  VkPhysicalDeviceVulkan12Features enabled12{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
  enabled12.timelineSemaphore = VK_TRUE;
  enabled12.shaderFloat16 = hw.shader_float16 ? VK_TRUE : VK_FALSE;
  VkPhysicalDeviceVulkan13Features enabled13{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
  VkPhysicalDeviceFeatures2 enabled2{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
  enabled2.features.shaderInt16 = hw.shader_int16 ? VK_TRUE : VK_FALSE;
  enabled2.features.shaderInt64 = hw.shader_int64 ? VK_TRUE : VK_FALSE;
  enabled16.pNext = &enabled12;
  enabled12.pNext = &enabled13;
  enabled13.pNext = &enabled_fa;
  if (hw.cooperative_matrix_f32_8) {
    enabled_fa.pNext = &enabled_cm;
  }
  enabled2.pNext = &enabled16;

  // M1 receipt: Mesa Honeykrisp exposes one compute queue (family 0,
  // queueCount 1). Request exactly that; never invent additional queues.
  float priority = 1.0f;
  VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
  qci.queueFamilyIndex = caps_.queue_family_index;
  qci.queueCount = 1;
  qci.pQueuePriorities = &priority;

  VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
  dci.pNext = &enabled2;
  dci.queueCreateInfoCount = 1;
  dci.pQueueCreateInfos = &qci;
  // Extensions enabled only from hardware truth: a simulated
  // cooperative-matrix or float-atomic claim can never enable an
  // extension the driver does not actually provide.
  std::vector<const char*> device_exts;
  if (hw.shader_atomic_float_add) {
    device_exts.push_back("VK_EXT_shader_atomic_float");
  }
  if (hw.cooperative_matrix_f32_8) {
    device_exts.push_back(VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME);
  }
  if (!device_exts.empty()) {
    dci.enabledExtensionCount = static_cast<uint32_t>(device_exts.size());
    dci.ppEnabledExtensionNames = device_exts.data();
  }
  VKX_CHECK(it.CreateDevice(pd, &dci, nullptr, &device_));

  auto& dt = vk::device_table();
  dt.GetDeviceProcAddr = reinterpret_cast<PFN_vkGetDeviceProcAddr>(
      vk::GetInstanceProcAddr(rt.instance, "vkGetDeviceProcAddr"));
  if (!dt.GetDeviceProcAddr) {
    throw std::runtime_error(
        "[omarchy] Vulkan device is missing vkGetDeviceProcAddr.");
  }

  VkPhysicalDeviceMemoryProperties2 mem2{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2};
  it.GetPhysicalDeviceMemoryProperties2(pd, &mem2);
  mem_props_ = mem2.memoryProperties;

#define VKX_LOAD_DEVICE_FN(member, symbol)                                    \
  dt.member =                                                                 \
      reinterpret_cast<PFN_##symbol>(dt.GetDeviceProcAddr(device_, #symbol)); \
  if (dt.member == nullptr) {                                                 \
    throw std::runtime_error(                                                 \
        std::string("[omarchy] Vulkan device is missing ") + #symbol);        \
  }

  VKX_LOAD_DEVICE_FN(DestroyDevice, vkDestroyDevice)
  VKX_LOAD_DEVICE_FN(GetDeviceQueue, vkGetDeviceQueue)
  VKX_LOAD_DEVICE_FN(DeviceWaitIdle, vkDeviceWaitIdle)
  VKX_LOAD_DEVICE_FN(AllocateMemory, vkAllocateMemory)
  VKX_LOAD_DEVICE_FN(FreeMemory, vkFreeMemory)
  VKX_LOAD_DEVICE_FN(MapMemory, vkMapMemory)
  VKX_LOAD_DEVICE_FN(UnmapMemory, vkUnmapMemory)
  VKX_LOAD_DEVICE_FN(CreateBuffer, vkCreateBuffer)
  VKX_LOAD_DEVICE_FN(DestroyBuffer, vkDestroyBuffer)
  VKX_LOAD_DEVICE_FN(GetBufferMemoryRequirements, vkGetBufferMemoryRequirements)
  VKX_LOAD_DEVICE_FN(BindBufferMemory, vkBindBufferMemory)
  VKX_LOAD_DEVICE_FN(CreateCommandPool, vkCreateCommandPool)
  VKX_LOAD_DEVICE_FN(DestroyCommandPool, vkDestroyCommandPool)
  VKX_LOAD_DEVICE_FN(AllocateCommandBuffers, vkAllocateCommandBuffers)
  VKX_LOAD_DEVICE_FN(FreeCommandBuffers, vkFreeCommandBuffers)
  VKX_LOAD_DEVICE_FN(ResetCommandPool, vkResetCommandPool)
  VKX_LOAD_DEVICE_FN(BeginCommandBuffer, vkBeginCommandBuffer)
  VKX_LOAD_DEVICE_FN(EndCommandBuffer, vkEndCommandBuffer)
  VKX_LOAD_DEVICE_FN(ResetCommandBuffer, vkResetCommandBuffer)
  VKX_LOAD_DEVICE_FN(CmdCopyBuffer, vkCmdCopyBuffer)
  VKX_LOAD_DEVICE_FN(CmdFillBuffer, vkCmdFillBuffer)
  VKX_LOAD_DEVICE_FN(CreateShaderModule, vkCreateShaderModule)
  VKX_LOAD_DEVICE_FN(DestroyShaderModule, vkDestroyShaderModule)
  VKX_LOAD_DEVICE_FN(CreateDescriptorSetLayout, vkCreateDescriptorSetLayout)
  VKX_LOAD_DEVICE_FN(DestroyDescriptorSetLayout, vkDestroyDescriptorSetLayout)
  VKX_LOAD_DEVICE_FN(CreateDescriptorPool, vkCreateDescriptorPool)
  VKX_LOAD_DEVICE_FN(DestroyDescriptorPool, vkDestroyDescriptorPool)
  VKX_LOAD_DEVICE_FN(AllocateDescriptorSets, vkAllocateDescriptorSets)
  VKX_LOAD_DEVICE_FN(UpdateDescriptorSets, vkUpdateDescriptorSets)
  VKX_LOAD_DEVICE_FN(CreatePipelineLayout, vkCreatePipelineLayout)
  VKX_LOAD_DEVICE_FN(DestroyPipelineLayout, vkDestroyPipelineLayout)
  VKX_LOAD_DEVICE_FN(CreateComputePipelines, vkCreateComputePipelines)
  VKX_LOAD_DEVICE_FN(DestroyPipeline, vkDestroyPipeline)
  VKX_LOAD_DEVICE_FN(CmdBindPipeline, vkCmdBindPipeline)
  VKX_LOAD_DEVICE_FN(CmdBindDescriptorSets, vkCmdBindDescriptorSets)
  VKX_LOAD_DEVICE_FN(CmdPushConstants, vkCmdPushConstants)
  VKX_LOAD_DEVICE_FN(CmdDispatch, vkCmdDispatch)
  VKX_LOAD_DEVICE_FN(CmdPipelineBarrier, vkCmdPipelineBarrier)
  VKX_LOAD_DEVICE_FN(CreateEvent, vkCreateEvent)
  VKX_LOAD_DEVICE_FN(GetEventStatus, vkGetEventStatus)
  VKX_LOAD_DEVICE_FN(ResetEvent, vkResetEvent)
  VKX_LOAD_DEVICE_FN(CmdSetEvent, vkCmdSetEvent)
  VKX_LOAD_DEVICE_FN(CreateQueryPool, vkCreateQueryPool)
  VKX_LOAD_DEVICE_FN(GetQueryPoolResults, vkGetQueryPoolResults)
  VKX_LOAD_DEVICE_FN(CmdResetQueryPool, vkCmdResetQueryPool)
  VKX_LOAD_DEVICE_FN(CmdWriteTimestamp, vkCmdWriteTimestamp)
  VKX_LOAD_DEVICE_FN(QueueSubmit, vkQueueSubmit)
  VKX_LOAD_DEVICE_FN(QueueWaitIdle, vkQueueWaitIdle)
  VKX_LOAD_DEVICE_FN(CreateFence, vkCreateFence)
  VKX_LOAD_DEVICE_FN(DestroyFence, vkDestroyFence)
  VKX_LOAD_DEVICE_FN(ResetFences, vkResetFences)
  VKX_LOAD_DEVICE_FN(WaitForFences, vkWaitForFences)
  VKX_LOAD_DEVICE_FN(CreateSemaphore, vkCreateSemaphore)
  VKX_LOAD_DEVICE_FN(DestroySemaphore, vkDestroySemaphore)
  VKX_LOAD_DEVICE_FN(GetSemaphoreCounterValue, vkGetSemaphoreCounterValue)
  VKX_LOAD_DEVICE_FN(WaitSemaphores, vkWaitSemaphores)
  VKX_LOAD_DEVICE_FN(FlushMappedMemoryRanges, vkFlushMappedMemoryRanges)
  VKX_LOAD_DEVICE_FN(
      InvalidateMappedMemoryRanges, vkInvalidateMappedMemoryRanges)

#undef VKX_LOAD_DEVICE_FN

  dt.GetDeviceQueue(device_, caps_.queue_family_index, 0, &queue_);
  // The live binding budget: what the backend wants, clamped by what this
  // physical device actually reports. The spec floor (4) always passes
  // through; larger kernels check compute().binding_limit() at eval time.
  uint32_t binding_limit = std::min(
      kComputeBindingBudget,
      std::min(
          caps_.max_per_stage_descriptor_storage_buffers,
          caps_.max_descriptor_set_storage_buffers));
  compute_ = std::make_unique<ComputeRuntime>(device_, binding_limit);
  completions_ = std::make_unique<CompletionDispatcher>(device_, this);
}

Device::~Device() {
  if (device_ != VK_NULL_HANDLE) {
    auto& dt = vk::device_table();
    // Order is load-bearing: the queue finishes every submission first
    // (DeviceWaitIdle), then the dispatcher is shut down and drained while
    // the VkDevice and its semaphores are still valid, and only then is
    // the VkDevice destroyed.
    if (dt.DeviceWaitIdle) {
      dt.DeviceWaitIdle(device_);
    }
    if (completions_) {
      completions_->shutdown();
    }
    compute_.reset();
    if (dt.DestroyDevice) {
      dt.DestroyDevice(device_, nullptr);
    }
  }
}

uint64_t Device::signal_timeline(VkSemaphore semaphore, uint64_t value) {
  std::lock_guard<std::mutex> lk(queue_mutex_);
  uint64_t completion_value = completions().reserve();
  // Host-signal both timelines. This used to be an empty signal-only
  // QueueSubmit (no command buffer, no waits); Honeykrisp swallows that
  // shape - the semaphores are never signaled and the reserved
  // completion value never publishes, so the next host wait deadlocks
  // at target=1 with the counter stuck at 0 (cos/sin, GDN chunked
  // async_eval, Ministral/Bonsai-8B first submissions). vkSignalSemaphore
  // from the host needs no queue involvement and cannot be dropped.
  static PFN_vkSignalSemaphore signal_fn = nullptr;
  if (!signal_fn) {
    signal_fn = reinterpret_cast<PFN_vkSignalSemaphore>(
        vk::device_table().GetDeviceProcAddr(
            handle(), "vkSignalSemaphore"));
  }
  VkSemaphoreSignalInfo ev{VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO};
  ev.semaphore = semaphore;
  ev.value = value;
  VKX_CHECK(signal_fn(handle(), &ev));
  VkSemaphoreSignalInfo cc{VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO};
  cc.semaphore = completions().semaphore();
  cc.value = completion_value;
  VKX_CHECK(signal_fn(handle(), &cc));
  completions().enqueue(completion_value, {}, {});
  return completion_value;
}

Device::RecoveryResult Device::recover_stalled_submissions(
    uint64_t target_value,
    uint32_t round) {
  // Budget: two recovery rounds per stalled wait (|round| is owned by the
  // waiting loop, so a later wait starts fresh). Recovery is for the
  // swallow class, not for a genuinely wedged GPU - a real hang must
  // still surface as the typed watchdog error, not as an unbounded
  // retry loop.
  if (round >= 2) {
    return RecoveryResult::kExhausted;
  }

  std::lock_guard<std::mutex> lk(queue_mutex_);
  auto& completions = this->completions();
  // Judge and take against ALL reserved values, not just the waited one:
  // a previous recovery round re-retains resubmitted batches at fresh
  // values above the original target, and they must not be stranded.
  const uint64_t reserved_through = completions.last_reserved();
  bool executing = completions.has_active_submission(reserved_through);
  if (std::getenv("MLX_OMARCHY_TRACE_DISPATCH") != nullptr) {
    fprintf(stderr,
            "[rtmod] RECOVER-ENTER tid=%lu target=%llu through=%llu active=%d\n",
            (unsigned long)syscall(SYS_gettid),
            (unsigned long long)target_value,
            (unsigned long long)reserved_through,
            executing ? 1 : 0);
  }

  // Execution evidence: the batch's started event (CmdSetEvent at
  // TOP_OF_PIPE) fired, so the kernels ran and only the completion
  // signal was lost. Mesa will not publish ANY queue-based signal once
  // it has wedged a submission's timeline points (measured: fresh-value
  // resubmission executed, timeline stayed at 0) - but vkSignalSemaphore
  // from the host cannot be dropped (the in-tree empty-shape fix relies
  // on exactly that). The wait has seen a full no-progress interval by
  // definition of this call, so short kernels have long finished;
  // host-publish the completion at a freshly reserved value.
  if (executing) {
    // The retained batches' user (event) semaphores are equally
    // stranded: their ride-along signals were swallowed with the
    // submission. Host-signal each at a bumped fresh value (timeline
    // waits are >=), then publish the completion timeline.
    auto stranded = completions.take_resubmit_batches(reserved_through);
    for (auto& batch : stranded) {
      for (size_t i = 0; i + 1 < batch.signal_values.size(); ++i) {
        uint64_t current = 0;
        if (vk::device_table().GetSemaphoreCounterValue(
                device_, batch.signal_sems[i], &current) != VK_SUCCESS) {
          current = 0;
        }
        batch.signal_values[i] =
            std::max(batch.signal_values[i], current) + 1;
      }
    }
    uint64_t fresh = completions.reserve();
    static PFN_vkSignalSemaphore signal_fn = nullptr;
    if (!signal_fn) {
      signal_fn = reinterpret_cast<PFN_vkSignalSemaphore>(
          vk::device_table().GetDeviceProcAddr(
              handle(), "vkSignalSemaphore"));
    }
    VkSemaphoreSignalInfo cc{VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO};
    cc.semaphore = completions.semaphore();
    cc.value = fresh;
    VKX_CHECK(signal_fn(handle(), &cc));
    for (auto& batch : stranded) {
      for (size_t i = 0; i + 1 < batch.signal_values.size(); ++i) {
        VkSemaphoreSignalInfo us{VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO};
        us.semaphore = batch.signal_sems[i];
        us.value = batch.signal_values[i];
        VKX_CHECK(signal_fn(handle(), &us));
      }
    }
    fprintf(
        stderr,
        "[rtmod] SUBMIT-RECOVER tid=%lu host-signaled completion cv=%llu after "
        "executed-but-unsignaled batch (round %u)\n",
        (unsigned long)syscall(SYS_gettid),
        (unsigned long long)fresh,
        round + 1);
    // Every reserved value needs a pending completion entry: join paths
    // wait drained_value_ >= counter, and the fresh value's drain is what
    // lets drained_value_ catch up past the original completion.
    completions.enqueue(fresh, {}, {});
    return RecoveryResult::kRecovered;
  }

  // No execution evidence: the batch never began. Resubmit it with
  // fresh signal values - the dropped submission still holds its
  // original timeline points inside Mesa, and re-signaling the same
  // value creates a duplicate point Mesa does not publish. Timeline
  // waits are >=, so strictly higher values satisfy every existing
  // waiter.
  std::vector<CompletionDispatcher::ResubmitBatch> batches =
      completions.take_resubmit_batches(reserved_through);
  if (batches.empty()) {
    if (std::getenv("MLX_OMARCHY_TRACE_DISPATCH") != nullptr) {
      fprintf(stderr, "[rtmod] RECOVER-FALSE tid=%lu empty-batches\n", (unsigned long)syscall(SYS_gettid));
    }
    // Another waiter's recovery already took the batches (or the
    // producing stream has not submitted yet): not ours to recover, and
    // not a hang verdict. The caller keeps waiting.
    return RecoveryResult::kNotRecoverable;
  }

  auto& dt = vk::device_table();
  // First, an empty queue-touch submit: if the loss was a missed
  // doorbell rather than a dropped batch, this alone flushes the queue
  // and the original submission completes. No semaphores, no command
  // buffer - nothing to signal, so ordering cannot be violated.
  VkSubmitInfo kick{VK_STRUCTURE_TYPE_SUBMIT_INFO};
  dt.QueueSubmit(queue_, 1, &kick, VK_NULL_HANDLE);

  size_t count = batches.size();
  for (auto& batch : batches) {
    for (size_t i = 0; i + 1 < batch.signal_values.size(); ++i) {
      uint64_t current = 0;
      if (vk::device_table().GetSemaphoreCounterValue(
              device_, batch.signal_sems[i], &current) != VK_SUCCESS) {
        current = 0;
      }
      batch.signal_values[i] =
          std::max(batch.signal_values[i], current) + 1;
    }
    uint64_t fresh_completion = completions.reserve();
    batch.value = fresh_completion;
    batch.signal_values.back() = fresh_completion;

    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    VkTimelineSemaphoreSubmitInfo timeline{
        VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO};
    std::vector<VkPipelineStageFlags> wait_stages(
        batch.wait_sems.size(), VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
    timeline.waitSemaphoreValueCount =
        static_cast<uint32_t>(batch.wait_values.size());
    timeline.pWaitSemaphoreValues = batch.wait_values.data();
    timeline.signalSemaphoreValueCount =
        static_cast<uint32_t>(batch.signal_values.size());
    timeline.pSignalSemaphoreValues = batch.signal_values.data();
    si.pNext = &timeline;
    si.waitSemaphoreCount = static_cast<uint32_t>(batch.wait_sems.size());
    si.pWaitSemaphores = batch.wait_sems.data();
    si.pWaitDstStageMask = wait_stages.data();
    si.commandBufferCount = batch.cmd != VK_NULL_HANDLE ? 1u : 0u;
    si.pCommandBuffers = batch.cmd != VK_NULL_HANDLE ? &batch.cmd : nullptr;
    si.signalSemaphoreCount =
        static_cast<uint32_t>(batch.signal_sems.size());
    si.pSignalSemaphores = batch.signal_sems.data();
    VKX_CHECK(dt.QueueSubmit(queue_, 1, &si, VK_NULL_HANDLE));
    // The batch is pending again at its fresh value; retain it for a
    // further round.
    completions.retain_for_resubmit(fresh_completion, std::move(batch));
    // Every reserved value needs a pending completion entry: join paths
    // wait drained_value_ >= counter, and the fresh value's drain is what
    // lets drained_value_ catch up past the original completion. Empty
    // payload - the original completion still owns handlers/temporaries.
    completions.enqueue(fresh_completion, {}, {});
  }
  fprintf(
      stderr,
      "[rtmod] SUBMIT-RECOVER tid=%lu resubmitted %zu stalled batch(es) through "
      "cv>=%llu (round %u, fresh signals)\n",
      (unsigned long)syscall(SYS_gettid),
      count,
      (unsigned long long)target_value,
      round + 1);
  return RecoveryResult::kRecovered;
}

void Device::join_completed_handlers() {
  if (!completions_) {
    return;
  }
  uint64_t current = 0;
  if (vk::device_table().GetSemaphoreCounterValue(
          device_, completions_->semaphore(), &current) != VK_SUCCESS ||
      current == 0) {
    return;
  }
  completions_->wait(current);
}

// --- CompletionDispatcher -------------------------------------------------

namespace {

// Bounded dispatcher wait granularity (plan R16): a wedged queue delays
// shutdown and handler dispatch by at most one interval, never forever.
constexpr uint64_t kCompletionPollNs = 100ull * 1000 * 1000;

// No-progress watchdog poll cadence. Must be a small fraction of the
// no-progress interval so a stalled counter is detected quickly. 100 ms
// matches kCompletionPollNs (the dispatcher background loop) and gives
// 100 samples per no-progress interval at the 10 s default.
constexpr uint64_t kHangWatchPollNs = kCompletionPollNs;

// Parsed env override. Returns the default when the env var is unset or
// invalid; positive non-zero values only. Negative or zero values would
// cause the watchdog to fire instantly, so reject them.
uint64_t env_ns_override(const char* name, uint64_t fallback) {
  const char* v = std::getenv(name);
  if (!v || *v == '\0') {
    return fallback;
  }
  char* end = nullptr;
  long long parsed = std::strtoll(v, &end, 10);
  if (end == v || parsed <= 0) {
    return fallback;
  }
  return static_cast<uint64_t>(parsed);
}

} // namespace

uint64_t submit_hang_no_progress_ns() {
  static const uint64_t v =
      env_ns_override("MLX_OMARCHY_HANG_NO_PROGRESS_NS",
                      kSubmitHangNoProgressNsDefault);
  return v;
}

uint64_t submit_max_wall_ns() {
  static const uint64_t v =
      env_ns_override("MLX_OMARCHY_MAX_WALL_NS", kSubmitMaxWallNsDefault);
  return v;
}

void wait_for_timeline_progress(
    VkDevice device,
    VkSemaphore semaphore,
    uint64_t target_value,
    CompletionDispatcher* progress,
    std::function<uint64_t()> progress_generation,
    Device* recovery) {
  using clock = std::chrono::steady_clock;
  const uint64_t hang_ns = submit_hang_no_progress_ns();
  const uint64_t max_wall_ns = submit_max_wall_ns();
  const auto start = clock::now();
  const auto wall_deadline = start + std::chrono::nanoseconds(max_wall_ns);
  uint64_t last_observed = 0;
  auto last_advance = start;
  uint32_t recovery_round = 0;
  // Refused recoveries (empty retained-batch set) are bounded by the same
  // budget as resubmission rounds: a stall whose target is already
  // reserved but whose signalling batch does not exist will never be
  // satisfied, and waiting for "the owner" only defers the typed error
  // to the wall deadline (minutes) with the wrong message. Real
  // foreign/stale waiters never get here - their target exceeds
  // last_reserved and the branch above refuses them without counting.
  uint32_t refused_rounds = 0;
  bool foreign_traced = false;

  VkSemaphoreWaitInfo info{VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
  info.semaphoreCount = 1;
  info.pSemaphores = &semaphore;
  info.pValues = &target_value;

  for (;;) {
    VkResult res = vk::device_table().WaitSemaphores(
        device, &info, kHangWatchPollNs);
    if (res == VK_SUCCESS) {
      return;
    }
    if (res != VK_TIMEOUT) {
      throw std::runtime_error(
          std::string("[omarchy] Vulkan timeline wait failed (") +
          vk::result_string(res) +
          "). The device may be lost; no CPU fallback is available.");
    }

    uint64_t current = 0;
    if (vk::device_table().GetSemaphoreCounterValue(
            device, semaphore, &current) == VK_SUCCESS &&
        current > last_observed) {
      last_observed = current;
      last_advance = clock::now();
    }

    const auto now = clock::now();
    if (now >= wall_deadline) {
      throw std::runtime_error(
          std::string(
              "[omarchy] Vulkan submission did not complete within ") +
          std::to_string(max_wall_ns / 1000000ull) +
          " ms (Timeout). The device may be hung; no CPU fallback is"
          " available.");
    }

    const uint64_t progress_through =
        progress_generation ? progress_generation() : 0;
    const bool executing = progress && progress_through != 0 &&
        progress->has_active_submission(progress_through);
    if (executing) {
      last_advance = now;
    } else if ((now - last_advance) > std::chrono::nanoseconds(hang_ns)) {
      // Foreign/stale waiter: the waited value was never reserved by any
      // submission (target > last_reserved), so the batch that would
      // signal it has not even been handed to the driver yet - typically
      // a scheduler-thread Event::wait parking ahead of the owning
      // stream's submit. Nothing is recoverable here (no retained batch
      // can exist above last_reserved), and throwing kills the process
      // ahead of the owning wait's own recovery ladder (observed in the
      // build13 decision trace). Refuse recovery and keep waiting: the
      // owning wait runs the ladder for a real stall once it reserves,
      // and the wall deadline below still bounds a producer that never
      // submits.
      if (progress && target_value > progress->last_reserved()) {
        if (!foreign_traced &&
            std::getenv("MLX_OMARCHY_TRACE_DISPATCH") != nullptr) {
          foreign_traced = true;
          fprintf(stderr,
                  "[rtmod] STALL-FOREIGN tid=%lu target=%llu through=%llu (owner"
                  " has not submitted; continuing wait)\n",
                  (unsigned long)syscall(SYS_gettid),
                  (unsigned long long)target_value,
                  (unsigned long long)progress->last_reserved());
        }
        last_advance = clock::now();
        continue;
      }
      // No progress for a full no-progress interval. Before declaring the
      // device hung, give the recovery ladder a chance: Honeykrisp
      // provably swallows submissions (the empty signal-only shape was
      // deterministic and is fixed; burst observations add real command
      // buffers whose started event never fires). A successful resubmit
      // restarts the no-progress clock; a refused or exhausted recovery
      // throws exactly as before.
      if (std::getenv("MLX_OMARCHY_TRACE_DISPATCH") != nullptr) {
        fprintf(stderr,
                "[rtmod] STALL tid=%lu target=%llu observed=%llu round=%u "
                "recovery=%d\n",
                (unsigned long)syscall(SYS_gettid),
                (unsigned long long)target_value,
                (unsigned long long)last_observed,
                recovery_round,
                recovery != nullptr ? 1 : 0);
      }
      if (recovery) {
        switch (recovery->recover_stalled_submissions(
            target_value, recovery_round)) {
          case Device::RecoveryResult::kRecovered:
            recovery_round++;
            last_advance = clock::now();
            continue;
          case Device::RecoveryResult::kNotRecoverable:
            // Someone else may own the recovery (or own the submission
            // that has not happened yet). Throwing on the FIRST refusal
            // killed healthy processes ahead of the real recovery
            // (build13 decision trace), so refuse once per round - but
            // only for a bounded number of rounds. An unbounded wait is
            // wrong for a stall the retained set can never satisfy: the
            // bounded typed watchdog error is the contract.
            if (++refused_rounds >= 2) {
              break;
            }
            last_advance = clock::now();
            continue;
          case Device::RecoveryResult::kExhausted:
            break;
        }
      }
      throw std::runtime_error(
          std::string(
              "[omarchy] Vulkan timeline counter failed to advance for ") +
          std::to_string(hang_ns / 1000000ull) + " ms (last observed=" +
          std::to_string(last_observed) + ", target=" +
          std::to_string(target_value) +
          "). The device may be hung; no CPU fallback is available.");
    }
  }
}
CompletionDispatcher::CompletionDispatcher(VkDevice device, Device* owner)
    : device_(device), owner_(owner) {
  VkSemaphoreTypeCreateInfo type{VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
  type.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
  type.initialValue = 0;
  VkSemaphoreCreateInfo ci{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
  ci.pNext = &type;
  VKX_CHECK(
      vk::device_table().CreateSemaphore(device_, &ci, nullptr, &semaphore_));
  thread_ = std::thread([this] { run(); });
}

CompletionDispatcher::~CompletionDispatcher() {
  shutdown();
}

void CompletionDispatcher::wait(uint64_t value) {
  // No-progress watchdog: blocks until the timeline counter reaches
  // |value| OR throws a typed Omarchy error if the counter fails to
  // advance for the no-progress interval. vkWaitSemaphores has a
  // wall-clock cap that misclassifies long legitimate work as a hang;
  // observing the counter's motion avoids that failure mode without
  // giving up on real hang detection.
  wait_for_timeline_progress(
      device_, semaphore_, value, this, [value] { return value; }, owner_);
  // Inline fast path: drain and run every ready completion whose value
  // is <= |value| on this thread, serialized end-to-end through
  // drain_mutex_ with the background thread so handlers cannot interleave
  // across separate completion values. With this serialization in place,
  // drained_value_ already covers |value| when drain_through returns and
  // the cv_.wait_until below acts as a defensive join only.
  drain_through(value);
  // Defensive join: the timeline counter reached |value| but a handler
  // from a prior completion could still be running on the dispatcher
  // thread. The watchdog already validated counter motion; this join
  // is bounded by kSubmitHangNoProgressNs because any handler still
  // outstanding belongs to a completion whose counter value advanced,
  // and the dispatcher drains through that completion every poll
  // cycle. A short cv_ wait here would be unsafe without a bound, so
  // use the same no-progress interval as the watchdog ceiling.
  const auto join_deadline = std::chrono::steady_clock::now() +
      std::chrono::nanoseconds(submit_hang_no_progress_ns());
  std::unique_lock<std::mutex> lk(mutex_);
  cv_.wait_until(
      lk, join_deadline,
      [this, value] { return stop_ || drained_value_ >= value; });
  if (drained_value_ < value) {
    throw std::runtime_error(
        std::string(
            "[omarchy] Vulkan dispatcher drain did not catch up within ") +
        std::to_string(submit_hang_no_progress_ns() / 1000000ull) +
        " ms after the timeline counter reached " +
        std::to_string(value) +
        " (Timeout). The device may be hung; no CPU fallback is"
        " available.");
  }
}
void CompletionDispatcher::shutdown() {
  {
    std::lock_guard<std::mutex> lk(mutex_);
    stop_ = true;
  }
  cv_.notify_all();
  if (thread_.joinable()) {
    thread_.join();
  }
  // Static-destruction order can run this drain after the GPU allocator
  // registry is gone. Handlers still run (the scheduler is leaked by
  // design), but remaining buffer temporaries are intentionally leaked:
  // freeing arrays here would call into already-destroyed statics.
  static auto* leaked_temporaries = new std::vector<std::shared_ptr<void>>();
  while (!pending_.empty()) {
    auto completion = std::move(pending_.front());
    pending_.pop_front();
    for (auto& handler : completion.handlers) {
      handler();
    }
    leaked_temporaries->insert(
        leaked_temporaries->end(),
        std::make_move_iterator(completion.temporaries.begin()),
        std::make_move_iterator(completion.temporaries.end()));
  }
  leaked_temporaries->insert(
      leaked_temporaries->end(),
      std::make_move_iterator(retired_temporaries_.begin()),
      std::make_move_iterator(retired_temporaries_.end()));
  retired_temporaries_.clear();
  if (semaphore_ != VK_NULL_HANDLE) {
    vk::device_table().DestroySemaphore(device_, semaphore_, nullptr);
    semaphore_ = VK_NULL_HANDLE;
  }
}

uint64_t CompletionDispatcher::reserve() {
  // Called with the queue mutex held: the value order matches queue
  // execution order, as required for timeline signals.
  std::lock_guard<std::mutex> lk(mutex_);
  return ++next_value_;
}

uint64_t CompletionDispatcher::last_reserved() {
  std::lock_guard<std::mutex> lk(mutex_);
  return next_value_;
}

void CompletionDispatcher::enqueue(
    uint64_t value,
    std::vector<std::shared_ptr<void>> temporaries,
    std::vector<std::function<void()>> handlers,
    VkEvent started) {
  std::lock_guard<std::mutex> lk(mutex_);
  pending_.push_back(Completion{
      value, std::move(temporaries), std::move(handlers), started});
  cv_.notify_all();
}

void CompletionDispatcher::reset_progress_event(VkEvent event) {
  std::lock_guard<std::mutex> lk(mutex_);
  VKX_CHECK(vk::device_table().ResetEvent(device_, event));
}

bool CompletionDispatcher::has_active_submission(uint64_t through_value) {
  uint64_t reached = 0;
  VKX_CHECK(vk::device_table().GetSemaphoreCounterValue(
      device_, semaphore_, &reached));
  std::lock_guard<std::mutex> lk(mutex_);
  for (const auto& completion : pending_) {
    if (completion.value > through_value) {
      break;
    }
    if (completion.value <= reached) {
      continue;
    }
    if (completion.started != VK_NULL_HANDLE &&
        vk::device_table().GetEventStatus(device_, completion.started) ==
            VK_EVENT_SET) {
      return true;
    }
  }
  return false;
}

void CompletionDispatcher::retain_for_resubmit(
    uint64_t value,
    ResubmitBatch batch) {
  std::lock_guard<std::mutex> lk(mutex_);
  resubmitable_[value] = std::move(batch);
}

std::vector<CompletionDispatcher::ResubmitBatch>
CompletionDispatcher::take_resubmit_batches(uint64_t through_value) {
  std::lock_guard<std::mutex> lk(mutex_);
  std::vector<ResubmitBatch> out;
  for (auto& [value, batch] : resubmitable_) {
    if (value > through_value) {
      break;
    }
    out.push_back(std::move(batch));
  }
  resubmitable_.erase(resubmitable_.begin(), resubmitable_.upper_bound(through_value));
  return out;
}

void CompletionDispatcher::drain_through(uint64_t max_value) {
  std::lock_guard<std::mutex> drain_lk(drain_mutex_);
  std::vector<Completion> ready;
  {
    std::lock_guard<std::mutex> lk(mutex_);
    // reserve() and enqueue() preserve increasing value order.
    while (!pending_.empty() && pending_.front().value <= max_value) {
      ready.push_back(std::move(pending_.front()));
      pending_.pop_front();
    }
    // Their retained submit records completed - recovery must never
    // resubmit them again.
    resubmitable_.erase(
        resubmitable_.begin(), resubmitable_.upper_bound(max_value));
  }
  if (ready.empty()) {
    return;
  }
  // Completion boundary: make noncoherent host-visible writes from the
  // GPU visible before handlers and later host reads observe results.
  omarchy::allocator().invalidate_noncoherent(device_);
  uint64_t ready_value = ready.back().value;
  for (auto& completion : ready) {
    for (auto& handler : completion.handlers) {
      handler();
    }
  }
  // Completion boundary: buffers freed since their submissions recorded
  // leave the quarantine here. Cleanup for ready_value - 1 has provably
  // run once this value is observable (Mesa signals a submission's
  // semaphores before its submit-final cleanup retires the timeline
  // points), so quarantined buffers recycle exactly one generation after
  // their own completion.
  omarchy::allocator().release_quarantine(ready_value - 1);
  // Semaphore-keepalive payloads retire one completion generation late:
  // Mesa signals a submission's semaphores (including the completion
  // timeline read above) BEFORE its submit-final cleanup releases that
  // submission's timeline points, so observing this timeline does not
  // prove the driver finished the submission. Submits execute serially
  // in value order, so once completion V+1 is observable, cleanup for V
  // has run. Buffer payloads use the allocator quarantine instead.
  std::vector<std::shared_ptr<void>> retired;
  for (auto& completion : ready) {
    retired.insert(
        retired.end(),
        std::make_move_iterator(completion.temporaries.begin()),
        std::make_move_iterator(completion.temporaries.end()));
  }
  ready.clear();
  std::vector<std::shared_ptr<void>> release = std::move(retired_temporaries_);
  retired_temporaries_ = std::move(retired);
  std::lock_guard<std::mutex> lk(mutex_);
  drained_value_ = std::max(drained_value_, ready_value);
  cv_.notify_all();
  // |release| frees when this function returns: by then drained_value_
  // names a later generation, so the previous one is provably finished.
}

uint64_t CompletionDispatcher::drained_value() {
  std::lock_guard<std::mutex> lk(mutex_);
  return drained_value_;
}

void CompletionDispatcher::run() {
  // Polling, not vkWaitSemaphores: a host wait issued from this thread
  // while the queue executes the signal stalls some drivers (observed on
  // Mesa 22 lavapipe). GetSemaphoreCounterValue is a plain host query.
  // Explicit wait() drains inline. This thread releases payloads for
  // asynchronous commits that have no later wait.
  for (;;) {
    {
      std::unique_lock<std::mutex> lk(mutex_);
      if (pending_.empty()) {
        if (stop_) {
          return;
        }
        cv_.wait_for(lk, std::chrono::nanoseconds(kCompletionPollNs), [this] {
          return stop_ || !pending_.empty();
        });
        continue;
      }
    }
    uint64_t current = 0;
    if (vk::device_table().GetSemaphoreCounterValue(
            device_, semaphore_, &current) != VK_SUCCESS) {
      if (stop_) {
        return;
      }
      std::this_thread::sleep_for(std::chrono::nanoseconds(kCompletionPollNs));
      continue;
    }
    drain_through(current);

    // Do not busy-spin while the oldest pending submission is still on the
    // GPU. A caller-side drain or shutdown wakes this wait immediately.
    std::unique_lock<std::mutex> lk(mutex_);
    if (stop_) {
      return;
    }
    cv_.wait_for(lk, std::chrono::nanoseconds(kCompletionPollNs), [this] {
      return stop_ || pending_.empty();
    });
  }
}

} // namespace mlx::core::omarchy
