// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

// U2 Omarchy Vulkan spike: real GPU matmul and attention kernels for matched
// prefill and decode shapes, timed against the pinned same-machine llama.cpp
// Vulkan comparator with an explicit 80 percent go/no-go gate.
//
// What it runs (all real compute pipelines on the qualifying Vulkan device):
//   prefill  fp16/fp32 tiled GEMM      C[M,N] = A[M,K] * W[N,K]^T
//   decode   Q4_K quantized GEMV       out[n] = sum_k x[k] * W_q4k[n,k]
//   decode   fp32 GEMV (info row)
//   prefill  causal GQA SDPA, online softmax over KV tiles
//   decode   single-query GQA SDPA over the KV cache
//
// Kernel sources live in glsl/*.comp and are compiled to SPIR-V at build
// time with glslc or glslangValidator.
//
// Every kernel is validated against a CPU reference before timing; a
// mismatch aborts with the kernel name and error. Comparator inputs are
// required: missing or non-positive llama.cpp numbers are a hard error with
// the exact recovery steps. No comparator numbers are ever invented here.
//
// Exit codes: 0 = GO, 3 = NO-GO, 2 = usage/comparator/config error,
// 1 = runtime error (no device, validation mismatch, Vulkan failure).

#include <vulkan/vulkan.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

#include "benchmarks/omarchy/json_io.h"
#include "benchmarks/omarchy/reference.h"
#include "mlx/backend/omarchy/device.h"
#include "mlx/backend/omarchy/vulkan.h"

// User-space CPU pause for fence-status spins, mirroring the pinned ggml
// YIELD() (ggml-vulkan.cpp): ARM yield instruction, x86 pause, no syscalls.
#if defined(__aarch64__) || defined(__arm__)
#define OMARCHY_CPU_YIELD() asm volatile("yield")
#elif defined(__x86_64__) || defined(__i386__)
#define OMARCHY_CPU_YIELD() __builtin_ia32_pause()
#else
#define OMARCHY_CPU_YIELD() \
  do {                      \
  } while (0)
#endif

namespace omarchy = mlx::core::omarchy;
using omarchy_spike::json::Object;
using omarchy_spike::json::Value;

namespace omarchy_spike {

namespace {

constexpr double kGoNoGoThreshold = 0.80;
constexpr uint32_t kDefaultWarmup = 5;
constexpr uint32_t kDefaultReps = 30;
constexpr uint32_t kCompletionSpinUs = 2000;
constexpr uint64_t kCompletionTimeoutNs = 10'000'000'000ULL;
constexpr auto kCompletionPollDelay = std::chrono::microseconds(50);

[[noreturn]] void die(const std::string& msg, int code) {
  std::cerr << "ERROR: " << msg << std::endl;
  std::exit(code);
}

std::string read_file(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    return "";
  }
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

std::string utc_now() {
  std::time_t t = std::time(nullptr);
  char buf[32];
  std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", std::gmtime(&t));
  return buf;
}

// --- Pinned model dims ------------------------------------------------------

struct Dims {
  uint32_t hidden = 2048;
  uint32_t intermediate = 6144;
  uint32_t vocab = 248320;
  uint32_t layers = 24;
  uint32_t attn_layers = 6;
  uint32_t heads = 16;
  uint32_t kv_heads = 2;
  uint32_t head_dim = 256;

  uint32_t q_out() const {
    return heads * head_dim;
  }
  uint32_t kv_dim() const {
    return kv_heads * head_dim;
  }
  uint32_t qkv_out() const {
    return q_out() + 2 * kv_dim();
  }
  uint32_t mlp_up_out() const {
    return 2 * intermediate;
  }

  Value to_json() const {
    Object o;
    o["hidden_size"] = Value(double(hidden));
    o["intermediate_size"] = Value(double(intermediate));
    o["vocab_size"] = Value(double(vocab));
    o["num_hidden_layers"] = Value(double(layers));
    o["num_attention_layers"] = Value(double(attn_layers));
    o["num_attention_heads"] = Value(double(heads));
    o["num_key_value_heads"] = Value(double(kv_heads));
    o["head_dim"] = Value(double(head_dim));
    o["shape_source"] = Value(
        "pinned defaults from the Qwen3.8-2B contract (hidden 2048 confirmed "
        "by ane-linux-experiments receipts); head counts assume the Qwen3-Next "
        "GQA lineage. The comparator file's model_dims must match these or "
        "override them via --dims-json.");
    return Value(std::move(o));
  }
};

// --- Options -----------------------------------------------------------------

struct Options {
  std::string comparator_path;
  std::string ggml_comparator_path;
  std::string output_path;
  std::string dims_json_path;
  std::string spirv_dir;
  std::string source_commit;
  uint32_t warmup = kDefaultWarmup;
  uint32_t reps = kDefaultReps;
  uint32_t rounds = 10;
  uint32_t idle_wait_s = 0;
  uint32_t device_index = 0;
  bool profile_stages = false;
  bool source_dirty = false;
};

uint32_t parse_u32(const std::string& v, const char* name) {
  try {
    return uint32_t(std::stoul(v));
  } catch (const std::exception&) {
    die(std::string(name) + " expects a non-negative integer, got '" + v + "'",
        2);
  }
}

Options parse_args(int argc, char** argv) {
  Options opt;
  opt.comparator_path = "benchmarks/omarchy/llama_cpp_reference.json";
  for (int i = 1; i < argc; i++) {
    std::string a = argv[i];
    auto next = [&](const char* name) -> std::string {
      if (i + 1 >= argc) {
        die(std::string("missing value for ") + name, 2);
      }
      return argv[++i];
    };
    if (a == "--comparator") {
      opt.comparator_path = next("--comparator");
    } else if (a == "--ggml-comparator") {
      opt.ggml_comparator_path = next("--ggml-comparator");
    } else if (a == "--output") {
      opt.output_path = next("--output");
    } else if (a == "--source-commit") {
      opt.source_commit = next("--source-commit");
    } else if (a == "--source-dirty") {
      opt.source_dirty = true;
    } else if (a == "--dims-json") {
      opt.dims_json_path = next("--dims-json");
    } else if (a == "--spirv-dir") {
      opt.spirv_dir = next("--spirv-dir");
    } else if (a == "--warmup") {
      opt.warmup = parse_u32(next("--warmup"), "--warmup");
    } else if (a == "--reps") {
      opt.reps = parse_u32(next("--reps"), "--reps");
    } else if (a == "--rounds") {
      opt.rounds = parse_u32(next("--rounds"), "--rounds");
    } else if (a == "--idle-wait") {
      opt.idle_wait_s = parse_u32(next("--idle-wait"), "--idle-wait");
    } else if (a == "--device") {
      opt.device_index = parse_u32(next("--device"), "--device");
    } else if (a == "--profile-stages") {
      opt.profile_stages = true;
    } else if (a == "--help") {
      std::cout
          << "usage: omarchy_matmul_attention --comparator FILE [--output FILE]\n"
             "  [--source-commit HASH] [--source-dirty]\n"
             "  [--warmup N] [--reps N] [--rounds N] [--idle-wait S]\n"
             "  [--device N] [--dims-json FILE] [--spirv-dir DIR]\n"
             "  [--profile-stages] report the GPU-timestamp breakdown of the\n"
             "      attention stages (prefill dispatch, decode split/merge)\n"
             "Exit codes: 0 GO, 3 NO-GO, 2 usage/comparator error, 1 runtime error\n";
      std::exit(0);
    } else {
      die("unknown argument: " + a + " (see --help)", 2);
    }
  }
  if (opt.warmup == 0 || opt.reps == 0 || opt.rounds == 0) {
    die("--warmup, --reps, and --rounds must be at least 1", 2);
  }
  if (opt.ggml_comparator_path.empty()) {
    opt.ggml_comparator_path = "benchmarks/omarchy/ggml_op_comparator.json";
  }
  if (opt.spirv_dir.empty()) {
#ifdef OMARCHY_SPIKE_SPIRV_DIR
    opt.spirv_dir = OMARCHY_SPIKE_SPIRV_DIR;
#else
    die("no SPIR-V directory: build through CMake or pass --spirv-dir", 2);
#endif
  }
  return opt;
}

// --- Comparator --------------------------------------------------------------

struct Comparator {
  std::string commit;
  std::string build_id;
  std::string build_flags;
  std::string vulkan_device;
  std::string model;
  std::string model_sha256;
  double model_bytes = 0;
  double params_billions = 0;
  double prefill_tps = 0;
  double decode_tps = 0;
  double idle_baseline_c = -1000;
  Value raw;
};

double require_number(
    const Value& v,
    const std::string& path,
    const std::string& source_hint) {
  if (!v.is_number()) {
    die("comparator field '" + path + "' must be a number; " + source_hint, 2);
  }
  double d = v.number();
  if (!(d > 0) || !std::isfinite(d)) {
    die("comparator field '" + path + "' must be a positive measured value; " +
            source_hint,
        2);
  }
  return d;
}

std::string require_string(const Value& v, const std::string& path) {
  if (!v.is_string() || v.string().empty()) {
    die("comparator field '" + path +
            "' must be a non-empty string recorded from the llama.cpp build and "
            "llama-bench output",
        2);
  }
  return v.string();
}

// Loads and validates the pinned comparator. Every failure path prints the
// exact steps needed to produce the missing input.
Comparator load_comparator(const std::string& path) {
  const std::string kHowTo =
      "\nHow to fix:\n"
      "  1. Build the pinned llama.cpp release v0.3.0 (commit\n"
      "     c1d0e7a004015f23bc0233470b747b596f29b264; the llama-vulkan-build\n"
      "     unit on m1-test-host-linux produces it) with Vulkan.\n"
      "  2. On the same M1 machine, run against the release model\n"
      "     (~/ane-models/Qwen3.8-2B-Q4_K_M.gguf, sha256 4aa0fb13...f0ff):\n"
      "       llama-bench -m ~/ane-models/Qwen3.8-2B-Q4_K_M.gguf \\\n"
      "         -p 512 -n 128 -ngl 99 -r 5\n"
      "  3. Copy benchmarks/omarchy/llama_cpp_reference.example.json to\n"
      "     " +
      path +
      " and fill every field from the llama-bench output.\n"
      "  4. Re-run this spike. Real measured numbers only.\n";

  Comparator c;
  std::string text = read_file(path);
  if (text.empty()) {
    die("pinned llama.cpp comparator file not found: " + path +
            "\nThe spike refuses to report go/no-go without real comparator "
            "inputs." +
            kHowTo,
        2);
  }
  try {
    c.raw = omarchy_spike::json::parse(text);
  } catch (const std::exception& e) {
    die("comparator file " + path + " is not valid JSON: " + e.what() + kHowTo,
        2);
  }
  if (!c.raw.is_object()) {
    die("comparator file must contain a JSON object" + kHowTo, 2);
  }

  // Pin identity.
  if (!c.raw.has("pin") || !c.raw.at("pin").is_object()) {
    die("comparator is missing the 'pin' object (llama.cpp build identity)" +
            kHowTo,
        2);
  }
  const Value& pin = c.raw.at("pin");
  c.commit = require_string(pin.at("commit", "pin"), "pin.commit");
  c.build_id = require_string(pin.at("build_id", "pin"), "pin.build_id");
  c.build_flags =
      require_string(pin.at("build_flags", "pin"), "pin.build_flags");
  c.vulkan_device =
      require_string(pin.at("vulkan_device", "pin"), "pin.vulkan_device");
  c.model = require_string(pin.at("model", "pin"), "pin.model");
  c.model_sha256 =
      require_string(pin.at("model_sha256", "pin"), "pin.model_sha256");

  const std::string kBenchHint =
      "copy it from the llama-bench table columns (params, test t/s)";

  c.model_bytes = require_number(
      pin.at("model_bytes", "pin"), "pin.model_bytes", kBenchHint);
  c.params_billions = require_number(
      pin.at("params_billions", "pin"), "pin.params_billions", kBenchHint);

  // Workload numbers.
  if (!c.raw.has("prefill") || !c.raw.at("prefill").is_object() ||
      !c.raw.at("prefill").has("tps")) {
    die("comparator is missing 'prefill.tps' (llama-bench pp512 t/s)" + kHowTo,
        2);
  }
  if (!c.raw.has("decode") || !c.raw.at("decode").is_object() ||
      !c.raw.at("decode").has("tps")) {
    die("comparator is missing 'decode.tps' (llama-bench tg128 t/s)" + kHowTo,
        2);
  }
  c.prefill_tps = require_number(
      c.raw.at("prefill").at("tps", "prefill"), "prefill.tps", kBenchHint);
  c.decode_tps = require_number(
      c.raw.at("decode").at("tps", "decode"), "decode.tps", kBenchHint);

  if (c.raw.at("prefill").has("test") &&
      !c.raw.at("prefill").at("test").is_null()) {
    std::string t = c.raw.at("prefill").at("test").string();
    if (t != "pp512") {
      die("comparator prefill test must be 'pp512' (found '" + t + "')" +
              kHowTo,
          2);
    }
  }
  if (c.raw.at("decode").has("test") &&
      !c.raw.at("decode").at("test").is_null()) {
    std::string t = c.raw.at("decode").at("test").string();
    if (t != "tg128") {
      die("comparator decode test must be 'tg128' (found '" + t + "')" + kHowTo,
          2);
    }
  }

  // Model dims are optional context here; when present they must match the
  // effective dims. Gate dims are enforced through the ggml per-op cases.
  if (c.raw.has("model_dims") && !c.raw.at("model_dims").is_object()) {
    die("comparator 'model_dims' must be an object when present" + kHowTo, 2);
  }
  if (c.raw.has("thermal") && c.raw.at("thermal").is_object() &&
      c.raw.at("thermal").has("idle_baseline_c") &&
      c.raw.at("thermal").at("idle_baseline_c").is_number()) {
    c.idle_baseline_c = c.raw.at("thermal").at("idle_baseline_c").number();
  }
  return c;
}

void require_dims_match(const Value& model_dims, const Dims& dims) {
  auto need = [&](const char* key, double expected, double got) {
    if (got != expected) {
      die("comparator model_dims." + std::string(key) + " = " +
              std::to_string(got) + " but the spike uses " +
              std::to_string(expected) +
              ". Fix the comparator or pass --dims-json with the model's real "
              "hyperparameters.",
          2);
    }
  };
  auto num = [&](const char* key) -> double {
    const Value& v = model_dims.at(key, "model_dims");
    if (!v.is_number()) {
      die("comparator model_dims." + std::string(key) + " must be a number", 2);
    }
    return v.number();
  };
  need("hidden_size", dims.hidden, num("hidden_size"));
  need("intermediate_size", dims.intermediate, num("intermediate_size"));
  need("num_attention_heads", dims.heads, num("num_attention_heads"));
  need("num_key_value_heads", dims.kv_heads, num("num_key_value_heads"));
  need("head_dim", dims.head_dim, num("head_dim"));
  need("num_attention_layers", dims.attn_layers, num("num_attention_layers"));
}

// --- Vulkan context ----------------------------------------------------------

struct DeviceBuffer {
  VkBuffer buffer = VK_NULL_HANDLE;
  VkDeviceMemory memory = VK_NULL_HANDLE;
  void* mapped = nullptr;
  VkDeviceSize size = 0;
  bool coherent = true;
};

class Context {
 public:
  void init() {
    if (!omarchy::vk::load_loader()) {
      die("[omarchy] cannot load the Vulkan loader (libvulkan.so.1). "
          "Install the Vulkan loader and the Mesa Honeykrisp driver.",
          1);
    }
    auto& it = omarchy::vk::GetInstanceProcAddr;
    PFN_vkCreateInstance pCreateInstance =
        reinterpret_cast<PFN_vkCreateInstance>(
            it(VK_NULL_HANDLE, "vkCreateInstance"));
    PFN_vkEnumerateInstanceVersion pEnumVer =
        reinterpret_cast<PFN_vkEnumerateInstanceVersion>(
            it(VK_NULL_HANDLE, "vkEnumerateInstanceVersion"));
    if (!pCreateInstance) {
      die("[omarchy] Vulkan loader lacks vkCreateInstance.", 1);
    }
    uint32_t loader_ver = VK_API_VERSION_1_3;
    if (pEnumVer) {
      pEnumVer(&loader_ver);
    }
    uint32_t api = std::min(loader_ver, VK_API_VERSION_1_3);

    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "omarchy-matmul-attention-spike";
    app.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    app.pEngineName = "mlx-omarchy-spike";
    app.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    app.apiVersion = api;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &app;
    VKX_CHECK(pCreateInstance(&ici, nullptr, &instance_));

    resolve_instance();

    uint32_t count = 0;
    VKX_CHECK(it_.EnumeratePhysicalDevices(instance_, &count, nullptr));
    if (count == 0) {
      die("[omarchy] no Vulkan physical devices found. Is the Honeykrisp "
          "driver loaded? (Development override: "
          "MLX_OMARCHY_ALLOW_NON_APPLE=1)",
          1);
    }
    std::vector<VkPhysicalDevice> pds(count);
    VKX_CHECK(it_.EnumeratePhysicalDevices(instance_, &count, pds.data()));

    bool allow_non_apple =
        std::getenv("MLX_OMARCHY_ALLOW_NON_APPLE") != nullptr;
    std::string first_refusal;
    uint32_t supported_seen = 0;
    for (uint32_t i = 0; i < count; i++) {
      VkPhysicalDeviceDriverProperties driver{
          VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES};
      VkPhysicalDeviceProperties2 props2{
          VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
      props2.pNext = &driver;
      it_.GetPhysicalDeviceProperties2(pds[i], &props2);
      omarchy::DeviceSupport support = omarchy::classify_physical_device(
          props2.properties, int32_t(driver.driverID), allow_non_apple);
      if (support.supported) {
        if (supported_seen == device_index_) {
          pd_ = pds[i];
          support_ = support;
          props_ = props2.properties;
          driver_name_ = driver.driverName;
          break;
        }
        supported_seen++;
        continue;
      }
      if (first_refusal.empty()) {
        first_refusal = support.reason;
      }
    }
    if (pd_ == VK_NULL_HANDLE) {
      die("[omarchy] no qualifying Vulkan device" +
              (supported_seen > 0
                   ? " at index " + std::to_string(device_index_) + " (" +
                       std::to_string(supported_seen) + " supported found)"
                   : ": " +
                       (first_refusal.empty() ? std::string("none found")
                                              : first_refusal)),
          1);
    }

    // The matched ggml comparator stores activations and F16 weights in
    // 16-bit storage. Require both storage and arithmetic support together.
    VkPhysicalDeviceVulkan12Features f12{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    VkPhysicalDeviceShaderFloat16Int8Features f16f{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES};
    VkPhysicalDevice16BitStorageFeatures storage16{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES};
    VkPhysicalDeviceFeatures2 feats2{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    feats2.pNext = &storage16;
    storage16.pNext = &f16f;
    f16f.pNext = &f12;
    it_.GetPhysicalDeviceFeatures2(pd_, &feats2);
    fp16_math_ = storage16.storageBuffer16BitAccess == VK_TRUE &&
        f12.shaderFloat16 == VK_TRUE && f16f.shaderFloat16 == VK_TRUE;
    VkPhysicalDeviceVulkan12Features enabled12{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    enabled12.timelineSemaphore = VK_TRUE;
    enabled12.shaderFloat16 = fp16_math_ ? VK_TRUE : VK_FALSE;
    VkPhysicalDeviceShaderFloat16Int8Features enabled16{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES};
    enabled16.shaderFloat16 = fp16_math_ ? VK_TRUE : VK_FALSE;
    enabled16.pNext = &enabled12;
    VkPhysicalDevice16BitStorageFeatures enabled_storage16{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES};
    enabled_storage16.storageBuffer16BitAccess =
        fp16_math_ ? VK_TRUE : VK_FALSE;
    enabled_storage16.pNext = &enabled16;

    uint32_t qfamily = 0;
    uint32_t qcount = 0;
    it_.GetPhysicalDeviceQueueFamilyProperties(pd_, &qcount, nullptr);
    std::vector<VkQueueFamilyProperties> qs(qcount);
    it_.GetPhysicalDeviceQueueFamilyProperties(pd_, &qcount, qs.data());
    bool found = false;
    for (uint32_t i = 0; i < qcount; i++) {
      if (qs[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
        qfamily = i;
        timestamp_valid_bits_ = qs[i].timestampValidBits;
        found = true;
        break;
      }
    }
    if (!found) {
      die("[omarchy] device has no compute queue family.", 1);
    }
    if (timestamp_valid_bits_ == 0) {
      die("[omarchy] compute queue family does not support timestamp queries.",
          1);
    }

    float priority = 1.0f;
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex = qfamily;
    qci.queueCount = 1;
    qci.pQueuePriorities = &priority;
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.pNext = &enabled_storage16;
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    PFN_vkCreateDevice pCreateDevice =
        reinterpret_cast<PFN_vkCreateDevice>(it(instance_, "vkCreateDevice"));
    VKX_CHECK(pCreateDevice(pd_, &dci, nullptr, &device_));

    resolve_device();

    dt_.GetDeviceQueue(device_, qfamily, 0, &queue_);
    it_.GetPhysicalDeviceMemoryProperties(pd_, &mem_props_);

    VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pci.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT |
        VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pci.queueFamilyIndex = qfamily;
    VKX_CHECK(dt_.CreateCommandPool(device_, &pci, nullptr, &pool_));
    VkCommandBufferAllocateInfo cai{
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cai.commandPool = pool_;
    cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cai.commandBufferCount = 1;
    VKX_CHECK(dt_.AllocateCommandBuffers(device_, &cai, &cmd_));
    VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VKX_CHECK(dt_.CreateFence(device_, &fci, nullptr, &fence_));
    VkQueryPoolCreateInfo qpi{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
    qpi.queryType = VK_QUERY_TYPE_TIMESTAMP;
    qpi.queryCount = 1;
    VKX_CHECK(
        dt_.CreateQueryPool(device_, &qpi, nullptr, &completion_query_pool_));
  }

  ~Context() {
    if (device_ != VK_NULL_HANDLE) {
      if (completion_query_pool_ != VK_NULL_HANDLE) {
        dt_.DestroyQueryPool(device_, completion_query_pool_, nullptr);
      }
      if (fence_ != VK_NULL_HANDLE)
        dt_.DestroyFence(device_, fence_, nullptr);
      if (pool_ != VK_NULL_HANDLE)
        dt_.DestroyCommandPool(device_, pool_, nullptr);
      dt_.DestroyDevice(device_, nullptr);
    }
    if (instance_ != VK_NULL_HANDLE) {
      reinterpret_cast<PFN_vkDestroyInstance>(omarchy::vk::GetInstanceProcAddr(
          instance_, "vkDestroyInstance"))(instance_, nullptr);
    }
  }

  // --- Buffers -------------------------------------------------------------

  DeviceBuffer make_buffer(VkDeviceSize bytes) {
    DeviceBuffer b;
    b.size = bytes;
    VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bi.size = bytes;
    bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VKX_CHECK(dt_.CreateBuffer(device_, &bi, nullptr, &b.buffer));
    VkMemoryRequirements req{};
    dt_.GetBufferMemoryRequirements(device_, b.buffer, &req);
    uint32_t type = find_memory_type(
        req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize = req.size;
    ai.memoryTypeIndex = type;
    VKX_CHECK(dt_.AllocateMemory(device_, &ai, nullptr, &b.memory));
    VKX_CHECK(dt_.BindBufferMemory(device_, b.buffer, b.memory, 0));
    VKX_CHECK(dt_.MapMemory(device_, b.memory, 0, VK_WHOLE_SIZE, 0, &b.mapped));
    b.coherent = (mem_props_.memoryTypes[type].propertyFlags &
                  VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0;
    return b;
  }

  void flush(const DeviceBuffer& b) {
    if (b.coherent)
      return;
    VkMappedMemoryRange r{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
    r.memory = b.memory;
    r.size = VK_WHOLE_SIZE;
    VKX_CHECK(dt_.FlushMappedMemoryRanges(device_, 1, &r));
  }
  void invalidate(const DeviceBuffer& b) {
    if (b.coherent)
      return;
    VkMappedMemoryRange r{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
    r.memory = b.memory;
    r.size = VK_WHOLE_SIZE;
    VKX_CHECK(dt_.InvalidateMappedMemoryRanges(device_, 1, &r));
  }
  void free_buffer(DeviceBuffer& b) {
    if (b.buffer != VK_NULL_HANDLE)
      dt_.DestroyBuffer(device_, b.buffer, nullptr);
    if (b.memory != VK_NULL_HANDLE)
      dt_.FreeMemory(device_, b.memory, nullptr);
    b = {};
  }

  // --- Pipelines -------------------------------------------------------------

  struct Pipeline {
    VkDescriptorSetLayout set_layout = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkDescriptorPool pool = VK_NULL_HANDLE;
    VkDescriptorSet set = VK_NULL_HANDLE;
    uint32_t bindings = 0;
  };

  // Spec constants use sequential ids 0..n-1 of uint32 words, matching the
  // layout(constant_id = N) declarations in glsl/*.comp.
  Pipeline make_pipeline(
      const std::string& spv_file,
      uint32_t bindings,
      const void* spec_data,
      size_t spec_size,
      const std::vector<DeviceBuffer*>& buffers) {
    Pipeline p;
    p.bindings = bindings;
    std::vector<VkSpecializationMapEntry> spec_entries;
    for (uint32_t i = 0; i < spec_size / sizeof(uint32_t); i++) {
      spec_entries.push_back(
          {i, uint32_t(i * sizeof(uint32_t)), sizeof(uint32_t)});
    }

    std::vector<VkDescriptorSetLayoutBinding> lbs;
    for (uint32_t i = 0; i < bindings; i++) {
      VkDescriptorSetLayoutBinding lb{};
      lb.binding = i;
      lb.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      lb.descriptorCount = 1;
      lb.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
      lbs.push_back(lb);
    }
    VkDescriptorSetLayoutCreateInfo li{
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    li.bindingCount = bindings;
    li.pBindings = lbs.data();
    VKX_CHECK(
        dt_.CreateDescriptorSetLayout(device_, &li, nullptr, &p.set_layout));

    VkPipelineLayoutCreateInfo pli{
        VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pli.setLayoutCount = 1;
    pli.pSetLayouts = &p.set_layout;
    pli.pushConstantRangeCount = 0;
    pli.pPushConstantRanges = nullptr;
    VKX_CHECK(dt_.CreatePipelineLayout(device_, &pli, nullptr, &p.layout));

    std::string path = spirv_dir_ + "/" + spv_file;
    std::string code = read_file(path);
    if (code.empty()) {
      die("SPIR-V file missing: " + path +
              ". Build through benchmarks/omarchy/CMakeLists.txt (glslc or "
              "glslangValidator compiles glsl/*.comp) or pass --spirv-dir.",
          2);
    }
    if (code.size() % 4 != 0) {
      die("SPIR-V file " + path + " is not word aligned.", 2);
    }
    VkShaderModuleCreateInfo mi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    mi.codeSize = code.size();
    mi.pCode = reinterpret_cast<const uint32_t*>(code.data());
    VkShaderModule module = VK_NULL_HANDLE;
    VkResult res = dt_.CreateShaderModule(device_, &mi, nullptr, &module);
    if (res != VK_SUCCESS) {
      die("vkCreateShaderModule failed for " + path + " (" +
              omarchy::vk::result_string(res) +
              "); the compiled kernel is invalid.",
          1);
    }

    VkSpecializationInfo spec{};
    spec.mapEntryCount = uint32_t(spec_entries.size());
    spec.pMapEntries = spec_entries.data();
    spec.dataSize = spec_size;
    spec.pData = spec_data;

    VkPipelineShaderStageRequiredSubgroupSizeCreateInfo rssi{
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO};
    VkPipelineShaderStageCreateInfo stage{
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stage.module = module;
    stage.pName = "main";
    stage.pSpecializationInfo = spec_entries.empty() ? nullptr : &spec;
    VkComputePipelineCreateInfo cpi{
        VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpi.stage = stage;
    cpi.layout = p.layout;
    VkResult pres = dt_.CreateComputePipelines(
        device_, VK_NULL_HANDLE, 1, &cpi, nullptr, &p.pipeline);
    dt_.DestroyShaderModule(device_, module, nullptr);
    if (pres != VK_SUCCESS) {
      die("vkCreateComputePipelines failed for " + path + " (" +
              omarchy::vk::result_string(pres) + ").",
          1);
    }

    VkDescriptorPoolSize ps{};
    ps.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    ps.descriptorCount = bindings;
    VkDescriptorPoolCreateInfo pi{
        VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pi.maxSets = 1;
    pi.poolSizeCount = 1;
    pi.pPoolSizes = &ps;
    VKX_CHECK(dt_.CreateDescriptorPool(device_, &pi, nullptr, &p.pool));
    VkDescriptorSetAllocateInfo ai{
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    ai.descriptorPool = p.pool;
    ai.descriptorSetCount = 1;
    ai.pSetLayouts = &p.set_layout;
    VKX_CHECK(dt_.AllocateDescriptorSets(device_, &ai, &p.set));

    std::vector<VkDescriptorBufferInfo> infos;
    for (auto* b : buffers) {
      infos.push_back({b->buffer, 0, VK_WHOLE_SIZE});
    }
    std::vector<VkWriteDescriptorSet> writes;
    for (uint32_t i = 0; i < bindings; i++) {
      VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
      w.dstSet = p.set;
      w.dstBinding = i;
      w.descriptorCount = 1;
      w.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      w.pBufferInfo = &infos[i];
      writes.push_back(w);
    }
    dt_.UpdateDescriptorSets(
        device_, uint32_t(writes.size()), writes.data(), 0, nullptr);
    return p;
  }

  void destroy_pipeline(Pipeline& p) {
    if (p.pipeline)
      dt_.DestroyPipeline(device_, p.pipeline, nullptr);
    if (p.layout)
      dt_.DestroyPipelineLayout(device_, p.layout, nullptr);
    if (p.set_layout)
      dt_.DestroyDescriptorSetLayout(device_, p.set_layout, nullptr);
    p = {};
  }

  double
  dispatch_and_wait(const Pipeline& p, uint32_t gx, uint32_t gy, uint32_t gz) {
    VKX_CHECK(dt_.ResetCommandBuffer(cmd_, 0));
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VKX_CHECK(dt_.BeginCommandBuffer(cmd_, &bi));
    dt_.ResetQueryPool(device_, completion_query_pool_, 0, 1);
    dt_.CmdBindPipeline(cmd_, VK_PIPELINE_BIND_POINT_COMPUTE, p.pipeline);
    dt_.CmdBindDescriptorSets(
        cmd_,
        VK_PIPELINE_BIND_POINT_COMPUTE,
        p.layout,
        0,
        1,
        &p.set,
        0,
        nullptr);
    dt_.CmdDispatch(cmd_, gx, gy, gz);
    VkMemoryBarrier bar{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    bar.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    bar.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    dt_.CmdPipelineBarrier(
        cmd_,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_HOST_BIT,
        0,
        1,
        &bar,
        0,
        nullptr,
        0,
        nullptr);
    dt_.CmdWriteTimestamp(
        cmd_, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, completion_query_pool_, 0);
    VKX_CHECK(dt_.EndCommandBuffer(cmd_));
    auto t0 = std::chrono::steady_clock::now();
    submit_one();
    wait_completion();
    auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::micro>(t1 - t0).count();
  }
  double dispatch2_and_wait(
      const Pipeline& p1,
      uint32_t gx1,
      uint32_t gy1,
      uint32_t gz1,
      const Pipeline& p2,
      uint32_t gx2,
      uint32_t gy2,
      uint32_t gz2) {
    VKX_CHECK(dt_.ResetCommandBuffer(cmd_, 0));
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VKX_CHECK(dt_.BeginCommandBuffer(cmd_, &bi));
    dt_.ResetQueryPool(device_, completion_query_pool_, 0, 1);
    dt_.CmdBindPipeline(cmd_, VK_PIPELINE_BIND_POINT_COMPUTE, p1.pipeline);
    dt_.CmdBindDescriptorSets(
        cmd_,
        VK_PIPELINE_BIND_POINT_COMPUTE,
        p1.layout,
        0,
        1,
        &p1.set,
        0,
        nullptr);
    dt_.CmdDispatch(cmd_, gx1, gy1, gz1);
    VkMemoryBarrier bar{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    bar.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    bar.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    dt_.CmdPipelineBarrier(
        cmd_,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0,
        1,
        &bar,
        0,
        nullptr,
        0,
        nullptr);
    dt_.CmdBindPipeline(cmd_, VK_PIPELINE_BIND_POINT_COMPUTE, p2.pipeline);
    dt_.CmdBindDescriptorSets(
        cmd_,
        VK_PIPELINE_BIND_POINT_COMPUTE,
        p2.layout,
        0,
        1,
        &p2.set,
        0,
        nullptr);
    dt_.CmdDispatch(cmd_, gx2, gy2, gz2);
    VkMemoryBarrier bar2{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    bar2.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    bar2.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    dt_.CmdPipelineBarrier(
        cmd_,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_HOST_BIT,
        0,
        1,
        &bar2,
        0,
        nullptr,
        0,
        nullptr);
    dt_.CmdWriteTimestamp(
        cmd_, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, completion_query_pool_, 0);
    VKX_CHECK(dt_.EndCommandBuffer(cmd_));
    auto t0 = std::chrono::steady_clock::now();
    submit_one();
    wait_completion();
    auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::micro>(t1 - t0).count();
  }

  // --- GPU timestamp profiling (opt-in via --profile-stages) -----------------

  VkQueryPool make_query_pool(uint32_t count) {
    VkQueryPoolCreateInfo qi{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
    qi.queryType = VK_QUERY_TYPE_TIMESTAMP;
    qi.queryCount = count;
    VkQueryPool pool = VK_NULL_HANDLE;
    VKX_CHECK(dt_.CreateQueryPool(device_, &qi, nullptr, &pool));
    return pool;
  }
  void destroy_query_pool(VkQueryPool pool) {
    if (pool != VK_NULL_HANDLE) {
      dt_.DestroyQueryPool(device_, pool, nullptr);
    }
  }

  void get_timestamp_queries(
      VkQueryPool pool,
      uint32_t first_query,
      uint32_t count,
      uint64_t* out_ticks) {
    VKX_CHECK(dt_.GetQueryPoolResults(
        device_,
        pool,
        first_query,
        count,
        size_t(count) * sizeof(uint64_t),
        out_ticks,
        sizeof(uint64_t),
        VK_QUERY_RESULT_64_BIT));
  }

  void wait_fence() {
    VkResult res =
        dt_.WaitForFences(device_, 1, &fence_, VK_TRUE, kCompletionTimeoutNs);
    if (res == VK_TIMEOUT) {
      die("Vulkan stage profiling did not complete within 10 seconds.", 1);
    }
    VKX_CHECK(res);
  }

  // dispatch_and_wait bracketed by GPU timestamps: first_query is written at
  // TOP_OF_PIPE (dispatch not started), first_query+1 at BOTTOM_OF_PIPE
  // (dispatch complete). Otherwise byte-for-byte the timed path.
  void dispatch_timestamped_and_wait(
      const Pipeline& p,
      uint32_t gx,
      uint32_t gy,
      uint32_t gz,
      VkQueryPool pool,
      uint32_t first_query) {
    VKX_CHECK(dt_.ResetCommandBuffer(cmd_, 0));
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VKX_CHECK(dt_.BeginCommandBuffer(cmd_, &bi));
    dt_.CmdResetQueryPool(cmd_, pool, first_query, 2);
    dt_.CmdWriteTimestamp(
        cmd_, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, pool, first_query);
    dt_.CmdBindPipeline(cmd_, VK_PIPELINE_BIND_POINT_COMPUTE, p.pipeline);
    dt_.CmdBindDescriptorSets(
        cmd_,
        VK_PIPELINE_BIND_POINT_COMPUTE,
        p.layout,
        0,
        1,
        &p.set,
        0,
        nullptr);
    dt_.CmdDispatch(cmd_, gx, gy, gz);
    dt_.CmdWriteTimestamp(
        cmd_, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, pool, first_query + 1);
    VkMemoryBarrier bar{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    bar.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    bar.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    dt_.CmdPipelineBarrier(
        cmd_,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_HOST_BIT,
        0,
        1,
        &bar,
        0,
        nullptr,
        0,
        nullptr);
    VKX_CHECK(dt_.EndCommandBuffer(cmd_));
    VKX_CHECK(dt_.ResetFences(device_, 1, &fence_));
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd_;
    VKX_CHECK(dt_.QueueSubmit(queue_, 1, &si, fence_));
    wait_fence();
  }

  // dispatch2_and_wait bracketed by GPU timestamps: first_query = split not
  // started, first_query+1 = split complete, first_query+2 = merge complete.
  // The split interval is q1-q0; the merge interval q2-q1 includes the
  // launch gap behind the existing split->merge barrier.
  void dispatch2_timestamped_and_wait(
      const Pipeline& p1,
      uint32_t gx1,
      uint32_t gy1,
      uint32_t gz1,
      const Pipeline& p2,
      uint32_t gx2,
      uint32_t gy2,
      uint32_t gz2,
      VkQueryPool pool,
      uint32_t first_query) {
    VKX_CHECK(dt_.ResetCommandBuffer(cmd_, 0));
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VKX_CHECK(dt_.BeginCommandBuffer(cmd_, &bi));
    dt_.CmdResetQueryPool(cmd_, pool, first_query, 3);
    dt_.CmdWriteTimestamp(
        cmd_, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, pool, first_query);
    dt_.CmdBindPipeline(cmd_, VK_PIPELINE_BIND_POINT_COMPUTE, p1.pipeline);
    dt_.CmdBindDescriptorSets(
        cmd_,
        VK_PIPELINE_BIND_POINT_COMPUTE,
        p1.layout,
        0,
        1,
        &p1.set,
        0,
        nullptr);
    dt_.CmdDispatch(cmd_, gx1, gy1, gz1);
    VkMemoryBarrier bar{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    bar.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    bar.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    dt_.CmdPipelineBarrier(
        cmd_,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0,
        1,
        &bar,
        0,
        nullptr,
        0,
        nullptr);
    dt_.CmdWriteTimestamp(
        cmd_, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, pool, first_query + 1);
    dt_.CmdBindPipeline(cmd_, VK_PIPELINE_BIND_POINT_COMPUTE, p2.pipeline);
    dt_.CmdBindDescriptorSets(
        cmd_,
        VK_PIPELINE_BIND_POINT_COMPUTE,
        p2.layout,
        0,
        1,
        &p2.set,
        0,
        nullptr);
    dt_.CmdDispatch(cmd_, gx2, gy2, gz2);
    dt_.CmdWriteTimestamp(
        cmd_, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, pool, first_query + 2);
    VkMemoryBarrier bar2{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    bar2.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    bar2.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    dt_.CmdPipelineBarrier(
        cmd_,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_HOST_BIT,
        0,
        1,
        &bar2,
        0,
        nullptr,
        0,
        nullptr);
    VKX_CHECK(dt_.EndCommandBuffer(cmd_));
    VKX_CHECK(dt_.ResetFences(device_, 1, &fence_));
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd_;
    VKX_CHECK(dt_.QueueSubmit(queue_, 1, &si, fence_));
    wait_fence();
  }

  void wait_idle() {
    VKX_CHECK(dt_.DeviceWaitIdle(device_));
  }

  void wait_completion() {
    const auto start = std::chrono::steady_clock::now();
    const auto spin_deadline =
        start + std::chrono::microseconds(kCompletionSpinUs);
    const auto hard_deadline =
        start + std::chrono::nanoseconds(kCompletionTimeoutNs);
    uint64_t tick = 0;
    for (;;) {
      VkResult res = dt_.GetQueryPoolResults(
          device_,
          completion_query_pool_,
          0,
          1,
          sizeof(tick),
          &tick,
          sizeof(tick),
          VK_QUERY_RESULT_64_BIT);
      if (res == VK_SUCCESS) {
        return;
      }
      if (res != VK_NOT_READY) {
        VKX_CHECK(res);
      }
      const auto now = std::chrono::steady_clock::now();
      if (now >= hard_deadline) {
        die("Vulkan benchmark dispatch did not complete within 10 seconds.", 1);
      }
      if (now >= spin_deadline) {
        std::this_thread::sleep_for(kCompletionPollDelay);
        continue;
      }
      for (uint32_t i = 0; i < 800; i++) {
        OMARCHY_CPU_YIELD();
      }
    }
  }

  void submit_one() {
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd_;
    VKX_CHECK(dt_.QueueSubmit(queue_, 1, &si, VK_NULL_HANDLE));
  }

  bool fp16_math() const {
    return fp16_math_;
  }
  const VkPhysicalDeviceProperties& props() const {
    return props_;
  }
  const std::string& driver_name() const {
    return driver_name_;
  }
  const omarchy::DeviceSupport& support() const {
    return support_;
  }
  const VkPhysicalDeviceMemoryProperties& mem_props() const {
    return mem_props_;
  }
  uint32_t max_shared() const {
    VkPhysicalDeviceProperties2 p2{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    it_.GetPhysicalDeviceProperties2(pd_, &p2);
    return p2.properties.limits.maxComputeSharedMemorySize;
  }
  uint32_t max_invocations() const {
    VkPhysicalDeviceProperties2 p2{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    it_.GetPhysicalDeviceProperties2(pd_, &p2);
    return p2.properties.limits.maxComputeWorkGroupInvocations;
  }
  void set_spirv_dir(std::string dir) {
    spirv_dir_ = std::move(dir);
  }
  void set_device_index(uint32_t index) {
    device_index_ = index;
  }
  // GPU timestamp capability of the compute queue family, captured in init().
  uint32_t timestamp_valid_bits() const {
    return timestamp_valid_bits_;
  }
  float timestamp_period_ns() const {
    return props_.limits.timestampPeriod;
  }
  bool gpu_timestamps_supported() const {
    return timestamp_valid_bits_ > 0 && props_.limits.timestampPeriod > 0.0f;
  }

 private:
  void resolve_instance() {
    auto load = [&](auto& fn, const char* name) {
      fn = reinterpret_cast<std::remove_reference_t<decltype(fn)>>(
          omarchy::vk::GetInstanceProcAddr(instance_, name));
      if (!fn)
        die(std::string("[omarchy] missing instance function ") + name, 1);
    };
    load(it_.EnumeratePhysicalDevices, "vkEnumeratePhysicalDevices");
    load(it_.GetPhysicalDeviceProperties2, "vkGetPhysicalDeviceProperties2");
    load(it_.GetPhysicalDeviceFeatures2, "vkGetPhysicalDeviceFeatures2");
    load(
        it_.GetPhysicalDeviceQueueFamilyProperties,
        "vkGetPhysicalDeviceQueueFamilyProperties");
    load(
        it_.GetPhysicalDeviceMemoryProperties,
        "vkGetPhysicalDeviceMemoryProperties");
  }

  void resolve_device() {
    PFN_vkGetDeviceProcAddr gpa = reinterpret_cast<PFN_vkGetDeviceProcAddr>(
        omarchy::vk::GetInstanceProcAddr(instance_, "vkGetDeviceProcAddr"));
    if (!gpa)
      die("[omarchy] missing vkGetDeviceProcAddr", 1);
    auto load = [&](auto& fn, const char* name) {
      fn = reinterpret_cast<std::remove_reference_t<decltype(fn)>>(
          gpa(device_, name));
      if (!fn)
        die(std::string("[omarchy] missing device function ") + name, 1);
    };
    load(dt_.GetDeviceQueue, "vkGetDeviceQueue");
    load(dt_.DeviceWaitIdle, "vkDeviceWaitIdle");
    load(dt_.DestroyDevice, "vkDestroyDevice");
    load(dt_.AllocateMemory, "vkAllocateMemory");
    load(dt_.FreeMemory, "vkFreeMemory");
    load(dt_.MapMemory, "vkMapMemory");
    load(dt_.UnmapMemory, "vkUnmapMemory");
    load(dt_.CreateBuffer, "vkCreateBuffer");
    load(dt_.DestroyBuffer, "vkDestroyBuffer");
    load(dt_.GetBufferMemoryRequirements, "vkGetBufferMemoryRequirements");
    load(dt_.BindBufferMemory, "vkBindBufferMemory");
    load(dt_.CreateCommandPool, "vkCreateCommandPool");
    load(dt_.DestroyCommandPool, "vkDestroyCommandPool");
    load(dt_.AllocateCommandBuffers, "vkAllocateCommandBuffers");
    load(dt_.CreateShaderModule, "vkCreateShaderModule");
    load(dt_.DestroyShaderModule, "vkDestroyShaderModule");
    load(dt_.CreateDescriptorSetLayout, "vkCreateDescriptorSetLayout");
    load(dt_.DestroyDescriptorSetLayout, "vkDestroyDescriptorSetLayout");
    load(dt_.CreateDescriptorPool, "vkCreateDescriptorPool");
    load(dt_.DestroyDescriptorPool, "vkDestroyDescriptorPool");
    load(dt_.AllocateDescriptorSets, "vkAllocateDescriptorSets");
    load(dt_.UpdateDescriptorSets, "vkUpdateDescriptorSets");
    load(dt_.CreatePipelineLayout, "vkCreatePipelineLayout");
    load(dt_.DestroyPipelineLayout, "vkDestroyPipelineLayout");
    load(dt_.CreateComputePipelines, "vkCreateComputePipelines");
    load(dt_.DestroyPipeline, "vkDestroyPipeline");
    load(dt_.CmdBindPipeline, "vkCmdBindPipeline");
    load(dt_.CmdBindDescriptorSets, "vkCmdBindDescriptorSets");
    load(dt_.CmdDispatch, "vkCmdDispatch");
    load(dt_.CmdPipelineBarrier, "vkCmdPipelineBarrier");
    load(dt_.QueueSubmit, "vkQueueSubmit");
    load(dt_.WaitForFences, "vkWaitForFences");
    load(dt_.CreateFence, "vkCreateFence");
    load(dt_.DestroyFence, "vkDestroyFence");
    load(dt_.ResetFences, "vkResetFences");
    load(dt_.BeginCommandBuffer, "vkBeginCommandBuffer");
    load(dt_.EndCommandBuffer, "vkEndCommandBuffer");
    load(dt_.ResetCommandBuffer, "vkResetCommandBuffer");
    load(dt_.CreateQueryPool, "vkCreateQueryPool");
    load(dt_.DestroyQueryPool, "vkDestroyQueryPool");
    load(dt_.GetQueryPoolResults, "vkGetQueryPoolResults");
    load(dt_.ResetQueryPool, "vkResetQueryPool");
    load(dt_.CmdResetQueryPool, "vkCmdResetQueryPool");
    load(dt_.CmdWriteTimestamp, "vkCmdWriteTimestamp");
    load(dt_.FlushMappedMemoryRanges, "vkFlushMappedMemoryRanges");
    load(dt_.InvalidateMappedMemoryRanges, "vkInvalidateMappedMemoryRanges");
  }

  uint32_t find_memory_type(uint32_t bits, VkMemoryPropertyFlags want) {
    for (uint32_t i = 0; i < mem_props_.memoryTypeCount; i++) {
      if ((bits & (1u << i)) &&
          (mem_props_.memoryTypes[i].propertyFlags & want) == want) {
        return i;
      }
    }
    die("[omarchy] no host-visible memory type on the qualifying device.", 1);
  }

  struct InstanceTable {
    PFN_vkEnumeratePhysicalDevices EnumeratePhysicalDevices{nullptr};
    PFN_vkGetPhysicalDeviceProperties2 GetPhysicalDeviceProperties2{nullptr};
    PFN_vkGetPhysicalDeviceFeatures2 GetPhysicalDeviceFeatures2{nullptr};
    PFN_vkGetPhysicalDeviceQueueFamilyProperties
        GetPhysicalDeviceQueueFamilyProperties{nullptr};
    PFN_vkGetPhysicalDeviceMemoryProperties GetPhysicalDeviceMemoryProperties{
        nullptr};
  };

  VkInstance instance_{VK_NULL_HANDLE};
  InstanceTable it_{};
  VkPhysicalDevice pd_{VK_NULL_HANDLE};
  omarchy::DeviceSupport support_{};
  VkPhysicalDeviceProperties props_{};
  std::string driver_name_;
  VkDevice device_{VK_NULL_HANDLE};
  struct DeviceTable {
    PFN_vkGetDeviceQueue GetDeviceQueue{nullptr};
    PFN_vkDeviceWaitIdle DeviceWaitIdle{nullptr};
    PFN_vkDestroyDevice DestroyDevice{nullptr};
    PFN_vkAllocateMemory AllocateMemory{nullptr};
    PFN_vkFreeMemory FreeMemory{nullptr};
    PFN_vkMapMemory MapMemory{nullptr};
    PFN_vkUnmapMemory UnmapMemory{nullptr};
    PFN_vkCreateBuffer CreateBuffer{nullptr};
    PFN_vkDestroyBuffer DestroyBuffer{nullptr};
    PFN_vkGetBufferMemoryRequirements GetBufferMemoryRequirements{nullptr};
    PFN_vkBindBufferMemory BindBufferMemory{nullptr};
    PFN_vkCreateCommandPool CreateCommandPool{nullptr};
    PFN_vkDestroyCommandPool DestroyCommandPool{nullptr};
    PFN_vkAllocateCommandBuffers AllocateCommandBuffers{nullptr};
    PFN_vkCreateShaderModule CreateShaderModule{nullptr};
    PFN_vkDestroyShaderModule DestroyShaderModule{nullptr};
    PFN_vkCreateDescriptorSetLayout CreateDescriptorSetLayout{nullptr};
    PFN_vkDestroyDescriptorSetLayout DestroyDescriptorSetLayout{nullptr};
    PFN_vkCreateDescriptorPool CreateDescriptorPool{nullptr};
    PFN_vkDestroyDescriptorPool DestroyDescriptorPool{nullptr};
    PFN_vkAllocateDescriptorSets AllocateDescriptorSets{nullptr};
    PFN_vkUpdateDescriptorSets UpdateDescriptorSets{nullptr};
    PFN_vkCreatePipelineLayout CreatePipelineLayout{nullptr};
    PFN_vkDestroyPipelineLayout DestroyPipelineLayout{nullptr};
    PFN_vkCreateComputePipelines CreateComputePipelines{nullptr};
    PFN_vkDestroyPipeline DestroyPipeline{nullptr};
    PFN_vkCmdBindPipeline CmdBindPipeline{nullptr};
    PFN_vkCmdBindDescriptorSets CmdBindDescriptorSets{nullptr};
    PFN_vkCmdDispatch CmdDispatch{nullptr};
    PFN_vkCmdPipelineBarrier CmdPipelineBarrier{nullptr};
    PFN_vkQueueSubmit QueueSubmit{nullptr};
    PFN_vkWaitForFences WaitForFences{nullptr};
    PFN_vkCreateFence CreateFence{nullptr};
    PFN_vkDestroyFence DestroyFence{nullptr};
    PFN_vkResetFences ResetFences{nullptr};
    PFN_vkBeginCommandBuffer BeginCommandBuffer{nullptr};
    PFN_vkEndCommandBuffer EndCommandBuffer{nullptr};
    PFN_vkResetCommandBuffer ResetCommandBuffer{nullptr};
    PFN_vkCreateQueryPool CreateQueryPool{nullptr};
    PFN_vkDestroyQueryPool DestroyQueryPool{nullptr};
    PFN_vkGetQueryPoolResults GetQueryPoolResults{nullptr};
    PFN_vkResetQueryPool ResetQueryPool{nullptr};
    PFN_vkCmdResetQueryPool CmdResetQueryPool{nullptr};
    PFN_vkCmdWriteTimestamp CmdWriteTimestamp{nullptr};
    PFN_vkFlushMappedMemoryRanges FlushMappedMemoryRanges{nullptr};
    PFN_vkInvalidateMappedMemoryRanges InvalidateMappedMemoryRanges{nullptr};
  };
  DeviceTable dt_{};
  VkQueue queue_{VK_NULL_HANDLE};
  VkCommandPool pool_{VK_NULL_HANDLE};
  VkCommandBuffer cmd_{VK_NULL_HANDLE};
  VkFence fence_{VK_NULL_HANDLE};
  VkQueryPool completion_query_pool_{VK_NULL_HANDLE};
  VkPhysicalDeviceMemoryProperties mem_props_{};
  bool fp16_math_{false};
  std::string spirv_dir_;
  uint32_t device_index_{0};
  uint32_t timestamp_valid_bits_{0};
};

struct Stats {
  double median_us = 0;
  double min_us = 0;
  double max_us = 0;
  double p95_us = 0;
  double mean_us = 0;
  std::vector<double> samples;
};

Stats summarize_samples(std::vector<double> samples) {
  if (samples.empty()) {
    die("cannot summarize an empty timing sample set", 1);
  }
  std::sort(samples.begin(), samples.end());
  Stats stats;
  stats.min_us = samples.front();
  stats.max_us = samples.back();
  stats.median_us = samples[samples.size() / 2];
  stats.p95_us = samples[std::min(
      samples.size() - 1, size_t(0.95 * (samples.size() - 1)))];
  double sum = 0;
  for (double sample : samples) {
    sum += sample;
  }
  stats.mean_us = sum / samples.size();
  stats.samples = std::move(samples);
  return stats;
}

Stats time_row(
    Context& ctx,
    const Context::Pipeline& pipeline,
    uint32_t gx,
    uint32_t gy,
    uint32_t gz,
    uint32_t warmup,
    uint32_t reps) {
  for (uint32_t i = 0; i < warmup; i++) {
    ctx.dispatch_and_wait(pipeline, gx, gy, gz);
  }
  std::vector<double> samples;
  samples.reserve(reps);
  for (uint32_t i = 0; i < reps; i++) {
    samples.push_back(ctx.dispatch_and_wait(pipeline, gx, gy, gz));
  }
  return summarize_samples(std::move(samples));
}

Stats time_row2(
    Context& ctx,
    const Context::Pipeline& first,
    uint32_t first_gx,
    uint32_t first_gy,
    uint32_t first_gz,
    const Context::Pipeline& second,
    uint32_t second_gx,
    uint32_t second_gy,
    uint32_t second_gz,
    uint32_t warmup,
    uint32_t reps) {
  for (uint32_t i = 0; i < warmup; i++) {
    ctx.dispatch2_and_wait(
        first,
        first_gx,
        first_gy,
        first_gz,
        second,
        second_gx,
        second_gy,
        second_gz);
  }
  std::vector<double> samples;
  samples.reserve(reps);
  for (uint32_t i = 0; i < reps; i++) {
    samples.push_back(ctx.dispatch2_and_wait(
        first,
        first_gx,
        first_gy,
        first_gz,
        second,
        second_gx,
        second_gy,
        second_gz));
  }
  return summarize_samples(std::move(samples));
}

// --- Thermal zones
// --------------------------------------------------------------

struct ThermalSample {
  std::string zone;
  double temp_c = 0;
};

std::vector<ThermalSample> read_thermal_zones() {
  std::vector<ThermalSample> out;
  for (int i = 0; i < 32; i++) {
    std::string base = "/sys/class/thermal/thermal_zone" + std::to_string(i);
    std::string type = read_file(base + "/type");
    std::string temp = read_file(base + "/temp");
    if (type.empty() || temp.empty()) {
      continue;
    }
    while (!type.empty() && (type.back() == '\n'))
      type.pop_back();
    double milli = std::atof(temp.c_str());
    if (milli == 0 && temp[0] != '0') {
      continue;
    }
    out.push_back({type, milli / 1000.0});
  }
  return out;
}

// Verbatim lm-sensors output when /usr/bin/sensors exists; empty otherwise.
std::string read_sensors_output() {
  FILE* p = popen("/usr/bin/sensors 2>/dev/null", "r");
  if (!p) {
    return "";
  }
  std::string out;
  char buf[4096];
  size_t n;
  while ((n = fread(buf, 1, sizeof(buf), p)) > 0) {
    out.append(buf, n);
  }
  pclose(p);
  return out;
}

Value thermal_json(
    const std::vector<ThermalSample>& zones,
    const std::string& sensors_output) {
  Object o;
  omarchy_spike::json::Array zs;
  double max_c = -1000;
  for (auto& z : zones) {
    Object zj;
    zj["zone"] = Value(z.zone);
    zj["temp_c"] = Value(z.temp_c);
    zs.push_back(Value(std::move(zj)));
    max_c = std::max(max_c, z.temp_c);
  }
  o["zones"] = Value(std::move(zs));
  o["max_zone_temp_c"] = Value(max_c > -1000 ? Value(max_c) : Value());
  if (!sensors_output.empty()) {
    o["sensors_output"] = Value(sensors_output);
  }
  return Value(std::move(o));
}

} // namespace
} // namespace omarchy_spike

// --- Benchmark rows and gate (part 2)
// -----------------------------------------

namespace omarchy_spike {

struct Row {
  std::string name;
  std::string section; // prefill | decode | info
  std::string kind; // gemm | gemv | sdpa_prefill | sdpa_decode
  bool gate = false; // contributes to the 80 percent go/no-go
  std::string math; // fp16 | fp32
  std::string quant; // none | q4_k
  uint32_t M = 0, K = 0, N = 0;
  uint32_t sdpa_h = 0, sdpa_hkv = 0, sdpa_hd = 0, sdpa_kv = 0;
  double flops = 0; // one execution
  double weight_bytes = 0;
  double metric = 0; // gflops (gate metric for every gate row)
  double cmp_gflops = 0; // matched ggml comparator throughput (gate rows)
  double cmp_median_us = 0;
  Stats stats;
};

// --- Pinned ggml per-op comparator -------------------------------------------

// Loads the machine-generated ggml_op_comparator.json produced by the runner
// from the pinned llama.cpp checkout. Missing or non-positive values are a
// hard error: the gate never runs against invented numbers.
Value load_ggml_comparator(const std::string& path) {
  const std::string kHowTo =
      "\nHow to fix:\n"
      "  The runner builds benchmarks/omarchy/ggml_op_probe.cpp against the\n"
      "  pinned llama.cpp tree (LLAMA_CPP_DIR, default ~/src/llama.cpp) with\n"
      "  GGML_VULKAN=ON and writes ggml_op_comparator.json automatically.\n"
      "  Run tools/ci/run-omarchy-spike.sh, or fix the path with\n"
      "  --ggml-comparator FILE.\n";
  std::string text = read_file(path);
  if (text.empty()) {
    die("pinned ggml per-op comparator file not found: " + path +
            "\nThe 80 percent gate compares matched kernels; without this "
            "probe output the spike refuses to report go/no-go." +
            kHowTo,
        2);
  }
  Value v;
  try {
    v = omarchy_spike::json::parse(text);
  } catch (const std::exception& e) {
    die("ggml comparator file " + path + " is not valid JSON: " + e.what() +
            kHowTo,
        2);
  }
  if (!v.is_object() || !v.has("cases") || !v.at("cases").is_array() ||
      v.at("cases").array().empty()) {
    die("ggml comparator file " + path + " has no 'cases' array" + kHowTo, 2);
  }
  return v;
}

struct GgmlCase {
  std::string op;
  double median_us = 0;
  double flops = 0;
  double m = 0, k = 0, n = 0, kv = 0, h = 0, hkv = 0, d = 0;
};

// Maps spike rows to comparator cases by name and enforces that dimensions
// and quantization match before any ratio is computed.
std::map<std::string, GgmlCase> match_ggml_cases(
    const Value& gcmp,
    const std::vector<Row>& rows,
    const Dims& dims) {
  std::map<std::string, GgmlCase> out;
  for (auto& r : rows) {
    if (!r.gate)
      continue;
    const Value* found = nullptr;
    for (auto& c : gcmp.at("cases").array()) {
      if (c.is_object() && c.has("name") && c.at("name").is_string() &&
          c.at("name").string() == r.name) {
        found = &c;
        break;
      }
    }
    if (!found) {
      die("ggml comparator is missing case '" + r.name +
              "'. The probe in ggml_op_probe.cpp must cover every gate row.",
          2);
    }
    GgmlCase gc;
    gc.op = found->at("op").string();
    const Value& us = found->at("median_us");
    if (!us.is_number() || !(us.number() > 0) || !std::isfinite(us.number())) {
      die("ggml comparator case '" + r.name +
              "' has no positive measured median_us; the gate refuses invented values.",
          2);
    }
    gc.median_us = us.number();
    gc.flops = found->at("flops").number();
    std::string want_op;
    if (r.kind == "gemm")
      want_op = "mul_mat_f16";
    if (r.kind == "gemv" && r.quant == "q4_k")
      want_op = "mul_mat_q4k";
    if (r.kind == "sdpa_prefill" || r.kind == "sdpa_decode")
      want_op = "flash_attn_ext";
    if (gc.op != want_op) {
      die("ggml comparator case '" + r.name + "' op '" + gc.op +
              "' does not match the spike row op '" + want_op + "'",
          2);
    }
    auto num = [&](const char* key) -> double {
      return found->at(key).number();
    };
    if (r.kind == "gemm" || r.kind == "gemv") {
      gc.m = num("M");
      gc.k = num("K");
      gc.n = num("N");
      if (gc.m != r.M || gc.k != r.K || gc.n != r.N) {
        die("ggml comparator case '" + r.name +
                "' dims M=" + std::to_string(gc.m) +
                " K=" + std::to_string(gc.k) + " N=" + std::to_string(gc.n) +
                " do not match the spike row M=" + std::to_string(r.M) +
                " K=" + std::to_string(r.K) + " N=" + std::to_string(r.N),
            2);
      }
    } else {
      gc.m = num("M");
      gc.kv = num("KV");
      gc.h = num("H");
      gc.hkv = num("HKV");
      gc.d = num("D");
      if (gc.m != r.M || gc.kv != r.sdpa_kv || gc.h != r.sdpa_h ||
          gc.hkv != r.sdpa_hkv || gc.d != r.sdpa_hd) {
        die("ggml comparator case '" + r.name +
                "' dims do not match the spike sdpa row (M/KV/H/HKV/D)",
            2);
      }
      if (gc.h != dims.heads || gc.hkv != dims.kv_heads ||
          gc.d != dims.head_dim) {
        die("ggml comparator case '" + r.name +
                "' H/HKV/D do not match model_dims",
            2);
      }
    }
    out.emplace(r.name, gc);
  }
  return out;
}

double ggml_case_gflops(const GgmlCase& c) {
  return c.flops / (c.median_us / 1e6) / 1e9;
}

Value row_json(const Row& r, uint32_t warmup, uint32_t reps) {
  Object o;
  o["name"] = Value(r.name);
  o["section"] = Value(r.section);
  o["kind"] = Value(r.kind);
  o["gate"] = Value(r.gate);
  o["math"] = Value(r.math);
  o["quantization"] = Value(r.quant);
  if (r.quant == "q4_k") {
    o["io_dtype"] = Value("fp16_activation_q4_k_weights_fp32_output");
  } else if (r.math == "fp16" || r.kind == "sdpa_decode") {
    o["io_dtype"] = Value("fp16_inputs_fp32_output");
  } else {
    o["io_dtype"] = Value("fp32");
  }
  o["math_dtype"] = Value(r.math);
  if (r.kind == "gemm" || r.kind == "gemv") {
    o["m"] = Value(double(r.M));
    o["k"] = Value(double(r.K));
    o["n"] = Value(double(r.N));
  } else {
    o["queries"] = Value(double(r.M));
    o["kv_len"] = Value(double(r.sdpa_kv));
    o["heads"] = Value(double(r.sdpa_h));
    o["kv_heads"] = Value(double(r.sdpa_hkv));
    o["head_dim"] = Value(double(r.sdpa_hd));
  }
  o["warmup_iters"] = Value(double(warmup));
  o["repetitions"] = Value(double(reps));
  o["flops_per_exec"] = Value(r.flops);
  o["weight_bytes"] = Value(r.weight_bytes);
  Object st;
  st["median_us"] = Value(r.stats.median_us);
  st["min_us"] = Value(r.stats.min_us);
  st["max_us"] = Value(r.stats.max_us);
  st["p95_us"] = Value(r.stats.p95_us);
  st["mean_us"] = Value(r.stats.mean_us);
  o["timing"] = Value(std::move(st));
  o["metric"] = Value("gflops");
  o["metric_value"] = Value(r.metric);
  if (r.section == "decode" && r.quant == "q4_k") {
    o["decode_bandwidth_gbps"] =
        Value(r.weight_bytes / (r.stats.median_us / 1000.0 / 1000.0) / 1e9);
  }
  if (r.gate && r.cmp_median_us > 0) {
    o["comparator_gflops"] = Value(r.cmp_gflops);
    o["comparator_median_us"] = Value(r.cmp_median_us);
  }
  return Value(std::move(o));
}

struct SelfCheck {
  std::string kernel;
  double max_rel_err = 0;
  double tolerance = 0;
  bool pass = false;
};

struct Rng {
  std::mt19937 mt{20260830};
  std::uniform_real_distribution<float> uni{-1.0f, 1.0f};
  float next() {
    return uni(mt);
  }
};

void fill_random(DeviceBuffer& b, size_t count, Rng& rng, Context& ctx) {
  auto* p = static_cast<float*>(b.mapped);
  for (size_t i = 0; i < count; i++) {
    p[i] = rng.next();
  }
  ctx.flush(b);
}
void fill_random_f16(DeviceBuffer& b, size_t count, Rng& rng, Context& ctx) {
  auto* p = static_cast<uint16_t*>(b.mapped);
  for (size_t i = 0; i < count; i++) {
    p[i] = ref::f32_to_f16(rng.next());
  }
  ctx.flush(b);
}

std::vector<float> read_f16_values(const DeviceBuffer& b, size_t count) {
  const auto* p = static_cast<const uint16_t*>(b.mapped);
  std::vector<float> values(count);
  for (size_t i = 0; i < count; i++) {
    values[i] = ref::f16_to_f32(p[i]);
  }
  return values;
}

double max_rel_err(
    const std::vector<float>& got,
    const std::vector<float>& want) {
  double worst = 0;
  for (size_t i = 0; i < want.size(); i++) {
    double denom = std::max(1.0, std::fabs(double(want[i])));
    worst =
        std::max(worst, std::fabs(double(got[i]) - double(want[i])) / denom);
  }
  return worst;
}

std::vector<uint32_t> spec_words(std::initializer_list<uint32_t> values) {
  return std::vector<uint32_t>(values);
}

Value self_check_json(const std::vector<SelfCheck>& checks) {
  omarchy_spike::json::Array a;
  for (auto& c : checks) {
    Object o;
    o["kernel"] = Value(c.kernel);
    o["max_rel_err"] = Value(c.max_rel_err);
    o["tolerance"] = Value(c.tolerance);
    o["pass"] = Value(c.pass);
    a.push_back(Value(std::move(o)));
  }
  return Value(std::move(a));
}

// Validates one GEMM pipeline against the CPU reference.
void validate_gemm(
    Context& ctx,
    const std::string& spv,
    const std::string& label,
    bool fp16_math,
    uint32_t M,
    uint32_t N,
    uint32_t K,
    double tol,
    std::vector<SelfCheck>& checks) {
  Rng rng;
  size_t input_bytes = fp16_math ? 2 : 4;
  DeviceBuffer a = ctx.make_buffer(size_t(M) * K * input_bytes);
  DeviceBuffer w = ctx.make_buffer(size_t(N) * K * input_bytes);
  DeviceBuffer c = ctx.make_buffer(size_t(M) * N * 4);
  if (fp16_math) {
    fill_random_f16(a, size_t(M) * K, rng, ctx);
    fill_random_f16(w, size_t(N) * K, rng, ctx);
  } else {
    fill_random(a, size_t(M) * K, rng, ctx);
    fill_random(w, size_t(N) * K, rng, ctx);
  }
  std::vector<uint32_t> words = spec_words({M, N, K});
  auto p =
      ctx.make_pipeline(spv, 3, words.data(), words.size() * 4, {&a, &w, &c});
  ctx.dispatch_and_wait(p, (N + 63) / 64, (M + 63) / 64, 1);
  ctx.wait_idle();
  ctx.invalidate(c);
  std::vector<float> ref;
  std::vector<float> av = fp16_math
      ? read_f16_values(a, size_t(M) * K)
      : std::vector<float>(
            static_cast<float*>(a.mapped),
            static_cast<float*>(a.mapped) + size_t(M) * K);
  std::vector<float> wv = fp16_math
      ? read_f16_values(w, size_t(N) * K)
      : std::vector<float>(
            static_cast<float*>(w.mapped),
            static_cast<float*>(w.mapped) + size_t(N) * K);
  ref::gemm(av, wv, ref, M, N, K);
  double err = max_rel_err(
      std::vector<float>(
          static_cast<float*>(c.mapped),
          static_cast<float*>(c.mapped) + size_t(M) * N),
      ref);
  SelfCheck sc{label, err, tol, err <= tol};
  checks.push_back(sc);
  if (!sc.pass) {
    die("self-check FAILED for " + label + ": max rel err " +
            std::to_string(err) + " > tolerance " + std::to_string(tol) +
            ". The kernel is wrong; refusing to time it.",
        1);
  }

  // Controls: the tile-algorithm simulation must reproduce the reference
  // with the fixed transposed W load, and must reject the historical load
  // that indexed K by lx (every kk consumed the same element).
  std::vector<float> algorithm_ref;
  ref::gemm(av, wv, algorithm_ref, M, N, K);
  std::vector<float> sim;
  ref::gemm_tile_simulate(av, wv, sim, M, N, K, false);
  double sim_err = max_rel_err(sim, algorithm_ref);
  SelfCheck pos{label + "_sim_fixed", sim_err, 1e-9, sim_err <= 1e-9};
  checks.push_back(pos);
  if (!pos.pass) {
    die("self-check FAILED for " + label +
            "_sim_fixed: the tile simulation "
            "disagrees with the reference (rel err " +
            std::to_string(sim_err) + ")",
        1);
  }
  ref::gemm_tile_simulate(av, wv, sim, M, N, K, true);
  double legacy_err = max_rel_err(sim, algorithm_ref);
  SelfCheck neg{
      label + "_sim_legacy_negative_control",
      legacy_err,
      tol,
      legacy_err > tol};
  checks.push_back(neg);
  if (!neg.pass) {
    die("self-check FAILED for " + label +
            "_sim_legacy_negative_control: "
            "the broken legacy W load was expected to mismatch the reference "
            "but matched (rel err " +
            std::to_string(legacy_err) + ")",
        1);
  }
  ctx.destroy_pipeline(p);
  ctx.free_buffer(a);
  ctx.free_buffer(w);
  ctx.free_buffer(c);
}

void validate_gemv_f32(
    Context& ctx,
    uint32_t N,
    uint32_t K,
    double tol,
    std::vector<SelfCheck>& checks) {
  Rng rng;
  DeviceBuffer x = ctx.make_buffer(size_t(K) * 4);
  DeviceBuffer w = ctx.make_buffer(size_t(N) * K * 4);
  DeviceBuffer o = ctx.make_buffer(size_t(N) * 4);
  fill_random(x, K, rng, ctx);
  fill_random(w, size_t(N) * K, rng, ctx);
  std::vector<uint32_t> words = spec_words({N, K});
  auto p = ctx.make_pipeline(
      "gemv_f32.spv", 3, words.data(), words.size() * 4, {&x, &w, &o});
  ctx.dispatch_and_wait(p, N, 1, 1);
  ctx.wait_idle();
  ctx.invalidate(o);
  std::vector<float> ref;
  ref::gemv(
      std::vector<float>(
          static_cast<float*>(x.mapped), static_cast<float*>(x.mapped) + K),
      std::vector<float>(
          static_cast<float*>(w.mapped),
          static_cast<float*>(w.mapped) + size_t(N) * K),
      ref,
      N,
      K);
  double err = max_rel_err(
      std::vector<float>(
          static_cast<float*>(o.mapped), static_cast<float*>(o.mapped) + N),
      ref);
  SelfCheck sc{"gemv_f32", err, tol, err <= tol};
  checks.push_back(sc);
  if (!sc.pass) {
    die("self-check FAILED for gemv_f32: max rel err " + std::to_string(err) +
            " > tolerance " + std::to_string(tol),
        1);
  }
  ctx.destroy_pipeline(p);
  ctx.free_buffer(x);
  ctx.free_buffer(w);
  ctx.free_buffer(o);
}

// GPU Q4_K layout: 40 words (160 bytes) per 256-value block. Words 0..7 hold
// precomputed fp16 pairs d*scale and dmin*min; words 8..39 are canonical
// quant words 4..35 unchanged.
constexpr uint32_t kQ4kGpuWordsPerBlock = 40; // 160 bytes

// Converts one row of canonical 36-word Q4_K blocks into the 40-word GPU
// layout. k must be an exact multiple of the 256-value block size.
void q4k_row_to_gpu(
    const std::vector<uint32_t>& src,
    std::vector<uint32_t>& dst,
    uint32_t k) {
  if (k == 0 || k % ref::kQkK != 0) {
    die("q4k_row_to_gpu: K=" + std::to_string(k) +
            " is not an exact multiple of the 256-value Q4_K block size",
        2);
  }
  uint32_t nb = k / ref::kQkK;
  dst.assign(size_t(nb) * kQ4kGpuWordsPerBlock, 0);
  for (uint32_t b = 0; b < nb; b++) {
    uint32_t s = b * ref::kQ4kWordsPerBlock;
    uint32_t t = b * kQ4kGpuWordsPerBlock;
    float d = ref::f16_to_f32(src[s] & 0xFFFF);
    float dmin = ref::f16_to_f32(src[s] >> 16);
    for (uint32_t j = 0; j < 8; j++) {
      uint32_t sc, mn;
      ref::q4k_get_scale_min_k4(j, src, s, sc, mn);
      uint32_t g = (j / 2) % 2; // v_im half holding scale index j
      uint32_t h = j / 4; // low or high pair inside that half
      uint32_t p = j % 2; // fp16 slot inside the word
      dst[t + 4 * g + h] |= uint32_t(ref::f32_to_f16(d * float(sc)))
          << (16 * p);
      dst[t + 4 * g + 2 + h] |= uint32_t(ref::f32_to_f16(dmin * float(mn)))
          << (16 * p);
    }
    std::memcpy(&dst[t + 8], &src[s + 4], 32 * sizeof(uint32_t));
  }
}

void validate_gemv_q4k(
    Context& ctx,
    uint32_t N,
    uint32_t K,
    double tol,
    std::vector<SelfCheck>& checks) {
  Rng rng;
  uint32_t row_words = (K / ref::kQkK) * kQ4kGpuWordsPerBlock;
  std::vector<std::vector<uint32_t>> packed;
  for (uint32_t n = 0; n < N; n++) {
    packed.push_back(ref::q4k_make_row(rng.mt, K));
  }
  DeviceBuffer x = ctx.make_buffer(size_t(K) * 2);
  DeviceBuffer w = ctx.make_buffer(size_t(N) * row_words * 4);
  DeviceBuffer o = ctx.make_buffer(size_t(N) * 4);
  fill_random_f16(x, K, rng, ctx);
  {
    std::vector<uint32_t> gpu;
    auto* p = static_cast<uint32_t*>(w.mapped);
    for (uint32_t n = 0; n < N; n++) {
      q4k_row_to_gpu(packed[n], gpu, K);
      std::memcpy(p + size_t(n) * row_words, gpu.data(), row_words * 4);
    }
    ctx.flush(w);
  }
  std::vector<uint32_t> words = spec_words({N, K});
  auto pipe = ctx.make_pipeline(
      "gemv_q4k.spv", 3, words.data(), words.size() * 4, {&x, &w, &o});
  ctx.dispatch_and_wait(pipe, (N + 1) / 2, 1, 1);
  ctx.wait_idle();
  ctx.invalidate(o);

  double worst = 0;
  std::vector<double> expected(N);
  std::vector<float> wq;
  std::vector<float> x_values = read_f16_values(x, K);
  auto* op = static_cast<float*>(o.mapped);
  for (uint32_t n = 0; n < N; n++) {
    ref::q4k_dequant_row(packed[n], wq, K);
    double acc = 0;
    for (uint32_t k = 0; k < K; k++) {
      acc += double(x_values[k]) * double(wq[k]);
    }
    expected[n] = acc;
    double denom = std::max(1.0, std::fabs(acc));
    worst = std::max(worst, std::fabs(double(op[n]) - acc) / denom);
  }
  SelfCheck sc{"gemv_q4k_K" + std::to_string(K), worst, tol, worst <= tol};
  checks.push_back(sc);
  if (!sc.pass) {
    for (uint32_t n = 0; n < N; n++) {
      std::cerr << "q4k[" << n << "]: got=" << op[n] << " want=" << expected[n]
                << "\n";
    }
    die("self-check FAILED for gemv_q4k: max rel err " + std::to_string(worst) +
            " > tolerance " + std::to_string(tol) +
            ". The Q4_K layout disagrees with the pinned ggml layout.",
        1);
  }
  ctx.destroy_pipeline(pipe);
  ctx.free_buffer(x);
  ctx.free_buffer(w);
  ctx.free_buffer(o);
}

void validate_sdpa_prefill(
    Context& ctx,
    bool fp16_math,
    uint32_t M,
    uint32_t KV,
    uint32_t HD,
    uint32_t H,
    uint32_t HKV,
    double tol,
    std::vector<SelfCheck>& checks) {
  Rng rng;
  size_t qs = size_t(M) * H * HD;
  size_t kvs = size_t(KV) * HKV * HD;
  size_t input_bytes = fp16_math ? 2 : 4;
  DeviceBuffer q = ctx.make_buffer(qs * input_bytes);
  DeviceBuffer k = ctx.make_buffer(kvs * input_bytes);
  DeviceBuffer v = ctx.make_buffer(kvs * input_bytes);
  DeviceBuffer o = ctx.make_buffer(qs * 4);
  if (fp16_math) {
    fill_random_f16(q, qs, rng, ctx);
    fill_random_f16(k, kvs, rng, ctx);
    fill_random_f16(v, kvs, rng, ctx);
  } else {
    fill_random(q, qs, rng, ctx);
    fill_random(k, kvs, rng, ctx);
    fill_random(v, kvs, rng, ctx);
  }
  float scale = 1.0f / std::sqrt(float(HD));
  uint32_t scale_bits;
  std::memcpy(&scale_bits, &scale, 4);
  std::vector<uint32_t> words = spec_words({M, KV, HD, H, HKV, scale_bits});
  auto p = ctx.make_pipeline(
      fp16_math ? "sdpa_prefill_f16.spv" : "sdpa_prefill_f32.spv",
      4,
      words.data(),
      words.size() * 4,
      {&q, &k, &v, &o});
  ctx.dispatch_and_wait(p, H, (M + 7) / 8, 1);
  ctx.wait_idle();
  ctx.invalidate(o);
  std::vector<float> q_values = fp16_math
      ? read_f16_values(q, qs)
      : std::vector<float>(
            static_cast<float*>(q.mapped), static_cast<float*>(q.mapped) + qs);
  std::vector<float> k_values = fp16_math
      ? read_f16_values(k, kvs)
      : std::vector<float>(
            static_cast<float*>(k.mapped), static_cast<float*>(k.mapped) + kvs);
  std::vector<float> v_values = fp16_math
      ? read_f16_values(v, kvs)
      : std::vector<float>(
            static_cast<float*>(v.mapped), static_cast<float*>(v.mapped) + kvs);
  std::vector<float> ref;
  ref::sdpa(q_values, k_values, v_values, ref, M, KV, HD, H, HKV, scale);
  double err = max_rel_err(
      std::vector<float>(
          static_cast<float*>(o.mapped), static_cast<float*>(o.mapped) + qs),
      ref);
  std::string label =
      std::string("sdpa_prefill_") + (fp16_math ? "f16" : "f32");
  SelfCheck sc{label, err, tol, err <= tol};
  checks.push_back(sc);
  if (!sc.pass) {
    die("self-check FAILED for " + label + ": max rel err " +
            std::to_string(err) + " > tolerance " + std::to_string(tol),
        1);
  }
  ctx.destroy_pipeline(p);
  ctx.free_buffer(q);
  ctx.free_buffer(k);
  ctx.free_buffer(v);
  ctx.free_buffer(o);
}

// Split-k geometry for the decode SDPA, following the pinned ggml host
// policy (ggml-vulkan.cpp ggml_vk_flash_attn): split_k = 2 * cores /
// workgroups, where drivers without a core count -- honeykrisp on the M1 --
// use the 16-placeholder; split_kv then rounds up to a multiple of the
// Bc=32 block and split_k recomputes as ceil(KV / split_kv).
void sdpa_decode_split_geom(
    uint32_t KV,
    uint32_t HKV,
    uint32_t& splits,
    uint32_t& split_kv) {
  const uint32_t Bc = 32;
  uint32_t split_k = 16u * 2u / (1u * HKV);
  split_kv = KV;
  if (split_k > 1) {
    split_kv = std::max(1u, KV / split_k);
    split_kv = (split_kv + Bc - 1) & ~(Bc - 1);
    split_k = (KV + split_kv - 1) / split_kv;
  }
  splits = split_k;
}

// Decode split/merge scratch holds O/L/M as float16_t (the merge kernel
// widens to fp32 for its accumulation and writes the fp32 output), so the
// pinned [split][head][HD] + 2*[split][head] layout costs 2 bytes per
// element.
size_t sdpa_decode_scratch_bytes(uint32_t splits, uint32_t h, uint32_t hd) {
  return size_t(splits) * h * (2u + hd) * 2u;
}

void validate_sdpa_decode(
    Context& ctx,
    uint32_t KV,
    uint32_t HD,
    uint32_t H,
    uint32_t HKV,
    double tol,
    std::vector<SelfCheck>& checks) {
  Rng rng;
  size_t qs = size_t(H) * HD;
  size_t kvs = size_t(KV) * HKV * HD;
  // Q arrives as F32; the split shader stages it into shared f16 (the
  // pinned flash_attn.comp Q contract).
  DeviceBuffer q = ctx.make_buffer(qs * 4);
  DeviceBuffer k = ctx.make_buffer(kvs * 2);
  DeviceBuffer v = ctx.make_buffer(kvs * 2);
  DeviceBuffer o = ctx.make_buffer(qs * 4);
  fill_random(q, qs, rng, ctx);
  fill_random_f16(k, kvs, rng, ctx);
  fill_random_f16(v, kvs, rng, ctx);
  float scale = 1.0f / std::sqrt(float(HD));
  uint32_t scale_bits;
  std::memcpy(&scale_bits, &scale, 4);
  uint32_t splits = 0;
  uint32_t split_kv = 0;
  sdpa_decode_split_geom(KV, HKV, splits, split_kv);
  std::vector<uint32_t> words =
      spec_words({KV, HD, H, HKV, scale_bits, splits, split_kv});
  DeviceBuffer scratch =
      ctx.make_buffer(sdpa_decode_scratch_bytes(splits, H, HD));
  auto ps = ctx.make_pipeline(
      "sdpa_decode_split.spv",
      4,
      words.data(),
      words.size() * 4,
      {&q, &k, &v, &scratch});
  auto pm = ctx.make_pipeline(
      "sdpa_decode_merge.spv",
      2,
      words.data(),
      words.size() * 4,
      {&scratch, &o});
  ctx.dispatch2_and_wait(ps, splits, HKV, 1, pm, H, (HD + 31) / 32, 1);
  ctx.wait_idle();
  ctx.invalidate(o);
  std::vector<float> q_values(
      static_cast<const float*>(q.mapped),
      static_cast<const float*>(q.mapped) + qs);
  std::vector<float> k_values = read_f16_values(k, kvs);
  std::vector<float> v_values = read_f16_values(v, kvs);
  std::vector<float> ref;
  // Decode attends the whole KV cache: the reference runs non-causal.
  ref::sdpa(
      q_values,
      k_values,
      v_values,
      ref,
      1,
      KV,
      HD,
      H,
      HKV,
      scale,
      /*causal=*/false);
  double err = max_rel_err(
      std::vector<float>(
          static_cast<float*>(o.mapped), static_cast<float*>(o.mapped) + qs),
      ref);
  SelfCheck sc{"sdpa_decode", err, tol, err <= tol};
  checks.push_back(sc);
  if (!sc.pass) {
    die("self-check FAILED for sdpa_decode: max rel err " +
            std::to_string(err) + " > tolerance " + std::to_string(tol),
        1);
  }

  // Controls: the algorithm simulation must reproduce the reference with
  // the fixed reduction, and must reject the historical broken gather
  // (spart[4*j+c]) so a shader regression cannot slip through as noise.
  std::vector<float> sim;
  ref::sdpa_decode_simulate(
      q_values, k_values, v_values, sim, KV, HD, H, HKV, scale, false);
  double sim_err = max_rel_err(sim, ref);
  SelfCheck pos{"sdpa_decode_sim_fixed", sim_err, 2e-4, sim_err <= 2e-4};
  checks.push_back(pos);
  if (!pos.pass) {
    die("self-check FAILED for sdpa_decode_sim_fixed: the reduction "
        "simulation disagrees with the reference (rel err " +
            std::to_string(sim_err) + ")",
        1);
  }
  ref::sdpa_decode_simulate(
      q_values, k_values, v_values, sim, KV, HD, H, HKV, scale, true);
  double legacy_err = max_rel_err(sim, ref);
  SelfCheck neg{
      "sdpa_decode_sim_legacy_negative_control",
      legacy_err,
      tol,
      legacy_err > tol};
  checks.push_back(neg);
  if (!neg.pass) {
    die("self-check FAILED for sdpa_decode_sim_legacy_negative_control: "
        "the broken legacy reduction was expected to mismatch the "
        "reference but matched (rel err " +
            std::to_string(legacy_err) + "); the negative control is vacuous",
        1);
  }
  ctx.destroy_pipeline(ps);
  ctx.destroy_pipeline(pm);
  ctx.free_buffer(scratch);
  ctx.free_buffer(q);
  ctx.free_buffer(k);
  ctx.free_buffer(v);
  ctx.free_buffer(o);
}

// Times one benchmark row end to end.
void run_row(Context& ctx, Row& row, uint32_t warmup, uint32_t reps) {
  Rng rng;
  if (row.kind == "gemm") {
    size_t input_bytes = row.math == "fp16" ? 2 : 4;
    DeviceBuffer a = ctx.make_buffer(size_t(row.M) * row.K * input_bytes);
    DeviceBuffer w = ctx.make_buffer(size_t(row.N) * row.K * input_bytes);
    DeviceBuffer c = ctx.make_buffer(size_t(row.M) * row.N * 4);
    if (row.math == "fp16") {
      fill_random_f16(a, size_t(row.M) * row.K, rng, ctx);
      fill_random_f16(w, size_t(row.N) * row.K, rng, ctx);
    } else {
      fill_random(a, size_t(row.M) * row.K, rng, ctx);
      fill_random(w, size_t(row.N) * row.K, rng, ctx);
    }
    std::vector<uint32_t> words = spec_words({row.M, row.N, row.K});
    std::string spv = row.math == "fp16" ? "gemm_f16.spv" : "gemm_f32.spv";
    auto p =
        ctx.make_pipeline(spv, 3, words.data(), words.size() * 4, {&a, &w, &c});
    row.stats =
        time_row(ctx, p, (row.N + 63) / 64, (row.M + 63) / 64, 1, warmup, reps);
    ctx.destroy_pipeline(p);
    ctx.free_buffer(a);
    ctx.free_buffer(w);
    ctx.free_buffer(c);
  } else if (row.kind == "gemv") {
    uint32_t row_words = row.quant == "q4_k"
        ? (row.K / ref::kQkK) * kQ4kGpuWordsPerBlock
        : row.K;
    size_t x_bytes = row.quant == "q4_k" ? 2 : 4;
    DeviceBuffer x = ctx.make_buffer(size_t(row.K) * x_bytes);
    DeviceBuffer w = ctx.make_buffer(size_t(row.N) * size_t(row_words) * 4);
    DeviceBuffer o = ctx.make_buffer(size_t(row.N) * 4);
    if (row.quant == "q4_k") {
      fill_random_f16(x, row.K, rng, ctx);
    } else {
      fill_random(x, row.K, rng, ctx);
    }
    if (row.quant == "q4_k") {
      std::vector<uint32_t> gpu;
      auto* wp = static_cast<uint32_t*>(w.mapped);
      for (uint32_t n = 0; n < row.N; n++) {
        auto packed = ref::q4k_make_row(rng.mt, row.K);
        q4k_row_to_gpu(packed, gpu, row.K);
        std::memcpy(wp + size_t(n) * row_words, gpu.data(), row_words * 4);
      }
      ctx.flush(w);
    } else {
      fill_random(w, size_t(row.N) * row.K, rng, ctx);
    }
    std::vector<uint32_t> words = spec_words({row.N, row.K});
    std::string spv = row.quant == "q4_k" ? "gemv_q4k.spv" : "gemv_f32.spv";
    auto p =
        ctx.make_pipeline(spv, 3, words.data(), words.size() * 4, {&x, &w, &o});
    uint32_t xgroups = row.quant == "q4_k" ? (row.N + 1) / 2 : row.N;
    row.stats = time_row(ctx, p, xgroups, 1, 1, warmup, reps);
    ctx.destroy_pipeline(p);
    ctx.free_buffer(x);
    ctx.free_buffer(w);
    ctx.free_buffer(o);
  } else if (row.kind == "sdpa_prefill") {
    size_t qs = size_t(row.M) * row.sdpa_h * row.sdpa_hd;
    size_t kvs = size_t(row.sdpa_kv) * row.sdpa_hkv * row.sdpa_hd;
    size_t input_bytes = row.math == "fp16" ? 2 : 4;
    DeviceBuffer q = ctx.make_buffer(qs * input_bytes);
    DeviceBuffer k = ctx.make_buffer(kvs * input_bytes);
    DeviceBuffer v = ctx.make_buffer(kvs * input_bytes);
    DeviceBuffer o = ctx.make_buffer(qs * 4);
    if (row.math == "fp16") {
      fill_random_f16(q, qs, rng, ctx);
      fill_random_f16(k, kvs, rng, ctx);
      fill_random_f16(v, kvs, rng, ctx);
    } else {
      fill_random(q, qs, rng, ctx);
      fill_random(k, kvs, rng, ctx);
      fill_random(v, kvs, rng, ctx);
    }
    float scale = 1.0f / std::sqrt(float(row.sdpa_hd));
    uint32_t scale_bits;
    std::memcpy(&scale_bits, &scale, 4);
    std::vector<uint32_t> words = spec_words(
        {row.M,
         row.sdpa_kv,
         row.sdpa_hd,
         row.sdpa_h,
         row.sdpa_hkv,
         scale_bits});
    std::string spv =
        row.math == "fp16" ? "sdpa_prefill_f16.spv" : "sdpa_prefill_f32.spv";
    auto p = ctx.make_pipeline(
        spv, 4, words.data(), words.size() * 4, {&q, &k, &v, &o});
    row.stats = time_row(ctx, p, row.sdpa_h, (row.M + 7) / 8, 1, warmup, reps);
    ctx.destroy_pipeline(p);
    ctx.free_buffer(q);
    ctx.free_buffer(k);
    ctx.free_buffer(v);
    ctx.free_buffer(o);
  } else { // sdpa_decode
    size_t qs = size_t(row.sdpa_h) * row.sdpa_hd;
    size_t kvs = size_t(row.sdpa_kv) * row.sdpa_hkv * row.sdpa_hd;
    DeviceBuffer q = ctx.make_buffer(qs * 4);
    DeviceBuffer k = ctx.make_buffer(kvs * 2);
    DeviceBuffer v = ctx.make_buffer(kvs * 2);
    DeviceBuffer o = ctx.make_buffer(qs * 4);
    fill_random(q, qs, rng, ctx);
    fill_random_f16(k, kvs, rng, ctx);
    fill_random_f16(v, kvs, rng, ctx);
    float scale = 1.0f / std::sqrt(float(row.sdpa_hd));
    uint32_t scale_bits;
    std::memcpy(&scale_bits, &scale, 4);
    uint32_t splits = 0;
    uint32_t split_kv = 0;
    sdpa_decode_split_geom(row.sdpa_kv, row.sdpa_hkv, splits, split_kv);
    std::vector<uint32_t> words = spec_words(
        {row.sdpa_kv,
         row.sdpa_hd,
         row.sdpa_h,
         row.sdpa_hkv,
         scale_bits,
         splits,
         split_kv});
    DeviceBuffer scratch = ctx.make_buffer(
        sdpa_decode_scratch_bytes(splits, row.sdpa_h, row.sdpa_hd));
    auto ps = ctx.make_pipeline(
        "sdpa_decode_split.spv",
        4,
        words.data(),
        words.size() * 4,
        {&q, &k, &v, &scratch});
    auto pm = ctx.make_pipeline(
        "sdpa_decode_merge.spv",
        2,
        words.data(),
        words.size() * 4,
        {&scratch, &o});
    row.stats = time_row2(
        ctx,
        ps,
        splits,
        row.sdpa_hkv,
        1,
        pm,
        row.sdpa_h,
        (row.sdpa_hd + 31) / 32,
        1,
        warmup,
        reps);
    ctx.destroy_pipeline(ps);
    ctx.destroy_pipeline(pm);
    ctx.free_buffer(scratch);
    ctx.free_buffer(q);
    ctx.free_buffer(k);
    ctx.free_buffer(v);
    ctx.free_buffer(o);
  }

  if (row.kind == "gemm" || row.kind == "gemv") {
    row.flops = 2.0 * double(row.M) * double(row.N) * double(row.K);
    row.weight_bytes = row.quant == "q4_k"
        ? double(row.N) * double(row.K) * 160.0 / 256.0
        : double(row.N) * double(row.K) * (row.math == "fp16" ? 2.0 : 4.0);
  } else if (row.kind == "sdpa_prefill") {
    row.flops = 2.0 * 2.0 * double(row.sdpa_h) * double(row.sdpa_hd) *
        (double(row.M) * (double(row.M) + 1.0) / 2.0);
    row.weight_bytes = 2.0 * double(row.sdpa_kv) * double(row.sdpa_hkv) *
        double(row.sdpa_hd) * (row.math == "fp16" ? 2.0 : 4.0);
  } else {
    row.flops = 2.0 * 2.0 * double(row.sdpa_h) * double(row.sdpa_hd) *
        double(row.sdpa_kv);
    row.weight_bytes = 2.0 * double(row.sdpa_kv) * double(row.sdpa_hkv) *
        double(row.sdpa_hd) * 2.0;
  }
  double seconds = row.stats.median_us / 1000.0 / 1000.0;
  row.metric = row.flops / seconds / 1e9;
}

// --- Optional attention stage profile (--profile-stages)
// ------------------------

struct AttentionStageProfile {
  std::string prefill_row;
  std::string decode_row;
  double timestamp_period_ns = 0;
  uint32_t timestamp_valid_bits = 0;
  Stats prefill_dispatch; // one sdpa_prefill dispatch, GPU-side
  Stats decode_split; // decode split dispatch, GPU-side
  Stats decode_merge; // decode merge dispatch (incl. launch gap), GPU-side
};

// The largest of the three stage medians, with its median written through
// median_us when non-null. Shared by the stderr summary and the receipt.
std::string dominant_attention_stage(
    const AttentionStageProfile& p,
    double* median_us) {
  double dm = p.prefill_dispatch.median_us;
  std::string name = p.prefill_row + " dispatch";
  if (p.decode_split.median_us > dm) {
    dm = p.decode_split.median_us;
    name = p.decode_row + " split";
  }
  if (p.decode_merge.median_us > dm) {
    dm = p.decode_merge.median_us;
    name = p.decode_row + " merge";
  }
  if (median_us) {
    *median_us = dm;
  }
  return name;
}

// Profiles the two attention gate rows with GPU timestamps: one pass reports
// the prefill dispatch interval and the decode split/merge intervals for the
// exact shapes the timed rows use (same fills, spec constants, grids, and
// pipelines). Runs only after all timed rounds, so the gate medians are
// untouched. Fails clearly when the device cannot timestamp or a query
// result is implausible.
AttentionStageProfile profile_attention_stages(
    Context& ctx,
    const std::vector<Row>& rows,
    uint32_t warmup,
    uint32_t reps) {
  if (!ctx.gpu_timestamps_supported()) {
    die("--profile-stages requires GPU timestamp queries; this queue family "
        "reports timestampValidBits=" +
            std::to_string(ctx.timestamp_valid_bits()) +
            " and timestampPeriod=" +
            std::to_string(ctx.timestamp_period_ns()) +
            " ns. The driver cannot produce the stage breakdown.",
        1);
  }
  const Row* prefill = nullptr;
  const Row* decode = nullptr;
  for (const auto& r : rows) {
    if (r.kind == "sdpa_prefill")
      prefill = &r;
    if (r.kind == "sdpa_decode")
      decode = &r;
  }
  if (!prefill || !decode) {
    die("--profile-stages could not find the sdpa_prefill and sdpa_decode "
        "rows that define the balanced attention workload.",
        2);
  }

  AttentionStageProfile profile;
  profile.prefill_row = prefill->name;
  profile.decode_row = decode->name;
  profile.timestamp_period_ns = ctx.timestamp_period_ns();
  profile.timestamp_valid_bits = ctx.timestamp_valid_bits();
  const uint64_t tick_mask = ctx.timestamp_valid_bits() >= 64
      ? ~0ull
      : ((1ull << ctx.timestamp_valid_bits()) - 1ull);
  auto stage_us = [&](uint64_t begin, uint64_t end) -> double {
    uint64_t delta = (end - begin) & tick_mask;
    double ns = double(delta) * double(ctx.timestamp_period_ns());
    if (ns <= 0.0 || !std::isfinite(ns) || ns > 1e9) {
      die("attention stage profile got an implausible timestamp interval (" +
              std::to_string(delta) + " ticks, " + std::to_string(ns) +
              " ns); the GPU timestamp results are not trustworthy on this "
              "driver",
          1);
    }
    return ns / 1000.0;
  };

  VkQueryPool pool = ctx.make_query_pool(3);
  Rng rng;

  // Prefill: timestamps around the single SDPA dispatch.
  {
    size_t qs = size_t(prefill->M) * prefill->sdpa_h * prefill->sdpa_hd;
    size_t kvs =
        size_t(prefill->sdpa_kv) * prefill->sdpa_hkv * prefill->sdpa_hd;
    size_t input_bytes = prefill->math == "fp16" ? 2 : 4;
    DeviceBuffer q = ctx.make_buffer(qs * input_bytes);
    DeviceBuffer k = ctx.make_buffer(kvs * input_bytes);
    DeviceBuffer v = ctx.make_buffer(kvs * input_bytes);
    DeviceBuffer o = ctx.make_buffer(qs * 4);
    if (prefill->math == "fp16") {
      fill_random_f16(q, qs, rng, ctx);
      fill_random_f16(k, kvs, rng, ctx);
      fill_random_f16(v, kvs, rng, ctx);
    } else {
      fill_random(q, qs, rng, ctx);
      fill_random(k, kvs, rng, ctx);
      fill_random(v, kvs, rng, ctx);
    }
    float scale = 1.0f / std::sqrt(float(prefill->sdpa_hd));
    uint32_t scale_bits;
    std::memcpy(&scale_bits, &scale, 4);
    std::vector<uint32_t> words = spec_words(
        {prefill->M,
         prefill->sdpa_kv,
         prefill->sdpa_hd,
         prefill->sdpa_h,
         prefill->sdpa_hkv,
         scale_bits});
    std::string spv = prefill->math == "fp16" ? "sdpa_prefill_f16.spv"
                                              : "sdpa_prefill_f32.spv";
    auto p = ctx.make_pipeline(
        spv, 4, words.data(), words.size() * 4, {&q, &k, &v, &o});
    uint32_t gx = prefill->sdpa_h;
    uint32_t gy = (prefill->M + 7) / 8;
    for (uint32_t i = 0; i < warmup; i++) {
      ctx.dispatch_and_wait(p, gx, gy, 1);
    }
    std::vector<double> us;
    us.reserve(reps);
    for (uint32_t i = 0; i < reps; i++) {
      ctx.dispatch_timestamped_and_wait(p, gx, gy, 1, pool, 0);
      uint64_t ticks[2];
      ctx.get_timestamp_queries(pool, 0, 2, ticks);
      us.push_back(stage_us(ticks[0], ticks[1]));
    }
    profile.prefill_dispatch = summarize_samples(std::move(us));
    ctx.destroy_pipeline(p);
    ctx.free_buffer(q);
    ctx.free_buffer(k);
    ctx.free_buffer(v);
    ctx.free_buffer(o);
  }

  // Decode: timestamps around split and merge in the same single submission
  // the timed row uses.
  {
    size_t qs = size_t(decode->sdpa_h) * decode->sdpa_hd;
    size_t kvs = size_t(decode->sdpa_kv) * decode->sdpa_hkv * decode->sdpa_hd;
    DeviceBuffer q = ctx.make_buffer(qs * 4);
    DeviceBuffer k = ctx.make_buffer(kvs * 2);
    DeviceBuffer v = ctx.make_buffer(kvs * 2);
    DeviceBuffer o = ctx.make_buffer(qs * 4);
    fill_random(q, qs, rng, ctx);
    fill_random_f16(k, kvs, rng, ctx);
    fill_random_f16(v, kvs, rng, ctx);
    float scale = 1.0f / std::sqrt(float(decode->sdpa_hd));
    uint32_t scale_bits;
    std::memcpy(&scale_bits, &scale, 4);
    uint32_t splits = 0;
    uint32_t split_kv = 0;
    sdpa_decode_split_geom(decode->sdpa_kv, decode->sdpa_hkv, splits, split_kv);
    std::vector<uint32_t> words = spec_words(
        {decode->sdpa_kv,
         decode->sdpa_hd,
         decode->sdpa_h,
         decode->sdpa_hkv,
         scale_bits,
         splits,
         split_kv});
    DeviceBuffer scratch = ctx.make_buffer(
        sdpa_decode_scratch_bytes(splits, decode->sdpa_h, decode->sdpa_hd));
    auto ps = ctx.make_pipeline(
        "sdpa_decode_split.spv",
        4,
        words.data(),
        words.size() * 4,
        {&q, &k, &v, &scratch});
    auto pm = ctx.make_pipeline(
        "sdpa_decode_merge.spv",
        2,
        words.data(),
        words.size() * 4,
        {&scratch, &o});
    uint32_t mgx2 = decode->sdpa_h;
    uint32_t mgy2 = (decode->sdpa_hd + 31) / 32;
    for (uint32_t i = 0; i < warmup; i++) {
      ctx.dispatch2_and_wait(
          ps, splits, decode->sdpa_hkv, 1, pm, mgx2, mgy2, 1);
    }
    std::vector<double> split_us;
    std::vector<double> merge_us;
    split_us.reserve(reps);
    merge_us.reserve(reps);
    for (uint32_t i = 0; i < reps; i++) {
      ctx.dispatch2_timestamped_and_wait(
          ps, splits, decode->sdpa_hkv, 1, pm, mgx2, mgy2, 1, pool, 0);
      uint64_t ticks[3];
      ctx.get_timestamp_queries(pool, 0, 3, ticks);
      split_us.push_back(stage_us(ticks[0], ticks[1]));
      merge_us.push_back(stage_us(ticks[1], ticks[2]));
    }
    profile.decode_split = summarize_samples(std::move(split_us));
    profile.decode_merge = summarize_samples(std::move(merge_us));
    ctx.destroy_pipeline(ps);
    ctx.destroy_pipeline(pm);
    ctx.free_buffer(scratch);
    ctx.free_buffer(q);
    ctx.free_buffer(k);
    ctx.free_buffer(v);
    ctx.free_buffer(o);
  }

  ctx.destroy_query_pool(pool);
  ctx.wait_idle();
  return profile;
}

Value stage_profile_json(
    const AttentionStageProfile& p,
    uint32_t warmup,
    uint32_t reps) {
  auto stage =
      [](const std::string& row_name, const char* stage_name, const Stats& s) {
        Object so;
        so["row"] = Value(row_name);
        so["stage"] = Value(std::string(stage_name));
        Object st;
        st["median_us"] = Value(s.median_us);
        st["min_us"] = Value(s.min_us);
        st["max_us"] = Value(s.max_us);
        st["p95_us"] = Value(s.p95_us);
        st["mean_us"] = Value(s.mean_us);
        so["timing_us"] = Value(std::move(st));
        return Value(std::move(so));
      };
  Object o;
  o["method"] = Value(
      "VK_QUERY_TYPE_TIMESTAMP written around the same dispatches the timed "
      "rows run (TOP_OF_PIPE before, BOTTOM_OF_PIPE after), one submission "
      "per repetition. Intervals are GPU-side and exclude the submission and "
      "fence overhead that host wall-clock row medians include; the decode "
      "merge interval includes the launch gap behind the split->merge "
      "barrier.");
  o["timestamp_period_ns"] = Value(double(p.timestamp_period_ns));
  o["timestamp_valid_bits"] = Value(double(p.timestamp_valid_bits));
  o["warmup"] = Value(double(warmup));
  o["repetitions"] = Value(double(reps));
  omarchy_spike::json::Array stages;
  stages.push_back(stage(p.prefill_row, "dispatch", p.prefill_dispatch));
  stages.push_back(stage(p.decode_row, "split", p.decode_split));
  stages.push_back(stage(p.decode_row, "merge", p.decode_merge));
  o["stages"] = Value(std::move(stages));
  double dom_us = 0;
  o["dominant_stage"] = Value(dominant_attention_stage(p, &dom_us));
  o["dominant_stage_median_us"] = Value(dom_us);
  return Value(std::move(o));
}

} // namespace omarchy_spike

int main(int argc, char** argv) {
  using namespace omarchy_spike;
  Options opt = parse_args(argc, argv);

  // 1. Comparators first: the spike refuses to run without real inputs.
  //    llama_cpp_reference.json carries the pin and model-level context;
  //    ggml_op_comparator.json carries the matched per-op kernel timings
  //    that the 80 percent gates are computed from.
  Comparator cmp;
  Value gcmp;
  try {
    cmp = load_comparator(opt.comparator_path);
  } catch (const std::exception& e) {
    die("comparator field error in " + opt.comparator_path + ": " + e.what() +
            " (every required field must be present with a real measured value)",
        2);
  }
  try {
    gcmp = load_ggml_comparator(opt.ggml_comparator_path);
  } catch (const std::exception& e) {
    die("ggml comparator field error in " + opt.ggml_comparator_path + ": " +
            e.what(),
        2);
  }
  if (!gcmp.has("position_balance_complete") ||
      gcmp.at("position_balance_complete").kind() != Value::Kind::Bool ||
      !gcmp.at("position_balance_complete").boolean()) {
    die("ggml comparator is not position-balanced; run the probe with one "
        "round per comparator case",
        2);
  }
  {
    std::string ggml_commit =
        gcmp.has("pin_commit") && gcmp.at("pin_commit").is_string()
        ? gcmp.at("pin_commit").string()
        : "";
    if (!ggml_commit.empty() && ggml_commit != cmp.commit) {
      die("ggml comparator was measured on llama.cpp commit " + ggml_commit +
              " but the reference pin is " + cmp.commit +
              ". Re-run the probe on the pinned checkout.",
          2);
    }
  }

  // 2. Dims: pinned defaults, optionally overridden, then matched against
  // the comparator.
  Dims dims;
  if (!opt.dims_json_path.empty()) {
    std::string text = read_file(opt.dims_json_path);
    if (text.empty()) {
      die("--dims-json file not found: " + opt.dims_json_path, 2);
    }
    Value dv;
    try {
      dv = omarchy_spike::json::parse(text);
    } catch (const std::exception& e) {
      die("--dims-json file is not valid JSON: " + std::string(e.what()), 2);
    }
    auto num = [&](const char* key) -> const Value {
      return dv.has(key) ? dv.at(key) : Value();
    };
    auto set = [&](const char* key, uint32_t& field) {
      const Value v = num(key);
      if (v.is_number()) {
        if (v.number() < 1 || v.number() != std::floor(v.number())) {
          die(std::string("dims field '") + key +
                  "' must be a positive integer",
              2);
        }
        field = uint32_t(v.number());
      }
    };
    set("hidden_size", dims.hidden);
    set("intermediate_size", dims.intermediate);
    set("vocab_size", dims.vocab);
    set("num_hidden_layers", dims.layers);
    set("num_attention_layers", dims.attn_layers);
    set("num_attention_heads", dims.heads);
    set("num_key_value_heads", dims.kv_heads);
    set("head_dim", dims.head_dim);
    if (dims.head_dim != 256) {
      die("the attention gate requires head_dim 256 (spike kernels); got " +
              std::to_string(dims.head_dim) +
              ". The spike cannot gate attention at this head_dim; fix "
              "model_dims.",
          2);
    }
  }
  if (cmp.raw.has("model_dims")) {
    try {
      require_dims_match(cmp.raw.at("model_dims"), dims);
    } catch (const std::exception& e) {
      die("comparator model_dims error: " + std::string(e.what()), 2);
    }
  }

  // 3. Thermal baseline before any work.
  if (opt.idle_wait_s > 0) {
    std::cerr << "idling " << opt.idle_wait_s << "s before the run...\n";
    std::this_thread::sleep_for(std::chrono::seconds(opt.idle_wait_s));
  }
  auto thermal_before = read_thermal_zones();
  auto sensors_before = read_sensors_output();
  if (thermal_before.empty() && sensors_before.empty()) {
    die("[omarchy] no temperature source: /sys/class/thermal is empty and "
        "/usr/bin/sensors is missing. Install lm_sensors so the thermal "
        "receipt is real.",
        1);
  }

  // 4. Device.
  Context ctx;
  ctx.set_spirv_dir(opt.spirv_dir);
  ctx.set_device_index(opt.device_index);
  ctx.init();
  if (!ctx.fp16_math()) {
    die("the matched ggml comparator requires shaderFloat16 and "
        "storageBuffer16BitAccess; this device cannot run the U2 gate",
        1);
  }
  if (ctx.support().non_apple_dev) {
    std::cerr << "WARNING: development device " << ctx.props().deviceName
              << " (non-Honeykrisp). Receipts from this run are not Omarchy "
                 "hardware receipts.\n";
  }

  // 5. Self-checks: every kernel against its CPU reference before timing.
  std::vector<SelfCheck> checks;
  double gemm_tol = ctx.fp16_math() ? 3e-2 : 1e-4;
  validate_gemm(
      ctx,
      ctx.fp16_math() ? "gemm_f16.spv" : "gemm_f32.spv",
      ctx.fp16_math() ? "gemm_f16" : "gemm_f32",
      ctx.fp16_math(),
      20,
      19,
      48,
      gemm_tol,
      checks);
  validate_gemm(
      ctx,
      ctx.fp16_math() ? "gemm_f16.spv" : "gemm_f32.spv",
      ctx.fp16_math() ? "gemm_f16_K2048" : "gemm_f32_K2048",
      ctx.fp16_math(),
      8,
      64,
      2048,
      ctx.fp16_math() ? 3e-3 : 1e-4,
      checks);
  validate_gemv_f32(ctx, 8, 512, 1e-4, checks);
  validate_gemv_q4k(ctx, 8, 512, 2e-3, checks);
  validate_gemv_q4k(ctx, 8, 6144, 6e-3, checks);
  validate_sdpa_prefill(
      ctx,
      ctx.fp16_math(),
      16,
      24,
      256,
      2,
      1,
      ctx.fp16_math() ? 6e-2 : 1e-4,
      checks);
  validate_sdpa_decode(ctx, 70, 256, 2, 1, 3e-2, checks);
  validate_sdpa_decode(ctx, 512, 256, 16, 2, 3e-2, checks);
  std::cerr << "all self-checks passed\n";

  // 6. Benchmark rows (matched to the Qwen3.8-2B full-attention layers).
  const uint32_t M_PREFILL = 512;
  const uint32_t KV_LEN = 512;
  std::vector<Row> rows;
  auto add_gemm = [&](const char* name, uint32_t k, uint32_t n) {
    Row r;
    r.name = name;
    r.section = "prefill";
    r.kind = "gemm";
    r.gate = true;
    r.math = ctx.fp16_math() ? "fp16" : "fp32";
    r.quant = "none";
    r.M = M_PREFILL;
    r.K = k;
    r.N = n;
    rows.push_back(r);
  };
  add_gemm("prefill_gemm_qkv", dims.hidden, dims.qkv_out());
  add_gemm("prefill_gemm_o", dims.q_out(), dims.hidden);
  add_gemm("prefill_gemm_mlp_up", dims.hidden, dims.mlp_up_out());
  add_gemm("prefill_gemm_mlp_down", dims.intermediate, dims.hidden);

  auto add_gemv_q4k = [&](const char* name, uint32_t k, uint32_t n) {
    Row r;
    r.name = name;
    r.section = "decode";
    r.kind = "gemv";
    r.gate = true;
    r.math = "fp32";
    r.quant = "q4_k";
    r.M = 1;
    r.K = k;
    r.N = n;
    rows.push_back(r);
  };
  add_gemv_q4k("decode_gemv_q4k_qkv", dims.hidden, dims.qkv_out());
  add_gemv_q4k("decode_gemv_q4k_o", dims.q_out(), dims.hidden);
  add_gemv_q4k("decode_gemv_q4k_mlp_up", dims.hidden, dims.mlp_up_out());
  add_gemv_q4k("decode_gemv_q4k_mlp_down", dims.intermediate, dims.hidden);

  {
    Row r;
    r.name = "decode_gemv_f32_qkv";
    r.section = "info";
    r.kind = "gemv";
    r.gate = false;
    r.math = "fp32";
    r.quant = "none";
    r.M = 1;
    r.K = dims.hidden;
    r.N = dims.qkv_out();
    rows.push_back(r);
  }
  {
    Row r;
    r.name = "prefill_sdpa";
    r.section = "attention";
    r.kind = "sdpa_prefill";
    r.gate = true;
    r.math = ctx.fp16_math() ? "fp16" : "fp32";
    r.quant = "none";
    r.M = M_PREFILL;
    r.sdpa_h = dims.heads;
    r.sdpa_hkv = dims.kv_heads;
    r.sdpa_hd = dims.head_dim;
    r.sdpa_kv = KV_LEN;
    rows.push_back(r);

    Row d;
    d.name = "decode_sdpa";
    d.section = "attention";
    d.kind = "sdpa_decode";
    d.gate = true;
    d.math = "fp32";
    d.quant = "none";
    d.M = 1;
    d.sdpa_h = dims.heads;
    d.sdpa_hkv = dims.kv_heads;
    d.sdpa_hd = dims.head_dim;
    d.sdpa_kv = KV_LEN;
    rows.push_back(d);
  }

  std::map<std::string, GgmlCase> ggml_cases;
  try {
    ggml_cases = match_ggml_cases(gcmp, rows, dims);
  } catch (const std::exception& e) {
    die("ggml comparator case error: " + std::string(e.what()), 2);
  }
  std::vector<size_t> gate_row_indices;
  for (size_t i = 0; i < rows.size(); i++) {
    if (rows[i].gate) {
      gate_row_indices.push_back(i);
    }
  }
  if (opt.rounds % gate_row_indices.size() != 0) {
    die("--rounds must be a multiple of the " +
            std::to_string(gate_row_indices.size()) + " gated cases",
        2);
  }

  std::vector<std::vector<double>> pooled_samples(rows.size());
  for (uint32_t round = 0; round < opt.rounds; round++) {
    for (size_t slot = 0; slot < gate_row_indices.size(); slot++) {
      size_t row_index =
          gate_row_indices[(slot + round) % gate_row_indices.size()];
      Row& row = rows[row_index];
      std::cerr << "round " << (round + 1) << "/" << opt.rounds << ": "
                << row.name << " ...\n";
      run_row(ctx, row, opt.warmup, opt.reps);
      auto& pooled = pooled_samples[row_index];
      pooled.insert(
          pooled.end(), row.stats.samples.begin(), row.stats.samples.end());
    }
  }
  for (size_t row_index : gate_row_indices) {
    Row& row = rows[row_index];
    row.stats = summarize_samples(std::move(pooled_samples[row_index]));
    double seconds = row.stats.median_us / 1000.0 / 1000.0;
    row.metric = row.flops / seconds / 1e9;
  }
  for (Row& row : rows) {
    if (!row.gate) {
      std::cerr << "running " << row.name << " ...\n";
      run_row(ctx, row, opt.warmup, opt.reps);
    }
    auto it = ggml_cases.find(row.name);
    if (it != ggml_cases.end()) {
      row.cmp_median_us = it->second.median_us;
      row.cmp_gflops = ggml_case_gflops(it->second);
    }
  }
  ctx.wait_idle();
  auto thermal_after = read_thermal_zones();
  auto sensors_after = read_sensors_output();

  // 6b. Optional stage profile: GPU-timestamp breakdown of the attention
  //     dispatches. Runs after every timed round and the thermal capture, so
  //     the gate medians and the thermal receipt are untouched. With
  //     --profile-stages absent this block does not execute at all.
  AttentionStageProfile stage_profile;
  if (opt.profile_stages) {
    std::cerr << "profiling attention stages with GPU timestamps...\n";
    stage_profile = profile_attention_stages(ctx, rows, opt.warmup, opt.reps);
    double dom_us = 0;
    std::string dom = dominant_attention_stage(stage_profile, &dom_us);
    std::cerr << "stage profile (GPU timestamps, "
              << stage_profile.timestamp_period_ns << " ns period):\n"
              << "  " << stage_profile.prefill_row << " dispatch: median "
              << stage_profile.prefill_dispatch.median_us << " us\n"
              << "  " << stage_profile.decode_row << " split: median "
              << stage_profile.decode_split.median_us << " us\n"
              << "  " << stage_profile.decode_row << " merge: median "
              << stage_profile.decode_merge.median_us << " us\n"
              << "  dominant attention stage: " << dom << " (" << dom_us
              << " us)\n";
  }

  // 7. Matched-kernel ratios and the explicit gate. Every gate row compares
  // against the pinned ggml Vulkan timing of the same op, shape, and
  // quantization; llama-bench throughput is context only.
  double worst_prefill = 1e30;
  double worst_decode = 1e30;
  double worst_attention = 1e30;
  for (auto& r : rows) {
    if (!r.gate)
      continue;
    double ratio = r.metric / r.cmp_gflops;
    if (r.section == "prefill")
      worst_prefill = std::min(worst_prefill, ratio);
    if (r.section == "decode")
      worst_decode = std::min(worst_decode, ratio);
    if (r.section == "attention")
      worst_attention = std::min(worst_attention, ratio);
  }
  bool go = worst_prefill >= kGoNoGoThreshold &&
      worst_decode >= kGoNoGoThreshold && worst_attention >= kGoNoGoThreshold;

  // 8. Receipt.
  Object out;
  out["schema"] = Value("omarchy-spike/v1");
  out["timestamp_utc"] = Value(utc_now());
  if (!opt.source_commit.empty()) {
    Object source;
    source["commit"] = Value(opt.source_commit);
    source["dirty"] = Value(opt.source_dirty);
    out["source"] = Value(std::move(source));
  }
  out["timing_method"] = Value(
      "host wall clock around Vulkan submission through timestamp-query "
      "completion (the query is host-reset before reuse, written after the "
      "final host-read barrier, then polled with bounded user-space yields); "
      "gated cases rotate through every order position; medians pool equal "
      "warmup and repetition counts from all rounds");
  out["thermal_procedure"] = Value(
      "record /sys/class/thermal zones and verbatim sensors output before "
      "and after the run; runner supports --idle-wait S before the run; "
      "paired comparisons start within 2 C of the comparator's recorded "
      "idle baseline (thermal_delta_vs_baseline_c)");

  Object dev;
  dev["device_name"] = Value(ctx.props().deviceName);
  dev["vendor_id"] = Value(double(ctx.props().vendorID));
  dev["device_id"] = Value(double(ctx.props().deviceID));
  dev["api_version"] = Value(double(ctx.props().apiVersion));
  dev["driver"] = Value(ctx.driver_name());
  dev["driver_id"] = Value(double(ctx.support().driver_id));
  dev["non_apple_dev"] = Value(ctx.support().non_apple_dev);
  dev["fp16_math"] = Value(ctx.fp16_math());
  out["device"] = Value(std::move(dev));

  // Runtime bridge provenance: what the mlx-omarchy backend itself sees.
  Object rt;
  rt["mlx_backend_available"] = Value(omarchy::is_available());
  rt["mlx_backend_error"] = Value(omarchy::init_error());
  if (omarchy::is_available()) {
    try {
      const auto& cap = omarchy::capability_report(opt.device_index);
      rt["capability_report_device"] = Value(cap.device_name);
      rt["capability_report_fp16"] = Value(cap.shader_float16);
      rt["capability_report_storage16"] =
          Value(cap.storage_buffer_16bit_access);
      rt["capability_report_shared_mem"] =
          Value(double(cap.max_compute_shared_memory_size));
    } catch (const std::exception& e) {
      rt["capability_report_error"] = Value(e.what());
    }
  }
  out["mlx_runtime"] = Value(std::move(rt));

  out["model_dims"] = dims.to_json();
  out["config"] = ([&] {
    Object c;
    c["warmup_iters_per_round"] = Value(double(opt.warmup));
    c["repetitions_per_round"] = Value(double(opt.reps));
    c["rounds"] = Value(double(opt.rounds));
    c["position_balance_complete"] = Value(true);
    c["idle_wait_s"] = Value(double(opt.idle_wait_s));
    c["comparator_path"] = Value(opt.comparator_path);
    return Value(std::move(c));
  })();
  out["self_check"] = self_check_json(checks);
  out["thermal_before"] = thermal_json(thermal_before, sensors_before);
  out["thermal_after"] = thermal_json(thermal_after, sensors_after);

  Object cmpj;
  cmpj["role"] = Value(
      "pin identity and model-level context; matched-kernel gate numbers come from ggml_op_comparator.json");
  if (cmp.raw.has("pin"))
    cmpj["pin"] = cmp.raw.at("pin");
  if (cmp.raw.has("model_dims"))
    cmpj["model_dims"] = cmp.raw.at("model_dims");
  if (cmp.raw.has("prefill"))
    cmpj["prefill"] = cmp.raw.at("prefill");
  if (cmp.raw.has("decode"))
    cmpj["decode"] = cmp.raw.at("decode");
  if (cmp.raw.has("thermal"))
    cmpj["thermal"] = cmp.raw.at("thermal");
  if (cmp.raw.has("source_receipt"))
    cmpj["source_receipt"] = cmp.raw.at("source_receipt");
  out["comparator"] = Value(std::move(cmpj));
  out["ggml_comparator"] = gcmp;

  double thermal_delta = 0;
  if (!thermal_before.empty() && cmp.idle_baseline_c > -100) {
    double now_max = 0;
    for (auto& z : thermal_before)
      now_max = std::max(now_max, z.temp_c);
    thermal_delta = now_max - cmp.idle_baseline_c;
  }
  auto section_comparables = [&](const char* section, double worst) {
    Object o;
    o["metric"] = Value("gflops");
    o["comparator"] = Value(
        "pinned ggml Vulkan per-op median, matched shape and quantization (ggml_op_comparator.json)");
    o["ratio_formula"] = Value(
        "spike row gflops / comparator case gflops; case gflops = flops / median_us");
    o["spike_worst_row_ratio"] = Value(worst);
    o["gate"] = Value(worst >= kGoNoGoThreshold);
    return Value(std::move(o));
  };
  out["comparables_prefill"] = section_comparables("prefill", worst_prefill);
  out["comparables_decode"] = section_comparables("decode", worst_decode);
  out["comparables_attention"] =
      section_comparables("attention", worst_attention);

  if (opt.profile_stages) {
    out["stage_profile"] =
        stage_profile_json(stage_profile, opt.warmup, opt.reps);
  }

  omarchy_spike::json::Array rowjs;
  for (auto& r : rows) {
    rowjs.push_back(row_json(r, opt.warmup, opt.reps));
  }
  out["rows"] = Value(std::move(rowjs));

  Object gate;
  gate["threshold"] = Value(kGoNoGoThreshold);
  gate["prefill_ratio"] = Value(worst_prefill);
  gate["decode_ratio"] = Value(worst_decode);
  gate["attention_ratio"] = Value(worst_attention);
  gate["thermal_delta_vs_baseline_c"] =
      cmp.idle_baseline_c > -1000 && !thermal_before.empty()
      ? Value(thermal_delta)
      : Value();
  gate["verdict"] = Value(go ? "GO" : "NO-GO");
  gate["scope"] = Value(
      "every gate row compares matched kernels: spike op gflops versus the "
      "pinned ggml Vulkan per-op median for the same op, shape, and "
      "quantization (prefill mul_mat F16, decode mul_mat Q4_K M=1, causal "
      "GQA flash_attn_ext). llama-bench pp512/tg128 is recorded as "
      "model-level context only.");
  out["go_no_go"] = Value(std::move(gate));

  std::string receipt = Value(std::move(out)).dump(2);
  if (!opt.output_path.empty()) {
    std::ofstream f(opt.output_path);
    f << receipt << "\n";
    if (!f) {
      die("cannot write receipt to " + opt.output_path, 1);
    }
  }
  std::cout << receipt << std::endl;

  std::cerr << "verdict: " << (go ? "GO" : "NO-GO") << " (prefill "
            << worst_prefill << ", decode " << worst_decode << ", attention "
            << worst_attention << ", threshold " << kGoNoGoThreshold << ")\n";
  return go ? 0 : 3;
}
