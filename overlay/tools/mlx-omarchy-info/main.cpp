// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

// mlx-omarchy-info: capability report and runtime smoke tool for the Omarchy
// Vulkan backend of MLX. Also reports ANE FDT/accel0/module visibility and
// Core ML frontend/cache presence (plan §73 installed-state surface).
//
//   mlx-omarchy-info                 human-readable capability report
//   mlx-omarchy-info --json          the same report as JSON
//   mlx-omarchy-info --trace-smoke   execute a real Vulkan buffer round trip
//                                  and print the backend trace counters
//   mlx-omarchy-info --device N      report device N instead of the default
//   mlx-omarchy-info --check-bundle D
//                                  validate the ANE bundle at D and print its
//                                  parsed contract as [receipt] lines
//
// Exit codes: 0 success, 1 the backend, smoke, or bundle check failed,
// 2 usage error or bundle directory not found.

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <string>
#include <sys/stat.h>
#include <vector>

#include "mlx/backend/omarchy/allocator.h"
#include "mlx/backend/omarchy/ane/bundle.h"
#include "mlx/backend/omarchy/device.h"
#include "mlx/backend/omarchy/encoder.h"
#include "mlx/backend/omarchy/trace.h"
#include "mlx/device.h"
#include "mlx/stream.h"

namespace omarchy = mlx::core::omarchy;

namespace {
std::string json_escape(const std::string& in) {
  std::string out;
  for (char c : in) {
    switch (c) {
      case '"':
        out += "\\\"";
        break;
      case '\\':
        out += "\\\\";
        break;
      case '\n':
        out += "\\n";
        break;
      default:
        out += c;
    }
  }
  return out;
}

std::string version_string(uint32_t version) {
  return std::to_string(VK_API_VERSION_MAJOR(version)) + "." +
      std::to_string(VK_API_VERSION_MINOR(version)) + "." +
      std::to_string(VK_API_VERSION_PATCH(version));
}

bool env_flag(const char* name) {
  const char* v = std::getenv(name);
  if (!v) {
    return false;
  }
  std::string s = v;
  for (auto& c : s) {
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  }
  return s == "1" || s == "on" || s == "true" || s == "yes";
}

std::filesystem::path sysroot_path() {
  const char* raw = std::getenv("MLX_OMARCHY_SYSROOT");
  if (raw != nullptr && raw[0] != '\0') {
    return raw;
  }
  return "/";
}

std::string read_text_file(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    return {};
  }
  std::string value((std::istreambuf_iterator<char>(input)), {});
  while (!value.empty() &&
         (value.back() == '\0' || value.back() == '\n' || value.back() == '\r')) {
    value.pop_back();
  }
  return value;
}

bool ane_compatible_token(const std::string& token) {
  return token == "apple,ane" ||
      (token.size() > 4 && token.compare(token.size() - 4, 4, "-ane") == 0);
}

struct AneCapability {
  bool fdt_node{false};
  std::vector<std::string> fdt_compatible;
  bool accel0{false};
  bool accel0_character_device{false};
  bool module_present{false};
  std::string module_version;
  bool available{false};
};

struct CoremlCapability {
  bool frontend_present{false};
  bool command_present{false};
  std::string cache_root;
  bool cache_present{false};
  unsigned long long cache_entries{0};
};

AneCapability collect_ane() {
  AneCapability out;
  const auto root = sysroot_path();
  const auto dt = root == "/"
      ? std::filesystem::path("/sys/firmware/devicetree/base")
      : root / "sys/firmware/devicetree/base";
  std::error_code ec;
  if (std::filesystem::exists(dt, ec)) {
    for (auto it = std::filesystem::recursive_directory_iterator(
             dt,
             std::filesystem::directory_options::skip_permission_denied,
             ec);
         it != std::filesystem::recursive_directory_iterator();
         it.increment(ec)) {
      if (ec || !it->is_directory()) {
        continue;
      }
      const std::string name = it->path().filename().string();
      const bool named = name == "ane" || name.rfind("ane@", 0) == 0;
      const std::string raw = read_text_file(it->path() / "compatible");
      std::vector<std::string> tokens;
      size_t offset = 0;
      while (offset < raw.size()) {
        size_t end = raw.find('\0', offset);
        if (end == std::string::npos) {
          end = raw.size();
        }
        std::string token = raw.substr(offset, end - offset);
        if (!token.empty()) {
          tokens.push_back(std::move(token));
        }
        offset = end + 1;
      }
      bool hit = false;
      for (const auto& token : tokens) {
        if (ane_compatible_token(token)) {
          hit = true;
          out.fdt_compatible.push_back(token);
        }
      }
      if (named || hit) {
        out.fdt_node = true;
        if (!hit) {
          out.fdt_compatible.insert(
              out.fdt_compatible.end(), tokens.begin(), tokens.end());
        }
      }
    }
  }
  std::sort(out.fdt_compatible.begin(), out.fdt_compatible.end());
  out.fdt_compatible.erase(
      std::unique(out.fdt_compatible.begin(), out.fdt_compatible.end()),
      out.fdt_compatible.end());
  if (out.fdt_compatible.size() > 8) {
    out.fdt_compatible.resize(8);
  }

  std::filesystem::path accel = root == "/"
      ? std::filesystem::path("/dev/accel/accel0")
      : root / "dev/accel/accel0";
  const char* accel_env = std::getenv("MLX_OMARCHY_ACCEL_DEV");
  if (accel_env != nullptr && accel_env[0] != '\0') {
    accel = accel_env;
  }
  struct stat status {};
  if (::lstat(accel.c_str(), &status) == 0) {
    out.accel0 = true;
    out.accel0_character_device = S_ISCHR(status.st_mode);
  }

  const auto module = root == "/"
      ? std::filesystem::path("/sys/module/ane")
      : root / "sys/module/ane";
  if (std::filesystem::is_directory(module, ec)) {
    out.module_present = true;
    out.module_version = read_text_file(module / "version");
  }
  out.available =
      out.fdt_node && out.accel0_character_device && out.module_present;
  return out;
}

CoremlCapability collect_coreml() {
  CoremlCapability out;
  const char* home = std::getenv("HOME");
  const std::filesystem::path home_path =
      (home != nullptr && home[0] != '\0') ? home : std::filesystem::path();
  const char* cache_env = std::getenv("MLX_OMARCHY_CACHE_DIR");
  if (cache_env != nullptr && cache_env[0] != '\0') {
    out.cache_root = (std::filesystem::path(cache_env) / "coreml").string();
  } else if (!home_path.empty()) {
    out.cache_root = (home_path / ".cache/mlx-omarchy/coreml").string();
  }
  std::error_code ec;
  out.cache_present = !out.cache_root.empty() &&
      std::filesystem::is_directory(out.cache_root, ec);
  if (out.cache_present) {
    for (const auto& entry :
         std::filesystem::directory_iterator(out.cache_root, ec)) {
      const std::string name = entry.path().filename().string();
      if (entry.is_directory() && !name.empty() && name.front() != '.') {
        out.cache_entries++;
      }
    }
  }
  const char* frontend = std::getenv("MLX_OMARCHY_COREML_ROOT");
  if (frontend != nullptr && frontend[0] != '\0') {
    out.frontend_present = std::filesystem::is_regular_file(
        std::filesystem::path(frontend) / "__init__.py", ec);
  }
  if (!home_path.empty()) {
    out.command_present = std::filesystem::is_regular_file(
        home_path / ".local/bin/mlx-omarchy-coreml", ec);
  }
  return out;
}

void print_ane_json(const AneCapability& ane) {
  std::cout << "  \"ane\": {\n";
  std::cout << "    \"fdt_node\": " << (ane.fdt_node ? 1 : 0) << ",\n";
  std::cout << "    \"fdt_compatible\": [";
  for (size_t i = 0; i < ane.fdt_compatible.size(); ++i) {
    if (i > 0) {
      std::cout << ", ";
    }
    std::cout << "\"" << json_escape(ane.fdt_compatible[i]) << "\"";
  }
  std::cout << "],\n";
  std::cout << "    \"accel0\": " << (ane.accel0 ? 1 : 0) << ",\n";
  std::cout << "    \"accel0_character_device\": "
            << (ane.accel0_character_device ? 1 : 0) << ",\n";
  std::cout << "    \"module_present\": " << (ane.module_present ? 1 : 0)
            << ",\n";
  std::cout << "    \"module_version\": \"" << json_escape(ane.module_version)
            << "\",\n";
  std::cout << "    \"available\": " << (ane.available ? 1 : 0) << "\n";
  std::cout << "  }";
}

void print_coreml_json(const CoremlCapability& coreml) {
  std::cout << "  \"coreml\": {\n";
  std::cout << "    \"frontend_present\": " << (coreml.frontend_present ? 1 : 0)
            << ",\n";
  std::cout << "    \"command_present\": " << (coreml.command_present ? 1 : 0)
            << ",\n";
  std::cout << "    \"cache_root\": \"" << json_escape(coreml.cache_root)
            << "\",\n";
  std::cout << "    \"cache_present\": " << (coreml.cache_present ? 1 : 0)
            << ",\n";
  std::cout << "    \"cache_entries\": " << coreml.cache_entries << "\n";
  std::cout << "  }";
}

void print_ane_text(const AneCapability& ane) {
  std::cout << "  ane fdt node:      " << (ane.fdt_node ? "yes" : "no") << "\n";
  std::cout << "  ane compatible:    ";
  if (ane.fdt_compatible.empty()) {
    std::cout << "none\n";
  } else {
    for (size_t i = 0; i < ane.fdt_compatible.size(); ++i) {
      if (i > 0) {
        std::cout << ",";
      }
      std::cout << ane.fdt_compatible[i];
    }
    std::cout << "\n";
  }
  std::cout << "  ane accel0:        "
            << (ane.accel0_character_device ? "yes" : "no") << "\n";
  std::cout << "  ane module:        " << (ane.module_present ? "yes" : "no")
            << "\n";
  if (!ane.module_version.empty()) {
    std::cout << "  ane module ver:    " << ane.module_version << "\n";
  }
  std::cout << "  ane available:     " << (ane.available ? "yes" : "no")
            << "\n";
}

void print_coreml_text(const CoremlCapability& coreml) {
  std::cout << "  coreml frontend:   "
            << (coreml.frontend_present ? "yes" : "no") << "\n";
  std::cout << "  coreml command:    "
            << (coreml.command_present ? "yes" : "no") << "\n";
  std::cout << "  coreml cache:      "
            << (coreml.cache_root.empty() ? "none" : coreml.cache_root) << "\n";
  std::cout << "  coreml cache ents: " << coreml.cache_entries
            << (coreml.cache_present ? "" : " (absent)") << "\n";
}


void print_json(uint32_t index) {
  const auto& caps = omarchy::capability_report(index);
  const auto& trace = omarchy::trace::counters();
  bool non_apple = env_flag("MLX_OMARCHY_ALLOW_NON_APPLE");
  std::cout << "{\n";
  auto str_field = [&](const char* k, const std::string& v, bool comma) {
    std::cout << "  \"" << k << "\": \"" << json_escape(v) << "\""
              << (comma ? ",\n" : "\n");
  };
  auto num_field = [&](const char* k, unsigned long long v, bool comma) {
    std::cout << "  \"" << k << "\": " << v << (comma ? ",\n" : "\n");
  };
  str_field("tool", "mlx-omarchy-info", true);
  str_field("device_name", caps.device_name, true);
  str_field("driver_name", caps.driver_name, true);
  str_field("api_version", version_string(caps.api_version), true);
  str_field("driver_version_raw", std::to_string(caps.driver_version), true);
  num_field("vendor_id", caps.vendor_id, true);
  num_field("device_id", caps.device_id, true);
  num_field("queue_family_index", caps.queue_family_index, true);
  num_field("queue_count", caps.queue_count, true);
  num_field(
      "total_memory_bytes",
      static_cast<unsigned long long>(caps.total_memory),
      true);
  num_field(
      "max_allocation_bytes",
      static_cast<unsigned long long>(caps.max_allocation_size),
      true);
  num_field(
      "max_buffer_bytes",
      static_cast<unsigned long long>(caps.max_buffer_size),
      true);
  num_field(
      "max_storage_buffer_range_bytes",
      static_cast<unsigned long long>(caps.max_storage_buffer_range),
      true);
  num_field(
      "max_compute_shared_memory_bytes",
      caps.max_compute_shared_memory_size,
      true);
  num_field(
      "max_compute_work_group_invocations",
      caps.max_compute_work_group_invocations,
      true);
  num_field(
      "max_compute_work_group_size_x",
      caps.max_compute_work_group_size[0],
      true);
  num_field("unified_memory", caps.unified_memory ? 1 : 0, true);
  num_field("host_visible_coherent", caps.host_visible_coherent ? 1 : 0, true);
  num_field("timeline_semaphore", caps.timeline_semaphore ? 1 : 0, true);
  num_field("shader_float16", caps.shader_float16 ? 1 : 0, true);
  num_field("shader_int16", caps.shader_int16 ? 1 : 0, true);
  num_field(
      "storage_buffer_16bit_access",
      caps.storage_buffer_16bit_access ? 1 : 0,
      true);
  num_field(
      "max_per_stage_descriptor_storage_buffers",
      caps.max_per_stage_descriptor_storage_buffers,
      true);
  num_field(
      "max_descriptor_set_storage_buffers",
      caps.max_descriptor_set_storage_buffers,
      true);
  num_field("non_apple_dev_override", non_apple ? 1 : 0, true);
  if (caps.simulated) {
    num_field("simulated", 1, true);
    str_field("simulation_profile", caps.simulation_profile, true);
    str_field("simulated_driver_variant", caps.simulated_driver_variant, true);
  }
  std::cout << "  \"trace\": {\n";
  num_field(
      "gpu_primitive_dispatches", trace.gpu_primitive_dispatches.load(), true);
  num_field("vk_submissions", trace.vk_submissions.load(), true);
  num_field("vk_buffer_copies", trace.vk_buffer_copies.load(), true);
  num_field("vk_buffer_fills", trace.vk_buffer_fills.load(), true);
  num_field("vk_compute_dispatches", trace.vk_compute_dispatches.load(), true);
  num_field(
      "compiled_tape_dispatches", trace.compiled_tape_dispatches.load(), true);
  num_field(
      "compiled_tape_node_evaluations",
      trace.compiled_tape_node_evaluations.load(),
      false);
  std::cout << "  },\n";
  print_ane_json(collect_ane());
  std::cout << ",\n";
  print_coreml_json(collect_coreml());
  std::cout << "\n}\n";
}

void print_text(uint32_t index) {
  const auto& caps = omarchy::capability_report(index);
  std::cout << "mlx-omarchy-info\n";
  std::cout << "  device:            " << caps.device_name << "\n";
  std::cout << "  driver:            " << caps.driver_name << "\n";
  std::cout << "  api version:       " << version_string(caps.api_version)
            << "\n";
  std::cout << "  vendor:device id:  0x" << std::hex << caps.vendor_id << ":0x"
            << caps.device_id << std::dec << "\n";
  std::cout << "  queue family:      " << caps.queue_family_index << " ("
            << caps.queue_count << " queues)\n";
  std::cout << "  total memory:      " << (caps.total_memory >> 20)
            << " MiB (device-local heap)\n";
  std::cout << "  unified memory:    " << (caps.unified_memory ? "yes" : "no")
            << "\n";
  std::cout << "  host coherent:     "
            << (caps.host_visible_coherent ? "yes" : "no")
            << (caps.host_visible_coherent
                    ? ""
                    : " (explicit flush/invalidate required)")
            << "\n";
  std::cout << "  timeline semaphore:"
            << (caps.timeline_semaphore ? "yes" : "no") << "\n";
  std::cout << "  shader float16:    " << (caps.shader_float16 ? "yes" : "no")
            << "\n";
  std::cout << "  shader int16:      " << (caps.shader_int16 ? "yes" : "no")
            << "\n";
  std::cout << "  16-bit storage:    "
            << (caps.storage_buffer_16bit_access ? "yes" : "no") << "\n";
  std::cout << "  max compute shm:   " << caps.max_compute_shared_memory_size
            << " B\n";
  std::cout << "  max wg invocations:"
            << caps.max_compute_work_group_invocations << "\n";
  std::cout << "  storage buffer bindings (per stage / per set): "
            << caps.max_per_stage_descriptor_storage_buffers << " / "
            << caps.max_descriptor_set_storage_buffers << "\n";
  if (env_flag("MLX_OMARCHY_ALLOW_NON_APPLE")) {
    std::cout << "  NOTE: MLX_OMARCHY_ALLOW_NON_APPLE=1; this is a"
                 " development-only device, not Omarchy Honeykrisp.\n";
  }
  if (caps.simulated) {
    std::cout << "  SIMULATED:         capability profile '"
              << caps.simulation_profile << "' (stands in for driver_variant "
              << caps.simulated_driver_variant << "); NOT hardware results\n";
  }
  print_ane_text(collect_ane());
  print_coreml_text(collect_coreml());
}

// Execute one real buffer round trip: host write -> vkCmdCopyBuffer ->
// fence -> host read back.
int trace_smoke(uint32_t index) {
  auto& dev = omarchy::device(index);
  mlx::core::Stream s = mlx::core::new_stream(mlx::core::Device::gpu);
  auto& encoder = omarchy::get_command_encoder(s);

  constexpr size_t kBytes = 1 << 16;
  auto src = omarchy::allocator().malloc(kBytes);
  auto dst = omarchy::allocator().malloc(kBytes);
  auto* src_ptr = static_cast<uint8_t*>(src.raw_ptr());
  auto* dst_ptr = static_cast<uint8_t*>(dst.raw_ptr());
  if (!src_ptr || !dst_ptr) {
    std::cerr << "[mlx-omarchy-info] smoke failed: unmapped buffer\n";
    return 1;
  }
  for (size_t i = 0; i < kBytes; ++i) {
    src_ptr[i] = static_cast<uint8_t>(i % 251);
  }
  std::memset(dst_ptr, 0, kBytes);

  auto* src_buf = static_cast<omarchy::VulkanBuffer*>(src.ptr());
  auto* dst_buf = static_cast<omarchy::VulkanBuffer*>(dst.ptr());
  encoder.copy_buffer(src_buf->buffer, dst_buf->buffer, kBytes);
  encoder.synchronize();

  int mismatches = 0;
  for (size_t i = 0; i < kBytes; ++i) {
    if (src_ptr[i] != dst_ptr[i]) {
      mismatches++;
    }
  }
  auto& trace = omarchy::trace::counters();
  if (mismatches != 0) {
    std::cerr << "[mlx-omarchy-info] smoke FAILED: " << mismatches
              << " mismatching bytes after device copy\n";
    return 1;
  }
  std::cout << "buffer round trip: OK (" << kBytes << " bytes, device "
            << dev.capabilities().device_name << ")\n";
  std::cout << "trace: vk_submissions=" << trace.vk_submissions.load()
            << " vk_buffer_copies=" << trace.vk_buffer_copies.load()
            << " vk_buffer_fills=" << trace.vk_buffer_fills.load()
            << " vk_compute_dispatches="
            << trace.vk_compute_dispatches.load()
            << " gpu_primitive_dispatches="
            << trace.gpu_primitive_dispatches.load() << "\n";
  return 0;
}

// Prints one tensor list of the parsed bundle contract as [receipt] lines.
void print_tensor_list(
    const char* kind,
    const std::vector<omarchy::ane::AneTensor>& tensors) {
  if (tensors.empty()) {
    std::cout << "[receipt] " << kind << ": none\n";
    return;
  }
  for (const auto& tensor : tensors) {
    std::cout << "[receipt] " << kind << " " << tensor.name << ": index="
              << tensor.index << " dtype=" << tensor.dtype << " shape=[";
    for (size_t i = 0; i < tensor.shape.size(); ++i) {
      if (i > 0) {
        std::cout << ",";
      }
      std::cout << tensor.shape[i];
    }
    std::cout << "] byte_size=" << tensor.byte_size
              << " stride=" << tensor.stride << "\n";
  }
}

void print_logical_results(
    const std::vector<omarchy::ane::AneLogicalResult>& results) {
  for (size_t index = 0; index < results.size(); ++index) {
    const auto& result = results[index];
    std::cout << "[receipt] logical_result " << index << " " << result.name
              << ": dtype=" << result.dtype << " shape=[";
    for (size_t i = 0; i < result.shape.size(); ++i) {
      if (i > 0) {
        std::cout << ",";
      }
      std::cout << result.shape[i];
    }
    std::cout << "] tensor=" << result.tensor
              << " element_offset=" << result.element_offset
              << " element_count=" << result.element_count
              << " conversion=" << result.conversion << "\n";
  }
}

void print_anec_nchw(
    const omarchy::ane::AneAnecHeader& header,
    uint32_t channel) {
  std::cout << " nchw=[";
  for (size_t i = 0; i < header.nchw[channel].size(); ++i) {
    if (i > 0) {
      std::cout << ",";
    }
    std::cout << header.nchw[channel][i];
  }
  std::cout << "]";
}

void print_anec_binding(
    const char* kind,
    const omarchy::ane::AneProgramBinding& binding,
    const omarchy::ane::AneAnecHeader& header) {
  std::cout << "[receipt] anec " << kind << " " << binding.tensor
            << ": channel=" << binding.channel
            << " logical_bytes=" << binding.logical_bytes
            << " allocation_bytes=" << binding.allocation_bytes
            << " element_offset=" << binding.element_offset
            << " element_count=" << binding.element_count
            << " physical_elements=" << binding.physical_elements;
  print_anec_nchw(header, static_cast<uint32_t>(binding.channel));
  std::cout << "\n";
}

void print_anec_program(
    size_t index,
    const omarchy::ane::AneValidatedProgram& validated,
    const omarchy::ane::AneManifest& manifest) {
  const auto& program = manifest.programs.at(validated.manifest_index);
  const auto& header = validated.anec_header;
  std::cout << "[receipt] dispatch " << index
            << ": program=" << validated.manifest_index
            << " payload=" << program.payload
            << " operation=" << program.operation
            << " encoder=" << program.encoder
            << " scratch_bytes=" << program.scratch_bytes
            << " payload_size=" << header.payload_size
            << " td_size=" << header.task_descriptor_size
            << " td_count=" << header.task_descriptor_count
            << " sources=" << header.source_count
            << " destinations=" << header.destination_count
            << "\n";
  for (const auto& binding : program.inputs) {
    print_anec_binding("input", binding, header);
  }
  for (const auto& binding : program.outputs) {
    print_anec_binding("output", binding, header);
  }
}

// Validates one bundle directory and prints its parsed contract. This runs
// load_bundle only (see mlx/backend/omarchy/ane/bundle.h): no Vulkan device
// is opened and nothing reaches a descriptor submission.
// Exit codes: 0 valid, 1 named loader error, 2 directory not found.
int check_bundle(const std::string& dir_arg) {
  std::filesystem::path dir(dir_arg);
  try {
    omarchy::ane::AneBundle bundle = omarchy::ane::load_bundle(dir);
    const omarchy::ane::AneManifest& m = bundle.manifest;
    std::cout << "[receipt] bundle: " << dir.string() << "\n";
    std::cout << "[receipt] graph: " << m.name << "\n";
    std::cout << "[receipt] graph_hash: " << m.graph_hash << "\n";
    std::cout << "[receipt] task_descriptors: " << m.task_descriptors << "\n";
    print_tensor_list("input", m.inputs);
    print_tensor_list("output", m.outputs);
    print_logical_results(m.logical_results);
    print_tensor_list("state", m.state);
    print_tensor_list("intermediate", m.intermediates);
    std::cout << "[receipt] dispatch_plan:";
    for (uint64_t index : m.dispatch_plan) {
      std::cout << " " << index;
    }
    std::cout << "\n";
    for (size_t index = 0; index < bundle.programs.size(); ++index) {
      print_anec_program(index, bundle.programs[index], m);
    }
    for (const auto& payload : m.payloads) {
      std::cout << "[receipt] payload " << payload.role << ": " << payload.path
                << " sha256=" << payload.sha256
                << " byte_size=" << payload.byte_size << "\n";
    }
    std::cout << "[receipt] compiler: host_build=" << m.compiler.host_build
              << " toolchain=" << m.compiler.toolchain
              << " target=" << m.compiler.target << "\n";
    std::cout << "[receipt] driver_abi_major: " << m.driver_abi_major << "\n";
    std::cout << "[receipt] provenance: repo=" << m.provenance.source_repo
              << " commit=" << m.provenance.source_commit << "\n";
    std::cout << "[receipt] OK: bundle valid\n";
    return 0;
  } catch (const omarchy::ane::AneBundleNotFound&) {
    std::cerr << "[mlx-omarchy-info] bundle not found (region stays on "
                 "Vulkan): "
              << dir.string() << "\n";
    return 2;
  } catch (const std::exception& ex) {
    std::cerr << "[mlx-omarchy-info] check-bundle failed: " << ex.what()
              << "\n";
    return 1;
  }
}

void usage() {
  std::cerr
      << "usage: mlx-omarchy-info [--json] [--trace-smoke] [--device N]\n"
         "       mlx-omarchy-info --check-bundle <dir>\n";
}

} // namespace

int main(int argc, char** argv) {
  uint32_t index = 0;
  bool json = false;
  bool smoke = false;
  std::string check_bundle_dir;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--json") {
      json = true;
    } else if (arg == "--trace-smoke") {
      smoke = true;
    } else if (arg == "--check-bundle" && i + 1 < argc) {
      check_bundle_dir = argv[++i];
    } else if (arg == "--check-bundle") {
      usage();
      return 2;
    } else if (arg == "--device" && i + 1 < argc) {
      index = static_cast<uint32_t>(std::atoi(argv[++i]));
    } else if (arg == "--help" || arg == "-h") {
      usage();
      return 0;
    } else {
      usage();
      return 2;
    }
  }

  if (!check_bundle_dir.empty()) {
    return check_bundle(check_bundle_dir);
  }

  try {
    if (smoke) {
      return trace_smoke(index);
    }
    if (!omarchy::is_available()) {
      std::cerr << "[mlx-omarchy-info] backend unavailable:\n  "
                << omarchy::init_error() << "\n";
      return 1;
    }
    json ? print_json(index) : print_text(index);
    return 0;
  } catch (const std::exception& ex) {
    std::cerr << "[mlx-omarchy-info] error: " << ex.what() << "\n";
    return 1;
  }
}
