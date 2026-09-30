// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

#include "mlx/backend/omarchy/ane/runtime_detail.h"

#include "mlx/backend/omarchy/ane/bundle.h"

#include <ane.h>

#include <cerrno>
#include "mlx/backend/omarchy/ane/driver_abi.h"

#include <libdrm/drm.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/utsname.h>
#include <unistd.h>
#include <vector>

namespace mlx::core::omarchy::ane::detail {
namespace {

// Per-ABI eligibility profile. The runtime gate picks one profile based on
// the MLX_OMARCHY_ANE_ABI env var at process start (default 1 = the audited
// M1 / T8103 / T6001 lane). Each ABI carries its own DT-compatible set,
// kernel-module name, driver version, and pinned libane commit; per-ABI
// pins keep one ABI from loosening the other.
struct AbiProfile {
  int abi{1};
  const char* env_name{""};
  // Any one of these DT compatibles passes the eligibility check.
  std::initializer_list<const char*> compatibles;
  // The /sys/module/<module_name> directory the gate reads.
  const char* module_name{""};
  // Exact libane commit the worker will require under this ABI.
  const char* libane_commit{""};
  // Exact string written into the readiness identity's "driver_abi=" field.
  const char* identity_abi_tag{""};
};

constexpr AbiProfile kAbiProfiles[] = {
    {1,
     "ABI 1 (M1 / T8103 / T6001)",
     {"apple,t8103-ane"},
     "ane",
     "6fa243ac7241119a9eb229abbf8cb4dd8949f915",
     "1"},
    {2,
     "ABI 2 (M2 / T6021)",
     {"apple,t6021-ane"},
     "ane_t6021",
     // Smallest libane that loads every h14 ANEC currently on disk in the
     // M2 fixtures (bmm islands + rms + select-runtime). Bumped by the E5/E6
     // owners when select numerics land.
     "8b010938aeb64bfa04b95e89da0bedd2ef9e3e72",
     "2"},
};

constexpr int kAneDefaultAbi = 1;

const AbiProfile& requested_abi_profile() {
  const char* value = std::getenv("MLX_OMARCHY_ANE_ABI");
  if (value == nullptr || value[0] == '\0') {
    return kAbiProfiles[kAneDefaultAbi - 1];
  }
  // Strtol with a non-null end-pointer to reject whitespace and trailing
  // junk; std::stoi would accept "2foo".
  char* end = nullptr;
  errno = 0;
  const long parsed = std::strtol(value, &end, 10);
  if (errno != 0 || end == value || (end != nullptr && *end != '\0')) {
    throw runtime_error(
        "MLX_OMARCHY_ANE_ABI='" + std::string(value) +
        "' is not a valid ABI integer (expected 1 or 2)");
  }
  for (const auto& profile : kAbiProfiles) {
    if (profile.abi == parsed) {
      return profile;
    }
  }
  throw runtime_error(
      "MLX_OMARCHY_ANE_ABI=" + std::to_string(parsed) +
      " is not a supported ABI (expected 1 or 2)");
}

using Clock = std::chrono::steady_clock;

class UncertainCompletion : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

std::string trim_system_value(std::string value) {
  while (!value.empty() &&
         (value.back() == '\0' || value.back() == '\n' || value.back() == '\r')) {
    value.pop_back();
  }
  return value;
}

std::string read_value(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw runtime_error("cannot read hardware identity " + path.string());
  }
  return trim_system_value(
      std::string(std::istreambuf_iterator<char>(input), {}));
}

bool has_compatible(const std::string& values, const std::string& expected) {
  size_t offset = 0;
  while (offset < values.size()) {
    size_t end = values.find('\0', offset);
    if (end == std::string::npos) {
      end = values.size();
    }
    if (values.substr(offset, end - offset) == expected) {
      return true;
    }
    offset = end + 1;
  }
  return false;
}

std::filesystem::path single_entry(
    const std::filesystem::path& directory,
    const std::string& prefix,
    const std::string& suffix) {
  std::filesystem::path match;
  for (const auto& entry : std::filesystem::directory_iterator(directory)) {
    const std::string name = entry.path().filename().string();
    if (name.rfind(prefix, 0) != 0 ||
        (name.size() < suffix.size() ||
         name.compare(name.size() - suffix.size(), suffix.size(), suffix) != 0)) {
      continue;
    }
    if (!match.empty()) {
      throw runtime_error("hardware identity has multiple matching entries in " +
                          directory.string());
    }
    match = entry.path();
  }
  if (match.empty()) {
    throw runtime_error("hardware identity has no matching entry in " +
                        directory.string());
  }
  return match;
}

int query_driver_abi_major(int expected) {
  const int fd = ::open("/dev/accel/accel0", O_RDWR | O_CLOEXEC);
  if (fd < 0) {
    throw runtime_error(
        "ANE driver ABI mismatch: runtime requires major " +
        std::to_string(expected) + ", driver reports unavailable (open failed: " +
        std::strerror(errno) + ")");
  }
  drm_version version{};
  if (::ioctl(fd, DRM_IOCTL_VERSION, &version) != 0) {
    const int error = errno;
    ::close(fd);
    throw runtime_error(
        "ANE driver ABI mismatch: runtime requires major " +
        std::to_string(expected) + ", driver reports unavailable (DRM_IOCTL_VERSION: " +
        std::strerror(error) + ")");
  }
  ::close(fd);
  return version.version_major;
}


std::string verify_hardware_eligibility(const AbiProfile& profile) {
  utsname system{};
  if (::uname(&system) != 0) {
    throw runtime_error("uname failed: " + std::string(std::strerror(errno)));
  }
  if (std::string(system.sysname) != "Linux" ||
      std::string(system.machine) != "aarch64") {
    throw runtime_error(
        "ANE execution requires Linux aarch64 on " + std::string(profile.env_name));
  }

  const auto node = single_entry("/proc/device-tree/soc", "ane@", "");
  const std::string compatible = read_value(node / "compatible");
  bool matched_compatible = false;
  const char* matched_compatible_str = nullptr;
  for (const char* expected : profile.compatibles) {
    if (has_compatible(compatible, expected)) {
      matched_compatible = true;
      matched_compatible_str = expected;
      break;
    }
  }
  if (!matched_compatible) {
    std::string expected_list;
    bool first_expected = true;
    for (const char* expected : profile.compatibles) {
      if (!first_expected) {
        expected_list += " or ";
      }
      expected_list += expected;
      first_expected = false;
    }
    throw runtime_error(
        "device-tree ANE node does not match " + expected_list +
        " for " + std::string(profile.env_name));
  }
  const std::string status = read_value(node / "status");
  if (status != "okay") {
    throw runtime_error("device-tree ANE status is '" + status + "', not 'okay'");
  }

  const std::string module_version_path =
      "/sys/module/" + std::string(profile.module_name) + "/version";
  std::string module_version = "<not exposed>";
  bool module_version_exposed = false;
  struct stat version_stat {};
  if (::stat(module_version_path.c_str(), &version_stat) == 0) {
    module_version = read_value(module_version_path);
    module_version_exposed = true;
  }
  const std::string module_source_path =
      "/sys/module/" + std::string(profile.module_name) + "/srcversion";
  std::string module_source;
  bool module_source_exposed = false;
  struct stat src_stat {};
  if (::stat(module_source_path.c_str(), &src_stat) == 0) {
    module_source = read_value(module_source_path);
    module_source_exposed = true;
  } else {
    module_source = "<not exposed>";
  }

  const auto platform = single_entry("/sys/bus/platform/devices", "", ".ane");
  const auto driver = std::filesystem::canonical(platform / "driver").filename();
  if (driver != "ane") {
    throw runtime_error(
        "ANE platform device is not bound to the ane driver (found '" +
        driver.string() + "')");
  }
  const std::string power_control = read_value(platform / "power/control");
  const std::string runtime_status = read_value(platform / "power/runtime_status");
  if (power_control != "on" || runtime_status != "active") {
    throw runtime_error(
        "ANE runtime power must be on and active, found control='" +
        power_control + "' status='" + runtime_status + "'");
  }

  struct stat device_status {};
  if (::stat("/dev/accel/accel0", &device_status) != 0 ||
      !S_ISCHR(device_status.st_mode) ||
      ::access("/dev/accel/accel0", R_OK | W_OK) != 0) {
    throw runtime_error("qualified /dev/accel/accel0 is not readable and writable");
  }
  const int driver_abi_major = query_driver_abi_major(profile.abi);
  require_driver_abi_major(profile.abi, driver_abi_major);

  std::ostringstream identity;
  identity << "host=" << system.nodename << " kernel=" << system.release
           << " machine=" << system.machine
           << " abi=" << profile.identity_abi_tag
           << " dt_node=" << node.filename().string()
           << " dt_compatible=" << matched_compatible_str
           << " driver_module=" << profile.module_name
           << " driver_version=" << (module_version_exposed ? module_version : "<not exposed>")
           << " driver_srcversion=" << (module_source_exposed ? module_source : "<not exposed>")
           << " runtime_pm=" << power_control << '/' << runtime_status
           << " libane_commit=" << profile.libane_commit
           << " driver_abi=" << profile.identity_abi_tag;
  return identity.str();
}

std::vector<uint8_t> read_kernel(const AneValidatedProgram& program) {
  const uint64_t aligned_task =
      (program.anec_header.task_size + 15) & ~uint64_t{15};
  const uint64_t offset = kAnecPayloadOffset + aligned_task;
  const size_t size = checked_size(program.anec_header.kernel_size, "kernel byte count");
  std::vector<uint8_t> kernel(size);
  if (kernel.empty()) {
    return kernel;
  }
  std::ifstream input(program.anec, std::ios::binary);
  input.seekg(static_cast<std::streamoff>(offset));
  input.read(reinterpret_cast<char*>(kernel.data()), kernel.size());
  if (input.gcount() != static_cast<std::streamsize>(kernel.size())) {
    throw runtime_error("cannot read validated kernel section from " + program.anec.string());
  }
  return kernel;
}

struct AneHandleDeleter {
  void operator()(ane_nn* handle) const {
    if (handle != nullptr) {
      __ane_free(handle);
    }
  }
};

struct LoadedProgram {
  const AneProgram* manifest{nullptr};
  size_t manifest_index{0};
  std::unique_ptr<ane_nn, AneHandleDeleter> handle;
  std::vector<std::vector<uint8_t>> inputs;
  std::vector<std::vector<uint8_t>> outputs;
};

std::vector<LoadedProgram> load_programs(
    const AneBundle& bundle,
    const AbiProfile& profile) {
  std::vector<LoadedProgram> loaded;
  loaded.reserve(bundle.programs.size());
  for (const auto& program : bundle.programs) {
    errno = 0;
    std::unique_ptr<ane_nn, AneHandleDeleter> handle(
        __ane_init(program.anec.c_str(), 0));
    if (!handle) {
      throw runtime_error(
          "libane rejected program " + std::to_string(program.manifest_index) +
          " or the ANE device; qualified driver " +
          std::string(profile.env_name) + " is required (errno=" +
          std::to_string(errno) + ")");
    }
    const auto& manifest = bundle.manifest.programs.at(program.manifest_index);
    for (size_t i = 0; i < manifest.inputs.size(); ++i) {
      const uint64_t size = __ane_src_size(handle.get(), static_cast<uint32_t>(i));
      if (size != manifest.inputs[i].allocation_bytes) {
        throw runtime_error(
            "libane input allocation differs for program " +
            std::to_string(program.manifest_index));
      }
    }
    for (size_t i = 0; i < manifest.outputs.size(); ++i) {
      const uint64_t size = __ane_dst_size(handle.get(), static_cast<uint32_t>(i));
      if (size != manifest.outputs[i].allocation_bytes) {
        throw runtime_error(
            "libane output allocation differs for program " +
            std::to_string(program.manifest_index));
      }
    }

    auto kernel = read_kernel(program);
    if (!kernel.empty()) {
      if (ane_kernel_capacity(handle.get()) < kernel.size()) {
        throw runtime_error(
            "libane kernel capacity differs for program " +
            std::to_string(program.manifest_index));
      }
      errno = 0;
      int result = ane_bind_kernel(handle.get(), kernel.data(), kernel.size());
      if (result != 0) {
        throw runtime_error(
            "ane_bind_kernel failed for program " +
            std::to_string(program.manifest_index) + " with result " +
            std::to_string(result) + " errno=" + std::to_string(errno));
      }
    }
    std::vector<std::vector<uint8_t>> inputs;
    for (const auto& binding : manifest.inputs) {
      inputs.emplace_back(
          checked_size(binding.allocation_bytes, "input allocation"));
    }
    std::vector<std::vector<uint8_t>> outputs;
    for (const auto& binding : manifest.outputs) {
      outputs.emplace_back(
          checked_size(binding.allocation_bytes, "output allocation"));
    }
    loaded.push_back({
        &manifest,
        program.manifest_index,
        std::move(handle),
        std::move(inputs),
        std::move(outputs)});
  }
  return loaded;
}

size_t staging_size(const AneManifest& manifest) {
  uint64_t total = 0;
  auto add = [&](const std::vector<AneTensor>& tensors) {
    for (const auto& tensor : tensors) {
      if (tensor.byte_size > std::numeric_limits<uint64_t>::max() - total) {
        throw runtime_error("host staging size overflows uint64");
      }
      total += tensor.byte_size;
    }
  };
  add(manifest.inputs);
  add(manifest.outputs);
  return checked_size(total, "host staging size");
}

void read_inputs(
    const AneManifest& manifest,
    const uint8_t* staging,
    std::map<std::string, std::vector<uint8_t>>& dense) {
  size_t offset = 0;
  for (const auto& tensor : manifest.inputs) {
    auto& data = dense.at(tensor.name);
    std::memcpy(data.data(), staging + offset, data.size());
    offset += data.size();
  }
  for (const auto& tensor : manifest.intermediates) {
    std::fill(dense.at(tensor.name).begin(), dense.at(tensor.name).end(), 0);
  }
  for (const auto& tensor : manifest.outputs) {
    std::fill(dense.at(tensor.name).begin(), dense.at(tensor.name).end(), 0);
  }
}

void write_outputs(
    const AneManifest& manifest,
    uint8_t* staging,
    const std::map<std::string, std::vector<uint8_t>>& dense) {
  size_t offset = 0;
  for (const auto& tensor : manifest.inputs) {
    offset += checked_size(tensor.byte_size, "input byte count");
  }
  for (const auto& tensor : manifest.outputs) {
    const auto& data = dense.at(tensor.name);
    std::memcpy(staging + offset, data.data(), data.size());
    offset += data.size();
  }
}

void execute_program(
    LoadedProgram& program,
    std::map<std::string, std::vector<uint8_t>>& dense,
    Clock::time_point deadline,
    bool& submitted) {
  for (size_t i = 0; i < program.manifest->inputs.size(); ++i) {
    const auto& binding = program.manifest->inputs[i];
    auto& packed = program.inputs[i];
    pack_binding(binding, dense.at(binding.tensor), packed);
    ensure_worker_before(deadline, "ANE input transfer");
    __ane_send(program.handle.get(), packed.data(), static_cast<uint32_t>(i));
  }

  ensure_worker_before(deadline, "ANE program execution submission");
  submitted = true;
  errno = 0;
  int result = ane_exec(program.handle.get());
  if (result != 0) {
    throw UncertainCompletion(
        "ane_exec failed for program " + std::to_string(program.manifest_index) +
        " with result " + std::to_string(result) + " errno=" +
        std::to_string(errno));
  }

  for (size_t i = 0; i < program.manifest->outputs.size(); ++i) {
    const auto& binding = program.manifest->outputs[i];
    auto& packed = program.outputs[i];
    __ane_read(program.handle.get(), packed.data(), static_cast<uint32_t>(i));
    unpack_binding(binding, packed, dense.at(binding.tensor));
  }
}

std::string error_reason(std::string value) {
  constexpr const char* prefix = "[omarchy-ane] runtime: ";
  if (value.rfind(prefix, 0) == 0) {
    value.erase(0, std::strlen(prefix));
  }
  if (!value.empty() && value.back() == '.') {
    value.pop_back();
  }
  return value;
}

void set_detail(WorkerReply& reply, const std::string& value) {
  std::strncpy(reply.detail, value.c_str(), sizeof(reply.detail) - 1);
}

bool send_reply(int fd, const WorkerReply& reply) {
  return ::send(fd, &reply, sizeof(reply), MSG_NOSIGNAL) ==
      static_cast<ssize_t>(sizeof(reply));
}

bool receive_command(int fd, WorkerCommand& command) {
  ssize_t received = ::recv(fd, &command, sizeof(command), 0);
  return received == static_cast<ssize_t>(sizeof(command));
}

std::map<std::string, std::vector<uint8_t>> allocate_dense(
    const AneManifest& manifest) {
  std::map<std::string, std::vector<uint8_t>> dense;
  auto add = [&](const std::vector<AneTensor>& tensors) {
    for (const auto& tensor : tensors) {
      dense.emplace(
          tensor.name,
          std::vector<uint8_t>(checked_size(tensor.byte_size, "tensor byte count")));
    }
  };
  add(manifest.inputs);
  add(manifest.outputs);
  add(manifest.state);
  add(manifest.intermediates);
  return dense;
}

} // namespace

int run_worker(
    int control_fd,
    int staging_fd,
    size_t expected_staging_size,
    const std::filesystem::path& manifest_path,
    const std::map<std::string, std::filesystem::path>& payload_paths) {
  uint8_t* staging = nullptr;
  std::vector<LoadedProgram> programs;
  try {
    const AbiProfile& profile = requested_abi_profile();
    std::string identity = verify_hardware_eligibility(profile);
    AneBundle bundle = load_bundle_snapshot(manifest_path, payload_paths);
    // Cross-check the bundle's declared ABI against the gate's profile:
    // an h14 bundle must not run against an h13 host and vice versa.
    // The manifest parser enforces target<->abi pairing (h13<->1, h14<->2);
    // this check makes the host-half of that invariant explicit on every
    // worker load.
    if (bundle.manifest.driver_abi_major != static_cast<uint64_t>(profile.abi)) {
      throw runtime_error(
          "bundle declares driver_abi_major " +
          std::to_string(bundle.manifest.driver_abi_major) +
          " but the worker gate is " + std::string(profile.env_name) +
          " (set MLX_OMARCHY_ANE_ABI to switch lanes)");
    }
    if (staging_size(bundle.manifest) != expected_staging_size) {
      throw runtime_error("parent and worker staging sizes differ");
    }
    void* mapping = ::mmap(
        nullptr,
        expected_staging_size,
        PROT_READ | PROT_WRITE,
        MAP_SHARED,
        staging_fd,
        0);
    if (mapping == MAP_FAILED) {
      throw runtime_error("worker host staging mmap failed: " +
                          std::string(std::strerror(errno)));
    }
    staging = static_cast<uint8_t*>(mapping);
    programs = load_programs(bundle, profile);
    auto dense = allocate_dense(bundle.manifest);

    WorkerReply ready;
    ready.kind = WorkerReplyKind::ready;
    set_detail(
        ready,
        "graph_hash=" + bundle.manifest.graph_hash + "\ncontract_sha256=" +
            sha256_file(manifest_path) + "\nmodel_sha256=" +
            bundle.manifest.release_asset.model_sha256 + "\n" + identity);
    if (!send_reply(control_fd, ready)) {
      throw runtime_error("parent closed before worker readiness reply");
    }

    while (true) {
      WorkerCommand command;
      if (!receive_command(control_fd, command)) {
        break;
      }
      WorkerReply reply;
      reply.serial = command.serial;
      if (command.version != kWorkerProtocolVersion ||
          command.deadline_monotonic_nanoseconds <= 0) {
        reply.kind = WorkerReplyKind::failed;
        set_detail(reply, "invalid worker command or deadline");
        send_reply(control_fd, reply);
        break;
      }
      if (command.operation == WorkerOperation::shutdown) {
        const uint64_t count = programs.size();
        programs.clear();
        reply.kind = WorkerReplyKind::stopped;
        reply.released_programs = count;
        send_reply(control_fd, reply);
        break;
      }
      if (command.operation != WorkerOperation::execute) {
        reply.kind = WorkerReplyKind::failed;
        set_detail(reply, "unknown worker operation");
        send_reply(control_fd, reply);
        break;
      }

      bool submitted = false;
      try {
        const auto deadline = Clock::time_point(
            std::chrono::nanoseconds(command.deadline_monotonic_nanoseconds));
        read_inputs(bundle.manifest, staging, dense);
        for (auto& program : programs) {
          execute_program(program, dense, deadline, submitted);
        }
        write_outputs(bundle.manifest, staging, dense);
        reply.kind = WorkerReplyKind::executed;
        set_detail(reply, "ordered dispatch completed");
        if (!send_reply(control_fd, reply)) {
          break;
        }
      } catch (const UncertainCompletion& error) {
        reply.kind = WorkerReplyKind::uncertain;
        set_detail(reply, error_reason(error.what()));
        send_reply(control_fd, reply);
        break;
      } catch (const std::exception& error) {
        reply.kind = submitted
            ? WorkerReplyKind::uncertain
            : WorkerReplyKind::failed;
        set_detail(reply, error_reason(error.what()));
        send_reply(control_fd, reply);
        break;
      }
    }
    programs.clear();
    ::munmap(staging, expected_staging_size);
    return 0;
  } catch (const std::exception& error) {
    WorkerReply reply;
    reply.kind = WorkerReplyKind::failed;
    set_detail(reply, error_reason(error.what()));
    send_reply(control_fd, reply);
    programs.clear();
    if (staging != nullptr) {
      ::munmap(staging, expected_staging_size);
    }
    return 1;
  }
}

} // namespace mlx::core::omarchy::ane::detail
