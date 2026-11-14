// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

// Lane 3 device (see worker_h16.h). Compiled only when the build is
// configured with MLX_OMARCHY_ANE_DEVICE=ON, like worker_libane.cpp. It needs
// no omarchy-ane checkout: the ABI 3 uapi header is vendored under uapi/.

#ifdef MLX_OMARCHY_ANE_DEVICE

#include "mlx/backend/omarchy/ane/worker_h16.h"

#include "mlx/backend/omarchy/ane/h16_hwx.h"
#include "mlx/backend/omarchy/ane/tile_layout.h"
#include "mlx/backend/omarchy/ane/uapi/drm/ane_accel.h"

#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/utsname.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstring>
#include <filesystem>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace mlx::core::omarchy::ane {

namespace {

static_assert(DRM_ANE_H16_SECTIONS == kAneH16SectionCount);
static_assert(sizeof(drm_ane_h16_load) == 0x58);
static_assert(sizeof(drm_ane_h16_section) == 0x10);
static_assert(sizeof(drm_ane_h16_buffer) == 0x10);
static_assert(sizeof(drm_ane_h16_exec) == 0x10);

// What the ABI 3 driver means by each errno (drivers/accel/ane
// ane_h16_accel.c and ane_h16_program.c at the vendored uapi commit).
const char* h16_errno_meaning(int error) {
  switch (error) {
    case EINVAL:
      return "the driver refused an argument (flags, section table, surface "
             "count or buffer size), or this file holds no open program";
    case EFAULT:
      return "the driver could not read or write a user buffer";
    case ENOEXEC:
      return "the driver refused the HWX (validation failed or it was "
             "compiled for another engine generation)";
    case EBUSY:
      return "this file already holds a program, or the transport cannot "
             "keep a second program loaded";
    case EIO:
      return "the H16 session is quarantined; reboot before the next ANE "
             "submission";
    case ETIMEDOUT:
      return "the firmware did not report completion in time; the driver "
             "quarantined the session";
    case EPROTO:
      return "the firmware sent a malformed reply; the driver quarantined "
             "the session";
    case EREMOTEIO:
      return "the firmware reported a failed request";
    case ENOSPC:
      return "the program window or a firmware ID pool is exhausted "
             "(program mappings stay until reboot)";
    case ENOMEM:
      return "the kernel could not allocate memory";
    case ENOTTY:
      return "the node does not implement this ioctl (not an ABI 3 driver)";
    case ENODEV:
      return "the device is gone";
    default:
      return "unexpected driver error";
  }
}

std::string errno_name(int error) {
  const char* name = ::strerrorname_np(error);
  return name ? std::string(name) : "errno " + std::to_string(error);
}

std::string kernel_release() {
  utsname system{};
  return ::uname(&system) == 0 ? std::string(system.release) : "unknown";
}

class SystemKernel : public AneH16Kernel {
 public:
  int open_node(const std::string& path) override {
    const int fd = ::open(path.c_str(), O_RDWR | O_CLOEXEC);
    return fd >= 0 ? fd : -errno;
  }
  int ioctl_node(int fd, unsigned long request, void* argument) override {
    // drmIoctl's rule: a signal before the driver took its lock restarts
    // the call; nothing was submitted.
    for (;;) {
      if (::ioctl(fd, request, argument) == 0) {
        return 0;
      }
      if (errno != EINTR) {
        return -errno;
      }
    }
  }
  void close_node(int fd) override {
    ::close(fd);
  }
  std::vector<std::string> list_nodes() override {
    std::vector<std::string> nodes;
    std::error_code error;
    for (const auto& entry :
         std::filesystem::directory_iterator("/dev/accel", error)) {
      const std::string name = entry.path().filename().string();
      if (name.rfind("accel", 0) == 0) {
        nodes.push_back(entry.path().string());
      }
    }
    std::sort(nodes.begin(), nodes.end());
    return nodes;
  }
};

class H16Device : public AneDevice {
 public:
  H16Device(
      const std::string& node,
      std::shared_ptr<AneH16Kernel> kernel,
      AneH16Residency residency)
      : kernel_(std::move(kernel)), residency_(residency) {
    if (!kernel_) {
      throw AneDeviceError("ABI 3 device requires a kernel interface");
    }
    identity_ = probe_h16_node(node, *kernel_, &probe_fd_);
  }

  ~H16Device() override {
    release();
    if (probe_fd_ >= 0) {
      kernel_->close_node(probe_fd_);
    }
  }

  std::string describe() const override {
    return identity_.describe();
  }

  std::string status() const override {
    size_t resident = 0;
    for (const auto& entry : programs_) {
      resident += entry.second.fd >= 0 ? 1 : 0;
    }
    std::ostringstream line;
    line << "h16 residency=" << h16_residency_name(residency_)
         << " programs=" << programs_.size() << " resident=" << resident
         << " kernel_loads=" << kernel_loads_ << " evictions=" << evictions_
         << " limit="
         << (limit_ ? std::to_string(limit_) : std::string("unbounded"));
    return line.str();
  }

  void load(const AneValidatedProgram& program) override {
    const std::string subject = "program " +
        std::to_string(program.manifest_index) + " (" +
        program.payload_path().filename().string() + ")";
    if (program.format != AneProgramFormat::hwx) {
      throw AneDeviceError(
          "the ABI 3 device cannot load ANEC " + subject +
          "; ANEC bundles run on libane (--libane)");
    }
    if (programs_.count(program.manifest_index)) {
      throw AneDeviceError(subject + " is already loaded");
    }
    const auto& contract = program.h16_surfaces;
    if (contract.size() < 2 || contract.size() > DRM_ANE_H16_SURFACES) {
      throw AneDeviceError(
          subject + " declares " + std::to_string(contract.size()) +
          " surfaces; ABI 3 programs have 2 to " +
          std::to_string(DRM_ANE_H16_SURFACES));
    }

    // The sealed HWX image the validator approved; the sections are built
    // again from exactly these bytes, so the pair handed to H16_LOAD always
    // describes one file. The mapping stays with the program: a program the
    // device unloads to make room (swap) is loaded again from these same
    // bytes, never from a path that could have changed.
    Program loaded;
    loaded.subject = subject;
    loaded.hwx = std::make_shared<AneMappedFile>(program.hwx.string());
    try {
      loaded.sections =
          h16_build_sections(loaded.hwx->data(), loaded.hwx->size());
    } catch (const AneH16HwxError& error) {
      throw AneDeviceError(subject + " HWX refused: " + error.reason);
    }
    if (loaded.sections.tasks != program.h16_tasks ||
        loaded.sections.surfaces.size() != contract.size()) {
      throw AneDeviceError(subject + " HWX differs from the validated bundle");
    }
    for (size_t i = 0; i < contract.size(); ++i) {
      if (loaded.sections.surfaces[i].allocation != contract[i].bytes ||
          loaded.sections.surfaces[i].direction != contract[i].direction) {
        throw AneDeviceError(subject + " HWX differs from the validated bundle");
      }
      loaded.directions.push_back(contract[i].direction);
      // H16_LOAD reports each surface's tensor span, which EXEC buffers
      // carry; the manifest binds the allocation, the span's upper bound.
      loaded.sizes.push_back(loaded.sections.surfaces[i].span);
      loaded.buffers.emplace_back(static_cast<size_t>(contract[i].bytes), 0);
    }

    // Load on the kernel now, even when the program may be unloaded again
    // before it runs: the driver's own validation (ENOEXEC and the like)
    // then fails the session open, not a submit inside its deadline.
    auto entry = programs_.emplace(program.manifest_index, std::move(loaded));
    try {
      make_resident(entry.first->second);
    } catch (...) {
      programs_.erase(entry.first);
      throw;
    }
  }

  void send(
      size_t manifest_index,
      uint32_t channel,
      const AneProgramBinding& binding,
      const uint8_t* data,
      size_t size) override {
    (void)channel; // the binding's surface index addresses H16 surfaces
    Program& program = find(manifest_index, "send");
    std::vector<uint8_t>& surface = surface_of(program, binding, 1, "send");
    if (binding.raw) {
      // libane's raw staging: the staged bytes go to the surface verbatim.
      std::memcpy(surface.data(), data, std::min(size, surface.size()));
      return;
    }
    ane_pack_rows(
        binding, data, surface.data(),
        binding.logical_bytes / ane_element_size(binding));
  }

  void exec(size_t manifest_index) override {
    Program& program = find(manifest_index, "exec");
    // The surfaces live in host buffers, so a program the device had to
    // unload comes back with its staged inputs intact.
    make_resident(program);
    std::array<drm_ane_h16_buffer, DRM_ANE_H16_SURFACES> buffers{};
    for (size_t i = 0; i < program.buffers.size(); ++i) {
      buffers[i].ptr = reinterpret_cast<uintptr_t>(program.buffers[i].data());
      // The staging buffer covers the allocation; the tensor lies in its
      // first span bytes, which is what the driver copies.
      buffers[i].size = program.sizes[i];
    }
    drm_ane_h16_exec request{};
    request.buffers_ptr = reinterpret_cast<uintptr_t>(buffers.data());
    request.count = static_cast<uint32_t>(program.buffers.size());
    const int result =
        kernel_->ioctl_node(program.fd, DRM_IOCTL_ANE_H16_EXEC, &request);
    if (result != 0) {
      throw AneDeviceError(h16_ioctl_error(
          "DRM_IOCTL_ANE_H16_EXEC", program.subject + " on " + identity_.node,
          -result));
    }
  }

  void read(
      size_t manifest_index,
      uint32_t channel,
      const AneProgramBinding& binding,
      uint8_t* out,
      size_t size) override {
    (void)channel;
    Program& program = find(manifest_index, "read");
    const std::vector<uint8_t>& surface = surface_of(program, binding, 2, "read");
    if (binding.raw) {
      std::memcpy(out, surface.data(), std::min(size, surface.size()));
      return;
    }
    ane_unpack_rows(
        binding, surface.data(), out,
        binding.logical_bytes / ane_element_size(binding));
  }

  // Closing a program's file makes the driver send TERMINATE and UNLOAD.
  void release() override {
    for (auto& entry : programs_) {
      unload(entry.second);
    }
    programs_.clear();
  }

 private:
  struct Program {
    int fd{-1}; // -1: not on the kernel right now (never loaded or evicted)
    std::string subject;
    std::shared_ptr<AneMappedFile> hwx;
    AneH16Sections sections;
    std::vector<uint32_t> directions;
    std::vector<uint64_t> sizes; // tensor spans, what H16_EXEC copies
    std::vector<std::vector<uint8_t>> buffers; // H16_LOAD surface order
    uint64_t last_used{0};
  };

  size_t resident_count() const {
    size_t count = 0;
    for (const auto& entry : programs_) {
      count += entry.second.fd >= 0 ? 1 : 0;
    }
    return count;
  }

  void unload(Program& program) {
    if (program.fd >= 0) {
      kernel_->close_node(program.fd);
      program.fd = -1;
    }
  }

  // Unloads the least recently used resident program other than `keep`.
  bool evict_one(const Program* keep) {
    Program* victim = nullptr;
    for (auto& entry : programs_) {
      Program& candidate = entry.second;
      if (&candidate == keep || candidate.fd < 0) {
        continue;
      }
      if (!victim || candidate.last_used < victim->last_used) {
        victim = &candidate;
      }
    }
    if (!victim) {
      return false;
    }
    unload(*victim);
    ++evictions_;
    return true;
  }

  // One open + H16_LOAD for `program`. Returns 0 (program resident), or the
  // negative errno of the ioctl, with the file closed again. Anything that
  // is not a driver refusal (open failure, a report that differs from the
  // manifest) throws and leaves no file behind.
  int load_once(Program& program) {
    // One program per open file: a fresh file for every load.
    const int fd = kernel_->open_node(identity_.node);
    if (fd < 0) {
      throw AneDeviceError(
          "open(" + identity_.node + ") failed for " + program.subject + ": " +
          errno_name(-fd) + " (" + std::strerror(-fd) + ")");
    }
    std::array<drm_ane_h16_section, DRM_ANE_H16_SECTIONS> table{};
    for (size_t kind = 0; kind < table.size(); ++kind) {
      table[kind].kind = static_cast<uint32_t>(DRM_ANE_H16_SECTION_GENERIC + kind);
      table[kind].size =
          static_cast<uint32_t>(program.sections.sections[kind].size());
      table[kind].data_ptr =
          reinterpret_cast<uintptr_t>(program.sections.sections[kind].data());
    }
    drm_ane_h16_load request{};
    request.hwx_ptr = reinterpret_cast<uintptr_t>(program.hwx->data());
    request.hwx_size = program.hwx->size();
    request.sections_ptr = reinterpret_cast<uintptr_t>(table.data());
    request.section_count = DRM_ANE_H16_SECTIONS;
    const int result =
        kernel_->ioctl_node(fd, DRM_IOCTL_ANE_H16_LOAD, &request);
    if (result != 0) {
      kernel_->close_node(fd);
      return result;
    }
    ++kernel_loads_;

    // The kernel's view of the program must equal the manifest's bindings.
    // Closing the file unloads the program again.
    auto refuse = [&](const std::string& reason) {
      kernel_->close_node(fd);
      throw AneDeviceError(
          "DRM_IOCTL_ANE_H16_LOAD for " + program.subject + " reported " +
          reason);
    };
    if (request.surface_count != program.sizes.size()) {
      refuse(
          std::to_string(request.surface_count) +
          " surfaces; the manifest binds " +
          std::to_string(program.sizes.size()));
    }
    for (size_t i = 0; i < program.sizes.size(); ++i) {
      if (request.surface_size[i] != program.sizes[i] ||
          request.surface_direction[i] != program.directions[i]) {
        refuse(
            "surface " + std::to_string(i) + " with " +
            std::to_string(request.surface_size[i]) + " bytes, direction " +
            std::to_string(request.surface_direction[i]) +
            "; the HWX spans " + std::to_string(program.sizes[i]) +
            " bytes, direction " + std::to_string(program.directions[i]));
      }
    }
    program.fd = fd;
    return 0;
  }

  // Residency policy (see AneH16Residency). Brings `program` onto the
  // kernel, unloading other programs where the policy and the driver need
  // it. On a throw the program has no file and nothing else changed except
  // programs already unloaded to make room (they reload on their next use).
  void make_resident(Program& program) {
    if (program.fd >= 0) {
      program.last_used = ++clock_;
      return;
    }
    if (residency_ == AneH16Residency::Swap) {
      while (evict_one(&program)) {
      }
    } else if (limit_ != 0) {
      while (resident_count() >= limit_ && evict_one(&program)) {
      }
    }
    for (;;) {
      const int result = load_once(program);
      if (result == 0) {
        break;
      }
      if (result != -EBUSY) {
        throw AneDeviceError(h16_ioctl_error(
            "DRM_IOCTL_ANE_H16_LOAD",
            program.subject + " on " + identity_.node, -result));
      }
      const size_t others = resident_count();
      if (residency_ == AneH16Residency::Concurrent) {
        throw AneDeviceError(
            h16_ioctl_error(
                "DRM_IOCTL_ANE_H16_LOAD",
                program.subject + " on " + identity_.node, EBUSY) +
            "; " + std::to_string(others) +
            " program(s) of this device are loaded and the kernel does not "
            "keep another one resident (concurrent residency needs the "
            "switch-safe driver patch); run with --h16-residency swap or "
            "auto to load programs on demand");
      }
      if (others == 0) {
        throw AneDeviceError(
            h16_ioctl_error(
                "DRM_IOCTL_ANE_H16_LOAD",
                program.subject + " on " + identity_.node, EBUSY) +
            "; this device holds no program, so another client or an earlier "
            "worker still owns the node's program slot");
      }
      // The driver's capacity is the number of programs that were resident
      // when it said no. Remember it so later loads unload first instead of
      // asking the kernel again, and unload the least recently used.
      limit_ = limit_ == 0 ? others : std::min(limit_, others);
      evict_one(&program);
    }
    program.last_used = ++clock_;
  }

  Program& find(size_t manifest_index, const char* what) {
    auto found = programs_.find(manifest_index);
    if (found == programs_.end()) {
      throw AneDeviceError(
          std::string(what) + ": program " + std::to_string(manifest_index) +
          " is not loaded");
    }
    return found->second;
  }

  std::vector<uint8_t>& surface_of(
      Program& program,
      const AneProgramBinding& binding,
      uint32_t direction,
      const char* what) {
    if (binding.channel >= program.buffers.size() ||
        program.directions[binding.channel] != direction) {
      throw AneDeviceError(
          std::string(what) + ": " + program.subject + " has no " +
          (direction == 1 ? "input" : "output") + " surface " +
          std::to_string(binding.channel) + " for tensor '" + binding.tensor +
          "'");
    }
    return program.buffers[binding.channel];
  }

  std::shared_ptr<AneH16Kernel> kernel_;
  AneH16Residency residency_;
  AneH16Identity identity_;
  int probe_fd_{-1};
  std::unordered_map<size_t, Program> programs_;
  uint64_t clock_{0};
  size_t limit_{0}; // 0 = no refusal seen yet
  uint64_t kernel_loads_{0};
  uint64_t evictions_{0};
};

} // namespace

std::shared_ptr<AneH16Kernel> make_h16_system_kernel() {
  return std::make_shared<SystemKernel>();
}

std::string AneH16Identity::describe() const {
  std::ostringstream line;
  line << "lane=3 node=" << node << " driver=" << driver
       << " drm_version=" << version_major << '.' << version_minor << '.'
       << version_patch << " abi=" << abi << " chip=h" << chip
       << " caps_size=" << caps_size << " kernel=" << kernel_release();
  return line.str();
}

std::string h16_ioctl_error(
    const char* ioctl,
    const std::string& subject,
    int error) {
  return std::string(ioctl) + " failed for " + subject + ": " +
      errno_name(error) + " (" + std::strerror(error) + "): " +
      h16_errno_meaning(error);
}

AneH16Identity probe_h16_node(
    const std::string& node,
    AneH16Kernel& kernel,
    int* fd_out) {
  const bool explicit_node = !node.empty();
  const std::vector<std::string> candidates =
      explicit_node ? std::vector<std::string>{node} : kernel.list_nodes();
  std::vector<std::string> open_errors;
  for (const auto& path : candidates) {
    const int fd = kernel.open_node(path);
    if (fd < 0) {
      open_errors.push_back(
          "open(" + path + "): " + errno_name(-fd) + " (" +
          std::strerror(-fd) + ")");
      continue;
    }
    drm_ane_get_caps caps{};
    int result = kernel.ioctl_node(fd, DRM_IOCTL_ANE_GET_CAPS, &caps);
    if (result != 0) {
      kernel.close_node(fd);
      if (explicit_node) {
        throw AneDeviceError(h16_ioctl_error(
            "DRM_IOCTL_ANE_GET_CAPS", path, -result));
      }
      continue; // another accel driver; not ours
    }
    if (caps.abi_version != DRM_ANE_ABI_V3 || caps.chip_family != DRM_ANE_CHIP_H16) {
      kernel.close_node(fd);
      if (explicit_node) {
        throw AneDeviceError(
            path + " reports ANE ABI " + std::to_string(caps.abi_version) +
            " chip " + std::to_string(caps.chip_family) +
            "; lane 3 requires ABI 3 on H16 (chip 16)");
      }
      continue;
    }
    std::array<char, 64> name{};
    drm_version version{};
    version.name = name.data();
    version.name_len = name.size() - 1;
    result = kernel.ioctl_node(fd, DRM_IOCTL_VERSION, &version);
    if (result != 0) {
      kernel.close_node(fd);
      throw AneDeviceError(h16_ioctl_error("DRM_IOCTL_VERSION", path, -result));
    }
    AneH16Identity identity;
    identity.node = path;
    identity.driver = std::string(
        name.data(), std::min<size_t>(version.name_len, name.size() - 1));
    identity.version_major = version.version_major;
    identity.version_minor = version.version_minor;
    identity.version_patch = version.version_patchlevel;
    identity.abi = caps.abi_version;
    identity.chip = caps.chip_family;
    identity.caps_size = caps.size;
    if (identity.driver != "ane" || identity.version_major != DRM_ANE_ABI_V3) {
      kernel.close_node(fd);
      throw AneDeviceError(
          path + " is DRM driver '" + identity.driver + "' major " +
          std::to_string(identity.version_major) +
          "; lane 3 requires driver 'ane' major 3");
    }
    if (fd_out) {
      *fd_out = fd;
    } else {
      kernel.close_node(fd);
    }
    return identity;
  }
  if (!open_errors.empty()) {
    std::string joined;
    for (const auto& error : open_errors) {
      joined += (joined.empty() ? "" : "; ") + error;
    }
    throw AneDeviceError("ABI 3 accel node cannot be opened: " + joined);
  }
  throw AneDeviceError(
      "no ABI 3 H16 accel node (the H16 program stage was not reached)");
}

const char* h16_residency_name(AneH16Residency residency) {
  switch (residency) {
    case AneH16Residency::Concurrent:
      return "concurrent";
    case AneH16Residency::Swap:
      return "swap";
    case AneH16Residency::Auto:
      break;
  }
  return "auto";
}

AneH16Residency parse_h16_residency(const std::string& text) {
  if (text == "auto") {
    return AneH16Residency::Auto;
  }
  if (text == "concurrent") {
    return AneH16Residency::Concurrent;
  }
  if (text == "swap") {
    return AneH16Residency::Swap;
  }
  throw std::invalid_argument(
      "--h16-residency expects auto, concurrent or swap, got '" + text + "'");
}

std::unique_ptr<AneDevice> make_h16_device(
    const std::string& node,
    std::shared_ptr<AneH16Kernel> kernel,
    AneH16Residency residency) {
  return std::make_unique<H16Device>(node, std::move(kernel), residency);
}

std::unique_ptr<AneDevice> make_h16_device(
    const std::string& node,
    AneH16Residency residency) {
  return make_h16_device(node, make_h16_system_kernel(), residency);
}

} // namespace mlx::core::omarchy::ane

#endif // MLX_OMARCHY_ANE_DEVICE
