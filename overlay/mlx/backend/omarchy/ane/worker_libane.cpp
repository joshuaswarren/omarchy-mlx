// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

// libane-backed device for the bounded ANE worker. Compiled only when
// the build is configured with MLX_OMARCHY_ANE_DEVICE=ON and points at
// an omarchy-ane checkout. Host builds compile nothing from this file:
// the mlx library never links libane, the symbols are dlopen'd from the
// library path the caller names, and no inference-server surface exists
// (plan section 25).

#ifdef MLX_OMARCHY_ANE_DEVICE

#include "mlx/backend/omarchy/ane/tile_layout.h"
#include "mlx/backend/omarchy/ane/worker.h"

#include <dlfcn.h>

#include <cstring>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

extern "C" {
#include "ane.h"
}

namespace mlx::core::omarchy::ane {

namespace {

// ABI-2 port table entry (omarchy-ane libane/ane_m2.h): one runtime
// buffer per io surface. dir 0 = input, 1 = output, 2 = scratch; the
// scratch buffer_id is fixed at 0x40 by the library. Copied here because
// the worker dlopens libane and the shipped header needs kernel types
// the overlay does not vendor.
struct ane_m2_port_spec {
  const char* name;
  uint32_t dir;
  uint32_t buffer_id;
  uint32_t bar_slot;
  uint64_t tile_bytes;
};

struct LibaneApi {
  struct ane_nn* (*init)(const char* path, int dev_id);
  // Tile-shift-aware load (omarchy-ane agent/encoder-whole-program and
  // later): whole-program containers denominate tiles[] in 512-B units.
  // Optional: absent, every load uses the library's default shift.
  struct ane_nn* (*init_shift)(const char* path, int dev_id, uint32_t shift);
  void (*free)(struct ane_nn* nn);
  int (*exec)(struct ane_nn* nn);
  uint64_t (*src_size)(struct ane_nn* nn, uint32_t idx);
  uint64_t (*dst_size)(struct ane_nn* nn, uint32_t idx);
  void (*send)(struct ane_nn* nn, void* from, uint32_t idx);
  void (*read)(struct ane_nn* nn, void* to, uint32_t idx);
  // ABI-2 (T6021) port-table entry points. Optional: absent on builds
  // without the M2 backend; bundles that need them fail at load with a
  // clear error instead of a dlsym crash.
  struct ane_nn* (*m2_init_ports)(
      const char* path,
      const struct ane_m2_port_spec* ports,
      uint32_t port_count,
      int dev_id);
  int (*m2_exec)(struct ane_nn* nn);
  int (*m2_send)(struct ane_nn* nn, const void* from, uint32_t idx);
  int (*m2_read)(struct ane_nn* nn, void* to, uint32_t idx);
  bool m2_ok() const {
    return m2_init_ports && m2_exec && m2_send && m2_read;
  }
  bool ok() const {
    return init && free && exec && src_size && dst_size && send && read;
  }
};

LibaneApi load_api(void* handle) {
  LibaneApi api{};
  if (!handle) return api;
  api.init = reinterpret_cast<decltype(api.init)>(dlsym(handle, "__ane_init"));
  api.init_shift = reinterpret_cast<decltype(api.init_shift)>(
      dlsym(handle, "__ane_init_shift"));
  api.free = reinterpret_cast<decltype(api.free)>(dlsym(handle, "__ane_free"));
  api.exec = reinterpret_cast<decltype(api.exec)>(dlsym(handle, "ane_exec"));
  api.src_size = reinterpret_cast<decltype(api.src_size)>(
      dlsym(handle, "__ane_src_size"));
  api.dst_size = reinterpret_cast<decltype(api.dst_size)>(
      dlsym(handle, "__ane_dst_size"));
  api.send = reinterpret_cast<decltype(api.send)>(dlsym(handle, "__ane_send"));
  api.read = reinterpret_cast<decltype(api.read)>(dlsym(handle, "__ane_read"));
  api.m2_init_ports = reinterpret_cast<decltype(api.m2_init_ports)>(
      dlsym(handle, "ane_m2_init_ports"));
  api.m2_exec = reinterpret_cast<decltype(api.m2_exec)>(
      dlsym(handle, "ane_m2_exec"));
  api.m2_send = reinterpret_cast<decltype(api.m2_send)>(
      dlsym(handle, "ane_m2_send"));
  api.m2_read = reinterpret_cast<decltype(api.m2_read)>(
      dlsym(handle, "ane_m2_read"));
  return api;
}

class LibaneDevice : public AneDevice {
 public:
  explicit LibaneDevice(std::string library)
      : library_(std::move(library)) {
    handle_ = ::dlopen(library_.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!handle_) {
      const char* reason = ::dlerror();
      throw AneDeviceError(
          "cannot load libane from " + library_ + ": " +
          (reason ? reason : "unknown dlopen failure"));
    }
    api_ = load_api(handle_);
    if (!api_.ok()) {
      ::dlclose(handle_);
      handle_ = nullptr;
      throw AneDeviceError("libane is missing required symbols");
    }
  }

  ~LibaneDevice() override {
    release();
    if (handle_) {
      ::dlclose(handle_);
    }
  }

  std::string describe() const override {
    return "libane(" + library_ + ")";
  }

  void load(
      const AneValidatedProgram& program,
      const AneProgram& manifest_program) override {
    if (networks_.count(program.manifest_index)) {
      throw AneDeviceError("program already loaded");
    }
    if (!manifest_program.aux_outputs.empty()) {
      load_ports(program, manifest_program);
      return;
    }
    struct ane_nn* nn = api_.init_shift
        ? api_.init_shift(program.anec.c_str(), 0, program.tile_shift)
        : api_.init(program.anec.c_str(), 0);
    if (!nn) {
      throw AneDeviceError(
          "ane_init failed for " + program.anec.string());
    }
    networks_.emplace(program.manifest_index, nn);
  }

  void send(
      size_t manifest_index,
      uint32_t channel,
      const AneProgramBinding& binding,
      const uint8_t* data,
      size_t size) override {
    struct ane_nn* nn = network(manifest_index, "send");
    // libane addresses sources by their sequential position among the
    // anec's inputs; the worker passes the manifest binding whose
    // channel order matches (ane.h: ane_send(nn, input, 0), (nn, input, 1)).
    auto& tile = send_buffers_[channel];
    if (tile.size() < binding.allocation_bytes) {
      tile.resize(binding.allocation_bytes, 0);
    }
    const bool ports_mode = ports_networks_.count(manifest_index);
    const auto send_fn = [&](void* staged) {
      const int rc = ports_mode
          ? api_.m2_send(nn, staged, channel)
          : (api_.send(nn, staged, channel), 0);
      if (rc < 0) {
        throw AneDeviceError(
            "ane_m2_send failed on channel " + std::to_string(channel));
      }
    };
    if (binding.raw) {
      std::memcpy(tile.data(), data, size);
      send_fn(tile.data());
      return;
    }
    ane_pack_rows(
        binding, data, tile.data(), binding.logical_bytes / ane_element_size(binding));
    send_fn(tile.data());
  }

  void exec(size_t manifest_index) override {
    struct ane_nn* nn = network(manifest_index, "exec");
    const int rc = ports_networks_.count(manifest_index)
        ? api_.m2_exec(nn)
        : api_.exec(nn);
    if (rc < 0) {
      throw AneDeviceError(
          "ane_exec failed for program " + std::to_string(manifest_index));
    }
  }

  void read(
      size_t manifest_index,
      uint32_t channel,
      const AneProgramBinding& binding,
      uint8_t* out,
      size_t size) override {
    struct ane_nn* nn = network(manifest_index, "read");
    auto& tile = read_buffers_[channel];
    if (tile.size() < binding.allocation_bytes) {
      tile.resize(binding.allocation_bytes, 0);
    }
    const bool ports_mode = ports_networks_.count(manifest_index);
    const auto read_fn = [&](void* staged) {
      const int rc = ports_mode
          ? api_.m2_read(nn, staged, channel)
          : (api_.read(nn, staged, channel), 0);
      if (rc < 0) {
        throw AneDeviceError(
            "ane_m2_read failed on channel " + std::to_string(channel));
      }
    };
    if (binding.raw) {
      read_fn(tile.data());
      std::memcpy(out, tile.data(), size);
      return;
    }
    read_fn(tile.data());
    ane_unpack_rows(
        binding, tile.data(), out, binding.logical_bytes / ane_element_size(binding));
    (void)size;
  }

  void release() override {
    for (auto& entry : networks_) {
      api_.free(entry.second);
    }
    networks_.clear();
    ports_networks_.clear();
  }

 private:
  // ABI-2 port-table load: inputs, then outputs (positional then aux),
  // then the scratch port the library binds at BAR slot 3 / buffer 0x40.
  // Port order is also the send/read index order, matching the worker's
  // positional convention.
  void load_ports(
      const AneValidatedProgram& program,
      const AneProgram& manifest_program) {
    if (!api_.m2_ok()) {
      throw AneDeviceError(
          "bundle requires the ABI-2 port-table API (aux_outputs) but " +
          library_ + " does not export ane_m2_init_ports");
    }
    std::vector<ane_m2_port_spec> ports;
    ports.reserve(
        manifest_program.inputs.size() + manifest_program.outputs.size() +
        manifest_program.aux_outputs.size() + 1);
    const auto push = [](const AneProgramBinding& binding, uint32_t dir) {
      return ane_m2_port_spec{
          binding.tensor.c_str(),
          dir,
          static_cast<uint32_t>(binding.buffer_id),
          static_cast<uint32_t>(binding.bar_slot),
          binding.allocation_bytes};
    };
    for (const auto& binding : manifest_program.inputs) {
      ports.push_back(push(binding, 0));
    }
    for (const auto& binding : manifest_program.outputs) {
      ports.push_back(push(binding, 1));
    }
    for (const auto& binding : manifest_program.aux_outputs) {
      ports.push_back(push(binding, 1));
    }
    ports.push_back(ane_m2_port_spec{
        "scratch", 2, 0x40u, 3u, manifest_program.scratch_bytes});
    struct ane_nn* nn = api_.m2_init_ports(
        program.anec.c_str(), ports.data(),
        static_cast<uint32_t>(ports.size()), 0);
    if (!nn) {
      throw AneDeviceError(
          "ane_m2_init_ports failed for " + program.anec.string());
    }
    networks_.emplace(program.manifest_index, nn);
    ports_networks_.insert(program.manifest_index);
  }

  struct ane_nn* network(size_t manifest_index, const char* what) const {
    auto found = networks_.find(manifest_index);
    if (found == networks_.end()) {
      throw AneDeviceError(
          std::string(what) + ": program " +
          std::to_string(manifest_index) + " is not loaded");
    }
    return found->second;
  }


  std::string library_;
  void* handle_{nullptr};
  LibaneApi api_{};
  std::unordered_map<size_t, struct ane_nn*> networks_;
  std::set<size_t> ports_networks_;
  std::unordered_map<uint32_t, std::vector<uint8_t>> send_buffers_;
  std::unordered_map<uint32_t, std::vector<uint8_t>> read_buffers_;
};

} // namespace

std::unique_ptr<AneDevice> make_libane_device(const std::string& library) {
  return std::make_unique<LibaneDevice>(library);
}

} // namespace mlx::core::omarchy::ane

#endif // MLX_OMARCHY_ANE_DEVICE
