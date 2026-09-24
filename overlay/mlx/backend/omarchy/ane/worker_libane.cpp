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
#include <string>
#include <unordered_map>

extern "C" {
#include "ane.h"
}

namespace mlx::core::omarchy::ane {

namespace {

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

  void load(const AneValidatedProgram& program) override {
    if (networks_.count(program.manifest_index)) {
      throw AneDeviceError("program already loaded");
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
    (void)channel;
    if (binding.raw) {
      // Selector-addressed surface: staged bytes go in verbatim, zero
      // padding beyond them, exactly like libane's ane_send.
      std::vector<uint8_t> tile(binding.allocation_bytes, 0);
      std::memcpy(tile.data(), data, size);
      api_.send(nn, tile.data(), channel);
      return;
    }
    std::vector<uint8_t> tile(binding.allocation_bytes, 0);
    ane_pack_rows(
        binding, data, tile.data(), binding.logical_bytes / ane_element_size(binding));
    api_.send(nn, tile.data(), channel);
  }

  void exec(size_t manifest_index) override {
    struct ane_nn* nn = network(manifest_index, "exec");
    if (api_.exec(nn) < 0) {
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
    (void)channel;
    if (binding.raw) {
      std::vector<uint8_t> tile(binding.allocation_bytes, 0);
      api_.read(nn, tile.data(), channel);
      std::memcpy(out, tile.data(), size);
      return;
    }
    std::vector<uint8_t> tile(binding.allocation_bytes, 0);
    api_.read(nn, tile.data(), channel);
    ane_unpack_rows(
        binding, tile.data(), out, binding.logical_bytes / ane_element_size(binding));
    (void)size;
  }

  void release() override {
    for (auto& entry : networks_) {
      api_.free(entry.second);
    }
    networks_.clear();
  }

 private:
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
};

} // namespace

std::unique_ptr<AneDevice> make_libane_device(const std::string& library) {
  return std::make_unique<LibaneDevice>(library);
}

} // namespace mlx::core::omarchy::ane

#endif // MLX_OMARCHY_ANE_DEVICE
