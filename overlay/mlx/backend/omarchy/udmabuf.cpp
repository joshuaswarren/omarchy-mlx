// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

#include "mlx/backend/omarchy/udmabuf.h"

#include <fcntl.h>
#include <linux/dma-buf.h>
#include <linux/udmabuf.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <unordered_set>
#include <vector>

#include "mlx/backend/omarchy/device.h"

namespace mlx::core::omarchy::udmabuf {

namespace {

// Live slabs. A serve process holds one slab set per layer plus the union
// slabs: dozens at most, so a linear scan beats a map.
std::mutex g_mu;
std::vector<Region> g_regions;

CreateFn g_create_override = nullptr;

// Same parse rules as host-import's kill switch (allocator.cpp).
bool env_off(const char* name) {
  const char* v = std::getenv(name);
  if (v == nullptr || *v == '\0') {
    return false;
  }
  std::string s = v;
  std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return s == "0" || s == "off" || s == "false" || s == "no";
}

size_t page_size() {
  long page = sysconf(_SC_PAGESIZE);
  return page > 0 ? static_cast<size_t>(page) : 4096;
}

// Per-buffer cap from the kernel; 0 means unknown (the sysfs file is
// missing on hosts where udmabuf is a module that never loaded), and
// unknown means the ioctl decides.
size_t size_limit_bytes() {
  std::FILE* f = std::fopen("/sys/module/udmabuf/parameters/size_limit_mb", "r");
  if (f == nullptr) {
    return 0;
  }
  unsigned long mb = 0;
  int got = std::fscanf(f, "%lu", &mb);
  std::fclose(f);
  if (got != 1) {
    return 0;
  }
  return mb * 1024 * 1024;
}

} // namespace

bool find_region(const void* ptr, Region* out) {
  std::unique_lock lk(g_mu);
  for (const auto& r : g_regions) {
    auto* p = static_cast<const char*>(ptr);
    auto* base = static_cast<const char*>(r.ptr);
    if (p >= base && p < base + r.size) {
      *out = r;
      return true;
    }
  }
  return false;
}

void register_region(Region r) {
  std::unique_lock lk(g_mu);
  g_regions.push_back(r);
}

bool unregister_region(const void* ptr, Region* out) {
  std::unique_lock lk(g_mu);
  for (auto it = g_regions.begin(); it != g_regions.end(); ++it) {
    if (it->ptr == ptr) {
      *out = *it;
      g_regions.erase(it);
      return true;
    }
  }
  return false;
}

bool disabled() {
  // Re-read every call (same as allocator's host_import_disabled): the
  // env-kill test case flips the variable mid-process.
  return env_off("MLX_OMARCHY_UDMABUF");
}

// MLX_OMARCHY_HOST_IMPORT_DEBUG=1: one stderr line per process per
// distinct key, same shape as allocator.cpp's host-import lines, so a
// device receipt greps one prefix for both import paths.
void note(const std::string& key, const std::string& detail) {
  static const bool enabled = env_flag("MLX_OMARCHY_HOST_IMPORT_DEBUG");
  if (!enabled) {
    return;
  }
  static std::mutex log_mu;
  static std::unordered_set<std::string> logged;
  std::unique_lock lk(log_mu);
  if (!logged.insert(key).second) {
    return;
  }
  std::fprintf(stderr, "[omarchy] udmabuf: %s%s\n", key.c_str(), detail.c_str());
  std::fflush(stderr);
}

void set_create_for_test(CreateFn fn) {
  g_create_override = fn;
}

namespace {

// Shared tail of every decline: everything built so far goes away.
int fail(
    void* map,
    size_t rounded,
    int memfd,
    int extra_fd,
    const char* reason,
    const std::string& detail,
    int code) {
  note(
      std::string("decline=") + reason + " ",
      detail + " rounded=" + std::to_string(rounded));
  if (extra_fd >= 0) {
    close(extra_fd);
  }
  if (map != nullptr) {
    munmap(map, rounded);
  }
  if (memfd >= 0) {
    close(memfd);
  }
  return code;
}

} // namespace

} // namespace mlx::core::omarchy::udmabuf

extern "C" {

int omarchy_udmabuf_slab_create(size_t size, struct omarchy_udmabuf_slab* out) {
  namespace u = mlx::core::omarchy::udmabuf;
  const size_t page = u::page_size();
  if (u::disabled()) {
    u::note("decline=env-disabled ", "MLX_OMARCHY_UDMABUF=0");
    return 1;
  }
  if (size == 0) {
    u::note("decline=size-zero ", "");
    return 2;
  }
  // The kernel rejects any size or offset that is not a multiple of the
  // CPU page size (16 KiB on the lab Macs), so round the slab up to one.
  // The loader preads only the requested bytes; the tail is never read.
  const size_t rounded = ((size + page - 1) / page) * page;

  int memfd = memfd_create("omarchy-udmabuf-slab", MFD_ALLOW_SEALING | MFD_CLOEXEC);
  if (memfd < 0) {
    return u::fail(
        nullptr, rounded, -1, -1, "memfd-failed",
        "errno=" + std::to_string(errno), 3);
  }
  if (ftruncate(memfd, static_cast<off_t>(rounded)) != 0) {
    return u::fail(
        nullptr, rounded, memfd, -1, "truncate-failed",
        "errno=" + std::to_string(errno), 4);
  }
  // udmabuf requires F_SEAL_SHRINK on the backing memfd.
  if (fcntl(memfd, F_ADD_SEALS, F_SEAL_SHRINK) != 0) {
    return u::fail(
        nullptr, rounded, memfd, -1, "seal-failed",
        "errno=" + std::to_string(errno), 5);
  }
  void* map = mmap(nullptr, rounded, PROT_READ | PROT_WRITE, MAP_SHARED, memfd, 0);
  if (map == MAP_FAILED) {
    return u::fail(
        nullptr, rounded, memfd, -1, "mmap-failed",
        "errno=" + std::to_string(errno), 6);
  }

  // The kernel step: export the memfd pages as a dma-buf. Test builds
  // replace this whole step with a hook because the CT has no
  // /dev/udmabuf.
  struct udmabuf_create req {
    .memfd = static_cast<std::uint32_t>(memfd),
    .flags = UDMABUF_FLAGS_CLOEXEC,
    .offset = 0,
    .size = static_cast<std::uint64_t>(rounded),
  };
  int fd = -1;
  if (u::g_create_override != nullptr) {
    errno = 0;
    fd = u::g_create_override(&req);
    if (fd < 0) {
      int e = errno != 0 ? errno : EINVAL;
      return u::fail(
          map, rounded, memfd, -1, "ioctl-failed", "errno=" + std::to_string(e),
          9);
    }
  } else {
    const size_t limit = u::size_limit_bytes();
    if (limit != 0 && rounded > limit) {
      return u::fail(
          map, rounded, memfd, -1, "size-limit",
          "limit_mb=" + std::to_string(limit / (1024 * 1024)), 8);
    }
    int dev = open("/dev/udmabuf", O_RDWR | O_CLOEXEC);
    if (dev < 0) {
      return u::fail(
          map, rounded, memfd, -1, "device-missing",
          "errno=" + std::to_string(errno) + " (kernel needs CONFIG_UDMABUF)",
          7);
    }
    fd = ioctl(dev, UDMABUF_CREATE, &req);
    if (fd < 0) {
      int e = errno;
      close(dev);
      return u::fail(
          map, rounded, memfd, -1, "ioctl-failed", "errno=" + std::to_string(e),
          9);
    }
    close(dev);
  }

  u::register_region(
      u::Region{.ptr = map, .size = rounded, .fd = fd, .memfd = memfd});
  u::note(
      "created ", "rounded=" + std::to_string(rounded) + " page=" +
          std::to_string(page) + " fd=" + std::to_string(fd));
  out->ptr = map;
  out->size = rounded;
  out->fd = fd;
  return 0;
}

void omarchy_udmabuf_slab_destroy(struct omarchy_udmabuf_slab* slab) {
  namespace u = mlx::core::omarchy::udmabuf;
  if (slab == nullptr || slab->ptr == nullptr) {
    return;
  }
  u::Region r{};
  if (!u::unregister_region(slab->ptr, &r)) {
    // Unknown slab: still release what the caller holds.
    r = u::Region{.ptr = slab->ptr, .size = slab->size, .fd = slab->fd, .memfd = -1};
  }
  if (r.fd >= 0) {
    close(r.fd);
  }
  munmap(r.ptr, r.size);
  if (r.memfd >= 0) {
    close(r.memfd);
  }
  slab->ptr = nullptr;
  slab->size = 0;
  slab->fd = -1;
}

int omarchy_udmabuf_slab_sync(const struct omarchy_udmabuf_slab* slab, int start) {
  if (slab == nullptr || slab->fd < 0) {
    return -EINVAL;
  }
  struct dma_buf_sync s {
    .flags = (start != 0 ? DMA_BUF_SYNC_START : DMA_BUF_SYNC_END) |
        DMA_BUF_SYNC_RW,
  };
  if (ioctl(slab->fd, DMA_BUF_IOCTL_SYNC, &s) != 0) {
    return -errno;
  }
  return 0;
}

} // extern "C"
