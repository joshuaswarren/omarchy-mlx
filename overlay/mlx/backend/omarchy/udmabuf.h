// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <string>

// udmabuf-backed expert slabs (the Linux Honeykrisp zero-copy path).
// VK_EXT_external_memory_host is not advertised by Honeykrisp, so a slab
// that must be BOTH host-writable (the loader preads into it) and GPU
// readable (the allocator aliases it into an array) goes through
// /dev/udmabuf instead: the slab is a sealed memfd whose pages are
// exported as a dma-buf; the allocator imports that dma-buf
// (VK_EXT_external_memory_dma_buf) and hands back the ORIGINAL pointer.
// Proven end to end on an M2 Max: memfd -> UDMABUF_CREATE ->
// VkImportMemoryFdInfoKHR -> compute shader, CPU reads through the
// original memfd mapping, 0 wrong words from 16 KiB to 512 MiB, with and
// without DMA_BUF_IOCTL_SYNC.
//
// Kernel contract (measured): size and offset must be multiples of the
// CPU page size (16 KiB there); /sys/module/udmabuf/parameters/
// size_limit_mb caps one buffer (default 64 MB, runtime-writable as
// root); udmabuf takes memfd/shmem pages only and requires
// F_SEAL_SHRINK; the kernel needs CONFIG_UDMABUF.

struct omarchy_udmabuf_slab {
  void* ptr;   // mmap base, page aligned (also the array data pointer)
  size_t size; // actual size, rounded up to the CPU page size
  int fd;      // dma-buf fd (owned; closed by _destroy)
};

// Kernel UAPI (linux/udmabuf.h), forward-declared so the test hook below
// does not force every includer to pull the Linux headers in.
struct udmabuf_create;

#define OMARCHY_UDMABUF_EXPORT __attribute__((visibility("default")))

extern "C" {

// Create a slab of at least |size| bytes: memfd_create + F_SEAL_SHRINK +
// mmap + UDMABUF_CREATE, registered so VulkanAllocator::make_buffer can
// import it. Returns 0 and fills |out| on success. On failure returns a
// positive reason code (out untouched) and, under
// MLX_OMARCHY_HOST_IMPORT_DEBUG=1, prints one stderr line with the
// errno/driver state:
//   1 MLX_OMARCHY_UDMABUF=0 (kill switch)
//   2 size == 0
//   3 memfd_create failed
//   4 ftruncate failed
//   5 F_SEAL_SHRINK failed
//   6 mmap failed
//   7 /dev/udmabuf missing or unopenable (kernel lacks CONFIG_UDMABUF)
//   8 size exceeds size_limit_mb
//   9 UDMABUF_CREATE ioctl failed
// Callers (sushi) treat any nonzero as "fall back to today's path".
OMARCHY_UDMABUF_EXPORT int omarchy_udmabuf_slab_create(
    size_t size,
    struct omarchy_udmabuf_slab* out);

// Destroy a slab from _create. Null-safe. Unregisters the region, closes
// the dma-buf fd and the memfd, unmaps. The array must be dead first
// (same caller-owns-the-bytes contract as the host-import path).
OMARCHY_UDMABUF_EXPORT void omarchy_udmabuf_slab_destroy(
    struct omarchy_udmabuf_slab* slab);

// DMA_BUF_IOCTL_SYNC START (start != 0) / END (start == 0), read-write,
// around a CPU fill batch. Coherency does not require it on Apple UMA
// (the M2 run was correct with and without); it is the cheap safe option
// for kernels where the dma-buf exporter needs CPU-access brackets.
// Returns 0, or the ioctl's errno as a negative value.
OMARCHY_UDMABUF_EXPORT int omarchy_udmabuf_slab_sync(
    const struct omarchy_udmabuf_slab* slab,
    int start);

} // extern "C"

namespace mlx::core::omarchy::udmabuf {

struct Region {
  void* ptr{nullptr};
  size_t size{0};
  int fd{-1};    // dma-buf
  int memfd{-1}; // backing memfd
};

// True when ptr lies inside a registered slab. Copies the region out so
// concurrent registrations cannot dangle.
bool find_region(const void* ptr, Region* out);
void register_region(Region r);
// Removes and returns the region; false when ptr is not a slab base.
bool unregister_region(const void* ptr, Region* out);

// MLX_OMARCHY_UDMABUF=0|off|false|no kills the whole path.
bool disabled();

// MLX_OMARCHY_HOST_IMPORT_DEBUG=1: one stderr line per process per
// distinct key, same shape as allocator.cpp's host-import lines.
void note(const std::string& key, const std::string& detail);

// Test-only seam: CT hosts have no /dev/udmabuf. When set, create()
// replaces the open(/dev/udmabuf) + UDMABUF_CREATE step with this hook,
// which receives the fully built request and returns a live fd (it will
// be dup'd/closed by the registry) or -1 with errno set. Null restores
// the real kernel path. Test code only; never set in production.
using CreateFn = int (*)(::udmabuf_create* req);
// Default visibility: the test binary links this across the libmlx
// boundary (the mlx build compiles with -fvisibility=hidden).
OMARCHY_UDMABUF_EXPORT void set_create_for_test(CreateFn fn);

} // namespace mlx::core::omarchy::udmabuf
