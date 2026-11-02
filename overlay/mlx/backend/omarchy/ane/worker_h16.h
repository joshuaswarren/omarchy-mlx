// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT
#pragma once

// Lane 3: the ABI version 3 device for the bounded ANE worker. It drives the
// in-kernel accel/ane H16 node (DRM_IOCTL_ANE_GET_CAPS, DRM_IOCTL_ANE_H16_LOAD,
// DRM_IOCTL_ANE_H16_EXEC) directly; no libane is loaded. Compiled only with
// MLX_OMARCHY_ANE_DEVICE, like the libane device.
//
// Kernel model this device follows (include/uapi/drm/ane_accel.h, ABI 3):
// one program per open file, any number of files with a loaded program at
// the same time, the program unloaded when its file closes. Every surface is
// copied by the kernel on each EXEC (inputs in, the one output out), so the
// device keeps one host buffer per surface and passes all of them, in the
// order H16_LOAD reported, to every EXEC.

#include "mlx/backend/omarchy/ane/worker.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace mlx::core::omarchy::ane {

// The syscall seam: every open, ioctl and close the lane-3 device performs
// goes through one of these. The system implementation calls the kernel;
// tests put a fake driver behind it.
struct AneH16Kernel {
  virtual ~AneH16Kernel() = default;
  // A file descriptor, or -errno.
  virtual int open_node(const std::string& path) = 0;
  // 0, or -errno.
  virtual int ioctl_node(int fd, unsigned long request, void* argument) = 0;
  virtual void close_node(int fd) = 0;
  // Candidate accel nodes in the order the reference client probes them
  // (sorted /dev/accel/accel* names).
  virtual std::vector<std::string> list_nodes() = 0;
};

std::shared_ptr<AneH16Kernel> make_h16_system_kernel();

struct AneH16Identity {
  std::string node;
  std::string driver;
  int version_major{0};
  int version_minor{0};
  int version_patch{0};
  uint32_t abi{0};
  uint32_t chip{0};
  uint32_t caps_size{0};
  // One line for receipts and the worker's startup output.
  std::string describe() const;
};

// Finds the node: `node` empty means the first /dev/accel/accel* whose
// GET_CAPS reports ABI 3 / H16; otherwise exactly that node, which must.
// Returns the identity and leaves `fd_out` open on the node (or closes it
// when fd_out is null). A node that exists but cannot be opened is a
// failure, not a missing lane (the reference client's rule).
AneH16Identity probe_h16_node(
    const std::string& node,
    AneH16Kernel& kernel,
    int* fd_out = nullptr);

// The ioctl failure text: the ioctl name, the program, errno name and what
// the ABI 3 driver means by it.
std::string h16_ioctl_error(
    const char* ioctl,
    const std::string& subject,
    int error);

std::unique_ptr<AneDevice> make_h16_device(const std::string& node);
std::unique_ptr<AneDevice> make_h16_device(
    const std::string& node,
    std::shared_ptr<AneH16Kernel> kernel);

} // namespace mlx::core::omarchy::ane
