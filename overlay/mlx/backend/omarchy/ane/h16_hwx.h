// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

#pragma once

// Host side of DRM_IOCTL_ANE_H16_LOAD (driver ABI 3).
//
// H16_LOAD takes the compiled HWX together with four small sections that
// the client derives from it: generic (the buffer table), operation (the
// BAR slot bindings), procedure and text properties. This header reads an
// H16G or H17 HWX and derives those four sections. It never opens a
// device; the bundle validator and the lane-3 device share it, so a bundle
// that validates is a bundle the device can load.
//
// The byte layouts are written out in h16_hwx.cpp next to the code that
// emits them. The output is checked byte for byte against the add-128
// fixture of the kernel tree (aurora-silicon/linux#1477) and, through
// h16_sections_diff, against the kernel tree's reference client on
// thousands of generated variants.

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "mlx/api.h"

namespace mlx::core::omarchy::ane {

// CPU subtype word of the HWX header for the two supported targets.
constexpr uint32_t kAneHwxSubtypeH16G = 7;
constexpr uint32_t kAneHwxSubtypeH17 = 9;

// Number of host-built sections and their file names, in the kind order
// of DRM_ANE_H16_SECTION_GENERIC .. DRM_ANE_H16_SECTION_TEXT_PROPERTIES.
constexpr size_t kAneH16SectionCount = 4;
constexpr std::array<const char*, kAneH16SectionCount> kAneH16SectionNames{
    "generic.bin", "operation.bin", "procedure.bin", "text-properties.bin"};

// Raised for an HWX the builder will not derive sections for. `reason` is
// the short refusal text shared with the kernel tree's reference client,
// so a refusal reads the same from either tool.
struct AneH16HwxError : std::runtime_error {
  explicit AneH16HwxError(const std::string& why)
      : std::runtime_error("[omarchy-ane] h16 hwx: " + why + "."),
        reason(why) {}
  std::string reason;
};

// A tensor surface: one section of a __FVMLIB segment. Surfaces keep the
// order of the HWX segments, which is the order H16_LOAD reports them in
// and H16_EXEC expects its buffers in.
struct AneH16Surface {
  uint64_t vm{0};         // section VM address (the BAR base that names it)
  uint64_t allocation{0}; // vmsize of the enclosing segment
  uint32_t direction{0};  // segment protection: 1 input, 2 output
  std::optional<std::string> name;
  uint64_t span{0};       // tensor element span: count * batch
  uint32_t id{0};         // buffer ID in the generic section
};

// A section placed in VM: address, size and file offset.
struct AneH16Region {
  uint64_t vm{0};
  uint64_t size{0};
  uint32_t offset{0};
};

// What the builder reads out of the HWX.
struct AneH16Parsed {
  uint32_t subtype{0};
  std::vector<AneH16Surface> surfaces;
  AneH16Region text;                   // __TEXT,__text: the task stream
  std::optional<uint64_t> constants;   // __TEXT,__const VM address
  std::vector<AneH16Region> kernels;   // __KERN_0,__kern_0 .. in order
  std::optional<AneH16Region> scratch; // __DATA,__bss (offset not used)
  size_t descriptor{0};                // file offset of the program descriptor
};

// The four sections and the bookkeeping the device needs next to them.
struct AneH16Sections {
  std::array<std::vector<uint8_t>, kAneH16SectionCount> sections;
  std::vector<AneH16Surface> surfaces;              // HWX order, IDs set
  std::optional<uint32_t> scratch_id;
  uint64_t scratch_bytes{0};                        // __bss size, 0 if none
  std::vector<uint32_t> kernel_ids;                 // one per kernel section
                                                    // when they are buffers
  std::vector<std::pair<uint32_t, uint32_t>> refs;  // (BAR slot, tag)
  uint32_t tasks{0};
  uint32_t subtype{0};
};

// Reads the load commands of an HWX. Throws AneH16HwxError.
MLX_API AneH16Parsed h16_parse_hwx(const uint8_t* hwx, size_t size);

// Walks a task stream that must open with `prologue`; returns the number
// of tasks and the BAR slots the task records load from, ascending.
// Throws AneH16HwxError.
MLX_API std::pair<uint32_t, std::vector<uint32_t>> h16_walk_tasks(
    const uint8_t* stream,
    size_t size,
    uint32_t prologue);

// Derives the four H16_LOAD sections of one HWX. Throws AneH16HwxError.
MLX_API AneH16Sections h16_build_sections(const uint8_t* hwx, size_t size);

// Read-only private mapping of a whole file. A model's HWX can be hundreds
// of megabytes, so it is mapped rather than read into memory.
class MLX_API AneMappedFile {
 public:
  explicit AneMappedFile(const std::string& path);
  ~AneMappedFile();
  AneMappedFile(const AneMappedFile&) = delete;
  AneMappedFile& operator=(const AneMappedFile&) = delete;
  const uint8_t* data() const {
    return data_;
  }
  size_t size() const {
    return size_;
  }

 private:
  const uint8_t* data_{nullptr};
  size_t size_{0};
};

} // namespace mlx::core::omarchy::ane
