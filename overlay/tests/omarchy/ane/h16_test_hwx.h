// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT
#pragma once

// Test-only synthetic H16G/H17 HWX containers and the add-128 fixture path.
//
// The builder writes the load commands the ABI 3 section builder reads:
// one __FVMLIB segment per surface, __TEXT,__text with a task stream, an
// optional __KERN_0,__kern_0 and __DATA,__bss, the kind-4 program descriptor
// (BAR bases at +0x10 * (slot + 1)), one kind-3 tensor descriptor and one
// 0x40 name command per surface. Every task references every BAR slot in
// `slots`. Nothing here claims to be a runnable program.

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace h16_test {

inline std::filesystem::path add128_dir() {
  return std::filesystem::path(__FILE__).parent_path() / "fixtures" /
      "h16-add128";
}

inline std::vector<uint8_t> read_all(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  return std::vector<uint8_t>(
      (std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
}

struct Surface {
  std::string name;
  uint32_t direction{1};
  uint64_t vm{0};
  uint64_t allocation{0x4000};
  uint32_t n{1};
  uint64_t batch{0x2000};
};

struct Spec {
  uint32_t subtype{7};
  std::vector<Surface> surfaces{
      {"a", 1, 0x30000000, 0x4000, 1, 0x2000},
      {"y", 2, 0x30010000, 0x4000, 1, 0x2000},
      {"b", 1, 0x30020000, 0x4000, 1, 0x2000}};
  std::optional<std::pair<uint64_t, uint64_t>> scratch; // vm, size
  std::optional<uint64_t> kernel_vm;
  uint32_t tasks{1};
  // (BAR slot, descriptor base); a zero base is a config-word slot.
  std::vector<std::pair<uint32_t, uint64_t>> slots{
      {6, 0x30000000}, {7, 0x30010000}, {8, 0x30020000}};
  bool descriptor{true};
};

inline void put32(std::vector<uint8_t>& b, size_t o, uint32_t v) {
  if (b.size() < o + 4) b.resize(o + 4, 0);
  for (int i = 0; i < 4; ++i) b[o + i] = uint8_t(v >> (8 * i));
}
inline void put64(std::vector<uint8_t>& b, size_t o, uint64_t v) {
  put32(b, o, uint32_t(v));
  put32(b, o + 4, uint32_t(v >> 32));
}
inline void put_name(std::vector<uint8_t>& b, size_t o, const std::string& s) {
  if (b.size() < o + s.size()) b.resize(o + s.size(), 0);
  std::memcpy(b.data() + o, s.data(), s.size());
}

inline std::vector<uint8_t> task_stream(const Spec& spec) {
  std::vector<uint8_t> stream(16, 0);
  put32(stream, 0, spec.subtype == 9 ? 0 : 1);
  for (uint32_t t = 0; t < spec.tasks; ++t) {
    const size_t pos = stream.size();
    const uint32_t words = 9 + 2 * static_cast<uint32_t>(spec.slots.size());
    stream.resize(pos + 4 * words, 0);
    put32(stream, pos, words << 16);
    size_t rec = pos + 4 * 9;
    for (const auto& slot : spec.slots) {
      // Header bit 29 names BAR slot (h >> 23) & 0x3f; register word 1 is
      // inside the first register range; one data word follows.
      put32(stream, rec, (1u << 29) | (slot.first << 23) | 1u);
      put32(stream, rec + 4, 0x1234);
      rec += 8;
    }
    stream.resize((stream.size() + 15) & ~size_t(15), 0);
  }
  return stream;
}

inline std::vector<uint8_t> build(const Spec& spec) {
  std::vector<uint8_t> b(32, 0);
  uint32_t commands = 0;
  auto segment = [&](const char* segname, uint64_t vm, uint64_t vmsize,
                     uint32_t prot, const char* sectname, uint64_t addr,
                     uint64_t size, uint32_t offset) {
    const size_t pos = b.size();
    b.resize(pos + 72 + 80, 0);
    put32(b, pos, 0x19);
    put32(b, pos + 4, 72 + 80);
    put_name(b, pos + 8, segname);
    put64(b, pos + 24, vm);
    put64(b, pos + 32, vmsize);
    put32(b, pos + 56, prot);
    put32(b, pos + 60, prot);
    put32(b, pos + 64, 1);
    put_name(b, pos + 72, sectname);
    put_name(b, pos + 72 + 16, segname);
    put64(b, pos + 72 + 32, addr);
    put64(b, pos + 72 + 40, size);
    put32(b, pos + 72 + 48, offset);
    ++commands;
    return pos;
  };
  for (const auto& s : spec.surfaces) {
    segment("__FVMLIB", s.vm, s.allocation, s.direction, "__fvmlib", s.vm,
            s.allocation, 0);
  }
  const std::vector<uint8_t> stream = task_stream(spec);
  const size_t text_command = segment(
      "__TEXT", 0x10000000, 0x4000, 5, "__text", 0x10000000, stream.size(), 0);
  if (spec.kernel_vm) {
    segment("__KERN_0", *spec.kernel_vm, 0x4000, 1, "__kern_0",
            *spec.kernel_vm, 0x100, 0);
  }
  if (spec.scratch) {
    segment("__DATA", spec.scratch->first, spec.scratch->second, 3, "__bss",
            spec.scratch->first, spec.scratch->second, 0);
  }
  if (spec.descriptor) {
    const size_t pos = b.size();
    b.resize(pos + 0x10 * 65 + 0x10, 0);
    put32(b, pos, 4);
    put32(b, pos + 4, 0x10 * 65 + 0x10);
    put32(b, pos + 8, 4);
    for (const auto& slot : spec.slots) {
      put64(b, pos + 0x10 * (slot.first + 1), slot.second);
    }
    ++commands;
  }
  for (const auto& s : spec.surfaces) {
    const size_t pos = b.size();
    b.resize(pos + 0x80 + 16, 0);
    put32(b, pos, 4);
    put32(b, pos + 4, 0x80 + 16);
    put32(b, pos + 8, 3);
    put32(b, pos + 0x14, s.direction);
    put32(b, pos + 0x28, s.n);
    put64(b, pos + 0x50, s.batch);
    put32(b, pos + 0x7c, 0x80);
    put_name(b, pos + 0x80, s.name);
    ++commands;
  }
  for (const auto& s : spec.surfaces) {
    const size_t pos = b.size();
    b.resize(pos + 24 + 16, 0);
    put32(b, pos, 0x40);
    put32(b, pos + 4, 24 + 16);
    put64(b, pos + 16, s.vm);
    put_name(b, pos + 24, s.name);
    ++commands;
  }
  const uint32_t sizeofcmds = static_cast<uint32_t>(b.size() - 32);
  const size_t text_offset = (b.size() + 0x3fff) & ~size_t(0x3fff);
  b.resize(text_offset, 0);
  b.insert(b.end(), stream.begin(), stream.end());
  put32(b, text_command + 72 + 48, static_cast<uint32_t>(text_offset));
  put32(b, 0, 0xbeefface);
  put32(b, 4, 0x80);
  put32(b, 8, spec.subtype);
  put32(b, 16, commands);
  put32(b, 20, sizeofcmds);
  return b;
}

} // namespace h16_test
