// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

#pragma once

#include "mlx/backend/omarchy/ane/manifest.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "mlx/api.h"

namespace mlx::core::omarchy::ane {

constexpr size_t kAnecHeaderSize = 0x6a8;
constexpr size_t kAnecPayloadOffset = 0x1000;
constexpr size_t kAnecTileCount = 0x20;

struct AneAnecHeader {
  uint64_t payload_size{0};
  uint32_t task_descriptor_size{0};
  uint32_t task_descriptor_count{0};
  uint64_t task_size{0};
  uint64_t kernel_size{0};
  uint32_t source_count{0};
  uint32_t destination_count{0};
  uint64_t bootstrap_channel_size{0};
  std::array<uint32_t, kAnecTileCount> tiles{};
  std::array<std::array<uint64_t, 6>, kAnecTileCount> nchw{};
};

struct AneBundleNotFound : std::runtime_error {
  using std::runtime_error::runtime_error;
};

struct AneValidatedProgram {
  size_t manifest_index{0};
  AneAnecHeader anec_header;
  std::filesystem::path anec;
  uint64_t tile_shift{kAneTileShiftDefault};
};

struct AneBundle {
  AneManifest manifest;
  std::vector<AneValidatedProgram> programs;
  std::optional<std::filesystem::path> weights;
};

MLX_API AneAnecHeader parse_anec_header(
    const std::filesystem::path& path,
    uint64_t tile_shift = kAneTileShiftDefault);
MLX_API AneBundle load_bundle(const std::filesystem::path& dir);
MLX_API AneBundle load_bundle_snapshot(
    const std::filesystem::path& manifest,
    const std::map<std::string, std::filesystem::path>& payloads);
MLX_API std::string sha256_hex(const uint8_t* data, size_t size);
MLX_API std::string sha256_file(const std::filesystem::path& path);

} // namespace mlx::core::omarchy::ane
