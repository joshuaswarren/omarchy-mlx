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
#include <utility>
#include <vector>

#include "mlx/api.h"

#ifndef _WIN32
#include <unistd.h>
#endif

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

// One bundle file snapshotted into a sealed memfd: the process consumes
// these bytes for the rest of its life, and on-disk mutation of the
// original cannot reach them. `sha256` is measured over exactly the
// sealed bytes. Move-only: the owner closes the descriptor.
struct AneSealedFile {
  int fd{-1};
  std::string sha256;
  AneSealedFile() = default;
  AneSealedFile(int fd, std::string sha256)
      : fd(fd), sha256(std::move(sha256)) {}
#ifndef _WIN32
  ~AneSealedFile() {
    if (fd >= 0) {
      ::close(fd);
    }
  }
  AneSealedFile(const AneSealedFile&) = delete;
  AneSealedFile& operator=(const AneSealedFile&) = delete;
  AneSealedFile(AneSealedFile&& other) noexcept
      : fd(std::exchange(other.fd, -1)),
        sha256(std::move(other.sha256)) {}
  AneSealedFile& operator=(AneSealedFile&& other) noexcept {
    if (this != &other) {
      if (fd >= 0) {
        ::close(fd);
      }
      fd = std::exchange(other.fd, -1);
      sha256 = std::move(other.sha256);
    }
    return *this;
  }
#endif
};

MLX_API AneAnecHeader parse_anec_header(
    const std::filesystem::path& path,
    uint64_t tile_shift = kAneTileShiftDefault);
MLX_API AneBundle load_bundle(const std::filesystem::path& dir);
MLX_API AneBundle load_bundle_snapshot(
    const std::filesystem::path& manifest,
    const std::map<std::string, std::filesystem::path>& payloads,
    const std::map<std::filesystem::path, std::string>& known_digests = {});
MLX_API std::string sha256_hex(const uint8_t* data, size_t size);
MLX_API std::string sha256_file(const std::filesystem::path& path);

// Unit-test hook: when true, hashing pins the scalar compress path even
// where the CPU reports the ARMv8 crypto extension, so the FIPS vectors
// exercise both implementations in one suite run.
MLX_API void sha256_force_scalar_compress(bool force);

// Snapshot one regular file from `directory_fd` into a sealed memfd and
// return it with the digest of the sealed bytes. The digest is computed
// during the copy, the write seal is applied afterwards, and F_GET_SEALS
// must confirm every seal stuck or this fails; the hash therefore always
// describes exactly the immutable bytes the caller consumes. The name
// must be a flat entry of the directory (no '/'); links are refused.
MLX_API AneSealedFile sealed_file_at(int directory_fd, const std::string& name);

// The load-boundary seal: every file the session consumes (manifest.json
// and each manifest payload of `dir`) is hashed and bound to `expected`
// (file name -> sha256) before any parse that can reach execution. Every
// consumed file needs an expectation and every expectation needs a
// consumed file; a mismatch, a missing pin, an unknown file, or a link is
// a refusal. The returned bundle's program and weights paths point at the
// consumed images (/proc/self/fd/<fd>); the appended AneSealedFile entries
// own those descriptors and must outlive the bundle and the device session
// that consumes them.
//
// Per-file shape: the default is the sealed memfd snapshot (sealed_file_at,
// above) — the snapshot closes the check-then-use window because the hash
// describes exactly the immutable bytes the session reads. The snapshot's
// read pass overlaps the memfd copy (hash inline, writer thread drains the
// previous chunk), so the TOCTOU guarantee costs near the sha256 floor.
// Warm-path exception (Main review, 2026-10-03): the source descriptor is
// handed over without a snapshot ONLY when the canonical file AND every
// parent directory are root-owned and not group/other-writable — bytes the
// process could not modify even in principle — AND the identity-keyed
// digest sidecar (path|dev|ino|size|mtime_ns|ctime_ns, the same store the
// unsealed path uses) carries this exact identity's digest equal to the
// pin. Any identity change, a missing or stale entry, or
// OMARCHY_ANE_SEAL_VERIFY (truthy: 1/true/yes/on) restores the full sealed
// snapshot; the pin comparison runs in both paths. The sidecar remains a
// mismatch detector for accidental corruption or stale deploys, not an
// anti-tamper boundary: it adds no capability the writer of a root-owned
// read-only chain (root only) did not already have.
MLX_API AneBundle load_bundle_sealed(
    const std::filesystem::path& dir,
    const std::map<std::string, std::string>& expected,
    std::vector<AneSealedFile>& sealed);

} // namespace mlx::core::omarchy::ane
