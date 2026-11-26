// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

#pragma once

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "mlx/api.h"
#include "mlx/backend/omarchy/ane/manifest.h"

namespace mlx::core::omarchy::ane {

using AneBuffer = std::vector<uint8_t>;
using AneBufferMap = std::map<std::string, AneBuffer>;

struct AneShutdownReceipt {
  int worker_pid{-1};
  uint64_t released_programs{0};
};

class MLX_API AneRuntime {
 public:
  static std::unique_ptr<AneRuntime> load(
      const std::filesystem::path& bundle,
      std::chrono::milliseconds startup_deadline,
      const std::filesystem::path& diagnostic_path);

  ~AneRuntime();

  AneRuntime(const AneRuntime&) = delete;
  AneRuntime& operator=(const AneRuntime&) = delete;

  AneBufferMap execute(
      const AneBufferMap& inputs,
      std::chrono::milliseconds deadline);
  const std::vector<AneLogicalResult>& output_layout() const;
  AneShutdownReceipt shutdown(std::chrono::milliseconds deadline);

  bool usable() const;
  int worker_pid() const;
  std::string runtime_identity() const;

 private:
  struct Impl;
  explicit AneRuntime(std::unique_ptr<Impl> implementation);

  std::unique_ptr<Impl> implementation_;
};

} // namespace mlx::core::omarchy::ane
