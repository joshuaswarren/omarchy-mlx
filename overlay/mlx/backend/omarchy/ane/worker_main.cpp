// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

#include "mlx/backend/omarchy/ane/runtime_detail.h"
#include "mlx/backend/omarchy/ane/bundle.h"

#include <cerrno>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <iostream>
#include <limits>
#include <map>
#include <sys/file.h>
#include <sys/stat.h>
#include <string>

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "[omarchy-ane] worker: expected STAGING_BYTES\n";
    return 2;
  }
  try {
    struct stat inherited_lock {};
    if (::fstat(
            mlx::core::omarchy::ane::detail::kWorkerHardwareLockFd,
            &inherited_lock) != 0 || !S_ISREG(inherited_lock.st_mode) ||
        inherited_lock.st_nlink != 1 ||
        ::flock(
            mlx::core::omarchy::ane::detail::kWorkerHardwareLockFd,
            LOCK_EX | LOCK_NB) != 0) {
      throw std::invalid_argument("missing inherited ANE hardware lock");
    }
    errno = 0;
    char* end = nullptr;
    unsigned long long size = std::strtoull(argv[1], &end, 10);
    if (errno != 0 || end == argv[1] || *end != '\0' || size == 0 ||
        size > std::numeric_limits<size_t>::max()) {
      throw std::invalid_argument("invalid staging byte count");
    }
    const auto manifest_path = std::filesystem::path("/proc/self/fd") /
        std::to_string(mlx::core::omarchy::ane::detail::kWorkerManifestFd);
    const auto manifest = mlx::core::omarchy::ane::parse_ane_manifest(manifest_path);
    std::map<std::string, std::filesystem::path> payloads;
    for (size_t i = 0; i < manifest.payloads.size(); ++i) {
      payloads.emplace(
          manifest.payloads[i].path,
          std::filesystem::path("/proc/self/fd") /
              std::to_string(
                  mlx::core::omarchy::ane::detail::kWorkerPayloadFdBase + i));
    }
    return mlx::core::omarchy::ane::detail::run_worker(
        mlx::core::omarchy::ane::detail::kWorkerControlFd,
        mlx::core::omarchy::ane::detail::kWorkerStagingFd,
        static_cast<size_t>(size),
        manifest_path,
        payloads);
  } catch (const std::exception& error) {
    std::cerr << "[omarchy-ane] worker: " << error.what() << '\n';
    return 2;
  }
}
