// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace mlx::core::omarchy {

// Packaged Vulkan identity files (v0.7.7 recipe contract, names fixed).
// The package installs the Honeykrisp ICD JSON and the Mesa git SHA file
// under this directory; OMARCHY_MLX_SYSTEM_PREFIX overrides the prefix so
// tests and hardware rehearsals can stage a fake tree.
#ifndef OMARCHY_MLX_SYSTEM_PREFIX
#define OMARCHY_MLX_SYSTEM_PREFIX "/usr/lib/omarchy-mlx"
#endif

inline std::string omarchy_system_prefix() {
  const char* override_prefix = std::getenv("OMARCHY_MLX_SYSTEM_PREFIX");
  return (override_prefix != nullptr && override_prefix[0] != '\0')
      ? std::string(override_prefix)
      : std::string(OMARCHY_MLX_SYSTEM_PREFIX);
}

inline std::string packaged_honeykrisp_icd_path(const std::string& prefix) {
  return prefix + "/vulkan/honeykrisp_icd.aarch64.json";
}

inline std::string packaged_mesa_git_sha_path(const std::string& prefix) {
  return prefix + "/vulkan/mesa-git-sha";
}

inline bool is_honeykrisp_icd(const std::string& path) {
  std::string lower = path;
  std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return lower.find("honeykrisp") != std::string::npos ||
      lower.find("asahi_icd") != std::string::npos ||
      lower.find("libvulkan_asahi") != std::string::npos;
}

struct HoneykrispIcdSelection {
  std::string path;
  // True when the packaged recipe ICD was selected (no user override).
  bool packaged{false};
  // True when VK_DRIVER_FILES or VK_ICD_FILENAMES picked the ICD.
  bool user_override{false};
};

inline HoneykrispIcdSelection resolve_honeykrisp_icd_detail(
    const std::vector<std::string>& candidates,
    const char* user_files,
    const std::string& packaged_path) {
  if (user_files != nullptr && user_files[0] != '\0') {
    std::string value(user_files);
    size_t start = 0;
    while (start <= value.size()) {
      const size_t end = value.find(':', start);
      const std::string item = value.substr(start, end - start);
      if (is_honeykrisp_icd(item)) {
        std::error_code existence;
        if (!std::filesystem::exists(item, existence) || existence) {
          throw std::runtime_error(
              "Honeykrisp ICD selection refused: user Vulkan ICD JSON does"
              " not exist: " +
              item);
        }
        return {item, false, true};
      }
      if (end == std::string::npos) {
        break;
      }
      start = end + 1;
    }
    throw std::runtime_error(
        "Honeykrisp ICD selection refused: user Vulkan ICD value excludes Honeykrisp: " +
        value);
  }
  // The packaged recipe ICD wins over a stock system ICD so a machine
  // with both never silently runs the system driver build.
  if (!packaged_path.empty()) {
    std::error_code existence;
    if (std::filesystem::exists(packaged_path, existence) && !existence) {
      return {packaged_path, true, false};
    }
  }
  for (const auto& candidate : candidates) {
    if (is_honeykrisp_icd(candidate)) {
      return {candidate, false, false};
    }
  }
  throw std::runtime_error("Honeykrisp Vulkan ICD JSON was not found");
}

inline std::string resolve_honeykrisp_icd(
    const std::vector<std::string>& candidates,
    const char* user_files) {
  return resolve_honeykrisp_icd_detail(candidates, user_files, {}).path;
}

inline std::string mesa_git_sha(const std::string& driver_info) {
  const size_t marker = driver_info.find("git-");
  if (marker == std::string::npos) {
    return {};
  }
  size_t end = marker + 4;
  while (end < driver_info.size() &&
         std::isxdigit(static_cast<unsigned char>(driver_info[end]))) {
    ++end;
  }
  const size_t length = end - marker - 4;
  return length >= 7 ? driver_info.substr(marker + 4, length) : std::string{};
}

// Reads the packaged one-line Mesa git SHA (the string the driver reports
// after "git-"). A missing or unreadable file yields an empty string: the
// caller then records identity without enforcing an expectation.
inline std::string packaged_mesa_git_sha(const std::string& prefix) {
  const std::filesystem::path path = std::filesystem::path(prefix) /
      "vulkan" / "mesa-git-sha";
  std::error_code error;
  if (!std::filesystem::is_regular_file(path, error) || error) {
    return {};
  }
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    return {};
  }
  std::string line;
  std::getline(input, line);
  const auto is_pad = [](char c) {
    return c == '\n' || c == '\r' || c == ' ' || c == '\t';
  };
  while (!line.empty() && is_pad(line.back())) {
    line.pop_back();
  }
  size_t start = 0;
  while (start < line.size() && is_pad(line[start])) {
    ++start;
  }
  return line.substr(start);
}

inline void require_expected_honeykrisp_sha(
    const std::string& expected,
    const std::string& actual) {
  if (!expected.empty() && expected != actual) {
    throw std::runtime_error(
        "Honeykrisp Mesa git SHA mismatch: expected " + expected +
        ", found " + (actual.empty() ? std::string("unavailable") : actual));
  }
}

} // namespace mlx::core::omarchy
