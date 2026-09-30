// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <cctype>
#include <stdexcept>
#include <string>
#include <vector>

namespace mlx::core::omarchy {

inline bool is_honeykrisp_icd(const std::string& path) {
  std::string lower = path;
  std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return lower.find("honeykrisp") != std::string::npos ||
      lower.find("asahi_icd") != std::string::npos ||
      lower.find("libvulkan_asahi") != std::string::npos;
}

inline std::string resolve_honeykrisp_icd(
    const std::vector<std::string>& candidates,
    const char* user_files) {
  if (user_files != nullptr && user_files[0] != '\0') {
    std::string value(user_files);
    size_t start = 0;
    while (start <= value.size()) {
      const size_t end = value.find(':', start);
      const std::string item = value.substr(start, end - start);
      if (is_honeykrisp_icd(item)) {
        return item;
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
  for (const auto& candidate : candidates) {
    if (is_honeykrisp_icd(candidate)) {
      return candidate;
    }
  }
  throw std::runtime_error("Honeykrisp Vulkan ICD JSON was not found");
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
