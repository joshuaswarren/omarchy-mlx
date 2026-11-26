// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

#pragma once

#include <filesystem>
#include <string>

namespace omarchy_info {

inline std::filesystem::path find_ane_module(
    const std::filesystem::path& sysroot) {
  const auto modules = sysroot / "sys/module";
  std::error_code ec;
  const auto generic = modules / "ane";
  if (std::filesystem::is_directory(generic, ec)) {
    return generic;
  }

  std::filesystem::path match;
  ec.clear();
  for (std::filesystem::directory_iterator it(modules, ec), end;
       !ec && it != end; it.increment(ec)) {
    if (!it->is_directory(ec)) {
      ec.clear();
      continue;
    }
    const std::string name = it->path().filename().string();
    if (name.rfind("ane_", 0) == 0 &&
        (match.empty() || name < match.filename().string())) {
      match = it->path();
    }
  }
  return match;
}

} // namespace omarchy_info
