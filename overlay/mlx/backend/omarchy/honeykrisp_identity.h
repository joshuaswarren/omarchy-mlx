// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <mutex>
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

// Apple M3/M4-family chip ids (omarchy-ane data/ane-soc records; t6034
// is the M3 Max variant whose ADT reports arm-io,t6031). Read only on
// the not-found error path: the packaged Honeykrisp ICD targets M1/M2,
// and an M3/M4 hitting this error must learn that from the message,
// not a bare "not found". Empty string when the chip is not M3/M4
// family or the DT is unreadable.
inline std::string apple_m3_m4_chip_note() {
  std::ifstream compatible_file("/proc/device-tree/compatible");
  if (!compatible_file) {
    return {};
  }
  std::string compatible((std::istreambuf_iterator<char>(compatible_file)),
                         std::istreambuf_iterator<char>());
  struct NewGenEntry {
    const char* chip_id;
    const char* name;
  };
  static constexpr NewGenEntry kNewGenChips[] = {
      {"apple,t8122", "Apple M3"},
      {"apple,t6030", "Apple M3 Pro"},
      {"apple,t6031", "Apple M3 Max"},
      {"apple,t6034", "Apple M3 Max variant"},
      {"apple,t8132", "Apple M4"},
      {"apple,t6040", "Apple M4 Pro"},
      {"apple,t6041", "Apple M4 Max"},
  };
  for (const auto& entry : kNewGenChips) {
    if (compatible.find(entry.chip_id) != std::string::npos) {
      return " (detected " + std::string(entry.name) + ", " + entry.chip_id +
             ": the packaged Honeykrisp ICD supports M1/M2 silicon and M3"
             "/M4 GPU compute is not certified yet; on an M3 Pro, aurora's"
             " mesa-m3 graphics driver is not selected for compute. Collect"
             " M3/M4 state with scripts/m3m4_kit.sh in"
             " joshuaswarren/omarchy-mlx)";
    }
  }
  return {};
}

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
  throw std::runtime_error("Honeykrisp Vulkan ICD JSON was not found" +
                           apple_m3_m4_chip_note());
}

// The system Honeykrisp ICD to retry with when the packaged recipe ICD
// enumerates no supported device (the packaged Mesa targets M1/M2; a newer
// chip may only be served by the system driver). Empty when the packaged
// ICD was not the one selected (an explicit override is never second-guessed)
// or when no other Honeykrisp ICD is installed.
inline std::string system_icd_fallback(
    const std::vector<std::string>& candidates,
    const HoneykrispIcdSelection& selection) {
  if (!selection.packaged || selection.user_override) {
    return {};
  }
  for (const auto& candidate : candidates) {
    if (candidate != selection.path && is_honeykrisp_icd(candidate)) {
      return candidate;
    }
  }
  return {};
}

// vkCreateInstance failed because the loader found no ICD it could load
// (VK_ERROR_INCOMPATIBLE_DRIVER, -9): a missing or broken packaged driver
// library. Takes the raw VkResult so the header needs no Vulkan include.
inline bool instance_failure_is_no_driver(int vk_result) {
  return vk_result == -9;
}

struct IcdFallbackResult {
  bool ok{false};
  // Set only when the system ICD was tried: the ICD that found no device.
  std::string packaged_path;
  // Set only when the system ICD was tried: the one the retry used.
  std::string fallback_path;
};

// Device-discovery policy around one attempt (Runtime::init): run |attempt|;
// when it fails because the selected packaged ICD enumerated no supported
// device (|no_supported_device|, set by the attempt), point the loader at the
// system ICD with |set_icd| and attempt once more. |selection|, |icd_retry|
// and |error| are updated for the second attempt; a failed retry leaves an
// error that carries both reasons. Never retries an override, a refusal, or
// a second time.
template <class Attempt, class SetIcd>
IcdFallbackResult init_with_icd_fallback(
    HoneykrispIcdSelection& selection,
    bool& no_supported_device,
    bool& icd_retry,
    std::string& error,
    const std::vector<std::string>& candidates,
    Attempt attempt,
    SetIcd set_icd) {
  IcdFallbackResult result;
  if (attempt()) {
    result.ok = true;
    return result;
  }
  const std::string fallback =
      no_supported_device ? system_icd_fallback(candidates, selection) : std::string();
  if (fallback.empty()) {
    return result;
  }
  const std::string packaged_error = error;
  result.packaged_path = selection.path;
  selection = HoneykrispIcdSelection{fallback, false, false};
  icd_retry = true;
  if (!set_icd(fallback)) {
    return result;
  }
  if (attempt()) {
    result.ok = true;
    result.fallback_path = fallback;
    return result;
  }
  error = packaged_error + " The system ICD " + fallback + " was tried next: " + error;
  return result;
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

namespace detail {
// Storage shared by note_stock_driver_identity and
// stock_driver_warning_recorded (inline-function statics are unique per
// function, so both must go through one slot).
inline std::string& stock_driver_warning_slot() {
  static std::string slot;
  return slot;
}
} // namespace detail

// Parses the first major.minor.patch triple in a driver info string
// ("Mesa 26.2.3 (git-0f1e2d3c4b)", "Mesa 26.3.0-devel"). Returns false
// when the string holds no such triple; callers then stay silent
// instead of guessing a version.
inline bool first_mesa_version_triple(
    const std::string& driver_info,
    unsigned& major,
    unsigned& minor,
    unsigned& patch) {
  size_t index = 0;
  while (index < driver_info.size()) {
    if (!std::isdigit(static_cast<unsigned char>(driver_info[index]))) {
      ++index;
      continue;
    }
    unsigned values[3] = {0, 0, 0};
    bool complete = true;
    size_t cursor = index;
    for (int component = 0; component < 3; ++component) {
      size_t digits = 0;
      unsigned long long value = 0;
      while (cursor < driver_info.size() &&
             std::isdigit(static_cast<unsigned char>(driver_info[cursor]))) {
        value = value * 10 + static_cast<unsigned long long>(
                                  driver_info[cursor] - '0');
        ++cursor;
        ++digits;
      }
      if (digits == 0) {
        complete = false;
        break;
      }
      values[component] = static_cast<unsigned>(value);
      if (component < 2) {
        if (cursor < driver_info.size() && driver_info[cursor] == '.') {
          ++cursor;
        } else {
          complete = false;
          break;
        }
      }
    }
    if (complete) {
      major = values[0];
      minor = values[1];
      patch = values[2];
      return true;
    }
    index = cursor > index ? cursor : index + 1;
  }
  return false;
}

// Whether a warning is owed, decided without a device so every case is
// unit-tested. The stock system driver only became safe for variable
// shifts in the 26.2.4 release; older stock builds miscompile
// shift-heavy kernels (int8 among them) on Apple GPUs.
struct StockDriverCheck {
  bool warn{false};
  std::string message;
};

inline StockDriverCheck stock_driver_warning(
    const std::string& icd_source,
    const std::string& driver_info,
    const std::string& driver_sha,
    const std::string& expected_sha) {
  StockDriverCheck check;
  // The packaged recipe build comes from the known-good fork: never
  // warn, whatever its version string reads.
  if (icd_source == "packaged") {
    return check;
  }
  // A driver that reports the known-good commit is safe even when its
  // version string reads older (a fork build without a git- token).
  if (!expected_sha.empty() && !driver_sha.empty() &&
      driver_sha == expected_sha) {
    return check;
  }
  unsigned major = 0;
  unsigned minor = 0;
  unsigned patch = 0;
  if (!first_mesa_version_triple(driver_info, major, minor, patch)) {
    return check;
  }
  const bool older_than_fix = major < 26 ||
      (major == 26 &&
       (minor < 2 || (minor == 2 && patch < 4)));
  if (!older_than_fix) {
    return check;
  }
  check.warn = true;
  check.message = "The system Vulkan driver is Mesa " +
      std::to_string(major) + "." + std::to_string(minor) + "." +
      std::to_string(patch) +
      ". Mesa older than 26.2.4 can return wrong results from int8 and"
      " other shift-heavy kernels. Install the omarchy-mlx-vulkan package"
      " or update Mesa to 26.2.4 or newer.";
  return check;
}

// Resolves the warning decision once per process at the first device
// identity. When the selected driver is an old stock build it prints
// one stderr line with the stable "[omarchy] warning: " prefix and
// keeps running; the same text (empty when silent) is stored for
// device info reporting. Call it for every resolved device: the guard
// keeps one decision per process.
inline const std::string& note_stock_driver_identity(
    const std::string& icd_source,
    const std::string& driver_info,
    const std::string& driver_sha,
    const std::string& expected_sha) {
  static std::once_flag once;
  std::call_once(once, [&] {
    StockDriverCheck check =
        stock_driver_warning(icd_source, driver_info, driver_sha, expected_sha);
    if (check.warn) {
      std::fprintf(stderr, "[omarchy] warning: %s\n", check.message.c_str());
    }
    detail::stock_driver_warning_slot() = std::move(check.message);
  });
  return detail::stock_driver_warning_slot();
}

// The warning text this process printed, or empty when it stayed silent.
inline const std::string& stock_driver_warning_recorded() {
  return detail::stock_driver_warning_slot();
}


// mlx-lm serve patches (docs/INSTALL.md). Stock mlx-lm 0.31.x runs the
// gated-delta update of hybrid models (Qwen3.5, Qwen3.6, Qwen3.8 and others)
// as a Python loop of tiny ops on a non-Metal GPU: one token, four dispatches
// per linear-attention layer and a host synchronization per layer. The vendored
// patch series routes it to mx.fast.gated_delta_update. On a 16-inch M1 Max the
// Qwen3.6-35B-A3B 4-bit prefill reads 44 tokens per second on stock mlx-lm and
// 181 with the series (receipts/2026-10-30-moe-prefill-35b). The unpatched
// loop is never one primitive, so the backend cannot see it as a fallback;
// the check reads the one file that decides it, beside the loaded libmlx.
struct MlxLmPatchCheck {
  bool warn{false};
  std::string message;
};

// `lib_dir` is the directory of the loaded libmlx
// (<site-packages>/mlx/lib). Silent when mlx-lm is not installed beside it,
// when the file is unreadable, or when it carries the fused route.
inline MlxLmPatchCheck mlx_lm_patch_warning(const std::string& lib_dir) {
  MlxLmPatchCheck check;
  if (lib_dir.empty()) {
    return check;
  }
  const std::filesystem::path file = std::filesystem::path(lib_dir) / ".." /
      ".." / "mlx_lm" / "models" / "gated_delta.py";
  std::ifstream in(file);
  if (!in) {
    return check;
  }
  const std::string text(
      (std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  if (text.find("mx.fast.gated_delta_update") != std::string::npos) {
    return check;
  }
  check.warn = true;
  check.message =
      "The mlx-lm in this environment has no omarchy serve patches. Hybrid"
      " models (Qwen3.5, Qwen3.6, Qwen3.8 and others) prefill"
      " several times slower on stock mlx-lm (measured 4.1x on Qwen3.6-35B-A3B"
      " prefill). Fix: install.sh --patch-venv <this venv>. Silence with"
      " MLX_OMARCHY_NO_MLXLM_WARNING=1.";
  return check;
}

// Prints the warning once per process, the first time a device identity is
// resolved. `lib_dir` empty means the library path could not be determined.
inline void note_mlx_lm_patches(const std::string& lib_dir) {
  static std::once_flag once;
  std::call_once(once, [&] {
    const char* off = std::getenv("MLX_OMARCHY_NO_MLXLM_WARNING");
    if (off != nullptr && off[0] != '\0' && off[0] != '0') {
      return;
    }
    const MlxLmPatchCheck check = mlx_lm_patch_warning(lib_dir);
    if (check.warn) {
      std::fprintf(stderr, "[omarchy] warning: %s\n", check.message.c_str());
    }
  });
}

} // namespace mlx::core::omarchy
