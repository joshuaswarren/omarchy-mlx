// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

// Header-only, host-only ABI profile table extracted from
// runtime_worker.cpp so the gate decision logic can be unit-tested on
// any architecture without touching sysfs, /proc, or libane. The
// production code in runtime_worker.cpp keeps its own copy of the
// kAbiProfiles table (because the worker process is a separate
// address space and the table is constexpr); this header mirrors the
// table byte-for-byte and is linked only into test binaries.
//
// KEEP IN SYNC with runtime_worker.cpp::kAbiProfiles. The
// runtime_worker_gate_test.cpp TEST_CASE asserts the mirrors match.

#pragma once

#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <string>

namespace mlx::core::omarchy::ane::detail {

inline constexpr int kAneDefaultAbiMirror = 1;

struct AbiProfileMirror {
  int abi;
  const char* env_name;
  std::initializer_list<const char*> compatibles;
  const char* module_name;
  const char* libane_commit;
  const char* identity_abi_tag;
};

inline constexpr AbiProfileMirror kAbiProfilesMirror[] = {
    {1,
     "ABI 1 (M1 / T8103 / T6001)",
     {"apple,t8103-ane", "apple,t6000-ane"},
     "ane",
     "6fa243ac7241119a9eb229abbf8cb4dd8949f915",
     "1"},
    {2,
     "ABI 2 (M2 / T6021)",
     {"apple,t6021-ane"},
     "ane_t6021",
     "8b010938aeb64bfa04b95e89da0bedd2ef9e3e72",
     "2"},
};

// Pure helper: parse the env-style ABI string and return the matching
// profile index into kAbiProfilesMirror. Throws std::invalid_argument
// on a missing, empty, non-numeric, or out-of-set value. The env var
// itself is read by the caller and passed in.
inline int resolve_abi_profile_index(const char* env_value) {
  if (env_value == nullptr || env_value[0] == '\0') {
    return kAneDefaultAbiMirror - 1;
  }
  char* end = nullptr;
  errno = 0;
  const long parsed = std::strtol(env_value, &end, 10);
  if (errno != 0 || end == env_value || (end != nullptr && *end != '\0')) {
    throw std::invalid_argument(
        "MLX_OMARCHY_ANE_ABI='" + std::string(env_value) +
        "' is not a valid ABI integer (expected 1 or 2)");
  }
  for (size_t i = 0; i < sizeof(kAbiProfilesMirror) / sizeof(kAbiProfilesMirror[0]);
       ++i) {
    if (kAbiProfilesMirror[i].abi == parsed) {
      return static_cast<int>(i);
    }
  }
  throw std::invalid_argument(
      "MLX_OMARCHY_ANE_ABI=" + std::to_string(parsed) +
      " is not a supported ABI (expected 1 or 2)");
}

// Pure helper: render the comma-separated list of expected DT
// compatibles a profile accepts.
inline std::string abi_profile_expected_compatibles_text(
    const AbiProfileMirror& profile) {
  std::string text;
  size_t i = 0;
  for (const char* c : profile.compatibles) {
    if (i > 0) {
      text += " or ";
    }
    text += c;
    ++i;
  }
  return text;
}

// Pure helper: the per-ABI pinned libane commit. The worker identity
// records this string so a peer can confirm which ABI lane the worker
// was started under.
inline const char* abi_profile_libane_commit(const AbiProfileMirror& profile) {
  return profile.libane_commit;
}

// Pure helper: the worker identity's `driver_abi=N` tag.
inline const char* abi_profile_identity_tag(const AbiProfileMirror& profile) {
  return profile.identity_abi_tag;
}

// Pure helper: detect a compatible string in the DT NUL-separated
// compatible blob. Mirrors the worker's has_compatible exactly so the
// test exercises the same prefix/equality semantics the worker uses
// against the live DT node.
inline bool compatible_matches(
    const std::string& compatible_blob,
    const char* expected) {
  if (expected == nullptr) {
    return false;
  }
  size_t offset = 0;
  while (offset < compatible_blob.size()) {
    size_t end = compatible_blob.find('\0', offset);
    if (end == std::string::npos) {
      end = compatible_blob.size();
    }
    if (end - offset == std::strlen(expected) &&
        compatible_blob.compare(offset, end - offset, expected) == 0) {
      return true;
    }
    offset = end + 1;
  }
  return false;
}

// Pure helper: pick the first compatible from a profile that matches
// the DT compatible blob. Returns nullptr when none match (which is
// the worker's "device-tree ANE node does not match ..." error).
inline const char* abi_profile_first_matching_compatible(
    const AbiProfileMirror& profile,
    const std::string& compatible_blob) {
  for (const char* c : profile.compatibles) {
    if (compatible_matches(compatible_blob, c)) {
      return c;
    }
  }
  return nullptr;
}

// Pure helper: cross-check a bundle's declared driver_abi_major
// against the gate profile's ABI. A bundle whose ABI does not match
// the gate's profile must be rejected before any device access; this
// is the host-half of the "h14 bundle must not run on an h13 host and
// vice versa" contract.
inline void assert_bundle_abi_matches_profile(
    uint64_t bundle_driver_abi_major,
    const AbiProfileMirror& profile) {
  if (bundle_driver_abi_major != static_cast<uint64_t>(profile.abi)) {
    throw std::invalid_argument(
        "bundle declares driver_abi_major " +
        std::to_string(bundle_driver_abi_major) +
        " but the worker gate is " + std::string(profile.env_name) +
        " (set MLX_OMARCHY_ANE_ABI to switch lanes)");
  }
}

} // namespace mlx::core::omarchy::ane::detail
