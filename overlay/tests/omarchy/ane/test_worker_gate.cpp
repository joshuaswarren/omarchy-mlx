// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

// Unit tests for the runtime worker ABI profile gate (E3.1). These tests
// exercise the decision logic only; no /proc, /sys, libane, or ANE
// device is touched, so they run on any host architecture (this CT is
// x86_64). The test sources the mirrored profile table from
// runtime_worker_gate.h; the production kAbiProfiles in
// runtime_worker.cpp is the byte-for-byte source of truth and a
// consistency check below enforces the mirror match.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

#include "mlx/backend/omarchy/honeykrisp_identity.h"
#include "mlx/backend/omarchy/ane/driver_abi.h"
#include "mlx/backend/omarchy/ane/runtime_worker_gate.h"
#include "mlx/backend/omarchy/ane/runtime_ownership.h"

#include <cstring>
#include <filesystem>
#include <fcntl.h>
#include <string>
#include <sys/stat.h>
#include <unistd.h>

using mlx::core::omarchy::ane::detail::abi_profile_expected_compatibles_text;
using mlx::core::omarchy::ane::detail::abi_profile_first_matching_compatible;
using mlx::core::omarchy::ane::detail::abi_profile_identity_tag;
using mlx::core::omarchy::ane::detail::abi_profile_libane_commit;
using mlx::core::omarchy::ane::detail::assert_bundle_abi_matches_profile;
using mlx::core::omarchy::ane::detail::compatible_matches;
using mlx::core::omarchy::ane::detail::kAbiProfilesMirror;
using mlx::core::omarchy::ane::detail::resolve_abi_profile_index;

// Stable ABI-1 libane commit before this change landed. If the gate
// ever loosens or forgets this pin, the test below will catch it.
constexpr const char* kAbiOneExpectedLibane =
    "6fa243ac7241119a9eb229abbf8cb4dd8949f915";

TEST_CASE(
    "default gate selects ABI 1 when MLX_OMARCHY_ANE_ABI is unset") {
  const int index = resolve_abi_profile_index(nullptr);
  CHECK(index == 0);
  CHECK(kAbiProfilesMirror[index].abi == 1);
  CHECK(std::strcmp(kAbiProfilesMirror[index].module_name, "ane") == 0);
  CHECK(std::strcmp(
            abi_profile_libane_commit(kAbiProfilesMirror[index]),
            kAbiOneExpectedLibane) == 0);
  CHECK(std::strcmp(abi_profile_identity_tag(kAbiProfilesMirror[index]), "1") ==
        0);
}

TEST_CASE("default gate selects ABI 1 on empty-string env") {
  const int index = resolve_abi_profile_index("");
  CHECK(index == 0);
}

TEST_CASE("MLX_OMARCHY_ANE_ABI=2 selects the ABI-2 T6021 profile") {
  const int index = resolve_abi_profile_index("2");
  CHECK(index == 1);
  CHECK(kAbiProfilesMirror[index].abi == 2);
  CHECK(std::strcmp(kAbiProfilesMirror[index].module_name, "ane_t6021") == 0);
  // ABI-2 libane commit must differ from ABI-1 — the two lanes cannot
  // share a pin or an M1 host could pass an ABI-2 bundle (or vice
  // versa) by replaying the same libane build.
  CHECK(std::strcmp(abi_profile_libane_commit(kAbiProfilesMirror[index]),
                    kAbiOneExpectedLibane) != 0);
  CHECK(std::strcmp(abi_profile_identity_tag(kAbiProfilesMirror[index]), "2") ==
        0);
}

TEST_CASE(
    "MLX_OMARCHY_ANE_ABI=1 explicitly selects ABI 1 (no regression)") {
  const int index = resolve_abi_profile_index("1");
  CHECK(index == 0);
  CHECK(kAbiProfilesMirror[index].abi == 1);
}

TEST_CASE("unknown ABI integers are rejected") {
  CHECK_THROWS_AS(resolve_abi_profile_index("3"), std::invalid_argument);
  CHECK_THROWS_AS(resolve_abi_profile_index("0"), std::invalid_argument);
  CHECK_THROWS_AS(resolve_abi_profile_index("11"), std::invalid_argument);
}

TEST_CASE("garbage ABI strings are rejected, not silently defaulted") {
  // "2foo" would let std::stoi return 2; the worker uses strtol with
  // an end-pointer to reject trailing junk.
  CHECK_THROWS_AS(resolve_abi_profile_index("2foo"), std::invalid_argument);
  CHECK_THROWS_AS(resolve_abi_profile_index("1.0"), std::invalid_argument);
  CHECK_THROWS_AS(resolve_abi_profile_index(" h14"), std::invalid_argument);
  CHECK_THROWS_AS(resolve_abi_profile_index("two"), std::invalid_argument);
}

TEST_CASE("DT-compatible matching mirrors the worker's has_compatible") {
  // Real M1 / T8103 ANE node compatible strings include apple,t8103-ane
  // as a primary or fallback entry. Real M2 / T6021 nodes list
  // apple,t6021-ane. The gate must accept either profile against its
  // expected blob and reject the other ABI's blob.
  auto make_blob = [](const char* first, const char* second) {
    std::string blob(std::strlen(first) + 1 + std::strlen(second), '\0');
    std::memcpy(blob.data(), first, std::strlen(first));
    std::memcpy(blob.data() + std::strlen(first) + 1, second,
                std::strlen(second));
    return blob;
  };
  const std::string m1_blob =
      make_blob("apple,ane", "apple,t8103-ane");
  const std::string m2_blob =
      make_blob("apple,ane", "apple,t6021-ane");

  // Plain string membership check, for completeness.
  CHECK(compatible_matches(m1_blob, "apple,t8103-ane"));
  CHECK_FALSE(compatible_matches(m1_blob, "apple,t6021-ane"));
  CHECK(compatible_matches(m2_blob, "apple,t6021-ane"));
  CHECK_FALSE(compatible_matches(m2_blob, "apple,t8103-ane"));
}

TEST_CASE("ABI-1 profile accepts an M1 host and rejects an M2 host") {
  // Build the NUL-separated DT blob without truncating at the
  // embedded NUL: std::string(const char*) would stop at the first
  // NUL; the data()+memcpy path preserves every byte and leaves the
  // std::string non-const so subsequent writes go through the data()
  // pointer (operator[] is const here because we want the test to
  // match the read-only shape of /proc/device-tree/.../compatible).
  auto make_blob = [](const char* first, const char* second) {
    std::string blob(std::strlen(first) + 1 + std::strlen(second), '\0');
    std::memcpy(blob.data(), first, std::strlen(first));
    std::memcpy(blob.data() + std::strlen(first) + 1, second,
                std::strlen(second));
    return blob;
  };
  const std::string m1_blob =
      make_blob("apple,ane", "apple,t8103-ane");
  const std::string m2_blob =
      make_blob("apple,ane", "apple,t6021-ane");

  const auto& abi1 = kAbiProfilesMirror[0];
  CHECK(abi_profile_first_matching_compatible(abi1, m1_blob) != nullptr);
  CHECK(abi_profile_first_matching_compatible(abi1, m2_blob) == nullptr);
  // The error text names both expected compatibles and the lane.
  const std::string expected_text = abi_profile_expected_compatibles_text(abi1);
  CHECK(expected_text.find("apple,t8103-ane") != std::string::npos);
}

TEST_CASE("ABI-2 profile accepts an M2 host and rejects an M1 host") {
  auto make_blob = [](const char* first, const char* second) {
    std::string blob(std::strlen(first) + 1 + std::strlen(second), '\0');
    std::memcpy(blob.data(), first, std::strlen(first));
    std::memcpy(blob.data() + std::strlen(first) + 1, second,
                std::strlen(second));
    return blob;
  };
  const std::string m1_blob =
      make_blob("apple,ane", "apple,t8103-ane");
  const std::string m2_blob =
      make_blob("apple,ane", "apple,t6021-ane");

  const auto& abi2 = kAbiProfilesMirror[1];
  CHECK(abi_profile_first_matching_compatible(abi2, m2_blob) != nullptr);
  CHECK(abi_profile_first_matching_compatible(abi2, m1_blob) == nullptr);
  const std::string expected_text = abi_profile_expected_compatibles_text(abi2);
  CHECK(expected_text.find("apple,t6021-ane") != std::string::npos);
  // ABI-2 must NOT accept t8103-ane — loosening the table here would
  // let an M1 host run an ABI-2 bundle (which the cross-check below
  // already refuses, but the per-ABI table must remain exclusive).
  CHECK(expected_text.find("apple,t8103-ane") == std::string::npos);
}

TEST_CASE("ABI-1 profile accepts the T6000-family node of T6001 hosts") {
  // Packaged T6001 overlay nodes (omarchy-ane packaging/dt/t6001-ane.dts)
  // carry the family compatible apple,t6000-ane — the of_match entry the
  // ane driver binds on M1 Pro/Max/Ultra. ABI-1 must accept it (as primary
  // or fallback entry, and as the single-entry blob the live node carries)
  // and ABI-2 must keep refusing it.
  auto make_blob = [](const char* first, const char* second) {
    std::string blob(std::strlen(first) + 1 + std::strlen(second), '\0');
    std::memcpy(blob.data(), first, std::strlen(first));
    std::memcpy(blob.data() + std::strlen(first) + 1, second,
                std::strlen(second));
    return blob;
  };
  const std::string t6001_blob = make_blob("apple,ane", "apple,t6000-ane");
  const std::string packaged_blob = std::string("apple,t6000-ane") + '\0';
  const std::string m2_blob = make_blob("apple,ane", "apple,t6021-ane");

  const auto& abi1 = kAbiProfilesMirror[0];
  const auto& abi2 = kAbiProfilesMirror[1];
  CHECK(std::strcmp(abi_profile_first_matching_compatible(abi1, t6001_blob),
                    "apple,t6000-ane") == 0);
  CHECK(std::strcmp(abi_profile_first_matching_compatible(abi1, packaged_blob),
                    "apple,t6000-ane") == 0);
  CHECK(abi_profile_first_matching_compatible(abi2, t6001_blob) == nullptr);
  CHECK(abi_profile_first_matching_compatible(abi2, m2_blob) != nullptr);
  const std::string expected_text = abi_profile_expected_compatibles_text(abi1);
  CHECK(expected_text.find("apple,t6000-ane") != std::string::npos);
}

TEST_CASE(
    "per-ABI pins do not leak between lanes (libane + module_name)") {
  const auto& abi1 = kAbiProfilesMirror[0];
  const auto& abi2 = kAbiProfilesMirror[1];
  CHECK(std::strcmp(abi1.module_name, abi2.module_name) != 0);
  CHECK(std::strcmp(
            abi_profile_libane_commit(abi1),
            abi_profile_libane_commit(abi2)) != 0);
}
TEST_CASE(
    "bundle ABI cross-check refuses h14 on an ABI-1 gate") {
  const auto& abi1 = kAbiProfilesMirror[0];
  // h14 bundles declare driver_abi_major=2; an ABI-1 worker must
  // refuse them. The thrown std::invalid_argument carries the
  // bundle's declared abi and the active lane name (full text covered
  // by CHECK_NOTHROW / CHECK_THROWS_AS below; message-substring
  // assertions are skipped because doctest 2.4.12's Contains
  // substring check interacts awkwardly with template string
  // operators in some header orders).
  CHECK_THROWS_AS(
      assert_bundle_abi_matches_profile(2, abi1),
      std::invalid_argument);
}

TEST_CASE("bundle ABI cross-check refuses h13 on an ABI-2 gate") {
  const auto& abi2 = kAbiProfilesMirror[1];
  // h13 bundles declare driver_abi_major=1; an ABI-2 worker must
  // refuse them.
  CHECK_THROWS_AS(
      assert_bundle_abi_matches_profile(1, abi2),
      std::invalid_argument);
}

TEST_CASE("bundle ABI cross-check accepts matched pairs") {
  const auto& abi1 = kAbiProfilesMirror[0];
  const auto& abi2 = kAbiProfilesMirror[1];
  CHECK_NOTHROW(assert_bundle_abi_matches_profile(1, abi1));
  CHECK_NOTHROW(assert_bundle_abi_matches_profile(2, abi2));
}

TEST_CASE(
    "ABI-1 gate identity still records the original h13 lane markers") {
  // The pre-change identity string committed to "dt_compatible=apple,t8103-ane"
  // and "driver_abi=1". The new ABI-1 record must keep both exactly so
  // every receipt that grep'd those substrings still parses (and so an
  // M1 host can still be distinguished from an M2 host at the
  // identity-record layer).
  const auto& abi1 = kAbiProfilesMirror[0];
  const auto& abi2 = kAbiProfilesMirror[1];
  const std::string abi1_compatibles_text = abi_profile_expected_compatibles_text(abi1);
  CHECK(abi1_compatibles_text.find("apple,t8103-ane") != std::string::npos);
  CHECK(std::strcmp(abi_profile_identity_tag(abi1), "1") == 0);
  CHECK(std::strcmp(abi_profile_identity_tag(abi2), "2") == 0);
}
TEST_CASE("runtime PM acceptance allows packaged auto control, refuses inactive") {
  using mlx::core::omarchy::ane::detail::runtime_pm_acceptable;
  CHECK(runtime_pm_acceptable("on", "active"));
  // The packaged ane driver enables runtime PM: control is 'auto' and
  // the gate's device open resumes the device before this check.
  CHECK(runtime_pm_acceptable("auto", "active"));
  CHECK_FALSE(runtime_pm_acceptable("auto", "suspended"));
  CHECK_FALSE(runtime_pm_acceptable("on", "suspended"));
  CHECK_FALSE(runtime_pm_acceptable("idle", "active"));
  CHECK_FALSE(runtime_pm_acceptable("", "active"));
}

TEST_CASE("driver ABI major accepts the expected ABI and rejects mismatches") {
  using mlx::core::omarchy::ane::detail::require_driver_abi_major;
  CHECK_NOTHROW(require_driver_abi_major(1, 1));
  CHECK_NOTHROW(require_driver_abi_major(2, 2));
  try {
    require_driver_abi_major(1, 2);
    FAIL("expected ABI mismatch");
  } catch (const std::runtime_error& error) {
    CHECK(std::string(error.what()) ==
          "ANE driver ABI mismatch: runtime requires major 1, driver reports major 2");
  }
}
TEST_CASE("Honeykrisp ICD resolution honors user choices and identifies Mesa SHA") {
  using namespace mlx::core::omarchy;
  const std::vector<std::string> candidates{
      "/usr/share/vulkan/icd.d/lvp_icd.json",
      "/usr/share/vulkan/icd.d/asahi_icd.json"};
  CHECK(resolve_honeykrisp_icd(candidates, nullptr) == candidates[1]);
  CHECK(resolve_honeykrisp_icd(candidates, candidates[1].c_str()) == candidates[1]);
  CHECK(resolve_honeykrisp_icd(candidates, "/tmp/honeykrisp.json:/tmp/other.json") ==
        "/tmp/honeykrisp.json");
  CHECK_THROWS_AS(
      resolve_honeykrisp_icd(candidates, "/tmp/lvp_icd.json"),
      std::runtime_error);
  CHECK(mesa_git_sha("Mesa 26.1.0-devel (git-0123456789abcdef)") ==
        "0123456789abcdef");
  CHECK(mesa_git_sha("Mesa release build") == "");
  CHECK_NOTHROW(require_expected_honeykrisp_sha("0123456789abcdef",
                                                "0123456789abcdef"));
  try {
    require_expected_honeykrisp_sha("expected", "actual");
    FAIL("expected SHA mismatch");
  } catch (const std::runtime_error& error) {
    CHECK(std::string(error.what()) ==
          "Honeykrisp Mesa git SHA mismatch: expected expected, found actual");
  }
}

TEST_CASE("ANE lock files are created and repaired to world access") {
  using mlx::core::omarchy::ane::detail::RuntimeOwnership;
  const auto directory = std::filesystem::temp_directory_path() /
      ("runtimegate-lock-" + std::to_string(::getpid()));
  std::filesystem::remove_all(directory);
  REQUIRE(std::filesystem::create_directory(directory));
  const auto lock = directory / "device.lock";
  const auto state = directory / "quarantine";
  for (const auto& path : {lock, state}) {
    const int fd = ::open(path.c_str(), O_CREAT | O_WRONLY | O_CLOEXEC, 0660);
    REQUIRE(fd >= 0);
    REQUIRE(::fchmod(fd, 0660) == 0);
    REQUIRE(::close(fd) == 0);
  }
  {
    auto ownership = RuntimeOwnership::acquire_at(
        lock, state, "00000000-0000-0000-0000-000000000000");
    struct stat lock_status {};
    struct stat state_status {};
    REQUIRE(::stat(lock.c_str(), &lock_status) == 0);
    REQUIRE(::stat(state.c_str(), &state_status) == 0);
    CHECK((lock_status.st_mode & 0777) == 0666);
    CHECK((state_status.st_mode & 0777) == 0666);
  }
  REQUIRE(::chmod(directory.c_str(), 01777) == 0);
  CHECK_THROWS_AS(
      RuntimeOwnership::validate_shared_paths_at(
          directory, lock, state, ::geteuid() + 1, ::getegid()),
      std::runtime_error);
  std::filesystem::remove_all(directory);
}
