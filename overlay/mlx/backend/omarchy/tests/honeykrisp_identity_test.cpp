// Test-first contract for the foreign-ICD escape hatch on the omarchy
// identity gate. Run with tests/run-identity-tests.sh; exits nonzero on the
// first failed expectation.
//
// Contract (Main decision 2026-11-22 01:15 CST):
//   1. default-refuses: with no hatch, a user ICD that excludes
//      Honeykrisp/Borealis is refused.
//   2. hatch-accepts-named-discrete: with MLX_OMARCHY_FOREIGN_ICD naming the
//      exact user ICD, that ICD is accepted and the selection is marked
//      foreign.
//   3. hatch-still-refuses-CPU: the hatch never accepts a software/CPU ICD
//      (llvmpipe, lavapipe, swrast, softpipe, swiftshader).
#include <cassert>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

#include "../honeykrisp_identity.h"

namespace {

using mlx::core::omarchy::HoneykrispIcdSelection;
using mlx::core::omarchy::resolve_honeykrisp_icd_detail;

std::string write_json(const std::filesystem::path& dir,
                       const std::string& name, const std::string& library) {
  std::filesystem::create_directories(dir);
  const auto file = dir / name;
  std::ofstream out(file);
  out << "{\"file_format_version\":\"1.0.0\",\"ICD\":{"
      << "\"library_path\":\"" << library << "\"}}\n";
  return file.string();
}

int failures = 0;

void expect_throws(const std::string& what, const char* needle,
                   const std::vector<std::string>& candidates,
                   const char* user_files, const std::string& packaged,
                   const char* foreign) {
  try {
    (void)resolve_honeykrisp_icd_detail(candidates, user_files, packaged,
                                        foreign);
    std::cout << "FAIL " << what << " (accepted, expected refusal)\n";
    failures++;
  } catch (const std::exception& e) {
    if (std::string(e.what()).find(needle) == std::string::npos) {
      std::cout << "FAIL " << what << " (refusal message misses '" << needle
                << "': " << e.what() << ")\n";
      failures++;
    } else {
      std::cout << "ok " << what << "\n";
    }
  }
}

void expect_selects(const std::string& what,
                    const HoneykrispIcdSelection& selection,
                    const std::string& path, bool user_override,
                    bool foreign) {
  if (selection.path != path || selection.user_override != user_override ||
      selection.foreign != foreign) {
    std::cout << "FAIL " << what << " (path=" << selection.path
              << " user_override=" << selection.user_override
              << " foreign=" << selection.foreign << ")\n";
    failures++;
  } else {
    std::cout << "ok " << what << "\n";
  }
}

}  // namespace

int main() {
  const auto dir = std::filesystem::temp_directory_path() /
                   "omarchy-identity-hatch-test";
  std::filesystem::remove_all(dir);
  const auto intel = write_json(dir, "intel_icd.json",
                                "/usr/lib/x86_64-linux-gnu/libvulkan_intel.so");
  const auto lvp = write_json(dir, "lavapipe_icd.json",
                              "/usr/lib/x86_64-linux-gnu/libvulkan_lavapipe.so");

  // 1. default-refuses: without the hatch the non-Apple ICD is refused.
  expect_throws("default-refuses", "excludes Honeykrisp", {}, intel.c_str(),
                "", nullptr);
  // and the packaged/system path still resolves when no user override exists
  {
    const auto packaged = write_json(
        dir, "honeykrisp_icd.aarch64.json",
        "/usr/lib/omarchy-mlx/vulkan/libvulkan_honeykrisp.so");
    const auto selection = resolve_honeykrisp_icd_detail({}, nullptr,
                                                         packaged, nullptr);
    expect_selects("packaged-path-unaffected", selection, packaged, false,
                   false);
  }

  // 2. hatch-accepts-named-discrete: hatch == the user ICD path is accepted
  // and marked foreign.
  {
    const auto selection = resolve_honeykrisp_icd_detail(
        {}, intel.c_str(), "", intel.c_str());
    expect_selects("hatch-accepts-named-discrete", selection, intel, true,
                   true);
  }

  // 3. hatch-still-refuses-CPU: a software ICD is refused even when named.
  expect_throws("hatch-still-refuses-CPU", "software", {}, lvp.c_str(), "",
                lvp.c_str());

  // 3b. the real Mesa lavapipe JSON names its library libvulkan_lvp.so;
  // the marker filter must catch that spelling too (wB1 review finding).
  const auto real_lvp = write_json(dir, "lvp_icd.x86_64.json",
                                   "libvulkan_lvp.so");
  expect_throws("hatch-refuses-real-mesa-lvp-json", "software", {},
                real_lvp.c_str(), "", real_lvp.c_str());

  // 3c. a symlink named by the hatch compares against its target, so a
  // symlink that resolves into the marker-less real Mesa lvp JSON is
  // refused: canonicalization sees through the alias (wB1 review finding).
  {
    const auto sub = dir / "alias";
    std::filesystem::create_directories(sub);
    const auto alias = sub / "intel_icd.json";
    std::filesystem::create_symlink(real_lvp, alias);
    expect_throws("hatch-refuses-symlink-to-software-icd", "software", {},
                  alias.string().c_str(), "", alias.string().c_str());
  }

  // hatch mismatch: a user ICD other than the named one stays refused.
  expect_throws("hatch-mismatch-refuses", "does not match", {}, intel.c_str(),
                "", lvp.c_str());

  // honeykrisp names stay accepted with and without the hatch.
  {
    const auto hk = write_json(dir, "honeykrisp_icd.aarch64.json",
                               "/usr/lib/omarchy-mlx/vulkan/libvulkan.so");
    const auto selection =
        resolve_honeykrisp_icd_detail({}, hk.c_str(), "", hk.c_str());
    expect_selects("honeykrisp-override-still-accepted", selection, hk, true,
                   false);
  }

  std::filesystem::remove_all(dir);
  if (failures != 0) {
    std::cout << failures << " failure(s)\n";
    return 1;
  }
  std::cout << "identity-hatch tests pass\n";
  return 0;
}
