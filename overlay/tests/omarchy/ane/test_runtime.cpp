// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

#include "mlx/backend/omarchy/ane/bundle.h"
#include "mlx/backend/omarchy/ane/runtime.h"
#include "mlx/backend/omarchy/ane/runtime_detail.h"
#include "mlx/backend/omarchy/ane/runtime_ownership.h"

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>
#include <sys/stat.h>
#include <unistd.h>
#include <sys/wait.h>

using namespace mlx::core::omarchy::ane;

namespace {

constexpr uint64_t kRuntimePayloadBytes = 0x4000;
const std::array<uint64_t, 6> kRuntimeNchw{1, 64, 1, 1, 64, 64};

template <typename T>
void write_le(std::string& bytes, size_t offset, T value) {
  static_assert(std::is_integral_v<T>);
  REQUIRE(offset + sizeof(T) <= bytes.size());
  std::memcpy(bytes.data() + offset, &value, sizeof(T));
}

void write_nchw(std::string& bytes, uint32_t channel) {
  size_t offset = 40 + kAnecTileCount * sizeof(uint32_t) +
      channel * kRuntimeNchw.size() * sizeof(uint64_t);
  for (uint64_t value : kRuntimeNchw) {
    write_le(bytes, offset, value);
    offset += sizeof(value);
  }
}

std::string runtime_anec_bytes() {
  std::string bytes(kAnecPayloadOffset + kRuntimePayloadBytes, '\0');
  write_le<uint64_t>(bytes, 0, kRuntimePayloadBytes);
  write_le<uint32_t>(bytes, 8, 0x274);
  write_le<uint32_t>(bytes, 12, 1);
  write_le<uint64_t>(bytes, 16, 0x1f8);
  write_le<uint64_t>(bytes, 24, 0x400);
  write_le<uint32_t>(bytes, 32, 2);
  write_le<uint32_t>(bytes, 36, 1);
  write_le<uint32_t>(bytes, 40, 1);
  for (uint32_t channel : {4u, 5u, 6u}) {
    write_le<uint32_t>(bytes, 40 + channel * sizeof(uint32_t), 1);
    write_nchw(bytes, channel);
  }
  return bytes;
}

void write_file(const std::filesystem::path& path, const std::string& bytes) {
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  output.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  REQUIRE(output.good());
}

std::string runtime_tensor(const std::string& name, uint64_t index) {
  return "{\"name\":\"" + name + "\",\"index\":" + std::to_string(index) +
      ",\"dtype\":\"float16\",\"shape\":[1,64,1,1],\"byte_size\":128,"
      "\"stride\":16384}";
}

std::string runtime_binding(const std::string& name, uint32_t channel) {
  return "{\"tensor\":\"" + name + "\",\"channel\":" +
      std::to_string(channel) +
      ",\"dtype\":\"float16\",\"shape\":[1,64,1,1],"
      "\"nchw\":[1,64,1,1,64,64],\"logical_bytes\":128,"
      "\"allocation_bytes\":16384,\"element_offset\":0,"
      "\"element_count\":64,\"physical_elements\":64}";
}

class RuntimeBundle {
 public:
  explicit RuntimeBundle(int manifest_version = 4) {
    std::string path_template =
        (std::filesystem::temp_directory_path() / "mlx-omarchy-runtime-XXXXXX").string();
    char* created = ::mkdtemp(path_template.data());
    if (created == nullptr) {
      throw std::runtime_error("cannot create runtime bundle test directory");
    }
    path_ = created;

    const std::string payload = runtime_anec_bytes();
    const std::string payload_sha = sha256_hex(
        reinterpret_cast<const uint8_t*>(payload.data()), payload.size());
    const std::string payload_record =
        "[{\"byte_size\":" + std::to_string(payload.size()) +
        ",\"path\":\"program-0.anec\",\"role\":\"anec\",\"sha256\":\"" +
        payload_sha + "\"}]";
    const std::string model_sha = sha256_hex(
        reinterpret_cast<const uint8_t*>(payload_record.data()), payload_record.size());
    const std::string manifest =
        R"json({"manifest_version":)json" + std::to_string(manifest_version) +
        R"json(,"name":"runtime-directory-test","graph_hash":")json" +
        std::string(64, '1') +
        R"json(","task_descriptors":1,"inputs":[)json" +
        runtime_tensor("a", 0) + "," + runtime_tensor("b", 1) +
        R"json(],"outputs":[)json" + runtime_tensor("y", 0) +
        R"json(],"logical_results":[{"name":"y","dtype":"float16","shape":[1,64,1,1],"tensor":"y","element_offset":0,"element_count":64,"conversion":"identity"}],"state":[],"intermediates":[],"programs":[{)json"
        "\"payload\":\"program-0.anec\",\"operation\":\"add\","
        "\"encoder\":\"h13-oracle-parity\",\"task_descriptors\":1,"
        "\"scratch_bytes\":0,\"inputs\":[" + runtime_binding("a", 5) + "," +
        runtime_binding("b", 6) + "],\"outputs\":[" + runtime_binding("y", 4) +
        "]}],\"dispatch_plan\":[0],\"payloads\":[{\"role\":\"anec\","
        "\"path\":\"program-0.anec\",\"sha256\":\"" + payload_sha +
        "\",\"byte_size\":" + std::to_string(payload.size()) +
        "}],\"compiler\":{\"host_build\":\"Linux test host\","
        "\"toolchain\":\"mil-hwxc test source\",\"target\":\"h13\"},"
        "\"driver_abi_major\":1,\"provenance\":{"
        "\"source_repo\":\"joshuaswarren/omarchy-mlx\",\"source_commit\":\"" +
        std::string(40, 'c') +
        "\",\"exported_at\":\"2026-09-12\"},\"release_asset\":{"
        "\"model\":\"runtime-directory-test\",\"model_sha256\":\"" +
        model_sha + "\"}}";
    write_file(path_ / "program-0.anec", payload);
    write_file(path_ / "manifest.json", manifest);
  }

  ~RuntimeBundle() {
    std::error_code error;
    std::filesystem::remove_all(path_, error);
  }

  const std::filesystem::path& path() const {
    return path_;
  }

 private:
  std::filesystem::path path_;
};

AneProgramBinding lane_binding() {
  AneProgramBinding binding;
  binding.tensor = "x";
  binding.channel = 5;
  binding.dtype = "float16";
  binding.shape = {1, 64, 1, 1};
  binding.nchw = {1, 64, 1, 1, 64, 64};
  binding.logical_bytes = 128;
  binding.allocation_bytes = 0x4000;
  binding.element_count = 64;
  binding.physical_elements = 64;
  return binding;
}

std::vector<uint8_t> dense_values(size_t elements) {
  std::vector<uint8_t> dense(elements * 2);
  for (size_t i = 0; i < elements; ++i) {
    dense[i * 2] = static_cast<uint8_t>(i);
    dense[i * 2 + 1] = static_cast<uint8_t>(0x80 + i);
  }
  return dense;
}

} // namespace

TEST_CASE("ANE runtime packs and unpacks qualified 64-byte lanes") {
  const auto binding = lane_binding();
  const auto dense = dense_values(64);

  const auto packed = detail::pack_binding(binding, dense);
  REQUIRE(packed.size() == binding.allocation_bytes);
  for (size_t i = 0; i < 64; ++i) {
    CHECK(packed[i * 64] == dense[i * 2]);
    CHECK(packed[i * 64 + 1] == dense[i * 2 + 1]);
    CHECK(std::all_of(
        packed.begin() + i * 64 + 2,
        packed.begin() + (i + 1) * 64,
        [](uint8_t value) { return value == 0; }));
  }

  std::vector<uint8_t> round_trip(dense.size(), 0xff);
  detail::unpack_binding(binding, packed, round_trip);
  CHECK(round_trip == dense);
}

TEST_CASE("ANE runtime preserves tensor regions outside a sliced binding") {
  auto binding = lane_binding();
  binding.shape = {1, 4, 1, 1};
  binding.logical_bytes = 8;
  binding.element_offset = 2;
  binding.element_count = 4;
  binding.physical_elements = 4;
  binding.nchw = {1, 4, 1, 1, 64, 64};
  const auto dense = dense_values(8);

  const auto packed = detail::pack_binding(binding, dense);
  std::vector<uint8_t> restored(dense.size(), 0x5a);
  detail::unpack_binding(binding, packed, restored);

  CHECK(restored[0] == 0x5a);
  CHECK(restored[3] == 0x5a);
  CHECK(std::equal(restored.begin() + 4, restored.begin() + 12, dense.begin() + 4));
  CHECK(restored[12] == 0x5a);
  CHECK(restored.back() == 0x5a);
}

TEST_CASE("ANE runtime honors plane stride for multiple channels") {
  AneProgramBinding binding;
  binding.tensor = "x";
  binding.dtype = "float16";
  binding.shape = {1, 2, 2, 2};
  binding.nchw = {1, 2, 2, 2, 32, 8};
  binding.logical_bytes = 16;
  binding.allocation_bytes = 0x4000;
  binding.element_count = 8;
  binding.physical_elements = 8;
  const auto dense = dense_values(8);

  const auto packed = detail::pack_binding(binding, dense);
  const std::array<size_t, 8> offsets = {0, 2, 8, 10, 32, 34, 40, 42};
  for (size_t i = 0; i < offsets.size(); ++i) {
    CHECK(packed[offsets[i]] == dense[i * 2]);
    CHECK(packed[offsets[i] + 1] == dense[i * 2 + 1]);
  }
  CHECK(packed[16] == 0);

  std::vector<uint8_t> restored(dense.size());
  detail::unpack_binding(binding, packed, restored);
  CHECK(restored == dense);
}

TEST_CASE("ANE runtime rejects invalid staging and deadlines before device work") {
  const auto binding = lane_binding();
  CHECK_THROWS_WITH_AS(
      detail::pack_binding(binding, std::vector<uint8_t>(126)),
      "[omarchy-ane] runtime: tensor 'x' dense staging byte count is 126, expected at least 128.",
      std::runtime_error);
  CHECK_THROWS_WITH_AS(
      detail::checked_deadline(std::chrono::milliseconds(0)),
      "[omarchy-ane] runtime: deadline must be positive.",
      std::invalid_argument);
  CHECK_THROWS_WITH_AS(
      detail::checked_deadline(std::chrono::milliseconds::max()),
      "[omarchy-ane] runtime: deadline exceeds the monotonic clock range.",
      std::invalid_argument);
}

TEST_CASE("ANE runtime validates every bundle directory entry before device work") {
  RuntimeBundle bundle;

  SUBCASE("unknown regular file") {
    write_file(bundle.path() / "extra.bin", "unexpected");
    CHECK_THROWS_WITH_AS(
        AneRuntime::load(bundle.path(), std::chrono::seconds(1), {}),
        "[omarchy-ane] bundle: unknown payload file 'extra.bin' not listed in manifest.",
        std::runtime_error);
  }

  SUBCASE("unknown symlink") {
    std::filesystem::create_symlink(
        bundle.path() / "program-0.anec", bundle.path() / "alias.anec");
    CHECK_THROWS_WITH_AS(
        AneRuntime::load(bundle.path(), std::chrono::seconds(1), {}),
        "[omarchy-ane] bundle: unexpected link 'alias.anec' inside bundle.",
        std::runtime_error);
  }

  SUBCASE("unknown directory") {
    std::filesystem::create_directory(bundle.path() / "nested");
    CHECK_THROWS_WITH_AS(
        AneRuntime::load(bundle.path(), std::chrono::seconds(1), {}),
        "[omarchy-ane] bundle: unexpected directory 'nested' inside bundle.",
        std::runtime_error);
  }

  SUBCASE("positional synthetic ANEC refused under STRICT_BIND before "
          "device work") {
    // The synthetic ANEC carries no task stream, so strict libane
    // (LIBANE_CONFIG_STRICT_BIND) refuses its positional channel map
    // before the diagnostic-path guard is reachable.
    CHECK_THROWS_WITH_AS(
        AneRuntime::load(bundle.path(), std::chrono::seconds(1), {}),
        "[omarchy-ane] bundle: program 0 task stream does not name every "
        "surface; channel map is positional.",
        std::runtime_error);
  }
}

TEST_CASE("ANE runtime rejects old schema before worker initialization") {
  RuntimeBundle bundle(3);
  CHECK_THROWS_WITH_AS(
      AneRuntime::load(bundle.path(), std::chrono::seconds(1), {}),
      "[omarchy-ane] manifest: unsupported manifest_version (expected 4).",
      std::runtime_error);
}

TEST_CASE("ANE output layout API is an immutable runtime-lifetime view") {
  CHECK((std::is_same_v<
         decltype(std::declval<const AneRuntime&>().output_layout()),
         const std::vector<AneLogicalResult>&>));
}

TEST_CASE("ANE installed worker follows the loaded library prefix") {
  CHECK(
      detail::installed_worker_path("/opt/venv/lib/python3.14/site-packages/mlx/lib/libmlx.so") ==
      "/opt/venv/lib/python3.14/site-packages/mlx/bin/mlx-omarchy-ane-worker");
}
TEST_CASE("ANE build worker fallback is restricted to the exact build library") {
  const auto root = std::filesystem::temp_directory_path() /
      ("mlx-omarchy-ane-worker-selection-" + std::to_string(::getpid()));
  const auto installed_library = root / "installed/mlx/lib/libmlx.so";
  const auto build_library = root / "build/libmlx.so";
  const auto build_worker = root / "build/mlx-omarchy-ane-worker";
  const auto installed_worker = detail::installed_worker_path(installed_library);
  const auto stale_derived_worker = detail::installed_worker_path(build_library);
  std::filesystem::remove_all(root);
  std::filesystem::create_directories(installed_library.parent_path());
  std::filesystem::create_directories(build_library.parent_path());
  std::filesystem::create_directories(stale_derived_worker.parent_path());
  std::ofstream(installed_library).put('i');
  std::ofstream(build_library).put('b');
  std::ofstream(build_worker).put('w');
  std::ofstream(stale_derived_worker).put('s');
  REQUIRE(::chmod(build_worker.c_str(), 0700) == 0);
  REQUIRE(::chmod(stale_derived_worker.c_str(), 0700) == 0);

  CHECK_THROWS_WITH_AS(
      detail::worker_executable_path(
          std::filesystem::canonical(installed_library),
          build_library,
          build_worker),
      "[omarchy-ane] runtime: private ANE worker executable not found.",
      std::runtime_error);
  CHECK(
      detail::worker_executable_path(
          std::filesystem::canonical(build_library),
          build_library,
          build_worker) == std::filesystem::canonical(build_worker));
  std::filesystem::remove(build_worker);
  CHECK_THROWS_WITH_AS(
      detail::worker_executable_path(
          std::filesystem::canonical(build_library),
          build_library,
          build_worker),
      "[omarchy-ane] runtime: private ANE worker executable not found.",
      std::runtime_error);

  std::filesystem::create_directories(installed_worker.parent_path());
  std::ofstream(installed_worker).put('w');
  REQUIRE(::chmod(installed_worker.c_str(), 0700) == 0);
  std::filesystem::remove(build_library);
  CHECK(
      detail::worker_executable_path(
          std::filesystem::canonical(installed_library),
          build_library,
          build_worker) == std::filesystem::canonical(installed_worker));
  std::filesystem::remove_all(root);
}
TEST_CASE("ANE worker descriptors stay above every fixed child destination") {
  CHECK(detail::worker_source_fd_floor(0, 64) == 7);
  CHECK(detail::worker_source_fd_floor(3, 64) == 19);
  CHECK_THROWS_WITH_AS(
      detail::worker_source_fd_floor(3, 24),
      "[omarchy-ane] runtime: bundle payload count exceeds worker descriptor limit.",
      std::runtime_error);
}

TEST_CASE("ANE child receives each payload through a collision-free descriptor") {
  const auto root = std::filesystem::temp_directory_path() /
      ("mlx-omarchy-ane-descriptors-" + std::to_string(::getpid()));
  std::filesystem::remove_all(root);
  std::filesystem::create_directory(root);
  constexpr size_t payload_count = 3;
  const int floor = detail::worker_source_fd_floor(payload_count, 64);
  std::array<int, payload_count> sources{};
  for (size_t i = 0; i < payload_count; ++i) {
    const auto path = root / std::to_string(i);
    std::ofstream(path) << static_cast<char>('A' + i);
    const int original = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    REQUIRE(original >= 0);
    sources[i] = ::fcntl(original, F_DUPFD_CLOEXEC, floor);
    ::close(original);
    REQUIRE(sources[i] >= floor);
  }

  const pid_t child = ::fork();
  REQUIRE(child >= 0);
  if (child == 0) {
    for (size_t i = 0; i < payload_count; ++i) {
      if (::dup2(
              sources[i],
              detail::kWorkerPayloadFdBase + static_cast<int>(i)) < 0) {
        ::_exit(10);
      }
    }
    for (size_t i = 0; i < payload_count; ++i) {
      char value = 0;
      const int fd = detail::kWorkerPayloadFdBase + static_cast<int>(i);
      if (::lseek(fd, 0, SEEK_SET) < 0 || ::read(fd, &value, 1) != 1 ||
          value != static_cast<char>('A' + i)) {
        ::_exit(20 + static_cast<int>(i));
      }
    }
    ::_exit(0);
  }
  int status = 0;
  REQUIRE(::waitpid(child, &status, 0) == child);
  for (int fd : sources) {
    ::close(fd);
  }
  CHECK(WIFEXITED(status));
  CHECK(WEXITSTATUS(status) == 0);
  std::filesystem::remove_all(root);
}

TEST_CASE("ANE worker rejects a deadline at the hardware submission boundary") {
  CHECK_THROWS_WITH_AS(
      detail::ensure_worker_before(
          detail::RuntimeClock::now() - std::chrono::milliseconds(1),
          "ANE program execution submission"),
      "[omarchy-ane] runtime: deadline expired before ANE program execution submission.",
      std::runtime_error);
}
TEST_CASE("ANE ownership excludes peers and preserves a same-boot quarantine") {
  const auto root = std::filesystem::temp_directory_path() /
      ("mlx-omarchy-ane-ownership-" + std::to_string(::getpid()));
  std::filesystem::remove_all(root);
  std::filesystem::create_directory(root);
  const auto lock = root / "runtime.lock";
  const auto state = root / "quarantine";
  const std::string boot_a = "11111111-1111-1111-1111-111111111111";
  const std::string boot_b = "22222222-2222-2222-2222-222222222222";
  struct stat first_state {};

  {
    auto owner = detail::RuntimeOwnership::acquire_at(lock, state, boot_a);
    CHECK_THROWS_WITH_AS(
        detail::RuntimeOwnership::acquire_at(lock, state, boot_a),
        "[omarchy-ane] runtime: another ANE runtime owns the host device.",
        std::runtime_error);
    const pid_t peer = ::fork();
    REQUIRE(peer >= 0);
    if (peer == 0) {
      try {
        auto second = detail::RuntimeOwnership::acquire_at(lock, state, boot_a);
        (void)second;
        ::_exit(2);
      } catch (const std::runtime_error& error) {
        ::_exit(std::string(error.what()) ==
                "[omarchy-ane] runtime: another ANE runtime owns the host device."
            ? 0
            : 3);
      }
    }
    int peer_status = 0;
    REQUIRE(::waitpid(peer, &peer_status, 0) == peer);
    CHECK(WIFEXITED(peer_status));
    CHECK(WEXITSTATUS(peer_status) == 0);
    REQUIRE(::stat(state.c_str(), &first_state) == 0);
    owner.release_cleanly();
  }
  struct stat clean_state {};
  REQUIRE(::stat(state.c_str(), &clean_state) == 0);
  CHECK(clean_state.st_dev == first_state.st_dev);
  CHECK(clean_state.st_ino == first_state.st_ino);
  CHECK(std::filesystem::exists(state));
  CHECK(std::filesystem::file_size(state) == 0);
  {
    auto owner = detail::RuntimeOwnership::acquire_at(lock, state, boot_a);
    owner.arm();
    owner.quarantine();
  }
  CHECK_THROWS_WITH_AS(
      detail::RuntimeOwnership::acquire_at(lock, state, boot_a),
      "[omarchy-ane] runtime: ANE runtime is quarantined for this boot; reboot is required.",
      std::runtime_error);
  {
    auto owner = detail::RuntimeOwnership::acquire_at(lock, state, boot_b);
    owner.release_cleanly();
  }
  CHECK(std::filesystem::exists(state));
  CHECK(std::filesystem::file_size(state) == 0);

  std::ofstream(state) << std::string(129, 'x');
  CHECK_THROWS_WITH_AS(
      detail::RuntimeOwnership::acquire_at(lock, state, boot_a),
      "[omarchy-ane] runtime: ANE ownership state is invalid.",
      std::runtime_error);
  CHECK(std::filesystem::exists(state));
  CHECK_THROWS_WITH_AS(
      detail::RuntimeOwnership::acquire_at(lock, state, boot_a),
      "[omarchy-ane] runtime: ANE ownership state is invalid.",
      std::runtime_error);
  std::filesystem::remove_all(root);
}

TEST_CASE("ANE shared ownership paths require frozen provisioned identities") {
  const auto root = std::filesystem::temp_directory_path() /
      ("mlx-omarchy-ane-shared-ownership-" + std::to_string(::getpid()));
  std::filesystem::remove_all(root);
  std::filesystem::create_directory(root);
  REQUIRE(::chmod(root.c_str(), 01777) == 0);
  const auto lock = root / "device.lock";
  const auto state = root / "quarantine";
  std::ofstream{lock};
  std::ofstream{state};
  REQUIRE(::chmod(lock.c_str(), 0666) == 0);
  REQUIRE(::chmod(state.c_str(), 0666) == 0);

  detail::RuntimeOwnership::validate_shared_paths_at(
      root, lock, state, ::geteuid(), ::getegid());
  REQUIRE(::chmod(root.c_str(), 03777) == 0);
  CHECK_THROWS_AS(
      detail::RuntimeOwnership::validate_shared_paths_at(
          root, lock, state, ::geteuid(), ::getegid()),
      std::runtime_error);
  REQUIRE(::chmod(root.c_str(), 01777) == 0);
  REQUIRE(::chmod(lock.c_str(), 0640) == 0);
  CHECK_THROWS_WITH_AS(
      detail::RuntimeOwnership::validate_shared_paths_at(
          root, lock, state, ::geteuid(), ::getegid()),
      doctest::String{
          ("[omarchy-ane] runtime: ANE ownership state is not a provisioned shared regular file: " +
           lock.string() + ".")
              .c_str()},
      std::runtime_error);
  REQUIRE(::chmod(lock.c_str(), 0666) == 0);
  std::filesystem::remove(state);
  std::filesystem::create_symlink(lock, state);
  CHECK_THROWS_WITH_AS(
      detail::RuntimeOwnership::validate_shared_paths_at(
          root, lock, state, ::geteuid(), ::getegid()),
      doctest::String{
          ("[omarchy-ane] runtime: ANE ownership state is not a provisioned shared regular file: " +
           state.string() + ".")
              .c_str()},
      std::runtime_error);
  std::filesystem::remove_all(root);
}
