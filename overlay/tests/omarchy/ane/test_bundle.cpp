// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

#include "mlx/backend/omarchy/ane/bundle.h"
#include "json.hpp"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <system_error>
#include <type_traits>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

using namespace mlx::core::omarchy::ane;

namespace {

constexpr uint64_t kPayloadBytes = 0x4000;
constexpr uint64_t kAllocationBytes = 0x4000;
const std::array<uint64_t, 6> kNchw{1, 64, 1, 1, 64, 64};

std::string hex(size_t size, char value) {
  return std::string(size, value);
}

template <typename T>
void write_le(std::string& bytes, size_t offset, T value) {
  static_assert(std::is_integral_v<T>);
  REQUIRE(offset + sizeof(T) <= bytes.size());
  std::memcpy(bytes.data() + offset, &value, sizeof(T));
}

void write_nchw(
    std::string& bytes,
    uint32_t channel,
    const std::array<uint64_t, 6>& nchw = kNchw) {
  size_t offset = 40 + kAnecTileCount * sizeof(uint32_t) +
      channel * nchw.size() * sizeof(uint64_t);
  for (uint64_t value : nchw) {
    write_le(bytes, offset, value);
    offset += sizeof(value);
  }
}
void write_h13_task(
    std::string& bytes,
    uint32_t selectors,
    uint32_t src1_cfg = 0x00033881u,
    uint32_t src2_cfg = 0x00033881u,
    uint32_t dst_cfg = 0x040000c1u) {
  uint32_t td_size = 0;
  std::memcpy(&td_size, bytes.data() + 8, sizeof(td_size));
  REQUIRE(td_size >= 64);
  REQUIRE(td_size % 8 == 0);
  REQUIRE(kAnecPayloadOffset + td_size <= bytes.size());
  std::fill(
      bytes.begin() + kAnecPayloadOffset,
      bytes.begin() + kAnecPayloadOffset + td_size,
      0);
  write_le<uint32_t>(
      bytes,
      kAnecPayloadOffset + 4,
      ((td_size / 4 - 1) & 0x1ff) << 16);
  write_le<uint32_t>(bytes, kAnecPayloadOffset + 8 * 4, selectors);
  const uint32_t registers[3] = {0x13800u, 0x13804u, 0x17800u};
  const uint32_t configs[3] = {src1_cfg, src2_cfg, dst_cfg};
  size_t at = kAnecPayloadOffset + 40;
  for (int slot = 0; slot < 3; ++slot) {
    write_le<uint32_t>(bytes, at, registers[slot]);
    write_le<uint32_t>(bytes, at + 4, configs[slot]);
    at += 8;
  }
  while (at + 8 <= kAnecPayloadOffset + td_size) {
    at += 8;
  }
  REQUIRE(at == kAnecPayloadOffset + td_size);
}



std::string anec_bytes(char seed) {
  std::string bytes(kAnecPayloadOffset + kPayloadBytes, '\0');
  for (uint64_t i = 0; i < kPayloadBytes; ++i) {
    bytes[kAnecPayloadOffset + i] = static_cast<char>(seed + (i % 7));
  }
  write_le<uint64_t>(bytes, 0, kPayloadBytes);
  write_le<uint32_t>(bytes, 8, 64);
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
  write_h13_task(bytes, 0x00024966u);
  return bytes;

}

nlohmann::json tensor(const std::string& name, uint64_t index) {
  return {
      {"name", name},
      {"index", index},
      {"dtype", "float16"},
      {"shape", {1, 64, 1, 1}},
      {"byte_size", 128},
      {"stride", kAllocationBytes},
  };
}
nlohmann::json logical_result(
    const std::string& name,
    const std::string& tensor_name,
    std::vector<uint64_t> shape = {1, 64, 1, 1},
    uint64_t element_offset = 0,
    uint64_t element_count = 64) {
  return {
      {"name", name},
      {"dtype", "float16"},
      {"shape", std::move(shape)},
      {"tensor", tensor_name},
      {"element_offset", element_offset},
      {"element_count", element_count},
      {"conversion", "identity"},
  };
}

nlohmann::json binding(const std::string& name, uint64_t channel) {
  return {
      {"tensor", name},
      {"channel", channel},
      {"dtype", "float16"},
      {"shape", {1, 64, 1, 1}},
      {"nchw", kNchw},
      {"logical_bytes", 128},
      {"allocation_bytes", kAllocationBytes},
      {"element_offset", 0},
      {"element_count", 64},
      {"physical_elements", 64},
  };
}

nlohmann::json program(
    const std::string& payload,
    const std::string& operation,
    const std::array<std::string, 2>& inputs,
    const std::string& output) {
  return {
      {"payload", payload},
      {"operation", operation},
      {"encoder", "h13-oracle-parity"},
      {"task_descriptors", 1},
      {"scratch_bytes", 0},
      {"inputs", {binding(inputs[0], 5), binding(inputs[1], 6)}},
      {"outputs", {binding(output, 4)}},
  };
}

class TempDir {
 public:
  TempDir() {
    auto base = std::filesystem::temp_directory_path() / "mlx-omarchy-ane-bundle-test";
    std::filesystem::create_directories(base);
    for (int attempt = 0; attempt < 64; ++attempt) {
      auto candidate = base /
          ("case-" + std::to_string(::getpid()) + "-" + std::to_string(attempt));
      std::error_code error;
      if (std::filesystem::create_directory(candidate, error) && !error) {
        path_ = std::move(candidate);
        return;
      }
    }
    throw std::runtime_error("cannot create test directory");
  }

  ~TempDir() {
    std::error_code error;
    std::filesystem::remove_all(path_, error);
  }

  const std::filesystem::path& path() const {
    return path_;
  }

 private:
  std::filesystem::path path_;
};

void write_file(const std::filesystem::path& path, const std::string& bytes) {
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  output.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  REQUIRE(output.good());
}

std::string payload_collection_identity(const nlohmann::json& payloads) {
  nlohmann::json records = payloads;
  std::sort(records.begin(), records.end(), [](const auto& lhs, const auto& rhs) {
    return lhs.at("path").template get<std::string>() <
        rhs.at("path").template get<std::string>();
  });
  const std::string encoded = records.dump(-1, ' ', true);
  return sha256_hex(
      reinterpret_cast<const uint8_t*>(encoded.data()), encoded.size());
}

struct Fixture {
  TempDir dir;
  std::array<std::string, 2> payload_bytes{anec_bytes('A'), anec_bytes('K')};
  nlohmann::json manifest;

  explicit Fixture(uint64_t tile_shift = kAneTileShiftDefault) {
    // Same byte allocations under any tile unit: the header's tiles[] counts
    // are denominated in 1<<tile_shift byte units, so a shift-9 header scales
    // each count by 32 to describe the identical channel bytes.
    const uint32_t count = uint32_t(1) << (14 - tile_shift);
    for (auto& payload : payload_bytes) {
      for (uint32_t channel : {0u, 4u, 5u, 6u}) {
        write_le<uint32_t>(
            payload, 40 + channel * sizeof(uint32_t), count);
      }
    }
    manifest = {
        {"manifest_version", 4},
        {"tile_shift", tile_shift},
        {"name", "h13-chain-add-mul"},
        {"graph_hash", hex(64, '1')},
        {"task_descriptors", 2},
        {"inputs", {tensor("a", 0), tensor("b", 1)}},
        {"outputs", {tensor("y", 0)}},
        {"logical_results",
         {logical_result("matrix", "y", {2, 32}),
          logical_result("tail", "y", {8}, 56, 8),
          logical_result("tail", "y", {8}, 56, 8)}},
        {"state", nlohmann::json::array()},
        {"intermediates", {tensor("sum", 0)}},
        {"programs",
         {program("program-0.anec", "add", {"a", "b"}, "sum"),
          program("program-1.anec", "mul", {"sum", "b"}, "y")}},
        {"dispatch_plan", {0, 1}},
        {"payloads",
         {{{"role", "anec"},
           {"path", "program-0.anec"},
           {"sha256", digest(0)},
           {"byte_size", payload_bytes[0].size()}},
          {{"role", "anec"},
           {"path", "program-1.anec"},
           {"sha256", digest(1)},
           {"byte_size", payload_bytes[1].size()}}}},
        {"compiler",
         {{"host_build", "Linux test host"},
          {"toolchain", "mil-hwxc test source"},
          {"target", "h13"}}},
        {"driver_abi_major", 1},
        {"provenance",
         {{"source_repo", "joshuaswarren/mlx-omarchy"},
          {"source_commit", hex(40, 'c')},
          {"exported_at", "2026-09-12"}}},
        {"release_asset",
         {{"model", "h13-chain-add-mul"},
          {"model_sha256", ""}}},
    };
    manifest["release_asset"]["model_sha256"] =
        payload_collection_identity(manifest["payloads"]);
  }

  std::string digest(size_t index) const {
    const auto& bytes = payload_bytes.at(index);
    return sha256_hex(reinterpret_cast<const uint8_t*>(bytes.data()), bytes.size());
  }

  void refresh_payload(size_t index) {
    manifest["payloads"][index]["sha256"] = digest(index);
    manifest["payloads"][index]["byte_size"] = payload_bytes[index].size();
    manifest["release_asset"]["model_sha256"] =
        payload_collection_identity(manifest["payloads"]);
  }

  void write() const {
    write_file(dir.path() / "program-0.anec", payload_bytes[0]);
    write_file(dir.path() / "program-1.anec", payload_bytes[1]);
    write_file(dir.path() / "manifest.json", manifest.dump(2) + "\n");
  }
};

template <typename Function>
void check_error(Function&& function, const std::string& expected) {
  try {
    function();
    FAIL("expected exception");
  } catch (const std::exception& error) {
    CHECK(std::string(error.what()).find(expected) != std::string::npos);
  }
}

} // namespace

TEST_CASE("sha256 digests survive the active compress path") {
  // The compress step dispatches at runtime (ARMv8 crypto when the CPU
  // reports it, the scalar path otherwise). These FIPS 180-4 answers pin
  // whichever path the host took, across the padding boundaries where a
  // block-compression rewrite breaks first.
  CHECK(sha256_hex(nullptr, 0) ==
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
  CHECK(sha256_hex(reinterpret_cast<const uint8_t*>("abc"), 3) ==
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
  std::string big(1000, 'a');
  CHECK(sha256_hex(reinterpret_cast<const uint8_t*>(big.data()), big.size()) ==
        "41edece42d63e8d9bf515a9ba6932e1c20cbc9f5a5d134645adb5db1b9737ea3");
  for (const auto& [size, want] : std::array<std::pair<size_t, const char*>, 9>{
           {{1, "ca978112ca1bbdcafac231b39a23dc4da786eff8147c4e72b9807785afee48bb"},
            {3, "9834876dcfb05cb167a5c24953eba58c4ac89b1adf57f28f2f9d09af107ee8f0"},
            {55, "9f4390f8d30c2dd92ec9f095b65e2b9ae9b0a925a5258e241c9f1e910f734318"},
            {56, "b35439a4ac6f0948b6d6f9e3c6af0f5f590ce20f1bde7090ef7970686ec6738a"},
            {63, "7d3e74a05d7db15bce4ad9ec0658ea98e3f06eeecf16b4c6fff2da457ddc2f34"},
            {64, "ffe054fe7ae0cb6dc65c3af9b61d5209f439851db43d0ba5997337df154668eb"},
            {65, "635361c48bb9eab14198e76ea8ab7f1a41685d6ad62aa9146d301d4f17eb0ae0"},
            {119, "31eba51c313a5c08226adf18d4a359cfdfd8d2e816b13f4af952f7ea6584dcfb"},
            {120, "2f3d335432c70b580af0e8e1b3674a7c020d683aa5f73aaaedfdc55af904c21c"}}}) {
    std::string block(size, 'a');
    INFO("size=", size);
    CHECK(sha256_hex(reinterpret_cast<const uint8_t*>(block.data()),
                     block.size()) == want);
  }
  std::string tail(1024 * 1024, '\x5a');
  CHECK(sha256_hex(reinterpret_cast<const uint8_t*>(tail.data()), tail.size()) ==
        "bf63d8a95fcc2e64619813aae35fdcbe871fdd9264caa3f365eb3aed0f679129");
}

TEST_CASE("valid multi-program bundle preserves dispatch and bindings") {
  Fixture fixture;
  fixture.write();
  AneBundle bundle = load_bundle(fixture.dir.path());
  REQUIRE(bundle.programs.size() == 2);
  CHECK(bundle.manifest.manifest_version == 4);
  CHECK(bundle.manifest.driver_abi_major == 1);
  CHECK(bundle.manifest.dispatch_plan == std::vector<uint64_t>{0, 1});
  CHECK(bundle.manifest.programs[bundle.programs[0].manifest_index].inputs[0].tensor == "a");
  CHECK(bundle.manifest.programs[bundle.programs[1].manifest_index].inputs[0].tensor == "sum");
  CHECK(bundle.programs[1].anec_header.source_count == 2);
  REQUIRE(bundle.manifest.logical_results.size() == 3);
  CHECK(bundle.manifest.logical_results[0].name == "matrix");
  CHECK(bundle.manifest.logical_results[0].shape == std::vector<uint64_t>{2, 32});
  CHECK(bundle.manifest.logical_results[1].name == "tail");
  CHECK(bundle.manifest.logical_results[1].element_offset == 56);
  CHECK(bundle.manifest.logical_results[2].name == "tail");
  CHECK(bundle.manifest.logical_results[2].tensor == "y");
}

TEST_CASE("snapshot loader uses supplied immutable payload paths") {
  Fixture caller;
  caller.write();
  TempDir snapshot;
  write_file(snapshot.path() / "manifest.json", caller.manifest.dump(2) + "\n");
  write_file(snapshot.path() / "program-0.anec", caller.payload_bytes[0]);
  write_file(snapshot.path() / "program-1.anec", caller.payload_bytes[1]);
  std::map<std::string, std::filesystem::path> payloads{
      {"program-0.anec", snapshot.path() / "program-0.anec"},
      {"program-1.anec", snapshot.path() / "program-1.anec"}};

  AneBundle loaded =
      load_bundle_snapshot(snapshot.path() / "manifest.json", payloads);
  write_file(caller.dir.path() / "program-0.anec", anec_bytes('Z'));

  CHECK(sha256_file(loaded.programs[0].anec) == caller.digest(0));
  CHECK(loaded.programs[0].anec.parent_path() == snapshot.path());
}

TEST_CASE("release identity uses the producer's canonical payload byte domain") {
  SUBCASE("mismatch") {
    Fixture fixture;
    fixture.manifest["release_asset"]["model_sha256"] = hex(64, '4');
    fixture.write();
    check_error(
        [&] { load_bundle(fixture.dir.path()); },
        "model_sha256 does not match compiled payload collection");
  }

  SUBCASE("non-ASCII paths use escaped canonical JSON") {
    Fixture fixture;
    const std::string renamed = "prögram-0.anec";
    fixture.manifest["programs"][0]["payload"] = renamed;
    fixture.manifest["payloads"][0]["path"] = renamed;
    fixture.manifest["release_asset"]["model_sha256"] =
        payload_collection_identity(fixture.manifest["payloads"]);
    fixture.write();
    std::filesystem::rename(fixture.dir.path() / "program-0.anec",
                            fixture.dir.path() / renamed);
    CHECK_NOTHROW(load_bundle(fixture.dir.path()));
  }
}

TEST_CASE("old schemas are rejected without compatibility shim") {
  Fixture fixture;
  fixture.manifest["manifest_version"] = 3;
  fixture.write();
  check_error([&] { load_bundle(fixture.dir.path()); }, "unsupported manifest_version");
}
TEST_CASE("manifest v4 requires strict ordered logical results") {
  SUBCASE("unreferenced physical output") {
    Fixture fixture;
    fixture.manifest["outputs"].push_back(tensor("sum", 1));
    fixture.manifest["intermediates"] = nlohmann::json::array();
    fixture.manifest["programs"][1]["inputs"][0]["tensor"] = "a";
    fixture.write();
    check_error(
        [&] { load_bundle(fixture.dir.path()); },
        "logical_results must reference every physical output");
  }
  SUBCASE("missing list") {
    Fixture fixture;
    fixture.manifest.erase("logical_results");
    fixture.write();
    check_error(
        [&] { load_bundle(fixture.dir.path()); },
        "missing field 'logical_results'");
  }
  SUBCASE("empty list") {
    Fixture fixture;
    fixture.manifest["logical_results"] = nlohmann::json::array();
    fixture.write();
    check_error(
        [&] { load_bundle(fixture.dir.path()); },
        "field 'logical_results' must be a non-empty array");
  }
  SUBCASE("missing member") {
    Fixture fixture;
    fixture.manifest["logical_results"][0].erase("conversion");
    fixture.write();
    check_error([&] { load_bundle(fixture.dir.path()); }, "missing field 'conversion'");
  }
  SUBCASE("unknown member") {
    Fixture fixture;
    fixture.manifest["logical_results"][0]["physical_elements"] = 64;
    fixture.write();
    check_error(
        [&] { load_bundle(fixture.dir.path()); },
        "unknown field 'physical_elements'");
  }
}

TEST_CASE("logical results reject invalid physical views") {
  SUBCASE("unknown physical tensor") {
    Fixture fixture;
    fixture.manifest["logical_results"][0]["tensor"] = "missing";
    fixture.write();
    check_error(
        [&] { load_bundle(fixture.dir.path()); },
        "references unknown physical output tensor 'missing'");
  }
  SUBCASE("dtype mismatch") {
    Fixture fixture;
    fixture.manifest["logical_results"][0]["dtype"] = "bfloat16";
    fixture.write();
    check_error(
        [&] { load_bundle(fixture.dir.path()); },
        "dtype does not match physical output tensor 'y'");
  }
  SUBCASE("zero element count") {
    Fixture fixture;
    fixture.manifest["logical_results"][0]["element_count"] = 0;
    fixture.write();
    check_error([&] { load_bundle(fixture.dir.path()); }, "field 'element_count' must be positive");
  }
  SUBCASE("zero shape dimension") {
    Fixture fixture;
    fixture.manifest["logical_results"][0]["shape"] = {0, 64};
    fixture.write();
    check_error([&] { load_bundle(fixture.dir.path()); }, "must contain positive integers");
  }
  SUBCASE("shape product overflow") {
    Fixture fixture;
    fixture.manifest["logical_results"][0]["shape"] = {
        std::numeric_limits<uint64_t>::max(), 2};
    fixture.write();
    check_error([&] { load_bundle(fixture.dir.path()); }, "geometry overflows uint64");
  }
  SUBCASE("shape product differs from slice") {
    Fixture fixture;
    fixture.manifest["logical_results"][0]["shape"] = {8};
    fixture.write();
    check_error(
        [&] { load_bundle(fixture.dir.path()); },
        "shape does not match element_count");
  }
  SUBCASE("maximal offset cannot wrap past storage") {
    Fixture fixture;
    fixture.manifest["logical_results"][0]["element_offset"] =
        std::numeric_limits<uint64_t>::max();
    fixture.write();
    check_error(
        [&] { load_bundle(fixture.dir.path()); },
        "range exceeds physical output tensor 'y'");
  }
  SUBCASE("out of bounds") {
    Fixture fixture;
    fixture.manifest["logical_results"][0] =
        logical_result("bad", "y", {8}, 60, 8);
    fixture.write();
    check_error(
        [&] { load_bundle(fixture.dir.path()); },
        "range exceeds physical output tensor 'y'");
  }
  SUBCASE("unsupported conversion") {
    Fixture fixture;
    fixture.manifest["logical_results"][0]["conversion"] = "cast";
    fixture.write();
    check_error(
        [&] { load_bundle(fixture.dir.path()); },
        "unsupported conversion 'cast'");
  }
}

TEST_CASE("removed firmware field is rejected") {
  Fixture fixture;
  fixture.manifest["firmware"] = {{"min", "13.0"}, {"max", "13.5"}};
  fixture.write();
  check_error([&] { load_bundle(fixture.dir.path()); }, "unknown field 'firmware'");
}

TEST_CASE("driver ABI is required, unsigned, and exact") {
  SUBCASE("missing") {
    Fixture fixture;
    fixture.manifest.erase("driver_abi_major");
    fixture.write();
    check_error([&] { load_bundle(fixture.dir.path()); }, "missing field 'driver_abi_major'");
  }
  SUBCASE("signed") {
    Fixture fixture;
    fixture.manifest["driver_abi_major"] = -1;
    fixture.write();
    check_error([&] { load_bundle(fixture.dir.path()); }, "must be a non-negative integer");
  }
  SUBCASE("wrong major") {
    Fixture fixture;
    fixture.manifest["driver_abi_major"] = 2;
    fixture.write();
    check_error([&] { load_bundle(fixture.dir.path()); }, "unsupported driver_abi_major 2");
  }
}

TEST_CASE("dispatch plan must be a complete permutation") {
  Fixture fixture;
  fixture.manifest["dispatch_plan"] = {0, 0};
  fixture.write();
  check_error([&] { load_bundle(fixture.dir.path()); }, "must be a permutation");
}

TEST_CASE("dispatch plan rejects use-before-write") {
  Fixture fixture;
  fixture.manifest["dispatch_plan"] = {1, 0};
  fixture.write();
  check_error([&] { load_bundle(fixture.dir.path()); }, "reads tensor 'sum' before its range is written");
}
TEST_CASE("declared final outputs require complete dispatched range coverage") {
  const std::array<uint64_t, 6> physical_128{1, 128, 1, 1, 64, 64};

  SUBCASE("a gap is rejected") {
    Fixture fixture;
    fixture.manifest["outputs"][0]["shape"] = {1, 128, 1, 1};
    fixture.manifest["outputs"][0]["byte_size"] = 256;
    auto& output = fixture.manifest["programs"][1]["outputs"][0];
    output["nchw"] = physical_128;
    output["physical_elements"] = 128;
    write_nchw(fixture.payload_bytes[1], 4, physical_128);
    fixture.refresh_payload(1);
    fixture.write();
    check_error(
        [&] { load_bundle(fixture.dir.path()); },
        "output tensor 'y' is not fully written");
  }

  SUBCASE("disjoint program tiles can cover the output") {
    Fixture fixture;
    fixture.manifest["outputs"][0]["shape"] = {1, 128, 1, 1};
    fixture.manifest["outputs"][0]["byte_size"] = 256;
    fixture.manifest["intermediates"] = nlohmann::json::array();
    for (size_t p = 0; p < 2; ++p) {
      fixture.manifest["programs"][p]["inputs"][0]["tensor"] = "a";
      auto& output = fixture.manifest["programs"][p]["outputs"][0];
      output["tensor"] = "y";
      output["nchw"] = physical_128;
      output["physical_elements"] = 128;
      output["element_offset"] = p * 64;
      write_nchw(fixture.payload_bytes[p], 4, physical_128);
      fixture.refresh_payload(p);
    }
    fixture.write();
    CHECK_NOTHROW(load_bundle(fixture.dir.path()));
  }
}

TEST_CASE("each ANEC payload must map to one program") {
  Fixture fixture;
  fixture.manifest["programs"][1]["payload"] = "program-0.anec";
  fixture.write();
  check_error([&] { load_bundle(fixture.dir.path()); }, "referenced more than once");
}

TEST_CASE("bindings reject unknown tensors and invalid ranges") {
  SUBCASE("unknown tensor") {
    Fixture fixture;
    fixture.manifest["programs"][0]["inputs"][0]["tensor"] = "missing";
    fixture.write();
    check_error([&] { load_bundle(fixture.dir.path()); }, "references unknown tensor 'missing'");
  }
  SUBCASE("range") {
    Fixture fixture;
    fixture.manifest["programs"][0]["inputs"][0]["element_offset"] = 1;
    fixture.write();
    check_error([&] { load_bundle(fixture.dir.path()); }, "range or allocation exceeds tensor 'a'");
  }
}
TEST_CASE("binding physical geometry is exact while slices may be smaller") {
  SUBCASE("physical elements must equal NCHW elements") {
    for (uint64_t physical : {uint64_t{1024}, uint64_t{20000}}) {
      Fixture fixture;
      fixture.manifest["programs"][0]["inputs"][0]["physical_elements"] = physical;
      fixture.write();
      check_error(
          [&] { load_bundle(fixture.dir.path()); },
          "physical_elements does not match NCHW geometry");
    }
  }

  SUBCASE("a valid slice can be smaller than physical geometry") {
    Fixture fixture;
    const std::array<uint64_t, 6> physical_512{1, 512, 1, 1, 32, 32};
    fixture.manifest["inputs"][0]["shape"] = {1, 1024, 1, 1};
    fixture.manifest["inputs"][0]["byte_size"] = 2048;
    auto& input = fixture.manifest["programs"][0]["inputs"][0];
    input["shape"] = {1, 384, 1, 1};
    input["nchw"] = physical_512;
    input["logical_bytes"] = 768;
    input["element_offset"] = 512;
    input["element_count"] = 384;
    input["physical_elements"] = 512;
    write_nchw(fixture.payload_bytes[0], 5, physical_512);
    fixture.refresh_payload(0);
    fixture.write();
    CHECK_NOTHROW(load_bundle(fixture.dir.path()));
  }
}

TEST_CASE("ordinary tensors cannot use zero geometry") {
  Fixture fixture;
  fixture.manifest["inputs"][0]["shape"] = {0};
  fixture.manifest["inputs"][0]["byte_size"] = 0;
  fixture.write();
  check_error([&] { load_bundle(fixture.dir.path()); }, "must contain positive integers");
}

TEST_CASE("binding channel mapping must match ANEC order") {
  Fixture fixture;
  fixture.manifest["programs"][0]["inputs"][0]["channel"] = 6;
  fixture.manifest["programs"][0]["inputs"][1]["channel"] = 5;
  fixture.write();
  check_error([&] { load_bundle(fixture.dir.path()); }, "channel does not match ANEC binding order");
}


TEST_CASE("derived reverse map rejects a positional declaration") {
  Fixture fixture;
  write_le<uint32_t>(fixture.payload_bytes[0], 32, 1);
  write_h13_task(fixture.payload_bytes[0], 0x00025864u);
  fixture.manifest["programs"][0]["inputs"] = nlohmann::json::array({binding("a", 5)});
  fixture.refresh_payload(0);
  fixture.write();
  check_error(
      [&] { load_bundle(fixture.dir.path()); },
      "channel does not match ANEC binding order");
}

TEST_CASE("derived reverse map accepts the task-stream channels") {
  Fixture fixture;
  write_le<uint32_t>(fixture.payload_bytes[0], 32, 1);
  write_h13_task(fixture.payload_bytes[0], 0x00025864u);
  fixture.manifest["programs"][0]["inputs"] = nlohmann::json::array({binding("a", 4)});
  fixture.manifest["programs"][0]["outputs"] = nlohmann::json::array({binding("sum", 5)});
  fixture.refresh_payload(0);
  fixture.write();
  CHECK_NOTHROW(load_bundle(fixture.dir.path()));
}


TEST_CASE("binding allocation must match ANEC channel") {
  Fixture fixture;
  fixture.manifest["programs"][0]["inputs"][0]["allocation_bytes"] = 0x8000;
  fixture.manifest["inputs"][0]["stride"] = 0x8000;
  fixture.write();
  check_error([&] { load_bundle(fixture.dir.path()); }, "allocation_bytes does not match ANEC");
}

TEST_CASE("program task count and scratch allocation match each ANEC") {
  SUBCASE("task descriptors") {
    Fixture fixture;
    fixture.manifest["programs"][0]["task_descriptors"] = 2;
    fixture.manifest["task_descriptors"] = 3;
    fixture.write();
    check_error([&] { load_bundle(fixture.dir.path()); }, "task_descriptors does not match ANEC");
  }
  SUBCASE("scratch") {
    Fixture fixture;
    fixture.manifest["programs"][0]["scratch_bytes"] = 1;
    fixture.write();
    check_error([&] { load_bundle(fixture.dir.path()); }, "scratch_bytes does not match ANEC");
  }
  SUBCASE("understated positive scratch") {
    Fixture fixture;
    write_le<uint32_t>(fixture.payload_bytes[0], 40 + 3 * sizeof(uint32_t), 1);
    fixture.refresh_payload(0);
    fixture.manifest["programs"][0]["scratch_bytes"] = 1;
    fixture.write();
    check_error([&] { load_bundle(fixture.dir.path()); }, "scratch_bytes does not match ANEC");
  }
  SUBCASE("exact positive scratch") {
    Fixture fixture;
    write_le<uint32_t>(fixture.payload_bytes[0], 40 + 3 * sizeof(uint32_t), 1);
    fixture.refresh_payload(0);
    fixture.manifest["programs"][0]["scratch_bytes"] = kAllocationBytes;
    fixture.write();
    CHECK_NOTHROW(load_bundle(fixture.dir.path()));
  }
}

TEST_CASE("ANEC task descriptor fields stay within driver submit limits") {
  SUBCASE("task descriptor count accepts 0xffff") {
    Fixture fixture;
    write_le<uint32_t>(fixture.payload_bytes[0], 12, 0xffff);
    fixture.manifest["programs"][0]["task_descriptors"] = 0xffff;
    fixture.manifest["task_descriptors"] = 0x10000;
    fixture.refresh_payload(0);
    fixture.write();
    CHECK_NOTHROW(load_bundle(fixture.dir.path()));
  }

  SUBCASE("task descriptor count rejects 0x10000") {
    Fixture fixture;
    write_le<uint32_t>(fixture.payload_bytes[0], 12, 0x10000);
    fixture.manifest["programs"][0]["task_descriptors"] = 0x10000;
    fixture.manifest["task_descriptors"] = 0x10001;
    fixture.refresh_payload(0);
    fixture.write();
    check_error(
        [&] { load_bundle(fixture.dir.path()); },
        "ANEC task descriptor count exceeds driver limit 0xffff");
  }

  SUBCASE("task descriptor size accepts 0x40000") {
    Fixture fixture;
    fixture.payload_bytes[0].resize(kAnecPayloadOffset + 0x40000);
    write_le<uint64_t>(fixture.payload_bytes[0], 0, 0x40000);
    write_le<uint32_t>(fixture.payload_bytes[0], 8, 0x40000);
    write_le<uint32_t>(fixture.payload_bytes[0], 40, 16);
    write_h13_task(fixture.payload_bytes[0], 0x00024966u);
    fixture.refresh_payload(0);
    fixture.write();
    CHECK_NOTHROW(load_bundle(fixture.dir.path()));
  }

  SUBCASE("task descriptor size rejects 0x40004") {
    Fixture fixture;
    fixture.payload_bytes[0].resize(kAnecPayloadOffset + 0x40004);
    write_le<uint64_t>(fixture.payload_bytes[0], 0, 0x40004);
    write_le<uint32_t>(fixture.payload_bytes[0], 8, 0x40004);
    write_le<uint32_t>(fixture.payload_bytes[0], 40, 17);
    fixture.refresh_payload(0);
    fixture.write();
    check_error(
        [&] { load_bundle(fixture.dir.path()); },
        "ANEC task descriptor size exceeds driver limit 0x40000");
  }
}

TEST_CASE("all payload digests are checked before ANEC parsing") {
  Fixture fixture;
  fixture.manifest["payloads"][1]["sha256"] = hex(64, '0');
  fixture.payload_bytes[0].resize(8);
  fixture.manifest["payloads"][0]["byte_size"] = 8;
  fixture.manifest["payloads"][0]["sha256"] = fixture.digest(0);
  fixture.manifest["release_asset"]["model_sha256"] =
      payload_collection_identity(fixture.manifest["payloads"]);
  fixture.write();
  check_error([&] { load_bundle(fixture.dir.path()); }, "program-1.anec sha256 mismatch");
}

TEST_CASE("ANEC header declares the exact file size") {
  Fixture fixture;
  fixture.payload_bytes[0] += "TRAILER";
  fixture.refresh_payload(0);
  fixture.write();
  check_error(
      [&] { load_bundle(fixture.dir.path()); },
      "ANEC file size does not match payload_size");
}

TEST_CASE("unknown files and missing payloads fail closed") {
  SUBCASE("unknown") {
    Fixture fixture;
    fixture.write();
    write_file(fixture.dir.path() / "extra.bin", "unexpected");
    check_error([&] { load_bundle(fixture.dir.path()); }, "unknown payload file 'extra.bin'");
  }
  SUBCASE("missing") {
    Fixture fixture;
    fixture.write();
    std::filesystem::remove(fixture.dir.path() / "program-1.anec");
    check_error([&] { load_bundle(fixture.dir.path()); }, "payload file missing: program-1.anec");
  }
}

TEST_CASE("directory links fail closed") {
  SUBCASE("listed payload") {
    Fixture fixture;
    fixture.write();
    TempDir outside;
    const auto payload = fixture.dir.path() / "program-0.anec";
    const auto target = outside.path() / "program-0.anec";
    std::filesystem::rename(payload, target);
    std::filesystem::create_symlink(target, payload);
    check_error(
        [&] { load_bundle(fixture.dir.path()); },
        "unexpected link 'program-0.anec'");
  }
  SUBCASE("unlisted entry") {
    Fixture fixture;
    fixture.write();
    std::filesystem::create_symlink(
        fixture.dir.path() / "program-0.anec",
        fixture.dir.path() / "alias.anec");
    check_error(
        [&] { load_bundle(fixture.dir.path()); },
        "unexpected link 'alias.anec'");
  }
}

TEST_CASE("missing bundle is a normal not-found outcome") {
  TempDir parent;
  try {
    load_bundle(parent.path() / "absent");
    FAIL("expected AneBundleNotFound");
  } catch (const AneBundleNotFound& error) {
    CHECK(std::string(error.what()).find("stays on Vulkan") != std::string::npos);
  }
}

namespace {

// Isolates each digest-cache test from the real user sidecar and from other
// tests; every digest-cache TEST_CASE must call this first.
struct SidecarGuard {
  TempDir dir;
  SidecarGuard() {
    setenv("MLX_OMARCHY_ANE_DIGEST_CACHE_PATH",
           (dir.path() / "sidecar.txt").string().c_str(), 1);
  }
  std::filesystem::path sidecar() const { return dir.path() / "sidecar.txt"; }
};

} // namespace

TEST_CASE("digest cache skips re-hash on unchanged file identity") {
  // Default state: MLX_OMARCHY_ANE_DIGEST_CACHE unset in the test process.
  unsetenv("MLX_OMARCHY_ANE_DIGEST_CACHE");
  SidecarGuard sidecar;
  Fixture fixture;
  fixture.write();
  REQUIRE(load_bundle(fixture.dir.path()).programs.size() == 2);

  // Same size, same mtime, different content: the documented identity-keyed
  // trade — a cache hit serves the previously verified digest.
  const auto payload = fixture.dir.path() / "program-0.anec";
  struct ::stat st {};
  REQUIRE(::stat(payload.c_str(), &st) == 0);
  struct timespec times[2]{{st.st_atim.tv_sec, st.st_atim.tv_nsec},
                           {st.st_mtim.tv_sec, st.st_mtim.tv_nsec}};
  write_file(payload, anec_bytes('Z'));
  REQUIRE(::utimensat(AT_FDCWD, payload.c_str(), times, 0) == 0);
  CHECK(load_bundle(fixture.dir.path()).programs.size() == 2);

  // Same content change with a NEW mtime: cache miss, full re-verify, the
  // mismatch against the manifest is caught.
  write_file(payload, anec_bytes('Q'));
  check_error(
      [&] { load_bundle(fixture.dir.path()); }, "program-0.anec sha256 mismatch");

  // The sidecar was primed (2 initial) plus the re-verified miss (1); a
  // cache hit adds no line.
  std::ifstream input(sidecar.sidecar());
  std::vector<std::string> lines;
  for (std::string line; std::getline(input, line);) {
    lines.push_back(line);
  }
  REQUIRE(lines.size() == 3);
  // Every line must carry the full identity key — a moved-from key would
  // serialize an empty path and silently never hit across processes.
  for (const auto& line : lines) {
    INFO("line=", line);
    CHECK(line.rfind(fixture.dir.path().string(), 0) == 0);
    CHECK(line.find("|0|0|") == std::string::npos);
  }
}

TEST_CASE("digest cache kill-switch forces full verification") {
  SidecarGuard sidecar;
  Fixture fixture;
  fixture.write();
  unsetenv("MLX_OMARCHY_ANE_DIGEST_CACHE");
  REQUIRE(load_bundle(fixture.dir.path()).programs.size() == 2);

  const auto payload = fixture.dir.path() / "program-0.anec";
  struct ::stat st0 {};
  REQUIRE(::stat(payload.c_str(), &st0) == 0);
  struct timespec keep[2]{{st0.st_atim.tv_sec, st0.st_atim.tv_nsec},
                          {st0.st_mtim.tv_sec, st0.st_mtim.tv_nsec}};
  write_file(payload, anec_bytes('Z'));
  REQUIRE(::utimensat(AT_FDCWD, payload.c_str(), keep, 0) == 0);

  // Every accepted off-value forces the re-hash the cache would have skipped;
  // the content change is caught even though the identity is unchanged.
  for (const char* value : {"0", "false", "no", "off", ""}) {
    setenv("MLX_OMARCHY_ANE_DIGEST_CACHE", value, 1);
    INFO("value=", value);
    check_error(
        [&] { load_bundle(fixture.dir.path()); }, "program-0.anec sha256 mismatch");
  }

  // Any other value keeps the cache enabled: stale identity is served.
  setenv("MLX_OMARCHY_ANE_DIGEST_CACHE", "1", 1);
  CHECK(load_bundle(fixture.dir.path()).programs.size() == 2);
  unsetenv("MLX_OMARCHY_ANE_DIGEST_CACHE");
}

TEST_CASE("digest cache sidecar serves a fresh process without re-hash") {
  // Simulates a fresh worker process: prime the sidecar directly, never load
  // the bundle in this process first.
  unsetenv("MLX_OMARCHY_ANE_DIGEST_CACHE");
  SidecarGuard sidecar;
  Fixture fixture;
  fixture.write();
  const auto payload = fixture.dir.path() / "program-0.anec";

  // helper: identity key line for the CURRENT file state
  auto key_line = [&](const std::string& digest) {
    struct ::stat st {};
    REQUIRE(::stat(payload.c_str(), &st) == 0);
    char line[512];
    std::snprintf(
        line,
        sizeof(line),
        "%s|%llx|%llx|%llx|%llx %s\n",
        payload.c_str(),
        static_cast<unsigned long long>(st.st_dev),
        static_cast<unsigned long long>(st.st_ino),
        static_cast<unsigned long long>(st.st_size),
        static_cast<unsigned long long>(st.st_mtim.tv_sec) * 1000000000ull +
            static_cast<unsigned long long>(st.st_mtim.tv_nsec),
        digest.c_str());
    return std::string(line);
  };
  auto bump_mtime = [&] {
    struct ::stat st {};
    REQUIRE(::stat(payload.c_str(), &st) == 0);
    struct timespec times[2]{
        {st.st_atim.tv_sec, st.st_atim.tv_nsec},
        {st.st_mtim.tv_sec, st.st_atim.tv_nsec + 1000}};
    REQUIRE(::utimensat(AT_FDCWD, payload.c_str(), times, 0) == 0);
  };

  // 1. Correct sidecar entry, no prior in-process state: the fresh process
  // serves the digest from the sidecar and the bundle loads.
  write_file(sidecar.sidecar(), key_line(fixture.digest(0)));
  CHECK(load_bundle(fixture.dir.path()).programs.size() == 2);

  // 2. A sidecar entry with a WRONG digest under a NEW identity must be
  // caught: the cached digest is always compared against the manifest.
  bump_mtime();
  write_file(sidecar.sidecar(), key_line(std::string(64, '0')));
  check_error(
      [&] { load_bundle(fixture.dir.path()); }, "program-0.anec sha256 mismatch");

  // 3. Kill-switch ignores every cache: correct sidecar restored, payload
  // tampered with the identity preserved — the forced re-hash catches what
  // the cache would have served.
  write_file(payload, anec_bytes('Z'));
  bump_mtime();
  bump_mtime();
  write_file(sidecar.sidecar(), key_line(fixture.digest(0)));
  setenv("MLX_OMARCHY_ANE_DIGEST_CACHE", "0", 1);
  check_error(
      [&] { load_bundle(fixture.dir.path()); }, "program-0.anec sha256 mismatch");
  unsetenv("MLX_OMARCHY_ANE_DIGEST_CACHE");
}

TEST_CASE("tile_shift defaults to the H13 island unit") {
  Fixture fixture;
  fixture.manifest.erase("tile_shift");
  fixture.write();
  auto bundle = load_bundle(fixture.dir.path());
  CHECK(bundle.manifest.tile_shift == kAneTileShiftDefault);
  CHECK(bundle.programs[0].tile_shift == kAneTileShiftDefault);
}

TEST_CASE("explicit tile_shift 9 validates the same byte allocations") {
  Fixture fixture(kAneTileShiftWholeProgram);
  fixture.write();
  auto bundle = load_bundle(fixture.dir.path());
  CHECK(bundle.manifest.tile_shift == kAneTileShiftWholeProgram);
  CHECK(bundle.programs[0].tile_shift == kAneTileShiftWholeProgram);
  // Identical channel bytes: shift 9 with 32x counts equals shift 14 with 1x.
  CHECK(
      bundle.programs[0].anec_header.tiles[4] ==
      uint32_t(32));
}

TEST_CASE("unsupported tile_shift is refused") {
  Fixture fixture;
  fixture.manifest["tile_shift"] = 10;
  fixture.write();
  check_error(
      [&] { load_bundle(fixture.dir.path()); }, "unsupported tile_shift 10");
}

TEST_CASE("whole-program manifest with raw bindings loads") {
  // Raw bindings carry only channel + staged bytes; the ANEC channel
  // allocation is the sole geometry. Matches the parakeet whole-encoder
  // bundle shape (tile_shift 9, one 13701-TD program).
  Fixture fixture(kAneTileShiftWholeProgram);
  // Channel allocations big enough for the staged surfaces: 1500 tiles of
  // 512 B = 768000 bytes, the features-sized window.
  for (auto& payload : fixture.payload_bytes) {
    for (uint32_t channel : {4u, 5u, 6u}) {
      write_le<uint32_t>(payload, 40 + channel * sizeof(uint32_t), 1500);
    }
  }
  fixture.refresh_payload(0);
  const uint64_t alloc = 1500 * 512;
  auto binding = [](const char* tensor, uint64_t channel, uint64_t logical,
                    uint64_t allocation) {
    return nlohmann::json{
        {"tensor", tensor},
        {"channel", channel},
        {"dtype", "float16"},
        {"raw", true},
        {"logical_bytes", logical},
        {"allocation_bytes", allocation},
    };
  };
  fixture.manifest["name"] = "parakeet-encoder-whole";
  fixture.manifest["task_descriptors"] = 1;
  fixture.manifest["inputs"] = {
      {{"name", "attention_mask"}, {"index", 0}, {"dtype", "float16"},
       {"shape", {3000, 1, 1, 1}}, {"byte_size", 6000},
       {"stride", alloc}},
      {{"name", "input_features"}, {"index", 1}, {"dtype", "float16"},
       {"shape", {3000, 128, 1, 1}}, {"byte_size", 768000},
       {"stride", alloc}},
  };
  fixture.manifest["outputs"] = {
      {{"name", "y"}, {"index", 0}, {"dtype", "float16"},
       {"shape", {240000, 1, 1}}, {"byte_size", 480000},
       {"stride", alloc}},
  };
  fixture.manifest["logical_results"] = {
      {{"name", "y"}, {"dtype", "float16"}, {"shape", {240000, 1, 1}},
       {"tensor", "y"}, {"element_offset", 0}, {"element_count", 240000},
       {"conversion", "identity"}},
  };
  fixture.manifest["intermediates"] = nlohmann::json::array();
  fixture.manifest["dispatch_plan"] = {0};
  fixture.manifest["payloads"] = nlohmann::json::array({fixture.manifest["payloads"][0]});
  fixture.manifest["release_asset"]["model_sha256"] =
      payload_collection_identity(fixture.manifest["payloads"]);
  fixture.manifest["programs"] = nlohmann::json::array({nlohmann::json{
      {"payload", "program-0.anec"},
      {"operation", "whole-encoder"},
      {"encoder", "apple-whole-encoder-hwxv2"},
      {"task_descriptors", 1},
      {"scratch_bytes", 0},
      {"inputs",
       {binding("attention_mask", 5, 6000, alloc),
        binding("input_features", 6, 768000, alloc)}},
      {"outputs", {binding("y", 4, 480000, alloc)}},
  }});
  // Only program-0 exists in this bundle: the directory must not carry a
  // payload the manifest does not list.
  write_file(fixture.dir.path() / "program-0.anec", fixture.payload_bytes[0]);
  write_file(fixture.dir.path() / "manifest.json",
             fixture.manifest.dump(2) + "\n");
  auto bundle = load_bundle(fixture.dir.path());
  CHECK(bundle.manifest.programs[0].operation == "whole-encoder");
  CHECK(bundle.manifest.programs[0].inputs[0].raw);
  CHECK(bundle.manifest.programs[0].inputs[0].logical_bytes == 6000);
  CHECK(bundle.manifest.programs[0].inputs[0].element_count == 3000);
}
