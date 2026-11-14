// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

// Host tests for the ABI 3 (H16) section builder. The add-128 case compares
// against the four sections in the kernel tree's KUnit fixture, which the
// kernel's reference client h16-sections.py reproduces byte for byte
// (fixtures/h16-add128/source.json). No device is touched.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"
#include "mlx/backend/omarchy/ane/bundle.h"
#include "mlx/backend/omarchy/ane/h16_hwx.h"
#include "h16_test_hwx.h"

#include <string>
#include <vector>

using namespace mlx::core::omarchy::ane;

namespace {

std::string refusal(const std::vector<uint8_t>& hwx) {
  try {
    h16_build_sections(hwx.data(), hwx.size());
  } catch (const AneH16HwxError& error) {
    return error.reason;
  }
  return "<accepted>";
}

uint32_t le32(const std::vector<uint8_t>& b, size_t o) {
  return uint32_t(b[o]) | (uint32_t(b[o + 1]) << 8) |
      (uint32_t(b[o + 2]) << 16) | (uint32_t(b[o + 3]) << 24);
}

uint64_t le64(const std::vector<uint8_t>& b, size_t o) {
  return uint64_t(le32(b, o)) | (uint64_t(le32(b, o + 4)) << 32);
}

} // namespace

TEST_CASE("add-128 sections equal the kernel fixture byte for byte") {
  const auto dir = h16_test::add128_dir();
  const auto hwx = h16_test::read_all(dir / "model.hwx");
  REQUIRE(hwx.size() == 49152);
  REQUIRE(
      sha256_hex(hwx.data(), hwx.size()) ==
      "231f54c28ea6b14a2867fac78eac1a2cb22aa3b0eeaa28113ed03c520b83a1d3");
  const AneH16Sections built = h16_build_sections(hwx.data(), hwx.size());
  for (size_t kind = 0; kind < kAneH16SectionCount; ++kind) {
    const auto expected = h16_test::read_all(dir / kAneH16SectionNames[kind]);
    REQUIRE(!expected.empty());
    CHECK(built.sections[kind] == expected);
  }
  CHECK(built.subtype == kAneHwxSubtypeH16G);
  CHECK(built.tasks == 1);
  CHECK(!built.scratch_id);
  CHECK(built.scratch_bytes == 0);
  // Kernel order input, output, input: the output takes ID 4, inputs 5, 6.
  REQUIRE(built.surfaces.size() == 3);
  CHECK(*built.surfaces[0].name == "i0");
  CHECK(built.surfaces[0].direction == 1);
  CHECK(built.surfaces[0].id == 5);
  CHECK(*built.surfaces[1].name == "o0");
  CHECK(built.surfaces[1].direction == 2);
  CHECK(built.surfaces[1].id == 4);
  CHECK(*built.surfaces[2].name == "i1");
  CHECK(built.surfaces[2].id == 6);
  for (const auto& surface : built.surfaces) {
    CHECK(surface.allocation == 32768);
  }
  const std::vector<std::pair<uint32_t, uint32_t>> refs{{6, 5}, {7, 4}, {8, 6}};
  CHECK(built.refs == refs);
}

TEST_CASE("scratch and kernel slots bind the scratch ID and resource 1007") {
  h16_test::Spec spec;
  spec.scratch = {{0x40000000, 0x123456}};
  spec.kernel_vm = 0x50000000;
  spec.tasks = 3;
  spec.slots = {{3, 0x40000000}, {4, 0x50000000}, {6, 0x30000000},
                {7, 0x30010000}, {8, 0x30020000}, {9, 0}};
  const auto hwx = h16_test::build(spec);
  const AneH16Sections built = h16_build_sections(hwx.data(), hwx.size());
  REQUIRE(built.scratch_id);
  CHECK(*built.scratch_id == 7);
  CHECK(built.scratch_bytes == 0x123456);
  CHECK(built.tasks == 3);
  // Slot 9 has a zero base (a config-word slot) and stays unbound.
  const std::vector<std::pair<uint32_t, uint32_t>> refs{
      {3, 7}, {4, 1007}, {6, 5}, {7, 4}, {8, 6}};
  CHECK(built.refs == refs);

  const auto& generic = built.sections[0];
  REQUIRE(generic.size() == 0x208 + 4 * 0x30);
  CHECK(le32(generic, 0) == 1);
  CHECK(le32(generic, 4) == 0x10);
  CHECK(le32(generic, 0x204) == 4);
  // Entry 3 is the scratch: an input-typed buffer of the __bss size, last.
  const size_t scratch = 0x208 + 3 * 0x30;
  CHECK(le32(generic, scratch) == 1);
  CHECK(le32(generic, scratch + 4) == 7);
  CHECK(le32(generic, scratch + 8) == 0);
  CHECK(le32(generic, scratch + 0x10) == 1);
  CHECK(le64(generic, scratch + 0x20) == 0x123456);
  CHECK(le32(generic, scratch + 0x28) == 0xffff);
  // The output entry is typed 1, direction 2.
  CHECK(le32(generic, 0x208 + 0x30 + 4) == 4);
  CHECK(le32(generic, 0x208 + 0x30 + 8) == 1);
  CHECK(le32(generic, 0x208 + 0x30 + 0x10) == 2);

  const auto& operation = built.sections[1];
  REQUIRE(operation.size() == 4 + 0x40c);
  CHECK(le32(operation, 0) == 1);
  CHECK(le32(operation, 4 + 8) == 5);
  CHECK(le32(operation, 4 + 0x0c + 8) == 4);
  CHECK(le32(operation, 4 + 0x0c + 12) == 1007);

  const auto& properties = built.sections[3];
  REQUIRE(properties.size() == 0x28);
  CHECK(le32(properties, 12) == 3);
  CHECK(le64(properties, 0x20) == h16_test::task_stream(spec).size());

  const auto& procedure = built.sections[2];
  REQUIRE(procedure.size() == 0x38);
  CHECK(le64(procedure, 8) == 0x18);
  CHECK(le32(procedure, 0x18 + 4 * 6) == 0xffffffffu);
}

TEST_CASE("an H17 program opens its task stream with 0") {
  h16_test::Spec spec;
  spec.subtype = kAneHwxSubtypeH17;
  const auto hwx = h16_test::build(spec);
  const AneH16Sections built = h16_build_sections(hwx.data(), hwx.size());
  CHECK(built.subtype == kAneHwxSubtypeH17);

  // The H16G prologue under the H17 subtype is refused.
  auto wrong = h16_test::build(h16_test::Spec{});
  h16_test::put32(wrong, 8, kAneHwxSubtypeH17);
  CHECK(refusal(wrong) == "task stream lacks the prologue of its target");
}

TEST_CASE("malformed programs get the reference client's refusals") {
  SUBCASE("magic and subtype") {
    auto hwx = h16_test::build(h16_test::Spec{});
    h16_test::put32(hwx, 0, 0xfeedfacf);
    CHECK(refusal(hwx) == "not an H16G or H17 HWX");
    hwx = h16_test::build(h16_test::Spec{});
    h16_test::put32(hwx, 8, 8);
    CHECK(refusal(hwx) == "not an H16G or H17 HWX");
    CHECK(refusal(std::vector<uint8_t>(16, 0)) == "not an H16G or H17 HWX");
  }
  SUBCASE("two outputs") {
    h16_test::Spec spec;
    spec.surfaces[2].direction = 2;
    CHECK(refusal(h16_test::build(spec)) == "an H16 program has exactly one output");
  }
  SUBCASE("missing program descriptor") {
    h16_test::Spec spec;
    spec.descriptor = false;
    CHECK(refusal(h16_test::build(spec)) ==
          "HWX lacks a task stream, program descriptor or surfaces");
  }
  SUBCASE("a BAR base that names nothing") {
    h16_test::Spec spec;
    spec.slots.push_back({10, 0x99990000});
    CHECK(refusal(h16_test::build(spec)) ==
          "BAR slot 10 base 0x99990000 names no surface, scratch, kernel or constants");
  }
  SUBCASE("no BAR reference at all") {
    h16_test::Spec spec;
    spec.slots = {{6, 0}};
    CHECK(refusal(h16_test::build(spec)) ==
          "program binds no buffer through a BAR slot");
  }
  SUBCASE("a surface without its tensor descriptor") {
    h16_test::Spec spec;
    spec.surfaces[0].batch = 0;
    CHECK(refusal(h16_test::build(spec)) == "surface a has no matching tensor");
    spec = h16_test::Spec{};
    spec.surfaces[1].batch = 0x8000; // span larger than the allocation
    CHECK(refusal(h16_test::build(spec)) == "surface y has no matching tensor");
  }
  SUBCASE("a scratch segment that is not read/write") {
    h16_test::Spec spec;
    spec.scratch = {{0x40000000, 0x1000}};
    auto hwx = h16_test::build(spec);
    // The __DATA segment is the fifth load command (3 surfaces, __TEXT).
    const size_t data = 32 + 4 * (72 + 80);
    REQUIRE(std::string(reinterpret_cast<const char*>(&hwx[data + 8])) == "__DATA");
    h16_test::put32(hwx, data + 60, 1);
    CHECK(refusal(hwx) == "scratch segment is not read/write");
  }
  SUBCASE("task stream records") {
    auto hwx = h16_test::build(h16_test::Spec{});
    const size_t text = 0x4000;
    // A register index outside every H16 register range.
    h16_test::put32(hwx, text + 16 + 36, (1u << 29) | (6u << 23) | 0x7000u);
    CHECK(refusal(hwx) == "task record outside the register image");
    hwx = h16_test::build(h16_test::Spec{});
    h16_test::put32(hwx, text + 8, 1);
    CHECK(refusal(hwx) == "task stream lacks the prologue of its target");
  }
  SUBCASE("truncated container") {
    auto hwx = h16_test::build(h16_test::Spec{});
    hwx.resize(200);
    CHECK(refusal(hwx) == "truncated load command");
  }
  SUBCASE("a load-command name that is not UTF-8") {
    auto hwx = h16_test::build(h16_test::Spec{});
    hwx[32 + 8] = 0xff; // first segment name
    CHECK(refusal(hwx) == "HWX name is not valid UTF-8");
  }
}

TEST_CASE("the task walk counts tasks and collects BAR slots") {
  h16_test::Spec spec;
  spec.tasks = 4;
  spec.slots = {{9, 0x30000000}, {2, 0x30010000}};
  const auto stream = h16_test::task_stream(spec);
  const auto [tasks, slots] = h16_walk_tasks(stream.data(), stream.size(), 1);
  CHECK(tasks == 4);
  CHECK(slots == std::vector<uint32_t>{2, 9});
  std::vector<uint8_t> empty(16, 0);
  h16_test::put32(empty, 0, 1);
  CHECK_THROWS_AS(h16_walk_tasks(empty.data(), empty.size(), 1), AneH16HwxError);
  std::vector<uint8_t> padded = empty;
  padded.resize(32, 0);
  padded[20] = 1;
  try {
    h16_walk_tasks(padded.data(), padded.size(), 1);
    FAIL("nonzero padding accepted");
  } catch (const AneH16HwxError& error) {
    CHECK(error.reason == "nonzero padding in the task stream");
  }
}
