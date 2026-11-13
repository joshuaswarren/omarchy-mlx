// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

// Host tests for the lane-3 (ABI 3, H16) device. A fake driver sits behind
// the AneH16Kernel seam: it answers GET_CAPS, DRM_IOCTL_VERSION, H16_LOAD
// and H16_EXEC with the uapi structures, keeps one program per open file,
// and records every byte the device hands it. No /dev/accel node, no ANE.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"
#include "mlx/backend/omarchy/ane/h16_hwx.h"
#include "mlx/backend/omarchy/ane/uapi/drm/ane_accel.h"
#include "mlx/backend/omarchy/ane/worker_h16.h"
#include "h16_test_hwx.h"

#include <unistd.h>

#include <array>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <set>
#include <string>
#include <vector>

using namespace mlx::core::omarchy::ane;

namespace {

struct FakeNode {
  int open_error{0};      // -errno for open()
  int caps_error{0};      // -errno for GET_CAPS (another accel driver)
  uint32_t abi{DRM_ANE_ABI_V3};
  uint32_t chip{DRM_ANE_CHIP_H16};
  std::string driver{"ane"};
  int major{DRM_ANE_ABI_V3};
};

struct FakeProgram {
  std::vector<uint8_t> hwx;
  std::array<std::vector<uint8_t>, 4> sections;
  std::vector<uint64_t> sizes;
  std::vector<uint32_t> directions;
  int execs{0};
};

// One fake ABI 3 driver. Outputs are a deterministic function of the inputs
// in kernel surface order: output[k] = first_input[k] + 2 * second_input[k]
// (bytes, mod 256), or a copy of the only input.
struct FakeKernel : AneH16Kernel {
  std::map<std::string, FakeNode> nodes{{"/dev/accel/accel0", FakeNode{}}};
  std::map<int, std::string> open_files;
  std::map<int, FakeProgram> programs;
  int next_fd{100};
  int load_error{0};
  int exec_error{0};
  // Overrides of what H16_LOAD reports, per surface index.
  std::map<size_t, uint64_t> report_size;
  std::vector<std::vector<uint8_t>> last_inputs;

  int open_node(const std::string& path) override {
    auto found = nodes.find(path);
    if (found == nodes.end()) return -ENOENT;
    if (found->second.open_error) return found->second.open_error;
    open_files[next_fd] = path;
    return next_fd++;
  }

  void close_node(int fd) override {
    CHECK(open_files.erase(fd) == 1); // never throws: runs from destructors
    programs.erase(fd);
  }

  std::vector<std::string> list_nodes() override {
    std::vector<std::string> out;
    for (const auto& entry : nodes) out.push_back(entry.first);
    return out;
  }

  int ioctl_node(int fd, unsigned long request, void* argument) override {
    if (open_files.count(fd) != 1) return -EBADF;
    const FakeNode& node = nodes.at(open_files.at(fd));
    if (request == DRM_IOCTL_ANE_GET_CAPS) {
      if (node.caps_error) return node.caps_error;
      auto* caps = static_cast<drm_ane_get_caps*>(argument);
      if (caps->flags || caps->pad) return -EINVAL;
      *caps = {};
      caps->size = sizeof(*caps);
      caps->abi_version = node.abi;
      caps->chip_family = node.chip;
      caps->section_size = sizeof(drm_ane_h16_section);
      caps->exec_io_size = sizeof(drm_ane_h16_buffer);
      return 0;
    }
    if (request == DRM_IOCTL_VERSION) {
      auto* version = static_cast<drm_version*>(argument);
      version->version_major = node.major;
      version->version_minor = 0;
      version->version_patchlevel = 0;
      const size_t n = std::min(version->name_len, node.driver.size());
      std::memcpy(version->name, node.driver.data(), n);
      version->name_len = node.driver.size();
      return 0;
    }
    if (request == DRM_IOCTL_ANE_H16_LOAD) {
      auto* load = static_cast<drm_ane_h16_load*>(argument);
      if (load->flags || load->pad || load->section_count != DRM_ANE_H16_SECTIONS)
        return -EINVAL;
      if (programs.count(fd)) return -EBUSY;
      if (load_error) return load_error;
      FakeProgram program;
      const auto* hwx = reinterpret_cast<const uint8_t*>(load->hwx_ptr);
      program.hwx.assign(hwx, hwx + load->hwx_size);
      const auto* table = reinterpret_cast<const drm_ane_h16_section*>(load->sections_ptr);
      for (uint32_t i = 0; i < load->section_count; ++i) {
        const uint32_t kind = table[i].kind - DRM_ANE_H16_SECTION_GENERIC;
        if (kind >= 4 || !program.sections[kind].empty() ||
            table[i].size < DRM_ANE_H16_SECTION_MIN ||
            table[i].size > DRM_ANE_H16_SECTION_MAX || table[i].size % 4)
          return -EINVAL;
        const auto* bytes = reinterpret_cast<const uint8_t*>(table[i].data_ptr);
        program.sections[kind].assign(bytes, bytes + table[i].size);
      }
      // The kernel derives the surfaces from the HWX itself.
      const AneH16Sections built =
          h16_build_sections(program.hwx.data(), program.hwx.size());
      load->surface_count = static_cast<uint32_t>(built.surfaces.size());
      for (size_t i = 0; i < built.surfaces.size(); ++i) {
        // The kernel reports each surface's tensor span.
        const uint64_t size = report_size.count(i) ? report_size.at(i)
                                                   : built.surfaces[i].span;
        load->surface_size[i] = size;
        load->surface_direction[i] = built.surfaces[i].direction;
        program.sizes.push_back(size);
        program.directions.push_back(built.surfaces[i].direction);
      }
      programs[fd] = std::move(program);
      return 0;
    }
    if (request == DRM_IOCTL_ANE_H16_EXEC) {
      auto* exec = static_cast<drm_ane_h16_exec*>(argument);
      auto found = programs.find(fd);
      if (exec->flags || found == programs.end() ||
          exec->count != found->second.sizes.size())
        return -EINVAL;
      const auto* buffers = reinterpret_cast<const drm_ane_h16_buffer*>(exec->buffers_ptr);
      for (uint32_t i = 0; i < exec->count; ++i)
        if (buffers[i].size != found->second.sizes[i]) return -EINVAL;
      if (exec_error) return exec_error;
      last_inputs.clear();
      size_t output = 0;
      for (uint32_t i = 0; i < exec->count; ++i) {
        const auto* bytes = reinterpret_cast<const uint8_t*>(buffers[i].ptr);
        if (found->second.directions[i] == 1)
          last_inputs.emplace_back(bytes, bytes + buffers[i].size);
        else
          output = i;
      }
      auto* out = reinterpret_cast<uint8_t*>(buffers[output].ptr);
      for (size_t k = 0; k < buffers[output].size; ++k) {
        const uint8_t a = k < last_inputs[0].size() ? last_inputs[0][k] : 0;
        const uint8_t b = last_inputs.size() > 1 && k < last_inputs[1].size()
            ? last_inputs[1][k] : 0;
        out[k] = static_cast<uint8_t>(a + 2 * b);
      }
      found->second.execs++;
      return 0;
    }
    return -ENOTTY;
  }
};

// The pinned add-128 program as the validator hands it to a device:
// kernel order input (surface 0), output (1), input (2), 32768 bytes each.
AneValidatedProgram add128_program(size_t index = 0) {
  AneValidatedProgram program;
  program.manifest_index = index;
  program.format = AneProgramFormat::hwx;
  program.hwx = h16_test::add128_dir() / "model.hwx";
  program.h16_surfaces = {{32768, 1}, {32768, 2}, {32768, 1}};
  program.h16_tasks = 1;
  return program;
}

AneProgramBinding raw_binding(const std::string& tensor, uint64_t surface) {
  AneProgramBinding binding;
  binding.tensor = tensor;
  binding.channel = surface;
  binding.raw = true;
  binding.dtype = "float16";
  binding.logical_bytes = 32768;
  binding.allocation_bytes = 32768;
  binding.element_count = 16384;
  return binding;
}

std::vector<uint8_t> pattern(uint8_t seed, size_t size = 32768) {
  std::vector<uint8_t> bytes(size);
  for (size_t i = 0; i < size; ++i) bytes[i] = static_cast<uint8_t>(seed + i * 7);
  return bytes;
}

std::string message_of(const std::function<void()>& call) {
  try {
    call();
  } catch (const AneDeviceError& error) {
    return error.what();
  }
  return "<no error>";
}

bool contains(const std::string& text, const std::string& part) {
  return text.find(part) != std::string::npos;
}

} // namespace

TEST_CASE("the probe picks the first ABI 3 H16 node and names it") {
  FakeKernel kernel;
  kernel.nodes.clear();
  FakeNode other_driver;
  other_driver.caps_error = -ENOTTY;
  FakeNode abi2;
  abi2.abi = DRM_ANE_ABI_V2;
  abi2.chip = DRM_ANE_CHIP_H14;
  kernel.nodes["/dev/accel/accel0"] = other_driver;
  kernel.nodes["/dev/accel/accel1"] = abi2;
  kernel.nodes["/dev/accel/accel2"] = FakeNode{};
  const AneH16Identity identity = probe_h16_node("", kernel);
  CHECK(identity.node == "/dev/accel/accel2");
  CHECK(identity.driver == "ane");
  CHECK(identity.abi == 3);
  CHECK(identity.chip == 16);
  CHECK(contains(identity.describe(), "lane=3 node=/dev/accel/accel2 driver=ane"));
  CHECK(contains(identity.describe(), "drm_version=3.0.0 abi=3 chip=h16"));
  CHECK(kernel.open_files.empty());

  CHECK(contains(
      message_of([&] { probe_h16_node("/dev/accel/accel1", kernel); }),
      "/dev/accel/accel1 reports ANE ABI 2 chip 14; lane 3 requires ABI 3 on H16"));
  CHECK(contains(
      message_of([&] { probe_h16_node("/dev/accel/accel0", kernel); }),
      "DRM_IOCTL_ANE_GET_CAPS failed for /dev/accel/accel0: ENOTTY"));
  CHECK(kernel.open_files.empty());
}

TEST_CASE("the probe separates a missing lane from an unopenable node") {
  FakeKernel kernel;
  kernel.nodes.clear();
  CHECK(contains(
      message_of([&] { probe_h16_node("", kernel); }),
      "no ABI 3 H16 accel node (the H16 program stage was not reached)"));
  FakeNode locked;
  locked.open_error = -EACCES;
  kernel.nodes["/dev/accel/accel0"] = locked;
  CHECK(contains(
      message_of([&] { probe_h16_node("", kernel); }),
      "ABI 3 accel node cannot be opened: open(/dev/accel/accel0): EACCES"));
  FakeNode wrong_driver;
  wrong_driver.driver = "ane_t6021";
  kernel.nodes["/dev/accel/accel0"] = wrong_driver;
  CHECK(contains(
      message_of([&] { probe_h16_node("", kernel); }),
      "is DRM driver 'ane_t6021' major 3; lane 3 requires driver 'ane' major 3"));
}

TEST_CASE("load hands the HWX and the reference sections to H16_LOAD") {
  auto kernel = std::make_shared<FakeKernel>();
  auto device = make_h16_device("", kernel);
  CHECK(contains(device->describe(), "lane=3 node=/dev/accel/accel0"));
  CHECK(kernel->open_files.size() == 1); // the probe file
  device->load(add128_program());
  REQUIRE(kernel->programs.size() == 1);
  const FakeProgram& loaded = kernel->programs.begin()->second;
  CHECK(loaded.hwx == h16_test::read_all(h16_test::add128_dir() / "model.hwx"));
  for (size_t kind = 0; kind < 4; ++kind) {
    CHECK(loaded.sections[kind] ==
          h16_test::read_all(h16_test::add128_dir() / kAneH16SectionNames[kind]));
  }
  device->release();
  CHECK(kernel->programs.empty());
  CHECK(kernel->open_files.size() == 1);
  device.reset();
  CHECK(kernel->open_files.empty());
}

TEST_CASE("every program gets its own file and all stay loaded together") {
  auto kernel = std::make_shared<FakeKernel>();
  auto device = make_h16_device("/dev/accel/accel0", kernel);
  device->load(add128_program(0));
  device->load(add128_program(1));
  device->load(add128_program(7));
  CHECK(kernel->programs.size() == 3);
  CHECK(kernel->open_files.size() == 4);
  CHECK(contains(
      message_of([&] { device->load(add128_program(1)); }),
      "program 1 (model.hwx) is already loaded"));
  device->release();
  CHECK(kernel->open_files.size() == 1);
}

TEST_CASE("send, exec and read move every surface in kernel order") {
  auto kernel = std::make_shared<FakeKernel>();
  auto device = make_h16_device("", kernel);
  device->load(add128_program());
  const auto a = pattern(3);
  const auto b = pattern(11);
  // execute_plan passes the binding position; the surface index is the
  // binding's own (a = surface 0, b = surface 2).
  device->send(0, 0, raw_binding("a", 0), a.data(), a.size());
  device->send(0, 1, raw_binding("b", 2), b.data(), b.size());
  device->exec(0);
  REQUIRE(kernel->last_inputs.size() == 2);
  CHECK(kernel->last_inputs[0] == a);
  CHECK(kernel->last_inputs[1] == b);
  std::vector<uint8_t> y(32768);
  device->read(0, 0, raw_binding("y", 1), y.data(), y.size());
  bool exact = true;
  for (size_t k = 0; k < y.size(); ++k) {
    exact = exact && y[k] == static_cast<uint8_t>(a[k] + 2 * b[k]);
  }
  CHECK(exact);
  CHECK(kernel->programs.begin()->second.execs == 1);
  // Surfaces are checked by direction.
  CHECK(contains(
      message_of([&] { device->send(0, 0, raw_binding("y", 1), a.data(), a.size()); }),
      "has no input surface 1 for tensor 'y'"));
  CHECK(contains(
      message_of([&] { device->read(0, 0, raw_binding("a", 0), y.data(), y.size()); }),
      "has no output surface 0 for tensor 'a'"));
  CHECK(contains(
      message_of([&] { device->exec(5); }), "exec: program 5 is not loaded"));
}

TEST_CASE("tiled bindings use the libane packing on the way in and out") {
  // Two 16 KiB surfaces: input x (0) and output y (1). The fake copies the
  // input surface to the output, so the round trip exposes the placement.
  h16_test::Spec spec;
  spec.surfaces = {{"x", 1, 0x30000000, 0x4000, 1, 0x2000},
                   {"y", 2, 0x30010000, 0x4000, 1, 0x2000}};
  spec.slots = {{6, 0x30000000}, {7, 0x30010000}};
  const auto hwx = h16_test::build(spec);
  const auto path = std::filesystem::temp_directory_path() /
      ("mlx-omarchy-h16-tiled-" + std::to_string(::getpid()) + ".hwx");
  {
    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<const char*>(hwx.data()), hwx.size());
  }
  AneValidatedProgram program;
  program.format = AneProgramFormat::hwx;
  program.hwx = path;
  program.h16_surfaces = {{0x4000, 1}, {0x4000, 2}};
  program.h16_tasks = 1;

  AneProgramBinding x;
  x.tensor = "x";
  x.channel = 0;
  x.dtype = "float16";
  x.shape = {1, 1, 4, 16};
  x.nchw = {1, 1, 4, 16, 256, 64}; // rows of 16 fp16 at a 64-byte stride
  x.logical_bytes = 4 * 16 * 2;
  x.allocation_bytes = 0x4000;
  x.element_count = 64;
  x.physical_elements = 64;
  AneProgramBinding y = x;
  y.tensor = "y";
  y.channel = 1;

  auto kernel = std::make_shared<FakeKernel>();
  auto device = make_h16_device("", kernel);
  device->load(program);
  const auto dense = pattern(1, 128);
  device->send(0, 0, x, dense.data(), dense.size());
  device->exec(0);
  REQUIRE(kernel->last_inputs.size() == 1);
  const auto& tile = kernel->last_inputs[0];
  for (size_t row = 0; row < 4; ++row) {
    CHECK(std::memcmp(tile.data() + row * 64, dense.data() + row * 32, 32) == 0);
    CHECK(tile[row * 64 + 32] == 0); // row padding stays untouched
  }
  std::vector<uint8_t> back(128);
  device->read(0, 0, y, back.data(), back.size());
  CHECK(back == dense);
  device.reset();
  std::filesystem::remove(path);
}

TEST_CASE("driver errors name the ioctl, the program and the errno") {
  auto kernel = std::make_shared<FakeKernel>();
  auto device = make_h16_device("", kernel);
  kernel->load_error = -ENOEXEC;
  const std::string load = message_of([&] { device->load(add128_program()); });
  CHECK(contains(
      load,
      "DRM_IOCTL_ANE_H16_LOAD failed for program 0 (model.hwx) on "
      "/dev/accel/accel0: ENOEXEC"));
  CHECK(contains(load, "the driver refused the HWX"));
  CHECK(kernel->open_files.size() == 1); // the failed program's file is closed

  kernel->load_error = 0;
  device->load(add128_program());
  kernel->exec_error = -ETIMEDOUT;
  const std::string exec = message_of([&] { device->exec(0); });
  CHECK(contains(exec, "DRM_IOCTL_ANE_H16_EXEC failed for program 0 (model.hwx)"));
  CHECK(contains(exec, "ETIMEDOUT"));
  CHECK(contains(exec, "quarantined the session"));
  CHECK(contains(h16_ioctl_error("DRM_IOCTL_ANE_H16_EXEC", "p", EIO),
                 "EIO (Input/output error): the H16 session is quarantined"));
}

TEST_CASE("a kernel surface report that disagrees with the manifest is refused") {
  auto kernel = std::make_shared<FakeKernel>();
  kernel->report_size[1] = 16384;
  auto device = make_h16_device("", kernel);
  CHECK(contains(
      message_of([&] { device->load(add128_program()); }),
      "DRM_IOCTL_ANE_H16_LOAD for program 0 (model.hwx) reported surface 1 "
      "with 16384 bytes, direction 2; the HWX spans 32768 bytes, direction 2"));
  CHECK(kernel->programs.empty()); // closing the file unloaded it
  CHECK(kernel->open_files.size() == 1);
}

TEST_CASE("the lane-3 device refuses ANEC programs and stale contracts") {
  auto kernel = std::make_shared<FakeKernel>();
  auto device = make_h16_device("", kernel);
  AneValidatedProgram anec;
  anec.anec = "/dev/null";
  CHECK(contains(
      message_of([&] { device->load(anec); }),
      "the ABI 3 device cannot load ANEC program 0"));
  AneValidatedProgram stale = add128_program();
  stale.h16_surfaces[0].bytes = 16384;
  CHECK(contains(
      message_of([&] { device->load(stale); }),
      "HWX differs from the validated bundle"));
  CHECK(kernel->open_files.size() == 1);
}

TEST_CASE("a supervised worker run reaches the fake driver end to end") {
  AneBundle bundle;
  bundle.manifest.manifest_version = kAneManifestVersionHwx;
  bundle.manifest.driver_abi_major = 3;
  AneTensor tensor;
  tensor.dtype = "float16";
  tensor.shape = {16384};
  tensor.byte_size = 32768;
  tensor.name = "a";
  bundle.manifest.inputs.push_back(tensor);
  tensor.name = "b";
  bundle.manifest.inputs.push_back(tensor);
  tensor.name = "y";
  bundle.manifest.outputs.push_back(tensor);
  AneProgram program;
  program.payload = "model.hwx";
  program.inputs = {raw_binding("a", 0), raw_binding("b", 2)};
  program.outputs = {raw_binding("y", 1)};
  bundle.manifest.programs = {program};
  bundle.manifest.dispatch_plan = {0};
  bundle.programs = {add128_program()};

  AneWorkerOptions options;
  options.deadline = std::chrono::milliseconds(4000);
  options.iterations = 2;
  AneWorker worker(
      [] { return make_h16_device("", std::make_shared<FakeKernel>()); },
      options);
  const auto a = pattern(5);
  const auto b = pattern(9);
  std::map<std::string, AneWorker::Buffer> outputs;
  const auto report = worker.run(bundle, {{"a", a}, {"b", b}}, &outputs);
  REQUIRE(report.status == AneWorkerStatus::Completed);
  CHECK(report.iterations == 2);
  CHECK(report.released_programs == 1);
  REQUIRE(outputs.count("y") == 1);
  bool exact = outputs["y"].size() == 32768;
  for (size_t k = 0; exact && k < 32768; ++k) {
    exact = outputs["y"][k] == static_cast<uint8_t>(a[k] + 2 * b[k]);
  }
  CHECK(exact);
}
