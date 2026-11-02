// SPDX-License-Identifier: MIT
// Standalone calibration only; never linked into a wheel or the production router.
#include "contract.h"
#include <dlfcn.h>
#include <fcntl.h>
#include <unistd.h>
#include <bit>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <sstream>
#include <string>
#include "mlx/ops.h"
#include "mlx/random.h"
#include "mlx/transforms.h"
#include "mlx/backend/omarchy/encoder.h"

using namespace mlx::core;
namespace om = mlx::core::omarchy;
namespace fs = std::filesystem;
namespace {
constexpr unsigned guard = 8;
constexpr uint16_t canary = 0x7fc1;
using Clock = std::chrono::steady_clock;

void require(bool ok, const char* why) {
  if (!ok) throw std::runtime_error(why);
}
void* bytes(const array& a) {
  auto buffer = a.buffer();
  return static_cast<char*>(buffer.raw_ptr()) + a.offset();
}
std::vector<uint16_t> bits(const array& a) {
  auto* p = static_cast<const uint16_t*>(bytes(a));
  return {p, p + a.size()};
}
void save(const fs::path& path, const void* data, size_t size) {
  std::ofstream out(path, std::ios::binary);
  out.write(static_cast<const char*>(data), size);
  out.close();
  require(!out.fail(), "cannot retain raw output");
}
array allocate(Shape shape) {
  array a(shape, bfloat16, nullptr, {});
  allocator::Allocator& alloc = om::allocator();
  a.set_data(alloc.malloc(a.nbytes()));
  return a;
}
// Retain the entire native trace, including stderr on a failed dispatch.
std::vector<int> traced(const fs::path& path, const std::function<void()>& run) {
  std::fflush(stderr);
  int saved = dup(2);
  int sink = open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0644);
  require(saved >= 0 && sink >= 0, "cannot retain dispatch trace");
  require(dup2(sink, 2) >= 0, "cannot redirect dispatch trace");
  close(sink);
  setenv("MLX_OMARCHY_TRACE_DISPATCH", "1", 1);
  try { run(); }
  catch (...) {
    std::fflush(stderr); dup2(saved, 2); close(saved);
    unsetenv("MLX_OMARCHY_TRACE_DISPATCH"); throw;
  }
  std::fflush(stderr); dup2(saved, 2); close(saved);
  unsetenv("MLX_OMARCHY_TRACE_DISPATCH");
  std::ifstream in(path);
  std::string line;
  std::vector<int> ids;
  while (std::getline(in, line)) {
    int id;
    if (std::sscanf(line.c_str(), "[rtmod] DISPATCH kernel=%d", &id) == 1)
      ids.push_back(id);
  }
  require(in.eof(), "cannot read complete dispatch trace");
  return ids;
}
void gate(om::Device& dev) {
  const auto& c = dev.capabilities();
  const auto& hw = dev.hardware_capabilities();
  require(c.device_name.find("G14C") != std::string::npos &&
          hw.device_name.find("G14C") != std::string::npos,
          "this calibration is G14C only; no other-chip substitution");
  for (const auto* caps : {&c, &hw}) {
    require(caps->subgroup_size == 32 &&
            (caps->subgroup_operations & VK_SUBGROUP_FEATURE_ARITHMETIC_BIT) &&
            caps->storage_buffer_16bit_access && caps->shader_int16,
            "real and reported 32-lane arithmetic/int16 storage required");
  }
  require(hw.required_subgroup_size == 0 || hw.required_subgroup_size == 32,
          "incompatible required subgroup width");
  require(dev.compute().binding_limit() >= 5, "five storage bindings required");
  std::cerr << "CALIBRATION device=" << hw.device_name << " subgroup="
            << hw.subgroup_size << " required=" << hw.required_subgroup_size << '\n';
}
// Refuse inherited routing experiments; do not rewrite the default arm's flags.
void clean_route_environment() {
  for (char** p = ::environ; *p; ++p) {
    std::string name(*p, std::strchr(*p, '=') - *p);
    require(!(name.starts_with("MLX_OMARCHY_QM") ||
              name == "MLX_OMARCHY_NO_COOPMAT" ||
              name == "MLX_OMARCHY_NO_QMM_FMA" ||
              name == "MLX_OMARCHY_CAPS_SIM" ||
              name == "MLX_OMARCHY_ALLOW_NON_APPLE" ||
              name == "MLX_OMARCHY_TRACE_DISPATCH"),
            "routing/simulation/trace override must be absent at entry");
  }
}

void run_case(unsigned k, unsigned n, unsigned t, const fs::path& out, Stream stream) {
  auto& encoder = om::get_command_encoder(stream);
  const auto block_plan = calibration::plan(k, n, t, guard, guard);
  random::seed(7919u * k + 31u * n + 101u * t);
  auto w = multiply(random::normal(Shape{int(n), int(k)}, float32,
                                   std::nullopt, stream), array(0.05f), stream);
  auto q = quantize(w, 64, 4, "affine", std::nullopt, stream);
  array scales = astype(q[1], bfloat16, stream);
  array biases = astype(q[2], bfloat16, stream);
  array x = astype(random::normal(Shape{1, int(t), int(k)}, float32,
                                  std::nullopt, stream), bfloat16, stream);
  eval({q[0], scales, biases, x});
  encoder.synchronize("calibration inputs ready");
  require(x.offset() == 0 && q[0].offset() == 0 &&
          scales.offset() == 0 && biases.offset() == 0,
          "generated operands must have zero base offsets");
  std::vector<std::pair<const array*, std::vector<char>>> originals;
  for (const auto* a : {&x, &q[0], &scales, &biases}) {
    const char* p = static_cast<const char*>(bytes(*a));
    originals.push_back({a, {p, p + a->nbytes()}});
  }
  const char* names[] = {"x.bf16", "wq.u32", "scales.bf16", "biases.bf16"};
  for (size_t i = 0; i < originals.size(); ++i)
    save(out / names[i], originals[i].second.data(), originals[i].second.size());

  // Exercise a real aligned nonzero input base; the default arm uses identical bytes.
  array padded = allocate({int(t * k + 2 * guard)});
  auto* xp = static_cast<uint16_t*>(bytes(padded));
  std::fill(xp, xp + padded.size(), canary);
  std::memcpy(xp + guard, bytes(x), x.nbytes());
  auto default_arm = [&]() {
    array y = quantized_matmul(x, q[0], scales, biases, true, 64, 4, "affine", stream);
    y.eval(); encoder.synchronize("calibration default tile");
    return bits(y);
  };
  auto batch4_arm = [&]() {
    array y = allocate({int(t * n + 2 * guard)});
    auto* yp = static_cast<uint16_t*>(bytes(y));
    std::fill(yp, yp + y.size(), canary);
    std::array<om::ComputeBinding, 5> bindings{
        om::binding(padded), om::binding(q[0]), om::binding(scales),
        om::binding(biases), om::binding(y)};
    for (auto block : block_plan) {
      const auto params = calibration::push_constants<om::ComputeParams>(block, k, n, t);
      encoder.dispatch_compute(om::ComputeKernel::QmmVecQ4WordSubgroupBatch4BF16,
                               bindings, params, block.groups, 1, 1);
    }
    encoder.synchronize("calibration direct Batch4 4+tail");
    for (unsigned i = 0; i < guard; ++i)
      require(yp[i] == canary && yp[guard + t * n + i] == canary,
              "Batch4 output canary changed");
    return std::vector<uint16_t>(yp + guard, yp + guard + t * n);
  };
  std::vector<uint16_t> base, direct;
  const auto base_trace = traced(out / "default.trace", [&] { base = default_arm(); });
  const auto direct_trace = traced(out / "calibration.trace", [&] { direct = batch4_arm(); });
  const int batch4 = static_cast<int>(om::ComputeKernel::QmmVecQ4WordSubgroupBatch4BF16);
  calibration::require_batch4_trace(direct_trace, batch4);
  using K = om::ComputeKernel;
  const std::vector<int> tiles{
      int(K::QmmPrefillCoopmatM16BF16X32FullNLdsPad),
      int(K::QmmPrefillCoopmatM16BF16X32), int(K::QmmPrefillCoopmatBF16X32FullN),
      int(K::QmmPrefillCoopmatBF16X32), int(K::QmmTileBF16)};
  unsigned tile_count = 0;
  for (int id : base_trace) {
    if (std::find(tiles.begin(), tiles.end(), id) != tiles.end()) ++tile_count;
    else require(id == int(K::CastBF16F32), "unexpected default-arm dispatch");
  }
  require(tile_count == 1, "default arm must run exactly one supported default tile");
  for (const auto& output : {base, direct}) {
    require(output.size() == t * n, "incomplete output");
    for (uint16_t b : output)
      require((b & 0x7f80u) != 0x7f80u, "nonfinite or unwritten BF16 output");
  }
  save(out / "default.bf16", base.data(), base.size() * 2);
  save(out / "calibration.bf16", direct.data(), direct.size() * 2);
  unsigned differing = 0;
  for (size_t i = 0; i < base.size(); ++i) differing += base[i] != direct[i];
  // Paired wall time includes allocation, recording, synchronization, host output
  // copy and (direct arm only) canary initialization/checks; not GPU timestamps.
  std::ostringstream result;
  result.precision(17);
  result << "{\"purpose\":\"calibration_only\",\"production_engagement\":false,"
               "\"arm\":\"" << calibration::arm << "\",\"k\":" << k
            << ",\"n\":" << n << ",\"t\":" << t << ",\"seed\":0,\"two_word\":"
            << (calibration::two_word(k, n) ? "true" : "false")
            << ",\"differing_bf16_elements\":" << differing
            << ",\"x_shape\":[1," << t << ',' << k << "],\"wq_shape\":["
            << n << ',' << k / 8 << "],\"affine_shape\":[" << n << ',' << k / 64
            << "],\"x_scales_biases_dtype\":\"bfloat16\",\"wq_dtype\":\"uint32\","
               "\"layout\":\"row_major\",\"lhs_base_elements\":8,\"output_guard_elements\":8,"
               "\"timing\":\"paired_host_wall_with_sync_and_output_copy\","
               "\"rounds\":[";
  for (unsigned round = 0; round < 3; ++round) {
    double times[2]{};
    for (unsigned position = 0; position < 2; ++position) {
      const unsigned arm = (round + position) % 2;
      auto start = Clock::now();
      const auto got = arm == 0 ? default_arm() : batch4_arm();
      times[arm] = std::chrono::duration<double>(Clock::now() - start).count();
      require(got == (arm == 0 ? base : direct), "arm output changed during timing");
    }
    if (round) result << ',';
    result << "{\"default_seconds\":" << times[0]
           << ",\"calibration_seconds\":" << times[1] << '}';
  }
  for (const auto& [a, raw] : originals)
    require(std::memcmp(bytes(*a), raw.data(), raw.size()) == 0, "input bytes changed");
  require(std::memcmp(xp + guard, bytes(x), x.nbytes()) == 0, "direct input differs");
  for (unsigned i = 0; i < guard; ++i)
    require(xp[i] == canary && xp[guard + t * k + i] == canary, "input canary changed");
  result << "],\"inputs_unchanged\":true,\"qualification\":null}\n";
  const auto text = result.str();
  save(out / "result.json", text.data(), text.size());
  std::cout << text;
}
} // namespace

int main(int argc, char** argv) {
  try {
    require(argc == 6, "usage: qmv-small-t-calibrate K N T FRESH_OUT EXPECTED_LIB");
    auto number = [](const char* value) {
      size_t used; unsigned long n = std::stoul(value, &used);
      require(used == std::strlen(value) && n <= UINT32_MAX, "invalid integer");
      return static_cast<unsigned>(n);
    };
    unsigned k = number(argv[1]), n = number(argv[2]), t = number(argv[3]);
    calibration::plan(k, n, t, guard, guard);
    require(std::any_of(calibration::cells.begin(), calibration::cells.end(),
            [&](auto cell) { return k == cell.k && n == cell.n; }), "unknown calibration cell");
    require(std::endian::native == std::endian::little, "little-endian host required");
    fs::path out(argv[4]);
    require(!fs::exists(out) && !fs::is_symlink(out), "output must be absent, no resume");
    clean_route_environment();
    Dl_info info{};
    require(dladdr(reinterpret_cast<void*>(&om::get_command_encoder), &info) && info.dli_fname,
            "cannot resolve loaded library");
    require(fs::canonical(info.dli_fname) == fs::canonical(argv[5]), "loaded library path mismatch");
    std::cerr << "CALIBRATION loaded_lib=" << fs::canonical(info.dli_fname) << '\n';
    Stream stream = new_stream(Device::gpu);
    gate(om::get_command_encoder(stream).device());
    require(fs::create_directory(out), "cannot create fresh output directory");
    run_case(k, n, t, out, stream);
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "CALIBRATION REFUSED: " << e.what() << '\n';
    return 1;
  }
}
