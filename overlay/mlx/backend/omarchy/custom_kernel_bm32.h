// Nemotron BM32 steel linear NT hand port.
//
// mlx_vlm nemotron_labs_diffusion wraps a Steel GEMMKernel<BM=32, BN=64,
// BK=16, WM=2, WN=2> in mx.fast.metal_kernel to compute x @ weight.T for
// sequence-length-32 projections. simdgroup_matrix arithmetic cannot
// survive the generic translator, so the exact shipped body, ABI, and
// launch shape are pinned here and the same 32x64 BF16 tile is emitted in
// GLSL. A source that merely resembles the shipped kernel falls through
// to the generic path (which then refuses the steel constructs), while a
// matching body with a wrong ABI or launch is a hard porting error.

#include "bm32_msl.inc"

constexpr int kBm32TileM = 32;
constexpr int kBm32TileN = 64;
constexpr int kBm32Threads = 128;

constexpr std::string_view bm32_glsl = R"GLSL(
// Dense NT tile GEMM: OUT[r,c] = sum_k X[r,k] * W[c,k]. Each workgroup owns
// one 32x64 output tile; each of the 128 lanes owns a 4x4 block. BF16
// storage, FP32 accumulation, RNE BF16 store.
shared float tile_a[16][32];
shared float tile_b[16][64];

void main() {
  uint row0 = gl_WorkGroupID.y * 32u;
  uint col0 = gl_WorkGroupID.x * 64u;
  uint row = (gl_LocalInvocationIndex / 16u) * 4u;
  uint col = (gl_LocalInvocationIndex % 16u) * 4u;
  uint lane_k = gl_LocalInvocationIndex % 8u;
  uint lane_row = gl_LocalInvocationIndex / 8u;
  vec4 acc[4] = vec4[4](vec4(0.0), vec4(0.0), vec4(0.0), vec4(0.0));

  for (uint kb = 0u; kb < K; kb += 16u) {
    for (uint r = lane_row; r < 32u; r += 16u) {
      uint base = (row0 + r) * K + kb + 2u * lane_k;
      tile_a[2u * lane_k][r] = _mlx_bf16_to_float(X[base]);
      tile_a[2u * lane_k + 1u][r] = _mlx_bf16_to_float(X[base + 1u]);
    }
    for (uint c = lane_row; c < 64u; c += 16u) {
      uint base = (col0 + c) * K + kb + 2u * lane_k;
      tile_b[2u * lane_k][c] = _mlx_bf16_to_float(W[base]);
      tile_b[2u * lane_k + 1u][c] = _mlx_bf16_to_float(W[base + 1u]);
    }
    barrier();
    for (uint t = 0u; t < 16u; ++t) {
      vec4 b = vec4(tile_b[t][col], tile_b[t][col + 1u],
                    tile_b[t][col + 2u], tile_b[t][col + 3u]);
      for (uint r = 0u; r < 4u; ++r) {
        acc[r] = fma(vec4(tile_a[t][row + r]), b, acc[r]);
      }
    }
    barrier();
  }

  for (uint r = 0u; r < 4u; ++r) {
    uint base = (row0 + row + r) * O + col0 + col;
    OUT[base] = _mlx_float_to_bf16(acc[r].x);
    OUT[base + 1u] = _mlx_float_to_bf16(acc[r].y);
    OUT[base + 2u] = _mlx_float_to_bf16(acc[r].z);
    OUT[base + 3u] = _mlx_float_to_bf16(acc[r].w);
  }
}
)GLSL";

std::optional<Translation> translate_bm32(
    std::string_view source,
    size_t marker,
    size_t arguments_open,
    size_t arguments_close,
    size_t body_open,
    size_t body_close,
    std::vector<Parameter>& parameters,
    const std::tuple<int, int, int>& grid,
    const std::tuple<int, int, int>& threadgroup,
    size_t output_count) {
  constexpr std::string_view kernel_marker = "[[kernel]] void ";
  const auto entry = source.substr(
      marker + kernel_marker.size(), arguments_open - marker - kernel_marker.size());
  constexpr std::string_view prefix =
      "custom_kernel_nemotron_bm32_steel_linear_nt_";
  if (!entry.starts_with(prefix)) {
    return std::nullopt;
  }
  const auto wrapped = source.substr(body_open + 1, body_close - body_open - 1);
  if (wrapped.size() != nemotron_bm32_steel_linear_nt.size() + 2 ||
      wrapped.front() != '\n' || wrapped.back() != '\n' ||
      wrapped.substr(1, wrapped.size() - 2) != nemotron_bm32_steel_linear_nt) {
    return std::nullopt;
  }

  const auto refuse = []() {
    throw std::runtime_error(
        "unsupported BM32 kernel ABI or launch: nemotron_bm32_steel_linear_nt");
  };

  // The generated source is <expanded steel header>\n followed by the
  // template declaration; the body pin above fixes the semantics, so the
  // header itself only has to precede the expected declaration.
  constexpr std::string_view decl = "template <typename T, int K, int O>\n";
  const auto preamble = source.substr(0, marker);
  if (preamble.size() < decl.size() ||
      preamble.substr(preamble.size() - decl.size()) != decl ||
      output_count != 1) {
    refuse();
  }

  const std::string entry_name(entry);
  const std::string arg_types = "bfloat16_t, ";
  const std::string tail_prefix = "template [[host_name(\"" + entry_name +
      "\")]] [[kernel]] decltype(" + entry_name + "<" + arg_types;
  const std::string tail = trim(std::string(source.substr(body_close + 1)));
  if (!tail.starts_with(tail_prefix)) {
    refuse();
  }
  const auto end = tail.find('>', tail_prefix.size());
  if (end == std::string::npos) {
    refuse();
  }
  const std::string dims = tail.substr(tail_prefix.size(), end - tail_prefix.size());
  if (tail.substr(end) != ">) " + entry_name + "<" + arg_types + dims + ">;") {
    refuse();
  }
  const auto parts = split(dims, ',');
  if (parts.size() != 2) {
    refuse();
  }
  const auto dimension = [&](const std::string& text) {
    int value = 0;
    const auto [end, ec] = std::from_chars(text.data(), text.data() + text.size(), value);
    if (ec != std::errc{} || end != text.data() + text.size() || value <= 0) {
      refuse();
    }
    return value;
  };
  const int k = dimension(parts[0]), o = dimension(parts[1]);

  // grid is in threads: (128 * O/64, rows/32, 1) over a (128,1,1)
  // threadgroup; each workgroup covers one 32x64 output tile.
  const auto [grid_x, grid_y, grid_z] = grid;
  if (k % 16 != 0 || o % kBm32TileN != 0 || grid_z != 1 || grid_y < 1 ||
      grid_x != kBm32Threads * (o / kBm32TileN) ||
      threadgroup != std::make_tuple(kBm32Threads, 1, 1)) {
    refuse();
  }
  const int rows = grid_y * kBm32TileM;

  constexpr auto bf = "bfloat16_t";
  constexpr std::string_view abi_names[] = {"x", "weight", "out"};
  if (parameters.size() != 3) {
    refuse();
  }
  std::ostringstream signature;
  signature << '\n';
  std::ostringstream glsl;
  glsl << "#version 460\n"
       << "#extension GL_EXT_shader_explicit_arithmetic_types_int16 : require\n"
       << "#extension GL_EXT_shader_16bit_storage : require\n"
       << "layout(local_size_x=" << kBm32Threads << ") in;\n"
       << "#define K " << k << "u\n#define O " << o << "u\n";
  constexpr std::string_view aliases[] = {"X", "W", "OUT"};
  for (size_t i = 0; i < 3; ++i) {
    const auto& p = parameters[i];
    const bool out = i == 2;
    if (p.scalar || p.atomic || p.name != abi_names[i] || p.type != bf ||
        p.constant_space || p.binding != i) {
      refuse();
    }
    signature << "  " << (out ? "" : "const ") << "device " << p.type << "* "
              << abi_names[i] << " [[buffer(" << i << ")]],\n";
    glsl << buffer_declaration(p, out);
    glsl << "#define " << aliases[i] << " _b" << i << ".data\n";
  }
  signature << "  uint3 simdgroup_index_in_threadgroup "
               "[[simdgroup_index_in_threadgroup]],\n"
            << "  uint3 thread_index_in_simdgroup "
               "[[thread_index_in_simdgroup]],\n"
            << "  uint3 threadgroup_position_in_grid "
               "[[threadgroup_position_in_grid]]";
  if (source.substr(arguments_open + 1, arguments_close - arguments_open - 1) !=
          signature.str() ||
      source.substr(arguments_close + 1, body_open - arguments_close - 1) != " ") {
    refuse();
  }
  glsl << bf16_glsl_helpers << bm32_glsl;
  return Translation{glsl.str(), std::move(parameters),
                     {SteelBm32, k, o, rows}};
}

// The launch contract fixes x (rows,K), weight (O,K) and out (rows,O) as
// contiguous BF16 row-major arrays, matching the suite spec.
bool bm32_dispatch_matches(
    const CbqDispatch& dispatch,
    const std::vector<array>& inputs,
    const std::vector<array>& outputs) {
  return inputs.size() == 2 && outputs.size() == 1 &&
      cbq_array_matches(
          inputs[0], bfloat16, {dispatch.r, dispatch.k}) &&
      cbq_array_matches(
          inputs[1], bfloat16, {dispatch.o, dispatch.k}) &&
      cbq_array_matches(
          outputs[0], bfloat16, {dispatch.r, dispatch.o});
}
