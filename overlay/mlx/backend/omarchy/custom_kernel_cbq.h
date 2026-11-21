// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

// Included inside the custom-kernel translator's private namespace.
#include "cbq_msl.inc"

constexpr uint32_t kCbqThreads = 64;

constexpr std::string_view cbq_glsl = R"GLSL(
const uint SB = K / 32u;
const uint SUP = K / 256u;

#if !CBQ_GLU
#define QH_B QH_A
#define QS_B QS_A
#define D_B D_A
#endif

uint cbq_header(uint row, uint sb, bool up) {
  uint i = row * SB + sb;
  return uint(up ? QH_B[i] : QH_A[i]);
}

uint cbq_codes(uint row, uint sb, bool up) {
  uint i = row * SB + sb;
  return up ? QS_B[i] : QS_A[i];
}

uint cbq_code(uint codes, uint l, uint h) {
  return ((codes >> (8u * l)) & 0xffu) | (((h >> (3u * l)) & 7u) << 8u);
}

float cbq_coeff(uint word, uint j, float delta) {
  return float(bitfieldExtract(int(word), int(8u * (j & 3u)), 8)) + delta;
}

float cbq_delta(uint h) {
  return (h & 0x8000u) != 0u ? -0.125 : 0.125;
}

float cbq_multiplier(uint h) {
  return float(2u * ((h >> 12u) & 7u) + 1u);
}

#if !CBQ_GRAD_D
float cbq_scale(uint row, uint sup, uint h, bool up) {
  uint i = row * SUP + sup;
  return _mlx_bf16_to_float(up ? D_B[i] : D_A[i]) * cbq_multiplier(h);
}
#endif

#if CBQ_GRAD_D || CBQ_GRAD_X
// The reference uses fp64 and a relative-only gradient tolerance. Recover
// product and summation roundoff without requiring Vulkan shaderFloat64.
void cbq_add_product(inout float sum, inout float error, float a, float b) {
  precise float product = a * b;
  precise float next = sum + product;
  precise float residual = abs(sum) >= abs(product)
      ? (sum - next) + product : (product - next) + sum;
  precise float product_error = fma(a, b, -product);
  error += residual + product_error;
  sum = next;
}
#endif

#if !CBQ_GRAD_D && !CBQ_GRAD_X
float cbq_gather(uint row, uint xbase, bool up) {
  float sum = 0.0;
  for (uint sb = 0u; sb < SB; ++sb) {
    uint h = cbq_header(row, sb, up);
    uint codes = cbq_codes(row, sb, up);
    float delta = cbq_delta(h);
    float dot32 = 0.0;
    for (uint l = 0u; l < 4u; ++l) {
      uint gi = cbq_code(codes, l, h);
      uvec2 words = uvec2(LUT[gi * 2u], LUT[gi * 2u + 1u]);
      uint base = xbase + sb * 32u + l * 8u;
      for (uint j = 0u; j < 8u; ++j) {
        dot32 = fma(cbq_coeff(words[j / 4u], j, delta),
                    _mlx_bf16_to_float(ACT[base + j]), dot32);
      }
    }
    sum = fma(cbq_scale(row, sb / 8u, h, up), dot32, sum);
  }
  return sum;
}
#endif

void main() {
  const uint step = gl_NumWorkGroups.x * gl_WorkGroupSize.x;
  const uint experts = uint(QS_A.length()) / SB / O;
#if !CBQ_GRAD_X
  const uint tokens = uint(ACT.length()) / K;
#endif
#if CBQ_GRAD_D
  for (uint i = gl_GlobalInvocationID.x; i < uint(OUT_A.length()); i += step) {
    uint row = i / SUP, sup = i % SUP;
    uint e = row / O, o = row % O;
    float sum = uintBitsToFloat(OUT_A[i]), error = 0.0;
    for (uint r = 0u; r < R; ++r) {
      uint tok = TOK[r];
      if (EXP[r] != e || tok >= tokens) continue;
      float ct = _mlx_bf16_to_float(COT[r * O + o]);
      if (ct == 0.0) continue;
      for (uint sb = sup * 8u; sb < sup * 8u + 8u; ++sb) {
        uint h = cbq_header(row, sb, false);
        uint codes = cbq_codes(row, sb, false);
        float delta = cbq_delta(h);
        float scaled = ct * cbq_multiplier(h);
        for (uint l = 0u; l < 4u; ++l) {
          uint gi = cbq_code(codes, l, h);
          uvec2 words = uvec2(LUT[gi * 2u], LUT[gi * 2u + 1u]);
          uint base = tok * K + sb * 32u + l * 8u;
          for (uint j = 0u; j < 8u; ++j) {
            cbq_add_product(sum, error,
                scaled * _mlx_bf16_to_float(ACT[base + j]),
                cbq_coeff(words[j / 4u], j, delta));
          }
        }
      }
    }
    OUT_A[i] = floatBitsToUint(sum + error);
  }
#elif CBQ_GRAD_X
  for (uint i = gl_GlobalInvocationID.x; i < uint(OUT_A.length()); i += step) {
    uint tok = i / K, k = i % K;
    uint sb = k / 32u, l = (k / 8u) & 3u, j = k & 7u;
    float sum = uintBitsToFloat(OUT_A[i]), error = 0.0;
    for (uint r = 0u; r < R; ++r) {
      uint e = EXP[r];
      if (TOK[r] != tok || e >= experts) continue;
      for (uint o = 0u; o < O; ++o) {
        float ct = _mlx_bf16_to_float(COT[r * O + o]);
        if (ct == 0.0) continue;
        uint row = e * O + o;
        uint h = cbq_header(row, sb, false);
        uint gi = cbq_code(cbq_codes(row, sb, false), l, h);
        cbq_add_product(sum, error, ct * cbq_scale(row, k / 256u, h, false),
                        cbq_coeff(LUT[gi * 2u + j / 4u], j, cbq_delta(h)));
      }
    }
    OUT_A[i] = floatBitsToUint(sum + error);
  }
#else
  for (uint i = gl_GlobalInvocationID.x; i < R * O; i += step) {
    uint r = i / O, o = i % O;
    uint e = EXP[r], tok = TOK[r];
    if (e >= experts || tok >= tokens) {
      OUT_A[i] = uint16_t(0x7fc0u);
#if CBQ_GLU
      OUT_B[i] = uint16_t(0x7fc0u);
#endif
      continue;
    }
    uint row = e * O + o;
    float value = cbq_gather(row, tok * K, false);
#if CBQ_SITU
    float u = _mlx_bf16_round(value);
    float g = _mlx_bf16_to_float(GATE[i]);
    value = (4.0 * tanh(g / 4.0)) * (1.0 / (1.0 + exp(-g))) * (25.0 * tanh(u / 25.0));
#endif
    OUT_A[i] = _mlx_float_to_bf16(value);
#if CBQ_GLU
    OUT_B[i] = _mlx_float_to_bf16(cbq_gather(row, tok * K, true));
#endif
  }
#endif
}
)GLSL";

std::optional<Translation> translate_cbq(
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
  constexpr std::string_view prefix = "custom_kernel_cbq_";
  if (!entry.starts_with(prefix)) {
    return std::nullopt;
  }
  struct Case {
    std::string_view name;
    std::string_view body;
    CbqPort port;
    int threads;
    int tile;
  };
  constexpr Case cases[] = {
      {"gather_mm", cbq_gather_mm, CbqGather, 64, 0},
      {"gather_mm_v2", cbq_gather_mm_v2, CbqGather, 256, -1},
      {"gather_mm_v3", cbq_gather_mm_v3, CbqGather, 256, 8},
      {"gather_mm_v4", cbq_gather_mm_v4, CbqGather, 128, 32},
      {"gather_mm_v3_situ", cbq_gather_mm_v3_situ, CbqSitu, 256, 8},
      {"gather_mm_v4_situ", cbq_gather_mm_v4_situ, CbqSitu, 128, 32},
      {"gather_mm_glu", cbq_gather_mm_glu, CbqGlu, 256, 4},
      {"grad_d", cbq_grad_d, CbqGradD, 64, 0},
      {"grad_x", cbq_grad_x, CbqGradX, 64, 0},
  };
  const auto delim = entry.find("__", prefix.size());
  const auto name = entry.substr(prefix.size(), delim - prefix.size());
  const auto* spec = std::find_if(std::begin(cases), std::end(cases),
      [&](const Case& c) { return c.name == name; });
  const auto wrapped = source.substr(body_open + 1, body_close - body_open - 1);
  if (spec == std::end(cases) || wrapped.size() != spec->body.size() + 2 ||
      wrapped.front() != '\n' || wrapped.back() != '\n' ||
      wrapped.substr(1, wrapped.size() - 2) != spec->body) {
    return std::nullopt;
  }

  const bool gd = spec->port == CbqGradD, gx = spec->port == CbqGradX;
  const bool glu = spec->port == CbqGlu, situ = spec->port == CbqSitu;
  const auto refuse = [&]() {
    throw std::runtime_error("unsupported CBQ kernel ABI or launch: " + std::string(name));
  };
  const std::string decl = std::string("template <typename InT, ") +
      (gd ? "" : "typename DT, ") + "int K, int O, int R>\n";
  if (source.substr(0, marker) != decl || output_count != (glu ? 2u : 1u)) {
    refuse();
  }

  const std::string entry_name(entry);
  const std::string arg_types = gd ? "bfloat16_t, " : "bfloat16_t, bfloat16_t, ";
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
  if (parts.size() != 3) {
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
  const int k = dimension(parts[0]), o = dimension(parts[1]), r = dimension(parts[2]);
  const int64_t x = gx ? k / 8 : spec->tile < 0 ? spec->threads : spec->tile == 0 ? o :
      (static_cast<int64_t>(o) + spec->tile - 1) / spec->tile * spec->threads;
  if (k % 256 != 0 || (!glu && spec->tile > 0 && o % spec->tile != 0) ||
      grid != std::make_tuple(x, r, 1) ||
      threadgroup != std::make_tuple(spec->threads, 1, 1)) {
    refuse();
  }

  struct Buffer {
    const char* type;
    const char* name;
    const char* alias;
  };
  constexpr auto bf = "bfloat16_t";
  std::vector<Buffer> abi;
  if (!gx) abi.push_back({bf, "x", "ACT"});
  if (gd || gx) abi.push_back({bf, "cot", "COT"});
  abi.push_back({"uint", "eidx", "EXP"});
  abi.push_back({"uint", "tokidx", "TOK"});
  abi.push_back({"uint8_t", glu ? "a_qs" : "cb_qs", "QS_A"});
  abi.push_back({"uint16_t", glu ? "a_qh" : "cb_qh", "QH_A"});
  if (!gd) abi.push_back({bf, glu ? "a_d" : "cb_d", "D_A"});
  if (glu) {
    abi.push_back({"uint8_t", "b_qs", "QS_B"});
    abi.push_back({"uint16_t", "b_qh", "QH_B"});
    abi.push_back({bf, "b_d", "D_B"});
  }
  abi.push_back({"int8_t", "grid", "LUT"});
  if (situ) abi.push_back({bf, "gbuf", "GATE"});
  const size_t inputs = abi.size();
  abi.push_back({gd || gx ? "float" : bf,
                 gd ? "grad_d" : gx ? "grad_x" : glu ? "yg" : "y", "OUT_A"});
  if (glu) abi.push_back({bf, "yu", "OUT_B"});
  if (parameters.size() != abi.size()) {
    refuse();
  }
  std::ostringstream signature, glsl;
  signature << '\n';
  glsl << "#version 460\n"
       << "#extension GL_EXT_shader_explicit_arithmetic_types_int16 : require\n"
       << "#extension GL_EXT_shader_16bit_storage : require\n"
       << "layout(local_size_x=" << kCbqThreads << ") in;\n"
       << "#define K " << k << "u\n#define O " << o << "u\n#define R " << r << "u\n"
       << "#define CBQ_GRAD_D " << gd << "\n#define CBQ_GRAD_X " << gx << "\n"
       << "#define CBQ_GLU " << glu << "\n#define CBQ_SITU " << situ << '\n';
  for (size_t i = 0; i < abi.size(); ++i) {
    const auto& p = parameters[i];
    const auto& a = abi[i];
    const bool out = i >= inputs, atomic = out && (gd || gx);
    // MLX spells the two uint32 index types uint32_t in live signatures.
    const bool type_matches =
        p.type == a.type ||
        (std::string_view(a.type) == "uint" && p.type == "uint32_t");
    if (p.scalar || p.atomic != atomic || p.name != a.name || !type_matches ||
        (out && p.constant_space)) {
      refuse();
    }
    signature << "  " << (out ? "" : "const ")
              << (p.constant_space ? "constant " : "device ")
              << (atomic ? "atomic<" : "") << p.type << (atomic ? ">" : "")
              << "* " << a.name << " [[buffer(" << i << ")]],\n";
    if (p.type == "int8_t" || p.type == "uint8_t") {
      // The logical device does not enable storageBuffer8BitAccess.
      glsl << buffer_declaration({"uint", p.name, p.binding, false, false, false}, out);
    } else {
      glsl << buffer_declaration(p, out);
    }
    glsl << "#define " << a.alias << " _b" << i << ".data\n";
  }
  if (spec->tile == 0) {
    signature << "  uint3 thread_position_in_grid [[thread_position_in_grid]]";
  } else {
    signature << "  uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]],\n"
              << "  uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]";
  }
  if (source.substr(arguments_open + 1, arguments_close - arguments_open - 1) != signature.str() ||
      source.substr(arguments_close + 1, body_open - arguments_close - 1) != " ") {
    refuse();
  }
  glsl << bf16_glsl_helpers << cbq_glsl;
  return Translation{glsl.str(), std::move(parameters), {spec->port, k, o, r}};
}

bool cbq_array_matches(const array& a, Dtype dtype, std::initializer_list<int> dims) {
  return a.dtype() == dtype && a.flags().row_contiguous && a.offset() == 0 &&
      a.data_size() == a.size() && a.size() <= INT32_MAX &&
      std::equal(a.shape().begin(), a.shape().end(), dims.begin(), dims.end());
}

bool cbq_dispatch_matches(
    const CbqDispatch& plan, const std::vector<array>& inputs,
    const std::vector<array>& outputs, bool initialized) {
  const auto [port, k, o, r] = plan;
  const bool gd = port == CbqGradD, gx = port == CbqGradX;
  const bool glu = port == CbqGlu, situ = port == CbqSitu;
  const size_t expected = gd || gx ? 7 : glu ? 10 : situ ? 8 : 7;
  if (inputs.size() != expected || outputs.size() != (glu ? 2u : 1u) ||
      ((gd || gx) && !initialized)) {
    return false;
  }
  size_t i = 0;
  if (!gx) {
    const auto& x = inputs[i++];
    if (x.ndim() != 2 || x.shape(0) <= 0 ||
        !cbq_array_matches(x, bfloat16, {x.shape(0), k})) {
      return false;
    }
  }
  if ((gd || gx) && !cbq_array_matches(inputs[i++], bfloat16, {r, o})) {
    return false;
  }
  if (!cbq_array_matches(inputs[i++], uint32, {r}) ||
      !cbq_array_matches(inputs[i++], uint32, {r})) {
    return false;
  }
  const auto& qs = inputs[i];
  if (qs.ndim() != 3 || qs.shape(0) <= 0) {
    return false;
  }
  const int e = qs.shape(0);
  for (int stack = 0; stack < (glu ? 2 : 1); ++stack) {
    if (!cbq_array_matches(inputs[i++], uint8, {e, o, k / 8}) ||
        !cbq_array_matches(inputs[i++], uint16, {e, o, k / 32}) ||
        (!gd && !cbq_array_matches(inputs[i++], bfloat16, {e, o, k / 256}))) {
      return false;
    }
  }
  if (!cbq_array_matches(inputs[i++], int8, {2048 * 8}) ||
      (situ && !cbq_array_matches(inputs[i], bfloat16, {r, o}))) {
    return false;
  }
  const auto& out = outputs.front();
  if (gd) return cbq_array_matches(out, float32, {e, o, k / 256});
  if (gx) return out.ndim() == 2 && out.shape(0) > 0 &&
      cbq_array_matches(out, float32, {out.shape(0), k});
  return cbq_array_matches(out, bfloat16, {r, o}) &&
      (!glu || cbq_array_matches(outputs[1], bfloat16, {r, o}));
}
