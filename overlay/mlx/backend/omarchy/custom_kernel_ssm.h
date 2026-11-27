// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT
//
// Nemotron-H / Mamba-2 selective scan decode step (ssm_kernel) hand port.
//
// mlx_lm/models/ssm.py wraps this 1-dispatch decode recurrence in
// mx.fast.metal_kernel("ssm_kernel"). Its MSL uses integer `auto` locals
// (n, h_idx, g_idx, ds_idx, d_idx, s_idx, idx), which the generic translator
// refuses by name to prevent float-precision index math. The exact shipped
// body, ABI, and launch shape are pinned here and lowered directly to
// Vulkan 1.3 GLSL (32-lane subgroupAdd fast path on 32-wide subgroups,
// shared-memory tree reduction fallback on non-32 subgroups).

#include "ssm_msl.inc"

constexpr bool ssm_supported_type(std::string_view type) {
  return type == "float" || type == "float16_t" || type == "half" ||
      type == "bfloat16_t";
}

constexpr bool ssm_is_fp16(std::string_view type) {
  return type == "float16_t" || type == "half";
}

constexpr std::string_view ssm_read_expr(std::string_view type) {
  if (type == "bfloat16_t") return "_mlx_bf16_to_float";
  if (ssm_is_fp16(type)) return "float";
  return "";
}

constexpr std::string_view ssm_write_expr(std::string_view type) {
  if (type == "bfloat16_t") return "_mlx_float_to_bf16";
  if (ssm_is_fp16(type)) return "float16_t";
  return "";
}

std::optional<Translation> translate_ssm(
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
  constexpr std::string_view prefix = "custom_kernel_ssm_kernel_";
  if (!entry.starts_with(prefix)) {
    return std::nullopt;
  }

  // Exact shipped body pin: any body mutation falls through to the generic
  // translator (which refuses integer `auto` by name). Trim leading/trailing
  // whitespace only (verbatim mlx_lm source has trailing indent before `"""`).
  const auto raw_body = source.substr(body_open + 1, body_close - body_open - 1);
  if (trim(std::string(raw_body)) != trim(std::string(ssm_kernel))) {
    return std::nullopt;
  }

  const auto refuse = []() {
    throw std::runtime_error(
        "unsupported Mamba-2 SSM kernel ABI or launch: ssm_kernel");
  };

  constexpr std::string_view decl =
      "template <typename T, typename U, int Dh, int Ds, int H, int G>\n";
  const auto preamble = source.substr(0, marker);
  if (preamble.size() < decl.size() ||
      preamble.substr(preamble.size() - decl.size()) != decl ||
      output_count != 2) {
    refuse();
  }

  const std::string entry_name(entry);
  const std::string tail_prefix = "template [[host_name(\"" + entry_name +
      "\")]] [[kernel]] decltype(" + entry_name + "<";
  const std::string tail = trim(std::string(source.substr(body_close + 1)));
  if (!tail.starts_with(tail_prefix)) {
    refuse();
  }
  const auto end = tail.find('>', tail_prefix.size());
  if (end == std::string::npos) {
    refuse();
  }
  const std::string targs =
      tail.substr(tail_prefix.size(), end - tail_prefix.size());
  if (tail.substr(end) != ">) " + entry_name + "<" + targs + ">;") {
    refuse();
  }
  const auto parts = split(targs, ',');
  if (parts.size() != 6) {
    refuse();
  }
  const std::string t_type = trim(parts[0]);
  const std::string u_type = trim(parts[1]);
  if (!ssm_supported_type(t_type) || !ssm_supported_type(u_type)) {
    refuse();
  }
  const auto parse_pos = [&](const std::string& raw) {
    const std::string text = trim(raw);
    int value = 0;
    const auto [ptr, ec] =
        std::from_chars(text.data(), text.data() + text.size(), value);
    if (ec != std::errc{} || ptr != text.data() + text.size() || value <= 0) {
      refuse();
    }
    return value;
  };
  const int dh = parse_pos(parts[2]);
  const int ds = parse_pos(parts[3]);
  const int h = parse_pos(parts[4]);
  const int g = parse_pos(parts[5]);
  if (ds % 32 != 0 || h % g != 0) {
    refuse();
  }

  const auto [grid_x, grid_y, grid_z] = grid;
  const auto [tg_x, tg_y, tg_z] = threadgroup;
  if (grid_x != 32 || grid_y != dh || grid_z < h || grid_z % h != 0 ||
      tg_x != 32 || tg_y < 1 || tg_y > 32 || tg_z != 1) {
    refuse();
  }
  const int batch = grid_z / h;

  constexpr std::string_view abi_names[9] = {
      "X", "A_log", "B", "C", "D", "dt", "state_in", "out", "state_out"};
  if (parameters.size() != 9) {
    refuse();
  }
  bool needs_bf16 = false;
  bool needs_fp16 = false;
  std::ostringstream signature;
  signature << '\n';
  for (size_t i = 0; i < 9; ++i) {
    const auto& p = parameters[i];
    const bool is_out = i >= 7;
    if (p.scalar || p.atomic || p.name != abi_names[i] ||
        !ssm_supported_type(p.type) || (is_out && p.constant_space) ||
        p.binding != i) {
      refuse();
    }
    if ((i == 0 || i == 7) && p.type != t_type) refuse();
    if ((i == 6 || i == 8) && p.type != u_type) refuse();
    if (p.type == "bfloat16_t") needs_bf16 = true;
    if (ssm_is_fp16(p.type)) needs_fp16 = true;
    signature << "  " << (is_out ? "" : "const ")
              << (p.constant_space ? "constant " : "device ") << p.type << "* "
              << abi_names[i] << " [[buffer(" << i << ")]],\n";
  }
  signature << "  uint thread_index_in_simdgroup [[thread_index_in_simdgroup]],\n"
            << "  uint3 thread_position_in_grid [[thread_position_in_grid]],\n"
            << "  uint3 thread_position_in_threadgroup "
               "[[thread_position_in_threadgroup]]";
  if (source.substr(arguments_open + 1, arguments_close - arguments_open - 1) !=
          signature.str() ||
      source.substr(arguments_close + 1, body_open - arguments_close - 1) != " ") {
    refuse();
  }

  std::ostringstream glsl;
  glsl << "#version 460\n"
       << "#extension GL_KHR_shader_subgroup_basic : require\n"
       << "#extension GL_KHR_shader_subgroup_arithmetic : require\n";
  if (needs_bf16) {
    glsl << "#extension GL_EXT_shader_explicit_arithmetic_types_int16 : require\n"
         << "#extension GL_EXT_shader_16bit_storage : require\n";
  }
  if (needs_fp16) {
    glsl << "#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require\n";
    if (!needs_bf16) {
      glsl << "#extension GL_EXT_shader_16bit_storage : require\n";
    }
  }
  glsl << "layout(local_size_x=32, local_size_y=" << tg_y
       << ", local_size_z=1) in;\n"
       << "#define SSM_DH " << dh << "u\n"
       << "#define SSM_DS " << ds << "u\n"
       << "#define SSM_H " << h << "u\n"
       << "#define SSM_G " << g << "u\n"
       << "#define SSM_N_PER_T " << (ds / 32) << "u\n"
       << "#define SSM_GRID_Z " << grid_z << "u\n"
       << "#define SSM_TY " << tg_y << "u\n";

  constexpr std::string_view aliases[9] = {
      "SSM_X", "SSM_ALOG", "SSM_B", "SSM_C", "SSM_D",
      "SSM_DT", "SSM_SIN", "SSM_OUT", "SSM_SOUT"};
  for (size_t i = 0; i < 9; ++i) {
    glsl << buffer_declaration(parameters[i], i >= 7);
    glsl << "#define " << aliases[i] << " _b" << i << ".data\n";
  }
  if (needs_bf16) {
    glsl << bf16_glsl_helpers;
  }
  auto emit_rd = [&](const char* fn, const std::string& ty) {
    const auto conv = ssm_read_expr(ty);
    glsl << "float " << fn << "(" << glsl_type(ty) << " v) { return "
         << (conv.empty() ? "v" : std::string(conv) + "(v)") << "; }\n";
  };
  auto emit_wr = [&](const char* fn, const std::string& ty) {
    const auto conv = ssm_write_expr(ty);
    glsl << glsl_type(ty) << " " << fn << "(float v) { return "
         << (conv.empty() ? "v" : std::string(conv) + "(v)") << "; }\n";
  };
  emit_rd("_ssm_rd_x", parameters[0].type);
  emit_rd("_ssm_rd_alog", parameters[1].type);
  emit_rd("_ssm_rd_b", parameters[2].type);
  emit_rd("_ssm_rd_c", parameters[3].type);
  emit_rd("_ssm_rd_d", parameters[4].type);
  emit_rd("_ssm_rd_dt", parameters[5].type);
  emit_rd("_ssm_rd_sin", parameters[6].type);
  emit_wr("_ssm_wr_out", parameters[7].type);
  emit_wr("_ssm_wr_sout", parameters[8].type);

  glsl << R"GLSL(
shared float _ssm_sh[SSM_TY][32];

void main() {
  uint ds_idx = gl_LocalInvocationID.x;
  uint ly = gl_LocalInvocationID.y;
  uint d_idx = gl_GlobalInvocationID.y;
  uint n = gl_GlobalInvocationID.z;
  bool in_bounds = (d_idx < SSM_DH) && (n < SSM_GRID_Z);

  float acc = 0.0;
  float x_val = 0.0;
  uint h_idx = 0u;

  if (in_bounds) {
    h_idx = n % SSM_H;
    uint g_idx = n / SSM_G;
    uint x_off = n * SSM_DH + d_idx;
    uint st_off = (n * SSM_DH + d_idx) * SSM_DS;
    uint bc_off = g_idx * SSM_DS;

    float dt_val = _ssm_rd_dt(SSM_DT[n]);
    float A_val = -exp(_ssm_rd_alog(SSM_ALOG[h_idx]));
    float dA = exp(A_val * dt_val);
    x_val = _ssm_rd_x(SSM_X[x_off]);

    for (uint i = 0u; i < SSM_N_PER_T; ++i) {
      uint s_idx = SSM_N_PER_T * ds_idx + i;
      uint idx = st_off + s_idx;
      float dB_by_x = x_val * dt_val * _ssm_rd_b(SSM_B[bc_off + s_idx]);
      float st = dA * _ssm_rd_sin(SSM_SIN[idx]) + dB_by_x;
      SSM_SOUT[idx] = _ssm_wr_sout(st);
      acc += st * _ssm_rd_c(SSM_C[bc_off + s_idx]);
    }
  }

  if (gl_SubgroupSize == 32u) {
    acc = subgroupAdd(acc);
  } else {
    _ssm_sh[ly][ds_idx] = acc;
    barrier();
    for (uint stride = 16u; stride > 0u; stride >>= 1u) {
      if (ds_idx < stride) {
        _ssm_sh[ly][ds_idx] += _ssm_sh[ly][ds_idx + stride];
      }
      barrier();
    }
    acc = _ssm_sh[ly][0];
  }

  if (in_bounds && ds_idx == 0u) {
    SSM_OUT[n * SSM_DH + d_idx] =
        _ssm_wr_out(acc + x_val * _ssm_rd_d(SSM_D[h_idx]));
  }
}
)GLSL";

  return Translation{glsl.str(), std::move(parameters),
                     {Mamba2Ssm, dh, ds, batch, h, g}};
}

bool ssm_dtype_matches(const array& a, const std::string& msl_type) {
  if (msl_type == "float") return a.dtype() == float32;
  if (ssm_is_fp16(msl_type)) return a.dtype() == float16;
  if (msl_type == "bfloat16_t") return a.dtype() == bfloat16;
  return false;
}

bool ssm_dispatch_matches(
    const Translation& translation,
    const std::vector<array>& inputs,
    const std::vector<array>& outputs,
    bool has_init_value) {
  if (has_init_value || inputs.size() != 7 || outputs.size() != 2 ||
      translation.parameters.size() != 9) {
    return false;
  }
  const auto& d = translation.cbq;
  const int dh = d.k;
  const int ds = d.o;
  const int batch = d.r;
  const int h = d.h;
  const int g = d.g;
  if (dh <= 0 || ds <= 0 || ds % 32 != 0 || batch <= 0 || h <= 0 || g <= 0 ||
      h % g != 0) {
    return false;
  }
  const int hb = h / g;
  auto ok_shape = [](const array& a, std::initializer_list<int> shape) {
    if (!a.flags().row_contiguous || a.ndim() != shape.size()) return false;
    size_t i = 0;
    for (int dim : shape) {
      if (a.shape(i++) != dim) return false;
    }
    return true;
  };
  for (size_t i = 0; i < 7; ++i) {
    if (!ssm_dtype_matches(inputs[i], translation.parameters[i].type)) {
      return false;
    }
  }
  for (size_t i = 0; i < 2; ++i) {
    if (!ssm_dtype_matches(outputs[i], translation.parameters[7 + i].type)) {
      return false;
    }
  }
  const bool dt_ok = inputs[5].flags().row_contiguous &&
      inputs[5].size() == static_cast<size_t>(batch) * static_cast<size_t>(h);
  const bool alog_ok = inputs[1].flags().row_contiguous &&
      inputs[1].size() == static_cast<size_t>(h);
  const bool d_ok = inputs[4].flags().row_contiguous &&
      inputs[4].size() == static_cast<size_t>(h);
  return ok_shape(inputs[0], {batch, 1, h, dh}) && alog_ok &&
      ok_shape(inputs[2], {batch, 1, hb, ds}) &&
      ok_shape(inputs[3], {batch, 1, hb, ds}) && d_ok && dt_ok &&
      ok_shape(inputs[6], {batch, h, dh, ds}) &&
      ok_shape(outputs[0], {batch, 1, h, dh}) &&
      ok_shape(outputs[1], {batch, h, dh, ds});
}
