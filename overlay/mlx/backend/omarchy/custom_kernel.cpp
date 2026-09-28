// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

#include "mlx/fast_primitives.h"

#include <fcntl.h>
#include <signal.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <mutex>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "mlx/backend/gpu/copy.h"
#include "mlx/backend/omarchy/allocator.h"
#include "mlx/backend/omarchy/ane/bundle.h"
#include "mlx/backend/omarchy/compute.h"
#include "mlx/backend/omarchy/encoder.h"
#include "mlx/backend/omarchy/unsupported.h"

namespace mlx::core::fast {
namespace {

struct Parameter {
  std::string type;
  std::string name;
  uint32_t binding;
  bool scalar;
  bool atomic;
};

struct Translation {
  std::string glsl;
  std::vector<Parameter> parameters;
};

struct TemporaryFile {
  explicit TemporaryFile(const char* suffix) {
    std::string pattern = "/tmp/mlx-omarchy-custom-XXXXXX";
    pattern += suffix;
    path.assign(pattern.begin(), pattern.end());
    path.push_back('\0');
    fd = mkstemps(path.data(), static_cast<int>(std::strlen(suffix)));
    if (fd < 0) {
      throw std::runtime_error("cannot create a shader compiler temporary file");
    }
    path.pop_back();
  }

  ~TemporaryFile() {
    if (fd >= 0) {
      close(fd);
    }
    if (!path.empty()) {
      unlink(path.c_str());
    }
  }

  TemporaryFile(const TemporaryFile&) = delete;
  TemporaryFile& operator=(const TemporaryFile&) = delete;

  std::string path;
  int fd{-1};
};

std::string trim(std::string value) {
  const auto first = value.find_first_not_of(" \t\r\n");
  if (first == std::string::npos) {
    return {};
  }
  const auto last = value.find_last_not_of(" \t\r\n");
  return value.substr(first, last - first + 1);
}

std::vector<std::string> split(const std::string& value, char delimiter) {
  std::vector<std::string> parts;
  std::istringstream stream(value);
  std::string part;
  while (std::getline(stream, part, delimiter)) {
    parts.push_back(trim(std::move(part)));
  }
  return parts;
}

std::string regex_escape(const std::string& value) {
  static const std::regex special(R"([.^$|()\[\]{}*+?\\])");
  return std::regex_replace(value, special, R"(\$&)");
}

void replace_all(
    std::string& value,
    const std::string& needle,
    const std::string& replacement) {
  size_t position = 0;
  while ((position = value.find(needle, position)) != std::string::npos) {
    value.replace(position, needle.size(), replacement);
    position += replacement.size();
  }
}

void replace_word(
    std::string& value,
    const std::string& needle,
    const std::string& replacement) {
  value = std::regex_replace(
      value, std::regex("\\b" + regex_escape(needle) + "\\b"), replacement);
}

size_t matching_delimiter(
    const std::string& value,
    size_t opening,
    char open,
    char close) {
  int depth = 0;
  for (size_t index = opening; index < value.size(); ++index) {
    if (value[index] == open) {
      ++depth;
    } else if (value[index] == close && --depth == 0) {
      return index;
    }
  }
  throw std::runtime_error("generated MSL has an unbalanced delimiter");
}

std::string glsl_type(const std::string& msl_type) {
  static const std::unordered_map<std::string, std::string> types = {
      {"float", "float"},
      {"float16_t", "float16_t"},
      {"half", "float16_t"},
      {"bfloat16_t", "uint16_t"},
      {"int", "int"},
      {"int8_t", "int8_t"},
      {"int16_t", "int16_t"},
      {"int32_t", "int"},
      {"int64_t", "int64_t"},
      {"uint", "uint"},
      {"uint8_t", "uint8_t"},
      {"uint16_t", "uint16_t"},
      {"uint32_t", "uint"},
      {"uint64_t", "uint64_t"},
  };
  const auto found = types.find(msl_type);
  if (found == types.end()) {
    throw std::runtime_error("unsupported buffer type `" + msl_type + "`");
  }
  return found->second;
}

void translate_types(std::string& code) {
  const std::vector<std::pair<std::string, std::string>> replacements = {
      {"float4", "vec4"},
      {"float3", "vec3"},
      {"float2", "vec2"},
      {"half4", "f16vec4"},
      {"half3", "f16vec3"},
      {"half2", "f16vec2"},
      {"uint4", "uvec4"},
      {"uint3", "uvec3"},
      {"uint2", "uvec2"},
      {"int4", "ivec4"},
      {"int3", "ivec3"},
      {"int2", "ivec2"},
      {"int32_t", "int"},
      {"uint32_t", "uint"},
      {"half", "float16_t"},
  };
  for (const auto& [from, to] : replacements) {
    replace_word(code, from, to);
  }
  code = std::regex_replace(
      code,
      std::regex(R"(static_cast\s*<\s*([A-Za-z_][A-Za-z0-9_]*)\s*>\s*\(([^()]*)\))"),
      "$1($2)");
  code = std::regex_replace(
      code,
      std::regex(R"(as_type\s*<\s*uint\s*>\s*\(([^()]*)\))"),
      "floatBitsToUint($1)");
  code = std::regex_replace(
      code,
      std::regex(R"(as_type\s*<\s*float\s*>\s*\(([^()]*)\))"),
      "uintBitsToFloat($1)");
}

std::vector<Parameter> parse_parameters(const std::string& signature) {
  const std::regex parameter_pattern(
      R"((?:const\s+)?(?:device|constant)\s+(atomic<)?([A-Za-z_][A-Za-z0-9_]*)(?:>)?\s*([*&])\s*([A-Za-z_][A-Za-z0-9_]*)\s*\[\[buffer\(([0-9]+)\)\]\])");
  std::vector<Parameter> parameters;
  for (std::sregex_iterator it(
           signature.begin(), signature.end(), parameter_pattern),
       end;
       it != end;
       ++it) {
    parameters.push_back(
        {(*it)[2].str(),
         (*it)[4].str(),
         static_cast<uint32_t>(std::stoul((*it)[5].str())),
         (*it)[3].str() == "&",
         (*it)[1].matched});
  }
  std::sort(
      parameters.begin(),
      parameters.end(),
      [](const Parameter& lhs, const Parameter& rhs) {
        return lhs.binding < rhs.binding;
      });
  for (size_t index = 0; index < parameters.size(); ++index) {
    if (parameters[index].binding != index) {
      throw std::runtime_error("generated MSL has non-contiguous buffer bindings");
    }
  }
  if (parameters.empty()) {
    throw std::runtime_error("generated MSL has no buffer parameters");
  }
  return parameters;
}

void resolve_kernel_templates(
    const std::string& source,
    size_t marker,
    std::string& header,
    std::string& body) {
  const auto template_position = source.rfind("template <", marker);
  if (template_position == std::string::npos) {
    header = source.substr(0, marker);
    return;
  }
  const auto template_end = source.find('>', template_position);
  if (template_end == std::string::npos || template_end > marker) {
    throw std::runtime_error("generated MSL has an invalid kernel template");
  }
  if (!trim(source.substr(template_end + 1, marker - template_end - 1)).empty()) {
    header = source.substr(0, marker);
    return;
  }
  header = source.substr(0, template_position);
  const std::string declarations =
      source.substr(template_position + 10, template_end - template_position - 10);
  const auto decltype_position = source.find("decltype(", marker);
  if (decltype_position == std::string::npos) {
    throw std::runtime_error("generated MSL kernel template has no instantiation");
  }
  const auto values_open = source.find('<', decltype_position);
  const auto values_close = source.find('>', values_open);
  if (values_open == std::string::npos || values_close == std::string::npos) {
    throw std::runtime_error("generated MSL kernel template has invalid values");
  }
  auto declaration_parts = split(declarations, ',');
  auto value_parts = split(
      source.substr(values_open + 1, values_close - values_open - 1), ',');
  if (declaration_parts.size() != value_parts.size()) {
    throw std::runtime_error("generated MSL kernel template arity mismatch");
  }
  for (size_t index = 0; index < declaration_parts.size(); ++index) {
    const auto name_position = declaration_parts[index].find_last_of(" \t");
    if (name_position == std::string::npos) {
      throw std::runtime_error("generated MSL kernel template parameter is invalid");
    }
    const auto name = trim(declaration_parts[index].substr(name_position + 1));
    auto value = value_parts[index];
    if (declaration_parts[index].find("bool") != std::string::npos) {
      value = value == "0" ? "false" : "true";
    } else if (declaration_parts[index].find("typename") != std::string::npos) {
      value = glsl_type(value);
    }
    replace_word(header, name, value);
    replace_word(body, name, value);
  }
}

void translate_header(std::string& header) {
  replace_all(header, "#include <metal_stdlib>", "");
  replace_all(header, "using namespace metal;", "");
  replace_all(header, "metal::precise::", "");
  replace_all(header, "metal::fast::", "");
  replace_all(header, "metal::", "");
  header = std::regex_replace(
      header,
      std::regex(
          R"(template\s*<\s*typename\s+([A-Za-z_][A-Za-z0-9_]*)\s*>\s*\1\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(\s*\1\s+([A-Za-z_][A-Za-z0-9_]*)\s*\))"),
      "float $2(float $3)");
  translate_types(header);
  if (header.find("template") != std::string::npos ||
      header.find("[[") != std::string::npos) {
    throw std::runtime_error("unsupported user header declaration");
  }
}

std::string buffer_declaration(const Parameter& parameter, bool output) {
  auto type = glsl_type(parameter.type);
  if (parameter.atomic && parameter.type == "float") {
    type = "uint";
  }
  const auto qualifier = output ? "" : "readonly ";
  return "layout(set=0, binding=" + std::to_string(parameter.binding) +
      ", std430) " + qualifier + "buffer _MlxBuffer" +
      std::to_string(parameter.binding) + " { " + type + " data[]; } _b" +
      std::to_string(parameter.binding) + ";\n";
}

void translate_bfloat_parameter(
    std::string& body,
    const Parameter& parameter,
    bool output) {
  const std::string escaped = regex_escape(parameter.name);
  const std::string storage = "_b" + std::to_string(parameter.binding) + ".data";
  if (output) {
    body = std::regex_replace(
        body,
        std::regex("\\b" + escaped + R"(\s*\[([^\]]+)\]\s*=\s*([^;]+);)"),
        storage + "[$1] = _mlx_float_to_bf16($2);");
    if (std::regex_search(body, std::regex("\\b" + escaped + R"(\s*\[)"))) {
      throw std::runtime_error(
          "bfloat16 output supports direct assignment only");
    }
  } else if (parameter.scalar) {
    replace_word(body, parameter.name, "_mlx_bf16_to_float(" + storage + "[0])");
  } else {
    body = std::regex_replace(
        body,
        std::regex("\\b" + escaped + R"(\s*\[([^\]]+)\])"),
        "_mlx_bf16_to_float(" + storage + "[$1])");
    if (std::regex_search(body, std::regex("\\b" + escaped + R"(\s*\[)"))) {
      throw std::runtime_error("unsupported bfloat16 buffer expression");
    }
  }
}

void translate_atomic_parameter(
    std::string& body,
    const Parameter& parameter) {
  const std::string operation = "atomic_fetch_add_explicit";
  size_t search_from = 0;
  while (true) {
    const auto position = body.find(operation, search_from);
    if (position == std::string::npos) {
      return;
    }
    const auto open = body.find('(', position + operation.size());
    const auto close = matching_delimiter(body, open, '(', ')');
    const auto arguments = split(body.substr(open + 1, close - open - 1), ',');
    if (arguments.size() != 3 || arguments[2] != "memory_order_relaxed") {
      search_from = close + 1;
      continue;
    }
    std::smatch target;
    const std::regex target_pattern(
        "^\\s*&\\s*" + regex_escape(parameter.name) +
        "\\s*\\[([^\\]]+)\\]\\s*$");
    if (!std::regex_match(arguments[0], target, target_pattern)) {
      search_from = close + 1;
      continue;
    }
    std::string replacement;
    if (parameter.type == "float") {
      const auto statement_end = body.find_first_not_of(" \t\r\n", close + 1);
      if (statement_end == std::string::npos || body[statement_end] != ';') {
        throw std::runtime_error(
            "float atomic add return values are unsupported");
      }
      const auto storage = "_b" + std::to_string(parameter.binding) +
          ".data[" + target[1].str() + "]";
      replacement = "{ uint _mlx_old = " + storage +
          "; for (;;) { uint _mlx_next = floatBitsToUint(uintBitsToFloat(_mlx_old) + " +
          arguments[1] + "); uint _mlx_observed = atomicCompSwap(" + storage +
          ", _mlx_old, _mlx_next); if (_mlx_observed == _mlx_old) break; _mlx_old = _mlx_observed; } }";
    } else if (
        parameter.type == "int" || parameter.type == "int32_t" ||
        parameter.type == "uint" || parameter.type == "uint32_t") {
      replacement = "atomicAdd(_b" + std::to_string(parameter.binding) +
          ".data[" + target[1].str() + "], " + arguments[1] + ")";
    } else {
      throw std::runtime_error(
          "unsupported atomic output type " + parameter.type);
    }
    body.replace(position, close - position + 1, replacement);
    search_from = position + replacement.size();
  }
}

Translation translate_msl(
    const std::string& source,
    const std::tuple<int, int, int>& grid,
    const std::tuple<int, int, int>& threadgroup,
    size_t output_count,
    int compile_mode) {
  const auto marker = source.find("[[kernel]] void ");
  if (marker == std::string::npos) {
    throw std::runtime_error("generated MSL kernel entry point is missing");
  }
  const auto arguments_open = source.find('(', marker);
  const auto arguments_close =
      matching_delimiter(source, arguments_open, '(', ')');
  const auto body_open = source.find('{', arguments_close);
  if (body_open == std::string::npos) {
    throw std::runtime_error("generated MSL kernel body is missing");
  }
  const auto body_close = matching_delimiter(source, body_open, '{', '}');
  const auto parameters = parse_parameters(
      source.substr(arguments_open + 1, arguments_close - arguments_open - 1));
  if (output_count == 0 || output_count > parameters.size()) {
    throw std::runtime_error("generated MSL output arity is invalid");
  }

  std::string header;
  std::string body = source.substr(body_open + 1, body_close - body_open - 1);
  resolve_kernel_templates(source, marker, header, body);
  translate_header(header);

  const std::vector<std::string> forbidden = {
      "texture", "sampler", "imageblock", "raytracing", "simdgroup_matrix",
      "quadgroup", "visible_function", "intersection_function", "object_data"};
  for (const auto& token : forbidden) {
    if (header.find(token) != std::string::npos ||
        body.find(token) != std::string::npos) {
      throw std::runtime_error("unsupported MSL feature `" + token + "`");
    }
  }

  const auto [grid_x, grid_y, grid_z] = grid;
  const auto [threads_x, threads_y, threads_z] = threadgroup;
  if (grid_x < 0 || grid_y < 0 || grid_z < 0 || threads_x <= 0 ||
      threads_y <= 0 || threads_z <= 0) {
    throw std::runtime_error("grid and threadgroup dimensions must be positive");
  }
  const auto local_x = std::max(1, std::min(grid_x, threads_x));
  const auto local_y = std::max(1, std::min(grid_y, threads_y));
  const auto local_z = std::max(1, std::min(grid_z, threads_z));
  const uint64_t local_total = static_cast<uint64_t>(local_x) * local_y * local_z;
  if (local_total > 1024) {
    throw std::runtime_error("threadgroup contains more than 1024 threads");
  }
  const auto groups_x = grid_x == 0 ? 0 : (grid_x + local_x - 1) / local_x;
  const auto groups_y = grid_y == 0 ? 0 : (grid_y + local_y - 1) / local_y;
  const auto groups_z = grid_z == 0 ? 0 : (grid_z + local_z - 1) / local_z;

  const std::vector<std::pair<std::string, std::string>> attributes = {
      {"dispatch_threads_per_threadgroup",
       "uvec3(" + std::to_string(local_x) + "," +
           std::to_string(local_y) + "," + std::to_string(local_z) + ")"},
      {"dispatch_simdgroups_per_threadgroup", "gl_NumSubgroups"},
      {"simdgroup_index_in_threadgroup", "gl_SubgroupID"},
      {"simdgroups_per_threadgroup", "gl_NumSubgroups"},
      {"thread_execution_width", "gl_SubgroupSize"},
      {"thread_index_in_simdgroup", "gl_SubgroupInvocationID"},
      {"thread_index_in_threadgroup", "gl_LocalInvocationIndex"},
      {"thread_position_in_grid", "gl_GlobalInvocationID"},
      {"thread_position_in_threadgroup", "gl_LocalInvocationID"},
      {"threadgroup_position_in_grid", "gl_WorkGroupID"},
      {"threadgroups_per_grid",
       "uvec3(" + std::to_string(groups_x) + "," +
           std::to_string(groups_y) + "," + std::to_string(groups_z) + ")"},
      {"threads_per_grid",
       "uvec3(" + std::to_string(grid_x) + "," +
           std::to_string(grid_y) + "," + std::to_string(grid_z) + ")"},
      {"threads_per_simdgroup", "gl_SubgroupSize"},
      {"threads_per_threadgroup",
       "uvec3(" + std::to_string(local_x) + "," +
           std::to_string(local_y) + "," + std::to_string(local_z) + ")"},
      {"grid_origin", "uvec3(0)"},
      {"grid_size",
       "uvec3(" + std::to_string(grid_x) + "," +
           std::to_string(grid_y) + "," + std::to_string(grid_z) + ")"},
  };
  for (const auto& [from, to] : attributes) {
    replace_word(body, from, to);
  }

  replace_all(body, "metal::precise::", "");
  replace_all(body, "metal::fast::", "");
  replace_all(body, "metal::", "");
  replace_all(body, "threadgroup_barrier(mem_flags::mem_threadgroup)", "barrier()" );
  replace_all(body, "simd_sum", "subgroupAdd");
  replace_all(body, "simd_max", "subgroupMax");
  body = std::regex_replace(
      body,
      std::regex(R"(float16_t\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([^;]+);)"),
      "float16_t $1 = float16_t($2);");
  replace_all(body, "simd_min", "subgroupMin");
  replace_all(body, "simd_broadcast", "subgroupBroadcast");
  replace_all(body, "simd_shuffle_down", "subgroupShuffleDown");
  replace_all(body, "simd_shuffle_up", "subgroupShuffleUp");
  replace_all(body, "simd_shuffle_xor", "subgroupShuffleXor");
  replace_all(body, "simd_shuffle", "subgroupShuffle");
  replace_word(body, "constexpr", "const");
  translate_types(body);

  std::string shared_declarations;
  const std::regex shared_pattern(
      R"(threadgroup\s+([A-Za-z_][A-Za-z0-9_]*)\s+([A-Za-z_][A-Za-z0-9_]*)\s*\[\s*([0-9]+)\s*\]\s*;)");
  std::smatch shared_match;
  while (std::regex_search(body, shared_match, shared_pattern)) {
    shared_declarations += "shared " + glsl_type(shared_match[1].str()) + " " +
        shared_match[2].str() + "[" + shared_match[3].str() + "];\n";
    body.replace(
        shared_match.position(), shared_match.length(), "");
  }

  const size_t output_start = parameters.size() - output_count;
  std::string element_helpers;
  size_t elem_search = 0;
  while (true) {
    const auto position = body.find("elem_to_loc", elem_search);
    if (position == std::string::npos) {
      break;
    }
    const auto open = body.find('(', position + 11);
    const auto close = matching_delimiter(body, open, '(', ')');
    const auto arguments = split(body.substr(open + 1, close - open - 1), ',');
    if (arguments.size() != 4) {
      throw std::runtime_error("unsupported elem_to_loc expression");
    }
    constexpr std::string_view shape_suffix = "_shape";
    if (arguments[1].size() <= shape_suffix.size() ||
        arguments[1].compare(
            arguments[1].size() - shape_suffix.size(),
            shape_suffix.size(),
            shape_suffix) != 0) {
      throw std::runtime_error("unsupported elem_to_loc shape argument");
    }
    const auto base = arguments[1].substr(
        0, arguments[1].size() - shape_suffix.size());
    if (arguments[2] != base + "_strides" || arguments[3] != base + "_ndim") {
      throw std::runtime_error("unsupported elem_to_loc metadata arguments");
    }
    const auto shape = std::find_if(
        parameters.begin(), parameters.end(), [&](const Parameter& parameter) {
          return parameter.name == arguments[1];
        });
    const auto strides = std::find_if(
        parameters.begin(), parameters.end(), [&](const Parameter& parameter) {
          return parameter.name == arguments[2];
        });
    const auto ndim = std::find_if(
        parameters.begin(), parameters.end(), [&](const Parameter& parameter) {
          return parameter.name == arguments[3];
        });
    if (shape == parameters.end() || strides == parameters.end() ||
        ndim == parameters.end()) {
      throw std::runtime_error("elem_to_loc metadata bindings are missing");
    }
    const auto helper = "_mlx_elem_to_loc_" + base;
    body.replace(position, close - position + 1, helper + "(" + arguments[0] + ")");
    if (element_helpers.find("uint " + helper + "(") == std::string::npos) {
      element_helpers += "uint " + helper + "(uint elem) { uint loc = 0u; for (int axis = int(_b" +
          std::to_string(ndim->binding) + ".data[0]) - 1; axis >= 0; --axis) { uint extent = uint(_b" +
          std::to_string(shape->binding) + ".data[axis]); uint coord = elem % extent; elem /= extent; loc += coord * uint(_b" +
          std::to_string(strides->binding) + ".data[axis]); } return loc; }\n";
    }
    elem_search = position + helper.size();
  }

  bool needs_bfloat = false;
  bool needs_int8 = false;
  bool needs_int16 = false;
  bool needs_int64 = false;
  bool needs_subgroup = body.find("subgroup") != std::string::npos;
  std::string declarations;
  std::string macros;
  for (size_t index = 0; index < parameters.size(); ++index) {
    const auto& parameter = parameters[index];
    const bool output = index >= output_start;
    declarations += buffer_declaration(parameter, output);
    if (parameter.type == "bfloat16_t") {
      needs_bfloat = true;
      translate_bfloat_parameter(body, parameter, output);
    } else if (parameter.atomic) {
      translate_atomic_parameter(body, parameter);
    } else {
      if (output) {
        const auto escaped = regex_escape(parameter.name);
        body = std::regex_replace(
            body,
            std::regex("\\b" + escaped + R"(\s*\[([^\]]+)\]\s*=\s*([^;]+);)"),
            parameter.name + "[$1] = " + glsl_type(parameter.type) + "($2);");
      }
      macros += "#define " + parameter.name + " _b" +
          std::to_string(parameter.binding) + ".data";
      if (parameter.scalar) {
        macros += "[0]";
      }
      macros += "\n";
    }
    needs_int8 = needs_int8 || parameter.type == "int8_t" ||
        parameter.type == "uint8_t";
    needs_int16 = needs_int16 || parameter.type == "int16_t" ||
        parameter.type == "uint16_t" || parameter.type == "bfloat16_t";
    needs_int64 = needs_int64 || parameter.type == "int64_t" ||
        parameter.type == "uint64_t";
  }

  if (body.find("threadgroup") != std::string::npos ||
      body.find("memory_order") != std::string::npos ||
      body.find("atomic_fetch") != std::string::npos ||
      body.find("[[") != std::string::npos) {
    throw std::runtime_error("unsupported MSL syntax remains after translation");
  }

  std::ostringstream glsl;
  glsl << "#version 460\n";
  glsl << "#extension GL_EXT_scalar_block_layout : require\n";
  if (needs_int8) {
    glsl << "#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require\n"
         << "#extension GL_EXT_shader_8bit_storage : require\n";
  }
  if (needs_int16) {
    glsl << "#extension GL_EXT_shader_explicit_arithmetic_types_int16 : require\n"
         << "#extension GL_EXT_shader_16bit_storage : require\n";
  }
  if (needs_int64) {
    glsl << "#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require\n";
  }
  if (source.find("float16_t") != std::string::npos ||
      source.find("half") != std::string::npos) {
    glsl << "#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require\n";
  }
  if (needs_subgroup) {
    glsl << "#extension GL_KHR_shader_subgroup_basic : require\n"
         << "#extension GL_KHR_shader_subgroup_arithmetic : require\n"
         << "#extension GL_KHR_shader_subgroup_shuffle : require\n";
  }
  glsl << "#define __FAST_MATH__ " << (compile_mode == 2 ? 1 : 0) << "\n";
  glsl << "layout(local_size_x=" << local_x << ", local_size_y=" << local_y
       << ", local_size_z=" << local_z << ") in;\n";
  glsl << declarations << shared_declarations;
  if (needs_bfloat) {
    glsl << "float _mlx_bf16_to_float(uint16_t value) { return uintBitsToFloat(uint(value) << 16); }\n"
         << "uint16_t _mlx_float_to_bf16(float value) { uint bits = floatBitsToUint(value); uint rounded = bits + 0x7fffu + ((bits >> 16) & 1u); return uint16_t(rounded >> 16); }\n";
  }
  glsl << header << "\n" << element_helpers << macros;
  glsl << "void main() {\n"
       << "if (gl_GlobalInvocationID.x >= " << grid_x
       << "u || gl_GlobalInvocationID.y >= " << grid_y
       << "u || gl_GlobalInvocationID.z >= " << grid_z << "u) return;\n"
       << body << "\n}\n";
  return {glsl.str(), parameters};
}

std::string find_executable(const std::string& name) {
  const char* path_value = std::getenv("PATH");
  if (path_value == nullptr) {
    return {};
  }
  for (const auto& directory : split(path_value, ':')) {
    const std::string candidate = (directory.empty() ? "." : directory) + "/" + name;
    if (access(candidate.c_str(), X_OK) == 0) {
      return candidate;
    }
  }
  return {};
}

std::string read_file(const std::string& path, size_t limit = SIZE_MAX) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    return {};
  }
  std::string value;
  char buffer[4096];
  while (input && value.size() < limit) {
    input.read(buffer, std::min(sizeof(buffer), limit - value.size()));
    value.append(buffer, static_cast<size_t>(input.gcount()));
  }
  return value;
}

int run_compiler(
    const std::string& executable,
    const std::vector<std::string>& arguments,
    int log_fd) {
  const pid_t child = fork();
  if (child < 0) {
    throw std::runtime_error("cannot start the runtime shader compiler");
  }
  if (child == 0) {
    dup2(log_fd, STDOUT_FILENO);
    dup2(log_fd, STDERR_FILENO);
    std::vector<char*> argv;
    argv.reserve(arguments.size() + 2);
    argv.push_back(const_cast<char*>(executable.c_str()));
    for (const auto& argument : arguments) {
      argv.push_back(const_cast<char*>(argument.c_str()));
    }
    argv.push_back(nullptr);
    execv(executable.c_str(), argv.data());
    _exit(127);
  }

  int status = 0;
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
  while (waitpid(child, &status, WNOHANG) == 0) {
    if (std::chrono::steady_clock::now() >= deadline) {
      kill(child, SIGKILL);
      waitpid(child, &status, 0);
      throw std::runtime_error("runtime shader compilation exceeded 30 seconds");
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  if (!WIFEXITED(status)) {
    return 128;
  }
  return WEXITSTATUS(status);
}

// The shader compiler is resolved once.  A cache entry has to be named by the
// exact binary and flags that produced it, never by the source alone.
struct ShaderCompiler {
  std::string executable;
  std::vector<std::string> flags;
  std::string version;
};

std::string compiler_version(const std::string& executable) {
  TemporaryFile log(".log");
  if (run_compiler(executable, {"--version"}, log.fd) != 0) {
    return {};
  }
  return trim(read_file(log.path, 4096));
}

const ShaderCompiler& shader_compiler() {
  static const ShaderCompiler resolved = [] {
    ShaderCompiler value;
    value.executable = find_executable("glslc");
    if (!value.executable.empty()) {
      value.flags = {"-O", "--target-env=vulkan1.3"};
    } else {
      value.executable = find_executable("glslangValidator");
      if (value.executable.empty()) {
        return value;
      }
      value.flags = {"-V", "-Os", "--target-env", "vulkan1.3", "-S", "comp"};
    }
    value.version = compiler_version(value.executable);
    return value;
  }();
  return resolved;
}

std::vector<uint32_t> compile_glsl(const std::string& source) {
  const ShaderCompiler& compiler = shader_compiler();
  if (compiler.executable.empty()) {
    throw std::runtime_error(
        "neither glslc nor glslangValidator is available on PATH");
  }
  TemporaryFile input(".comp");
  TemporaryFile output(".spv");
  TemporaryFile log(".log");
  if (ftruncate(input.fd, 0) != 0 ||
      write(input.fd, source.data(), source.size()) !=
          static_cast<ssize_t>(source.size())) {
    throw std::runtime_error("cannot write runtime shader source");
  }
  close(input.fd);
  input.fd = -1;

  std::vector<std::string> arguments = compiler.flags;
  arguments.push_back(input.path);
  arguments.emplace_back("-o");
  arguments.push_back(output.path);
  const int status = run_compiler(compiler.executable, arguments, log.fd);
  if (status != 0) {
    const auto diagnostics = trim(read_file(log.path, 8192));
    throw std::runtime_error(
        "runtime shader compilation failed" +
        (diagnostics.empty() ? std::string{} : ": " + diagnostics));
  }
  close(output.fd);
  output.fd = -1;
  const auto bytes = read_file(output.path);
  if (bytes.size() < sizeof(uint32_t) || bytes.size() % sizeof(uint32_t) != 0) {
    throw std::runtime_error("runtime shader compiler produced invalid SPIR-V");
  }
  std::vector<uint32_t> words(bytes.size() / sizeof(uint32_t));
  std::memcpy(words.data(), bytes.data(), bytes.size());
  if (words.front() != 0x07230203u) {
    throw std::runtime_error("runtime shader compiler produced invalid SPIR-V magic");
  }
  return words;
}

// `glslc -O` runs the SPIR-V optimizer, and on a kernel assembled out of
// software binary32 helpers that costs seconds: the Parakeet mel frontend's
// eight custom kernels spend 9.8 s there against 0.17 s of actual compute,
// once per process, because the memo below dies with the process.  The
// optimizer's output is a pure function of the source and the invocation, so
// it belongs on disk.
constexpr char kSpirvCacheMagic[] = "MLXOSPV1";
constexpr size_t kSpirvCacheMagicSize = 8;

std::string spirv_cache_root() {
  if (const char* configured = std::getenv("MLX_OMARCHY_SPIRV_CACHE")) {
    // Empty or "0" turns the disk layer off; any other value relocates it.
    const std::string value = configured;
    return (value.empty() || value == "0") ? std::string{} : value;
  }
  if (const char* xdg = std::getenv("XDG_CACHE_HOME"); xdg != nullptr && *xdg) {
    return std::string(xdg) + "/mlx-omarchy/spirv";
  }
  if (const char* home = std::getenv("HOME"); home != nullptr && *home) {
    return std::string(home) + "/.cache/mlx-omarchy/spirv";
  }
  return {};
}

bool make_directories(const std::string& path) {
  for (size_t index = 1; index <= path.size(); ++index) {
    if (index != path.size() && path[index] != '/') {
      continue;
    }
    const std::string prefix = path.substr(0, index);
    if (::mkdir(prefix.c_str(), 0700) != 0 && errno != EEXIST) {
      return false;
    }
  }
  return true;
}

std::string spirv_cache_path(const std::string& glsl) {
  const std::string root = spirv_cache_root();
  if (root.empty()) {
    return {};
  }
  const ShaderCompiler& compiler = shader_compiler();
  if (compiler.version.empty()) {
    // Without a compiler identity an entry cannot be invalidated on upgrade,
    // and a stale entry is a wrong binary rather than a slow one.
    return {};
  }
  std::string material = "mlx-omarchy custom kernel spirv 1\n";
  material += compiler.executable + "\n" + compiler.version + "\n";
  for (const auto& flag : compiler.flags) {
    material += flag;
    material += ' ';
  }
  material += "\n";
  material += glsl;
  return root + "/" +
      omarchy::ane::sha256_hex(
             reinterpret_cast<const uint8_t*>(material.data()),
             material.size()) +
      ".spv";
}

bool load_cached_spirv(const std::string& path, std::vector<uint32_t>& words) {
  const std::string blob = read_file(path);
  if (blob.size() <= kSpirvCacheMagicSize ||
      blob.compare(0, kSpirvCacheMagicSize, kSpirvCacheMagic, kSpirvCacheMagicSize) != 0) {
    return false;
  }
  const size_t payload = blob.size() - kSpirvCacheMagicSize;
  if (payload % sizeof(uint32_t) != 0) {
    return false;
  }
  words.resize(payload / sizeof(uint32_t));
  std::memcpy(words.data(), blob.data() + kSpirvCacheMagicSize, payload);
  if (words.front() != 0x07230203u) {
    words.clear();
    return false;
  }
  return true;
}

void store_cached_spirv(
    const std::string& path,
    const std::vector<uint32_t>& words) {
  const size_t slash = path.rfind('/');
  if (slash == std::string::npos || !make_directories(path.substr(0, slash))) {
    return;
  }
  const std::string temporary = path + ".tmp-" + std::to_string(::getpid());
  {
    std::ofstream out(temporary, std::ios::binary | std::ios::trunc);
    if (!out) {
      return;
    }
    out.write(kSpirvCacheMagic, kSpirvCacheMagicSize);
    out.write(
        reinterpret_cast<const char*>(words.data()),
        static_cast<std::streamsize>(words.size() * sizeof(uint32_t)));
    out.flush();
    if (!out) {
      ::unlink(temporary.c_str());
      return;
    }
  }
  if (::rename(temporary.c_str(), path.c_str()) != 0) {
    ::unlink(temporary.c_str());
  }
}

const std::vector<uint32_t>& cached_compile(const std::string& glsl) {
  static std::mutex mutex;
  static std::unordered_map<std::string, std::vector<uint32_t>> cache;
  std::lock_guard lock(mutex);
  auto found = cache.find(glsl);
  if (found != cache.end()) {
    return found->second;
  }
  const std::string entry = spirv_cache_path(glsl);
  if (entry.empty()) {
    return cache.emplace(glsl, compile_glsl(glsl)).first->second;
  }
  std::vector<uint32_t> words;
  if (!load_cached_spirv(entry, words)) {
    words = compile_glsl(glsl);
    store_cached_spirv(entry, words);
  }
  return cache.emplace(glsl, std::move(words)).first->second;
}

// Translation is pure in the kernel's source and launch geometry, and it is
// regex-heavy: a decode-time kernel dispatched per token paid milliseconds per
// call for it, several times the cost of the compute it was dispatching. The
// SPIR-V below is already cached, so cache the step that produces it too.
// The in-process memo alone still makes every fresh process re-run the full
// translate_msl pass for each dynamic kernel before its first dispatch (the
// Parakeet mel frontend pays ~100 ms there across its seven kernels), so the
// memo is backed by the same disk layer the SPIR-V cache uses. The material
// version must be bumped whenever any translate_* pass changes its output.
constexpr char kTranslationCacheMagic[] = "MLXOTR1";
constexpr size_t kTranslationCacheMagicSize = 8;
constexpr char kTranslationCacheVersion[] = "1";

std::string translation_cache_path(const std::string& identity) {
  const std::string root = spirv_cache_root();
  if (root.empty()) {
    return {};
  }
  std::string material = "mlx-omarchy custom kernel translation ";
  material += kTranslationCacheVersion;
  material += "\n";
  material += identity;
  return root + "/" +
      omarchy::ane::sha256_hex(
             reinterpret_cast<const uint8_t*>(material.data()),
             material.size()) +
      ".tr";
}

void put_u64(std::string& out, uint64_t value) {
  for (int i = 0; i < 8; ++i) {
    out.push_back(static_cast<char>((value >> (8 * i)) & 0xff));
  }
}

bool get_u64(
    const std::string& blob,
    size_t& cursor,
    size_t& remaining,
    uint64_t& value) {
  if (cursor + sizeof(uint64_t) > blob.size() || remaining < sizeof(uint64_t)) {
    return false;
  }
  value = 0;
  for (int i = 0; i < 8; ++i) {
    value |= static_cast<uint64_t>(
                 static_cast<unsigned char>(blob[cursor + i])) <<
        (8 * i);
  }
  cursor += sizeof(uint64_t);
  remaining -= sizeof(uint64_t);
  return true;
}

void put_string(std::string& out, const std::string& value) {
  put_u64(out, value.size());
  out += value;
}

bool get_string(
    const std::string& blob,
    size_t& cursor,
    size_t& remaining,
    std::string& value) {
  uint64_t length = 0;
  if (!get_u64(blob, cursor, remaining, length) || length > remaining) {
    return false;
  }
  if (cursor + length > blob.size()) {
    return false;
  }
  value.assign(blob, cursor, static_cast<size_t>(length));
  cursor += static_cast<size_t>(length);
  remaining -= static_cast<size_t>(length);
  return true;
}

std::string serialize_translation(const Translation& translation) {
  std::string payload;
  put_string(payload, translation.glsl);
  put_u64(payload, translation.parameters.size());
  for (const auto& parameter : translation.parameters) {
    put_string(payload, parameter.type);
    put_string(payload, parameter.name);
    put_u64(payload, parameter.binding);
    payload.push_back(parameter.scalar ? 1 : 0);
    payload.push_back(parameter.atomic ? 1 : 0);
  }
  std::string blob(kTranslationCacheMagic, kTranslationCacheMagicSize);
  blob += payload;
  return blob;
}

bool parse_translation(const std::string& blob, Translation& translation) {
  if (blob.size() <= kTranslationCacheMagicSize ||
      blob.compare(0, kTranslationCacheMagicSize, kTranslationCacheMagic, kTranslationCacheMagicSize) != 0) {
    return false;
  }
  size_t cursor = kTranslationCacheMagicSize;
  size_t remaining = blob.size() - kTranslationCacheMagicSize;
  uint64_t count = 0;
  if (!get_string(blob, cursor, remaining, translation.glsl) ||
      !get_u64(blob, cursor, remaining, count) || count > 4096) {
    return false;
  }
  translation.parameters.resize(static_cast<size_t>(count));
  for (auto& parameter : translation.parameters) {
    uint64_t binding = 0;
    if (!get_string(blob, cursor, remaining, parameter.type) ||
        !get_string(blob, cursor, remaining, parameter.name) ||
        !get_u64(blob, cursor, remaining, binding) || remaining < 2) {
      return false;
    }
    parameter.binding = static_cast<uint32_t>(binding);
    parameter.scalar = blob[cursor] != 0;
    parameter.atomic = blob[cursor + 1] != 0;
    cursor += 2;
    remaining -= 2;
  }
  if (remaining != 0) {
    return false;
  }
  return true;
}

bool load_cached_translation(
    const std::string& path,
    Translation& translation) {
  const std::string blob = read_file(path);
  return parse_translation(blob, translation);
}

void store_cached_translation(
    const std::string& path,
    const Translation& translation) {
  const size_t slash = path.rfind('/');
  if (slash == std::string::npos || !make_directories(path.substr(0, slash))) {
    return;
  }
  const std::string temporary = path + ".tmp-" + std::to_string(::getpid());
  {
    std::ofstream out(temporary, std::ios::binary | std::ios::trunc);
    if (!out) {
      return;
    }
    const std::string blob = serialize_translation(translation);
    out.write(blob.data(), static_cast<std::streamsize>(blob.size()));
    out.flush();
    if (!out) {
      ::unlink(temporary.c_str());
      return;
    }
  }
  if (::rename(temporary.c_str(), path.c_str()) != 0) {
    ::unlink(temporary.c_str());
  }
}

const Translation& cached_translation(
    const std::string& source,
    const std::tuple<int, int, int>& grid,
    const std::tuple<int, int, int>& threadgroup,
    size_t output_count,
    int compile_mode) {
  static std::mutex mutex;
  static std::unordered_map<std::string, Translation> cache;
  const auto [grid_x, grid_y, grid_z] = grid;
  const auto [threads_x, threads_y, threads_z] = threadgroup;
  std::ostringstream key;
  key << grid_x << ',' << grid_y << ',' << grid_z << ';' << threads_x << ','
      << threads_y << ',' << threads_z << ';' << output_count << ';'
      << compile_mode << '\n'
      << source;
  std::lock_guard lock(mutex);
  auto identity = key.str();
  auto found = cache.find(identity);
  if (found != cache.end()) {
    return found->second;
  }
  const std::string entry = translation_cache_path(identity);
  if (!entry.empty()) {
    Translation stored;
    if (load_cached_translation(entry, stored)) {
      return cache.emplace(std::move(identity), std::move(stored))
          .first->second;
    }
    auto& fresh = cache
        .emplace(
            std::move(identity),
            translate_msl(
                source, grid, threadgroup, output_count, compile_mode))
        .first->second;
    store_cached_translation(entry, fresh);
    return fresh;
  }
  return cache
      .emplace(
          std::move(identity),
          translate_msl(source, grid, threadgroup, output_count, compile_mode))
      .first->second;
}

omarchy::ComputeBinding binding(const array& value) {
  auto* buffer = static_cast<const omarchy::VulkanBuffer*>(value.buffer().ptr());
  return {buffer->buffer, 0, buffer->size, buffer};
}

template <typename T>
array metadata_array(
    const std::vector<T>& values,
    Dtype dtype,
    omarchy::CommandEncoder& encoder) {
  array metadata(Shape{static_cast<int>(values.size())}, dtype, nullptr, {});
  array::Flags flags;
  flags.contiguous = true;
  flags.row_contiguous = true;
  flags.col_contiguous = true;
  metadata.set_data(
      omarchy::allocator().malloc(metadata.nbytes()),
      metadata.size(),
      Strides{1},
      flags,
      0);
  auto* buffer = static_cast<omarchy::VulkanBuffer*>(metadata.buffer().ptr());
  std::memcpy(buffer->data, values.data(), metadata.nbytes());
  encoder.add_temporary(metadata);
  return metadata;
}

} // namespace

void CustomKernel::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& out_for_error = outputs.at(0);
  if (is_precompiled_) {
    omarchy::unsupported(
        "fast::CustomKernel MSL subset: precompiled Metal libraries", out_for_error);
  }
  if (!scalar_arguments_.empty()) {
    omarchy::unsupported(
        "fast::CustomKernel MSL subset: serialized scalar arguments",
        out_for_error);
  }
  if (shared_memory_ != 0) {
    omarchy::unsupported(
        "fast::CustomKernel MSL subset: dynamic threadgroup memory",
        out_for_error);
  }

  const Translation* translation = nullptr;
  try {
    translation = &cached_translation(
        source_, grid_, threadgroup_, outputs.size(), compile_options_);
  } catch (const std::exception& error) {
    omarchy::unsupported(
        std::string("fast::CustomKernel MSL subset: ") + error.what(),
        out_for_error);
  }

  auto& s = stream();
  auto& encoder = omarchy::get_command_encoder(s);
  std::vector<array> temporaries;
  temporaries.reserve(inputs.size() * 3 + outputs.size());
  for (auto& out : outputs) {
    if (init_value_) {
      temporaries.emplace_back(init_value_.value(), out.dtype());
      fill_gpu(temporaries.back(), out, s);
    } else {
      out.set_data(omarchy::allocator().malloc(out.nbytes()));
    }
  }

  std::vector<array> checked_inputs;
  checked_inputs.reserve(inputs.size());
  for (const auto& input : inputs) {
    if (ensure_row_contiguous_ &&
        (!input.flags().row_contiguous || input.offset() != 0)) {
      checked_inputs.push_back(contiguous_copy_gpu(input, s));
      encoder.add_temporary(checked_inputs.back());
    } else {
      if (!ensure_row_contiguous_ && input.offset() != 0) {
        omarchy::unsupported(
            "fast::CustomKernel MSL subset: nonzero input buffer offsets with ensure_row_contiguous=False",
            out_for_error);
      }
      checked_inputs.push_back(input);
    }
  }

  std::vector<omarchy::ComputeBinding> bindings;
  bindings.reserve(translation->parameters.size());
  for (size_t index = 0; index < checked_inputs.size(); ++index) {
    const auto& input = checked_inputs[index];
    bindings.push_back(binding(input));
    if (input.ndim() == 0) {
      continue;
    }
    const auto& [needs_shape, needs_strides, needs_ndim] = shape_infos_.at(index);
    if (needs_shape) {
      std::vector<int32_t> shape(input.shape().begin(), input.shape().end());
      auto metadata = metadata_array(shape, int32, encoder);
      bindings.push_back(binding(metadata));
      temporaries.push_back(std::move(metadata));
    }
    if (needs_strides) {
      std::vector<int64_t> strides(input.strides().begin(), input.strides().end());
      auto metadata = metadata_array(strides, int64, encoder);
      bindings.push_back(binding(metadata));
      temporaries.push_back(std::move(metadata));
    }
    if (needs_ndim) {
      auto metadata = metadata_array(
          std::vector<int32_t>{input.ndim()}, int32, encoder);
      bindings.push_back(binding(metadata));
      temporaries.push_back(std::move(metadata));
    }
  }
  for (const auto& output : outputs) {
    bindings.push_back(binding(output));
  }
  if (bindings.size() != translation->parameters.size()) {
    omarchy::unsupported(
        "fast::CustomKernel MSL subset: generated binding count mismatch",
        out_for_error);
  }

  const auto [grid_x, grid_y, grid_z] = grid_;
  if (grid_x == 0 || grid_y == 0 || grid_z == 0) {
    return;
  }
  const auto [threads_x, threads_y, threads_z] = threadgroup_;
  const auto local_x = std::min(grid_x, threads_x);
  const auto local_y = std::min(grid_y, threads_y);
  const auto local_z = std::min(grid_z, threads_z);
  const auto groups_x = static_cast<uint32_t>((grid_x + local_x - 1) / local_x);
  const auto groups_y = static_cast<uint32_t>((grid_y + local_y - 1) / local_y);
  const auto groups_z = static_cast<uint32_t>((grid_z + local_z - 1) / local_z);

  const std::vector<uint32_t>* spirv = nullptr;
  try {
    spirv = &cached_compile(translation->glsl);
  } catch (const std::exception& error) {
    omarchy::unsupported(
        std::string("fast::CustomKernel MSL subset: ") + error.what(),
        out_for_error);
  }
  omarchy::ComputeParams params;
  encoder.dispatch_compute(
      translation->glsl,
      *spirv,
      bindings,
      params,
      groups_x,
      groups_y,
      groups_z);
}

} // namespace mlx::core::fast
