// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

#include "mlx/fast_primitives.h"

#include <fcntl.h>
#include <dlfcn.h>
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
#include <map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "mlx/backend/gpu/copy.h"
#include "mlx/backend/omarchy/allocator.h"
#include "mlx/backend/omarchy/ane/bundle.h"
#include "mlx/backend/omarchy/compute.h"
#include "mlx/backend/omarchy/encoder.h"
#include "mlx/backend/omarchy/unsupported.h"

#include "translation_version.h"

namespace mlx::core::fast {
namespace {

struct Parameter {
  std::string type;
  std::string name;
  uint32_t binding;
  bool scalar;
  bool atomic;
  bool constant_space;
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
    // Diagnostic (MLX_OMARCHY_KEEP_SHADER=1): keep the intermediate GLSL
    // instead of unlinking it — shader-debugging sessions diff the emitted
    // text; default stays delete-on-exit.
    static const bool keep_shader =
        std::getenv("MLX_OMARCHY_KEEP_SHADER") != nullptr;
    if (!path.empty() && !keep_shader) {
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

void translate_as_type(std::string& code, const std::vector<Parameter>& parameters);
void translate_c_style_casts(std::string& code);
void map_numeric_limits(std::string& code);
void translate_pointer_advance(std::string& body, const std::vector<Parameter>& parameters);

void translate_types(std::string& code, const std::vector<Parameter>& parameters = {}) {
  // Single-token C typedefs (`typedef float U;` in the mlx_vlm qwen3_5
  // ragged-SDPA kernels) have no GLSL meaning; drop the declaration and
  // expand the alias to its type everywhere.
  {
    const std::regex typedef_decl(
        R"(typedef\s+([A-Za-z_][A-Za-z0-9_]*)\s+([A-Za-z_][A-Za-z0-9_]*)\s*;)");
    std::string rewritten;
    rewritten.reserve(code.size());
    size_t last = 0;
    std::vector<std::pair<std::string, std::string>> typedefs;
    for (std::sregex_iterator it(code.begin(), code.end(), typedef_decl), end;
         it != end;
         ++it) {
      const auto& match = *it;
      rewritten += code.substr(last, match.position() - last);
      typedefs.emplace_back(match[2].str(), match[1].str());
      last = match.position() + match.length();
    }
    rewritten += code.substr(last);
    if (!typedefs.empty()) {
      code = std::move(rewritten);
      for (const auto& [name, type] : typedefs) {
        replace_word(code, name, type);
      }
    }
  }
  // as_type<ushort>(bf16-derived value) is a 16-bit pattern bitcast: the
  // operand's bfloat16 storage pattern, which the bf16 input rewrite has
  // left as _mlx_bf16_to_float(...). Map it to _mlx_float_to_bf16 BEFORE
  // the type table below narrows ushort to uint (which would emit
  // floatBitsToUint on the widened float — a 32-bit index into a 65536-row
  // per-dtype table; mlx-serve's fused SwiGLU sigmoid lookup, 2026-10-07).
  // A 16-bit as_type on anything else is left alone and surfaces as the
  // existing named refusal (as_type between mismatched sizes is invalid
  // MSL anyway).
  {
    size_t bitcast_search = 0;
    while (true) {
      const auto marker = code.find("as_type<", bitcast_search);
      if (marker == std::string::npos) {
        break;
      }
      const auto open_angle = marker + 8;
      const auto close_angle = code.find('>', open_angle);
      if (close_angle == std::string::npos) {
        break;
      }
      const auto destination =
          trim(code.substr(open_angle, close_angle - open_angle));
      const bool narrow16 = destination == "ushort" ||
          destination == "short" || destination == "uint16_t" ||
          destination == "int16_t";
      const auto open_paren = code.find('(', close_angle + 1);
      if (open_paren == std::string::npos) {
        break;
      }
      const auto close_paren = matching_delimiter(code, open_paren, '(', ')');
      const auto operand =
          code.substr(open_paren + 1, close_paren - open_paren - 1);
      if (narrow16) {
        // A 16-bit as_type is a storage-pattern bitcast of a 16-bit value.
        // The bf16 input rewrite has already widened every bf16 load to
        // f32, so the operand's f32 value round-trips exactly through
        // _mlx_float_to_bf16 — the pattern the (unwidened) source meant.
        // This holds whether the operand is the load itself or a local
        // holding it (mlx-serve's SwiGLU: `float g = gate[i];
        // sigtab[as_type<ushort>(g)]`). as_type between mismatched sizes
        // is invalid MSL, so no genuine 32-bit bitcast lands here.
        code.replace(
            marker, close_paren - marker + 1,
            "_mlx_float_to_bf16(" + operand + ")");
        bitcast_search = marker;
        continue;
      }
      bitcast_search = marker + 8;
    }
  }
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
      // Metal promotes bfloat16_t LOCALS to fp32 for arithmetic and
      // rounds back at bf16 buffer writes; mirror that exactly: the
      // body token becomes float, while bf16 buffer parameters keep
      // uint16_t storage (buffer_declaration) and every load/store
      // converts via _mlx_bf16_to_float / the _mlx_float_to_bf16 store
      // casts. Integer uint16_t locals do integer math instead — the
      // root cause of the Qwen3.5-2B GDN conv+sigmoid failure
      // (OmlxLinux M2 repro: uint16_t(1) / (uint16_t(1) + exp(...))).
      {"bfloat16_t", "float"},
      // GLSL has no 16-bit integer scalars; MSL short/ushort hold small
      // integers in these kernels (tile indices, e.g. TensorFold _LINEAR's
      // `short erow[CAP]`), so the int-range mapping is value-exact.
      {"short", "int"},
      {"ushort", "uint"},
      // size_t locals (oMLX GDN chunk kernels: 'const size_t row =
      // (size_t)Hk * Dk') hold buffer offsets; the uint mapping is
      // value-exact for every offset below 4 GiB, and the old
      // blanket \bsize_t\b refusal blocked whole kernels that only
      // used size_t for scalar index arithmetic.
      {"size_t", "uint"},
      {"half", "float16_t"},
  };
  for (const auto& [from, to] : replacements) {
    replace_word(code, from, to);
  }
  // Metal's generic vec<T,N> (template kernels parameterized on the
  // element type, e.g. 'const device vec<T,4>*' instantiated with
  // bfloat16_t) maps to the GLSL vector of the promoted element — the
  // bfloat16_t->float promotion above has already run, so the float
  // forms cover the bf16 instantiation.
  for (const auto& [from, to] : std::vector<std::pair<std::string,
                                                     std::string>>{
           {"vec<float,4>", "vec4"},
           {"vec<float,3>", "vec3"},
           {"vec<float,2>", "vec2"},
           {"vec<int,4>", "ivec4"},
           {"vec<int,3>", "ivec3"},
           {"vec<int,2>", "ivec2"},
           {"vec<uint,4>", "uvec4"},
           {"vec<uint,3>", "uvec3"},
           {"vec<uint,2>", "uvec2"},
       }) {
    replace_all(code, from, to);
  }
  code = std::regex_replace(
      code,
      std::regex(R"(static_cast\s*<\s*([A-Za-z_][A-Za-z0-9_]*)\s*>\s*\(([^()]*)\))"),
      "$1($2)");
  translate_as_type(code, parameters);
}

// Translate C-style scalar casts `(type)expr` into functional `type(expr)`
// with a balanced-paren argument. Necessary for H3 MPP int8 sources and any
// kernel that uses `(int64_t)idx` to narrow a 64-bit index for an SSBO
// subscript (glslang rejects 64-bit SSBO subscripts). Catches balanced-arg
// forms like `(int8_t)clamp(int(rint(v[j])), -127, 127)` by extending the
// argument through any nested call chain.
//
// Defensive notes (KernelBattery 2026-10-04 asan chase):
//  * The previous scanner reported 'free(): invalid next size' on the H3
//    _QUANTIZE MSL. The corruption was not reproducible in a standalone
//    asan driver of the cast scanner alone (6/20 runs on this dev box hit
//    ASAN's DEADLYSIGNAL report, 5/40 in a trivial file-read program —
//    a libasan + libstdc++ teardown race on this kernel/glibc, not the
//    scanner itself). The re-landed scanner below:
//      - asserts every indexing position against `code.size()`;
//      - recomputes `search_from` against the post-replace `code.size()`
//        instead of by string-length arithmetic;
//      - throws on unbalanced delimiters rather than returning npos into
//        the call chain.
//  * The intended replace span is `(open, end_of_argument)`, not
//    `consumed + (next - open)`; the old arithmetic was correct in normal
//    cases but unsafe when the call-chain scan early-exited at a `)` of an
//    enclosing form. The new path stores the explicit end and uses it.
int translate_c_style_casts_once(std::string& code) {
  static const std::unordered_map<std::string, std::string> casts = {
      {"int8_t", "int8_t"}, {"uint8_t", "uint8_t"},
      {"int", "int"}, {"uint", "uint"},
      {"float", "float"}, {"bool", "bool"},
      {"int32_t", "int"}, {"uint32_t", "uint"},
      {"int64_t", "uint"}, {"uint64_t", "uint"},
      {"size_t", "uint"},
  };
  int replacements = 0;
  size_t search_from = 0;
  while (search_from < code.size()) {
    const auto open = code.find('(', search_from);
    if (open == std::string::npos) {
      return 0;
    }
    const auto close = code.find(')', open + 1);
    if (close == std::string::npos) {
      return 0;
    }
    const auto candidate = trim(code.substr(open + 1, close - open - 1));
    const auto mapped = casts.find(candidate);
    if (mapped == casts.end() || close + 1 >= code.size()) {
      search_from = open + 1;
      continue;
    }
    const auto next = code.find_first_not_of(" \t\r\n", close + 1);
    if (next == std::string::npos) {
      return 0;
    }
    // Find the end of the cast's argument: either a balanced call `(...)`,
    // or an identifier extended through any `(...)` call chain it leads.
    size_t argument_end = 0;
    if (code[next] == '(') {
      argument_end = matching_delimiter(code, next, '(', ')') + 1;
    } else if (
        std::isalnum(static_cast<unsigned char>(code[next])) ||
        code[next] == '_') {
      size_t end = next;
      while (end < code.size()) {
        const char c = code[end];
        if (std::isalnum(static_cast<unsigned char>(c)) || c == '_') {
          ++end;
        } else if (c == '(') {
          end = matching_delimiter(code, end, '(', ')') + 1;
        } else if (c == '[') {
          // C binds the cast to the whole postfix expression: `(float)inp[c]`
          // means float(inp[c]). Without consuming the subscript the scanner
          // rewrote it to float(inp)[c], splitting the cast from the index —
          // bf16/f16 buffer reads then missed their widening rewrite and
          // every `(float)rel[...]`-style read in the corpus died as
          // "unsupported bfloat16 buffer expression" (2026-10-08
          // KernelRecheck).
          end = matching_delimiter(code, end, '[', ']') + 1;
        } else if (c == '.') {
          // Float-literal tails: `(bfloat16_t)0.0f` must cast the WHOLE
          // literal `0.0f`; stopping at the digit gave float(0).0f
          // (2026-10-08 KernelRecheck, moe_route_fused).
          ++end;
        } else if (
            (c == 'e' || c == 'E' || c == 'p' || c == 'P') &&
            end + 1 < code.size() &&
            (std::isalnum(static_cast<unsigned char>(code[end + 1])) ||
             ((code[end + 1] == '+' || code[end + 1] == '-') &&
              end + 2 < code.size() &&
              std::isalnum(static_cast<unsigned char>(code[end + 2]))))) {
          // hex-float / scientific exponent tails (1e-5, 0x1p-3)
          end += 2;
        } else {
          break;
        }
      }
      argument_end = end;
    } else {
      search_from = open + 1;
      continue;
    }
    // The replace span is the slice from the cast's opening paren to the
    // end of the argument; the replacement is `mapped + "(" + argument + ")"`.
    // Argument text is `code.substr(next, argument_end - next)`. The span
    // size is `argument_end - open`. Clamp against `code.size()` so an
    // unbalanced argument end (impossible with the new matching_delimiter
    // contract, but defensive) cannot pass a too-large count to replace.
    if (argument_end > code.size() || next > code.size() ||
        open > code.size() || argument_end < open) {
      return replacements;
    }
    const std::string argument = code.substr(next, argument_end - next);
    const std::string replacement =
        mapped->second + "(" + argument + ")";
    code.replace(open, argument_end - open, replacement);
    // Skip past the replacement in THIS pass; a cast inside the replacement
    // (e.g. `(float)((float)cq * (float)sq_)`) is rewritten by the NEXT full
    // pass — translate_c_style_casts runs to a fixpoint. Resuming inside the
    // replacement instead produced pathological rewrites like
    // `float((float))(float(cq) * float(sq_)))`
    // (2026-10-08 KernelRecheck, kda_glue_pre).
    search_from = std::min(open + replacement.size(), code.size());
    ++replacements;
  }
  return replacements;
}

void translate_c_style_casts(std::string& code) {
  // Collapse chained scalar-family casts first: `(float)((float)x)` and
  // `(float)(bfloat16_t)(x)` otherwise survive as C-style casts inside
  // constructor arguments after the scan (glslc: GL_NV_explicit_typecast /
  // syntax error — kda_glue_pre/post, 2026-10-08 KernelRecheck). Repeat
  // until stable for chains longer than two.
  static const std::regex chain(
      R"(\(\s*(?:float|int|uint|bool|bfloat16_t|float16_t|int8_t|uint8_t|int16_t|uint16_t|int32_t|uint32_t|int64_t|uint64_t|size_t)\s*\)\s*\(\s*(?:float|int|uint|bool|bfloat16_t|float16_t|int8_t|uint8_t|int16_t|uint16_t|int32_t|uint32_t|int64_t|uint64_t|size_t)\s*\))");
  std::string prev;
  int guard = 0;
  while (code != prev && guard++ < 32) {
    prev = code;
    code = std::regex_replace(code, chain, "(");
  }
  // Run to a fixpoint: each pass rewrites the OUTERMOST remaining casts;
  // casts inside a rewritten constructor argument are handled by later
  // passes (bounded — every pass removes at least one `(cast)` token).
  for (int pass = 0; pass < 16; ++pass) {
    if (translate_c_style_casts_once(code) == 0) {
      break;
    }
  }
}

// Rewrite local device-pointer aliases into (buffer, offset) indexing.
//
// The MSL idiom `device const T* p = base + off; ... p[i]` (37 sites across
// the pinned omlx/TensorFold kernels) has no GLSL form: GLSL has no pointer
// type, but a `T* p = B + O` alias of a [[buffer]] parameter is exactly
// `B[O + i]` at every use. This pass:
//   1. finds declarations `((const)? device (const)? T* NAME = INIT;)`
//      whose INIT is `(cast)? BASE (+ EXPR)?` with BASE a buffer parameter
//      name (or another alias — offsets compose);
//   2. deletes the declaration;
//   3. rewrites every `NAME[EXPR]` to `BASE[(OFF) + (EXPR)]`;
//   4. throws the exact named error when a vector-pointee alias (uint4*,
//      bfloat4*, float4* — indexing granularity differs from the scalar
//      buffer), an array of pointers, or any surviving bare use of the
//      alias remains: those need a wider rewrite and must fail loudly,
//      never silently mis-index.
void translate_device_pointer_aliases(
    std::string& body,
    const std::vector<Parameter>& parameters,
    std::string& prologue_helpers) {
  // Three rigid per-order declaration patterns. GCC's ECMAScript engine
  // mis-binds groups when an alternation and optional const are combined
  // (it shifted the type/name groups by one on
  // 'const device uchar* krow = ...'), so no nested optionals here.
  // Group 1 = address space, group 2 = pointee type, group 3 = alias name,
  // group 4 = initializer. `constant` joins `device`: the CBQ/K3 kernels
  // declare `constant float* cs = cons;` read-only LUT aliases of
  // constant-bound buffers, and the rewrite is identical (2026-10-08
  // KernelRecheck). Without it they died at the surviving-pointer guard.
  static const std::regex alias_patterns[] = {
      std::regex(
          R"((const\s+)?(device|constant)\s+(const\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*\*\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([^;]+);)"),
      std::regex(
          R"((device|constant)\s+(const\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*\*\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([^;]+);)"),
      std::regex(
          R"((device|constant)\s+([A-Za-z_][A-Za-z0-9_]*)\s*\*\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([^;]+);)"),
  };
  struct Alias {
    std::string base;
    std::string offset;
  };
  std::unordered_map<std::string, Alias> aliases;
  std::unordered_set<std::string> vector_aliases;
  auto parameter_exists = [&](const std::string& name) {
    return std::any_of(
        parameters.begin(), parameters.end(),
        [&](const Parameter& parameter) { return parameter.name == name; });
  };
  // Vector pointees change indexing granularity; READ aliases of
  // 4-element bf16 vectors are supported via the _mlx_bf16_to_float4
  // helper (one alias index = four consecutive bf16 elements widened
  // to a float vec4 — the 'float4(Kt4[d])' idiom in the oMLX GDN chunk
  // kernels). Vector WRITES and other vector types still refuse: the
  // elementwise scatter has no consumer here and must fail loudly.
  auto is_vector_type = [](const std::string& type) {
    return !type.empty() && type.find_first_of("234", type.size() - 1) ==
        type.size() - 1;
  };
  bool vector_read_helpers = false;
  bool progressed = true;
  // Per-pattern match-group indices for (type, name, initializer); the
  // optional const/address-space groups shift them per pattern.
  static const int alias_groups[][3] = {{4, 5, 6}, {3, 4, 5}, {2, 3, 4}};
  while (progressed) {
    progressed = false;
    for (size_t pattern_index = 0; pattern_index < std::size(alias_patterns);
         ++pattern_index) {
      const auto& alias_pattern = alias_patterns[pattern_index];
      const auto& groups = alias_groups[pattern_index];
      for (std::sregex_iterator it(body.begin(), body.end(), alias_pattern),
               end;
           it != end; ++it) {
        const auto type = (*it)[groups[0]].str();
        const auto name = (*it)[groups[1]].str();
        if (aliases.count(name)) {
          continue;
        }
        std::string init = trim((*it)[groups[2]].str());
        // Strip a leading C-style device-pointer cast: `(const device T*)`.
        static const std::regex cast_prefix(
            R"(^\(\s*(?:const\s+)?(?:device|constant)\s+(?:const\s+)?[A-Za-z_][A-Za-z0-9_]*\s*\*\s*\)\s*)");
        init = std::regex_replace(init, cast_prefix, "");
        // Split BASE (+ EXPR)?.
        static const std::regex base_offset(
            R"(([A-Za-z_][A-Za-z0-9_]*)\s*(?:\+\s*([^;]+))?)");
        std::smatch parts;
        if (!std::regex_match(init, parts, base_offset)) {
          throw std::runtime_error(
              "unsupported MSL feature `device pointer arithmetic` is not "
              "implemented for the Omarchy Vulkan backend");
        }
        const auto base = parts[1].str();
        std::string offset = parts[2].matched ? trim(parts[2].str()) : "0";
        const bool base_is_parameter = parameter_exists(base);
        const bool base_is_alias = aliases.count(base) > 0;
        if (!base_is_parameter && !base_is_alias) {
          // Aliased through a function call or another buffer expression:
          // fail by name rather than guess.
          throw std::runtime_error(
              "unsupported MSL feature `device pointer arithmetic` is not "
              "implemented for the Omarchy Vulkan backend");
        }
        if (is_vector_type(type)) {
          // vec4-of-bf16 READ aliases are supported through the
          // _mlx_bf16_to_float4 helper (below, at the use-rewrite);
          // every other vector alias (writes, non-4 widths, non-bf16
          // bases the helper cannot serve) still refuses by name.
          const Alias probe{base_is_alias ? aliases[base].base : base,
                            base_is_alias ? aliases[base].offset : offset};
          const auto base_param = std::find_if(
              parameters.begin(), parameters.end(),
              [&](const Parameter& parameter) {
                return parameter.name == probe.base;
              });
          const bool bf16_base = base_is_parameter &&
              base_param != parameters.end() &&
              base_param->type == "bfloat16_t";
          if (type != "vec4" || !bf16_base) {
            throw std::runtime_error(
                "unsupported MSL feature `device pointer alias of vector type "
                "`" + type + "*` is not implemented for the Omarchy Vulkan "
                "backend");
          }
          aliases[name] = {probe.base, probe.offset};
          // Mark as a vector alias: the use-rewrite emits the helper.
          vector_aliases.insert(name);
          body.replace(
              static_cast<size_t>(it->position()), it->length(), "");
          progressed = true;
          break;  // iterators invalidated by the erase; rescan
        }
        if (base_is_alias) {
          const auto& parent = aliases[base];
          offset = "(" + parent.offset + " + " + offset + ")";
          aliases[name] = {parent.base, offset};
        } else {
          aliases[name] = {base, offset};
        }
        body.replace(
            static_cast<size_t>(it->position()), it->length(), "");
        progressed = true;
        break;  // iterators invalidated by the erase; rescan
      }
      if (progressed) {
        break;
      }
    }
  }
  for (const auto& [name, alias] : aliases) {
    const bool is_vector_alias = vector_aliases.count(name) > 0;
    size_t search_from = 0;
    // First: scalar-constructor casts on the alias — `float(NAME)`,
    // `int(NAME)`, etc. (the Qwen3.5 GDN chunk kernels: `float(U0_o)`,
    // `(float)U0_o[(size_t)j * Dv + d]`). Two forms:
    //   (a) functional cast: `T(NAME)`, optionally with whitespace.
    //       Rewrite to `T( base[(off)+0] )` (a first-element read).
    //   (b) C-style cast over an indexed use: `(T)NAME[i]`.
    //       Strip the `(T)` prefix so the indexed path below rewrites
    //       `NAME[i]` normally; the cast stays wrapping the resulting
    //       `base[(off)+i]` expression.
    // The detection walks backward from the position right after NAME:
    //   - if the next non-whitespace char is `(` and the chars before it
    //     form a type token, this is a functional cast (a);
    //   - if NAME is followed by `[`, the chars before NAME form a
    //     parenthesised type token, and the chars immediately before
    //     the `(` are `(` and a type token, this is a C-style cast
    //     over an indexed use (b).
    size_t p = 0;
    while (true) {
      const auto pos = body.find(name, p);
      if (pos == std::string::npos) break;
      p = pos + name.size();
      const size_t after_name = pos + name.size();
      while (p < body.size() && (body[p] == ' ' || body[p] == '\t')) p++;
      if (p >= body.size()) continue;
      if (body[p] == '(') {
        // (a) functional cast
        const auto close = matching_delimiter(body, p, '(', ')');
        size_t e = pos;
        while (e > 0 && (body[e - 1] == ' ' || body[e - 1] == '\t' ||
                           body[e - 1] == '\n' || body[e - 1] == '\r'))
          e--;
        size_t tend = e;
        while (e > 0 &&
               (std::isalnum(static_cast<unsigned char>(body[e - 1])) ||
                body[e - 1] == '_'))
          e--;
        if (e >= tend) continue;
        if (e > 0 && (std::isalnum(static_cast<unsigned char>(body[e - 1])) ||
                      body[e - 1] == '_'))
          continue;
        if (tend < body.size() &&
            (std::isalnum(static_cast<unsigned char>(body[tend])) ||
             body[tend] == '_' || body[tend] == '(' || body[tend] == '['))
          continue;
        const std::string cast_name = body.substr(e, tend - e);
        body.replace(
            e, close - e + 1,
            cast_name + "(" + alias.base + "[(" + alias.offset + ") + 0u])");
        const std::string repl =
            cast_name + "(" + alias.base + "[(" + alias.offset + ") + 0u])";
        p = e + repl.size();
        continue;
      }
      if (body[p] == '[') {
        // possibly (b) C-style cast `(T)NAME[i]`. Walk back past the
        // type token then look for the open paren (skipping anything
        // non-alphanumeric between them: e.g. `)` for the type's
        // closing paren, or a template like `vec<T,4>`).
        size_t paren = pos;
        while (paren > 0 && (body[paren - 1] == ' ' || body[paren - 1] == '\t'))
          paren--;
        // skip any non-alnum chars between the type and '('
        // (e.g. ')' of `(T)`, '>' of `vec<T,4>`, whitespace).
        while (paren > 0 &&
               !std::isalnum(static_cast<unsigned char>(body[paren - 1])) &&
               body[paren - 1] != '(')
          paren--;
        // eat the type token (alnum / underscore)
        while (paren > 0 &&
               (std::isalnum(static_cast<unsigned char>(body[paren - 1])) ||
                body[paren - 1] == '_'))
          paren--;
        // skip whitespace between the type token and '('
        while (paren > 0 && (body[paren - 1] == ' ' || body[paren - 1] == '\t'))
          paren--;
        if (paren == 0 || body[paren - 1] != '(') continue;
        const size_t open_paren = paren - 1;
        const size_t close_paren = body.find(')', open_paren + 1);
        if (close_paren == std::string::npos ||
            close_paren > pos /* closing ')' must be before NAME */)
          continue;
        // type token between '(' and ')'
        std::string t = trim(body.substr(open_paren + 1, close_paren - open_paren - 1));
        if (t.empty()) continue;
        if (!(std::isalnum(static_cast<unsigned char>(t.front())) || t.front() == '_') ||
            !(std::isalnum(static_cast<unsigned char>(t.back())) || t.back() == '_'))
          continue;
        // strip the cast '(T)': from open_paren to close_paren inclusive
        body.erase(open_paren, close_paren - open_paren + 1);
        p = pos - (close_paren - open_paren + 1);  // rescan at NAME start
        continue;
      }
      // not a cast form: handled by the indexed / bare-use path below.
    }
    // Then: indexed uses and bare (non-cast) uses follow the existing
    // rewrite path below.
    while (true) {
      const auto position = body.find(name, search_from);
      if (position == std::string::npos) {
        break;
      }
      const bool left_ok = position == 0 ||
          !(std::isalnum(static_cast<unsigned char>(body[position - 1])) ||
            body[position - 1] == '_');
      if (!left_ok) {
        search_from = position + name.size();
        continue;
      }
      const size_t after = position + name.size();
      if (after >= body.size() ||
          std::isalnum(static_cast<unsigned char>(body[after])) ||
          body[after] == '_') {
        // A longer identifier (e.g. base_weight) merely contains the name;
        // but an alias at end-of-body has no indexing use left either.
        if (after >= body.size()) {
          throw std::runtime_error(
              "unsupported MSL feature `device pointer arithmetic` is not "
              "implemented for the Omarchy Vulkan backend");
        }
        search_from = position + name.size();
        continue;
      }
      if (body[after] != '[') {
        // Bare use (call argument, arithmetic, address-of): fail loudly.
        throw std::runtime_error(
            "unsupported MSL feature `device pointer arithmetic` is not "
            "implemented for the Omarchy Vulkan backend");
      }
      const auto close = matching_delimiter(body, after, '[', ']');
      const std::string inner = body.substr(after + 1, close - after - 1);
      std::string replacement;
      if (is_vector_alias) {
        // One alias index spans four consecutive bf16 elements: widen
        // them to a float vec4 (the float4(Kt4[d]) read idiom).
        vector_read_helpers = true;
        replacement = "_mlx_bf16_to_float4(" + alias.base + "[(" +
            alias.offset + ") + (4u * (" + inner + "))], " + alias.base +
            "[( " + alias.offset + ") + (4u * (" + inner + ")) + 1u], " +
            alias.base + "[(" + alias.offset + ") + (4u * (" + inner +
            ")) + 2u], " + alias.base + "[(" + alias.offset +
            ") + (4u * (" + inner + ")) + 3u])";
      } else {
        replacement = alias.base + "[(" + alias.offset + ") + (" + inner +
            ")]";
      }
      body.replace(position, close - position + 1, replacement);
      search_from = position + replacement.size();
    }
  }
  if (vector_read_helpers) {
    // Emitted into the GLSL prologue next to the other bf16 helpers
    // (they define _mlx_bf16_to_float, which this helper calls).
    prologue_helpers =
        "vec4 _mlx_bf16_to_float4(uint16_t a, uint16_t b, uint16_t c, "
        "uint16_t d) { return vec4(_mlx_bf16_to_float(a), "
        "_mlx_bf16_to_float(b), _mlx_bf16_to_float(c), "
        "_mlx_bf16_to_float(d)); }\n";
  }
}

// MSL `auto` local declarations have no GLSL counterpart: `auto` is a
// reserved word, and glslang fails the surviving declaration with
// 'syntax error, unexpected IDENTIFIER, expecting COMMA or SEMICOLON'
// (Qwen3.5-9B serve, first decode token, bf16 q_out [1,1,16,128], the
// omlx_qwen35_gdn_prework kernel's
// `const auto sy = 1 / (1 + metal::precise::exp(metal::abs(conv)));` —
// 2026-10-09, offline repro at .comp:72/73; v0.7.31 9b5c938 fails
// identically at .comp:60, so the construct was never supported, only
// newly reached). Two corpus shapes:
//   * value declarations (`const auto sy = 1 / (1 + ...)`) become float:
//     Metal deduces the initializer's type and every value auto in the
//     serving corpus is an fp32 expression (GLSL's implicit
//     int->float declaration conversion covers degenerate integer forms
//     at float precision);
//   * pointer declarations (`const auto x_row = x + row * K;` — the MoE
//     router/decode row-walk idiom, half the Qwen3.5 decode kernels)
//     deduce the base buffer's element type; they are rewritten to the
//     explicit `const device T*` form the device-pointer alias pass
//     already translates, and scoped rebinds (`const auto x = x_row;`,
//     x_row itself an alias) chain through the same type map.
void translate_auto_declarations(
    std::string& body,
    const std::vector<Parameter>& parameters) {
  static const std::regex auto_decl(
      R"((const\s+)?auto\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([^;{}]+);)");
  static const std::regex base_offset(
      R"(^\s*([A-Za-z_][A-Za-z0-9_]*)\s*(?:\+|$))");
  std::unordered_map<std::string, std::string> pointer_types;
  for (const auto& parameter : parameters) {
    if (!parameter.scalar) {
      pointer_types[parameter.name] = parameter.type;
    }
  }
  std::string rewritten;
  rewritten.reserve(body.size());
  size_t last = 0;
  for (std::sregex_iterator it(body.begin(), body.end(), auto_decl), end;
       it != end;
       ++it) {
    const auto& match = *it;
    rewritten += body.substr(last, match.position() - last);
    const std::string name = match[2].str();
    const std::string init = trim(match[3].str());
    std::smatch parts;
    if (std::regex_search(init, parts, base_offset)) {
      const auto base = pointer_types.find(parts[1].str());
      if (base != pointer_types.end()) {
        rewritten += match[1].str() + "device " + base->second + "* " + name +
            " = " + init + ";";
        pointer_types[name] = base->second;
        last = match.position() + match.length();
        continue;
      }
    }
    // Value declaration: Metal deduces the initializer's type. Float is
    // correct only when the initializer is provably float-valued: a float
    // literal, a math builtin, or a read through a pointer alias or a
    // declared float local (bf16 buffer reads widen to float). Integer
    // initializers (`auto n = thread_position_in_grid.z; auto hv_idx =
    // n % Hv;`, gated-delta step) would silently compute at float
    // precision and then fail on `%` and indexing — refuse them by name.
    const std::regex float_local_decl(
        R"(\bfloat\s+([A-Za-z_][A-Za-z0-9_]*)\s*(?:=|\[|;))");
    const std::regex pointer_decl(
        R"(\b(?:device|constant)\s+(?:const\s+)?[A-Za-z_][A-Za-z0-9_]*\s*\*\s*([A-Za-z_][A-Za-z0-9_]*))");
    std::unordered_set<std::string> float_locals;
    for (std::sregex_iterator f(body.begin(), body.end(), float_local_decl),
             fend;
         f != fend;
         ++f) {
      float_locals.insert((*f)[1].str());
    }
    for (std::sregex_iterator f(body.begin(), body.end(), pointer_decl), fend;
         f != fend;
         ++f) {
      float_locals.insert((*f)[1].str());
    }
    static const std::regex float_literal(
        R"([0-9]+\.[0-9]*(?:[eE][+-]?[0-9]+)?f?|[0-9]+[eE][+-]?[0-9]+f?|[0-9]+f\b)");
    static const std::regex float_call(
        R"(\b(exp|exp2|log|log2|sqrt|rsqrt|sin|cos|tan|pow|floor|ceil|round|roundEven|inversesqrt)\s*\()");
    bool float_valued = std::regex_search(init, float_literal) ||
        std::regex_search(init, float_call);
    for (const auto& local : float_locals) {
      if (std::regex_search(init,
              std::regex("\\b" + regex_escape(local) + "\\b"))) {
        float_valued = true;
        break;
      }
    }
    if (!float_valued) {
      throw std::runtime_error(
          "unsupported MSL feature `auto` value declaration with a "
          "non-float initializer (declare the type explicitly)");
    }
    rewritten += match[1].str() + "float " + name + " = " + init + ";";
    last = match.position() + match.length();
  }
  rewritten += body.substr(last);
  body = std::move(rewritten);
}

// Metal's pointer-advance idiom walks buffer rows:
//   `const device T* kptr = keys + off; ... kptr[j] ...; kptr += step;`
// (the mlx_vlm qwen3_5 ragged-SDPA attention kernels — the Qwen3.6-35B-A3B
// prefill refusal 'device pointer arithmetic', 2026-10-09; the gated-delta
// step walks the same way, and its auto row pointers arrive here in the
// explicit form translate_auto_declarations emits). GLSL has no pointers,
// but the walk is exactly an index variable: the advance becomes an add on
// a uint location, the declaration initializes it, and indexed uses read
// through it. Buffer parameters advanced in place (`y += Hv * Dv;`) get a
// zero-initialized location declared at the top of the body.
void translate_pointer_advance(
    std::string& body,
    const std::vector<Parameter>& parameters) {
  const auto parameter_exists = [&](const std::string& name) {
    return std::any_of(
        parameters.begin(),
        parameters.end(),
        [&](const Parameter& parameter) { return parameter.name == name; });
  };
  // 1. Advanced buffer parameters.
  for (const auto& parameter : parameters) {
    if (parameter.scalar) {
      continue;
    }
    const auto escaped = regex_escape(parameter.name);
    const std::regex advance("\\b" + escaped + R"(\s*\+=\s*([^;]+);)");
    if (!std::regex_search(body, advance)) {
      continue;
    }
    body = std::regex_replace(
        body, advance, "_mlx_" + parameter.name + "_loc += $1;");
    body = std::regex_replace(
        body,
        std::regex("\\b" + escaped + R"(\s*\[([^\]]+)\])"),
        parameter.name + "[(_mlx_" + parameter.name + "_loc) + ($1)]");
    body.insert(0, "uint _mlx_" + parameter.name + "_loc = 0u;\n");
  }
  // 2. Advanced pointer aliases. Same three declaration shapes as the alias
  // pass (no nested optionals: GCC's ECMAScript engine mis-binds them).
  static const std::regex advance_alias_patterns[] = {
      std::regex(
          R"((const\s+)?(device|constant)\s+(const\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*\*\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([^;]+);)"),
      std::regex(
          R"((device|constant)\s+(const\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*\*\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([^;]+);)"),
      std::regex(
          R"((device|constant)\s+([A-Za-z_][A-Za-z0-9_]*)\s*\*\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([^;]+);)"),
  };
  // Per-pattern (name, initializer) match-group indices; the optional
  // const/address-space groups shift them per pattern (same layout rule as
  // alias_groups below).
  static const int advance_groups[][2] = {{5, 6}, {4, 5}, {3, 4}};
  bool progressed = true;
  while (progressed) {
    progressed = false;
    for (size_t pattern_index = 0;
         pattern_index < std::size(advance_alias_patterns);
         ++pattern_index) {
      const auto& pattern = advance_alias_patterns[pattern_index];
      const auto groups = advance_groups[pattern_index];
      for (std::sregex_iterator it(body.begin(), body.end(), pattern), end;
           it != end;
           ++it) {
        const auto name = (*it)[groups[0]].str();
        const std::regex is_advanced(
            "\\b" + regex_escape(name) + R"(\s*\+=)");
        if (!std::regex_search(body, is_advanced)) {
          continue;
        }
        std::string init = trim((*it)[groups[1]].str());
        static const std::regex cast_prefix(
            R"(^\(\s*(?:const\s+)?(?:device|constant)\s+(?:const\s+)?[A-Za-z_][A-Za-z0-9_]*\s*\*\s*\)\s*)");
        init = std::regex_replace(init, cast_prefix, "");
        static const std::regex base_offset(
            R"(([A-Za-z_][A-Za-z0-9_]*)\s*(?:\+\s*([^;]+))?)");
        std::smatch parts;
        if (!std::regex_match(init, parts, base_offset)) {
          throw std::runtime_error(
              "unsupported MSL feature `device pointer arithmetic` is not "
              "implemented for the Omarchy Vulkan backend");
        }
        const auto base = parts[1].str();
        const auto offset = parts[2].matched ? trim(parts[2].str()) : "0";
        if (!parameter_exists(base)) {
          // A walk whose base is another alias or an expression has no
          // single location variable; fail by name rather than guess.
          throw std::runtime_error(
              "unsupported MSL feature `device pointer arithmetic` is not "
              "implemented for the Omarchy Vulkan backend");
        }
        // Replace the full declaration first (the later rewrites shift
        // positions, and a name-only replacement would orphan the type
        // prefix for the alias pass).
        body.replace(
            static_cast<size_t>(it->position()),
            it->length(),
            "uint _mlx_" + name + "_loc = uint(" + offset + ");");
        body = std::regex_replace(
            body, is_advanced, "_mlx_" + name + "_loc +=");
        body = std::regex_replace(
            body,
            std::regex("\\b" + regex_escape(name) + R"(\s*\[([^\]]+)\])"),
            base + "[(_mlx_" + name + "_loc) + ($1)]");
        progressed = true;
        break;
      }
      if (progressed) {
        break;
      }
    }
  }
}

// Values of the body's foldable `const int NAME = <literal arithmetic>;`
// declarations (template substitution has already made the expressions
// numeric). Used by the array-initializer and shared-hoist passes, which
// need extents as compile-time integers. Parenthesized expressions do not
// fold and simply stay absent from the map.
std::map<std::string, int> fold_const_ints(const std::string& body) {
  static const std::regex const_int(
      R"(\bconst\s+int\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*((?:int\s*\(\s*)*)([0-9A-Za-z_][0-9A-Za-z_\s*/+\-]*[0-9A-Za-z_])\s*(?:\)\s*)*\;)");
  std::vector<std::pair<std::string, std::string>> pending;
  std::map<std::string, int> values;
  for (std::sregex_iterator it(body.begin(), body.end(), const_int), end;
       it != end;
       ++it) {
    pending.emplace_back((*it)[1].str(), (*it)[3].str());
  }
  // Extent expressions may reference other folded constants
  // (`constexpr int v_per_thread = D_SIZE / BD;`): substitute known values
  // and re-evaluate until a pass makes no progress.
  for (int round = 0; round < 4 && !pending.empty(); ++round) {
    std::vector<std::pair<std::string, std::string>> remaining;
    for (const auto& [name, expr] : pending) {
      std::string resolved = expr;
      for (const auto& [known, value] : values) {
        resolved = std::regex_replace(
            resolved,
            std::regex("\\b" + regex_escape(known) + "\\b"),
            std::to_string(value));
      }
      if (resolved.find_first_not_of("0123456789+-*/ \t") !=
          std::string::npos) {
        remaining.emplace_back(name, resolved);
        continue;
      }
      static const std::regex piece(R"(([0-9]+)|([+\-*/]))");
      std::vector<int> numbers;
      std::vector<char> ops;
      bool ok = true;
      for (std::sregex_iterator p(resolved.begin(), resolved.end(), piece),
               piece_end;
           p != piece_end;
           ++p) {
        if ((*p)[1].matched) {
          numbers.push_back(std::atoi((*p)[1].str().c_str()));
        } else {
          ops.push_back((*p)[2].str()[0]);
        }
      }
      if (numbers.empty() || ops.size() + 1 != numbers.size()) {
        remaining.emplace_back(name, resolved);
        continue;
      }
      std::vector<int> terms;
      terms.push_back(numbers[0]);
      std::vector<char> adds;
      for (size_t i = 0; i < ops.size(); ++i) {
        if (ops[i] == '*') {
          terms.back() *= numbers[i + 1];
        } else if (ops[i] == '/') {
          if (numbers[i + 1] == 0) {
            ok = false;
            break;
          }
          terms.back() /= numbers[i + 1];
        } else {
          adds.push_back(ops[i]);
          terms.push_back(numbers[i + 1]);
        }
      }
      if (!ok) {
        continue;
      }
      int value = terms[0];
      for (size_t i = 0; i < adds.size(); ++i) {
        value = adds[i] == '+' ? value + terms[i + 1] : value - terms[i + 1];
      }
      values[name] = value;
    }
    if (remaining.size() == pending.size()) {
      break;
    }
    pending = std::move(remaining);
  }
  return values;
}

void translate_as_type(
    std::string& code,
    const std::vector<Parameter>& parameters) {
  // as_type<Dest>(src) is a bitcast; src may contain nested parentheses, so
  // match the argument by balanced delimiters rather than a flat regex.
  // Float sources map to the floatBitsTo* / uintBitsToFloat reinterprets;
  // INTEGER sources are value-preserving already, so as_type<uint>(int_expr)
  // is the identity constructor uint(expr) — floatBitsToUint on an int
  // reinterprets the FLOAT bits of the converted value and produces garbage
  // (the llguidance mask kernel masks with mask bits read as int,
  // 2026-10-08 KernelRecheck).
  static const std::unordered_map<std::string, std::string> float_bitcasts = {
      {"float", "uintBitsToFloat"},
      {"uint", "floatBitsToUint"},
      {"uint32_t", "floatBitsToUint"},
      {"int", "floatBitsToInt"},
      {"int32_t", "floatBitsToInt"},
  };
  static const std::unordered_map<std::string, std::string> int_conversions = {
      {"float", "float"},
      {"uint", "uint"},
      {"uint32_t", "uint"},
      {"int", "int"},
      {"int32_t", "int"},
  };
  auto param_is_integer = [&parameters](const std::string& expression) {
    for (const auto& parameter : parameters) {
      const bool integer_type = parameter.type == "int" ||
          parameter.type == "int32_t" || parameter.type == "uint" ||
          parameter.type == "uint32_t" || parameter.type == "int8_t" ||
          parameter.type == "uint8_t" || parameter.type == "int16_t" ||
          parameter.type == "uint16_t" || parameter.type == "bool";
      if (!integer_type) {
        continue;
      }
      // The operand names the parameter directly (mask[...], scalar form).
      // NOT static: a static here captures the first integer parameter's
      // name for the whole process and every later kernel probes that name
      // instead of its own (llguidance masked half the vocabulary at
      // random, 2026-10-08 KernelRecheck).
      const std::regex direct(
          R"((^|[^.\w]))" + regex_escape(parameter.name) + R"((?!\w))");
      if (std::regex_search(expression, direct)) {
        return true;
      }
    }
    return false;
  };
  size_t search_from = 0;
  while (true) {
    const auto marker = code.find("as_type<", search_from);
    if (marker == std::string::npos) {
      return;
    }
    const auto open_angle = marker + 8;
    const auto close_angle = code.find('>', open_angle);
    if (close_angle == std::string::npos) {
      return;
    }
    const auto destination = trim(code.substr(open_angle, close_angle - open_angle));
    const auto mapped = float_bitcasts.find(destination);
    if (mapped == float_bitcasts.end()) {
      throw std::runtime_error("unsupported MSL feature `as_type<" + destination + ">`");
    }
    const auto open_paren = code.find('(', close_angle + 1);
    if (open_paren == std::string::npos) {
      return;
    }
    const auto close_paren = matching_delimiter(code, open_paren, '(', ')');
    const auto argument =
        code.substr(open_paren + 1, close_paren - open_paren - 1);
    std::string replacement;
    if (destination == "uint" || destination == "uint32_t" ||
        destination == "int" || destination == "int32_t") {
      const auto converted = int_conversions.find(destination);
      if (param_is_integer(argument)) {
        replacement =
            converted->second + "(" + argument + ")";
      } else {
        replacement = mapped->second + "(" + argument + ")";
      }
    } else {
      replacement = mapped->second + "(" + argument + ")";
    }
    code.replace(
        marker, close_paren - marker + 1, replacement);
    search_from = marker;
  }
}

std::vector<Parameter> parse_parameters(const std::string& signature) {
  const std::regex parameter_pattern(
      R"((?:const\s+)?(device|constant)\s+(atomic<)?([A-Za-z_][A-Za-z0-9_]*)(?:>)?\s*([*&])\s*([A-Za-z_][A-Za-z0-9_]*)\s*\[\[buffer\(([0-9]+)\)\]\])");
  std::vector<Parameter> parameters;
  for (std::sregex_iterator it(
           signature.begin(), signature.end(), parameter_pattern),
       end;
       it != end;
       ++it) {
    parameters.push_back(
        {(*it)[3].str(),
         (*it)[5].str(),
         static_cast<uint32_t>(std::stoul((*it)[6].str())),
         (*it)[4].str() == "&",
         (*it)[2].matched,
         (*it)[1].str() == "constant"});
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
    auto value = trim(value_parts[index]);
    if (declaration_parts[index].find("bool") != std::string::npos) {
      value = value == "0" ? "false" : "true";
    } else if (declaration_parts[index].find("typename") != std::string::npos) {
      // A typename instantiated as any bf16 representation substitutes
      // float — but NOT as a plain widen: Metal's bfloat16_t operators
      // round EVERY assignment and constructor to bfloat16 (compute fp32,
      // round on store). The plain substitution rounded only once at the
      // final output store, and the shader diverged from the composed
      // bf16 op chain it replaces (mlx-serve SwiGLU: 26% of elements off
      // by 1 bf16 ulp, 2026-10-07 parity probe). Mirror Metal:
      // constructor casts `T(expr)` and declarations/assignments of
      // T-typed locals round through the bf16 storage pattern.
      // Buffer parameters keep uint16_t storage (buffer_declaration uses
      // glsl_type separately) and narrow via _mlx_float_to_bf16 at
      // stores. Without the float substitution, the Qwen3.5-2B GDN decode
      // leg emits 'uint16_t sy = uint16_t(1) / (uint16_t(1) +
      // exp(abs(conv)));' — integer math where Metal computes in fp32.
      if (value == "bfloat16_t" || value == "bfloat16" ||
          value == "uint16_t") {
        const auto bfloat_name = name;  // the template parameter name (e.g. T)
        // Constructor casts: `T(expr)` -> `_mlx_bf16_round_trip(expr)`.
        // Word-bounded: a longer identifier ending in the template name
        // (POST in the residual+RMSNorm kernel) must not lose its tail.
        replace_word(body, bfloat_name + "(", "_mlx_bf16_round_trip(");
        replace_word(header, bfloat_name + "(", "_mlx_bf16_round_trip(");
        // Declarations `T name = expr;`: the declaration becomes float and
        // the initializing expression rounds like Metal's constructor.
        const std::regex decl_pattern(
            "\\b" + bfloat_name +
            R"(\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([^;]+);)");
        std::vector<std::string> bf16_locals;
        for (std::sregex_iterator it(body.begin(), body.end(), decl_pattern);
             it != std::sregex_iterator();
             ++it) {
          bf16_locals.push_back((*it)[1].str());
        }
        replace_word(body, bfloat_name + " ", "float ");
        replace_word(header, bfloat_name + " ", "float ");
        for (const auto& local : bf16_locals) {
          body = std::regex_replace(
              body,
              std::regex(
                  R"((^|[^=!<>+\-*/%&|^[:alnum:]_]))" + regex_escape(local) +
                  R"(\s*=\s*([^;]+);)"),
              "$1" + local + " = _mlx_bf16_round_trip($2);");
        }
        value = "float";
      } else if (value == "float16_t" || value == "half") {
        // Metal promotes half arithmetic to float (half storage, fp32
        // compute); float locals mirror that and the buffer reads widen in
        // the float16 parameter pass below (2026-10-08 KernelRecheck,
        // bitlinear_matmul: `1 / weight_scale[0]` cannot compile as
        // int / float16_t in GLSL).
        value = "float";
      } else if (value == "true" || value == "false" ||
                 std::isdigit(static_cast<unsigned char>(value.front()))) {
        // boolean and numeric template values pass through unchanged;
        // only other type names (alphabetic first char) map through
        // glsl_type.
      } else if (!value.empty() &&
                 std::isalpha(static_cast<unsigned char>(value.front()))) {
        value = glsl_type(value);
      }
    }
    replace_word(header, name, value);
    replace_word(body, name, value);
  }
}

// Helpers in the user header may take `threadgroup T*` parameters (shared
// scratch passed by pointer, e.g. the msv_row_inv_rms reduction stage in
// mlx-serve's fused residual+RMSNorm kernel). GLSL has neither pointer
// parameters nor storage-qualified parameters. When every call site in the
// kernel body passes the shared array under its own declared name, the
// parameter is dropped from the signature: the helper body's references then
// resolve to the kernel body's global `shared` arrays directly. Anything
// else — an alias, an expression, a missing argument — is refused with the
// construct named.
void specialize_threadgroup_helper_params(
    std::string& header,
    std::string& body) {
  static const std::regex helper_signature(
      R"(\b([A-Za-z_][A-Za-z0-9_]*)\s+([A-Za-z_][A-Za-z0-9_]*)\s*\()",
      std::regex_constants::optimize);
  static const std::regex threadgroup_param(
      R"(threadgroup\s+([A-Za-z_][A-Za-z0-9_]*)\s*\*\s*([A-Za-z_][A-Za-z0-9_]*))",
      std::regex_constants::optimize);
  // Each pass erases at most one parameter; erasing invalidates match
  // positions, so rescan from the start until no parameter remains.
  while (true) {
    bool erased = false;
    for (std::sregex_iterator it(
             header.begin(), header.end(), helper_signature);
         it != std::sregex_iterator();
         ++it) {
      const std::string helper = (*it)[2].str();
      const size_t paren = header.find('(', (*it).position(0));
      if (paren == std::string::npos) {
        continue;
      }
      const size_t close = matching_delimiter(header, paren, '(', ')');
      const std::string params = header.substr(paren + 1, close - paren - 1);
      std::sregex_iterator tp(
          params.begin(), params.end(), threadgroup_param);
      if (tp == std::sregex_iterator()) {
        continue;
      }
      const std::string ptype = (*tp)[1].str();
      const std::string pname = (*tp)[2].str();
      const size_t param_index = static_cast<size_t>(std::count(
          params.begin(), params.begin() + (*tp).position(0), ','));
      // Every call site in the kernel body must pass the same-named array.
      size_t search = 0;
      size_t calls = 0;
      while (true) {
        const size_t call = body.find(helper + "(", search);
        if (call == std::string::npos) {
          break;
        }
        if (call > 0 &&
            (std::isalnum(static_cast<unsigned char>(body[call - 1])) ||
             body[call - 1] == '_')) {
          search = call + 1;
          continue;
        }
        const size_t call_open = call + helper.size();
        const size_t call_close =
            matching_delimiter(body, call_open, '(', ')');
        const auto arguments =
            split(body.substr(call_open + 1, call_close - call_open - 1), ',');
        if (param_index >= arguments.size()) {
          throw std::runtime_error(
              "unsupported threadgroup helper parameter: call of `" +
              helper + "` is missing argument " +
              std::to_string(param_index));
        }
        std::string argument = arguments[param_index];
        argument.erase(0, argument.find_first_not_of(" \t\r\n"));
        argument.erase(argument.find_last_not_of(" \t\r\n") + 1);
        if (argument != pname) {
          throw std::runtime_error(
              "unsupported threadgroup helper parameter: call of `" +
              helper + "` passes `" + argument + "` for shared array `" +
              pname + "` (GLSL cannot pass shared storage as a parameter)");
        }
        ++calls;
        search = call_close + 1;
      }
      if (calls == 0) {
        throw std::runtime_error(
            "unsupported threadgroup helper parameter: `" + helper +
            "` takes `threadgroup " + ptype + "* " + pname +
            "` but the kernel body never calls it");
      }
      // Drop the matching argument from every call site (reverse order so
      // the collected positions stay valid while erasing). The source
      // separates arguments with ", ", so each preceding argument consumes
      // its trimmed length plus the two separator characters.
      std::vector<std::pair<size_t, size_t>> cuts;
      search = 0;
      while (true) {
        const size_t call = body.find(helper + "(", search);
        if (call == std::string::npos) {
          break;
        }
        if (call > 0 &&
            (std::isalnum(static_cast<unsigned char>(body[call - 1])) ||
             body[call - 1] == '_')) {
          search = call + 1;
          continue;
        }
        const size_t call_open = call + helper.size();
        const size_t call_close =
            matching_delimiter(body, call_open, '(', ')');
        const auto arguments =
            split(body.substr(call_open + 1, call_close - call_open - 1), ',');
        size_t arg_start = call_open + 1;
        for (size_t index = 0; index < param_index; ++index) {
          arg_start += arguments[index].size() + 2;
        }
        size_t begin = arg_start;
        size_t end = begin + arguments[param_index].size();
        if (param_index + 1 < arguments.size()) {
          end += 2;  // ", "
        } else if (param_index > 0) {
          begin -= 2;  // ", "
        }
        cuts.emplace_back(begin, end);
        search = call_close + 1;
      }
      for (auto it = cuts.rbegin(); it != cuts.rend(); ++it) {
        body.erase(it->first, it->second - it->first);
      }
      // Erase the parameter from this signature (with one adjacent comma).
      const std::string bare = "threadgroup " + ptype + "* " + pname;
      size_t at = header.find(bare, paren);
      if (at == std::string::npos || at + bare.size() > close) {
        throw std::runtime_error(
            "threadgroup helper parameter vanished while rewriting");
      }
      size_t end = at + bare.size();
      if (header.compare(end, 2, ", ") == 0) {
        end += 2;
      } else if (at > paren + 1 && header.compare(at - 2, 2, ", ") == 0) {
        at -= 2;
      }
      header.erase(at, end - at);
      erased = true;
      break;
    }
    if (!erased) {
      return;
    }
  }
}

void translate_header(std::string& header) {
  // Any #include has no GLSL meaning: the OMARCHY ICD compiles the shader
  // standalone, and a kernel that actually calls the included APIs fails
  // later on its own tokens (metal_stdlib helpers are mapped above; MPP
  // tensor ops surface as matmul2d_descriptor / tensor<> in the body and
  // fail GLSL compilation by name). Stripping all includes keeps pure
  // marker includes -- e.g. the H3 _QUANTIZE kernel's
  // MetalPerformancePrimitives include, which its body never uses --
  // from blocking an otherwise translatable kernel.
  header = std::regex_replace(
      header, std::regex(R"(#include\s*[<\"][^>\"]*[>\"])"), "");
  replace_all(header, "#include <metal_stdlib>", "");
  // Namespace-qualified using-directives (using namespace mpp::tensor_ops;)
  // have no GLSL equivalent; strip the whole directive. Unqualified using
  // declarations survive and fail later by name if they reference
  // Metal-only symbols.
  header = std::regex_replace(
      header,
      std::regex(
          R"(using\s+namespace\s+[A-Za-z_][A-Za-z0-9_]*(::[A-Za-z_][A-Za-z0-9_]*)*\s*;)"),
      "");
  replace_all(header, "using namespace metal;", "");
  // After resolve_kernel_templates has substituted concrete types for
  // template parameters, 'template <typename T>' declarations in the
  // header become vestigial (GLSL has no C++ templates; the functions
  // are effectively monomorphized by the type substitution). Strip the
  // template prefix so the functions become regular GLSL functions.
  // Without this, the QMV header's 'template <typename T> inline float
  // load_vector(...)' triggers 'unsupported user header declaration'.
  header = std::regex_replace(
      header, std::regex(R"(template\s*<\s*typename\s+[A-Za-z_][A-Za-z0-9_]*\s*>\s*)"),
      "");
  replace_all(header, "using namespace metal;", "");
  replace_all(header, "metal::precise::", "");
  replace_all(header, "metal::fast::", "");
  replace_all(header, "metal::", "");
  map_numeric_limits(header);
  header = std::regex_replace(
      header,
      std::regex(
          R"(template\s*<\s*typename\s+([A-Za-z_][A-Za-z0-9_]*)\s*>\s*\1\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(\s*\1\s+([A-Za-z_][A-Za-z0-9_]*)\s*\))"),
      "float $2(float $3)");
  translate_types(header);
  // Header helpers keep C-style casts on their parameters
  // (`(float)exp(abs((float)x))` in the K3 kda glue sigmoid); GLSL rejects
  // them, so the body cast pass runs here too (2026-10-08 KernelRecheck).
  translate_c_style_casts(header);
  // Helper functions in the user header (e.g. the msv_row_inv_rms stage
  // helper in mlx-serve's fused residual+RMSNorm kernel) keep their MSL
  // spellings unless the statement-level body rewrites also run here.
  // METAL_FUNC marks file-local helpers; GLSL has no storage-class keyword
  // for them, so it is simply dropped. simd_* reductions and
  // threadgroup_barrier map to subgroup ops exactly as in the body.
  // `threadgroup T*` parameters have no GLSL equivalent (shared arrays
  // cannot be passed as pointer parameters), so a header that uses one is
  // refused cleanly with the construct named.
  replace_all(header, "METAL_FUNC", "");
  // `inline` marks MSL header helpers (the K3 kda glue sigmoid); GLSL has no
  // such qualifier and glslang rejects it, so drop it (2026-10-08
  // KernelRecheck). Header-only: `inline` stays a legal body identifier.
  replace_word(header, "inline", "");
  replace_all(
      header, "threadgroup_barrier(mem_flags::mem_threadgroup)", "barrier()");
  replace_all(header, "threadgroup_barrier(mem_flags::mem_device)", "barrier()");
  replace_all(header, "simd_sum", "subgroupAdd");
  replace_all(header, "simd_max", "subgroupMax");
  replace_all(header, "simd_min", "subgroupMin");
  // metal::precise::rsqrt survives the metal:: strip above as rsqrt; GLSL
  // names it inversesqrt (same mapping the body rewrites apply).
  replace_word(header, "rsqrt", "inversesqrt");
  replace_word(header, "fabs", "abs");
  if (header.find("threadgroup") != std::string::npos) {
    throw std::runtime_error(
        "unsupported MSL feature `threadgroup` in a helper function "
        "(GLSL allows shared storage only as a file-scope declaration in "
        "the kernel body; helper parameters must be specialized away)");
  }
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

// A float16_t buffer read keeps its half type where the context cannot take
// a float: the whole right-hand side of a plain `=` (`sh_a[i] = embedding[j];`
// into a float16_t lvalue) and a function-call argument (`exact_fma16(acc,
// x, weights[j])` into a float16_t parameter). GLSL widens float16_t to float
// implicitly, never the other way, so the unwidened read is valid wherever the
// widened one is, and valid in the two places the widened one is not. Any
// operator context still widens: mixed int/f16 arithmetic needs the float.
bool half_read_is_plain_value(const std::string& s, size_t begin, size_t end) {
  size_t before = begin;
  while (before > 0 && std::isspace(static_cast<unsigned char>(s[before - 1]))) {
    --before;
  }
  size_t after = end;
  while (after < s.size() && std::isspace(static_cast<unsigned char>(s[after]))) {
    ++after;
  }
  if (before == 0 || after >= s.size()) {
    return false;
  }
  const char prev = s[before - 1];
  const char next = s[after];
  if (prev == '=') {
    const bool plain_assign = before < 2 ||
        std::strchr("=!<>+-*/%&|^", s[before - 2]) == nullptr;
    return plain_assign && next == ';';
  }
  if (prev == ',') {
    return next == ',' || next == ')' || next == '}';
  }
  if (prev == '{') {
    // First element of a brace initializer (`float16_t arr[2] = {x[0], x[1]};`):
    // the array constructor takes the element type, and GLSL widens half to
    // float in it but not float to half.
    return next == ',' || next == '}';
  }
  if (prev == '(' && (next == ',' || next == ')')) {
    size_t name_end = before - 1;
    while (name_end > 0 &&
           std::isspace(static_cast<unsigned char>(s[name_end - 1]))) {
      --name_end;
    }
    return name_end > 0 &&
        (std::isalnum(static_cast<unsigned char>(s[name_end - 1])) ||
         s[name_end - 1] == '_');
  }
  return false;
}

// Statements that assign into a float16_t local or shared array (`acc += x;`,
// `acc = acc + x;`, `sh[i] = x;`) store through float16_t(): the right-hand
// side may be float after the half reads were widened, and GLSL does not
// convert float to float16_t on assignment. Compound forms expand to
// `lhs = float16_t(lhs op (rhs));`. A statement whose right-hand side is
// already a whole float16_t(...) call is left alone. A match inside a for
// header (the scan meets an unmatched `)` before the `;`) is skipped.
void wrap_half_lvalue_assignments(
    std::string& body,
    const std::string& hoisted_declarations,
    const std::unordered_set<std::string>& half_outputs) {
  static const std::regex declaration(
      R"(\bfloat16_t\s+([A-Za-z_][A-Za-z0-9_]*))");
  std::unordered_set<std::string> half_names;
  const std::string* const sources[2] = {&body, &hoisted_declarations};
  for (const std::string* text : sources) {
    for (std::sregex_iterator it(text->begin(), text->end(), declaration), stop;
         it != stop;
         ++it) {
      half_names.insert((*it)[1].str());
    }
  }
  if (half_names.empty() && half_outputs.empty()) {
    return;
  }
  auto is_ident = [](char c) {
    return std::isalnum(static_cast<unsigned char>(c)) || c == '_';
  };
  auto is_space = [](char c) {
    return std::isspace(static_cast<unsigned char>(c)) != 0;
  };
  const size_t size = body.size();
  std::string out;
  size_t copied = 0;
  size_t i = 0;
  while (i < size) {
    if (!(std::isalpha(static_cast<unsigned char>(body[i])) || body[i] == '_') ||
        (i > 0 && (is_ident(body[i - 1]) || body[i - 1] == '.'))) {
      ++i;
      continue;
    }
    size_t name_end = i;
    while (name_end < size && is_ident(body[name_end])) {
      ++name_end;
    }
    const size_t start = i;
    i = name_end;
    const std::string lhs_name = body.substr(start, name_end - start);
    const bool output_only = half_names.count(lhs_name) == 0;
    if (output_only && half_outputs.count(lhs_name) == 0) {
      continue;
    }
    size_t before = start;
    while (before > 0 && is_space(body[before - 1])) {
      --before;
    }
    if (before > 0 && std::strchr(";{})", body[before - 1]) == nullptr) {
      continue;
    }
    size_t cursor = name_end;
    while (cursor < size && body[cursor] == '[') {
      int depth = 0;
      size_t close = cursor;
      for (; close < size; ++close) {
        depth += body[close] == '[' ? 1 : body[close] == ']' ? -1 : 0;
        if (depth == 0) {
          break;
        }
      }
      if (close >= size) {
        break;
      }
      cursor = close + 1;
    }
    const size_t lhs_end = cursor;
    while (cursor < size && is_space(body[cursor])) {
      ++cursor;
    }
    if (cursor >= size) {
      break;
    }
    char op = 0;
    size_t rhs_begin = 0;
    if (body[cursor] == '=' && (cursor + 1 >= size || body[cursor + 1] != '=')) {
      if (output_only) {
        continue;
      }
      rhs_begin = cursor + 1;
    } else if (
        std::strchr("+-*/", body[cursor]) != nullptr && cursor + 1 < size &&
        body[cursor + 1] == '=' && (cursor + 2 >= size || body[cursor + 2] != '=')) {
      // The compound form repeats the left-hand side inside its own
      // expansion, so an index that has side effects or calls something
      // (`h[i++] += x`, `h[f(i)] += x`) must not be duplicated: it is left
      // as written, as before the widening commit. A plain `=` names the
      // left-hand side once and needs no such care.
      {
        const std::string lhs_text = body.substr(start, lhs_end - start);
        if (lhs_text.find("++") != std::string::npos ||
            lhs_text.find("--") != std::string::npos ||
            lhs_text.find('(') != std::string::npos ||
            lhs_text.find('=') != std::string::npos) {
          continue;
        }
      }
      op = body[cursor];
      rhs_begin = cursor + 2;
    } else {
      continue;
    }
    int depth = 0;
    size_t end = rhs_begin;
    bool found = false;
    for (; end < size; ++end) {
      const char c = body[end];
      if (c == '(' || c == '[') {
        ++depth;
      } else if (c == ')' || c == ']') {
        if (--depth < 0) {
          break;
        }
      } else if (c == '{' || c == '}') {
        break;
      } else if (c == ';' && depth == 0) {
        found = true;
        break;
      }
    }
    if (!found) {
      continue;
    }
    size_t rb = rhs_begin;
    while (rb < end && is_space(body[rb])) {
      ++rb;
    }
    size_t re = end;
    while (re > rb && is_space(body[re - 1])) {
      --re;
    }
    const std::string rhs = body.substr(rb, re - rb);
    if (rhs.empty()) {
      continue;
    }
    if (op == 0 && rhs.rfind("float16_t(", 0) == 0) {
      int d = 0;
      size_t k = std::string("float16_t").size();
      for (; k < rhs.size(); ++k) {
        d += rhs[k] == '(' ? 1 : rhs[k] == ')' ? -1 : 0;
        if (d == 0) {
          break;
        }
      }
      if (k == rhs.size() - 1) {
        i = end + 1;
        continue;
      }
    }
    const std::string lhs = body.substr(start, lhs_end - start);
    out.append(body, copied, start - copied);
    out += lhs + " = float16_t(";
    if (op != 0) {
      out += lhs + " " + op + " (" + rhs + ")";
    } else {
      out += rhs;
    }
    out += ");";
    copied = end + 1;
    i = end + 1;
  }
  out.append(body, copied, std::string::npos);
  body = std::move(out);
}

// A float16_t helper parameter takes a float16_t value; GLSL widens half to
// float implicitly and never the other way, so an argument that is an operator
// expression (`f(bc, x, w[j] + w[j + 1])`, a float once its reads are widened)
// is wrapped in float16_t(). Only the helper's declared signature decides
// which positions are half: a float parameter is never narrowed, and a plain
// value (identifier, element read, call, literal) is left exactly as written,
// so kernels that compiled keep their text.
void wrap_half_call_arguments(std::string& body, const std::string& header) {
  static const std::regex definition(
      R"(\b[A-Za-z_][A-Za-z0-9_]*\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(([^()]*)\)\s*\{)");
  std::unordered_map<std::string, std::vector<bool>> half_positions;
  for (std::sregex_iterator it(header.begin(), header.end(), definition), stop;
       it != stop;
       ++it) {
    std::vector<bool> positions;
    bool any_half = false;
    const std::string params = (*it)[2].str();
    size_t cursor = 0;
    while (cursor <= params.size()) {
      size_t comma = params.find(',', cursor);
      if (comma == std::string::npos) {
        comma = params.size();
      }
      std::string one = params.substr(cursor, comma - cursor);
      static const std::regex half_scalar(
          R"(^\s*(?:const\s+)?(?:float16_t|half)\s+[A-Za-z_][A-Za-z0-9_]*\s*$)");
      const bool is_half = std::regex_match(one, half_scalar);
      positions.push_back(is_half);
      any_half = any_half || is_half;
      cursor = comma + 1;
    }
    if (any_half) {
      half_positions[(*it)[1].str()] = std::move(positions);
    }
  }
  if (half_positions.empty()) {
    return;
  }
  auto is_ident = [](char c) {
    return std::isalnum(static_cast<unsigned char>(c)) || c == '_';
  };
  auto has_operator = [](const std::string& arg) {
    int depth = 0;
    for (size_t i = 0; i < arg.size(); ++i) {
      const char c = arg[i];
      if (c == '(' || c == '[') {
        ++depth;
      } else if (c == ')' || c == ']') {
        --depth;
      } else if (depth == 0) {
        if (c == '+' || c == '*' || c == '/' || c == '%' || c == '?') {
          return true;
        }
        if (c == '-' &&
            arg.find_first_not_of(" \t\n", 0) != i) {
          return true;
        }
      }
    }
    return false;
  };
  size_t i = 0;
  while (i < body.size()) {
    if (!(std::isalpha(static_cast<unsigned char>(body[i])) || body[i] == '_') ||
        (i > 0 && (is_ident(body[i - 1]) || body[i - 1] == '.'))) {
      ++i;
      continue;
    }
    size_t name_end = i;
    while (name_end < body.size() && is_ident(body[name_end])) {
      ++name_end;
    }
    const std::string name = body.substr(i, name_end - i);
    const auto helper = half_positions.find(name);
    if (helper == half_positions.end() || name_end >= body.size() ||
        body[name_end] != '(') {
      i = name_end;
      continue;
    }
    size_t close = name_end;
    int depth = 0;
    for (; close < body.size(); ++close) {
      depth += body[close] == '(' ? 1 : body[close] == ')' ? -1 : 0;
      if (depth == 0) {
        break;
      }
    }
    if (close >= body.size()) {
      break;
    }
    std::string rebuilt = "(";
    size_t arg_begin = name_end + 1;
    size_t position = 0;
    int arg_depth = 0;
    bool changed = false;
    for (size_t k = arg_begin; k <= close; ++k) {
      const char c = body[k];
      if (c == '(' || c == '[') {
        ++arg_depth;
      } else if ((c == ')' || c == ']') && k != close) {
        --arg_depth;
      }
      if ((c == ',' && arg_depth == 0) || k == close) {
        std::string arg = body.substr(arg_begin, k - arg_begin);
        const bool half_slot =
            position < helper->second.size() && helper->second[position];
        const bool already =
            arg.find("float16_t(") != std::string::npos &&
            arg.find("float16_t(") == arg.find_first_not_of(" \t\n");
        if (half_slot && !already && has_operator(arg)) {
          const size_t lead = arg.find_first_not_of(" \t\n");
          const size_t tail = arg.find_last_not_of(" \t\n");
          arg = arg.substr(0, lead) + "float16_t(" +
              arg.substr(lead, tail - lead + 1) + ")" + arg.substr(tail + 1);
          changed = true;
        }
        rebuilt += arg;
        rebuilt += k == close ? ")" : ",";
        arg_begin = k + 1;
        ++position;
      }
    }
    if (changed) {
      body.replace(name_end, close - name_end + 1, rebuilt);
      i = name_end + rebuilt.size();
    } else {
      i = name_end + 1;
    }
  }
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
    // The read pass rewrote in-place stores into a bf16 INPUT buffer
    // (state-cache updates, e.g. Qwen3.5 GDN recurrent_state) into
    // '_mlx_bf16_to_float(slot) = rhs;' — an assignment to a call.
    // Narrow instead: slot = _mlx_float_to_bf16(rhs), which is exactly
    // the round-to-nearest-even bf16 store the MSL performed.
    body = std::regex_replace(
        body,
        std::regex(
            R"(_mlx_bf16_to_float\()" + regex_escape(storage) +
            R"(\[([^\]]*)\]\)\s*=\s*([^;]+);)"),
        storage + "[$1] = _mlx_float_to_bf16($2);");
    if (std::regex_search(body, std::regex("\\b" + escaped + R"(\s*\[)"))) {
      throw std::runtime_error("unsupported bfloat16 buffer expression");
    }
    // Bare (non-indexed) uses — float(NAME) of a whole bf16 buffer — have
    // no GLSL meaning either; the scalar form (&) is the supported route.
    // Swizzle positions do not count: `.x` on a uint3 vector is not the
    // buffer, and the inkling sconv decode kernel names its bf16 buffer `x`
    // while indexing `thread_position_in_grid.x` (2026-10-08 KernelRecheck).
    if (std::regex_search(
            body, std::regex("(^|[^.\\w])" + escaped + "(?!\\w)"))) {
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

// numeric_limits<T>::infinity()/max()/lowest()/min()/epsilon() have no GLSL
// form; the IEEE bit patterns and C literals are exact. The mlx-vlm
// llguidance mask kernel masks with -infinity() (2026-10-08 KernelRecheck).
// The whole numeric_limits<...>::fn() expression must match as one — the
// bare words are ordinary calls (metal::max) and must survive untouched.
// Runs before the INFINITY/NAN defines are emitted, on the body AND on
// header helpers (the GDN kernels' omlx_log1p helper tests against
// numeric_limits<float>::max(); body-only mapping left `numeric_limits`
// in the emitted helper and glslang died at the helper line, 2026-10-09).
void map_numeric_limits(std::string& code) {
  static const std::regex limits(
      R"(numeric_limits\s*<\s*[A-Za-z_][A-Za-z0-9_]*\s*>\s*::\s*(infinity|lowest|max|min|epsilon)\s*\(\s*\))");
  static const std::unordered_map<std::string, std::string> limit_values = {
      {"infinity", "INFINITY"},
      {"lowest", "(-3.4028234663852886e+38f)"},
      {"max", "3.4028234663852886e+38f"},
      {"min", "1.1754943508222875e-38f"},
      {"epsilon", "1.1920928955078125e-07f"},
  };
  std::string rewritten;
  rewritten.reserve(code.size());
  size_t last = 0;
  for (std::sregex_iterator it(code.begin(), code.end(), limits), end;
       it != end; ++it) {
    const auto& match = *it;
    rewritten += code.substr(last, match.position() - last);
    rewritten += limit_values.at((*it)[1].str());
    last = match.position() + match.length();
  }
  rewritten += code.substr(last);
  code = std::move(rewritten);
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
  // Strip MSL comments (// line and /* block */). Comments can contain
  // Metal keywords ('// One threadgroup per row' in the Qwen3.5 MoE
  // router kernel) that would trigger the final guard's substring check
  // even though they are not code. Stripping before all passes also
  // prevents the type/token replacements from corrupting comment text.
  {
    size_t search = 0;
    while (true) {
      const auto line_start = body.find("//", search);
      if (line_start == std::string::npos) break;
      const auto line_end = body.find('\n', line_start);
      const auto count = (line_end == std::string::npos)
                             ? body.size() - line_start
                             : line_end - line_start;
      body.erase(line_start, count);
      search = line_start;
    }
    search = 0;
    while (true) {
      const auto block_start = body.find("/*", search);
      if (block_start == std::string::npos) break;
      const auto block_end = body.find("*/", block_start + 2);
      if (block_end == std::string::npos) break;
      body.erase(block_start, block_end - block_start + 2);
      search = block_start;
    }
  }
  specialize_threadgroup_helper_params(header, body);
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
  map_numeric_limits(body);
  replace_all(body, "threadgroup_barrier(mem_flags::mem_threadgroup)", "barrier()" );
  replace_all(body, "threadgroup_barrier(mem_flags::mem_device)", "barrier()");
  replace_all(body, "simd_sum", "subgroupAdd");
  replace_all(body, "simd_max", "subgroupMax");
  // MSL metal::precise::rsqrt survives the metal:: strip as rsqrt; GLSL
  // names it inversesqrt. (Qwen3.5 GDN norm-gate, OmlxLinux M2 repro.)
  replace_word(body, "rsqrt", "inversesqrt");
  // MSL rint() rounds half-to-even in the current direction; GLSL
  // roundEven() is the same function. Needed by the H3 _QUANTIZE kernel
  // (int8 rounding of activations) and any quantize-style kernel.
  replace_word(body, "rint", "roundEven");
  // Metal's fabs() is C-named; GLSL only has abs() (inkling_moe_route's
  // softplus stage, 2026-10-08 KernelRecheck).
  replace_word(body, "fabs", "abs");
  body = std::regex_replace(
      body,
      std::regex(R"(\bfloat16_t\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([^;]+);)"),
      "float16_t $1 = float16_t($2);");
  replace_all(body, "simd_min", "subgroupMin");
  replace_all(body, "simd_broadcast", "subgroupBroadcast");
  replace_all(body, "simd_shuffle_down", "subgroupShuffleDown");
  replace_all(body, "simd_shuffle_up", "subgroupShuffleUp");
  replace_all(body, "simd_shuffle_xor", "subgroupShuffleXor");
  replace_all(body, "simd_shuffle", "subgroupShuffle");
  replace_word(body, "constexpr", "const");
  translate_types(body, parameters);
  translate_c_style_casts(body);
  // MSL `auto` locals have no GLSL form: `auto` is a reserved word, and
  // glslang fails the declaration with 'syntax error, unexpected
  // IDENTIFIER, expecting COMMA or SEMICOLON' (the Qwen3.5 GDN decode
  // prework's `const auto sy = 1 / (1 + exp(abs(conv)))`, serve failure
  // 2026-10-09, bf16 q_out [1,1,16,128]; nothing in the tree ever mapped
  // the construct — v0.7.31 included). Value autos become float;
  // pointer-style autos rewrite to the explicit device-pointer alias form,
  // so this must run before the alias pass below consumes those forms.
  translate_auto_declarations(body, parameters);
  // Pointer-advance walks (`kptr += step;`) become index variables; this
  // must also run before the alias pass, which refuses bare alias uses.
  translate_pointer_advance(body, parameters);
  std::string vector_alias_helpers;
  translate_device_pointer_aliases(body, parameters, vector_alias_helpers);

  // MSL implicitly narrows/widens in scalar declarations; GLSL 460
  // rejects implicit conversions. Wrap const scalar-declaration
  // initializers in an explicit constructor of the declared type:
  //  * const int from uint operands (H3 _QUANTIZE `const int first =
  //    g * GROUP + thread_position_in_threadgroup.x * PER` — int() is a
  //    no-op on int expressions, the same modulo-wrap as MSL on uint);
  //  * const uint16_t/int16_t from float/uint operands (the bf16
  //    pack/unpack idiom in the Qwen3.5-2B GDN kernel, OmlxLinux M2
  //    repro 2026-10-05: float -> const uint16_t).
  {
    static const std::regex const_decl(
        R"(\b(const\s+(int|uint|int16_t|uint16_t)\s+[A-Za-z_][A-Za-z0-9_]*\s*=\s*)([^;{}]+)(;))");
    std::string rewritten;
    rewritten.reserve(body.size());
    size_t last = 0;
    for (std::sregex_iterator it(body.begin(), body.end(), const_decl), end;
         it != end; ++it) {
      const auto& m = *it;
      rewritten += body.substr(last, m.position() - last);
      // Multi-declarator lines (`const int K = mp[0], KHI = mp[1];`,
      // moe_route_fused) must NOT fold into one constructor call — GLSL
      // keeps comma declarators legal as written.
      if (m[3].str().find(',') != std::string::npos) {
        rewritten += m[0].str();
      } else {
        rewritten += m[1].str() + m[2].str() + "(" + m[3].str() + ")" +
            m[4].str();
      }
      last = m.position() + m.length();
    }
    rewritten += body.substr(last);
    body = std::move(rewritten);
  }

  // MSL allows any integer expression as a condition (`if (flag)`); GLSL
  // requires a bool. Wrap the narrow forms — a bare identifier, an indexed
  // element, or a zero-argument call — in _mlx_nonzero(...), whose
  // overloads accept bool (identity), int/uint (!= 0) and float (!= 0.0f):
  // a plain `!= 0` would break bool conditions ('bool' != 'int' has no
  // overload in GLSL — the templates/bfloat smoke case hit this).
  // Compound conditions (comparisons, && / ||, !) do not match this
  // pattern and need no wrap.
  std::string condition_helpers;
  static const std::regex integer_condition(
      R"(\b(if|while)\s*\(\s*(([A-Za-z_][A-Za-z0-9_]*|[0-9]+u?)(\s*\[[^\[\]]*\])?(\(\))?)\s*\))");
  if (std::regex_search(body, integer_condition)) {
    condition_helpers =
        "bool _mlx_nonzero(bool v) { return v; }\n"
        "bool _mlx_nonzero(int v) { return v != 0; }\n"
        "bool _mlx_nonzero(uint v) { return v != 0u; }\n"
        "bool _mlx_nonzero(float v) { return v != 0.0f; }\n";
    body = std::regex_replace(body, integer_condition,
                              "$1 (_mlx_nonzero($2))");
  }

  // MSL converts integers to bool inside `&&` chains (the llguidance mask:
  // `word < S && ((bits >> bit) & 1u)`); GLSL requires bool operands. Split
  // the top-level `&&` of a bool declaration and route the operands through
  // an overloaded and-helper (identity on bool, != 0 on integers/floats).
  std::string bool_and_helpers;
  {
    static const std::regex bool_decl(
        R"(\bbool\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([^;]+);)");
    std::string rewritten;
    rewritten.reserve(body.size());
    size_t last = 0;
    bool any_split = false;
    auto split_top_level = [](const std::string& init) {
      std::vector<std::string> parts;
      int depth = 0;
      size_t start = 0;
      for (size_t i = 0; i < init.size(); ++i) {
        char c = init[i];
        if (c == '(' || c == '[' || c == '{') {
          ++depth;
        } else if (c == ')' || c == ']' || c == '}') {
          --depth;
        } else if (depth == 0 && c == '&' && i + 1 < init.size() &&
                   init[i + 1] == '&') {
          parts.push_back(init.substr(start, i - start));
          start = i + 2;
          ++i;
        }
      }
      parts.push_back(init.substr(start));
      return parts;
    };
    for (std::sregex_iterator it(body.begin(), body.end(), bool_decl), end;
         it != end; ++it) {
      const auto& m = *it;
      rewritten += body.substr(last, m.position() - last);
      const std::string init = trim(m[2].str());
      auto parts = split_top_level(init);
      if (parts.size() > 1) {
        any_split = true;
        std::string expr = parts[0];
        for (size_t k = 1; k < parts.size(); ++k) {
          expr = "_mlx_bool_and(" + expr + ", " + parts[k] + ")";
        }
        rewritten += "bool " + m[1].str() + " = " + expr + ";";
      } else {
        rewritten += m[0].str();
      }
      last = m.position() + m.length();
    }
    rewritten += body.substr(last);
    body = std::move(rewritten);
    if (any_split) {
      bool_and_helpers =
          "bool _mlx_bool_and(bool a, bool b) { return a && b; }\n"
          "bool _mlx_bool_and(bool a, uint b) { return a && b != 0u; }\n"
          "bool _mlx_bool_and(uint a, bool b) { return a != 0u && b; }\n"
          "bool _mlx_bool_and(uint a, uint b) { return a != 0u && b != 0u; }\n"
          "bool _mlx_bool_and(bool a, int b) { return a && b != 0; }\n"
          "bool _mlx_bool_and(int a, bool b) { return a != 0 && b; }\n"
          "bool _mlx_bool_and(int a, int b) { return a != 0 && b != 0; }\n"
          "bool _mlx_bool_and(uint a, int b) { return a != 0u && b != 0; }\n"
          "bool _mlx_bool_and(int a, uint b) { return a != 0 && b != 0u; }\n"
          "bool _mlx_bool_and(bool a, float b) { return a && b != 0.0f; }\n"
          "bool _mlx_bool_and(float a, bool b) { return a != 0.0f && b; }\n"
          "bool _mlx_bool_and(float a, float b) { return a != 0.0f && b != 0.0f; }\n";
    }
  }

  // MSL array initializers may under-supply elements (`float sum[4] =
  // {0.0};` zero-fills in C); GLSL requires the exact element count. Expand
  // the single-constant form; complete lists pass through unchanged
  // (2026-10-08 KernelRecheck, bitlinear_matmul). The extent may be a
  // constant-int identifier (`float o[v_per_thread] = {0};`, mlx_vlm
  // qwen3_5 ragged SDPA) whose foldable value is read from the body's
  // `const int NAME = <literal arithmetic>;` declarations.
  {
    const auto const_ints = fold_const_ints(body);
    static const std::regex array_init(
        R"(\b([A-Za-z_][A-Za-z0-9_]*)\s+([A-Za-z_][A-Za-z0-9_]*)\s*\[\s*([A-Za-z_][A-Za-z0-9_]*|[0-9]+)\s*\]\s*=\s*\{\s*([^{}]*?)\s*\})");
    std::string rewritten;
    rewritten.reserve(body.size());
    size_t last = 0;
    for (std::sregex_iterator it(body.begin(), body.end(), array_init), end;
         it != end; ++it) {
      const auto& m = *it;
      rewritten += body.substr(last, m.position() - last);
      int size = std::atoi(m[3].str().c_str());
      if (size == 0) {
        const auto folded = const_ints.find(m[3].str());
        if (folded == const_ints.end()) {
          throw std::runtime_error(
              "unsupported MSL feature `array initializer with an "
              "unresolved constant extent `" + m[3].str() + "`");
        }
        size = folded->second;
      }
      const std::string extent = m[3].str();
      const std::string values = trim(m[4].str());
      const bool single = values.find(',') == std::string::npos;
      if (size > 1 && single) {
        rewritten += m[1].str() + " " + m[2].str() + "[" + extent + "] = {";
        for (int i = 0; i < size; ++i) {
          if (i) {
            rewritten += ", ";
          }
          rewritten += values;
        }
        rewritten += "}";
      } else {
        rewritten += m[0].str();
      }
      last = m.position() + m.length();
    }
    rewritten += body.substr(last);
    body = std::move(rewritten);
  }

  bool needs_bfloat = false;

  // _Pragma("clang loop unroll(full)") and friends are optimization hints
  // with no GLSL equivalent; dropping them keeps the loop semantics.
  body = std::regex_replace(body, std::regex(R"(_Pragma\s*\([^()]*\))"), "");
  replace_all(header, "_Pragma", "_MLX_NO_PRAGMA");
  header = std::regex_replace(header, std::regex(R"(_MLX_NO_PRAGMA\s*\([^()]*\))"), "");

  // The `device` address space has no GLSL meaning; buffer parameters are
  // already macro aliases for SSBO data, so the keyword is redundant.
  replace_word(body, "device", "");
  replace_word(header, "device", "");

  // Bare `thread` locals (e.g. `thread float q_frag[4];`) are ordinary
  // GLSL locals; the address-space qualifier has no counterpart. This
  // runs after the threadgroup/shared pass above, and the word boundary
  // keeps `threadgroup` and the translated `thread_*` attribute macros
  // (which no longer contain a bare `thread` token) untouched.
  replace_word(body, "thread", "");
  replace_word(header, "thread", "");

  // `fast::` is the Metal stdlib namespace some JIT kernels use
  // (`fast::exp`, `fast::exp2`); GLSL builtins carry no namespace, so
  // the qualifier just disappears. The `metal::` strips above have
  // already removed the qualified forms.
  replace_all(body, "fast::", "");
  replace_all(header, "fast::", "");

  // `const T* name = BUFFER;` re-declares a buffer base under another name.
  // Drop the statement: the parameter macro already provides that alias.
  body = std::regex_replace(
      body,
      std::regex(
          R"(const\s+[A-Za-z_][A-Za-z0-9_]*\s*\*\s*[A-Za-z_][A-Za-z0-9_]*\s*=\s*([A-Za-z_][A-Za-z0-9_]*)\s*;)"),
      "");

  // Explicit bfloat(expr) casts round through bfloat16; keep the value in the
  // float domain with a round-trip helper so later float math stays exact.
  if (body.find("bfloat") != std::string::npos) {
    body = std::regex_replace(
        body, std::regex(R"(\bbfloat\s*\()"), "_mlx_bf16_round(");
    replace_word(body, "bfloat", "float");
    needs_bfloat = true;
  }

  // Any surviving pointer declaration has no GLSL translation; fail by
  // name rather than emit a broken shader. (size_t is mapped to uint in
  // translate_types, so it no longer trips this guard — the oMLX GDN
  // chunk kernels use size_t for scalar offsets throughout.)
  if (std::regex_search(
          body,
          std::regex(R"(const\s+[A-Za-z_][A-Za-z0-9_]*\s*\*)"))) {
    throw std::runtime_error(
        "unsupported MSL feature `device pointer arithmetic` is not "
        "implemented for the Omarchy Vulkan backend");
  }

  std::string shared_declarations;
  const std::regex shared_pattern(
      R"(threadgroup\s+([A-Za-z_][A-Za-z0-9_]*)\s+([A-Za-z_][A-Za-z0-9_]*)\s*\[\s*([^\[\];]+)\s*\]\s*;)");
  std::smatch shared_match;
  // Extent identifiers must be visible at file scope: the shared arrays
  // hoist above main(), so `shared float outputs[BN * BD];` would leave BN
  // undeclared if BN stayed a local const (mlx_vlm qwen3_5 ragged SDPA,
  // Qwen3.6-35B-A3B prefill 2026-10-09). Any extent identifier with a
  // foldable `const int` declaration in the body hoists with the array.
  std::string hoisted_consts;
  {
    const auto const_ints = fold_const_ints(body);
    const std::regex identifier("[A-Za-z_][A-Za-z0-9_]*");
    std::unordered_set<std::string> needed;
    for (std::sregex_iterator it(body.begin(), body.end(), shared_pattern),
             end;
         it != end;
         ++it) {
      for (std::sregex_iterator id((*it)[3].first, (*it)[3].second, identifier),
               id_end;
           id != id_end;
           ++id) {
        if (const_ints.count(id->str())) {
          needed.insert(id->str());
        }
      }
    }
    for (const auto& name : needed) {
      const auto entry = const_ints.find(name);
      const std::regex decl("\\bconst\\s+int\\s+" + regex_escape(name) +
                            R"(\s*=\s*[^;{}]+;\s*)");
      std::smatch dm;
      if (std::regex_search(body, dm, decl)) {
        hoisted_consts +=
            "const int " + name + " = " + std::to_string(entry->second) + ";\n";
        body.replace(dm.position(), dm.length(), "");
      }
    }
  }
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

  bool needs_int8 = false;
  bool needs_int16 = false;
  bool needs_int64 = false;
  bool needs_subgroup = body.find("subgroup") != std::string::npos ||
      body.find("gl_Subgroup") != std::string::npos ||
      body.find("gl_NumSubgroups") != std::string::npos;
  std::string declarations;
  std::string macros;
  // MSL permits implicit uint->int in scalar declarations (`int d = elem
  // % D;` with uint elem, as in the MiniMax M3 K2 combine); GLSL rejects
  // the mixed-sign assignment outright. Wrap every integer scalar
  // declaration's initializer in the matching constructor cast: it is a
  // no-op when the expression already has the declared type, so the pass
  // is safe on declarations that never needed it. Declarations with
  // multiple comma declarators (`int a = 1, b = 2;`) are left alone —
  // none of the supported kernels use them and wrapping the whole
  // initializer would fold two declarators into one constructor call.
  {
    const std::regex int_decl(
        R"(\b(int|uint)\s+([A-Za-z_]\w*)\s*=\s*([^;{}]+);)");
    std::string rewritten;
    auto begin = std::sregex_iterator(body.begin(), body.end(), int_decl);
    auto end = std::sregex_iterator();
    if (begin != end) {
      size_t last = 0;
      for (auto it = begin; it != end; ++it) {
        const auto& m = *it;
        const std::string rhs = m[3].str();
        rewritten += body.substr(last, m.position() - last);
        if (rhs.find(',') == std::string::npos) {
          rewritten += m[1].str() + " " + m[2].str() + " = " + m[1].str() +
              "(" + rhs + ");";
        } else {
          rewritten += m[0].str();
        }
        last = m.position() + m.length();
      }
      rewritten += body.substr(last);
      body = std::move(rewritten);
    }
  }
  for (size_t index = 0; index < parameters.size(); ++index) {
    auto parameter = parameters[index];
    const bool output = index >= output_start;
    declarations += buffer_declaration(parameter, output);
    if (parameter.type == "bfloat16_t") {
      needs_bfloat = true;
      translate_bfloat_parameter(body, parameter, output);
    } else if (parameter.atomic) {
      translate_atomic_parameter(body, parameter);
    } else {
      // float16_t buffer reads widen to float exactly (Metal promotes half
      // arithmetic to fp32); mixed int/f16 expressions then compile as
      // int/float (2026-10-08 KernelRecheck, bitlinear_matmul).
      if (parameter.type == "float16_t" && !output) {
        const auto escaped16 = regex_escape(parameter.name);
        const std::string storage =
            "_b" + std::to_string(parameter.binding) + ".data[";
        const std::regex read_pattern(
            "\\b" + escaped16 + R"(\s*\[([^\]]+)\])");
        std::string rewritten;
        size_t last = 0;
        for (std::sregex_iterator it(body.begin(), body.end(), read_pattern), stop;
             it != stop;
             ++it) {
          const size_t begin = static_cast<size_t>(it->position(0));
          const size_t end = begin + static_cast<size_t>(it->length(0));
          const std::string element = storage + (*it)[1].str() + "]";
          rewritten.append(body, last, begin - last);
          rewritten += half_read_is_plain_value(body, begin, end)
              ? element
              : "float(" + element + ")";
          last = end;
        }
        rewritten.append(body, last, std::string::npos);
        body = std::move(rewritten);
      }
      // A small array bound in the constant space that the body never
      // indexes is used as a value: mlx-audio's phonon unpack divides
      // `in_features / 16` directly (2026-10-08 KernelRecheck). Mark it
      // scalar so the macro takes element 0; indexed small arrays keep the
      // array macro.
      if (parameter.constant_space && !parameter.scalar && !parameter.atomic &&
          !std::regex_search(
              body,
              std::regex("\\b" + regex_escape(parameter.name) + R"(\s*\[)"))) {
        parameter.scalar = true;
      }
      // Indexed stores into a non-bfloat buffer of a narrower int type
      // need the same explicit cast as outputs when the RHS is float.
      // Cover bfloat16_t INPUT parameters as well as outputs: kernels
      // store converted floats into bf16 state/KV caches passed as
      // inputs (Qwen3.5 GDN state update, [1,1,16,128] bf16, seen in
      // the OmlxLinux M2 repro 2026-10-05).
      if (output || parameter.type == "bfloat16_t") {
        const auto escaped = regex_escape(parameter.name);
        body = std::regex_replace(
            body,
            std::regex("\\b" + escaped + R"(\s*\[([^\]]+)\]\s*=\s*([^;]+);)"),
            parameter.name + "[$1] = " + glsl_type(parameter.type) + "($2);");
      }
      macros += "#define _mlx_arg" +
          std::to_string(parameter.binding) + " ";
      if (parameter.type == "float16_t" && parameter.scalar) {
        // half storage widened at the macro: int/f16 arithmetic compiles as
        // int/float (Metal promotes half math to fp32)
        macros += "float(_b" + std::to_string(parameter.binding) + ".data[0])";
      } else {
        macros += "_b" + std::to_string(parameter.binding) + ".data";
        if (parameter.scalar) {
          macros += "[0]";
        }
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
  // A float16_t local, shared array or output element receives float
  // arithmetic wherever a half read was widened: `acc += w[i];`,
  // `acc = acc + w[i];` and `out[t] += w[j];` must store through float16_t().
  // An operator expression passed to a float16_t helper parameter converts
  // the same way. Both run once after every parameter's reads are rewritten.
  {
    std::unordered_set<std::string> half_outputs;
    for (size_t index = output_start; index < parameters.size(); ++index) {
      if (parameters[index].type == "float16_t") {
        half_outputs.insert(parameters[index].name);
      }
    }
    wrap_half_lvalue_assignments(body, shared_declarations, half_outputs);
  }
  wrap_half_call_arguments(body, header);
  // The body can reference 16/64-bit types the parameters never name —
  // e.g. a local `const uint16_t m = ...` in the bf16 pack/unpack idiom
  // (Qwen3.5-2B GDN, OmlxLinux M2 repro 2026-10-05). Without the
  // extension the type is unknown and glslang fails with 'syntax error,
  // unexpected IDENTIFIER' on the declaration.
  needs_int8 = needs_int8 || body.find("int8_t") != std::string::npos ||
      body.find("uint8_t") != std::string::npos;
  needs_int16 = needs_int16 || body.find("int16_t") != std::string::npos ||
      body.find("uint16_t") != std::string::npos ||
      body.find("float16_t") != std::string::npos;
  needs_int64 = needs_int64 || body.find("int64_t") != std::string::npos ||
      body.find("uint64_t") != std::string::npos;
  // Rename parameter tokens in the body to the safe internal aliases the
  // macros above define. A parameter literally named `x` (or y/z/w/...)
  // must not survive as a preprocessor macro: any GLSL swizzle spelled
  // with the same letter (`gl_GlobalInvocationID.x`, `vec.y`) would
  // expand through the alias and fail to compile (measured on the M2
  // Honeykrisp ticket, 2026-10-04, test_three_float_scalars). The token
  // match refuses names directly preceded by a '.' (swizzle position) or
  // a word character. std::regex has no lookbehind, so the leading
  // character is captured and re-emitted.
  for (const auto& parameter : parameters) {
    if (parameter.atomic || parameter.type == "bfloat16_t") {
      // Those paths already rewrote every use to the storage name.
      continue;
    }
    const std::string alias =
        "_mlx_arg" + std::to_string(parameter.binding);
    body = std::regex_replace(
        body,
        std::regex("(^|[^.\\w])" + regex_escape(parameter.name) + "(?!\\w)"),
        "$1" + alias);
  }

  // GLSL reserves words MSL allows as identifiers (`bool in = ...` in the
  // mlx-serve residual+RMSNorm row walk; `out` likewise). By this point
  // every buffer parameter is aliased to _mlx_argN, so a surviving token is
  // a body-local identifier — rename it out of the reserved set.
  replace_word(body, "in", "_mlx_in");
  replace_word(body, "out", "_mlx_out");

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
  // MSL kernels use the C math-huge-value identifiers freely
  // (`-INFINITY` causal guards in the MiniMax M3 attention kernels,
  // `NAN` sentinels elsewhere). GLSL has neither identifier; the IEEE
  // bit patterns are exact and fold as constant expressions.
  if (body.find("INFINITY") != std::string::npos ||
      header.find("INFINITY") != std::string::npos) {
    glsl << "#define INFINITY uintBitsToFloat(0x7F800000u)\n";
  }
  if (body.find("NAN") != std::string::npos ||
      header.find("NAN") != std::string::npos) {
    glsl << "#define NAN uintBitsToFloat(0x7FC00000u)\n";
  }
  glsl << "layout(local_size_x=" << local_x << ", local_size_y=" << local_y
       << ", local_size_z=" << local_z << ") in;\n";
  glsl << declarations << hoisted_consts << shared_declarations;
  if (needs_bfloat) {
    glsl << "float _mlx_bf16_to_float(uint16_t value) { return uintBitsToFloat(uint(value) << 16); }\n"
         << "uint16_t _mlx_float_to_bf16(float value) { uint bits = floatBitsToUint(value); uint rounded = bits + 0x7fffu + ((bits >> 16) & 1u); return uint16_t(rounded >> 16); }\n"
         << "float _mlx_bf16_round(float value) { return _mlx_bf16_to_float(_mlx_float_to_bf16(value)); }\n"
         << "float _mlx_bf16_round_trip(float value) { return _mlx_bf16_round(value); }\n";
  }
  glsl << header << "\n" << element_helpers << vector_alias_helpers
       << condition_helpers << bool_and_helpers << macros;
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
constexpr char kTranslationCacheVersion[] = "2";

// Stamp of the translation pipeline itself, generated at build time from
// the SHA-256 of this source file (see omarchy_shader translation_version
// rule in CMakeLists.txt). Without it a rebuilt binary with changed
// translation logic would keep serving stale cached GLSL from an earlier
// binary: the identity of a kernel alone does not describe the translator
// that renders it. Observed live 2026-10-04: translator fixes stayed
// invisible until MLX_OMARCHY_SPIRV_CACHE=0 because old .tr entries
// matched by identity.
#ifndef MLX_OMARCHY_TRANSLATOR_SOURCE_SHA
#define MLX_OMARCHY_TRANSLATOR_SOURCE_SHA "unknown"
#endif
constexpr char kTranslatorSourceSha[] = MLX_OMARCHY_TRANSLATOR_SOURCE_SHA;

// Runtime identity of the loaded translator: the SHA-256 of the shared
// object that provides this code, plus the build-time source hash when the
// CMake generation produced one. Pip-built wheels may bake the fallback
// "unknown" (the CMake shader-generation target is not always on the pip
// compile's include path); the runtime library hash ensures that two
// different libmlx builds can NEVER share a translation-cache entry,
// regardless of what the build scripts injected. One-time cost: a single
// read of the .so (~50-400 MB, cached by the OS page cache).
const std::string& translator_runtime_identity() {
  static const std::string identity = [] {
    Dl_info info;
    std::string lib_hash = "no-library";
    if (dladdr((void*)&translator_runtime_identity, &info) &&
        info.dli_fname) {
      std::ifstream so(info.dli_fname, std::ios::binary);
      auto data = std::string(std::istreambuf_iterator<char>(so),
                              std::istreambuf_iterator<char>());
      if (!data.empty()) {
        lib_hash = omarchy::ane::sha256_hex(
            reinterpret_cast<const uint8_t*>(data.data()), data.size());
      }
    }
    return kTranslatorSourceSha + std::string(":") + lib_hash;
  }();
  return identity;
}

std::string translation_cache_material(const std::string& identity) {
  std::string material = "mlx-omarchy custom kernel translation ";
  material += kTranslationCacheVersion;
  material += " ";
  material += translator_runtime_identity();
  material += "\n";
  material += identity;
  return material;
}

std::string translation_cache_path(const std::string& identity) {
  const std::string root = spirv_cache_root();
  if (root.empty()) {
    return {};
  }
  std::string material = translation_cache_material(identity);
  return root + "/" +
      omarchy::ane::sha256_hex(
             reinterpret_cast<const uint8_t*>(material.data()),
             material.size()) +
      ".tr";
}

// Exposed for testing: the cache-key material must change when the
// translator identity changes, so two different libmlx builds
// can never share a .tr cache entry.
std::string translation_cache_material_for_test(
    const std::string& identity,
    const std::string& source_sha,
    const std::string& library_hash) {
  std::string material = "mlx-omarchy custom kernel translation ";
  material += kTranslationCacheVersion;
  material += " ";
  material += source_sha;
  material += ":";
  material += library_hash;
  material += "\n";
  material += identity;
  return material;
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

using omarchy::binding;

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

#ifdef MLX_OMARCHY_TEST_TRANSLATE
// Test-only entry: drive translate_msl on a raw MSL source and return
// the GLSL the translator would emit (or throw). The harness in
// harness/kernel_battery.py links against this when computing the
// per-kernel table for the parity matrix. Intended for dev-box
// classification only; the GPU dispatch row stays on the M2 lane.

std::string translation_cache_material_for_test(
    const std::string& identity,
    const std::string& source_sha,
    const std::string& library_hash) {
  std::string material = "mlx-omarchy custom kernel translation ";
  material += kTranslationCacheVersion;
  material += " ";
  material += source_sha;
  material += ":";
  material += library_hash;
  material += "\n";
  material += identity;
  return material;
}

std::string mlx_omarchy_translate_msl_for_test(
    const std::string& source,
    int grid_x,
    int grid_y,
    int grid_z,
    int threads_x,
    int threads_y,
    int threads_z,
    std::size_t output_count) {
  const auto translation = translate_msl(
      source,
      std::make_tuple(grid_x, grid_y, grid_z),
      std::make_tuple(threads_x, threads_y, threads_z),
      output_count,
      0);
  return translation.glsl;
}
#endif

} // namespace mlx::core::fast
