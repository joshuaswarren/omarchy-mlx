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

void translate_as_type(std::string& code);
void translate_c_style_casts(std::string& code);

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
  translate_as_type(code);
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
void translate_c_style_casts(std::string& code) {
  static const std::unordered_map<std::string, std::string> casts = {
      {"int8_t", "int8_t"}, {"uint8_t", "uint8_t"},
      {"int", "int"}, {"uint", "uint"},
      {"float", "float"}, {"bool", "bool"},
      {"int32_t", "int"}, {"uint32_t", "uint"},
      {"int64_t", "uint"}, {"uint64_t", "uint"},
      {"size_t", "uint"},
  };
  size_t search_from = 0;
  while (search_from < code.size()) {
    const auto open = code.find('(', search_from);
    if (open == std::string::npos) {
      return;
    }
    const auto close = code.find(')', open + 1);
    if (close == std::string::npos) {
      return;
    }
    const auto candidate = trim(code.substr(open + 1, close - open - 1));
    const auto mapped = casts.find(candidate);
    if (mapped == casts.end() || close + 1 >= code.size()) {
      search_from = open + 1;
      continue;
    }
    const auto next = code.find_first_not_of(" \t\r\n", close + 1);
    if (next == std::string::npos) {
      return;
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
      return;
    }
    const std::string argument = code.substr(next, argument_end - next);
    const std::string replacement =
        mapped->second + "(" + argument + ")";
    code.replace(open, argument_end - open, replacement);
    // Recompute search_from against the post-replace size; the old
    // `open + replacement.size()` arithmetic could land past `code.size()`
    // when argument_end < open (impossible here) and ties the next scan
    // position to the just-rewritten text rather than to the source.
    search_from = std::min(open + replacement.size(), code.size());
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
  // Group 1 = pointee type, group 2 = alias name, group 3 = initializer.
  static const std::regex alias_patterns[] = {
      std::regex(
          R"(const\s+device\s+([A-Za-z_][A-Za-z0-9_]*)\s*\*\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([^;]+);)"),
      std::regex(
          R"(device\s+const\s+([A-Za-z_][A-Za-z0-9_]*)\s*\*\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([^;]+);)"),
      std::regex(
          R"(device\s+([A-Za-z_][A-Za-z0-9_]*)\s*\*\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([^;]+);)"),
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
  while (progressed) {
    progressed = false;
    for (const auto& alias_pattern : alias_patterns) {
      for (std::sregex_iterator it(body.begin(), body.end(), alias_pattern),
               end;
           it != end; ++it) {
        const auto type = (*it)[1].str();
        const auto name = (*it)[2].str();
        if (aliases.count(name)) {
          continue;
        }
        std::string init = trim((*it)[3].str());
        // Strip a leading C-style device-pointer cast: `(const device T*)`.
        static const std::regex cast_prefix(
            R"(^\(\s*(?:const\s+)?device\s+(?:const\s+)?[A-Za-z_][A-Za-z0-9_]*\s*\*\s*\)\s*)");
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

void translate_as_type(std::string& code) {
  // as_type<Dest>(src) is a bitcast; src may contain nested parentheses, so
  // match the argument by balanced delimiters rather than a flat regex.
  static const std::unordered_map<std::string, std::string> bitcasts = {
      {"float", "uintBitsToFloat"},
      {"uint", "floatBitsToUint"},
      {"uint32_t", "floatBitsToUint"},
      {"int", "floatBitsToInt"},
      {"int32_t", "floatBitsToInt"},
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
    const auto mapped = bitcasts.find(destination);
    if (mapped == bitcasts.end()) {
      throw std::runtime_error("unsupported MSL feature `as_type<" + destination + ">`");
    }
    const auto open_paren = code.find('(', close_angle + 1);
    if (open_paren == std::string::npos) {
      return;
    }
    const auto close_paren = matching_delimiter(code, open_paren, '(', ')');
    code.replace(
        marker, close_paren - marker + 1,
        mapped->second + "(" + code.substr(open_paren + 1, close_paren - open_paren - 1) + ")");
    search_from = marker;
  }
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
    auto value = trim(value_parts[index]);
    if (declaration_parts[index].find("bool") != std::string::npos) {
      value = value == "0" ? "false" : "true";
    } else if (declaration_parts[index].find("typename") != std::string::npos) {
      // Metal promotes bfloat16_t LOCALS to fp32 for arithmetic; mirroring
      // that means a typename instantiated as any bf16 representation
      // substitutes float, not the uint16_t storage type. Buffer parameters
      // keep uint16_t storage (buffer_declaration uses glsl_type separately)
      // and narrow via _mlx_float_to_bf16 at stores. Without this, the
      // Qwen3.5-2B GDN decode leg emits 'uint16_t sy = uint16_t(1) /
      // (uint16_t(1) + exp(abs(conv)));' — integer math where Metal
      // computes in fp32.
      // Integer/bool template values (HK=16, L2=true) pass through as-is;
      // only type names (alphabetic first char) map through glsl_type.
      if (value == "bfloat16_t" || value == "bfloat16" || value == "uint16_t") {
        value = "float";
      } else if (value == "true" || value == "false" ||
                 std::isdigit(static_cast<unsigned char>(value.front()))) {
        // boolean and numeric template values pass through unchanged
      } else if (!value.empty() &&
                 std::isalpha(static_cast<unsigned char>(value.front()))) {
        value = glsl_type(value);
      }
    }
    replace_word(header, name, value);
    replace_word(body, name, value);
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
    if (std::regex_search(body, std::regex("\\b" + escaped + "\\b"))) {
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
  // MSL metal::precise::rsqrt survives the metal:: strip as rsqrt; GLSL
  // names it inversesqrt. (Qwen3.5 GDN norm-gate, OmlxLinux M2 repro.)
  replace_word(body, "rsqrt", "inversesqrt");
  // MSL rint() rounds half-to-even in the current direction; GLSL
  // roundEven() is the same function. Needed by the H3 _QUANTIZE kernel
  // (int8 rounding of activations) and any quantize-style kernel.
  replace_word(body, "rint", "roundEven");
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
  translate_types(body);
  translate_c_style_casts(body);
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
  body = std::regex_replace(
      body,
      std::regex(
          R"(\b(const\s+(int|uint|int16_t|uint16_t)\s+[A-Za-z_][A-Za-z0-9_]*\s*=\s*)([^;{}]+)(;))"),
      "$1$2($3)$4");

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
    const auto& parameter = parameters[index];
    const bool output = index >= output_start;
    declarations += buffer_declaration(parameter, output);
    if (parameter.type == "bfloat16_t") {
      needs_bfloat = true;
      translate_bfloat_parameter(body, parameter, output);
    } else if (parameter.atomic) {
      translate_atomic_parameter(body, parameter);
    } else {
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
          std::to_string(parameter.binding) + " _b" +
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
  glsl << declarations << shared_declarations;
  if (needs_bfloat) {
    glsl << "float _mlx_bf16_to_float(uint16_t value) { return uintBitsToFloat(uint(value) << 16); }\n"
         << "uint16_t _mlx_float_to_bf16(float value) { uint bits = floatBitsToUint(value); uint rounded = bits + 0x7fffu + ((bits >> 16) & 1u); return uint16_t(rounded >> 16); }\n"
         << "float _mlx_bf16_round(float value) { return _mlx_bf16_to_float(_mlx_float_to_bf16(value)); }\n";
  }
  glsl << header << "\n" << element_helpers << vector_alias_helpers << condition_helpers << macros;
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

std::string translation_cache_path(const std::string& identity) {
  const std::string root = spirv_cache_root();
  if (root.empty()) {
    return {};
  }
  std::string material = "mlx-omarchy custom kernel translation ";
  material += kTranslationCacheVersion;
  material += " ";
  material += kTranslatorSourceSha;
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

#ifdef MLX_OMARCHY_TEST_TRANSLATE
// Test-only entry: drive translate_msl on a raw MSL source and return
// the GLSL the translator would emit (or throw). The harness in
// harness/kernel_battery.py links against this when computing the
// per-kernel table for the parity matrix. Intended for dev-box
// classification only; the GPU dispatch row stays on the M2 lane.
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
