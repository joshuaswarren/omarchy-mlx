// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

#include "m2v_route.h"

#include <dirent.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <signal.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <string_view>
#include <unordered_map>

namespace mlx::core::omarchy::m2v {
namespace {

// ---------------------------------------------------------------------------
// SHA-256, self-contained so the cache key is testable without the ANE
// bundle's symbol table.

struct Sha256 {
  uint32_t state[8] = {0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
                       0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u};
  uint64_t bits = 0;
  uint8_t block[64] = {};
  size_t used = 0;

  void update(const uint8_t* data, size_t size) {
    bits += static_cast<uint64_t>(size) * 8;
    while (size > 0) {
      const size_t take = std::min(size, sizeof(block) - used);
      std::memcpy(block + used, data, take);
      used += take;
      data += take;
      size -= take;
      if (used == sizeof(block)) {
        compress();
        used = 0;
      }
    }
  }

  void compress() {
    static const uint32_t k[64] = {
        0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu,
        0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u, 0xd807aa98u, 0x12835b01u,
        0x243185beu, 0x550c7dc3u, 0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u,
        0xc19bf174u, 0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,
        0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau, 0x983e5152u,
        0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u,
        0x06ca6351u, 0x14292967u, 0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu,
        0x53380d13u, 0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
        0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u, 0xd192e819u,
        0xd6990624u, 0xf40e3585u, 0x106aa070u, 0x19a4c116u, 0x1e376c08u,
        0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu,
        0x682e6ff3u, 0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
        0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u};
    uint32_t w[64];
    for (int i = 0; i < 16; ++i) {
      w[i] = (uint32_t(block[i * 4]) << 24) | (uint32_t(block[i * 4 + 1]) << 16) |
          (uint32_t(block[i * 4 + 2]) << 8) | uint32_t(block[i * 4 + 3]);
    }
    for (int i = 16; i < 64; ++i) {
      const uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^
          (w[i - 15] >> 3);
      const uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^
          (w[i - 2] >> 10);
      w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    uint32_t a = state[0], b = state[1], c = state[2], d = state[3];
    uint32_t e = state[4], f = state[5], g = state[6], h = state[7];
    for (int i = 0; i < 64; ++i) {
      const uint32_t s1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
      const uint32_t ch = (e & f) ^ (~e & g);
      const uint32_t t1 = h + s1 + ch + k[i] + w[i];
      const uint32_t s0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
      const uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
      const uint32_t t2 = s0 + maj;
      h = g;
      g = f;
      f = e;
      e = d + t1;
      d = c;
      c = b;
      b = a;
      a = t1 + t2;
    }
    state[0] += a;
    state[1] += b;
    state[2] += c;
    state[3] += d;
    state[4] += e;
    state[5] += f;
    state[6] += g;
    state[7] += h;
  }

  static uint32_t rotr(uint32_t value, int amount) {
    return (value >> amount) | (value << (32 - amount));
  }

  std::string hex() {
    const uint64_t bit_count = bits;
    const uint8_t one = 0x80;
    update(&one, 1);
    const uint8_t zero = 0;
    while (used != 56) {
      update(&zero, 1);
    }
    uint8_t tail[8];
    for (int i = 0; i < 8; ++i) {
      tail[i] = static_cast<uint8_t>(bit_count >> (56 - 8 * i));
    }
    bits -= 64;  // the padding is not part of the message
    update(tail, 8);
    std::string out;
    out.reserve(64);
    char buf[9];
    for (int i = 0; i < 8; ++i) {
      std::snprintf(buf, sizeof(buf), "%08x", state[i]);
      out += buf;
    }
    return out;
  }
};

std::string sha256_hex(const std::string& data) {
  Sha256 hash;
  hash.update(reinterpret_cast<const uint8_t*>(data.data()), data.size());
  return hash.hex();
}

// ---------------------------------------------------------------------------
// A minimal JSON reader, enough for the gate table and the m2v-compile
// reflection documents. Throws std::runtime_error naming the problem.

struct Json {
  enum class Kind { Null, Bool, Number, String, Array, Object };
  Kind kind = Kind::Null;
  bool boolean = false;
  double number = 0;
  std::string text;
  std::vector<Json> array;
  std::map<std::string, Json> object;

  const Json* find(const std::string& key) const {
    if (kind != Kind::Object) {
      return nullptr;
    }
    const auto it = object.find(key);
    return it == object.end() ? nullptr : &it->second;
  }

  int as_int(const char* field) const {
    if (kind != Kind::Number) {
      throw std::runtime_error(
          std::string("metal2vk reflection field ") + field +
          " is not a number");
    }
    return static_cast<int>(number);
  }

  std::string as_string(const char* field) const {
    if (kind != Kind::String) {
      throw std::runtime_error(
          std::string("metal2vk reflection field ") + field +
          " is not a string");
    }
    return text;
  }
};

struct JsonParser {
  explicit JsonParser(const std::string& input) : text(input) {}

  Json parse() {
    Json value = parse_value();
    skip();
    if (position != text.size()) {
      fail("trailing characters after the JSON document");
    }
    return value;
  }

 private:
  const std::string& text;
  size_t position = 0;

  [[noreturn]] void fail(const std::string& why) {
    throw std::runtime_error(
        std::string("metal2vk JSON is malformed at byte ") +
        std::to_string(position) + ": " + why);
  }

  void skip() {
    while (position < text.size() &&
           (text[position] == ' ' || text[position] == '\t' ||
            text[position] == '\n' || text[position] == '\r')) {
      ++position;
    }
  }

  char peek() {
    skip();
    if (position >= text.size()) {
      fail("unexpected end of document");
    }
    return text[position];
  }

  void expect(char c) {
    if (peek() != c) {
      fail(std::string("expected '") + c + "'");
    }
    ++position;
  }

  bool consume(char c) {
    if (position < text.size() && text[position] == c) {
      ++position;
      return true;
    }
    return false;
  }

  Json parse_value() {
    const char c = peek();
    if (c == '{') {
      return parse_object();
    }
    if (c == '[') {
      return parse_array();
    }
    if (c == '"') {
      Json value;
      value.kind = Json::Kind::String;
      value.text = parse_string();
      return value;
    }
    if (consume('t')) {
      literal("rue");
      return {Json::Kind::Bool, true};
    }
    if (consume('f')) {
      literal("alse");
      return {Json::Kind::Bool, false};
    }
    if (consume('n')) {
      literal("ull");
      return {};
    }
    return parse_number();
  }

  void literal(const char* rest) {
    for (; *rest; ++rest) {
      if (position >= text.size() || text[position] != *rest) {
        fail("bad literal");
      }
      ++position;
    }
  }

  Json parse_object() {
    expect('{');
    Json value;
    value.kind = Json::Kind::Object;
    if (consume('}')) {
      return value;
    }
    while (true) {
      const std::string key = parse_string();
      expect(':');
      value.object[key] = parse_value();
      if (consume(',')) {
        continue;
      }
      expect('}');
      return value;
    }
  }

  Json parse_array() {
    expect('[');
    Json value;
    value.kind = Json::Kind::Array;
    if (consume(']')) {
      return value;
    }
    while (true) {
      value.array.push_back(parse_value());
      if (consume(',')) {
        continue;
      }
      expect(']');
      return value;
    }
  }

  std::string parse_string() {
    expect('"');
    std::string out;
    while (position < text.size() && text[position] != '"') {
      char c = text[position++];
      if (c == '\\') {
        if (position >= text.size()) {
          fail("unterminated escape");
        }
        c = text[position++];
        switch (c) {
          case '"': out.push_back('"'); break;
          case '\\': out.push_back('\\'); break;
          case '/': out.push_back('/'); break;
          case 'b': out.push_back('\b'); break;
          case 'f': out.push_back('\f'); break;
          case 'n': out.push_back('\n'); break;
          case 'r': out.push_back('\r'); break;
          case 't': out.push_back('\t'); break;
          case 'u': {
            if (position + 4 > text.size()) {
              fail("short unicode escape");
            }
            unsigned code = 0;
            for (int i = 0; i < 4; ++i) {
              const char h = text[position++];
              code <<= 4;
              if (h >= '0' && h <= '9') {
                code |= h - '0';
              } else if (h >= 'a' && h <= 'f') {
                code |= h - 'a' + 10;
              } else if (h >= 'A' && h <= 'F') {
                code |= h - 'A' + 10;
              } else {
                fail("bad unicode escape");
              }
            }
            // The documents here are ASCII; encode as UTF-8.
            if (code < 0x80) {
              out.push_back(static_cast<char>(code));
            } else if (code < 0x800) {
              out.push_back(static_cast<char>(0xC0 | (code >> 6)));
              out.push_back(static_cast<char>(0x80 | (code & 0x3F)));
            } else {
              out.push_back(static_cast<char>(0xE0 | (code >> 12)));
              out.push_back(
                  static_cast<char>(0x80 | ((code >> 6) & 0x3F)));
              out.push_back(static_cast<char>(0x80 | (code & 0x3F)));
            }
            break;
          }
          default: fail("unknown escape");
        }
      } else {
        out.push_back(c);
      }
    }
    expect('"');
    return out;
  }

  Json parse_number() {
    const size_t start = position;
    if (position < text.size() && text[position] == '-') {
      ++position;
    }
    while (position < text.size() &&
           ((text[position] >= '0' && text[position] <= '9') ||
            text[position] == '.' || text[position] == 'e' ||
            text[position] == 'E' || text[position] == '+' ||
            text[position] == '-')) {
      ++position;
    }
    if (position == start) {
      fail("expected a value");
    }
    Json value;
    value.kind = Json::Kind::Number;
    value.number = std::strtod(text.substr(start, position - start).c_str(),
                               nullptr);
    return value;
  }
};

Json parse_json(const std::string& text) {
  return JsonParser(text).parse();
}

// ---------------------------------------------------------------------------
// The shipped gate table. tools/fill_m2v_aot.py reads the JSON between the
// markers below to check that every kernel it compiles is routed here, so
// keep the markers intact.
//
// M2V_GATE_TABLE_BEGIN
const char kGateTableJson[] = R"json({
  "kernels": {
    "omlx_qwen35_moe_router_topk": {"state": "verified", "evidence": "parity sweep 2026-10-10: exact against the translator output"},
    "omlx_qwen35_moe_router_gemv": {"state": "verified", "evidence": "parity sweep 2026-10-10: matches the MSL reference"},
    "omlx_qwen35_moe_router_softmax_topk_row": {"state": "verified", "evidence": "parity sweep 2026-10-10: matches the MSL reference"},
    "omlx_qwen35_moe_router_softmax_topk_rows": {"state": "verified", "evidence": "parity sweep 2026-10-10: matches the MSL reference"},
    "omlx_qwen35_moe_combine_row": {"state": "verified", "evidence": "parity sweep 2026-10-10: matches the MSL reference"},
    "omlx_qwen35_moe_gate_up_decode_b4g32f_shared_b4g32f_gate_b4g32s": {"state": "verified", "evidence": "parity sweep 2026-10-10: matches the MSL reference within the accepted simd_sum rounding class"},
    "omlx_qwen35_moe_gate_up_window_b4g32f_shared_b4g32f_gate_b4g32s": {"state": "verified", "evidence": "parity sweep 2026-10-10: matches the MSL reference within the accepted simd_sum rounding class"},
    "omlx_qwen35_moe_gate_up_topk_b4g32f_shared_b4g32f_gate_b4g32s": {"state": "verified", "evidence": "parity sweep 2026-10-10: matches the MSL reference within the accepted simd_sum rounding class"},
    "omlx_qwen35_moe_down_combine_decode_b4g32f_shared_b4g32f": {"state": "verified", "evidence": "parity sweep 2026-10-10: bit exact against the fixed MSL reference"},
    "omlx_qwen35_moe_down_combine_window_b4g32f_shared_b4g32f": {"state": "verified", "evidence": "parity sweep 2026-10-10: bit exact against the fixed MSL reference"},
    "omlx_gdn_sigmoid_probe": {"state": "verified", "evidence": "parity sweep 2026-10-10: exact against the translator output"},
    "omlx_gdn_sigmoid_probe_float_float": {"state": "verified", "evidence": "parity sweep 2026-10-10: exact against the translator output"},
    "omlx_gdn_norm_gate_eps1em06": {"state": "verified", "evidence": "parity sweep 2026-10-10: matches the MSL reference"},
    "omlx_gdn_verify_main_replay": {"state": "verified", "evidence": "parity sweep 2026-10-10: matches the MSL reference"},
    "omlx_verify_attn_wide_combine": {"state": "verified", "evidence": "parity sweep 2026-10-10: exact against the translator output"},
    "omlx_verify_attn_gqa_combine": {"state": "verified", "evidence": "parity sweep 2026-10-10: exact against the translator output"},
    "omlx_chain_attn_partial": {"state": "unverified", "evidence": "the module was regenerated from the call contract dispatch assembles; it compiles and validates CPU-side, GPU parity of this exact module is owed, so auto does not route it"},
    "omlx_chain_attn_combine": {"state": "verified", "evidence": "parity sweep 2026-10-10: exact against the translator output"},
    "omlx_qwen35_gdn_prework_S1": {"state": "unverified", "evidence": "the module was regenerated from the call contract dispatch assembles; it compiles and validates CPU-side, GPU parity of this exact module is owed, so auto does not route it"},
    "omlx_qwen4_gdn_decode_prework": {"state": "unverified", "evidence": "the module was regenerated from the call contract dispatch assembles; it compiles and validates CPU-side, GPU parity of this exact module is owed, so auto does not route it"},
    "omlx_qwen4_gdn_decode_step": {"state": "unverified", "evidence": "the module was regenerated from the call contract dispatch assembles; it compiles and validates CPU-side, GPU parity of this exact module is owed, so auto does not route it"},
    "omlx_qwen4_gdn_batch_decode_step": {"state": "unverified", "evidence": "the module was regenerated from the call contract dispatch assembles; it compiles and validates CPU-side, GPU parity of this exact module is owed, so auto does not route it"},
    "omlx_qwen4_gdn_decode_norm_gate": {"state": "unverified", "evidence": "the module was regenerated from the call contract dispatch assembles; it compiles and validates CPU-side, GPU parity of this exact module is owed, so auto does not route it"},
    "omlx_qwen4_gdn_prefill_prework": {"state": "unverified", "evidence": "the module was regenerated from the call contract dispatch assembles; it compiles and validates CPU-side, GPU parity of this exact module is owed, so auto does not route it"},
    "omlx_qwen4_gdn_prefill_norm_gate": {"state": "unverified", "evidence": "the module was regenerated from the call contract dispatch assembles; it compiles and validates CPU-side, GPU parity of this exact module is owed, so auto does not route it"},
    "omlx_qwen4_gdn_verify_step": {"state": "unverified", "evidence": "the module was regenerated from the call contract dispatch assembles; it compiles and validates CPU-side, GPU parity of this exact module is owed, so auto does not route it"},
    "omlx_qwen4_gdn_verify_step_states": {"state": "unverified", "evidence": "the module was regenerated from the call contract dispatch assembles; it compiles and validates CPU-side, GPU parity of this exact module is owed, so auto does not route it"},
    "qwen35_gated_delta_step": {"state": "failed", "evidence": "the parity-rig call passes T as a (1,) array where the source reads a scalar; the assembled MSL does not compile (uint(T) on a constant pointer); production dispatches gated_delta_step instead"},
    "gated_delta_step": {"state": "unverified", "evidence": "the module was regenerated from the call contract dispatch assembles; it compiles and validates CPU-side, GPU parity of this exact module is owed, so auto does not route it"},
    "qwen3_5_ragged_sdpa_1p_bf16_d256_v256": {"state": "unverified", "evidence": "the module was regenerated from the call contract dispatch assembles; it compiles and validates CPU-side, GPU parity of this exact module is owed, so auto does not route it"},
    "qwen3_5_ragged_sdpa_2p1_bf16_d256_v256_b4": {"state": "unverified", "evidence": "production mlx_vlm call; awaiting the grid-corrected parity re-run, mirroring qwen35_ragged_sdpa_2p1"},
    "qwen3_5_ragged_sdpa_2p2_bf16_v256_b4": {"state": "unverified", "evidence": "the module was regenerated from the call contract dispatch assembles; it compiles and validates CPU-side, GPU parity of this exact module is owed, so auto does not route it"},
    "qwen35_ragged_sdpa_1p": {"state": "unverified", "evidence": "the module was regenerated from the call contract dispatch assembles; it compiles and validates CPU-side, GPU parity of this exact module is owed, so auto does not route it"},
    "qwen35_ragged_sdpa_2p1": {"state": "unverified", "evidence": "awaiting the grid-corrected parity re-run before acceptance"},
    "qwen35_ragged_sdpa_2p2": {"state": "unverified", "evidence": "the module was regenerated from the call contract dispatch assembles; it compiles and validates CPU-side, GPU parity of this exact module is owed, so auto does not route it"},
    "omlx_verify_attn_wide_partial": {"state": "failed", "evidence": "clspv emits an invalid module; the producer fix is in flight"},
    "omlx_verify_attn_gqa_partial": {"state": "failed", "evidence": "metal2vk refuses it by name (Metal 4 tensor ops)"}
  }
})json";
// M2V_GATE_TABLE_END

GateTable parse_gate_table(const std::string& json) {
  const Json document = parse_json(json);
  const Json* kernels = document.find("kernels");
  if (kernels == nullptr || kernels->kind != Json::Kind::Object) {
    throw std::runtime_error(
        "the metal2vk gate table needs a \"kernels\" object");
  }
  GateTable table;
  for (const auto& [name, entry] : kernels->object) {
    GateEntry gate;
    if (const Json* state = entry.find("state"); state != nullptr) {
      gate.state = state->text;
      if (gate.state != "verified" && gate.state != "unverified" &&
          gate.state != "failed") {
        throw std::runtime_error(
            "the metal2vk gate table entry for " + name +
            " has an unknown state: " + gate.state);
      }
    } else {
      throw std::runtime_error(
          "the metal2vk gate table entry for " + name + " has no state");
    }
    if (const Json* evidence = entry.find("evidence"); evidence != nullptr) {
      gate.evidence = evidence->text;
    }
    table.emplace(name, std::move(gate));
  }
  return table;
}

std::string read_env(const char* name) {
  const char* value = std::getenv(name);
  return value == nullptr ? std::string{} : std::string(value);
}

} // namespace

GateTable parse_gate_table_file(const std::string& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error(
        "cannot read the metal2vk gate table override at " + path);
  }
  std::string text((std::istreambuf_iterator<char>(input)),
                   std::istreambuf_iterator<char>());
  return parse_gate_table(text);
}

const GateTable& gate_table() {
  static const GateTable table = [] {
    if (const std::string override_path = read_env("MLX_OMARCHY_M2V_POLICY_FILE");
        !override_path.empty()) {
      return parse_gate_table_file(override_path);
    }
    return parse_gate_table(kGateTableJson);
  }();
  return table;
}

std::string kernel_base_name(const std::string& full_name) {
  std::string name = full_name;
  const std::string prefix = "custom_kernel_";
  if (name.rfind(prefix, 0) == 0) {
    name = name.substr(prefix.size());
  }
  const size_t suffix = name.find("__");
  if (suffix != std::string::npos) {
    name = name.substr(0, suffix);
  }
  return name;
}

std::string kernel_entry_name(const std::string& msl_source) {
  const std::string marker = "[[host_name(\"";
  size_t at = msl_source.find(marker);
  if (at != std::string::npos) {
    const size_t begin = at + marker.size();
    const size_t end = msl_source.find('"', begin);
    if (end != std::string::npos) {
      return msl_source.substr(begin, end - begin);
    }
  }
  // A plain kernel: the single [[kernel]] function name.
  at = msl_source.find("[[kernel]]");
  if (at == std::string::npos) {
    return {};
  }
  const size_t declaration = msl_source.find("void ", at);
  if (declaration == std::string::npos) {
    return {};
  }
  const size_t begin = declaration + 5;
  size_t end = begin;
  while (end < msl_source.size() &&
         (msl_source[end] == '_' ||
          (msl_source[end] >= 'a' && msl_source[end] <= 'z') ||
          (msl_source[end] >= 'A' && msl_source[end] <= 'Z') ||
          (msl_source[end] >= '0' && msl_source[end] <= '9'))) {
    ++end;
  }
  return msl_source.substr(begin, end - begin);
}

Route decide_route(const std::string& base_name, const GateTable& table) {
  std::string override_key = "MLX_OMARCHY_M2V_KERNEL_";
  for (const char c : base_name) {
    override_key.push_back(
        c >= 'a' && c <= 'z' ? static_cast<char>(c - 'a' + 'A') : c);
  }
  if (const std::string override_value = read_env(override_key.c_str());
      !override_value.empty()) {
    if (override_value == "m2v") {
      return Route::M2v;
    }
    if (override_value == "translator") {
      return Route::Translator;
    }
    if (override_value == "refuse") {
      return Route::Refuse;
    }
    throw std::runtime_error(
        override_key + "=" + override_value +
        " is not one of m2v, translator, refuse");
  }
  const std::string mode = [] {
    const std::string value = read_env("MLX_OMARCHY_METAL_KERNEL_BACKEND");
    if (value.empty()) {
      return std::string("translator");
    }
    if (value != "translator" && value != "auto" && value != "m2v") {
      throw std::runtime_error(
          "MLX_OMARCHY_METAL_KERNEL_BACKEND=" + value +
          " is not one of translator, auto, m2v");
    }
    return value;
  }();
  if (mode == "translator") {
    return Route::Translator;
  }
  const auto entry = table.find(base_name);
  if (entry == table.end()) {
    // A kernel the table does not know keeps today's route.
    return Route::Translator;
  }
  if (entry->second.state == "verified") {
    return Route::M2v;
  }
  if (mode == "m2v" && entry->second.state == "unverified") {
    return Route::M2v;
  }
  return Route::Translator;
}

namespace {

std::mutex counters_mutex;
RouteCounters counters;
std::map<std::pair<std::string, std::string>, bool> logged_fallbacks;

void dump_summary_at_exit() {
  if (read_env("MLX_OMARCHY_M2V_SUMMARY").empty()) {
    return;
  }
  const std::string summary = route_summary();
  if (!summary.empty()) {
    std::fwrite(summary.data(), 1, summary.size(), stderr);
    std::fflush(stderr);
  }
}

} // namespace

void record_route(const std::string& base_name, Route route) {
  std::lock_guard lock(counters_mutex);
  static const bool registered = [] {
    std::atexit(dump_summary_at_exit);
    return true;
  }();
  (void)registered;
  RouteCounters& slot = counters.per_kernel[base_name];
  switch (route) {
    case Route::M2v:
      ++counters.m2v_dispatches;
      ++slot.m2v_dispatches;
      break;
    case Route::Translator:
      ++counters.translator_dispatches;
      ++slot.translator_dispatches;
      break;
    case Route::Refuse:
      ++counters.refusals;
      ++slot.refusals;
      break;
  }
}

void record_fallback(const std::string& full_name, const std::string& reason) {
  std::lock_guard lock(counters_mutex);
  ++counters.fallbacks;
  ++counters.per_kernel[kernel_base_name(full_name)].fallbacks;
}

void log_fallback_once(const std::string& full_name, const std::string& reason) {
  {
    std::lock_guard lock(counters_mutex);
    auto& logged = logged_fallbacks[{full_name, reason}];
    if (logged) {
      return;
    }
    logged = true;
  }
  std::fprintf(
      stderr,
      "[omarchy] metal_kernel %s: metal2vk route failed (%s), using translator\n",
      full_name.c_str(),
      reason.c_str());
  std::fflush(stderr);
}

RouteCounters route_counters() {
  std::lock_guard lock(counters_mutex);
  return counters;
}

std::string route_summary() {
  std::lock_guard lock(counters_mutex);
  std::ostringstream out;
  out << "[omarchy] metal_kernel route summary: m2v="
      << counters.m2v_dispatches << " translator="
      << counters.translator_dispatches << " fallbacks="
      << counters.fallbacks << " refusals=" << counters.refusals << "\n";
  for (const auto& [name, slot] : counters.per_kernel) {
    out << "[omarchy]   " << name << ": m2v=" << slot.m2v_dispatches
        << " translator=" << slot.translator_dispatches
        << " fallbacks=" << slot.fallbacks << " refusals=" << slot.refusals
        << "\n";
  }
  return out.str();
}

std::string aot_cache_key(const std::string& msl, const std::string& name) {
  return sha256_hex(msl + std::string(1, '\0') + name);
}

std::string aot_dir() {
  if (const std::string configured = read_env("MLX_OMARCHY_M2V_AOT_DIR");
      !configured.empty()) {
    return configured;
  }
  static const std::string next_to_library = [] {
    Dl_info info;
    if (dladdr(reinterpret_cast<void*>(&aot_dir), &info) != 0 &&
        info.dli_fname != nullptr) {
      const std::string path(info.dli_fname);
      const size_t slash = path.rfind('/');
      if (slash != std::string::npos) {
        return path.substr(0, slash) + "/m2v_aot";
      }
    }
    return std::string{};
  }();
  return next_to_library;
}

M2vReflection parse_reflection(const std::string& json) {
  const Json document = parse_json(json);
  M2vReflection reflection;
  if (const Json* name = document.find("name"); name != nullptr) {
    reflection.name = name->as_string("name");
  } else {
    throw std::runtime_error(
        "the metal2vk reflection has no module name");
  }
  if (const Json* coop = document.find("uses_cooperative_matrix");
      coop != nullptr) {
    if (coop->kind != Json::Kind::Bool) {
      throw std::runtime_error(
          "the metal2vk reflection field uses_cooperative_matrix is not a boolean");
    }
    reflection.uses_cooperative_matrix = coop->boolean;
  }
  auto read_size3 =
      [&](const char* field) -> std::optional<std::array<int, 3>> {
    const Json* value = document.find(field);
    if (value == nullptr || value->kind == Json::Kind::Null) {
      return std::nullopt;
    }
    if (value->kind != Json::Kind::Array || value->array.size() != 3) {
      throw std::runtime_error(
          std::string("the metal2vk reflection field ") + field +
          " is not three numbers");
    }
    return std::array<int, 3>{
        value->array[0].as_int(field),
        value->array[1].as_int(field),
        value->array[2].as_int(field)};
  };
  reflection.workgroup_size = read_size3("workgroup_size");
  reflection.workgroup_size_spec_constant_ids =
      read_size3("workgroup_size_spec_constant_ids");
  if (const Json* kernel = document.find("kernel");
      kernel != nullptr && kernel->kind == Json::Kind::Object) {
    if (const Json* args = kernel->find("args");
        args != nullptr && args->kind == Json::Kind::Array) {
      for (const Json& arg : args->array) {
        M2vArgument parsed;
        if (const Json* ordinal = arg.find("ordinal"); ordinal != nullptr) {
          parsed.ordinal = ordinal->as_int("ordinal");
        }
        if (const Json* kind = arg.find("kind"); kind != nullptr) {
          parsed.kind = kind->as_string("kind");
        }
        if (const Json* value = arg.find("set"); value != nullptr) {
          parsed.set = value->as_int("set");
        }
        if (const Json* value = arg.find("binding"); value != nullptr) {
          parsed.binding = value->as_int("binding");
        }
        if (const Json* value = arg.find("offset"); value != nullptr) {
          parsed.offset = value->as_int("offset");
        }
        if (const Json* value = arg.find("size"); value != nullptr) {
          parsed.size = value->as_int("size");
        }
        if (const Json* value = arg.find("spec_id"); value != nullptr) {
          parsed.spec_id = value->as_int("spec_id");
        }
        if (const Json* value = arg.find("metal_name"); value != nullptr) {
          parsed.metal_name = value->as_string("metal_name");
        }
        reflection.args.push_back(std::move(parsed));
      }
    }
  }
  if (const Json* regions = document.find("push_constant_regions");
      regions != nullptr && regions->kind == Json::Kind::Object) {
    for (const auto& [region_name, region] : regions->object) {
      M2vPushRegion parsed;
      parsed.name = region_name;
      if (const Json* value = region.find("offset"); value != nullptr) {
        parsed.offset = value->as_int("offset");
      }
      if (const Json* value = region.find("size"); value != nullptr) {
        parsed.size = value->as_int("size");
      }
      reflection.push_constant_regions.push_back(std::move(parsed));
    }
  }
  return reflection;
}

std::string validate_dispatch(
    const M2vReflection& reflection,
    size_t binding_count,
    const std::array<uint32_t, 3>& threadgroup) {
  if (reflection.uses_cooperative_matrix) {
    return "the module uses cooperative matrix, which the metal2vk route "
           "does not enable on this backend";
  }
  if (reflection.args.empty()) {
    return "the reflection lists no arguments";
  }
  int push_end = 0;
  for (const auto& region : reflection.push_constant_regions) {
    push_end = std::max(push_end, region.offset + region.size);
  }
  int position = 0;
  int previous_ordinal = -1;
  for (const auto& arg : reflection.args) {
    if (arg.kind != "storage_buffer") {
      return "argument " + std::to_string(arg.ordinal) + " (" +
          (arg.metal_name.empty() ? "unnamed" : arg.metal_name) +
          ") has kind " + arg.kind +
          "; the metal2vk route in this build supports storage-buffer "
          "arguments only";
    }
    if (arg.set != 0) {
      return "argument " + std::to_string(arg.ordinal) +
          " binds in descriptor set " + std::to_string(arg.set) +
          "; the backend only provides set 0";
    }
    if (arg.binding != position++) {
      return "argument " + std::to_string(arg.ordinal) + " binds at slot " +
          std::to_string(arg.binding) +
          "; the module's bindings are not dense in argument order";
    }
    if (arg.ordinal <= previous_ordinal ||
        static_cast<size_t>(arg.ordinal) >= binding_count) {
      return "argument ordinal " + std::to_string(arg.ordinal) +
          " is out of order or beyond the kernel's " +
          std::to_string(binding_count) + " buffers";
    }
    previous_ordinal = arg.ordinal;
  }
  if (reflection.workgroup_size_spec_constant_ids.has_value()) {
    if (*reflection.workgroup_size_spec_constant_ids !=
        std::array<int, 3>{0, 1, 2}) {
      return "the module drives the workgroup size from spec constants " +
          std::to_string((*reflection.workgroup_size_spec_constant_ids)[0]) +
          "," +
          std::to_string((*reflection.workgroup_size_spec_constant_ids)[1]) +
          "," +
          std::to_string((*reflection.workgroup_size_spec_constant_ids)[2]) +
          "; the backend specialises ids 0,1,2";
    }
  } else if (reflection.workgroup_size.has_value()) {
    const auto& fixed = *reflection.workgroup_size;
    if (static_cast<uint32_t>(fixed[0]) != threadgroup[0] ||
        static_cast<uint32_t>(fixed[1]) != threadgroup[1] ||
        static_cast<uint32_t>(fixed[2]) != threadgroup[2]) {
      return "the module fixes the workgroup size to " +
          std::to_string(fixed[0]) + "," + std::to_string(fixed[1]) + "," +
          std::to_string(fixed[2]) + " but the dispatch uses " +
          std::to_string(threadgroup[0]) + "," +
          std::to_string(threadgroup[1]) + "," +
          std::to_string(threadgroup[2]);
    }
  } else {
    return "the reflection states no workgroup size and no spec constant ids";
  }
  if (push_end > 128) {
    return "the push constant block ends at byte " +
        std::to_string(push_end) +
        "; the shared pipeline layout provides 128 bytes to compute";
  }
  return {};
}

M2vCompileRun run_m2v_compile(
    const std::string& tool,
    const std::string& msl_path,
    const std::string& out_dir,
    const std::string& name,
    int stage_timeout_s,
    int wall_cap_s) {
  int out_pipe[2];
  int err_pipe[2];
  if (pipe(out_pipe) != 0 || pipe(err_pipe) != 0) {
    return {};
  }
  std::vector<std::string> arguments = {
      tool,
      "--msl",
      msl_path,
      "--out",
      out_dir,
      "--name",
      name,
      "--timeout",
      std::to_string(stage_timeout_s),
      "--json"};
  std::vector<char*> argv;
  argv.reserve(arguments.size() + 1);
  for (auto& argument : arguments) {
    argv.push_back(argument.data());
  }
  argv.push_back(nullptr);

  const pid_t child = fork();
  if (child < 0) {
    close(out_pipe[0]);
    close(out_pipe[1]);
    close(err_pipe[0]);
    close(err_pipe[1]);
    return {};
  }
  if (child == 0) {
    dup2(out_pipe[1], STDOUT_FILENO);
    dup2(err_pipe[1], STDERR_FILENO);
    close(out_pipe[0]);
    close(out_pipe[1]);
    close(err_pipe[0]);
    close(err_pipe[1]);
    setpgid(0, 0);
    execvp(tool.c_str(), argv.data());
    _exit(127);
  }
  close(out_pipe[1]);
  close(err_pipe[1]);

  // The tool keeps its diagnostics in a log file and prints one line per
  // stream, so both pipes fit their kernel buffers; read them after exit.
  M2vCompileRun run;
  int status = 0;
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(wall_cap_s);
  bool timed_out = false;
  while (true) {
    const pid_t done = waitpid(child, &status, WNOHANG);
    if (done == child || done < 0) {
      break;
    }
    if (std::chrono::steady_clock::now() >= deadline) {
      // Kill the whole process group: the tool is past its own deadline.
      kill(-child, SIGKILL);
      waitpid(child, &status, 0);
      timed_out = true;
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  auto drain = [&](int fd, std::string& into) {
    char buffer[4096];
    while (true) {
      const ssize_t got = read(fd, buffer, sizeof(buffer));
      if (got <= 0) {
        break;
      }
      into.append(buffer, static_cast<size_t>(got));
    }
  };
  drain(out_pipe[0], run.json_line);
  drain(err_pipe[0], run.diagnostics);
  close(out_pipe[0]);
  close(err_pipe[0]);

  if (timed_out) {
    run.exit_code = 4;
    run.diagnostics =
        "m2v-compile exceeded the " + std::to_string(wall_cap_s) +
        " s wall-clock cap";
    return run;
  }
  run.exit_code =
      WIFEXITED(status) ? WEXITSTATUS(status) : 4;
  return run;
}

namespace {

std::string find_on_path(const std::string& name) {
  const char* path_value = std::getenv("PATH");
  if (path_value == nullptr) {
    return {};
  }
  std::string path(path_value);
  size_t start = 0;
  while (start <= path.size()) {
    const size_t colon = path.find(':', start);
    const std::string directory = colon == std::string::npos
        ? path.substr(start)
        : path.substr(start, colon - start);
    if (!directory.empty()) {
      const std::string candidate = directory + "/" + name;
      if (access(candidate.c_str(), X_OK) == 0) {
        return candidate;
      }
    }
    if (colon == std::string::npos) {
      break;
    }
    start = colon + 1;
  }
  return {};
}

std::string read_whole_file(const std::string& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    return {};
  }
  return std::string(std::istreambuf_iterator<char>(input),
                     std::istreambuf_iterator<char>());
}

struct TempDir {
  explicit TempDir(const char* pattern) {
    path.assign(pattern);
    path += "-XXXXXX";
    if (mkdtemp(path.data()) == nullptr) {
      throw std::runtime_error("cannot create a metal2vk temporary directory");
    }
  }
  ~TempDir() {
    // Best effort: --out holds a flat pair of artifacts plus the MSL.
    if (DIR* entries = ::opendir(path.c_str()); entries != nullptr) {
      while (const struct dirent* item = ::readdir(entries)) {
        if (item->d_name[0] == '.') {
          continue;
        }
        ::unlink((path + "/" + item->d_name).c_str());
      }
      ::closedir(entries);
    }
    ::rmdir(path.c_str());
  }
  TempDir(const TempDir&) = delete;
  TempDir& operator=(const TempDir&) = delete;
  std::string path;
};

void store_aot(
    const std::string& dir,
    const std::string& key,
    const std::string& spv,
    const std::string& json) {
  for (int i = 1; i <= static_cast<int>(dir.size()); ++i) {
    if (i != static_cast<int>(dir.size()) && dir[i] != '/') {
      continue;
    }
    ::mkdir(dir.substr(0, i).c_str(), 0755);
  }
  for (const auto& [suffix, content] :
       {std::pair{".spv", spv}, {".json", json}}) {
    const std::string target = dir + "/" + key + suffix;
    const std::string temporary =
        target + ".tmp-" + std::to_string(::getpid());
    {
      std::ofstream out(temporary, std::ios::binary | std::ios::trunc);
      if (!out) {
        return;
      }
      out.write(content.data(), static_cast<std::streamsize>(content.size()));
      out.flush();
      if (!out) {
        ::unlink(temporary.c_str());
        return;
      }
    }
    if (::rename(temporary.c_str(), target.c_str()) != 0) {
      ::unlink(temporary.c_str());
    }
  }
}

std::string first_line(const std::string& text) {
  const size_t newline = text.find('\n');
  std::string line = newline == std::string::npos ? text
                                                  : text.substr(0, newline);
  while (!line.empty() &&
         (line.back() == '\r' || line.back() == ' ' || line.back() == '\t')) {
    line.pop_back();
  }
  return line;
}

} // namespace

M2vModule resolve_module(const std::string& msl, const std::string& entry_name) {
  const std::string key = aot_cache_key(msl, entry_name);
  const std::string dir = aot_dir();
  if (!dir.empty()) {
    const std::string spv_blob = read_whole_file(dir + "/" + key + ".spv");
    const std::string json_blob = read_whole_file(dir + "/" + key + ".json");
    if (!spv_blob.empty() && !json_blob.empty()) {
      M2vModule module;
      module.key = key;
      module.spv.resize(spv_blob.size() / sizeof(uint32_t));
      std::memcpy(
          module.spv.data(), spv_blob.data(), module.spv.size() * sizeof(uint32_t));
      if (module.spv.empty() || module.spv.front() != 0x07230203u) {
        throw std::runtime_error(
            "the ahead-of-time metal2vk module at " + dir + "/" + key +
            ".spv is not SPIR-V");
      }
      module.reflection = parse_reflection(json_blob);
      return module;
    }
  }

  std::string tool = read_env("MLX_OMARCHY_M2V_COMPILE");
  if (!tool.empty() && tool.find('/') == std::string::npos) {
    tool = find_on_path(tool);
  } else if (tool.empty()) {
    tool = find_on_path("m2v-compile");
  }
  if (tool.empty()) {
    throw std::runtime_error(
        "no ahead-of-time module is shipped for kernel " + entry_name +
        " and m2v-compile is not available");
  }

  TempDir temp("/tmp/mlx-m2v");
  const std::string msl_path = temp.path + "/kernel.metal";
  {
    std::ofstream out(msl_path, std::ios::binary | std::ios::trunc);
    if (!out) {
      throw std::runtime_error("cannot write the MSL for m2v-compile");
    }
    out.write(msl.data(), static_cast<std::streamsize>(msl.size()));
    out.flush();
    if (!out) {
      throw std::runtime_error("cannot write the MSL for m2v-compile");
    }
  }
  const M2vCompileRun run =
      run_m2v_compile(tool, msl_path, temp.path, entry_name);
  if (run.exit_code == 0) {
    // The --json line names the files (long names truncate the stem).
    std::string spv_name = entry_name + ".spv";
    std::string json_name = entry_name + ".json";
    try {
      const Json result = parse_json(run.json_line);
      if (const Json* value = result.find("spv"); value != nullptr) {
        spv_name = value->as_string("spv");
      }
      if (const Json* value = result.find("json"); value != nullptr) {
        json_name = value->as_string("json");
      }
    } catch (const std::exception&) {
      throw std::runtime_error(
          "m2v-compile reported success but its result is unreadable: " +
          first_line(run.json_line));
    }
    const std::string spv_blob = read_whole_file(
        spv_name.front() == '/' ? spv_name : temp.path + "/" + spv_name);
    const std::string json_blob = read_whole_file(
        json_name.front() == '/' ? json_name : temp.path + "/" + json_name);
    if (spv_blob.empty() || json_blob.empty()) {
      throw std::runtime_error(
          "m2v-compile reported success but wrote no module");
    }
    if (!dir.empty()) {
      store_aot(dir, key, spv_blob, json_blob);
    }
    M2vModule module;
    module.key = key;
    module.spv.resize(spv_blob.size() / sizeof(uint32_t));
    std::memcpy(
        module.spv.data(), spv_blob.data(), module.spv.size() * sizeof(uint32_t));
    if (module.spv.empty() || module.spv.front() != 0x07230203u) {
      throw std::runtime_error(
          "m2v-compile produced a module that is not SPIR-V");
    }
    module.reflection = parse_reflection(json_blob);
    return module;
  }
  if (run.exit_code == 3) {
    std::string construct = "untranslated constructs";
    try {
      const Json result = parse_json(run.json_line);
      if (const Json* value = result.find("refused"); value != nullptr) {
        construct = value->as_string("refused");
      }
    } catch (const std::exception&) {
    }
    throw std::runtime_error(
        "metal2vk refuses kernel " + entry_name + ": " + construct);
  }
  if (run.exit_code == 2) {
    throw std::runtime_error(
        "m2v-compile rejected the call: " + first_line(run.diagnostics));
  }
  if (run.exit_code == 5) {
    throw std::runtime_error(
        "the metal2vk module failed spirv-val: " +
        first_line(run.diagnostics));
  }
  // 4 covers compile errors, stage timeouts and the wall-clock cap.
  throw std::runtime_error(
      "m2v-compile failed: " + first_line(run.diagnostics));
}

namespace {

struct ModuleSlot {
  std::optional<M2vModule> module;
  std::string error;
};

std::mutex module_cache_mutex;
std::unordered_map<std::string, ModuleSlot> module_cache;

} // namespace

const M2vModule& cached_module(
    const std::string& msl,
    const std::string& entry_name) {
  // ponytail: one lock across the compile, so a cold JIT blocks other cold
  // kernels; per-key futures if cold-start concurrency ever matters.
  std::lock_guard lock(module_cache_mutex);
  std::string key = entry_name;
  key.push_back('\0');
  key += msl;
  auto found = module_cache.find(key);
  if (found == module_cache.end()) {
    ModuleSlot slot;
    try {
      slot.module = resolve_module(msl, entry_name);
    } catch (const std::exception& error) {
      slot.error = error.what();
    }
    found = module_cache.emplace(std::move(key), std::move(slot)).first;
  }
  if (!found->second.module.has_value()) {
    throw std::runtime_error(found->second.error);
  }
  return *found->second.module;
}

bool route_env_active() {
  static const bool active = [] {
    if (!read_env("MLX_OMARCHY_METAL_KERNEL_BACKEND").empty()) {
      return true;
    }
    constexpr std::string_view prefix = "MLX_OMARCHY_M2V_KERNEL_";
    for (char** entry = environ; entry != nullptr && *entry != nullptr;
         ++entry) {
      if (std::string_view(*entry).substr(0, prefix.size()) == prefix) {
        return true;
      }
    }
    return false;
  }();
  return active;
}

} // namespace mlx::core::omarchy::m2v
