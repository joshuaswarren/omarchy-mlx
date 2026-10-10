// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

// CPU tests for the metal2vk routing policy, the m2v-compile reflection
// contract, the ahead-of-time cache layout, and the subprocess error
// mapping. No GPU, no Vulkan: m2v_route.cpp is deliberately free of both.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

#include "m2v_route.h"

#include <sys/stat.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>

namespace {

// Sets a variable for the scope and restores the previous state after.
class EnvVar {
 public:
  EnvVar(const char* name, const std::string& value) : name_(name) {
    if (const char* previous = std::getenv(name); previous != nullptr) {
      previous_ = previous;
    }
    setenv(name, value.c_str(), 1);
  }
  EnvVar(const EnvVar&) = delete;
  EnvVar& operator=(const EnvVar&) = delete;
  ~EnvVar() {
    if (previous_.has_value()) {
      setenv(name_.c_str(), previous_->c_str(), 1);
    } else {
      unsetenv(name_.c_str());
    }
  }

 private:
  std::string name_;
  std::optional<std::string> previous_;
};

class ScopedDir {
 public:
  ScopedDir() {
    const auto base =
        std::filesystem::temp_directory_path() / "mlx-m2v-route-tests";
    std::filesystem::create_directories(base);
    path_ = base / std::to_string(counter_++);
    std::filesystem::create_directories(path_);
    path_string_ = path_.string();
  }
  ~ScopedDir() {
    std::error_code ignored;
    std::filesystem::remove_all(path_, ignored);
  }
  ScopedDir(const ScopedDir&) = delete;
  ScopedDir& operator=(const ScopedDir&) = delete;
  const std::string& path() const {
    return path_string_;
  }
  std::string file(const std::string& name) const {
    return (path_ / name).string();
  }

 private:
  inline static int counter_ = 0;
  std::filesystem::path path_;
  std::string path_string_;
};

const char* kWorkedExample = R"json({
 "module": "block_sum_f32.spv",
 "capabilities": ["Shader"],
 "extensions": ["SPV_KHR_non_semantic_info"],
 "uses_cooperative_matrix": false,
 "workgroup_size_spec_constant_ids": [0, 1, 2],
 "push_constant_regions": {
  "PushConstantRegionOffset": {"offset": 0, "size": 12},
  "PushConstantRegionGroupOffset": {"offset": 16, "size": 12}
 },
 "name": "block_sum_f32",
 "kernel": {
  "name": "block_sum_f32",
  "workgroup_size": null,
  "args": [
   {"ordinal": 0, "kind": "storage_buffer", "set": 0, "binding": 0,
    "metal_name": "values", "metal_type": "device const float*"},
   {"ordinal": 1, "kind": "storage_buffer", "set": 0, "binding": 1,
    "metal_name": "partials", "metal_type": "device float*"},
   {"ordinal": 2, "kind": "pod_push_constant", "offset": 32, "size": 4,
    "metal_name": "count", "metal_type": "const uint"}
  ]
 },
 "toolchain": {"clang": "clang version 23.1.1", "opt": "LLVM",
  "clspv": "LLVM", "clspv_sha256": "00", "include_sha256": "46",
  "clspv_patches_sha256": "95", "sources_sha256": "36"}
})json";

using mlx::core::omarchy::m2v::GateTable;
using mlx::core::omarchy::m2v::Route;

GateTable test_table() {
  return GateTable{
      {"verified_kernel", {"verified", "test"}},
      {"unverified_kernel", {"unverified", "test"}},
      {"failed_kernel", {"failed", "test"}},
  };
}

} // namespace

TEST_CASE("kernel base names") {
  using mlx::core::omarchy::m2v::kernel_base_name;
  CHECK(kernel_base_name(
            "custom_kernel_omlx_qwen35_moe_router_topk__bfloat16_t_512_8_"
            "bfloat16_t_uint32_t_bfloat16_t") == "omlx_qwen35_moe_router_topk");
  CHECK(kernel_base_name(
            "custom_kernel_qwen35_gated_delta_step__bfloat16_t_float_128_128_"
            "16_32__bfloat16_t") == "qwen35_gated_delta_step");
  CHECK(kernel_base_name(
            "custom_kernel_omlx_gdn_sigmoid_probe_float_float") ==
        "omlx_gdn_sigmoid_probe_float_float");
  CHECK(kernel_base_name("user_kernel") == "user_kernel");
}

TEST_CASE("kernel entry names") {
  using mlx::core::omarchy::m2v::kernel_entry_name;
  const std::string templated =
      "template <typename T_, int G>\n"
      "[[kernel]] void custom_kernel_a__bfloat16_t_8_16(\n"
      "  device float* x [[buffer(0)]]) { }\n"
      "template [[host_name(\"custom_kernel_a__bfloat16_t_8_16\")]] [[kernel]]"
      " decltype(custom_kernel_a__bfloat16_t_8_16<float, 8>)"
      " custom_kernel_a__bfloat16_t_8_16<float, 8>;\n";
  CHECK(kernel_entry_name(templated) == "custom_kernel_a__bfloat16_t_8_16");
  const std::string plain =
      "[[kernel]] void custom_kernel_probe_float_float(\n"
      "  device float* x [[buffer(0)]]) { }\n";
  CHECK(kernel_entry_name(plain) == "custom_kernel_probe_float_float");
  CHECK(kernel_entry_name("no kernels here").empty());
}

TEST_CASE("route decision order") {
  const auto table = test_table();
  using mlx::core::omarchy::m2v::decide_route;

  {
    EnvVar backend("MLX_OMARCHY_METAL_KERNEL_BACKEND", "");
    EnvVar kernel("MLX_OMARCHY_M2V_KERNEL_VERIFIED_KERNEL", "");
    CHECK(decide_route("verified_kernel", table) == Route::Translator);
    CHECK(decide_route("unknown_kernel", table) == Route::Translator);
  }
  {
    EnvVar backend("MLX_OMARCHY_METAL_KERNEL_BACKEND", "auto");
    CHECK(decide_route("verified_kernel", table) == Route::M2v);
    CHECK(decide_route("unverified_kernel", table) == Route::Translator);
    CHECK(decide_route("failed_kernel", table) == Route::Translator);
    CHECK(decide_route("unknown_kernel", table) == Route::Translator);
  }
  {
    EnvVar backend("MLX_OMARCHY_METAL_KERNEL_BACKEND", "m2v");
    CHECK(decide_route("verified_kernel", table) == Route::M2v);
    CHECK(decide_route("unverified_kernel", table) == Route::M2v);
    CHECK(decide_route("failed_kernel", table) == Route::Translator);
    CHECK(decide_route("unknown_kernel", table) == Route::Translator);
  }
  {
    EnvVar backend("MLX_OMARCHY_METAL_KERNEL_BACKEND", "auto");
    EnvVar kernel("MLX_OMARCHY_M2V_KERNEL_VERIFIED_KERNEL", "translator");
    CHECK(decide_route("verified_kernel", table) == Route::Translator);
  }
  {
    EnvVar backend("MLX_OMARCHY_METAL_KERNEL_BACKEND", "translator");
    EnvVar kernel("MLX_OMARCHY_M2V_KERNEL_UNKNOWN_KERNEL", "m2v");
    CHECK(decide_route("unknown_kernel", table) == Route::M2v);
  }
  {
    EnvVar backend("MLX_OMARCHY_METAL_KERNEL_BACKEND", "auto");
    EnvVar kernel("MLX_OMARCHY_M2V_KERNEL_VERIFIED_KERNEL", "refuse");
    CHECK(decide_route("verified_kernel", table) == Route::Refuse);
  }
  {
    EnvVar backend("MLX_OMARCHY_METAL_KERNEL_BACKEND", "auto");
    EnvVar kernel("MLX_OMARCHY_M2V_KERNEL_UNVERIFIED_KERNEL", "m2v");
    CHECK(decide_route("unverified_kernel", table) == Route::M2v);
  }
  SUBCASE("unknown values fail loudly") {
    bool threw = false;
    try {
      EnvVar backend("MLX_OMARCHY_METAL_KERNEL_BACKEND", "maybe");
      decide_route("verified_kernel", table);
    } catch (const std::exception& error) {
      threw = true;
      CHECK(std::string(error.what()).find(
                "MLX_OMARCHY_METAL_KERNEL_BACKEND=maybe") !=
            std::string::npos);
    }
    CHECK(threw);
  }
  {
    bool threw = false;
    try {
      EnvVar backend("MLX_OMARCHY_METAL_KERNEL_BACKEND", "auto");
      EnvVar kernel("MLX_OMARCHY_M2V_KERNEL_VERIFIED_KERNEL", "no");
      decide_route("verified_kernel", table);
    } catch (const std::exception& error) {
      threw = true;
      CHECK(std::string(error.what()).find(
                "MLX_OMARCHY_M2V_KERNEL_VERIFIED_KERNEL=no") !=
            std::string::npos);
    }
    CHECK(threw);
  }
}

TEST_CASE("gate table parsing") {
  using mlx::core::omarchy::m2v::gate_table;
  using mlx::core::omarchy::m2v::parse_gate_table_file;
  const auto& table = gate_table();
  CHECK(table.size() >= 33);
  const auto& topk = table.at("omlx_qwen35_moe_router_topk");
  CHECK(topk.state == "verified");
  CHECK_FALSE(topk.evidence.empty());
  CHECK(table.at("qwen35_ragged_sdpa_2p1").state == "unverified");
  CHECK(table.at("omlx_verify_attn_wide_partial").state == "failed");
  CHECK(table.at("omlx_verify_attn_gqa_partial").state == "failed");
  CHECK(table.count("omlx_gdn_sigmoid_probe") == 1);
  CHECK(table.count("omlx_gdn_sigmoid_probe_float_float") == 1);

  SUBCASE("an override file replaces the table") {
    ScopedDir dir;
    const std::string path = dir.file("policy.json");
    {
      std::ofstream out(path);
      out << R"json({"kernels": {"k": {"state": "verified",
        "evidence": "e"}}})json";
    }
    const auto overridden = parse_gate_table_file(path);
    CHECK(overridden.size() == 1);
    CHECK(overridden.at("k").state == "verified");
  }
  SUBCASE("a missing override file names the path") {
    CHECK_THROWS_WITH(
        parse_gate_table_file("/nonexistent/policy.json"),
        doctest::Contains("cannot read the metal2vk gate table"));
  }
  SUBCASE("a malformed entry names the kernel") {
    ScopedDir dir;
    const std::string path = dir.file("policy.json");
    {
      std::ofstream out(path);
      out << R"json({"kernels": {"k": {"state": "sometimes"}}})json";
    }
    CHECK_THROWS_WITH(
        parse_gate_table_file(path),
        doctest::Contains("unknown state: sometimes"));
  }
}

TEST_CASE("reflection parsing") {
  using mlx::core::omarchy::m2v::parse_reflection;
  const auto reflection = parse_reflection(kWorkedExample);
  CHECK(reflection.name == "block_sum_f32");
  CHECK_FALSE(reflection.uses_cooperative_matrix);
  REQUIRE(reflection.workgroup_size_spec_constant_ids.has_value());
  CHECK(*reflection.workgroup_size_spec_constant_ids ==
        std::array<int, 3>{0, 1, 2});
  CHECK_FALSE(reflection.workgroup_size.has_value());
  REQUIRE(reflection.args.size() == 3);
  CHECK(reflection.args[0].kind == "storage_buffer");
  CHECK(reflection.args[0].binding == 0);
  CHECK(reflection.args[0].metal_name == "values");
  CHECK(reflection.args[2].kind == "pod_push_constant");
  CHECK(reflection.args[2].offset == 32);
  CHECK(reflection.args[2].size == 4);
  REQUIRE(reflection.push_constant_regions.size() == 2);
  int group_offset_end = -1;
  int plain_offset_end = -1;
  for (const auto& region : reflection.push_constant_regions) {
    if (region.name == "PushConstantRegionGroupOffset") {
      group_offset_end = region.offset + region.size;
    }
    if (region.name == "PushConstantRegionOffset") {
      plain_offset_end = region.offset + region.size;
    }
  }
  CHECK(group_offset_end == 28);
  CHECK(plain_offset_end == 12);

  SUBCASE("malformed documents name the problem") {
    CHECK_THROWS(parse_reflection("{\"no_name\": 1}"));
    CHECK_THROWS(parse_reflection("{\"name\": 7}"));
    CHECK_THROWS(parse_reflection("{\"name\": \"k\", "
                                  "\"workgroup_size\": [1, 2]}"));
    CHECK_THROWS(parse_reflection("{not json"));
  }
}

TEST_CASE("dispatch validation") {
  using mlx::core::omarchy::m2v::parse_reflection;
  using mlx::core::omarchy::m2v::validate_dispatch;
  auto storage_only = R"json({
    "name": "k",
    "uses_cooperative_matrix": false,
    "workgroup_size_spec_constant_ids": [0, 1, 2],
    "push_constant_regions": {
      "PushConstantRegionOffset": {"offset": 0, "size": 12},
      "PushConstantRegionGroupOffset": {"offset": 16, "size": 12}},
    "kernel": {"name": "k", "workgroup_size": null, "args": [
      {"ordinal": 0, "kind": "storage_buffer", "set": 0, "binding": 0},
      {"ordinal": 1, "kind": "storage_buffer", "set": 0, "binding": 1}]}})json";
  const auto good = parse_reflection(storage_only);
  CHECK(validate_dispatch(good, 2, {256, 1, 1}).empty());
  CHECK_FALSE(validate_dispatch(good, 3, {256, 1, 1}).empty());

  SUBCASE("the worked example carries a POD argument") {
    const auto example = parse_reflection(kWorkedExample);
    const std::string reason = validate_dispatch(example, 3, {256, 1, 1});
    CHECK(reason.find("pod_push_constant") != std::string::npos);
    CHECK(reason.find("count") != std::string::npos);
  }
  SUBCASE("moved bindings are refused") {
    auto moved = std::string(storage_only);
    moved.replace(
        moved.find("\"binding\": 1"), 12, "\"binding\": 5");
    const std::string reason =
        validate_dispatch(parse_reflection(moved), 2, {256, 1, 1});
    CHECK(reason.find("binds at slot 5") != std::string::npos);
  }
  SUBCASE("an oversized push constant block is refused") {
    auto big = std::string(storage_only);
    big.replace(
        big.find("\"PushConstantRegionOffset\": {\"offset\": 0, \"size\": 12}"),
        53,
        "\"PushConstantRegionOffset\": {\"offset\": 0, \"size\": 200}");
    const std::string reason =
        validate_dispatch(parse_reflection(big), 2, {256, 1, 1});
    CHECK(reason.find("push constant block") != std::string::npos);
  }
  SUBCASE("a fixed workgroup size must match the dispatch") {
    const auto fixed = parse_reflection(R"json({
      "name": "k",
      "uses_cooperative_matrix": false,
      "workgroup_size": [8, 8, 1],
      "push_constant_regions": {},
      "kernel": {"name": "k", "args": [
        {"ordinal": 0, "kind": "storage_buffer", "set": 0, "binding": 0}]}})json");
    CHECK(validate_dispatch(fixed, 1, {8, 8, 1}).empty());
    const std::string reason = validate_dispatch(fixed, 1, {256, 1, 1});
    CHECK(reason.find("fixes the workgroup size") != std::string::npos);
  }
  SUBCASE("cooperative matrix is refused") {
    auto coop = std::string(storage_only);
    coop.replace(coop.find("false"), 5, "true");
    const std::string reason =
        validate_dispatch(parse_reflection(coop), 2, {256, 1, 1});
    CHECK(reason.find("cooperative matrix") != std::string::npos);
  }
}

TEST_CASE("cache key") {
  using mlx::core::omarchy::m2v::aot_cache_key;
  CHECK(aot_cache_key("hello", "k") ==
        "50a96fa44679a8f1108d673d612dbdd78a0ec66baf413338a4a4c6f294e0a051");
  CHECK(aot_cache_key("hello", "k") == aot_cache_key("hello", "k"));
  CHECK(aot_cache_key("hello", "k") != aot_cache_key("hello", "j"));
  CHECK(aot_cache_key("hello ", "k") != aot_cache_key("hello", "k"));
}

// A fake m2v-compile driven by bash: parses --out and writes the artifact
// pair, then exits with the code named in its path.
class FakeTool {
 public:
  FakeTool(const std::string& behaviour, int exit_code)
      : dir_(), script_(dir_.file("m2v-compile")) {
    std::ofstream out(script_);
    out << "#!/usr/bin/env bash\n"
        << "set -euo pipefail\n"
        << "out=\"\"; name=\"\"\n"
        << "while [[ $# -gt 0 ]]; do case \"$1\" in\n"
        << "  --out) out=\"$2\"; shift 2;;\n"
        << "  --name) name=\"$2\"; shift 2;;\n"
        << "  *) shift;;\n"
        << "esac; done\n"
        << behaviour << "\n"
        << "exit " << exit_code << "\n";
    out.close();
    chmod(script_.c_str(), 0755);
  }
  const std::string& path() const {
    return script_;
  }

 private:
  ScopedDir dir_;
  std::string script_;
};

TEST_CASE("module resolution against a fake tool") {
  using namespace mlx::core::omarchy::m2v;
  // Own the AOT directory so subcases cannot leak entries into the default
  // location next to this binary.
  ScopedDir aot_root;
  EnvVar aot("MLX_OMARCHY_M2V_AOT_DIR", aot_root.path());
  const std::string msl = "[[kernel]] void custom_kernel_k() { }";
  const std::string entry = "custom_kernel_k";
  const std::string reflection =
      R"({"name": "custom_kernel_k", "uses_cooperative_matrix": false,)"
      R"("workgroup_size_spec_constant_ids": [0, 1, 2],)"
      R"("push_constant_regions": {},)"
      R"("kernel": {"name": "custom_kernel_k", "workgroup_size": null,)"
      R"("args": [{"ordinal": 0, "kind": "storage_buffer", "set": 0,)"
      R"("binding": 0}]}})";

  SUBCASE("success compiles once, then the AOT copy answers") {
    FakeTool tool(
        "printf '\\003\\002\\043\\007\\000\\000\\000\\000\\000\\000\\000"
        "\\000' > \"$out/$name.spv\"\n"
        "cat > \"$out/$name.json\" <<'JSONEOF'\n" +
            std::string(reflection) +
            "\nJSONEOF\n"
            "printf '{\"status\": \"ok\", \"cache\": \"miss\", \"name\": "
            "\"%s\", \"spv\": \"%s.spv\", \"json\": \"%s.json\"}' "
            "\"$name\" \"$name\" \"$name\"\n",
        0);
    EnvVar tool_env("MLX_OMARCHY_M2V_COMPILE", tool.path());
    const auto module = resolve_module(msl, entry);
    CHECK(module.key == aot_cache_key(msl, entry));
    REQUIRE(module.spv.size() == 3);
    CHECK(module.spv.front() == 0x07230203u);
    CHECK(module.reflection.name == entry);
    // Break the tool: the AOT copy must still resolve.
    EnvVar no_tool("MLX_OMARCHY_M2V_COMPILE", "/nonexistent/m2v-compile");
    const auto again = resolve_module(msl, entry);
    CHECK(again.key == module.key);
  }
  SUBCASE("refusal by name") {
    FakeTool tool(
        "printf '{\"status\": \"refused\", \"refused\": \"tensor<\", "
        "\"kernel\": \"k\"}'\n",
        3);
    EnvVar tool_env("MLX_OMARCHY_M2V_COMPILE", tool.path());
    CHECK_THROWS_WITH(
        resolve_module(msl, entry),
        doctest::Contains(("metal2vk refuses kernel " + entry).c_str()));
  }
  SUBCASE("compile error surfaces the diagnostic") {
    FakeTool tool(
        "echo 'error: clang failed somewhere (log: /tmp/x)' >&2\n", 4);
    EnvVar tool_env("MLX_OMARCHY_M2V_COMPILE", tool.path());
    CHECK_THROWS_WITH(
        resolve_module(msl, entry),
        doctest::Contains("m2v-compile failed: error: clang failed"));
  }
  SUBCASE("spirv-val failure") {
    FakeTool tool("echo 'error: error: line 12: invalid' >&2\n", 5);
    EnvVar tool_env("MLX_OMARCHY_M2V_COMPILE", tool.path());
    CHECK_THROWS_WITH(
        resolve_module(msl, entry),
        doctest::Contains("failed spirv-val"));
  }
  SUBCASE("usage error") {
    FakeTool tool("echo 'error: no such file: /nope' >&2\n", 2);
    EnvVar tool_env("MLX_OMARCHY_M2V_COMPILE", tool.path());
    CHECK_THROWS_WITH(
        resolve_module(msl, entry),
        doctest::Contains("rejected the call"));
  }
  SUBCASE("no tool anywhere names the kernel") {
    // The variable unset and a PATH without the tool: the not-installed
    // machine.
    EnvVar no_tool("MLX_OMARCHY_M2V_COMPILE", "");
    EnvVar path("PATH", "/usr/bin:/bin");
    CHECK_THROWS_WITH(
        resolve_module(msl, entry),
        doctest::Contains("no ahead-of-time module is shipped"));
  }
}

TEST_CASE("the subprocess wall-clock cap kills a wedged tool") {
  using namespace mlx::core::omarchy::m2v;
  ScopedDir dir;
  FakeTool tool("sleep 30\n", 0);
  const std::string msl_path = dir.file("kernel.metal");
  {
    std::ofstream out(msl_path);
    out << "[[kernel]] void k() { }";
  }
  const auto run =
      run_m2v_compile(tool.path(), msl_path, dir.path(), "k", 1, 1);
  CHECK(run.exit_code == 4);
  CHECK(run.diagnostics.find("wall-clock cap") != std::string::npos);
}

TEST_CASE("route counters and the once-only fallback log") {
  using namespace mlx::core::omarchy::m2v;
  record_route("counter_kernel", Route::M2v);
  record_route("counter_kernel", Route::M2v);
  record_route("counter_kernel", Route::Translator);
  const auto before = route_counters().per_kernel.at("counter_kernel");
  CHECK(before.m2v_dispatches == 2);
  CHECK(before.translator_dispatches == 1);
  record_fallback("custom_kernel_counter_kernel__x", "some reason");
  CHECK(route_counters().per_kernel.at("counter_kernel").fallbacks == 1);
  log_fallback_once("custom_kernel_counter_kernel__x", "some reason");
  log_fallback_once("custom_kernel_counter_kernel__x", "some reason");
  CHECK(route_counters().per_kernel.at("counter_kernel").fallbacks == 1);
  const std::string summary = route_summary();
  CHECK(summary.find("counter_kernel") != std::string::npos);
  CHECK(summary.find("m2v=2") != std::string::npos);
}
