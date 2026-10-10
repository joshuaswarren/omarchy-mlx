// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

#pragma once

// Routing policy and module resolution for running selected
// fast::CustomKernel kernels through metal2vk (clspv-produced SPIR-V)
// instead of the MSL-to-GLSL translator. See docs/custom-kernel-metal2vk.md
// for the environment variables, the route order, and the rollback table.
//
// Everything here is free of Vulkan and MLX array dependencies so the
// policy, reflection parsing, cache layout, and the m2v-compile subprocess
// contract are unit-testable on a box with no GPU.

#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace mlx::core::omarchy::m2v {

// The route a kernel takes for its next dispatch.
enum class Route {
  Translator,
  M2v,
  Refuse,
};

// One entry of the gate table: how much the metal2vk parity run trusts a
// kernel. `verified` routes through metal2vk in the auto mode, `unverified`
// only in the m2v mode, `failed` never does.
struct GateEntry {
  std::string state;  // "verified" | "unverified" | "failed"
  std::string evidence;
};

using GateTable = std::map<std::string, GateEntry>;

// The embedded gate table shipped with the binary (between the
// M2V_GATE_TABLE markers in m2v_route.cpp). Throws std::runtime_error on a
// malformed override file named by MLX_OMARCHY_M2V_POLICY_FILE.
const GateTable& gate_table();

// Parses one gate-table JSON document from disk; the override-file path of
// gate_table(). Throws when the file is unreadable or malformed.
GateTable parse_gate_table_file(const std::string& path);

// "custom_kernel_omlx_a__bfloat16_t_8_16" -> "omlx_a".
// A plain (non-template) kernel keeps its whole name after the prefix:
// "custom_kernel_omlx_probe_float_float" -> "omlx_probe_float_float".
std::string kernel_base_name(const std::string& full_name);

// The entry name m2v-compile must be asked for: the [[host_name("...")]]
// instantiation when the MSL has one, else the single [[kernel]] function
// name. Empty when neither is present.
std::string kernel_entry_name(const std::string& msl_source);

// Full route decision for one dispatch, in order: the per-kernel
// MLX_OMARCHY_M2V_KERNEL_<BASE> override, the whole-feature
// MLX_OMARCHY_METAL_KERNEL_BACKEND mode, the gate table, the default
// (translator). Both environment variables are read on every call so a
// caller can flip them between dispatches in one process. An unknown value
// of either variable throws std::runtime_error naming it.
Route decide_route(const std::string& base_name, const GateTable& table);

// Counters behind the process route summary.
struct RouteCounters {
  uint64_t m2v_dispatches{0};
  uint64_t translator_dispatches{0};
  uint64_t fallbacks{0};
  uint64_t refusals{0};
  std::map<std::string, RouteCounters> per_kernel;
};

void record_route(const std::string& base_name, Route route);
void record_fallback(const std::string& full_name, const std::string& reason);
RouteCounters route_counters();
// One line per kernel that did not take a silent route, plus totals.
std::string route_summary();
// Log once per (kernel, reason): the fallback is visible exactly once even
// when the dispatch runs per token.
void log_fallback_once(const std::string& full_name, const std::string& reason);

// A resolved metal2vk module: SPIR-V words plus the parsed reflection.
struct M2vArgument {
  int ordinal{-1};
  std::string kind;
  int set{-1};
  int binding{-1};
  int offset{-1};
  int size{-1};
  int spec_id{-1};
  std::string metal_name;
};

struct M2vPushRegion {
  std::string name;
  int offset{0};
  int size{0};
};

struct M2vReflection {
  std::string name;  // the entry point name inside the module
  bool uses_cooperative_matrix{false};
  // Fixed workgroup size when the module declares one.
  std::optional<std::array<int, 3>> workgroup_size;
  // The three spec constant ids for x, y, z when the size is
  // spec-constant driven (what MSL-derived modules use).
  std::optional<std::array<int, 3>> workgroup_size_spec_constant_ids;
  std::vector<M2vArgument> args;
  std::vector<M2vPushRegion> push_constant_regions;
};

struct M2vModule {
  std::string key;  // the content key in the AOT cache
  std::vector<uint32_t> spv;
  M2vReflection reflection;
};

// The cache key of the ahead-of-time index: sha256 over the exact MSL text
// and the entry name, so a kernel's module is found by what mlx assembles.
std::string aot_cache_key(const std::string& msl, const std::string& name);

// The AOT directory: MLX_OMARCHY_M2V_AOT_DIR, else "m2v_aot" next to the
// loaded mlx library. Empty when neither resolves.
std::string aot_dir();

// Reflection contract of the m2v-compile <name>.json output. Throws
// std::runtime_error naming the field on a malformed document.
M2vReflection parse_reflection(const std::string& json);

// Dispatch-time check of a resolved module against what the encoder will
// bind: every argument a storage buffer at its own position, a push
// constant block inside the shared pipeline layout, and a workgroup size
// story. Returns an empty string when the dispatch is safe, else the
// reason the metal2vk route must not run this module.
std::string validate_dispatch(
    const M2vReflection& reflection,
    size_t binding_count,
    const std::array<uint32_t, 3>& threadgroup);

// Runs `m2v-compile --msl <path> --out <dir> --name <name> --json` as a
// direct child (no shell), with a wall-clock cap that kills the child's
// whole process group.
struct M2vCompileRun {
  int exit_code{-1};  // 0 ok, 2 usage, 3 refused, 4 compile error or
                      // timeout, 5 spirv-val failure, -1 no result
  std::string json_line;
  std::string diagnostics;
};

M2vCompileRun run_m2v_compile(
    const std::string& tool,
    const std::string& msl_path,
    const std::string& out_dir,
    const std::string& name,
    int stage_timeout_s = 120,
    int wall_cap_s = 600);

// Resolves the module for one kernel: the shipped AOT directory first,
// then a JIT compile through m2v-compile (whose own M2V_CACHE_DIR the child
// consults), writing the result back into the AOT directory when it is
// writable. Throws std::runtime_error with a one-line reason on every
// failure path (no tool, refusal, compile error, validator failure).
M2vModule resolve_module(
    const std::string& msl,
    const std::string& entry_name);

} // namespace mlx::core::omarchy::m2v
