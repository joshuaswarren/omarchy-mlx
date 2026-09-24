// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

#include "mlx/backend/omarchy/ane/manifest.h"

#include <json.hpp>

#include <algorithm>
#include <fstream>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace mlx::core::omarchy::ane {
namespace {

constexpr const char* kManifestName = "manifest.json";

std::runtime_error manifest_error(const std::string& reason) {
  return std::runtime_error("[omarchy-ane] manifest: " + reason + ".");
}

const nlohmann::json& required_field(
    const nlohmann::json& object,
    const std::string& field) {
  auto found = object.find(field);
  if (found == object.end()) {
    throw manifest_error("missing field '" + field + "'");
  }
  return *found;
}

void reject_unknown_fields(
    const nlohmann::json& object,
    const std::vector<const char*>& allowed) {
  std::set<std::string> allowed_set(allowed.begin(), allowed.end());
  for (auto it = object.begin(); it != object.end(); ++it) {
    if (allowed_set.count(it.key()) == 0) {
      throw manifest_error("unknown field '" + it.key() + "'");
    }
  }
}

std::string require_string(const nlohmann::json& object, const char* field) {
  const auto& value = required_field(object, field);
  if (!value.is_string()) {
    throw manifest_error(std::string("field '") + field + "' must be a string");
  }
  return value.get<std::string>();
}

std::string require_non_empty_string(
    const nlohmann::json& object,
    const char* field) {
  std::string value = require_string(object, field);
  if (value.empty()) {
    throw manifest_error(std::string("field '") + field + "' must not be empty");
  }
  return value;
}

uint64_t require_unsigned(const nlohmann::json& object, const char* field) {
  const auto& value = required_field(object, field);
  if (!value.is_number_unsigned()) {
    throw manifest_error(
        std::string("field '") + field + "' must be a non-negative integer");
  }
  return value.get<uint64_t>();
}

uint64_t require_positive(const nlohmann::json& object, const char* field) {
  uint64_t value = require_unsigned(object, field);
  if (value == 0) {
    throw manifest_error(std::string("field '") + field + "' must be positive");
  }
  return value;
}

bool is_lower_hex(const std::string& value, size_t length) {
  if (value.size() != length) {
    return false;
  }
  for (char c : value) {
    if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'))) {
      return false;
    }
  }
  return true;
}

std::string require_hex(
    const nlohmann::json& object,
    const char* field,
    size_t length,
    const char* what) {
  std::string value = require_string(object, field);
  if (!is_lower_hex(value, length)) {
    throw manifest_error(
        std::string("field '") + field + "' must be " + what + " (" +
        std::to_string(length) + " lowercase hex characters)");
  }
  return value;
}

uint64_t checked_mul(uint64_t lhs, uint64_t rhs, const std::string& field) {
  if (lhs != 0 && rhs > std::numeric_limits<uint64_t>::max() / lhs) {
    throw manifest_error(field + " geometry overflows uint64");
  }
  return lhs * rhs;
}

uint64_t checked_add(uint64_t lhs, uint64_t rhs, const std::string& field) {
  if (rhs > std::numeric_limits<uint64_t>::max() - lhs) {
    throw manifest_error(field + " range overflows uint64");
  }
  return lhs + rhs;
}

uint64_t dtype_size(const std::string& dtype) {
  if (dtype == "float16" || dtype == "bfloat16") return 2;
  if (dtype == "float32" || dtype == "int32") return 4;
  if (dtype == "uint8" || dtype == "bool") return 1;
  return 0;
}

std::vector<uint64_t> require_shape(
    const nlohmann::json& object,
    const char* field,
    const std::string& where) {
  const auto& value = required_field(object, field);
  if (!value.is_array() || value.empty()) {
    throw manifest_error(where + " field '" + field + "' must be a non-empty array");
  }
  std::vector<uint64_t> shape;
  for (const auto& dim : value) {
    if (!dim.is_number_unsigned() || dim.get<uint64_t>() == 0) {
      throw manifest_error(where + " field '" + field + "' must contain positive integers");
    }
    shape.push_back(dim.get<uint64_t>());
  }
  return shape;
}

uint64_t element_count(
    const std::vector<uint64_t>& shape,
    const std::string& where) {
  uint64_t count = 1;
  for (uint64_t dim : shape) {
    count = checked_mul(count, dim, where);
  }
  return count;
}

AneTensor parse_tensor(
    const nlohmann::json& value,
    const std::string& list_name,
    size_t position,
    uint64_t tile_alignment) {
  std::string where = list_name + "[" + std::to_string(position) + "]";
  if (!value.is_object()) {
    throw manifest_error(where + " must be an object");
  }
  reject_unknown_fields(
      value, {"name", "index", "dtype", "shape", "byte_size", "stride"});

  AneTensor tensor;
  tensor.name = require_non_empty_string(value, "name");
  tensor.index = require_unsigned(value, "index");
  tensor.dtype = require_non_empty_string(value, "dtype");
  tensor.shape = require_shape(value, "shape", where);
  tensor.byte_size = require_positive(value, "byte_size");
  tensor.stride = require_positive(value, "stride");

  uint64_t size = dtype_size(tensor.dtype);
  if (size == 0) {
    throw manifest_error(where + " has unsupported dtype '" + tensor.dtype + "'");
  }
  uint64_t expected = checked_mul(element_count(tensor.shape, where), size, where);
  if (tensor.byte_size != expected) {
    throw manifest_error(
        where + " byte_size " + std::to_string(tensor.byte_size) +
        " does not match dtype geometry " + std::to_string(expected));
  }
  if (tensor.stride < tensor.byte_size || tensor.stride % tile_alignment != 0) {
    throw manifest_error(
        where + " stride must cover byte_size and be " +
        std::to_string(tile_alignment) + "-aligned");
  }
  return tensor;
}

std::vector<AneTensor> parse_tensor_list(
    const nlohmann::json& root,
    const char* field,
    bool required_non_empty,
    uint64_t tile_alignment) {
  const auto& values = required_field(root, field);
  if (!values.is_array() || (required_non_empty && values.empty())) {
    throw manifest_error(
        std::string("field '") + field + "' must be " +
        (required_non_empty ? "a non-empty array" : "an array"));
  }
  std::vector<AneTensor> tensors;
  std::set<uint64_t> indices;
  for (size_t i = 0; i < values.size(); ++i) {
    AneTensor tensor = parse_tensor(values[i], field, i, tile_alignment);
    if (!indices.insert(tensor.index).second) {
      throw manifest_error(std::string("field '") + field + "' has duplicate index");
    }
    tensors.push_back(std::move(tensor));
  }
  return tensors;
}

AneLogicalResult parse_logical_result(
    const nlohmann::json& value,
    size_t position) {
  const std::string where =
      "logical_results[" + std::to_string(position) + "]";
  if (!value.is_object()) {
    throw manifest_error(where + " must be an object");
  }
  reject_unknown_fields(
      value,
      {"name", "dtype", "shape", "tensor", "element_offset",
       "element_count", "conversion"});

  AneLogicalResult result;
  result.name = require_non_empty_string(value, "name");
  result.dtype = require_non_empty_string(value, "dtype");
  result.shape = require_shape(value, "shape", where);
  result.tensor = require_non_empty_string(value, "tensor");
  result.element_offset = require_unsigned(value, "element_offset");
  result.element_count = require_positive(value, "element_count");
  result.conversion = require_non_empty_string(value, "conversion");
  if (result.conversion != "identity") {
    throw manifest_error(
        where + " has unsupported conversion '" + result.conversion + "'");
  }
  if (element_count(result.shape, where) != result.element_count) {
    throw manifest_error(where + " shape does not match element_count");
  }
  return result;
}

std::vector<AneLogicalResult> parse_logical_results(
    const nlohmann::json& root) {
  const auto& values = required_field(root, "logical_results");
  if (!values.is_array() || values.empty()) {
    throw manifest_error("field 'logical_results' must be a non-empty array");
  }
  std::vector<AneLogicalResult> results;
  results.reserve(values.size());
  for (size_t i = 0; i < values.size(); ++i) {
    results.push_back(parse_logical_result(values[i], i));
  }
  return results;
}

void validate_logical_results(const AneManifest& manifest) {
  struct OutputRecord {
    const AneTensor* tensor;
    bool referenced{false};
  };
  std::map<std::string, OutputRecord> outputs;
  for (const auto& output : manifest.outputs) {
    outputs.emplace(output.name, OutputRecord{&output});
  }
  for (size_t i = 0; i < manifest.logical_results.size(); ++i) {
    const auto& result = manifest.logical_results[i];
    const std::string where =
        "logical_results[" + std::to_string(i) + "]";
    auto found = outputs.find(result.tensor);
    if (found == outputs.end()) {
      throw manifest_error(
          where + " references unknown physical output tensor '" +
          result.tensor + "'");
    }
    const AneTensor& output = *found->second.tensor;
    found->second.referenced = true;
    if (result.dtype != output.dtype) {
      throw manifest_error(
          where + " dtype does not match physical output tensor '" +
          result.tensor + "'");
    }
    const uint64_t physical_count = element_count(output.shape, where);
    if (result.element_offset > physical_count ||
        result.element_count > physical_count - result.element_offset) {
      throw manifest_error(
          where + " range exceeds physical output tensor '" +
          result.tensor + "'");
    }
  }
  for (const auto& [name, output] : outputs) {
    if (!output.referenced) {
      throw manifest_error(
          "logical_results must reference every physical output; missing " + name);
    }
  }
}

AneProgramBinding parse_binding(
    const nlohmann::json& value,
    const std::string& where,
    uint64_t tile_alignment) {
  if (!value.is_object()) {
    throw manifest_error(where + " must be an object");
  }
  reject_unknown_fields(
      value,
      {"tensor", "channel", "dtype", "shape", "nchw", "logical_bytes",
       "allocation_bytes", "element_offset", "element_count", "physical_elements",
       "raw"});

  // Raw staging (whole-program containers): the task stream addresses the
  // surface through selector registers, not NCHW tile placement, so the
  // binding carries only the channel, the staged byte count, and the ANEC
  // channel allocation. Staged bytes are memcpy'd straight into (and back
  // out of) the channel buffer, exactly like libane's ane_send/ane_read.
  if (value.value("raw", false)) {
    AneProgramBinding binding;
    binding.raw = true;
    binding.tensor = require_non_empty_string(value, "tensor");
    binding.channel = require_unsigned(value, "channel");
    binding.dtype = require_non_empty_string(value, "dtype");
    binding.logical_bytes = require_positive(value, "logical_bytes");
    binding.allocation_bytes = require_positive(value, "allocation_bytes");
    if (binding.dtype != "float16" && binding.dtype != "bfloat16" &&
        binding.dtype != "bool") {
      throw manifest_error(where + " requires a 16-bit or 1-byte bool dtype");
    }
    if (binding.allocation_bytes < binding.logical_bytes ||
        binding.allocation_bytes % tile_alignment != 0) {
      throw manifest_error(
          where + " allocation is smaller than logical data or unaligned");
    }
    uint64_t size = dtype_size(binding.dtype);
    if (size == 0 || binding.logical_bytes % size != 0) {
      throw manifest_error(
          where + " logical_bytes must be a multiple of the dtype size");
    }
    binding.element_count = binding.logical_bytes / size;
    return binding;
  }

  AneProgramBinding binding;
  binding.tensor = require_non_empty_string(value, "tensor");
  binding.channel = require_unsigned(value, "channel");
  binding.dtype = require_non_empty_string(value, "dtype");
  binding.shape = require_shape(value, "shape", where);
  const auto& nchw = required_field(value, "nchw");
  if (!nchw.is_array() || nchw.size() != binding.nchw.size()) {
    throw manifest_error(where + " field 'nchw' must contain exactly 6 integers");
  }
  for (size_t i = 0; i < binding.nchw.size(); ++i) {
    if (!nchw[i].is_number_unsigned() || nchw[i].get<uint64_t>() == 0) {
      throw manifest_error(where + " field 'nchw' must contain positive integers");
    }
    binding.nchw[i] = nchw[i].get<uint64_t>();
  }
  binding.logical_bytes = require_positive(value, "logical_bytes");
  binding.allocation_bytes = require_positive(value, "allocation_bytes");
  binding.element_offset = require_unsigned(value, "element_offset");
  binding.element_count = require_positive(value, "element_count");
  binding.physical_elements = require_positive(value, "physical_elements");

  uint64_t size = dtype_size(binding.dtype);
  if (size == 0) {
    throw manifest_error(where + " has unsupported dtype '" + binding.dtype + "'");
  }
  uint64_t shape_elements = element_count(binding.shape, where);
  if (binding.element_count != shape_elements ||
      binding.logical_bytes != checked_mul(shape_elements, size, where)) {
    throw manifest_error(where + " logical allocation does not match dtype geometry");
  }
  uint64_t physical_elements = 1;
  for (size_t i = 0; i < 4; ++i) {
    physical_elements = checked_mul(physical_elements, binding.nchw[i], where);
  }
  if (binding.physical_elements != physical_elements) {
    throw manifest_error(where + " physical_elements does not match NCHW geometry");
  }
  if (binding.element_count > binding.physical_elements) {
    throw manifest_error(where + " element_count exceeds physical_elements");
  }
  if (binding.allocation_bytes < binding.logical_bytes ||
      binding.allocation_bytes % tile_alignment != 0) {
    throw manifest_error(where + " allocation is smaller than logical data or unaligned");
  }
  if (binding.nchw[4] % binding.nchw[5] != 0 ||
      binding.nchw[5] % size != 0 ||
      binding.nchw[4] / binding.nchw[5] < binding.nchw[2] ||
      binding.nchw[5] / size < binding.nchw[3]) {
    throw manifest_error(where + " NCHW packed tile is smaller than logical data");
  }
  uint64_t physical_bytes = checked_mul(
      checked_mul(binding.nchw[0], binding.nchw[1], where),
      binding.nchw[4],
      where);
  if (physical_bytes > binding.allocation_bytes) {
    throw manifest_error(where + " NCHW physical bytes exceed allocation_bytes");
  }
  return binding;
}

std::vector<AneProgramBinding> parse_bindings(
    const nlohmann::json& program,
    const char* field,
    uint64_t tile_alignment,
    size_t program_index,
    bool require_non_empty) {
  const auto& values = required_field(program, field);
  std::string prefix = "programs[" + std::to_string(program_index) + "]." + field;
  if (!values.is_array() || (require_non_empty && values.empty())) {
    throw manifest_error(prefix + " must be " +
                         (require_non_empty ? "a non-empty array" : "an array"));
  }
  std::vector<AneProgramBinding> bindings;
  std::set<uint64_t> channels;
  for (size_t i = 0; i < values.size(); ++i) {
    AneProgramBinding binding = parse_binding(
        values[i], prefix + "[" + std::to_string(i) + "]", tile_alignment);
    if (!channels.insert(binding.channel).second) {
      throw manifest_error(prefix + " has duplicate channel");
    }
    bindings.push_back(std::move(binding));
  }
  return bindings;
}

AneProgram parse_program(
    const nlohmann::json& value,
    size_t index,
    uint64_t tile_alignment) {
  std::string where = "programs[" + std::to_string(index) + "]";
  if (!value.is_object()) {
    throw manifest_error(where + " must be an object");
  }
  reject_unknown_fields(
      value,
      {"payload", "operation", "encoder", "task_descriptors", "scratch_bytes",
       "inputs", "outputs"});
  AneProgram program;
  program.payload = require_non_empty_string(value, "payload");
  program.operation = require_non_empty_string(value, "operation");
  program.encoder = require_non_empty_string(value, "encoder");
  program.task_descriptors = require_positive(value, "task_descriptors");
  program.scratch_bytes = require_unsigned(value, "scratch_bytes");
  program.inputs = parse_bindings(value, "inputs", tile_alignment, index, false);
  program.outputs =
      parse_bindings(value, "outputs", tile_alignment, index, true);
  return program;
}

AnePayload parse_payload(const nlohmann::json& value, size_t position) {
  std::string where = "payloads[" + std::to_string(position) + "]";
  if (!value.is_object()) {
    throw manifest_error(where + " must be an object");
  }
  reject_unknown_fields(value, {"role", "path", "sha256", "byte_size"});
  AnePayload payload;
  payload.role = require_non_empty_string(value, "role");
  if (payload.role != "anec" && payload.role != "weights") {
    throw manifest_error(where + " field 'role' must be 'anec' or 'weights'");
  }
  payload.path = require_non_empty_string(value, "path");
  std::filesystem::path path(payload.path);
  if (path.is_absolute() || path.has_parent_path() || path.filename() != path) {
    throw manifest_error(where + " field 'path' must be a plain filename");
  }
  payload.sha256 = require_hex(value, "sha256", 64, "a SHA-256 digest");
  payload.byte_size = require_positive(value, "byte_size");
  return payload;
}

void parse_compiler(const nlohmann::json& value, AneManifest& manifest) {
  if (!value.is_object()) {
    throw manifest_error("field 'compiler' must be an object");
  }
  reject_unknown_fields(value, {"host_build", "toolchain", "target"});
  manifest.compiler.host_build = require_non_empty_string(value, "host_build");
  manifest.compiler.toolchain = require_non_empty_string(value, "toolchain");
  manifest.compiler.target = require_non_empty_string(value, "target");
  if (manifest.compiler.target != "h13") {
    throw manifest_error("field 'compiler.target' must be exactly 'h13'");
  }
}

void parse_provenance(const nlohmann::json& value, AneManifest& manifest) {
  if (!value.is_object()) {
    throw manifest_error("field 'provenance' must be an object");
  }
  reject_unknown_fields(value, {"source_repo", "source_commit", "exported_at"});
  manifest.provenance.source_repo = require_non_empty_string(value, "source_repo");
  manifest.provenance.source_commit =
      require_hex(value, "source_commit", 40, "a source commit");
  manifest.provenance.exported_at = require_non_empty_string(value, "exported_at");
}

void parse_release(const nlohmann::json& value, AneManifest& manifest) {
  if (!value.is_object()) {
    throw manifest_error("field 'release_asset' must be an object");
  }
  reject_unknown_fields(value, {"model", "model_sha256"});
  manifest.release_asset.model = require_non_empty_string(value, "model");
  manifest.release_asset.model_sha256 =
      require_hex(value, "model_sha256", 64, "a compiled-payload collection SHA-256 digest");
}

void validate_bindings(AneManifest& manifest) {
  struct TensorRecord {
    const AneTensor* tensor;
    std::string role;
    bool read{false};
    bool written{false};
  };
  std::map<std::string, TensorRecord> tensors;
  auto add = [&](const std::vector<AneTensor>& list, const char* role) {
    for (const auto& tensor : list) {
      if (!tensors.emplace(tensor.name, TensorRecord{&tensor, role}).second) {
        throw manifest_error("tensor name '" + tensor.name + "' appears in multiple roles");
      }
    }
  };
  add(manifest.inputs, "input");
  add(manifest.outputs, "output");
  add(manifest.state, "state");
  add(manifest.intermediates, "intermediate");

  auto validate = [&](const AneProgramBinding& binding, bool output, const std::string& where) {
    auto found = tensors.find(binding.tensor);
    if (found == tensors.end()) {
      throw manifest_error(where + " references unknown tensor '" + binding.tensor + "'");
    }
    TensorRecord& record = found->second;
    bool role_ok = output ? (record.role != "input") : (record.role != "output");
    if (!role_ok) {
      throw manifest_error(where + " uses tensor '" + binding.tensor + "' in the wrong direction");
    }
    if (binding.dtype != record.tensor->dtype) {
      throw manifest_error(where + " dtype does not match tensor '" + binding.tensor + "'");
    }
    uint64_t total = element_count(record.tensor->shape, where);
    uint64_t end = checked_add(binding.element_offset, binding.element_count, where);
    if (end > total || binding.allocation_bytes > record.tensor->stride) {
      throw manifest_error(where + " range or allocation exceeds tensor '" + binding.tensor + "'");
    }
    record.read = record.read || !output;
    record.written = record.written || output;
  };

  uint64_t total_descriptors = 0;
  std::set<std::string> program_payloads;
  for (size_t p = 0; p < manifest.programs.size(); ++p) {
    const AneProgram& program = manifest.programs[p];
    total_descriptors = checked_add(total_descriptors, program.task_descriptors, "task_descriptors");
    if (!program_payloads.insert(program.payload).second) {
      throw manifest_error("program payload '" + program.payload + "' is referenced more than once");
    }
    for (size_t i = 0; i < program.inputs.size(); ++i) {
      validate(program.inputs[i], false,
               "programs[" + std::to_string(p) + "].inputs[" + std::to_string(i) + "]");
    }
    for (size_t i = 0; i < program.outputs.size(); ++i) {
      validate(program.outputs[i], true,
               "programs[" + std::to_string(p) + "].outputs[" + std::to_string(i) + "]");
    }
  }
  if (total_descriptors != manifest.task_descriptors) {
    throw manifest_error("field 'task_descriptors' does not equal the program total");
  }

  std::set<std::string> anec_payloads;
  for (const auto& payload : manifest.payloads) {
    if (payload.role == "anec") {
      anec_payloads.insert(payload.path);
    }
  }
  if (anec_payloads != program_payloads) {
    throw manifest_error("program payload mapping does not bind every ANEC payload exactly once");
  }

  for (const auto& [name, record] : tensors) {
    if ((record.role == "input" && !record.read) ||
        (record.role == "output" && !record.written) ||
        ((record.role == "state" || record.role == "intermediate") &&
         (!record.read || !record.written))) {
      throw manifest_error("tensor '" + name + "' is not fully bound for role '" + record.role + "'");
    }
  }

  using Range = std::pair<uint64_t, uint64_t>;
  std::map<std::string, std::vector<Range>> available;
  auto add_range = [&](const std::string& name, uint64_t begin, uint64_t end) {
    auto& ranges = available[name];
    ranges.emplace_back(begin, end);
    std::sort(ranges.begin(), ranges.end());
    std::vector<Range> merged;
    for (const Range& range : ranges) {
      if (merged.empty() || range.first > merged.back().second) {
        merged.push_back(range);
      } else {
        merged.back().second = std::max(merged.back().second, range.second);
      }
    }
    ranges = std::move(merged);
  };
  auto add_tensor = [&](const AneTensor& tensor) {
    add_range(tensor.name, 0, element_count(tensor.shape, tensor.name));
  };
  for (const auto& tensor : manifest.inputs) add_tensor(tensor);
  for (const auto& tensor : manifest.state) add_tensor(tensor);
  for (uint64_t program_index : manifest.dispatch_plan) {
    const AneProgram& program = manifest.programs[program_index];
    for (const auto& binding : program.inputs) {
      uint64_t end = binding.element_offset + binding.element_count;
      bool covered = false;
      for (const Range& range : available[binding.tensor]) {
        covered = covered ||
            (range.first <= binding.element_offset && range.second >= end);
      }
      if (!covered) {
        throw manifest_error(
            "dispatch_plan reads tensor '" + binding.tensor + "' before its range is written");
      }
    }
    for (const auto& binding : program.outputs) {
      add_range(
          binding.tensor,
          binding.element_offset,
          binding.element_offset + binding.element_count);
    }
  }
  for (const auto& output : manifest.outputs) {
    const uint64_t total = element_count(output.shape, output.name);
    auto found = available.find(output.name);
    bool complete = found != available.end() &&
        std::any_of(
            found->second.begin(),
            found->second.end(),
            [&](const Range& range) { return range.first == 0 && range.second >= total; });
    if (!complete) {
      throw manifest_error("output tensor '" + output.name + "' is not fully written");
    }
  }
}

} // namespace

AneManifest parse_ane_manifest(const std::filesystem::path& manifest_path) {
  std::ifstream input(manifest_path);
  if (!input) {
    throw manifest_error(
        std::string("cannot open ") + kManifestName + " at " +
        manifest_path.parent_path().string());
  }

  nlohmann::json root;
  try {
    input >> root;
  } catch (const nlohmann::json::exception& error) {
    throw manifest_error(std::string("invalid JSON: ") + error.what());
  }
  if (!root.is_object()) {
    throw manifest_error("root must be an object");
  }
  reject_unknown_fields(
      root,
      {"manifest_version", "tile_shift", "name", "graph_hash",
       "task_descriptors", "inputs", "outputs", "logical_results", "state",
       "intermediates", "programs", "dispatch_plan", "payloads", "compiler",
       "driver_abi_major", "provenance", "release_asset"});

  AneManifest manifest;
  const auto& version = required_field(root, "manifest_version");
  bool supported_version =
      (version.is_number_unsigned() && version.get<uint64_t>() == kAneManifestVersion) ||
      (version.is_number_integer() && !version.is_number_unsigned() &&
       version.get<int64_t>() == kAneManifestVersion);
  if (!supported_version) {
    throw manifest_error(
        "unsupported manifest_version (expected " +
        std::to_string(kAneManifestVersion) + ")");
  }
  manifest.manifest_version = kAneManifestVersion;
  // Optional per-bundle tile-count unit: log2 of the byte unit the ANEC
  // header's tiles[] counts are denominated in. Absent means shift 14, the
  // H13 island unit, so every pre-existing manifest parses unchanged.
  if (root.contains("tile_shift")) {
    manifest.tile_shift = require_unsigned(root, "tile_shift");
    if (manifest.tile_shift != kAneTileShiftDefault &&
        manifest.tile_shift != kAneTileShiftWholeProgram) {
      throw manifest_error(
          "unsupported tile_shift " + std::to_string(manifest.tile_shift) +
          " (expected " + std::to_string(kAneTileShiftDefault) + " or " +
          std::to_string(kAneTileShiftWholeProgram) + ")");
    }
  }
  const uint64_t tile_alignment = uint64_t{1} << manifest.tile_shift;
  manifest.name = require_non_empty_string(root, "name");
  manifest.graph_hash = require_hex(root, "graph_hash", 64, "a graph SHA-256 digest");
  manifest.task_descriptors = require_positive(root, "task_descriptors");
  manifest.inputs = parse_tensor_list(root, "inputs", true, tile_alignment);
  manifest.outputs = parse_tensor_list(root, "outputs", true, tile_alignment);
  manifest.logical_results = parse_logical_results(root);
  manifest.state = parse_tensor_list(root, "state", false, tile_alignment);
  manifest.intermediates = parse_tensor_list(root, "intermediates", false, tile_alignment);

  const auto& programs = required_field(root, "programs");
  if (!programs.is_array() || programs.empty()) {
    throw manifest_error("field 'programs' must be a non-empty array");
  }
  for (size_t i = 0; i < programs.size(); ++i) {
    manifest.programs.push_back(parse_program(programs[i], i, tile_alignment));
  }

  const auto& dispatch = required_field(root, "dispatch_plan");
  if (!dispatch.is_array() || dispatch.size() != manifest.programs.size()) {
    throw manifest_error("field 'dispatch_plan' must contain every program exactly once");
  }
  std::set<uint64_t> dispatched;
  for (const auto& index : dispatch) {
    if (!index.is_number_unsigned() || index.get<uint64_t>() >= manifest.programs.size() ||
        !dispatched.insert(index.get<uint64_t>()).second) {
      throw manifest_error("field 'dispatch_plan' must be a permutation of program indices");
    }
    manifest.dispatch_plan.push_back(index.get<uint64_t>());
  }

  const auto& payloads = required_field(root, "payloads");
  if (!payloads.is_array() || payloads.empty()) {
    throw manifest_error("field 'payloads' must be a non-empty array");
  }
  std::set<std::string> paths;
  size_t anec_count = 0;
  size_t weights_count = 0;
  for (size_t i = 0; i < payloads.size(); ++i) {
    AnePayload payload = parse_payload(payloads[i], i);
    if (!paths.insert(payload.path).second) {
      throw manifest_error("duplicate payload path '" + payload.path + "'");
    }
    anec_count += payload.role == "anec";
    weights_count += payload.role == "weights";
    manifest.payloads.push_back(std::move(payload));
  }
  if (anec_count == 0 || weights_count > 1) {
    throw manifest_error("payloads require at least one ANEC and at most one weights file");
  }

  parse_compiler(required_field(root, "compiler"), manifest);
  manifest.driver_abi_major = require_unsigned(root, "driver_abi_major");
  if (manifest.driver_abi_major != kAneDriverAbiMajor) {
    throw manifest_error(
        "unsupported driver_abi_major " + std::to_string(manifest.driver_abi_major) +
        " (expected " + std::to_string(kAneDriverAbiMajor) + ")");
  }
  parse_provenance(required_field(root, "provenance"), manifest);
  parse_release(required_field(root, "release_asset"), manifest);
  validate_logical_results(manifest);
  validate_bindings(manifest);
  return manifest;
}

} // namespace mlx::core::omarchy::ane
