// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

// Wave 11 compiled-tape coverage: every upstream-fusable op class that the
// Omarchy tape interpreter admits must match its eager path. Equivalence
// tolerances are stated per case: float32 compares compiled against eager
// at 1e-6, float16 at 2e-3 after both sides are widened to float32, and
// integer, bitwise, boolean, and select classes compare exactly.
//
// Compiled tapes run by default on every device class; the stale-shape
// defect that corrupted them on Honeykrisp is fixed at the interpreter
// (eval-time shape derivation) and pinned by the shapeless reuse cases
// below. No override is needed or honoured: MLX_OMARCHY_ALLOW_UNSAFE_COMPILE
// is retired. bf16 tapes run on the same interpreter: the 2026-09-02
// bf16 refusal fenced the stale-shape corruption root-caused below, not
// a dtype defect. The trigonometric domain gate is carried per-primitive.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

#include <cmath>
#include <complex>
#include <cstdlib>
#include <functional>
#include <string>
#include <vector>

#include "mlx/backend/gpu/device_info.h"
#include "mlx/backend/omarchy/device.h"
#include "mlx/backend/omarchy/encoder.h"
#include "mlx/backend/omarchy/fused_chain.h"
#include "mlx/backend/omarchy/trace.h"
#include "mlx/compile.h"
#include "mlx/device.h"
#include "mlx/ops.h"
#include "mlx/stream.h"

using namespace mlx::core;

namespace {

void skip(const char* reason) {
  std::cout << "Skipping: " << reason << "\n";
}

Stream gpu_stream() {
  set_default_device(Device::gpu);
  return new_stream(Device::gpu);
}

bool compute_available() {
  if (!gpu::is_available()) {
    skip(
        "no qualifying Vulkan device (set MLX_OMARCHY_ALLOW_NON_APPLE=1 on"
        " a development machine).");
    return false;
  }
  return true;
}

void sync_stream(const Stream& stream) {
  omarchy::get_command_encoder(stream).synchronize();
}

// Widens an array to float32 on the host path so non-f32 buffers can be
// read through one comparison loop.
array as_float32(const array& value, const Stream& stream) {
  if (value.dtype() == float32) {
    return value;
  }
  return astype(value, float32, stream);
}

// Compiled-versus-eager equivalence. Both sides run the same fn, once with
// compilation disabled and once fused, and the buffers must agree within
// the stated tolerance. epsilon == 0 means exact (integer and boolean).
void check_compiled_matches_eager(
    const std::function<std::vector<array>(const std::vector<array>&)>& fn,
    const std::vector<array>& inputs,
    Dtype dtype,
    const Stream& stream,
    double epsilon) {
  set_compile_mode(CompileMode::disabled);
  std::vector<array> eager_outputs = fn(inputs);
  for (auto& out : eager_outputs) {
    out.eval();
  }
  sync_stream(stream);

  set_compile_mode(CompileMode::enabled);
  auto compiled_fn = compile(fn);
  std::vector<array> compiled_outputs = compiled_fn(inputs);
  for (auto& out : compiled_outputs) {
    out.eval();
  }
  sync_stream(stream);
  set_compile_mode(CompileMode::disabled);

  REQUIRE_EQ(eager_outputs.size(), compiled_outputs.size());
  for (size_t j = 0; j < eager_outputs.size(); ++j) {
    REQUIRE_EQ(eager_outputs[j].shape(), compiled_outputs[j].shape());
    if (dtype == bool_ && eager_outputs[j].dtype() == bool_) {
      const bool* eager = eager_outputs[j].data<bool>();
      const bool* compiled = compiled_outputs[j].data<bool>();
      for (size_t index = 0; index < eager_outputs[j].size(); ++index) {
        INFO("bool mismatch at ", index, " eager=",
             static_cast<int>(eager[index]), " compiled=",
             static_cast<int>(compiled[index]));
        CHECK_EQ(static_cast<int>(eager[index]),
                 static_cast<int>(compiled[index]));
      }
      continue;
    }
    if (dtype == int32 && eager_outputs[j].dtype() == int32) {
      const int32_t* eager = eager_outputs[j].data<int32_t>();
      const int32_t* compiled = compiled_outputs[j].data<int32_t>();
      for (size_t index = 0; index < eager_outputs[j].size(); ++index) {
        INFO("int32 mismatch at ", index, " eager=", eager[index],
             " compiled=", compiled[index]);
        CHECK_EQ(eager[index], compiled[index]);
      }
      continue;
    }
    if (dtype == uint32 && eager_outputs[j].dtype() == uint32) {
      const uint32_t* eager = eager_outputs[j].data<uint32_t>();
      const uint32_t* compiled = compiled_outputs[j].data<uint32_t>();
      for (size_t index = 0; index < eager_outputs[j].size(); ++index) {
        INFO("uint32 mismatch at ", index, " eager=", eager[index],
             " compiled=", compiled[index]);
        CHECK_EQ(eager[index], compiled[index]);
      }
      continue;
    }
    if (dtype == float32) {
      const float* eager = eager_outputs[j].data<float>();
      const float* compiled = compiled_outputs[j].data<float>();
      for (size_t index = 0; index < eager_outputs[j].size(); ++index) {
        INFO("f32 mismatch at ", index, " eager=", eager[index],
             " compiled=", compiled[index]);
        CHECK(
            compiled[index] == doctest::Approx(eager[index]).epsilon(epsilon));
      }
    } else {
      array eager_wide = as_float32(eager_outputs[j], stream);
      array compiled_wide = as_float32(compiled_outputs[j], stream);
      eager_wide.eval();
      compiled_wide.eval();
      sync_stream(stream);
      const float* eager = eager_wide.data<float>();
      const float* compiled = compiled_wide.data<float>();
      for (size_t index = 0; index < eager_outputs[j].size(); ++index) {
        bool same =
            compiled[index] == doctest::Approx(eager[index]).epsilon(epsilon);
        if (!same) {
          std::printf(
              "WIDE MISMATCH idx=%zu eager=%.10g compiled=%.10g\n",
              index,
              (double)eager[index],
              (double)compiled[index]);
        }
        CHECK(same);
      }
    }
  }
}
std::string evaluation_error(array value) {
  try {
    value.eval();
  } catch (const std::exception& error) {
    return error.what();
  }
  return {};
}

using UnaryFn = std::function<array(const array&, const Stream&)>;
using BinaryFn =
    std::function<array(const array&, const array&, const Stream&)>;

} // namespace

TEST_CASE("compiled tape matches eager for every float unary class") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();

  struct UnaryCase {
    std::string name;
    std::vector<float> input;
    UnaryFn op;
  };

  std::vector<UnaryCase> cases = {
      {"abs",
       {0.2f, -0.5f, 1.7f, -3.4f},
       [](const array& a, const Stream& s) { return abs(a, s); }},
      {"arcsin",
       {-0.3f, 0.0f, 0.25f, 0.9f},
       [](const array& a, const Stream& s) { return arcsin(a, s); }},
      {"arccos",
       {-0.3f, 0.0f, 0.25f, 0.9f},
       [](const array& a, const Stream& s) { return arccos(a, s); }},
      {"arctan",
       {-1.5f, 0.4f, 2.2f, -0.8f},
       [](const array& a, const Stream& s) { return arctan(a, s); }},
      {"arcsinh",
       {-1.5f, 0.4f, 2.2f, -0.8f},
       [](const array& a, const Stream& s) { return arcsinh(a, s); }},
      {"arccosh",
       {1.05f, 1.5f, 3.0f, 7.25f},
       [](const array& a, const Stream& s) { return arccosh(a, s); }},
      {"arctanh",
       {-0.9f, 0.1f, 0.45f, -0.2f},
       [](const array& a, const Stream& s) { return arctanh(a, s); }},
      {"ceil",
       {-1.7f, -0.2f, 0.3f, 2.9f},
       [](const array& a, const Stream& s) { return ceil(a, s); }},
      {"cos",
       {0.1f, -0.4f, 1.1f, 2.5f},
       [](const array& a, const Stream& s) { return cos(a, s); }},
      {"cosh",
       {0.1f, -0.4f, 1.1f, 2.5f},
       [](const array& a, const Stream& s) { return cosh(a, s); }},
      {"erf",
       {-1.2f, 0.0f, 0.5f, 2.0f},
       [](const array& a, const Stream& s) { return erf(a, s); }},
      {"erfinv",
       {-0.9f, -0.1f, 0.1f, 0.9f},
       [](const array& a, const Stream& s) { return erfinv(a, s); }},
      {"expm1",
       {-1.2f, 0.0f, 0.5f, 2.0f},
       [](const array& a, const Stream& s) { return expm1(a, s); }},
      {"floor",
       {-1.7f, -0.2f, 0.3f, 2.9f},
       [](const array& a, const Stream& s) { return floor(a, s); }},
      {"log",
       {0.25f, 0.5f, 1.0f, 4.0f},
       [](const array& a, const Stream& s) { return log(a, s); }},
      {"log1p",
       {0.25f, 0.5f, 1.0f, 4.0f},
       [](const array& a, const Stream& s) { return log1p(a, s); }},
      {"round",
       {-1.6f, -0.5f, 0.4f, 2.5f},
       [](const array& a, const Stream& s) { return round(a, s); }},
      {"sigmoid",
       {-1.2f, 0.0f, 0.5f, 2.0f},
       [](const array& a, const Stream& s) { return sigmoid(a, s); }},
      {"sign",
       {-3.2f, -0.1f, 0.0f, 5.5f},
       [](const array& a, const Stream& s) { return sign(a, s); }},
      {"sin",
       {0.1f, -0.4f, 1.1f, 2.5f},
       [](const array& a, const Stream& s) { return sin(a, s); }},
      {"sinh",
       {0.1f, -0.4f, 1.1f, 2.5f},
       [](const array& a, const Stream& s) { return sinh(a, s); }},
      {"tan",
       {0.1f, -0.4f, 0.9f, -1.2f},
       [](const array& a, const Stream& s) { return tan(a, s); }},
      {"tanh",
       {0.1f, -0.4f, 1.1f, 2.5f},
       [](const array& a, const Stream& s) { return tanh(a, s); }},
      {"negative",
       {0.2f, -0.5f, 1.7f, -3.4f},
       [](const array& a, const Stream& s) { return negative(a, s); }},
      {"square",
       {0.2f, -0.5f, 1.7f, -3.4f},
       [](const array& a, const Stream& s) { return square(a, s); }},
      {"sqrt",
       {0.25f, 0.5f, 1.0f, 4.0f},
       [](const array& a, const Stream& s) { return sqrt(a, s); }},
      {"exp",
       {-1.2f, 0.0f, 0.5f, 2.0f},
       [](const array& a, const Stream& s) { return exp(a, s); }},
  };

  for (const auto& unary : cases) {
    INFO("unary class: ", unary.name);
    array x(unary.input.begin(), Shape{4}, float32);
    auto fn = [&unary, &stream](const std::vector<array>& inputs) {
      // Two fusable nodes per graph so the tracer builds a tape.
      return std::vector<array>{add(unary.op(inputs[0], stream), array(1.0f), stream)};
    };
    check_compiled_matches_eager(fn, {x}, float32, stream, 1e-6);
  }
}

TEST_CASE("compiled tape matches eager for every float binary class") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();

  struct BinaryCase {
    std::string name;
    std::vector<float> lhs;
    std::vector<float> rhs;
    BinaryFn op;
  };

  std::vector<BinaryCase> cases = {
      {"add",
       {0.2f, -0.5f, 1.7f, -3.4f},
       {1.0f, 0.25f, -2.0f, 0.5f},
       [](const array& a, const array& b, const Stream& s) {
         return add(a, b, s);
       }},
      {"subtract",
       {0.2f, -0.5f, 1.7f, -3.4f},
       {1.0f, 0.25f, -2.0f, 0.5f},
       [](const array& a, const array& b, const Stream& s) {
         return subtract(a, b, s);
       }},
      {"multiply",
       {0.2f, -0.5f, 1.7f, -3.4f},
       {1.0f, 0.25f, -2.0f, 0.5f},
       [](const array& a, const array& b, const Stream& s) {
         return multiply(a, b, s);
       }},
      {"divide",
       {0.2f, -0.5f, 1.7f, -3.4f},
       {1.0f, 0.25f, -2.0f, 0.5f},
       [](const array& a, const array& b, const Stream& s) {
         return divide(a, b, s);
       }},
      {"maximum",
       {0.2f, -0.5f, 1.7f, -3.4f},
       {1.0f, 0.25f, -2.0f, 0.5f},
       [](const array& a, const array& b, const Stream& s) {
         return maximum(a, b, s);
       }},
      {"minimum",
       {0.2f, -0.5f, 1.7f, -3.4f},
       {1.0f, 0.25f, -2.0f, 0.5f},
       [](const array& a, const array& b, const Stream& s) {
         return minimum(a, b, s);
       }},
      {"power",
       {0.5f, 1.5f, 2.0f, -0.5f},
       {2.0f, 3.0f, 0.5f, 3.0f},
       [](const array& a, const array& b, const Stream& s) {
         return power(a, b, s);
       }},
      {"logaddexp",
       {0.2f, -0.5f, 1.7f, -3.4f},
       {1.0f, 0.25f, -2.0f, 0.5f},
       [](const array& a, const array& b, const Stream& s) {
         return logaddexp(a, b, s);
       }},
      {"arctan2",
       {0.2f, -0.5f, 1.7f, -3.4f},
       {1.0f, 0.25f, -2.0f, 0.5f},
       [](const array& a, const array& b, const Stream& s) {
         return arctan2(a, b, s);
       }},
      {"remainder",
       {5.5f, -6.25f, 7.0f, -3.5f},
       {2.0f, 1.5f, 3.0f, 2.0f},
       [](const array& a, const array& b, const Stream& s) {
         return remainder(a, b, s);
       }},
  };

  for (const auto& binary : cases) {
    INFO("binary class: ", binary.name);
    array x(binary.lhs.begin(), Shape{4}, float32);
    array y(binary.rhs.begin(), Shape{4}, float32);
    auto fn = [&binary, &stream](const std::vector<array>& inputs) {
      return std::vector<array>{
          add(binary.op(inputs[0], inputs[1], stream), array(1.0f), stream)};
    };
    check_compiled_matches_eager(fn, {x, y}, float32, stream, 1e-6);
  }
}

TEST_CASE("compiled tape matches eager for integer and bitwise classes") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  std::vector<int32_t> iv = {17, -5, 1024, -4096};
  std::vector<int32_t> jv = {3, 2, 7, 4095};
  std::vector<uint32_t> uv = {12, 345678, 7, 4294967290u};
  array x(iv.begin(), Shape{4}, int32);
  array y(jv.begin(), Shape{4}, int32);
  array u(uv.begin(), Shape{4}, uint32);

  // Integer Add, Multiply, and Square now route through the integer
  // kernel alongside the bitwise, remainder, Power, Abs, and Sign
  // classes. Two fusable nodes per graph so the tracer builds a tape.
  // All exact against eager.

  // BitwiseInvert, wrapped by a bitwise identity.
  auto invert_fn = [&stream](const std::vector<array>& inputs) {
    return std::vector<array>{
        bitwise_xor(bitwise_invert(inputs[0], stream), array(1), stream)};
  };
  check_compiled_matches_eager(invert_fn, {x}, int32, stream, 0);

  // BitwiseBinary and/or/xor through tapes.
  std::vector<BinaryFn> bitwise_cases = {
      [](const array& a, const array& b, const Stream& s) {
        return bitwise_xor(bitwise_and(a, b, s), a, s);
      },
      [](const array& a, const array& b, const Stream& s) {
        return bitwise_and(bitwise_or(a, b, s), a, s);
      },
      [](const array& a, const array& b, const Stream& s) {
        return bitwise_or(bitwise_xor(a, b, s), b, s);
      },
  };
  for (const auto& bitwise : bitwise_cases) {
    auto fn = [&bitwise, &stream](const std::vector<array>& inputs) {
      return std::vector<array>{bitwise(inputs[0], inputs[1], stream)};
    };
    check_compiled_matches_eager(fn, {x, y}, int32, stream, 0);
  }

  // Integer remainder wrapped by a bitwise identity.
  auto rem_fn = [&stream](const std::vector<array>& inputs) {
    return std::vector<array>{
        bitwise_xor(remainder(inputs[0], inputs[1], stream), array(0), stream)};
  };
  check_compiled_matches_eager(rem_fn, {x, y}, int32, stream, 0);

  // Integer Power into integer Abs.
  auto int_power_fn = [&stream](const std::vector<array>& inputs) {
    return std::vector<array>{
        abs(power(inputs[0], array(2), stream), stream)};
  };
  check_compiled_matches_eager(int_power_fn, {x}, int32, stream, 0);

  // Integer Sign wrapped by a bitwise identity.
  auto int_sign_fn = [&stream](const std::vector<array>& inputs) {
    return std::vector<array>{
        bitwise_xor(sign(inputs[0], stream), array(0), stream)};
  };
  check_compiled_matches_eager(int_sign_fn, {x}, int32, stream, 0);

  // uint32 bitwise through a tape.
  auto uint_or_fn = [&stream](const std::vector<array>& inputs) {
    return std::vector<array>{
        bitwise_and(bitwise_or(inputs[0], inputs[1], stream), inputs[0], stream)};
  };
  check_compiled_matches_eager(uint_or_fn, {u, u}, uint32, stream, 0);

  // Integer Add and Multiply through a tape.
  auto int_arith_fn = [&stream](const std::vector<array>& inputs) {
    return std::vector<array>{
        multiply(add(inputs[0], inputs[1], stream), inputs[0], stream)};
  };
  check_compiled_matches_eager(int_arith_fn, {x, y}, int32, stream, 0);

  // Integer Square over boundary magnitudes (INT_MAX, INT_MIN, 2^16)
  // whose squares wrap, wrapped by a bitwise identity so the graph has
  // two fusable nodes.
  std::vector<int32_t> bv = {2147483647, -2147483648, 65536, -1};
  array b(bv.begin(), Shape{4}, int32);
  auto int_square_fn = [&stream](const std::vector<array>& inputs) {
    return std::vector<array>{
        bitwise_xor(square(inputs[0], stream), inputs[0], stream)};
  };
  check_compiled_matches_eager(int_square_fn, {b}, int32, stream, 0);

  // uint32 Add wraps through the tape: 4294967290 + 4294967290 goes to
  // 4294967284, then adding 4294967290 again goes to 4294967278.
  auto uint_wrap_fn = [&stream](const std::vector<array>& inputs) {
    return std::vector<array>{
        add(add(inputs[0], inputs[1], stream), inputs[0], stream)};
  };
  check_compiled_matches_eager(uint_wrap_fn, {u, u}, uint32, stream, 0);
}

TEST_CASE(
    "compiled tape matches eager for comparison, logical, select, and"
    " cast classes") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  std::vector<float> xv = {0.0f, 0.5f, -1.0f, 2.0f};
  std::vector<float> yv = {0.5f, 0.5f, 1.0f, -2.0f};
  std::vector<bool> cv = {true, false, true, false};
  array x(xv.begin(), Shape{4}, float32);
  array y(yv.begin(), Shape{4}, float32);
  array cond(cv.begin(), Shape{4}, bool_);
  // Float comparisons (the backend gates GreaterEqual to int32, so it is
  // covered separately below).
  std::vector<BinaryFn> comparisons = {
      [](const array& a, const array& b, const Stream& s) {
        return equal(a, b, s);
      },
      [](const array& a, const array& b, const Stream& s) {
        return not_equal(a, b, s);
      },
      [](const array& a, const array& b, const Stream& s) {
        return greater(a, b, s);
      },
      [](const array& a, const array& b, const Stream& s) {
        return less(a, b, s);
      },
      [](const array& a, const array& b, const Stream& s) {
        return less_equal(a, b, s);
      },
  };
  for (const auto& compare : comparisons) {
    auto fn = [&compare, &stream](const std::vector<array>& inputs) {
      return std::vector<array>{
          compare(multiply(inputs[0], inputs[0], stream), inputs[1], stream)};
    };
    check_compiled_matches_eager(fn, {x, y}, bool_, stream, 0);
  }

  // GreaterEqual is int32-only in this backend (its causal-mask kernel):
  // feed it an int-capable node and compare exact against eager.
  {
    std::vector<int32_t> iv = {1, 2, 3, 4};
    std::vector<int32_t> jv = {2, 2, 2, 2};
    array xi(iv.begin(), Shape{4}, int32);
    array yi(jv.begin(), Shape{4}, int32);
    auto ge_fn = [&stream](const std::vector<array>& inputs) {
      return std::vector<array>{greater_equal(
          bitwise_xor(inputs[0], inputs[1], stream), inputs[1], stream)};
    };
    check_compiled_matches_eager(ge_fn, {xi, yi}, bool_, stream, 0);
  }

  // Logical classes on boolean tape inputs, exact against eager.
  auto logical_fn = [&stream](const std::vector<array>& inputs) {
    auto both = logical_and(inputs[0], inputs[1], stream);
    auto either = logical_or(both, inputs[1], stream);
    return std::vector<array>{logical_not(either, stream)};
  };
  check_compiled_matches_eager(logical_fn, {cond, cond}, bool_, stream, 0);

  // Select with a comparison feeding it inside one tape, plus an AsType
  // cast layered on top, exact against eager.
  auto select_fn = [&stream](const std::vector<array>& inputs) {
    auto pick = where(inputs[0], inputs[1], inputs[2], stream);
    return std::vector<array>{astype(pick, int32, stream)};
  };
  check_compiled_matches_eager(select_fn, {cond, x, y}, int32, stream, 0);

  // Casts across the float/int boundary through one tape.
  auto cast_fn = [&stream](const std::vector<array>& inputs) {
    auto scaled = multiply(inputs[0], array(3.0f), stream);
    return std::vector<array>{
        add(astype(astype(scaled, int32, stream), float32, stream),
            array(0.5f),
            stream)};
  };
  check_compiled_matches_eager(cast_fn, {x}, float32, stream, 1e-6);
}

TEST_CASE("compiled tape returns multiple outputs exactly") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  std::vector<float> xv = {0.5f, -1.5f, 2.0f, 0.0f};
  std::vector<float> yv = {1.0f, 0.25f, -2.0f, 4.0f};
  array x(xv.begin(), Shape{4}, float32);
  array y(yv.begin(), Shape{4}, float32);

  auto fn = [&stream](const std::vector<array>& inputs) {
    auto sum = add(inputs[0], inputs[1], stream);
    auto prod = multiply(sum, inputs[0], stream);
    return std::vector<array>{sum, prod};
  };
  check_compiled_matches_eager(fn, {x, y}, float32, stream, 1e-6);
}

TEST_CASE("compiled tape interleaves proven and widened classes in f16") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  std::vector<float> gate_v = {0.35f, -1.1f, 2.2f, -0.4f};
  std::vector<float> up_v = {1.5f, 0.25f, -0.75f, 2.0f};
  array gate(gate_v.begin(), Shape{4}, float16);
  array up(up_v.begin(), Shape{4}, float16);

  // The mlx-lm swiglu fragment shape (sigmoid and multiply, the proven
  // subset) with widened tanh and abs nodes interleaved.
  auto fn = [&stream](const std::vector<array>& inputs) {
    auto activated = multiply(sigmoid(inputs[0], stream), inputs[0], stream);
    auto blended = add(activated, inputs[1], stream);
    auto bounded =
        multiply(tanh(blended, stream), abs(blended, stream), stream);
    return std::vector<array>{bounded};
  };
  // f16 keeps the host-independent comparison within 2e-3.
  check_compiled_matches_eager(fn, {gate, up}, float16, stream, 2e-3);
}

TEST_CASE("bf16 tape runs the widened op set and matches eager exactly") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  std::vector<float> xv = {0.0f, 0.1f, 0.2f, 0.3f};
  array x(xv.begin(), Shape{4}, bfloat16);

  // The widened classes fuse through the chain's own bf16 kernels,
  // which round every instruction to the storage dtype exactly like
  // per-node dispatch, so compiled output is bit-exact.
  // Bits compare as uint16: the bf16 bit pattern IS the contract.
  auto fn = [&stream](const std::vector<array>& inputs) {
    auto rounded = round(multiply(inputs[0], inputs[0], stream), stream);
    return std::vector<array>{abs(rounded, stream)};
  };
  set_compile_mode(CompileMode::disabled);
  std::vector<array> eager_outputs = fn({x});
  for (auto& out : eager_outputs) {
    out.eval();
  }
  sync_stream(stream);
  set_compile_mode(CompileMode::enabled);
  auto compiled_fn = compile(fn);
  std::vector<array> compiled_outputs = compiled_fn({x});
  for (auto& out : compiled_outputs) {
    out.eval();
  }
  sync_stream(stream);
  set_compile_mode(CompileMode::disabled);

  REQUIRE_EQ(eager_outputs.size(), compiled_outputs.size());
  for (size_t j = 0; j < eager_outputs.size(); ++j) {
    REQUIRE_EQ(eager_outputs[j].shape(), compiled_outputs[j].shape());
    const uint16_t* eager =
        eager_outputs[j].data<uint16_t>();
    const uint16_t* compiled =
        compiled_outputs[j].data<uint16_t>();
    for (size_t index = 0; index < eager_outputs[j].size(); ++index) {
      INFO("bf16 bits mismatch at ", index, " eager=0x",
           std::hex, eager[index], " compiled=0x", compiled[index],
           std::dec);
      CHECK_EQ(eager[index], compiled[index]);
    }
  }
}

TEST_CASE("compiled tape computes Real, Imag, and Conjugate on complex64") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  // Upstream fuses the complex trio (mlx/compile.cpp is_unary); the
  // backend kernels landed with complex64 transport, so the tape runs
  // them instead of refusing by name. Eager is the reference.
  std::vector<complex64_t> zv = {
      complex64_t{1.0f, 0.5f},
      complex64_t{-2.0f, 3.0f},
      complex64_t{0.0f, -1.25f},
      complex64_t{4.5f, 0.0f}};
  array z(zv.begin(), Shape{4}, complex64);

  auto fn = [&stream](const std::vector<array>& inputs) {
    auto projected = add(
        real(inputs[0], stream),
        multiply(imag(inputs[0], stream), array(2.0f), stream),
        stream);
    auto mirrored = multiply(conjugate(inputs[0], stream), inputs[0], stream);
    return std::vector<array>{projected, mirrored};
  };
  set_compile_mode(CompileMode::disabled);
  auto eager = fn({z});
  set_compile_mode(CompileMode::enabled);
  auto compiled = compile(fn)({z});
  for (auto& out : eager) {
    out.eval();
  }
  for (auto& out : compiled) {
    out.eval();
  }
  sync_stream(stream);
  set_compile_mode(CompileMode::disabled);

  REQUIRE_EQ(compiled[0].dtype(), float32);
  REQUIRE_EQ(compiled[1].dtype(), complex64);
  for (size_t i = 0; i < 4; ++i) {
    INFO("element ", i);
    CHECK_EQ(compiled[0].data<float>()[i], eager[0].data<float>()[i]);
    CHECK_EQ(
        compiled[1].data<complex64_t>()[i].real(),
        eager[1].data<complex64_t>()[i].real());
    CHECK_EQ(
        compiled[1].data<complex64_t>()[i].imag(),
        eager[1].data<complex64_t>()[i].imag());
  }
  // The values themselves: real + 2*imag, and |z|^2 with zero imaginary.
  CHECK_EQ(compiled[0].data<float>()[1], -2.0f + 6.0f);
  CHECK_EQ(compiled[1].data<complex64_t>()[1].real(), 13.0f);
  CHECK_EQ(compiled[1].data<complex64_t>()[1].imag(), 0.0f);
}

// The tape debug switches (docs/install-omarchy.md) are diagnostics for
// the Honeykrisp layer isolation. Defaults must stay inert: with no env
// var set the scoped switch state is false and every other case in this
// battery exercises the unswitched path. Each switch, once set, must
// leave the tape matching eager on a development device. On a real Apple
// GPU the case skips with the rest: there the switches exist so the
// hardware protocol can bisect the defect, not to promise correct
// values.
TEST_CASE(
    "compiled-tape debug switches stay inert by default and correct when set") {
  CHECK_FALSE(omarchy::tape_full_barriers());
  CHECK_FALSE(omarchy::tape_no_reuse());
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  // No stream capture on purpose: ops recorded through an explicit
  // non-trace stream materialize outside the compiled tape, which
  // truncated this fn's tape to a broadcast add and made the submission
  // assertion below count eager per-op commits instead of tape nodes.
  // With plain ops the tracer captures the whole graph.
  auto swiglu = [](const std::vector<array>& inputs) {
    array gate = sigmoid(inputs[0]);
    return std::vector<array>{
        multiply(inputs[0], gate),
        add(gate, array(1.0f))};
  };
  // A single-output linear variant for the submission-shape measurement
  // below: every node has one consumer, so the tracer tapes the whole
  // chain (the multi-output swiglu above splits at its shared gate and
  // tapes only a remnant, which cannot demonstrate per-node shape).
  auto swiglu1 = [](const std::vector<array>& inputs) {
    array gate = sigmoid(inputs[0]);
    return std::vector<array>{
        add(multiply(inputs[0], gate), array(1.0f))};
  };

  std::vector<float> xv(64);
  for (size_t i = 0; i < xv.size(); ++i) {
    xv[i] = static_cast<float>(i) * 0.125f - 4.0f;
  }
  array x(xv.begin(), Shape{2, 32}, float32);
  std::vector<array> inputs = {x};

  // NO_BUFFER_CACHE is absent here on purpose: it is read once at
  // runtime init, which earlier cases in this binary have already
  // triggered, so a mid-process setenv cannot arm it. It is verified by
  // a full-process env run in the receipt instead.
  const char* switches[] = {
      "MLX_OMARCHY_TAPE_PER_NODE_SUBMIT",
      "MLX_OMARCHY_TAPE_FULL_BARRIERS",
      "MLX_OMARCHY_TAPE_NO_REUSE",
      "MLX_OMARCHY_TAPE_SYNC_EVERY"};
  for (const char* name : switches) {
    INFO("switch ", name);
    setenv(name, "1", 1);
    check_compiled_matches_eager(swiglu, inputs, float32, stream, 1e-6);
    if (std::string(name) == "MLX_OMARCHY_TAPE_PER_NODE_SUBMIT") {
      // The decisive-bisector property: the switched tape must actually
      // change submission shape. Contract (7c3d6b4 batching): eager ops
      // accumulate and flush only at finalize, event contracts, the
      // 100-node budget, or host-read sync; an explicit commit() still
      // submits the open batch immediately; the tape runner commits
      // after every dispatching node; a view-only node (Broadcast)
      // records no commands and produces no submission either way.
      // The tracer splits this fn at its multi-consumer intermediate
      // (gate feeds both outputs), so the taped unit is a 2-node
      // [Broadcast, Add] and the split-off ops run eager inside the
      // compiled call - the switch must still strictly dominate the
      // batched baseline for the same fn, which is the property the
      // hardware bisect relies on. Assert dominance, not a constant:
      // the count legitimately tracks whatever the tracer tapes.
      unsetenv(name);
      set_compile_mode(CompileMode::enabled);
      auto baseline_fn = compile(swiglu1);
      uint64_t base_t0 = omarchy::trace::counters().vk_submissions.load();
      auto base_out = baseline_fn(inputs);
      for (auto& out : base_out) {
        out.eval();
      }
      sync_stream(stream);
      uint64_t delta_off =
          omarchy::trace::counters().vk_submissions.load() - base_t0;

      setenv(name, "1", 1);
      auto fused = compile(swiglu1);
      uint64_t on_t0 = omarchy::trace::counters().vk_submissions.load();
      auto outputs = fused(inputs);
      for (auto& out : outputs) {
        out.eval();
      }
      sync_stream(stream);
      uint64_t delta_on =
          omarchy::trace::counters().vk_submissions.load() - on_t0;
      set_compile_mode(CompileMode::disabled);
      CHECK(delta_off < delta_on);
      CHECK(delta_on >= 3);
    }
    unsetenv(name);
  }
}

TEST_CASE(
    "shapeless compiled tape reused at a new shape matches eager"
    " (stale-traced-shape regression)") {
  // The Honeykrisp compiled-tape corruption root cause: the interpreter
  // used to materialize every node at the traced shape, so a shapeless
  // fragment reused across a sequence-length change computed into
  // stale-shape outputs and read past its eval-time input buffers. On
  // Honeykrisp the reads landed in recycled allocator pages (the
  // f16-impossible ~8e8 Cos-gate magnitudes); on lavapipe they read
  // zeros, so nothing but a shape-mismatch refusal ever showed. This
  // case traces at one shape and requires value-and-shape equality with
  // eager at other shapes, which the stale-shape path cannot pass.
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();

  auto swiglu = [](const std::vector<array>& inputs) {
    array gate = sigmoid(inputs[0]);
    return std::vector<array>{
        add(multiply(inputs[0], gate), array(0.5f))};
  };

  auto make_input = [](Shape shape) {
    std::vector<float> data(shape[0] * shape[1]);
    for (size_t i = 0; i < data.size(); ++i) {
      data[i] = static_cast<float>(i % 17) * 0.25f - 2.0f;
    }
    return array(data.begin(), shape, float32);
  };

  set_compile_mode(CompileMode::enabled);
  auto compiled_fn = compile(swiglu, /*shapeless=*/true);

  // First call traces the tape at this shape.
  std::vector<array> trace_inputs = {make_input(Shape{2, 32})};
  auto trace_out = compiled_fn(trace_inputs);
  for (auto& out : trace_out) {
    out.eval();
  }
  sync_stream(stream);

  // Reuse the same traced fragment at other shapes: grow, shrink, and
  // an unrelated width. Grow is the recorded refusal direction.
  std::vector<Shape> shapes = {Shape{4, 32}, Shape{1, 32}, Shape{3, 48}};
  for (size_t case_index = 0; case_index < shapes.size(); ++case_index) {
    INFO("shape case ", case_index);
    std::vector<array> inputs = {make_input(shapes[case_index])};
    set_compile_mode(CompileMode::disabled);
    std::vector<array> eager = swiglu(inputs);
    for (auto& out : eager) {
      out.eval();
    }
    sync_stream(stream);
    set_compile_mode(CompileMode::enabled);
    auto outputs = compiled_fn(inputs);
    for (auto& out : outputs) {
      out.eval();
    }
    sync_stream(stream);
    set_compile_mode(CompileMode::disabled);
    REQUIRE_EQ(eager.size(), outputs.size());
    for (size_t j = 0; j < eager.size(); ++j) {
      REQUIRE_EQ(eager[j].shape(), outputs[j].shape());
      const float* eager_data = eager[j].data<float>();
      const float* compiled_data = outputs[j].data<float>();
      for (size_t index = 0; index < eager[j].size(); ++index) {
        INFO("element ", index, " eager=", eager_data[index],
             " compiled=", compiled_data[index]);
        CHECK(std::abs(eager_data[index] - compiled_data[index]) <=
              1e-6 * std::max(1.0f, std::abs(eager_data[index])));
      }
    }
  }
}

TEST_CASE(
    "bf16 shapeless tape reused at a new shape matches eager exactly"
    " (gate-lift regression)") {
  // The configuration the lifted bf16 gate fenced: mlx-lm traces swiglu
  // in bfloat16 at prefill shapes and serves every decode step from the
  // same shapeless fragment. The stale-shape corruption this interpreter
  // fixed (see the f32 case above) was the defect the 2026-09-02 bf16
  // refusal was installed against; the dtype was never at fault. This
  // case runs that exact trace-then-reuse pattern in bf16 and requires
  // bit-exact agreement with eager after widening.
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();

  auto swiglu = [](const std::vector<array>& inputs) {
    array gate = sigmoid(inputs[0]);
    return std::vector<array>{
        add(multiply(inputs[0], gate), array(0.5f))};
  };

  auto make_input = [](Shape shape) {
    std::vector<float> data(shape[0] * shape[1]);
    for (size_t i = 0; i < data.size(); ++i) {
      data[i] = static_cast<float>(i % 17) * 0.25f - 2.0f;
    }
    return array(data.begin(), shape, bfloat16);
  };

  set_compile_mode(CompileMode::enabled);
  auto compiled_fn = compile(swiglu, /*shapeless=*/true);

  std::vector<array> trace_inputs = {make_input(Shape{2, 32})};
  auto trace_out = compiled_fn(trace_inputs);
  for (auto& out : trace_out) {
    out.eval();
  }
  sync_stream(stream);

  std::vector<Shape> shapes = {Shape{4, 32}, Shape{1, 32}, Shape{3, 48}};
  for (size_t case_index = 0; case_index < shapes.size(); ++case_index) {
    INFO("shape case ", case_index);
    std::vector<array> inputs = {make_input(shapes[case_index])};
    set_compile_mode(CompileMode::disabled);
    std::vector<array> eager = swiglu(inputs);
    for (auto& out : eager) {
      out.eval();
    }
    sync_stream(stream);
    set_compile_mode(CompileMode::enabled);
    auto outputs = compiled_fn(inputs);
    for (auto& out : outputs) {
      out.eval();
    }
    sync_stream(stream);
    set_compile_mode(CompileMode::disabled);
    REQUIRE_EQ(eager.size(), outputs.size());
    for (size_t j = 0; j < eager.size(); ++j) {
      REQUIRE_EQ(eager[j].shape(), outputs[j].shape());
      const uint16_t* eager_data = eager[j].data<uint16_t>();
      const uint16_t* compiled_data = outputs[j].data<uint16_t>();
      for (size_t index = 0; index < eager[j].size(); ++index) {
        INFO("element ", index, " eager=0x", std::hex, eager_data[index],
             " compiled=0x", compiled_data[index], std::dec);
        CHECK_EQ(eager_data[index], compiled_data[index]);
      }
    }
  }
}

TEST_CASE("shapeless reuse works on the default path with no override set") {
  // The user-visible contract the re-enable restores: a plain process
  // with no MLX_OMARCHY_* override and no MLX_DISABLE_COMPILE compiles,
  // serves new shapes from the shapeless fragment, and matches eager.
  // Complements the stale-traced-shape regression above (which pins the
  // interpreter fix); this case pins the default path a user actually
  // runs, and refuses to run under any override so it cannot silently
  // test the wrong path.
  for (const char* name :
       {"MLX_OMARCHY_ALLOW_UNSAFE_COMPILE", "MLX_DISABLE_COMPILE"}) {
    if (std::getenv(name) != nullptr) {
      skip("override environment set; this case proves the default path");
      return;
    }
  }
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();

  auto swiglu = [](const std::vector<array>& inputs) {
    array gate = sigmoid(inputs[0]);
    return std::vector<array>{
        add(multiply(inputs[0], gate), array(0.25f))};
  };

  auto make_input = [](Shape shape) {
    std::vector<float> data(shape[0] * shape[1]);
    for (size_t i = 0; i < data.size(); ++i) {
      data[i] = static_cast<float>(i % 23) * 0.125f - 1.5f;
    }
    return array(data.begin(), shape, float32);
  };

  set_compile_mode(CompileMode::enabled);
  auto compiled_fn = compile(swiglu, /*shapeless=*/true);

  std::vector<array> trace_inputs = {make_input(Shape{2, 48})};
  auto trace_out = compiled_fn(trace_inputs);
  for (auto& out : trace_out) {
    out.eval();
  }
  sync_stream(stream);

  std::vector<Shape> shapes = {Shape{1, 48}, Shape{5, 64}, Shape{2, 96}};
  for (size_t case_index = 0; case_index < shapes.size(); ++case_index) {
    INFO("shape case ", case_index);
    std::vector<array> inputs = {make_input(shapes[case_index])};
    set_compile_mode(CompileMode::disabled);
    std::vector<array> eager = swiglu(inputs);
    for (auto& out : eager) {
      out.eval();
    }
    sync_stream(stream);
    set_compile_mode(CompileMode::enabled);
    auto outputs = compiled_fn(inputs);
    for (auto& out : outputs) {
      out.eval();
    }
    sync_stream(stream);
    set_compile_mode(CompileMode::disabled);
    REQUIRE_EQ(eager.size(), outputs.size());
    for (size_t j = 0; j < eager.size(); ++j) {
      REQUIRE_EQ(eager[j].shape(), outputs[j].shape());
      const float* eager_data = eager[j].data<float>();
      const float* compiled_data = outputs[j].data<float>();
      for (size_t index = 0; index < eager[j].size(); ++index) {
        INFO("element ", index, " eager=", eager_data[index],
             " compiled=", compiled_data[index]);
        CHECK(std::abs(eager_data[index] - compiled_data[index]) <=
              1e-6 * std::max(1.0f, std::abs(eager_data[index])));
      }
    }
  }
}

TEST_CASE(
    "compiled AdaIN + Snake: a Sin after a fused chain matches eager"
    " (no-primitive regression)") {
  // Kokoro's AdaINResBlock1 shape: instance norm with a style affine from
  // a split matmul, then Snake x + (1/a) sin(a x)^2 with a captured alpha.
  // The norm/affine run closes as a fused chain whose tail operand is a
  // tracing-graph array, and Sin's argument gate settle()s a graph over
  // its input. Before the fix that nested eval walked into the trace and
  // threw "[eval] Attempting to eval an array without a primitive".
  if (!compute_available()) {
    return;
  }
  if (!omarchy::fused_chain_enabled()) {
    skip("MLX_OMARCHY_FUSED_CHAIN is off; this case needs the fused path");
    return;
  }
  Stream stream = gpu_stream();
  const int C = 8, T = 64, S = 16;
  auto ramp = [](int n, float scale, float offset) {
    std::vector<float> v(n);
    for (int i = 0; i < n; ++i) {
      v[i] = static_cast<float>(i % 13) * scale + offset;
    }
    return v;
  };
  auto wv = ramp(2 * C * S, 0.01f, -0.05f);
  auto bv = ramp(2 * C, 0.02f, -0.1f);
  auto av = ramp(C, 0.1f, 0.5f);
  auto xv = ramp(C * T, 0.21f, -1.3f);
  auto sv = ramp(S, 0.3f, -2.0f);
  for (Dtype dtype : {float32, bfloat16}) {
    INFO("dtype ", dtype);
    array w(wv.begin(), {2 * C, S}, dtype);
    array b(bv.begin(), {2 * C}, dtype);
    array alpha(av.begin(), {1, C, 1}, dtype);
    auto adain_snake = [w, b, alpha](const std::vector<array>& in) {
      const array& x = in[0];
      auto affine = split(expand_dims(addmm(b, in[1], transpose(w)), 2), 2, 1);
      array xn = divide(
          subtract(x, mean(x, {2}, true)),
          sqrt(add(var(x, {2}, true), array(1e-5f, x.dtype()))));
      array y = add(multiply(add(array(1.0f, x.dtype()), affine[0]), xn),
                    affine[1]);
      array s2 = power(sin(multiply(alpha, y)), array(2.0f, x.dtype()));
      return std::vector<array>{
          add(y, multiply(divide(array(1.0f, x.dtype()), alpha), s2))};
    };
    std::vector<array> inputs = {
        array(xv.begin(), {1, C, T}, dtype), array(sv.begin(), {1, S}, dtype)};
    // bf16: 1e-7 relative is far below one bf16 ulp (2^-8), so after
    // widening this demands bit-exact agreement with eager.
    check_compiled_matches_eager(
        adain_snake, inputs, dtype, stream, dtype == float32 ? 1e-5 : 1e-7);
  }
}
