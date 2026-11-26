// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

// Pinned ggml Vulkan per-op comparator for the U2 spike.
//
// llama-bench reports model-level throughput, not matched kernel timings,
// and test-backend-ops uses a fixed internal shape matrix that does not
// expose the spike's dimensions. This probe measures the pinned llama.cpp
// build's ggml Vulkan backend directly, at the exact matched shapes:
//   mul_mat F16   : prefill GEMM rows (fp16 math class)
//   mul_mat Q4_K  : decode GEMV rows (Q4_K quantization)
//   flash_attn_ext: causal GQA SDPA rows
// The case names equal the spike row names; the benchmark enforces that
// dims and quantization match per row before any ratio is computed.
// The runner compiles this against the pinned checkout and records the
// commit it was actually built from.
//
// Exit codes: 0 ok, 2 usage error, 1 runtime error.

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"

#include <unistd.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace {

[[noreturn]] void die(const std::string& msg, int code) {
  std::cerr << "ERROR: " << msg << std::endl;
  std::exit(code);
}

std::string arg_value(int argc, char** argv, int& i, const char* name) {
  if (i + 1 >= argc) {
    die(std::string("missing value for ") + name, 2);
  }
  return argv[++i];
}

uint16_t f32_to_f16(float value) {
  uint32_t x;
  std::memcpy(&x, &value, 4);
  uint32_t sign = (x >> 16) & 0x8000;
  int32_t biased = int32_t((x >> 23) & 0xff);
  uint32_t mant = x & 0x7fffff;
  if (biased == 0xff)
    return uint16_t(sign | 0x7c00 | (mant ? 0x200 : 0));
  if (biased == 0)
    return uint16_t(sign);
  int32_t exp = biased - 127 + 15;
  if (exp >= 0x1f)
    return uint16_t(sign | 0x7c00);
  if (exp <= 0) {
    if (exp < -10)
      return uint16_t(sign);
    mant |= 0x800000;
    int shift = 14 - exp;
    uint32_t half = mant >> shift;
    uint32_t rem = mant & ((1u << shift) - 1);
    uint32_t mid = 1u << (shift - 1);
    if (rem > mid || (rem == mid && (half & 1)))
      half++;
    return uint16_t(sign | half);
  }
  uint32_t half = (uint32_t(exp) << 10) | (mant >> 13);
  uint32_t rem = mant & 0x1fff;
  if (rem > 0x1000 || (rem == 0x1000 && (half & 1)))
    half++;
  return uint16_t(sign | half);
}

template <typename Fill>
void fill_tensor(ggml_tensor* t, ggml_backend_t backend, Fill fill) {
  std::vector<uint16_t> host(ggml_nbytes(t) / sizeof(uint16_t));
  fill(host);
  ggml_backend_tensor_set(t, host.data(), 0, ggml_nbytes(t));
}

struct CaseResult {
  std::string name;
  std::string op; // mul_mat_f16 | mul_mat_q4k | flash_attn
  int64_t m = 0, k = 0, n = 0;
  int64_t kv = 0, h = 0, hkv = 0, d = 0;
  double median_us = 0, min_us = 0, max_us = 0;
  double flops = 0;
  std::vector<double>
      samples; // pooled across balanced order before aggregation
};

// ggml mul_mat(A[M,K], B[N,K]^T) with row-major weights: weights are the
// B operand typed q4_K or f16, shape [K, N] in ggml ne order.
CaseResult run_mul_mat(
    ggml_backend_t backend,
    const std::string& name,
    bool quantized,
    int64_t m,
    int64_t k,
    int64_t n,
    int warmup,
    int reps,
    std::mt19937& rng) {
  ggml_type wtype = quantized ? GGML_TYPE_Q4_K : GGML_TYPE_F16;
  size_t meta = ggml_tensor_overhead() * 16 + ggml_graph_overhead();
  struct ggml_init_params ip{};
  ip.mem_size = meta;
  ip.mem_buffer = nullptr;
  ip.no_alloc = true;
  ggml_context* ctx = ggml_init(ip);
  if (!ctx)
    die("ggml_init failed", 1);

  // Activations f16 [K, M]; weights [K, N]; out f32 [N, M].
  ggml_tensor* a = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, k, m);
  ggml_tensor* w = ggml_new_tensor_2d(ctx, wtype, k, n);
  struct ggml_cgraph* gf = ggml_new_graph(ctx);
  // llama.cpp convention (llama-graph.cpp: "ggml_mul_mat(ctx0, w, cur)"):
  // weights are the first operand, activations the second. Result is
  // [N, M]: N outputs per token row.
  ggml_tensor* out = ggml_mul_mat(ctx, w, a);
  ggml_build_forward_expand(gf, out);
  ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
  if (!buf)
    die("backend buffer allocation failed", 1);

  fill_tensor(a, backend, [&](std::vector<uint16_t>& host) {
    std::uniform_real_distribution<float> uni(-1.0f, 1.0f);
    for (auto& v : host)
      v = f32_to_f16(uni(rng));
  });
  if (quantized) {
    // Random Q4_K payload; values only affect magnitudes, not timing.
    std::vector<uint8_t> bytes(ggml_nbytes(w));
    for (auto& b : bytes)
      b = uint8_t(rng() & 0xFF);
    ggml_backend_tensor_set(w, bytes.data(), 0, ggml_nbytes(w));
  } else {
    fill_tensor(w, backend, [&](std::vector<uint16_t>& host) {
      std::uniform_real_distribution<float> uni(-1.0f, 1.0f);
      for (auto& v : host)
        v = f32_to_f16(uni(rng));
    });
  }

  for (int i = 0; i < warmup; i++) {
    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
      die("ggml graph compute failed for " + name, 1);
    }
    ggml_backend_synchronize(backend);
  }
  std::vector<double> samples;
  samples.reserve(reps);
  for (int i = 0; i < reps; i++) {
    auto t0 = std::chrono::steady_clock::now();
    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
      die("ggml graph compute failed for " + name, 1);
    }
    // Explicit: wall time must cover GPU completion, not enqueue.
    ggml_backend_synchronize(backend);
    auto t1 = std::chrono::steady_clock::now();
    samples.push_back(
        std::chrono::duration<double, std::micro>(t1 - t0).count());
  }

  CaseResult r;
  r.name = name;
  r.op = quantized ? "mul_mat_q4k" : "mul_mat_f16";
  r.m = m;
  r.k = k;
  r.n = n;
  r.flops = 2.0 * double(m) * double(k) * double(n);
  r.samples = std::move(samples);
  ggml_backend_buffer_free(buf);
  ggml_free(ctx);
  return r;
}

CaseResult run_flash_attn(
    ggml_backend_t backend,
    const std::string& name,
    int64_t m,
    int64_t kv,
    int64_t h,
    int64_t hkv,
    int64_t d,
    int warmup,
    int reps,
    std::mt19937& rng) {
  size_t meta = ggml_tensor_overhead() * 16 + ggml_graph_overhead();
  struct ggml_init_params ip{};
  ip.mem_size = meta;
  ip.mem_buffer = nullptr;
  ip.no_alloc = true;
  ggml_context* ctx = ggml_init(ip);
  if (!ctx)
    die("ggml_init failed", 1);

  ggml_tensor* q = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, d, m, h);
  ggml_tensor* k = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, d, kv, hkv);
  ggml_tensor* v = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, d, kv, hkv);
  ggml_tensor* mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, kv, m);
  struct ggml_cgraph* gf = ggml_new_graph(ctx);
  ggml_tensor* out = ggml_flash_attn_ext(
      ctx, q, k, v, mask, 1.0f / std::sqrt((float)d), 0.0f, 0.0f);
  ggml_build_forward_expand(gf, out);
  ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
  if (!buf)
    die("backend buffer allocation failed", 1);

  auto uni_fill = [&](std::vector<uint16_t>& host) {
    std::uniform_real_distribution<float> uni(-1.0f, 1.0f);
    for (auto& v : host)
      v = f32_to_f16(uni(rng));
  };
  fill_tensor(q, backend, uni_fill);
  fill_tensor(k, backend, uni_fill);
  fill_tensor(v, backend, uni_fill);
  if (m == 1) {
    // Decode: the single query attends the whole KV cache (causal=false):
    // an all-zero mask row permits every cached token.
    std::vector<uint16_t> zeros(size_t(m) * kv, 0);
    ggml_backend_tensor_set(mask, zeros.data(), 0, ggml_nbytes(mask));
  } else {
    // Causal mask [KV, M]: 0 when t <= row else -INF.
    std::vector<uint16_t> host(size_t(m) * kv);
    const uint16_t neg_inf = 0xFC00;
    for (int64_t row = 0; row < m; row++) {
      for (int64_t t = 0; t < kv; t++) {
        host[size_t(row) * kv + t] = (t <= row) ? uint16_t(0) : neg_inf;
      }
    }
    ggml_backend_tensor_set(mask, host.data(), 0, ggml_nbytes(mask));
  }

  for (int i = 0; i < warmup; i++) {
    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
      die("ggml graph compute failed for " + name, 1);
    }
    ggml_backend_synchronize(backend);
  }
  std::vector<double> samples;
  samples.reserve(reps);
  for (int i = 0; i < reps; i++) {
    auto t0 = std::chrono::steady_clock::now();
    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
      die("ggml graph compute failed for " + name, 1);
    }
    ggml_backend_synchronize(backend);
    auto t1 = std::chrono::steady_clock::now();
    samples.push_back(
        std::chrono::duration<double, std::micro>(t1 - t0).count());
  }

  CaseResult r;
  r.name = name;
  r.op = "flash_attn_ext";
  r.m = m;
  r.kv = kv;
  r.h = h;
  r.hkv = hkv;
  r.d = d;
  r.flops =
      2.0 * 2.0 * double(h) * double(d) * (double(m) * (double(m) + 1.0) / 2.0);
  if (m == 1) {
    r.flops = 2.0 * 2.0 * double(h) * double(d) * double(kv);
  }
  r.samples = std::move(samples);
  ggml_backend_buffer_free(buf);
  ggml_free(ctx);
  return r;
}

} // namespace

int main(int argc, char** argv) {
  std::string pin_commit;
  std::string output_path;
  int warmup = 5;
  int reps = 30;
  int rounds = 10;

  for (int i = 1; i < argc; i++) {
    std::string a = argv[i];
    if (a == "--pin-commit") {
      pin_commit = arg_value(argc, argv, i, "--pin-commit");
    } else if (a == "--output") {
      output_path = arg_value(argc, argv, i, "--output");
    } else if (a == "--warmup") {
      warmup = std::stoi(arg_value(argc, argv, i, "--warmup"));
    } else if (a == "--reps") {
      reps = std::stoi(arg_value(argc, argv, i, "--reps"));
    } else if (a == "--rounds") {
      rounds = std::stoi(arg_value(argc, argv, i, "--rounds"));
    } else {
      die("unknown argument: " + a +
              " (usage: ggml_op_probe [--pin-commit HASH] [--output FILE] [--warmup N] [--reps N] [--rounds N])",
          2);
    }
  }
  if (rounds < 1 || warmup < 0 || reps < 1)
    die("--rounds >= 1, --reps >= 1, --warmup >= 0 required", 2);

  ggml_backend_load_all();
  ggml_backend_dev_t dev = nullptr;
  for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
    auto candidate = ggml_backend_dev_get(i);
    if (std::strstr(ggml_backend_dev_name(candidate), "Vulkan")) {
      dev = candidate;
      break;
    }
  }
  if (!dev) {
    die("no Vulkan device registered in ggml. Build the pinned llama.cpp "
        "with GGML_VULKAN=ON and point the runner at that build directory.",
        1);
  }
  ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
  if (!backend) {
    die("ggml Vulkan backend init failed", 1);
  }

  std::mt19937 rng(20260830);
  // Names equal the spike row names (matched shapes, Qwen3.8-2B
  // full-attention layer set). Each spec re-runs its case with identical
  // warmup/reps, so the only variable between rounds is order position.
  struct CaseSpec {
    std::function<CaseResult(int, int)> run;
  };
  std::vector<CaseSpec> specs;
  specs.push_back({[&](int w, int r) {
    return run_mul_mat(
        backend, "prefill_gemm_qkv", false, 512, 2048, 5120, w, r, rng);
  }});
  specs.push_back({[&](int w, int r) {
    return run_mul_mat(
        backend, "prefill_gemm_o", false, 512, 4096, 2048, w, r, rng);
  }});
  specs.push_back({[&](int w, int r) {
    return run_mul_mat(
        backend, "prefill_gemm_mlp_up", false, 512, 2048, 12288, w, r, rng);
  }});
  specs.push_back({[&](int w, int r) {
    return run_mul_mat(
        backend, "prefill_gemm_mlp_down", false, 512, 6144, 2048, w, r, rng);
  }});
  specs.push_back({[&](int w, int r) {
    return run_mul_mat(
        backend, "decode_gemv_q4k_qkv", true, 1, 2048, 5120, w, r, rng);
  }});
  specs.push_back({[&](int w, int r) {
    return run_mul_mat(
        backend, "decode_gemv_q4k_o", true, 1, 4096, 2048, w, r, rng);
  }});
  specs.push_back({[&](int w, int r) {
    return run_mul_mat(
        backend, "decode_gemv_q4k_mlp_up", true, 1, 2048, 12288, w, r, rng);
  }});
  specs.push_back({[&](int w, int r) {
    return run_mul_mat(
        backend, "decode_gemv_q4k_mlp_down", true, 1, 6144, 2048, w, r, rng);
  }});
  specs.push_back({[&](int w, int r) {
    return run_flash_attn(
        backend, "prefill_sdpa", 512, 512, 16, 2, 256, w, r, rng);
  }});
  specs.push_back({[&](int w, int r) {
    return run_flash_attn(
        backend, "decode_sdpa", 1, 512, 16, 2, 256, w, r, rng);
  }});

  // One complete rotation places every case in every order position once.
  const bool position_balance_complete =
      rounds % static_cast<int>(specs.size()) == 0;
  std::vector<CaseResult> cases;
  for (int round = 0; round < rounds; round++) {
    for (size_t slot = 0; slot < specs.size(); slot++) {
      const CaseSpec& spec = specs[(slot + round) % specs.size()];
      CaseResult r = spec.run(warmup, reps);
      if (round == 0) {
        cases.push_back(std::move(r));
      } else {
        CaseResult& agg = cases[(slot + size_t(round)) % specs.size()];
        if (agg.name != r.name)
          die("case order bookkeeping mismatch: " + agg.name + " vs " + r.name,
              1);
        agg.samples.insert(
            agg.samples.end(), r.samples.begin(), r.samples.end());
      }
    }
  }
  for (auto& c : cases) {
    std::sort(c.samples.begin(), c.samples.end());
    c.min_us = c.samples.front();
    c.max_us = c.samples.back();
    c.median_us = c.samples[c.samples.size() / 2];
  }
  ggml_backend_free(backend);

  std::ostringstream os;
  char host[256] = {0};
  if (gethostname(host, sizeof(host) - 1) != 0) {
    snprintf(host, sizeof(host), "unknown");
  }
  os << "{\n";
  os << "  \"comparator\": \"pinned ggml Vulkan per-op timing\",\n";
  os << "  \"pin_commit\": \"" << pin_commit << "\",\n";
  os << "  \"device\": \"" << ggml_backend_dev_name(dev) << "\",\n";
  os << "  \"device_description\": \"" << ggml_backend_dev_description(dev)
     << "\",\n";
  os << "  \"machine\": \"" << host << "\",\n";
  os << "  \"timing_method\": \"host wall clock around "
        "ggml_backend_graph_compute followed by ggml_backend_synchronize; "
        "case list rotated one slot per round; per-case samples pooled after "
        "equal warmup/reps per round; full positional balance requires rounds "
        "to be a multiple of case count\",\n";
  os << "  \"capture\": \""
     << (position_balance_complete ? "balanced-rotation" : "partial-rotation")
     << "\",\n";
  os << "  \"position_balance_complete\": "
     << (position_balance_complete ? "true" : "false") << ",\n";
  os << "  \"rounds\": " << rounds << ",\n";
  os << "  \"warmup\": " << warmup << ",\n";
  os << "  \"repetitions\": " << reps << ",\n";
  os << "  \"test_backend_ops_note\": \"test-backend-ops -r perf uses a fixed internal shape matrix that does not expose these dimensions, so the ops are measured directly at matched shapes\",\n";
  os << "  \"llama_bench_note\": \"llama-bench pp512/tg128 throughput stays in llama_cpp_reference.json as model-level context only\",\n";
  os << "  \"cases\": [\n";
  for (size_t i = 0; i < cases.size(); i++) {
    auto& c = cases[i];
    os << "    {\"name\": \"" << c.name << "\", \"op\": \"" << c.op << "\"";
    if (c.op == "flash_attn_ext") {
      os << ", \"M\": " << c.m << ", \"KV\": " << c.kv << ", \"H\": " << c.h
         << ", \"HKV\": " << c.hkv << ", \"D\": " << c.d
         << ", \"causal\": " << (c.m > 1 ? "true" : "false");
    } else {
      os << ", \"M\": " << c.m << ", \"K\": " << c.k << ", \"N\": " << c.n;
    }
    os << ", \"flops\": " << c.flops << ", \"median_us\": " << c.median_us
       << ", \"min_us\": " << c.min_us << ", \"max_us\": " << c.max_us << "}"
       << (i + 1 < cases.size() ? "," : "") << "\n";
  }
  os << "  ]\n}\n";

  std::string text = os.str();
  if (!output_path.empty()) {
    FILE* f = fopen(output_path.c_str(), "w");
    if (!f) {
      die("cannot write " + output_path, 1);
    }
    fputs(text.c_str(), f);
    fclose(f);
  }
  fputs(text.c_str(), stdout);
  return 0;
}
