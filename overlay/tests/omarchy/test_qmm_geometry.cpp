// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

// MLX_OMARCHY_QMM_GEOM: the decode Qmm column geometry. Every default-eligible
// geometry (2x4, 2x4w2, 2x8w2, 2x16, 2x16w2) runs the same strided per-lane
// chain per column, so the bytes must be identical to the base 2x4 kernels;
// only the columns and threads per workgroup change.
//
// Experimental 2x16kcont / 2x16kcs16 assign contiguous K-chunks per lane
// (qmm_vec.comp QMM_KCONT), which reassociates FP32 per-lane sums at K >= 1024
// (+/-1 bf16 ULP, flipping exact greedy ties on Qwen3.5-9B). They are marked
// not bit-identical and excluded from every default selection.
//
// The geometry kernels need a 32-lane arithmetic subgroup, so on a device with
// smaller subgroups every scenario takes the base kernels and the byte checks
// pass trivially; on qualifying devices (e.g. Apple G13/G14/G18, discrete
// 32-lane Vulkan parts) they exercise the real variants. An unknown value must
// refuse loudly.
//
// The flag is read once per process, so each scenario runs in a child
// process (this binary re-execs itself) and the parent compares the bytes.

#include <dlfcn.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "mlx/backend/omarchy/allocator.h"
#include "mlx/backend/omarchy/device.h"
#include "mlx/backend/omarchy/device_local_activations.h"
#include "mlx/backend/omarchy/fused_chain.h"
#include "mlx/backend/gpu/device_info.h"
#include "mlx/fast.h"
#include "mlx/ops.h"
#include "mlx/stream.h"
#include "mlx/transforms.h"

using namespace mlx::core;

namespace {

struct Lin {
  array w;
  array s;
  array b;
};

Lin make_lin(int n, int k, std::mt19937& rng) {
  std::normal_distribution<float> nd(0.0f, 0.05f);
  std::vector<float> host(static_cast<size_t>(n) * k);
  for (auto& v : host) {
    v = nd(rng);
  }
  array w = array(host.data(), Shape{n, k}, float32);
  auto parts = quantize(w, 64, 4);
  return Lin{parts[0], astype(parts[1], bfloat16), astype(parts[2], bfloat16)};
}

array qmm(const array& x, const Lin& l) {
  return quantized_matmul(x, l.w, l.s, l.b, true, 64, 4);
}

array make_x(int k, std::mt19937& rng) {
  std::normal_distribution<float> nd(0.0f, 0.6f);
  std::vector<float> host(k);
  for (auto& v : host) {
    v = nd(rng);
  }
  host[17] = 3000.0f;
  return astype(
      array(host.data(), Shape{1, 1, k}, float32), bfloat16);
}

// One scenario: build the decode-shaped graphs and evaluate everything.
int scenario(const std::string& name, const std::string& out) {
  set_default_device(Device::gpu);
  if (!gpu::is_available()) {
    std::ofstream f(out);
    f << "skip";
    return 0;
  }
  Stream stream = new_stream(Device::gpu);
  std::mt19937 rng(20261104);
  std::vector<array> outs;
  if (name == "single") {
    auto x = make_x(2048, rng);
    auto l = make_lin(6144, 2048, rng);
    eval(x, l.w, l.s, l.b);
    auto o = qmm(x, l);
    eval(o);
    outs.push_back(o);
    auto small = make_lin(100, 2048, rng);
    eval(small.w, small.s, small.b);
    auto o2 = qmm(x, small);
    eval(o2);
    outs.push_back(o2);
  } else if (name == "group") {
    auto x = make_x(2048, rng);
    auto a = make_lin(4096, 2048, rng);
    auto b = make_lin(512, 2048, rng);
    auto c = make_lin(512, 2048, rng);
    eval(x, a.w, a.s, a.b, b.w, b.s, b.b, c.w, c.s, c.b);
    auto resid = astype(array(0.25f), bfloat16);
    auto oa = add(qmm(x, a), resid);
    auto ob = qmm(x, b);
    auto oc = qmm(x, c);
    eval(oa, ob, oc);
    outs.push_back(oa);
    outs.push_back(ob);
    outs.push_back(oc);
  } else if (name == "qkv3m") {
    auto x = make_x(2048, rng);
    auto a = make_lin(4096, 2048, rng);
    auto b = make_lin(512, 2048, rng);
    auto c = make_lin(512, 2048, rng);
    eval(x, a.w, a.s, a.b, b.w, b.s, b.b, c.w, c.s, c.b);
    auto oa = qmm(x, a);
    auto ob = qmm(x, b);
    auto oc = qmm(x, c);
    eval(oa, ob, oc);
    outs.push_back(oa);
    outs.push_back(ob);
    outs.push_back(oc);
  } else if (name == "swiglu") {
    auto x = make_x(2048, rng);
    auto gate = make_lin(6144, 2048, rng);
    auto up = make_lin(6144, 2048, rng);
    auto down = make_lin(2048, 6144, rng);
    eval(x, gate.w, gate.s, gate.b, up.w, up.s, up.b, down.w, down.s, down.b);
    auto g = qmm(x, gate);
    auto act = multiply(multiply(g, sigmoid(g)), qmm(x, up));
    auto d = qmm(act, down);
    eval(act, d);
    outs.push_back(act);
    outs.push_back(d);
  } else if (name == "k512" || name == "k768" || name == "k1024") {
    const int k = (name == "k512") ? 512 : ((name == "k768") ? 768 : 1024);
    auto x = make_x(k, rng);
    auto a = make_lin(4096, k, rng);
    auto b = make_lin(1024, k, rng);
    eval(x, a.w, a.s, a.b, b.w, b.s, b.b);
    auto oa = qmm(x, a);
    auto ob = qmm(x, b);
    eval(oa, ob);
    outs.push_back(oa);
    outs.push_back(ob);
  } else if (name == "resnorm2048" || name == "resnorm4096" ||
             name == "resnorm2048_liveqmm") {
    const int n = (name == "resnorm4096") ? 4096 : 2048;
    const int k = 2048;
    auto x = make_x(k, rng);
    auto w = make_lin(n, k, rng);
    auto resid = make_x(n, rng);
    auto nw = make_x(n, rng);
    nw = reshape(nw, Shape{n});
    eval(x, w.w, w.s, w.b, resid, nw);
    auto q = qmm(x, w);
    auto sum = add(q, resid);
    auto normed = fast::rms_norm(sum, nw, 1e-6f);
    if (name == "resnorm2048_liveqmm") {
      eval(q, sum, normed);
      outs.push_back(q);
    } else {
      eval(sum, normed);
    }
    outs.push_back(sum);
    outs.push_back(normed);
  } else if (name == "name") {
    std::ofstream f(out);
    f << mlx::core::omarchy::qmm_geometry_name();
    return 0;
  } else {
    return 2;
  }
  std::ofstream f(out, std::ios::binary);
  for (const auto& o : outs) {
    array bits = astype(view(o, uint16), uint32);
    eval(bits);
    const uint32_t* p = bits.data<uint32_t>();
    f.write(
        reinterpret_cast<const char*>(p),
        static_cast<std::streamsize>(bits.size() * sizeof(uint32_t)));
  }
  return 0;
}

struct Bf16Diff {
  size_t count{0};
  int32_t max_ulp{0};
};

int32_t bf16_ordered(uint16_t u) {
  int32_t v = static_cast<int32_t>(u);
  return (v & 0x8000) != 0 ? (0x8000 - (v & 0x7fff)) : (0x8000 + v);
}

Bf16Diff diff_u32_bf16_files(const std::string& path_a, const std::string& path_b) {
  std::ifstream fa(path_a, std::ios::binary);
  std::ifstream fb(path_b, std::ios::binary);
  std::vector<uint32_t> a, b;
  uint32_t v = 0;
  while (fa.read(reinterpret_cast<char*>(&v), sizeof(v))) {
    a.push_back(v);
  }
  while (fb.read(reinterpret_cast<char*>(&v), sizeof(v))) {
    b.push_back(v);
  }
  Bf16Diff d;
  if (a.size() != b.size()) {
    d.count = a.size() > b.size() ? a.size() - b.size() : b.size() - a.size();
    d.max_ulp = INT32_MAX;
    return d;
  }
  for (size_t i = 0; i < a.size(); ++i) {
    if (a[i] != b[i]) {
      ++d.count;
      int32_t ulp = std::abs(
          bf16_ordered(static_cast<uint16_t>(a[i])) -
          bf16_ordered(static_cast<uint16_t>(b[i])));
      if (ulp > d.max_ulp) {
        d.max_ulp = ulp;
      }
    }
  }
  return d;
}

uint64_t fnv(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  uint64_t h = 1469598103934665603ull;
  char buf[65536];
  while (f.read(buf, sizeof(buf)) || f.gcount() > 0) {
    for (std::streamsize i = 0; i < f.gcount(); ++i) {
      h ^= static_cast<unsigned char>(buf[i]);
      h *= 1099511628211ull;
    }
    if (!f) {
      break;
    }
  }
  return h;
}

int run_child(const char* name, const char* out) {
  std::string binary = "/proc/self/exe";
  std::vector<char*> argv{
      const_cast<char*>(binary.c_str()),
      const_cast<char*>("--child"),
      const_cast<char*>(name),
      const_cast<char*>(out),
      nullptr};
  pid_t pid = fork();
  if (pid == 0) {
    execv(binary.c_str(), argv.data());
    _exit(127);
  }
  int status = 0;
  waitpid(pid, &status, 0);
  if (!WIFEXITED(status)) {
    return -1;
  }
  return WEXITSTATUS(status);
}

} // namespace

int main(int argc, char** argv) {
  if (argc == 4 && std::strcmp(argv[1], "--child") == 0) {
    try {
      return scenario(argv[2], argv[3]);
    } catch (const std::exception& e) {
      std::cerr << "child threw: " << e.what() << "\n";
      return 3;
    }
  }
  set_default_device(Device::gpu);
  if (!gpu::is_available()) {
    std::cout << "Skipping: no qualifying Vulkan device.\n";
    return 0;
  }
  const auto& caps = mlx::core::omarchy::device().capabilities();
  std::cout << "device " << caps.device_name << " subgroup "
            << caps.subgroup_size << "\n";
  int failures = 0;
  // An unknown geometry value must refuse loudly where the geometry is read
  // (the probe reads it; a device without the 32-lane subgroup column never
  // consults the geometry at all).
  {
    setenv("MLX_OMARCHY_QMM_GEOM", "9x9", 1);
    int rc = run_child("name", "/tmp/w7p-geom-invalid.bin");
    unsetenv("MLX_OMARCHY_QMM_GEOM");
    if (rc != 3) {
      std::cout << "FAIL: an invalid geometry value did not refuse (rc "
                << rc << ")\n";
      ++failures;
    }
  }
  // The reported name follows the chip default (the parent process never
  // sets the override, so the child inherits the unset environment).
  {
    const auto& dev_obj = mlx::core::omarchy::device();
    const auto& caps_ref = dev_obj.capabilities();
    const std::string& dev = caps_ref.device_name;
    const mlx::core::omarchy::QmmWideCaps probe_caps{
        caps_ref.subgroup_size,
        caps_ref.subgroup_operations,
        caps_ref.max_compute_work_group_invocations,
        caps_ref.max_compute_work_group_size[0]};
    const bool vram = mlx::core::omarchy::device_local_activations_enabled();
    std::string want = mlx::core::omarchy::select_default_qmm_geometry_name(
        dev, dev_obj.physical_type(), probe_caps, vram);
    int rc = run_child("name", "/tmp/w7p-geom-name.bin");
    std::ifstream f("/tmp/w7p-geom-name.bin");
    std::string got;
    std::getline(f, got);
    if (rc != 0 || got != want) {
      std::cout << "FAIL: default geometry name is " << got << ", want "
                << want << " (rc " << rc << ")\n";
      ++failures;
    }
    if (got == "2x16kcont" || got == "2x16kcs16") {
      std::cout << "FAIL: non-bit-identical kcont geometry selected by default: "
                << got << "\n";
      ++failures;
    }
  }
  // Every default-eligible geometry must produce the exact 2x4 bytes.
  for (const char* geometry : {"2x8w2", "2x16", "2x16w2", "2x4w2", "2x4"}) {
    for (const char* name : {"single", "group", "swiglu", "k512", "k768", "k1024"}) {
      std::string base = "/tmp/w7p-geom-base.bin";
      std::string other = "/tmp/w7p-geom-other.bin";
      setenv("MLX_OMARCHY_QMM_GEOM", "2x4", 1);
      int rc = run_child(name, base.c_str());
      setenv("MLX_OMARCHY_QMM_GEOM", geometry, 1);
      int rc2 = run_child(name, other.c_str());
      unsetenv("MLX_OMARCHY_QMM_GEOM");
      if (rc != 0 || rc2 != 0) {
        std::cout << "FAIL: " << geometry << " " << name << " child rc "
                  << rc << "/" << rc2 << "\n";
        ++failures;
        continue;
      }
      if (fnv(base) != fnv(other)) {
        std::cout << "FAIL: " << geometry << " " << name
                  << " bytes differ from 2x4\n";
        ++failures;
      } else {
        std::cout << "ok: " << geometry << " " << name << " identical\n";
      }
    }
  }
  // Experimental 2x16kcont is NOT bit-identical to 2x4 for K >= 1024
  // (FP32 lane-chunk reassociation -> +/-1 bf16 ULP), and is excluded from
  // default selection above. Verify it remains exact at K=512 (Q4_FAST_ITERS=1)
  // and K=768 (Q4_KCONT_OK=0 fallback), and stays within 1 bf16 ULP at K=1024.
  for (const char* name : {"k512", "k768", "k1024"}) {
    std::string base = "/tmp/w7p-geom-base.bin";
    std::string other = "/tmp/w7p-geom-other.bin";
    setenv("MLX_OMARCHY_QMM_GEOM", "2x4", 1);
    int rc = run_child(name, base.c_str());
    setenv("MLX_OMARCHY_QMM_GEOM", "2x16kcont", 1);
    int rc2 = run_child(name, other.c_str());
    unsetenv("MLX_OMARCHY_QMM_GEOM");
    if (rc != 0 || rc2 != 0) {
      std::cout << "FAIL: 2x16kcont " << name << " child rc "
                << rc << "/" << rc2 << "\n";
      ++failures;
      continue;
    }
    const Bf16Diff d = diff_u32_bf16_files(base, other);
    if (std::strcmp(name, "k1024") != 0) {
      if (d.count != 0) {
        std::cout << "FAIL: 2x16kcont " << name
                  << " differed from 2x4 where strided/fallback must match (diffs="
                  << d.count << " max_ulp=" << d.max_ulp << ")\n";
        ++failures;
      } else {
        std::cout << "ok: 2x16kcont " << name << " identical\n";
      }
    } else if (d.max_ulp > 1) {
      std::cout << "FAIL: 2x16kcont k1024 exceeded 1 bf16 ULP reassociation bound (diffs="
                << d.count << " max_ulp=" << d.max_ulp << ")\n";
      ++failures;
    } else {
      std::cout << "ok: 2x16kcont k1024 marked non-bit-identical (diffs="
                << d.count << " max_ulp=" << d.max_ulp << ")\n";
    }
  }
  // Part-default checks for G14C low-register SwiGLU (R1S8W2) and coalesced
  // uvec4 ResNorm (R2S16W2).
  {
    if (!mlx::core::omarchy::qmm_swiglu_r1_default("Apple M2 Max (G14C B1)") ||
        !mlx::core::omarchy::qmm_swiglu_r1_default("Apple M1 (G13G B1)") ||
        mlx::core::omarchy::qmm_swiglu_r1_default("Apple M1 Max (G13C C0)") ||
        mlx::core::omarchy::qmm_swiglu_r1_default("llvmpipe (LLVM 15.0.6, 256 bits)")) {
      std::cout << "FAIL: qmm_swiglu_r1_default part check\n";
      ++failures;
    }
    if (!mlx::core::omarchy::qmm_resnorm_vec4_default("Apple M2 Max (G14C B1)") ||
        !mlx::core::omarchy::qmm_resnorm_vec4_default("Apple M1 (G13G B1)") ||
        mlx::core::omarchy::qmm_resnorm_vec4_default("Apple M1 Max (G13C C0)") ||
        mlx::core::omarchy::qmm_resnorm_vec4_default("llvmpipe (LLVM 15.0.6, 256 bits)")) {
      std::cout << "FAIL: qmm_resnorm_vec4_default part check\n";
      ++failures;
    }
    if (!mlx::core::omarchy::qmm_resnorm_r2s32_default("Apple M2 Max (G14C B1)") ||
        mlx::core::omarchy::qmm_resnorm_r2s32_default("Apple M1 (G13G B1)") ||
        mlx::core::omarchy::qmm_resnorm_r2s32_default("Apple M1 Max (G13C C0)") ||
        mlx::core::omarchy::qmm_resnorm_r2s32_default("llvmpipe (LLVM 15.0.6, 256 bits)")) {
      std::cout << "FAIL: qmm_resnorm_r2s32_default part check\n";
      ++failures;
    }
    if (!mlx::core::omarchy::qmm_lm_r2s32_default("Apple M2 Max (G14C B1)") ||
        !mlx::core::omarchy::qmm_lm_r2s32_default("Apple M1 (G13G B1)") ||
        mlx::core::omarchy::qmm_lm_r2s32_default("Apple M1 Max (G13C C0)") ||
        mlx::core::omarchy::qmm_lm_r2s32_default("llvmpipe (LLVM 15.0.6, 256 bits)")) {
      std::cout << "FAIL: qmm_lm_r2s32_default part check\n";
      ++failures;
    }
    if (!mlx::core::omarchy::qmm_swiglu_r1s4_default("Apple M2 Max (G14C B1)") ||
        mlx::core::omarchy::qmm_swiglu_r1s4_default("Apple M1 Max (G13C C0)") ||
        mlx::core::omarchy::qmm_swiglu_r1s4_default("llvmpipe (LLVM 15.0.6, 256 bits)")) {
      std::cout << "FAIL: qmm_swiglu_r1s4_default part check\n";
      ++failures;
    }
    if (!mlx::core::omarchy::qmm_word_r2s4_default("Apple M2 Max (G14C B1)") ||
        !mlx::core::omarchy::qmm_word_r2s4_default("Apple M1 (G13G B1)") ||
        mlx::core::omarchy::qmm_word_r2s4_default("Apple M1 Max (G13C C0)") ||
        mlx::core::omarchy::qmm_word_r2s4_default("llvmpipe (LLVM 15.0.6, 256 bits)")) {
      std::cout << "FAIL: qmm_word_r2s4_default part check\n";
      ++failures;
    }
    if (!mlx::core::omarchy::qmm_multi_upfront_default("Apple M2 Max (G14C B1)") ||
        !mlx::core::omarchy::qmm_multi_upfront_default("Apple M1 (G13G B1)") ||
        mlx::core::omarchy::qmm_multi_upfront_default("Apple M1 Max (G13C C0)") ||
        mlx::core::omarchy::qmm_multi_upfront_default("llvmpipe (LLVM 15.0.6, 256 bits)")) {
      std::cout << "FAIL: qmm_multi_upfront_default part check\n";
      ++failures;
    }
    unsetenv("MLX_OMARCHY_QMM_FAST_ONLY");
    if (!mlx::core::omarchy::qmm_fast_only_default_part("Apple M1 Max (G13C C0)") ||
        !mlx::core::omarchy::qmm_fast_only_default_part("Apple M1 (G13G B1)") ||
        mlx::core::omarchy::qmm_fast_only_default_part("Apple M2 Max (G14C B1)") ||
        mlx::core::omarchy::qmm_fast_only_default_part("llvmpipe (LLVM 15.0.6, 256 bits)")) {
      std::cout << "FAIL: qmm_fast_only_default_part part check\n";
      ++failures;
    }
    setenv("MLX_OMARCHY_QMM_FAST_ONLY", "0", 1);
    if (mlx::core::omarchy::qmm_fast_only_default_part("Apple M1 (G13G B1)")) {
      std::cout << "FAIL: qmm_fast_only_default_part =0 override\n";
      ++failures;
    }
    setenv("MLX_OMARCHY_QMM_FAST_ONLY", "1", 1);
    if (!mlx::core::omarchy::qmm_fast_only_default_part("Apple M2 Max (G14C B1)")) {
      std::cout << "FAIL: qmm_fast_only_default_part =1 override\n";
      ++failures;
    }
    unsetenv("MLX_OMARCHY_QMM_FAST_ONLY");
    unsetenv("MLX_OMARCHY_G14C_FAST_ONLY");
    if (!mlx::core::omarchy::qmm_g14c_fast_only_default_part("Apple M2 Max (G14C B1)") ||
        !mlx::core::omarchy::qmm_g14c_fast_only_default_part("Apple M1 (G13G B1)") ||
        mlx::core::omarchy::qmm_g14c_fast_only_default_part("Apple M1 Max (G13C C0)") ||
        mlx::core::omarchy::qmm_g14c_fast_only_default_part("llvmpipe (LLVM 15.0.6, 256 bits)")) {
      std::cout << "FAIL: qmm_g14c_fast_only_default_part part check\n";
      ++failures;
    }
    setenv("MLX_OMARCHY_G14C_FAST_ONLY", "0", 1);
    if (mlx::core::omarchy::qmm_g14c_fast_only_default_part("Apple M2 Max (G14C B1)") ||
        mlx::core::omarchy::qmm_g14c_fast_only_default_part("Apple M1 (G13G B1)")) {
      std::cout << "FAIL: qmm_g14c_fast_only_default_part =0 override\n";
      ++failures;
    }
    setenv("MLX_OMARCHY_G14C_FAST_ONLY", "1", 1);
    if (!mlx::core::omarchy::qmm_g14c_fast_only_default_part("Apple M1 Max (G13C C0)")) {
      std::cout << "FAIL: qmm_g14c_fast_only_default_part =1 override\n";
      ++failures;
    }
    unsetenv("MLX_OMARCHY_G14C_FAST_ONLY");
    unsetenv("MLX_OMARCHY_MULTI_R2S16");
    if (!mlx::core::omarchy::qmm_g14c_multi_r2s16_default_part("Apple M2 Max (G14C B1)") ||
        mlx::core::omarchy::qmm_g14c_multi_r2s16_default_part("Apple M1 Max (G13C C0)") ||
        mlx::core::omarchy::qmm_g14c_multi_r2s16_default_part("Apple M1 (G13G B1)") ||
        mlx::core::omarchy::qmm_g14c_multi_r2s16_default_part("llvmpipe (LLVM 15.0.6, 256 bits)")) {
      std::cout << "FAIL: qmm_g14c_multi_r2s16_default_part part check\n";
      ++failures;
    }
    setenv("MLX_OMARCHY_MULTI_R2S16", "0", 1);
    if (mlx::core::omarchy::qmm_g14c_multi_r2s16_default_part("Apple M2 Max (G14C B1)")) {
      std::cout << "FAIL: qmm_g14c_multi_r2s16_default_part =0 override\n";
      ++failures;
    }
    setenv("MLX_OMARCHY_MULTI_R2S16", "1", 1);
    if (!mlx::core::omarchy::qmm_g14c_multi_r2s16_default_part("Apple M1 Max (G13C C0)")) {
      std::cout << "FAIL: qmm_g14c_multi_r2s16_default_part =1 override\n";
      ++failures;
    }
    unsetenv("MLX_OMARCHY_MULTI_R2S16");
    unsetenv("MLX_OMARCHY_SDPA_HD256_FAST");
    if (!mlx::core::omarchy::sdpa_g14c_hd256_fast_default_part("Apple M2 Max (G14C B1)") ||
        mlx::core::omarchy::sdpa_g14c_hd256_fast_default_part("Apple M1 Max (G13C C0)") ||
        mlx::core::omarchy::sdpa_g14c_hd256_fast_default_part("llvmpipe (LLVM 15.0.6, 256 bits)")) {
      std::cout << "FAIL: sdpa_g14c_hd256_fast_default_part part check\n";
      ++failures;
    }
    setenv("MLX_OMARCHY_SDPA_HD256_FAST", "0", 1);
    if (mlx::core::omarchy::sdpa_g14c_hd256_fast_default_part("Apple M2 Max (G14C B1)")) {
      std::cout << "FAIL: sdpa_g14c_hd256_fast_default_part =0 override\n";
      ++failures;
    }
    unsetenv("MLX_OMARCHY_SDPA_HD256_FAST");
    unsetenv("MLX_OMARCHY_PROLOGUE_NORM");
    if (!mlx::core::omarchy::prologue_norm_default_part("Apple M2 Max (G14C B1)") ||
        mlx::core::omarchy::prologue_norm_default_part("Apple M1 Max (G13C C0)") ||
        mlx::core::omarchy::prologue_norm_default_part("llvmpipe (LLVM 15.0.6, 256 bits)")) {
      std::cout << "FAIL: prologue_norm_default_part part check\n";
      ++failures;
    }
    setenv("MLX_OMARCHY_PROLOGUE_NORM", "0", 1);
    if (mlx::core::omarchy::prologue_norm_default_part("Apple M2 Max (G14C B1)")) {
      std::cout << "FAIL: prologue_norm_default_part =0 override\n";
      ++failures;
    }
    unsetenv("MLX_OMARCHY_PROLOGUE_NORM");
    unsetenv("MLX_OMARCHY_SDPA_SPLIT_WIDE_MIN_KEYS");
    if (mlx::core::omarchy::sdpa_split_wide_min_keys("Apple M2 Max (G14C B1)") != 768u) {
      std::cout << "FAIL: sdpa_split_wide_min_keys G14C check\n";
      ++failures;
    }
  }
  // Low-register SwiGLU (R1S8W2 and R1S4W2) must be bit-identical to 2x4.
  {
    std::string base = "/tmp/w7p-geom-base.bin";
    std::string other = "/tmp/w7p-geom-other.bin";
    setenv("MLX_OMARCHY_QMM_GEOM", "2x4", 1);
    setenv("MLX_OMARCHY_SWIGLU_R1", "0", 1);
    int rc = run_child("swiglu", base.c_str());
    setenv("MLX_OMARCHY_QMM_GEOM", "2x8w2", 1);
    setenv("MLX_OMARCHY_SWIGLU_R1", "1", 1);
    setenv("MLX_OMARCHY_SWIGLU_R1S4", "0", 1);
    int rc2 = run_child("swiglu", other.c_str());
    if (rc != 0 || rc2 != 0 || fnv(base) != fnv(other)) {
      std::cout << "FAIL: swiglu R1S8W2 differed from 2x4 (rc " << rc << "/" << rc2 << ")\n";
      ++failures;
    } else {
      std::cout << "ok: swiglu R1S8W2 identical\n";
    }
    setenv("MLX_OMARCHY_SWIGLU_R1S4", "1", 1);
    int rc3 = run_child("swiglu", other.c_str());
    unsetenv("MLX_OMARCHY_QMM_GEOM");
    unsetenv("MLX_OMARCHY_SWIGLU_R1");
    unsetenv("MLX_OMARCHY_SWIGLU_R1S4");
    if (rc != 0 || rc3 != 0 || fnv(base) != fnv(other)) {
      std::cout << "FAIL: swiglu R1S4W2 differed from 2x4 (rc " << rc << "/" << rc3 << ")\n";
      ++failures;
    } else {
      std::cout << "ok: swiglu R1S4W2 identical\n";
    }
  }
  // 128-thread Q4 word decode GEMV (R2S4W2) must be bit-identical to 2x4.
  {
    std::string base = "/tmp/w7p-geom-base.bin";
    std::string other = "/tmp/w7p-geom-other.bin";
    setenv("MLX_OMARCHY_QMM_GEOM", "2x4", 1);
    setenv("MLX_OMARCHY_WORD_R2S4", "0", 1);
    int rc = run_child("single", base.c_str());
    setenv("MLX_OMARCHY_QMM_GEOM", "2x8w2", 1);
    setenv("MLX_OMARCHY_WORD_R2S4", "1", 1);
    int rc2 = run_child("single", other.c_str());
    unsetenv("MLX_OMARCHY_QMM_GEOM");
    unsetenv("MLX_OMARCHY_WORD_R2S4");
    if (rc != 0 || rc2 != 0 || fnv(base) != fnv(other)) {
      std::cout << "FAIL: word R2S4W2 differed from 2x4 (rc " << rc << "/" << rc2 << ")\n";
      ++failures;
    } else {
      std::cout << "ok: word R2S4W2 identical\n";
    }
  }
  // Upfront 8-load Multi R2S8W2Upfront must be bit-identical to 2x4.
  {
    std::string base = "/tmp/w7p-geom-base.bin";
    std::string other = "/tmp/w7p-geom-other.bin";
    setenv("MLX_OMARCHY_QMM_GEOM", "2x4", 1);
    setenv("MLX_OMARCHY_Q4_UPFRONT", "0", 1);
    int rc = run_child("group", base.c_str());
    setenv("MLX_OMARCHY_QMM_GEOM", "2x8w2", 1);
    setenv("MLX_OMARCHY_Q4_UPFRONT", "1", 1);
    int rc2 = run_child("group", other.c_str());
    unsetenv("MLX_OMARCHY_QMM_GEOM");
    unsetenv("MLX_OMARCHY_Q4_UPFRONT");
    if (rc != 0 || rc2 != 0 || fnv(base) != fnv(other)) {
      std::cout << "FAIL: multi R2S8W2Upfront differed from 2x4 (rc " << rc << "/" << rc2 << ")\n";
      ++failures;
    } else {
      std::cout << "ok: multi R2S8W2Upfront identical\n";
    }
  }
  // Coalesced uvec4 ResNorm (R2S16W2) must be bit-identical to unfused and
  // scalar ResNorm at both 2048 and 4096 row lengths.
  for (const char* name : {"resnorm2048", "resnorm4096"}) {
    std::string base = "/tmp/w7p-geom-base.bin";
    std::string other = "/tmp/w7p-geom-other.bin";
    setenv("MLX_OMARCHY_QMM_GEOM", "2x8w2", 1);
    setenv("MLX_OMARCHY_QMM_RESIDUAL_NORM", "0", 1);
    unsetenv("MLX_OMARCHY_RESNORM_VEC4");
    int rc = run_child(name, base.c_str());
    setenv("MLX_OMARCHY_QMM_RESIDUAL_NORM", "1", 1);
    setenv("MLX_OMARCHY_RESNORM_VEC4", "1", 1);
    setenv("MLX_OMARCHY_RESNORM_R2S32", "0", 1);
    int rc2 = run_child(name, other.c_str());
    if (rc != 0 || rc2 != 0 || fnv(base) != fnv(other)) {
      std::cout << "FAIL: " << name << " ResNormVec4 R2S16W2 differed from unfused (rc "
                << rc << "/" << rc2 << ")\n";
      ++failures;
    } else {
      std::cout << "ok: " << name << " ResNormVec4 R2S16W2 identical\n";
    }
    setenv("MLX_OMARCHY_RESNORM_R2S32", "1", 1);
    int rc3 = run_child(name, other.c_str());
    unsetenv("MLX_OMARCHY_QMM_GEOM");
    unsetenv("MLX_OMARCHY_QMM_RESIDUAL_NORM");
    unsetenv("MLX_OMARCHY_RESNORM_VEC4");
    unsetenv("MLX_OMARCHY_RESNORM_R2S32");
    if (rc != 0 || rc3 != 0 || fnv(base) != fnv(other)) {
      std::cout << "FAIL: " << name << " ResNormVec4 R2S32W2 differed from unfused (rc "
                << rc << "/" << rc3 << ")\n";
      ++failures;
    } else {
      std::cout << "ok: " << name << " ResNormVec4 R2S32W2 identical\n";
    }
  }
  // G14C fast-tile-only upfront variants must be 0-ULP bit-identical to
  // fallback-carrying upfront variants across Word, Multi, SwiGLU, ResNormVec4.
  {
    struct FoCase {
      const char* label;
      const char* scenario;
      const char* extra_env;
      const char* extra_val;
    };
    const FoCase cases[] = {
        {"word_r2s4_fast", "single", "MLX_OMARCHY_WORD_R2S4", "1"},
        {"multi_up_fast", "group", "MLX_OMARCHY_Q4_UPFRONT", "1"},
        {"swiglu_r1s4_fast", "swiglu", "MLX_OMARCHY_SWIGLU_R1S4", "1"},
        {"swiglu_r1s8_fast", "swiglu", "MLX_OMARCHY_SWIGLU_R1S4", "0"},
        {"resnorm2048_fast", "resnorm2048", "MLX_OMARCHY_RESNORM_R2S32", "1"},
        {"resnorm2048_liveqmm_fast", "resnorm2048_liveqmm", "MLX_OMARCHY_RESNORM_R2S32", "1"},
        {"resnorm4096_fast", "resnorm4096", "MLX_OMARCHY_RESNORM_R2S32", "1"},
        {"resnorm2048_r2s16_fast", "resnorm2048", "MLX_OMARCHY_RESNORM_R2S32", "0"},
        {"resnorm2048_r2s16_liveqmm_fast", "resnorm2048_liveqmm", "MLX_OMARCHY_RESNORM_R2S32", "0"},
        {"resnorm4096_r2s16_fast", "resnorm4096", "MLX_OMARCHY_RESNORM_R2S32", "0"},
    };
    for (const auto& c : cases) {
      std::string base = "/tmp/w7p-geom-base.bin";
      std::string other = "/tmp/w7p-geom-other.bin";
      setenv("MLX_OMARCHY_QMM_GEOM", "2x8w2", 1);
      setenv("MLX_OMARCHY_SWIGLU_R1", "1", 1);
      setenv("MLX_OMARCHY_QMM_RESIDUAL_NORM", "1", 1);
      setenv("MLX_OMARCHY_RESNORM_VEC4", "1", 1);
      setenv("MLX_OMARCHY_MULTI_R2S16", "0", 1);
      setenv(c.extra_env, c.extra_val, 1);
      setenv("MLX_OMARCHY_G14C_FAST_ONLY", "0", 1);
      int rc1 = run_child(c.scenario, base.c_str());
      setenv("MLX_OMARCHY_G14C_FAST_ONLY", "1", 1);
      int rc2 = run_child(c.scenario, other.c_str());
      unsetenv("MLX_OMARCHY_G14C_FAST_ONLY");
      unsetenv(c.extra_env);
      unsetenv("MLX_OMARCHY_MULTI_R2S16");
      unsetenv("MLX_OMARCHY_RESNORM_VEC4");
      unsetenv("MLX_OMARCHY_QMM_RESIDUAL_NORM");
      unsetenv("MLX_OMARCHY_SWIGLU_R1");
      unsetenv("MLX_OMARCHY_QMM_GEOM");
      if (rc1 != 0 || rc2 != 0 || fnv(base) != fnv(other)) {
        std::cout << "FAIL: " << c.label << " differed from fallback variant (rc "
                  << rc1 << "/" << rc2 << ")\n";
        ++failures;
      } else {
        std::cout << "ok: " << c.label << " identical\n";
      }
    }
    // Multi R2S16W2UpfrontFast (with epilogue) and MultiTrim3 R2S16 (plain
    // 3-member QKV) must both be 0-ULP bit-identical to Multi R2S8W2.
    {
      std::string base = "/tmp/w7p-geom-base.bin";
      std::string other = "/tmp/w7p-geom-other.bin";
      setenv("MLX_OMARCHY_QMM_GEOM", "2x8w2", 1);
      setenv("MLX_OMARCHY_Q4_UPFRONT", "0", 1);
      setenv("MLX_OMARCHY_MULTI_R2S16", "0", 1);
      int rc1 = run_child("group", base.c_str());
      setenv("MLX_OMARCHY_MULTI_R2S16", "1", 1);
      int rc2 = run_child("group", other.c_str());
      unsetenv("MLX_OMARCHY_MULTI_R2S16");
      unsetenv("MLX_OMARCHY_Q4_UPFRONT");
      unsetenv("MLX_OMARCHY_QMM_GEOM");
      if (rc1 != 0 || rc2 != 0 || fnv(base) != fnv(other)) {
        std::cout << "FAIL: multi_r2s16_up_fast differed from R2S8W2 (rc "
                  << rc1 << "/" << rc2 << ")\n";
        ++failures;
      } else {
        std::cout << "ok: multi_r2s16_up_fast identical\n";
      }
      // Plain 2-member group (k512, no epilogue) exercises R2S16W2UpfrontFast3M.
      setenv("MLX_OMARCHY_QMM_GEOM", "2x8w2", 1);
      setenv("MLX_OMARCHY_Q4_UPFRONT", "0", 1);
      setenv("MLX_OMARCHY_MULTI_R2S16", "0", 1);
      int rc3 = run_child("k512", base.c_str());
      setenv("MLX_OMARCHY_MULTI_R2S16", "1", 1);
      int rc4 = run_child("k512", other.c_str());
      unsetenv("MLX_OMARCHY_MULTI_R2S16");
      unsetenv("MLX_OMARCHY_Q4_UPFRONT");
      unsetenv("MLX_OMARCHY_QMM_GEOM");
      if (rc3 != 0 || rc4 != 0 || fnv(base) != fnv(other)) {
        std::cout << "FAIL: multi_r2s16_up_fast_3m differed from R2S8W2 (rc "
                  << rc3 << "/" << rc4 << ")\n";
        ++failures;
      } else {
        std::cout << "ok: multi_r2s16_up_fast_3m identical\n";
      }
    }
    // G14C compile-time ResNormVec4 D2048/D4096 specializations and batch==1
    // Word R2S4W2FastB1 must be 0-ULP bit-identical to the non-specialized
    // fast variants.
    for (const char* name : {"resnorm2048", "resnorm4096"}) {
      std::string base = "/tmp/w7p-geom-base.bin";
      std::string other = "/tmp/w7p-geom-other.bin";
      setenv("MLX_OMARCHY_QMM_GEOM", "2x8w2", 1);
      setenv("MLX_OMARCHY_QMM_RESIDUAL_NORM", "1", 1);
      setenv("MLX_OMARCHY_RESNORM_VEC4", "1", 1);
      setenv("MLX_OMARCHY_RESNORM_R2S32", "1", 1);
      setenv("MLX_OMARCHY_G14C_FAST_ONLY", "1", 1);
      setenv("MLX_OMARCHY_RESNORM_SPEC", "0", 1);
      int rc1 = run_child(name, base.c_str());
      setenv("MLX_OMARCHY_RESNORM_SPEC", "1", 1);
      int rc2 = run_child(name, other.c_str());
      unsetenv("MLX_OMARCHY_RESNORM_SPEC");
      unsetenv("MLX_OMARCHY_G14C_FAST_ONLY");
      unsetenv("MLX_OMARCHY_RESNORM_R2S32");
      unsetenv("MLX_OMARCHY_RESNORM_VEC4");
      unsetenv("MLX_OMARCHY_QMM_RESIDUAL_NORM");
      unsetenv("MLX_OMARCHY_QMM_GEOM");
      if (rc1 != 0 || rc2 != 0 || fnv(base) != fnv(other)) {
        std::cout << "FAIL: " << name
                  << "_spec differed from non-spec fast (rc " << rc1 << "/"
                  << rc2 << ")\n";
        ++failures;
      } else {
        std::cout << "ok: " << name << "_spec identical\n";
      }
    }
    {
      std::string base = "/tmp/w7p-geom-base.bin";
      std::string other = "/tmp/w7p-geom-other.bin";
      setenv("MLX_OMARCHY_QMM_GEOM", "2x8w2", 1);
      setenv("MLX_OMARCHY_WORD_R2S4", "1", 1);
      setenv("MLX_OMARCHY_G14C_FAST_ONLY", "1", 1);
      setenv("MLX_OMARCHY_WORD_B1", "0", 1);
      int rc1 = run_child("single", base.c_str());
      setenv("MLX_OMARCHY_WORD_B1", "1", 1);
      int rc2 = run_child("single", other.c_str());
      unsetenv("MLX_OMARCHY_WORD_B1");
      unsetenv("MLX_OMARCHY_G14C_FAST_ONLY");
      unsetenv("MLX_OMARCHY_WORD_R2S4");
      unsetenv("MLX_OMARCHY_QMM_GEOM");
      if (rc1 != 0 || rc2 != 0 || fnv(base) != fnv(other)) {
        std::cout << "FAIL: word_r2s4_fast_b1 differed from non-b1 fast (rc "
                  << rc1 << "/" << rc2 << ")\n";
        ++failures;
      } else {
        std::cout << "ok: word_r2s4_fast_b1 identical\n";
      }
    }
    // G13C ResNormVec Fast (QMM_RESNORM_PACK32 + dead output0 +
    // QMM_RESNORM_VEC_REG_V4) must be 0-ULP bit-identical to ResNormVec
    // fallback across dead and live qmm_out paths (D=2048 and D=4096).
    for (const char* name : {"resnorm2048", "resnorm2048_liveqmm", "resnorm4096"}) {
      std::string base = "/tmp/w7p-geom-base.bin";
      std::string other = "/tmp/w7p-geom-other.bin";
      setenv("MLX_OMARCHY_QMM_GEOM", "2x8w2", 1);
      setenv("MLX_OMARCHY_QMM_RESIDUAL_NORM", "1", 1);
      setenv("MLX_OMARCHY_RESNORM_VEC4", "0", 1);
      setenv("MLX_OMARCHY_RESNORM_VEC", "1", 1);
      setenv("MLX_OMARCHY_QMM_FAST_ONLY", "0", 1);
      int rc1 = run_child(name, base.c_str());
      setenv("MLX_OMARCHY_QMM_FAST_ONLY", "1", 1);
      int rc2 = run_child(name, other.c_str());
      unsetenv("MLX_OMARCHY_QMM_FAST_ONLY");
      unsetenv("MLX_OMARCHY_RESNORM_VEC");
      unsetenv("MLX_OMARCHY_RESNORM_VEC4");
      unsetenv("MLX_OMARCHY_QMM_RESIDUAL_NORM");
      unsetenv("MLX_OMARCHY_QMM_GEOM");
      if (rc1 != 0 || rc2 != 0 || fnv(base) != fnv(other)) {
        std::cout << "FAIL: " << name
                  << "_g13c_resnorm_vec_fast differed from fallback (rc "
                  << rc1 << "/" << rc2 << ")\n";
        ++failures;
      } else {
        std::cout << "ok: " << name << "_g13c_resnorm_vec_fast identical\n";
      }
  // DEC-44: the fused-multi dispatch-order probe (MLX_OMARCHY_QMM_ORDER=
  // stride, qmm_vec.comp flag bit 19) is a pure WG -> row-block permutation,
  // so every scenario must stay byte-identical with the probe on.
  for (const char* name : {"single", "group", "swiglu"}) {
    std::string base = "/tmp/w7p-geom-order-base.bin";
    std::string other = "/tmp/w7p-geom-order-stride.bin";
    unsetenv("MLX_OMARCHY_QMM_ORDER");
    int rc = run_child(name, base.c_str());
    setenv("MLX_OMARCHY_QMM_ORDER", "stride", 1);
    int rc2 = run_child(name, other.c_str());
    unsetenv("MLX_OMARCHY_QMM_ORDER");
    if (rc != 0 || rc2 != 0) {
      std::cout << "FAIL: order " << name << " child rc " << rc << "/"
                << rc2 << "\n";
      ++failures;
      continue;
    }
    if (fnv(base) != fnv(other)) {
      std::cout << "FAIL: order " << name << " bytes differ\n";
      ++failures;
    } else {
      std::cout << "ok: order " << name << " identical\n";
    }
  }
    }
  }
  // The dedicated SwiGLU dispatch (2x8w2's paired epilogue) sends one
  // workgroup per gate column pair: the permute's bijection must key on
  // that grid (groups0), not the members' group sum, which doubled it and
  // left the top half of the rows unwritten.
  {
    std::string base = "/tmp/w7p-geom-order-sw-base.bin";
    std::string other = "/tmp/w7p-geom-order-sw-stride.bin";
    setenv("MLX_OMARCHY_QMM_GEOM", "2x8w2", 1);
    int rc = run_child("swiglu", base.c_str());
    setenv("MLX_OMARCHY_QMM_ORDER", "stride", 1);
    int rc2 = run_child("swiglu", other.c_str());
    unsetenv("MLX_OMARCHY_QMM_ORDER");
    unsetenv("MLX_OMARCHY_QMM_GEOM");
    if (rc != 0 || rc2 != 0) {
      std::cout << "FAIL: order swiglu-2x8w2 child rc " << rc << "/" << rc2
                << "\n";
      ++failures;
    } else if (fnv(base) != fnv(other)) {
      std::cout << "FAIL: order swiglu-2x8w2 bytes differ\n";
      ++failures;
    } else {
      std::cout << "ok: order swiglu-2x8w2 identical\n";
    }
    // Compile-time ResNormVec4 R2S16W2Fast D2048/D4096 specializations must be
    // 0-ULP bit-identical to the non-specialized R2S16W2Fast variant (including
    // live QMM output).
    for (const char* name :
         {"resnorm2048", "resnorm2048_liveqmm", "resnorm4096"}) {
      std::string base = "/tmp/w7p-geom-base.bin";
      std::string other = "/tmp/w7p-geom-other.bin";
      setenv("MLX_OMARCHY_QMM_GEOM", "2x8w2", 1);
      setenv("MLX_OMARCHY_QMM_RESIDUAL_NORM", "1", 1);
      setenv("MLX_OMARCHY_RESNORM_VEC4", "1", 1);
      setenv("MLX_OMARCHY_RESNORM_R2S32", "0", 1);
      setenv("MLX_OMARCHY_G14C_FAST_ONLY", "1", 1);
      setenv("MLX_OMARCHY_RESNORM_SPEC", "0", 1);
      int rc1 = run_child(name, base.c_str());
      setenv("MLX_OMARCHY_RESNORM_SPEC", "1", 1);
      int rc2 = run_child(name, other.c_str());
      unsetenv("MLX_OMARCHY_RESNORM_SPEC");
      unsetenv("MLX_OMARCHY_G14C_FAST_ONLY");
      unsetenv("MLX_OMARCHY_RESNORM_R2S32");
      unsetenv("MLX_OMARCHY_RESNORM_VEC4");
      unsetenv("MLX_OMARCHY_QMM_RESIDUAL_NORM");
      unsetenv("MLX_OMARCHY_QMM_GEOM");
      if (rc1 != 0 || rc2 != 0 || fnv(base) != fnv(other)) {
        std::cout << "FAIL: " << name
                  << "_r2s16_spec differed from non-spec R2S16W2Fast (rc "
                  << rc1 << "/" << rc2 << ")\n";
        ++failures;
      } else {
        std::cout << "ok: " << name << "_r2s16_spec identical\n";
      }
    }
    // 16-binding 3-member plain+KV trimmed Multi R2S8W2UpfrontFast3M must be
    // 0-ULP bit-identical to 25-binding R2S8W2UpfrontFast across 2-member and
    // 3-member plain groups.
    for (const char* name : {"k512", "qkv3m"}) {
      std::string base = "/tmp/w7p-geom-base.bin";
      std::string other = "/tmp/w7p-geom-other.bin";
      setenv("MLX_OMARCHY_QMM_GEOM", "2x8w2", 1);
      setenv("MLX_OMARCHY_MULTI_R2S16", "0", 1);
      setenv("MLX_OMARCHY_Q4_UPFRONT", "1", 1);
      setenv("MLX_OMARCHY_G14C_FAST_ONLY", "1", 1);
      setenv("MLX_OMARCHY_MULTI_3M", "0", 1);
      int rc1 = run_child(name, base.c_str());
      setenv("MLX_OMARCHY_MULTI_3M", "1", 1);
      int rc2 = run_child(name, other.c_str());
      unsetenv("MLX_OMARCHY_MULTI_3M");
      unsetenv("MLX_OMARCHY_G14C_FAST_ONLY");
      unsetenv("MLX_OMARCHY_Q4_UPFRONT");
      unsetenv("MLX_OMARCHY_MULTI_R2S16");
      unsetenv("MLX_OMARCHY_QMM_GEOM");
      if (rc1 != 0 || rc2 != 0 || fnv(base) != fnv(other)) {
        std::cout << "FAIL: multi_r2s8_up_fast_3m_" << name
                  << " differed from R2S8W2UpfrontFast (rc " << rc1 << "/"
                  << rc2 << ")\n";
        ++failures;
      } else {
        std::cout << "ok: multi_r2s8_up_fast_3m_" << name << " identical\n";
      }
    }
  }
  std::cout << (failures == 0 ? "GEOM-TEST-PASS" : "GEOM-TEST-FAIL")
            << " failures " << failures << "\n";
  return failures == 0 ? 0 : 1;
}
