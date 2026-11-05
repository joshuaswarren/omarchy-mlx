// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

// MLX_OMARCHY_QMM_GEOM: the decode Qmm column geometry. Every geometry runs
// the same per-lane chain per column, so the bytes must be identical to the
// base 2x4 kernels; only the columns and threads per workgroup change. The
// geometry kernels need a 32-lane arithmetic subgroup, so on a device with
// smaller subgroups every scenario takes the base kernels and the byte
// checks pass trivially; on qualifying devices (the M1-class parts) they
// exercise the real variants. An unknown value must refuse loudly.
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

#include "mlx/backend/omarchy/device.h"
#include "mlx/backend/omarchy/fused_chain.h"
#include "mlx/backend/gpu/device_info.h"
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
    const auto& dev = mlx::core::omarchy::device().capabilities().device_name;
    std::string want =
        (dev.find("G13G") != std::string::npos || dev.find("G13C") != std::string::npos)
        ? "2x16"
        : "2x4";
    int rc = run_child("name", "/tmp/w7p-geom-name.bin");
    std::ifstream f("/tmp/w7p-geom-name.bin");
    std::string got;
    std::getline(f, got);
    if (rc != 0 || got != want) {
      std::cout << "FAIL: default geometry name is " << got << ", want "
                << want << " (rc " << rc << ")\n";
      ++failures;
    }
  }
  // Every accepted geometry produces the base bytes.
  for (const char* geometry : {"2x16", "2x4"}) {
    for (const char* name : {"single", "group", "swiglu"}) {
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
  std::cout << (failures == 0 ? "GEOM-TEST-PASS" : "GEOM-TEST-FAIL")
            << " failures " << failures << "\n";
  return failures == 0 ? 0 : 1;
}
