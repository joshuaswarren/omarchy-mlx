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
//
// The pair test checks exact bytes and dispatches; bytes alone cannot prove
// that the optional kernel ran.

#include <dlfcn.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "mlx/backend/omarchy/compute.h"
#include "mlx/backend/omarchy/device.h"
#include "mlx/backend/omarchy/encoder.h"
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

// Batched QMM can emit an input-copy dispatch before the QMM dispatch.
bool is_qmm_word_kernel(int id) {
  using K = omarchy::ComputeKernel;
  return id == static_cast<int>(K::QmmVecQ4WordSubgroupBF16) ||
      id == static_cast<int>(K::QmmVecQ4WordSubgroupBF16R2S16) ||
      id == static_cast<int>(K::QmmVecQ4WordSubgroupBatch4BF16) ||
      id == static_cast<int>(K::QmmVecQ4WordPairSubgroupBF16);
}

std::vector<std::pair<int, uint32_t>> traced_dispatch(
    const std::function<void()>& run);

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
  } else if (name == "wpair") {
    struct WCase {
      const char* label;
      int n;
      int k;
      int batch;
      bool qmv_batch_off;
    };
    const WCase cases[] = {
        {"full", 6144, 2048, 1, false},
        {"partial_fast", 96, 2048, 1, false},
        {"partial_tail", 100, 2048, 1, false},
        {"tail_k", 2176, 2176, 1, false},
        {"tiny", 65, 2048, 1, false},
        {"batch4_default", 6144, 2048, 3, false},
        {"batch_geom_noopt", 6144, 2048, 3, true},
        {"batch5_ineligible", 6144, 2048, 5, false},
    };
    std::vector<std::string> labels;
    std::vector<std::pair<int, uint32_t>> dispatches;
    std::vector<int> counts;
    for (const auto& c : cases) {
      std::mt19937 rng(20261104u + c.n * 7u + c.k * 13u + c.batch);
      auto l = make_lin(c.n, c.k, rng);
      array x = make_x(c.k, rng);
      if (c.batch != 1) {
        x = repeat(x, c.batch, 0);
      }
      eval(x, l.w, l.s, l.b);
      if (c.qmv_batch_off) {
        setenv("MLX_OMARCHY_QMV_BATCH", "0", 1);
      }
      array o = qmm(x, l);
      auto seen = traced_dispatch([&] {
        eval(o);
        omarchy::get_command_encoder(default_stream(Device::gpu)).synchronize();
      });
      if (c.qmv_batch_off) {
        unsetenv("MLX_OMARCHY_QMV_BATCH");
      }
      int qmm_id = -1;
      uint32_t qmm_gx = 0;
      int qmm_rows = 0;
      for (const auto& d : seen) {
        if (is_qmm_word_kernel(d.first)) {
          ++qmm_rows;
          qmm_id = d.first;
          qmm_gx = d.second;
        }
      }
      labels.push_back(c.label);
      dispatches.push_back({qmm_id, qmm_gx});
      counts.push_back(qmm_rows);
      outs.push_back(o);
    }
    std::ofstream ids(out + ".ids");
    for (size_t i = 0; i < dispatches.size(); ++i) {
      ids << labels[i] << " " << dispatches[i].first << " "
          << dispatches[i].second << " " << counts[i] << "\n";
    }
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

bool identical_files(const std::string& first, const std::string& second) {
  std::ifstream a(first, std::ios::binary);
  std::ifstream b(second, std::ios::binary);
  if (!a || !b || a.peek() == std::char_traits<char>::eof() ||
      b.peek() == std::char_traits<char>::eof()) {
    return false;
  }
  return std::equal(
      std::istreambuf_iterator<char>(a),
      std::istreambuf_iterator<char>(),
      std::istreambuf_iterator<char>(b),
      std::istreambuf_iterator<char>());
}

// Ordered (kernel id, workgroup count) pairs dispatched while `run`
// evaluates, from the encoder's MLX_OMARCHY_TRACE_DISPATCH lines on stderr
// (the test_qmv_batch.cpp capture pattern; the workgroup count is kept so
// the W2 pair check sees the changed grid, not only a changed id).
std::vector<std::pair<int, uint32_t>> traced_dispatch(
    const std::function<void()>& run) {
  std::fflush(stderr);
  int saved = dup(2);
  std::FILE* sink = std::tmpfile();
  dup2(fileno(sink), 2);
  setenv("MLX_OMARCHY_TRACE_DISPATCH", "1", 1);
  run();
  unsetenv("MLX_OMARCHY_TRACE_DISPATCH");
  std::fflush(stderr);
  dup2(saved, 2);
  close(saved);
  std::rewind(sink);
  std::vector<std::pair<int, uint32_t>> dispatches;
  char line[512];
  while (std::fgets(line, sizeof(line), sink)) {
    int id = -1;
    uint32_t gx = 0;
    if (std::sscanf(
            line, "[rtmod] DISPATCH kernel=%d count=%*u gx=%u", &id, &gx) ==
        2) {
      dispatches.push_back({id, gx});
    }
  }
  std::fclose(sink);
  return dispatches;
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
      if (!identical_files(base, other)) {
        std::cout << "FAIL: " << geometry << " " << name
                  << " bytes differ from 2x4\n";
        ++failures;
      } else {
        std::cout << "ok: " << geometry << " " << name << " identical\n";
      }
    }
  }
  // Equal bytes are insufficient when the optional route does not engage.
  {
    const auto& gcaps = mlx::core::omarchy::device().capabilities();
    bool qualifies = gcaps.subgroup_size == 32u &&
        (gcaps.subgroup_operations & VK_SUBGROUP_FEATURE_ARITHMETIC_BIT) != 0u;
    if (!qualifies) {
      std::cout << "skip: W2 word-pair needs a 32-lane arithmetic subgroup"
                << " (device subgroup " << gcaps.subgroup_size << ")\n";
    } else {
      const std::vector<std::string> expected = {
          "full",
          "partial_fast",
          "partial_tail",
          "tail_k",
          "tiny",
          "batch4_default",
          "batch_geom_noopt",
          "batch5_ineligible"};
      const size_t kSingleCases = 5;
      const auto prefix = (std::filesystem::temp_directory_path() /
                           ("omarchy-qmm-pair-" + std::to_string(getpid())))
                              .string();
      std::string base = prefix + "-base.bin";
      std::string cand = prefix + "-cand.bin";
      setenv("MLX_OMARCHY_QMM_GEOM", "2x16", 1);
      unsetenv("MLX_OMARCHY_QMM_WORD_PAIR");
      int rb = run_child("wpair", base.c_str());
      setenv("MLX_OMARCHY_QMM_WORD_PAIR", "1", 1);
      int rc = run_child("wpair", cand.c_str());
      unsetenv("MLX_OMARCHY_QMM_WORD_PAIR");
      unsetenv("MLX_OMARCHY_QMM_GEOM");
      auto read_rows = [](const std::string& path) {
        std::vector<std::array<std::string, 4>> rows;
        std::ifstream f(path);
        std::string label, id, gx, count;
        while (f >> label >> id >> gx >> count) {
          rows.push_back({label, id, gx, count});
        }
        return rows;
      };
      auto rows_b = read_rows(base + ".ids");
      auto rows_c = read_rows(cand + ".ids");
      if (rb != 0 || rc != 0) {
        std::cout << "FAIL: W2 word-pair child rc " << rb << "/" << rc
                  << "\n";
        ++failures;
      } else {
        auto rows_ok = [&](const std::vector<std::array<std::string, 4>>& rows,
                           const char* arm) {
          if (rows.size() != expected.size()) {
            std::cout << "FAIL: W2 word-pair " << arm << " trace has "
                      << rows.size() << " rows, want " << expected.size()
                      << " (one dispatch per named case)\n";
            ++failures;
            return false;
          }
          for (size_t i = 0; i < rows.size(); ++i) {
            if (rows[i][0] != expected[i]) {
              std::cout << "FAIL: W2 word-pair " << arm << " row " << i
                        << " is " << rows[i][0] << ", want " << expected[i]
                        << "\n";
              ++failures;
              return false;
            }
            if (rows[i][3] != "1") {
              std::cout << "FAIL: W2 word-pair " << arm << " case "
                        << rows[i][0] << " saw " << rows[i][3]
                        << " QMM dispatches, want 1 (capture broken)\n";
              ++failures;
              return false;
            }
          }
          return true;
        };
        bool ok_b = rows_ok(rows_b, "baseline");
        bool ok_c = rows_ok(rows_c, "candidate");
        if (ok_b && ok_c) {
          size_t engaged = 0;
          size_t batch_moved = 0;
          for (size_t i = 0; i < rows_b.size(); ++i) {
            bool differs = rows_b[i][1] != rows_c[i][1] ||
                rows_b[i][2] != rows_c[i][2];
            if (i < kSingleCases) {
              if (differs) {
                ++engaged;
              }
            } else if (differs) {
              ++batch_moved;
            }
          }
          if (engaged != kSingleCases || batch_moved != 0) {
            std::cout << "FAIL: WORD_PAIR=1 engagement " << engaged << "/"
                      << kSingleCases << " single-row cases, batch routes moved "
                      << batch_moved << " (the candidate dispatch must differ"
                      << " on every single-row case and match on every batch"
                      << " case)\n";
            ++failures;
          }
          const std::vector<std::array<uint32_t, 2>> pair_grids = {
              {192, 96},  // full, n = 6144
              {3, 2},     // partial_fast, n = 96
              {4, 2},     // partial_tail, n = 100
              {68, 34},   // tail_k, n = 2176
              {3, 2},     // tiny, n = 65
          };
          const int base_id = static_cast<int>(
              omarchy::ComputeKernel::QmmVecQ4WordSubgroupBF16R2S16);
          const int pair_id =
              static_cast<int>(omarchy::ComputeKernel::QmmVecQ4WordPairSubgroupBF16);
          bool grids_ok = true;
          for (size_t i = 0; i < kSingleCases; ++i) {
            if (rows_b[i][1] != std::to_string(base_id) ||
                std::stoul(rows_b[i][2]) != pair_grids[i][0] ||
                rows_c[i][1] != std::to_string(pair_id) ||
                std::stoul(rows_c[i][2]) != pair_grids[i][1]) {
              std::cout << "FAIL: W2 word-pair case " << rows_b[i][0]
                        << " dispatched kernel " << rows_b[i][1] << "/"
                        << rows_c[i][1] << " on grid " << rows_b[i][2] << "/"
                        << rows_c[i][2] << ", want baseline " << base_id << "/"
                        << pair_grids[i][0] << " and pair " << pair_id << "/"
                        << pair_grids[i][1] << "\n";
              grids_ok = false;
              ++failures;
            }
          }
          if (!identical_files(base, cand)) {
            std::cout << "FAIL: W2 word-pair bytes differ from the unpaired"
                      << " 2x16 kernel\n";
            ++failures;
          } else if (engaged == kSingleCases && batch_moved == 0 && grids_ok) {
            std::cout << "ok: W2 word-pair engaged and byte-identical over "
                      << kSingleCases << " single-row and "
                      << (rows_b.size() - kSingleCases) << " batch cases\n";
          }
        }
      }
    }
  }
  std::cout << (failures == 0 ? "GEOM-TEST-PASS" : "GEOM-TEST-FAIL")
            << " failures " << failures << "\n";
  return failures == 0 ? 0 : 1;
}
