// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

// Differential driver for the C++ section builder (h16_hwx.cpp), used by
// h16_sections_diff.py. "emit DIR" writes synthetic HWX variants (scratch,
// kernel, H17, zero-base slot, error cases, random byte flips); with no
// arguments it reads HWX paths from stdin and prints one line per file:
// "OK <section hex>... tasks=N refs=..." or "ERR <reason>".
//
//   g++ -std=c++20 -I overlay -I <mlx api shim> h16_sections_diff.cpp
//       overlay/mlx/backend/omarchy/ane/h16_hwx.cpp -o h16_sections_diff

#include "mlx/backend/omarchy/ane/h16_hwx.h"
#include "h16_test_hwx.h"

#include <cstdio>
#include <fstream>
#include <iostream>
#include <iterator>
#include <random>
#include <string>
#include <vector>

using namespace mlx::core::omarchy::ane;
using namespace h16_test;

static std::string hexs(const std::vector<uint8_t>& v) {
  std::string s;
  char b[3];
  for (auto c : v) {
    std::snprintf(b, 3, "%02x", c);
    s += b;
  }
  return s;
}

static void save(const std::string& p, const std::vector<uint8_t>& b) {
  std::ofstream o(p, std::ios::binary);
  o.write(reinterpret_cast<const char*>(b.data()), b.size());
}

static int emit(const std::string& dir) {
    std::vector<Spec> specs;
  specs.push_back(Spec{});
  { Spec s; s.scratch = {{0x40000000, 0x123456}}; s.kernel_vm = 0x50000000; s.tasks = 3;
    s.slots = {{3, 0x40000000}, {4, 0x50000000}, {6, 0x30000000}, {7, 0x30010000}, {8, 0x30020000}, {9, 0}}; specs.push_back(s); }
  { Spec s; s.subtype = 9; specs.push_back(s); }
  { Spec s; s.surfaces = {{"y",2,0x30010000,0x8000,2,0x2000},{"a",1,0x30000000,0x4000,1,0x2000}}; s.slots = {{7,0x30010000},{6,0x30000000}}; specs.push_back(s); }
  { Spec s; s.surfaces.push_back({"c",1,0x30030000,0x4000,1,0x2000}); s.surfaces.push_back({"d",1,0x30040000,0x4000,1,0x2000}); s.slots.push_back({9,0x30030000}); specs.push_back(s); }
  { Spec s; s.slots.push_back({10, 0x99990000}); specs.push_back(s); }
  { Spec s; s.surfaces[2].direction = 2; specs.push_back(s); }
  { Spec s; s.descriptor = false; specs.push_back(s); }
  { Spec s; s.scratch = {{0x40000000, 0x1000}}; s.slots = {{3, 0x40000000}}; specs.push_back(s); }
  std::mt19937 rng(7);
  int n = 0;
  for (size_t k = 0; k < specs.size(); ++k) {
    auto base = build(specs[k]);
    save(dir + "/s" + std::to_string(n++) + ".hwx", base);
    for (int m = 0; m < 300; ++m) {
      auto b = base;
      int flips = 1 + rng() % 3;
      for (int f = 0; f < flips; ++f) {
        size_t o = rng() % 2 ? rng() % std::min<size_t>(b.size(), 0x800) : (b.size() - 0x100 + rng() % 0x100);
        b[o % b.size()] = uint8_t(rng());
      }
      save(dir + "/s" + std::to_string(n++) + ".hwx", b);
    }
  }
  std::printf("%d\n", n);
  return 0;
}

int main(int argc, char** argv) {
  if (argc == 3 && std::string(argv[1]) == "emit") return emit(argv[2]);
  std::string path;
  while (std::getline(std::cin, path)) {
    std::ifstream in(path, std::ios::binary);
    std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(in)), {});
    try {
      auto out = h16_build_sections(bytes.data(), bytes.size());
      std::cout << "OK";
      for (auto& s : out.sections) std::cout << " " << hexs(s);
      std::cout << " tasks=" << out.tasks << " refs=";
      for (auto& r : out.refs) std::cout << r.first << ":" << r.second << ",";
      std::cout << " ids=";
      for (auto& s : out.surfaces) std::cout << (s.name ? *s.name : "None") << "=" << s.id << ",";
      if (out.scratch_id) std::cout << "scratch=" << *out.scratch_id << ",";
      std::cout << "\n";
    } catch (const AneH16HwxError& e) {
      std::cout << "ERR " << e.reason << "\n";
    }
  }
}
