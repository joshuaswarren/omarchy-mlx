// SPDX-License-Identifier: MIT
// Per-kernel translator harness driver.
//
// Reads an MSL source from a file and invokes translate_msl on it via the
// test-only export MLX_OMARCHY_TRANSLATE_EXPORTS provides. Reports the
// translator verdict on stdout:
//
//   line 1: VERDICT accept|reject
//   line 2 (only on reject): ERR <error text>
//   line 3 (only on accept): GLSL <first 200 chars of translated GLSL>
//
// Build:
//   g++ -std=gnu++20 -O0 -g \
//       -DMLX_OMARCHY_TEST_TRANSLATE \
//       -DMLX_OMARCHY_TRANSLATE_EXPORTS \
//       translator_driver.cpp -L. -lmlx -o translator_driver

#include <cstddef>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include "mlx/fast.h"
#include "mlx/ops.h"

namespace mlx::core::fast {
// Test-only export from custom_kernel.cpp.
std::string mlx_omarchy_translate_msl_for_test(
    const std::string& source,
    int grid_x, int grid_y, int grid_z,
    int threads_x, int threads_y, int threads_z,
    std::size_t output_count);
}

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "usage: " << argv[0] << " <msl_source.txt>\n";
    return 2;
  }
  std::ifstream in(argv[1]);
  if (!in) {
    std::cerr << "cannot open " << argv[1] << "\n";
    return 2;
  }
  std::ostringstream ss; ss << in.rdbuf();
  std::string source = ss.str();

  // Defaults match a (256, 1, 1) threadgroup with a 1-tile grid; the
  // harness cares only about the translation verdict, not dispatch shape.
  try {
    auto glsl = mlx::core::fast::mlx_omarchy_translate_msl_for_test(
        source, 1, 1, 1, 32, 1, 1, 1);
    std::cout << "VERDICT accept\n";
    if (!glsl.empty()) {
      std::cout << "GLSL " << glsl.substr(0, 200) << "\n";
    }
    return 0;
  } catch (const std::exception& error) {
    std::cout << "VERDICT reject\n";
    std::cout << "ERR " << error.what() << "\n";
    return 0;
  }
}
