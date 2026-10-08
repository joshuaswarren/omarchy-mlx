// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT
//
// Debug entry: translate one generated-MSL file and print the GLSL (or the
// exact refusal) so KernelRecheck can classify translator gaps on a CPU host.
// Usage: omarchy_custom_kernel_translate_dump FILE GX GY GZ TX TY TZ OUTPUTS

#define MLX_OMARCHY_TEST_TRANSLATE
#include "mlx/fast.h"

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace mlx::core::fast {
std::string mlx_omarchy_translate_msl_for_test(
    const std::string& source,
    int grid_x,
    int grid_y,
    int grid_z,
    int threads_x,
    int threads_y,
    int threads_z,
    std::size_t output_count);
}

int main(int argc, char** argv) {
  if (argc < 9) {
    std::cerr << "usage: " << argv[0]
              << " FILE GX GY GZ TX TY TZ OUTPUTS\n";
    return 64;
  }
  std::ifstream in(argv[1]);
  std::ostringstream buffer;
  buffer << in.rdbuf();
  const std::string source = buffer.str();
  try {
    const auto glsl = mlx::core::fast::mlx_omarchy_translate_msl_for_test(
        source,
        std::atoi(argv[2]), std::atoi(argv[3]), std::atoi(argv[4]),
        std::atoi(argv[5]), std::atoi(argv[6]), std::atoi(argv[7]),
        static_cast<std::size_t>(std::atoi(argv[8])));
    std::cout << glsl;
    return 0;
  } catch (const std::exception& error) {
    std::cout << "TRANSLATE-ERROR: " << error.what() << "\n";
    return 1;
  }
}
