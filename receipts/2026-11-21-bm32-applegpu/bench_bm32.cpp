#include <algorithm>
#include <chrono>
#include <cstdio>
#include <random>
#include <vector>

#include "mlx/backend/gpu/device_info.h"
#include "mlx/backend/omarchy/encoder.h"
#include "mlx/backend/omarchy/trace.h"
#include "mlx/device.h"
#include "mlx/ops.h"
#include "mlx/stream.h"
#include "mlx/transforms.h"

using namespace mlx::core;

// Timing only: the same dense projection shape the probe pins bit-exactly.
// The route is selected by MLX_OMARCHY_MATMUL_BM32 in the environment.
int main() {
  if (!gpu::is_available()) {
    std::printf("no qualifying GPU\n");
    return 2;
  }
  set_default_device(Device::gpu);
  Stream s = new_stream(Device::gpu);
  constexpr int m = 32, k = 4096, n = 14336;
  std::mt19937 rng(7);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::vector<bfloat16_t> a_host(static_cast<size_t>(m) * k);
  std::vector<bfloat16_t> w_host(static_cast<size_t>(n) * k);
  for (auto& v : a_host) v = bfloat16_t(dist(rng));
  for (auto& v : w_host) v = bfloat16_t(dist(rng));
  array a(a_host.begin(), Shape{m, k}, bfloat16);
  array w(w_host.begin(), Shape{n, k}, bfloat16);
  array wt = transpose(w, {1, 0}, s);
  eval({a, wt});
  omarchy::get_command_encoder(s).synchronize();
  for (int it = 0; it < 9; ++it) {
    auto t0 = std::chrono::steady_clock::now();
    array out = matmul(a, wt, s);
    eval(out);
    omarchy::get_command_encoder(s).synchronize();
    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    auto last = omarchy::trace::counters().last_dispatched_kernel.load();
    std::printf("iter %d %.3f ms last_kernel %lld\n", it, ms,
                static_cast<long long>(last));
  }
  return 0;
}
