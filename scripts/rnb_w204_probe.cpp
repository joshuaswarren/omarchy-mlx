
// width-204 cell error analysis: fused rope_rms_norm vs composed
// rms_norm->rope vs a host fp64 reference. Inputs are EXACTLY the
// doctest's (pattern seeds 235/251, bf16, eps 1e-6, base 10000,
// scale 1, offset 0, non-traditional, shape (1,9,1,204)).
// Rotation is identity here (position 0 => cos=1, sin=0), so the
// comparison isolates the norm-stage rounding of the two routes.
#include <cstdio>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <vector>
#include <algorithm>
#include "mlx/fast.h"
#include "mlx/ops.h"
#include "mlx/transforms.h"
using namespace mlx::core;

static std::vector<float> pattern(size_t count, uint32_t seed) {
  std::vector<float> values;
  values.reserve(count);
  uint32_t state = seed;
  for (size_t index = 0; index < count; ++index) {
    state = state * 1664525u + 1013904223u;
    values.push_back(
        static_cast<float>(static_cast<double>(state % 20000u) / 10000.0) -
        1.0f);
  }
  return values;
}

static uint16_t bf16_bits(float v) {
  uint32_t bits;
  std::memcpy(&bits, &v, 4);
  return (uint16_t)(bits >> 16);
}
static float bf16_from_bits(uint16_t b) {
  uint32_t bits = (uint32_t)b << 16;
  float v;
  std::memcpy(&v, &bits, 4);
  return v;
}
static double ulp_bf16_distance(double ref, float rounded) {
  if (ref == 0.0 && rounded == 0.0f) return 0.0;
  int e;
  std::frexp(std::max(std::fabs(ref), std::fabs((double)rounded)), &e);
  double ulp = std::ldexp(1.0, e - 8);  // 7 mantissa bits: spacing 2^(e-8)
  return std::fabs((double)rounded - ref) / ulp;
}

int main() {
  const int rows = 9, D = 204;
  const float eps = 1e-6f;
  auto x_data = pattern((size_t)rows * D, 235u);
  auto w_data = pattern((size_t)D, 251u);
  array x = astype(array(x_data.data(), Shape{1, rows, 1, D}, float32), bfloat16);
  array w = astype(array(w_data.data(), Shape{D}, float32), bfloat16);
  eval(x);
  eval(w);
  array fused = fast::rope_rms_norm(x, D, w, eps, false, 10000.0f, 1.0f, 0);
  array composed = fast::rope(fast::rms_norm(x, w, eps), D, false, 10000.0f, 1.0f, 0);
  eval(fused);
  eval(composed);
  auto flat = [](array a) {
    a = reshape(astype(a, float32), {a.size()});
    eval(a);
    const float* d = a.data<float>();
    return std::vector<float>(d, d + a.size());
  };
  std::vector<float> fh = flat(fused);
  std::vector<float> ch = flat(composed);
  std::vector<float> xh = flat(x);
  std::vector<float> wh = flat(w);

  double worst_f = 0, worst_c = 0, sum_f = 0, sum_c = 0;
  int idx_worst_f = -1, idx_worst_c = -1;
  double f_ulp_max = 0, c_ulp_max = 0, f_ulp_sum = 0, c_ulp_sum = 0;
  int disagree = 0;
  for (int r = 0; r < rows; ++r) {
    double ms = 0.0;
    for (int c = 0; c < D; ++c) {
      double v = (double)bf16_from_bits(bf16_bits(xh[r * D + c]));
      ms += v * v;
    }
    double rms = std::sqrt(ms / D + (double)eps);
    for (int c = 0; c < D; ++c) {
      double v = (double)bf16_from_bits(bf16_bits(xh[r * D + c]));
      double wv = (double)bf16_from_bits(bf16_bits(wh[c]));
      double ref = v * wv / rms;  // rotation is identity at position 0
      float fv = fh[r * D + c];
      float cv = ch[r * D + c];
      if (fv != cv) ++disagree;
      double ef = std::fabs(ref - (double)fv);
      double ec = std::fabs(ref - (double)cv);
      if (ef > worst_f) { worst_f = ef; idx_worst_f = r * D + c; }
      if (ec > worst_c) { worst_c = ec; idx_worst_c = r * D + c; }
      sum_f += ef; sum_c += ec;
      f_ulp_sum += ulp_bf16_distance(ref, fv);
      c_ulp_sum += ulp_bf16_distance(ref, cv);
      f_ulp_max = std::max(f_ulp_max, ulp_bf16_distance(ref, fv));
      c_ulp_max = std::max(c_ulp_max, ulp_bf16_distance(ref, cv));
    }
  }
  std::printf("disagreeing elements (fused != composed bitwise): %d / %d\n",
              disagree, rows * D);
  std::printf("fp64 max abs err: fused=%.6e (idx %d)  composed=%.6e (idx %d)\n",
              worst_f, idx_worst_f, worst_c, idx_worst_c);
  std::printf("fp64 mean abs err: fused=%.6e  composed=%.6e\n",
              sum_f / (rows * D), sum_c / (rows * D));
  std::printf("bf16 ULP err: fused max=%.4f mean=%.6f | composed max=%.4f mean=%.6f\n",
              f_ulp_max, f_ulp_sum / (rows * D), c_ulp_max, c_ulp_sum / (rows * D));
  {
    int r = 1586 / D, c = 1586 % D;
    double ms = 0.0;
    for (int k = 0; k < D; ++k)
      ms += std::pow((double)bf16_from_bits(bf16_bits(xh[r * D + k])), 2.0);
    double rms = std::sqrt(ms / D + (double)eps);
    double v = (double)bf16_from_bits(bf16_bits(xh[r * D + c]));
    double wv = (double)bf16_from_bits(bf16_bits(wh[c]));
    double ref = v * wv / rms;
    std::printf("element 1586: fp64=%.10f fused=%.10f (0x%04x) composed=%.10f (0x%04x)\n",
                ref, (double)fh[1586], bf16_bits(fh[1586]),
                (double)ch[1586], bf16_bits(ch[1586]));
    std::printf("element 1586 err: fused=%.6e composed=%.6e -> fused %s composed\n",
                std::fabs(ref - (double)fh[1586]), std::fabs(ref - (double)ch[1586]),
                std::fabs(ref - (double)fh[1586]) <= std::fabs(ref - (double)ch[1586])
                    ? "<=" : ">");
    std::printf("element 1586 ULP distances: fused=%.4f composed=%.4f\n",
                ulp_bf16_distance(ref, fh[1586]), ulp_bf16_distance(ref, ch[1586]));
  }
  return 0;
}
