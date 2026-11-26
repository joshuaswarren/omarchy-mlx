// Dense matmul best-of-8 timing against the built libmlx (no Python). Run
// through the mesa overlay ICD with ASAHI_AGX3B_COOPERATIVE_MATRIX 0 vs 1.
// Build: compile with the omarchy test target's -I/-D flags, link
//   c++ bench.o -o bench libmlx.a -ldl -lpthread /usr/lib64/libopenblas.so -lm -ldl mlx/io/libgguflib.a -lgcc_s
// usage: bench <n> [f16|f32]
#include "mlx/mlx.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
using namespace mlx::core;
int main(int argc, char** argv){
  int n = argc>1?atoi(argv[1]):2048;
  bool f16 = argc>2 && std::string(argv[2])=="f16";
  Dtype dt = f16?float16:float32;
  auto a = astype(random::normal({n,n}), dt);
  auto b = astype(random::normal({n,n}), dt);
  eval(a); eval(b);
  auto run=[&](){ auto c=matmul(a,b); eval(c); };
  for(int i=0;i<2;i++) run();
  double best=1e9;
  for(int i=0;i<8;i++){ auto t0=std::chrono::steady_clock::now(); run(); auto t1=std::chrono::steady_clock::now(); double ms=std::chrono::duration<double,std::milli>(t1-t0).count(); if(ms<best)best=ms; }
  double tflops = 2.0*(double)n*n*n/(best/1000.0)/1e12;
  printf("n=%d dt=%s best=%.3f ms %.3f TFLOP/s dev=%s\n", n, f16?"f16":"f32", best, tflops, default_device()==Device::gpu?"gpu":"cpu");
  return 0;
}
