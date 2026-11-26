// Regression test for the dense BF16 decode-shape vector matmul: every
// m=1 shape the decode chain dispatches (k % 128 == 0, n % 4 == 0,
// aligned fresh operands, weights stored [n, k]) must match the exact
// host reference, whatever workgroup span the dispatch uses. The vec
// shader grid-strides row_groups (4 output columns per group), so the
// per-column lane mapping and accumulation order are span-invariant;
// this case fails if any dispatch-side change reorders the reduction.
TEST_CASE("dense bf16 single-row decode shapes match the host reference") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  auto bf16_weight = [&](const std::vector<float>& values, int rows,
                         int cols) {
    return transpose(
        astype(
            array(values.begin(), Shape{rows, cols}, float32),
            bfloat16,
            stream),
        {0, 1},
        stream);
  };
  // [k, n] pairs: k/v projection (896 -> 128), q/o and down projection
  // (896 -> 896, 4864 -> 896), gate/up (896 -> 4864) and lm_head
  // (896 -> 151936). Integer operands: every product is exact in f32
  // and every partial sum stays integral below 2^24, so the shader's f32
  // accumulation is exact and the bf16-rounded result must equal the
  // host's f64 reference for every element, on every column group.
  for (auto [k, n] : std::vector<std::pair<int, int>>{
           {896, 128}, {896, 896}, {896, 4864}, {4864, 896}, {896, 151936}}) {
    std::vector<float> a_values(k);
    for (int i = 0; i < k; ++i) {
      a_values[i] = float(1 + i % 7);
    }
    std::vector<float> b_values(static_cast<size_t>(n) * k);
    for (int r = 0; r < n; ++r) {
      for (int c = 0; c < k; ++c) {
        b_values[static_cast<size_t>(r) * k + c] = float(1 + (r + c) % 3);
      }
    }
    std::vector<float> exact(n, 0.0f);
    for (int col = 0; col < n; ++col) {
      double sum = 0.0;
      for (int i = 0; i < k; ++i) {
        sum += double(a_values[i]) *
            double(b_values[static_cast<size_t>(col) * k + i]);
      }
      exact[col] = host_bf16_round(float(sum));
    }
    array x = astype(
        array(a_values.begin(), Shape{1, 1, k}, float32), bfloat16, stream);
    array w = bf16_weight(b_values, n, k);
    std::vector<float> got = readback_f32(stream, matmul(x, w, stream));
    REQUIRE_EQ(got.size(), exact.size());
    for (int col = 0; col < n; ++col) {
      INFO("k=", k, " n=", n, " col=", col);
      CHECK_EQ(got[col], exact[col]);
    }
    std::cout << "[matmul-bf16-vec] " << k << "x" << n << " exact"
              << std::endl;
  }
}
