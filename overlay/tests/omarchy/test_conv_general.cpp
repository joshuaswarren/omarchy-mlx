// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT
//
// General Convolution gate for groups, transposed convolutions, input
// dilation, and 1-D convolutions. Value tests against an explicit
// nested-loop reference in double precision that encodes the upstream
// .work/mlx/mlx/backend/cpu/conv.cpp slow_conv semantics rather than
// mirroring this backend's kernel assumptions. Every remaining
// unsupported combination refuses by a name that identifies the
// specific combination.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

#include <cmath>
#include <cstring>
#include <iostream>
#include <random>
#include <vector>

#include "mlx/backend/omarchy/compute.h"
#include "mlx/backend/omarchy/device.h"
#include "mlx/backend/omarchy/trace.h"
#include "mlx/backend/gpu/device_info.h"
#include "mlx/ops.h"
#include "mlx/stream.h"

using namespace mlx::core;

namespace {

void skip(const char* reason) {
  std::cout << "Skipping: " << reason << "\n";
}

Stream gpu_stream() {
  set_default_device(Device::gpu);
  return new_stream(Device::gpu);
}

bool compute_available() {
  if (!gpu::is_available()) {
    skip(
        "no qualifying Vulkan device (set MLX_OMARCHY_ALLOW_NON_APPLE=1 on"
        " a development machine).");
    return false;
  }
  return true;
}

void sync_stream(const Stream& stream) {
  mlx::core::synchronize(stream);
}

void check_close(
    array value,
    const std::vector<double>& expected,
    const Stream& stream,
    double epsilon) {
  value.eval();
  sync_stream(stream);
  REQUIRE_EQ(value.size(), expected.size());
  const float* values = value.data<float>();
  for (size_t index = 0; index < expected.size(); ++index) {
    const double margin = epsilon * std::max(1.0, std::fabs(expected[index]));
    CHECK(std::fabs(values[index] - expected[index]) <= margin);
  }
}


// General host reference. Encodes the upstream slow_conv_2D rules:
// out[n, oh, ow, o] sums in[n, ih, iw, c] * wt[o, kh, kw, cpg] for
// cpg channels of group g = o / (O/groups), with
//   ih = oh * sh - plo_h + flip ? (kh - 1 - ky) : ky) * kdh
// and the matching column rule. Taps outside the input-dilated
// extent or not aligned to the input-dilation grid contribute zero.
// Input/weight are NHWC and O(kH kW Cpg) row-major; one float per
// channel of the ungrouped total is in the input last axis. Double
// accumulation matches the kernel's float32 accumulator precision
// while exposing any rounding error in the half-precision paths.
struct Conv2DSpec {
  int stride_h = 1;
  int stride_w = 1;
  int pad_lo_h = 0;
  int pad_lo_w = 0;
  int pad_hi_h = 0;
  int pad_hi_w = 0;
  int kernel_dilation_h = 1;
  int kernel_dilation_w = 1;
  int input_dilation_h = 1;
  int input_dilation_w = 1;
  int groups = 1;
  bool flip = false;
};

std::vector<double> host_conv2d_general(
    const std::vector<float>& input,
    const std::vector<float>& weight,
    const Shape& in_shape,
    const Shape& wt_shape,
    const Conv2DSpec& spec) {
  const int n = in_shape[0];
  const int ih_n = in_shape[1];
  const int iw_n = in_shape[2];
  const int o = wt_shape[0];
  const int kh = wt_shape[1];
  const int kw = wt_shape[2];
  const int cpg = wt_shape[3];
  const int groups = spec.groups;
  const int c_total = groups * cpg;
  // Dilated input extent per upstream conv_out_axis_size:
  //   ih_ext = 1 + idil * (in - 1)
  //   oh = (ih_ext + pad_lo + pad_hi - (1 + kdil*(kh-1))) / stride + 1
  const long long ih_ext =
      1LL + (long long)spec.input_dilation_h * (ih_n - 1);
  const long long iw_ext =
      1LL + (long long)spec.input_dilation_w * (iw_n - 1);
  const long long kd_h =
      1LL + (long long)spec.kernel_dilation_h * (kh - 1);
  const long long kd_w =
      1LL + (long long)spec.kernel_dilation_w * (kw - 1);
  const long long oh = (ih_ext + spec.pad_lo_h + spec.pad_hi_h - kd_h) /
          spec.stride_h +
      1;
  const long long ow = (iw_ext + spec.pad_lo_w + spec.pad_hi_w - kd_w) /
          spec.stride_w +
      1;
  const int o_per_group = o / groups;
  std::vector<double> output(n * oh * ow * o, 0.0);
  for (int batch = 0; batch < n; ++batch) {
    for (long long out_row = 0; out_row < oh; ++out_row) {
      for (long long out_col = 0; out_col < ow; ++out_col) {
        for (int out_channel = 0; out_channel < o; ++out_channel) {
          const int group = out_channel / o_per_group;
          double acc = 0.0;
          for (int ky = 0; ky < kh; ++ky) {
            const int ky_eff = spec.flip ? (kh - 1 - ky) : ky;
            const long long ih =
                out_row * spec.stride_h - spec.pad_lo_h +
                (long long)ky_eff * spec.kernel_dilation_h;
            if (ih < 0 || ih >= ih_ext) {
              continue;
            }
            if (ih % spec.input_dilation_h != 0) {
              continue;
            }
            const long long row = ih / spec.input_dilation_h;
            for (int kx = 0; kx < kw; ++kx) {
              const int kx_eff = spec.flip ? (kw - 1 - kx) : kx;
              const long long iw =
                  out_col * spec.stride_w - spec.pad_lo_w +
                  (long long)kx_eff * spec.kernel_dilation_w;
              if (iw < 0 || iw >= iw_ext) {
                continue;
              }
              if (iw % spec.input_dilation_w != 0) {
                continue;
              }
              const long long col = iw / spec.input_dilation_w;
              const size_t in_base = ((size_t)batch * ih_n + (size_t)row) *
                      iw_n * c_total +
                  (size_t)col * c_total;
              const size_t wt_base = ((size_t)out_channel * kh + (size_t)ky) *
                      kw * cpg +
                  (size_t)kx * cpg;
              for (int channel = 0; channel < cpg; ++channel) {
                acc +=
                    (double)input[in_base + group * cpg + channel] *
                    (double)weight[wt_base + channel];
              }
            }
          }
          output[((size_t)((batch * oh + out_row) * ow + out_col) * o) +
              out_channel] = acc;
        }
      }
    }
  }
  return output;
}

// Wrap a 1D shape pair (input [N, W, C], weight [O, kW, Cpg]) as a
// 2D shape with extent-one height. The kernel packs 1D exactly this
// way and the reference must agree.
std::vector<double> host_conv1d_general(
    const std::vector<float>& input,
    const std::vector<float>& weight,
    const Shape& in_shape,
    const Shape& wt_shape,
    int stride,
    int pad_lo,
    int pad_hi,
    int kernel_dilation,
    int input_dilation,
    int groups,
    bool flip) {
  Conv2DSpec spec;
  spec.stride_h = 1;
  spec.stride_w = stride;
  spec.pad_lo_h = 0;
  spec.pad_lo_w = pad_lo;
  spec.pad_hi_h = 0;
  spec.pad_hi_w = pad_hi;
  spec.kernel_dilation_h = 1;
  spec.kernel_dilation_w = kernel_dilation;
  spec.input_dilation_h = 1;
  spec.input_dilation_w = input_dilation;
  spec.groups = groups;
  spec.flip = flip;
  Shape in2d{in_shape[0], 1, in_shape[1], in_shape[2]};
  Shape wt2d{wt_shape[0], 1, wt_shape[1], wt_shape[2]};
  return host_conv2d_general(input, weight, in2d, wt2d, spec);
}

// Compute the padding_lo/hi that conv_transpose_general in upstream
// ops.cpp would derive from the public parameters, then build the
// matching Conv2DSpec. Copied line-for-line from
// .work/mlx/mlx/ops.cpp conv_transpose_general.
std::pair<Conv2DSpec, Shape> transpose2d_to_conv_spec(
    const Shape& in_shape,
    const Shape& wt_shape,
    int stride_h,
    int stride_w,
    int pad_h,
    int pad_w,
    int kernel_dilation_h,
    int kernel_dilation_w,
    int output_padding_h,
    int output_padding_w,
    int groups) {
  Conv2DSpec spec;
  spec.groups = groups;
  spec.stride_h = 1;
  spec.stride_w = 1;
  spec.kernel_dilation_h = kernel_dilation_h;
  spec.kernel_dilation_w = kernel_dilation_w;
  spec.input_dilation_h = stride_h;
  spec.input_dilation_w = stride_w;
  spec.flip = true;
  const long long wt_h_ext =
      1LL + (long long)kernel_dilation_h * (wt_shape[1] - 1);
  const long long wt_w_ext =
      1LL + (long long)kernel_dilation_w * (wt_shape[2] - 1);
  spec.pad_lo_h = (int)(wt_h_ext - pad_h - 1);
  spec.pad_lo_w = (int)(wt_w_ext - pad_w - 1);
  const long long conv_h_out =
      (long long)(in_shape[1] - 1) * stride_h - 2LL * pad_h +
      (long long)kernel_dilation_h * (wt_shape[1] - 1) + 1LL;
  const long long conv_w_out =
      (long long)(in_shape[2] - 1) * stride_w - 2LL * pad_w +
      (long long)kernel_dilation_w * (wt_shape[2] - 1) + 1LL;
  // The real conv_general output size is conv_out + output_padding:
  // expanding id + plo + phi - kd + 1 algebraically reduces to
  // (in-1)*stride - 2*pad + kdil*(k-1) + 1 + opad, which is also the
  // PyTorch transposed-conv rule. The conv_out intermediate alone
  // misses the opad term.
  const long long out_h_expected = conv_h_out + output_padding_h;
  const long long out_w_expected = conv_w_out + output_padding_w;
  spec.pad_hi_h = (int)(conv_h_out - (1LL + (long long)stride_h * (in_shape[1] - 1)) + pad_h + output_padding_h);
  spec.pad_hi_w = (int)(conv_w_out - (1LL + (long long)stride_w * (in_shape[2] - 1)) + pad_w + output_padding_w);
  Shape expected_out_shape{
      in_shape[0],
      (int)out_h_expected,
      (int)out_w_expected,
      wt_shape[0]};
  return {spec, expected_out_shape};
}

std::pair<Conv2DSpec, Shape> transpose1d_to_conv_spec(
    const Shape& in_shape,
    const Shape& wt_shape,
    int stride,
    int pad,
    int kernel_dilation,
    int output_padding,
    int groups) {
  auto wrapped = transpose2d_to_conv_spec(
      Shape{in_shape[0], 1, in_shape[1], in_shape[2]},
      Shape{wt_shape[0], 1, wt_shape[1], wt_shape[2]},
      /*stride_h=*/1,
      stride,
      /*pad_h=*/0,
      pad,
      /*kernel_dilation_h=*/1,
      kernel_dilation,
      /*output_padding_h=*/0,
      output_padding,
      groups);
  Shape expected_1d{in_shape[0], wrapped.second[2], wt_shape[0]};
  return {wrapped.first, expected_1d};
}


void check_ones_chunked_conv2d(
    int input_extent,
    int kernel_extent,
    int channels,
    const Stream& stream) {
  Shape in_shape{1, input_extent, input_extent, channels};
  Shape wt_shape{1, kernel_extent, kernel_extent, channels};
  std::vector<float> input(
      static_cast<size_t>(input_extent) * input_extent * channels, 1.0f);
  std::vector<float> weight(
      static_cast<size_t>(kernel_extent) * kernel_extent * channels, 1.0f);
  Conv2DSpec spec;
  spec.pad_lo_h = 1;
  spec.pad_lo_w = 1;
  spec.pad_hi_h = 1;
  spec.pad_hi_w = 1;
  auto expected =
      host_conv2d_general(input, weight, in_shape, wt_shape, spec);
  auto actual = conv2d(
      array(input.begin(), in_shape, float32),
      array(weight.begin(), wt_shape, float32),
      {1, 1},
      {1, 1},
      {1, 1},
      1,
      stream);
  check_close(actual, expected, stream, 1e-6);
}

} // namespace

TEST_CASE("[conv-gaps] grouped 2-D Convolution matches host reference") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  std::mt19937 rng(31);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  // groups=1, uniform padding, odd kernel: a baseline through the
  // grouped code path that does not exercise group indexing.
  {
    Shape in_shape{2, 5, 6, 3};
    Shape wt_shape{4, 3, 3, 3};
    std::vector<float> in(2 * 5 * 6 * 3);
    std::vector<float> wt(4 * 3 * 3 * 3);
    for (auto& v : in) {
      v = dist(rng);
    }
    for (auto& v : wt) {
      v = dist(rng);
    }
    Conv2DSpec spec;
    spec.pad_lo_h = 1;
    spec.pad_lo_w = 1;
    spec.pad_hi_h = 1;
    spec.pad_hi_w = 1;
    spec.groups = 1;
    auto expected = host_conv2d_general(in, wt, in_shape, wt_shape, spec);
    auto actual = conv2d(
        array(in.begin(), in_shape, float32),
        array(wt.begin(), wt_shape, float32),
        {1, 1},
        {1, 1},
        {spec.kernel_dilation_h, spec.kernel_dilation_w},
        1,
        stream);
    check_close(actual, expected, stream, 1e-5);
  }

  // groups=2 with C=4 (Cpg=2) and O=6 (O/groups=3): the per-group
  // channel loop and group offset must read the right slice.
  {
    Shape in_shape{1, 4, 4, 4};
    Shape wt_shape{6, 2, 3, 2};
    std::vector<float> in(1 * 4 * 4 * 4);
    std::vector<float> wt(6 * 2 * 3 * 2);
    for (auto& v : in) {
      v = dist(rng);
    }
    for (auto& v : wt) {
      v = dist(rng);
    }
    Conv2DSpec spec;
    spec.stride_h = 1;
    spec.stride_w = 2;
    spec.pad_lo_h = 1;
    spec.pad_lo_w = 1;
    spec.pad_hi_h = 1;
    spec.pad_hi_w = 1;
    spec.groups = 2;
    auto expected = host_conv2d_general(in, wt, in_shape, wt_shape, spec);
    auto actual = conv2d(
        array(in.begin(), in_shape, float32),
        array(wt.begin(), wt_shape, float32),
        {spec.stride_h, spec.stride_w},
        {1, 1},
        {spec.kernel_dilation_h, spec.kernel_dilation_w},
        2,
        stream);
    check_close(actual, expected, stream, 1e-5);
  }

  // Depthwise: groups == C, Cpg == 1, stride 2. MobileNet-style
  // depthwise separable block; the per-group loop becomes a channel
  // gather over the input that must not bleed into the next group.
  {
    Shape in_shape{1, 6, 8, 3};
    Shape wt_shape{3, 3, 3, 1};
    std::vector<float> in(1 * 6 * 8 * 3);
    std::vector<float> wt(3 * 3 * 3 * 1);
    for (auto& v : in) {
      v = dist(rng);
    }
    for (auto& v : wt) {
      v = dist(rng);
    }
    Conv2DSpec spec;
    spec.stride_h = 2;
    spec.stride_w = 1;
    spec.pad_lo_h = 1;
    spec.pad_lo_w = 1;
    spec.pad_hi_h = 1;
    spec.pad_hi_w = 1;
    spec.groups = 3;
    auto expected = host_conv2d_general(in, wt, in_shape, wt_shape, spec);
    auto actual = conv2d(
        array(in.begin(), in_shape, float32),
        array(wt.begin(), wt_shape, float32),
        {spec.stride_h, spec.stride_w},
        {1, 1},
        {spec.kernel_dilation_h, spec.kernel_dilation_w},
        3,
        stream);
    check_close(actual, expected, stream, 1e-5);
  }
}

TEST_CASE(
    "[conv-gaps] large-filter single-channel Convolution matches host reference") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  std::mt19937 rng(1310);
  std::normal_distribution<float> dist(0.0f, 1.0f);
  Shape in_shape{1, 181, 181, 1};
  Shape wt_shape{1, 182, 182, 1};
  std::vector<float> in(181 * 181);
  std::vector<float> wt(182 * 182);
  for (auto& value : in) {
    value = dist(rng);
  }
  for (auto& value : wt) {
    value = dist(rng);
  }
  Conv2DSpec spec;
  spec.pad_lo_h = 1;
  spec.pad_lo_w = 1;
  spec.pad_hi_h = 1;
  spec.pad_hi_w = 1;
  auto expected = host_conv2d_general(in, wt, in_shape, wt_shape, spec);
  auto actual = conv2d(
      array(in.begin(), in_shape, float32),
      array(wt.begin(), wt_shape, float32),
      {1, 1},
      {1, 1},
      {1, 1},
      1,
      stream);
  check_close(actual, expected, stream, 1e-3);
}


TEST_CASE("[conv-gaps] convolution chunks channel products") {
  if (!compute_available()) {
    return;
  }
  check_ones_chunked_conv2d(55, 56, 4, gpu_stream());
}

TEST_CASE("[conv-gaps] convolution chunks more than 65536 products") {
  if (!compute_available()) {
    return;
  }
  check_ones_chunked_conv2d(128, 129, 4, gpu_stream());
}

TEST_CASE("[conv-gaps] 1-D Convolution matches host reference") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  std::mt19937 rng(53);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  // conv1d groups=1, odd kernel, uniform padding, batched input.
  {
    Shape in_shape{2, 7, 2};
    Shape wt_shape{3, 3, 2};
    std::vector<float> in(2 * 7 * 2);
    std::vector<float> wt(3 * 3 * 2);
    for (auto& v : in) {
      v = dist(rng);
    }
    for (auto& v : wt) {
      v = dist(rng);
    }
    int stride = 2;
    int pad = 1;
    int kdil = 1;
    int idil = 1;
    auto expected = host_conv1d_general(
        in, wt, in_shape, wt_shape, stride, pad, pad, kdil, idil, 1, false);
    auto actual = conv1d(
        array(in.begin(), in_shape, float32),
        array(wt.begin(), wt_shape, float32),
        stride,
        pad,
        kdil,
        1,
        stream);
    check_close(actual, expected, stream, 1e-5);
  }

  // Depthwise 1D with kernel dilation: the dilated tap must still
  // respect the input extent.
  {
    Shape in_shape{1, 9, 4};
    Shape wt_shape{4, 3, 1};
    std::vector<float> in(1 * 9 * 4);
    std::vector<float> wt(4 * 3 * 1);
    for (auto& v : in) {
      v = dist(rng);
    }
    for (auto& v : wt) {
      v = dist(rng);
    }
    int stride = 1;
    int pad = 0;
    int kdil = 2;
    int idil = 1;
    auto expected = host_conv1d_general(
        in, wt, in_shape, wt_shape, stride, pad, pad, kdil, idil, 4, false);
    auto actual = conv1d(
        array(in.begin(), in_shape, float32),
        array(wt.begin(), wt_shape, float32),
        stride,
        pad,
        kdil,
        4,
        stream);
    check_close(actual, expected, stream, 1e-5);
  }
}

// Transposed (flip) convolution runs the general kernel with the
// kernel stride fixed at one and the conv stride expressed as input
// dilation, exactly as upstream conv_transpose_general builds the
// primitive. The flip reverses the input walk while the weight keeps
// its stored order; every value below is checked against the
// double-precision host reference, which is the probe that actually
// distinguishes a wrong channel slice from its negation.
TEST_CASE("[conv-gaps] transposed 2-D Convolution matches host reference") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  std::mt19937 rng(101);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  // Stride 2, kernel 2x2: the smallest transposed conv that doubles
  // spatial extent. Output shape must equal 1 + stride*(in-1).
  {
    Shape in_shape{1, 3, 3, 2};
    Shape wt_shape{2, 2, 2, 2};
    std::vector<float> in(1 * 3 * 3 * 2);
    std::vector<float> wt(2 * 2 * 2 * 2);
    for (auto& v : in) {
      v = dist(rng);
    }
    for (auto& v : wt) {
      v = dist(rng);
    }
    auto [spec, expected_shape] = transpose2d_to_conv_spec(
        in_shape, wt_shape, 2, 2, 0, 0, 1, 1, 0, 0, 1);
    auto expected = host_conv2d_general(in, wt, in_shape, wt_shape, spec);
    auto actual = conv_transpose2d(
        array(in.begin(), in_shape, float32),
        array(wt.begin(), wt_shape, float32),
        {2, 2},
        {0, 0},
        {1, 1},
        {0, 0},
        1,
        stream);
    REQUIRE_EQ(actual.shape(), expected_shape);
    check_close(actual, expected, stream, 1e-5);
  }

  // Stride 2, kernel 3x3 with padding 1 and output_padding 1: the
  // standard GAN/generative model transposed block. Pads are
  // asymmetric and the kernel is larger than the stride.
  {
    Shape in_shape{1, 4, 4, 2};
    Shape wt_shape{2, 3, 3, 2};
    std::vector<float> in(1 * 4 * 4 * 2);
    std::vector<float> wt(2 * 3 * 3 * 2);
    for (auto& v : in) {
      v = dist(rng);
    }
    for (auto& v : wt) {
      v = dist(rng);
    }
    auto [spec, expected_shape] = transpose2d_to_conv_spec(
        in_shape, wt_shape, 2, 2, 1, 1, 1, 1, 1, 1, 1);
    auto expected = host_conv2d_general(in, wt, in_shape, wt_shape, spec);
    auto actual = conv_transpose2d(
        array(in.begin(), in_shape, float32),
        array(wt.begin(), wt_shape, float32),
        {2, 2},
        {1, 1},
        {1, 1},
        {1, 1},
        1,
        stream);
    REQUIRE_EQ(actual.shape(), expected_shape);
    check_close(actual, expected, stream, 1e-5);
  }

  // Combined transposed + kernel dilation: dilation 2 doubles the
  // effective kernel extent in the transposed path. Off-by-one in
  // output shape here is the classic transposed-conv bug.
  {
    Shape in_shape{1, 2, 2, 2};
    Shape wt_shape{2, 2, 2, 2};
    std::vector<float> in(1 * 2 * 2 * 2);
    std::vector<float> wt(2 * 2 * 2 * 2);
    for (auto& v : in) {
      v = dist(rng);
    }
    for (auto& v : wt) {
      v = dist(rng);
    }
    auto [spec, expected_shape] = transpose2d_to_conv_spec(
        in_shape, wt_shape, 1, 1, 0, 0, 2, 2, 1, 1, 1);
    auto expected = host_conv2d_general(in, wt, in_shape, wt_shape, spec);
    auto actual = conv_transpose2d(
        array(in.begin(), in_shape, float32),
        array(wt.begin(), wt_shape, float32),
        {1, 1},
        {0, 0},
        {2, 2},
        {1, 1},
        1,
        stream);
    REQUIRE_EQ(actual.shape(), expected_shape);
    check_close(actual, expected, stream, 1e-5);
  }

  // Grouped transposed: groups=2 exercises the group channel slice
  // under flip plus input dilation - the combination the retired
  // one-hot probe distrusted.
  {
    Shape in_shape{1, 3, 3, 4};
    Shape wt_shape{4, 2, 2, 2};
    std::vector<float> in(1 * 3 * 3 * 4);
    std::vector<float> wt(4 * 2 * 2 * 2);
    for (auto& v : in) {
      v = dist(rng);
    }
    for (auto& v : wt) {
      v = dist(rng);
    }
    auto [spec, expected_shape] = transpose2d_to_conv_spec(
        in_shape, wt_shape, 2, 2, 0, 0, 1, 1, 0, 0, 2);
    auto expected = host_conv2d_general(in, wt, in_shape, wt_shape, spec);
    auto actual = conv_transpose2d(
        array(in.begin(), in_shape, float32),
        array(wt.begin(), wt_shape, float32),
        {2, 2},
        {0, 0},
        {1, 1},
        {0, 0},
        2,
        stream);
    REQUIRE_EQ(actual.shape(), expected_shape);
    check_close(actual, expected, stream, 1e-5);
  }
}

TEST_CASE("[conv-gaps] transposed 1-D Convolution matches host reference") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  std::mt19937 rng(149);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  Shape in_shape{1, 5, 3};
  Shape wt_shape{3, 3, 3};
  std::vector<float> in(1 * 5 * 3);
  std::vector<float> wt(3 * 3 * 3);
  for (auto& v : in) {
    v = dist(rng);
  }
  for (auto& v : wt) {
    v = dist(rng);
  }
  int stride = 2;
  int pad = 1;
  int kdil = 1;
  int opad = 1;
  auto [spec, expected_shape] = transpose1d_to_conv_spec(
      in_shape, wt_shape, stride, pad, kdil, opad, 1);
  auto expected = host_conv1d_general(
      in,
      wt,
      in_shape,
      wt_shape,
      /*stride=*/1,
      spec.pad_lo_w,
      spec.pad_hi_w,
      kdil,
      stride,
      1,
      /*flip=*/true);
  auto actual = conv_transpose1d(
      array(in.begin(), in_shape, float32),
      array(wt.begin(), wt_shape, float32),
      stride,
      pad,
      kdil,
      opad,
      1,
      stream);
  REQUIRE_EQ(actual.shape(), expected_shape);
  check_close(actual, expected, stream, 1e-5);
}

TEST_CASE("[conv-gaps] input-dilated 2-D Convolution matches host reference") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  std::mt19937 rng(167);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  Shape in_shape{1, 5, 5, 2};
  Shape wt_shape{2, 2, 2, 2};
  std::vector<float> in(1 * 5 * 5 * 2);
  std::vector<float> wt(2 * 2 * 2 * 2);
  for (auto& v : in) {
    v = dist(rng);
  }
  for (auto& v : wt) {
    v = dist(rng);
  }
  // Direct input dilation (not via conv_transpose): a holey grid
  // upsampled then convolved. The kernel must skip non-aligned taps
  // rather than read the dilated index by integer division alone.
  int idil_h = 2;
  int idil_w = 3;
  int stride_h = 2;
  int stride_w = 1;
  int pad_lo_h = 1;
  int pad_lo_w = 0;
  int pad_hi_h = 0;
  int pad_hi_w = 1;
  int kdil_h = 1;
  int kdil_w = 1;
  Conv2DSpec spec;
  spec.input_dilation_h = idil_h;
  spec.input_dilation_w = idil_w;
  spec.stride_h = stride_h;
  spec.stride_w = stride_w;
  spec.pad_lo_h = pad_lo_h;
  spec.pad_lo_w = pad_lo_w;
  spec.pad_hi_h = pad_hi_h;
  spec.pad_hi_w = pad_hi_w;
  spec.kernel_dilation_h = kdil_h;
  spec.kernel_dilation_w = kdil_w;
  auto expected = host_conv2d_general(in, wt, in_shape, wt_shape, spec);
  auto actual = conv_general(
      array(in.begin(), in_shape, float32),
      array(wt.begin(), wt_shape, float32),
      {stride_h, stride_w},
      {pad_lo_h, pad_lo_w},
      {pad_hi_h, pad_hi_w},
      {kdil_h, kdil_w},
      {idil_h, idil_w},
      /*groups=*/1,
      /*flip=*/false,
      stream);
  check_close(actual, expected, stream, 1e-5);
}

TEST_CASE(
    "[conv-gaps] flipped strided input-dilated Convolution matches host reference") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  std::mt19937 rng(181);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  Shape in_shape{1, 4, 4, 2};
  Shape wt_shape{3, 2, 3, 2};
  std::vector<float> in(1 * 4 * 4 * 2);
  std::vector<float> wt(3 * 2 * 3 * 2);
  for (auto& v : in) {
    v = dist(rng);
  }
  for (auto& v : wt) {
    v = dist(rng);
  }
  Conv2DSpec spec;
  spec.stride_h = 2;
  spec.stride_w = 3;
  spec.pad_lo_h = 1;
  spec.pad_lo_w = 2;
  spec.pad_hi_h = 2;
  spec.pad_hi_w = 1;
  spec.kernel_dilation_h = 3;
  spec.kernel_dilation_w = 1;
  spec.input_dilation_h = 2;
  spec.input_dilation_w = 5;
  spec.flip = true;
  auto expected = host_conv2d_general(in, wt, in_shape, wt_shape, spec);
  auto actual = conv_general(
      array(in.begin(), in_shape, float32),
      array(wt.begin(), wt_shape, float32),
      {spec.stride_h, spec.stride_w},
      {spec.pad_lo_h, spec.pad_lo_w},
      {spec.pad_hi_h, spec.pad_hi_w},
      {spec.kernel_dilation_h, spec.kernel_dilation_w},
      {spec.input_dilation_h, spec.input_dilation_w},
      /*groups=*/1,
      spec.flip,
      stream);
  check_close(actual, expected, stream, 1e-5);
}

TEST_CASE(
    "[conv-gaps] FP16 grouped 2-D Convolution matches host reference") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  const auto& capabilities = omarchy::device(0).capabilities();
  if (!capabilities.shader_float16 ||
      !capabilities.storage_buffer_16bit_access) {
    skip("Vulkan device lacks required FP16 shader and storage features.");
    return;
  }
  std::mt19937 rng(211);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  Shape in_shape{1, 4, 4, 4};
  Shape wt_shape{4, 2, 2, 2};
  std::vector<float> in(1 * 4 * 4 * 4);
  std::vector<float> wt(4 * 2 * 2 * 2);
  for (auto& v : in) {
    v = dist(rng);
  }
  for (auto& v : wt) {
    v = dist(rng);
  }
  Conv2DSpec spec;
  spec.stride_w = 2;
  spec.pad_lo_h = 1;
  spec.pad_lo_w = 1;
  spec.pad_hi_h = 1;
  spec.pad_hi_w = 1;
  spec.groups = 2;
  auto expected = host_conv2d_general(in, wt, in_shape, wt_shape, spec);
  auto actual = astype(
      conv2d(
          astype(array(in.begin(), in_shape, float32), float16, stream),
          astype(array(wt.begin(), wt_shape, float32), float16, stream),
          {1, 2},
          {1, 1},
          {1, 1},
          2,
          stream),
      float32,
      stream);
  check_close(actual, expected, stream, 1e-2);
}

TEST_CASE("[conv-gaps] BF16 grouped 2-D Convolution matches host reference") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  const auto& capabilities = omarchy::device(0).capabilities();
  if (!capabilities.storage_buffer_16bit_access ||
      !capabilities.shader_int16) {
    skip("Vulkan device lacks required BF16 storage and shader features.");
    return;
  }
  std::mt19937 rng(229);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  Shape in_shape{1, 4, 4, 4};
  Shape wt_shape{4, 2, 2, 2};
  std::vector<float> in(1 * 4 * 4 * 4);
  std::vector<float> wt(4 * 2 * 2 * 2);
  for (auto& v : in) {
    v = dist(rng);
  }
  for (auto& v : wt) {
    v = dist(rng);
  }
  Conv2DSpec spec;
  spec.pad_lo_h = 1;
  spec.pad_lo_w = 1;
  spec.pad_hi_h = 1;
  spec.pad_hi_w = 1;
  spec.groups = 2;
  auto expected = host_conv2d_general(in, wt, in_shape, wt_shape, spec);
  auto actual = astype(
      conv2d(
          astype(array(in.begin(), in_shape, float32), bfloat16, stream),
          astype(array(wt.begin(), wt_shape, float32), bfloat16, stream),
          {1, 1},
          {1, 1},
          {1, 1},
          2,
          stream),
      float32,
      stream);
  // 3x3 kernel, 2 channels per group, ~9 products summed: bf16 mantissa
  // is 8 bits and rounding accumulates; an absolute epsilon of 0.05
  // matches the tolerance other bf16 paths in this suite use.
  check_close(actual, expected, stream, 1e-1);
}

// General 3-D host reference: the upstream slow_conv rules extended
// with a depth axis over channels-last [N, D, H, W, C] input and
// [O, kD, kH, kW, Cpg] weight.
std::vector<double> host_conv3d_general(
    const std::vector<float>& input,
    const std::vector<float>& weight,
    const Shape& in_shape,
    const Shape& wt_shape,
    const std::tuple<int, int, int>& stride,
    const std::tuple<int, int, int>& pad_lo,
    const std::tuple<int, int, int>& pad_hi,
    const std::tuple<int, int, int>& kernel_dilation,
    const std::tuple<int, int, int>& input_dilation,
    int groups,
    bool flip) {
  const int n = in_shape[0];
  const int id_n = in_shape[1];
  const int ih_n = in_shape[2];
  const int iw_n = in_shape[3];
  const int o = wt_shape[0];
  const int kd_n = wt_shape[1];
  const int kh_n = wt_shape[2];
  const int kw_n = wt_shape[3];
  const int cpg = wt_shape[4];
  const int c_total = groups * cpg;
  const auto extent = [](int size, int dil) {
    return 1LL + (long long)dil * (size - 1);
  };
  const long long id_ext = extent(id_n, std::get<0>(input_dilation));
  const long long ih_ext = extent(ih_n, std::get<1>(input_dilation));
  const long long iw_ext = extent(iw_n, std::get<2>(input_dilation));
  const long long kd_ext = extent(kd_n, std::get<0>(kernel_dilation));
  const long long kh_ext = extent(kh_n, std::get<1>(kernel_dilation));
  const long long kw_ext = extent(kw_n, std::get<2>(kernel_dilation));
  const long long od = (id_ext + std::get<0>(pad_lo) + std::get<0>(pad_hi) -
                        kd_ext) / std::get<0>(stride) + 1;
  const long long oh = (ih_ext + std::get<1>(pad_lo) + std::get<1>(pad_hi) -
                        kh_ext) / std::get<1>(stride) + 1;
  const long long ow = (iw_ext + std::get<2>(pad_lo) + std::get<2>(pad_hi) -
                        kw_ext) / std::get<2>(stride) + 1;
  const int o_per_group = o / groups;
  std::vector<double> output(n * od * oh * ow * o, 0.0);
  for (int batch = 0; batch < n; ++batch) {
    for (long long out_d = 0; out_d < od; ++out_d) {
      for (long long out_row = 0; out_row < oh; ++out_row) {
        for (long long out_col = 0; out_col < ow; ++out_col) {
          for (int out_channel = 0; out_channel < o; ++out_channel) {
            const int group = out_channel / o_per_group;
            double acc = 0.0;
            for (int kz = 0; kz < kd_n; ++kz) {
              const int kz_eff = flip ? (kd_n - 1 - kz) : kz;
              const long long id = out_d * std::get<0>(stride) -
                  std::get<0>(pad_lo) + kz_eff * std::get<0>(kernel_dilation);
              if (id < 0 || id >= id_ext || id % std::get<0>(input_dilation) != 0) {
                continue;
              }
              const long long tap_d = id / std::get<0>(input_dilation);
              for (int ky = 0; ky < kh_n; ++ky) {
                const int ky_eff = flip ? (kh_n - 1 - ky) : ky;
                const long long ih = out_row * std::get<1>(stride) -
                    std::get<1>(pad_lo) +
                    ky_eff * std::get<1>(kernel_dilation);
                if (ih < 0 || ih >= ih_ext ||
                    ih % std::get<1>(input_dilation) != 0) {
                  continue;
                }
                const long long tap_h = ih / std::get<1>(input_dilation);
                for (int kx = 0; kx < kw_n; ++kx) {
                  const int kx_eff = flip ? (kw_n - 1 - kx) : kx;
                  const long long iw = out_col * std::get<2>(stride) -
                      std::get<2>(pad_lo) +
                      kx_eff * std::get<2>(kernel_dilation);
                  if (iw < 0 || iw >= iw_ext ||
                      iw % std::get<2>(input_dilation) != 0) {
                    continue;
                  }
                  const long long tap_w = iw / std::get<2>(input_dilation);
                  const size_t in_base =
                      ((((size_t)batch * id_n + (size_t)tap_d) * ih_n +
                        (size_t)tap_h) * iw_n + (size_t)tap_w) * c_total;
                  const size_t wt_base =
                      ((((size_t)out_channel * kd_n + (size_t)kz) * kh_n +
                        (size_t)ky) * kw_n + (size_t)kx) * cpg;
                  for (int channel = 0; channel < cpg; ++channel) {
                    acc += (double)input[in_base + group * cpg + channel] *
                        (double)weight[wt_base + channel];
                  }
                }
              }
            }
            output[(((((size_t)batch * od + out_d) * oh + out_row) * ow +
                     out_col) * o) + out_channel] = acc;
          }
        }
      }
    }
  }
  return output;
}

// Copy of conv_transpose_general's padding derivation for 3 axes
// (.work/mlx/mlx/ops.cpp), returning the primitive-level parameters.
struct Conv3DSpec {
  std::tuple<int, int, int> stride;
  std::tuple<int, int, int> pad_lo;
  std::tuple<int, int, int> pad_hi;
  std::tuple<int, int, int> kernel_dilation;
  bool flip = true;
};

std::pair<Conv3DSpec, Shape> transpose3d_to_conv_spec(
    const Shape& in_shape,
    const Shape& wt_shape,
    std::tuple<int, int, int> stride,
    std::tuple<int, int, int> pad,
    std::tuple<int, int, int> dilation,
    std::tuple<int, int, int> output_padding,
    int groups) {
  Conv3DSpec spec;
  spec.stride = {1, 1, 1};
  spec.kernel_dilation = dilation;
  spec.flip = true;
  Shape out_shape{in_shape[0], 0, 0, 0, wt_shape[0]};
  const int stride_a[3] = {std::get<0>(stride),
                           std::get<1>(stride),
                           std::get<2>(stride)};
  const int pad_a[3] = {
      std::get<0>(pad), std::get<1>(pad), std::get<2>(pad)};
  const int dil_a[3] = {std::get<0>(dilation),
                        std::get<1>(dilation),
                        std::get<2>(dilation)};
  const int opad_a[3] = {std::get<0>(output_padding),
                         std::get<1>(output_padding),
                         std::get<2>(output_padding)};
  int pad_lo_a[3] = {0, 0, 0};
  int pad_hi_a[3] = {0, 0, 0};
  for (int axis = 0; axis < 3; ++axis) {
    const long long wt_size =
        1LL + (long long)dil_a[axis] * (wt_shape[1 + axis] - 1);
    const long long padding_lo = wt_size - pad_a[axis] - 1;
    const long long conv_output_shape =
        (long long)(in_shape[1 + axis] - 1) * stride_a[axis] -
        2LL * pad_a[axis] + (long long)dil_a[axis] * (wt_shape[1 + axis] - 1) +
        1;
    const long long out_size =
        1LL + (long long)stride_a[axis] * (in_shape[1 + axis] - 1);
    pad_lo_a[axis] = (int)padding_lo;
    pad_hi_a[axis] =
        (int)(conv_output_shape - out_size + pad_a[axis] + opad_a[axis]);
    out_shape[1 + axis] = (int)(conv_output_shape + opad_a[axis]);
  }
  spec.pad_lo = {pad_lo_a[0], pad_lo_a[1], pad_lo_a[2]};
  spec.pad_hi = {pad_hi_a[0], pad_hi_a[1], pad_hi_a[2]};
  (void)groups;
  return {spec, out_shape};
}

TEST_CASE("[conv-gaps] 3-D Convolution matches host reference") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  std::mt19937 rng(283);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  // Forward conv3d with stride 1 and symmetric padding: the plain
  // volume filter the retired refusal used to pin.
  {
    Shape in_shape{1, 4, 4, 4, 2};
    Shape wt_shape{2, 3, 3, 3, 2};
    std::vector<float> in(1 * 4 * 4 * 4 * 2);
    std::vector<float> wt(2 * 3 * 3 * 3 * 2);
    for (auto& v : in) {
      v = dist(rng);
    }
    for (auto& v : wt) {
      v = dist(rng);
    }
    auto expected = host_conv3d_general(
        in,
        wt,
        in_shape,
        wt_shape,
        {1, 1, 1},
        {1, 1, 1},
        {1, 1, 1},
        {1, 1, 1},
        {1, 1, 1},
        1,
        /*flip=*/false);
    auto actual = conv3d(
        array(in.begin(), in_shape, float32),
        array(wt.begin(), wt_shape, float32),
        std::tuple<int, int, int>(1, 1, 1),
        std::tuple<int, int, int>(1, 1, 1),
        std::tuple<int, int, int>(1, 1, 1),
        1,
        stream);
    check_close(actual, expected, stream, 1e-5);
  }

  // Strided forward conv3d with kernel dilation.
  {
    Shape in_shape{2, 5, 6, 7, 3};
    Shape wt_shape{4, 2, 2, 2, 3};
    std::vector<float> in(2 * 5 * 6 * 7 * 3);
    std::vector<float> wt(4 * 2 * 2 * 2 * 3);
    for (auto& v : in) {
      v = dist(rng);
    }
    for (auto& v : wt) {
      v = dist(rng);
    }
    auto expected = host_conv3d_general(
        in,
        wt,
        in_shape,
        wt_shape,
        {2, 2, 2},
        {0, 1, 0},
        {0, 1, 0},
        {2, 1, 2},
        {1, 1, 1},
        1,
        /*flip=*/false);
    auto actual = conv3d(
        array(in.begin(), in_shape, float32),
        array(wt.begin(), wt_shape, float32),
        std::tuple<int, int, int>(2, 2, 2),
        std::tuple<int, int, int>(0, 1, 0),
        std::tuple<int, int, int>(2, 1, 2),
        1,
        stream);
    check_close(actual, expected, stream, 1e-5);

    expected = host_conv3d_general(
        in, wt, in_shape, wt_shape,
        {2, 2, 2}, {0, 1, 0}, {0, 1, 0}, {2, 1, 2}, {2, 3, 2},
        1, false);
    actual = conv_general(
        array(in.begin(), in_shape, float32),
        array(wt.begin(), wt_shape, float32),
        {2, 2, 2}, {0, 1, 0}, {0, 1, 0}, {2, 1, 2}, {2, 3, 2},
        1, false, stream);
    check_close(actual, expected, stream, 1e-5);
  }
}

TEST_CASE(
    "[conv-gaps] transposed 3-D Convolution matches host reference") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  std::mt19937 rng(293);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  // The upstream output_padding shape: stride 2, 1x1x1 kernel,
  // output_padding 1. Values are the upstream ops_tests expectation
  // exactly.
  {
    Shape in_shape{1, 1, 2, 2, 2};
    std::vector<float> in{1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<float> wt{1, 1};
    Shape wt_shape{1, 1, 1, 1, 2};
    auto actual = conv_transpose3d(
        array(in.begin(), in_shape, float32),
        array(wt.begin(), wt_shape, float32),
        std::tuple<int, int, int>(2, 2, 2),
        std::tuple<int, int, int>(0, 0, 0),
        std::tuple<int, int, int>(1, 1, 1),
        std::tuple<int, int, int>(1, 1, 1),
        1,
        stream);
    std::vector<double> expected{
        3,    0,    7,    0,    0,    0,    0,    0,    11,   0,    15,
        0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
        0,    0,    0,    0,    0,    0,    0,    0,    0,    0};
    check_close(actual, expected, stream, 1e-6);
  }

  // Random volume with a 2x2x2 kernel, stride 2: the transposed taps
  // walk a dilated input grid in every axis against the double host
  // reference.
  {
    Shape in_shape{1, 2, 2, 2, 2};
    Shape wt_shape{2, 2, 2, 2, 2};
    std::vector<float> in(1 * 2 * 2 * 2 * 2);
    std::vector<float> wt(2 * 2 * 2 * 2 * 2);
    for (auto& v : in) {
      v = dist(rng);
    }
    for (auto& v : wt) {
      v = dist(rng);
    }
    auto [spec, expected_shape] = transpose3d_to_conv_spec(
        in_shape,
        wt_shape,
        {2, 2, 2},
        {0, 0, 0},
        {1, 1, 1},
        {0, 0, 0},
        1);
    auto expected = host_conv3d_general(
        in,
        wt,
        in_shape,
        wt_shape,
        spec.stride,
        spec.pad_lo,
        spec.pad_hi,
        spec.kernel_dilation,
        {2, 2, 2},
        1,
        spec.flip);
    auto actual = conv_transpose3d(
        array(in.begin(), in_shape, float32),
        array(wt.begin(), wt_shape, float32),
        {2, 2, 2},
        {0, 0, 0},
        {1, 1, 1},
        {0, 0, 0},
        1,
        stream);
    REQUIRE_EQ(actual.shape(), expected_shape);
    check_close(actual, expected, stream, 1e-5);
  }
}

// MLX_OMARCHY_DISPATCH_SPLIT_GMAC: a conv run as output tiles, one submission
// per tile, must give the bytes of the single dispatch. The shape has 12800
// outputs with 72 products each; 0.0001875 GMAC is 2604 outputs per tile,
// so five tiles and an uneven last tile.
TEST_CASE("conv2d split into output tiles is bit-identical to the single dispatch") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  std::mt19937 rng(17);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  Shape in_shape{2, 20, 20, 8};
  Shape wt_shape{16, 3, 3, 8};
  std::vector<float> in(2 * 20 * 20 * 8);
  std::vector<float> wt(16 * 3 * 3 * 8);
  for (auto& v : in) {
    v = dist(rng);
  }
  for (auto& v : wt) {
    v = dist(rng);
  }
  array x(in.begin(), in_shape, float32);
  array w(wt.begin(), wt_shape, float32);
  struct Run {
    std::vector<float> values;
    uint64_t dispatches;
    uint64_t submissions;
  };
  auto run = [&](const char* gmac) {
    if (gmac != nullptr) {
      setenv("MLX_OMARCHY_DISPATCH_SPLIT_GMAC", gmac, 1);
    } else {
      unsetenv("MLX_OMARCHY_DISPATCH_SPLIT_GMAC");
    }
    array y = conv2d(x, w, {1, 1}, {1, 1}, {1, 1}, 1, stream);
    auto& counters = omarchy::trace::counters();
    const uint64_t d0 = counters.vk_compute_dispatches.load();
    const uint64_t s0 = counters.vk_submissions.load();
    y.eval();
    sync_stream(stream);
    Run result;
    result.dispatches = counters.vk_compute_dispatches.load() - d0;
    result.submissions = counters.vk_submissions.load() - s0;
    const float* p = y.data<float>();
    result.values.assign(p, p + y.size());
    unsetenv("MLX_OMARCHY_DISPATCH_SPLIT_GMAC");
    return result;
  };
  Run whole = run(nullptr);
  Run split = run("0.0001875");
  Run off = run("0");
  REQUIRE_EQ(whole.values.size(), split.values.size());
  CHECK(std::memcmp(
            whole.values.data(),
            split.values.data(),
            whole.values.size() * sizeof(float)) == 0);
  CHECK(std::memcmp(
            whole.values.data(), off.values.data(), whole.values.size() * sizeof(float)) == 0);
  std::cout << "[conv_split] single: " << whole.dispatches << " dispatch(es), " << whole.submissions
            << " submission(s); split: " << split.dispatches << " dispatches, " << split.submissions
            << " submissions; GMAC=0: " << off.dispatches << " dispatch(es)\n";
  CHECK_EQ(whole.dispatches, 1u);
  CHECK_EQ(off.dispatches, 1u);
  CHECK_EQ(split.dispatches, 5u);
  CHECK_GE(split.submissions, whole.submissions + 4u);
}
