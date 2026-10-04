// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

// Distributed contract after the GPU-stream cutover. The ring transport
// is compiled and linked (MLX_BUILD_CPU=ON compiles
// mlx/distributed/ring/ring.cpp and the upstream CPU backend). On a host
// with a usable GPU device the ring communication stream is the
// accelerator stream (patches/mlx-ring-gpu-transport.patch:
// RingGroup::communication_stream -> to_stream(s, Device::gpu)), and the
// five distributed primitives evaluate as
// omarchy GPU operations (overlay/mlx/backend/omarchy/primitives.cpp,
// namespace distributed): host-side byte transport; tensor arithmetic
// stays on the GPU. A missing GPU cannot select upstream CPU tensor eval.
//
// A single-process run forms NO multi-rank group: without MLX_HOSTFILE and
// MLX_RANK, init() yields the singleton EmptyGroup (size 1), upstream's op
// layer short-circuits every collective at group.size() == 1
// (mlx/distributed/ops.cpp: all_reduce line 29, all_gather line 69,
// sum_scatter line 149) and throws invalid_argument for send (line 94) and
// recv (line 118). THIS suite pins that singleton contract only.
//
// The primitives themselves are value-proven at two ranks by
// distributed/test_two_rank.cpp, driven by
// distributed/run-two-rank.sh: every case there asserts a real two-rank
// group first, so a lone run fails instead of passing vacuously. The
// identity checks below are deliberately NOT primitive coverage.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

#include <stdexcept>
#include <vector>

#include "mlx/transforms.h"
#include "mlx/distributed/distributed.h"
#include "mlx/distributed/ops.h"
#include "mlx/ops.h"

using namespace mlx::core;

TEST_CASE("ring links; nccl and jaccl do not; init is the singleton") {
  // Ring availability is a build property: MLX_BUILD_CPU=ON compiles
  // ring.cpp, so is_available("ring") is true on any host. A multi-rank
  // group forms only with MLX_HOSTFILE and MLX_RANK; without them init()
  // yields the singleton EmptyGroup and every collective short-circuits.
  // Three agents in one night mistook these three bare failures for a
  // device defect, so they name the cause: the only way for them to
  // fail is a tree configured with MLX_BUILD_CPU=OFF, which leaves
  // ring.cpp uncompiled. Every build script here pins the flag ON.
  // std::string, not const char*: doctest stringifies a raw pointer as
  // its address, so the hint would print as 0x... and help nobody.
  const std::string ring_hint =
      "ring is unavailable, so this tree was configured with"
      " MLX_BUILD_CPU=OFF and ring.cpp was never compiled. This is a"
      " build-configuration mismatch, not a device defect: pass"
      " -DMLX_BUILD_CPU=ON, as every build script in scripts/ does.";
  CHECK_MESSAGE(distributed::is_available(), ring_hint);
  CHECK_MESSAGE(distributed::is_available("any"), ring_hint);
  CHECK_MESSAGE(distributed::is_available("ring"), ring_hint);
  // nccl is gated on MLX_BUILD_CUDA (OFF here); jaccl on Darwin with
  // SDK >= 26.2. Both remain compile-time stubs (no_nccl.cpp:7-9,
  // no_jaccl.cpp:7-9).
  CHECK_FALSE(distributed::is_available("nccl"));
  CHECK_FALSE(distributed::is_available("jaccl"));
  // mpi.cpp is also compiled under MLX_BUILD_CPU, but availability is
  // runtime dlopen("libmpi.so") (mpi.cpp MPIWrapper). This box has no
  // OpenMPI; if this check ever fails, a libmpi appeared on the host and
  // the mpi rows of the contract must be re-derived.
  CHECK_FALSE(distributed::is_available("mpi"));
  auto group = distributed::init();
  CHECK_EQ(group.size(), 1);
  CHECK_EQ(group.rank(), 0);
}

TEST_CASE("all_sum is identity at one rank (op-layer short-circuit)") {
  // ops.cpp line 29 returns the input before any primitive is
  // constructed. This pins the SINGLETON contract, not AllReduce; the
  // primitive's sum semantics are proven at two ranks in
  // distributed/test_two_rank.cpp.
  std::vector<float> data = {1.0f, -2.0f, 3.5f, 0.25f};
  array x = array(data.data(), {2, 2}, float32);
  eval(x);
  array y = distributed::all_sum(x);
  eval(y);
  CHECK_EQ(y.shape(), x.shape());
  CHECK_EQ(y.dtype(), x.dtype());
  CHECK_EQ(std::vector<float>(y.data<float>(), y.data<float>() + 4), data);
}

TEST_CASE("all_max and all_min are identity at one rank") {
  std::vector<float> data = {4.0f, -1.0f, 7.5f};
  array x = array(data.data(), {3}, float32);
  eval(x);
  array ymax = distributed::all_max(x);
  array ymin = distributed::all_min(x);
  eval(ymax);
  eval(ymin);
  CHECK_EQ(ymax.shape(), x.shape());
  CHECK_EQ(std::vector<float>(ymax.data<float>(), ymax.data<float>() + 3),
           data);
  CHECK_EQ(ymin.shape(), x.shape());
  CHECK_EQ(std::vector<float>(ymin.data<float>(), ymin.data<float>() + 3),
           data);
}

TEST_CASE("all_gather is identity at one rank (shape stays rank-one)") {
  // ops.cpp line 69 returns x unchanged, so the first dim does NOT grow
  // by the group size; the rank-ordered concatenation is proven at two
  // ranks in distributed/test_two_rank.cpp.
  std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
  array x = array(data.data(), {4}, float32);
  eval(x);
  array y = distributed::all_gather(x);
  eval(y);
  CHECK_EQ(y.shape(), x.shape());
  CHECK_EQ(y.dtype(), x.dtype());
  CHECK_EQ(std::vector<float>(y.data<float>(), y.data<float>() + 4), data);
}

TEST_CASE("sum_scatter is identity at one rank (shape stays whole)") {
  // ops.cpp line 149 returns x unchanged; the first dim is NOT divided
  // at one rank.
  std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  array x = array(data.data(), {2, 3}, float32);
  eval(x);
  array y = distributed::sum_scatter(x);
  eval(y);
  CHECK_EQ(y.shape(), x.shape());
  CHECK_EQ(y.dtype(), x.dtype());
  CHECK_EQ(std::vector<float>(y.data<float>(), y.data<float>() + 6), data);
}

TEST_CASE("send to a singleton group throws invalid_argument") {
  // ops.cpp line 94: the guard fires before the Send primitive exists.
  // Real delivery is proven at two ranks in distributed/test_two_rank.cpp.
  array x = array({1.0f, 2.0f}, {2}, float32);
  CHECK_THROWS_AS(distributed::send(x, 0), std::invalid_argument);
}

TEST_CASE("recv from a singleton group throws invalid_argument") {
  CHECK_THROWS_AS(distributed::recv({2}, float32, 0), std::invalid_argument);
}

TEST_CASE(
    "strict init: ring demands env, mpi and nccl still refuse, jaccl quirks") {
  // ring strict init without MLX_RANK/MLX_HOSTFILE throws the env guard
  // at ring.cpp:829-837 ("[ring] You need to provide...") -- a real
  // transport demanding a hostfile, NOT a stub refusal. A two-rank group
  // forms when the launcher sets both variables.
  CHECK_THROWS_AS(distributed::init(true, "ring"), std::runtime_error);
  // nccl remains a stub (no_nccl.cpp:11-17): strict throws.
  CHECK_THROWS_AS(distributed::init(true, "nccl"), std::runtime_error);
  // mpi strict init throws while libmpi.so is absent; the same canary as
  // the is_available("mpi") check above.
  CHECK_THROWS_AS(distributed::init(true, "mpi"), std::runtime_error);
  // jaccl is the one exception and this is upstream behavior on a Mac too:
  // the any-branch of init() reassigns bk_ to "jaccl" unconditionally
  // (distributed.cpp) even when jaccl::init returns nullptr, so
  // register_group caches the EmptyGroup under the key "jaccl" and every
  // later init(true, "jaccl") cache-hits instead of throwing. Whether a
  // strict jaccl init throws therefore depends on what ran earlier in
  // the process (discovery runs one case per process; the aggregate
  // binary runs many), so pin only the post-init() state a caller sees.
  distributed::init();
  auto jaccl = distributed::init(true, "jaccl");
  CHECK_EQ(jaccl.size(), 1);
  CHECK_EQ(jaccl.rank(), 0);
}

TEST_CASE("non-strict init of ungrouped backends yields a one-rank group") {
  // Without MLX_RANK/MLX_HOSTFILE: ring::init returns nullptr
  // (ring.cpp:838), mpi and nccl stubs return nullptr, and every key
  // resolves to the shared EmptyGroup of size 1.
  for (auto bk : {"ring", "mpi", "nccl", "jaccl"}) {
    auto group = distributed::init(false, bk);
    CHECK_EQ(group.size(), 1);
    CHECK_EQ(group.rank(), 0);
  }
}
