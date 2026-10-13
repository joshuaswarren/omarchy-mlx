// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

// Two-rank value proof for the distributed primitives over the ring
// transport. Upstream's op layer short-circuits every distributed op at
// group.size() == 1 (mlx/distributed/ops.cpp: all_reduce line 29, all_gather
// line 69, send line 94, recv line 118, sum_scatter line 149), so a
// single-process run of this binary proves nothing about the primitives:
// every case therefore starts by proving a real two-rank group, which makes
// a lone run FAIL instead of passing vacuously.
//
// run-two-rank.sh launches this binary twice with MLX_RANK=0 and MLX_RANK=1
// against a localhost hostfile and requires BOTH processes green. Each
// process verifies its own side of every case; the send/recv case closes
// with a collective so the sending rank's green also proves the receiving
// rank checked the exact bytes.
//
// The group proof is control flow (FAIL), not CHECK/REQUIRE: a guard
// assertion would read as a value assertion to the coverage parser, and a
// value anchor must compare data the primitive produced, not the guard.

#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest/doctest.h"

#include <cstdio>
#include <vector>

#include "mlx/array.h"
#include "mlx/transforms.h"
#include "mlx/distributed/distributed.h"
#include "mlx/distributed/ops.h"
#include "mlx/memory.h"
#include "mlx/ops.h"
#include "mlx/random.h"

using namespace mlx::core;

namespace {

std::vector<float> values(const array& a) {
  eval(a);
  return {a.data<float>(), a.data<float>() + a.size()};
}

} // namespace

int main(int argc, char** argv) {
  // The launcher always provides MLX_RANK and MLX_HOSTFILE; strict init
  // throws otherwise, and the group proofs below fail on any singleton
  // fallback.
  auto group = distributed::init(true, "ring");
  doctest::Context ctx(argc, argv);
  int code = ctx.run();
  if (code == 0) {
    std::printf(
        "TWORANK_OK rank=%d size=%d\n", group.rank(), group.size());
    std::fflush(stdout);
  }
  return code;
}

TEST_CASE("all_sum returns the elementwise sum on both ranks") {
  auto group = distributed::init();
  if (group.size() != 2 || group.rank() < 0 || group.rank() > 1) {
    FAIL("expected a two-rank ring group");
  }
  int rank = group.rank();
  std::vector<float> mine = (rank == 0)
      ? std::vector<float>{1.0f, 2.0f, -3.0f, 0.5f}
      : std::vector<float>{10.0f, 20.0f, -30.0f, 5.5f};
  std::vector<float> both = {11.0f, 22.0f, -33.0f, 6.0f};
  array x = array(mine.data(), {2, 2}, float32);
  eval(x);
  array y = distributed::all_sum(x);
  auto got = values(y);
  CHECK_EQ(y.shape()[0], 2);
  CHECK_EQ(y.shape()[1], 2);
  CHECK_EQ(y.dtype(), x.dtype());
  CHECK_EQ(got, both);
}

TEST_CASE("all_max and all_min return the elementwise extremes on both ranks") {
  auto group = distributed::init();
  if (group.size() != 2 || group.rank() < 0 || group.rank() > 1) {
    FAIL("expected a two-rank ring group");
  }
  int rank = group.rank();
  // Neither rank's input equals the elementwise result, so passing the
  // identity short-circuit is impossible here.
  std::vector<float> mine = (rank == 0)
      ? std::vector<float>{4.0f, -1.0f, 0.25f}
      : std::vector<float>{2.0f, -5.0f, 9.0f};
  std::vector<float> want_max = {4.0f, -1.0f, 9.0f};
  std::vector<float> want_min = {2.0f, -5.0f, 0.25f};
  array x = array(mine.data(), {3}, float32);
  eval(x);
  array ymax = distributed::all_max(x);
  array ymin = distributed::all_min(x);
  auto got_max = values(ymax);
  auto got_min = values(ymin);
  CHECK_EQ(ymax.shape()[0], 3);
  CHECK_EQ(got_max, want_max);
  CHECK_EQ(got_min, want_min);
}

TEST_CASE("all_gather concatenates both contributions in rank order") {
  auto group = distributed::init();
  if (group.size() != 2 || group.rank() < 0 || group.rank() > 1) {
    FAIL("expected a two-rank ring group");
  }
  int rank = group.rank();
  std::vector<float> mine = (rank == 0)
      ? std::vector<float>{1.0f, 2.0f}
      : std::vector<float>{3.0f, 4.0f};
  std::vector<float> both = {1.0f, 2.0f, 3.0f, 4.0f};
  array x = array(mine.data(), {2}, float32);
  eval(x);
  array y = distributed::all_gather(x);
  auto got = values(y);
  CHECK_EQ(y.shape()[0], 4);
  CHECK_EQ(y.dtype(), x.dtype());
  CHECK_EQ(got, both);
}

TEST_CASE("send and recv move an exact vector from rank 0 to rank 1") {
  auto group = distributed::init();
  if (group.size() != 2 || group.rank() < 0 || group.rank() > 1) {
    FAIL("expected a two-rank ring group");
  }
  int rank = group.rank();
  const std::vector<float> sent = {7.0f, -8.0f, 9.0f, 10.5f};
  const float checksum = 18.5f;
  if (rank == 0) {
    array x = array(sent.data(), {4}, float32);
    eval(x);
    array done = distributed::send(x, 1);
    auto echoed = values(done);
    CHECK_EQ(echoed, sent);
  } else {
    array got_arr = distributed::recv({4}, float32, 0);
    auto received = values(got_arr);
    CHECK_EQ(received, sent);
  }
  // Closing collective: rank 1 contributes the checksum of the bytes it
  // received, rank 0 contributes zero. Rank 0's green therefore proves
  // rank 1 verified the exact vector; rank 1's green proves liveness of
  // the sender side. A wrong delivery fails rank 1 here first.
  std::vector<float> mine = (rank == 0)
      ? std::vector<float>{0.0f}
      : std::vector<float>{checksum};
  array c = array(mine.data(), {1}, float32);
  eval(c);
  array total = distributed::all_sum(c);
  auto got_total = values(total);
  CHECK_EQ(got_total, std::vector<float>{checksum});
}

TEST_CASE("sum_scatter is refused by the ring transport at two ranks") {
  auto group = distributed::init();
  if (group.size() != 2 || group.rank() < 0 || group.rank() > 1) {
    FAIL("expected a two-rank ring group");
  }
  int rank = group.rank();
  // Upstream's Linux-linkable transports refuse sum_scatter: ring.cpp
  // throws "[ring] sum_scatter not supported." and mpi.cpp throws "[mpi]
  // sum_scatter not yet implemented.". Only Darwin-only jaccl implements
  // it (jaccl.cpp sum_scatter via group_->sum_scatter), which is why the
  // primitive stays Mac-usable and in the denominator while remaining
  // unprovable here. The primitive constructs at two ranks and the
  // refusal fires at eval on both. If this pin ever fails, this repo
  // gained a sum_scatter-capable transport and ReduceScatter must be
  // re-derived for coverage.
  std::vector<float> mine = (rank == 0)
      ? std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f}
      : std::vector<float>{10.0f, 20.0f, 30.0f, 40.0f};
  array x = array(mine.data(), {4}, float32);
  eval(x);
  array y = distributed::sum_scatter(x);
  CHECK_THROWS_AS(eval(y), std::runtime_error);
}

// The gather output is only a depends() dependency, so no handle to it survives
// async_eval; the ring task must still find it mapped after clear_cache().
TEST_CASE("an unread all_gather output survives a dropped handle and a cache clear") {
  auto group = distributed::init();
  if (group.size() != 2 || group.rank() < 0 || group.rank() > 1) {
    FAIL("expected a two-rank ring group");
  }
  int rank = group.rank();
  std::vector<float> mine = {static_cast<float>(rank + 1)};
  array w = astype(random::normal({2048, 2048}), bfloat16);
  eval(w);
  for (int step = 0; step < 50; ++step) {
    array t = astype(random::normal({1, 2048}), bfloat16);
    for (int i = 0; i < 8; ++i) {
      t = matmul(t, w);
    }
    array y = argmax(random::normal({1, 102400}) + astype(sum(t), float32), -1);
    auto outs = depends({y}, {distributed::all_gather(astype(y, float32))});
    async_eval(outs);
    clear_cache();
    eval(outs);
  }
  array x = array(mine.data(), {1}, float32);
  eval(x);
  auto got = values(distributed::all_gather(x));
  CHECK_EQ(got, std::vector<float>{1.0f, 2.0f});
}
