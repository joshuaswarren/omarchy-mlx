// Host-side unit test for the norm tail fold's static contract
// (tail_contract.h). No device, no dispatch: the function under test is
// pure array-metadata validation.
//
// Regression: at e7c2596bf the same checks lived inline in
// dispatch_quantized_gemv_group and ran whole_dense() over the SUM,
// which requires an allocated buffer. The sum is the dispatch's own
// Add-epilogue output — the group fires before that allocation — so
// every tail group refused at its first dispatch ("prologue group
// refused after fold adoption"). This test pins the fixed contract: an
// UNALLOCATED sum must pass; only dtype and the evaluated weight's
// density are refused here.
//
// What this test does NOT cover: the dispatch-time readiness checks
// (input_ready on weight and x), the width block (2048/2560, %256,
// node.size()==width), the member loop, and the shader itself — those
// need a device and stay under the M2 A/B.

#include "doctest/doctest.h"

#include "mlx/array.h"
#include "mlx/allocator.h"
#include "mlx/backend/omarchy/tail_contract.h"

using namespace mlx::core;

namespace {

// One-row bf16 array whose buffer exists: offset 0, dense data_size.
array dense_row(size_t width) {
  array a(Shape{static_cast<int>(width)}, bfloat16, nullptr, {});
  a.set_data(allocator::malloc(a.nbytes()));
  return a;
}

} // namespace

TEST_CASE("norm tail static contract") {
  const size_t width = 2560;
  // An unallocated sum: the shape the dispatch will write, no buffer.
  array sum(Shape{static_cast<int>(width)}, bfloat16, nullptr, {});
  array weight = dense_row(width);

  CHECK(sum.data_shared_ptr() == nullptr);
  CHECK(omarchy::norm_tail_static_contract(sum, weight, bfloat16, width));
  CHECK_FALSE(omarchy::norm_tail_static_contract(sum, weight, float16, width));

  SUBCASE("non-bf16 sum refused") {
    array f16_sum(Shape{static_cast<int>(width)}, float16, nullptr, {});
    CHECK_FALSE(
        omarchy::norm_tail_static_contract(f16_sum, weight, bfloat16, width));
  }
  SUBCASE("weight narrower than the sum width refused") {
    array narrow_weight = dense_row(2048);
    CHECK_FALSE(omarchy::norm_tail_static_contract(
        sum, narrow_weight, bfloat16, width));
  }
  SUBCASE("unallocated weight (not an evaluated producer) refused") {
    array unallocated_weight(Shape{static_cast<int>(width)}, bfloat16, nullptr, {});
    CHECK_FALSE(omarchy::norm_tail_static_contract(
        sum, unallocated_weight, bfloat16, width));
  }
  SUBCASE("oversized weight data_size refused") {
    array big(Shape{static_cast<int>(width)}, bfloat16, nullptr, {});
    // A weight row whose buffer claim spans width+2 elements: not a
    // whole dense width-element row.
    big.set_data(
        allocator::malloc((width + 2) * big.itemsize()),
        width + 2,
        Strides{1},
        big.flags(),
        0);
    CHECK_FALSE(
        omarchy::norm_tail_static_contract(sum, big, bfloat16, width));
  }
}
