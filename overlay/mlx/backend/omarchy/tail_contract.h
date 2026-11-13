#pragma once

#include "mlx/array.h"

namespace mlx::core::omarchy {

// Whole, dense, unoffset allocated buffer of `count` elements. Same
// contract primitives.cpp's whole_dense applies to real inputs; kept
// here so the tail contract is testable without a device.
bool tail_dense_input(const array& value, size_t count);

// Static (host-side, no device) contract the norm tail fold needs from
// the deleted RMSNorm's inputs, checked in dispatch_quantized_gemv_group
// before any buffer exists. The SUM is the dispatch's own Add-epilogue
// output: the group fires before member.epilogue->set_data() allocates
// it, so the sum must be validated on metadata alone (dtype here, size
// at the caller's width check) — an allocation requirement here refused
// every tail group at its first dispatch (the da4ca7acf fix). The
// weight, by contrast, is an evaluated producer input and must be a
// whole dense `width`-element row of the member dtype.
bool norm_tail_static_contract(
    const array& sum,
    const array& weight,
    Dtype member_dtype,
    size_t width);

} // namespace mlx::core::omarchy
