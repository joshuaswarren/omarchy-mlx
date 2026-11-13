#include "tail_contract.h"

namespace mlx::core::omarchy {

bool tail_dense_input(const array& value, size_t count) {
    return value.data_shared_ptr() != nullptr && value.offset() == 0 &&
        value.flags().row_contiguous && value.data_size() == count;
}

bool norm_tail_static_contract(
    const array& sum,
    const array& weight,
    Dtype member_dtype,
    size_t width) {
    return sum.dtype() == member_dtype && weight.dtype() == member_dtype &&
        tail_dense_input(weight, width);
}

} // namespace mlx::core::omarchy
