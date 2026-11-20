#include "sdpa_decode_mask_layout.h"

using mlx::core::omarchy::sdpa_decode_mask_layout;

int main() {
  const auto left_padding = sdpa_decode_mask_layout(1, 720, 360, 0, 1, 2, 32, 319);
  if (!left_padding || left_padding->offset != 1 || left_padding->batch != 360 ||
      left_padding->head != 0 || left_padding->key != 1) {
    return 1;
  }
  const auto dense = sdpa_decode_mask_layout(0, 319 * 64, 319 * 32, 319, 1, 2, 32, 319);
  if (!dense || dense->head != 319 || dense->batch != 319 * 32) {
    return 2;
  }
  const auto all_false = sdpa_decode_mask_layout(3, 1, 0, 0, 0, 2, 32, 319);
  if (!all_false || all_false->offset != 3 || all_false->key != 0) {
    return 3;
  }
  if (sdpa_decode_mask_layout(0, 637, 319, 0, 1, 2, 32, 319) ||
      sdpa_decode_mask_layout(0, 720, -360, 0, 1, 2, 32, 319) ||
      sdpa_decode_mask_layout(UINT32_MAX, 720, 360, 0, 1, 2, 32, 319)) {
    return 4;
  }
}
