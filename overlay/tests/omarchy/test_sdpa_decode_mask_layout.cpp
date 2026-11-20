#include "sdpa_decode_mask_layout.h"

// Host-only layout checks; they do not exercise GPU dispatch.

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
  // Broadcast: zero batch and head strides fan one key row across every row.
  const auto broadcast = sdpa_decode_mask_layout(4, 319, 0, 0, 1, 2, 32, 319);
  if (!broadcast || broadcast->offset != 4 || broadcast->batch != 0 ||
      broadcast->head != 0 || broadcast->key != 1) {
    return 5;
  }
  if (sdpa_decode_mask_layout(4, 318, 0, 0, 1, 2, 32, 319) ||
      sdpa_decode_mask_layout(0, 100, 0, 7, 1, 4, 32, 2)) {
    return 5;
  }
  // Positive strides: the exact fit passes, one element short fails.
  const auto strided = sdpa_decode_mask_layout(7, 13, 3, 5, 2, 2, 2, 3);
  if (!strided || strided->offset != 7 || strided->batch != 3 ||
      strided->head != 5 || strided->key != 2) {
    return 6;
  }
  if (sdpa_decode_mask_layout(7, 12, 3, 5, 2, 2, 2, 3)) {
    return 6;
  }
  // Sliced base offsets: the accessible window is counted from the offset.
  if (!sdpa_decode_mask_layout(1000, 250, 0, 0, 1, 1, 1, 250) ||
      sdpa_decode_mask_layout(1000, 249, 0, 0, 1, 1, 1, 250)) {
    return 7;
  }
  // The uint32 span window clamps what a huge accessible count can promise.
  if (!sdpa_decode_mask_layout(UINT32_MAX - 5, uint64_t{1} << 40, 0, 0, 1, 1, 1, 6) ||
      sdpa_decode_mask_layout(UINT32_MAX - 5, uint64_t{1} << 40, 0, 0, 1, 1, 1, 7)) {
    return 7;
  }
  // Unsigned overflow: offsets and strides past uint32 must refuse...
  if (sdpa_decode_mask_layout(uint64_t{UINT32_MAX} + 1, uint64_t{1} << 40, 360, 0, 1, 2, 32, 319) ||
      sdpa_decode_mask_layout(0, uint64_t{1} << 40, uint64_t{UINT32_MAX} + 1, 0, 1, 2, 1, 1)) {
    return 8;
  }
  // ...while a uint32-max stride with no extent is exact and must pass...
  const auto wide_stride = sdpa_decode_mask_layout(0, 1, UINT32_MAX, 0, 0, 1, 1, 1);
  if (!wide_stride || wide_stride->batch != UINT32_MAX) {
    return 8;
  }
  // ...and the same stride over two spans must refuse.
  if (sdpa_decode_mask_layout(0, uint64_t{1} << 40, UINT32_MAX, 0, 1, 3, 1, 1)) {
    return 8;
  }
  // Reject an empty accessible window or any zero dimension.
  if (sdpa_decode_mask_layout(0, 0, 1, 1, 1, 2, 2, 2) ||
      sdpa_decode_mask_layout(5, 10, 1, 1, 1, 0, 2, 2) ||
      sdpa_decode_mask_layout(5, 10, 1, 1, 1, 2, 0, 2) ||
      sdpa_decode_mask_layout(5, 10, 1, 1, 1, 2, 2, 0)) {
    return 9;
  }
}
