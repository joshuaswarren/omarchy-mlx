#include "sdpa_decode_mask_layout.h"

// Host-only option checks; they do not exercise GPU dispatch.

using mlx::core::omarchy::sdpa_decode_mask_fused_enabled;

int main() {
  // Unset and empty select the fused route by default.
  if (!sdpa_decode_mask_fused_enabled(nullptr) ||
      !sdpa_decode_mask_fused_enabled("")) {
    return 1;
  }
  // A literal "0" is the escape: it selects the composed route.
  if (sdpa_decode_mask_fused_enabled("0")) {
    return 2;
  }
  // Every other value keeps the fused route.
  if (!sdpa_decode_mask_fused_enabled("1") ||
      !sdpa_decode_mask_fused_enabled("00") ||
      !sdpa_decode_mask_fused_enabled("yes")) {
    return 3;
  }
  return 0;
}
