#include "sdpa_prefill_select.h"

using namespace mlx::core::omarchy;

int main() {
  // Env pins the route either way, coopmat device or not.
  if (!sdpa_bf16_fast_enabled("1", true) ||
      !sdpa_bf16_fast_enabled("1", false) ||
      !sdpa_bf16_fast_enabled("", false) ||
      sdpa_bf16_fast_enabled("0", true) ||
      sdpa_bf16_fast_enabled("0", false)) {
    return 1;
  }
  // Unset: coopmat-capable devices keep the f32 composed route.
  if (sdpa_bf16_fast_enabled(nullptr, true)) {
    return 2;
  }
  // Unset on a device without the coopmat composed route defaults to
  // bf16 scores; the f32 fallback it replaces is strictly slower.
  if (!sdpa_bf16_fast_enabled(nullptr, false)) {
    return 3;
  }
  return 0;
}
