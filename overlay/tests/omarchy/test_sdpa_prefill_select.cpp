#include "sdpa_prefill_select.h"

using namespace mlx::core::omarchy;

int main() {
  // Env pins the route either way, coopmat device or not, either shape.
  if (!sdpa_bf16_fast_enabled("1", true, true) ||
      !sdpa_bf16_fast_enabled("1", true, false) ||
      !sdpa_bf16_fast_enabled("1", false, true) ||
      !sdpa_bf16_fast_enabled("1", false, false) ||
      !sdpa_bf16_fast_enabled("", false, false) ||
      sdpa_bf16_fast_enabled("0", true, true) ||
      sdpa_bf16_fast_enabled("0", false, false)) {
    return 1;
  }
  // Unset: coopmat-capable devices keep the f32 composed route.
  if (sdpa_bf16_fast_enabled(nullptr, true, true) ||
      sdpa_bf16_fast_enabled(nullptr, true, false)) {
    return 2;
  }
  // Unset on a device without the coopmat composed route defaults to
  // bf16 scores on prefill-shaped calls; the f32 fallback it replaces
  // is strictly slower.
  if (!sdpa_bf16_fast_enabled(nullptr, false, true)) {
    return 3;
  }
  // Unset on the same device keeps bf16_fast off for single-query
  // decode: turning it on there would veto the masked fused decode
  // route, which requires !bf16_fast.
  if (sdpa_bf16_fast_enabled(nullptr, false, false)) {
    return 4;
  }
  return 0;
}
