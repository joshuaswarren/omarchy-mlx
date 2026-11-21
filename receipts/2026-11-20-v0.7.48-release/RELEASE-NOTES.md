# omarchy-mlx v0.7.48

Draft. Entries accumulate as changes land on main; the release owner fills
the asset, gate and measured sections at cut time.

## What changed

- **Fused masked batch-2 BF16 decode is the default.** Two-stream
  single-query attention with 128-wide heads under a mask keeps the fused
  shader instead of falling back to the composed route.
  `MLX_OMARCHY_SDPA_DECODE_BOOL_MASK=0` restores the composed route; unset,
  empty and every other value keep the fused route. Unsupported layouts
  and other shapes still use the composed route. The option resolution
  moved into `omarchy::sdpa_decode_mask_fused_enabled`
  (`sdpa_decode_mask_layout.h`), covered by the device-free
  `omarchy_sdpa_decode_mask_flag_tests` CTest. Measured on the 16-inch
  M1 Max over five AB and BA pairs: two streams 53.895 -> 68.934 tokens a
  second, token digests identical, zero GPU timeouts. Today's cap sweep:
  one stream is unchanged at every cap, two streams gain 26 to 34 percent.
