# Online-kernel inventory: per-part default caps

Date: 2026-11-26. mlx-omarchy commit `eface087f6` (MLX 0.32.4, overlay). Wheels ship online kernels compiled into libmlx.so via CMake.

## Online kernels in the wheel

| Kernel (source) | File path (eface087f6) | Description |
|---|---|---|
| `sdpa_decode_online` | `overlay/mlx/backend/omarchy/shaders/sdpa_decode_online.comp` | Single-dispatch online-softmax bf16 decode SDPA for head width 128 (MLX_OMARCHY_SDPA_ONLINE=16|32). Uses subgroup arithmetic, zero atomics on 32-core Apple GPUs. |
| `fast_rope` | `overlay/mlx/backend/omarchy/shaders/fast_rope.comp` | Fused rotary embedding matching `mlx::core::fast::RoPE` (mlx/fast.cpp, pinned 0.32.2). Supports forward/inverse, half-split and passthrough layouts, float32/f16/bf16 storage. |
| `fast_rope_norm` | `overlay/mlx/backend/omarchy/shaders/fast_rope_norm.comp` | Fused RoPE + RMSNorm. Exposed as `FastRopeNormBF16` and `FastRopeNormAppleBF16` compute kernels. |
| `fast_norm` | `overlay/mlx/backend/omarchy/shaders/fast_norm.comp` | RMSNorm (float32/f16/bf16). Exposed as `FastRmsNormF32/F16/BF16` compute kernels. |
| `fast_norm_gated` | `overlay/mlx/backend/omarchy/shaders/fast_norm_gated.comp` | Gated RMSNorm. Exposed as `FastNormGatedBF16` compute kernel. |
| `fast_norm_apple` | `overlay/mlx/backend/omarchy/shaders/fast_norm_apple.comp` | Apple-optimized RMSNorm. Uses Apple SIMD for the reduction. Exposed as `FastRmsNormAppleBF16` compute kernel. |
| `fast_norm_vjp_dw` | `overlay/mlx/backend/omarchy/shaders/fast_norm_vjp_dw.comp` | VJP dw (weight gradient) for RMSNorm. Exposed as `FastRmsNormVjpDwF32/F16/BF16` kernels. |
| `fast_norm_vjp_dx` | `overlay/mlx/backend/omarchy/shaders/fast_norm_vjp_dx.comp` | VJP dx (input gradient) for RMSNorm. Exposed as `FastRmsNormVjpDxF32/F16/BF16` kernels. |
| `fast_norm_vjp_dw_reduce` | `overlay/mlx/backend/omarchy/shaders/fast_norm_vjp_dw_reduce.comp` | VJP dw reduce for RMSNorm (reduces partial gradients). Exposed as `FastRmsNormVjpDwReduceF32/F16/BF16` kernels. |

Note: VJP kernels are compiled conditionally based on MLX build flags; not all variants are present in every wheel.

## Per-part default caps

Default per-part caps are pure functions in `mlx/backend/omarchy/device.h`. An explicit environment variable (e.g., `MLX_OMARCHY_SDPA_SPLIT_MIN_KEYS`) overrides the default for the given kernel/route.

### SDPA decode split (k_len threshold)

**Function**: `sdpa_split_min_keys(const std::string& device_name)` (device.h:735-748). Returns the minimum cache length `k_len` from which the split-k decode route is the default.

| Part (device name) | Default k_len (keys) | Function lines | Env var override |
|---|---|---|---|
| G13G (Apple M1, 13") | 128 | 739 | `MLX_OMARCHY_SDPA_SPLIT_MIN_KEYS` |
| G13C (Apple M1 Max, 16") | 320 | 740 | `MLX_OMARCHY_SDPA_SPLIT_MIN_KEYS` |
| G14C (Apple M2 Max, 14") | 320 | 741 | `MLX_OMARCHY_SDPA_SPLIT_MIN_KEYS` |
| G17C (Apple M5 Max, pre-calibration) | 320 | 742 | `MLX_OMARCHY_SDPA_SPLIT_MIN_KEYS` |
| G13S (Apple M1 Pro, pre-calibration) | 192 | 743 | `MLX_OMARCHY_SDPA_SPLIT_MIN_KEYS` |
| G14G (Apple M2 Pro, pre-calibration) | 192 | 744 | `MLX_OMARCHY_SDPA_SPLIT_MIN_KEYS` |
| G16S (Apple M4 Pro, cooperative-matrix driver) | 320 | 745 | `MLX_OMARCHY_SDPA_SPLIT_MIN_KEYS` |
| Unmeasured | 0 (no split route) | 748 | `MLX_OMARCHY_SDPA_SPLIT_MIN_KEYS` |

### RoPE half-split

**Function**: `rope_half_split_default_part(const std::string& device_name)` (device.h:876-878). Returns true when the half-split RoPE variant is the default for the given part.

| Part (device name) | Default half-split | Function lines | Env var override |
|---|---|---|---|
| G16G (Apple M1 Pro, Borealis) | true | 876-878 | `MLX_OMARCHY_NO_ROPE_HALF_SPLIT` |

When true, the half-split variant is selected; when false (default), the traditional (full-stride) variant is used. The env var `MLX_OMARCHY_NO_ROPE_HALF_SPLIT` can force the general (traditional) builds on every part (`0` forces half-split; any other value, including unset, leaves the per-part default).

### GatedDelta conv + norm fused

**Function**: `gdn_conv_delta_norm_default_part(const std::string& device_name)` (device.h:618-620). Returns true when the conv+delta+gated-norm fused route is the default.

| Part (device name) | Default fused | Function lines | Env var override |
|---|---|---|---|
| G13G (Apple M1, 13") | true | 618-620 | `MLX_OMARCHY_GDN_CONV_DELTA` |
| G14C (Apple M2 Max, 14") | true | 618-620 | `MLX_OMARCHY_GDN_CONV_DELTA` |
| G17C (Apple M5 Max, pre-calibration) | true | 618-620 | `MLX_OMARCHY_GDN_CONV_DELTA` |
| G13C (Apple M1 Max, 16") | false (stock route) | 618-620 | `MLX_OMARCHY_GDN_CONV_DELTA` |

When true, the fused route has mode 2 (conv + delta + gated norm); when false, mode 0 (stock route, op not called). An explicit `MLX_OMARCHY_GDN_CONV_DELTA` env var ("0", "1", "2") overrides the per-chip default.

### Vocab-tail chunked

**Function**: `vocab_tail_chunked_default(const std::string& device_name)` (device.h:642-645). Returns true when the chunked vocab-tail argmax/LSE route is the default.

| Part (device name) | Default chunked | Function lines | Env var override |
|---|---|---|---|
| G14C (Apple M2 Max, 14") | true | 642-645 | `MLX_OMARCHY_VOCAB_TAIL_CHUNKED` |
| G17C (Apple M5 Max, pre-calibration) | true | 642-645 | `MLX_OMARCHY_VOCAB_TAIL_CHUNKED` |
| G13C (Apple M1 Max, 16") | false (single-workgroup) | 642-645 | `MLX_OMARCHY_VOCAB_TAIL_CHUNKED` |
| G13G (Apple M1, 13") | false (single-workgroup) | 642-645 | `MLX_OMARCHY_VOCAB_TAIL_CHUNKED` |

When true, the chunked route uses one workgroup per 4096-column chunk plus a merge dispatch; when false, the single-workgroup kernels are used. An explicit `MLX_OMARCHY_VOCAB_TAIL_CHUNKED` env var overrides the per-chip default.

### GDN wide pref

**Function**: `gdn_pf_wide_default(const std::string& device_name, int hk, int hv)` (device.h:602-606). Returns true when the wide GDN prefill kernel is the default for the given head-configuration.

This function consults `gdn_bf16a_untiled_default(device_name, hk, hv)` to determine the untiled path; on parts where it returns false, the wide path is also false. No per-part hard-coded default; the wide path is gated on the untiled path's result.

### Encoder record order

**Function**: `encoder_record_in_earliest_wave_default(const std::string& device_name)` (device.h:655-659). Returns true when encoder records each batch in earliest-wave order by default.

| Part (device name) | Default earliest-wave | Function lines |
|---|---|---|
| G16S (Apple M4 Pro) | true | 655-659 |
| G14C (Apple M2 Max, pre-calibration) | false (stock order) | 655-659 |
| G17C (Apple M5 Max, pre-calibration) | false (stock order) | 655-659 |
| G13G (Apple M1, 13") | false (stock order) | 655-659 |
| G13C (Apple M1 Max, 16") | false (stock order) | 655-659 |

### SDPA causal flash coopmat

**Function**: `sdpa_causal_flash_coopmat_default(const std::string& device_name)` (device.h:679-684). Returns true when the fused causal flash coopmat SDPA route is the default.

| Part (device name) | Default flash coopmat | Function lines |
|---|---|---|
| G16S (Apple M4 Pro, cooperative-matrix driver) | true | 679-684 |
| G14C (Apple M2 Max, pre-calibration) | false (stock route) | 679-684 |
| G17C (Apple M5 Max, pre-calibration) | false (stock route) | 679-684 |
| G13C (Apple M1 Max, 16") | false (stock route) | 679-684 |
| G13G (Apple M1, 13") | false (stock route) | 679-684 |

When true, the fused causal flash coopmat route is the default for eligible runs; when false, the stock route is used. An explicit `MLX_OMARCHY_SDPA_CAUSAL_FLASH` env var forces the route (`0` disables it, `1` forces it).

## Per-part default summary

This table summarizes the online-kernel defaults that have hard-coded per-part rules. The default values are the function results before any env var override.

| Part | k_len threshold (split decode) | Half-split RoPE | GDN fused | Vocab-tail chunked | Earliest-wave encoder order | Causal flash coopmat |
|---|---|---|---|---|---|---|
| G13G | 128 | false | true | false | false | false |
| G13C | 320 | false | false | false | false | false |
| G13S | 192 | false | false | false | false | false |
| G14C | 320 | false | true | true | false | false |
| G14G | 192 | false | false | false | false | false |
| G16G | 0 (no rule) | true | false | false | false | false |
| G16S | 320 | false | false | false | true | true |
| G17C | 320 | false | true | true | false | false |
| Unmeasured | 0 | false | false | false | false | false |

## References

- Commit: eface087f6 (MLX 0.32.4 + overlay). Verified with `git cat-file -t eface087f6` and `git show eface087f6 --summary`.
- Part-rule code: `mlx/backend/omarchy/device.h` (910 lines). Functions are pure and keyed on `device_name` (part ID from Vulkan device advert).
- Environment variables: `MLX_OMARCHY_SDPA_SPLIT_MIN_KEYS`, `MLX_OMARCHY_NO_ROPE_HALF_SPLIT`, `MLX_OMARCHY_GDN_CONV_DELTA`, `MLX_OMARCHY_VOCAB_TAIL_CHUNKED`, `MLX_OMARCHY_SDPA_CAUSAL_FLASH`. Setting any env var overrides the per-part default.
- ABBA receipts for default verification: `receipts/2026-10-23-sdpa-split-decode/README.md` (split decode), `receipts/2026-10-22-gdn-wide-g13c/README.md` (GDN wide), `receipts/2026-11-14-g14c-gdn-conv-delta/README.md` (GDN fused), `receipts/2026-11-15-m4-pro-g16s-sdpa-causal-flash/README.md` (causal flash).
- Kernel files: all live in `overlay/mlx/backend/omarchy/shaders/`. Compiled into libmlx.so via CMake (`omarchy_shader` macro in CMakeLists.txt).
