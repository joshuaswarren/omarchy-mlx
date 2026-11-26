# Long-SDPA flash prefill route (hd128)

## Scope

`SdpaPrefillFlashBF16Hd128` streams K/V through 32-key shared tiles with f32 online softmax. A 256-lane workgroup maps 32 query rows, with one query tile per dispatch. It supports bf16 Q/K/V/output, head/value dimension 128, innermost stride 1, and no mask, causal mode, or sinks. By default it replaces composed attention only when the f32 score matrix exceeds 2^30 elements or 25% of device heap; `MLX_OMARCHY_SDPA_PREFILL_FLASH_MIN_L` forces an A/B threshold and `MLX_OMARCHY_SDPA_PREFILL_FLASH=0` disables it.

## M2 build and numerical/live-context gates

The fresh wheel built from `057d1920` (based on main `5f77de8a`) is `/var/tmp/longsdpa-current-057d1920/dist/mlx_omarchy-0.32.4.dev202610050436+057d192-cp314-cp314-linux_aarch64.whl`, 416,577,952 bytes, SHA-256 `fb11a05a3f3883084ecdfe1f3ea9efe88d8cee6a67b4806c30261e63e8887e48`. It was installed in `/var/tmp/longsdpa-m2-venv`; import reported version `0.32.4.dev202610050436+057d192`.

Flash-vs-FP64 sampled-row relative L2 errors, determinism, and forced-flash medians were measured in isolated processes at H=56: L=2048 0.001654, deterministic, 237.0 ms (64 query tiles; 3.70 ms/tile); L=4096 0.001658, deterministic, 955.3 ms (128; 7.46 ms/tile); L=8192 0.001656, deterministic, 3751.2 ms (256; 14.65 ms/tile); L=13365 0.001664, deterministic, 9938.1 ms (418; 23.78 ms/tile). At L=2048 flash/composed max absolute difference was 0.000977. The L=4096 per-tile timing falls in the 5–12 ms register-promotion proxy band; the longer lengths scale with K length and are not a 10x scratch-spill signature. The mixed numerics script separately terminated in Mesa's `addr_range.range <= UINT32_MAX` descriptor assertion; composed large-length cases are not supported by that backend limit.

The live-context harness passed: at L=2048, attention maxdiff 0.000488 and final maxdiff 0.015625; at L=13365, two full-context forwards took 20.43 s and 20.42 s and were bit-identical.

## Performance implication

At `[1,56,13365,128]`, the scalar route measured 9.9381 s for 5,121,485,107,200 FLOPs, about 0.515 TFLOP/s. This is roughly 14–20x slower than the 0.5–0.7 s/call target. For 48 layers × 8 denoise steps, 384 calls project to 3,816 s (63.6 min) of attention alone, versus 192–269 s at target. The projection excludes all other model work. A cooperative-matrix QKᵀ/PV kernel is required to approach the target.

## Native suite evidence and landing scope

`omarchy_fast_ops_tests`, `omarchy_fast_regression_tests`, and `omarchy_runtime_tests` passed on the earlier LongSdpa base. Those suite results are not claimed for the landed commit. The M2 wheel/numerics/live-context gates above were run on `057d1920`, based on main `5f77de8a`. The landed source was rebased onto main and differs from that tested source only by the intervening maskless test namespace fix and receipt updates; the native suites were not rerun on the landing SHA.
