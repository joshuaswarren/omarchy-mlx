# gather_qmm subgroup decode for 3-bit and 2-bit weights

The scalar `gather_qmm` kernel gives each output to one thread. The thread walks K serially. At the routed-MoE decode shape that costs 14 ms per call. This change moves two bit widths to the subgroup decode kernels.

- 3-bit, group 64, f16, transposed, with biases, m == 1, N divisible by 8: a new compiled variant of `gather_qmm_sub.comp` (`-DBITS3`, kernel `GatherQmmSubDirectR8Q3F16`). It decodes the 3-bit stream. The kernel is appended to the enum, so no earlier kernel id moves.
- 2-bit, group 64 or 128, transposed: the existing generic subgroup kernels. Only the host gate in `dispatch_gather_qmm` changes. A 2-bit code never straddles a word. The decode by `32 / bits` is exact.

## Rule fixed before the run

For every case the GPU error against an fp64 dequantize reference is at most 3 times the CPU stream's error plus 1e-3 of the output scale. Both routes must pass. They are the default route and the scalar route (`MLX_OMARCHY_GATHER_QMM_SUB=0`). At the real expert shape the default route must take at most half the scalar route's time per call.

## What was run

`gather_bits_gpu.py <bits> <group>` ran on the 14-inch M2 Max with Honeykrisp. The boot was `7dcada54`. The kernel was `7.1.12-2-20261013.18-sep-ARCH` with the performance governor. The wheel is `mlx_omarchy-0.32.4.dev202610160214+39f00215-cp314-cp314-linux_aarch64.whl` (sha256 `e14d0a5f8a70255204ceae2ca8b872b029cb09aa211de0e113d50eb5f2f18cba`). The loaded `libmlx.so` has sha256 `80107e18113638e220dbd391cbe95a967f3bcb26a646a7dbe86992c1f1acf0ef`. The log is `g3d-gpu-m2-1.run.log`. The per-config outputs are the `.out` files. Paths in them are shortened to `$HOME`.

| Config | Result |
|---|---|
| 3-bit group 64, f16 | PASS |
| 2-bit group 64, f16 | PASS |
| 2-bit group 128, f16 | PASS |

Each config ran five shapes. They are T 1 with 16 experts (K 2048, N 1024), T 1 with 64 experts (K 4096, N 512), T 32 with 16 experts, T 3 with 4 experts, and the real Muse Spark Open expert shape. All outputs were finite. Stderr was empty. The Vulkan device opened again after the run.

## Time per call at the real expert shape

The shape is K 4096, N 4096, top-8, f16, 24 experts, one row. Each figure is the median of five calls after a warm-up. These are cost figures on random weights. They are not a decode speed.

| Config | Default route | Scalar kernel |
|---|---|---|
| 3-bit group 64 | 2.05 ms | 14.69 ms |
| 2-bit group 64 | 1.76 ms | 14.25 ms |
| 2-bit group 128 | 1.60 ms | 14.21 ms |

The ratio is 0.11 to 0.14, inside the rule of 0.5. One decode token of a 62-layer model makes 186 gather calls. At these figures the calls add up to about 0.3 seconds per token.

## Error at the real shape

| Config | GPU vs fp64 | CPU vs fp64 | Output scale |
|---|---|---|---|
| 3-bit group 64 | 3.893e-03 | 3.893e-03 | 14.5 |
| 2-bit group 64 | 3.903e-03 | 3.903e-03 | 13.6 |
| 2-bit group 128 | 3.894e-03 | 3.894e-03 | 13.4 |

That is the f16 rounding of the output. The default route differs from the scalar route by at most one f16 step (3.906e-03). The lane tree changes the f32 sum order.

## Limits

- The weights are random. No checkpoint was loaded.
- The default route is not bit-identical to the scalar route.
- f16 only. f32 2-bit and 3-bit stay on the scalar kernel.
- The check ran on one chip and one boot.
- The 3-bit subgroup kernel needs group 64. 3-bit group 128 uses the scalar kernel.
