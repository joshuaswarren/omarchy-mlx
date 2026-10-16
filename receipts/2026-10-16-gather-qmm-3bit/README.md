# gather_qmm with 3-bit weights

`mx.gather_qmm` on the Vulkan backend refused every bit width except 4 and 8. The routed experts of the mlx-community Muse Spark Open 3bit checkpoint are 3-bit. This change lets `dispatch_gather_qmm` take 3-bit affine weights in the transposed layout. The code is in `overlay/mlx/backend/omarchy/primitives.cpp` (the bits gate) and `overlay/mlx/backend/omarchy/shaders/gather_qmm.comp` (the decode).

## What the layout is

A 3-bit row of K weights is K times 3 over 32 words. The codes are a little-endian bit stream. Code j starts at bit 3j, so some codes straddle two words. The kernel reads the bit position, takes the bits from the first word, and takes the rest from the next word when the code crosses the boundary. K must be a multiple of 32. Only the scalar kernel handles 3 bits. The subgroup and tile kernels stay 4-bit. Non-transposed 3-bit weights and the fp modes still raise the named `GatherQMM bits` refusal.

## What was run

| Check | Where | Result |
|---|---|---|
| Four 3-bit cases in `omarchy_matmul_family_tests` against an fp64 host reference | llvmpipe on the x86 dev box | 7595 assertions, 0 failed |
| `gather3_gpu_check.py`: eight routed-MoE shapes, 3-bit, group 64, f32 and f16, GPU against the CPU stream and an fp64 reference | 14-inch M2 Max, Honeykrisp | PASS |
| `gather3_real_shape.py`: the real expert shape (K 4096, N 4096), 24 experts, top-8, f16 | 14-inch M2 Max, Honeykrisp | PASS |

The M2 runs used the wheel `mlx_omarchy-0.32.4.dev202610152345+619b5578-cp314-cp314-linux_aarch64.whl` (sha256 `aaced956ee440d0070b0a141342e38e0a956aaa7cc5039883b04183bb51d68ee`). The loaded `libmlx.so` has sha256 `e544f1198fe51668d2ce0915d29246499f6efc782208c65913cd91480cef12b3`. The kernel was `7.1.12-2-20261013.18-sep-ARCH` with the performance governor, boot `7dcada54`. The logs are `g3-gpu-m2-1.run.log` and `g3-real-m2-1.run.log`. Paths in them are shortened to `$HOME`.

## Numbers

`gather3_gpu_check.py` pass rule: the GPU error against the fp64 reference must be at most 3 times the CPU stream's error plus 1e-3 of the output scale.

| dtype | T | top-k | experts | K | N | GPU vs fp64 | CPU vs fp64 | output scale |
|---|---|---|---|---|---|---|---|---|
| f32 | 1 | 8 | 16 | 2048 | 1024 | 1.230e-04 | 3.086e-05 | 89.0 |
| f32 | 1 | 8 | 64 | 4096 | 512 | 2.320e-04 | 4.033e-05 | 118.1 |
| f32 | 32 | 8 | 16 | 2048 | 1024 | 1.553e-04 | 2.524e-05 | 102.4 |
| f32 | 3 | 2 | 4 | 192 | 37 | 6.841e-06 | 3.049e-06 | 18.5 |
| f16 | 1 | 8 | 16 | 2048 | 1024 | 3.057e-02 | 3.057e-02 | 87.2 |
| f16 | 1 | 8 | 64 | 4096 | 512 | 5.648e-02 | 5.648e-02 | 140.3 |
| f16 | 32 | 8 | 16 | 2048 | 1024 | 3.133e-02 | 3.125e-02 | 120.2 |
| f16 | 3 | 2 | 4 | 192 | 37 | 7.697e-03 | 7.697e-03 | 17.2 |

The f16 errors are f16 output rounding. The f32 GPU error is about 4 to 6 times the CPU error and well inside the rule.

The real expert shape, f16, 24 experts, top-8, GPU against the CPU stream: the largest absolute gap was 3.906e-03 at one row and 7.812e-03 at eight rows, at an output scale of 13 to 14. Time per call is a cost figure, not a benchmark: median 14.63 ms at one row and 253.61 ms at eight rows (five calls after one warm-up). Peak memory was 2.41 GiB.

## Limits

- The weights are random. No Muse Spark Open checkpoint was loaded.
- The 3-bit route is the scalar kernel. It is correct and slow. One decode token of a 62-layer model makes 186 gather calls, so at 14.63 ms per call this op alone is about 2.7 seconds per token.
- One chip and one boot.
- The real checkpoint holds 256 experts per layer. The check used 24, so the staged scale and bias copy per call is smaller here than on the real model.

## Reproduce

Install the wheel. Then run both scripts with its Python. Each prints `RESULT PASS` or `RESULT FAIL`.

```sh
python3 gather3_gpu_check.py
python3 gather3_real_shape.py
```

The first script needs the CPU stream for its reference. The CPU stream of stock MLX 0.32.3 on x86 Linux also runs 3-bit `gather_qmm` and matches the fp64 reference.
