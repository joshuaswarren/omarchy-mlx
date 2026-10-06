# Dense f16 GEMM: fp16 cooperative matrices loaded straight from the buffers (Apple M1 G13G)

Date 2026-10-06. Lane matmul-gap. Notebook: MatmulGap H1 (private).

## Finding
The dense f16 matmul ran 0.598 TFLOP/s at 4096^3 on the M1 against 2.30 on macOS Metal on the same die. Every dense dtype was slow (bf16 0.66, f32 0.48), so the cause was not the f16 kernel alone. The AGX disassembly of the staged kernels (`AGX_MESA_DEBUG=shaders`) shows the matrix unit starved by instruction issue, not by bytes: a staged 128x64 coopmat tile issues 32 `simd_matrix_fmadd32` per k step against about 266 `iadd`, 127 `imadd`, 66 `csel`, 50 `if`, 48 `lload`, and 44 `lstore` (staging address math, bounds checks, and one address add per shared load/store). Bigger staged tiles help only to 0.99 TFLOP/s.

The fix removes the staging. The device lists an 8x8x8 subgroup shape with fp16 A/B and an fp32 accumulator. `shaders/matmul_coopmat_direct.comp` loads every operand tile straight from the storage buffers with `coopMatLoad` (no shared memory, no barriers) and compiles the orientation in (four pipelines), so the loop body is 88 instructions per 16 `simd_matrix_fmadd32` at 103 GPRs and full occupancy.

## Numerics
f16 x f16 products are exact in f32, and the accumulator visits k in ascending order, so the stored bits equal `MatmulRbF16` and the 16x16 tile.
- Standalone harness (`tools/gemm-bench`), float64 truth on sampled outputs: 0 bit mismatches against `matmul_rb.comp` over all 16.7M outputs at 4096^3 (nn and nt), and over every output of 24 edge cases across all four orientations (m, n not multiples of the tile, m = 33, k = 8..4096). Exact-rounded fraction 0.99487 for both kernels at 4096^3 nn.
- Through the wheel (`m1/mlxcheck-*.json`): seven cases (nn 4096^3, nt 512x4096x4096, tn, tt, batched nt, broadcast batch with odd m, sliced operands with an element offset) give identical output hashes with the direct route and with `MLX_OMARCHY_NO_COOPMAT=1`; fp64 error is identical. `m1/mlxcheck-trace.log` shows every case dispatching one of the four new pipelines (kernel ids 521-524 at commit 4aaa210e; later enum insertions on main renumber them).
- New test "direct cooperative-matrix f16 matmul matches the 16x16 tile in every orientation" (`omarchy_matmul_family_tests`).

## Performance (M1, T8103 G13G, 8 GPU cores; Linux 7.1.12-2-11.36; omarchy-mlx-vulkan 0.7.22, Mesa e7631595df, libvulkan_asahi.so sha256 26c93b8b...)
| cell | before | after | |
|---|---|---|---|
| MLX `a @ b` fp16 4096^2, 20 s x3 (`m1/run.log`, mm_trace) | 0.598 x3 (`MLX_OMARCHY_NO_COOPMAT=1`) | 1.718 x3 | 2.87x |
| standalone nn 4096^3 (`m1/h8-prod2.log`) | 0.599 | 1.723 | 2.88x |
| standalone nt 4096^3 (`m1/h8-prod2-nt.log`) | 0.604 | 1.476 | 2.44x |
| standalone nt 512x4096x4096 | 0.604 | 1.461 | 2.42x |
| standalone nt 100x2048x2048 (`m1/h6-f16-prod-100nt.log`) | 0.472 | 1.136 | 2.41x |
macOS Metal on the same die: 2.30 TFLOP/s, so the cell moves from 0.26x to 0.75x. Wheel `0.32.4.dev202610062138+4aaa210e` (sha256 a29c6527...). Standalone sides run interleaved round by round in one process; every run held `/dev/cpu_dma_latency` at 0 (the same CPU PD hold the backend applies on G13G).

M1 Max (T6001, 32 GPU cores, same driver build, `m1max/`): nn 2.402 -> 4.574 (1.90x), nt 2.439 -> 2.430 (no change), bits identical. The nt orientation is limited by something else on the larger die; it does not regress.

## Battery (M1, commit 4aaa210e, `m1/run.log`; tip e741c1d8, `m1/run10.log`)
25 standing suites green (runtime 49, primitive 104, matmul_family 25, fast_ops 43, kv 16, indexing 57, reduce 35, shape 25, linalg 30, copy_offset 27, distributed 9, compiled_tape 13, fft 19, fft_general 14, eig 9, take_fill 9, conv 13, complex 34, select_layout 13, fast_regression 2, scatter_determinism 21, eq_math 7, fused_chain 38, error_contract 3, ane_bundle 48). At the tip: matmul_family 26/26 with the new test; capability simulation 7/7 under each of the six profiles.

Rebased on main (MoE subgroup gather) and re-verified at 5998b745 (`m1-tip/`, wheel `0.32.4.dev202610062155+5998b745`): runtime 49, primitive 104, matmul_family 27, fast_ops 43, indexing 57, capability simulation 7/7 under m1-honeykrisp-fork and no-cooperative-matrix; mm_trace 1.718 direct vs 0.598 `MLX_OMARCHY_NO_COOPMAT=1`; mlxcheck hashes identical on all 7 cases; 24 standalone edge cases 0 bit mismatches (`m1-tip/edge.log`).

## Not changed
bf16 and f32 matmuls keep their kernels. Shapes the gate declines (k % 8 != 0, n < 32, odd n, odd m with a column-major lhs, odd offsets or strides, bias C, the causal attention shortcuts) keep `MatmulRbF16`. `MLX_OMARCHY_NO_COOPMAT=1` selects the previous route.

## Measured next lever (not in this change)
bf16 through the same direct kernel on exactly widened f32 operands (`m1/h10-bf16-direct32.log`): 1.112 vs 0.599 TFLOP/s on the M1, 2.898 vs 2.426 on the M1 Max, bits identical. It needs an exact bf16->f32 widening pass in the dispatch.
