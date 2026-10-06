# coopMatLoad address lowering in Honeykrisp: A/B on the M1 and the M1 Max

Date 2026-10-06. Lane matmul-gap (runbook item 5). Notebook: MatmulGap H5, H6 (private).

## Change under test (not landed)
joshuaswarren/mesa-1 branch `agent/w7q-simdmat-addr`, commit 5ecb8b7, on `honeykrisp-omarchy-v3` 01de0431ed2. In `src/asahi/compiler/agx_nir_lower_address.c`, a `ulea_agx`/`ilea_agx` address whose shift is below the access format's shift (the cooperative-matrix loads arrive as `ulea(base, byte_offset, 0)` for 16-bit elements) is now indexed by `offset >> (format_shift - shift)` with the base kept, instead of falling back to a materialized 64-bit address with index 0. The branch also carries an inert, environment-gated LICM experiment (0713304, `AGX_LICM=1`), which changed no code in these tests. Full diff: `candidate.diff`.

Both drivers were built from the same meson options on the M1 Max (`jw16/build.log`: base sha256 7b1a4b84..., candidate 17a923ed...) and selected per process with `VK_ICD_FILENAMES` (shader cache disabled). Arms alternate process by process.

## Code effect (shaderdb, `MatmulDirectF16Nn`)
Inner loop 94 -> 72 instructions (the 22 moves that built 64-bit address pairs are gone, 10 shifts added); 111 -> 95 GPRs; 896 -> 1024 resident threads. Identical on both chips.

## Results
Every output is bit-identical between the arms: the standalone output hashes match per cell on both chips, the 21 MLX cases (f16, bf16, f32 x seven shapes) hash-identical, and the 2B prefill logits and 32 greedy tokens identical.

M1, T8103 G13G (`jwm1/ab.log`, aurora 11.38, CPU PD hold on), `tools/gemm-bench`, TFLOP/s, 3 rounds:
| cell | base | candidate | |
|---|---|---|---|
| f16 4096^3 `a @ b` | 1.696 / 1.696 / 1.696 | 1.952 / 1.953 / 1.951 | +15.1 % |
| f16 4096^3 `a @ b.T` | 1.468 / 1.460 / 1.470 | 1.489 / 1.490 / 1.489 | +1.6 % |

M1 Max, T6001 G13C (`jw16/h6.log`, `jw16/h6b.log`, wheel 0.32.4.dev202610062245+41ca6e47, 3 rounds unless noted):
| cell | base | candidate | |
|---|---|---|---|
| gemm-bench f16 4096^3 `a @ b` | 5.663 / 5.586 / 5.706 | 4.718 / 4.687 / 4.716 | -17 % |
| gemm-bench f16 4096^3 `a @ b.T` | 2.691 / 2.467 / 2.537 | 2.523 / 2.491 / 2.498 | -2 % (within spread) |
| gemm-bench f16 512x4096x4096 `a @ b.T` | 2.788 / 2.706 / 2.539 | 2.709 / 2.727 / 2.649 | 0 % (within spread) |
| MLX 4096^2 `a @ b` f16 / bf16 / f32 (one 10 s run each) | 5.956 / 2.648 / 2.653 | 5.194 / 2.641 / 2.445 | -13 % / 0 % / -8 % |
| Qwen3.8-2B 4-bit pf512 prefill, ms | 372.93 / 372.46 / 372.77 | 371.10 / 370.69 / 369.93 | -0.6 % (faster) |
| Qwen3.8-2B 4-bit decode, tok/s | 45.57 / 45.14 / 45.84 | 45.38 / 45.54 / 45.56 | 0 % |

## Verdict
Not landable. The same lowering that buys the M1 15 % on dense `a @ b` costs the M1 Max 13-17 % on that cell and 8 % on f32; quantized prefill and decode do not move on the M1 Max. A shared-memory pad that should lower occupancy did not change either arm on the M1 Max (`MatmulGap H5`), so the M1 Max loss is not yet explained. The candidate is not proposed for mesa-1 or the packaged ICD.
