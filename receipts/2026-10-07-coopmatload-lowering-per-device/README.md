# coopMatLoad address lowering, per device: on for the M1, off for the M1 Max

Date 2026-10-07. Lane matmul-gap. Notebook: MatmulGap H7 (private). Follows `receipts/2026-10-06-coopmatload-lowering-ab/`, where the ungated lowering gave the M1 +15 % on dense `a @ b` and cost the M1 Max 13-17 %.

## Change (landed)
joshuaswarren/mesa-1 `honeykrisp-omarchy-v3` a4648d3b8bc, "asahi: fold sub-format lea offsets into the access index on single-cluster G13", fast-forward on f24b68097f8. Diff: `candidate.diff`.

- New `agx_device_key::fold_subformat_address`. `agx_gather_device_key` sets it to `gpu_generation == 13 && num_clusters_total == 1 && num_dies == 1`: true on T8103 (M1), false on T6001 (M1 Max) and on every chip not yet measured.
- `agx_optimize_nir` passes the bit to `agx_nir_lower_address`. The H5 fold (`ulea`/`ilea` with a shift below the format shift is indexed by `offset >> (format_shift - shift)`) runs only when the bit is set. With the bit clear, the pass takes the base code path.
- No LICM experiment (the H5 branch carried an inert one).

The A/B ran the same five files on `01de0431ed2` (commit 6e8cde53065, built on the M1 Max with the base build's meson options; base `libvulkan_asahi.so` sha256 7b1a4b84..., candidate f9a1f2ec...). The landed commit was rebased over three later `honeykrisp-omarchy-v3` commits (a9a668a4059, ed28f1d8c2c, f24b68097f8). Those commits change only `agx_nir_lower_math.c` and tests. The five files are byte-identical between 6e8cde53065 and a4648d3b8bc, and the three commits reference none of the changed symbols. The rebased tree itself was not built in this receipt; the GLM cluster run froze builds on the aarch64 hosts.

Drivers selected per process with `VK_ICD_FILENAMES`, shader cache disabled, arms alternated process by process, wheel 0.32.4.dev202610062245+41ca6e47 on both chips (`version_match: true`).

## Code effect (shaderdb, `MatmulDirectF16Nn`)
| chip | base | candidate |
|---|---|---|
| M1 Max (gate off) | 375 instrs, 111 GPRs, 896 threads | 375 instrs, 111 GPRs, 896 threads (identical) |
| M1 (gate on) | 375 instrs, 111 GPRs, 896 threads | 353 instrs, 95 GPRs, 1024 threads |

## Results
All outputs are bit-identical between the arms on both chips: `gemm-bench` `out_fnv` per cell, 21/21 MLX cases (f16, bf16, f32 x seven shapes), and on the M1 Max the Qwen3.8-2B 4-bit prefill logits and 32 greedy tokens.

M1, T8103 G13G (`jwm1/h7.log`, 2026-10-07T00:47Z, 3 rounds, TFLOP/s):
| cell | base | candidate | |
|---|---|---|---|
| gemm-bench f16 4096^3 `a @ b` | 1.695 / 1.669 / 1.671 | 1.892 / 1.857 / 1.863 | +11.5 % |
| gemm-bench f16 4096^3 `a @ b.T` | 1.468 / 1.458 / 1.473 | 1.483 / 1.474 / 1.469 | +0.6 % |
| gemm-bench f16 512x4096x4096 `a @ b.T` | 1.066 / 1.043 / 1.048 | 1.114 / 1.095 / 1.126 | +5.7 % |
| MLX 4096^2 `a @ b` f16 / bf16 / f32 (one 10 s run each) | 1.686 / 0.667 / 1.124 | 1.941 / 0.666 / 1.240 | +15.1 % / 0 % / +10.3 % |

The 2B LM cell did not run on the M1: the venv there has no `mlx_lm`, and the run had to finish before the 01:00Z cluster window.

M1 Max, T6001 G13C (`jw16/h7.log`, 2026-10-06T23:58Z, 3 rounds):
| cell | base | candidate | |
|---|---|---|---|
| gemm-bench f16 4096^3 `a @ b` | 5.695 / 5.632 / 5.648 | 5.621 / 5.621 / 5.680 | -0.3 % |
| gemm-bench f16 4096^3 `a @ b.T` | 2.453 / 2.474 / 2.507 | 2.426 / 2.373 / 2.466 | -2.3 % |
| gemm-bench f16 512x4096x4096 `a @ b.T` | 2.522 / 2.702 / 2.662 | 2.712 / 2.548 / 2.759 | +2.1 % |
| MLX 4096^2 `a @ b` f16 / bf16 / f32 | 5.988 / 2.640 / 2.641 | 5.949 / 2.637 / 2.605 | -0.7 % / 0 % / -1.4 % |
| Qwen3.8-2B 4-bit pf512 prefill, ms | 372.61 / 372.85 / 372.99 | 373.84 / 373.37 / 373.35 | +0.2 % |
| Qwen3.8-2B 4-bit decode, tok/s | 45.52 / 45.43 / 45.18 | 45.05 / 44.90 / 45.67 | -0.4 % |

The M1 Max arms compile the same shader code (shaderdb identical; with the bit clear the pass runs the base path), so the M1 Max differences are run-to-run spread. The H6 base runs of the same cells spanned 2.467-2.691 (`a @ b.T` 4096) and 2.539-2.788 (512). In the first block of `jw16/h7.log` the LM and mlxcheck cells fail because that script passes ICD paths that MLX refuses; the second block (`*.honeykrisp_icd.json`) holds those cells.

## Verdict
Landed. The M1 keeps the H5 win (+11.5 % gemm-bench `a @ b`, +15 % MLX f16, +10 % MLX f32), and the M1 Max keeps base code and base speed. Unmeasured chips keep base code until an A/B turns the gate on for them.
