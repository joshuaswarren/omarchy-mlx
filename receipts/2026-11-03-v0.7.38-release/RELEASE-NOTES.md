# omarchy-mlx v0.7.38

Headline: the decode Qmm column geometry now defaults to the 2x16 workgroup form on both Apple M1 machines. The wider workgroups cut kernel submissions per decoded token by more than half. Measured on the 13-inch M1 (G13G): +1.8 to +4.0 percent decoded tokens per second. Tokens stay bit-identical.

## Decode Qmm geometry, per chip (#350, #351)

The bf16 subgroup Q4 decode kernels now run 2 columns per slot and 16 slots per workgroup on the 13-inch M1 and the 16-inch M1 Max. This covers the single-token word and multi-weight kernels. Other parts keep the previous 2x4 form. The per-lane chain and byte count are unchanged; only the workgroup shape changes. Dispatches per token stay the same. Submissions per token drop: 8 to 3 on the 2B, 14 to 5 on the 4B, 18 to 6 on the 9B.

Measured on the 13-inch M1. Five balanced pairs per cell. Default env. Both arms used the same patched mlx-lm; A was the v0.7.37 release wheel, B was this release's source. Tokens were identical in every run.

| Model | Context | Delta vs v0.7.37 |
| --- | --- | --- |
| Qwen3.8-2B 4-bit | 512 | +3.43% |
| Qwen3.8-2B 4-bit | 2048 | +3.99% |
| 4B | 512 | +1.76% |
| 4B | 2048 | +2.22% |
| 4B | 4096 | +2.19% |
| 9B | 512 | +1.83% |
| 9B | 2048 | +2.42% |
| 9B | 4096 | +2.04% |

The full grid is 40 of 40 pairs with identical tokens: eight model/context cells on the 13-inch M1, five balanced pairs each, +1.76 to +3.99 percent, within-arm spread under 2 percent in every cell. On the 16-inch M1 Max the same interleaved A/B measured +4.12 percent at 2B with 512 tokens of context (#351); isolated per-shape cells are not comparable across geometries there, so the model-level summary is the 16-inch record. `MLX_OMARCHY_QMM_GEOM=<rows>x<slots>` still overrides the default. Any value outside the listed geometries is refused by name.

## Other changes

- qmm_vec: 4-bit and 8-bit weight fields decode in place, without word shifts (#318).
- sdpa_decode_native runs the bf16 decode attention arm on 256 lanes (#320).
- The backend no longer sets `HK_SUBMIT_POLL_US` at device init (#345). The 2000 us poll cost about 5.4 s of process CPU and about 1 W of system power on a 13-inch M1 decode. On a 16-inch M1 Max it cost 1.7 s of CPU and about 1.6 W of package power. Tokens per second were unchanged in both cases. A value the user sets still reaches the driver unchanged.

## Known issues

- Killing a process while it has two GPU submissions in flight still logs a GPU page fault and a firmware timeout (linux-aurora #335). The SIGTERM drain for the product workers is being prepared separately.
- A trial carry of this stack onto upstream 7.2.15 is running. This release stays on the 7.1.12 base.
