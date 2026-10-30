# PREREG: why Qwen3.6-35B-A3B 4-bit prefills at 13 tokens per second on the Omarchy MLX stack (v0.7 line; nothing run yet)

Written 2026-10-30 (committed 22:15Z) before any measurement of this question.

## Question

On a 14-inch M2 Max, `mlx-community/Qwen3.6-35B-A3B-4bit` prefills at 13.4 to 13.9 tokens per second (prompts of 8,192 and 32,768 tokens, quiet machine, three interleaved rounds, spread under 0.1 percent) and decodes at 3 to 4 tokens per second, against 293 and 235 prefill tokens per second for llama.cpp Vulkan on the same chip with the Q4_K_M build of the same model (write-up: `receipts/w9C/2026-10-30-m2-mlx-35b-prefill-gap.md` in the private tree; the measured rows are on the site). The dense Qwen3.5-9B on the same stack prefills at 260 to 340 tokens per second. What is the dominant cost in one prefill chunk of the MoE model, and is there a smallest change that removes it?

## Known before the run (from the source, not measured)

- The model has 40 layers: 30 gated-delta (linear attention) and 10 full attention (`full_attention_interval` 4). Every layer has a sparse MoE block: 256 experts, top-8, `moe_intermediate_size` 512, plus a shared expert of width 512. Expert weights are 4-bit group 64; the router gate and the shared-expert gate are 8-bit group 64 (`config.json` quantization overrides).
- mlx-lm sorts the routed rows by expert id when there are at least 64 of them and calls `gather_qmm` with `sorted_indices=True`. In the backend, `dispatch_gather_qmm` takes the tiled route (`gather_qmm_tile.comp`, one weight tile dequantized per run of up to 32 rows) only when `right_sorted`, `m == 1`, `index_count >= 64`, `index_count >= 4 * experts`, 4-bit group 64, K and N multiples of 64, bf16 or f16 output and enough shared memory (`primitives.cpp`, `use_tile`). At a prefill chunk of 512 tokens with top-8 the index count is 4096 against 4 x 256 = 1024, so the gate is met on paper. Where the gate is not met, the per-row subgroup kernel or the scalar kernel runs, and every row of a run re-reads the expert's weights.
- The dense 9B model has no 8-bit quantized matmul and no expert path, and it prefills 20 times faster. So the gap sits in the MoE block or in the 8-bit router matmuls, not in the gated-delta or attention layers shared with the dense model (the 9B uses the same layer types).

## Measurement

16-inch M1 Max (the M2 is not available), diagnostics wheel `42b30a2d`, one process per run, the model from the local Hugging Face cache, mlx-lm 0.31.3, real English text (the repository docs), a 512-token warmup chunk then 2048 tokens in chunks of 512 (`--prefill-step-size 512` as in the Darkmux runs), one bracketed window per chunk.
- P: GPU profile on (per-dispatch GPU timestamps; dispatches and GPU milliseconds by kernel name). The profiler adds a barrier per dispatch, so only shares are used.
- U: unprofiled prefill tokens per second.
- T0: unprofiled with `MLX_OMARCHY_GATHER_QMM_TILE=0`, to see whether the tile route is taken at all (equal to U means it is not).

## Hypotheses (exclusive on the question "what dominates GPU time in a chunk") and predictions, before the run

- H1, the expert matmul runs a per-row kernel, not the tile route: `GatherQmm*` kernels take at least 60 percent of GPU time and T0 equals U within 3 percent. P 40 percent. Fix: find which gate condition fails for this model and relax it with a measured test.
- H2, the tile route runs but is slow at 256 experts (runs of about 16 rows per expert, 3 in 4 tiles half empty): `GatherQmmTile*` at least 60 percent of GPU time and T0 slower than U. P 15 percent. Fix: tile shape or row-group size.
- H3, the 8-bit router or shared-expert-gate quantized matmuls are slow at M = 512: `Qmm*` kernels with N = 256 or 1 take at least 25 percent. P 15 percent.
- H4, the step is host-bound (GPU busy under 40 percent of the window, as in the unpadded-batch finding): P 20 percent. Fix: whatever the dispatch census shows (a composed fallback or per-expert Python loop).
- Other (none of the above holds): P 10 percent; reported as measured.
- Unprofiled rate U on the 16-inch M1 Max: between 8 and 40 tokens per second (P 80 percent). If it is above 100, the 13 on the M2 is not a property of this stack and the question changes.

## Stop rule

If the profile has no usable timestamps on this chip (GPU busy above 100 percent of the window, or under 2 percent), I fall back to dispatch counts by kernel name plus wall time per chunk and say that the GPU split was not measured. I make no code change before this receipt names a kernel.

## Amendment 1 (2026-10-30, committed 22:27Z; written after the first profile run, which ran 22:17 to 22:22Z, and before the second)

First run (`t7b-moeprof.sh`, 16-inch M1 Max, diagnostics wheel `42b30a2d`): H4 holds and H1 to H3 do not. Per 512-token chunk the GPU runs about 83,600 dispatches for 3.2 s of GPU time inside a 19.5 s profiled window; the sorted-expert tile route is taken (`GatherQmmTileBF16`, 40 dispatches per chunk, 5 ms total GPU time; with `MLX_OMARCHY_GATHER_QMM_TILE=0` the unprofiled rate falls from 44.3 to 36.4 tokens per second). The bulk of the dispatches is 307,200 each of `CastBF16F32` (n 4096), `ElementwiseLiteF32` (n 524288) and `ReduceF32` (n 4096) over the five chunks of the run, which is 512 tokens x 30 linear-attention layers x 4 x 5 chunks: the composed per-token gated-delta loop of mlx-lm, with a host synchronization per layer. The venv of that run carries pristine mlx-lm 0.31.3 (no `.omarchy-serve-patches` record, no `mx.fast.gated_delta_update` route, no ssm-maskless patch), so the run measured the unpatched stack. The unprofiled rate on this chip with that stack is 44.3 tokens per second, not 13.

Second run (`t7b-moeprof2.sh`): the same wheel, with the venv's own mlx-lm (U0) and with the current patch series applied to pristine 0.31.3 (F), two interleaved rounds, then one profiled F run. Predictions, before the run:
- F is at least 3x faster than U0 in prefill tokens per second (P 80 percent) and the two U0 rounds agree within 3 percent, as do the two F rounds (P 85 percent).
- In the profiled F run the dispatch count per 512-token chunk falls below 5,000 (P 80 percent) and GPU busy rises above 50 percent of the window (P 60 percent).
- What is left in the profiled F run is the MoE block and the dense layers: the share of `GatherQmm*` kernels in GPU time is above 20 percent (P 55 percent).
- Unprofiled F is between 150 and 450 tokens per second (P 60 percent). Above 450 would mean the M2's llama.cpp figure (293) is not a ceiling; below 150 would mean a second cost remains behind the gated-delta fallback and the next step is its own profile.

The open question that decides what the fix is: the M2 rows that show 13 tokens per second may have run on an unpatched venv. The M2 is down and its venv cannot be read. The Darkmux server command names a venv `main-venv-fresh` with mlx-lm 0.31.3; whether the serve patch series was applied to it is not known to me.

## Amendment 2 (2026-10-30, committed 22:40Z; after the matched-pair run, before this receipt was written)

Correction to Amendment 1 and to the first status message: the profile summary reports the median per window (one window is one 512-token chunk), not a total over the run. The unpatched chunk therefore has 250,867 dispatches (`ElementwiseLiteF32` 107,640, `CastBF16F32` 77,311, `ReduceF32` 30,720, `CopyGeneralBF16` 15,680, `CastF32BF16` 15,390), not the 83,600 written above, and the patched chunk has 4,986: 50x fewer. The other figures in Amendment 1 (GPU time 3.2 s of a 19.5 s profiled window, tile route taken, 44.3 against 36.4 tokens per second with the tile route off) come from the same summary and stand.

Scoring of the Amendment 1 predictions against the matched-pair run (`data/matched-pair-run.log`):
- F at least 3x faster than U0: held (4.1x: 180.7 and 180.7 against 43.7 and 44.2 tokens per second).
- Rounds agree within 3 percent: held (U0 11.75 s and 11.65 s per chunk, 0.9 percent apart; F 2.876 s and 2.877 s).
- Dispatches per chunk below 5,000: held, barely (4,986).
- GPU busy above 50 percent of the window: failed (107 ms of 3,069 ms in the profiled run, 3.5 percent). The 2026-10-03 prefill receipt calls the GPU timestamps of a single-join prefill unusable on this Mesa line, so this number is not read as a host-bound finding.
- `GatherQmm*` above 20 percent of GPU time: failed (5 of 107 ms, same caveat).
- Unprofiled F between 150 and 450 tokens per second: held (180.7).
