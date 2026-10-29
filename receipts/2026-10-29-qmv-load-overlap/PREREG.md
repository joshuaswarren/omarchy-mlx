# PREREG: overlapped weight loads in the Vulkan 4-bit affine qmv kernel

Written 2026-10-29 (about 13:00Z), before any code or measurement of this change. Assignment: the 2026-10-29 plan for an own-repo decode speedup.

## Change

In `overlay/mlx/backend/omarchy/shaders/qmm_vec.comp`, the single-row 4-bit group-64 decode path (`QmmVecQ4WordSubgroupBF16`, and the macro it shares): issue the loads of the next loop iteration (weight word or words, scale, bias, the x words) before the arithmetic of the current one, so the memory latency of consecutive iterations overlaps. The arithmetic does not change: the same operations in the same order, the accumulators stay `precise`. Technique source: TensorFold unit 8a (credit in the receipt; the diff and the compiler lesson are requested from its owner).

Selection: a new pipeline variant chosen at run time by `MLX_OMARCHY_QMV_LOADOVERLAP` (read live, `0` = the current kernel, `1` = the overlapped kernel). The A/B runs both arms in the same build and the same boot. If the bar below is met the default becomes on and `=0` stays as the off switch. If it is not met nothing is merged on this change.

## Bar (all must hold)

Model Qwen3.8-2B 4-bit (`SiddhJagani/Qwen3.8-2B-mlx-4Bit@0867d98b`), greedy, 256 tokens, the installed wheel built from the change, `scripts/bench_decode.py` (decode tok/s over the generated tokens, prefill excluded, EOS suppressed).

1. Decode tok/s at least +8% on the 13-inch M1 (G13G) and on the 16-inch M1 Max (G13C): median of 3 interleaved quiet pairs, same boot, arm order alternating. "Quiet" = nothing else on the machine in that window (guard queue, load below 0.5 before each pair).
2. The generated token ids are identical in every run of both arms.
3. Logits byte-identical: the full logit vector of every one of the 256 decode steps hashes identically (sha256 over the bf16 bytes) between the arms, on both chips.
4. The mlx-omarchy test suites that exercise the 4-bit qmv route pass on a software Vulkan device (`omarchy_primitive_tests`, `omarchy_matmul_family_tests`, `omarchy_fast_ops_tests`, `omarchy_qmv_batch_tests`), and a new test compares the two arms bit for bit on the shapes of the 2B model.

Not part of the bar and not claimed: any other model, the 14-inch M2 Max, prefill, rows above 1.

## Procedure

- Microbenchmark first (seconds per call, `mx.quantized_matmul` at 1 row on the 2B shapes, arms interleaved): used to choose among overlap shapes (prefetch-next, unroll-by-2 with loads first). The choice is made on the microbenchmark of ONE chip and then confirmed on the other; the end-to-end bar above is the only pass/fail.
- End to end: three interleaved pairs per chip, each arm in its own process.
- Identity: a separate pass with logits hashing (not timed).

## Fallback (from the plan)

If the bar is missed: techmore PR 296's route (`MLX_OMARCHY_QMV_SMALL_T`, merged opt-in, cap 4) becomes default-on only if a quiet A/B shows at least +5% at 2 to 4 row passes on both chips, with identical outputs. Its output differs from the shipped route (the GEMV order), so "identical" must be argued on the numerics gate, not assumed; if it cannot be, nothing merges and the receipt says why.

## Known risks

- The kernel already uses `precise` accumulators (qmm_vec.comp, receipts/2026-09-09-q4-gemv-order) and explicit fma chains, so the change is about load placement only; the device compiler may already overlap loads, in which case the gain is zero and nothing merges.
- Decode of the 2B model also spends time outside this kernel (attention, delta-net, lm head). Even a faster kernel moves tok/s by its share only.

## Gates adopted from the TensorFold unit 8a prereg, added before any run

1. Kernel-level BYTE equality, old kernel against new, on random activation rows (bf16 standard normal, never ones), every served (N, K) including the head, at least 250,000 outputs per chip, ON THE DEVICE. `qmv_micro.py` fingerprints 64 fixed-seed random rows per shape (5 shapes, the head included: about 16 million outputs) and prints the sha256; arm A and arm B fingerprints must be equal on each chip.
2. Model level: all 256 decode steps' full logits hash (`logits_hash.py`) and the generated ids equal between arms.
3. Speed: three interleaved pairs per chip, quiet, boot id recorded; median of the pairs; per-pair gains reported (decode tok/s from `bench_decode.py`, 256 tokens).
4. Per-shape per-dispatch table (the microbench chains 16 dependent calls per eval, so submit overhead does not count) in the PR text, with the dispatch counts per token and the sum against the measured saving.
5. Build hygiene: arm A is the published v0.7.34 wheel; arm B is built by the same release script from the branch; the two libmlx digests are printed in every log. A SPIR-V module-by-module identity check against the v0.7.34 build (only the qmm_vec modules that use `Q4_ROWS` may differ) is done before any merge, with the same shader compiler on both sides, or the merge waits.

## Predictions (author), written 2026-10-29 before the first run of this change

- The existing loop already issues the loads of a step (weight words, then x words) at the top of the step and does the scale and bias loads in the epilogue of the same step. At K = 2048 a lane walks only 4 steps, at K = 6144 12 steps, so the overlap can hide the latency of at most 3 to 11 step boundaries. I expect the device compiler to already issue part of these loads early.
- Kernel level (per dispatch, per-shape table): 1.00x to 1.12x on the K = 6144 shape (down projection), 1.00x to 1.06x on the K = 2048 shapes, about 1.00x on N = 512. The head (N = 248320) 1.00x to 1.08x.
- Model level (decode tok/s, 256 tokens): +1% to +5% on the 13-inch M1, +2% to +7% on the 16-inch M1 Max. The M1 Max has more bandwidth headroom (about 160 of 400 GB/s used at 2B decode, against about 44 of 68 GB/s on the M1), so I expect the larger gain there.
- Probability that the +8% bar is met on BOTH chips: 15%. On the M1 Max alone: 30%.
- Output identity: 90% that both gates 1 and 2 hold on the first build (the accumulators and their feeding expressions are already `precise`; the new code only moves loads). If a difference appears it is fixed by declaring more expressions precise, not by relaxing the check.
- Decision rule (fixed now): both chips at least +8% median AND gates 1 and 2 byte-identical: merge after the usual review. Either missed: nothing merges from this change; the fallback is the #296 default-on test above, which I expect to fail on the "identical outputs" requirement (its rows follow the GEMV summation order, not the coopmat chain). Numbers go in the receipt either way. No threshold changes after the first run.
