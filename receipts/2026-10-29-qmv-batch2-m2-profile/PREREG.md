# PREREG: why decode batch 2 takes longer on a 14-inch M2 Max with the hoisted batch qmv kernel (issue 304; nothing run yet)

Written 2026-10-29 (about 22:10Z) before any measurement of this question.

## Question

With PR 303 (hoisted nibble conversions in the batch-shared qmv kernel), a decode step at batch 2 on a 14-inch M2 Max takes 5.6 percent longer (29.1 against 27.6 ms, three interleaved pairs, ids identical). Batches 3 and 4 on the same chip are faster (1.102x, 1.152x). Which kernels account for the extra time at batch 2?

## Known before the run

- With `MLX_OMARCHY_QMV_BATCH=0` the shipped and hoisted wheels are equal (28.03 against 28.13 ms), so the loss sits in the hoisted batch kernel or in how it interacts with its neighbours (`batch2-route-diag-14-inch-M2-Max.log`).
- With the route on, the shipped wheel is no faster at batch 2 than with it off (28.02 against 28.03 ms); on the 13-inch M1 the route is worth 3 percent.
- The kernel microbenchmark at 2 rows shows no slowdown for the hoisted kernel on this chip (1.03x, 0.99x, 1.08x).
- A per-quad hoist (8 conversion floats live instead of 32) gave 0.970x at batch 2, the same as the whole-word form. Register pressure is not the cause.
- The dispatch trace shows the same batch kernel 210.38 times per step at batch 2, 3 and 4. The batch 2 step has 1522 dispatches, batch 3 has 883 and batch 4 has 888 (`dispatch-trace-14-inch-M2-Max.log`). The extra batch-2 dispatches come mostly from elementwise, cast, general copy, binary and mask-clear kernels (ids 469, 9, 58, 416, 208, 53: names from the header at the traced commit). Batch 2 takes 28 ms and batch 3 takes 17 ms on this chip; on the 13-inch M1 (69, 89, 116 ms) and the 16-inch M1 Max (29.8, 31.4, 36.9 ms) the step time grows with the batch. So batch 2 on this chip is slow in itself, apart from the hoist.

## Measurement

Two diagnostics wheels that differ only by PR 303: SHIPPED at `a736744a` (the parent of PR 303) and HOISTED at `42b30a2d` (main after PR 303). Both are built with `--diagnostics` (per-dispatch GPU timestamps through `MLX_OMARCHY_GPU_PROFILE`; the profiler inserts a barrier per dispatch, so absolute times inflate and only shares and differences between the two arms are used). 14-inch M2 Max, Qwen3.8-2B 4-bit, mlx_lm 0.31.3 `BatchGenerator`, greedy, four short prompts, 16 warmup steps then 32 measured steps, each run its own process, one boot.

Runs (each its own process, interleaved, three rounds): batch 2 SHIPPED, batch 2 HOISTED, batch 3 HOISTED (reference for the batch-2 kernel mix). Per step I read: wall time (host window), the sum of GPU kernel time by kernel name, the dispatch count by kernel name, and GPU busy as a share of the window.

Quantities:
- dWall = median wall per step, HOISTED minus SHIPPED, batch 2.
- dBatch = the same difference of the summed GPU time of the batch qmv kernel (`QmmVecQ4WordSubgroupBatch4BF16`).
- dOther = the same difference of the summed GPU time of every other kernel.
- dGap = dWall minus the difference of GPU busy time (window time with no kernel running, mostly host and submission).
- For batch 2 against batch 3 (HOISTED): the kernel names and dispatch counts that account for the extra 640 dispatches, and their GPU time.

## Hypotheses and what each predicts

- H1, the hoisted kernel is slower in the real graph: dBatch is at least 70 percent of dWall. Prediction P 45 percent. Consequence: the kernel microbenchmark does not represent the in-graph conditions (cache state, concurrency with neighbours, weight residency). Next step is a per-shape table of the batch kernel's in-graph time to see which of the six decode shapes loses, then a variant for those shapes.
- H2, neighbours get slower: dOther is at least 50 percent of dWall and dBatch is under 30 percent. P 20 percent. Consequence: the hoisted kernel changes what runs beside it (barrier pattern, cache residency); find which kernel family moves.
- H3, the loss is outside kernel time: dGap is at least 60 percent of dWall (the GPU is equally busy and the window is longer). P 35 percent. Consequence: host or submission timing, not a kernel property; look at submission and barrier counts per step in both wheels.
- Mixed (none of the three holds): reported as measured.

Second question, separate from the hoist: the extra batch-2 dispatches. Prediction: P 70 percent that they come from one composed-operation region (the gated-delta update or the mask handling, i.e. elementwise, cast, copy and mask-clear kernels in a repeating per-layer pattern), P 30 percent that they come from sampling or logit processing outside the layers. If it is the first, the 11 ms between batch 2 and batch 3 is a larger prize than the 1.5 ms hoist loss, and a fix goes in its own pull request.

## Stop rule

If the profile has no usable GPU timestamps on this chip (the 2026-10-03 prefill receipt found them unusable in a single-join prefill; the batch decode loop here syncs every step), I fall back to dispatch counts by kernel name plus wall time, state that the GPU split was not measured, and do not claim H1, H2 or H3.

## Not done in this experiment

No kernel change, no routing change by row count, no new wheel behavior. A fix needs the cause first and goes in a separate pull request with its own preregistration amendment.
