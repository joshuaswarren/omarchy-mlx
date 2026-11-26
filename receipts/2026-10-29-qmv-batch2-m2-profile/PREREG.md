# PREREG: why decode batch 2 takes longer on a 14-inch M2 Max with the hoisted batch qmv kernel (issue 304; nothing run yet)

Written 2026-10-29 (committed 21:54Z) before any measurement of this question.

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

## Amendment 1 (2026-10-29, committed 23:25Z; written after a first profile run, which ran 23:19 to 23:21Z, and before the second, which ran 23:33 to 23:36Z)

What the first run (`t7b-b2prof.sh`, three rounds, kernel stock, GPU timestamps on) showed, and what went wrong with it:

- The summarizer resolved kernel names with a line-by-line regex that skipped one enum entry, so the batch kernel (id 547) showed under the next name and `batch_kernel_ms` read 0 in every row. The raw profiles were deleted after summarizing. The run cannot answer H1, H2 or H3 and is not evidence for any of them. The summarizer now parses the enum as a whole and keeps the raw files.
- Two facts from that run stand, since neither depends on the names: with the profiler on (one barrier per dispatch) a batch-2 step takes 69.3 ms for the hoisted wheel and 69.4 ms for the shipped wheel, so the profiler does not reproduce the 5.6 percent loss; and the batch-2 step has 1225 dispatches against 628 at batch 3, with the GPU busy for about a quarter of the window (17.5 ms of 69 ms).

New observation, from the prompt lengths, not from a measurement: the four prompts are 12, 12, 13 and 13 tokens. Batch 2 uses the first two, which have equal length, so `BatchGenerator` pads nothing and passes no mask. Batches 3 and 4 include a 13-token prompt, so they are left padded and carry a mask. Batch 2 is therefore the only unpadded case in every table of this receipt. That may explain why batch 2 is slow in itself on this chip (28 ms against 17 ms at batch 3, 1522 against 883 dispatches).

H4, the unpadded path is the heavy one: with a padded two-row set (prompts of 12 and 13 tokens) the step has about 600 to 900 dispatches and takes about the batch-3 time, and with an unpadded three-row set (12, 12, 12) the step has about 1200 to 1500 dispatches. P 75 percent for the first half, P 65 percent for the second. If H4 holds, the "batch 2 on the 14-inch M2 Max" regression is a regression of the unpadded two-row path, and the extra dispatches are the mask-free branch of the model or the cache code, which I then locate by kernel names.

Second run (`t7b-b2prof2.sh`): shipped and hoisted diagnostics wheels, four row sets (E2 = prompts 0,1 unpadded; P2 = prompts 0,2 padded; E3 = prompts 0,1,4 unpadded, all 12 tokens; P3 = prompts 0,1,2 padded), three rounds, each cell once with GPU timestamps and once without (64 steps, step wall time only: the unprofiled pass is the one that can show the 5.6 percent loss, since the profiler serializes the dispatches). Predictions, before the run:

- Unprofiled wall, hoisted against shipped: the loss shows in E2 (speed ratio 0.94 to 0.97, P 70 percent); P2 shows no loss (0.98 to 1.03, P 70 percent); E3 shows a loss smaller than or equal to E2's (P 55 percent); P3 is faster hoisted (1.08 to 1.12, P 80 percent).
- Under the profiler the hoisted and shipped steps are equal within 0.5 ms in every set (P 75 percent), as in the first run.
- If the unprofiled loss follows the unpadded sets, the cause is in what the unpadded path does around the batch kernel, and the profiled kernel table (batch kernel time per shape, hoisted against shipped) says whether the batch kernel itself is slower there.

Not changed: no kernel change or routing change comes out of this experiment; a fix goes in its own pull request.

## Amendment 2 (2026-10-30, committed 00:13Z; written after the second run and a CPU probe, before the third)

Results that are in (second run, stock kernel, `t7b-b2prof2.sh`): the unpadded sets E2 and E3 run 1225 dispatches per step and take 29.2 ms (E2) and about 30 to 32 ms (E3); the padded sets P2 and P3 run 628 dispatches and take 14.3 and 15.8 to 17.3 ms. H4 held. In the unpadded sets the 18 gated-delta layers show no `GatedDeltaDecodeBF16PfWide` dispatch (0 against 18 per step) and about 600 more dispatches of elementwise, cast, clear, logical-or, select and reduce kernels, which is the composed fallback. Hoisted against shipped, unprofiled, speed ratio: E2 0.995, P2 1.020, E3 1.048, P3 1.093. Under the profiler the batch kernel itself is not slower in any set (hoisted minus shipped 0.02 to 0.07 ms faster per step).

Cause, from a CPU probe of the arguments `gated_delta_update` receives (`mask_probe.py`, Qwen3.8-2B, mlx-lm 0.31.3): for two prompts of equal length every decode call carries an all-true mask of shape (2, 1); for prompts of 12 and 13 tokens decode calls carry `mask=None`. The backend takes the fused decode kernel at B > 1 only without a mask (`fused_ready` in `GatedDeltaUpdate::eval_gpu`: `B == 1 || (T == 1 && !has_mask)`). `BatchGenerator._process_prompts` calls `finalize()` on the caches only when some prompt was padded; `ArraysCache.finalize` is what drops the left padding, so an unpadded batch keeps a left padding of zeros and `make_mask` builds an all-true mask at every decode step.

Third run (`t7b-nopad.sh`): the same hoisted diagnostics wheel with and without a one-block mlx-lm patch that calls `finalize()` for every batch (`patches/mlx-lm-batch-finalize.patch`), row sets E2, P2, E3, P3, three rounds, profiled pass (dispatch counts) and unprofiled pass (64 steps, step wall time, token-stream hash). Predictions, before the run:

- E2 and E3 patched: 628 dispatches per step and the batch kernel unchanged in count (P 90 percent); unprofiled step wall within 5 percent of the matching padded set (E2 patched about 14 to 15 ms; E3 patched about 16 to 18 ms) (P 80 percent).
- P2 and P3 patched: identical to unpatched, dispatch counts equal and token-stream hash equal (P 95 percent).
- Token-stream hash, E2 and E3, patched against unpatched: equal (P 60 percent). The fused decode kernel and the composed ops are not bit-equal in general; they may agree over 64 greedy tokens or not. A difference is reported, not treated as a failure.

The finding does not change the status of #303: the hoisted kernel is faster in the padded sets and equal or slightly slower only in the unpadded path, where the step is dominated by the composed fallback. The extra dispatches and the 2x step time belong to the mask handling above and have their own fix.

## Amendment 3 (2026-10-30, about 00:50Z; written after the CPU probes of three mlx-lm states and before the next GPU run)

Amendment 2 proposed a one-block mlx-lm patch (`finalize()` on every batch). It is withdrawn. The omarchy-mlx patch series already carries the fix: `scripts/patch-mlx-lm-ssm-maskless.py` makes `ArraysCache.make_mask` return `None` for a batch with no padded row, and its current form tests `all(p <= 0 ...)` on a host copy of the left padding, which stays true after `advance()` lowers the padding below zero. The copy of mlx-lm in the 0.7.28 system package, which every measurement in this receipt used through the diagnostics venvs, carries an older form of the same patch (`not any(left_padding_list)`), true only before the first token. The CPU probes (`mask-probe/`) show what `gated_delta_update` receives in decode for two prompts of equal length: pristine mlx-lm 0.31.3, an all-true (2, 1) mask at every call; the packaged copy, an all-true mask at every decode call and none in prefill; the current patch series, no mask in prefill or decode. For prompts of 12 and 13 tokens every state gives no mask in decode.

So the slow unpadded batch decode follows from the older mlx-lm patch state. The 2 to 5 percent batch-2 loss is not explained by it: the unprofiled profile-driver ratio at E2 is 0.995 and the decode A/B on the same prompts is 0.978 in all three pairs, and neither shows a mechanism. What this receipt has not shown on a GPU is that the current series removes the slowness and what the hoist does at 2 rows under it. The next run (`t7b-nopad2.sh`) crosses the shipped and hoisted diagnostics wheels with the packaged mlx-lm (U) and the current patch series applied to a pristine 0.31.3 tree (F), for the unpadded (E2) and padded (P2) two-row sets, three rounds, profiled and unprofiled. Predictions, before the run:

- Under F, E2 runs the fused gated-delta decode kernel: `GatedDeltaDecodeBF16PfWide` 18 per step and about 628 dispatches (P 90 percent), and the unprofiled step is within 10 percent of the P2 step under F (P 80 percent).
- Under U the run reproduces the second run: E2 1225 dispatches and no `PfWide` (P 95 percent).
- Under F the hoisted wheel is not slower than the shipped wheel at E2: speed ratio 1.00 or more (P 70 percent). The 2.2 to 5.6 percent loss measured under U is then specific to the fallback graph; if it persists under F, it is unexplained.
- P2: F and U give the same dispatch count and the same step time within 3 percent (P 85 percent), because both pass no mask there.

If the first prediction fails, the all-true mask is not the whole cause and the cause stays open. Nothing here changes the code of #303.
