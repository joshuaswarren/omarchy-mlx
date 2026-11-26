# CORRECTION — 2026-09-29T18:08Z (after orchestrator audit by Main)

**My previous dispatch counts were wrong by ~5 orders of magnitude.**

I treated the `count=M` field in `[rtmod] DISPATCH kernel=K count=M gx=A gy=B gz=C` lines as a per-call work-item count, then summed across lines. Main correctly flagged this as impossible (87 ms / 33 M = 2.6 ns per dispatch — clearly wrong). `count=M` is a **running work-group count from process start**, not per-call. The correct per-call count is the number of `[rtmod] DISPATCH` lines (each line = one Vulkan dispatch).

## Corrected dispatch counts (single probe, fresh KVCache per iter, 10 iters)

| Probe | DISPATCH lines / iter | Notes |
|---|---|---|
| 1 — talker only (1 forward + 1 mx.eval) | **1,529** | 28 layers × ~55 dispatches/layer for seq_len=1 |
| 2 — code_predictor (15 passes + 1 mx.eval at end) | 3,605 | ~240 dispatches/pass when serialized |
| 3 — full inner step (talker + 15 cp + sample + embed) | **1,600** | talker dominates; CP add only ~70 dispatches when graph-merged |

Per-call wall time (10 iters, fresh cache):
- Baseline talker (probe 1): **87 ms / step** (28 layers × 55 dispatches × ~57 μs/dispatch)

## Retraction

- **"compile reduces dispatches 46.5%"** claim in receipt README is INVALID — based on the wrong running-total sum. Will re-measure with line-count dispatch once the compile test runs again with the corrected methodology.
- The "33M dispatches / step" number cited everywhere (notebook entry, README, this addendum) was wrong; the **1,529 dispatches / step** figure above is correct.

## Next (in order)

Per Main's audit:
1. ✅ Line-count dispatch table — this correction.
2. dtype/quantization of talker and predictor weights (bf16 vs 4/8-bit). Check for 4-bit/8-bit Qwen3-TTS CustomVoice on HF.
3. KV-cache update pattern (concatenate per step vs preallocated slice-update).
4. Per-op profile of one talker step with the backend profiler.

The dispatch floor is ~1,500 per step. At ~57 μs/dispatch that gives the 87 ms wall time. If we can collapse 1,500 dispatches to a much smaller number via compile, wall time will drop proportionally.
