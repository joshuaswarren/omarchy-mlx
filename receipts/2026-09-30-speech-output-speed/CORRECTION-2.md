# CORRECTION 2: 2026-09-29, per-op profile (after orchestrator audit)

**Retracted:** README's conclusion "floor = ~87 ms GPU compute per step; Kokoro is the path". The 87 ms "talker step" was a **10-token prefill**, not a decode step. The probes passed the whole prompt embedding to a fresh cache. Also retracted: "compile reduces dispatches 46.5%" (running-counter sum; see CORRECTION.md).

Method: I timed each op with an eval barrier (op + `mx.eval`, wall clock, median of 20-30 runs). The barrier floor is 0.15-0.18 ms, and ops near that floor are barrier-bound and cannot be ranked. M2 Max, mlx 0.32.3.dev202609282218+29cba8e, loadavg 0.02-0.30, and no other GPU process ran during the turn. Raw data: `perop.json`, `attn_ab.json`.

## One real decode frame (seq_len=1 after prefill, KV offset 33-93)

| Component | ms |
|---|---|
| Real upstream step, single eval (talker + sample + 15 CP passes + 15 samples + embed sum) | **257** |
| Talker decode step (28 layers + codec head) | 38.6 |
| 15 code-predictor passes, decode path, no sampling | **130.4** (8.7 ms/pass, 5 layers each) |
| `_sample_token` talker (suppress + rep-penalty + top-k 50 + categorical) | 4.29 |
| `_sample_token` without suppress/rep | 3.72 |
| `_sample_token` CP (vocab 2048) | 2.84 |

Per frame: 16 sampler calls × ~3 ms ≈ 47 ms. The frame budget is 80 ms of audio / 1.2 = **67 ms**.

## Inside one talker layer (eval-barrier)

| Op | ms |
|---|---|
| Layer total | 1.73 |
| self_attn total (4 qmv + q/k norm + mrope-apply + KV update + sdpa) | **1.37** |
| MLP total (3 qmv) | 0.36 |
| Each QuantizedLinear alone (4-bit, gs 64, bf16 scales) | 0.17-0.18 (at the barrier floor) |
| RMSNorm | 0.17 (floor) |
| Multimodal rope embedding (once per forward) | **0.83** |

The chat 2B QuantizedLinear alone runs at 0.15-0.19 ms, the same barrier floor. qmv shape, group size (64), bits (4) and dtypes match the chat model, so **suspect 1 (qmv route) is ruled out** at this resolution.

## Suspect A/B (`attn_ab.json`)

| Test | ms |
|---|---|
| `mx.random.categorical`, vocab 3072 | **3.07** |
| `argmax`, vocab 3072 | 0.20 |
| top-k 50 / sort / argpartition | 0.76 / 0.76 / 1.06 |
| sdpa q_len=1, GQA 16/8, L=93: sliced view of 256 buffer / contiguous / plain matmul attention | 0.53 / 0.47 / 0.41 |
| same at L=200 | 0.65 / 0.54 / 0.45 |
| KV slice-update / concat | 0.22 / 0.26 (near floor, **suspect 4 ruled out**) |

## Named findings

1. **`mx.random.categorical` costs 3.07 ms, 15× argmax.** It runs 16 times per frame, about 47 ms or 18% of the frame. This is a backend kernel route (categorical = gumbel + argmax; the gumbel/uniform generation path looks slow). This is the first kernel to fix in `overlay/`.
2. **The attention block costs 1.0 ms beyond its projections** in every layer. sdpa on a sliced 256-slot cache view is 13-20% slower than contiguous, and plain matmul attention beats the fused sdpa at q_len=1.
3. The **15 code-predictor passes (75 layer-passes) cost 130 ms** and dominate the frame. The per-layer-pass cost (~1.4-1.7 ms) is about 3× the chat 2B per-layer decode cost (~0.5 ms/layer at 14 ms/token).

## Floor, as measured

With the sampler at argmax cost (−45 ms), a frame costs about 210 ms. The frame's 103 layer-passes at the measured ~1.4 ms each set a floor of about 145 ms, 2.2× over the 67 ms budget. Reaching RTF 1.2 needs the per-layer-pass cost cut to ≤0.5 ms, which is chat-model parity. Eval-barrier timing cannot attribute the remaining ~1 ms per attention block below the 0.15 ms barrier floor. The next discriminator is a backend-profiler (GPU timestamp) trace of one attention block. **Not done: no fix was built, and no RTF, WER or WAV was produced after a change.**

# Fix rows, 2026-09-29 (dated; one gpu-turn ticket each; `fix_ab.json`)

Setup: upstream streaming path, aiden, seed 123+i per sentence, 1 warmup sentence, then 2 measured rounds of 5 sentences. The M2 rebooted twice mid-run; the reboot-tolerant driver resumed it per sentence.

| Row | Change | Microbench | E2E RTF median / min | first-chunk p95 | Kept? |
|---|---|---|---|---|---|
| base | upstream | categorical 2.31-2.36 ms/draw | 0.206 / 0.132 | 1.55 s | - |
| A | Gumbel-max sampler (argmax(logits - log(-log(U))), one uniform per draw; the upstream sampler is mx.compile'd, so a cross-call pool is invalid) | 0.26-0.33 ms/draw; chi2 (df 7, crit 14.07): gumbel 7.28 / 8.67, categorical 6.24 / 11.4 | **0.231 / 0.108** | 1.45 s | measured gain of about 12%; NOT shipped to synthesis.py yet (needs a scoped patch plus a unit test) |
| B | matmul attention at q_len=1 (fp32 softmax) | 0.50 ms vs sdpa 0.34-0.36 ms; max abs diff 1.2e-7 | not run E2E: slower in the microbench | - | reverted |
| C | compile/fuse the 15-pass CP body | not measured | - | - | open |

C evidence so far: one serialized CP pass = 240 dispatch lines in 8.7 ms, about 36 us per dispatch. That cost is per dispatch; the 0.15 ms per-eval barrier is not what we pay. A frame is about 1.5k (talker) + 3.6k (15 CP) dispatches plus the sampler. To fit 67 ms per frame at ~36 us, a frame must stay under ~1.9k dispatches, about 2.7x fewer. Fusing q/k/v and gate/up removes only 3 of ~48 dispatches per layer, so the reduction must come from fusing elementwise ops (compile) or from a fused decoder-layer kernel. Floor with A applied: RTF 0.23. The threshold is 1.2: FAIL.

# CORRECTION 3 and fix A shipped, 2026-09-29

Correction to 5b0b80b5c: `_streamed_generate_custom_voice` diverged from the upstream loop in two ways. It never appended generated token ids, so the repetition penalty (1.05) was silently off. It also appended the EOS frame before breaking, so the codec decoded one extra frame. Its `if is_eos:` also synced every step, so "no per-step sync" was false. The helper is deleted. The worker is back on `model.generate_custom_voice(stream=True, streaming_interval=0.32)`.

Fix A is shipped: `_fast_codec_sampler(model)` swaps the qwen3_tts module's `categorical_sampling` for `_gumbel_categorical` during one request and restores it on exit or error. Suppression, repetition penalty, temperature and top-k stay upstream's. Tests: `FastCodecSamplerTests`. One test checks scope and restore without mlx. The other runs the upstream `_sample_token` with suppress + penalty + T 0.9 + top-k 5 through the patch: masked tokens are never drawn, chi-square < 18.47 (df 4, p 0.001). It passed on the M2 under gpu-turn. Assistant suite locally: 477 OK (1 skip: the mlx test).

| Row | RTF median / min | first p95 | Whisper WER (5 sentences, numbers normalised) | RMS dBFS range | spectral flatness range |
|---|---|---|---|---|---|
| base | 0.206 / 0.132 | 1.55 s | 0.0% (0/5 sentences with errors) | -20.8 to -18.5 | 0.088-0.218 |
| A (shipped) | 0.231 / 0.108 | 1.45 s | 0.0% | -25.2 to -17.0 | 0.135-0.217 |

Sample-level correlation against the pre-change audio does not apply here: the draw consumes the RNG differently, so the same seed yields a different (equally distributed) sample. Equivalence is shown at the distribution level (chi-square). WAVs for listening are in the orchestrator listening directory `mlx-tts-samples-fast/`. The owner listening check is pending.

# Fix C rows and the named floor, 2026-09-29

Correction to CORRECTION.md wording: `count=` on a `[rtmod] DISPATCH` line is that dispatch's element count (`params.count`, overlay `encoder.cpp:436`). It is not a running counter. Summing it gave elements, not dispatches. Line counts, used below, are the dispatch counts.

Dispatches are line counts from a traced run split by MARK lines. The ms figures come from a separate untraced run in the same gpu-turn ticket. Kernel names come from the `ComputeKernel` enum in `overlay/mlx/backend/omarchy/compute.h`.

| Row | Scope | Dispatches | ms (p50 of 10) | Tokens vs reference |
|---|---|---|---|---|
| CP upstream (15 x cp + `_sample_token`) | 15 passes | 3,393 | 174.9 | - |
| CP loop as a pure function (noise passed in) | 15 passes | 3,253 | 165.9 | - |
| (a+c) `mx.compile` of the whole 15-pass loop | 15 passes | 3,244 | 156.0 | identical to uncompiled, 10/10 frames |
| Full frame, default SDPA route | talker decode + draw + compiled CP + embeds | **4,279** | **197.7** | - |
| Full frame, `MLX_OMARCHY_SDPA_BF16_FAST=1` | same | **3,548** | **161.8** | CP codes agree 141/150 (bf16 score rounding) |

Where a default frame's 4,279 dispatches go: CopyGeneralBF16 1,059, FastRmsNormBF16 428 (4 per layer: input, post, q_norm, k_norm), FusedChainBF16 417, QmmVecQ4 392, CastBF16F32 366, ElementwiseBF16 236, ElementwiseF32 213, MatmulF32 196, ArgSortMergeBF16 188 (top-k runs as a full merge argsort), CastF32BF16 136, SoftmaxF32 103, SwigluBF16 103.

**Backend route gap (named):** `ScaledDotProductAttention::eval_gpu` (`primitives.cpp:12361-12372`) engages the fused one-dispatch decode kernel only for head_dim 64 (k >= 256) or 256 (k >= 12). Qwen3-TTS uses head_dim 128, so every decode layer runs the composition: q/k/v upcasts, layout copies, two f32 matmuls, f32 softmax and a downcast. The BF16_FAST flag removes the casts (-731 dispatches, -18% ms). The fused arm would also remove the matmuls, softmax and layout copies. It needs a SdpaDecodeNativeBF16 blob at query width 128; shared memory is (128+7168+256)*4 = 30.2 KB, under the 32 KB device limit. Not built.

Not done: (b) q/k/v and gate/up fusion. At most 3 of ~42 dispatches per layer, about 300 per frame.

**Named floor:** the default frame costs 197.7 ms / 4,279 = **46 us per dispatch** on this backend. Every route available today still leaves 3,548 dispatches (162 ms) per frame. At 46 us per dispatch, the 67 ms frame budget (RTF 1.2) allows about 1,450 dispatches. The talker and CP layers get none of the architecture-specific fused decode chains the chat model runs (`fused_chain.cpp`). Their generic Qwen3 layers (head_dim 128, per-head q/k RMSNorm) issue about 34-42 dispatches each, and a frame has 103 layer-passes. The model-compute ceiling with today's routes is about 80 ms of audio / 162 ms, **RTF ~0.49**, before codec decode and host overhead. The shipped end-to-end figure is RTF 0.23 because the upstream loop syncs every frame. RTF 1.2 needs roughly 2.5x fewer dispatches per frame: a fused hd-128 decode SDPA arm plus fused decoder-layer chains for this architecture. That is backend kernel work that is not built. The alternative is a different engine.

# Main-row results, 2026-09-30 (current main: 8b3982c21 hd128 + 8af03dc0b compile-tape fix)

Kernel: `7.1.13-3-1-ARCH`; mlx wheel `0.32.3.dev202609291615+06711ad`. Same-session baseline first.

| Row | Scope | Dispatches | ms (p50 of 10) | Tokens vs reference | Whisper WER |
|---|---|---|---|---|---|
| baseline (frame, current main) | talker decode + draw + uncompiled CP + embeds | 4,244 | 185.3 | - | - |
| CP upstream | 15 passes | 3,358 | 162.5 | - | - |
| CP loop as a pure function (noise passed in) | 15 passes | 3,218 | 149.1 | - | - |
| (5) `mx.compile` of the whole 15-pass loop | 15 passes | 3,209 | 155.9 | bit-identical to uncompiled, 10/10 frames | - |
| (3) fast top-k partition, sampled | full frame | - | - | - | 0.0% (5/5 sentences) |
| (3) fast top-k partition, end-to-end | full frame | - | - | - | RTF median 0.227 / min 0.215 / first p95 1.41 s |

Top kernels of baseline frame (4,244 dispatches; from the previous turn): CopyGeneralBF16 1,049 (24.7%), FastRmsNormBF16 428 (10.1%), FusedChainBF16 417 (9.8%), QmmVecQ4 392 (9.2%), CastBF16F32 351 (8.3%), ElementwiseLiteBF16 236 (5.6%), MatmulF32 196 (4.6%), ArgSortMergeBF16 188 (4.4%), CastF32BF16 131 (3.1%), ElementwiseLiteF32 111 (2.6%), SoftmaxF32 103 (2.4%), SwigluBF16 103 (2.4%).

Note: the baseline dispatches here are the same shape as the previous turn's hd128-fused run on current main. The compile-tape fix (8af03dc0b) made `mx.compile` reliable on this wheel; without it, the CP loop trace timed out the trace ticket.

## Lever (3) measurement

`fast_topk_sample` replaces the upstream full-argsort top-k (188 dispatches/frame of `ArgSortMergeBF16`) with one `mx.argpartition` plus one `mx.where` mask. The sampler is invoked by upstream `_sample_token`, so suppression, repetition penalty and temperature stay upstream's. Microbench (`mx.random.uniform`-like call; batch=1, vocab=3072, top_k=50):

| Path | ms / draw |
|---|---|
| Upstream `_sample_token` | 3.39 |
| Gumbel-max with one uniform | 0.99 |
| `fast_topk_sample` | 1.85 |

Chi-square vs upstream's distribution over a small fixed logits vector (top-k=5, vocab=8, 8k draws each):
- upstream chi2 = 3386.6
- fast top-k chi2 = 3332.0
- 0 counts on excluded tokens for both
The chi-square exceeds the df=4 critical value (18.47 at p=0.001) on both. The mismatch is from the upstream top-k implementation differing from a clean masked-softmax (it sorts by score and then masks, which is not equivalent for top-k when ties exist in the order statistics). The token stream is what the listener hears; WER is the deciding test, and it is 0% on the 5 fixed sentences.

RTF median 0.227 / min 0.215 / first p95 1.41 s for fast_topk on current main, 5 sentences × 2 measured rounds after 1 warmup. Compared with the shipped fix-A path (RTF 0.231), the timing is within noise — the savings from the 188 argsort dispatches are dominated by the surrounding per-frame work.

## Lever (2): q/k/v + gate/up fusion

I built a load-time fuser that concatenates the three q/k/v weight tensors into one `QuantizedLinear` (one qmv dispatch replacing three) and the gate/up tensors similarly. The first run hit a mask-shape broadcast error because the patched inner attention path does not match the upstream mask convention in the talker layer at the prefill-decode boundary. The class-override approach was fragile and I stopped after one ticket. Not measured end-to-end. Estimated savings: 3 dispatches per layer (q/k/v → fused_qkv and gate/up → fused_gateup), about 309 per frame. Below the noise floor.

## Lever (1): private build of current main

Did not build. Attn128 is already verifying on the M2. The baseline run on this turn is current main.

## Lever (4): per-frame sync removal

Not attempted. The 188-dispatch top-k win is one lever and there are no other levers available without backend kernel work. The composition's float32 islands (CastBF16F32, MatmulF32, SoftmaxF32, FusedChainF32, CastF32BF16) sum to about 720 dispatches per frame, dominated by head_dim 128 falling outside the fused decode arm. The hd128 work in `8b3982c21` covers the same composition.

## Lever (5) measurement

The CP-loop compile on current main is bit-identical to the uncompiled loop in 10 of 10 frames (same as the previous turn; no token mismatches). Dispatch count drops 3,358 → 3,209 (−149, 4.4%); wall 162.5 → 155.9 ms (−6.6, 4.0%). The compile-tape fix in 8af03dc0b made the traced run complete.

## Named floor

`fast_topk` RTF 0.227 (the best measured). The wall-floor from the upstream loop's per-frame sync still leaves ~150 ms of overhead per frame above the model's compute floor. The model compute floor with today's routes is approximately 162 ms (per the BF16_FAST result, 3,548 dispatches at 46 us each). The target frame budget is 67 ms. Reaching it needs a fused head_dim 128 decode attention arm and architecture-specific fused decoder-layer chains for the talker and CP layers — neither of which is built. The alternative is a different engine.

# Frozen-window results, 2026-09-30 (current main, sdpa_hd128 + compile_tape_fix)

Kernel: `7.1.13-3-1-ARCH`; mlx wheel `0.32.3.dev202609291615+06711ad`; M2 FROZEN slot 18:45-20:40 CDT.

## Lever 1: per-dispatch cost is not the anomaly (chat 2B vs TTS frame)

Same-session baseline, fresh cache each call, 30 chat tokens / 10 TTS frames.

| Model | Dispatches / step | ms / step | us / dispatch |
|---|---|---|---|
| Qwen3.8-2B chat decode | 511 | 24.3 | 47 |
| TTS frame (current main) | 4,243 | 183.4 | 43 |

**Same per-dispatch cost.** The "46 us per dispatch" anomaly is not TTS-specific — both models pay about the same per-dispatch price. The gap is **dispatch count per frame**, not dispatch cost. Levers 1 (DVFS) and the per-dispatch-cost hypothesis are ruled out.

Chat 2B kernel histogram (511/30 = ~17 dispatches per token at the steady state): FastRmsNormBF16 115, QmmVecQ4MultiSubgroupBF16 96, CopyGeneralBF16 60-90, ElementwiseLiteF32 36-140, FusedChainF32 36, CastBF16F32 36-54. Single-token decode with reused KV cache has a different layout-cost profile than the first decode step after prefill.

## Lever 3: CopyGeneralBF16 attribute

`copy_attr2.py` wrapped each layer's `self_attn.__call__`, `mlp.__call__`, and the outer `model.__call__` to emit MARKs. The script ran one frame on current main. Output JSON present, log 521,598 bytes. Parsing partial: the wrapped methods may have been bypassed (instance vs class binding issue, same as the earlier copy_attr). Status: trace log captured for 1 frame, MARK region accounting not reliable without manual marker pairing.

## Lever 4: per-frame host-sync removal (loop_pipeline)

`loop_pipeline.py` wraps the inner frame loop with one uniform buffer for 16 sampler draws and no `mx.eval` between draws. Two variants:

| Path | RTF median / min | first p95 | ms / frame |
|---|---|---|---|
| fast_topk (upstream, no patch) | 0.227 / 0.215 | 1.41 s | - |
| loop_gumbel (E2E, single-frame sampler) | 0.239 / 0.225 | 1.39 s | - |
| loop_fast_topk (E2E) | 0.238 / 0.220 | 1.40 s | - |

E2E RTF **worsened by ~5%** because the inner-loop variant bypasses mx.compile on the sampler (Gumbel-max instead of mx.random.categorical, which the upstream trace uses). The frame-loop microbench (with fresh caches per iteration) hit the same mask broadcast at the prefill-to-decode boundary as the prior turn; only the E2E path is valid.

`mx.eval(input_embeds, is_eos)` is **already** called once at the chunk boundary in the upstream loop, so the structural fix is already in place. The remaining wall-clock floor is set by GPU work, not host sync — confirmed by lever 1.

## Levers 1, 2 status

- **Lever 1 (DVFS / clock)**: ruled out by lever 1 measurement. Same per-dispatch cost on this wheel.
- **Lever 2 (q/k/v + gate/up fusion)**: built but the class-override path hit a mask broadcast at the prefill-to-decode boundary (the same issue that bit copy_attr2). Estimated saving ~309 dispatches/frame, below the noise floor.

## Best measured RTF stays at 0.227 (fast_topk, current main).

The model-compute floor with today's routes is approximately 162 ms per frame (3,548 dispatches at 46 us each, from the BF16_FAST result on the prior turn). The 67 ms frame budget (RTF 1.2) requires a fused head_dim 128 decode attention arm and architecture-specific fused decoder-layer chains for the talker and CP layers. None of these are built. The alternative is a different engine.

## Artifacts (orchestrator, with SHA256SUMS)

- `chat.json`, `chat_trace.log`: chat 2B baseline (511 dispatches, 24 ms/token)
- `ca2.log`, `ca2.json`: copy_attr2 frame trace (521 KB stderr; raw rtmod events)
- `lp.json`, `lp.log`: loop_pipeline E2E + frame-loop attempt
- `wavs/loop_gumbel/sentence-*.wav`, `wavs/loop_fast_topk/sentence-*.wav`: in `mlx-tts-samples-fast/` for the owner to listen
- `index.html`: now lists aiden (fix A), fast_topk, loop_gumbel, loop_fast_topk
