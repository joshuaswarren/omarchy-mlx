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
