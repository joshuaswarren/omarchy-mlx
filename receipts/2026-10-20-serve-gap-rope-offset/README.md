# Served-generator gap on a 13-inch M1: gemma-4 RoPE offset joins (2026-10-20)

`mlx_lm.server`'s generator ran gemma-4-E4B 3.7 to 3.9 ms per token (6 percent) slower than the `bench_decode` route on the same chip,
token ids equal. Stream choice, the sampler factory and the detokenizer were measured and are not causes. The cause is a host join per
own-cache layer in the RoPE trig gate. The two patchers in this change remove it: the served-generator gap closes to -0.13 ms (not real).

## Scope

One machine, one model: a 13-inch M1, gemma-4-E4B 4-bit, one prompt, greedy, 128 tokens, performance governor, ANE filler off.
Nothing here measures another model or chip. The patchers also apply to any mlx-lm 0.32 sliding-window model, but only gemma-4 builds
its RoPE offset with `mx.array(cache.offset)`, so only gemma-4 is expected to change.

## Identity

| Item | Value |
|---|---|
| Machine | 13-inch M1 (T8103), kernel `7.1.12-2-20261013.18-sep-ARCH`, boot `<id>` |
| mlx wheel | mlx-omarchy `0.32.4.dev202610102357+e57cb301`, `libmlx.so` sha256 prefix `9a225264dcf2dbcb` |
| mlx-lm tree | stock mlx-lm 0.32 plus this repo's mlx-lm patch series including #232 and #233: `generate.py` `5b077a35686d98cd`, `server.py` `2533fcc1f62bb3bf`, `models/cache.py` `7dcc85f3223f9786` |
| Fixed tree under test | the same tree plus `patch-mlx-lm-rotkv-host-offset.py` and `patch-mlx-lm-gemma4-rope-offset.py`: `models/cache.py` `99ffd1ef2a190719`, `models/gemma4_text.py` `27b8b85623f590ad` (stock-series `73ffa75e9069ce7b`) |
| Model | `mlx-community/gemma-4-e4b-it-4bit`, snapshot `475b9088d29754a3379866cf5aeb6b41acd313c2` |
| Prompt | "Write a short story about a lighthouse keeper who finds a message in a bottle." |
| Ids digest (128 tokens, all arms equal) | `6e98b674dcae2e22` |

## Method and results

Every ticket fixed its decision rule before the run: a gain is REAL at 3 percent of `m(A)` (`m` = milliseconds per token) with every
run median of the faster arm above every run median of the slower arm; spread above 5 percent is NOISY. Arms A (`bench_decode` through
the thread clamp), F (the `mlx_lm.server` generator in process, no HTTP), R (F on the fixed tree). Three rotated rounds, two request
pairs per process for F and R. The raw ticket logs are in `raw/`.

1. Gap ticket (`raw/out-gap-e4b-20261018T171949Z.log`, tree before the fix). A 15.52 tok/s (64.44 ms), F 14.63 (68.34 ms), gap +3.90 ms REAL.
   A new thread-local stream (+0.14 ms), the bench sampler factory (+0.17 ms, a verified no-op control) and bypassed detokenizer
   operations (+0.09 ms) moved nothing.
2. Dispatch census (`raw/out-gapcensus-e4b-20261019T205434Z.log`, trace on, untimed). F ran 1527.0 dispatches per decoded token against
   1507.0 for A; the only kernel id that differed by 0.5 or more was 103 (`ElementwiseI32`, `count=1`): 0 against 20.
   Removing those 20 dispatches alone (a host offset mirror for `BatchRotatingKVCache`) left the gap at +3.49 ms.
3. Counter census (`raw/out-bar-e4b-20261019T223410Z.log`, `BatchGenerator` on one thread, so the wrapper thread and queue are not the
   cause): B +4.10 ms, with the mirror +4.01 ms, with the mirror and no per-layer `mx.depends` +4.61 ms, all REAL. Extra submissions
   (+20), barriers (+20) and buffer copies (+24) per token stayed.
4. Trace-reason census (`raw/out-reason-e4b-20261020T022729Z.log`, traced, untimed). B minus A per decoded token:
   `JOIN reason=rope_offset_scalar` +24.0, submissions +20.0. B5 kept both.
5. Cause. `rope_trig_gate` (`overlay/mlx/backend/omarchy/primitives.cpp`) reads a one-element RoPE offset directly only when it is a host
   constant (`Status::available`, no primitive). Anything else takes `settle` plus `synchronize("rope_offset_scalar")`. `gemma4_text.py`
   built the offset as `mx.array(cache.offset)`: a Python int becomes a host constant, but a batch cache's offset array becomes a Copy
   node with a primitive, so each of the 24 own-cache layers joined the stream and the evaluator committed.
6. Fix ticket (`raw/out-rotfix-e4b-20261020T030426Z.log`, the patchers in this change). Census of R: `JOIN rope_offset_scalar` 0.0 per
   token (60 lines in each of the 24- and 40-token runs; difference 0), kernel 103 0.0 per token. Timing: A 15.48 tok/s (64.59 ms,
   spread 0.4%), F 14.64 (68.30 ms, 0.1%), R 15.51 (64.46 ms, 0.5%). Gap +3.71 ms REAL; gain F to R +3.84 ms (+5.9% of `m(A)`) REAL,
   103% of the gap; remainder R minus A -0.13 ms not real. Ids digests equal in every arm. An earlier run of the same fix with one more
   change (no per-layer `mx.depends`) read the same: gain +3.95 ms, remainder -0.12 ms; that change is not shipped because step 3 shows it
   does nothing alone.

## What the change does

- `scripts/patch-mlx-lm-rotkv-host-offset.py`: `BatchRotatingKVCache` keeps `offset_list`, the host mirror `BatchKVCache` already has,
  and stores `mx.array(offset_list)`. An outside assignment (`filter` runs on every request through `PromptProcessingBatch.split`)
  clears the mirror; the first decode step after that rebuilds it once from the device value.
- `scripts/patch-mlx-lm-gemma4-rope-offset.py`: an array cache offset goes to RoPE as is; an int still goes through `mx.array`.
- `MLX_OMARCHY_KV_HOST_OFFSET=0` keeps the lazy device offset in the rotating cache. The gemma-4 pass-through is value-identical either way.
- `tests/test_rotkv_host_offset_behavior.py`: bit-identical keys, values, offset and left padding against the device path through window
  rotation, with and without a `filter()` between prefill and decode, for four padding shapes; ten of the thirteen tests fail without the
  rotating patcher.

## Not shown

The gemma-4 pass-through has no CPU test (no tiny gemma-4 model); the ticket compares generated-id digests on the real model. No other
model or chip was run. The first attempt (mirror only) changed nothing in the real flow because `filter` cleared the mirror; the
CPU test now calls `filter`.
