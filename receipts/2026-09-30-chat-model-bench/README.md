# Chat-model benchmark for the default pairs — 2026-09-30

Question: why does the Everyday pair use a 2B chat model, and which chat model should each
default pair use? Data only; no product code changed.

**Answer.** On the fixed stack the 2B is no longer the right Everyday model. On the M2, four
larger models also meet the 2.0 s first-text budget, and two of them are much better on every
quality proxy. The 2B was reasonable on the stack as it stood this morning: larger GDN models were
slow, and the live wheel returned garbage for long prompts. Both defects are now fixed (details
below).

| Machine | Recommended chat model | Why |
|---|---|---|
| 96 GB M2-class | `mlx-community/Qwen3.5-9B-MLX-4bit` | TTFT 0.78 s; best native cards (6/8, incl. chart, facts, form); GSM8K 18/20, IFE 20/20; 6.5 GB peak |
| 16 GB | `mlx-community/Qwen3-4B-Instruct-2507-4bit` (new, needs a catalog entry) | TTFT 0.49 s; GSM8K 20/20, IFE 20/20; 6/8 cards renderable (2 native + 4 via markdown promotion); 3.7 GB peak |

Neither recommendation marks anything `recommended: true`. The 4B is not in the catalog yet
and still needs managed-launch and pair qualification.

## Results (release wheel, each timed turn alone on the GPU)

Environment: `<project-m2>` (Apple M2 Max T6021, 96 GB, Linux 7.1.13, Honeykrisp Vulkan). The
Vulkan device reports 50.6 GB total memory. The run used a private copy of the v0.7.6 release venv
(wheel `0.32.3.dev202609291615+06711ad`, mlx-lm 0.31.3 with the release patches) plus main
9ef622d14's `mlx-lm-gated-delta-fast-route-repeat.patch`. Provenance line from every turn:
`mlx-omarchy 0.32.3.dev202609291615+06711ad ... verified=match ... libmlx.so=sha256:004d24b6c951a253`.
Each chunk ran as one `gpu-turn -m 8` ticket; see "Methods" for how the harness was synced.
Every timed turn logged `/proc/loadavg` (0.3-1.4 at start) and a count of other processes holding
a render node; the count was 0 for every row below.

| model | cold load s | warm restart s | prefill 512 tok/s | prefill 2048 tok/s | decode tok/s | TTFT card prompt p50 / max s | peak GB (mx) | valid native cards /8 (+md-promotable) | prose kept prose /8 | GSM8K | IFE | replies hitting cap |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3.8-2B (Everyday today) | 1.79 | 1.53 | 843 | 1282 | 34.7 | 0.24 / 0.34 | 1.78 | 0/8 (+5 md) | 8/8 | 11/20 | 17/20 | 4 |
| Qwen3-4B-Instruct-2507 | 1.03 | 0.95 | 452 | 356 | 20.3 | 0.49 / 0.62 | 3.67 | 2/8 (+4 md) | 8/8 | 20/20 | 20/20 | 1 |
| Qwen3.5-9B | 1.90 | 2.08 | 344 | 401 | 26.0 | 0.78 / 0.78 | 6.47 | 6/8 (+0 md) | 7/8 | 18/20 | 20/20 | 3 |
| Ministral-3-8B-2512 | 2.58 | 1.73 | 323 | 357 | 9.3 | 0.84 / 0.85 | 6.18 | 4/8 (+1 md) | 7/8 | 2/2 (skipped rest) | not run (skipped) | 1 |
| Qwen3.8-27B (Quality today), reduced protocol | 8.29 | 9.04 | 103 | 97 | 8.6 | 2.81 / 3.03 | 18.22 | 5/8 (+1 md) | 6/8 | not run (reduced) | not run (reduced) | 1 |
| Qwen3.8-35B-A3B MoE (NovaeonStudio) | 8.47 | 12.2-40.1 (not reproducible) | 10 | out of GPU memory | 3.4 | 27.10 / 27.29 | 41.51 | not run (perf only) | not run | not run | not run | — |

Column notes:
- **TTFT card prompt**: the time to the first generated token for the coordinator's real
  ordinary-chat prompt. The system prompt is `Answer the user using their supplied facts. ` +
  `SCHEMA_PROMPT_COMPACT`, and the prompt is 235-244 tokens after the chat template. Every product
  turn carries this system prompt, so a 128-token prompt never occurs in the product. This prompt
  is therefore stricter than the design's 128-token case. p50 and max are over 5 runs, or 2 on the
  reduced protocol. Engine-level only: it excludes HTTP and coordinator overhead.
- **Prefill/decode**: synthetic prompts of 512/2048 tokens, greedy, 128 decode tokens with EOS
  ignored. Prefill is the time to the first token; decode is the steady rate after it. Median of 5
  (reduced protocol: 2, and 1 at 2048).
- **Peak GB**: `mx.get_peak_memory()` over the whole perf phase (weights + KV + workspace at the
  2048-token prompt). Process RSS was 0.3-0.44 GB for every model. The `/proc/meminfo` MemAvailable
  delta is recorded in each run file, but it is host-wide and other agents' processes contaminated
  it (up to 42 GB), so it is not used.
- **Cards**: 16 prompts in 3 groups: 8 that want a checklist, timeline, comparison, chart, facts,
  decision or form; 4 plain questions; 4 that use list or option words but want prose. The system
  prompt and the compact/full schema choice match the coordinator (`FULL_CARD_CUES`,
  `user_requested_full_schema`). Settings: repetition penalty 1.1, `max_tokens` 700, thinking
  disabled. A native card must pass `components.validate_components`. "md-promotable" counts
  replies without a valid fence that `card_promotion.extract_text` turns into a valid card; the
  coordinator does this today, so native + promotable is what a user would see.
- **GSM8K / IFE** are **proxies, not benchmarks** (n = 20 each). GSM8K uses test rows 0-19 of
  `openai/gsm8k` at revision 740312a, graded on the last "Final answer:" number. IFE uses 20
  deterministic checks written for this study. The corpora were hashed before any run:

```
35f959901540500e4a59db41a9319473aadb603b48f537687ef1c8526a5f6c67  cards_16.jsonl
42988891e6c2cd3da66bd9c30f4568e572a9cbb7e5de8ad0405f97d00f72ee77  gsm8k_20.jsonl
a79df7071d6b5f6d237b3e7d702601731a0bdcdda146da38405c428fa4971a0e  ife_20.jsonl
```

- **Replies hitting cap** counts replies that reached the token limit (700 cards, 768 GSM8K, 256
  IFE). The 2B filled the 700-token card cap on 3 of 16 card prompts.

Per-card outcomes (from `scripts/bench/details.py v3/*.json`):
- 2B: 0 native cards. It writes markdown (tables, lists) instead of the fence; 5 of those
  replies promote to a card (checklist or comparison). The chart, facts and form prompts produce
  no card.
- 4B: native chart and form. Checklist and comparison fences are invalid (component `type`
  missing). Timeline and facts render only through promotion; decision produces no card.
- 9B: native checklist, timeline, comparison, chart, facts and form. The Apollo timeline filled the
  700-token cap inside the fence (leaked fence). Decision produced prose. It emitted one unwanted
  checklist card on a list-word prose prompt ("why people like making lists").
- 27B: native timeline ×2, chart, facts, form. One checklist filled the cap, two comparison rows
  were malformed, one plain question got a `text` component, and one list-word prose prompt got a
  checklist.

## The 2.0 s first-text budget and memory admission

Design budget: first visible answer p95 ≤ 2.0 s (128-token prompt, 256-token reply). Measured on
the longer real card prompt:

- **Within budget:** 2B 0.34 s max, 4B 0.62 s, 9B 0.78 s, Ministral 0.85 s.
- **Over budget:** 27B, 3.03 s max (2.81 s p50). The MoE is far over (see below).
- **Full-schema turns** (chart/graph/form/decision/options/facts/sources cues) send a prompt of
  829-860 tokens. [INFERENCE from the measured 512-token prefill rates, not a measured TTFT]
  2B ≈ 1.0 s, 4B ≈ 1.9 s, 9B ≈ 2.5 s, 27B ≈ 8 s. So the 9B probably exceeds 2 s on those turns.
  The design budget does not cover them, but it is a real user-visible cost.
- **Reply completion** (not the design budget): 256 tokens take about 7.4 s on the 2B, 9.8 s on
  the 9B, 12.6 s on the 4B and 30 s on the 27B, at the measured decode rates.

Memory admission (design: weights + KV + workspace + the greater of 2 GiB and 10% of
MemAvailable), using measured peaks:

- **96 GB M2:** every row fits. The binding limit is the Vulkan device heap (50.6 GB), not host
  RAM. The MoE's 36.8 GB of weights leave too little for a 2048-token prefill.
- **16 GB machine:** reserve 2 GiB. On host-RAM arithmetic, 2B (1.8 GB), 4B (3.7 GB) and 9B
  (6.5 GB) fit and 27B (18.2 GB) does not. [INFERENCE, unverified: no 16 GB machine was measured]
  If its Vulkan heap is the same 53% share of RAM seen here, it is about 8.4 GB. The 9B plus Laya
  and voice would then be at or past that heap. The 4B leaves room, so it is the 16 GB default.

## Why the Everyday pair used a 2B, and why that no longer holds

The design picked the 2B as the "lower-memory, lower-latency choice". Today's measurements
found three stack defects. Each made larger or longer-prompt GDN models look worse than they are:

1. **9B prefill fell into a per-token loop** (Hk=16 ≠ Hv=32), giving 39 tok/s at 512 tokens and a
   6.2 s TTFT. It is fixed on main (9ef622d14) and now runs at 344 tok/s with a 0.78 s TTFT.
2. **The live wheel `+29cba8e` returned non-finite logits** (argmax token 0, replies of "!!!!")
   for GDN prompts past about 300-500 tokens. With the 2B, one forward pass was finite at 244-262
   tokens and non-finite at 512, 1024 and 2048 tokens and on every full-schema card prompt. On
   that wheel, the 2B answered every chart/form/facts/decision turn with garbage, and 13 of 56
   9B replies were garbage. Swapping wheels and mlx-lm trees showed the wheel is the cause. Main
   upgraded the live venv to the release wheel `+06711ad`.
3. **The first repeat patch broke decode:** its unconditional `if Hv != Hk` raised
   `UnboundLocalError` on every step with a state (all decode, all GDN models, including the
   2B). Its hunk header was also malformed. Main fixed both in 9ef622d14. Repeating only at T>1
   keeps 9B decode at 18.4 tok/s instead of 10.6 tok/s (old wheel).

On the fixed stack, latency no longer forces the 2B. It is still the fastest (0.24 s TTFT,
34.7 tok/s) and smallest, but it is last on quality: 0/8 native cards, GSM8K 11/20, IFE 17/20.
The 4B and 9B stay well inside the budget and score 18-20/20 on both proxies. That is a large
measured quality gain within the budget, so the stated rule (keep the 2B only if no larger model
gives one) removes the 2B as a default. It stays useful where speed matters most, such as a
**Fast** preference.

## Owner item 4: the Quality pair on an idle GPU

Qwen3.8-27B-4bit, release wheel, alone on the GPU:

| Measurement | Value |
|---|---|
| Prefill, 512 tokens | 103 tok/s |
| Prefill, 2048 tokens | 97 tok/s |
| Decode | 8.6 tok/s after 512 tokens, 8.0 after 2048 |
| TTFT on the card prompt | 2.81 s p50, 3.03 s max |
| Peak memory | 18.2 GB |
| Load | 8.3 s cold, 9.0 s warm restart |

The warm restart was slower than the cold load in this run; that is recorded, not explained.
Earlier "about 3 tok/s" figures were taken on a shared GPU. The 27B misses the 2.0 s budget on
the real prompt, and its card quality (5/8 native) is below the 9B's (6/8). On this stack it is
not a better Quality default than the 9B. An 8192-token prefill was not measured: the reduced
protocol Main assigned was 512/2048.

## Models that could not complete the protocol

- **NovaeonStudio/Qwen3.8-35B-A3B-Distill-oQ8-fp16-mtp** (`v3/qwen3.8-35b-a3b-moe.json` is the clean perf turn; `v3/...first-attempts.json` holds the 05:11Z turn with the 8.47 s cold load and the out-of-memory error, both alone on the GPU, and the discarded 05:36Z shared-GPU turn, whose `perf` block is not used): third-party, 36.8 GB of weights.
  mlx-lm 0.31.3 with the project patches **loads it on the Vulkan backend** (cold 8.5 s). The
  2048-token prefill fails with `VK_ERROR_OUT_OF_DEVICE_MEMORY` (Vulkan device total 50.6 GB),
  measured alone on the GPU. Measured alone with the 2048 point skipped (n=2): TTFT 27.1 s on the card prompt, prefill 10 tok/s, decode 3.4 tok/s, 41.5 GB peak. That is 13x over the 2.0 s budget. The first download omitted `chat_template.jinja`; the harness recorded the resulting template error, and the file was fetched at the pinned revision. [INFERENCE] About 3B active parameters should decode faster than the dense 9B, so 3.4 tok/s points at an unoptimized expert-gather or 8-bit/fp16 path on this backend rather than at model size. Cards, GSM8K and IFE were not run (perf-only, as
  assigned). It is not a default candidate on any measured count.
- **Qwen3.8-27B-mxfp4** and **gemma-4-31b-it**: not run. They were last in priority and marked
  "skip if the queue holds other jobs", and the queue never emptied.
- **Ministral-3-8B**: its IFE set and 18 of 20 GSM8K items were skipped under the same
  if-queue-busy rule. Its perf and card rows are complete. It also has the slowest decode
  (9.3 tok/s).

## Public download

All six measured repositories are public, ungated and Apache-2.0 on Hugging Face (API check,
2026-09-30). `mlx-community/Qwen3-4B-Instruct-2507-4bit` and `mlx-community/Qwen3.5-9B-MLX-4bit`
are conversions of official Qwen releases. `SiddhJagani/Qwen3.8-2B-mlx-4Bit` and the NovaeonStudio
MoE are community builds of a third-party distill (`empero-ai/...-Distill`). Pinned revisions:

| Model | Revision |
|---|---|
| Qwen3-4B-Instruct-2507-4bit | 50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b |
| Qwen3.5-9B-MLX-4bit | 938d8919941c6e7efd3c7150eff7fe9d12afa631 |
| Qwen3.8-2B | 0867d98bfb174b042d88461c0e7c97b86b34b381 |
| Qwen3.8-27B-4bit | 10c35caafbb80f7dc6a7a432cdd11af10a6d4818 |
| Ministral-3-8B | 182f003f01daa75f9de0f2c4d379722fd0bc1c61 |
| MoE | 5315bc8a40d751f5bded428eadd6a22007a3bf17 |

## Methods, deviations and invalid runs

- Harness: `scripts/bench/chat_model_bench.py` (commit b1bc4f3d9, idle guard added later). It
  runs one model per process, greedy decoding with thinking disabled, and stops at EOS. It
  checkpoints after every item and resumes after a kill or reboot (the M2 rebooted twice during
  the run). The run used a clone of main at d95e883 with the harness files from this worktree
  synced in, which is why the provenance line says `harness=d95e883-dirty`.
- The measurements call mlx-lm's `generate_step` in-process, with the same chat template,
  sampler settings and patched mlx-lm as the HTTP shim. They do not go through the shim's HTTP
  path. TTFT is therefore engine-level.
- **Invalid, kept for the record:** `summary.md`, `recommendation.md` and the top-level
  `qwen2b-baseline.json`, `qwen3.5-9b.json` and `ministral3-8b.json` are harness v1. That harness
  did not stop at EOS, left thinking on, used wrong GSM8K golds and mismatched IFE checks, and ran
  on the pre-fix wheel. The v2 runs on the old wheel (not committed) are also invalid because of
  the non-finite-logit defect.
- One MoE turn had three other processes holding the GPU: assistant servers left running outside
  gpu-turn. Its timings were discarded. The harness now refuses a timed phase when another process
  holds a render node.

Files: `v3/*.json` (full replies included), `scripts/bench/aggregate.py`, `details.py`,
`scan_degenerate.py` (0 degenerate replies on every v3 run), and `probe_finite.py`.
