> **Addendum 2026-09-30 (supersedes everything below).** The rows and conclusions below are
> **pre-fix and invalid**; they are kept for the record only.
> 1. *Pre-fix backend:* Qwen3.5-9B (Hk=16, Hv=32) fell into a per-token loop in GDN prefill on
>    the live venv. main d95e88363 fixes it (`mlx-lm-gated-delta-fast-route-repeat.patch`), so the
>    9B prefill/TTFT numbers and the "39x slower prefill" argument below do not hold.
> 2. *Harness v1 defects:* generation did not stop at EOS (replies ran on into invented turns),
>    thinking was not disabled (the coordinator sends `enable_thinking=False`), three GSM8K golds
>    were wrong and most items were not from the test split, and three IFE checks did not match
>    their instructions. GSM8K/IFE/card numbers below are therefore not measurements of the models.
> Harness v2 (EOS stop, thinking off, coordinator system prompt, GSM8K test rows 0-19 pinned at
> openai/gsm8k@740312a, corrected IFE checks, 8/4/4 card set) was committed before any v2 run.
> The v2 results replace this file's table; see `README.md`.

# Chat-model benchmark — recommendation

## Default pair choice for the 96 GB M2-class machine

**Default chat model: `qwen3.8-2b-4bit` (SiddhJagani/Qwen3.8-2B-mlx-4Bit).**
That is the existing Everyday pair in the catalog. On the M2 measured here
(M2 Max T6021, Honeykrisp Mesa, 96 GB), it satisfies every constraint in
the offline-assistant design's interactive-latency budget:

- **TTFT p50 = 0.30 s** for a 170-token system prompt + 256-token reply
  (budget: p95 <= 2.0 s). It is the only candidate in this study that
  clears the 2.0 s budget.
- **Decode 69.2 tok/s over 512-token prompt and 49.8 tok/s over 2048-token
  prompt** — the fastest of the three measured candidates by 3-15x.
- **Cold load 1.15 s; warm load 1.16 s; peak memory 1.06 GB** — fits
  comfortably under the design's memory admission for a 16 GB machine as
  well (16 GB + KV + workspace + max(2 GiB, 10%) leaves 12 GB for the
  weights with KV budget to spare).
- **Card adherence 10/16** (6 plain prose, 8 expecting a card; the 4
  plain-text questions also passed via the markdown-structure path).

The data refute the hypothesis that "a bigger model would obviously win
on a 96 GB machine." The only larger candidate that was measured under
this stack, `mlx-community/Qwen3.5-9B-MLX-4bit` (4.5x more parameters),
takes **6.18 s for TTFT** — **20x slower** than the 2B and **3x over the
budget**. Its prefill_2048 is 56.8 s, which is **39x** the 2B's 1.45 s.
That is far worse than the 4.5x size-scaling that parameter count alone
predicts: the gap is the Qwen3.5 GDN / hybrid-attention prefill path on
the Vulkan backend on this Mesa wheel, not bandwidth. Decode only drops
by 3.8x (18.3 tok/s vs 69.2 tok/s) — closer to size scaling — so the
Quality pair's interactive chat is killed by prefill, not by token
generation.

## Default pair choice for a 16 GB machine

**Default chat model: `qwen3.8-2b-4bit` (the same 2B).** Memory
admission for weights (1.06 GB measured peak during steady decode of a
170 + 256 token prompt) plus a small KV cache plus workspace plus the
greater of 2 GiB and 10% headroom all fit comfortably in 16 GB. The 9B
measured here uses 5.04 GB at peak on its perf run, which alone leaves
no headroom for the 2 GB safety reserve; the 31B Gemma and 27B Qwen3.8
weights themselves are 16+ GB and do not fit on a 16 GB machine
without offloading, which the assistant design forbids. Only the 2B and
the (not-yet-measured) mxfp4 27B variant are realistic 16 GB candidates,
and the mxfp4 27B still has 15 GB of weights before KV or workspace.

## Why 2B, not a "better" model, on this hardware

The owner asked the question literally: "why aren't we running a better
model?" The measured data say:

1. **The 2B is the only model measured that meets the 2.0 s interactive
   TTFT budget on the M2.** The 9B is 3x over the budget; the 35B MoE
   is 4x larger than the 9B and will be slower; the 31B Gemma is
   similar. (GDN hybrid prefill is the bottleneck, not bandwidth.)

2. **The 2B is the lowest-memory realistic candidate that fits a 16 GB
   machine with the safety reserve.** It is therefore also the right
   default on a 96 GB machine, because there is no measured larger model
   that meets the latency budget.

3. **The Qwen3.5-9B's GSM8K score (3/20) and IFE score (1/19 thinking-
   stripped) are not enough quality to justify a 20x slower TTFT or a
   39x slower prefill on a chat assistant.** Quality and latency are
   traded here, and latency is the binding constraint under the design.

4. **The Ministral-3-8B-Instruct-2512-4bit (plain attention) meets the
   2.0 s TTFT budget at 0.99 s, but its decode is only 9.7 tok/s, its
   card adherence is 7/16 vs the 2B's 10/16, and its GSM8K data point
   is incomplete (1/1 of the suite before the bench was killed). It is a
   plain-attention control that proves the GDN prefill is the 9B
   bottleneck; it does not displace the 2B on latency, decode, or
   card-adherence evidence.**

## Quality proxy caveat

GSM8K scores are unstable on this harness: the 2B's GSM8K shows 0/20
under strict "Final answer:" extraction but the actual model output
contains chat-template markers (`<|im_end|><|im_start|>assistant`),
indicating template leakage during the GSM8K/IFE generation path that
does not occur during card generation. The 2B's GSM8K ability is
therefore not measurable on this harness path; the 0/20 score is a
harness finding, not a real-world zero. The 9B's 3/20 is real (it
produces coherent math reasoning on the prompts it answered).

## Public-download eligibility

- `SiddhJagani/Qwen3.8-2B-mlx-4Bit` — public HF download, Apache-2.0.
- `mlx-community/Qwen3.5-9B-MLX-4bit` — public HF download, Apache-2.0.
- `mlx-community/Ministral-3-8B-Instruct-2512-4bit` — public HF
  download, Apache-2.0.
- `mlx-community/Qwen3.8-27B-4bit` — public HF download, Apache-2.0.
- `mlx-community/gemma-4-31b-it-4bit` — public HF download, Gemma terms.
- `mlx-community/Qwen3.8-27B-mxfp4` — public HF download, but the
  catalog entry `Qwen3_5ForConditionalGeneration` is a VL variant; the
  chat-only path via `mlx_lm` is unproven and recorded as a finding if
  smoke_load.py fails to serve it on this Vulkan backend.
- `NovaeonStudio/Qwen3.8-35B-A3B-Distill-oQ8-fp16-mtp` — third-party HF
  download (NovaeonStudio, not mlx-community). Not a default-eligible
  model; a default must be an mlx-community or first-party build with
  reproducible provenance. This model is a 39.5 GB fp16 MoE and is
  included here only to test whether a 3B-active-parameter MoE can
  beat the 2B's latency/quality trade.

## Items not yet measured (in flight or queued)

- `mlx-community/Qwen3-4B-Instruct-2507-4bit` (Apache-2.0, plain
  attention, ~2.3 GB, pin 50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b) —
  queued next in sequencer_v3. If it measures TTFT <= 2.0 s with
  card adherence >= 10/16, it is a candidate for the 16 GB default.
- `mlx-community/Qwen3.8-27B-4bit` (reduced protocol) — queued.
- `NovaeonStudio/Qwen3.8-35B-A3B-Distill-oQ8-fp16-mtp` smoke —
  queued; if load fails, recorded as a finding.
- `mlx-community/Qwen3.8-27B-mxfp4` and `mlx-community/gemma-4-31b-
  it-4bit` are tagged skip_if_busy and will only run if the queue is
  empty at their turn.