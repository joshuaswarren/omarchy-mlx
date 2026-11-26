# Qwen3.8-35B-A3B-Distill Q4/G64 device qualification plan

Status: FROZEN test vector, awaiting a device window (t6001-test-host preferred;
M2-GPU-interleave only if the boot owner lends a slot that fits
transfer + load). No HF publication, no public catalog entry; the
artifact is local-only by decision of the owner (2026-09-20).

## Artifact under test (local, immutable)

Path on conversion host: $HOME/models/qwen3.8-35b-a3b-distill-q4-g64
Source: empero-ai/Qwen3.8-35B-A3B-Distill @ bcc2dbe2f21b213625df2dc1a5a690212373af07
Quant: affine 4-bit / group 64, routers mlp.gate + shared_expert_gate
at 8-bit / group 64 (per-path overrides in config.json), mode affine.
License apache-2.0 (source and output).

Frozen shard digests (sha256, verified twice on the conversion host;
re-verify on the device BEFORE any run and abort on mismatch):

| shard | sha256 |
|---|---|
| model-1-of-6.safetensors | 640d25f27ffbf25a9daa5bc8db147f23edc914e1f5303f99e381af25a0ef28af |
| model-2-of-6.safetensors | fc10593d6feafc4ebc68e4d3767e94532e1a1d5374827c8f009e67b691b693ee |
| model-3-of-6.safetensors | a762747f333bd1f89b8cab326b2911c1a66094668fbf908fa4a991d393341461 |
| model-4-of-6.safetensors | ca8a62e28c6132f1fbaf5a47c7cff8bd3e3879720897bfed1d0b6fa42615e22e |
| model-5-of-6.safetensors | ef843bd297584ac2eaac01fe12e6186a0fb6ddc2d0a4adc6d7afcc8bff7e1ce1 |
| model-6-of-6.safetensors | 413a4e3e34ad1e17c245732ad3cbba8752944a98763584caf3e032ead8fce4af |

config.json / tokenizer files must come from the same directory; the
plan pins the directory, not an HF download.

## Frozen test vector

Exact prompt string: `The capital of France is`

Tokenization is performed by the artifact's own tokenizer
(tokenizer.json, vocab.json, merges.txt from the pinned revision).
CPU reference encoding (mlx-lm 0.31.3, transformers 5.17.0 fast
tokenizer): record the encoded IDs in the device receipt; the device
must tokenize to the same IDs with the artifact's tokenizer — assert
this before generating.

CPU reference generation (greedy, mlx-lm 0.31.3, 4 CPU cores,
RSS peak 18.62 GiB), 16 tokens:

```
prompt:  "The capital of France is"
ids:     [11751, 11, 264, 3177, 3750, 364, 1141, 8807, 3712,
          11, 1880, 11, 321, 7431, 13, 1049]
text:    " Paris, a city known for its rich history, art, and culture. It"
```

Receipt: forward_receipt.json in the artifact directory (admission,
hashes, logits shape (1, 5, 248320), determinism across two runs).
Evidence log: full-forward2.log; fixture-level loader-contract tests:
tools/test_qwen38_distill_quant.py (16 tests, real mlx_lm module tree).

### Pass rule (frozen before any device run)

1. Tokenizer gate: device-side encoded prompt IDs == CPU-encoded IDs.
2. Greedy generation, temperature 0, no sampler overrides,
   `max_tokens=16`, stop at EOS.
3. PASS: all 16 generated IDs equal the CPU reference sequence,
   and a second device run reproduces the same sequence
   (device determinism).
4. DIVERGENCE: any differing ID. Record the full sequence and the
   first differing position with both logits-space top-5. Divergence
   does NOT auto-fail the artifact — quantized GPU and CPU kernels
   may differ in low-order bits — but the qualification is NOT
   granted on divergence; the trace goes to the owner for review.
5. The first generated ID being anything other than 11751
   (` Paris`) additionally triggers immediate divergence analysis
   before any further tokens are compared.

No BF16-parity claim is made or derivable from this protocol; the CPU
run is the locked reference fixture per the repo's comparison rule.

## Managed-memory budget (for the device admission system)

All values derived from the pinned model config (text_config), not
generic estimates:

- Layers: 40 total = 30 linear_attention (gated delta) + 10
  full_attention (`full_attention_interval: 4`,
  `layer_types[3,7,...,39] == "full_attention"`).
- Attention KV per token (the 10 full-attn layers only):
  `num_key_value_heads (2) x head_dim (256) x 2 (K+V) x 2 B (bf16)
  x 10 layers = 20,480 B/token`.
- GDN ssm state (constant once materialized): 30 layers x
  `linear_num_value_heads (32) x linear_value_head_dim (128) x
  linear_key_head_dim (128) x 4 B (fp32, mamba_ssm_dtype) =
  62,914,560 B`.
- GDN conv state (constant): 30 layers x conv_dim
  (`2 x key_dim (16x128=2048) + value_dim (32x128=4096) = 8192`)
  x (kernel 4 - 1) x 2 B = 1,474,560 B.
- Packed weights (text-resident, measured artifact): 19,537,340,176 B.

| component | bytes |
|---|---|
| packed weights | 19,537,340,176 |
| GDN state total (ssm + conv) | 64,389,120 |
| KV cache | 20,480 B/token |
| workspace margin | device-owned; measured peak overrides (ServeCatalog budget API) |

### Reservation semantics (corrected per owner, 2026-09-20; amended after review)

- BEFORE load: the pending reservation holds the FULL estimated peak,
  placed with the ATOMIC `admit_and_reserve` API (no separate
  check-then-set): packed weights + GDN state + KV for the intended
  context + workspace margin. Weights are NOT marked resident before
  materialization.
- AFTER load completes: `resident_floor_bytes` is set to PARAMETER
  bytes only — the packed weights actually loaded =
  **19,537,340,176 B**. GDN state is NOT a parameter and is NOT
  counted in the floor; it keeps its own conservative reservation
  line (64,389,120 B) alongside KV (20,480 B/token at served context)
  and workspace. If GDN state is ever folded into the floor instead,
  its allocation residency must be proven first and it must be
  counted exactly once — the default here is the separate line.
- Exact figures: parameter floor 19,537,340,176 B; GDN state
  64,389,120 B (supersedes the rounded 64,440,320 figure circulated
  in chat).

## Device queue and host preference

1. t6001-test-host (preferred): persistent disk — the 19G artifact transfers
   once and survives reboots; serves later Bonsai/Laya/SPA-queued
   windows without re-transfer.
2. M2 GPU-interleave source gap: only if the boot owner lends a slot
   where transfer + load provably fit the memory budget and the
   transfer survives a reset cycle (repeated M2 resets cost repeated
   19G transfers — this is why t6001-test-host is preferred).

Transfer integrity: sha256 the six shards after copy on the device
host; abort on any mismatch. Transfers happen ONLY during an agreed
non-measurement window on the target host (transfer I/O pollutes
timing), and only after the persistent-disk headroom check confirms
room for the 19G artifact plus working space.

## Run protocol on the device

Stage G (GPU CLI): `mlx_lm.generate --model <artifact-dir> --prompt
"The capital of France is" --max-tokens 16 --temp 0` — record raw IDs.
Note: mlx_lm.generate_step takes a 1-D [S] prompt tensor (model()
takes [B, S]) — use the CLI or handle shapes explicitly.

Stage H (HTTP): `python -m mlx_lm.server --model <artifact-dir>
--host 127.0.0.1 --port 8080`; send the same prompt via a chat
request at temperature 0; record the generated text and IDs. Text-
level comparison only (the server applies its own chat template and
sampling defaults must be pinned to temperature 0 / greedy).

Both stages: two runs each; receipts include host placeholder (public
repo — no hostnames), device identity (chip/driver per receipts
convention), kernel/driver versions, artifact digests, exact commands,
generated IDs, timing separated cold/warm.

## Qualification status semantics

Device PASS (frozen rule above) upgrades the catalog family entry from
unqualified to generation-qualified (recommended still requires the
serve/http split per catalog contract). Until then the artifact stays
unqualified and unrecommended everywhere.

## Device gap follow-up (added 2026-09-20 after first device run)

First device run outcome (t6001-test-host window, frozen protocol): DIVERGENCE —
not qualified. Measured CPU-side facts at the step-2 decision point
(forced prefix = prompt + [11751], identical token IDs both sides):

- CPU quantized top-8: 11 -> 20.0, 13 -> 18.5, 318 -> 16.375 ...
- CPU chooses 11; the device chose 13; **margin top1-vs-13 = 1.5
  logits**. A 1.5-logit gap is NOT a razor tie: the "near-tie noise"
  explanation is unproven until the device-side logits are captured.
- CPU backend jitter floor (batched vs single forward): maxabs 0.0,
  relL2 0.0 — the CPU backend is self-consistent; the flip must come
  from accumulated cross-backend numeric difference, which is
  unmeasured until the Vulkan half runs.
- Bounded teacher-forced fixture (23 forced positions): 17/23 top-1
  match, min margin 0.125. Caveat: the first 5 positions are PROMPT
  tokens, where top-1 != forced is expected (the prompt is not the
  model's own greedy continuation); per-continuation-position stats
  were not persisted and the fixture should be re-run with
  continuation-only aggregation next window.
- CPU full position-2 logits saved:
  step2_cpu_logits.safetensors (fp16, 248,320 values) in the artifact
  directory; the device half must forward the SAME forced prefix and
  report maxabs / relL2 against this vector plus its own top-8 and
  the values for IDs 11 and 13.

### Pre-declared numeric acceptance contract (DRAFT for owner approval — applies to future device tests only; the 2026-09-20 run stays FAIL)

To be fixed BEFORE the next device window, by the owner:

1. Identity gates (hard, no tolerance): prompt token IDs, tokenizer
   files, shard sha256s must match the frozen vector exactly.
2. Per-step logit comparison under identical forced prefixes: report
   maxabs and relL2 over the full vocabulary, per generated position.
3. Top-1 agreement rule (proposal): a generated position counts as
   agreeing when the reference top-1 is the device top-1 OR the
   reference margin (top1 - device_choice) is below a pre-declared
   tie band T. Proposal: T = 2.0 logits (the measured step-2 margin
   was 1.5; the band must be justified from accumulated-delta
   measurements, not chosen to pass).
4. Sequence acceptance (proposal): qualification requires every
   position either agreeing under rule 3 or covered by the recorded
   device-vs-reference maxabs/relL2 within bands fixed in 2 — with
   band values pre-declared before the run and never tuned after.
5. Any position failing both rules = DIVERGENCE; qualification is
   granted only by explicit owner decision on the recorded trace.

### Amendment (owner review, 2026-09-20, later same day)

- The T = 2.0 logits tie band is WITHDRAWN: deriving the criterion
  from the observed failure (step-2 margin 1.5) is empirical tailoring,
  not root cause. Acceptance bands stay UNSET until the Vulkan-half
  accumulated-delta measurements exist; they will then be proposed
  from those measurements and approved by the owner before any
  qualifying run.
- The device half of the diagnostic covers ALL 23 forced positions at
  the LOGIT level: prompt-position logits must agree CPU-vs-GPU at
  the numeric-delta level even though prompt-token next-prediction
  differs from the actual text metric (different metrics, same
  numeric-agreement requirement). Alignment: position i logits
  predict token i+1 (verified in the CPU trace).
- First-divergent-layer bisect: the CPU trace
  (step2_cpu_layertrace.safetensors: embed + 40 per-layer last-
  position hidden states + final_norm + logits, fp16, frozen prefix
  prompt + [11751]) is mirrored to the device; the device script
  forwards the same prefix, diffs each layer output against the CPU
  trace (maxabs + relL2), and reports the first layer exceeding the
  measurement threshold plus per-layer deltas — localizing wrong
  op / order / precision empirically.
- Source precision survey (both backends accumulate in fp32; weights
  fp16 scales + affine biases; activations stored bf16/fp16 and
  widened to float in-kernel / in-loop): the per-op numeric shapes
  match, so candidate divergence sources are (1) accumulation ORDER
  in the quantized matmuls (shader tiling vs CPU sequential loop),
  (2) cast-point placement for bf16 activations across composite ops,
  (3) GDN composite state-update ordering, (4) SDPA reduction order.
  The bisect decides empirically; no source claim is accepted without
  it.
