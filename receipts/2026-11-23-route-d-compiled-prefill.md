# Route D — compiled prefill through `torch.compile` (M5)

Branch `w7Y/route-d-compiled-prefill`, on top of frozen `444518f61` (#754).
Host m1-host (M1, honeykrisp/Vulkan), `torch 2.15.0+cpu`,
`mlx 0.32.4.dev202610292155+42b30a2d`, `transformers 5.22.1`.
Prompt: `"The capital of France is"` tiled to 512 tokens; `NTOK=16` greedy,
`attn_implementation="sdpa"`, `StaticCache`.

## Qwen3-0.6B, fp32 — MATCH True, 16/16 CPU-identical ids

| phase | CPU eager | compiled cold | compiled warm |
|---|---|---|---|
| prefill T=512 | 3.761 s | 7.96 s (trace+mx.compile) | **1.299 s** |

Decode 10.57 tok/s at ctx-512 (vs ~22 tok/s at SMAX=64 — bigger cache mask
writes). Executors: prefill `fused_rms=113 state=28 nouts=85`, decode
`fused_rms=113 state=84 nouts=1`.

## Qwen3-4B-Instruct-2507, MV_QUANT=4, bf16

- Executors: prefill `fused_rms=145 state=36 quant=252 nouts=109`,
  decode `fused_rms=145 state=108 quant=252 nouts=1`.
- `prefill_cold_s=14.391`, `prefill_warm_s=3.986`, decode 6.43 tok/s @ ctx-512
  (clean re-run after the spontaneous m1-host reboot; first run measured
  15.017/4.175/6.44 under higher load).
- **Greedy ids verified against native `mlx-lm`** on the same
  `mlx-community/Qwen3-4B-Instruct-2507-4bit` checkpoint: 15/16 tokens
  identical; the sole first-token difference is `of`(1055) vs ` of`(315),
  a whitespace re-tokenization artifact of re-encoding the generated text —
  the decoded continuation is identical ("of France isThe capital of
  France is ...").
- CPU-eager bf16 prefill baseline did **not** finish within ~55 min under
  load (two attempts; killed to yield the shared GPU queue) — warm compiled
  prefill is therefore >700x the measured lower bound on eager TTFT.

## What it took (backend changes in `tools/torch-mlx-vulkan/mlx_vulkan.py`)

- **Structural mutation detection** (`_ph_of` over `index_copy_`/`add_`/
  `copy_`/`scatter_`/`fill_`/`index_put_` targets through `to`/`detach`/
  `clone`/`contiguous`/`float`/`half`) replaces `_cache_layers_` name
  sniffing — prefill and decode placeholder names differ.
- **Full output-spec preservation**: prefill graphs return logits + the 84
  mutated cache tensors inside the graph output tuple (`nouts=109`); decode
  appends them as tails (`nouts=1` + tail). `spec_pos`/`extra_mut` map both
  layouts back into `_MV_STATE`.
- **Shared state registries** `_MV_STATE`/`_MV_STATIC` keyed by placeholder
  name + seed-tensor identity: a separate compiled prefill executor writes
  into the same cache state the decode executor reads; a fresh `StaticCache`
  reseeds.
- `is_causal=True` + `attn_mask=None` -> `mask="causal"` for `mx.fast.sdpa`.
- **bf16 cache dtype preservation**: HF lazy-inits cache buffers as `zeros`
  inside the prefill trace; producers ignored the dtype kwarg and emitted
  fp32, so `mx.where` writes promoted K/V to fp32 and the next decode trace
  rejected them. `_mkdt` on `zeros/ones/empty/full` (+`_like`), and
  `index_copy_` casts updates to the target dtype.
- bf16-safe `_torch` (uint16 view; NumPy has no PEP-3118 bf16) and
  `MV_LOG` for a per-run lowering log (shared `/tmp` log was truncated by
  concurrent lanes).

## Caveats / operational hazards

- Near-tie flip at token 5 on the un-tiled prompt (fp32, `PLEN=0`): the
  0.0242-logit tie w7R bisected on B60 (PT-D5). Frozen `444518f61` picks
  15344, this branch picks 9625; a 16-token run continues fluently and
  correctly either way. Intrinsic tie instability under any pipeline
  perturbation, not a correctness regression — the PLEN=512 comparison above
  is the binding check (16/16).
- m1-host worktree was overwritten twice by queued lane jobs syncing the old
  backend (stale `IndexError` signature) and stale `__pycache__` bytecode —
  verify `grep -c nouts mlx_vulkan.py` > 0 and `rm -rf __pycache__` before
  every run. Bare `flock /tmp/m1-gpu.lock` runs also serialize against
  gpu-turn tickets, but they block queued tickets; prefer queue submission.
- `PHASE=mlx` without a CPU reference no longer prints a self-comparing
  `MATCH`; it writes `<REFJ>.mlx` and reports `NO CPU REFERENCE`.
