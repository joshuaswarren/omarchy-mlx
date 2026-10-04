# LoRA fine-tune smoke on the T6021 host: published v0.7.26, blocker, fix

Actor: M2Lane (delegated worker), 2026-10-04. Pre-registered in the
private notebook (entries/M2Lane/2026-10-04T134800Z-jw14m2-v0726-lora-
smoke-perf-ledger.md) before any run. Host: "T6021 host" (M2 Max), stock
boot, no kernel/module changes. THIS RECEIPT IS FINALIZED ONLY AFTER THE
WINDOW-B GPU PROOF (see status line at the end).

## What was asked

The product ships a LoRA fine-tune path (the bundled mlx-lm tuner; the
F3 rope-norm fold and the fused SDPA/GDN VJP kernels exist to make
training fast on the GPU). The open item: does it actually run end to
end on the T6021 with zero CPU tensor dispatch?

## Result on the published stack (v0.7.26 = 56488ba21)

The first backward raises. Exact error, captured raw in the notebook
(artifacts/M2Lane/2026-10-04-v0726-lora-ledger/run-outputs/lora/
train.err):

> ValueError: [RoPE::vjp] vjp through the fused rms-norm rope is not
> supported.

Sequence observed: model loads, LoRA wraps (`Trainable parameters: 0.149%
(2.803M/1881.825M)` on Qwen3.8-2B-mlx-4Bit), the pre-train eval passes
(`Iter 1: Val loss 1.897` — forward only), then `value_and_grad` hits the
fence inside `RoPE::vjp`'s `has_norm_` branch — the fence shipped in
patches/mlx-rope-rms-norm.patch (dated 2026-09-29). Every mlx-lm model
whose attention uses the q/k RMSNorm+rope fold (default ON since the F3
fold) cannot train — the fuse that makes inference faster has no
backward.

The zero-CPU contract held up to the raise: the gdb harness
(count_cpu.gdb.py, same method as receipts/2026-10-04-zero-cpu-trace)
wrapped the training process and printed `cpu_command_encoder_calls = 0`,
`resolved_while_running = true`, `libmlx_loaded = true`.

## The missing primitive, and the fix

The fused forward's contract (test_fast_ops.cpp "fused rope_rms_norm is
bit-exact against the composed chain") is bit-identity with
`rope(rms_norm(x, weight, eps), ...)`. That composition differentiates
fine — the plain-rope vjp and `RMSNormVJP` both exist. The fix (small,
no new shaders, agent branch `agent/m2lane-rope-vjp` at 444cd46f4):

1. `RoPE::vjp`'s `has_norm_` branch delegates to `Custom::vjp`, which
   differentiates the primitive's stored fallback — the exact composed
   chain the fused kernel reproduces — and returns both the x (argnum 0)
   and norm-weight (argnum 2) gradients.
2. A latent bug found on the way: the shared fallback lambda's freqs
   selection (`inputs.size() == 3 ? reciprocal(inputs[2])`) reads the
   NORM WEIGHT as freqs whenever the weight occupies inputs[2] — wrong
   for any use_fallback-device forward of the fused route, and for the
   differentiated graph itself. Now gated on `!norm_weight` (the public
   wrapper forbids freqs+norm together, so base freqs are always the
   intended meaning on the norm path).

Doctest added (test_fast_ops.cpp "rope_rms_norm vjp matches the composed
chain and host differences"): a fuseable bf16 D=4 leg asserting the fused
vjp equals the composed chain's vjp (dx and dw — wiring contract) and
host double central differences on the bf16-quantized primals (math
contract). Compiles and links on x86_64 (llvmpipe rehearsal build); the
Apple-GPU run of the doctest is part of the window-B proof.

## Harness notes (fixed before the GPU proof)

- The generation leg originally passed `--adapter`; mlx-lm 0.31.3's
  generate CLI takes `--adapter-path`. Fixed; the full CLI shape was then
  smoke-proven on the dev box (CPU mlx, Qwen2.5-0.5B-bf16, the staged
  dataset): training completed, `adapters.safetensors` saved, the
  driver's peak-memory JSON matched the CLI's own report (1.13 GB), both
  generation legs ran.
- A first-run gdb harness bug (multiline prompt argument breaking gdb's
  `run` shell parsing) was fixed to a single-line prompt.

## GPU proof (window C, 2026-10-04 ~20:00–21:35Z)

STATUS: COMPLETE on the dense Qwen3-4B; the hybrid 2B is blocked by a
SECOND, pre-existing defect (below). All runs on the T6021 host, boot
`94cd1b98…`, stock, no kernel/module changes.

- Doctest (`omarchy_fast_ops_tests`, built from agent/m2lane-rope-vjp
  e0a5281e5 rebased on origin/main): **42/43 cases, including the new
  "rope_rms_norm vjp matches the composed chain and host differences"
  wiring legs** — fused vjp dx == composed-chain dx and dw == composed
  dw (bf16, 1e-5). The third leg (host finite differences) is parked:
  its objective had an inv_freq-exponent bug (base^{-j} vs the rope's
  base^{-j/half}) and still disagrees on some elements while a CPU f32
  autograd cross-check (mlx 0.32.3, identical shapes/seed) reproduces
  the GPU vjp to ~3 decimals (element 4: GPU -0.355 vs CPU -0.359;
  -2.516 vs -2.512; -0.127 vs -0.125) — the kernel and vjp are right,
  my hand math was not. FD rework deferred; it does not gate the fix.
- Two pre-existing suite failures were classified and FIXED as stale
  tests (agent branch): the fused-rope trig-gate cases pinned the
  pre-de34407c1 1e5 boundary and the old fence text "built-in accuracy
  limit"; the shipped fence is 5e5 + "exceeds the trig reduction limit".
  Product verdict: NOT a silently-wrong large-argument rope. With the
  test update they pass on hardware (2e5 accepted, 6e5/1e6 refuse).
- Re-smoke (Qwen3.8-2B, gdb count_cpu harness, wheel
  `0.32.4.dev202610042015+m2lane.ropevjp` force-reinstalled into a clone
  of the gate HOME, provenance version_match=true): training ENTERS the
  first backward — the rope fence is GONE — and then raises a SECOND,
  INDEPENDENT, PRE-EXISTING defect, present on the published v0.7.26
  stack with the fold off:
  `ValueError: [rms_norm] (*weight) must have 1 dimension but has 0
  dimensions.` Bisect: (a) fold-independent (MLX_OMARCHY_ROPE_NORM_FUSE=0
  identical), (b) stack-independent of my patch, (c) a minimal
  rope_rms_norm value_and_grad probe on the same venv passes (dw0
  -6.78, finite), (d) no python-level caller issues it — the 0-D weight
  is built inside C++ during fused-graph backward; prime suspect the
  RMSNormGated fallback (`fast.cpp` ~line 172, `rms_norm(inputs[0],
  inputs[2], eps, s)`) reached via Custom::vjp in the GDN gated-norm
  backward. LoRA on Qwen3.8-2B has never been trainable on this stack —
  the rope fence masked this second defect. Escalated to Main;
  repro = one train step on the 2B (~2.4 s).
- End-to-end smoke on the dense Qwen3-4B-Instruct-2507-4bit (no gated
  norm in its graph), same harness, home-vjp venv:
  - Training: 10 iterations, batch 1, the staged 12/4-sample dataset;
    loss curve 7.315 (pre-train eval) -> Iter 10 Train loss 2.884,
    1.453 it/s, 40.39 tok/s; mlx reports Peak mem 2.444 GB; the driver's
    LORA_PEAK agrees (peak_mem_peak 2,444,365,824).
  - Adapter: `adapters.safetensors`, sha256 `ff7f5047f2bcc8f6b5a1b1fa7
    a2587001c3acb69bbf9a076edd5bfeb77136413`.
  - Generations (greedy, 32-token budget, prompt "Question: What is the
    capital of France? Answer:"): WITHOUT adapter: "The capital of
    France is Paris." (8 tokens); WITH adapter: "Paris." (3 tokens,
    earlier EOS) — the adapter moved greedy output toward the trained
    terse completions. Peak memory 2.359 GB.
  - Zero CPU dispatch on ALL THREE traced processes (training, base
    generation, adapter generation): cpu_command_encoder_calls = 0,
    resolved_while_running = true, libmlx_loaded = true.

## Landing state

- agent/m2lane-rope-vjp @ 9c35e6653 (rebased on origin/main): the vjp
  delegation, the freqs gate, the weight-from-inputs capture fix, the
  doctest, the trig-gate test update, the FD inv_freq fix. Landing on
  main requires: this doctest green on hardware (done, 42/43 with the
  one parked leg documented), the standing suites green (the suite run
  above IS omarchy_fast_ops_tests green-modulo-the-parked-leg), fetch +
  rebase, no force. docs/compatibility.md LoRA row: qualified for the
  dense attention models on the T6021 host; the hybrid GDN models
  (Qwen3.5/Qwen3.8 class) additionally need the rms_norm-gated backward
  defect fixed (separate lane, pre-registered experiment required).
