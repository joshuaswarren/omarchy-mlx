# 2026-10-05 PersistentMlp — MLX_OMARCHY_PERSISTENT_MLP lands: unit 2 integration + unit 3 M2 e2e A/B (numbers below)

Lane: PersistMlp (worker). Base: `origin/agent/GridBarrier` @ 94c01f669 (Main's
IRC directive, unit 1 = PersistentTailBF16 kernel land). Land tip: see git log
(agent/PersistMlp): unit 2 integration + per-stream gbb scratch fix.
Private notebook: `entries/PersistMlp/20261005T061500Z-jw14m2-persistent-mlp-tail-e2e.md`
(pre-registered 06:15Z, before any code change or hardware run). Artifacts:
`artifacts/PersistMlp/h294-m2-e2e-<stamp>/` (SHA256SUMS inside). Host: the
T6021 M2 (G14C), golden-build recipe (receipts/2026-10-05-golden-build),
wheel `mlx_omarchy-0.32.4.dev202610050436+<sha>` (sha256
a743d3596fe183323e7c5b95e4def0ad856f0220fbcf27e0f164d8fe9bfb1adf),
libmlx.so sha256 3f3df85f193aded13d95fbda36e96f41ed398274b9ec1c1ff2718c542746e7cb.

## What landed (unit 2)

- `fused_chain.cpp/.h`: the eager planner claims the decode tail pattern
  RmsNorm(x) -> one {gate,up} GemvGroup carrying the SwiGLU store fold ->
  one down GemvGroup whose Add epilogue adds x back, behind
  `MLX_OMARCHY_PERSISTENT_MLP` (default OFF; =0 kill switch). The norm node
  fires ONE `PersistentTailBF16` dispatch (fast_norm stage + qmm multi fold +
  qmm multi down+add, 2 in-kernel grid barriers, bounded 2^15 spins) at its
  eval; both gemv groups flip to done. Any refusal unwinds everything onto
  the shipped three-dispatch path with nothing allocated.
- `primitives.cpp`: `dispatch_persistent_mlp_tail` beside
  `dispatch_quantized_gemv_group`. Push-constant routing mirrors the
  generated persistent main: gu rides flags(bit 16)/matrix_k/shape[0..1];
  dn rides out_strides[0..2] = (n_dn, 1024 = bit 8+2, K_dn); norm rides
  reduce_size/output_size/alpha. Bindings 0..37 (blocks 0/1 = gate/up, block
  2 = down, 30/31/34 norm in/w/out, 35/36 x_mid/x_norm, 37 gbb_sync).
- G sizing: ONE probe per device, from the ACTUAL built pipeline —
  VK_KHR_pipeline_executable_properties register statistic -> HkTurnover
  formula G_fit = floor(cores * min(3072, floor(319488/gprs)) / 256),
  cross-checked with the measured-good ceilings (G13G 96, G14C 128; other
  chips formula-only). SKU-conservative core counts.
- Timeout flag: gbb_sync.word[2] read in a completion handler; any set flag
  logs loudly and permanently falls back (sticky) to the shipped path.
- Scratch (arrival/generation/timeout) is per-stream; zero-filled before
  every dispatch; never freed.

## Results (verified through 08:05Z, M2 wheel +e676874)

- Integration builds green (golden recipe, rc=0) and is INERT with the gate
  off: the OFF arm reproduces every 2B ledger pin (d64 76ec87cb… at
  106.97-108 tok/s class, d128 f6796407…, d512 c20359c7…) and the 4B d64
  ledger pin (42d27a8c…) — dispatch trace, digests, and rates unchanged
  from the shipped path, as the default-OFF contract requires.
- 2B d64 ON == the pin; 2B d128/d512 ON == OFF (see honest note below).
- NOT yet demonstrated: the route firing. The counter probe shows OFF==ON
  identical (4144 vk_compute_dispatches / 16 tokens, 0 fills) with no
  probe/refusal log line, so the ON arm took the shipped path: mlx_lm's
  swiglu() is mx.compile'd (Compiled node opaque to the planner's
  Multiply-chain scan) and patch-mlx-lm-swiglu-eager.py is not in
  apply-mlx-lm-patches.sh, so the SwiGLU store fold never plans and the
  tail's `!gu.swiglu_out` rung refuses. Root cause named without hardware
  (Main directive); fix staged: h294-build.sh applies the patcher to the
  venv (idempotent, bit-exact, pins unchanged) + planner refusal ladder
  behind MLX_OMARCHY_PERSISTENT_MLP_TRACE (commit 116fcae5c). 9B cells and
  the A/B are pending on the post-window rerun. The M2 parked at 08:11Z
  (Main); no tickets after the order; clones deleted per the parking
  directive.

## Verdict against the pre-registration

- Q1/Q3: NOT SETTLED — no route execution on hardware yet; no default-on
  decision is possible and none is requested.
- Q2 (probe): UNREACHED (the probe never runs while the planner refuses).
- Everything captured is in artifacts/PersistMlp/ (archive
  36c8e0c5f6ccfdafb584f0b248cdbf7e2bb31e9c6c2d589e36b8eab03e8c815a) and the
  notebook entry, including the negative results.
