# B60 (BMG G21, ANV/Xe2) — Xe16 cooperative-matrix prefill route

DEC-19. Capability-gated dense `Matmul` + `QmmPrefillCoopmat16Xe` on the
ANV `M8xN16` cooperative-matrix family (`K16`, fp16/bf16 operands, f32
accumulate), replacing the `bf16_fma` / `QmmPrefillFmaBF16` fallbacks on
Intel Xe2 where the M16 and 8x8x8 shape lists are empty.

## Environment

- Source commits: `cc3756a39` (implementation), `45f9937a7` + `df89dbce2`
  (test coverage), `1c43b95ad` (review: gate capability + shared-memory
  bound), `ac4d2554e` (review: qmm Xe16 emits bf16 operands, matching
  `qmm_intel_bf16`'s contract; earlier cells ran the fp16-operand
  build). Wheels: base `0.32.4.dev…+diag.cc74e7a` (origin/main), prop
  `0.32.4.dev…+diag.45f9937`; post-review cells re-run on prop
  `+diag.ac4d255` where marked.
- Kernel: `7.1.12-2`. Mesa (ANV): 26.2.6.
- Device: `Intel(R) Arc(tm) Pro B60 Graphics (BMG G21)`, eGPU over USB4,
  driver `xe`, subgroupSize 32 (pinned via requiredSubgroupSize).
  `VK_KHR_cooperative_matrix` shapes advertised: M8xN16 K16
  (f16→f32, bf16→f32) and int8 K32 only — no M16, no 8x8x8.
  `VK_EXT_subgroup_size_control` present.
- Model: `mlx-community/Qwen3-4B-Instruct-2507-4bit` (q4 group-64).
- Host placeholder: a Thunderbolt-eGPU Linux workstation.

## Probes

### Route admission (live dispatch)

`MLX_OMARCHY_TRACE_DISPATCH=1` on the prop wheel:

- `mx.matmul` bf16 64×64 → `DISPATCH kernel=735`
  (`MatmulBF16CoopmatXe16`). Base wheel dispatches `kernel=26`
  (`MatmulBF16`, `matmul_bm32_bf16.comp` sequential f32 FMA).
- `quantized_matmul` q4-g64 bf16, ragged m=20 → `kernel=736`
  (`QmmPrefillCoopmat16XeBF16`).

### 5-pair ABBA + propoff (greedy, temp 0, 128-token decode)

Commands (per cell): `flock <host-egpu-lockfile> env CELLS=<cell>
nice -n 10 bash run-abba.sh` — interleaved A-B-B-A blocks, plus 3
prop-off runs (prop wheel + `MLX_OMARCHY_NO_MATMUL_XE16=1
MLX_OMARCHY_NO_QMM_XE16=1`).

- ctx512: base/prop/propoff each deterministic (1 unique ids sha256 per
  arm). propoff ids == base ids. base != prop ids (one early argmax
  flip, paraphrase-level; see below). dmesg reset-keyword count 8→8.
- ctx2048: **all arms identical** ids sha256
  `e683dade…f0` across 23 runs; propoff bit-matches base. dmesg 8→8.
- 0 GPU hangs across all runs; no engine resets.

### Timing (steady state, per run)

- ctx2048 prefill: base median ≈ 23.5 s, propoff ≈ 23.55 s, **prop
  ≈ 21.8 s (−7.3 %)**; prop runs are tight (21.65–21.92 s).
- ctx2048 decode: base ≈ 39.4 ms/token, prop ≈ 39.9 ms/token (matched
  within noise). ctx512 decode: both arms ≈ 45 ms/token steady state.
- Per-kernel (ctx2048 kernel table): `MatmulBF16 n=134217728`
  17.5 ms/35 dispatches → `MatmulBF16CoopmatXe16` 9.8 ms/35;
  `QmmPrefillFmaBF16 n=1740800` (propoff arm) →
  `QmmPrefillCoopmat16XeBF16` 6.5 ms/70 on prop. The FMA fallback lines
  appear only on the propoff arm, i.e. the Xe16 route is the only
  non-FMA path for those shapes.

### Per-op error vs fp32 reference (prop wheel, B60)

`xe16_error_ppl.py` — random uniform inputs, error vs fp32 matmul of the
same bf16 operands, bound = max|ref|·2⁻⁸:

```
[dense-xe16-bf16] m=64   n=4096 k=4096 err=0.2498 bound=0.3951 ok
[dense-xe16-bf16] m=512  n=2048 k=512  err=0.1138 bound=0.1429 ok
[dense-xe16-bf16] m=2048 n=512  k=2048 err=0.2238 bound=0.3031 ok
[dense-xe16-bf16] m=77   n=1536 k=256  err=0.0617 bound=0.0946 ok
[qmm-xe16-bf16]   m=77   n=2048 k=2048 err=0.000302 bound=0.000412 ok
```

Dispatch trace confirms 76 dispatches of kernel 735 + 1 of 736 on the
prop arm, 0 on base. The m=512 qmm probe takes the pre-existing
`QmmPrefillIntelBF16` route on both wheels (m % 16 == 0) and is
identical bitwise on both arms — its 1.07× bound overshoot belongs to
that route, not this change.

### Perplexity (fixed 2000-token text, Qwen3-4B-4bit)

base ppl **1.346772**, prop ppl **1.346908** → +0.010 % (gate: ≤0.1 %).

## Known divergence note

Greedy ids at ctx512 differ between arms (paraphrase-level split around
token ~14); at ctx2048 they are bit-identical. The base kernel
accumulates K in strictly sequential f32 FMA order while ANV's
cooperative-matrix MMA accumulates in a hardware-defined order inside
each K16 tile; bitwise equality is therefore not a stable property of
the route change and lands stochastically per prompt.

## Watch items

- During the doctest battery (run once, before queue submission) one
  unrelated test case (`small-row qmm matches per-row single-row
  dispatch`) hit the backend's 10 s Vulkan timeline watchdog while the
  host was under heavy foreign build load; the Xe16 dispatch-pin cases
  around it had completed cleanly. Not reproduced in the model ABBA
  (46 model runs, 0 hangs). Flagged for follow-up.
- `test_xe16_covers_ragged_m` and the dense dispatch pin require an
  ANV device; they skip elsewhere.

## Device recovery

No hang, reset, or device-loss event occurred in any run; the device
reopens cleanly on subsequent runs (all runs after the ABBA used the
same device fd path without error).
