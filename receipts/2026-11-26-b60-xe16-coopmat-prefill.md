# B60 (BMG G21, ANV/Xe2) — Xe16 cooperative-matrix prefill route

DEC-19. Capability-gated dense `Matmul` + `QmmPrefillCoopmat16Xe` on the
ANV `M8xN16` cooperative-matrix family (`K16`, fp16/bf16 operands, f32
accumulate), replacing the `bf16_fma` / `QmmPrefillFmaBF16` fallbacks on
Intel Xe2 where the M16 and 8x8x8 shape lists are empty.

## Environment

- Source commits: rebased onto `origin/main` (`037fd3d0a`); implementation
  and review fixes (`4ce54161d`..`b56a1e26b`) plus `ea5cbfe6e` (scalarize
  Xe16 coopmat accumulators and drain per subgroup to eliminate NIR
  scratch spills). Wheels: base `0.32.4.dev…+diag.037fd3d`, prop
  `0.32.4.dev…+diag.ea5cbfe`. Early cells (`r3`/`r4`/`r5`) ran the
  pre-rebase head `dd8e638` vs `cc74e7a`.
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

- ctx512 (fp16-operand head `45f9937`): base/prop/propoff each
  deterministic (1 unique ids sha256 per arm). propoff ids == base ids.
  base != prop ids (one early argmax flip, paraphrase-level; see below).
  dmesg reset-keyword count 8→8.
- ctx2048 (bf16-operand head `dd8e63866`, boot `a79ce46b`): **all arms
  identical** ids sha256 `e683dade…f0` across 23 runs (10 ABBA pairs +
  3 propoff); propoff bit-matches base. dmesg 8→8,
  `GPU-TIMEOUTS delta=0`, `FOREIGN-GPU-USED count=0`.
- 0 GPU hangs across all runs; no engine resets.

### Decode regression root cause & fix (`dd8e638` → `ea5cbfe`)

On the pre-rebase head (`dd8e638` vs `cc74e7a`), ctx2048 decode regressed
+8.7 % over 5 ABBA blocks while `propoff` matched base, GT clock stayed
pinned at 2400 MHz, a 3 s post-prefill idle did not recover, and decode
dispatched identical kernel counts (371 dispatches/token, 0 Xe16
dispatches). Per-dispatch profiling (indexed by `ComputeKernel` id `e`),
`INTEL_DEBUG=cs`, and `INTEL_DEBUG=bat` isolated the exact cause:

1. **Unrolled-array spills in `matmul_coopmat_xe16.comp`**: indexing
   `acc[2][2]` across barriers (`acc[rb % 2u][cb]`) prevented NIR from
   scalarizing the four `coopmat` accumulators, producing **56 NIR scratch
   ops, 32:24 spills:fills, 1,272 SIMD32 instructions, 22,960 bytes** (vs
   `MatmulBF16`: 326 insns, 0:0 spills, 5,760 bytes) and bumping
   `CFE_STATE.Scratch Space Buffer` from 2 to 3.
2. **ANV `anv_shader_heap` `Kernel Start Pointer` (KSP) shift on Xe2**:
   every Qwen3-4B decode pipeline is lazily compiled on decode token 1
   *after* prefill. ANV packs shader binaries contiguously at 64-byte
   alignment in `device->shader_heap` (`anv_pipeline_cache.c:144`).
   Replacing `MatmulBF16` (`0x1680` B) + `QmmPrefillFmaBF16` (`0x7680` B)
   with `MatmulBF16CoopmatXe16` (`0x59c0` B) +
   `QmmPrefillCoopmat16XeBF16` (`0x5fe0` B) shifted every decode shader's
   KSP by `+0x2ca0` (`mod 0x800 = +0x4a0`). Controlled checks proved this
   placement shift — not coopmat/DPAS execution, buffer VAs, submission
   count, or `CFE_STATE` — caused the entire decode delta:
   - Executing `MatmulBF16CoopmatXe16` (10× `32×2048×2048`) and
     `QmmPrefillCoopmat16XeBF16` *after* decode pipelines are compiled
     changed `SdpaDecodeSplitP2` by 0.0 % (`17.2 → 17.3 → 17.4 µs`).
   - Compiling two unrelated non-coopmat shaders (`ElementwiseI32` +
     `ElementwiseLiteF32`, zero Xe16 shaders) before Qwen3-4B prefill
     reproduced the exact per-kernel fingerprint (`SdpaP2` layer 0
     `22.8 → 61.7 µs` vs prop `63.8 µs`; `FastRmsNormApple` +22 %;
     `SdpaP1` +11 %; `BinarySimple` −28 %; `LogSumExpChunk` −19 %; net
     decode GPU +6.3 %).
   - Diffing the full 10,252-line `INTEL_DEBUG=bat` command stream on a
     warmed 2400 MHz micro-loop showed bit-for-bit identical
     `COMPUTE_WALKER`s, shader ISA, `STATE_BASE_ADDRESS`, `CFE_STATE`,
     and binding tables — differing only in KSP — while KSP mod `0x800`
     (2 KiB) deterministically swung `SdpaP2` (`17.6 ↔ 23.6 ↔ 40.9 µs`,
     `0.1 µs` run-to-run repeatability; two allocations 30 KiB apart with
     the same `mod 0x800 = 0x780` gave `23.7` vs `23.6 µs`).
3. **Fix (`ea5cbfe6e`, rebased onto `037fd3d0a`)**: scalarized
   `acc00..acc11` and `mat_a0/a1`/`mat_b0/b1` in
   `matmul_coopmat_xe16.comp` (and `acc0/acc1` in
   `qmm_coopmat16_xe.comp`) with a per-subgroup `drain_s[2 * 8 * 16]`
   (1 KiB) so both subgroups drain in parallel. Under `INTEL_DEBUG=cs`,
   `MatmulBF16CoopmatXe16` drops from **32:24 spills:fills / 56 NIR
   scratch ops** to **0:0 spills:fills / 0 NIR scratch ops** (top-down
   scheduled), cutting prefill GPU time further and moving decode shader
   KSPs out of the slow `dd8e638` slot. (Because the residual decode
   gain also rides ANV's 64-byte-packed `anv_shader_heap` KSP placement
   on Xe2, the durable algorithmic gain of this PR is the ~30 % prefill
   GPU speedup; decode shows no regression.)

### Timing — 5-pair ABBA on rebased head (`037fd3d` vs `ea5cbfe`, solo B60)

5 A-B-B-A blocks (10 `base`, 10 `prop`) + 3 `propoff` per cell,
Qwen3-4B-Instruct-2507-4bit, greedy (`temp=0`, `seed=0`), 128 decode
tokens, per-dispatch GPU profiling (`period_ns = 52.083332`). 46 runs
total; `dmesg` reset-keyword count unchanged (222→222), 0 hangs.

**ctx2048** (23/23 runs bit-identical greedy `ids_sha256`
`e683dade…afb0`, `base == prop == propoff`):

| metric | base med (n=10) | prop med (n=10) | propoff med (n=3) | 5-pair ABBA Δ (blocks) | median Δ |
|---|---|---|---|---|---|
| prefill GPU | 7.58 s | 5.00 s | 7.11 s | −20.70, −29.69, −29.67, −22.81, −34.65 % | **−29.67 %** |
| prefill wall | 29.42 s | 23.41 s | 23.74 s | −3.70, −7.90, −7.92, +0.59, −21.15 % | **−7.90 %** |
| decode wall (med) | 48.22 ms/tok | 45.56 ms/tok | 47.63 ms/tok | −10.38, −5.50, −5.56, −3.09, −5.39 % | **−5.50 %** |
| decode GPU (med) | 40.48 ms/tok | 37.67 ms/tok | 39.88 ms/tok | −11.11, −6.53, −6.61, −0.92, −6.87 % | **−6.61 %** |
| total wall (128 tok) | 38.06 s | 29.40 s | 29.96 s | −6.38, −7.36, −13.73, −0.82, −18.35 % | **−7.36 %** |

**ctx512** (`base == propoff` `c387642c…18de`; `prop` deterministic
`a473cb22…1b49` — known accumulation-order split documented below):

| metric | base med (n=10) | prop med (n=10) | propoff med (n=3) | 5-pair ABBA Δ (blocks) | median Δ |
|---|---|---|---|---|---|
| prefill GPU | 1.30 s | 1.19 s | 1.29 s | −6.34, −8.18, −7.77, −7.87, −7.88 % | **−7.87 %** |
| decode wall (med) | 81.49 ms/tok | 74.25 ms/tok | 81.65 ms/tok | −12.22, −9.00, −8.93, −9.28, −9.17 % | **−9.17 %** |
| decode GPU (med) | 73.63 ms/tok | 66.40 ms/tok | 73.87 ms/tok | −8.15, −9.99, −9.91, −10.28, −10.22 % | **−9.99 %** |
| total wall (128 tok) | 27.66 s | 26.54 s | 27.63 s | −4.99, −4.14, −3.92, −4.06, −4.03 % | **−4.06 %** |

Per-kernel prefill GPU time (`ctx2048`, `base-3` vs `prop-3` vs
`propoff-1`):

- `MatmulBF16` (140 dispatches): `2,469.0 ms` (`propoff`: `2,470.1 ms`) →
  `MatmulBF16CoopmatXe16` (140 dispatches): **`1,140.0 ms`** (**2.17×**,
  −53.8 %).
- `QmmPrefillFmaBF16` (247 dispatches): `1,852.0 ms` (`propoff`:
  `1,864.1 ms`) → `QmmPrefillCoopmat16XeBF16` (247 dispatches):
  **`1,061.1 ms`** (**1.75×**, −42.7 %).
- `QmmPrefillIntelBF16` (247 dispatches): `2,320.2 ms` vs `2,320.6 ms`
  (`propoff`: `2,319.7 ms`), unchanged.
- `tests/qmm_intel_parity.py` on `+diag.ea5cbfe`: 3/3 tests passed in
  0.807 s; dense matmul parity vs base FMA: `max_abs_diff = 0` on
  `(32,32,32)`, `(33,65,48)`, `(64,128,64)`, `(128,256,128)` and `0.125`
  on `(2048,128,2048)`.

### Per-op error vs fp32 reference (prop wheel `+diag.dd8e638`, B60)

`xe16_error_ppl.py` — random uniform inputs, error vs fp32 matmul of the
same bf16 operands:

```
[dense-xe16-bf16] m=64   n=4096 k=4096 err=0.2498 bound=0.3951 ok
[dense-xe16-bf16] m=512  n=2048 k=512  err=0.1138 bound=0.1429 ok
[dense-xe16-bf16] m=2048 n=512  k=2048 err=0.2238 bound=0.3031 ok
[dense-xe16-bf16] m=77   n=1536 k=256  err=0.0617 bound=0.0946 ok
[qmm-xe16-bf16]   m=77   n=2048 k=2048 err=0.000428 fixed-bound=0.000412
[qmm-xe16-bf16]   m=512  n=2048 k=2048 err=0.000477 fixed-bound=0.000444
```

The acceptance bound is the k-scaled bound the doctest uses for the
bf16-operand route — `(3k+1)·M·2⁻²³ + 2k·M·2⁻⁸ + M·2⁻⁸`, `M =
max(2·max|got|, 1)` — not the probe's `max|ref|·2⁻⁸`, which ignores
reduction depth. At k=2048 the k-scaled bound is ≈16.0; both qmm
errors sit ~3.5×10⁴× inside it. For reference the fixed bound misses
are 1.04× (m=77) and 1.07× (m=512), and the shipped bf16-operand
`QmmPrefillIntelBF16` route also exceeds the fixed bound 1.07× on its
own shapes at the same k=2048 — the coopmat route is strictly more
accurate than the shipped bf16-operand path it replaces.

Dispatch trace confirms 76 dispatches of kernel 735 + 1 of 736 on the
prop arm (the m=77 qmm probe), 0 on base (26 x72 + 657 x1). The m=512
qmm probe takes the pre-existing `QmmPrefillIntelBF16` route on both
wheels (m % 16 == 0) and is identical bitwise on both arms — its fixed-
bound overshoot belongs to that route, not this change.

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

- The `== tok/s ==` tail of `run-abba.sh` greps `out-ctx512-*-1.txt`
  unconditionally; under `set -e` a ctx2048-only cell exits rc1 after
  all run cells complete (the r3 wrapper logged `lane-failure
  kmsg-clean` for this reason). The `== ids gates ==` pool also globs
  both cells' out files, so the pooled counts mix ctx512-r1 hashes with
  ctx2048 — the per-run `ids_sha256` lines are the cell-scoped record.
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
