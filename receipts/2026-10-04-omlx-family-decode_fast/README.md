# decode_fast family on omarchy (A25d) — hd-256 tiled O(L) decode SDPA

Status: IMPLEMENTED + DEV-PARITY PROVEN at every shape tried; M2 wheel +
ledger pending. Owner lane: FamDecodeFast.

## Family surface (oMLX 4d4f5a2 = v0.7.0)

`omlx/custom_kernels/decode_fast` = one native op: `sdpa_decode` (decode-mode
SDPA, q_len <= 8, hd {64,96,128,256} + 192->128), 1-pass, plus a
context-scaled 2-pass split-KV with fp32 partials on 'd'-class GPUs
(`sdpa_decode.cpp`: 2pass at k>=1024 GQA; blocks = ~256-key chunks clamped
32..256). oMLX-side companions: `memory_monitor.py` SDPA head-dim coverage
sets + unfused-score accounting, `patches/sdpa256_attention.py` (forces the
bounded route for hd-256 prefill at k>=8192; q_len<16 exempt). Those two are
dtype/shape bookkeeping in Python and run unmodified on any backend; the
native extension is the Metal-only part.

## omarchy coverage finding (this lane's "determine" deliverable)

Origin/main (14bf031a1) already fuses hd-256 bf16 decode (q_len==1) as the
composition-exact one-pass arm `SdpaDecodeNativeBF16Hd256`, engaged inside
the measured window k in [12, 7168] (`kDecodeBf16Windows`). Outside the
window (and at q_len 2..8) hd-256 rides the ~10-dispatch f32-score
composition (f32 KV upcast + full score/probs tensors per layer call).
There was NO split-KV two-pass at any bf16 width (the f16 pair covers hd
64/128 only, k>1024) — that is the missing tiled O(L) route for hd-256.

Memory behavior: at q_len==1 every route here is O(L) time and O(1)-ish
transient per call (composition materializes 2 f32 [b,h,1,k] tensors plus a
full f32 KV copy; the fused arms allocate no score matrix). The O(L^2)
memory spike the omlx patch guards is a PREFILL (q_len>=16) phenomenon and
stays with the composition on both backends.

## Implementation (commits 2437b4f91, 2e3dea274 on agent/famdecodefast)

- `sdpa_decode_native_p1.comp` + `_p2.comp` gain `-DBF16_IO` legs: pass 1
  grids (heads x blocks, flattened into x) with the one-pass arm's phase
  structure at block scale — serial full-width dots into a shared score
  stream (258 entries max), the same strided max/sum trees, one thread per
  output dim — and stores `[max, sum, o[256]]` fp32 per (head, block); pass
  2 folds blocks in ascending order and narrows once through the cast.comp
  RNE bf16 store. Block policy = decode_fast's 'd' formula, grown so every
  chunk fits the stream. No subgroup operations (lavapipe reports
  subgroup_size=8; the f16 pair's subgroup_add structure is unusable there
  and a per-key cross-thread reduce was miscompiled on llvmpipe).
- Dispatch: engages past the one-pass window (k > 7168) on hd-256 when the
  device meets a 3080-byte shared + 1024-thread workgroup check. No env
  gate: the route engages by default past the one-pass window.
- Enum entries appended: `SdpaDecodeNativeTwoPassP1/P2BF16Hd256`; two
  embedded blobs at SDPA_DIM=256; CMake entries beside the f16 pair.
- Parity test: `omarchy_sdpa_decode_fused_tests` "hd256 bf16 decode rides
  the two-pass split-KV past the window" — engagement (1 vs 2 dispatches)
  + composition agreement.

## Numerics evidence (dev box, lavapipe software driver, OMARCHY_MLX_SYSTEM_PREFIX
staged ICD + MLX_OMARCHY_ALLOW_NON_APPLE=1; exact commands in the notebook
artifacts):

- One-pass hd256 vs f32-score composition: BIT-IDENTICAL (4096/4096 bf16
  words) at k 13/128/512/2048 — route unchanged from main, now proven.
- Two-pass vs composition: max abs diff 3.0e-5 / 4092 of 4096 bf16 words
  identical at k=7169 (the boundary: first key past the window, short
  last blocks); 7.6e-6 / 4094 at k=8192; 7.6e-6 / 4091 at k=16384;
  3.0e-5 at k=40960; 3.8e-5 at k=65537. Versus an f64 host reference the
  composition itself matches to <=1e-4 and the one-pass to 5.8e-5.
- Shader toolchain: all six p1/p2 variants compile under glslangValidator
  12.0.0 and the glslc-style toolchain (glslang 16.6) on the dev box.

## Debugging record (kept short, evidence in notebook artifacts)

A boundary-shape NaN (k=7169) was isolated by dumping pass-1 scratch cells
through a debug pass 2: cells for blocks >= 1 held garbage because pass 1
sized EVERY block's key list by the global chunk (ceil(k/blocks)) instead
of its own (ceil((k - block)/blocks)), reading past the K cache on short
last blocks. Per-block chunk sizing fixed it; the parity grid now includes
the boundary shape and the 65537 chunk-fit shape.

## Remaining for A25d DONE

1. Root-cause the boundary NaN, drop the env gate, extend the parity grid
   to the boundary.
2. M2 (T6021/G13C) wheel build + op-level parity spot + the decode ledger:
   tok/s + peak transient at kv 64/512/4096/8192, family ON vs OFF
   (MLX_OMARCHY_SDPA_DECODE_NATIVE=0), Qwen3.5-9B-MLX-4bit bench_decode
   per-context; then the window-table re-measure this data implies
   (the {256,12,7168} upper bound is exactly what the two-pass replaces).
3. Teacher-forced top-1 / free-run gap / PPL numerics-gate per
   docs/numerics-gate.md for the route as deployed.

## Provenance

- Repo: joshuaswarren/omarchy-mlx, branch agent/famdecodefast
  (2437b4f91 -> 2e3dea274), base origin/main 14bf031a1.
- oMLX reference read-only: ~/src/omlx @ 4d4f5a280bc1739ba2cf39c1cee44fd5cc89cb40.
- Dev box: x86 PVE guest, kernel 6.17.2-1-pve, glslang 12.0.0 + glslang
  16.6 (glslc path), lavapipe ICD via staged prefix.
- Notebook: entries/FamDecodeFast/20261004T224400Z-omp-studio-local-omarchy-decode-fast-hd256.md,
  artifacts/FamDecodeFast/decode-fast-hd256/.
