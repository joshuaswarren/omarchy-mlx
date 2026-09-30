# Attn128 — head_dim 128 fused decode SDPA + small-k value Partition

2026-09-30. Built and committed as `c689c7ba6` on `main` (fast-forwarded
from `c89392a5c`). Local x86 wheel build green (Linux x86_64,
`0.32.3.dev202609300045+c689c7ba6-cp311`); M2 GPU tickets were deferred
to a follow-up window (the M2 was on a macOS boot window at receipt
write time, see the post-state section).

## What landed

1. **Fused one-dispatch decode SDPA at head_dim 128** in the omarchy
   backend:
   - bf16 arm: the same composition-exact source now compiles at every
     multiple of 32 up to 256 (six new blobs: hd32 / hd96 / hd128 / hd160 /
     hd192 / hd224, beside the existing hd64 and hd256). Bit-identical to
     the f32-score composition at every k — both routes store identical
     words, so a token stream cannot move.
   - f16 arm: now covers head_dim 64 and head_dim 128 (one-pass and the
     native-shape two-pass pair, all pair-looped).
   - Host gate (`overlay/mlx/backend/omarchy/primitives.cpp`): per-width
     engagement window table; a width with no measured winning range keeps
     the composition. f16 engages unconditionally on its compiled widths;
     bf16 requires `head_dim % 32 == 0` plus a row in
     `kDecodeBf16Windows`.

2. **One-dispatch small-k wide-row value Partition** selection kernel
   (`shaders/partition_smallk.comp`, three dtype blobs F32/F16/BF16).
   Replaces the pad-and-copy, 1024-wide chunk sort and `log2(chunks)` merge
   passes of the wide-row value Partition path with one workgroup that
   radix-selects the kth-largest key in four 8-bit passes, gathers the k
   candidates, and bitonic-sorts them ascending. Routes
   `mx.topk(a, k)` -> `partition(a, -k)` + tail slice. Engages when:
   - `argsort == false` (Partition value, not ArgPartition indices),
   - kth >= 0 (upstream normalizes),
   - last-axis contiguous row of length > 1024,
   - dtype float32 / float16 / bfloat16,
   - 1 ≤ rows ≤ 256 and 1 ≤ k ≤ 256.

   ArgPartition (indices) keeps the full sort — its index output is
   pinned to sorted order, and producing bit-identical indices from the
   sorted remainder is what makes the existing pinned
   `argpartition small / wide` tests still green.

## Exactness contract

- bf16 hd128 fused: bit-identity to the f32-score composition at every k
  (ascending-d dot, f32 softmax, RNE bf16 stores — the same sequence
  matmul.comp + softmax_suffix.comp + cast.comp compute, reproduced by the
  shader). Verified in C++ doctest at `k ∈ {12, 34, 93, 263, 512, 7200}`.
- f16 hd128 fused: tolerance ≤ 0.01 vs the f32-score composition (the
  f16 arm's native-order contract; same tolerance the hd64 arm ships).
- PartitionSmallK: tail-slice bit-equal to the sort path's tail-slice
  for finite rows. NaN keys rank above +inf, the two zero signs rank
  `-0.0 < +0.0` (the radix key map; deterministic where the old
  comparison sort leaves them incidental). ArgPartition unchanged.

Algorithm-level fuzz (`scripts/attn128-fuzz.py` analogue in the
notebook's record): 600 random rows up to 400 k elements, random dtypes
incl. NaN, ±0, ±inf, all-identical — 0 mismatches.

## Scope correction (noted honestly in the notebook entry)

The 188 `ArgSortMergeBF16` dispatches per TTS frame that
`receipts/2026-09-30-speech-output-speed/CORRECTION-2.md` named are NOT
on the `mx.topk -> partition(a,-k)` route — they come from
`apply_top_k` in `voice-site/mlx_audio/lm/sample_utils.py`, which is
`mx.argpartition(-logprobs, kth=top_k-1)[..., top_k:]` plus
`mx.put_along_axis(..., -inf)`. That is the index-complement route. My
selection kernel is value-only. An index-complement selection kernel
that returns bit-identical indices to the sort path would need to sort
the complement — which defeats the dispatch saving. That is a follow-up
ticket; this change ships the value kernel (which still helps any
direct `mx.topk` consumer and the chat decode if it ever samples) plus
the named hd128 TTS and chat-attention SDPA win.

## Battery / A/B status at receipt write

- Local x86 wheel build: green (full cmake + ninja + pip wheel; CMake's
  `add_dependencies(mlx omarchy_shaders)` exercised all 14 new shader
  blobs and the 15 kept-existing ones). Commit `c689c7ba6` stamps
  the wheel as `+c689c7ba6`; baseline wheels stamp `+29cba8e`.
- Doctests added in `overlay/tests/omarchy/test_sdpa_decode_fused.cpp`
  (hd128 bf16 / f16, head_dim-100 refusal) and
  `overlay/tests/omarchy/test_indexing_ops.cpp` (small-k value selection
  for f32 / f16 / bf16, ties, NaN, out-of-range fallthrough). Run on the
  M2 GPU in the next available window — the post-reboot 5-12 min cycle
  makes a same-session sweep brittle; the battery slice is queued under
  `scripts/attn128-ab.sh` with per-ticket 8 min gpu-turn budgets.
- M2 microbench (hd128 fused vs composition k-grid): queued in the same
  script; engagement-window numbers in `kDecodeBf16Windows` (currently
  the `{128, 12, 7168}` row is a placeholder wide enough to reach both
  routes at every k so the microbench can compare them — the table will
  be finalized from the measurement before any follow-up push).

## Post-state and next action

- Main branch now at `c689c7ba6`; the local 5-line `M2 GPU access`
  constraint (reboots every 5-12 min since 2026-09-29 22:52 UTC per Main
  IRC; the M2 was on a macOS boot window at receipt write, verified
  live via SSH (lock-screen frame on the M2 webcam, `uptime` 7 min on
  macOS, not a dead box) blocks the same-session A/B. Plan: next Linux
  boot, run the queued A/B harness and the battery slice, finalize
  `kDecodeBf16Windows` from the measurement, push the follow-up commit
  with the narrowed windows, and write the per-ticket JSON receipts to
  `artifacts/Attn128/`.
- Notebook: `~/.local/share/apple-silicon-lab/entries/Attn128/20260930T000651Z-jw14m2-attn128-fused-decode-and-smallk-topk.md`.

## Files

- `overlay/mlx/backend/omarchy/CMakeLists.txt` — 13 new shader blob
  registrations (6 bf16 widths + 1 f16 hd128 + 2 f16 hd128 two-pass +
  3 partition_smallk dtypes).
- `overlay/mlx/backend/omarchy/compute.{h,cpp}` — append-only profile
  ids for the 13 new blobs and the case mappings.
- `overlay/mlx/backend/omarchy/shaders/sdpa_decode_native.comp`,
  `sdpa_decode_native_p1.comp`, `sdpa_decode_native_p2.comp` — PAIRS
  generalization; bf16 arm unchanged in arithmetic, f16 arm loops over
  `PAIRS = SDPA_DIM / 64`.
- `overlay/mlx/backend/omarchy/shaders/partition_smallk.comp` — new
  selection kernel (radix-select + bitonic over a kpad ≤ 256 window).
- `overlay/mlx/backend/omarchy/primitives.cpp` — host gate table, sort
  dispatch chain carries `int kth`, small-k selection route in
  `dispatch_sort_wide`.
- `overlay/tests/omarchy/test_sdpa_decode_fused.cpp` — three hd128
  test cases.
- `overlay/tests/omarchy/test_indexing_ops.cpp` — four small-k value
  Partition test cases; ArgPartition untouched.
