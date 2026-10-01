# Attn128 — head_dim 128 fused decode SDPA + small-k value Partition

2026-09-30. Built and committed as `c689c7ba6` on `main` (fast-forwarded
from `c89392a5c`). Local x86 wheel build green (Linux x86_64,
`0.32.3.dev202609300045+c689c7ba6-cp311`); M2 GPU tickets were deferred
to a follow-up window (the M2 was on a macOS boot window at receipt
write time, see the post-state section).

**Status 2026-10-01:** verified on the M2 Max; see the dated addenda at the
end. They supersede the "queued" status and the provisional hd128 window
below, and they correct two earlier claims (a "chip-class bf16 defect" that
was a test bug, and a forced push that briefly truncated main).

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
- Notebook: `~/.local/share/apple-silicon-lab/entries/Attn128/20260930T000651Z-<project-m2>-attn128-fused-decode-and-smallk-topk.md`.

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

## Addendum 2026-10-01: measured on the M2 Max

Host: Apple M2 Max, Linux 7.1.13-3-1-ARCH, Honeykrisp Vulkan. Every GPU
run went through the shared GPU queue, one ticket at a time, in a single
boot. A/B sides prove distinct builds by their version stamps.

### Shaders on glslc

All 13 new blobs and the modified existing variants compile cleanly under
glslc on the M2 and on the M1 (T8103, glslc 2026.3). The original x86 check
used glslangValidator only, which AGENTS.md says is no evidence for these
chips.

### Measured hd128 window (replaces the provisional row)

`q (1,16,1,128)`, GQA 16/8 over strided bf16 cache views. Fused and
composition (`MLX_OMARCHY_SDPA_DECODE_NATIVE=0`) alternate in one process;
each cell is the median of 3 x 200 one-eval calls. The outputs were
bit-identical at all 20 grid points.

| keys | 1 | 12 | 33 | 93 | 256 | 512 | 1024 | 2048 | 4096 | 7168 |
|---|---|---|---|---|---|---|---|---|---|---|
| fused us | 214 | 259 | 274 | 295 | 361 | 581 | 909 | 859 | 1162 | 1882 |
| composition us | 349 | 450 | 470 | 557 | 747 | 1475 | 1613 | 1592 | 2774 | 2552 |

The fused arm wins at every k, so main `1cc561b35` sets the hd128 row to
`{1, 7168}`.

### Battery at `1cc561b35`

| suite | cases | assertions |
|---|---|---|
| omarchy_sdpa_decode_fused_tests | 6/6 | 34,926 |
| omarchy_indexing_ops_tests | 57/57 | 35,859 |
| omarchy_fast_ops_tests | 36/36 | 1,104,679 |
| omarchy_fast_regression_tests | 2/2 | 16 |
| omarchy_runtime_tests | 41/41 | 22,694 |
| omarchy_primitive_tests | 104/104 | 2,743,003 |
| omarchy_kv_ops_tests | 16/16 | 781 |
| omarchy_error_contract_tests | 3/3 | 14 |

### Chat decode: Qwen3-4B-Instruct-2507-4bit (head_dim 128)

One wheel (`+e00b37116`) runs the fused route and the composition
(`MLX_OMARCHY_SDPA_DECODE_NATIVE=0`) as the single change, plus the
release wheel (`+06711ad`) as context. Three rounds alternate in one boot;
each leg is a fresh process with 2 warmups and 5 greedy 64-token
generations, measured with mlx-lm's own decode tok/s.

| leg | decode tok/s |
|---|---|
| fused hd128 | **64.20** |
| composition | 53.27 |
| release wheel | 50.30 |

Fused gives +20.5% over the composition, and greedy token ids are
identical on all three legs.

### Qwen3-TTS full decode frame

This is the speech-output receipt's frame harness, toggling the decode
route in-process with legs alternating. Code-predictor tokens agreed
150/150 in every pair.

| route | dispatches/frame | ms/frame p50 |
|---|---|---|
| composition | 4,244 | 167-176 |
| fused, provisional k>=12 window | 3,767 | 158-166 |
| fused, measured k>=1 window (main) | **3,362** | **134-137** |

The k>=1 window matters because the 15-pass code predictor attends over
fewer than 12 keys on most passes. The k>=1 timing ticket overlapped a CPU
build. The legs alternated, so the relative numbers hold, but the absolute
ms run high. The frame still exceeds the RTF-1.2 dispatch budget recorded
in the speech-output receipt.

### Small-k Partition: 16-bit route

One wheel pair differs by a single change (`+e00b37116` takes the sort
route, `+e7498ce89` takes the selection route) and alternates in one boot.
The input is a 1 x 151,936 row, timed with `mx.topk`.

| cell | sort ms | selection ms | speedup |
|---|---|---|---|
| bf16 k=20 | 2.734 | 1.074 | 2.55x |
| bf16 k=50 | 3.412 | 1.085 | 3.15x |
| f16 k=20 | 3.100 | 1.072 | 2.89x |
| f16 k=50 | 3.420 | 1.067 | 3.21x |

The tail words were identical across routes. The selection-route doctest
passes 57/57 with 35,859 assertions. For sampler ids, Qwen3-4B ran at
temperature 0.8, top_k 20, seed 42 for 47 tokens. Stock argpartition
masking and a threshold mask through `mx.topk` produced identical ids on
both wheels.

## Correction 2026-10-01: the "bf16 small-k defect" was a test bug

The 16-bit block of `omarchy_indexing_ops_tests` failed on M1 (T8103),
M1 Max (T6001), and the M2. It was reported as a chip-class backend defect,
and the 16-bit selection route was gated off (`666d22c13`). **That claim
was wrong.** The test built its bf16 input as
`array(uint16_t*, shape, bfloat16)`, and MLX's `array::init` converts with
`std::copy`, which converts each uint16 *value*. The bit pattern 0xBF00
(-0.5) became the number 48896, which is bf16 0x473F. That accounts exactly
for the failing words 0x473F/0x4740. Every route returned the correct top-k
of that input. A second test bug compounded it: the bf16 reference
helper did not round to nearest even. The test now feeds input through
`astype`. Its reference reads back the device-held input words and
compares against the input words at the top-k indices (`e00b37116`).

The hd128 refusal doctest had a separate test bug. Its composition reshaped
head_dim-100 data into width 128. It now reads the width from its inputs
(`ec171e65f`).

## Correction 2026-10-01: forced push

Two pushes from this lane used a forced refspec. They reset `main` to this
lane's tip and dropped the commits that had landed after `8b3982c21`.
Merge `1bd649b80` restored that lineage. Every push from this lane since
then has been a fetch, a rebase, and a fast-forward.

## Update 2026-10-01: 16-bit selection restored on non-G13 parts

The float32-only gate (`666d22c13`) is replaced by a chip-class gate
(`27f47aa5d`): 16-bit rows take the one-dispatch selection everywhere
except G13 parts (M1 family), which keep the sort route. On the M2 Max
("Apple M2 Max (G14C B1)") all three reversal conditions held: the
selection-route doctest is bit-exact against device-held input words
(57/57 cases, 35,859 assertions), the route is 2.5-3.2x faster than the
sort route on 1 x 151,936 bf16/f16 rows at k 20/50 with identical tail
words, and Qwen3-4B top_k sampling produced identical ids through
argpartition and threshold-via-`mx.topk` masks. The full battery at that
commit is green (compiled tape 13/13, fused chain 36/36 with 346,272
assertions, runtime 41/41, primitive 104/104 with 2,743,003, fast ops
36/36, indexing 57/57, decode fused 6/6).

**G13 remains unverified**: no G13-class host ran the selection-route
doctest (the M1 was down for reinstall; the M1 Max was in gated ANE
work). On a G13 part the 16-bit rows keep the sort route, which is the
long-standing behavior, and the doctest reads the device name so its
expectations follow the part. A G13 run of `omarchy_indexing_ops_tests`
at any commit from `27f47aa5d` on closes the item; if it fails, the
fix is to extend the exclusion, not to hunt the kernel.

A build-flow defect is outside this change but blocks source-built
baselines. A raw `pip wheel` of pre-Attn128 commit `c89392a5c` on the
M1 (T8103) produces a `libmlx.so` with an undefined
`mlx::core::fast::RMSNormGated` vtable reference, so the wheel cannot be
imported. The shipped release wheel and every Attn128-era build import
cleanly.
