# 2026-10-10: mlx-omarchy v0.7.32 (int8 matmul exactness and speed, large-shape correctness fixes, GDN and SDPA serve routes, translator fixes)

Release: v0.7.32, tag `v0.7.32` = `ae5d0950bc4f2d47843c65112a212e21eb5e0060` (main at the freeze). aarch64 only.
Range v0.7.31..v0.7.32: 265 commits, 587 files, +47,151 / -972 lines. Gates ran on the packaged Honeykrisp driver
`omarchy-mlx-vulkan` 0.7.28-2 with Mesa 26.2.4-1 (see "Gates on the frozen wheel" and "Known issues").

Read the known issues before you install: training backward (item 2), a long-dispatch fault on the 14-inch M2 Max
(item 3), long attention that runs out of memory on the 16 GB M1 (item 7), and four failing test suites (item 1).

## Shipped (user impact)

### Serve/GPU: SDPA, GDN, allocator, gather_qmm

- **SDPA prefill composed-first route for non-causal bf16 on coopmat
  devices** (25052f05f, #43); chunked composed route writes GQA output
  through the regrouped view (268a33228, #46); causal flash launches the
  heaviest causal q tiles first, bit-identical, 1 to 5% faster
  (a8c7132b1, #49); opt-in causal flash (PR 41 family).
- **GDN prefill masked rows take the single-pass recur32 route on G13
  parts**: padded batched prefill 1.64x to 0.98x of sequential on G13C
  (cc9b891bc, #45; receipt in the PR).
- Allocator: submitted batches + quarantine drained before the OOM
  retry (065168651, #44); bin sizes above 1 MiB geometrically (8 per
  octave), which fixes the 16 GB offload out-of-memory (ad831f094, #47);
  receipt 830e87bf1: oMLX upstream pin, M1 16 GB A/B on the allocator-fix
  wheel: prefill 1.97x, decode 1.10x, token ids identical.
- gather_qmm fp16 decode takes the subgroup kernel: M1 64.1 to 4.13 ms
  per layer (ce4e77a5e, #42).
- 4 upstream backports (dbb15d69b: compile scalar output, Python dtype
  narrowing, .npy shape validation, pickle bfloat16 strides).

### Parakeet worker pin (#50)

- The built ANE worker is now PINNED: stamp step at wheel build, runtime
  checker, tests (adbdc6408). **Caveat: wheels built before the stamp
  step (pre-stamp) fail verify_runtime_assets**. That is the checker
  working as designed; rebuild, do not bypass.

### int8 matmul (chunked coop + swiglu coop, extends the headline below)

- Chunked int32 reduction on the f32 cooperative-matrix route
  (a7ce66b67), swiglu through two f32 coopmat matmuls (9f6e1e1ad),
  rhs_offset honored in the coopmat value-half W loads (17734eeb2).

### Stock driver warning

- One-shot warning when the selected driver is a stock Mesa older than
  26.2.4 (95e7b6f8a). Hosts are on Mesa 26.2.4 now (pacman -Q mesa);
  the warning is for stock-Mesa installs that never got the packaged
  Honeykrisp ICD.

### Docs

- TensorFold H3 video: replication guide + patches + render script
  (325d8491f); fresh-clone replication verified byte-identical mp4
  sha256 36bd5df0a7b4b060304944beeeebb28e89de75ede22b1eaf3e31ce165fafee07,
  58 min 44 s, M2 Max (2b007ac7b, docs/tensorfold-video.md); the same
  seed-1 render on wheels 53bc1e3 and 95e7b6f produced the same hash
  (the int8 kernels are exact through the whole pipeline).
- README "How close to macOS" table: Linux as a percentage of macOS, same
  MacBook Pro, 5 MLX models + 2 llama.cpp GGUF, with receipts
  (c264198be, 5444406ed).

### HEADLINE: int8 matmul speed + exactness (the TensorFold H3 DiT path)

- fast.int8_matmul f32 routes rebuilt (8185470ce..9a991ff4f, 19 commits
  in the family since v0.7.31): bitwise A/B test matrix + naive kill
  switch (8185470ce); tiled shared-memory kernel, bit-identical to naive
  (fa41e66e8, default since 83a490a18 which also fixes an out-of-bounds
  shared write in X staging); cooperative-matrix f16 by f16 with f32 accumulation default
  route, bit-identical (a97dfc7cf); **f32 cooperative-matrix route for
  the linear shapes** (8f1a2ff5c): the scalar f32-FMA kernel is flat at
  ~330 GMAC/s on G13C, so the coop route streams to exact f32 temps via
  the converting copy with the same exactness argument as the scalar
  route (commit message, 8f1a2ff5c); coopmat flush/quarters/64-row-pad
  fixes (59c3a8fb0, 88f53e4f6, 2e5f28e72); **chunked int32 reduction
  lifts the scalar f32 route to group <= 131072** (9a991ff4f);
  registers-resident FMA accumulators (79cd72eab); poison test + extended
  A/B matrix (e6504ef28, dedd0fba8); f32-FMA default + flags doc
  (a72046465, docs/kernel-flags.md).
- Measured on the frozen wheel (release gate, same run, default route against the scalar route; GMAC/s; every
  shape bit-identical to the naive kernel, `ab-summary ALL MATCH` on each chip). Clip shapes: qkv = 6417x5376x16128,
  fc1-swiglu = 6417x5376x14336 with swiglu, fc2 = 6417x14336x5376 (group size 1024 for fc2).

  | Shape | 13-inch M1 (G13G) | 16-inch M1 Max (G13C) | 14-inch M2 Max (G14C) |
  |---|---|---|---|
  | qkv | 343.8 vs 79.9 (4.3x) | 1367.1 vs 324.5 (4.2x) | 1829.2 vs 412.1 (4.4x) |
  | fc1-swiglu | 337.8 vs 80.1 (4.2x) | 1364.0 vs 324.0 (4.2x) | 1818.4 vs 411.9 (4.4x) |
  | fc2 | 280.2 vs 79.1 (3.5x) | 740.8 vs 319.6 (2.3x) | 1846.2 vs 410.4 (4.5x) |

  The scalar route stays the fallback below the coop-capable shapes. Single run per chip; the rows come from the
  release gate logs, not from a repeated benchmark.
- This sits ON TOP of the dispatch-clamp fix (2df1aed43, below): rows*n
  > 16,776,960 writes every output row.

### HEADLINE: correctness fixes for large shapes (defects present in v0.7.31 and earlier)

One defect class: the one-dispatch thread/matrix clamps silently dropped
every output past the first dispatch. All documented in
docs/known-defects.md (80c39075c, section "Int8Matmul, CastBool, and fused
rope_rms_norm went unwritten past one dispatch (16,776,960 threads)" +
"Batched linalg factorizations went unwritten past 65,535 matrices").

- **fast.int8_matmul wrong for rows*n > 16,776,960** (2df1aed43): tail
  never written past the 16,776,960-thread clamp (first bad element at
  row 2047 col 7936 = floor(16776960/8192) exactly); the TensorFold H3
  DiT hit this. Fix: `dispatch_logical_chunked` host loop (LogicalOrBool
  precedent). Regression: omarchy_int8_matmul_tests "writes every output
  row past the clamp" (21b920440).
- CastBool > 67,107,840 bools unwritten (ba6f6c4e6): stride-loop bool
  cast; regression omarchy_copy_offset_tests / test_copy_offsets.cpp
  (2^26 bools, first bad element 67,107,840; Honeykrisp M1 zero-return
  hazard).
- fused rope_rms_norm >= 65,535 rows unwritten (ba6f6c4e6): stride
  loop + word-tail guard placed after all barriers; regression
  omarchy_fast_ops_tests "fused" case (c57d5ea67).
- **Batched linalg (Cholesky/Inverse/LU/SVD/Eig/Eigh) unwritten past
  65,535 matrices** (2dd7dc0e6): chunk batched factorizations past the
  workgroup clamp; regressions omarchy_linalg_ops_tests + omarchy_eig_ops_tests
  (91582d56a "must write every matrix past the one-dispatch clamp").
- Bonsai/GDN-VJP refuse by name above the clamp instead of clamping
  silently (ba6f6c4e6, 2dd7dc0e6).
- Stuck-submit diagnostic (c8104dd48, 58e9881f8, 36703f534,
  73a20886c): names the started-but-never-retired park.

### Upstream correctness fixes (Vulkan backend primitives/shaders, landed after v0.7.31)

- bool MaskedScatter native (85c4a0b9a, merge f3dc90c94, upstream
  #4635): dedicated masked_scatter_bool.comp shader + dispatch; verified on
  G13G hardware, matrix row updated (receipts/2026-10-08-bool-masked-scatter/README.md,
  landed via 1e87dc129).
- ArgReduce NaN propagation (b42203b66): argreduce_suffix.comp now
  propagates NaN like the reference: argmax/argmin over rows containing
  NaN previously returned wrong indices deterministically (upstream suite
  test arg reduce NaN failed 6/6 before). Documented as
  OMARCHY-ARGREDUCE-NAN in docs/known-defects.md (:1057, FIXED same day);
  test coverage in overlay/tests/omarchy/test_reduce_ops.cpp; audit
  receipt receipts/2026-10-08-audit-followups/README.md (b0fe81f50).
- empty Sort / ArgSort short-circuit (e7f56fc3d): empty arrays
  short-circuit before the dtype refusal instead of erroring; defect entry
  in docs/known-defects.md (same receipt b0fe81f50).
- Scan::eval_gpu parent-aliased-output guard (a3ccf3a90): shared-buffer
  view allocation guard in Scan eval_gpu prevents aliasing a parent
  buffer's output; failing-before test 7c0317160 (tests: CumSum with a
  pre-allocated shared-buffer output); receipt b0fe81f50.

### Serve performance
- mlx-lm last-logits for the dense qwen3 family (0bdba12e8, both patch
  lines): cached prefill computes the quantized head only for the final
  prompt position on qwen3 dense models. Measured: 4B pf512 +9.0%,
  digests equal, fp64-reference error equal
  (receipts/2026-10-08-qwen3-last-logits). Proven on pristine wheels:
  g17-patch-series dev-box run 2026-10-08, both lines rc=0 + idempotent,
  "applied: mlx-lm-last-logits-qwen3.patch" in both
  (receipts/2026-10-08-release-0.7.32/g17-0732/). Self-contained patch
  fix (8fed8d8e1: local os import, no NameError on a stock 0.31.3 tree).
- Assistant wake word (338f63902): opt-in openWakeWord ONNX listener,
  `--wake-word`; bytes coercion + per-detector peak counter (3ccab703c);
  ALSA 'pulse' device with default fallback (573037bf3).
- Batched gated-delta prefill runs per-row on the fused kernel for B>1
  bf16 Dk=Dv=128 (95a879daf, #40, receipt + device tests).
- ssm-maskless: left_padding setter clears the host mirror (7156b508e,
  review fix).
- matmul direct-kernel route rows for G14C + lower m floors on G13G
  (edac8b4c6, #38); G13G f32 a @ b route row REMOVED (9b8ff10bc, #39: it
  lost 8 to 42% at k 2560 / n 9728).

### llama.cpp demo (new user-facing path, 13 commits)
- omarchy llama.cpp demo end to end: pacman dep check + omarchy update as
  step 0 (9677365d4, cc5feab0e; refuses to build with a partial
  upgrade: llvm 23 vs llvm-libs 22 breaks llvm-spirv), path-independent
  wrapper, every Step 2/3 command works from the repo folder
  (81b22f245), --reasoning-budget 0 so Qwen3.5 does not print a 400-token
  thinking block first (e7c0df84a), memory preflight (ab9cab4a6: refuses
  below model + 3 GiB MemAvailable, so small hosts need swap or zram;
  ef99f0b99: refuses below model + 1 GiB, warns below + 3 GiB),
  skip-update override (e2d10a3e0), honest 18-minute system-driver build
  note, base-M1 8 GB-class numbers in the README row (8b1d34551: 9B
  IQ2_M 0.34-4.27 tok/s, 4B IQ2_M 0.44-7.42 tok/s).

### Chip support
- M3/M4 family: chip table rows, M3-aware ICD error, unified m3m4 kit with
  README, data-only owner steps (284524e5d merge: 50170f716, dbd36ada4,
  91434d7fd, be7e3bb70).

### Docs
- TensorFold recipe facts fixed (02e027713), MiniMax H3 int8 recipe
  (37ced5128), upstream layout addendum (6dcc32387).

### Also in this release (landed 2026-10-09 and 2026-10-10)

- **FloorDivide runs on Vulkan** (a07be3ade, #53). Integer `floor_divide` is one primitive (upstream #4642, core hunks
  backported as `patches/mlx-floor-divide.patch`) and runs on the GPU. Integer divide by zero gives 0. For f16 and
  bf16 the quotient is rounded to the storage type before the floor, as upstream does. Hardware run on the 16-inch
  M1 Max: 4001 f16 pairs and 4001 bf16 pairs equal the CPU stream (a first run had 37 of 4001 differing; fixed in
  the PR).
- **bf16 attention sinks in SDPA** (0a04e534d, #55). gpt-oss-20b feeds bf16 sinks. The f32-score route accepted only
  f32 sinks. It now casts bf16, f16 and f32 sinks to the score dtype. Red and green runs on the 16-inch M1 Max are
  in the PR.
- **Gated-delta forward accepts Hv as a multiple of Hk** (e59d4e8bf, #60). Qwen3.5-9B has 16 key heads and 32 value
  heads. An mlx-lm that passes the un-repeated q and k used to take the composed per-token fallback. The backend now
  expands q and k on the device. The backward at Hk != Hv stays composed by default (see the known issues).
- **Gated-delta maskless prefill on the 13-inch M1 defaults to recur32 mode 0** (84b9175b7, #59). Mode 2 changed the
  greedy tokens of Qwen3.5-9B against the 16-inch M1 Max and macOS. Mode 0 matches (digest `80274aa7...` on both
  chips) at 3.3 percent lower prefill. Mode 2 stays opt-in with `MLX_OMARCHY_GDN_RECUR32=2`.
- **OpenBLAS isolation in the wheel** (1b0155cd8). The wheel ships a content-hashed OpenBLAS next to `libmlx.so`.
  Before, `import mlx.core` before `import torch` broke torch with `undefined symbol: sbgemm_`, because the first
  library loaded under the soname served both.
- **put_along_axis on a negative-stride view** (1c6efa398). It raised a false "more than UINT32_MAX elements" refusal
  on an 8x8 int32 input. It now materializes the view first.
- **oMLX on Linux** (bc80807ed, #61). oMLX pinned to v0.7.1.dev1, a qwen3_moe offload patch (0012), a gated-delta
  prefill probe (0013), and a custom-kernel kill switch.
- **Custom-kernel translator** (00d43bf7a, d3e40b015, 0b9616369 #62, 468cdc186 #65, ae5d0950b #64). MSL `auto`
  declarations and `numeric_limits` in header helpers translate. Pointer-advance walks become index variables.
  Typedefs expand. Half assignment types are preserved, nested C-style casts finish, and half helper arguments and
  stores narrow. The translator suite passes 36 of 36 cases at the tag. The text of #64 and #65 calls them
  post-freeze; the freeze moved to `ae5d0950`, so both ship in this release.
- **Comparison tables.** "How close to macOS" in the README grew rows for Qwen3.5-9B, Qwen3.8-27B, gpt-oss-20b, the
  base M1, and the Linux-only M2 Max rows, each with receipts (464f2b382, 1bb7c1ec8, 9ed6ac9b3, 5d54b2631, f53878ec6,
  3144139e9, 75dec70a1, a504d358d, a6e89622a, 44970dca3). The macOS collector probes were fixed (0df455d95, #57).

### Hardware scope of the Bonsai (1-bit) route

Hardware-qualified on the 13-inch M1 (G13G) only: `receipts/2026-10-08-bonsai-build/README.md`. No gate row in this
release covers it on the other chips.

## Gates on the frozen wheel

Frozen wheel sha256 `a91fbe5ec2a8e4c197106d749739f696b2689f5926a97349d17e8827cc2f4712`. Every host runs Mesa
26.2.4-1 and the packaged Honeykrisp driver (`omarchy-mlx-vulkan`) 0.7.28-2 (library sha256 prefix
`ac837b1c3aa08672`) for the rows below. Earlier runs on other driver builds were replaced, not mixed in: the
16-inch M1 Max first ran on 0.7.22-1 and the 14-inch M2 Max on 0.7.28-1; both were upgraded during the gates and
their rows rerun.

| Gate | 13-inch M1 (G13G) | 16-inch M1 Max (G13C) | 14-inch M2 Max (G14C) |
|---|---|---|---|
| g1: install the wheel in a fresh home, import, GPU smoke | PASS | PASS | PASS |
| stamp: wheel version and runtime asset check | PASS | PASS | PASS |
| g17: 15 patchers on mlx-lm 0.31.3 and 0.32.0, apply twice | 4 of 4 | 4 of 4 | 4 of 4 (CPU gate, run before the driver upgrade) |
| g7d: Parakeet transcript through the ANE worker, golden compare (sha256 `db501a8c...a0790`) | PASS | PASS | not applicable (M1 and M1 Max only) |
| ANE legs g7c, g13, g15 | not part of this host's plan | g7c exit 0; g13 6 of 6; g15 4 of 4 | not applicable |
| int8 matmul A/B (bit-identical to the naive kernel) and timing | PASS | PASS | PASS |
| fc2 long-dispatch repeat check (output sha256 prefix `5d2eb391db2925b6`) | match | match | match |
| standing battery, 45 suites | 41 green, 4 red (see known issues) | 42 green, 3 red (test defects) | 42 green, 3 red (test defects) |
| capability profiles, 6 profiles of 7 cases | 7 of 7 each | 7 of 7 each | 7 of 7 each |
| pinned decode, Qwen3-4B-Instruct-2507-4bit, 64 tokens, median of 5 | 22.04 tok/s (v0.7.31 on the same host: 22.08) | not run | not run |

Decode check (release step 6): the frozen wheel and the v0.7.31 wheel were installed in fresh venvs from the vendor
tar and ran interleaved, five runs each, temperature 0, 64 pinned tokens. Frozen wheel: 22.12, 22.18, 22.00, 21.96,
22.04 tok/s. v0.7.31: 22.09, 22.09, 21.99, 22.07, 22.08 tok/s. All ten runs produced the same 64 token ids (sha256
prefix `1c5bddb589b1edf8`). An ANE job from another test was submitted on that host three minutes before the run
and may have overlapped part of it; both wheels ran under the same conditions. The wheel was installed from the
local release files, not from the uploaded draft; the uploaded bytes are checked by sha256 against these files.

`fast_ops` reports 47 of 47 cases passed; 8 assertions fail inside 3 cases that are marked as allowed to fail (see
the training-backward item below). The g1, stamp, g7d, int8 and decode scripts check the wheel sha256 before they run.
The gate summary with the key log lines is in `receipts/2026-10-10-release-0.7.32/`.

## Known issues

1. **Four test suites fail in the standing battery.** Three are defects in the tests, and so is part of the fourth.
   `gdn_legacy_policy` has a precondition that went stale after #60. `conv_gemm_decomp` passes a CPU reference
   stream to a GPU-only call. `gdn_maskless_correctness` has a fixture path fixed at build time (6 of 6 pass with the
   fixtures in place). `sdpa_prefill_flash` asserts the flash route without checking the device capability. One of
   its cases also fails with an out-of-device-memory error on the 13-inch M1. That failure is a limit in the product,
   not in the test, and item 7 describes it. Fix: #72 changes only the tests and fixes the four test defects named
   above. #72 lands after the v0.7.32 tag and ships in v0.7.33.
2. **Training backward only; inference is not affected.** Two fused backward bugs. Both are the same in v0.7.31: the
   gated-delta backward kernel and its equal-heads routing, and the attention backward code, did not change between
   v0.7.31 and this release (checked in the source at both tags). At unequal key and value heads (GQA) both releases
   use the composed backward by default. The fused GQA backward is opt-in (`MLX_OMARCHY_FUSED_VJP_GQA=1`), and its
   battery case reports 3 failed assertions.
   - The fused gated-delta backward gives a wrong gate gradient (dg) at equal key and value heads (16), head
     dimension 128. In the battery, at 33 tokens the relative L2 error of dg is 0.96 (correct at 1 token). Cause:
     `gated_delta_vjp.comp` adds only one lane's partial of the state product into dg.
   - The fused attention backward gives about half the true value (finite-difference reference) for dk or dv at
     isolated elements on small multi-head shapes with equal heads (4 to 7 keys; for example `dv[28]` is 0.168
     against 0.349 at 4x4). No error is raised. Not tested at real model shapes.

   Workaround for both: set `MLX_OMARCHY_NO_FUSED_VJP=1`. Fix, v0.7.33: #79 fixes the dg kernel. The attention case
   gets a fix or a fence, and equal heads move to the composed backward if that costs under 5 percent.
3. **14-inch M2 Max only (G14C): a long GPU dispatch can return wrong values beside an animating window.** A 690 ms
   dispatch returned partly wrong values in 6 of 6 runs with an animating window open; no error was raised.
   Dispatches of 23 ms and 87 ms were correct in 12 of 12 runs. The threshold between 87 ms and 690 ms is not
   measured yet. The 690 ms dispatch and a 5.5 s dispatch were each correct with no window, a static window, or a
   stopped window. The 16-inch M1 Max and the 13-inch M1 were clean. Stock Mesa 26.2.4 and six Honeykrisp builds
   all show it on the M2 Max. Receipt: `receipts/2026-10-10-m2-long-dispatch-vs-animating-window.md` (#74). Fix:
   v0.7.33 caps the size of one dispatch; the first step is a screening run that times each candidate operation at
   real shapes and flags any over 50 ms.
4. **Mixture-of-experts prefill is far behind macOS** (see the "How close to macOS" tables in the README), and the
   gemma-4-E4B model stalls the GPU timeline on M1-class hardware. Both are under investigation; no fix date.
5. **Uint8 and int8 `take` refuse by name** (the LTX video-VAE decode). Fixed on main in #76 after this release's
   freeze; ships in v0.7.33.
6. **Packages.** The release files and `install.sh` carry the 0.7.28-x driver. One lab machine, installed from an
   older stable-pinned image, had a pacman repository entry that offered only 0.7.22-1. What a fresh Omarchy install
   configures was not checked.
7. **Long non-causal bf16 attention runs out of device memory on the 13-inch M1 (16 GB).** Attention over 16384
   tokens with 8 heads (head dimension 128, non-causal, bf16) fails on its first run with
   `VK_ERROR_OUT_OF_DEVICE_MEMORY`. This happens alone, in a fresh process, on an idle machine (device heap 7.55
   GiB). Afterwards the process cannot allocate even one 1 GiB block and holds about 9 GB of memory. The 16-inch M1
   Max runs the same call. By reading the code, the chunked attention route keeps the f32 scores of every chunk
   until the whole batch finishes, about 18 GiB at this shape. We have not measured which smaller shapes fit. We
   have not tested the causal and masked routes, other lengths, or the 14-inch M2 Max. Fix: a change that releases
   chunk buffers early is in progress for v0.7.33.

Not shipped: the H66 experiment flag. Float16 writes into custom-kernel input buffers are not a limitation (MLX emits
every input as a const pointer, so Metal rejects such a write too).

## Assets

- `mlx_omarchy-0.32.4.dev202610101057+ae5d0950-cp314-cp314-linux_aarch64.whl`
  sha256 `a91fbe5ec2a8e4c197106d749739f696b2689f5926a97349d17e8827cc2f4712` (421,859,122 B). Built natively on the
  16-inch M1 Max with glslc 2026.3 and shaderc 1.4.357, whole-encoder bundle pin-verified at build time.
- `omarchy-mlx-vendor-wheels-v0.7.32-cp314-aarch64.tar`
  sha256 `f0fcdf0fb0bd67f3898df338b460afcc9af9de1281e96c12c82164bca1539db0` (36 vendored wheels, 466,933,760 B)
  plus its `.sha256`.
- `SHA256SUMS` covers all three uploaded files.
- aarch64 only, as in v0.7.31. No x86_64 wheel is built for this release.

## Asset check

`python3 scripts/verify-release-assets.py v0.7.32 --platforms linux_aarch64` ran against the uploaded draft files and
printed `VERIFIED: every uploaded asset matches what the release claims`. The check covers the sha256 of each file
against `SHA256SUMS`, the version in the wheel, the build commit `ae5d0950` against the tag, and the feature strings
of a stable build. The platform override is recorded in the run: this release ships an aarch64 wheel only, as v0.7.31
did. The downloaded wheel and vendor tar are byte-identical to the files the gates ran, which each host checked by
sha256 before the run.
