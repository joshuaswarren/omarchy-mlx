# FUSE-3 fold 1 (the Q side): the rope norm+rotation inside the split pass 1

Receipt for the kernel half of the FUSE-3 card, 2026-11-24 (w7Q).
Branch `feature/fuse3-rope-fold-q`, PR 819, base `main` 25f823581c.

## What landed

`sdpa_decode_split_p1.comp` gained a `-DSDPA_QROPE=1` build (two
variants: the shared-memory tree and the Apple subgroup reduction) that
reads the **raw** query and computes `fast_rope_norm`'s RMS norm and
half-split rotation in its Q load, so one dispatch can replace the
`rope_rms_norm` + split-pass-1 pair. The kernel is registered
(`SdpaDecodeSplitP1R4BF16Hd256Qrope`, `...QropeApple`) but **not yet
reachable**: the planner that folds the rope dispatch is not in the PR,
so the production path still runs the unfused pair.

## Why the bit-identity is achievable (and proven)

The SDPA's thread `t` already owns query dims `{t, t + 128}`, which is
the rope kernel's own pairing: `fast_rope_norm.comp`'s reduction tree
adds lane `t` to lane `t + 128` first. So the 128-thread workgroup
reproduces the 256-lane rope reduction exactly:

- **Non-Apple tree** (`fast_rope_norm.comp` lines 208-220): the first
  step `sq(dim t) + sq(dim t + 128)` in thread, then strides 64..1 over
  the remaining 128 sums - the same sequence of f32 additions.
- **Apple subgroup path** (lines 176-200): the rope kernel's 256
  threads are 8 subgroups of 32, one dim each. The SDPA's subgroup `sg`
  holds rope lanes of rope subgroups `sg` and `sg + 4`, so two
  `subgroupAdd`s over this subgroup's 32 lanes reproduce two of the
  rope kernel's partials bit for bit; the 8 partials go through shared
  memory and subgroup 0's lanes 0-7 run the rope kernel's second level
  (`partial = lane < 8 ? s_part[lane] : 0; row_sum = subgroupAdd`).

The per-element chain is the rope kernel's own: `weight_norm = weight *
norm; rounded = rope_round(value * weight_norm)` with the value
**re-loaded from memory** after the reduction (the rope kernel
documents that carrying the squared register across the barrier lets
NIR fold the chain into an ffma that drifts one bf16 ULP at the
rounding tie line), then the half-split rotation with the rope's own
`rope_round` on each product and the same `bf16_bits` rounding on
store.

Two bugs the proof surfaced, both in the first draft:

1. The shared reduction state was rewritten by the next head's
   reduction while a slow thread still read `s_red[0]`. The rope kernel
   ends its row loop with a trailing barrier for exactly this reason;
   the fold now does the same.
2. The test passed `dims = head_dim`; the rope host passes
   `dims = dims_/2` (the half dims). With the wrong value the rope
   arm rotated pairs `(i, i + 256)` - past the row - and the two arms
   disagreed on half the heads.

## Test

`overlay/tests/omarchy/test_fuse3_rope_fold.cpp` drives the rope
kernel and both pass-1 builds directly at the route's guarded shape
(batch 1, one decode token, bf16, 8 query heads over 2 kv heads, 32-lane
subgroup arithmetic) and compares the scratch word for word:

```
[doctest] test cases: 1 | 1 passed | 0 failed | 0 skipped
```

0 of 2064 scratch words differ (the block max, the block exp sum and
the 256 weighted-value floats per head). Both arms ride the same
reduction path (the Apple kernel when the device has 32-lane subgroup
arithmetic, the tree otherwise), so the proof covers whichever path the
device takes.

Local build note: this host's glslc (shaderc 2023.2) segfaults on the
i64 G18 `int_elementwise` builds with `-O` and knows no
`GL_KHR_cooperative_matrix`, so the local doctest build routes through
`/tmp/shim-noglslc/glslc` (a wrapper: no `-O` for the i64 G18 files, a
trivial stand-in for the coopmat files). CI's modern glslc compiles
both for real; nothing in the repo depends on the shim.

## The planner: what the first attempt showed, and the sound shape

The naive shape - the planner skips the rope node and the attention
node's eval performs the folded dispatch - **crashes**, and the reason
is worth recording because it is not obvious.

The tape evaluates the rope node before the attention node, so the
skip removes the rope dispatch before anything knows which attention
route will run. The attention's own eval then finds a route it cannot
fold (on llvmpipe the split route is not taken at all) and falls
through to its ordinary path, which binds the query array - and that
array is the skipped rope output, which has no buffer. The crash is a
null dereference in `binding(q)`, not a wrong value, but it is fatal.


The fix has two parts, and the first one is a trap.

**The decision belongs at the attention node's turn, not the rope
node's.** It is tempting to let the rope node drive - its interception
calls the attention eval, which folds or refuses - but that is unsound:
the attention's keys and values are the KV cache after the update, and
that update sits between the rope node and the attention node in the
tape. At the rope node's turn the cache is not ready, so the attention
cannot be dispatched from there.

**So the rope node is skipped, and the attention eval must know, before
it dispatches anything, whether the split route will fold.** The
attention eval's route precedence is: the one-pass decode route, the
prefill routes, then the split-K block - and each of them binds the
query. With the rope skipped, a route that cannot fold would bind an
array with no buffer. The sound shape is therefore:

1. Extract the split route's **decision** (its guard chain and the
   `split_params`/kernel selection, which are pure) into a lambda
   defined in place inside `eval_gpu`, capturing the locals by
   reference - no parameter plumbing, one copy of the guards. The
   ordinary split block then becomes `if (auto route = split_route()) {
   dispatch }`, and the route's precedence is unchanged.
2. At the top of `eval_gpu`, when the planner has armed this node: call
   `split_route()`. If it folds (the `SdpaDecodeSplitP1R4BF16Hd256`
   variant, one token, width 256, the rope's `dims == head_dim`), the
   split block below takes the folded kernel. If it does not, run the
   skipped rope node in place **now** (`sdpa_rope_fold_refused`), so
   every later route finds a query with its data.
3. The planner skips the rope node whenever the attention node is armed
   and eligible, because step 2 makes a refusal safe at the attention
   node's turn: no predicate has to predict the route, and nothing can
   drift from it.

The refusal is cheap and exact: the rope node's primitive and inputs
are stashed at plan time, so the in-place run is the rope's own
`eval_gpu` on its own arrays.

## The planner: landed (f907bc5dc3 + 171bd51c66)

The attention side and the planner are implemented on this branch, and
the shape that landed is the one this section describes, with three
corrections the first working version needed:

- **The eval loop releases a tape node's inputs once its consumers have
  run.** The skipped rope node's raw query, weight and offsets are read
  at the attention node's turn, so the skip pins them to the submission
  (`encoder.add_temporary`), the way the rope-norm fold does.
- **`array::detach()` clears primitive and inputs on the descriptor the
  rope node shares with this record.** The refusal path therefore cannot
  use the rope node itself as the rope eval's output (the rope eval reads
  `out.primitive().stream()`): it builds a fresh output from the stashed
  primitive and inputs, runs the rope into it, and copies the bytes into
  the rope node's descriptor - which is exactly what the attention reads
  the query through.
- **The fold covers both width-256 pass-1 variants** (the plain pass 1
  and the value-phase pass 1 the M1 Max and the M3 Pro take by default):
  their Q loads are the same code, so `sdpa_decode_split_p1v.comp` takes
  the same `-DSDPA_QROPE` build and the fold accepts either.

## Hardware proof so far (the M6, the composed route's board)

The M6 does not take the split route (its decode guard refuses the
subgroup set), so the armed nodes exercised the **refusal** path there,
not the fold: the planner armed 35-36 attention nodes per eval
(`[fuse3] planned=36 cand=36`), the split block declined, the skipped
rope node ran in place, and the late safety net did not need to fire.
The gate - the same Qwen3-4B 35+128 decode with the split route forced,
the fold off and on on the same wheel - produced

```
arm off rc=0 dispatches=119399
arm on  rc=0 dispatches=119399
IDS_EQUAL True        # 128 greedy ids, fold off and on
```

identical dispatch histograms (the fold refuses on every armed node) and
identical ids: the skip, the pinning, the in-place rope and the
descriptor handback are sound on real hardware. The **firing** path is
proven next on the M1 Max (`jwm1`), whose default route is the pv-wide
split variant the fold covers; the doctest already proves the folded
kernel bit-identical for both variants on llvmpipe.

## What the planner needs

The concrete sequence, in the order that keeps every step verifiable:

1. **Hoist the decode locals** (`decode_env`, `decode_caps`,
   `decode_bf16_probe`, `split_rows_route`, `decode_path_override`)
   above the native one-pass route in
   `ScaledDotProductAttention::eval_gpu`. They are pure reads and the
   fold's decision needs them before any route dispatches.
2. **Extract the split route's decision** into a pure lambda defined in
   place (capturing the locals by reference), returning the params, the
   p1/p2 kernels, the block size and the row count, or `nullopt`. The
   ordinary split block becomes `if (auto route = split_route()) {
   dispatch }` - one copy of the guards, no precedence change.
3. **The fold at the top of `eval_gpu`:** when
   `MLX_OMARCHY_FUSE3=1` and the planner armed this node, call
   `split_route()`; fold only when the selected p1 is
   `SdpaDecodeSplitP1R4BF16Hd256` (no pv, no bk64, one token, width
   256), the raw query is bf16, and the rope's `dims == head_dim`.
   Otherwise run the skipped rope node in place now, before any route
   binds the query.
4. **The folded dispatch inside the split block:** the raw query, the
   norm weight and the offsets ride bindings 4-6, and the rope params
   ride the push fields the split kernels leave free: eps bits in
   `operation`, the rope scale bits in `lhs_size`, the freq exponent
   (`log(base) / half_dims`) in `beta`, the half dims in `dims`, the
   query row's time index in `matrix_m`, the offsets buffer offset in
   `output_offset` and the weight element offset in `aux_size`. Pass 2
   is unchanged.
5. **The late safety net:** right after the split block, if the plan
   was armed but the block did not run (a guard the arming predicate
   missed), run the rope in place - the later routes then find a query
   with its data instead of crashing.
6. **The planner** (`fused_chain.cpp`): the scan arms the attention
   node when its query is a single-use `fast::RoPE` output with the
   fused norm (bf16, forward, non-traditional, `dims == head_dim`, the
   rope's `use_fallback` false) and marks the rope node as skipped;
   `try_eval_eager_fusion` returns true for the skipped rope node; the
   plan accessor and the refusal hook are the two functions the
   attention eval calls.

Eligibility refuses everything else to the unfused pair: the rope's
f16 route, the freqs route, the non-fused norm, prefill shapes, and
every non-split route.

The flag: `MLX_OMARCHY_FUSE3=1` enables the fold; unset or `0` keeps
the unfused pair. It gates both the planner's skip and the attention's
fold, so a half-armed state cannot skip the rope without folding it.
- **The flag:** `MLX_OMARCHY_FUSE3=1` enables the fold; unset or `0`
  keeps the unfused pair. It must gate both the planner's arming and
  the attention's fold, so a half-armed state cannot skip the rope
  without folding it.
- **The gates still open:** the Apple reduction path on real hardware
  (the llvmpipe proof took the tree path or the subgroup path
  depending on the device's subgroup report), the ids identity on a
  decode run, and the dispatch-count check that the folded route is one
  dispatch lighter per attention layer.

## The applicability finding: the target models are partial-rotary

The Qwen3.8-2B config carries `"partial_rotary_factor": 0.25` with
`head_dim: 256` - the rope rotates **64 of the 256 dims**, and the rope
host passes `dims = 64` (the rotary width), not the head dim. The fold
as built requires `dims == head_dim` (the full-row rotation, whose
pairs are `(i, i + 128)` - the pairing this kernel's thread layout
already owns). On the Qwen3.8-2B the fold therefore refuses on every
armed node, and the M6's ids gate shows exactly that: identical ids,
identical dispatch histograms, fold on and off.

Making the fold fire for the Qwen3.8 family needs the passthrough
branch of the rope kernel reproduced in the fold's Q load: the rotation
pairs are `(i, i + 32)` (half dims 32), which the thread layout does
not own, so the rotation reads the normed-rounded values through shared
memory (the rope kernel's own rotation phase already does this for its
pairs). The reduction stays untouched. It is doable and it is a second
kernel contract to prove bit-identical; until it lands, the fold's
saving is zero for the Qwen3.8 family, and the card's minus-two-percent
target on the 2B decode is not reachable by this fold.

The fold remains correct and useful as landed for any future model with
full rotary width at head width 256 on the split route, and the
refusal-path machinery (the skip, the pinning, the descriptor handback)
is proven on hardware.

## The K side (the other dispatch)

The keys side is the harder half: the rotated key must also reach the
KV cache, and today the rope pair writes that window directly
(`find_rope_kv_redirect` in `fused_chain.cpp:2130`, the producer-direct
KV write). Folding the key rotation into the SDPA moves that write's
ownership to the attention dispatch, which changes the slice-update
planning and the dependency tracking. It should not be started before
the query side has its ids gate.
