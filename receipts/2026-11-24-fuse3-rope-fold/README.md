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

## The planner: what the first attempt showed, and the two shapes that work

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

Two shapes solve it:

1. **The rope node drives.** The planner arms the rope node with the
   attention node's inputs (keys, values, the scale, the output) and
   the attention node is marked "skip if the rope fold ran". The rope
   node's interception calls the attention primitive's eval with a
   thread-local fold context (the raw query, the weight, the offsets,
   the rope params). The attention eval either folds (the split route's
   folded variant) or returns "refused" **without dispatching**; on a
   refusal the rope node runs its ordinary rope dispatch and the
   attention node later runs its ordinary path, where the query now has
   its data. The refusal must be an early-out inside the attention
   eval: it may not fall through to a route that reads the query.
2. **The skip is conditional on a superset of the route's guards.** The
   planner skips the rope node only when it can prove the split route
   with the folded variant will run (batch 1, one token, head width
   256, the bf16 dtypes, the split-group set, the stride alignments,
   the device's subgroup and workgroup limits, and the split/pv/bk64
   env pins). That predicate duplicates the route's guard chain, so it
   must live next to it - a predicate that drifts from the route would
   skip the rope and leave the attention with no query.

Shape 1 keeps the route logic in one place (the attention eval) and
needs no duplicated guards; its cost is that the attention eval needs
a "refused, do not dispatch" early-out, which is a new contract on
that function.

## What the planner still needs after the shape is chosen

- The fold's dispatch belongs in `ScaledDotProductAttention::eval_gpu`
  (the split route at ~line 16830 of `primitives.cpp`), not in the
  planner: the route's guard chain, the scratch allocation and the two
  pass dispatch all live there, and duplicating them in
  `fused_chain.cpp` would be a second copy to keep in step. The fold
  swaps `split_p1` for the folded kernel, adds the weight and offset
  bindings, and packs the rope params into the push fields the split
  kernels leave free: eps bits in `operation`, the rope scale bits in
  `lhs_size`, the freq exponent (`log(base) / half_dims`) in `beta`,
  the half dims in `dims`, the query row's time index in `matrix_m`,
  the offsets buffer offset in `output_offset` and the weight element
  offset in `aux_size`.
- **Eligibility:** the query's primitive is `fast::RoPE` with
  `has_norm()` (the fused `rope_rms_norm`), `forward_`, not
  `traditional_`, `dims_ == head_dim` (the full-row rotation, so
  `half_dims = head_dim/2`), bf16 query and weight,
  `weight.size() == head_dim`, the offsets array present, and the
  rope's `use_fallback` false. The rope's f16 route, the freqs route
  and the non-fused norm keep the unfused pair.
- **The flag:** `MLX_OMARCHY_FUSE3=1` enables the fold; unset or `0`
  keeps the unfused pair. It must gate both the planner's arming and
  the attention's fold, so a half-armed state cannot skip the rope
  without folding it.
- **The gates still open:** the Apple reduction path on real hardware
  (the llvmpipe proof took the tree path or the subgroup path
  depending on the device's subgroup report), the ids identity on a
  decode run, and the dispatch-count check that the folded route is one
  dispatch lighter per attention layer.

## The K side (the other dispatch)

The keys side is the harder half: the rotated key must also reach the
KV cache, and today the rope pair writes that window directly
(`find_rope_kv_redirect` in `fused_chain.cpp:2130`, the producer-direct
KV write). Folding the key rotation into the SDPA moves that write's
ownership to the attention dispatch, which changes the slice-update
planning and the dependency tracking. It should not be started before
the query side has its ids gate.
