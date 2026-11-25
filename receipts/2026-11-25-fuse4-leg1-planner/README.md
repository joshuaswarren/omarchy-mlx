# FUSE-4 leg 1 — planner refusal sites + flag-gated dual-write adoption

Date: 2026-11-25. Branch: `fuse4-leg1-planner` (off `fuse4/dual-write-gemv`
at a793af40ad). Scope: the planner-adoption follow-up the jwm2 staged
brief names as the gate for leg 1 ("No leg runs until the branch's
planner-adoption follow-up exists"). No hardware run; everything below
is proven on this dev box's llvmpipe (Mesa 22.3.6 lavapipe,
`MLX_OMARCHY_ALLOW_NON_APPLE=1`).

## What landed

- `overlay/mlx/backend/omarchy/fused_chain.cpp`: behind
  `MLX_OMARCHY_QMM_DUAL_OUT` the eager GEMV group planner now detects
  the adoptable shape and hands strided parent outputs
  (`member.out_view`) to `dispatch_quantized_gemv_group`: every
  epilogue-free single-row group member is the sole reader of one
  unclaimed last-axis `Concatenate`, each member stores its flat slot
  (prefix sums of the concat input sizes), and the standalone
  Concatenate dispatch — one dispatch per input on the Vulkan path —
  never runs. The concat's eval fires the group; its parent storage is
  allocated at fire time, and on any route refusal the concat replays
  into that buffer, so unlike the out-gate fold no throw-on-refusal is
  needed (every deleted node is replayable). Flag unset or "0" plans
  exactly today's route.
- `overlay/tests/omarchy/test_fused_chain.cpp`: two planner-level
  cases (red before the adoption existed, verified).
- Receipt: this file.

## Refusal-site map (the code-side leg 0)

The jwm2 brief names three candidates for why prod plans no groups
(SC-20 census: 90 single QmmVecQ4MultiSubgroup dispatches/token
despite `MLX_OMARCHY_FUSED_GEMV` default-on for G14C; G14C is NOT in
`fused_gemv_off_default_part`, device.h:1021, so the master gate is
not the cause). The ordered contract a group must survive, and where
each candidate lives:

Plan-time scan (fused_chain.cpp, `EagerFusionScope`):

1. :1393-1421 node admission: `QuantizedMatmul`, 4 inputs, unclaimed,
   `node_gemv_enabled`, Affine/transpose/4-bit/group-64, weights not
   on the tape (`lookup`), x window (ndim >= 2, 1..8 rows, exact
   size), token gate (`gemv_token_multi_enabled`).
2. :1383-1389 single-consumer map (leg-0 candidate 3): an Add
   epilogue is planned only when `uses[input.id()] == 1` and the one
   consumer is the Add; a reader count above one kills the epilogue,
   not the group.
3. :1500 a group of one member without an epilogue is not planned.

Attach-time validation (fused_chain.cpp):

4. :1536-1593 out-gate fold: reader counts, dtypes, subgroup caps,
   binding limit, `qmm_vec_q4_word_enabled()`.

Dispatch-time route (`primitives.cpp`, `dispatch_quantized_gemv_group`):

5. :10131-10136 input_ready ordering (leg-0 candidate 2): x must be
   ready on the group's stream when the first member fires
   (`!input_ready(x, stream)` refuses).
6. :10147-10161 the dual-out route gate: flag unset or "0", a
   KV-window member in the mix, token route, out-gate fold, or
   SwiGLU fold each refuse the whole group (wAP review of #798).
7. :10240-10242 whole_dense on real checkpoint params (leg-0
   candidate 1) beside input_ready for w/scales/biases.
8. :10248-10259 per-member dual-write contract: no epilogue/window
   on a view member, parent storage allocated, dtype match,
   row-contiguous whole parent, offset 0, `view.offset + node.size()
   <= parent.size()`, parent <= 4 GiB.

Which site fires first on the prod wheel is what leg 0's wave-diag
names; this list is the 1:1 map from the diag's "first refused
contract line" to code. Adoption (new, fused_chain.cpp:1646-1753)
adds its own gates ahead of the route so the planner never deletes a
concat the route would refuse: flag + `qmm_vec_q4_word_enabled()`, no
outgate/swiglu ownership, single-row x only, no epilogue/window
members, one unclaimed full-member last-axis Concatenate, dtype and
stream match, prefix-sum offsets covering the whole parent.

## Contract items

1. Flag off (`MLX_OMARCHY_QMM_DUAL_OUT` unset or "0"): no
   `out_view` is ever planned; dispatch counts and bytes are today's
   route. Pinned by the unset and "0" arms of the adoption test.
2. Flag on + adoptable shape: ONE dispatch serves the group; the
   standalone Concatenate is deleted; every member aliases the parent
   buffer; parent bytes are byte-identical to the per-node baseline
   (bf16 and f16; FNV-1a digests over the raw stored 16-bit words
   match).
3. Flag on + non-adoptable shape (any gate above): today's route —
   group merged, concat dispatches per input, member outputs in their
   own buffers, results bit-exact.
4. The dual-out route refusals are unchanged: KV-window mix, token
   route, outgate, swiglu (dispatch gate at primitives.cpp:10154), and
   the planner never plans those combinations.
5. Any dispatch-time refusal after adoption unwinds: members take the
   per-node path, the concat replays into the pre-allocated buffer —
   correct output, no throw.
6. Hardware done-when (jwm2 leg 1, unchanged): env-only arms unset vs
   `MLX_OMARCHY_QMM_DUAL_OUT=1`, x8 ABBA, med ms/tok -2% or better
   with identical ids; if 0 dual-write dispatches are recorded the leg
   reports null. This branch unblocks that run.

## Evidence (llvmpipe, this box)

- Red first: with the tests added and the planner unchanged, the
  adoption case fails exactly on the planned assertions (dispatches
  4 vs 1; member buffers distinct from the parent) for both dtypes;
  the refusal case passes unchanged.
- Green after the adoption: `omarchy_fused_chain_tests` — planner
  cases 2/2 passed, 2750 assertions; full file 51 cases, 76780
  assertions, 0 assertion failures.
- Pre-existing, not this change: the token-multi case fails when the
  FULL file runs in order on this llvmpipe ("gemv group did not
  form"), and the failure is identical on the unmodified base
  (base: 48+1 of 49; branch: 50+1 of 51). In isolation the case
  passes on both. Driver caveat: Mesa 22.3.6 warns < 26.2.4 may
  miscompute shift-heavy kernels; the byte-identity gates are the
  controlling evidence, same as the leg-0 branch's llvmpipe proofs.
