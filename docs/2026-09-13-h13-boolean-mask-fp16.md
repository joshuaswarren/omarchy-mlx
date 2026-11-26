# H13 boolean/mask representation decision (2026-09-13)

Status: DECIDED (prototype verified). Scope: the Parakeet encoder's
boolean/mask path — 72 ops: `select` 48, `cast` 11, `less` 4, `floor` 3,
`floor_div` 3, `logical_and` 1, `logical_not` 1, `reduce_min` 1. Companion
receipt: `receipts/2026-09-13-depalettize/README.md` (streams 2–3 backlog);
prototype: `overlay/tools/coreml/mask_lowering.py` + host tests.

## Decision

**Masks are fp16 `0.0`/`1.0` tensors (`0` = false); no boolean dtype ever
reaches the H13 device.** Every boolean producer emits fp16 0/1; every
boolean consumer accepts fp16 0/1. The runtime converts the i32 input
mask at buffer-pack time. Bool→fp16 is exact (`false`→`+0.0`,
`true`→`1.0`); fp16→bool at unpack is `value != 0`.

## Per-op-class numerics

### `logical_and` — LOWERABLE, bit-exact (existing `mul`)

`and(a, b) = a * b` on 0/1: `1*1 = 1`, `0*1 = +0`, `0*0 = +0`. Zero
sign is always `+0` on false results. No rounding can occur (products of
0 and 1 are exact).

### `logical_not` — LOWERABLE, bit-exact (existing `mul` + `add`)

`not(a) = a * (-1.0) + 1.0`: `1 → -1 + 1 = +0`, `0 → -0 + 1 = 1`. The
intermediate `-0` (from `0 * -1`) is absorbed by `+1`; results are
exactly `1.0` / `+0.0`.

### `select` — SPLIT: half lowerable, half **not exactly representable**

The graph has two families (fills read from the pinned package):

1. **Conv-row zeroing (24 ops): fill `a = +0.0`** (`var_13`, bits
   `0x0000`). Lowered `select(m, +0.0, b) = b * not(m)`:
   - `m = 0` lanes: `b * 1 = b`, bit-exact (includes `-0.0` lanes of b).
   - `m = 1` lanes: `b * (+0) = sign(b) · 0` — value-equal to the `+0.0`
     fill; the only bit difference is a `-0.0` where `b < 0` on a filled
     lane. Value-exact by IEEE (`-0 == +0`), bitwise-exact whenever
     filled lanes carry `b ≥ 0`.
2. **General finite fill: `select(m, a, b) = m*a + not(m)*b`** — exact
   for finite `a`, `b` except the same zero-sign class (a selected
   `-0.0` normalizes to `+0.0`).
3. **Attention-bias fill (24 ops): fill `a = -inf`** (`var_8`, bits
   `0xFC00`) — **COUNTEREXAMPLE, not exactly representable in fp16
   arithmetic**: the identity computes `0 * (-inf) = NaN` on every
   unmasked lane, destroying the bias tensor. No arrangement of
   `+,-,*` over 0/1 masks avoids a `0 · inf` product on the branch that
   must be discarded. These 24 selects REQUIRE a compiler `select`
   (fp16 lane pick by 0/1 mask) — registry addition, stream-3
   mil-hwx-compiler work.
   - Documented fallback (NOT bit-exact, softmax-level equivalent only):
     replacing the `-inf` fill with a finite `-65504`-class constant
     makes `m*a + not(m)*b` finite, and every consumer of these selects
     is `add → softmax` (verified in the pinned graph: one hop to
     `add_0_cast_fp16 → softmax`); `exp(x - rowmax)` underflows to
     `+0` identically for both fills. This changes intermediate bits and
     is only acceptable if the acceptance contract moves from
     bit-exact-select to exact-softmax-rows. Default: do not use.

### `reduce_min` — LOWERABLE on the 0/1 domain (existing `reduce_max`)

`reduce_min(x) = not(reduce_max(not(x)))` over 0/1 masks: `not` maps 0/1
to 1/0, `max` picks 1 iff any 0 existed, outer `not` restores polarity.
Bit-exact on 0/1. (The graph's single instance reduces a bool→i32 cast
of a mask; after boundary conversion it is a 0/1 fp16 reduce.)
**General-domain** fp16 `reduce_min` (arbitrary values) is NOT derivable
from sum/max/mean — remains a compiler gap if a future graph needs it.

### `less` — NEEDS_COMPILER_OP

A comparison is a step function; no finite `+,-,*,/` expression over
fp16 produces `(x < y) → 0/1` exactly (rounding destroys any sign-based
construction, and `x == y` must map to 0 with `y - x = +0`
indistinguishable from tiny positives without a compare). Registry
addition: `less` emitting fp16 0/1. All 4 graph instances compare
exactly-representable domains (fp16 arange constants vs length scalars;
i32 positions ≤ 374 vs lengths), so fp16 compare is value-exact there —
the general contract should still state "compare in the operand dtype;
the frontend guarantees fp16-exact operands."

### `floor` — NEEDS_COMPILER_OP, but exact in fp16

`floor` is total and exact over fp16: for `|x| ≥ 2048` the fp16 spacing
is ≥ 1 so `x` is already integral (`floor(x) = x`); below 2048 every
integer result is representable. NaN/±inf pass through. It has no
arithmetic identity — registry addition: `floor`. The 3 graph instances
floor [1]-shaped length scalars.

### `floor_div` — NEEDS_COMPILER_OP (blocked on `floor`)

Lowered as `real_div` + `floor`. Two envelope notes: (1) `real_div`'s
const-divisor envelope must accept the scalar-constant splat (the graph
divides by the constant 2.0); (2) fp16 `floor_div` equals
`floor(fp16_div(x, y))`, which is **not** integer floor division in
general — counterexample: `3199 / 32` divides-and-rounds to exactly
`100.0` in fp16 (tie-to-even), so the fp16 result is `100` while exact
integer floor division gives `99`. This matches the reference: the
package itself casts lengths to fp16 before dividing, so replicating
fp16 semantics is reference-exact. The contract states this explicitly.

### `cast` — BOUNDARY (host pack/unpack), not a device op

- fp32→fp16 (input features) and i32→fp16 (lengths): round-to-nearest-
  even, bit-identical to the package's own cast op (macOS Core ML
  executes the same conversion).
- bool→fp16 / fp16→bool: the 0/1 mapping above.
- fp16→i32 (lengths after `floor`, integral by construction): exact.
- The [1]-shaped length-derivation chain (`reduce_sum` of the input
  mask → `floor_div` → `floor` → compare constants) is scalar control
  metadata rooted only in the input mask; the runtime may compute it on
  the host at pack time (plan §3.3 host-orchestration allowance), which
  removes those ops from the device graph entirely. The on-device
  lowerings above remain the contract for the tensor-shaped remainder.

## Registry additions implied (mil-hwx-compiler, stream 3)

| op | semantics | driven by |
|---|---|---|
| `less` | fp16 compare, emit fp16 0/1 | 4 encoder ops; no algebraic form |
| `floor` | fp16 floor (exact; NaN/±inf pass) | 3 encoder ops + `floor_div` |
| `select` | fp16 lane pick by 0/1 mask | 24 ops with `-inf` fill (counterexample above) |
| `floor_div` | `real_div` + `floor` (or native) | 3 ops; scalar-const divisor envelope note |

`logical_and`, `logical_not`, the `+0.0`-fill selects, 0/1-domain
`reduce_min`, and all `cast`s need no compiler change.

## Prototype results (`overlay/tools/coreml/mask_lowering.py`)

`plan_mask_lowering` classifies; `lower_mask_ops` rewrites the lowerable
subset in place (helpers re-emitted in SSA order). Host tests
(`overlay/tests/omarchy/coreml/test_mask_lowering.py`, 4/4 OK):

- `and`/`not`/`reduce_min`: bit-exact on adversarial masks.
- `select(m, +0.0, b)`: value-exact; bit-exact outside the
  characterized `m=1 ∧ b<0 → -0.0` fill lanes.
- General finite fill: value-exact (same zero-sign class).
- The `-inf` counterexample is executable: the planner flags
  `NEEDS_COMPILER_OP`, the rewriter refuses, and the naive identity is
  shown to produce NaN on unmasked lanes.
- Real pinned encoder classification (cache-backed test): exactly
  24 select NEEDS (`-inf`), 24 select LOWERABLE (`+0.0`), 1/1/1
  and/not/reduce_min LOWERABLE, 4 `less` + 3 `floor` + 3 `floor_div`
  NEEDS_COMPILER_OP, 11 cast BOUNDARY — 72 total, matching the coverage
  receipt.
