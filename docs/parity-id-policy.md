# When a candidate may change generated token ids

Decided 2026-09-10 while closing the M1 performance gap. The gate that had
been in force ("bit-identical to the previous Linux release on every leg")
was blocking the two largest remaining levers, and it was protecting the
wrong invariant: it froze Linux-specific arithmetic that does not match
macOS MLX, which is the parity target.

## The rule

A candidate may change a generated token stream only when both hold:

1. **The new arithmetic is native's arithmetic, proven.** The kernel must
   reproduce captured native intermediates exactly on fixed inputs (the
   oracle captures under `receipts/2026-09-09-*-control` and
   `receipts/native-baseline-2026-09-06`), not merely produce "plausible"
   output. A digest that is neither the old Linux value nor native's is
   rejected; that is what disqualified the fused SDPA attempt in
   `receipts/2026-09-09-decode-sdpa`.
2. **No leg that already matches native regresses.** Legs whose digest
   equals the native digest (today: Q4 short and Q4 1K context, BF16 1K
   context) must still equal it after the change.

Legs whose Linux digest already differs from native (today: Q4 262-token,
BF16 short, BF16 262-token) may move to native's digest. They may not move
to anything else.

## Consequences

- Release notes must state any digest change and which legs moved toward
  native. Both drivers (stock Mesa and the Honeykrisp fork) are checked;
  `receipts/2026-09-10-stock-prefill-digest` is the pattern.
- The stock-driver fallback paths are held to the same rule; a fork-only
  arithmetic change that silently alters the stock stream is a defect
  (`receipts/2026-09-10-v0.4.1-release.md`).
- Kernels kept only to preserve Linux-specific rounding, once native's
  order is implemented and proven, are deleted rather than retained behind
  a flag.

## Amendment 2026-09-10: the BF16 short and 262-token pins moved with the dense decode GEMV

The owner re-pinned the BF16 short (30/32) and 262-token (262/128)
generated-id digests when landing the native-order dense BF16 decode GEMV
(`receipts/2026-09-10-bf16-decode-gemv-land`; qualification evidence in
`receipts/2026-09-10-bf16-decode-gemv-requal`). The pins are per-driver
values, measured fresh on the installed fork and on stock Mesa. Rule 1's
"only native's digest" carve-out was waived for exactly these legs, on
measured grounds:

1. The old sequential kernel and the new subgroup kernel are float64
   round-to-nearest exact on every captured real decode projection,
   including the bias-cancellation elements where macOS Metal deviates. The
   moved digests carried no parity content: nothing on macOS ever produced
   them.
2. On synthetic large-N samples the new accumulation order is closer to
   float64 than the old one, never farther.
3. The BF16 legs already diverged from macOS before this change - that is
   why they sat in rule 1's "may move" list - and the 262-token leg already
   split across drivers before this change. Holding those streams frozen
   protected an f32 accumulation artifact, not parity.

The native-matching legs keep rule 2's full strength: BF16 1K context and
the Q4 short and Q4 1K-context digests must still equal native after any
change, and the landing receipt re-proves all six canonical Q4 digests
