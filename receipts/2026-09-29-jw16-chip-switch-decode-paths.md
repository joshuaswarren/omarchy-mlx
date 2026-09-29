# jw16/jwm1 chip-keyed GDN state tile and LSE/argreduce walk prefetch (2026-09-29, DecodeGap4)

## Change

Main, 2026-09-29 (IRC): mlx-omarchy main-tip measured on jwm1 (T8103/G13G) at
decode64 -3.8% / decode256 -3.7% vs its dad5621b7 wheel with digests pinned;
the loss was attributed to f0f142776 + aafad4062 (GDN decode state tile via
shared memory, 4 workgroups/head; 16-step logsumexp/argreduce walk prefetch),
which gained +3.1..3.9% on jw16 (G13C/T6001). Both paths are bit-exact
recompilations of the same arithmetic, so the fix is a per-chip default plus a
runtime override, not a revert:

- `MLX_OMARCHY_GDN_DECODE_TILE=0|1` — 1 (default on G13C and non-G13 parts)
  dispatches `GatedDeltaDecodeBF16` (staged tile, gridY = Dv/32); 0 dispatches
  the new append-only `GatedDeltaDecodeBF16Untiled` (the pre-tile per-row
  kernel restored verbatim from f0f142776^ as
  `shaders/gated_delta_decode_perrow.comp`, gridY = 1, raw-gates flag bit 8
  included).
- `MLX_OMARCHY_LSE_ARGREDUCE_PREFETCH=0|1` — 0 sets push-constant flag bit 31
  on the LogSumExp/ArgReduce suffix kernels, which selects the verbatim
  legacy one-load-per-element walk loops inside the same shaders; 1 (default
  on G13C and non-G13 parts) keeps the 16-step prefetch loops.
- Unset env resolves per device: `device_name` containing `G13` but not
  `G13C` (jwm1's G13G/T8103 class) -> legacy paths; everything else ->
  new paths (jw16's G13C/T6001 behavior and digests unchanged).

Commits: d445e7cf9 (change), ab1a183c2 (merge to main).

## Gates (jw16, boot a6a194f4, gpuwin window 2026-09-29T18:5xZ, llm-inference
restored health 200 + completion probe finish=length)

- Bit-equal arms (Main's requirement): `dg_bitcheck.py` (3084 rows: GDN
  raw/gb/special/chain, LSE 270, argmax/argmin 720, composed head, norm
  controls) in the candidate venv (wheel 0.32.3.dev202609291832+d445e7cf9)
  under four arms — default, both-legacy, tile1, pf0 — produced BYTE-IDENTICAL
  JSON (diff empty; sha256 4736b4609b1dafc0...). The serving-venv reference
  run differs only in the recorded mlx version string; every hash row is
  identical. Artifacts: apple-silicon-lab artifacts/Jw16DecodeGap4/w1/.
- Cells n=5, interleaved, digests pinned to production: d64 ctl 98.46/98.63
  vs cand-default 99.2 vs cand-legacy 98.8, all c84b3e7af640; d256 ctl
  97.55/97.59 vs cand-default 97.93 vs cand-legacy 98.1, all c6aabbf0a51d.
  The default arm holds jw16's deployed performance (within control range);
  no regression from adding the switch.
- jwm1 (G13G) validation is the jwm1 owner's pickup (w71): set nothing and
  the legacy paths engage; expected recovery of the ~3.8% decode loss with
  digests unchanged (the arms are bit-equal, so digests cannot move).

## Rollback

- Runtime: `MLX_OMARCHY_GDN_DECODE_TILE=0 MLX_OMARCHY_LSE_ARGREDUCE_PREFETCH=0`
  restores the pre-2026-09-29-decode kernel behavior on any host.
- Source: revert ab1a183c2.

## Limits

- Only G13C and G13G are measured; other G13 parts default to the legacy
  paths (the months-shipped behavior), everything non-G13 to the new paths.
- The legacy per-row GDN kernel carries the raw-gates prologue (flag bit 8)
  exactly as shipped before the tile commit; the LSE/argreduce legacy walks
  are the pre-f0f142776 loops verbatim inside a uniform branch of the same
  SPIR-V (validated by the identical-hash arms above, not by separate
  compilation).
