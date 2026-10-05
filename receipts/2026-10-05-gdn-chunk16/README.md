# 2026-10-05 — GDN prefill C=16 chunk coopmat (second attempt, post-HOIST body)

Lane: GdnChunk16 (worker). Hosts: the T6021 machine for build/doctest/A-B
(kktqkt pass-A + chunked coopmat pass-B kernels, env-gated CHUNK16);
jw16/jwm1 numbers come from Main/w71 via bundle (no direct jw16 access —
Main blocked jw16 ~21:50Z reboot for PR boot tests; resume after
'JW16 BACK'). Private notebook:
`entries/GdnChunk16/20261005T214500Z-jw14m2-gdn-c16-redesign.md`
(pre-registered before any M2 work). Artifacts
`artifacts/GdnChunk16/2026-10-05-build/` (SHA256SUMS sealed on close).
Branch: `agent/GdnChunk16` (off origin/main @ 04f825676, which has
the HOIST landing 7d84c4b27). Wheel under test:
0.32.4.dev2026100XXXXX+<commit> (lane build; provenance verified=match).

## Why a second attempt

The first C=16 attempt (Jw16GdnBlock16, e1b569cad) was REFUTED on G13C
at T=512/1000/2048 by 33–50% slower (the rolled single-buffered Dk
staging loop lost the load/MMA overlap that the C=8 batch kernel's
unrolled double-buffered staging kept — G13C's 32 KiB shared cap
blocked f32 double-buffering at C=16). The structural claim in the
goal ("C=16 halves the serial chunk steps") is sound; the prior
implementation just did not preserve the per-step overlap.

This second attempt ports the prior's block-2x2 decomposition (TR=0)
and the e1b569cad shared layout exactly, then plans to gate it on the
post-HOIST baseline (-12% kernel at T>=512 already on main). The
decision bit is what the previous entry failed to capture: **does
C=16 win on a G13G with HOIST's baseline, or does it regress the same
33-50%?** A win here would be the structural lever the chunk walk has
been waiting for.

## What landed (env-gated, default OFF)

`gated_delta_prefill_coopmat_c16.comp` +
`MLX_OMARCHY_GDN_CHUNK16=1` (kill switch: unset/=0 restores HOIST or
batch; default ON path is HOIST length-gated T >= 512). The prior
shader was renamed to `..._coopmat_c16.comp` (file) with the dispatch
profile id `GatedDeltaPrefillCoopmatChunk16BF16` (enum append-only).
The dispatch gate fires BEFORE the HOIST/batch block when the env is set,
so CHUNK16 wins on selection — but ONLY on the pre-registered gates.

## Gates on the T6021 lane build

- Captured-operand doctest: BIT-IDENTICAL against shipped (HOIST) on
  fixtures gdn_coopmat layer0+layer12, out AND state, byte-compare,
  same wheel OFF vs ON.
- Captured-operand doctest with NONZERO_STATE (the seeded non-zero
  initial state — the kernel-level state-path proof; production
  zeros do not exercise it).
- Determinism x3: T=512 chunk16 run hashes byte-identical across 3
  reps.
- Kernel-only micro T=512/1024 (3 gated pairs) against HOIST.
- e2e pf512/pf1024 paired A/B (bench_decode, raw prompt; same wheel).
- Greedy digests: d64 d128 d256 d512 d1024 d2048 token-identical
  (decoder-through-model; production data).
- Numerics per docs/numerics-gate.md on Qwen3.5/3.8 2B and 9B:
  teacher-forced top-1 >= 99%, every free-run first divergence
  composed top-2 gap <= 1 bf16 ULP at logit magnitude, PPL within
  0.1%.

## Gates on jwm1 (G13G, w71 via Main)

- kernel timing T=512/1024 vs HOIST (the G13G dispatch — the only
  host where the G13G chunk-walk lever matters; the M2 is already
  macOS parity on HOIST alone).
- e2e pf512/pf1024 against HOIST.
- All four pins exact: d64 eee1cf9635d6d4eb, d128 0756351401f5b3fb,
  d256 393a1cf303e9f362, pf512 509c19201275dfcc.

## Falsifier (kill the re-pin)

If M2 gates pass but the G13G measurement regresses >= 15% vs shipped
(at -33% per the prior attempt; -15% is a 50% recovery of the prior
loss) OR any of the numerics gates fails, the C=16 lever stays
default OFF. The decision rule for the re-pin: M2 gates green AND
G13G kernel -25% vs HOIST AND the G13G numerics pass; otherwise the
branch is a recorded negative and CHUNK16 stays OFF permanently
(swap rule: keeping the original attempt long after a refutation is
hoarding).

## VERDICT (M2, 2026-10-05T22:10Z): REFUTED — CHUNK16 stays OFF

The M2 run ticket measured C=16 at -51%/-52% vs HOIST (T=512/1024
kernel-only fast-op, 9 reps, median, quiet gates):

| T    | ship (HOIST) fast-op | chunk16 fast-op | delta |
|---|---|---|---|
| 64   | 5.71 ms      | 9.33 ms    | -39% |
| 128  | 10.30 ms     | 17.15 ms   | -40% |
| 512  | 31.62 ms     | 64.45 ms   | -51% |
| 1024 | 61.89 ms     | 129.78 ms  | -52% |

Per-call at T=512: 1.757 ms (ship) vs 3.580 ms (chunk16) = 112
us/chunk at C=16 vs 27 us/chunk at C=8 (7.0 us/token vs 3.4
us/token = 2.07x per token). This matches the prior G13C measurement
(110 vs 27.7 us/chunk at 2.0x per token) exactly — the regression is
architecture-independent and structural.

Determinism x3 (all four of layer0/12 × y/hf): IDENTICAL across 3
runs. Captured-operand doctest: chunk16 y/hf DIFFER from ship on all
four captures (the chunk-form rounding moves the digest by
construction, as pre-registered) and chunk16_nz differs from
chunk16 (the state path is exercised). The numerics gates were
therefore not reachable: the kernel is 2x slower and the digest
moves; both the -25% perf bar and the numerics re-pin path fail.

The branch (agent/GdnChunk16 @ 89625076c, default OFF,
MLX_OMARCHY_GDN_CHUNK16) is a recorded negative; the C=8 paths are
byte-identical and no default flips. Swap rule: the second refutation
closes the C=16 lever on this part class. The correct next structural
lever for the G13G chunk walk is the cross-slice N/Tinv sharing via
grid barrier (the GridBarrier/GduBar item the 2026-10-05-gdn-prefill
receipt named), which keeps C=8 numerics.

Artifacts (sealed SHA256SUMS, 34 entries):
`artifacts/GdnChunk16/2026-10-05-build/` (build.log, run.log,
doctest/*.SHA256 + binaries, micro/{ship,chunk16}.log). Notebook:
`entries/GdnChunk16/20261005T214500Z-jw14m2-gdn-c16-redesign.md`.

## Reproduce (lane)

```sh
git fetch origin agent/GdnChunk16
git checkout -b agent/GdnChunk16 FETCH_HEAD
bash scripts/check_gdn_variants.sh   # compiles all 4 GDN variants (incl. chunk16)
# then on M2, under gpu-turn:
bash gdnc16_m2_build.sh    # from the lab artifacts dir
```

## Artifacts (on close)

- `artifacts/GdnChunk16/2026-10-05-build/build.log` — golden_rebuild output.
- `artifacts/GdnChunk16/2026-10-05-build/doctest/{ship,chunk16,chunk16_nz}.SHA256`.
- `artifacts/GdnChunk16/2026-10-05-build/micro_{ship,chunk16}.log`.
- `artifacts/GdnChunk16/SHA256SUMS` — sealed on lane close.