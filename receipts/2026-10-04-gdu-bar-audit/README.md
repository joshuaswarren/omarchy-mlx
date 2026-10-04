# 2026-10-04 GduBar — owner 3-part numerics bar for v0.7.26 9B fused route on jw16 native tiled path

Lane: GduBar (independent audit; did not write the 9B route).
Base: origin/main 1c6c2a8a0 (DispatchFuse addendum 8).

## Tested bytes and setup

GitHub release tag `v0.7.26`, aarch64 cp314 wheel
`mlx_omarchy-0.32.4.dev202610040755+56488ba21-cp314-cp314-linux_aarch64.whl`,
sha256 `ff1e6dfdc1cec28ad7f5500e618ceb72c8ab24e1bb781b60425a9f918640891f`
(matched release SHA256SUMS). Fresh venv
`/var/tmp/gdubar-v0726-venv`; tag installer wheel/dependency/patch steps
reproduced using the v0.7.26 helper scripts and patches; serving venv
untouched. `mlx_provenance.py` reported `verified=match`,
`version_match=true`, mx version `0.32.4.dev202610040755+56488ba21`, GPU
backend. No diagnostic wheel, no `MLX_OMARCHY_GDN_DECODE_TILE` or
`MLX_OMARCHY_GDN_PF` overrides. The only A/B override was
`MLX_OMARCHY_GDN_RAW_REPEAT=0` versus the shipped default ON.

## Native tiled-route proof

`MLX_OMARCHY_TRACE_DISPATCH=1` with the addendum-4 d64 bench protocol
(`--limit 1 --warmup 1 --passes 1 --new-tokens 64 --prefill-tokens 512`):

| arm | tiled `GatedDeltaDecodeBF16` kernel 441 | Untiled 462 | Pf 463 | d64 digest | decode tok/s |
|---|---:|---:|---:|---|---:|
| default / raw route ON | **1776**; grid (32,4) | 0 | 0 | `26d569c86af27c5b…` | 37.45 |
| raw route OFF (`=0`) | 0 | 0 | 0 | `26d569c86af27c5b…` | 28.67 |

The ON count reproduces the expected 1776. Grid gy=4 is Dv/32 for
Dv=128; perrow/untiled would use gy=1. IDs 462/463 are absent. The OFF
log has kernel 441 absent and composed-fallback dispatches instead. Thus
the measurement exercised jw16's native tiled route, not perrow.

## Bar 1 — per-op error vs fp64: PASS

`gdu_fp64_probe.py --capture` monkeypatched the live model's first T=1
call and captured q/k/v/a/b/A_log/dt/state (q/k shapes `(1,1,16,128)`, v
`(1,1,32,128)`, state `(1,32,128,128)`). `--compare` evaluated composed
and fused paths on those same operands against the fp64 numpy reference.

| metric | composed vs fp64 | fused vs fp64 |
|---|---:|---:|
| output max_abs | 0.0016247568256038125 | 0.0016247568256038125 |
| output rel_max | 0.0017625064353020473 | 0.0017625064353020473 |
| state max_abs | 0.01607806977306403 | 0.01607806977306403 |
| state rel_max | 0.0009869427315465351 | 0.0009869427315465351 |

Composed and fused errors are identical in the reported precision; fused
is no worse. Private evidence:
`artifacts/GduBar/gdubar-audit/{gdu-operands.npz,fp64.json,bar1.json}`.

## Bar 2 — greedy >=95% identical OR all divergences gap <0.05: FAIL

10 prompts × 512 tokens. Prefix identity mean 29.43%; exact-position
agreement 31.113%; 10/10 prompts diverged. Six of ten first divergences
have composed gap >=0.05 (each is 0.125); four have composed gap 0.0.
Neither identity measure reaches 95%, and not every divergence is below
0.05.

| prompt | first divergence | composed top-2 gap | fused top-2 gap | exact match % |
|---:|---:|---:|---:|---:|
| 0 | 163 | 0.125 | 0.0 | 31.84 |
| 1 | 8 | 0.0 | 0.125 | 3.12 |
| 2 | 127 | 0.125 | 0.0 | 28.71 |
| 3 | 71 | 0.125 | 0.0 | 20.31 |
| 4 | 256 | 0.125 | 0.0 | 50.78 |
| 5 | 70 | 0.125 | 0.0 | 15.82 |
| 6 | 195 | 0.125 | 0.0 | 38.67 |
| 7 | 342 | 0.0 | 0.125 | 66.80 |
| 8 | 62 | 0.0 | 0.125 | 13.48 |
| 9 | 213 | 0.0 | 0.125 | 41.60 |

Six of ten divergences (60%) have composed gap >=0.05. A 1-bf16-ULP
gap of 0.125 at |logit| around 25–32 is above the explicit 0.05 cutoff
and does **not** qualify as a near-tie. The four 0.0 gaps are ties.

The earlier 100% free-run claim for the tiled path does not reproduce on
the published v0.7.26 wheel for this 10×512 run. This is consistent with
addendum 8's observed long-depth sensitivity, but this run alone does not
establish a universal claim across every build/environment. Private
outputs: `artifacts/GduBar/gdubar-audit/{free-run-gaps.json,bar2.txt}`.

## Bar 3 — route-sensitive S=1 PPL within 0.1%: PASS (49 scored tokens)

`ppl_s1.py` scores token i+1 after feeding token i, one token per model
call, so the decode route runs at each step. The audit fixed a script bug
that had scored token 0 against itself. The four selected prompts were
short after tokenization; the run scored **49 tokens per arm**, not the
192-token cap. Report this limited sample as measured, not as a large
corpus estimate.

| metric | composed | fused |
|---|---:|---:|
| tokens scored | 49 | 49 |
| mean NLL/token | 3.1837740528340244 | 3.181479940609056 |
| fused − composed NLL delta | — | **-0.0721%** |
| within 0.1% | — | **PASS** |

Private output: `artifacts/GduBar/gdubar-audit/ppl-s1.json`.

## Teacher-forced top-1 agreement (additional result)

10 × 512 = 5120 positions, teacher tokens from the composed free-run
arm. Agreement **5095/5120 = 99.5117%**; 25 disagreements, of which 11
have composed top-2 gap <0.05 (44%); max disagreement gap 0.25, mean
0.075. Private output: `artifacts/GduBar/gdubar-audit/tf-agree.json`.

## Script defects found and fixed in audit copies

- `h257_jwm1_runner.sh` forces `MLX_OMARCHY_GDN_DECODE_TILE=0` and
  `MLX_OMARCHY_GDN_PF=0` for its jwm1 forced-legacy run; those overrides
  would invalidate a jw16-native tiled audit. GduBar's window scripts
  omit both.
- `free_run_gaps.py` omitted the fused-side top-2 gap; added it.
- `free_run_gaps.py` used prefix-only identity for its 95% test; added
  exact-position agreement and retained both measures.
- `ppl_s1.py` scored the first prompt token as P(token0|token0); fixed
  the teacher-forcing shift. This biased both arms equally but did not
  represent valid absolute NLL.
- Runner prompt default `~/bench-scripts/qwen38-2b-prompts-10.jsonl`
  was absent; explicit existing prompt path used.
- `analyze_w1.py` expects GPU_PROFILE NDJSON, not present in the release
  wheel; TRACE_DISPATCH logs were counted directly instead.
- `gdu_fp64_probe.py` was checked and did use captured live-model T=1
  operands; no defect found.
- `tf9.py` labeled an unset raw-repeat env as `0`, although the shipped
  v0.7.26 default is ON. Fixed the metadata default to `1` and reran the
  ON arm; output records `gdn_raw_repeat=1`, the expected release version,
  and GPU backend. Agreement result remained 5095/5120.

## Verdict

1. Per-op fused error no worse than composed vs fp64 — **PASS**.
2. Greedy >=95% identical OR every divergence has composed gap <0.05 —
   **FAIL** (29.43% prefix identity, 31.113% exact-position agreement;
   6/10 first-divergence gaps >=0.05; 0.125 at |logit| 25–32 is not a
   near-tie under this bar).
3. S=1 route-sensitive PPL within 0.1% — **PASS** on 49 scored tokens
   (-0.0721%; small sample).

The v0.7.26 tiled route does not meet the complete owner bar because
bar 2 fails, despite passing per-op and measured PPL bars. The earlier

100% free-run claim does not reproduce in this audit.

## Addendum — 2026-10-04 revised scale-aware acceptance rule

The owner replaced the absolute 0.05 free-run cutoff with this rule:
teacher-forced top-1 agreement >=99%, and every free-run first divergence
has composed top-2 gap <= one bf16 ULP at the top-logit magnitude. Per-op
fp32/fp64 error must be no worse than deployed, and route-sensitive S=1
PPL must remain within 0.1%. At logit magnitude [16, 32), one bf16 ULP is
0.125; the former 0.05 absolute cutoff rejected that one-ULP difference.

Re-evaluated without changing the captured measurements, this audit passes
all three bars: per-op fused error equals composed error vs fp64;
teacher-forced agreement is 5095/5120 (99.5117%); the six 0.125 gaps are
one ULP at their recorded magnitudes and the four 0.0 gaps are within one
ULP; measured S=1 PPL delta is -0.0721% on 49 tokens. Prefix identity
remains 29.43% and is not the acceptance measure. The earlier verdict is
the conclusion under the superseded criterion. See docs/numerics-gate.md.
