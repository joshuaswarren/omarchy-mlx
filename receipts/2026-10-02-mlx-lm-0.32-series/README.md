# mlx-lm 0.32 patch series (so oMLX runs on the Omarchy stack)

Date: 2026-10-02. Source: this branch, based on origin/main 972772e6c. Upstream: mlx-lm commit
94cdcae13b266c337bcaca09b97b9c5a9c0e2cde, the commit oMLX 0.7.0 pins.

## Why

oMLX needs the mlx-lm 0.32 API (`mlx_lm.generate.StopSequences`), so it cannot run on the 0.31.3 every vendor
lock carries. On 94cdcae, 3 of the 10 0.31.3 patches apply at fuzz 0 (`qwen35-gdn-conv`, `conv-silu`,
`last-logits`), the rope-norm patcher applies, and `patch-mlx-lm-qknorm.py` refuses ("unrecognized content").

## What changed

- `patches/mlx-lm-0.32/`: the series rebuilt on 94cdcae in the same order by `rebuild_series.py` (exact-match
  edits that refuse on any drift; one diff per step).
  - gated-delta fast-route, repeat, raw: upstream moved the dispatch and added `lower_bound` and
    `allow_neg_eigval`. The raw decode route derives g and beta itself, so it runs only for the default
    variant; the others keep the composed g/beta path.
  - qwen35-qk-scaled: the q/k pair moved into `gated_delta.normalize_qk`, which now scales eps by
    `inv_scale**2`. The patch lives there and passes that eps through.
  - greedy-prune, gated-norm (`precise_swiglu` rename), ttft-early-submit: same edits, new context.
  - conv-ring: not ported (experimental, off by default).
- `scripts/apply-mlx-lm-patches.sh`: picks `patches/` or `patches/mlx-lm-0.32/` by the `StopSequences`
  marker, and refuses `MLX_OMARCHY_CONV_RING=1` on 0.32 before it touches the venv.
- `scripts/patch-mlx-lm-qknorm.py`: also patches the 0.32 site, gating the `normalize_qk` call and passing the
  fused epilogue the same scaled eps (`1e-6 * inv_scale**2`).

## Off-device checks

- On stock mlx-lm 0.31.3 with the 0.31.3 series, main's qknorm patcher and this one write identical trees
  (`qknorm_0313_check.sh`).
- On pristine 94cdcae, the apply script reproduces the staged series plus both patchers byte for byte.
- With `MLX_OMARCHY_CONV_RING=1` on 0.32 the script exits 6 and the venv is unchanged.

## Hardware

| | M1 test host | M2 Max test host |
|---|---|---|
| Chip, GPU | Apple M1, G13G B1 | Apple M2 Max, G14C B1 |
| Kernel | linux-aurora 7.1.12-2-7-ARCH | linux-aurora 7.1.13-3-1-ARCH |
| Vulkan | Honeykrisp, Mesa 26.3.0-devel (git-e7631595df) | Honeykrisp, Mesa 26.3.0-devel (git-7faf04c065) |
| mlx-omarchy | 0.32.4.dev202610012048+6cff5ea (omarchy-mlx 0.7.10-1 package) | 0.32.4.dev202610020523+36eb0c9 (wheel) |
| `mlx_provenance.py` | `mismatch` (see below) | `match` |

Model: SiddhJagani/Qwen3.8-2B-mlx-4Bit (Qwen3.5 architecture, 4-bit, group 64), `model.safetensors` sha256
prefix b0d5de688567bf4a, same bytes on both hosts.

The M1 mismatch is a packaging artifact, not a stale wheel: the pacman package ships stripped binaries
(`file`: stripped), `pacman -Qkk omarchy-mlx` reports 0 altered files, and the wheel RECORD holds the unstripped
hashes (libmlx.so on disk 2eada8ef..., RECORD 01af5c3b...). Every packaged install trips the gate. The M2 Max
uses an unstripped wheel and verifies. Lead with the M2 Max numbers.

## Gate: greedy tokens

`gate.sh` runs five fresh processes under the host GPU lock: upstream 94cdcae, the 0.32 series, the series with
`MLX_OMARCHY_GDN_QKNORM_FUSE=0`, then upstream and the series again. Each warms up, then generates 256 greedy
tokens for two prompts (477 and 1,939 tokens). Hashes are SHA-256 of the token lists, first 12 hex.

| | 477-token prompt, 64 / 128 / 256 | 1,939-token prompt, 64 / 128 / 256 |
|---|---|---|
| upstream 94cdcae (both hosts, both repeats) | d156202f9beb / 0d76e1b70abc / d062df33f279 | 3c31f671bb4f / 199c29cc6d58 / 4c99c9e8eeea |
| 0.32 series, qknorm fused or not (both hosts, both repeats) | 97fe4aa909bb / 11d200967f33 / 099cac0987eb | 3c31f671bb4f / 199c29cc6d58 / 4c99c9e8eeea |

The series changes token 11 of the 477-token prompt. `bisect.sh` applies the steps one at a time: tokens stay
equal to upstream through `gated-delta-fast-route-repeat`, change at `gated-delta-raw`, and no later step changes
them again.

This is not a port defect. `raw_0313_ab.sh` builds stock mlx-lm 0.31.3 with the shipping series, with and
without the raw patch, and runs the same prompts on the M2 Max:

| 64 tokens | 477-token prompt | 1,939-token prompt |
|---|---|---|
| 0.31.3 + shipping series | 97fe4aa909bb | 3c31f671bb4f |
| 0.31.3 + shipping series without raw | 4dd46bc0eda7 | 3c31f671bb4f |
| 94cdcae + this series | 97fe4aa909bb | 3c31f671bb4f |

So the 0.32 series emits exactly the tokens the shipping 0.31.3 stack emits. The raw decode route is not
token-exact against the composed route on 0.31.3 either; the 2026-09-29 bit check compared raw across wheel
builds, not against the composed route. That is a separate finding for the raw route, not part of this change.

## Speed (same runs, one cold request per prompt, warmup first)

| tok/s | M2 Max upstream | M2 Max series | M1 upstream | M1 series |
|---|---|---|---|---|
| prefill, 477-token prompt | 132.4 | 985.0 / 1,070.7 | 63.9 | 397.9 / 396.6 |
| prefill, 1,939-token prompt | 134.0 | 1,582.6 / 1,576.9 | 64.2 | 414.6 / 413.8 |
| decode, 477-token prompt | 54.1 | 91.1 / 90.9 | 28.2 | 41.9 / 42.8 |
| decode, 1,939-token prompt | 43.5 | 64.5 / 65.0 | 25.4 | 36.8 / 36.9 |

Series columns show both repeats. No thermal control beyond running sequentially; no dispatch trace was taken.

oMLX 0.7.0 end to end on the M2 Max (`omlx serve`, `--no-cache`, one cold streamed request, 477 + 128 tokens):
upstream 94cdcae decodes at 51.2 tok/s with a 3,622 ms first token; with this series, 79.3 tok/s and 510 ms.

## Not covered

- Repeat runs beyond two, batching, and long contexts.
- The rerun-idempotency of `apply-mlx-lm-patches.sh`: on a venv that already has the gated-delta patches,
  the reverse check of `fast-route` fails because `repeat` and `raw` edit the same lines. This exists on main
  for 0.31.3 too.

## Review notes (round 1)

- The first push of three patches (`qwen35-gdn-conv`, `conv-silu`, `last-logits`) carried `qwen3_5.py.orig`
  sections: GNU patch wrote backups while `rebuild_series.py` applied the 0.31.3 files, and the step diff picked
  them up. `rebuild_series.py` now passes `--no-backup-if-mismatch` and excludes `*.orig`/`*.rej`. The regenerated
  series applies with no backup files, and the `.py` tree it produces hashes to e0623638535c7446, the same as the
  tree the gates ran on (both exclude `_version.py`, which pip writes and no patch touches). Python never imported
  the stray `.orig` file, so the gate results stand.
- `gate.sh` and `bisect.sh` are kept as they ran. The `1c6d913aded2faec` line in `gate.sh` hashes the patch
  sources, not the applied tree; e0623638535c7446 above identifies the tree. `bisect.sh` ran without `set -e`,
  but each cumulative step is a prefix of the series the apply script (`set -euo pipefail`) applied cleanly on the
  same host minutes earlier, and the last step's tokens equal the full series' tokens.
- `qknorm_0313_check.sh` now runs from any checkout (`bash qknorm_0313_check.sh [BASE]`) and verifies the mlx-lm
  0.31.3 wheel's sha256 (758cfddf...); rerun after the fix: identical trees.
- `install.sh` pins mlx-lm 0.31.3 and downloads only that series. A venv later moved to the 0.32 line now gets a
  clear error that names the missing series directory, and the venv stays untouched.
