# jwm1 v0.7.31 receipts (generated from logs, 2026-10-07 ~19:5xZ)

Host: M1 base (G13G), uname 7.1.12-2-12.2-sep-ARCH (12.2). Queue empty, no holds.

## g17 patch-series (jwm1-g17-summary.log + per-run logs)

```
g17 mlx-lm 0.31.3: apply1 rc=0 idempotent rc=0
g17 mlx-lm 0.32.0: apply1 rc=0 idempotent rc=0
G17_JWM1_DONE ok=4/4
```

- Series detection: 0.31.3 -> "patches", 0.32.0 -> "patches/mlx-lm-0.32" (both series present in the tag tree).
- Patch counts: 0.31.3 log = 18 applied/patched lines; 0.32.0 log = 21 (the 0.32 line has 3 python patchers reporting "patched:").
- Idempotent second runs: rc=0 both (already-applied paths clean).
- Note: the first attempt used the gates-dir copy of the apply script, which expects scripts/ beside it (layout 1); rc=2 x4 (patcher path missing). Rerun used the repo checkout copy (layout 2, scripts/ + patches/ at repo root): rc=0. Not a wheel or patch defect; driver script path bug.

## Install check (v0.7.31 wheel, clean venv, CPU only)

```
a2f8c83e5c5f635d  (wheel sha256 prefix, matches SHA256SUMS)
mlx.core import OK, version: 0.32.4.dev202610071347+9b5c938
CPU matmul OK: 4096.0
INSTALL_CHECK_JWM1_DONE rc=0
```

## w7K final look

docs/lanes/pr-followups/ does not exist on origin/main (git ls-tree origin/main docs/ has no lanes/), nor on jwm1, the M2, or jw16 checkouts. No v0.7.31 entry found anywhere greppable (receipts, docs, AGENTS.md). w7K's verdict is NOT on disk; the review is still pending with w7K directly.
