# M2 battery — v0.7.30 STOPPED at g1 (release blocker), 2026-10-07

- Driver: single (m2-battery-final.sh, RESET=1), stopped cleanly at Main's
  order ~12:12Z (TERM driver group + g2 server; verified SURVIVORS: none;
  gpu-turn queue + lock holders empty; logs kept under
  ~/v0.7.30-gates/v0.7.30-gate-logs/).
- Host state: released for TensorFold (~60 min). Nothing of ours resident
  (verified). uname -r 7.1.12-2-12.1-sep-ARCH (aurora 12.1). Wheel
  sha256 0a3ab728a801ab6576b98b94d8e2cf979d10f85b1d70be668708de4be243b4c3.
  Tag cf71ea276.

## g1-clean-install RC=1 wall=20s — RELEASE BLOCKER (root cause confirmed)

Install trace tail (~/v0.7.30-gates/v0.7.30-gate-logs/g1-install.log):

```
==> Applying mlx-lm serve patches (conv-ring off unless MLX_OMARCHY_CONV_RING=1)
mlx-lm patch series: patches
...
applied: mlx-lm-gated-delta-raw.patch
patched: <gate-home>/.local/share/mlx-omarchy/venv/lib/python3.14/site-packages/mlx_lm/models/cache.py
unrecognized BatchKVCache content in <gate-home>/.local/share/mlx-omarchy/venv/lib/python3.14/site-packages/mlx_lm/models/cache.py; refusing to patch (mlx-lm version mismatch?)
INSTALL_EXIT 1 2026-10-07T12:02:55Z
env: '<gate-home>/.local/bin/mlx-omarchy-chat': No such file or directory
```

Root cause (w7K-reproduced): pristine mlx-lm 0.31.3 exits 1 at
apply-mlx-lm-patches.sh:143 — patch-mlx-lm-kv-maskless.py's anchors no
longer match cache.py after the NEW ssm-maskless rewrite (bc3f3c109/c1adc33c5)
changed make_mask; the two patches were never composition-tested. Pristine
0.32.0: rc=0 (w7K).

## Failure cascade

g2-online-9b: RC=1 — ran UNPATCHED mlx-lm (install died at the patch step);
stopped mid-download at Main's order. All later gates would also have run
unpatched → none count as release evidence.

## Gate-script changes after the cut (record per w7K item 2)

ZERO commits touch scripts/release-gates since cf71ea276. The runbook copies
in receipts/2026-10-06-release-0.7.29/gates/ changed post-tag (landed:
03077abb5, 3a987b098, 8b1a6f1d0, 37040046a, 341600d3c, 0a1470956, 4b4860211,
11b13a342, ef70b8cca, 3961d3acd, af582c147, 53a60b36e, 82166c9f9, b8dd2c6bf).
The v0.7.30 tarball carries the cf71ea276 scripts; the gates run from main.
