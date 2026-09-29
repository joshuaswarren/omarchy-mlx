# jw16 conv-silu decode boundary reduction (2026-09-29, DecodeGap5)

## Change

F1 of the dependent-boundary reduction ladder: fold `nn.silu(conv_out)` into
the `gdn_conv_update` dispatch via a new `activate=True` keyword on the
`GdnConvUpdate` primitive. Removes one dependent dispatch per GDN layer
(18/token of 315 on the window-1 census). Lands on top of the chip-keyed
GDN state tile + LSE/argreduce walk prefetch landed at 0e8c66307 -- the
chip arms are bit-equal and jwm1's legacy arm is unchanged. Implemented as:

- `overlay/mlx/backend/omarchy/shaders/gdn_conv_decode.comp` gains the
  `params.flags & 1u` epilogue: `STORAGE_TYPE conv_out = STORE_VALUE(accumulator)`,
  then `sig_b = STORE_VALUE(1/(1+exp(-v)))`, `out = STORE_VALUE(v * LOAD_VALUE(sig_b))`
  -- bit-exact to the composed swiglu SILU_ONLY chain (conv out rounds to
  storage first, then sigmoid and product each round exactly like the
  composed `sigmoid -> multiply` nodes).
- `overlay/mlx/backend/omarchy/primitives.cpp`: `GdnConvUpdate` gains
  `activates()` (params.flags bit 0).
- `patches/mlx-gdn-conv-decode.patch`: regen to apply on top of upstream mlx.
- `patches/mlx-lm-conv-silu.patch`: in `mlx_lm/models/qwen3_5.py`, change
  `mx.fast.gdn_conv_update(conv_state, qkv, self.conv1d.weight)` to `..., activate=True`
  and drop the standalone `nn.silu(conv_out)` in the decode branch.
- `scripts/apply-mlx-lm-patches.sh`: new `apply mlx-lm-conv-silu.patch`
  step (ordered AFTER the qwen35-gdn-conv step; before gated-norm).
- `tools/dg_bitcheck_conv.py`, `tools/win1-dg4.sh`, `tools/win2-dg5.sh`:
  DecodeGap4/5 gate scripts (raw-output SHA-256 per case + cell driver).

Commits: 51dd63fa9 (change), 636dc2627 (gate scripts),
13b1f04c2 + af5d70213 (rebased onto main 0e8c66307),
5b438c4b2 + fbdb6da62 (DecodeGap5 window script final); merged
into main as fast-forward from 0e8c66307.

## Gates (jw16, boot 8c3d0b5c, gpuwin windows W2 + battery, llm-inference
restored health 200 + completion probe finish=length after every window)

- **Bit-equality** (whole-model, n=5 interleaved): `dg_bitcheck.py` 3084
  rows in candidate vs serving venvs -> byte-identical JSON except the
  recorded wheel version. `dg_bitcheck_conv.py` (conv+silu): every row
  identical across both venvs (the production venv already carries the
  DecodeGap4 conv-capable wheel 1848+51dd63fa9; with both wheels
  implementing the same epilogue the cross-venv identity check is a
  stronger discriminator than the planned ABSENT probe -- see notebook
  entries/Jw16DecodeGap5/20260929T1904Z-jw16-conv-silu-boundary.md
  W2 observations). Fused == ref within candidate on all four cases
  (rand / special / bigstate / specialw).
- **Whole-model records** (full cells): all four digests identical at
  identical token counts:
    d64  ctl 98.33 / cand 100.29 / ctl2 98.99 -- all c84b3e7af6401c64...
    d128 ctl 98.79 / cand 100.11           -- all 07c515e0338b9108...
    d256 ctl 97.59 / cand 98.80 / ctl2 97.52 -- all c6aabbf0a51de38d...
    d512 ctl 93.67 / cand 94.87            -- all 5c120987f0e5869d...
  Candidate above the control min-max at every length (+1.2%..+2.0%);
  d64 +1.97% outside [98.33, 98.99] -> the >=+1% gate PASSES.
- **Race battery**: `gated_battery.sh 8` interleaved 10x10 contract
  pairs, cand = `dg5-venv` (X=1: differs by venv only), ctl = serving
  venv. All 16 runs returned `ordered_records_sha256 = dbf704971617`
  (no divergence beyond control's); per-run medians cand 100.00-100.34
  vs ctl 98.84-99.00 -- no overlap.
- **Census** (fused stream, diag wheel 0.32.3.dev202609291921+diag.5b438c4b2,
  driver 128 tokens warmup 16, profile_analyze + analyze_decode_profile):
  steady 301 dispatches/token (297 expected with 18 silu removed + 4
  resolved against the control-arm census; full named per-kernel table
  in w2/census-{analyze,edges}.txt). Race battery log: 32 files in
  w2-battery/. A control-arm census (same driver, diag wheel from main
  0e8c66307 on the composed stream) was staged but the btrfs reclaim
  after the journal vacuum has not yet returned the workspace to its
  pre-window room -- the artifact stash at `/tmp/dg5-wheels-jw16/` and
  the dg5-ctl-diag-venv (wheel 0.32.3.dev202609291955+diag.0e8c66307
  installed, no activate patch, composed stream) survive for the next
  continuation window.

## Deploy

```
bash /var/tmp/appbar/deploy_wheel.sh /var/tmp/dg5/wheels/mlx_omarchy-0.32.3.dev202609291923+fbdb6da62-cp314-cp314-linux_aarch64.whl
patch -d /var/tmp/v072-venv-fused/lib/python3.14/site-packages/mlx_lm/models < /var/tmp/dg5-build/patches/mlx-lm-conv-silu.patch
sudo systemctl start llm-inference
curl health 200 + completion finish=length + d64 digest = c84b3e7af640...
```

Deploy will be the next continuation. Rollback = the deploy_wheel.sh
backup venv (`$VENV.pre-<UTC>`) + the serving venv's pre-conv
qwen3_5.py backup. Restoration proof after deploy: same as window
restore proof (health 200 + completion probe + d64 digest =
c84b3e7af640). NOTE: production /var/tmp/v072-venv-fused already runs
the DecodeGap4 conv-capable wheel 1848+51dd63fa9 with the COMPOSED
mlx_lm patch (no activate call site) -- a half-deploy not recorded in
that entry. The deploy here closes the gap: new wheel fbdb6da62 +
activate=True patch at the same step.

## Mac ratios (this lane's candidate vs DecodeGap4 window-1 macOS legs)

Linux candidate vs DecodeGap4 window-1 macOS decode medians:
    d64  100.29 / 180.01 = 0.5571x (44.3% gap)
    d128 100.11 / 179.28 = 0.5584x (44.2% gap)
    d256 98.80  / 178.28 = 0.5542x (44.6% gap)
    d512 94.87  / 177.12 = 0.5356x (46.4% gap)

Within-session ratio cand/ctl (Linux-only, this window, paired):
    d64  1.0197, d128 1.0133, d256 1.0124, d512 1.0128 -- +1.2..+2.0%
    consistent with the ~220 us/token budget from removing 18 dependent
    boundaries at 12.2 us each (DecodeGap4).

## Rollback

- Runtime: switch the wheel back to the deploy backup venv
  (`/var/tmp/v072-venv-fused.pre-<UTC>`), restart llm-inference.
- Code: revert fbdb6da62 (this lane's tip after the ff-merge); the
  DecodeGap4 chip-switch ab1a183c2 / 0e8c66307 stay.

## Artifacts

apple-silicon-lab/entries/Jw16DecodeGap5/20260929T1904Z-jw16-conv-silu-boundary.md
apple-silicon-lab/artifacts/Jw16DecodeGap5/w2/ (21 files, SHA256SUMS verified)
apple-silicon-lab/artifacts/Jw16DecodeGap5/w2-battery/ (32 files)