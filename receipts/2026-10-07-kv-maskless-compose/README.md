# kv-maskless series composition fix (v0.7.30 g1-clean-install blocker)

2026-10-07. Lane: KvMasklessFix. Dev box: x86 container, CPU-only; no GPU used.

## Symptom

`scripts/apply-mlx-lm-patches.sh` exits 1 on a pristine pip mlx-lm **0.31.3**
tree (the version `install.sh` pins, `MLX_LM_VERSION=0.31.3`, and every vendor
lock carries via `packaging/requirements-lock.in`), found by the v0.7.30 M2
g1-clean-install gate:

```
patched: .../mlx_lm/models/cache.py (mlx-lm 0.31.3)
unrecognized BatchKVCache content in .../mlx_lm/models/cache.py; refusing to patch (mlx-lm version mismatch?)
RC=1
```

ssm-maskless (immediately before it) applies fine on 0.31.3 — the earlier
suspicion that bc3f3c109/c1adc33c5 rewrote the matched body is falsified; the
kv anchors never matched stock 0.31.3 at all.

## Root cause

`patch-mlx-lm-kv-maskless.py` (and `patch-mlx-lm-kv-host-offset.py`,
`patch-mlx-lm-batch-greedy-head.py`) anchor on the **mlx-lm 0.32**
BatchKVCache/generate API but `apply-mlx-lm-patches.sh` invoked them
unconditionally in both series. Per-anchor count against stock 0.31.3's
BatchKVCache body (pristine wheel, `cache.py` sha256 `819ed95dcbf75565…`,
matching the independent w7Q audit):

```
CLASS_OLD          count=1 OK
INIT_OLD           count=1 OK
PREPARE_OLD        count=0 FAIL <- patcher refuses
FINALIZE_OLD       count=0 FAIL <- patcher refuses
MASK_OLD           count=1 OK
FILTER_OLD         count=0 FAIL <- patcher refuses
EXTEND_EMPTY_OLD   count=1 OK
EXTEND_OLD         count=1 OK
```

0.31.3's older class has no prepare/finalize right-padding mirrors and filters
via a device sync (`min_left_pad = self.left_padding.min().item()`); upstream
reworked the class in 0.32, so there is nothing to port the anchors onto.

## Decision: gate, not port

- Shipped venv = 0.31.3 (`install.sh:25,388`, `pip install --no-deps`); the
  0.32 series exists for the oMLX lane (`packaging/omlx-linux/install.sh`,
  mlx-lm `94cdcae`).
- The managed mlx-lm serve route is single-concurrency (docs/serve.md), so the
  batched BatchKVCache decode path is not what 0.31.3 ships.
- 0.31.3 consequence without the three patchers: fallback, not failure —
  upstream `make_mask` keeps returning the array mask (composed SDPA decode),
  offset stays a lazy device array (RoPE host join), and 0.31.3 has no
  `GenerationBatch`. The omarchy wheel's gates self-guard on array provenance.
- v0.7.29 control: its series has zero references to the three patchers
  (`grep -c` = 0) and applies RC=0 to pristine 0.31.3 — the breakage arrived
  with the v0.7.29..v0.7.30 batched-decode lane (2dcd24ba9, f7c3168a6,
  6e1e8e13d, 1a80bc79e), so no earlier release was affected.

## Fix

`scripts/apply-mlx-lm-patches.sh`: run the three patchers only when
`$SERIES == patches/mlx-lm-0.32`; otherwise print an explicit skip line.
Refusal-on-unknown-content in every patcher is untouched (it is what caught
this). `docs/kernel-flags.md` documents the scope and the 0.31.3 consequence.

## Evidence

Fail-before (pre-fix main `b8dd2c6bf`), full series on pristine 0.31.3:

```
mlx-lm patch series: patches
... (6 .patch files apply; ssm-maskless patches 0.31.3 fine)
patched: .../cache.py (mlx-lm 0.31.3)
unrecognized BatchKVCache content in .../cache.py; refusing to patch (mlx-lm version mismatch?)
RC=1
```

Regression test before the fix (`tests/test_apply_mlxlm_series.py`):

```
1 failed in 0.64s
```
(0.31.3 leg: series rc 1 at kv-maskless)

Pass-after, pristine trees rebuilt from `mlx-lm==0.31.3` and `mlx-lm==0.32.0`
wheels (pristine sha256 cache.py / generate.py: 0.31.3
`819ed95dcbf75565…`/`270778ad53eaca55…`, 0.32.0
`c96cb169fd5e0ec4…`/`4a3bf57f5679dac7…`):

```
=== pristine 0.31.3, run1 ===  RC1=0
kv-maskless/kv-host-offset: 0.32 series only; skipped on patches
batch-greedy-head: 0.32 series only (no GenerationBatch); skipped on patches
=== 0.31.3 run2 === 5 "already patched:" lines, RC2=0
=== pristine 0.32.0, run1 ===  RC3=0
=== 0.32.0 run2 === 8 "already patched:" lines, RC4=0
```

Suite (8 cases: both lines × apply/idempotent/markers/CPU-mask-behavior, with
CPU mlx available):

```
8 passed in 1.87s
```

Behavior legs assert the shipped semantics end to end: 0.32 patched tree
`BatchKVCache([0,0]).make_mask(1) is None` and `make_mask(2) is not None`;
0.31.3 patched tree keeps the upstream array mask for both.

Neighboring contract tests (patch(1) contract, installer audit table, patcher
ordering) still green:

```
35 passed, 5 skipped in 2.46s
```

(skips are the mlx/transformers-gated behavior suites that need an Apple-side
ambient stack; they are unchanged and unaffected by the gate.)

v0.7.29 control on pristine 0.31.3:

```
RC=0   (16 applied/patched lines, no 0.32-only patchers present)
```

## Files changed

- `scripts/apply-mlx-lm-patches.sh` — SERIES gates for the three 0.32-only
  python patchers, with explicit skip lines.
- `docs/kernel-flags.md` — KV_MASKLESS / KV_HOST_OFFSET rows: 0.32-series-only
  scope + 0.31.3 consequence.
- `tests/test_apply_mlxlm_series.py` — new full-series regression (pristine
  pip trees, both supported mlx-lm lines, idempotency, per-line markers, CPU
  mask behavior; skips a line only if pip cannot fetch that wheel).

Repro scratch (venvs < 1 MB each: wheel unzip, no interpreter) was throwaway
and removed; the same trees rebuild via `pip download --no-deps mlx-lm==<ver>`
+ unzip (what the test does).
