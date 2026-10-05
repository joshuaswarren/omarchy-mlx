# oMLX parity rows A4/A5/A6/A8/A14/A15/A20/A21 (OmlxCache lane)

Owner: OmlxCache (worker). Pre-registered 2026-10-05; twin notebook entry at
`entries/OmlxCache/<ts>-omp-studio-local-omlx-rows-a4-a21-harness.md`.

## What is here

- `run_rows.sh` — the real-run harness. One self-contained leg per gpu-turn
  ticket (each ≤ 12 min), `--list` prints legs + acceptance. Runs on the M2
  (`ssh jw14m2-linux`) or jwm1 after `packaging/omlx-linux/install.sh`
  provisioned the venv.
- `devbox_protocol_test.py` — dev-box protocol tests (SUPPORTING EVIDENCE
  ONLY; this box has no accelerator). Current: 4/4 PASS.
- `patches/0004-linux-memory-monitor-active-memory-probe.patch` (in
  `packaging/omlx-linux/patches/`, auto-applied by install.sh's glob) — the
  A15 gate patch: probe `mx.get_active_memory` directly instead of keying the
  baseline on the custom-kernel gate.

## Pre-registered acceptance per row (real pass = M2/jwm1 run + receipt)

| Row | Real pass requires | Leg |
|---|---|---|
| A4 | 4-concurrent greedy == single digest on Qwen3-4B with fences OFF. Preconditions: wheel built from main >= `48ce2b25f` (rope_rms_norm per-request array offsets; NOT in golden wheel 60f80d2 — need the v0.7.28 cut) AND a FRESH mlx-lm 0.31.3 install in the venv (stale pre-patched qwen3.py silently keeps the v0.7.27 fence; RopeNormBatch gotcha). Leg asserts fence-marker absence from the installed qwen3.py, then digests. | `a4` |
| A5 | Shared ≥512-token prefix, two concurrent requests with divergent suffixes: both byte-identical to sequential references (copy-on-write honest) + zero corruption lines; paged cache active (log). | `a5` |
| A6 | `--paged-ssd-cache-dir` + small hot cache: long-prefix request, safetensors blocks on disk, server restart, same-prefix request returns IDENTICAL digest (state survived restart); restore log line if emitted. | `a6` |
| A8 | SpecPrefill settings on (`specprefill_enabled`, draft = Qwen2.5-0.5B snapshot, threshold 256, keep_pct 0.5) on DeepSeek-Coder-V2-Lite (MoE, 64 experts): long request served + specprefill scoring lines in server.log. Outputs vs off are RECORDED, not asserted (sparse prefill approximates by design). | `a8` |
| A14 | Under `--memory-guard-gb 3`: admitting Qwen3.8-2B (1.04G) with Qwen3-4B (2.21G) + pinned Qwen2.5-0.5B (0.27G) resident evicts the LRU unpinned model (Qwen3-4B), keeps the pinned one; `ttl_seconds: 75` auto-unloads the idle model; pinned model still serves. | `a14` |
| A15 | Enforcer startup line with real ceiling + `Baseline memory set: <nonzero>` (patch 0004; on omarchy `mx.get_active_memory` exists even with the custom-kernel gate closed) + `/api/stats`. | `a15` |
| A20 | TurboQuant KV via settings API at bits=8 and bits=4 vs off on a long prefix: HTTP 200s, turboquant lines in server.log (wrap actually applied), digests + memory recorded; bits=8 equality vs off noted honestly. | `a20` |
| A21 | MoE expert offload at 25% residency on DeepSeek-Coder-V2-Lite: greedy output BIT-IDENTICAL to the resident run (routing unchanged by construction) + real completion + offload log lines. | `a21` |

## Status

- Dev-box protocol tests: DONE (4/4, this receipt, `devbox_protocol_test.py`).
- Real runs: PENDING M2 FREE (w73 window) — queue below.

## M2 execution queue (after Main announces M2 FREE)

```sh
# 0) one-time venv refresh to the post-fix state (inside the first ticket)
scp -q ~/.config/superpowers/worktrees/mlx-omarchy/OmlxCache/receipts/2026-10-05-omlx-cache-rows/run_rows.sh \
  jw14m2-linux:/tmp/omlx-rows/run_rows.sh
#    v0.7.28-class wheel (A4 precondition), sha VERIFIED on the dev box 2026-10-05:
#    mlx_omarchy-0.32.4.dev202610050725+5c15fba-cp314-cp314-linux_aarch64.whl
#    sha256 68bb536fa4879ff6367b35617e800e9a3cfe1d35563458474ebdf33e2ce4c92d
#    staged at /tmp/v0728-wheel/ here; scp to the M2 and assert the sha there
#    before pip-install (draft release v0.7.28, tag 5c15fbaea = rope B>1 fix)
ssh -o ConnectTimeout=8 jw14m2-linux
#   install.sh (packaging/omlx-linux) as landed, then:
#   /tmp/omlx-home/.venvs/omlx/bin/pip install --no-deps 'mlx-lm==0.31.3'
#   re-apply mlx-lm patch series (scripts/apply-mlx-lm-patches.sh)
#   + install the v0.7.28-class wheel (>= 48ce2b25f) when Release0728 delivers it
# 1) tickets (gpu-turn FIFO, one leg each, <= 12 min):
/tmp/omlx-rows/run_rows.sh a4     # needs the post-fix wheel
/tmp/omlx-rows/run_rows.sh a15    # 0004 receipt (enforcer+baseline)
/tmp/omlx-rows/run_rows.sh a5
/tmp/omlx-rows/run_rows.sh a6
/tmp/omlx-rows/run_rows.sh a20    # after a15 (uses its accounting context)
/tmp/omlx-rows/run_rows.sh a21
/tmp/omlx-rows/run_rows.sh a8
/tmp/omlx-rows/run_rows.sh a14    # longest (85 s idle wait), run last
# artifacts land in /tmp/omlx-rows/<leg>-<ts>/ on the M2; copy back with
# scp -r jw14m2-linux:/tmp/omlx-rows/<leg>-* artifacts/OmlxCache/
```

## Dependencies / asks routed to Main

1. v0.7.28-class wheel (main >= `48ce2b25f`) from Release0728 — A4 cannot run
   on the golden wheel (60f80d2 predates the fix).
2. gpu-turn slots per the queue above after M2 FREE.
