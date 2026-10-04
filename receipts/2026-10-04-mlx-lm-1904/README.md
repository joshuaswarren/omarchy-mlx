# mlx-lm #1904 backport receipt

Branch: `agent/mlx-lm-1904` (rebased on origin/main 4744cda7e; v0.7.27 published from 6edd258f8 before this branch was pushed).

Upstream: ml-explore/mlx-lm `caed1943d38ecb58f6e466a72bed98310c9d45a5` (#1904), merged 2026-10-01. The upstream change only calls `json.loads` for string arguments and leaves object arguments unchanged. The backport is `patches/mlx-lm-tool-call-arguments.patch` (0.31.3 series) with an identical copy in `patches/mlx-lm-0.32/`; mlx-lm remains under its upstream MIT license and this patch adds no copied license text.

## Behavioral proof

The regression test (`tests/test_mlxlm_tool_call_args.py`) is self-contained and always runs: it writes the verbatim upstream 0.31.3 `process_message_content` region (byte-identical to the shipped file's hunk context) into a temp tree, applies `patches/mlx-lm-tool-call-arguments.patch` with the installer's exact `patch --strip=1 --forward --fuzz=0` invocation, executes the pristine function (asserting the bug: dict arguments raise `TypeError`, string arguments decode), then executes the patched function (asserting dict arguments pass through and string arguments still decode). No installed mlx-lm is required on the box running the test.

A second layer runs the same assertions against a real installed server module when one is discoverable, honoring `MLX_LM_SERVER_PY` (explicit file), `MLX_LM_VENV` (venv root), or the running interpreter's prefix; the test skips with that reason when no `mlx_lm/server.py` is found.

Observed modes:

```text
default (no installed mlx-lm):    patched-snippet ok, installed test skipped
MLX_LM_VENV=pristine 0.31.3:      FAILED errors=1 (installed tree reproduces the bug)
MLX_LM_VENV=patched 0.31.3:       OK
MLX_LM_SERVER_PY=0.32-line file:  OK
```

## Fresh install and repository contracts

A clean venv was created with `python3 -m venv`, followed by `pip install --no-deps mlx-lm==0.31.3`. `scripts/apply-mlx-lm-patches.sh <venv>` applied the new patch and all existing default patches (exit 0), and both parser regression tests passed against the installed patched module (repeated on a second clean venv after the apply-script change).

`python3 -m unittest tests.test_install_sh_contract tests.test_serve_bootstrap`: 41 tests passed, including both `PatcherCoverageTests` cases and the installer patch-fetch contract.

Full dev-box discovery (`python3 -m unittest discover -s tests -t .`): 1019 tests, 30 skipped, 16 problem tests — all 16 in the bonsai2/laya modules and all caused by `ModuleNotFoundError: No module named 'mlx'` (the aarch64-only wheel is absent on this x86 box). A pristine origin/main checkout fails the same module set with the identical `failures=3, errors=13` tally, so these are pre-existing environment-dependent tests, not regressions from this change; the remaining 1003 tests pass.

## Rerun idempotence fix (was pre-existing at v0.7.27)

Pre-existing confirmation: `scripts/apply-mlx-lm-patches.sh` is byte-identical between v0.7.27 (6edd258f8) and origin/main, and a second full run on a fully patched venv failed before the fix:

```text
already applied: mlx-lm-tool-call-arguments.patch
patch does not apply (mlx-lm version mismatch?): mlx-lm-gated-delta-fast-route.patch
rerun-exit:1
```

Cause: patches later in the series insert into the same `gated_delta.py` / `qwen3_5.py` regions, so earlier patches' exact reverse dry-run no longer matches after the full series lands. The fix adds a third outcome to `apply()`: when forward and reverse both fail, a LATER patch of the same series reverse-matching is content evidence the series already ran, and the patch is reported `already applied (hunks rewritten by later patches in this series)`. The ordered `SERIES_PATCHES` array keeps the evidence forward-looking only, so the loud `patch does not apply (mlx-lm version mismatch?)` error still fires on a tree the series never patched — verified by corrupting two context lines in `generate.py` of a pristine venv:

```text
applied: mlx-lm-gated-delta-raw.patch
patched: .../mlx_lm/models/cache.py (mlx-lm 0.31.3)
patch does not apply (mlx-lm version mismatch?): mlx-lm-greedy-prune.patch
exit 1
```

State matrix after the fix: fully patched rerun exit 0 (11 already-applied lines); fresh install exit 0 with parser tests OK; corrupted tree exit 1 with the loud error. A raw mlx-lm 0.32-line checkout (oMLX pin 94cdcae, which predates #1904) also takes the patch (series `patches/mlx-lm-0.32`, both parser tests OK) and reruns idempotently.

## Hardware serve smoke

Not run. The changed surface is the server's request parser; the model-free AST test executes the real installed `process_message_content` both pre- and post-patch, and the 0.32-line checkout exercise covers the second shipped series. No GPU-bound behavior changed.

## Scope note

The rerun fix touches the shared apply loop; unrelated patch contents were not modified.
