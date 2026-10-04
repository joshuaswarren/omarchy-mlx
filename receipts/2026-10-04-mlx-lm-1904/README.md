# mlx-lm #1904 backport receipt

Branch: `agent/mlx-lm-1904` (rebased on origin/main 4744cda7e; v0.7.27 published from 6edd258f8 before this branch was pushed).

Upstream: ml-explore/mlx-lm `caed1943d38ecb58f6e466a72bed98310c9d45a5` (#1904), merged 2026-10-01. The upstream change only calls `json.loads` for string arguments and leaves object arguments unchanged. The backport is `patches/mlx-lm-tool-call-arguments.patch` (0.31.3 series) with an identical copy in `patches/mlx-lm-0.32/`; mlx-lm remains under its upstream MIT license and this patch adds no copied license text.

## Behavioral proof

The model-free regression test (`tests/test_mlxlm_tool_call_args.py`) extracts the real `process_message_content` function from the installed `mlx_lm/server.py` AST. It tests object-valued arguments and the existing JSON-string path without importing MLX or constructing a model/tokenizer.

RED, pristine mlx-lm 0.31.3:

```text
MLX_LM_VENV=/tmp/mlxlm1904-red .../python -m unittest tests/test_mlxlm_tool_call_args.py -v
ERROR: test_object_arguments_are_preserved
TypeError: the JSON object must be str, bytes or bytearray, not dict
String-argument test: ok
```

GREEN, patched mlx-lm 0.31.3:

```text
Ran 2 tests in 0.021s
OK
```

## Fresh install and repository contracts

A clean venv was created with `python3 -m venv`, followed by `pip install --no-deps mlx-lm==0.31.3`. `scripts/apply-mlx-lm-patches.sh <venv>` applied the new patch and all existing default patches (exit 0), and both parser regression tests passed against the installed patched module (repeated on a second clean venv after the apply-script change).

`python3 -m unittest tests.test_install_sh_contract tests.test_serve_bootstrap`: 41 tests passed, including both `PatcherCoverageTests` cases and the installer patch-fetch contract.

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
