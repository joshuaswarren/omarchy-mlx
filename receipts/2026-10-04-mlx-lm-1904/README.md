# mlx-lm #1904 backport receipt

Branch: `agent/mlx-lm-1904` (kept off `main` until v0.7.27 publishes).

Upstream: ml-explore/mlx-lm `caed1943d38ecb58f6e466a72bed98310c9d45a5` (#1904), merged 2026-10-01. The upstream change only calls `json.loads` for string arguments and leaves object arguments unchanged. The backport is `patches/mlx-lm-tool-call-arguments.patch`; mlx-lm remains under its upstream MIT license and this patch adds no copied license text.

## Behavioral proof

The model-free regression test extracts the real `process_message_content` function from the installed `mlx_lm/server.py` AST. It tests object-valued arguments and the existing JSON-string path without importing MLX or constructing a model/tokenizer.

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

A clean venv was created with `python3 -m venv /tmp/mlxlm1904-fresh`, followed by `pip install --no-deps mlx-lm==0.31.3`. `scripts/apply-mlx-lm-patches.sh /tmp/mlxlm1904-fresh` applied the new patch and all existing default patches. The two parser regression tests passed against the installed patched module.

`python3 -m unittest tests.test_install_sh_contract -v`: 26 tests passed, including both `PatcherCoverageTests` cases. `python3 -m unittest tests.test_serve_bootstrap -v`: 15 tests passed, including the installer patch-fetch contract.

## Re-run caveat

On the already-patched venv, a second full invocation reported `already applied: mlx-lm-tool-call-arguments.patch`, then stopped at the pre-existing `mlx-lm-gated-delta-fast-route.patch` with `patch does not apply (mlx-lm version mismatch?)`. Its shared `gated_delta.py` hunk is subsequently changed by later GDN patches, so the current series cannot reverse-check that first patch after the full series has landed. The new parser patch itself is recognized as already applied. The full-series rerun gate remains open; this backport did not alter unrelated GDN patches or that existing idempotence behavior.

No hardware serve smoke was run; the model-free parser test covers the changed serve parser directly.
