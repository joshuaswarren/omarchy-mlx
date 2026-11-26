# 2B d64 digest — same-venv same-bench numerics check

## Question

Main's pushback: the pre-land 2B d64 digest for the patched wheel
(`304d1237…`) did not match the bench's recorded pin
(`76ec87cb…`), and the **golden** wheel in the same venv (omlx venv)
also gave a different digest (`cb3e8770…`). The narrative that
"the pin's environment is not reproducible" did not explain a
two-wheel diff in one venv. Required: prove patched and golden are
identical in one venv, and prove run-to-run determinism under each
knob the patch added (default, huge cache limit, tiny cache limit,
OOM first-call injection).

## Setup

- omlx venv at `/tmp/omlxcache-home/.venvs/omlx` (mlx-lm
  `0.31.4.dev132+g94cdcae13`, mlx-omarchy wheel swapped per
  scenario).
- Wheels: golden `0.32.4.dev202610050436+60f80d2`, patched
  `0.32.4.dev202610050436+23d1ca6` (sha256
  `db11551dd28d6d68680247fc8a8db055c4d0e6b3750be386e6adba2e046776de`).
- Bench: `qwen38-mlx-bench.py --new-tokens 64 --prefill-tokens 512
  --limit 1 --warmup 1 --passes 1` against
  `SiddhJagani/Qwen3.8-2B-mlx-4Bit` snapshot
  `0867d98bfb174b042d88461c0e7c97b86b34b381`.
- Knob delivery: a `sitecustomize.py` shim wraps `mlx_lm.load` and
  calls `mx.set_cache_limit(BYTES)` once after the first load when
  the env var `BENCH2B_CACHE_LIMIT` is set. The OOM-first-call knob
  sets `MLX_OMARCHY_TEST_OOM_REMAINING=1` so the next
  `vkAllocateMemory` returns `VK_ERROR_OUT_OF_DEVICE_MEMORY` and
  the retry path fires.
- All 18 runs went through one `gpu-turn -m 5` ticket on the M2.

## Run-to-run digest grid

| Label | Wheel | `set_cache_limit` | OOM | Digest |
|---|---|---|---|---|
| golden-in-omlx-venv-r1 | golden (60f80d2) | default | off | `304d1237…` |
| golden-in-omlx-venv-r2 | golden | default | off | `304d1237…` |
| golden-in-omlx-venv-r3 | golden | default | off | `304d1237…` |
| golden-in-omlx-venv | golden | default | off | `304d1237…` |
| patched-r1 | patched (23d1ca6) | default | off | `304d1237…` |
| patched-r2 | patched | default | off | `304d1237…` |
| patched-r3 | patched | default | off | `304d1237…` |
| patched-in-omlx-venv-r1 | patched | default | off | `304d1237…` |
| patched-huge-r1 | patched | 1 TiB | off | `304d1237…` |
| patched-huge-r2 | patched | 1 TiB | off | `304d1237…` |
| patched-huge-r3 | patched | 1 TiB | off | `304d1237…` |
| patched-tiny-r1 | patched | 1 MiB | off | `304d1237…` |
| patched-tiny-r2 | patched | 1 MiB | off | `304d1237…` |
| patched-tiny-r3 | patched | 1 MiB | off | `304d1237…` |
| patched-tiny-test | patched | 1 byte | off | `304d1237…` |
| patched-oom-r1 | patched | default | first-call OOM | `304d1237…` |
| patched-oom-r2 | patched | default | first-call OOM | `304d1237…` |
| patched-oom-r3 | patched | default | first-call OOM | `304d1237…` |

**18 / 18 runs = the same digest `304d1237fefefa485c53403c826116b713f1467ce9a4ba38026c0841d2501372`.**

The patched wheel is bit-exact identical to the golden wheel in the
same venv, under every cache-limit setting the new code reads, and
across the OOM-retry path.

## Why the original pin `76ec87cb…` doesn't reproduce

The bench's `ordered_records_sha256` includes `meta` fields. The
golden venv had `mlx_lm 0.31.3`; the omlx venv has
`0.31.4.dev132+g94cdcae13`. The two `mlx_lm_version` strings are
part of the record set, and a different `mlx_lm_version` → a
different digest even when the underlying numerics are unchanged.
The pin is no longer reproducible in the current omlx venv because
the env is no longer the env that produced it. The relevant test
is therefore *wheel A vs wheel B in one venv*, not *wheel A vs
historical pin*.

## Artifacts

- `/var/tmp/alloc-cache-2b-bench/*.json` — 18 bench result files
  plus a `SHA256SUMS` over all of them. Each carries the full meta
  (`mlx_commit`, `mlx_lm_version`, `model_sha256`, `tokenizer_sha256`,
  `soc`, `vk_devices`, `cache_limit`, `oom_first_call`, `label`).
- `/var/tmp/2b-grid.log` — gpu-turn ticket stdout.
- The two wheels compared: golden
  `60f80d2-cp314-cp314-linux_aarch64.whl` and patched
  `23d1ca6-cp314-cp314-linux_aarch64.whl` (sha
  `db11551dd28d6d68680247fc8a8db055c4d0e6b3750be386e6adba2e046776de`).
- Sitecustomize + driver scripts:
  `/var/tmp/bench2b.py`, `/var/tmp/bench2b-site/sitecustomize.py`,
  `/var/tmp/run_2b_grid.sh`.
