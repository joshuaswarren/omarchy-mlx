# oMLX parity rows A4/A5/A6/A8/A14/A15/A20/A21 (OmlxCache lane)

Owner: OmlxCache. Pre-registered 2026-10-05; notebook entry at
`entries/OmlxCache/20261005T102000Z-omp-studio-local-omlx-rows-a4-a21-harness.md`.

## Harnesses and limits

- `run_rows.sh` is the real-run harness. It supports `--list`; each row uses a separate gpu-turn ticket of at most 12 minutes. Artifacts are written to `/tmp/omlx-rows/<leg>-<ts>/`.
- `devbox_protocol_test.py` ran 4/4 protocol checks; supporting evidence only. Dev box has no accelerator capable of production Vulkan shaders.
- `packaging/omlx-linux/patches/0004-linux-memory-monitor-active-memory-probe.patch` lets A15 probe `mx.get_active_memory` independently of the custom-kernel gate.

## Row test plan and current result

A real pass requires a real model request on M2 or jwm1 and a linked receipt. Metal-only: no row here is intrinsically Metal-only; device-independent control plane and the Linux/Vulkan backend are exercised. A4 needs the post-fix rope-rms-norm wheel, not Metal. A15 requires its Linux probe compatibility patch.

| Row | oMLX surface (file:line) | Real pass observation / cheapest harness | Status and receipt |
|---|---|---|---|
| A4 | `omlx/settings.py:328`; `omlx/scheduler.py:6,11,1904,1922` | Qwen3-4B; four concurrent greedy requests equal single-request digest, fences off. `run_rows.sh a4`; requires wheel with `48ce2b25f` and fresh mlx-lm 0.31.3. | PASS `a4-20261005T122140Z`; see artifact folder. |
| A5 | `omlx/cache/paged_cache.py:490` | Two divergent continuations sharing ≥512-token prefix equal sequential references; confirm paged cache and no corruption. `run_rows.sh a5`. | PASS `a5-20261005T133636Z`; see artifact folder. |
| A6 | `omlx/cache/paged_ssd_cache.py:1623`; CLI in `omlx/settings.py:1404-1414` | Request with paged SSD cache, restart, same-prefix digest equal; safetensors persistence/restore log. `run_rows.sh a6`. | PASS `a6-20261005T133711Z`; see artifact folder. |
| A8 | `omlx/api/openai_models.py:370-374`; `omlx/model_settings.py:261-264`; `omlx/scheduler.py:10148` | Long uncached prompt above threshold with SpecPrefill scoring and successful sparse prefill, no fallback/error. `run_rows.sh a8`. | Linux/Vulkan BLOCKED on DeepSeek and Qwen3-4B RoPE-dimension errors (`a8-20261005T135842Z`, `a8-20261005T140444Z`). Stock macOS Qwen3-0.6B + self-draft passed same 845-token prompt (see `a8-macos-stock-qwen3-0.6b/`); model/draft differ, so A8 remains open pending same-model comparison.
| A14 | `omlx/admin/routes.py:309,384,2713,6837` | Three-GB memory guard; LRU evicts unpinned model, preserves pinned, TTL unloads idle model; pinned model still serves. `run_rows.sh a14`. | PASS `a14-20261005T135217Z`; see artifact folder. |
| A15 | `omlx/memory_monitor.py:36,313`; patch 0004 | Real enforcer ceiling and nonzero baseline plus stats endpoint. `run_rows.sh a15`. | PARTIAL: enforcer and stats work, but baseline not set; installed `set_baseline_memory` has no caller. Receipt `a15-20261005T133604Z`. |
| A20 | `omlx/model_settings.py:363-365`; `omlx/scheduler.py:787-891` | Valid completion JSON with TurboQuant off/8/4-bit settings and conversion logs; compare digests honestly. `run_rows.sh a20`. | PASS on cached Qwen2.5-0.5B (`a20-20261005T135541Z`). Qwen3-4B conversion hits Vulkan OOM (`a20-20261005T133736Z`). |
| A21 | `omlx/model_settings.py:46-53,407-408` | Supported cached MoE; resident and 25%-offload greedy valid completion JSON must have equal digest. `run_rows.sh a21`. | BLOCKED: supported Qwen3.8-35B activates offload but Vulkan request fails on unsupported `uint32` custom kernel; DeepSeek type rejected by settings API. Receipts `a21-20261005T133946Z`, `a21-20261005T140717Z`, `a21-20261005T140846Z`. |

Artifact root: `artifacts/OmlxCache/omlx-cache-rows/`; log and request/response artifacts are retained there. Failures are not passes. The A21 harness validates both completion bodies before comparing digests to prevent empty-response false equality.

## Minimal rerun commands

After the harness and install payload are staged on the M2 and a gpu-turn ticket is granted, invoke one row per ticket:

```sh
/tmp/omlx-rows/run_rows.sh --list
/tmp/omlx-rows/run_rows.sh a4
/tmp/omlx-rows/run_rows.sh a5
/tmp/omlx-rows/run_rows.sh a6
/tmp/omlx-rows/run_rows.sh a8
/tmp/omlx-rows/run_rows.sh a14
/tmp/omlx-rows/run_rows.sh a15
/tmp/omlx-rows/run_rows.sh a20
/tmp/omlx-rows/run_rows.sh a21
```

Do not download models. Use cached models only. Do not edit the parity matrix directly; row updates and receipt paths go to `agent://ParityMatrix`.