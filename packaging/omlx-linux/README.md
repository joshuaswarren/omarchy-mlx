# oMLX-on-Linux compat layer (omarchy-mlx)

Run jundot/omlx's open-source LLM inference server on top of the
omarchy-mlx wheel (Linux/Vulkan). Ships as a small patch series against the
pinned oMLX release, an installer, and AST/pytest guards — never a source
fork. Every guard that depends on a macOS-only API degrades honestly (no
fake values, no fallthrough to `default` for things we know).

- Upstream: <https://github.com/jundot/omlx>
- Pinned: `v0.7.0` (commit `4d4f5a280bc1739ba2cf39c1cee44fd5cc89cb40`,
  Apache-2.0; full text copied to `omlx-LICENSE-Apache-2.0.txt` here).
- mlx-lm pin: `94cdcae13b266c337bcaca09b97b9c5a9c0e2cde` (post-0.31.3
  main with 0.32 API; resolves to `0.31.4.devN`). The repo's
  `patches/mlx-lm-0.32/` series applies cleanly in order on this commit
  (verified 2026-10-04, 10/10 with no offsets); the installer's
  `apply-mlx-lm-patches.sh` selects that series automatically via the
  `StopSequences` marker.
- omarchy-mlx work: pinned tree at
  `joshuaswarren/omarchy-mlx` (`mlx.lock` → MLX 0.32.3 @ `9c3d35571a`;
  mlx-omarchy wheel distribution name).
- Hardware tested on: Apple M2 Max (T6021) Linux, kernel aurora, with the
  local Honeykrisp Vulkan ICD.

## What's in here

```
packaging/omlx-linux/
  install.sh                              # user-level venv installer
  ast_import_scan.py                      # AST guard scan for the patched tree
  test_compat_guards.py                   # pytest, runs on the dev box
  README.md                               # this file
  omlx-LICENSE-Apache-2.0.txt             # upstream license (recorded)
  patches/
    0001-linux-hardware-proc-meminfo.patch
    0002-linux-cli-cache-limit-total-memory.patch
    0003-linux-enforcer-no-wired-limit-log.patch
```

## Audit (what the patches actually do, file-by-file)

| oMLX site | macOS assumption | Linux disposition | Patch |
|---|---|---|---|
| `omlx/utils/hardware.py: get_total_memory_bytes` | sysctl `hw.memsize` then `mx.metal.device_info()["memory_size"]`; **falls back to fake 8 GB on a 96 GB M2** | read `/proc/meminfo` `MemTotal` first (unified-memory model: GPU draws from the same RAM, so total system RAM is the budget) | 0001 |
| `omlx/utils/hardware.py: get_chip_name` | sysctl `machdep.cpu.brand_string` | read `/proc/device-tree/model` (`"Apple M2 Max"` on Apple Silicon under Linux); otherwise honest `Unknown (<sys> <machine>)` | 0001 |
| `omlx/utils/hardware.py: parse_chip_info` | unrecognised → `("M1", "")` (a fake) | unrecognised → `("Unknown", "")` | 0001 |
| `omlx/utils/hardware.py: get_os_version` | macOS only | `Linux <release>` | 0001 |
| `omlx/utils/hardware.py: get_max_working_set_bytes` | Apple Metal cap | 75 % of unified system RAM (Apple's working-set heuristic on Apple Silicon) | — |
| `omlx/utils/hardware.py: device_total_memory` (new) | n/a | helper accepting either `"memory_size"` (Metal) or `"total_memory"` (omarchy Vulkan) | 0001 |
| `omlx/cli.py: server-start cache-limit` | `mx.device_info()["memory_size"]` | also try `"total_memory"` (omarchy key) via the helper | 0002 |
| `omlx/process_memory_enforcer.py: _apply_wired_limit` (else branch when sysctl & metal cap both 0) | silent debug log | one INFO log on non-darwin stating the guard budgets against unified system RAM and `mx.set_wired_limit` is a no-op on this backend; darwin branch untouched | 0003 |
| `mx.metal.is_available()` (≈25 patch modules) | `True` on Metal hosts | exists in the omarchy wheel (`no_metal.cpp` stub); returns `False`; every fast route self-gates with `hasattr(mx.fast, ...)` and falls back to composed paths — bit-identical output, slower (not a correctness regression) | — |
| `mx.set_wired_limit` (engine/dflash.py + enforcer) | per-process Metal cap | omarchy backend: honest no-op returning `0` (`mlx/backend/omarchy/allocator.cpp:414`) | — |
| `mx.device_info()` | Metal keys (`memory_size`, `max_recommended_working_set_size`, `architecture` per-family) | omarchy keys (`device_name`, `driver`, `total_memory`, `unified_memory`, `architecture ∈ {honeykrisp, vulkan}`); missing Metal keys return 0 from `.get()` | 0001, 0002 |
| `omlx/utils/proc_memory.py` | `libproc.dylib` ctypes for `phys_footprint`/`task_info` | `# if sys.platform == "darwin":` gated; returns 0 honestly elsewhere (circuit-breaker semantics, "fail open") | — |
| `omlx/utils/system_sampler.py` (cluster) | `host_processor_info` via libsystem | darwin-gated; cluster routing degrades | — |
| `omlx/cluster/*` (discovery, probe, transport, node_role, jaccl, system_socket_proxy, launch, pairing) | sysctl / system_profiler / ioreg / ifconfig / scutil / VM stats | each site `sys.platform == "darwin"` gated; single-host serving doesn't import any of them | — |
| `omlx/admin/ms_downloader.py` | modelscope at module level | unguarded module import → `try/except` in caller (server.py:2422) catches the ImportError and silently disables ModelScope downloads | — |
| `omlx/engine/{stt,sts,tts,vlm}.py` | `mlx_audio`, `mlx_vlm` inside functions | function-body guarded (`` in package, scanner sees them as `deferred (function body)`) | — |
| `omlx/patches/dflash_*.py` | `dflash_mlx` at module level | the installer's AST check runs with `--allow patches` because the dispatch is gated on `model_type` config branches that never fire for a Qwen3.5-2B text load | — |
| `omlx/custom_kernels/*` | prebuilt `*.metallib` / `*.dylib` | optional package; loader reports unavailable and never raises | — |
| `apps/omlx-mac/Sources/**` (Swift menubar app) | AppKit / NSStatusItem / Metal capture | mac-only by design, never installed (the wheel is Python-only) | — |

## Install

```sh
# from an omarchy-mlx checkout
OMARCHY_ROOT=$(pwd)

# The mlx-omarchy aarch64 wheel is built once per release window by the
# OmarchyDistributed lane and installed into a shared venv
# (e.g. /var/tmp/shared-omarchy-venv). Until Main announces that venv is
# ready, other lanes build dev-box-only — do NOT build a competing wheel
# on the M2 in parallel. Once announced, point --mlx-wheel at the shared
# wheel file, or copy it out of the shared venv's dist-info tree.
OMLX_WHEEL=/var/tmp/shared-omarchy-venv/dist/mlx_omarchy-*aarch64.whl
ls -1 "$OMLX_WHEEL" 2>/dev/null || echo "wait for shared-wheel announcement"

# install into a throwaway venv with a throwaway HOME
$OMARCHY_ROOT/packaging/omlx-linux/install.sh \
    --mlx-wheel "$OMLX_WHEEL" \
    --venv /tmp/omlx-venv \
    --home /tmp/omlx-home
```

What the installer does, in order:

1. Resolves `OMARCHY_ROOT` (the omarchy-mlx checkout you ran it from) and
   refuses to run without `scripts/apply-mlx-lm-patches.sh` and `patches/`.
2. Clones `https://github.com/jundot/omlx` to `--omlx-dir`, checks out
   `v0.7.0`, verifies the resolved commit equals the recorded pin
   `4d4f5a28`, and refuses if the work tree is dirty.
3. Applies `patches/*.patch` in lexical order (each first run through
   `git apply --check`; reverse-checked as already-applied safe).
4. Runs `ast_import_scan.py` on the patched tree with `--allow patches`:
   any unguarded module-level import of a watched package the install
   does not provide fails the build.
5. Creates `--venv` (defaults to `~/.venvs/omlx`) and upgrades pip.
6. Installs the omarchy mlx wheel, then writes a constraints file pinning
   `mlx-omarchy==<wheel version>`. There is **no** `mlx` pin: the omarchy
   wheel's distribution name is `mlx-omarchy`, and it must never be
   shadowed by upstream Metal mlx from PyPI.
7. Installs `mlx-lm` at `94cdcae` with `--no-deps` (its `mlx` dependency
   would otherwise pull the upstream Metal wheel).
8. Installs the pinned runtime dependency set (transformers>=5.14,<5.18,
   numpy, markitdown, etc.) under the constraints file.
9. Installs `mlx-vlm` at `ea79808` with `--no-deps`.
10. Installs `omlx` itself with `--no-deps` (deps were explicit above).
11. Runs `OMARCHY_ROOT/scripts/apply-mlx-lm-patches.sh` on the venv:
    the script auto-selects `patches/mlx-lm-0.32/` for the pinned
    commit (10/10 apply in order; the 0.32 series is the right one
    for oMLX).
12. Prints installed versions and the server command.

If the Python interpreter is `>=3.14`, the install uses
`--ignore-requires-python` (oMLX's `requires-python = ">=3.11,<3.14"` is
the upstream CI matrix, not a hard 3.14 incompat — the M2's installed
cp314 omarchy wheel and the rest of the stack load on 3.14, and the
smoke is the empirical check).

## Server smoke (M2, inside a gpu-turn ticket)

```sh
setsid nohup "$HOME/bin/gpu-turn" -m 30 -- env HOME=/tmp/omlx-home \
    /tmp/omlx-venv/bin/omlx serve --host 127.0.0.1 --port 8900 \
        --model mlx-community/Qwen3.5-2B-MLX-4bit 2>/tmp/omlx-server.log &

# /v1/models
curl -s --max-time 5 http://127.0.0.1:8900/v1/models

# non-streaming chat
curl -s --max-time 60 http://127.0.0.1:8900/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"mlx-community/Qwen3.5-2B-MLX-4bit","messages":[{"role":"user","content":"hi"}],"max_tokens":40,"temperature":0}' \
  -o /tmp/omlx-nonstream.json

# streaming chat (TTFT, tokens/s)
curl -sN --max-time 60 http://127.0.0.1:8900/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"mlx-community/Qwen3.5-2B-MLX-4bit","stream":true,"messages":[{"role":"user","content":"hi"}],"max_tokens":80,"temperature":0}' \
  -o /tmp/omlx-stream.sse
```

Greedy batched-vs-single digest equality is the numerics gate: same
prompt, one request at a time → save digest; four concurrent requests on
the same engine → save digest; they must match. Prefix-cache reuse is
checked by sending the same long system prompt twice and watching the
second TTFT drop.

## Dev-box checks (no mlx needed)

The compat layer ships two checks the dev box can run today:

```sh
# 1) AST import scan (no warnings required)
python3 packaging/omlx-linux/ast_import_scan.py /tmp/omlx-applied/omlx --allow patches

# 2) Hardware/parser/enforcer-patch unit tests (stub modules only)
python3 -m pytest packaging/omlx-linux/test_compat_guards.py
```

The M2 smoke above is the only check that requires a live accelerator
(gpu-turn ticket) and is not part of the dev-box suite. The receipt
record in `receipts/2026-10-04-omlx-linux/README.md` is what carries
the M2 evidence onto `omarchy-mlx` main.

## License

oMLX is Apache-2.0 (see `omlx-LICENSE-Apache-2.0.txt`). The patches here
are derivative works of oMLX and inherit the Apache-2.0 terms. The
installer and compat tools (`install.sh`, `ast_import_scan.py`,
`test_compat_guards.py`, this README) are also Apache-2.0.