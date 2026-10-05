# A20 server repro after the alloc-cache fix

- Apple Silicon M2 (Honeykrisp Vulkan heap; host redacted), lane tree at the standard
  golden-clone location on top of `/var/tmp/golden-wheel` (golden at
  origin `23d1ca6e`, warm `60f80d2` build, plus the heap-budget
  wired_limit patch `8f8d32956` ported in, plus the AllocCache patch
  on top).
- Wheel: `mlx_omarchy-0.32.4.dev202610050436+23d1ca6-cp314-cp314-linux_aarch64.whl`
  (incremental build).
- Same Qwen3-4B 651-token / max_tokens 96 workload, same harness
  (`/var/tmp/a20_server_probe.py` + `/var/tmp/a20site/sitecustomize.py`),
  same one-model server config (TurboQuant 8-bit, memory guard off,
  no-cache, no-hf-cache, max-concurrent 1).
- The omlx venv was reinstalled against the new wheel (it had
  `0.32.4.dev202610050725+5c15fba` before; now
  `0.32.4.dev202610050436+23d1ca6`).

## Pre-registered pass criterion

HTTP 200 chat-completion response, non-empty completion, control file
`request_success`, no `async_eval_failure` event in `memory.jsonl`.

## Result

PASS.

- `control.txt`: `request_success`.
- `response.json`: `status 200`, `elapsed_s 105.15`, `body 770` bytes,
  `id chatcmpl-baaa462c`, real completion text.
- `memory.jsonl`: 86 events, 0 `async_eval_failure` events, 0
  `async_eval_failure_after_clear_cache` events.
- `cache_memory` profile: 0 → 43.88 GiB (just before the first
  `gc_limit_` release at step 26) → 0.84 GiB (release) → climbs
  again to 42.22 → 0.87 → 15.43 (last `client_marker` after success).
  This is the bounded cycle the gate forces: each decode step adds
  ~1.6 GiB of KV-cache allocation; once `active + cache + size`
  would cross 95 % of `total_memory` (47.13 GiB → ceiling 48.07 GiB),
  `BufferCache::release_cached_buffers` evicts the LRU before the
  next `vkAllocateMemory`.
- `active_memory` 2.34 GiB peak, `peak_memory` 2.34 GiB (unchanged
  from the failing baseline — the bug was a cache leak, not active
  growth).
- The OOM-retry path in `malloc` (release cache + retry once on
  `VK_ERROR_OUT_OF_DEVICE_MEMORY` / `VK_ERROR_OUT_OF_HOST_MEMORY`)
  was not triggered: the gc_limit gate freed enough headroom for
  every step to succeed on the first try.

## Artifacts

- `a20-single-server-memory.jsonl`: 86 events, 103,421 B.
- `a20-single-server-response.json`: HTTP 200 + body + elapsed, 903 B.
- `a20-single-server-body.json`: the 651-token request body, 3,365 B.
- `control.txt`: `request_success`, 15 B.
- `SHA256SUMS`: sha256 of the four captured files above.

## Replay command

```sh
cd /var/tmp && HOME=/tmp/omlxcache-home setsid nohup \
  /usr/local/bin/gpu-turn -m 10 -- /var/tmp/run_a20.sh \
  >>/var/tmp/a20-runner-new.log 2>&1 </dev/null & disown
```

`run_a20.sh` just sets `HOME` and execs the staged
`a20_server_probe.py` under the rebuilt omlx venv.
