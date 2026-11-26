# 2026-10-01-jw16-llm-icd-env

Drop-in `/etc/systemd/system/llm-inference.service.d/icd-env.conf` with `Environment=AGX_SCHED_COMPUTE=1` restores llama.cpp pp512 to parity with the old ICD (`9d949d4-vec2`) on jw16 after MesaLand landed the `1432df0196-transfer` system ICD.

**Identified cause:** the new ICD's compute nosched pass (commit `8fa3a53dc98` / Mesa `agx_compile.c:3554`, kill switch `AGX_SCHED_COMPUTE=1`). The local-offset fold pass h81 is *not* the cause — disabling it alone makes pp512 *worse*.

## Evidence

Single gpuwin window, 5 arms, same binary (llama-bench built from llama.cpp `c1d0e7a004015f23bc0233470b747b596f29b264`, the running server build), same model (`qwen38-27b-q4_k_m.gguf`), `-p 512 -n 128 -r 3 -t 8`. Greedy completion fixed prompt/temp=0/seed=42/n_predict=128.

| arm | driverInfo | pp512 (t/s) | tg128 (t/s) | text sha256 |
|---|---|---|---|---|
| new_default (no env, system ICD `1432df0196-transfer`) | `Mesa 26.3.0-devel (git-1432df0196)` | 48.47 ± 0.01 | 8.01 ± 0.06 | `5f3801d3...` |
| new_sched (`AGX_SCHED_COMPUTE=1`) | `Mesa 26.3.0-devel (git-1432df0196)` | **53.41 ± 0.01** | 8.03 ± 0.04 | `5f3801d3...` |
| new_fold (`AGX_LOCAL_FOLD_OFF=1`) | `Mesa 26.3.0-devel (git-1432df0196)` | 47.95 ± 0.01 | 8.00 ± 0.08 | `5f3801d3...` |
| new_both (both kill switches) | `Mesa 26.3.0-devel (git-1432df0196)` | 53.02 ± 0.01 | 8.03 ± 0.03 | `5f3801d3...` |
| old_ctl (`VK_DRIVER_FILES=/var/tmp/mesaland-icd/asahi_icd_ctl9d949d4.json`) | `Mesa 26.3.0-devel` (no git suffix) | 53.03 ± 0.00 | 8.04 ± 0.02 | `5f3801d3...` |

Greedy completion byte-identical across all 5 arms (`cmp -s` confirmed).

## Drop-in

`/etc/systemd/system/llm-inference.service.d/icd-env.conf`:

```
[Service]
Environment=AGX_SCHED_COMPUTE=1
```

`sudo systemctl daemon-reload && sudo systemctl restart llm-inference.service`. Verified live: `systemctl show llm-inference.service -p Environment` returns `Environment=LD_LIBRARY_PATH=... AGX_SCHED_COMPUTE=1`; the running llama-server (pid 8939) inherits `AGX_SCHED_COMPUTE=1` via `/proc/<pid>/environ`. Health 200 + authenticated completion `finish_reason=length`.

## Post-restart probe (live system ICD + drop-in)

`llama-bench` pp512/tg128 x3: **pp512 = 53.41 ± 0.01 t/s**, tg128 = 7.91 ± 0.23 t/s — matches the in-window `new_sched` arm exactly. old_ctl was 53.03 → restored within 1% (`+0.72%`).

Live completion test (200 max_tokens): `completion_tokens=200, prompt_tokens=66, cached_tokens=42, finish_reason=length` — service healthy and serving.

## What's untouched

- MLX stack: keeps MesaLand ICD defaults (no env override there). MesaLand's pp cells measured pf512 +1.0%, pf2048 +0.9% with both passes active — MLX prefill tolerates the compute nosched pass even though llama.cpp's own shaders do not.
- System ICD json unchanged: `/usr/share/vulkan/icd.d/asahi_icd.aarch64.json` → `/usr/local/lib/libvulkan_asahi.so.1432df0196-transfer`. Rollback still = `sudo cp /var/tmp/asahi_icd.aarch64.json.pre-1432df0196-transfer.bak ...`.
- Wheel untouched (`0.32.3.dev202610010525+1e7cf5c45` in `/var/tmp/v072-venv-fused`).
- Local-offset fold pass stays on (it actually helps llama.cpp by ~1% — see `new_fold` arm).

## Artifacts

`~/.local/share/apple-silicon-lab/artifacts/Jw16LlmIcd/20261001-jw16-llm-icd-env/` (27 files: 6 bench + 6 completion.json + 6 text + 6 identity + 6 server.log + dropin-icd-env.conf; SHA256SUMS 27/27 verified).

Notebook entry: `~/.local/share/apple-silicon-lab/entries/Jw16LlmIcd/20261001T111144Z-jw16-llm-service-icd-env.md`.

## Coordination

Held GPU window twice for PMP lane: once during AnePmp2's variant boot (held ~5 min, AnePmp2 confirmed clear), once during AnePmp3's Phase 1b+2 cycles (their variant boot id `6014af1d-85b5-4cb0-80fa-87d35795d531` came up and reverted before drop-in install).