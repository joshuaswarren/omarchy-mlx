# jwm1 (T8103): early first submit around prompt processing + first token (2026-09-29)

Host record (python graph build + `async_eval` return) is fully exposed in synchronous paths: on jwm1 a synchronous decode token is python build 3.6-4.3 ms + record/submit 14.3-17.0 ms + GPU 25.6-26.0 ms (macOS exposes +4.8 ms per synchronous token, jwm1 Linux +21.7 ms; macOS same-die window receipts/2026-09-29-jwm1-macos-micro-window3 in ane-linux-experiments). kBatchNodeBudget 4096 (encoder.h) keeps a graph on one submit, so the GPU cannot start until recording ends.

## Measurements (jwm1, live venv or candidate venv of the same wheel family, interleaved arms, one GPU-lock hold each, digests identical in every run)

| Change | sync token ms | L=11 forward ms | qwen38 TTFT s | decode64 tok/s |
|---|---:|---:|---:|---:|
| baseline (4096) | 45.2-45.5 | 158-174 | 0.1837-0.1844 | 41.60-41.62 |
| constant budget 128 (H49) | 34.7 | 135 | 0.1778 | 41.01 (-1.5%) |
| constant budget 64 | 35.2 | 134 | 0.1761 | 39.98 (-3.9%) |
| first batch 128, always (H52) | 36.1-36.6 | 151-156 | 0.1603 (-12.7%) | 41.12 (-1.15%) |
| first batch 128, only when GPU idle at graph start (H53) | 36.2-36.6 | 153-156 | 0.1746 (-4.9%) | 41.48 (-0.27%) |
| first batch 128 only around prompt + first token (this patch, H54) | n/a (model() path) | n/a | **0.1590 (-13.8%)** | **41.48 (-0.34%)** (d128 40.73 vs 40.87) |

Idle gating loses most of the TTFT gain because the prompt graph starts while the previous generate_step's last look-ahead token is still finishing (trace: `state=2 last_completion=77 drained=76`). The patch instead scopes the setting to the phase that pays the exposed record cost: `generate_step` sets `MLX_OMARCHY_BATCH_FIRST=128` before prompt processing and pops it right after the first token.

## Other probes in the same investigation (not shipped)

- Buffer-cache limit 32 -> 256/1024/2048 MB: synchronous token 45.0 -> 38.5 ms (exact), decode/prefill/TTFT/Parakeet unchanged.
- Wake-up latency (jw16 poll driver) -1 ms and CPU frequency (busy spinners) -3 ms on the synchronous token: not the cause.
- Native stack sampling of synchronous tokens: 64% in the GPU-wait ioctl; no libvulkan_asahi frames on top among non-wait samples; allocator create/map/unmap visible.

Remaining gap on this cell: macOS TTFT 0.1248 s (same-die paired cell, receipts/2026-09-28-jwm1-macos-parity-legs): Linux now 1.27x latency (was 1.47x on the same protocol run), still a LOSS.
