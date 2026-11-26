# COS-419: G13C vocabulary-tail single pair

## Result: retain the existing default

On the Apple M1 Max, enabling PR 269's chunked vocabulary-tail default measured **40.4 → 40.2 decode tok/s (-0.50%)**, with identical token digests. Exactly one real before process and one real after process ran. The candidate was reverted; this receipt changes no backend code or kernel default. The pair does not establish a speedup.

| Metric | Before | After |
|---|---:|---:|
| Decode tok/s | 40.4 | 40.2 |
| Prefill tok/s | 301.9 | 303.1 |
| First-token latency, ms | 1468.1 | 1462.2 |
| Model load, seconds | 2.26 | 2.08 |
| Actual prompt / generated tokens | 443 / 128 | 443 / 128 |
| Token digest | `9a142db605d5939d` | `9a142db605d5939d` |
| Real measured processes | 1 | 1 |

The raw numerical results, token timestamps, device identity, conditions and loaded-library provenance are in [before.json](before.json) and [after.json](after.json). Model load is outside the generation timer. The small prefill and first-token differences carry no speedup claim.

## Baseline and change

[PR 269](https://github.com/joshuaswarren/omarchy-mlx/pull/269) was already merged at `48908ca1825d5c80feb5edaa1eb04c5c410fe9c0`. Its diff and comments were read. It defaults the existing chunked ArgMax/ArgMin and LogSumExp routes on G14C (M2 Max), with live flag overrides. Its [M2 Max gains](../2026-10-23-vocab-tail-g14c/README.md) are not COS-419 measurements.

Before: source `7337e3faa`, with the same backend as that merged base. Candidate: `3ef2819a3`, adding G13C to `vocab_tail_chunked_default`, updating the existing part test and flag documentation. No shader or arithmetic change. Revert: `237a4bae1`. The G13C default remains off.

Both native Release wheels were built from separate clean source/build directories with `MLX_BUILD_OMARCHY=ON`, `MLX_BUILD_CPU=OFF`, Metal/CUDA/tests/examples/benchmarks off, and two build jobs. Stamps are `0.32.4.dev202610231341+7337e3f` and `0.32.4.dev202610231356+3ef2819`. Compiled versions matched their installed distributions, RECORD hashes matched loaded files, and the two `libmlx.so` hashes differ. Wheel SHA-256: before `59ac887e027c5db8993f0d9cdf7414356280d92168e1581244ca9a4742982c65`; after `91997a56d971f76d991b6169fd354c1e54b385cddd8ec1d428b51d35e1877a71`.

## Hardware, model and command

Same physical 16-inch Apple M1 Max, 32 GPU cores, 64 GB memory; Vulkan device `Apple M1 Max (G13C C0)`. The pre-run hardware capture recorded Linux kernel `7.1.12-2-12.6-sep-ARCH`; Honeykrisp Mesa `26.3.0-devel (git-69bde15340)`, Vulkan API 1.4.362. Driver-library SHA-256 `40282811c6e14ef991f72936bb675008cb4d6b3199b1785bc34c7976ab89c425`. All CPU policies used `performance`; this session changed neither governors nor services.

Model: `mlx-community/Qwen3.5-9B-MLX-4bit`, snapshot `938d8919941c6e7efd3c7150eff7fe9d12afa631`, affine 4-bit/group-64. Shard SHA-256 values: `a68b87558c6ef43f74c2bd63ce7e9092ceddc3101f3def0030774bae5f42aadd` and `b0a770bf8469c7f3f18756a0e0283f1c1174344a83e059a4e483f6af4907352d`. Index SHA-256 `dd023913fb87cfdae27fb11dcf695117c925833796ccac3c64117d6652d8ff1e`.

The exact same model command ran on both builds, with `RUN_ROOT=$HOME/src/omarchy-mlx-cos419/.work/cos419-run` and `MODEL=$HOME/.cache/huggingface/hub/models--mlx-community--Qwen3.5-9B-MLX-4bit/snapshots/938d8919941c6e7efd3c7150eff7fe9d12afa631`:

```sh
timeout 600 "$RUN_ROOT/venv/bin/python" "$RUN_ROOT/llmstep.py" "$MODEL" 477 128
```

`llmstep.py` is the unchanged [PR 269 receipt harness](../2026-10-23-vocab-tail-g14c/scripts/llmstep.py): one eight-token warmup, then one measured greedy generation without a sampler. The `477` argument produces 443 prompt tokens with this tokenizer; both outputs record that actual count. The fixed text is in the linked script. Both chunk flags were unset. Both runs used offline model access, `TOKENIZERS_PARALLELISM=false`, the packaged Honeykrisp ICD and the same patched mlx-lm series through `PYTHONPATH=$HOME/.local/share/coreglass/mlx-lm-series-main/lib/python3.14/site-packages`. Its `generate.py` SHA-256 is `5b077a35686d98cd14caa3ac0063da1e8870985c96e7bda1a86802d50e604188`.

## Timing, competing work and checks

Before ran at 13:52:58–13:53:07Z; after at 14:12:41–14:12:49Z on 2026-10-23. Each held the nonblocking `/tmp/m1-gpu.lock`. Read-only process/load and GPU-client inspections preceded the pair; no other GPU benchmark client was observed. Two earlier load-gate refusals launched no model process. Initial load limit was 1.3. Start loads were 1.19 and 1.16. A CPU/ANE benchmark was active during the before run and had ended before the after run: these are different competing-load conditions, so the pair is not an isolated causal comparison. No process, session or service was stopped or changed. The single-run protocol precludes another baseline or sweep.

`thermal_zone0` read 29.9°C before and 29.5°C after; its sensor identity is unknown, so these are not GPU-temperature measurements. The successful fresh model processes reopen the Vulkan device. Both no-CPU wheel builds completed; both numerical token digests match. Receipt arithmetic, links, provenance, model/prompt/generation equality and checksums are checked separately. The failed performance candidate does not qualify a default change.

The 14:21:51Z post-pair capture found a different boot and kernel `7.1.12-2-20261013.18-egpu-ARCH`. This session issued no reboot or kernel change. Kernel and boot identity were not captured inside each measured process, so their equality across the pair is unverified. The post-state kernel logs cannot be differenced against the earlier boot as though they were one boot. Together with the changed ANE load, this precludes an isolated performance conclusion.

All authored changes remained in the assigned worktree. Native build artifacts used its assigned pathname on the target, not another checkout. The initial x86-only preflight was superseded by the authorized Apple Silicon execution route. No other worktree or installed/shared runtime was edited.

## Ledger target selection

[ledger-audit.json](ledger-audit.json) ranks all twelve decode/prefill median pairs for the six filenames shared by `receipts/2026-10-09-linux-vs-macos/{linux,macos}/`: GPT-OSS-20B, gemma-4-26B-A4B, gemma-4-31B, Qwen3.5-9B, Qwen3.6-35B-A3B and Qwen3.8-27B. The largest raw timing deficit is GPT-OSS prefill, **1.862 versus 581.120 tok/s**, a 312.095× per-token-time ratio. Its decode digests differ, so it is not a token-matched parity result. These are historical ledger values, not this pair.

Among the audited six-model ledger's decode pairs with equal output digests, the largest deficit is Qwen3.5-9B on the M1 Max, **37.052 versus 59.378 tok/s**, Linux at 62.400% of macOS; that motivated this PR 269 follow-on. See the [Linux](../2026-10-09-linux-vs-macos/linux/qwen35-9b.json), [macOS](../2026-10-09-linux-vs-macos/macos/qwen35-9b.json) and [benchmark definitions](../../docs/BENCHMARKS.md). This trial does not fix the raw GPT-OSS prefill deficit or claim fresh macOS parity.
