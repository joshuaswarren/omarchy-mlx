# Serve loop on a 13-inch M1: the rotating-cache mask and the per-token logprob read (2026-10-18)

This receipt backs the measurements cited by `scripts/patch-mlx-lm-rotkv-maskless.py` (PR 232) and `patches/mlx-lm-0.32/mlx-lm-server-skip-logprob-read.patch` (PR 233): 76.67 -> 71.49 ms per token, +568 dispatches and +5.18 ms per token for the mask, 3.48 ms per token for the logprob read, a 19 percent gap between the server's generator loop and `bench_decode`, and the end-to-end served result 13.03 -> 14.64 tok/s. Everything below is read from the files in `data/`; `SHA256SUMS` covers `data/` and `harness/`.

## Identity

| Item | Value |
|---|---|
| Machine | 13-inch M1 (Vulkan device `Apple M1 (G13G B1)`, printed in every `bench_decode` JSON line) |
| Kernel | `7.1.12-2-20261013.18-sep-ARCH` (printed by every run log) |
| Backend | `mlx-omarchy 0.32.4.dev202610102357+e57cb301` wheel, `libmlx.so` sha256 prefix `9a225264dcf2dbcb` (the `provenance:` lines of the G arms read `verified=match`) |
| mlx-lm | the 0.32.0 tree the wheel runs with, with the repo's mlx-lm patches (the "before" tree); the "after" tree is the same tree with the two patches of PR 232 and PR 233 applied (heads `11ba827fde47bbf692f3a2fcbecf40d209d973bb` and `41149cf8efb1dde9478a47bb5662f413cd3d7215`, merged as `8ffc8e7904f2fe6444b243d6b77c46543e34f3f9` and `a85a048f89a912fea99855ef1cea9569ce3a1be8`); `cache.py` / `server.py` sha256 prefixes of both trees are in `data/before-after/fixab-e4b.out` |
| Model | `mlx-community/gemma-4-e4b-it-4bit`, snapshot `475b9088d29754a3379866cf5aeb6b41acd313c2` (the snapshot directory name; the weights files were not hashed) |
| Governor | `performance` on both policies, read (never written) at the start of every run; P-cluster mean 2988 MHz in every arm (sampler summary lines in the run logs) |
| Not recorded | Mesa version string and driver library hash, GPU and firmware versions, memory use, temperature |

Same boot for all runs of the 13-inch (`boot af2c726b`). The 16-inch result is a second machine: 16-inch M1 Max (G13C), kernel `7.1.12-2-12.6-sep-ARCH`, model `mlx-community/gemma-4-26b-a4b-it-4bit` snapshot `0d77464eeb233a2da68ebf9d7dc4edaac7db956d`, tree `gprp` with its own wheel build (`libmlx.so` prefix `7c9c9fff09bbc988`, different from the 13-inch's).

## What was run

All runs: prompt "Write a short story about a lighthouse keeper who finds a message in a bottle.", 128 generated tokens, greedy (temperature 0), exact token ids hashed (`ids_sha256_16` / `ids_sha16`). Scripts are in `harness/` (sha256 in `SHA256SUMS`); each prints its own tree, hash and load lines first.

- Arm A: `bench_decode.py` through `run_clamped.py thread` (the thread-clamp placement): `stream_generate`, 4-token warm-up, then 128 tokens; rate = 127 token gaps (`decode N tok/s over 127 tokens`), one sample per process.
- Arm G: the `mlx_lm.server` `ResponseGenerator` + `BatchGenerator` driven in-process, no HTTP (`harness/servegen_inproc.py`): decode and prompt concurrency 1, prompt cache 0, an 8-token warm-up request, then request pairs of 32 and 128 tokens; `ts_tps` = (n - 1) / (last token time - first token time) of the 128-token request.
- Arms GM / GL / GS / GML / GMLT: arm G with one candidate cost removed by a driver monkeypatch (`install_flip` in `servegen_inproc.py`): GM no all-True mask array for the sliding-window batch cache, GL no per-token `r.logprobs[r.token].item()`, GS a new thread-local stream, GML both GM and GL, GMLT GML plus phase timers.
- Arms H / HF: the real server (`python -m mlx_omarchy_serve serve`), non-streamed requests, rate = (n_long - n_short) / (wall_long - wall_short) over a 32-token and a 128-token request (`harness/serve_client.py`); H on the before tree, HF on the after tree.
- Dispatch census: `MLX_OMARCHY_TRACE_DISPATCH=1` prints one `[rtmod] DISPATCH kernel=<id> count=<n> gx gy gz` line per dispatch; dispatches per decoded token = (histogram of a longer run - histogram of a shorter run) / decoded tokens added (A: 6 and 10 tokens, 4 decoded tokens added; G and GM: 24 and 40 tokens, 20 added). Kernel ids are the `ComputeKernel` enum of `overlay/mlx/backend/omarchy/compute.h` at commit `e57cb301` (`harness/kernel_names.py`).
- Order and counts: each round is a rotation of the arm list (round r starts at arm r), 3 or 4 rounds, 2 request pairs per G/H process, 15 s pause between processes. The timed scripts wait up to 300 s for the 1-minute load average to fall below 0.5 and abort otherwise; the census script has no load wait. The lab scheduler held a solo window for the timed runs, and every script aborts when an ANE process exists.
- Rule: a gain is REAL when it is at least 3 percent of the bench_decode time per token and every run median of the faster arm is above every run median of the slower arm; spread of run medians above 5 percent is NOISY. The exact rules are the headers of `flip_analyze.py` and `fixab_analyze.py`.

## Results

1. Census, 13-inch, gemma-4-E4B (`data/census-13inch-e4b/`): dispatches per decoded token A 1507.0, G 2096.0, GM 1528.0. In G the 35 sliding-window layers lose the fused `SdpaDecodeNativeBF16Hd256` (id 465: 35 -> 0 per token) and add `CastBF16F32` +140, `CopyGeneralBF16` +140, `MatmulF32` +70, `ElementwiseLiteF32` +70, `SoftmaxF32`, `CastF32BF16`, `SelectBF16`, `LogicalOrBool`, `ClearU32` +35 each; GM returns the fused kernel (+35) and removes the composed set (-568 in total). Token ids G = GM at both lengths. The driver counted `make_mask` returning None 41 and 61 times (3 prefills delegated).
2. Flips, first run (`data/flips-1/`, 4 rounds): tok/s medians A 15.47 (64.65 ms/token), G 13.04 (76.67), GM 13.99 (71.49), GL 13.66 (73.19), GS 13.10 (76.33); gap +12.02 ms (18.6 percent of the bench_decode time), REAL; GM +5.18 ms (43 percent of the gap), GL +3.48 ms (29 percent), GS +0.34 ms (not REAL); ids equal in all arms (`6e98b674dcae2e22`).
3. Additivity (`data/flips-2-additivity/`, 3 rounds): A 15.52 (64.45), G 13.04 (76.67), GM 14.04 (71.24), GL 13.65 (73.26), GML 14.63 (68.33); gains over G: GM +5.43, GL +3.41, GML +8.33 ms (sum 8.84; difference -0.50 ms: additive within the registered 1.0 ms); remainder against bench_decode +3.88 ms, REAL. Phase timers (GMLT, per decoded token): wall 68.32 ms, `BatchGenerator.next()` 68.31 ms of which `mx.async_eval` 67.03 ms, `mx.eval` about 0, host rest 1.28 ms, outside `next()` 0.02 ms. Whether the `async_eval` call waits for the GPU or records on the host is not separated by these counters.
4. Before and after (`data/before-after/`, 3 rounds): A 15.48 (64.59 ms/token), G 13.06 (76.60), GF 14.63 (68.37), H 13.03 (76.73), HF 14.64 (68.30). Served H -> HF +8.42 ms per token (13.03 -> 14.64 tok/s, 1.123x), REAL; in-process G -> GF +8.22 ms (1.120x), REAL. Ids equal across A, G, GF (`6e98b674dcae2e22`); served text digest equal in H and HF (`30e02a01d2c805d4`). Registered verdict: the served gain is REAL.
5. Census, 16-inch M1 Max, gemma-4-26B-A4B (`data/census-16inch-26b/`, untimed): dispatches per decoded token A 1707.0, G 2141.0 (+434), GM 1733.0 (-408 against G); the 25 sliding-window layers lose `SdpaDecodeNativeBF16Hd256` (-25) in G and get it back in GM; ids G = GM. No rate was measured on this machine.

## Limits

One boot and one prompt per machine, one model per machine, 3 or 4 round medians per arm; the arm order is a rotation, not a balanced design. The 13-inch gap to `bench_decode` after both patches is still 3.7 ms per token (HF 68.30 against A 64.59); the cause is not measured. The mask-patch gain on the 16-inch and the M2 is not measured (the 16-inch census shows the same dispatch pattern; the M2 has no sliding-window model cached). The patched numbers are for greedy requests that do not ask for logprobs.
