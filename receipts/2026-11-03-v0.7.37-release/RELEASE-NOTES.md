omarchy-mlx v0.7.37 fixes long-context prefill on models with head_dim 256. Direct mlx-lm use now picks its prefill chunk from the GPU heap headroom. The half-split RoPE builds are limited to the Apple M4 GPU. The tag is main at `06d1038d2aa26ed235e65d2aa68677b9e904d65b` (tag v0.7.37). The figures come from the pull requests and receipts they cite.

## Long-context prefill no longer fails on head_dim 256 models (#333, fixed by #338)

Causal prefill on a model with head_dim 256 failed once a chunk's float32 score buffer reached 2^31 bytes, one byte past the device's storage-buffer limit of 2^31-1. For a 16-head model at the default 2048-token chunk that happens when the cache holds 16384 keys. The chunked attention route already split the work by query rows under a 1 GiB cap, but it only took non-causal calls. It now takes causal calls too, with the same arithmetic per row.

Checked on a 16-inch M1 Max with Qwen3.5-9B 4-bit (the same attention geometry as the model in #333), a 32768-token prompt, the default chunk and no heap override:

| Build | Result |
| --- | --- |
| v0.7.36 wheel, default chunk | fails after about 5 minutes: the dispatch binds 2147483648 bytes against a limit of 2147483647 |
| v0.7.36 wheel, chunk 512 | completes, 64 generated tokens |
| this change, default chunk | completes, the same 64 tokens as the chunk-512 run, peak 11.52 GB |

On a 13-inch M1 with a 2B 4-bit model at 32768 tokens, prefill ran at 263.2 and 262.4 tokens per second on v0.7.36. With the change it ran at 265.3 and 265.2, with the same output. Two runs per arm show no slowdown. They do not show a gain.

Not checked: bit identity at the real cap beyond that 64-token comparison. The 9B run with the mlx-lm serve patches applied. Float16 models. The 13-inch M1 case that opened #333, which needs a larger heap than that machine has free. Details are in #338.

## mlx-lm picks the prefill chunk from the heap headroom (#331)

Direct `mlx_lm` use on a 16 GB machine could not prefill an 8k prompt with a 9B 4-bit model at the default 2048-token chunk. The 1 GiB allocation failed with `VK_ERROR_OUT_OF_DEVICE_MEMORY`. When no chunk is given, the patched `generate_step` now takes the largest of 2048 and its halvings. It picks the one whose estimated scratch fits the heap headroom. An explicit value always wins, and the result never exceeds 2048. The serve path, the wheel and every kernel are unchanged.

On a 13-inch M1 with 15.1 GiB, 5 fresh processes, Qwen3.8-9B 4-bit at 8k: the old default fails. With no chunk given, the patch picked 1024 and all 5 runs completed at 88.14 prefill tokens per second, peak 6.93 GB, decode 10.38.

## Half-split RoPE builds: Apple M4 GPU only by default (#334, by techmore)

v0.7.36 built the half-split RoPE kernels on every part, but the gain was measured only on the Apple M4 GPU. They are now the default there only. Every other part keeps the general builds until it is measured. `MLX_OMARCHY_NO_ROPE_HALF_SPLIT` still wins when set (`1` keeps the general builds everywhere, `0` takes the half-split builds everywhere). The output is bit-identical either way, and nothing changes on the M4.

## Also in this release

- The release gates pass the OpenBLAS include directories when g13 and g15 configure cmake (#336), so those two gates no longer need host preparation.
- The community-data schema identity check runs from the checked-out commit (#337).

## Known cost carried from v0.7.35

Batch 2 on a 14-inch M2 Max still takes about 5 percent longer per step than before the batch kernel change (issue #304). `MLX_OMARCHY_QMV_BATCH=0` removes it for that workload. It also turns the batch kernel off for 3 and 4 rows. No new measurement exists since v0.7.36.

## Known issue: a GPU page fault when a process with queued GPU work is killed (#339, kernel issue omacom/linux-aurora#335)

On Linux, the GPU driver logs a page fault when a process with two compute submissions in flight is killed. A firmware timeout and a recovery follow. The machine recovers each time and later GPU work runs. The killed process does not. An 11-line MLX script reproduces it on a 13-inch M1 in 10 of 10 runs, and a Vulkan-only program with no MLX in it reproduces it in 15 of 15 kills. Nothing faulted in 10 runs with one submission in flight at a time, or in 10 runs where the Vulkan program exits normally. This is what the voice gate g10 logged on v0.7.35 and v0.7.36. The gate stops the assistant with a signal while work is still queued. On this release g10 ran once on the 16-inch machine and logged no fault. The Vulkan-only program puts the fault below MLX, in the GPU driver. It is not new in this release, and the kernel side is tracked in linux-aurora#335.

## How this release was gated

The gates installed from the draft release's assets. The `SHA256SUMS` asset lists the five other files, and every digest in it equals the digest GitHub reports for the uploaded asset. `scripts/verify-release-assets.py v0.7.37` printed `VERIFIED` on the draft. Both wheels were built from clean checkouts of the tag commit.

On a 13-inch M1 (kernel `7.1.12-2-20261013.18-sep-ARCH`) every gate printed its own pass marker. Seven of them (g1, g3, g6, g7a, g7b, g8, g9) were rerun once after a first attempt failed in the harness, not in the release; the receipt lists both attempts:

- g1 clean install: `INSTALL_EXIT 0`, and the installed version `0.32.4.dev202611031039+06d1038d` equals the uploaded wheel's.
- g3 online 4B card: `CARD_CHECK PASS`.
- g6 codec: exit 0.
- g7a packaged ICD: `ICD_CONTRACT PASS`.
- g7b system install: `SYSINSTALL_EXIT 0`.
- g7c packaged ANE worker verify: `GATE7C_EXIT 0`.
- g7d fresh-image transcribe: `GOLDEN_MATCH PASS`.
- g8 Kokoro: `G8_KOKORO_EXIT 0`.
- g9 read-aloud: `SPEAK_QUEUE_SMOKE PASS`.

On a 16-inch M1 Max (kernel `7.1.12-2-20261103.2-lab-ARCH`, AGX vote off by that kernel's default) the rest passed. The 9B gates run on this machine.

- g2 online 9B: `GATE PASS`. The chat answered and the compare chose `cat` (0.7192).
- g4 offline: `GATE PASS`, with `OUTBOUND_DENIED True` in its log. g5 Laya: exit 0.
- g10 Kokoro primer: `GATE10 PASS`. Primed after 26.4 s (limit 90 s), first audio 0.248 s (limit 2.0 s).
- g12 Kokoro stream: `GATE12 PASS`. First audio 0.351 s (limit 1.5 s), 20 chunks, 10.90 s of audio, none starved, no errors.
- g11 card 9B: `CARD_CHECK PASS`.
- g13 GDN coopmat correctness: 6 of 6 cases, 1394 of 1394 assertions.
- g14 routing: `ROUTE_ON PASS` and `KILL_SWITCH PASS`.
- g15 trig contract: 7 of 7 cases (128 assertions) and 4 of 4 cases (49 assertions).

The driver logged no GPU timeout during any of these batches, g10 included (see the known issue below). g1, g3 and g8 also passed on this machine, and so did g7d. Each of those 16-inch rows ran once.

## Decode number on the uploaded wheel

The uploaded aarch64 wheel was installed into a fresh venv on the 13-inch M1 with `pip install --no-index --find-links` on the vendored wheel set from the draft (mlx-lm 0.31.3). Qwen3.8-2B 4-bit (`SiddhJagani/Qwen3.8-2B-mlx-4Bit@0867d98b`), greedy, seed 0, 128 tokens requested, `MLX_DISABLE_COMPILE=1`, five separate runs: 27.89, 28.19, 28.19, 27.95 and 28.19 tokens per second. The median is 28.19. The generated token ids hash to `775bf034e185c8f9` in all five runs, the digest the v0.7.34, v0.7.35 and v0.7.36 runs gave. The v0.7.36 median on the same machine was 27.66.
