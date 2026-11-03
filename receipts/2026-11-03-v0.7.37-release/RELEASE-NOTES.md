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

On Linux, the GPU driver logs a page fault when a process with two compute submissions in flight is killed. A firmware timeout and a recovery follow. The machine recovers each time and later GPU work runs. The killed process does not. An 11-line MLX script reproduces it on a 13-inch M1 in 10 of 10 runs, and a Vulkan-only program with no MLX in it reproduces it in 15 of 15 kills. Nothing faulted in 10 runs with one submission in flight at a time, or in 10 runs where the Vulkan program exits normally. This is what the voice gate g10 logged on v0.7.35 and v0.7.36. The gate stops the assistant with a signal while work is still queued. The Vulkan-only program puts the fault below MLX, in the GPU driver. It is not new in this release, and the kernel side is tracked in linux-aurora#335.

## How this release was gated

PENDING: filled in from the receipt once the draft assets pass the gates.

## Decode number on the uploaded wheel

PENDING: one pinned 4-bit decode run on the uploaded aarch64 wheel.
