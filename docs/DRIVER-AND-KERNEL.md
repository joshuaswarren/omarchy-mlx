# Driver and kernel notes

Back to the [README](../README.md).

## Qwen3.5-9B fused GDN decode

The fused raw decode route is on by default on every chip. The published-wheel audit found that free-run outputs diverged from the reference. See the acceptance criteria in docs/numerics-gate.md. Link the performance and numerical results before publishing figures.

## How it works

MLX lowers your graph to Vulkan compute and runs it on the Apple GPU.
The backend picks the Vulkan ICD
inside its own process. With no `VK_DRIVER_FILES` or
`VK_ICD_FILENAMES` set, it selects
`/usr/lib/omarchy-mlx/vulkan/honeykrisp_icd.aarch64.json` when that
file exists and points the loader at it for this process. Otherwise it
searches the standard ICD directories. When the packaged ICD is
selected and `mesa-git-sha` is non-empty, startup refuses a
Honeykrisp driver whose reported SHA differs from that file. The desktop keeps its
system Mesa driver. The ANE runs as a
separate worker process. It only executes hash-pinned program
bundles, so unverified ANE programs never run.
