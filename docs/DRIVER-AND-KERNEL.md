# Driver and kernel notes

Back to the [README](../README.md).

## Qwen3.5-9B fused GDN decode

The fused raw decode route is ON by default on every chip. The v0.7.26
published-wheel audit measured +31-35% decode on G14-class hardware and
+35-43% on G13. It found per-op fp64 error identical, 99.51% teacher-forced
top-1 agreement, and S=1 PPL -0.072%. All 10 free-runs diverged within 512
tokens: six first-divergence gaps were 0.125 (one bf16 ULP at those logits),
and four were zero; overall prefix identity was 29.43%. This meets the
scale-aware numerics gate. Set MLX_OMARCHY_GDN_RAW_REPEAT=0 to use the
composed route. See docs/numerics-gate.md for the acceptance criteria.

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
