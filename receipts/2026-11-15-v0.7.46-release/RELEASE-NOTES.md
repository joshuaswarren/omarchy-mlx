# omarchy-mlx v0.7.46

[Joshua's agent]

This release follows v0.7.45. It makes the runtime fall back to the system Vulkan driver when the packaged one finds no supported GPU (the M5 case), admits Apple GPUs by their `Apple M<digit>` device name, and raises the CPU utilization floor of GPU-submitting threads by default.

## What changed

- **Fallback to the system Vulkan driver (#460).** With the packaged ICD installed and no `VK_DRIVER_FILES` or `VK_ICD_FILENAMES`, the runtime used to point the loader only at the packaged file. The packaged Mesa knows the G13 and G14 chips only, so a chip it cannot serve (an M5 Max with the package installed) ended in "no Vulkan physical devices found". Now, when the first attempt finds no supported device, the runtime retries once on the system Honeykrisp ICD and prints one `[omarchy]` line naming both ICDs. An explicit override never falls back, and a checksum mismatch or any other refusal does not retry. Checked on a 13-inch M1 with a packaged ICD that loads and finds no device, and with a missing driver file: the v0.7.45 wheel fails with the no-device error, this wheel prints the line and decodes on the system ICD (same generated-ids digest as the pinned decode). **Not yet checked on an M5.**
- **Device guard admits `Apple M<digit>` names (#457, #458).** Aurora's packaged Mesa names Apple GPUs that way (for example `Apple M5 Max (G17C B1)`); the guard refused them. The rule needs `Apple M` followed by a digit, so `Apple Metal ...` is not admitted. Non-Apple devices are still refused without `MLX_OMARCHY_ALLOW_NON_APPLE`.
- **CPU utilization floor for GPU-submitting threads, default on (#463, `MLX_OMARCHY_CPU_UCLAMP=0` turns it off).** Greedy digests are identical with the clamp on and off on every host and model tried. Decode tok/s, clamp on against off (two runs per arm): Mac mini M6 Qwen3.8-2B 4-bit 2.88 and 2.90 against 2.02 and 2.01, Qwen3-4B 4-bit 4.46 and 4.52 against 3.45 and 3.45; 13-inch M1 (governor `performance`) Qwen3.8-2B 29.52 and 29.34 against 28.41 and 28.30, Qwen3-4B 22.92 and 22.11 against 22.98 and 22.94 (within noise); 14-inch M2 Max (`schedutil`) flat on both models.
- README: M5 users use the edge channel until the rc and stable repositories move past 0.7.28 (#461).
- Receipts: the v0.7.45 release notes and gate record (#459).

## Known issues carried forward

The GPU timeouts on the 16-inch M1 Max (G13C) are not understood, as stated in the v0.7.45 notes. Nothing in this release changes them. The split-rows verify route stays off unless `MLX_OMARCHY_SDPA_SPLIT_ROWS` is set; do not turn it on there.

## Measured

Release gates ran against the draft assets (tag at 44873d15, wheel `mlx_omarchy-0.32.4.dev202611151824+44873d15`) on a 13-inch M1 under the 7.3.0-2 lab kernel, with the installed ICD pinned:

- Clean install from the release assets (g1), online 4B model card (g3), g6, g8, g7a: pass.
- Pinned decode, Qwen3.8-2B 4-bit, 5 of 5 runs identical, digest `775bf034e185c8f9`: pass.
- System install (g7b), ANE worker verify (g7c), fresh-image transcription (g7d): pass.
- Speak-queue gate (g9): pass.
- GDN maskless build and run (g13), trig build and run (g15): pass.

Three gate batches stopped on a full 3 GB `/tmp` (no space left on device) on the first try and were rerun as new rows once space was freed; no gate was skipped or edited. The M5 fallback (#460) was checked on an M5 after the gates, not as part of them: with the installed 0.7.40-3 package and no environment variables, `import mlx.core` failed with "no Vulkan physical devices found"; with the v0.7.46 vendored wheels in a private venv it printed the `[omarchy]` fallback line and ran a 512x512 matmul on `Device(gpu, 0)`, twice with the same result. The pacman package route is not covered.
