# Troubleshooting

Back to the [README](../README.md).

- `import mlx.core` fails on a missing shared library: install
  `openblas`, `lapack`, and `blas` through pacman. The installers do
  this for you; a manual wheel install has to.
- `mlx-omarchy-info` does not report `icd_source` `packaged`: the
  wheels do not ship the driver. Install `omarchy-mlx-vulkan` (the
  `omarchy-mac-ml` package pulls it in). It places a private Honeykrisp
  ICD under `/usr/lib/omarchy-mlx/vulkan/` and leaves the system Mesa
  driver installed. See
  [docs/install-omarchy.md](install-omarchy.md).
- The backend refuses to start and names a non-Apple GPU. The refusal
  is on purpose. Set `MLX_OMARCHY_ALLOW_NON_APPLE=1` only on a dev
  box with software Vulkan.
- `mlx-omarchy-parakeet transcribe` refuses before it runs: run
  `mlx-omarchy-parakeet download` first. The ANE needs an M1 or
  M1 Max with `/dev/accel/accel0` present and the `ane` module
  loaded. The refusal names the missing piece.
- On an M2 Max, `mlx-omarchy-info` reports the ANE as available when the
  driver is loaded (read on 2026-10-10 with the installed 0.7.28 package:
  node, `accel0`, module and version all present). An older release could
  report it as missing; if yours does, update the package.
