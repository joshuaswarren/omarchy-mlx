# Muse Spark Open 2-bit-expert MLX build tools

The scripts and facts behind the mixed-bit MLX build of Muse Spark Open (816B total, 42B active), built from AMD's public FP8 checkpoint.

- `BUILD.md`: reproduce steps and the measured result (source revision, bits per group, 63 parts, 237,429,698,758 bytes).
- `recipe.json`: the quantization recipe and tool versions.
- `requant_stream.py`: streaming FP8 to MLX converter (each tensor is fetched by HTTP Range, nothing is staged on disk). Its sha256 is in `recipe.json`; do not edit it if you want to compare hashes.
- `finalize2.py`: checks every tensor against the published mlx-community 3-bit text layout, writes the index and config, renames the parts. It needs that layout as an inventory JSON (name, dtype, shape per tensor, read from the safetensors headers of the published 3-bit repo); the inventory generator is not part of this directory.
- `mac-manifest.py`: whole-file and per-tensor sha256 manifest.

The weights are not in this repository.
