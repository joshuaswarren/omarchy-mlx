# Muse Spark Open text model for mlx-lm

The mlx-community Muse Spark Open quantizations (3bit and 4bit) declare `model_type` `muse_spark`. They use the Hugging Face `text_config` layout. mlx-lm main has no loader for that name. Its `muse_spark_open` class reads the native checkpoint layout, and it calls seven fused Metal kernels.

`serve/mlx_omarchy_muse` adds the text-only class of the closed mlx-lm pull request 1975 (head `117b84cae44a`). That class uses only `mx` operations, so it needs no Metal kernel. The file is copied unchanged. Its sha256 is in `serve/mlx_omarchy_muse/__init__.py` and in [dependency-licenses.md](dependency-licenses.md). The licence text is `LICENSES/mlx-lm.txt`.

## Use

```python
import mlx_omarchy_muse
mlx_omarchy_muse.register()
# mlx_lm.load(...) now resolves model_type "muse_spark"
```

`register()` does nothing when mlx-lm already provides a `muse_spark` module. It changes no mlx-lm file. The package is in a repository checkout only. It is not in the release package.

## What the tests show

`tests/test_muse_spark_adapter.py` runs on the CPU. It compares the class with the transformers `MuseSparkTextModel` (transformers 5.20.0, fp32) on the same random weights. The weights are tiny: hidden size 64, four layers, four experts, a sliding window of 8 and 24 tokens. The reference logits are stored in `tests/fixtures/muse/muse_tiny_reference.json`. `generate_fixture.py` rebuilds them.

| Check | Result |
|---|---|
| One-shot logits, 24 positions | largest difference 9.6e-06, reference logits up to 0.90, top-1 equal at every position |
| Incremental decode across the sliding window | largest difference 9.3e-06, top-1 equal at every position |
| The published 3bit `config.json` | resolves to this class, 62 layers, 256 experts, 16 full and 46 sliding layers |

Both runs used mlx 0.32.3 on the CPU. The suite passes with mlx-lm 0.32.0 and with mlx-lm main at `06e03f7ce`.

## What it does not show

- No run of this class on the Vulkan backend yet.
- No bf16 run, no quantized run, and no run with the real weights.
- No speed figure.
- The class does not read ModelOpt NVFP4 checkpoints.

## Size

The 3bit checkpoint is 359.2 GB and the 4bit one is 461.7 GB (Hugging Face file sizes). No machine in the lab holds either. The Vulkan `gather_qmm` accepts 4-bit and 8-bit weights only, so the 3-bit experts of the 3bit checkpoint do not run on the GPU backend. This package does not change any of that.
