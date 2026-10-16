# Muse Spark Open text model for mlx-lm

The mlx-community Muse Spark Open quantizations (3bit and 4bit) declare `model_type` `muse_spark`. They use the Hugging Face `text_config` layout. mlx-lm main has no loader for that name. Its `muse_spark_open` class reads the native checkpoint layout, and it calls eight fused Metal kernels registered in six files.

`serve/mlx_omarchy_muse` adds the text-only class of the closed mlx-lm pull request 1975 (head `117b84cae44a`). That class uses only `mx` operations, so it needs no Metal kernel. The file is copied unchanged. Its sha256 is in `serve/mlx_omarchy_muse/__init__.py` and in [dependency-licenses.md](dependency-licenses.md). The licence text is `LICENSES/mlx-lm.txt`.

## Use

From a clean checkout, put `serve` on the path:

```sh
PYTHONPATH=serve python3 -c "
import mlx_omarchy_muse
mlx_omarchy_muse.register()
import mlx_lm  # mlx_lm.load(...) now resolves model_type 'muse_spark'
"
```

`register()` does nothing when mlx-lm already provides a `muse_spark` module. It changes no mlx-lm file. The package is in a repository checkout only. It is not in the release package.

## Tests and reference figures

`tests/test_muse_spark_adapter.py` runs on the CPU. It compares the class with the transformers `MuseSparkTextModel` (transformers 5.20.0, fp32) on the same random weights. The weights are tiny: hidden size 64, four layers, four experts, a sliding window of 8 and 24 tokens. The reference logits are stored in `tests/fixtures/muse/muse_tiny_reference.json`. `generate_fixture.py` rebuilds them.

| Check | Result |
|---|---|
| One-shot logits, 24 positions | largest difference 9.6e-06, reference logits up to 0.90, top-1 equal at every position |
| Incremental decode across the sliding window | largest difference 9.3e-06, top-1 equal at every position |
| The published 3bit `config.json` | resolves to this class, 62 layers, 256 experts, 16 full and 46 sliding layers |

Provenance of the figures above:

| Item | Value |
|---|---|
| Harness commit | `4e0f9d752` (this branch) |
| Backend | stock PyPI `mlx` 0.32.3 and `mlx-cpu` 0.32.3, CPU only |
| `scripts/mlx_provenance.py` | `verified: no-metadata`, `mx_version: 0.32.3`. The library is a source install, so no wheel RECORD exists to compare. |
| transformers, torch | 5.20.0, 2.14.1 (CPU) |

The suite also passes with mlx-lm 0.32.0 and with mlx-lm main at `06e03f7ce`. The figures were not produced with the mlx-omarchy wheel.

## Requirements

- Memory for the checkpoint: 359.2 GB for 3bit and 461.7 GB for 4bit (Hugging Face file sizes).
- The 3bit experts use 3-bit `gather_qmm`. The backend must run that layout. The Vulkan `gather_qmm` in this repository accepts 4-bit and 8-bit weights only.
- The class reads the Hugging Face layout of the mlx-community checkpoints. It does not read ModelOpt NVFP4 checkpoints.
