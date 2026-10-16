# Muse Spark Open text model for mlx-lm

The mlx-community Muse Spark Open quantizations (3bit and 4bit) declare `model_type` `muse_spark`. They use the Hugging Face `text_config` layout. mlx-lm main has no loader for that name. Its `muse_spark_open` class reads the native checkpoint layout, and it calls eight fused Metal kernels registered in six files.

`serve/mlx_omarchy_muse` adds the text-only class of the closed mlx-lm pull request 1975 (head `117b84cae44a`). That class uses only `mx` operations, so it needs no Metal kernel. The file is copied unchanged. Its sha256 is in `serve/mlx_omarchy_muse/__init__.py` and in [dependency-licenses.md](dependency-licenses.md). The licence text is `LICENSES/mlx-lm.txt`.

## Quick start

You need an Apple Silicon machine on Omarchy Linux, a release wheel that carries the 3-bit and 2-bit `gather_qmm`, mlx-lm 0.32.0, and enough memory for the checkpoint (see Requirements). The run uses mlx-lm through the adapter. It does not use the oMLX server.

1. Install the release wheel. See [INSTALL.md](INSTALL.md). The install script builds a private venv, and `mlx-omarchy` runs that venv's Python.
2. Install mlx-lm 0.32.0 into that venv. The `--no-deps` flag keeps the mlx-omarchy build of `mlx`.

   ```sh
   mlx-omarchy -m pip install --no-deps mlx-lm==0.32.0
   ```

3. Clone the repository. The adapter and the check scripts live in the checkout.

   ```sh
   git clone https://github.com/joshuaswarren/omarchy-mlx && cd omarchy-mlx
   ```

4. Check the expert kernels on your GPU. Each command prints `RESULT PASS`.

   ```sh
   mlx-omarchy receipts/2026-10-16-gather-qmm-subgroup-3bit-2bit/gather_bits_gpu.py 3 64
   mlx-omarchy receipts/2026-10-16-gather-qmm-subgroup-3bit-2bit/gather_bits_gpu.py 2 128
   ```

5. Register the class and load a checkpoint. The checkpoint directory is a download of an mlx-community Muse Spark Open repository.

   ```sh
   PYTHONPATH=serve mlx-omarchy -c "
   import mlx_omarchy_muse
   mlx_omarchy_muse.register()
   from mlx_lm import load, generate
   model, tokenizer = load('/path/to/checkpoint')
   print(generate(model, tokenizer, prompt='Say hello.', max_tokens=32))
   "
   ```

The routed experts run `gather_qmm` at their stored bit width. 4-bit and 8-bit use the kernels that ship with the wheel. 3-bit group 64 and 2-bit group 64 and 128 use the subgroup decode kernels at the decode shape. The check scripts print the time per call at the real expert shape.

## Use

From a clean checkout, put `serve` on the path:

```sh
PYTHONPATH=serve python3 -c "
import mlx_omarchy_muse
mlx_omarchy_muse.register()
import mlx_lm  # mlx_lm.load(...) now resolves model_type 'muse_spark'
"
```

`register()` installs the adapter class as `mlx_lm.models.muse_spark`. It replaces the module that mlx-lm 0.32.2 and later ship under that name, because that module reads the native checkpoint layout and imports the eight fused Metal kernels. Set `MLX_OMARCHY_MUSE_UPSTREAM=1` to keep the mlx-lm module instead. `register()` then does nothing. It changes no mlx-lm file. The package is in a repository checkout only. It is not in the release package.

| mlx-lm | Status |
|---|---|
| 0.32.0 | The adapter tests pass. The adapter adds the `muse_spark` module. |
| 0.32.2 | The adapter tests pass. `register()` replaces the module that mlx-lm ships. |

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

The suite passes with mlx-lm 0.32.0 and with mlx-lm 0.32.2 (nine tests each), and with mlx-lm main at `06e03f7ce`. The figures were not produced with the mlx-omarchy wheel.

## Requirements

- Memory for the checkpoint: 359.2 GB for 3bit and 461.7 GB for 4bit (Hugging Face file sizes).
- The routed experts run `gather_qmm` at their stored width. The Vulkan `gather_qmm` in this repository runs 2-bit, 3-bit, 4-bit and 8-bit affine weights in the transposed layout with biases.
- The class reads the Hugging Face layout of the mlx-community checkpoints. It does not read ModelOpt NVFP4 checkpoints.
