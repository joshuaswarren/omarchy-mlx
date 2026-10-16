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

## Run it across several machines (omarchy-cluster, MLX)

This route runs on omarchy-mlx through omarchy-cluster (mlx-lm). It does not use the oMLX server. It does not use llama.cpp.

| Item | State at this commit |
|---|---|
| Tiny 8-layer Muse-shaped model, three machines, split 3,3,2, through the commands below | Run. 8 tokens at 19.95 tok/s. There is no agreed single-process reference: single-process runs on two devices disagree on the first token too. So this is a plumbing result and not a correctness claim. |
| The raw completion request of step 6 | Unit test and a mock gateway only. Not run on real nodes. |
| The 2-bit 62-layer checkpoint on the cluster | Not run. The layer split is not final, so none is given here. |
| Quality, decode speed and first-token time of the 2-bit checkpoint | Not measured. |

1. Get the 2-bit checkpoint. It is the unpruned `amd/Muse-Spark-Open-816B-A42B-Quark-PTPC-FP8` (revision `52cace63f43ed9fc6acef6dcd391adb02f4bfbe1`) converted to MLX affine weights: routed experts 2-bit group 128, attention and the expert projections 4-bit group 64, embedding 4-bit, `lm_head` 6-bit group 64. It has 63 part files, 2,425 tensors and 237,429,698,758 bytes, and every tensor name, dtype and logical shape equals the published mlx-community 3-bit layout (the packed widths differ, because the bit widths differ). The converter, the finalizer and the recipe are in [tools/muse-2bit](../tools/muse-2bit/README.md).
2. Put the parts where every node can read them. Each node needs a Hugging Face cache snapshot directory (`models--<org>--<name>/snapshots/<revision>`) that holds links to the part files and to `config.json`, the tokenizer files and `model.safetensors.index.json`. On the machines that lack the disk space, the links pointed into read-only sshfs mounts of the disks that hold the parts. Each node then loads the model by its repository name, offline.
3. Install on every node: the omarchy-cluster agent (see the omarchy-cluster README) at or after merge commit `224c1b2` (raw prompts, step 6) and, on each Linux node, an omarchy-mlx wheel built from main at or after the commits that add the subgroup 3-bit and 2-bit `gather_qmm` and the compiled `rsqrt` fix. The Linux ranks of the tiny run used wheel `0.32.4.dev202610160322+64f85e71`. The macOS rank used stock `mlx` 0.32.3 and mlx-lm 0.32.1 from main.
4. Give every rank the adapter. Set `OMARCHY_MLX_SERVE=<checkout>/serve` in the rank environment. Point `--python-linux` and `--python-mac` at a small wrapper script that sets the environment and then runs the node's Python. On Linux the wrapper puts the omarchy-mlx tree first on `PYTHONPATH`.
5. Start the pipeline from one gateway machine.

   ```sh
   omarchy-cluster serve <model> --nodes <node1>,<node2>,<node3> --split 3,3,2 \
     --ring-ip <node1>=<ip1> --ring-ip <node2>=<ip2> --ring-ip <node3>=<ip3> \
     --python-mac <rank-wrapper> --python-linux <rank-wrapper> --port 18020 --engine-port 18031
   ```

   `--split` lists the layers per node, in node order. The first node is rank 0, the engine, and it runs the last layers: `--nodes a,b,c --split 3,3,2` puts layers 5 to 7 on `a`, 2 to 4 on `b` and 0 to 1 on `c`. The tiny model has 8 layers. The 2-bit checkpoint has 62.
6. Send a request. The gateway accepts temperature 0 only. The reply carries `timings` with `prefill_ms`, `decode_ms` and `tokens_per_s`. A chat request applies the checkpoint's chat template:

   ```sh
   curl -s http://<gateway>:18020/v1/chat/completions -H 'Content-Type: application/json' \
     -d '{"model":"<model>","messages":[{"role":"user","content":"hello"}],"max_tokens":8,"temperature":0}'
   ```

   The staged chat template has no switch that turns reasoning off. In llama.cpp runs Muse opened with private reasoning (recipient `self`) and spent its token budget before it answered; this has not yet been seen on MLX. To get the answer at once, send a raw completion that ends in the answer channel. A raw prompt is tokenized as written and the engine adds no beginning-of-text token, so the text carries its own. Render the prompt from the template, then append the answer channel. `AutoTokenizer` needs a transformers release that provides the `TokenizersBackend` class (4.46.3 does not). The template's default reasoning effort is `xhigh` (it prints `Reasoning strength: 512`); `reasoning_effort="low"` prints 8:

   ```python
   from transformers import AutoTokenizer
   tok = AutoTokenizer.from_pretrained("/path/to/checkpoint")
   prompt = tok.apply_chat_template([{"role": "user", "content": "What is the capital of France?"}], tokenize=False, add_generation_prompt=True,
                                    reasoning_effort="low")
   prompt += " to=user<|message|>"
   ```

   Check the text before you send it: it begins with `<|begin_of_text|><|start|>system` and ends with `<|start|>assistant to=user<|message|>`. Then send it with `"raw": true`:

   ```sh
   curl -s http://<gateway>:18020/v1/completions -H 'Content-Type: application/json' \
     -d '{"model":"<model>","prompt":"<the text above>","raw":true,"max_tokens":200,"temperature":0}'
   ```

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
