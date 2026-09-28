# Demo: chat on the Apple GPU under Omarchy

Five minutes, one M1 running Omarchy (Asahi Linux). Watch it first: [the recorded run](https://joshuaswarren.github.io/mlx-omarchy/) (2:47, unedited). Nothing here changes the
Mesa driver, Hyprland, or any Omarchy file; everything lands under `$HOME`.

## 1. Install

```bash
curl -fsSL https://raw.githubusercontent.com/joshuaswarren/mlx-omarchy/main/install.sh | bash
```

What you should see, in order: the two runtime packages (`lapack`, `blas`)
installed through `omarchy-pkg-add`, the release wheel downloaded and checked
against `SHA256SUMS`, a private venv created in
`~/.local/share/mlx-omarchy/venv`, and a smoke test that prints
`device: Apple M1 (G13G B1)` followed by `matmul OK`. If the device line says
`llvmpipe`, the Vulkan driver is not the Apple one; see Troubleshooting.

## 2. Open the chat app

The source installer adds `mlx-omarchy-chat` to open the web app.
`mlx-omarchy-demo` uses the same models in a terminal. It no longer loads its own model.
This change has not passed release tests. Published installers may still use the old terminal demo.

From this checkout, inspect the app without an install:

```bash
PYTHONPATH=serve python3 -m mlx_omarchy_assistant --help
PYTHONPATH=serve python3 -m mlx_omarchy_assistant --home /tmp/mlx-chat-check
```

Setup offers Everyday and Quality pairs and requires approval before downloads.
Neither pair currently has complete co-serving and disconnected-restart evidence.
Unqualified models refuse normal startup rather than claim readiness.

The terminal interface accepts `--terminal`, `--prompt`, and `--once`.
Use `--pair everyday --yes` only when you intend to approve that pair's downloads.
See the [serving guide](../docs/serve.md) for exact qualification limits.

## 3. Use it from your own code

`mlx-omarchy` is the interpreter of the private venv:

```bash
mlx-omarchy -c 'import mlx.core as mx; print(mx.device_info()); print((mx.ones((4,4)) @ mx.ones((4,4))).tolist())'
mlx-omarchy -m mlx_lm.generate --model MODEL_ID --prompt "Hello"
```

The API is upstream MLX (`import mlx.core as mx`, `mlx.nn`, `mlx_lm`), so
upstream examples run unchanged when they stay inside the supported
operations ([docs/compatibility.md](../docs/compatibility.md)).

## 4. Remove it

```bash
curl -fsSL https://raw.githubusercontent.com/joshuaswarren/mlx-omarchy/main/install.sh | bash -s -- --uninstall
```

## Troubleshooting

- `Python 3.14 is required`: Omarchy on Asahi ships Python 3.14; `pacman -Syu`
  if yours is older.
- `device: llvmpipe`: the Honeykrisp driver was not picked up. Check
  `vulkaninfo --summary` (package `vulkan-tools`) lists `Apple M1`; unset any
  `VK_ICD_FILENAMES`/`VK_DRIVER_FILES` you set for other software.
- Model download hangs: the demo reads `HF_HUB_OFFLINE`, `HF_ENDPOINT`, and
  the usual proxy variables, like any Hugging Face download.
- Something else: [docs/known-defects.md](../docs/known-defects.md), then an
  issue with the output of `mlx-omarchy -c 'import mlx.core as mx; print(mx.__version__, mx.device_info())'`.
