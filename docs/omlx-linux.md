# oMLX on omarchy-mlx

This guide runs oMLX v0.7.1.dev1 on an Omarchy M+ Linux install. The
backend is the omarchy-mlx Vulkan backend. It does not apply to stock Linux
or macOS. The backend uses the private Honeykrisp Vulkan ICD. Mesa remains
the system driver. Install this stack only on supported Omarchy M+
hardware.

## What is verified

The Linux compatibility package ran on an M2 Max (T6021). The basic server
path ran there too, on Linux with the Honeykrisp Vulkan ICD. The recorded
run installed oMLX 0.7.0 into a separate user venv. It added mlx-lm
`94cdcae1` and the omarchy-mlx `0.32.4.dev` wheel. It found 14 cached
models. It returned real chat completions from Qwen3-4B. See [the server
receipt](../receipts/2026-10-04-omlx-linux/README.md).

A separate [stub-render receipt](../receipts/2026-10-05-omlx-admin-ui/README.md)
records only template/layout and API-routing evidence. That receipt does
not qualify authentication. It does not qualify model loading, chat
generation, downloader behavior, integrations, or the benchmark panel. The
x86 dev box has no MLX module or accelerator. It suits device-independent
checks only.

## Install

First identify the installed omarchy-mlx Python environment and wheel. Do
not install into the shared environment. `install.sh` creates an isolated
user venv. Use the exact wheel built for the target architecture and
Python version.

```sh
cd ~/src/mlx-omarchy
python3 packaging/omlx-linux/install.sh \
  --mlx-wheel /path/to/compatible-mlx-omarchy.whl \
  --home "$HOME/.local/share/omlx-linux"
```

The installer defaults to oMLX upstream tag v0.7.1.dev1 (commit c0b1056b).
It creates its venv under the supplied private HOME. Supply the compatible
wheel built for the target machine and Python version. The compatibility
patches target oMLX commit
`c0b1056b41ebde9422af316cf5038a423eb8f24c`. Do not apply them to another
upstream revision. The installer applies the platform-gate patches, Linux
compatibility patches, and the repository's matching mlx-lm patch series.

The package README documents installer parameters and checks:
[`packaging/omlx-linux/README.md`](../packaging/omlx-linux/README.md).
Keep the baseline/shared MLX environment unchanged.

## Start the server

Use the venv and private HOME created by the installer. Keep the existing
Hugging Face cache path apart from the private HOME. Pre-downloaded weights
then stay visible. The server does not download them again. Start without
`--model` when you test dashboard loading. The Models panel then performs
the load.

```sh
HOST_HOME="$HOME"
export HOME=/path/to/omlx-home
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HOST_HOME/.cache/huggingface/hub}"
"$HOME/.venvs/omlx/bin/omlx" serve --host 127.0.0.1 --port 8900
```

Open `http://127.0.0.1:8900/admin` in a browser on the same host. Keep the
server bound to loopback. Change that only when you have configured
authentication and network access on purpose. On loopback, oMLX may
redirect straight to the dashboard. If it shows the login page, use the
configured API key. See upstream `omlx/admin/routes.py` and the v0.7.0
admin interface.

## Fresh-user walkthrough

1. Install into a new private HOME. Use `packaging/omlx-linux/install.sh`
   and the target's compatible MLX wheel.
2. Start the server as above. Visit `/admin`.
3. Log in with the configured admin credentials or API key.
4. In the Models panel, find a cached model. Or use the downloader to fetch
   a supported MLX model. Confirm its model ID in the UI before you load
   it.
5. Load the model from the dashboard. Wait for the loaded status. Loading
   is a real model operation. A server health response alone does not
   prove it.
6. Open Chat. Select the loaded model. Send a prompt. Confirm visible
   generated text.
7. Open Benchmark. Select that model and a context profile. Run a small
   prompt-length case. Record prefill (PP) and text-generation (TG)
   tokens/s from the result.
8. Save browser screenshots, server log, request/result data, model
   identity, package versions, host/kernel/backend, and wall-clock
   timestamps. Keep them with the hardware receipt.

The pinned benchmark implementation (`omlx/admin/benchmark.py`) makes a
point of two choices. Measured calls use unique UUID prefixes and
`skip_cache_store=True`. It treats unexpected cached input tokens as a
warning. This is a no-cache PP/TG benchmark. It is not a partial-prefix-hit
benchmark. No separate partial-prefix-hit control was found in the v0.7.0
benchmark page or its code. A separate cache-reuse experiment needs its own
controlled request sequence. It also needs observed cache-hit evidence. It
cannot be reported as a panel feature at this pin. README.md's claim of
partial-prefix-hit benchmarking at lines 253 to 260 is an upstream
documentation/code mismatch. It is not an omarchy defect. A draft note is
in [`receipts/2026-10-04-omlx-tensorfold-parity/drafts/a17-benchmark-prefix-wording.md`](../receipts/2026-10-04-omlx-tensorfold-parity/drafts/a17-benchmark-prefix-wording.md).

Admin routes in the pinned upstream code include login
(`omlx/admin/routes.py:1815,1906`), chat (`:1862`), model downloads
(`:7518`), and benchmark start/stream/results (`:8655-8870`). The benchmark
request captures `model_id`, `prompt_lengths`, `generation_length`,
`batch_sizes`, `context_profile`, and warmup mode
(`omlx/admin/benchmark.py:109-118`). A real benchmark number needs a real
model completion on the GPU. A stub backend or dev-box screenshot is not
evidence for PP/TG or prefix-cache behavior.

## Limitations and evidence

The package's compatibility and server-smoke receipt is the source of truth
for what has run. A green import, a `/health` response, a screenshot of an
unconnected mock, or an API response from a stub backend does not prove
model loading. It also does not prove chat generation, benchmark values, or
partial prefix-hit behavior. Record missing hardware/UI runs as unverified.
Do not infer success.
