# oMLX on omarchy-mlx

This guide runs oMLX v0.7.1.dev1 on an Omarchy M+ Linux install with the omarchy-mlx Vulkan backend. It does not apply to stock Linux or macOS. The backend uses the private Honeykrisp Vulkan ICD; Mesa remains the system driver. Install this stack only on supported Omarchy M+ hardware.

## What is verified

The Linux compatibility package and the basic server path were exercised on an M2 Max (T6021) running Linux with the Honeykrisp Vulkan ICD. The recorded run installed oMLX 0.7.0, mlx-lm `94cdcae1`, and the omarchy-mlx `0.32.4.dev` wheel into a separate user venv, discovered 14 cached models, and returned real chat completions from Qwen3-4B. See [the server receipt](../receipts/2026-10-04-omlx-linux/README.md).

A separate [stub-render receipt](../receipts/2026-10-05-omlx-admin-ui/README.md) records only template/layout and API-routing evidence. That receipt does not qualify authentication, model loading, chat generation, downloader behavior, integrations, or the benchmark panel. The x86 dev box has no MLX module or accelerator and is suitable only for device-independent checks.

## Install

First identify the installed omarchy-mlx Python environment and wheel. Do not install into the shared environment: `install.sh` creates an isolated user venv. Use the exact wheel built for the target architecture and Python version.

```sh
cd ~/src/mlx-omarchy
python3 packaging/omlx-linux/install.sh \
  --mlx-wheel /path/to/compatible-mlx-omarchy.whl \
  --home "$HOME/.local/share/omlx-linux"
```

The installer defaults to oMLX upstream tag v0.7.1.dev1 (commit c0b1056b) and creates its venv under the supplied private HOME. Supply the compatible wheel built for the target machine and Python version. The compatibility patches target oMLX commit `c0b1056b41ebde9422af316cf5038a423eb8f24c`; do not apply them to another upstream revision. The installer applies the platform-gate patches, Linux compatibility patches, and the repository's matching mlx-lm patch series.

The package README documents installer parameters and checks: [`packaging/omlx-linux/README.md`](../packaging/omlx-linux/README.md). Keep the baseline/shared MLX environment unchanged.

## Start the server

Use the venv and private HOME created by the installer. Keep the existing Hugging Face cache path separate from the private HOME so pre-downloaded weights remain visible and the server does not download them again. Start without `--model` when testing dashboard loading; this makes the Models panel perform the load.

```sh
HOST_HOME="$HOME"
export HOME=/path/to/omlx-home
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HOST_HOME/.cache/huggingface/hub}"
"$HOME/.venvs/omlx/bin/omlx" serve --host 127.0.0.1 --port 8900
```

Open `http://127.0.0.1:8900/admin` in a browser on the same host. Keep the server bound to loopback unless you have configured authentication and network access intentionally. On loopback, oMLX may redirect directly to the dashboard; if it presents the login page, use the configured API key. See upstream `omlx/admin/routes.py` and the v0.7.0 admin interface.

## Fresh-user walkthrough

1. Install into a new private HOME using `packaging/omlx-linux/install.sh` and the target's compatible MLX wheel.
2. Start the server as above and visit `/admin`.
3. Log in using the configured admin credentials/API key.
4. In the Models panel, discover a cached model or use the downloader to fetch a supported MLX model. Confirm its model ID in the UI before loading it.
5. Load the model from the dashboard and wait for the loaded status. Loading is a real model operation; a server health response alone does not establish it.
6. Open Chat, select the loaded model, send a prompt, and confirm visible generated text.
7. Open Benchmark, select that model and a context profile, run a small prompt-length case, and record prefill (PP) and text-generation (TG) tokens/s from the result.
8. Save browser screenshots, server log, request/result data, model identity, package versions, host/kernel/backend, and wall-clock timestamps with the hardware receipt.

The pinned benchmark implementation (`omlx/admin/benchmark.py`) deliberately generates unique UUID prefixes and sets `skip_cache_store=True` for measured calls. It treats unexpected cached input tokens as a warning; this is a no-cache PP/TG benchmark, not a partial-prefix-hit benchmark. No separate partial-prefix-hit control was found in the v0.7.0 benchmark page or implementation. A separate cache-reuse experiment needs its own controlled request sequence and observed cache-hit evidence; it cannot be reported as a panel feature at this pin. README.md's claim of partial-prefix-hit benchmarking at lines 253–260 is an upstream documentation/code discrepancy, not an omarchy defect. A draft note is in [`receipts/2026-10-04-omlx-tensorfold-parity/drafts/a17-benchmark-prefix-wording.md`](../receipts/2026-10-04-omlx-tensorfold-parity/drafts/a17-benchmark-prefix-wording.md).

Admin routes in the pinned upstream implementation include login (`omlx/admin/routes.py:1815,1906`), chat (`:1862`), model downloads (`:7518`), and benchmark start/stream/results (`:8655-8870`). The benchmark request captures `model_id`, `prompt_lengths`, `generation_length`, `batch_sizes`, `context_profile`, and warmup mode (`omlx/admin/benchmark.py:109-118`). A real benchmark number requires a real model completion on the GPU; a stub backend or dev-box screenshot is not evidence for PP/TG or prefix-cache behavior.

## What you get: measured numbers

These numbers come from runs of `oMLX serve` on Linux with greedy decoding, the omarchy-mlx Vulkan backend and `OMLX_LINUX_CUSTOM_KERNELS=0`. Each row is the first of two requests in one run of the v0.7.1.dev1 tree. For the three dense-model runs the second request decoded within 0.3 to 1.4 percent of the first (the offload rows differ more). Every cell, the run method and the failed runs are in [the numbers receipt](../receipts/2026-10-10-omlx-linux-numbers/README.md).

| machine | model | prompt | first token | decode tok/s |
|---|---|---|---|---|
| M2 Max, 96 GB | Qwen3.8-27B 4-bit | 267 tokens | 5.1 s | 9.35 |
| M1, 16 GB | Qwen3.5-9B 4-bit (MLX) | 356 tokens | 5.4 s | 8.29 |
| M1, 16 GB | Qwen3.5-9B 4-bit (MLX) | 476 tokens | 6.9 s | 8.31 |
| M2 Max, 96 GB | Qwen3.6-35B-A3B 4-bit, expert offload at 25 percent | 90 tokens | 10.0 s | 2.90 |
| M1, 16 GB | Qwen3.6-35B-A3B 4-bit, expert offload at 10 percent | 43 tokens | 19.8 s | 1.31 |

What these numbers say:

- The oMLX version changes decode speed little. The v0.7.1.dev1 tree and the previous pin (`cc1fdc9a`) differ by 0.4 percent on the 27B (9.35 against 9.39 tok/s) and by up to 8.6 percent on the other pairs, with no consistent direction (M1 35B offload: 1.31 against 1.43; M2 35B offload: 2.90 against 2.84).
- Prefill ran at about 35 tok/s on the M2 Max with the 27B model and at 41 tok/s on the M1 with the 9B model (the second, warm pilot request of 142 and 100 tokens).
- Several requests at once do not raise prefill speed. oMLX packs concurrent prefills into one forward pass, and that pass ran at 60 to 69 tok/s on the M1 with the 9B model, the same rate as one 476-token request (about 68 tok/s). With 4 prompts the slowest first token arrived after 14.5 s; with 8 prompts, after 26.2 s.
- Expert offload lets a mixture-of-experts model run on a machine that cannot hold it, at a high cost: 1.31 and 2.90 tok/s for the 35B model in the rows above.
- The memory the offload path needs is close to the weights it keeps resident. For Qwen3-30B-A3B the measured peak after a 128-token prefill was 3.0 to 7.2 percent above the admission estimate, at residencies 0.25, 0.4 and 0.5.

Limits of this data:

- One run per row, no confidence intervals. Decode speed with 4 concurrent requests varied by up to 1.45 times between two repeats in the same run (M2 Max, 35B offload: 0.98 and 0.67 tok/s), so do not read concurrent numbers as a difference between oMLX versions.
- Custom kernels on: not measured on this wheel. One run with custom kernels on and an older wheel returned no tokens; its server log shows a shader compile error at the decode step.
- Qwen3.6-35B-A3B without offload on the M2 Max: not measured. The run returned no tokens; its server log shows prefill failing on an unsupported kernel feature.
- Long prompts: not measured for memory. The estimate check above used 128 tokens.
- No macOS comparison for these oMLX runs. The run header records the Mesa package (26.2.4), not whether a private driver file was selected.

## Limitations and evidence

The package's compatibility and server-smoke receipt is the source of truth for what has run. A green import, `/health` response, screenshot of an unconnected mock, or API response from a stub backend does not prove model loading, chat generation, benchmark values, or partial prefix-hit behavior. Record missing hardware/UI runs as unverified instead of inferring success.
