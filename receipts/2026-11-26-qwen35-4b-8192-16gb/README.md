# Compact pair (`mlx-community/Qwen3.5-4B-MLX-4bit`) at 8192-token backend context on 16 GB M1 - 2026-11-26

[Joshua's agent]

**Question.** Pure full-attention `mlx-community/Qwen3-4B-Instruct-2507-4bit` (36/36 full-attention layers, 147,456 B/token KV cache) was previously qualified at 16384 tokens on a 96 GB M2 Max with a 16.88 GiB measured peak, which exceeds a 16 GB host and leaves the 16 GB 13-inch M1 stuck at the 4096-token default. When tested at 8192 tokens on the 16 GB 13-inch M1 against an omp-shaped prompt (6,067 tokens + `max_tokens=512`), `Qwen3-4B-Instruct-2507-4bit` failed at 4608 of 6067 tokens with `DRM_IOCTL_ASAHI_GEM_CREATE failed: Cannot allocate memory` (`BO creation failed`, HTTP 500 `generation_failed`; worker `VmHWM` 7.96 GiB, `RssShmem` 7.55 GiB, pair footprint 9.22 GiB, prompt cache 0.72 GB). Does hybrid `mlx-community/Qwen3.5-4B-MLX-4bit` (Apache-2.0, revision `32f3e8ecf65426fc3306969496342d504bfa13f3`, 3.03 GB weights, 8/32 full-attention layers, 32,768 B/token KV cache) pass managed pair startup, HTTP generation, omp-shaped prompts, needle recall, and over-cap refusal at 8192 tokens on the 16 GB 13-inch M1, and what is its measured peak?

## Provenance and environment

- **Machine:** 13-inch MacBook Pro M1, 16 GB unified memory (`apple,j293`, `apple,t8103`, `apple,arm-platform`).
- **Source commit & wheel:** installed packages `omarchy-mlx 0.7.50rc2-1`, `omarchy-mlx-vulkan 0.7.50rc2-1`, `mesa 1:26.2.4-1`; installed wheel `mlx-omarchy 0.32.4.dev202611222118+cf66ff1d0` (source commit `cf66ff1d0`), `mlx-lm 0.31.3`; staged `serve/` + `scripts/qualify_backend_context.py` from this PR branch atop `037fd3d0a` (`catalog.json` sha256 `df1e17aee7abeb50335bc291f5a283cdf2bc9c6178946b5484c1c6a03d84eb67` during the qualification run prior to recording the measured peak).
- **Loaded libraries:**
  - `/usr/lib/omarchy-mlx/venv/lib/python3.14/site-packages/mlx/core.cpython-314-aarch64-linux-gnu.so` sha256 `615f0eb19f8f7d178f79fafee02158c3ac33d1ad165f80d76c86cfb2b94682cc`
  - `/usr/lib/omarchy-mlx/venv/lib/python3.14/site-packages/mlx/lib/libmlx.so` sha256 `05df19bdf83e60ff1865362068c69133e8bae3746837d4303a3880803b778da8`
  - `/usr/lib/omarchy-mlx/vulkan/libvulkan_asahi.so` sha256 `e3b2c831489a1b4d71d65527a437d5d6c193a54c0ef0f47fa3efa1c51d0f5269`
  - `/usr/lib/omarchy-mlx/vulkan/honeykrisp_icd.aarch64.json` sha256 `c335b523edea89c8b71fe6e5e871f0ad8285b298319c4c29b6b512df1b49052b`
- **Kernel & DRM:** `7.3.0-2-20261110.1-lab-ARCH #4 SMP PREEMPT_DYNAMIC Tue Nov 10 06:03:56 UTC 2026 aarch64`, DRM driver `asahi 1.0.0` (`206400000.gpu`), boot `fb9ebb8c-5aca-4fc4-b917-5f42f681443d`.
- **Vulkan device & Mesa:** `Apple M1 (G13G B1)` (8 GPU cores, `unified_memory=1`, `max_recommended_working_set_size=8,111,783,936` B, `cooperative_matrix_{f16,bf16,f32}_8=1`), `Honeykrisp`, `Mesa 26.3.0-devel (git-1650fc1a33)`, Vulkan API `1.4.362`, `driver_version=109060195`.
- **Firmware (`/proc/device-tree/chosen`):** `os-fw-version 13.5`, `system-fw-version 27.0`, `iboot1 mBoot-20457.1.29`, `iboot2 iBoot-8422.141.2`, `m1n1-stage1 v1.6.1-dirty`, `m1n1-stage2 v1.6.1-omarchy.aurora14`.
- **Model & quantization hashes:**
  - Chat: `mlx-community/Qwen3.5-4B-MLX-4bit`, revision `32f3e8ecf65426fc3306969496342d504bfa13f3`, 4-bit affine (`group_size=64`), `model.safetensors` (`3,034,300,695` B) sha256 `5fb9acd0246866381cf8c5c354c6db1019f6498eec4ccb4f5edcc71ffeacb2db`, `config.json` sha256 `f3efc81b2ea8d96a45301037d3ccccbcccdef44a961845c87f286aaddbc6eaaa`.
  - Decision: `laya-mlx` (`convaiinnovations/laya`, revision `55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851`), converted `float16` weights (`Device(gpu, 0)`), `weights_sha256` `891102d372688fc2a094dac56a384bc537b87c63f21f9f3dac0be2b7cbc8d86c`.

## Method and qualification results

Ran `scripts/qualify_backend_context.py` alone on the GPU (`FOREIGN-GPU-USED count=0`, `GPU-TIMEOUTS before=0 after=0 delta=0`). The script prepares `compact` (`qwen3.5-4b-mlx-4bit` + `laya-mlx`) at 4096, starts both managed workers, calls `ensure_context(8192)` with `MLX_OMARCHY_PAIR_DEV_QUALIFICATION=1`, and sends the four HTTP requests directly to the chat worker while sampling system `MemAvailable` and worker RSS every 0.5 s.

| Step (`receipt.json`) | Result |
|---|---|
| `setup("compact", 4096)` + `start()` + `ensure_context(8192)` | ok, context 4096 -> 8192, `chat_prompt_cache_size=1`, chat worker restarted cleanly while `laya-mlx` stayed resident on `Device(gpu, 0)` |
| omp-shaped request (8.6 KB system + 28 tool schemas, `max_tokens=512`) | HTTP 200, 6191 prompt tokens, 44.52 s, reply `"Ok."` |
| needle at the start of a 7615-token prompt, `max_tokens=64` | HTTP 200, 56.12 s, `found: true`, reply `"7316"` |
| prompt over the cap (13384 tokens + 30) | HTTP 404 (`exceeds the admitted context budget of 8192`), 0.02 s |
| small request right after the refusal | HTTP 200 |

**Prefill step.** The worker ran with `--prefill-step-size 512` (`prefill_step_size` in `receipt.json`). Prompt cache at 8192 held 1 sequence at 0.25 GB (versus 0.72 GB for `Qwen3-4B-Instruct-2507-4bit`).

## Memory (GiB, 1024^3 bytes)

| Quantity | GiB |
|---|---|
| Baseline system `MemAvailable` before pair start | 11.46 |
| Chat worker peak `VmHWM` (`6,729,600 kB`) | 6.42 |
| of which peak `RssShmem` (`6,078,960 kB`, GPU buffers) | 5.80 |
| Chat worker `RssShmem` right after restart at 8192, idle | 0.07 |
| Lowest system `MemAvailable` during the run (`5,194,032 kB`) | 4.95 |
| Pair peak footprint (`7,301,744 kB`, baseline minus lowest `MemAvailable`) | 6.96 |
| Catalog estimate at 8192 without prefill scratch | 4.08 |
| Catalog estimate at 8192 with step-512 scratch | 5.32 |

## Backend dispatch trace and Vulkan device reopen

Run on the same boot (`fb9ebb8c-5aca-4fc4-b917-5f42f681443d`), same installed wheel and ICD, alone on the GPU (`FOREIGN-GPU-USED count=0`, `GPU-TIMEOUTS before=0 after=0 delta=0`):

- **Backend dispatch trace (`MLX_OMARCHY_TRACE_DISPATCH=1`):** 16-token `mlx_lm.generate` on the pinned `Qwen3.5-4B-MLX-4bit` snapshot with `mx.default_device() == Device(gpu, 0)` emitted `95,538` `[rtmod]` GPU dispatch/submit trace lines (top dispatched kernels: `kernel=664` x2176, `kernel=58` x1294, `kernel=233` x1168, `kernel=463` x432, `kernel=464` x408, `kernel=516` x408, `kernel=9` x387). (`MLX_OMARCHY_TRACE_DISPATCH=1` traces the GPU/Vulkan encoder and proves GPU kernel execution; a gdb breakpoint on `mlx::core::cpu::get_command_encoder` was not separately attached in this run, see `receipts/2026-09-30-pair-gates/README.md` for the Qwen3.5 GDN zero-CPU gdb audit.)
- **Vulkan device reopen (fresh subprocess):** Opened `Device(gpu, 0)` immediately after generation, ran a `64x256 @ 256x128` fp32 matmul (`max|gpu-cpu| = 0.00e+00`) and a 4-bit `group_size=64` `quantized_matmul` (`max|gpu-cpu| = 2.29e-05`), and exited with `REOPEN_OK` and 0 kernel GPU timeouts.

## Catalog effect

`qwen3.5-4b-mlx-4bit` is marked `qualified` for `generation`, `http`, and `managed` with `extension.backend_context_qualified_tokens = 8192` and `extension.backend_context_peak_bytes = 6,891,110,400` (`6,729,600 kB = 6.42 GiB`). The `compact` pair switches its `chat_model` from `qwen3-4b-instruct-2507-4bit` to `qwen3.5-4b-mlx-4bit` (while `qwen3-4b-instruct-2507-4bit` stays in the catalog with its 16384 / 16.88 GiB peak entry for 24 GiB+ hosts). Above the 4096 default, `pairs.chat_estimate` uses the measured 6.42 GiB peak: with `laya-mlx` (2.41 GiB) and the 2.00 GiB desktop reserve (10.83 GiB total), a 16 GB host with >= 10.83 GiB `MemAvailable` admits 8192 tokens cleanly and stays at <= 4096 when memory is tighter.

**Not measured.** Contexts above 8192 on `qwen3.5-4b-mlx-4bit`, concurrent chat requests at 8192, generation quality beyond the single needle (`7316`), gdb CPU-encoder breakpoint count on the 4B variant, and 8 GB hosts.

**Reproduce.**
1. Qualification run: `MLX_OMARCHY_PAIR_DEV_QUALIFICATION=1 MLX_OMARCHY_OFFLINE=1 HF_HUB_OFFLINE=1 MLX_OMARCHY_HOME=<temp home with converted laya> PYTHONPATH=<serve tree> python scripts/qualify_backend_context.py --home <temp home> --pair compact --tokens 8192 --omp-tools 28 --omp-max-tokens 512 --out receipt.json`
2. Dispatch trace + Vulkan reopen: `HF_HUB_OFFLINE=1 MLX_OMARCHY_TRACE_DISPATCH=1 python -c 'import mlx.core as mx; from mlx_lm import load, generate; m, t = load("<snapshot>"); print(mx.default_device(), repr(generate(m, t, prompt="Reply with the single word ok.", max_tokens=16, verbose=False)[:80]))'` followed by a fresh-process fp32 matmul + 4-bit `quantized_matmul` GPU-vs-CPU parity check.
