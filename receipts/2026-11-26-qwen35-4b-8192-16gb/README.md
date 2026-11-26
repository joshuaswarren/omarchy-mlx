# Compact pair (`mlx-community/Qwen3.5-4B-MLX-4bit`) at 8192-token backend context on 16 GB M1 - 2026-11-26

[Joshua's agent]

**Question.** Pure full-attention `mlx-community/Qwen3-4B-Instruct-2507-4bit` (36/36 full-attention layers, 147,456 B/token KV cache) was previously qualified at 16384 tokens on a 96 GB M2 Max with a 16.88 GiB measured peak, which exceeds a 16 GB host and leaves the 16 GB 13-inch M1 stuck at the 4096-token default. When tested at 8192 tokens on the 16 GB 13-inch M1 against an omp-shaped prompt (6,067 tokens + `max_tokens=512`), `Qwen3-4B-Instruct-2507-4bit` failed at 4608 of 6067 tokens with `DRM_IOCTL_ASAHI_GEM_CREATE failed: Cannot allocate memory` (`BO creation failed`, HTTP 500 `generation_failed`; worker `VmHWM` 7.96 GiB, `RssShmem` 7.55 GiB, pair footprint 9.22 GiB, prompt cache 0.72 GB). Does hybrid `mlx-community/Qwen3.5-4B-MLX-4bit` (Apache-2.0, revision `32f3e8ecf65426fc3306969496342d504bfa13f3`, 3.03 GB weights, 8/32 full-attention layers, 32,768 B/token KV cache) pass managed pair startup, HTTP generation, omp-shaped prompts, needle recall, and over-cap refusal at 8192 tokens on the 16 GB 13-inch M1, and what is its measured peak?

**Method.** `scripts/qualify_backend_context.py` on the 13-inch MacBook Pro M1 (16 GB unified memory, Linux `7.3.0-2-20261110.1-lab-ARCH`), `omarchy-mlx 0.7.50rc2-1`, `mlx-lm 0.31.3`, Honeykrisp Vulkan ICD (`/usr/lib/omarchy-mlx/vulkan/honeykrisp_icd.aarch64.json`), no other GPU workload running (`FOREIGN-GPU-USED count=0`, `GPU-TIMEOUTS before=0 after=0 delta=0`). The script prepares `compact` (`qwen3.5-4b-mlx-4bit` + `laya-mlx`) at 4096, starts both managed workers, calls `ensure_context(8192)` with `MLX_OMARCHY_PAIR_DEV_QUALIFICATION=1`, and sends the four HTTP requests directly to the chat worker while sampling system `MemAvailable` and worker RSS every 0.5 s.

| Step (`receipt.json`) | Result |
|---|---|
| `setup("compact", 4096)` + `start()` + `ensure_context(8192)` | ok, context 4096 -> 8192, `chat_prompt_cache_size=1`, chat worker restarted cleanly while `laya-mlx` stayed resident |
| omp-shaped request (8.6 KB system + 28 tool schemas, `max_tokens=512`) | HTTP 200, 6191 prompt tokens, 44.52 s, reply `"Ok."` |
| needle at the start of a 7615-token prompt, `max_tokens=64` | HTTP 200, 56.12 s, `found: true`, reply `"7316"` |
| prompt over the cap (13384 tokens + 30) | HTTP 404 (`exceeds the admitted context budget of 8192`), 0.02 s |
| small request right after the refusal | HTTP 200 |

**Prefill step.** The worker ran with `--prefill-step-size 512` (`prefill_step_size` in `receipt.json`). Prompt cache at 8192 held 1 sequence at 0.25 GB (versus 0.72 GB for `Qwen3-4B-Instruct-2507-4bit`).

**Memory (GiB, 1024^3 bytes).**

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

**What the catalog does with it.** `qwen3.5-4b-mlx-4bit` is marked `qualified` for `generation`, `http`, and `managed` with `extension.backend_context_qualified_tokens = 8192` and `extension.backend_context_peak_bytes = 6,891,110,400` (`6,729,600 kB = 6.42 GiB`). The `compact` pair switches its `chat_model` from `qwen3-4b-instruct-2507-4bit` to `qwen3.5-4b-mlx-4bit` (while `qwen3-4b-instruct-2507-4bit` stays in the catalog with its 16384 / 16.88 GiB peak entry for 24 GiB+ hosts). Above the 4096 default, `pairs.chat_estimate` uses the measured 6.42 GiB peak: with `laya-mlx` (2.41 GiB) and the 2.00 GiB desktop reserve (10.83 GiB total), a 16 GB host with >= 10.83 GiB `MemAvailable` admits 8192 tokens cleanly and stays at <= 4096 when memory is tighter.

**Not measured.** Contexts above 8192 on `qwen3.5-4b-mlx-4bit`, concurrent chat requests at 8192, generation quality beyond the single needle (`7316`), and 8 GB hosts.

**Reproduce.** `MLX_OMARCHY_PAIR_DEV_QUALIFICATION=1 MLX_OMARCHY_OFFLINE=1 HF_HUB_OFFLINE=1 MLX_OMARCHY_HOME=<temp home with converted laya> PYTHONPATH=<serve tree> python scripts/qualify_backend_context.py --home <temp home> --pair compact --tokens 8192 --omp-tools 28 --omp-max-tokens 512 --out receipt.json`
