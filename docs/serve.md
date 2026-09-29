# Local generation and HTTP serving on Omarchy

The installer includes the serving CLI, Laya decision server, and Bonsai server packages.
The v0.7.4 installer contains all three. Old installations need an explicit update.
Installed code and a successful generation request do not establish model or assistant qualification.

## Current application boundary

The source tree includes the shared application under `serve/mlx_omarchy_assistant/`.
It is not a qualified release. Both default pairs and voice still require end-to-end hardware evidence.
All catalog entries retain `recommended: false`. No code-only check promotes **Ready offline**.

Inspect the source application:

```bash
PYTHONPATH=serve python3 -m mlx_omarchy_assistant --help
PYTHONPATH=serve python3 -m mlx_omarchy_assistant --home /tmp/mlx-chat-check
```

The browser offers pair setup, comparisons, classification, ordinal scores, opt-in history, and local speech controls.
The LLM can draft comparison options, but scoring requires explicit confirmation of the editable draft.
If an explanation claims a different option and omits the Laya choice, the chat says they disagree and leaves the Laya result unchanged. That sentence is part of the saved turn.
History selection includes whole turns. Pinned constraints remain separate and visible.
The terminal uses the same coordinator with `--terminal` or `--prompt TEXT --once`.
Setup approval covers the selected pinned artifacts, including Laya conversion and optional voice assets.
After login, `mlx-omarchy-chat --resume` loads that saved pair and keeps both workers resident until logout.
A reboot clears GPU memory. The user service starts the same saved pair again at the next login.
Opening the launcher attaches to that process. It does not load the weights a second time.
The service does nothing until a pair has been set up, so a fresh install holds no model memory.
To stop the load at login, run `systemctl --user disable --now mlx-omarchy-chat.service`. `install.sh --uninstall` removes the unit.
Do not treat a successful download or process startup as pair qualification.
A missing Parakeet dictation module leaves speech unavailable. It must not stop text setup.

Transfer uses the **Transfer** dialog or `python3 -m mlx_omarchy_assistant.transfer --help`.
Preparation collects the app, speech tools, installed dependency pins, runtime wheels, model files, and approved licenses.
It follows version markers and requested extras, and refuses missing required dependencies.
Export-plan failures leave the inspection control available for retry.
Installation validates the archive and stages a venv without network access before replacing the active files.
Speech scheduling now interleaves: a queued read-aloud parks generation at a real decode boundary and synthesizes between chunks, with bounded waits and an honest busy refusal when the pause cannot be proven. On-hardware pacing qualification is still pending.
The [hardware smoke receipt](../receipts/2026-09-27-offline-assistant/receipt.json) records failed and incomplete gates, not release proof.
Automatic decision routing stays disabled. The held-out suite is frozen and unevaluated at `tests/fixtures/routing_held_out.json`.
Long-context admission still needs measured workspace and latency curves for each chip/runtime.
The complete [design](plans/2026-09-27-offline-assistant-design.md) remains binding.

### Everyday pair gate status, 2026-09-28

Host: M2 Max (T6021), kernel 7.1.13-ARCH-polltx, MLX 0.32.3.dev202609232032+4fd2130ed, source `134d0b67a` plus the fixes below.
The [receipt](../receipts/2026-09-28-everyday-resume/receipt.json) has the numbers.

| Gate | Result |
|---|---|
| Chat, compare, cancel | Pass. Laya chose `cat` over `elephant` (0.7192 / 0.2808). Cancel stopped the turn. |
| Resume after reboot | Pass. Saved pair loaded in 1.41 s. The next chat answered. |
| Restart with outbound sockets denied | Pass, twice. A network namespace with loopback only. Both connection tests failed as intended. |
| First visible answer, 30 warm turns | Pass after a fix. Before: p50 2.86 s, p95 3.24 s. After: p50 0.99 s, p95 1.06 s. Target is p95 2.00 s. |
| Voice output | Intelligible after a backend fix. Whisper large-v3-turbo transcribed the first voice at 4.3% word error and `aiden` at 0.0% on five sentences. The owner listened on 2026-09-28: the first voice was clear but Chinese-accented; `aiden` "sounds good" and is the default. Generation runs 5 to 6 times slower than real time (real-time factor 0.13 to 0.19). First audio p95 1.46 s. |
| Voice input | No Linux path. Parakeet needs the ANE encoder: 5.2 s median on the M1 against a 2 s target, and the T6021 ANE is not live. |
| Voice as a whole | Not qualified. It needs both directions. |
| Quality pair | Not qualified. Ten card prompts gave a valid card on 5 of 8 expected. The other 3 hit my 700-token test cap. One unrequested card appeared. Decode was about 3 tokens/s on a GPU shared with other jobs, so the interactive latency target is unproven. |
| Card generation, Everyday (2B) | Fails. 0 of 8 prompts produced a card, with the full schema, the compact schema, an example, or a reminder. The model writes a markdown list and ignores the fence. Invalid or absent blocks are dropped and the prose stays. |
| Routing held-out suite | Not evaluated. Automatic routing stays off. |
| UX screenshots and accessibility | Pass for six states at 375, 768, 1024, and 1440 px, plus a 200% zoom frame, keyboard, contrast, reduced motion, and semantics checks, with five defects fixed. See the [UI receipt](../receipts/2026-09-28-ui-qualification/README.md). Not run: a real screen reader. |
| Standing M1 battery, zero-CPU trace, peak memory, clean install | Not run. A clean install needs a release that contains the assistant. |

No pair is qualified. All catalog entries keep `recommended: false`.

Defects found by these runs and fixed in source:

- **Speech was a hum.** An elementwise add of two transposed views wrote its output at the wrong positions on the Vulkan backend (max absolute error 6 to 9 against NumPy). The speech decoder uses that add. The fix is in `overlay/mlx/backend/omarchy/primitives.cpp` with a focused test. It is a runtime fix: the v0.7.4 wheel does not contain it. See the [receipt](../receipts/2026-09-28-tts-fix/README.md).
- **Chat prompts carried a 792-token card schema on every turn.** Prefill cost about 2 s. Ordinary chat now sends a 209-token schema with three card types. A message that names a chart, graph, form, decision, options, facts, or sources gets the full schema, and so does every Laya turn.
- **Greedy decoding looped on the 2B model** until the token cap. Chat requests now send a repetition penalty of 1.1.
- **A reboot during pair start left an unclaimed reservation.** Every later start refused with "already held". The reaper now clears an unclaimed record when its creating process is gone.

Known gap: the one-line installer on `main` cannot install the assistant yet. `install.sh` fetches the assistant files from the latest release tag. v0.7.4 was cut before they existed, so those fetches return 404 and the install aborts. A new release that contains the assistant, promoted to latest after a clean install passes, fixes this. Until then, run the assistant from a source checkout.

### Voice options

The pinned voice pack is `mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-4bit`
(mlx-audio 0.5.6). The worker exposes every preset speaker; the default
is `aiden`, an American English male voice. The Details drawer surfaces a
`Voice` select with each speaker's native accent, a `Preview` button
that renders one fixed sentence in the selected voice, and a live-region
announcement for the new choice. The choice is persisted per home in
`voice/voice.json` (mode 0600, atomic) and read by the worker at the next
synthesis request; chat workers stay resident and no pack reload happens.

| Speaker | Native language / accent |
|---|---|
| aiden | American English (default) |
| ryan | English |
| serena, vivian, uncle_fu | Chinese-native; English with an accent |
| ono_anna | Japanese-native; English with an accent |
| sohee | Korean-native; English with an accent |
| eric | Sichuan dialect (Chinese) |
| dylan | Beijing dialect (Chinese) |

Honest note: the pack has no American English female voice. The
non-English-native speakers are surfaced as fully labelled options, accent
included, and refused to fall back to the default when the worker cannot
find the speaker. An unknown voice id is a 400 with the name repeated,
never a silent swap.

The worker asks mlx-audio for the `english` codec token whenever the
text is ASCII; non-ASCII text falls back to the model's auto-detection,
which is also how the dialect speakers (Eric, Dylan) keep their
Sichuan/Beijing dialect when they are used for Chinese text.

## Model status

The catalog records generation, HTTP, and managed-launch qualification separately.
Its records are not proof of a clean installation, paired memory use, or a disconnected assistant session.
Read the exact model revision, runtime, and scope in each receipt before using a result.

| Catalog ID | Role | Generation / HTTP / managed status in catalog | Remaining pair requirement |
|---|---|---|---|
| `qwen3.8-2b-4bit` | Small chat, MLX-LM | Qualified / qualified / untested | Managed launch and paired qualification |
| `qwen3.8-27b-4bit` | Larger chat, MLX-LM | Qualified / qualified / qualified | Resolve recorded numerical-equivalence limits and qualify the pair |
| `laya-mlx` | Typed decisions, dedicated module | Qualified / qualified / untested | Converted artifact, managed launch, paired qualification |

The IDs above reference pinned entries, not floating upstream model names.
The catalog also contains Bonsai and other Qwen variants; they are not automatic assistant defaults.
Some qualification references are missing from this checkout, and historical notes disagree with later catalog flags.
Resolve those references and preserve their original scope before promoting a recommendation.
This documentation update does not certify a new hardware result.

The pinned Qwen3.8-27B text-generation example and its raw output remain in the
[README](../README.md#quick-start) and [install receipt](../receipts/2026-09-20-qwen38-text-install/receipt.json).
That historical CLI run used MLX-VLM; the serving catalog selects MLX-LM through the project shim.
A loader-specific result does not qualify every loader or vision inference.
The 27B checkpoint's roughly 16 GB of weights are not its runtime memory requirement.
Do not select it automatically for a 16 GiB machine.

## Laya typed-decision serving

Laya is a roughly 421M-parameter non-autoregressive model for choices, ordinal scores, and yes/no answers.
It generates no prose. It uses `POST /v1/decisions` rather than chat completions.
The [Laya contract](../serve/mlx_omarchy_laya/CONTRACT.md) specifies request schemas, calibration, and conversion.
Its `rl_agent.act_probability` value is the probability of answering rather than escalating.
Its `confidence` value is one minus normalized entropy, not factual accuracy.

The catalog pins `convaiinnovations/laya` at `1c5edc17a7acd8701df6fc341c0d179f1c62c982`. Convert that source snapshot with the in-repository converter before serving it.
The serving CLI checks for a converted artifact and refuses the raw source snapshot with a conversion hint.
It does not currently perform the conversion for the user.
Chat and Laya retain independent processes, model IDs, and endpoints.
The assistant coordinates these workers without changing their model contracts.

## Serving catalog CLI

Use `mlx-omarchy-serve` after installation, or `omarchy mlx serve` when the command-center integration is available.
From a source checkout, include `PYTHONPATH=serve` so child processes can import the serving packages.
These commands only inspect the catalog or plan memory and disk use; they do not download or launch a model:

```bash
PYTHONPATH=serve python -m mlx_omarchy_serve catalog list --offline
PYTHONPATH=serve python -m mlx_omarchy_serve plan qwen3.8-2b-4bit --context 4096 --offline
PYTHONPATH=serve python -m mlx_omarchy_serve plan qwen3.8-27b-4bit --context 8192 --offline
```

To start one model, use an explicit target and context. This is a manual route, not paired assistant setup:

```bash
mlx-omarchy-serve serve qwen3.8-2b-4bit --context 4096
```

The command checks memory and disk before asking for download approval.
Manual unqualified targets print a warning. Noninteractive download requires `--yes` and an explicit target.
Targets can be catalog IDs, Hugging Face repository IDs, or local model directories.
Use `--help` to inspect the flags for `recommend`, `plan`, `serve`, `catalog`, `reserve`, and `unreserve`.
With all recommendation flags false, `recommend` selects no model.

## Memory and context

The budget counts full model weights, KV for the requested context, workspace, and a system safety reserve.
A mixture-of-experts model must budget all its weights, not only its active parameters.
Use a practical explicit context rather than inheriting the catalog's theoretical maximum.

Admission uses an atomic shared reservation transaction before managed model loading.
A pending reservation counts its full allocation. A resident reservation retains unmaterialized headroom,
subtracting only its verified materialized floor to avoid counting resident memory twice.
Without a verified floor, the full reservation remains counted.
Owner tokens prevent another process from clearing the reservation.
Shutdown releases it only after the owned worker has stopped.
See [budget.py](../serve/mlx_omarchy_serve/budget.py) for the implementation.

A preflight pass is not a measured peak or a guarantee against unrelated applications consuming memory later.
The assistant reserves both models and optional speech workers together through batch admission.
It must never make fit depend on swap, silent model substitution, or CPU tensor fallback.

On the MLX-LM route, [the project shim](../serve/mlx_omarchy_serve/_mlxlm_server.py) enforces prompt plus output within admitted context.
It pins MLX-LM 0.31.3, rejects invalid token arguments, and limits generation concurrency to one.
Prompt-cache entries add to the admitted memory bound.
The oMLX route has no verified server-side context cap and prints a warning.
Module routes use an in-repository allowlist for Laya and Bonsai; arbitrary catalog-named Python modules never execute.

## Offline operation

After the runtime and model artifacts are prepared, `--offline` or `MLX_OMARCHY_OFFLINE=1` disables catalog refresh and model downloads.
The CLI uses cached or bundled catalog data and checks local snapshot completeness before serving.
A missing model produces a refusal, not an attempted download.
This does not make the installer offline: Python dependencies, converted Laya files, and every other required artifact must already exist.

Catalog refresh uses the project's GitHub raw source, an ETag, a five-second timeout, and atomic cache replacement.
The TTL check runs inside serving/catalog commands, not a background daemon.
The [catalog implementation](../serve/mlx_omarchy_serve/catalog.py) defines refresh settings and validation.
The assistant's **Ready offline** gate requires a cold start with outbound traffic denied.
No such end-to-end assistant qualification is claimed here.

Keep all development servers on loopback. Do not expose these unauthenticated endpoints to a LAN or the internet.
No model load enables `trust_remote_code`. Unsupported operations must fail by name rather than run CPU tensors.

## Install and check the backend

```bash
curl -fsSL https://raw.githubusercontent.com/joshuaswarren/mlx-omarchy/main/install.sh | bash
mlx-omarchy -c 'import mlx.core as mx; print(mx.device_info())'
```

Confirm the Apple GPU / Honeykrisp backend, not a software Vulkan device.
Install additional loaders in the same environment as mlx-omarchy; do not
replace its wheel with the upstream macOS package.

## HTTP server contract

The installer includes `mlx-lm==0.31.3`. This low-level example bypasses the managed CLI's memory admission and context shim.
Prefer the managed command above. For direct server development, choose a checkpoint qualified with this exact loader and wheel.

```bash
: "${MODEL_ID:?Set a model already qualified with this server}"
mlx-omarchy -m mlx_lm.server \
  --model "$MODEL_ID" --host 127.0.0.1 --port 8080
```

Keep the server on loopback. These development servers are not an
authenticated production front door; do not expose them directly to a LAN
or the internet. A dummy API key is appropriate only for a loopback endpoint
that does not require authentication.

Inspect `/v1/models`, then use its actual model ID in requests. A successful
model listing is not a generation smoke test: `/v1/chat/completions` must
return a completion ID and nonempty assistant content.

```bash
curl --fail-with-body http://127.0.0.1:8080/v1/models
```

For OpenAI-compatible clients, configure:

```bash
export OPENAI_BASE_URL=http://127.0.0.1:8080/v1
export OPENAI_API_KEY=mlx
```

Select the reported model ID in the client's provider settings. Codex,
omp, pi, Hermes and OpenClaw have client-specific configuration; the common
contract is the endpoint and model ID, not necessarily these environment
variables. Claude Code uses Anthropic Messages rather than OpenAI chat;
it needs a separately configured compatible proxy. Do not assume an
OpenAI URL alone makes that protocol work. Stop a foreground server with
Ctrl-C.

## Historical server measurements — not current model recommendations

The 2026-09-19 comparison used **Qwen2.5-7B-Instruct-4bit**, v0.7.1 on
t6001-test-host (M1 Max), 37 prompt tokens, greedy decoding, 128 maximum output tokens,
and eight interleaved rounds. These are archival results, not predictions
for Qwen3.8 or evidence that those servers load it.

| Server | Historical decode tok/s | Linux scope at measurement |
|---|---:|---|
| mlx_lm.server | 10.36 | Bundled Python server |
| oMLX | 33.5 | Source installation |
| mlx-serve | 5.65 | `linux-vulkan-port` source build; MLX safetensors only |

The measured rate is end-to-end HTTP wall-clock divided by
`usage.completion_tokens`: it includes prefill, detokenization and HTTP
handling, and is not a decode-only figure. All legs were single-stream —
no concurrency was measured — and all four servers were co-resident on
one GPU in one shared window, which is fair across legs but understates
each server's solo absolute rate. mlx-serve's prompt-lookup decoding made
no measurable difference on this prompt (5.65 tok/s on vs 5.70 off). The
earlier two-leg run that day measured the mlx_lm.server leg at 9.33
tok/s versus 10.36 in the four-leg run; the four-leg numbers are the
canonical comparison because every leg shared that window.

See the [four-leg receipt](../receipts/2026-09-19-mlxserve-linux-port-t6001-test-host.md)
and [earlier comparison](../receipts/2026-09-19-serve-options-bench-t6001-test-host.md)
for reproduction, versions and limitations. These measurements establish
nothing about current-generation models: they predate Qwen3.8
qualification and must not be read as a "fastest server for current
models" claim. Linux mlx-serve's GGUF and ANE engines were not
operational in that comparison. Its `/v1/models` ID may be an internal
hash rather than a Hugging Face repository name.

Other historical loader experiments, including the specialized Bonsai
runtime, remain in the [v0.7.0 recertification receipt](../receipts/2026-09-18-v070-pretag-recert-t6001-test-host.md).
They are not drop-in HTTP-serving recommendations.
