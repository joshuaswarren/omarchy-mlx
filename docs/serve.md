# Local generation and HTTP serving on Omarchy

The installer includes the serving CLI, Laya decision server, and Bonsai server packages.
The v0.7.6 installer contains all three, plus the MLX Chat assistant. Old installations need an explicit update.
Installed code and a successful generation request do not establish model or assistant qualification.

## Current application boundary

The source tree includes the shared application under `serve/mlx_omarchy_assistant/`.
It is not a qualified release. All three default pairs and voice still require end-to-end hardware evidence.
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
Automatic decision routing stays off; [gate status](#per-pair-gate-status-2026-10-01) has the measured reason. The held-out suite is now spent. Explicit **Compare options** is unaffected.
Long-context admission still needs measured workspace and latency curves for each chip/runtime.
The complete [design](plans/2026-09-27-offline-assistant-design.md) remains binding.

### Per-pair gate status, 2026-10-01

Since `8a1e25843` the default pairs are Everyday = `qwen3.5-9b-mlx-4bit` + Laya,
Compact = `qwen3-4b-instruct-2507-4bit` + Laya, and Quality = `qwen3.8-27b-4bit`
+ Laya; the 2B left the catalog ([card promotion
receipt](../receipts/2026-09-30-card-promotion/README.md)). The 2026-09-28 runs
([receipt](../receipts/2026-09-28-everyday-resume/receipt.json)) measured the
retired 2B pair and stay history.

Why the defaults changed: on the fixed stack — the GDN prefill repeat fix
(`9ef622d14`, [receipt](../receipts/2026-09-30-gdn-prefill/README.md), 9B
prefill 46.8 to 316.9 tok/s) and the release wheel `+06711ad` (the earlier live
wheel returned non-finite logits on long GDN prompts) — the 2B is last on every
quality proxy measured (0/8 native cards, GSM8K 11/20, IFE 17/20) while the 9B
(0.78 s TTFT) and the 4B (0.44 s) hold the 2.0 s first-text budget ([chat-model
bench](../receipts/2026-09-30-chat-model-bench/README.md)).

Unless a row names another build, every number below was measured on the M2 Max
(T6021, 96 GB) with the v0.7.6 release wheel `0.32.3.dev202609291615+06711ad`
(provenance `verified: match`).

| Gate | Everyday (9B) | Compact (4B) | Quality (27B) |
|---|---|---|---|
| Cards, HELD-OUT v4 (frozen; pass needs 15/18 valid, 0 spurious) | 16/18, 0 spurious — pass | 18/18, 0 spurious — pass | 18/18, 0 spurious — pass |
| First-text p95, chosen config, stock kernel | 1.05 s — pass | 0.66 s — pass | not part of this gate |
| First text on the real card prompt, engine level | 0.78 s — within the 2.0 s design budget | 0.44 s — within | 2.45 s — over budget |
| Paired memory peak over baseline (whole system, run 1 per pair) | 10.88 GiB | 4.80 GiB | 19.26 GiB |
| Tier fit | 16 GB and 96 GB | 16 GB and 96 GB | 96 GB only |
| Pair qualified | No | No | No |

Tier fit is host-RAM arithmetic on the measured 96 GB peaks; no 16 GB machine was measured ([chat-model bench](../receipts/2026-09-30-chat-model-bench/README.md), [pair gates receipt](../receipts/2026-09-30-pair-gates/README.md)).

Cards: the product ships `card_format` unset everywhere (markdown promotion, no
schema on ordinary chat). The fenced-json schema was measured and rejected by
the pre-registered rule: the 4B wrote 18/18 cards with it but first-text p95 was
2.40 s, and the 9B had one spurious card (17/18) — see the [card promotion
receipt](../receipts/2026-09-30-card-promotion/README.md). The bench card
protocol leaves the same two prompts (the Apollo timeline and the decision)
without a card on both the 9B and the 4B ([chat-model
bench](../receipts/2026-09-30-chat-model-bench/README.md)).

**Quality performance — the design budget is not met.** Through the assistant
API with both workers resident ([pair gates
receipt](../receipts/2026-09-30-pair-gates/README.md), run 1): prefill 73–81
tok/s (about 75–85 % of the 97–103 tok/s the same wheel reaches alone on the
GPU), decode about 1.4 tok/s behind a 5,408-token prompt, and first visible
text 2.5–14.4 s on card turns against the 2 s design budget. A 20,956-token
prefill completed in about 258 s; the harness's own 600 s per-turn deadline
then stopped the turn before its one-token decode — a harness limit, recorded
as such. The 262,144-token admission is the catalog model maximum, not a
memory-admitted limit.

**Routing — the held-out suite passed on precision; routing stays off.** Frozen
policy 3 (commit `50ca49fae`) scored precision 1.000 (35/35, 0 false
positives), recall 1.000, and 0 of 15 injection cases routed to a decision; the
head was called on 0 of 100 held-out turns. The shipped head-free decision path
runs at p95 43 ms on the M2 CPU. The Laya head call itself took p95 347 ms
(p50 309 ms, 100 warm calls) against the 250 ms warm deadline. Automatic
routing stays off until the owner decides which latency the gate measures
([routing receipt](../receipts/2026-09-30-routing-gate/README.md)). The
held-out suite is now spent.

**Voice input — every frozen threshold passed; not qualified as a pair gate.**
On the 192-clip corpus with the pinned `parakeet-tdt-0.6b-v3` (`ed2b7e8c…`):
WER 3.42 % test-clean (≤ 6 %), 2.92 % test-other (≤ 14 %), 4.39 % accented
(≤ 20 %), and 11.43 % in 0 dB babble (≤ 30 %); silence and pink noise returned
empty on 10 of 10 each ([speech input
receipt](../receipts/2026-09-30-speech-input-gpu/README.md)). The test-clean
figure scores one 30.04 s clip through the product's 30.0 s cut; counted as a
refusal, test-clean is 8.15 % and fails. The recognition worker made 0
CPU-stream dispatches (152 before the models-package fix). Empty transcripts
are fixed: 3 of 96 clear-speech uploads returned nothing, and the voiced-clip
retry returns 0 of 96; padding every request and dithering the clip were
measured and rejected. Browser run, n=30 after the fix: 0 empty, p50 861.8 ms,
p95 1397.8 ms against the 2 s budget, and the 375/1440 px scenarios (permission
denied, device loss, 30 s limit, cancel) pass. The Orca pass below covers the
UI states; no screen-reader run has driven a live dictation.

**Voice output — the real-time threshold is not met, for either engine.** The
default Qwen3-TTS pack measures median RTF 0.22–0.23 (audio s over wall s)
against the 1.2 threshold, and the named floor is about 87 ms of GPU compute
per talker step — about 5.2 s of GPU work per 5 s of audio at 12.5 Hz, at
43–47 µs per dispatch ([speech output speed
receipt](../receipts/2026-09-30-speech-output-speed/README.md) and
[NAMED-FLOOR](../receipts/2026-09-30-speech-output-speed/NAMED-FLOOR.md)). The
Attn128 fused head_dim-128 decode attention cuts a full frame from 4,244 to
3,362 dispatches (167–176 to 134–137 ms p50) — still about 2.3× the
1,488-dispatch budget for RTF 1.2
([Attn128](../receipts/2026-09-30-attn128/README.md)). Per Main's direction the
floor ships: Qwen3-TTS stays for non-real-time synthesis, with a streaming
first sentence audible in about 1.4 s. The Kokoro-82M second engine fails its
own frozen thresholds — RTF 0.69 against a required 5 or more, and first-audio
p95 10.5–10.8 s against 1.0 s (WER 0.87 % passes) — so it ships behind the
picker, is not the default, and is not recommended; the owner listens before
any decision ([Kokoro receipt](../receipts/2026-09-30-speech-output-kokoro/README.md)).
Voice as a whole stays unqualified: it needs both directions.

**Zero-CPU traces.** The chat (2B), decision, TTS, and 9B GDN chat paths each
measured 0 CPU tensor-primitive dispatches through a gdb breakpoint on
`mlx::core::cpu::get_command_encoder`, with live controls: one `mx.add` on an
explicit `mx.cpu` stream fires 3 encoder calls, the same op on `mx.gpu` fires 0.
The `[rtmod] DISPATCH` facility alone cannot prove this — it instruments only
the GPU encoder ([pair gates receipt](../receipts/2026-09-30-pair-gates/README.md)).
Finding, still open: an explicit `stream=mx.cpu` still executes CPU primitives
in the release wheel (`binary_op_cpu<Add>` and similar instantiations live in
`libmlx.so`, untraced by the dispatch facility). The product paths never trip
it; a caller that passes a CPU stream can bypass the contract.

**Standing battery and pin state.** The 13-inch M1 battery passed 26/26 suites
at `db74f11ad` ([M1 battery
receipt](../receipts/2026-09-30-m1-battery/README.md)). At the [mlx pin
bump](../receipts/2026-10-01-mlx-pin-bump/README.md) tip (`aabe46c3c`) the
T6001 battery passed 29/30; its one failure, the bf16 block of
`omarchy_indexing_ops_tests`, was later shown to be two test bugs rather than a
backend defect and is fixed on main (`e00b37116`, [Attn128
corrections](../receipts/2026-09-30-attn128/README.md)). The pin bump's token
digests are bit-identical on both chips; its merge gate stays held pending the
G13G re-run after that host's reinstall. Mesa: the omacom v2/v3 driver stacks
keep every digest bit-exact on every chip, and v3 is pinnable on T6001
evidence, but both stacks collapse pure prefill about 5.4× on the M2 chip
(G14) against the deployed driver — consistent with the coopmat matmul path
the deployed pre-gating build uses on G14 and the omacom stacks gate to G13
(the receipt's labeled hypothesis; ready falsifier `AGX_SIMDMAT=1`) — so the
Mesa pin is blocked for that chip ([Mesa v2
parity](../receipts/2026-10-01-mesa-v2-parity/README.md)).

**Runtime and install gates.** The packaged DKMS ANE module passed the worker's
ABI-1 acceptance on T6001 (bit-exact h13 add-mul bundle, per-user fallback,
negative controls); Honeykrisp ICD selection now refuses a missing override
JSON, and release builds raise instead of silently falling back to the CPU
device ([runtime gate](../receipts/2026-09-30-runtime-gate/README.md)). The
offline `--system` stage ran green inside a network namespace on T6001 with 36
vendored aarch64 wheels; two bugs were found on hardware and fixed test-first
([system install receipt](../receipts/2026-09-30-system-install-hw/README.md)).

**Screen reader — pass.** A real Orca run drove the static UI through ten
states (setup, chat stream, escape, decision, card, compare, drawers, voice
unavailable, error, offline chip) in a container, with utterances captured
verbatim. One defect was found and fixed: the setup checkbox announced
"invalid entry." until `setup.js` set `aria-invalid="false"` ([receipt](../receipts/2026-09-30-screen-reader/README.md)).
Landmark, heading, and table navigation are not exercisable in that container.

Defects these runs found and fixed in source: chat requests carry a repetition
penalty of 1.1 because greedy decoding looped on the 2B until the token cap;
the STT worker made 152 CPU-stream calls through a float64 filterbank built at
models-package import until the worker registered that package without running
its `__init__`; the recorder requested browser noise suppression, which doubled
the captured level (median 2.14×) until it asked for unprocessed audio; and the
voiced-clip retry above. Two defects are still open: the Compact 4B card turn
that returned a silent empty reply (run 1, `status=complete`, `text_len` 0),
and the explicit-CPU-stream finding above.

### Open items for the owner

| # | Item | Why it blocks |
|---|---|---|
| 1 | Routing latency decision: does the 250 ms gate measure the Laya head call (p95 347 ms, fails) or the shipped head-free path (p95 43 ms, passes)? | Automatic routing stays off until decided. |
| 2 | TTS real-time route: Qwen3-TTS RTF 0.22–0.23 vs 1.2; Kokoro RTF 0.69 vs 5. | No engine meets its real-time threshold; voice output stays unqualified. |
| 3 | Pair qualification and the card-rule gap: no pair has passed the full gate set, and the 9B and 4B produce no card for the Apollo-timeline and decision prompts. | Nothing is qualified; `recommended` stays false everywhere. |
| 4 | G14 Mesa coopmat: both omacom stacks collapse prefill about 5.4× vs the deployed driver; the hypothesis has a ready falsifier (`AGX_SIMDMAT=1` on v3). | Blocks the Mesa pin for the M2 chip. |
| 5 | G13G re-run on the reinstalled jwm1, and one G13-class run of the Attn128 16-bit selection-route doctest. | Gates the mlx pin bump merge; the selection route stays float32-gated until then. |

No pair is qualified. All catalog entries keep `recommended: false`.

The one-line installer ships MLX Chat: `install.sh` fetches the assistant, the serve CLI, and the wheel from the promoted release tag (v0.7.6, installed-from-release gates green, including both Laya fresh-install paths; see `receipts/2026-09-30-v076-release.md`).

### Voice options

Two pinned engines are registered; the Details drawer groups the picker by
engine, and the default stays Qwen3-TTS until the owner accepts the second
engine by listening.

The default engine is `mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-4bit`
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

The second engine is `mlx-community/Kokoro-82M-bf16` (revision
`a71e4d38…b1c3c`, Apache-2.0, 24 kHz, files sha256-pinned in
`KOKORO_PACK`). It is non-autoregressive: one or a few forward passes per
sentence instead of one per 12.5 Hz frame, so streaming never underruns.
Its G2P front end is misaki 0.7.4 (English) with a user-space espeak-ng
wheel (`espeakng-loader`, `phonemizer`) as the out-of-vocabulary fallback —
no root and no system package; everything installs into the pack's own
runtime, and nothing touches the network at run time. Voices come from the
pack's own tensors:

| Speaker | Accent |
|---|---|
| af_heart (engine default) | American English |
| af_bella | American English |
| am_michael | American English |

Each engine keeps its own resident worker, so switching engines does not
reload the other. Picking a voice picks its engine; engine assets are
verified with the same hash gate as the default pack, and an engine whose
assets are not downloaded is shown disabled in the picker, not silently
offered.

The worker asks mlx-audio for the `english` codec token whenever the
text is ASCII; non-ASCII text falls back to the model's auto-detection,
which is also how the dialect speakers (Eric, Dylan) keep their
Sichuan/Beijing dialect when they are used for Chinese text.

## How cards are produced

A chat reply shows prose only, prose plus a card the model emitted, or prose
plus a card the application built from the reply.

1. **Text streams first.**  Text before an ```` ```assistant-ui ```` fence is
   shown as it arrives; the fenced JSON is buffered, validated by
   `components.validate_components`, and gets at most one repair call.  A
   valid fenced card always wins.  A block that stays invalid is dropped and
   the prose stays.
2. **Card promotion.**  When a chat turn ends without a valid fence,
   `card_promotion.extract_text(reply, user_text)` may build one card from
   the reply's markdown, outside fenced code.  It needs both a request for
   the artifact in the user's message and a matching structure in the
   reply:
   - checklist ("checklist", "to-do", "action items", "packing list", or
     "steps"): a task list or bullet/numbered list of 3 or more items;
   - comparison ("compare", "contrast", "versus", "side by side", "in a
     table"): a pipe table with 2-8 columns and 2 or more rows of equal
     width, or 2 or more headed sections of bullets, one row per section
     with a column per bullet label (`**Price:** ...`) the sections share;
   - timeline ("timeline", "schedule", "agenda", "roadmap", "milestones",
     "phases", "plan my Monday"): 3 or more items or headings that start with
     a time, date, weekday, `Week 2`-style period or a short label, or a table
     whose first column is the time;
   - facts ("facts", "key points"): 2 or more list items.

   "Explain", "what is", "why" and "how does" questions stay prose even when
   the reply has a table ("describe the phases of the moon" too), and "in a
   paragraph", "no bullets" or "without using a list" turns promotion off.
   The card is validated before it is emitted and its title
   ends with `(from reply)`.  Input over 1 MiB is refused and every pattern
   runs in linear time.
3. **Schema policy.**  Ordinary chat turns send no card schema.  The full
   schema is sent on Laya turns, when the message names something markdown
   cannot carry (chart, graph, form, decision, options, facts, sources), and
   for chat models whose catalog entry declares
   `extension.card_format: "fenced-json"` (at least 6 of 8 valid fenced
   cards, measured).  No catalog entry declares it today.

## Model status

The catalog records generation, HTTP, and managed-launch qualification separately.
Its records are not proof of a clean installation, paired memory use, or a disconnected assistant session.
Read the exact model revision, runtime, and scope in each receipt before using a result.

| Catalog ID | Role | Generation / HTTP / managed status in catalog | Remaining pair requirement |
|---|---|---|---|
| qwen3.5-9b-mlx-4bit | Everyday chat, MLX-LM | Untested / untested / untested | Generation, HTTP, managed launch, and pair qualification |
| qwen3-4b-instruct-2507-4bit | Compact chat, MLX-LM | Untested / untested / untested | Generation, HTTP, managed launch, and pair qualification |
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

The catalog pins `convaiinnovations/laya` at `55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851`. Convert that source snapshot with the in-repository converter before serving it.
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
PYTHONPATH=serve python -m mlx_omarchy_serve plan qwen3.5-9b-mlx-4bit --context 4096 --offline
PYTHONPATH=serve python -m mlx_omarchy_serve plan qwen3-4b-instruct-2507-4bit --context 4096 --offline
PYTHONPATH=serve python -m mlx_omarchy_serve plan qwen3.8-27b-4bit --context 8192 --offline
```

To start one model, use an explicit target and context. This is a manual route, not paired assistant setup:

```bash
mlx-omarchy-serve serve qwen3.5-9b-mlx-4bit --context 4096
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

System packaging builds this same serving stack offline: `packaging/build-venv.sh`
creates the private venv from vendored, hash-locked wheels and
`install.sh --system` stages `/usr/lib/omarchy-mlx/venv` with the serve
launchers (`packaging/PKGBUILD.example` documents the recipe shape). The
serve CLI and the assistant discover their venv through
`serve/mlx_omarchy_paths.py`: `$OMARCHY_MLX_VENV`, then
`/usr/lib/omarchy-mlx/venv`, then the legacy `~/.local/share/mlx-omarchy/venv`
(with a one-line hint naming `mlx-omarchy-retire-legacy`).

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
