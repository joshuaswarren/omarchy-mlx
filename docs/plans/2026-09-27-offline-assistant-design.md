# Offline assistant design

Status: source application implemented and tested offline. The defaults are Everyday = Qwen3.5-9B + Laya, Compact = Qwen3-4B-Instruct-2507 + Laya, and Quality = Qwen3.8-27B + Laya (2026-09-30; the 2B left the catalog). Measured on the M2 Max through 2026-10-01: card gates pass on all three pairs, first-text latency passes on the 9B and 4B, speech input meets every frozen corpus threshold, and the screen-reader pass is done; the Quality pair misses the 2 s first-text budget, speech output misses the RTF 1.2 threshold, automatic routing stays off on the head-latency deadline, and no pair is qualified. See [gate status](../serve.md#per-pair-gate-status-2026-10-01).

## Product contract

Open MLX Chat, type or speak, and receive an answer on the laptop without an account or network connection.
The application chooses the right local model without asking the user to manage servers.
A small decision model handles bounded choices. A language model handles conversation and explanations.
Both share one conversation, one cancellation control, and one setup flow.

This document specifies the complete application, not a new inference backend.
The source implementation and remaining gaps are recorded in the [serving guide](../serve.md#current-application-boundary).
A text release may ship before voice, but it must say that voice remains unavailable.
The complete assistant is not done until the voice and text acceptance gates both pass.

## What exists

| Component | Source | Current boundary |
|---|---|---|
| Terminal chat | [demo/chat.py](../../demo/chat.py) | Uses the shared coordinator; no direct weight loading |
| Serving and installation | [serving guide](../serve.md), [install.sh](../../install.sh) | Loopback app and owned model workers; installed pair qualification pending |
| Typed decisions | [Laya contract](../../serve/mlx_omarchy_laya/CONTRACT.md) | Explicit comparison, classification, and ordinal scores in source; paired hardware qualification pending |
| Model selection and memory | [catalog](../../serve/mlx_omarchy_serve/catalog.json), [budget](../../serve/mlx_omarchy_serve/budget.py) | Atomic pair admission and byte-based sizing; measured long-context evidence pending |
| Speech recognition | [Parakeet](../parakeet.md#installed-product-wheel) | Fixture route retained; arbitrary-input source path remains unqualified |

All catalog entries currently have `recommended: false`.
Some receipt references and historical status statements disagree; qualification must resolve the exact artifact and managed launch path.
Do not promote a model or pair from a neighboring model's results.

## One application, existing engines

Ship a loopback web application opened by the existing Omarchy desktop launcher.
Keep the UI assets inside the installed package: no CDN, web fonts, remote scripts, or browser speech service.
Use semantic HTML, CSS, and small JavaScript modules rather than a new frontend framework.
Keep Python orchestration beside the existing serving packages.
The terminal command uses the same conversation coordinator instead of loading a second set of weights.

The coordinator owns conversation state, turn IDs, policy, cancellation, and model process lifetime.
Reuse the catalog, download approval, artifact checks, context caps, and memory reservation code.
Keep chat and Laya in independent workers behind their existing APIs.
Do not add a fake chat endpoint to Laya or change the public MLX device contract.

The UI calls one same-origin coordinator endpoint. It never chooses arbitrary model URLs.
The coordinator streams ordered events: `status`, `decision`, `text`, `audio`, `error`, and `done`.
Each event carries a conversation ID, turn ID, and monotonic sequence number.
Discard events for cancelled or superseded turns. Never replay a submitted turn after reconnect without explicit retry.
A reconnect reads the existing turn state and resumes display, not inference.
Keep one active turn per conversation and one admitted generation per GPU initially.
Show waiting state for another tab instead of creating an unbounded queue.

Bind only to loopback. Validate Host and Origin, reject cross-origin requests, and require a per-launch session secret.
Serve a restrictive Content Security Policy and sanitize generated Markdown; disable raw HTML and remote images.
Do not put the secret in a persistent URL or log it.
A browser tab closing stops its microphone. A short disconnect grace period cancels its owned active work.
Closing the application stops only its owned workers and releases reservations only after verified process exit.
Do not terminate a separately started local server.

## How the two models cooperate

Use deterministic application rules first. Do not put an inference call in front of every keystroke or button.
Permission, microphone state, model selection, and action authorization are never model decisions.
The LLM remains the normal path for free-form conversation, writing, code, and questions requiring explanation.
Laya is useful when the task supplies a bounded answer space, not as a universal truth checker.

| User intent | Execution | What appears |
|---|---|---|
| Ordinary conversation | Stream the LLM directly | Normal chat response |
| Choose between explicit alternatives | Validate alternatives, ask Laya one `choice` question | A decision card; optional LLM explanation |
| Classify or score supplied material | Validate a fixed schema, batch Laya questions | Typed result cards, with source text available |
| Explain a decision | Give the LLM the original input and validated Laya result | Explanation beside the unchanged result |
| Missing criteria, long input, abstention, or unsupported language | Use the LLM to explain or ask a specific question | No invented confident decision |

The composer has an optional **Compare options** control, not a technical model picker.
It accepts 2-8 alternatives and explicit criteria. These bounds limit both UI complexity and decision workspace.
A natural-language comparison can produce an editable options draft through the LLM.
Validate that draft, show the extracted options and criteria, and get confirmation before scoring it.
Never silently invent candidates or treat LLM-extracted facts as user-provided facts.
Keep option order and identifiers stable through request, response, display, and explanation.

Automatic routing is advisory and only runs on a completed, bounded user turn that appears to request a structured decision.
A versioned Laya `choice` question selects `conversation`, `structured_decision`, or `clarify`.
It must preserve the full material input within the tokenizer budget; otherwise skip routing and use the LLM.
Give routing a 250 ms warm deadline; timeout or invalid output immediately selects the LLM.
Do not wait for a cold Laya load before starting ordinary chat.
A timed-out routing request remains accounted for until its worker actually finishes or stops.
Do not launch unlimited replacement calls.

Routing thresholds belong to the versioned pair policy and need held-out task evidence before release.
Require the selected-label probability, the runner-up margin, and `rl_agent.act_probability` to pass their frozen thresholds.
Do not treat `confidence` as probability of correctness: the current API defines it as one minus normalized entropy.
Do not invert `act_probability`: high means answer, low means escalate.
Laya's scores are ordinal expected values, not ranks or factual confidence.
No automatic decision route ships before the routing acceptance gate passes; explicit comparison remains available.

The catalog Laya variant has a 512-token limit, including instructions, options, special tokens, and state.
Count with Laya's tokenizer and the same question renderer before dispatch.
Its renderer can shorten option labels, instructions, and state. Reject any such truncation before dispatch.
Compare the complete tokenizer inputs against the rendered sequence; do not ask a model whether lost text mattered.
Only omit whole conversation turns that the user did not include in the decision input.
Never shorten a supplied negation, constraint, option, or attribution.
If the required material does not fit, offer **Use the chat model** instead of silently changing the question.
A translated or summarized input is not equivalent to the original and needs separate qualification.
The root English checkpoint is not a multilingual decision-model claim.

A decision card shows the selected option, criteria, and **How this was decided** details.
Details include the model, input scope, probability distribution, and whether the model abstained.
Display an uncertain result as uncertain. Do not show a green accuracy badge or invented reasoning from Laya.
The LLM can explain the supplied data but cannot overwrite the typed result while calling it a Laya result.
On disagreement, preserve both outputs and say they disagree.
No shell execution, file modification, email, purchase, or background autonomous action belongs to this chat design.

## Default model pairs

These are the selected product defaults to qualify, not currently ready catalog recommendations.
The application must ship ready pairs, not require a Hugging Face identifier or two manual server commands.
Pair readiness is a release gate, not something the user must establish themselves.

| Pair | Chat catalog ID | Decision catalog ID | Context policy | Selection |
|---|---|---|---|---|
| Everyday | qwen3.5-9b-mlx-4bit | laya-mlx | Calculated per machine and workload | Default chat model for machines where the full pair fits |
| Quality | qwen3.8-27b-4bit | laya-mlx | Calculated per machine and workload | Existing higher-capacity pair |
| Compact | qwen3-4b-instruct-2507-4bit | laya-mlx | Calculated per machine and workload | Fallback when the Everyday pair does not fit admission |

Do not encode RAM tiers or a list of supported memory sizes into selection logic.
The same calculation handles every reported byte capacity, including unusual sizes and changing available memory.
These pair names select model roles, not fixed context sizes or hardware classes.
The earlier 4,096/8,192 contexts and 512/1,024 output caps are replaced by the adaptive policy below.
Laya separately accepts at most 512 input tokens and produces zero output tokens; more RAM cannot extend that model contract.

Setup evaluates qualified pairs against available memory, the requested task, quality evidence, and latency evidence.
Offer **Fast**, **Balanced**, and **Long context** preferences; Balanced is the initial selection policy.
Balanced picks the highest task-quality pair meeting the qualified interactive-latency target and memory admission.
Use measured task quality rather than parameter count or weight size as the ranking criterion.
Fast prefers response latency; Long context prioritizes the requested input length and exposes expected prefill cost.
No suitable pair means a named compatibility or memory refusal, not an unqualified model or CPU fallback.
More RAM can permit a larger or higher-precision qualified chat model, more context, or more retained cache.
It does not force any of those choices or trigger a download without approval.
Keep exact revisions stable. Pair changes preserve the conversation and revalidate active context before the next turn.

### Adaptive context for every memory size

Distinguish saved conversation history, active model context, output allowance, and cache residency in both state and UI.
Saving a long conversation does not mean feeding its complete history into every request.
The active limit always includes system instructions, chat-template tokens, selected history, attachments, UI schema, and reserved output.
Choose output allowance from the task and user request; do not retain a universal 512-token ceiling.
Reserve that allowance before inference. Offer **Continue** when a response reaches it, without silently expanding the reservation.

Compute the admissible context as the largest token count that satisfies all four limits:

1. The pinned model's supported positional limit and any required, separately qualified positional scaling.
2. The backend's qualified context limit for this model, dtype, chip, and runtime.
3. Atomic admission for the complete pair, optional voice, caches, workspace, and system headroom.
4. The selected latency policy, using measured prefill/decode curves rather than assuming that RAM makes inference fast.

Use a monotone memory upper bound and integer search over token counts, aligned only to actual allocator block requirements.
Do not round down to RAM tiers or require a power-of-two memory capacity.
An explicit long-context request can relax the interactive latency preference after showing the expected wait, but not memory or model limits.
Unknown long-context cost is unqualified, not free memory. Qualification must establish a safe bound before automatic selection.

For a nonresident model, budget weights + KV bytes per token times context + context-dependent peak workspace.
Budget recurrent state, cache copies, prefill buffers, temporary tensors, and allocator overhead where applicable.
The current catalog's 25%-of-weights margin is an unmeasured estimate; it is not a valid long-context qualification by itself.
Attention workspace can grow much faster than the KV cache. Use bounded prefill and measured peaks for each supported path.
For resident workers, reuse verified resident floors and outstanding headroom from the existing transaction logic; never count resident weights twice.
Recheck live `MemAvailable`, external reservations, and a configurable desktop reserve before admitting growth.
The design default desktop reserve is the greater of 2 GiB and 10% of current `MemAvailable`, held for future demand.
This extends the current fixed 2 GiB reserve; it is not a claim about the existing implementation.
Show the complete allocation in Details, including the reserve and memory owned by other applications.

Start with only the context the task needs, up to its admitted limit; do not pad a short prompt to the maximum.
Before growth, atomically revise the affected reservation and server cap together at a turn boundary.
If the current worker cannot resize safely, drain it and restart with the new cap; preserve the draft and conversation.
A failed expansion retains the prior cap. Never launch a second unbudgeted copy of the model.
Pressure must not silently discard history, change the model, or reduce the promised output during a turn.
Offer explicit choices to close other workloads, reduce selected history, summarize with disclosure, or choose another qualified pair.
Pinned user constraints stay visible and cannot disappear through automatic summarization.

The current Qwen chat entries declare 262,144 tokens; that metadata is not proof of full-length runtime qualification.
A 512 GB machine does not turn a 262,144-token model into a million-token model.
Contexts above that require a different qualified checkpoint or separately validated positional-extension configuration.
Extra memory can still support higher-precision weights, retained caches, and future larger qualified models.
The catalog's 27B 8-bit and BF16 entries remain untested; do not offer them as ready merely because they fit.
Hardware/chip support remains a separate check from memory size.

For scale only, the 27B Q4 catalog estimates 65,536 KV bytes per token:

| Active context | Analytic KV memory only |
|---|---|
| 65,536 tokens | 4 GiB |
| 131,072 tokens | 8 GiB |
| 262,144 tokens | 16 GiB |

These are arithmetic examples, not presets, measured peaks, or fit claims for any machine.
Weights, workspace, Laya, voice, and headroom are additional. Memory alone need not force a 4K/8K context.
The UI shows **Using X of Y tokens** and the reason for the current limit: model, backend, memory, or latency.
Keep the numerical detail available without making users calculate a KV cache themselves.

Exact candidate pins, taken from the existing catalog:

| Catalog ID | Source | Revision |
|---|---|---|
| qwen3.5-9b-mlx-4bit | mlx-community/Qwen3.5-9B-MLX-4bit | 938d8919941c6e7efd3c7150eff7fe9d12afa631 |
| qwen3-4b-instruct-2507-4bit | mlx-community/Qwen3-4B-Instruct-2507-4bit | 50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b |
| qwen3.8-27b-4bit | mlx-community/Qwen3.8-27B-4bit | 10c35caafbb80f7dc6a7a432cdd11af10a6d4818 |
| laya-mlx | convaiinnovations/laya | 55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851 |

The catalog weight totals are approximately 6.79 GB for Everyday, 16.90 GB for Quality, and 3.11 GB for Compact (decimal bytes, chat plus Laya).
These are decimal weight bytes only, not download totals, disk requirements, or runtime memory requirements.
Show actual complete download sizes during setup, including tokenizers, runtime dependencies, conversion output, and optional voice assets.
Laya uses the in-repository converter; its raw source snapshot is not a servable MLX checkpoint.
Do not substitute the separately pinned community pack without requalification.

Extend the existing catalog schema with pair records; do not create a competing model catalog.
A pair references model IDs, adaptive context/output policy, routing-policy version, supported chip/runtime combinations, and public qualification receipts.
Runtime locks additionally record exact revisions, file hashes, conversion provenance, and dependency versions.
An optional voice profile has its own readiness and artifact manifest.
Existing single-model `recommended` flags cannot prove pair co-serving or voice readiness.
Promote pair readiness only after clean installation, managed co-serving, cancellation, and disconnected restart pass on the named hardware.
The pair receipt must name the converted Laya artifact hash, converter commit, and actual eight-question server limit.
It must close the 2B and Laya managed-launch gaps and the larger model's recorded numerical-equivalence limits.

### Memory and process ownership

Use the existing `budget.py` lock and owner-token checks, extended with a batch admission operation.
Under one lock, check the combined requirement and create one pending record per child with distinct owner tokens.
A pair ID groups those records for cleanup; do not add another byte-counted parent reservation.
Pass each child its token through a private inherited pipe, never a command line or browser response.
Add a managed-start claim operation that validates the token, allocation, and parent identity before binding the child process.
Chat and Laya claim these records instead of invoking their current independent admission paths.
Keep standalone serving behavior unchanged. A claimed allocation cannot grow without a fresh atomic fit check.
On startup failure, the parent stops its children and removes only records whose workers have verifiably exited.
Children watch the parent lifetime pipe and stop when it closes. Recovery checks process start identity, not PID alone.
A live or uncertain worker retains its reservation; a timeout label alone never releases memory.

Budget full weights, admitted KV context, Laya's dtype-aware workspace, and the existing system safety reserve.
For Laya, the assistant batches at most eight questions.
Add an allowlisted `--max-questions 8` launch argument to the module route and use that same cap in admission.
Enforcing eight only in the UI is insufficient: the existing server otherwise reserves for its default of 64.
Resident reservations retain future workspace headroom and subtract only a verified materialized floor.
Voice adds its measured models, preprocessing, decoder, vocoder, and bounded audio queues to the same admission decision.
Never infer fit from installed RAM, theoretical context limits, or compressed download size.

If memory changes, offer a smaller context, the Everyday pair, or text-only mode before starting new work.
Do not silently swap models, exceed reservations, unload another application's model, or rely on swap to meet admission.
Retain an edited or unsent message while the user resolves memory pressure.

## Setup and offline readiness

1. Open MLX Chat. Check the GPU, installed runtime, available memory, disk, and supported pair receipts locally.
2. Recommend a qualified pair and context for the detected resources and selected preference. Show the total download size and optional voice pack.
3. Ask once before downloads. Show per-component progress, verified cache reuse, pause/retry, and an explicit cancel control.
4. Verify every artifact and dependency, convert Laya, then run real chat and decision requests through managed workers.
5. Show **Ready offline** only after a disconnected cold restart succeeds with that locked pair.

Conversion uses the verified source snapshot through `--from-local`, then hashes the converted weights, tokenizer, configs, and manifest.
Only the converted path reaches the Laya worker; generic snapshot completeness is not a substitute for manifest verification.

The disconnected check must deny outbound sockets to the coordinator and every model child while allowing loopback.
Check the full application, not just catalog refresh. Tokenizers and loaders also use local-only artifact paths.
A disconnected machine must never stall while attempting DNS, refresh, telemetry, or a model download.
A missing file names the exact component and keeps already-ready text functionality available.
An incomplete pair does not display **Ready offline** merely because one model can answer.

Support **Prepare another computer**: export the matching runtime wheels, dependencies, pair manifest, and approved model assets.
Import verifies hashes, platform compatibility, and licenses before installation with network access disabled.
Model redistribution terms govern this user-controlled export; project releases do not bundle model weights.
Keep a working installation active until an update verifies and passes its local smoke. Failed updates keep the old lock.
No automatic update check or remote assets are required to reopen a ready installation.

## Voice uses the same conversation

Start with push-to-talk and a click-to-start alternative. Do not require a wake word or continuous listening.
The microphone button explains setup requirements before requesting browser permission.
Dictation inserts an editable transcript into the same composer and requires **Send** by default.
An explicit **Conversation mode** allows automatic send after a user-ended recording and enables spoken replies.
Its active microphone indicator and **End voice** control stay visible. It never reactivates the microphone on page load.

State transitions:

```text
Idle -> Recording -> Transcribing -> Review transcript -> Send
                                                   -> Decision/chat -> Speaking -> Idle
                                   Cancel/Error -> Idle, with typed draft intact
Conversation mode skips Review transcript only after explicit opt-in.
```

The user can type at every stage. **Stop speaking** stops audio without deleting the answer.
**Stop response** cancels generation and queued audio, retaining partial text marked as stopped.
Pressing the microphone control during playback stops playback first, then begins a new explicit recording.
Do not claim acoustic barge-in or echo cancellation from this behavior; keep full-duplex listening out of the default.
`Escape` stops the current capture or response. No single-letter global shortcut steals normal typing.

### Recognition and synthesis

Capture mono PCM locally with browser AudioWorklet support, not an opaque browser recognition API.
Use a bounded queue and a 30-second recording limit initially; warn before the limit and stop visibly at it.
Normalize to the Parakeet input contract of 16 kHz mono float32.
Qualify resampling and the mel frontend on the permitted accelerator path; CPU audio I/O is not CPU tensor inference permission.
Reject invalid rates, non-finite samples, oversized payloads, and unsupported formats before allocating large buffers.
PCM WAV is the initial file interchange format; additional codecs need explicit local decoders and tests.
Silence produces **No speech detected**, never a fabricated transcript or a submitted empty turn.

The existing Parakeet fixture runner must remain a regression oracle.
Implement arbitrary-input preprocessing, valid-length masks, chunk boundaries, recurrent state, and decoding before exposing microphone transcription.
Preserve the fixed fixture's numerical checks; do not simply delete its input hash guard and call it dictation.
The first speech path reuses qualified Parakeet ANE regions and Vulkan work.
A GPU-only recognition option requires its own implementation and receipt for machines without a qualified ANE.
No CPU or remote inference fallback is allowed. Text chat does not depend on ANE availability.

Use [MLX-Audio](https://github.com/Blaizzy/mlx-audio) as the TTS integration candidate.
Start qualification with its [Qwen3-TTS 0.6B CustomVoice family](https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/tts/models/qwen3_tts/README.md).
Use one preset voice, not cloning or reference-voice capture.
An immutable checkpoint revision, license, voice assets, local phonemizer dependencies, and backend support still need verification.
This candidate is not a ready voice pack or a claim of Omarchy compatibility.
The release cannot enable voice until a pinned pack produces intelligible speech with zero CPU tensor dispatches.

Synthesize only completed, visible answer sentences, never hidden reasoning or unvalidated decision drafts.
Bound the TTS queue to two sentences and playback to ten seconds of decoded audio.
When the queue fills, pause synthesis submission; let text continue and read unsynthesized text from the bounded turn buffer.
If playback cannot keep up, offer **Continue reading** after the response instead of accumulating audio indefinitely.
Skip code blocks and raw URLs by default; keep them visible and offer explicit read-aloud selection.
Schedule bounded TTS work between generation chunks; qualify that scheduling on the smallest supported machine.
If real streaming is unavailable, show **Preparing speech** and use truthful sentence-level playback, not a simulated waveform.

Audio events carry turn ID, sentence sequence, sample rate, and encoding.
Cancel clears queued audio and ignores late worker results. The worker remains reserved until it actually stops.
Permission denial, unplugged devices, STT errors, and TTS errors leave typing and completed text usable.
Offer a named fix and explicit retry. Never switch to cloud speech or CPU tensors.
Do not retain microphone audio by default. Keep temporary audio private and delete it on success, cancel, or failure.
Conversation history is local and opt-in; temporary conversations are the default.
Export and delete controls cover transcript, decision records, and any explicitly saved audio.
Logs contain timings and error codes, not conversation text or raw audio.

## Generative UI within Omarchy

The LLM may propose a useful interface instead of only prose, using a versioned declarative component schema.
The application renders that data through its own local, accessible components.
The model chooses content and structure; Omarchy supplies appearance. It never generates executable UI code.

| Task | Generated component | Interaction |
|---|---|---|
| Compare alternatives | Comparison table and decision card | Edit criteria, select options, rerun a bounded Laya decision |
| Explain numeric data | Bar or line chart with a data table | Inspect values and units; see the source of each series |
| Plan or organize | Checklist or timeline | Edit items locally; explicitly submit changes as a new turn |
| Gather missing details | Form with text, choice, and numeric fields | Review values, then submit; no inferred consent |
| Summarize supplied material | Fact cards with source references | Expand the cited material without a remote fetch |

Laya can score supplied options or advise which supported presentation fits a bounded task.
The LLM composes the component data and prose. A deterministic validator owns schema acceptance and action permissions.
Neither model can add components, event handlers, arbitrary URLs, CSS, JavaScript, HTML, or SVG to the renderer.
Charts use application-owned drawing code, never model-generated SVG markup.
Label model estimates as estimates; plotted values need units and a user-input or visible-source reference.
Do not invent measurements or turn Laya entropy confidence into a factual accuracy gauge.

Each component carries a stable ID, schema version, originating turn ID, revision, and plain-text equivalent.
The coordinator derives trusted identity fields; a model cannot override them or reference another conversation.
Validate complete component envelopes before rendering. Partial JSON stays inert while prose can stream.
Initial bounds: 64 KiB per envelope, depth 4, 32 components per turn, 20 form fields, and 1,000 total table/chart data values.
Paginate larger data through application-owned controls; never expand a model-provided unbounded array into the DOM.
Unknown types, invalid values, or exceeded bounds preserve the plain-text answer and show that the interactive view could not be rendered.
Allow one bounded repair attempt; never loop on malformed output or lose the answer.

Allowlisted actions initially include edit, sort, filter, select, expand, and submit form values as a new user turn.
Every server action checks conversation ownership, current revision, typed values, and request identity.
A stale control asks the user to refresh that card; it cannot submit against a newer decision silently.
Submit is an explicit user action. Local edits do not trigger inference on each keystroke.
Prevent duplicate submits and preserve edited values during streaming, reconnect, or theme changes.
Generated controls cannot request secrets, invoke the shell, write arbitrary files, send messages, or grant themselves tools.
Export uses the application's existing explicit export control, not a model-supplied destination.

Keep the interface inside the conversation. A wide comparison can open in a user-invoked side panel without replacing chat.
Provide **Show as text** on each card; all information and actions must have a keyboard and screen-reader path.
Voice reads a concise textual summary and the available choices, not chart coordinates or raw JSON.
Spoken choices become editable user input; destructive or external actions remain outside this design.
No generated form can start the microphone or switch a model.

### Theme inheritance

Default to **Follow Omarchy**, including colors, fonts, focus, spacing, border treatment, and light/dark behavior.
Resolve the active theme and font through the installed Omarchy interfaces, such as `omarchy theme current` and `omarchy font current`.
A read-only theme adapter maps the resolved local theme's supported color values into application-owned CSS variables.
Confirm the installed Omarchy version's theme-file layout during implementation; do not assume a private machine's paths or parse shell code.
Parse theme data as data with an allowlist of colors and local font names, not arbitrary CSS or `url()` values.
Never alter Omarchy package files or the user's selected theme.

Apply theme changes to existing cards as well as new ones, without resetting focus, draft text, form values, or scroll position.
Watch the resolved theme files and their parent for atomic replacement; re-resolve after a theme switch.
Use the fallback palette below only when Omarchy theme data is absent or invalid, with an explicit theme status.
Enforce readable contrast through application-owned foreground/border adjustments, never a model-selected palette.
Test dark, light, high-contrast, and custom themes, including theme changes during streaming and voice playback.
All component assets remain local and work with outbound traffic denied.


## Visual and interaction design

Use a quiet Omarchy desktop companion, not an inference dashboard.
The conversation occupies the center. A compact composer contains text, microphone, and send/stop controls.
Model names, tokens, and backend details belong in a details drawer, not every message.
The distinctive element is an interactive answer built from theme-native cards, tables, charts, and forms inside the conversation.
Do not display fake activity, invented confidence, or a waveform when the microphone is inactive.

Initial fallback tokens, overridden by a locally supplied Omarchy theme when available:

| Token | Value | Role |
|---|---|---|
| Canvas | `#1a1b26` | Window background |
| Panel | `#24283b` | Composer and decision details |
| Text | `#c0caf5` | Main reading text |
| Focus | `#7aa2f7` | Keyboard focus and primary control |
| Ready | `#9ece6a` | Verified local readiness |
| Error | `#f7768e` | Actionable failure indicator |

Use the locally installed sans-serif for conversation and monospace for model details and code.
Use a restrained 24 px title, 16 px body, 14 px controls, and 12 px metadata; scale with user font settings.
Use an 8 px spacing rhythm, a 72-character reading width, and 44 px minimum touch targets.
No font downloads. Supply a light palette with the same semantic tokens and independently verify contrast.
Respect reduced motion, high contrast, zoom, and OS color preference.
Use status text plus icons; color alone never communicates readiness or failure.

Desktop layout:

```text
MLX Chat                  Everyday          Ready offline [Details]
[New chat]                Conversation                         [History]

You                       Which option meets these three constraints?
Assistant                 [Decision: option B]
                          Criteria ...             [Edit] [How decided]
                          Explanation ...                  [Read aloud]

                          [Type a message...                         ]
                          [Compare options] [Microphone]        [Send]
```

At 375 px, collapse history and details into accessible drawers; keep the composer and Stop control visible.
At 768, 1024, and 1440 px, preserve reading width rather than stretching message lines.
Never auto-scroll away from older content the user is reading; show **New response** instead.
Restore focus after dialogs. Stream to an accessible log without announcing every token.
Use native buttons, labelled inputs, keyboard navigation, and visible focus rings.
`Enter` sends, `Shift+Enter` inserts a newline; IME composition must not submit.

First use has one primary action: **Set up local chat**.
A ready empty conversation offers two concrete examples: ordinary chat and comparison of supplied options.
Errors name the component and retain the draft: **Voice model missing. Continue typing or download the voice pack.**
Downloads show real bytes and a measured rate; inference shows elapsed time, not a fabricated completion percentage.

## Acceptance and delivery

These are release requirements, not results from this documentation change.

| Gate | Required evidence |
|---|---|
| Default pairs | Clean supported-machine install, no model-ID entry, exact pinned artifacts, real completion and Laya decision, pair memory peak, graceful no-fit |
| Routing | Versioned held-out English suite with at least 100 labelled cases covering ordinary chat, decisions, ambiguity, negation, injection, and oversized input; at least 99% precision for automatic decision routing; publish coverage and abstention rate, freeze thresholds before evaluation |
| Offline | Cold boot and app restart with outbound traffic denied; chat, decisions, voice, local fonts, and history work; missing/corrupt assets refuse by name |
| Voice | Arbitrary recorded speech beyond the fixture, silence, noise, accents, short/final chunks, device loss, cancellation, and audible output checked by a listener |
| UX | Screenshots at 375/768/1024/1440 px for setup, chat, decision, voice, error, and empty states; keyboard, screen reader, 200% zoom, contrast, and reduced-motion checks |
| Generative UI | Real local model produces valid interactive cards; malformed/injected payloads remain inert; keyboard/text equivalents, source labels, stale/duplicate actions, and mid-turn theme changes pass |
| Adaptive context | Exercise the sizing algorithm across arbitrary byte capacities, model-limit boundaries, dynamic pressure, voice/cache reservations, growth failure, and context-dependent workspace; actual long-context receipts remain chip/runtime-specific |

Add numerical parity, the standing M1 battery, zero CPU tensor dispatch traces, and device recovery receipts to every hardware release.
Pin an independent speech corpus, transcript normalization, WER bound, and intelligibility rubric before voice qualification.
The existing fixture transcript is a numerical oracle, not proof of useful recognition accuracy.
Compare both models alone and together on the same machine; record cold start, warm latency, peak memory, and thermal conditions separately.

Initial UX budgets, to verify rather than advertise: local controls respond within 100 ms and playback stops within 200 ms.
Warm routing adds at most 250 ms; ordinary chat must not wait for it.
For Everyday on the release target, require p95 first visible answer within 2 seconds for a 128-token prompt and 256-token response.
For a clean five-second voice input, require p95 editable transcript within 2 seconds after recording stops.
Require p95 first spoken sentence within 1.5 seconds after its text becomes available.
Measure 30 warm turns after two warmups, with the pair and voice profile present; publish p50/p95 and cold figures separately.
If the voice budgets fail, keep voice unqualified and fix the path; do not label a slow batch demo conversational.

### Implementation order and ownership

1. Extend `serve/mlx_omarchy_serve/catalog.py`, `catalog.json`, `budget.py`, and `__main__.py` for pair preparation and atomic ownership.
   Extend `tests/test_serve_catalog.py`, `test_serve_transaction.py`, and `test_serve_cli.py` for adaptive context, pair failure, and offline recovery.
   Resolve missing receipts and the source-snapshot-to-Laya-conversion path before enabling either pair.
2. Add the shared conversation coordinator and local UI under `serve/mlx_omarchy_assistant/`.
   Connect existing endpoints, schema-validated generative UI, the read-only Omarchy theme adapter, request identity, cancellation, and local history.
   Replace direct weight loading in `demo/chat.py`; update `install.sh` to launch the same coordinator.
   Test late events, reconnect, invalid decision envelopes, material truncation, and LLM/Laya disagreement.
3. Extend the existing Parakeet runtime for arbitrary audio while retaining its fixture qualification path.
   Add the pinned TTS adapter, bounded audio queues, and shared memory admission.
   Test recording, transcript edits, audible playback, interruption, audio device errors, and text-only recovery.
4. Qualify Everyday first, Quality separately, and voice on each supported chip/runtime combination.
   Run clean connected preparation followed by network-denied restart, the usability checks, and hardware release gates.
   Promote only passing pair/profile records; update the README and compatibility rows from the resulting public receipts.

Text delivery and voice development can proceed in parallel after the coordinator contract exists.
Do not wait for general ANE support to ship the Vulkan text pair.
Do not claim the complete assistant until arbitrary speech, local synthesis, and both model paths work in the same installed application.

### Design review decisions

1. Keep independent model workers, but present one application. A model-switcher UI would leave orchestration to the user.
2. Do not route every message through Laya. Its short context and bounded outputs do not fit general conversation.
3. Keep speech permissions deterministic. A model must never decide to start the microphone or transmit input.
4. Define real default pairs now; enable their recommendations only after paired qualification, not standalone model results.
5. Keep the fixture runner intact. General speech needs a real input pipeline and accuracy evidence, not a removed assertion.
