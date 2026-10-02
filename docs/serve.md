# Local generation and HTTP serving on Omarchy

The installer includes the serving CLI, Laya decision server, and Bonsai server packages.
The v0.7.17 installer contains all three, plus the MLX Chat assistant. Old installations need an explicit update.
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
(T6021, 96 GB). The bench, card, routing, and speech runs used the v0.7.6
release wheel `0.32.3.dev202609291615+06711ad` (provenance `verified: match`);
the paired-memory, card-timing, and idle-GPU runs used the harness at main
`a1251aaa`, which carries the card-reply fix.

| Gate | Everyday (9B) | Compact (4B) | Quality (27B) |
|---|---|---|---|
| Cards, HELD-OUT v4 (frozen; pass needs 15/18 valid, 0 spurious) | 16/18, 0 spurious — pass | 18/18, 0 spurious — pass | 18/18, 0 spurious — pass |
| First-text p95, chosen config, stock kernel | 1.05 s — pass | 0.66 s — pass | not part of this gate |
| First text on the real card prompt, engine level | 0.78 s — within the 2.0 s design budget | 0.44 s — within | 2.45 s — over budget |
| Paired memory peak over baseline (whole system; nine valid runs) | 9.5 GiB idle (one run 13.4 with page cache) | 5.2–6.8 GiB | 17.3–18.4 GiB |
| Backend peak (chat worker allocator) | 5.5 GiB | 2.9 GiB | 15.9 GiB |
| Tier fit | 16 GB and 96 GB (one 16 GB run borderline) | 16 GB and 96 GB | 96 GB only |
| Pair qualified | No | No | No |

Tier fit is host-RAM arithmetic on the measured peaks; no 16 GB machine was measured ([pair gates receipt](../receipts/2026-09-30-pair-gates/README.md), v3 tier analysis). The card turns in all nine v3 runs completed with visible text and a valid component.

Cards: the product ships `card_format` unset everywhere (markdown promotion, no
schema on ordinary chat). The fenced-json schema was measured and rejected by
the pre-registered rule: the 4B wrote 18/18 cards with it but first-text p95 was
2.40 s, and the 9B had one spurious card (17/18) — see the [card promotion
receipt](../receipts/2026-09-30-card-promotion/README.md). The bench card
protocol leaves the same two prompts (the Apollo timeline and the decision)
without a card on both the 9B and the 4B ([chat-model
bench](../receipts/2026-09-30-chat-model-bench/README.md)).

**Card timing on a quiet boot** (idle render node; [pair gates
receipt](../receipts/2026-09-30-pair-gates/README.md)): the release-relevant
number is the wait to the first *visible component*. The 9B streams prose from
5.75 s but its chart payload stays invisible until it validates at 115.9 s;
the 27B shows prose at 20.5 s and its chart at 53.8 s; the 4B emits a fenced
block that fails component validation, so its user sees nothing until the
coordinator's `invalid_component` notice plus raw text lands at about 91 s.
The coordinator fix (`003823df5`, in `a1251aaa`) is why that fallback exists:
an invalid fence is now shown as raw text and an empty reply is guarded
against — the earlier silent empty reply on the 4B is fixed. What remains on
the 4B is the model's own invalid fence, a model- or card-format change, not
a coordinator one.

**Quality performance — the design budget is not met, on an idle GPU.** The
quiet-window run (no other render-node holder, `fuser` empty before and after;
[pair gates receipt](../receipts/2026-09-30-pair-gates/README.md)): prefill
42 tok/s at a 600-token prompt and 76 tok/s at 2,100 tokens (about 25 % under
the 97–103 tok/s the same model does alone on the GPU in ModelBench, the cost
of the resident assistant and Laya workers); TTFT 6.31 s for the real
300-token chat prompt — about 3× the 2 s design budget. The 2026-10-02
[TTFT phase receipt](../receipts/2026-10-02-quality-ttft/README.md) instrumented
every phase and corrected the earlier attribution: the coordinator's whole
per-turn path (pair start, history, admission, speech probe, request dispatch)
is ~3 ms of the 6.4 s; the cost is the engine's own prompt path — prefill of
the full 325-token prompt at the resident-pair rate plus the first decode
step — not coordinator code; the same receipt fixed context admission,
which had counted every prompt as 2 tokens (`cd216fdc0`: `count()` took the
length of transformers 5's `BatchEncoding`); visible
decode about 5.4 tok/s (about 4× the superseded shared-GPU estimate of
1.4 tok/s, which that run's contention produced); card-turn first text
2.6–14.4 s. A shared-GPU run 1 (73–81 tok/s prefill, ~1.4 tok/s decode) is
kept in the receipt for comparison and is superseded.

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
denied, device loss, 30 s limit, cancel) pass. The voice-screen reader
receipt below now drives every state under real Orca, recording the
verbatim utterances for keyboard start, Stop, Escape, error, transcribing,
transcript insertion, TTS start/finish/truncate/resume, preview, and the
voice-state machine; seven accessibility defects were found and fixed
([voice screen-reader receipt](../receipts/2026-10-02-voice-screen-reader/README.md)).

**Voice output — Kokoro-82M is the default engine (owner decision, 2026-10-02); Qwen3-TTS stays below real time as the second engine.** The
Qwen3-TTS pack
measures median RTF 0.22–0.23 (audio s over wall s) against the 1.2
threshold, with the named floor of about 87 ms of GPU compute per talker step
([speech output speed
receipt](../receipts/2026-09-30-speech-output-speed/README.md) and
[NAMED-FLOOR](../receipts/2026-09-30-speech-output-speed/NAMED-FLOOR.md)); the
Attn128 fused decode attention narrows its dispatch gap and does not close it
([Attn128](../receipts/2026-09-30-attn128/README.md)). Kokoro-82M is measured
real time after the conv k-tap GEMM decomposition: direct RTF about 1.21
median (range 1.108–1.422; the same wheel without the fix measures 0.66), and
through the real serve worker with the Kokoro engine verified, RTF about 1.51
median over 16 sentences (warm sentences above 1.4), first audio 2.29–6.46 s,
Whisper WER 4.44 % overall with 13 of 16 sentences at 0.0 % (the worst is the
paragraph capped at 30 s mid-sentence), peak memory 1427 MB on that
paragraph, and 0 CPU dispatches in the serve worker ([OpCost receipt,
ADDENDUM 2](../receipts/2026-10-01-opcost-microbench/README.md)). The
receipt retracts the earlier serve-path streaming numbers (RTF 0.20–0.27) as
Qwen3-TTS runs mislabeled as Kokoro. A wheel from before this work refuses
Kokoro serve-path synthesis at the trig accuracy gate (Sin magnitude above
the limit); v0.7.17 is the first release that runs it end to end — its cut
carries the in-shader trig reduction and the conv decomposition, and its
release-battery serve-path smoke measured RTF 0.981 through the real
Synthesis class (smoke scope, not real-time qualification)
([v0.7.17 receipt](../receipts/2026-10-02-v0717-release.md)). The owner
decided on 2026-10-02 that Kokoro-82M is the default speech-output engine
with voice af_heart and no listening step, so the unset voice choice now
resolves to Kokoro everywhere (first-run setup downloads the 329 MB Kokoro
pack, not the 1.7 GB Qwen3-TTS pack); an explicitly saved voice choice
still wins, and Qwen3-TTS stays as the selectable second engine. Voice
output stays **unqualified** in status text: the frozen gates it still
misses are the real-time RTF threshold (serve path measures about 1.51
against the 1.2 gate), the 1.5 s first-audio design target (2.29–6.46 s;
needs vocoder output streaming), and a durable qualification receipt bound
to the Kokoro pack revision and wheel hash (`record_qualification` has not
run for the default engine). A pre-0.7.17 wheel is refused at Kokoro load
with an upgrade message instead of a backend crash
([Kokoro receipt](../receipts/2026-09-30-speech-output-kokoro/README.md)).
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
receipt](../receipts/2026-09-30-m1-battery/README.md)). The [mlx pin
bump](../receipts/2026-10-01-mlx-pin-bump/README.md) landed (`2e6a39d6a`, pin
`9c3d35571a`) with token digests bit-identical on both chips and shipped in
v0.7.10; the battery failure it recorded in `omarchy_indexing_ops_tests` was
two test bugs rather than a backend defect and is fixed on main (`e00b37116`,
[Attn128 corrections](../receipts/2026-09-30-attn128/README.md)); the 16-bit
selection doctest has since passed on the M2 and on T6001, and the T8103
(G13G) run of that doctest is the last box in the matrix. Mesa: the earlier
G14 prefill collapse was unblocked by opening the coopmat gate for G14X, and
the pin candidate `e7631595df6` now passes on G14X (+4.7 % prefill-512,
+9.3 % prefill-2048 over the deployed driver, dispatch count exactly equal)
and on G13X (parity on every leg), all digests bit-exact; the G13G arm of
that candidate is pending host availability, and the pin lands after it
([Mesa v2 parity](../receipts/2026-10-01-mesa-v2-parity/README.md)).

**Runtime and install gates.** The packaged DKMS ANE module passed the worker's
ABI-1 acceptance on T6001 (bit-exact h13 add-mul bundle, per-user fallback,
negative controls); Honeykrisp ICD selection now refuses a missing override
JSON, and release builds raise instead of silently falling back to the CPU
device ([runtime gate](../receipts/2026-09-30-runtime-gate/README.md)). The
offline `--system` stage ran green inside a network namespace on T6001 with 36
vendored aarch64 wheels; two bugs were found on hardware and fixed test-first
([system install receipt](../receipts/2026-09-30-system-install-hw/README.md)).

**Screen reader — pass (extended to the voice features).** Two real Orca
runs drove the static UI through every state in an isolated container.
The earlier run (Orca 3.38.4) covered setup, chat stream/escape,
decision, card, compare, drawers, voice unavailable, error, offline chip
([receipt](../receipts/2026-09-30-screen-reader/README.md)); the second run
(Orca 43.1, which ships the full web script) drove the current UI at
origin/main through the voice path: ready/missing/unqualified/usable
states, keyboard dictation start/stop/Escape, transcript insertion and
focus, transcribing 500, no-speech client gate, TTS playback (read aloud,
Stop speaking, play-to-end, truncation + Continue reading), and the voice
preview in the Details drawer ([voice screen-reader
receipt](../receipts/2026-10-02-voice-screen-reader/README.md)). Seven
a11y defects were found and fixed in this pass: the mic button was
pointer-only (no Space/Enter start), Escape cancelled only from the
textarea (no cancel from the focused mic), the recorder tick announced a
10 Hz polite-region flood, TTS playback start had no announcement, the
recognition state "usable" was unlabeled, history assistant bubbles
had no Read aloud control (and no `lastAssistantBubble` for
truncation), and `SpeakQueue` announced the truncation event twice.

Defects these runs found and fixed in source: chat requests carry a repetition
penalty of 1.1 because greedy decoding looped on the 2B until the token cap;
the STT worker made 152 CPU-stream calls through a float64 filterbank built at
models-package import until the worker registered that package without running
its `__init__`; the recorder requested browser noise suppression, which doubled
the captured level (median 2.14×) until it asked for unprocessed audio; the
the voice screen-reader pass added a click handler for keyboard dictation start, an
Escape keydown handler on the focused mic button, replaced a 10 Hz polite
live-region flood from the recorder tick with a single "Recording started."
announcement, announced the false-to-true edge of TTS playback, named the
"usable" recognition state in the status region and the mic title, rendered
the Read aloud control on history assistant bubbles, and deduplicated the
playback-cap truncation announcement. The card-reply fix (`003823df5`, in `a1251aaa`)
that shows an invalid fence as raw text and guards the empty reply. Still
open: the 4B's fenced card output keeps failing component validation (model
behavior — the user gets the notice and raw text, not a card), and the
explicit-CPU-stream finding above.

### Open items for the owner

| # | Item | Why it blocks |
|---|---|---|
| 1 | Routing latency decision: does the 250 ms gate measure the Laya head call (p95 347 ms, fails) or the shipped head-free path (p95 43 ms, passes)? | Automatic routing stays off until decided. |
| 2 | Quality design budget: first text is the engine prompt path on the resident pair (~6.4 s for a 300-token prompt vs the 2 s budget); the lever is engine-side prefill and first-decode-step work, and stable-prefix cache reuse is a memory-admission gate decision. | The Quality pair cannot qualify against a budget it misses. |
| 3 | Voice output first audio: 2.3–6.6 s against the 1.5 s design target (needs vocoder output streaming). Kokoro default decided by the owner 2026-10-02: Kokoro-82M default engine, af_heart default voice, no listening step; Qwen3-TTS stays as the selectable second engine. | Voice output stays unqualified: the RTF gate (serve path ~1.51 vs 1.2), the first-audio target, and a `record_qualification` receipt for the default engine are all still open. |
| 4 | Pair-level qualification: no pair has passed the full gate set; chart cards wait 53.8 s (27B) / 115.9 s (9B) to a visible component, and the 4B's fence never validates. | Nothing is qualified; `recommended` stays false everywhere. |
| 5 | G13G (jwm1) Mesa arm of pin candidate `e7631595df6`, plus the T8103 16-bit selection doctest and standing battery. | The Mesa pin and the Attn128 chip matrix each wait on that host. |

No pair is qualified. All catalog entries keep `recommended: false`.

The one-line installer ships MLX Chat from the promoted release tag. The
current release is v0.7.17, published and promoted Latest on 2026-10-02 from
cut `a33a4e395` ([receipt](../receipts/2026-10-02-v0717-release.md)). It is
the first release that ships the installer serve file-list fix, the Parakeet
transcribe self-healing deps and staged launcher, the build-time runtime-asset
checker with parsed seal errors, the per-submission cap and queue priority
(issue #19), `MLX_OMARCHY_UCLAMP_MIN` placement, the in-shader Cody-Waite
trig reduction (`6cbf55d8f`) and the conv k-tap GEMM decomposition
(`8db7abb73`) that make Kokoro run end to end, the voice accessibility
fixes, and the bare `--submit` collector default (the server-side tolerance
for older collectors was already live). Two earlier cuts were burned
unpublished by the draft gate battery, as designed: v0.7.15 missed
`perf_placement.py` in the release-lane serve file list (serve startup died
on every user install) and v0.7.16 carried a function-scope `import signal`
that raised `UnboundLocalError` at every assistant startup, caught by gate 2.
No asset of either was published ([receipt](../receipts/2026-10-02-v0717-release.md)).
Releases are cut draft-first: assets are verified as a draft, the
installed-from-release gate battery runs, and a draft that fails is deleted
unpublished. The repository is now `joshuaswarren/omarchy-mlx`; the Python
package names are unchanged. v0.7.13 is a privacy patch release: the
community-data collector redacts hostname-derived aliases from dmesg/journal
unit paths before submission and the service scans submissions for the same
aliases (`scripts/collect_deep.py`, `services/community-data/src/pii.ts`);
the worker redeploy is a separate packaging step.

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
| qwen3.5-9b-mlx-4bit | Everyday chat, MLX-LM | Qualified / qualified / qualified | Pair qualification |
| qwen3-4b-instruct-2507-4bit | Compact chat, MLX-LM | Qualified / qualified / qualified | Pair qualification |
| `qwen3.8-27b-4bit` | Larger chat, MLX-LM | Qualified / qualified / qualified | Resolve recorded numerical-equivalence limits and qualify the pair |
| `laya-mlx` | Typed decisions, dedicated module | Qualified / qualified / untested | Converted artifact, managed launch, paired qualification |

The IDs above reference pinned entries, not floating upstream model names.
The catalog also contains Bonsai and other Qwen variants; they are not automatic assistant defaults.
Some qualification references are missing from this checkout, and historical notes disagree with later catalog flags.
Resolve those references and preserve their original scope before promoting a recommendation.
This documentation update does not certify a new hardware result.

The pinned Qwen3.8-27B text-generation example remains in the
[README](../README.md#quick-start); its original install receipt is no longer
in this checkout.
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
curl -fsSL https://raw.githubusercontent.com/joshuaswarren/omarchy-mlx/main/install.sh | bash
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

The raw runs of that comparison predate this checkout's receipt set. These
measurements establish
nothing about current-generation models: they predate Qwen3.8
qualification and must not be read as a "fastest server for current
models" claim. Linux mlx-serve's GGUF and ANE engines were not
operational in that comparison. Its `/v1/models` ID may be an internal
hash rather than a Hugging Face repository name.

Other historical loader experiments, including the specialized Bonsai
runtime, are not drop-in HTTP-serving recommendations.
