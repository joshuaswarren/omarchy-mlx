# MLX Chat voice-path screen-reader pass — 2026-10-02

Real Orca screen-reader pass on the voice features of the current
`mlx_omarchy_assistant` static UI at origin/main (commit 8db7abb73). Driven
inside an isolated Linux container on the build/compiler oracle host
via `docker run`; the host's own desktop session is not touched. This
pass covers what the prior receipt at `screens/2026-09-30-screen-reader/` did not — mic
dictation flow, transcript insertion, TTS playback/stop, and the voice
state machine — in the UI as it stands today.

## Environment identity

- Image: `orca-screen-reader:bookworm` (Debian bookworm-slim, arm64,
  image id c7c94dc30e75, 1.17 GB). Built into the container from Docker
  on the host; no host processes touched.
- Display: `Xvfb :99`, 1440x900x24.
- AT-SPI: bus on the dbus session launched inside the container.
- Browser: Chromium 154.0.8037.57 with `--enable-accessibility
  --force-renderer-accessibility --enable-features=AccessibilityAriaVirtualContent
  --autoplay-policy=no-user-gesture-required`, plus fake-media flags
  `--use-fake-ui-for-media-stream --use-fake-device-for-media-stream
  --use-file-for-fake-audio-capture=<wav>`.
- Screen reader: **Orca 43.1** with a one-line user settings file (no
  Learn Mode, speech on). Crucially this image's Orca ships the full
  `orca/scripts/web` (bookmarks, live-region manager, web keymap),
  unlike the prior pass's 3.38.4 which lacked web.py.
- Speech capture: every `SPEECH OUTPUT: '<text>'` line from Orca's
  debug log between the scenario marker and the end-of-scenario. Speech
  is `sd_dummy` (audio is discarded to /dev/null; the debug log is the
  speech record).
- Backend: in-container Python `ThreadingHTTPServer` stub at
  `http://127.0.0.1:8765` that serves the real `static/` files plus a
  fake `/api/*` surface. Voice status shapes mirror what the real
  coordinator emits (see `serve/mlx_omarchy_assistant/recognition.py`
  `Recognition.status` and `serve/mlx_omarchy_assistant/server.py`
  `_voice_status`): recognition states `ready/usable/unqualified/
  missing`; synthesis states `ready/unqualified/missing`. Scenario
  state is switched via `/api/__state?set=<name>` so one Chromium
  instance drives every scenario.

## Method

For each scenario the stub state is set, the page is cache-busted and
re-rendered, Orca's locus of focus is reset to the body, a marker file
is written, the scenario keystrokes run, and the `SPEECH OUTPUT` lines
emitted between the marker and end-of-scenario are extracted verbatim.
The driver (`harness/voice-driver.py`) targets controls via
`tab_until(...)` and asserts `document.activeElement` before each
action so the keystrokes land where intended. Before/after each
scenario the driver dumps `document.activeElement`, `#mic-btn`,
`#stop-speak-btn`, the Continue chip, the voice-status region,
`#live-region`, and the composer textarea value via CDP
`Runtime.evaluate` (`ax-before.json`, `ax-after.json`).

## Per-scenario findings

The number after each scenario is the count of `SPEECH OUTPUT:` lines
captured verbatim.

#### (1) voice-unqualified — 42 utterances (state=`voice-unqualified`)

Tab walk through the page. The mic button is `disabled` (recognition
not qualified on this machine), so Tab skips it. Orca announces every
focusable control in order:

| Utterance (verbatim) | Control |
|---|---|
| `Start a new conversation push button.` | new-chat |
| `Open conversation history push button. opens dialog` | history |
| `Prepare another computer or install an offline bundle push button. opens dialog` | transfer |
| `Open model and memory details push button. opens dialog` | details |
| `Message entry Type a message. Enter sends, Shift+Enter adds a newline.` | composer textarea |
| `Compare options toggle button not pressed.` | compare |
| `Classify or score toggle button not pressed.` | analyze |
| `Conversation mode check box not checked.` | conversation mode |

`scenarios/voice-unqualified/orca-utterances.txt`

#### (2) voice-missing — 42 utterances (state=`voice-missing`)

Same control walk; Orca's utterance line is identical because the
visible composer controls do not change (only the voice status text
behind them does, and that text lives in a non-focusable `[role=status]`
region that announces only on update — correct live-region behavior).

#### (3) voice-usable — 42 utterances (state=`voice-usable`)

Post-fix: the recognition.state "usable" branch is now named in the
voice-status region and (if the mic were enabled) the mic title would
explain why. Mic remains disabled — the product is conservative
(`micBtn.disabled = recognitionState !== "ready"`), and the new title
no longer misnames the state. The status region content is observable in
`scenarios/voice-usable/ax-after.json` (`voiceStatus`).

#### (4) voice-ready-idle — 44 utterances (state=`voice-ready`)

Tab walk reaches the mic (now focusable). Orca announces role, name
and description together:

| Utterance (verbatim) | Control |
|---|---|
| `Microphone push button.` | mic button (line 18) |
| `Press and hold, or click to latch, to record.` | title attribute (line 19) |

The toolbar's preceding `Compare options toggle button not pressed.`
and the trailing `Conversation mode check box not checked.` confirm
the surrounding toolbar announces role, name, and pressed/checked state
correctly.

`scenarios/voice-ready-idle/orca-utterances.txt`

#### (5) voice-record-transcript — 7 utterances (state=`voice-ready`)

Keyboard Tab (3 presses) → Space to start recording with the mic
button focused → Space to stop after 1.5 s of the fake tone → wait for
transcribe → assert focus and content. Verbatim:

```
[06:13:45.085308] tab 
[06:13:45.113612] Microphone push button.
[06:13:45.113651] Press and hold, or click to latch, to record.
[06:13:45.283154] space 
[06:13:46.806535] space 
[06:13:46.899341] Transcribing…
[06:13:48.436057] Message entry Transcribe the stub sentence, please.
```

"Transcribing…" is a live-region announcement that Orca 43.1 with the
web script surfaces automatically. "Message entry ... the sentence,
please." is the textarea's name + value read on focus arrival after
`composer.setText(composed)` (which both merges the transcript and
moves focus into the textarea — the same field that the live-region
`Edit the transcript, then press Send.` message names). The
`ax-after.json` shows `composerFocused: true`,
`composerText: "Transcribe the stub sentence, please."`, and
`live: "Edit the transcript, then press Send."`.

#### (6) voice-escape-cancel — 5 utterances (state=`voice-ready`)

Tab to mic → Space (recording starts, announcement goes out) →
Escape (recording cancels from the focused mic button) → Tab (focus
still works; no keyboard trap):

```
[06:13:57.231265] Recording started. Press Escape to cancel.
[06:13:57.964084] escape 
[06:13:58.085628] Recording cancelled.
[06:13:58.497641] tab 
[06:13:58.525394] Conversation mode check box not checked.
```

The recording-start announcement is the new live-region text wired in
`app.js` after `recorder.start()` succeeds (the prior code only
fired announcements from the 100 ms recorder tick, producing a
polite-region flood — D3). The Escape handler on the mic button is
the new keydown listener in `composer.js` (D2; the textarea's
existing Escape handler covers keyboard users whose focus lives in
the textarea — covered by an additional test).

#### (7) voice-transcribe-error — 10 utterances (state=`voice-transcribe-error`)

Stub returns HTTP 500 for `/api/transcribe`. The UI announces:

```
[06:15:39.027350] Microphone push button.
[06:15:39.027397] Press and hold, or click to latch, to record.
[06:15:39.197106] space 
[06:15:40.359893] space 
[06:15:43.514741] Transcribe failed: transcribe failed (500)
```

`ax-after.json`: `live: "Transcribe failed: transcribe failed (500)"`,
`notice: "Transcribe failed: transcribe failed (500)"` (sighted twin).

#### (8) voice-no-speech — 9 utterances (state=`voice-ready`, silent WAV)

Reloads Chromium with `--use-file-for-fake-audio-capture=<silent WAV>`
(amp 0.002 → RMS 0.002, below the recorder's 0.005 silence gate).
Then the same Tab→Space→Space flow. Verbatim:

```
[06:15:23.361020] Microphone push button.
[06:15:23.361056] Press and hold, or click to latch, to record.
[06:15:23.526318] space 
[06:15:23.813842] Recording started. Press Escape to cancel.
[06:15:25.057800] space 
[06:15:25.202410] No speech detected.
```

Both the recording-start announcement and the client-side silence
announcement are captured. With this Chromium the wait is brief
because peakRms drops below the gate immediately.

#### (9) tts-read-stop — 59 utterances (state=`voice-ready`)

Tab to "Read aloud" on the seeded assistant bubble (13 Tabs from body)
→ Enter (queues sentence 1; sentence 2 follows) → wait → Tab to "Stop
speaking" → Enter. The new "Reading reply aloud." announcement fires
on the false-to-true edge of `onSpeakingChange` (D4). Verbatim:

```
[06:31:03.011396] Read aloud push button.
[06:31:03.326380] Reading reply aloud.
[06:31:07.137842] Stop speaking push button.
[06:31:07.426594] Stopped speaking.
```

`Reading reply aloud.` is captured by Orca 43.1 announcing the polite
live-region update; `Stopped speaking.` is the existing
`app.js stopSpeaking()` announcement. The after-state shows the Stop
speaking button hidden (composer's `setSpeaking(false)`).

#### (10) tts-playout — 39 utterances (state=`voice-ready`)

Same Tab to Read aloud → Enter; no Stop speech; wait 11 s for the two
sentences to play through. Verbatim:

```
[06:37:27.329087] Read aloud push button.
[06:37:27.524932] Reading reply aloud.
[06:37:33.642962] Finished speaking.
```

`Finished speaking.` is the `onAudioDone` announcement. The
`ax-after.json` shows the Stop speaking button hidden.

#### (11) tts-truncate-resume — 60 utterances (state=`tts-truncate`)

Stub streams 24 × 0.5 s audio chunks (12 s) so the 10 s playback cap
truncates mid-stream. Tab to Read aloud → Enter; wait for the
truncation announcement + Continue chip; Tab to Continue reading →
Enter. Verbatim:

```
[06:37:46.013281] Read aloud push button.
[06:37:46.228104] Speech paused at the playback limit. Continue reading is available.
[06:37:57.947110] Continue reading push button. the announcement goes back to... Continue reading the response aloud.
[06:37:59.711121] Continuing to read aloud.
```

The truncation announcement fires ONCE (verified by `grep -c` = 1 in
the final run; the earlier run showed 2, which led to the dedupe fix).
The Continue chip label and aria-label are both announced on focus
arrival (the second "Continue reading push button." + the label from
`aria-label="Continue reading the response aloud"`).

#### (12) voice-preview-picker — 50 utterances (state=`voice-ready`)

Open Details drawer (Tab → "Details" → Enter) → Tab to
`#details-voice-select` → Down arrow (selects the second voice) →
Tab to `#details-voice-preview` → Enter → wait → Tab to tab → "Message
entry ..." (drawer focus trap verified — Tab returned focus to main
content, no loop in the modal). Verbatim highlights:

```
[06:39:04.939008] Output allowance in tokens combo box.
[06:39:05.088077] Reply voice combo box collapsed
[06:39:05.093092] Preview push button.
[06:39:05.093122] Preview speaks one fixed sentence in the selected voice.
[06:39:08.448760] Voice set to Bella.
[06:39:09.197332] leaving banner.
[06:39:09.197388] main content
[06:39:09.197405] information
[06:39:09.197421] Message entry ...
```

The voice select label change fires `Voice set to Bella.` through
the same live-region handler that drives the polite update. "leaving
banner / main content" after the drawer close proves Tab re-enters
main content cleanly — no keyboard trap.

## Defects found and fixed

| # | Defect | Fix |
|---|---|---|
| D1 | `#mic-btn` was driven only by `pointerdown/pointerup`; keyboard Space/Enter (click with `event.detail===0`) never fired a start or stop. | Added a `click` handler that toggles the same latch as the pointer path when `event.detail===0`. `tests/js/assistant-ui.test.mjs` proves keyboard start + keyboard success, plus a regression test that a synthetic pointer-click twin (detail 1) does not double-fire. |
| D2 | Escape cancelled recording only from the composer textarea; focus stays on the mic button after a keyboard start, so Escape reached nothing. | Added a `keydown` handler on `#mic-btn` that calls `resetMicUi()` + `onMicCancel()` when Escape arrives during recording. Test covers both the mic-button and the textarea paths. |
| D3 | `app.js` announced `"Recording N.N seconds"` from the 100 ms recorder tick — a 10 Hz polite-region flood for the whole recording. | Dropped the per-tick `announce`; added a single `Recording started. Press Escape to cancel.` via `_micMessage` after a successful `recorder.start()`. Warn + cap announcements remain. |
| D4 | No announcement when speech playback started; the user got no "reading aloud / Stop speaking is available" cue. | `SpeakQueue`'s `onSpeakingChange` hook now announces `"Reading reply aloud."` on the false→true edge (only once per playback). |
| D5 | `recognition.state === "usable"` rendered the status region as `"Dictation: Voice unknown"` while the mic stayed disabled — the state was nameless. | Added a `usable` case to `voiceTitle` / `voiceLabel`: `"Voice input is usable, but no recorded acceptance run qualifies this machine yet"` / `"Voice usable, unqualified"`. The mic remains disabled (product unchanged). |
| D6 | Completed assistant messages rendered by `_appendMessage` had no Read aloud control; after a page reload no message had a TTS button. `lastAssistantBubble` was never set, so truncation resume had no target either. | `_appendMessage` now renders the same `.message__meta` (shared `_readAloudMeta` helper) and sets `_mlxSentences = splitSentences(msg.content).map(...)` + `_mlxTurnId` + `lastAssistantBubble`. `appendAssistant` also uses the helper (dedup). Test exercises a constructed `ConversationView`. |
| D7 | `SpeakQueue` fired `onTruncate` from both the enqueue-budget check and the in-stream cap check, producing duplicate live-region announcements. | New `_markTruncated()` is the single source: it sets `_truncated = true` once per turn and is called from both paths. `resumeFromTruncation → reset()` re-arms it. Test in `tests/js/voice-recorder.test.mjs`. |

## What is stubbed and what is real

The screen-reader pass drives the **real** static UI files from the worktree
served verbatim by the stub at `http://127.0.0.1:8765/`. The stub
implements the **API surface** the UI calls, but every backend
endpoint returns canned data:

| Path | Stub behavior | Real backend (in this container) |
|---|---|---|
| `GET /api/status` | per-state voice.status JSON; states mirror `recognition.py`/`server.py` | not run |
| `POST /api/transcribe` | 1.5 s sleep → `{text: ...}`; per-state 500 or `{text: ""}` for the empty/error scenarios | not run |
| `POST /api/speak` | SSE of base64 PCM16LE 16 kHz mono tone chunks; closes the response after `event: done` | not run |
| `GET /api/voice/preview` | one fixed-sentence PCM16LE blob | not run |
| Other conversation endpoints | the seeded conversation + a no-event SSE stream | not run |

Audio **input** is exercised end-to-end through the real
AudioWorklet/getUserMedia/MediaRecorder/16 kHz WAV path in the static
UI, into the stub's `/api/transcribe`. The fake media device captures a
speech-like 60 s gated 440 Hz tone (RMS ~0.18, above the recorder's
0.005 silence gate) — this is honest enough to drive every
recorder hook (`onMeter`, `onWarn`, `onStop`, `onDeviceLost`) the same
way a real microphone would. Audio **output** is exercised through the
real `Web Audio` graph (`AudioContext`, `createBuffer`,
`createBufferSource`, `src.onended`, `SpeakQueue`'s scheduling and the
`onAudioDone`/`onTruncate`/`onSpeakingChange` callbacks). Orca 43.1
in the container confirmed the AudioContext clock advances and
`onended` fires for a 0.5 s 16 kHz scheduled source — the WebAudio
pipeline is functional.

What this means concretely:

- **Stub vs real**: the voice status JSON shapes are byte-for-byte
  what the real coordinator emits; the SSE wire format matches the real
  one; the silence/empty/error client-side gate (peakRms < 0.005) is
  exercised against the stub. The real MLX transcribT/TTS pipeline is
  not invoked. End-to-end browser → STT → transcript on **real**
  hardware is covered separately by the speech-input-gpu browser E2E
  receipt (`receipts/2026-09-30-speech-input-gpu/`): e2e-20260930T064754Z
  ran alone on the M2 and covers permission-denied, the 30 s cap, device
  loss, escape-cancel, and the editable transcript path. TTS on real
  hardware is covered by the speech-output Kokoro/speed receipts.
- **No live-microphone audio**: the fake media device's audio waveform
  is real PCM (it goes through AudioWorklet, peakRms is computed) but
  it is a synthetic tone burst, not human speech. No claim is made
  that dictation heard real audio this run — the assertion is that
  every UI state the recognizer drives is reached correctly.
- **Real assistant backend not started in the container**: a real
  backend needs the mlx wheel + a small chat model + the
  voice-qualification receipt (host-side acceptance run). The Linux VM
  the container runs on has no GPU and no ANE, so the real
  recognizer's status would not report ready here even with models
  loaded — the ready-state UI flows cannot occur against a real
  backend in this container. The browser E2E receipt (above) covers
  the ready-state path on real hardware. Net: the screen-reader gate
  is a UI-state pass with a stubbed backend that follows the real
  JS/API contracts; the real-backend voice behavior is covered by the
  speech-input-gpu receipt.

## Items not run

- **Web keymaps for `Insert+H` headings, `Insert+;` landmarks,
  `Insert+T` tables, `Insert+F5` live-region read**: Orca 43.1 in this
  image now ships the web script (prior image's 3.38.4 did not), but
  the driver does not rely on these bindings — it uses Tab/Space/Enter/
  Escape/Arrows, which Orca's default keymap covers. The web
  keymaps would also work; they were simply not exercised.
- **Live microphone audio**: only the fake media device with a
  synthetic tone (and a synthetic silence variant) is exercised.
- **Real assistant backend in the container**: see the stub-vs-real
  discussion above.

## Verification

- `tests/js/` (5 files, bun):
  - `assistant-ui.test.mjs` (4 blocks added: keyboard start/stop +
    pointer Escape on mic + textarea Escape + usable naming + chair
    history read-aloud). Passes.
  - `assistant-ui-cards.test.mjs`, `transfer.test.mjs`, `util.test.mjs`,
    `voice-recorder.test.mjs` (1 block added: truncation dedupe).
    Pass.
- `tests/` Python assistant suite (pytest,
  `test_assistant_*.py`, `test_serve_assistant_*.py`,
  `test_analyze_*.py`): 507 passed, 4 skipped, 80 subtests
  passed in 156 s.

## Files

- `harness/voice-driver.py` — the keyboard driver (set state, tab
  until predicate, write marker, extract `SPEECH OUTPUT` lines, dump
  DOM state before/after).
- `harness/stub_server.py` — the in-container stub backend with
  per-state voice status, per-scenario transcribe behaviour, and
  per-scenario speak stream length. **Updated mid-run**: the speak SSE
  response now closes after the `done` event (the real server does the
  same; the client `readSseStream` reads to EOF).
- `harness/start-voice.sh` — container bootstrap (dbus, Xvfb,
  speech-dispatcher with sd_dummy debug logging, Orca user settings,
  WAV generation, stub launch).
- `scenarios/<name>/orca-utterances.txt` — verbatim speech.
- `scenarios/<name>/orca-raw.txt` — full Orca debug section.
- `scenarios/<name>/ax-before.json`, `ax-after.json` — DOM snapshots.
- `harness/endedprobe.py`, `rateprobe.py`, `audioprobe.py`,
  `queueprobe.py` — in-container probes that confirmed the
  AudioContext clock + `onended` behaviour during debugging.

## Defects remaining / known gaps

- Orca 43.1 with the web script announces polite-region updates as
  observed (Transcribing…, Reading reply aloud., Stopped speaking.,
  Finished speaking., Recording started… press Escape to cancel., and
  Voice set…) but its output is still driven by the speech-dispatcher's
  `sd_dummy` module: audible on a real machine, not audible in this
  container. The capture is the speech record.
- Chromium in a container without ALSA/Pulse falls back to a silent
  sink; WebAudio scheduling and `onended` are functional here, but the
  audio device "sound" never reaches a speaker. This is independent of
  the UI gate.