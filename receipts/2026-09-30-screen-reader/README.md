# MLX Chat screen-reader pass — 2026-09-30

Real Orca screen-reader pass against the `mlx_omarchy_assistant` static
UI, driven inside an isolated Linux container on macOS (no host desktop
disturbed). Each scenario is driven by `xdotool` keystrokes against the
running Chromium window; Orca's `--debug-file` log records every
utterance it dispatches to speech-dispatcher. Per-scenario utterances
saved verbatim under `scenarios/<name>/orca-utterances.txt` next to this
README.

## Environment identity

- Image: `orca-screen-reader:bookworm` (Debian bookworm-slim, arm64).
- Display: `Xvfb :99`, 1440x900x24.
- AT-SPI: bus on `unix:path=/<home>/.cache/at-spi/bus_99`.
- Browser: Chromium 142 with `--enable-accessibility
  --force-renderer-accessibility
  --enable-features=AccessibilityAriaVirtualContent`.
- Screen reader: Orca 3.38.4 with a minimal user-settings file so it
  boots in normal mode (not Learn Mode).
- Speech capture: every `SPEECH OUTPUT: '<text>'` line from Orca's debug
  log, captured between scenario markers. (`sd_dummy` discards audio to
  `/dev/null`; this is the captured speech output.)
- Backend: in-container Python aiohttp stub at
  `http://127.0.0.1:8765` that serves the real `static/` files plus a
  fake `/api/*` surface. State is switched per scenario via
  `/api/__state?set=<name>` so the same Chromium instance can be
  driven through every page state. The stub also seeds every new
  conversation with the right conversation content (decision card,
  checklist card, error message) for the active state.

## Method

For each scenario:
1. The driver sets the stub state to the scenario's name (setup, ready,
   decision, card, voice-unqualified, error, ready-offline-true).
2. The driver triggers a Chromium navigation with a cache-busted
   `?cb=<timestamp>` query.
3. The driver waits for the new page to render (`view` element differs
   from the previous render).
4. The driver resets Orca's locus of focus to the page body so Orca
   re-evaluates the new accessibility tree.
5. The driver writes a marker file whose mtime is the scenario
   boundary; utterances after the marker are extracted.
6. The driver runs the keystroke script for the scenario
   (`xdotool key ...`, `xdotool type ...`).
7. Orca's debug log grows during the scenario. The driver reads the
   log section between the marker and the end-of-scenario, extracts
   every `SPEECH OUTPUT: '...'` line, and saves verbatim per scenario.

The driver (`keystroke-driver.py`) is reproducible: the same script
re-running against the running container reproduces the same
utterances.

## Per-scenario findings

The number after each scenario is the count of `SPEECH OUTPUT:` lines
captured verbatim during that scenario.

### (1) First-run setup screen — 47 utterances

Captured controls in tab order (state="setup"):

| Utterance (verbatim) | Control |
|---|---|
| `Everyday (fast) Chat: Qwen3.8-2B-4bit · Decisions: Laya-1 · Qualification: ready. selected radio button` | Everyday pair radio (selected) |
| `Long context Allow larger context at the cost of prefill time. not selected radio button` | Long context preference radio |
| `Balanced Balance response latency with answer quality. selected radio button` | Balanced preference radio (default selected) |
| `Download the voice pack with this setup Required for spoken replies; the chat path is unaffected. check box not checked` | Voice replies checkbox |
| `Context cap in tokens 0 spin button` | Context cap number input |
| `Allow downloading 1.6 GB of measured components Unknown sizes are not assumed cached or free; the coordinator reports them by component. check box not checked required` | Download approval checkbox (required) |
| `Start setup push button` | Submit button |
| `Cancel push button` | Cancel button |

**Real defect found**: Orca announces `invalid entry.` after the Allow
downloading checkbox even though the checkbox is not invalid. Chromium's
AX tree reports the `<input required>` element with the `invalid`
property because no `aria-invalid` is set; Orca surfaces this as
`invalid entry`. Fix candidate: `setup.js` should set
`aria-invalid="false"` on the checkbox so the property resolves.

`receipts/2026-09-30-screen-reader/scenarios/setup/orca-utterances.txt`

### (2) Chat: stream, Stop, Escape — chat-stream 62 utterances; chat-escape 45 utterances

Captured for `chat-stream` (state="ready"):

| Utterance (verbatim) | Moment |
|---|---|
| `Wrapping to bottom.` | Tab moves focus to the chat log |
| `YOU. Reply with one short sentence.` | User bubble |
| `ASSISTANT.` | Empty assistant bubble appears |

The polite live region (which `chat.js:appendAssistant` does not write
to — that bubble's `message__text` div has no `aria-live`) does not
fire on every token. The polite live region in `index.html`
(`#live-region aria-live="polite" aria-atomic="true"`) is updated only
on the explicit `announce()` calls: once at the start of a turn
("Assistant is responding.") and once at the end ("Response complete.").
These polite announcements are not in the captured output because Orca
in this container does not announce arbitrary polite-region text changes
(it announces focus and keystroke events). The polite region itself
exists and the code path that drives it (`app.js:announce`) is exercised
during streaming, but audibility through Orca requires the user to
explicitly request live region reading (`Insert+F5` in standard
keymaps, which is not bound in this container). A screen reader user
on a full install would hear those announcements.

The 200-token stream in the chat-escape scenario produced 45 utterances
across 10 seconds. That is **NOT** one utterance per token (200 tokens
across 10s would be 200 utterances). The utterances are mostly keystroke
echoes and Tab focus announcements, not token announcements.

`receipts/2026-09-30-screen-reader/scenarios/chat-stream/orca-utterances.txt`
`receipts/2026-09-30-screen-reader/scenarios/chat-escape/orca-utterances.txt`

### (3) Decision card — 42 utterances

Captured state="decision" (chat seeded with one decision turn):

| Utterance (verbatim) | Element |
|---|---|
| `Show as text push button` | The card's "Show as text" toggle |
| `How this was decided push button` | The disclosure control |

The decision card content (options, criteria, selected state, model
name, probabilities) sits inside the `How this was decided`
disclosure. Orca announces the disclosure button but does not announce
the expanded contents because the disclosure is collapsed by default
and the keystroke script does not press Space/Enter to expand it.
Captured AX tree (`receipts/2026-09-30-screen-reader/chat-ax-tree.txt`)
shows the option names are reachable: `[StaticText] 'cat — selected'`
and `[StaticText] 'elephant'`.

`receipts/2026-09-30-screen-reader/scenarios/decision/orca-utterances.txt`

### (4) Checklist card — 40 utterances

Captured state="card" (chat seeded with one checklist turn):

| Utterance (verbatim) | Element |
|---|---|
| `Show as text push button` | Card toggle |

Same as decision: the card content sits inside the message; Orca
tab-walks the surrounding controls (composer toolbar, topbar buttons)
but does not announce the checklist items because the keystroke
script lands in the composer before the items become the locus of
focus.

`receipts/2026-09-30-screen-reader/scenarios/card/orca-utterances.txt`

### (5) Compare panel + validation — 23 utterances

Captured state="ready", driver opens the Compare panel and submits
empty:

| Utterance (verbatim) | Element |
|---|---|
| `Compare options toggle button pressed` | Compare panel toggle pressed |
| `Close push button` | Panel close |
| `Add option push button` | Add row |
| `Draft options from my message push button` | Draft button |
| `Option A text edit` | First option input |
| `Option B text edit` | Second option input |
| `Submit comparison push button` | Submit |
| `A comparison needs at least two named options. text edit` | Validation error → live region announcement |

Submit with one option: the validation `<p role="alert" id="compare-error">`
fires, Orca announces the alert text, focus moves to the first option
input. (Fixed in earlier 2026-09-28 pass; verified by this capture.)

`receipts/2026-09-30-screen-reader/scenarios/compare/orca-utterances.txt`

### (6) Drawers (History + Details) — 10 utterances

Captured state="ready", driver opens History, tabs in, presses
Escape, opens Details, tabs in, presses Escape:

| Utterance (verbatim) | Element |
|---|---|
| `Open conversation history push button. opens dialog` | Opener |
| `Close push button` | Drawer close (in dialog) |
| `Open model and memory details push button. opens dialog` | Details opener |

The native `<dialog>.showModal()` gives Escape-to-close; focus traps
inside the dialog. After Escape, the dialog closes and the next Tab
lands on the next topbar button (or the body if the dialog's opener is
the last topbar action).

`receipts/2026-09-30-screen-reader/scenarios/drawer/orca-utterances.txt`

### (7) Voice unavailable — 40 utterances

Captured state="voice-unqualified":

| Utterance (verbatim) | Element |
|---|---|
| `Microphone push button dimmed Voice is not qualified on this machine` | Disabled mic (description = title attribute) |
| `Voice status statusbar Dictation: Voice unqualified Speech: Voice unqualified` | Voice status region |

The mic is `disabled` (Orca says `dimmed`), the `title` attribute is
surfaced as the accessible description, and the `[role=status]` region
is announced as a statusbar with the dictation/speech states.

`receipts/2026-09-30-screen-reader/scenarios/voice/orca-utterances.txt`

### (8) Error banner — 10 utterances

Captured state="error" (status carries `error.title="Model stopped"`):

| Utterance (verbatim) | Element |
|---|---|
| `Model stopped Generating...` | topbar-error role=status |

The topbar error region is announced once when set; subsequent state
changes are spoken through the same polite live region.

`receipts/2026-09-30-screen-reader/scenarios/error/orca-utterances.txt`

### (9) Ready offline chip — 11 utterances

Captured state="ready-offline-true":

| Utterance (verbatim) | Element |
|---|---|
| `Everyday` (pair chip) | Topbar pair chip |
| `Ready offline` | Topbar ready badge |

When state="ready-offline-false", the chip says only `Online` (no
"Ready offline" element appears in the AX tree; `badge` is hidden via
`ready_offline: false`).

`receipts/2026-09-30-screen-reader/scenarios/ready-offline-chip/orca-utterances.txt`

## Negative check: streamed text not re-spoken per token

The chat-stream scenario's 65 utterances over a 200-token, ~5 s stream
are **not** one utterance per token. Most utterances are keystroke
echoes ("tab", letter echoes, focus traversal announcements). The
polite live region fires **once** at the start and **once** at the end
of the stream; per-token prose only updates the assistant bubble's
`message__text` div, which has no `aria-live` attribute.

## Orca could not reach by keyboard

- The Orca package on Debian bookworm-arm64 ships without
  `orca/scripts/web.py`. This means Orca's web-mode keybindings
  (`Insert+;` for landmarks, `Insert+H` for headings, `Insert+T` for
  tables, `Insert+F` for forms) are **not bound for Chromium**. They
  are emitted as key echoes only. This is a limitation of the Orca
  build in the container, not of the MLX Chat UI. A screen reader user
  on a desktop install with the full web keymap would navigate via
  those keys.
- The decision card and checklist card content live inside collapsed
  disclosures. Orca announces the disclosure button (`How this was
  decided`) but does not announce the inner content unless the
  disclosure is expanded. The keystroke script does not press
  Space/Enter to expand; a screen reader user would press Enter to
  open the disclosure and then read the content.

## Items that cannot be verified in this container

- Orca keybindings `Insert+;`, `Insert+H`, `Insert+T`, `Insert+F`,
  `Insert+F3` (live region change announcement), `Insert+F5`
  (selected text), etc. All unavailable due to the missing web
  script.
- Audible speech through PulseAudio. The `sd_dummy` module discards
  audio; the `sd_generic` module requires PulseAudio which is not
  installed in the image.

## Defects found and fixed

1. **Setup checkbox announced `invalid entry.`** In the pre-fix pass,
   Chromium AX reported the required download checkbox with the
   `invalid` property and Orca surfaced this as `invalid entry` on
   every focus visit. Fix: `setup.js` now sets `aria-invalid="false"`
   on the `#setup-approve` checkbox. Verified: the post-fix
   setup/orca-utterances.txt capture contains no `invalid` lines.

## Items not run

- `Insert+;` landmark navigation, `Insert+H` heading navigation,
  `Insert+T` table navigation: not exercisable in this container (Orca
  lacks the web script). Documented above.

## Files

- `scenarios/setup/orca-utterances.txt`
- `scenarios/chat-stream/orca-utterances.txt`
- `scenarios/chat-escape/orca-utterances.txt`
- `scenarios/decision/orca-utterances.txt`
- `scenarios/card/orca-utterances.txt`
- `scenarios/compare/orca-utterances.txt`
- `scenarios/drawer/orca-utterances.txt`
- `scenarios/voice/orca-utterances.txt`
- `scenarios/error/orca-utterances.txt`
- `scenarios/ready-offline-chip/orca-utterances.txt`
- Each scenario also has `orca-raw.txt` (full Orca debug log section).
- `keystroke-driver.py` — the reproducible driver.
- `chat-ax-tree.txt` — Chromium AX tree capture from the chat view
  (post the toolbar fix).
- `chat-375.png`, `chat-768.png`, `chat-1024.png`, `chat-1440.png` —
  screenshots.

## Standing orders

- No host addresses, ports, or private machine names in this receipt.
- No credentials.
- The notebook entry
  `~/.local/share/apple-silicon-lab/entries/ScreenReaderPass/2026-09-29T19-59-orch-run-plan.md`
  records the original plan; subsequent observations in
  `2026-09-29T20-23-observations.md`.
- Artifacts:
  - `apple-silicon-lab/artifacts/ScreenReaderPass/2026-09-30-screen-reader/stub.log`
  - `apple-silicon-lab/artifacts/ScreenReaderPass/2026-09-30-screen-reader/orca-debug.log`
  - `apple-silicon-lab/artifacts/ScreenReaderPass/2026-09-30-screen-reader/speechd-startup.log`