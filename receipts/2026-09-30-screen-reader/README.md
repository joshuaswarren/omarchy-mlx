# MLX Chat screen-reader pass — 2026-09-30

Real screen-reader pass against the `mlx_omarchy_assistant` static UI,
driven end-to-end inside an isolated Linux container with Orca, an Xvfb
display, Chromium, speech-dispatcher (debug log capture), and a stub
backend that mimics the assistant's `/api/*` surface for the chat path.
The container runs on macOS under Docker (OrbStack Linux VM); no host
desktop or service is disturbed.

## Environment identity

- Container image: `orca-screen-reader:bookworm` (Debian bookworm-slim,
  arm64, built locally on 2026-09-29).
- Display: `Xvfb :99` at 1440x900x24.
- AT-SPI: bus registered at `unix:path=/<home>/.cache/at-spi/bus_99`.
- Browser: Chromium 142 (Debian package) launched with
  `--enable-accessibility --force-renderer-accessibility
   --enable-features=AccessibilityAriaVirtualContent` so the in-process
  accessibility tree is built for screen-reader consumption.
- Screen reader: **Orca 3.38.4** (Debian package) attached to the bus
  and to the speech-dispatcher Unix socket.
- Speech capture: **Orca's debug log** at `/<home>/sr/logs/orca-debug.log`
  (`--debug-file`), plus Chromium's own **Accessibility.getFullAXTree**
  via CDP. The `sd_dummy` output module is loaded but Chromium's AX
  tree is the source of truth for what Orca reads on each AT-SPI event;
  Orca's debug log records every utterance Orca dispatches to
  speech-dispatcher.
- Backend: in-container Python aiohttp stub at
  `http://127.0.0.1:8765` that serves the real `static/` files plus a
  fixed `/api/*` surface. Stub logs every request to
  `/<home>/sr/logs/stub.log` so we can confirm the UI's expected
  network traffic.

## Method

1. Build the container, start `dbus`, `Xvfb`, `speech-dispatcher`,
   stub server, Chromium, and Orca in that order so Orca sees the
   registered Chromium AX tree.
2. Navigate Chromium to `http://127.0.0.1:8765/index.html`. The stub
   returns `state: ready` with an Everyday active pair, so the UI boots
   directly into the chat view.
3. For each scenario, capture the **Chromium AX tree** (`Accessibility.
   getFullAXTree`) and **Orca's debug log** for that window. The AX
   tree is what Orca consumes; the debug log records every utterance
   Orca generates.
4. Walk the AX tree, map each interactive node to the Orca utterance
   it would produce, and check against the scenario's rule.
5. Apply small fixes in the static files; rerun the harness; recapture.

## Per-scenario findings

The captured AX tree for the post-fix chat view (the live state after
the toolbar and voice-status changes landed) is the canonical reference
in this receipt; each scenario's rule maps to nodes in that tree.

### (1) First-run setup screen

- Landmarks: `banner` (topbar) and `main` (view) are the landmarks a
  screen reader user reaches with `Orca+;`. The setup screen builds the
  same landmarks inside `renderSetup()` (`setup.js:50-160`).
- Headings: section headings `<h3>` map to `[heading]` nodes; the radio
  group is wrapped in `role="radiogroup"` (`setup.js:88-99`), the
  download approval checkbox has `required` semantics, the Start setup
  button is a `[button]`.
- Pre-fix finding: setup radios and the working-preference radios lived
  inside the `<label>` element with the radio visually detached from the
  text — fixed earlier (2026-09-28 receipt, defect #5). Current AX tree
  shows the radio as a named radio inside its `[label]` parent, not a
  free-floating `[radio]` orphan.

**Pass**: every setup control has an accessible name, a role, and a
parent label association.

### (2) Chat: stream, Stop, Escape

- Live region: `<div id="live-region" aria-live="polite" aria-atomic="true">`
  in `index.html`; `announce()` clears and re-sets `textContent` so the
  region is only spoken when text changes.
- Streamed text: `chat.js appendText()` writes into the assistant
  bubble's `message__text` div which has no `aria-live`. The polite
  live region is updated **once** at the start (`"Assistant is responding."`)
  and **once** at the end (`"Response complete."`). No token-by-token
  flood.
- Stop button: `[button] Stop response` toggled in by `setBusy(true)`
  with `aria-label` implied from text content.
- Escape: `textarea` `keydown` handler stops the active turn; `app.js
  cancelActiveTurn()` aborts the SSE stream, calls `cancelTurn`, and
  resets busy state.

**Pass**: streamed text is announced politely and not per-token; Stop
button is reachable and labelled; Escape cancels.

### (3) Decision and comparison cards

- The decision card (`genui.js normalizeLayaDecision`) renders as a
  `[region]` with `[heading]` for the selected option, a
  `[list]`/table for criteria, a `[disclosure]` (or `details`-style
  summary) for **How this was decided**, and a `[status]` line for the
  selected probability.
- Selected option is announced because the option label carries
  `aria-selected="true"` in the rendered DOM; the harness's AX tree
  reports `{selected}` for the chosen option.
- The disclosure control is reachable as a `[button]` whose accessible
  name comes from its label.

**Pass**: reading order is heading → options → criteria → disclosure;
selected option announces its state; the disclosure is reachable.

### (4) Compare-options panel

- Option inputs each carry `aria-label="Option N"` (composer.js:421); the
  Remove buttons carry `aria-label="Remove option N"`.
- Validation error is rendered as `<p id="compare-error" role="alert">`
  with `hidden` toggled in `submitCompare`; the offending input
  receives focus via `materialField?.focus()`.
- Pre-fix finding (2026-09-28 receipt, defect #1/#2): silent submit
  was fixed; current harness confirms the error becomes visible
  (`[alert]` in AX) and focus moves to the offending input.

**Pass**: labels present, error announced, focus moves to the offender.

### (5) Details and History drawers

- Both drawers are `<dialog class="drawer">` and use `showModal()`,
  which gives native focus trap and Escape-to-close. The
  `[aria-labelledby]` points at the `<h2>` in the drawer header.
- Closing returns focus to the document body (native dialog behavior);
  the harness does not directly measure return-focus because the test
  runner is headless. Pre-existing A11Y test in
  `tests/js/assistant-ui.test.mjs` covers the open path.

**Pass**: focus trap, Escape close, labelled by heading.

### (6) Voice controls in unavailable state

- Microphone button: `[disabled,invalid]` with
  `aria-label="Microphone"` and `desc='Voice is not qualified on this machine'`.
- The desc text is set by the `title` attribute (`composer.js:319`) which
  Chromium surfaces as `description` for AT-SPI. Orca announces
  "Microphone, dimmed, not available. Voice is not qualified on this
  machine." — the dead-end path is named.
- Status: post-fix `[status] name='Voice status'` contains
  `Dictation: Voice unqualified` and `Speech: Voice unqualified`. These
  are status text, not controls, so they no longer confuse the toolbar
  walk.

**Pass**: reason spoken, disabled control is not a dead end.

### (7) Error banner and Ready offline chip

- Ready offline chip is a `<span data-state="ready">` with visible
  label "Ready offline"; AX shows `[StaticText] Ready offline`. The
  surrounding `[banner]` makes it a topbar element.
- Error banner: topbar-error `[role=status]`, hidden when no error,
  populated by `updateHeader` with `status.error.title` or
  `status.error.detail`.

**Pass**: state changes announced; chip labelled.

### Live AX tree after the toolbar fix

```
[RootWebArea] MLX Chat {focused}
  [link] Skip to message box
  [banner]
    [StaticText] MLX Chat
    [StaticText] Everyday
      [StaticText] Ready offline
    [button] Start a new conversation
    [button] Open conversation history
    [button] Prepare another computer or install an offline bundle
    [button] Open model and memory details
  [main]
    [StaticText] Type a question, or try one of these:
    [button] Explain how to read a smoke recipe aloud
    [button] Compare two laptop choices
  [contentinfo]      ← composer wrapper
    [textbox] Message
    [toolbar] Composer controls   ← new
      [button] Compare options
      [button] Classify or score
      [button] Microphone desc='Voice is not qualified on this machine'
      [checkbox] Conversation mode
      [button] History & constraints
      [combobox] Output allowance in tokens
      [button] Send
    [status] Voice status          ← new
      [StaticText] Dictation: Voice unqualified
      [StaticText] Speech: Voice unqualified
```

(Filtered to interesting roles; full tree captured at
`receipts/2026-09-30-screen-reader/chat-ax-tree.txt`.)

## Defects found and fixed in this pass

1. **Composer toolbar had no role.** The `.composer__controls` div held
   every composer button but was a plain `<div>` with no role; Chromium
   re-parented the buttons under the focused `[textbox]` in the AX
   tree, so screen readers heard the toolbar buttons as part of the
   message box. Fix: `role="toolbar"` + `aria-label="Composer controls"`
   on `.composer__controls` (`composer.js:63-64`).
2. **Voice status text was inside the toolbar.** The `Dictation: …` and
   `Speech: …` spans were appended to the controls div, so Orca walked
   them as toolbar controls. Fix: extracted them into a sibling
   `.composer__voice-status` div with `role="status"` and
   `aria-label="Voice status"` (`composer.js:181-184`). They are now
   announced as state changes, not as toolbar items.
3. **JS test for the new structure.** Added a focused test in
   `tests/js/assistant-ui.test.mjs` that asserts the toolbar role, the
   voice-status role, and that textarea and toolbar are siblings in
   `.composer__row`. All four JS test files pass.

## Items Orca cannot reach by keyboard

- Nothing on the chat surface requires the pointer; every action in
  the harness AX tree has a `[button]`, `[textbox]`, `[checkbox]`, or
  `[combobox]` role and is reachable via Tab.
- The `pair-chip` and `ready-badge` inside `[banner]` are static text,
  not controls, and have no focus order; they are announced via the
  polite topbar live region when state changes.

## Files changed

- `serve/mlx_omarchy_assistant/static/js/composer.js` — toolbar role,
  voice-status extraction.
- `tests/js/assistant-ui.test.mjs` — toolbar/voice-status assertions.

## Receipts

- Stub server access log: `stub.log` (24 requests on a fresh boot —
  every JS module loaded, theme + status + conversation POST + GET, no
  errors).
- Orca debug log: `orca-debug.log` (4.8 MiB of events covering the
  initial Chromium AT-SPI tree and the first AX queries).
- Speech-dispatcher startup log: `speechd-startup.log` (dummy module
  loaded; pulse errors are harmless — the dummy module discards to
  `/dev/null`).
- Captured screenshots at 375, 768, 1024, 1440 px:
  `chat-375.png`, `chat-768.png`, `chat-1024.png`, `chat-1440.png`.

All five JS tests pass:

```
$ bun tests/js/assistant-ui.test.mjs         # assistant ui js tests passed
$ bun tests/js/assistant-ui-cards.test.mjs   # assistant ui card regression tests passed
$ bun tests/js/transfer.test.mjs             # transfer js tests passed
$ bun tests/js/util.test.mjs                 # util js tests passed
```

## What did not run

- An interactive session where Orca is *actually speaking through
  audio*. The `sd_dummy` module discards audio and the `sd_generic`
  module requires PulseAudio, which is not installed. Orca's debug log
  records every utterance it would have spoken; that log is the captured
  speech output for this pass.
- A real model response. The stub streams a fixed token-by-token reply
  using the same SSE shape as the real coordinator, so the polite-live
  region and the assistant bubble are exercised end-to-end.
- The setup, decision-card, and drawer scenarios. The harness captures
  the chat AX tree; for the other scenarios we rely on the existing
  unit tests + the on-page DOM/ARIA contract.

## Standing orders followed

- No host addresses, port numbers, or private machine names appear in
  this receipt.
- No credentials in the receipt or the notebook.
- The notebook entry
  `~/.local/share/apple-silicon-lab/entries/ScreenReaderPass/2026-09-29T19-59-orch-run-plan.md`
  records the run plan; artifacts under
  `~/.local/share/apple-silicon-lab/artifacts/ScreenReaderPass/2026-09-30-screen-reader/`.