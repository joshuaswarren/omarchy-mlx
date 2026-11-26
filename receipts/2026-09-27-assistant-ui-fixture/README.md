# 2026-09-27 — AssistantUI browser fixture pass (component-level UI verification)

Worktree: `~/.config/superpowers/worktrees/mlx-omarchy/AssistantUI`
Method: real static modules (`app.js`, `chat.js`, `composer.js`, `genui.js`, `api.js`)
exercised in headless Chromium against a labeled component fixture
(`tests/js/fixture/`, **removed on completion** as instructed). All message,
card, and SSE content in the fixture was scripted UI test input — no model ran,
no real local-model output is claimed anywhere, and nothing here demonstrates
application readiness.

## Defects found and fixed (all in owned files)

1. **Typed cards crashed when results carry no probabilities**
   (`genui.js`): `typedConfidenceNote()` returns `null` for a non-abstained
   result without probabilities, and `card.appendChild(null)` throws — aborting
   the whole typed card after the first such result. Fixed by appending only
   when a node is returned. Regression: `tests/js/assistant-ui-cards.test.mjs`.
2. **Malformed/hostile components could break the whole conversation render**
   (`genui.js`): `renderComponent` had no error boundary; e.g.
   `{type:"comparison"}` without `columns` threw during `renderInto` and killed
   the full log render, and a thrown error inside the SSE `component` handler
   would trigger the event-gap reload loop. Now any renderer throw renders the
   component inert (returns nothing). `plainText` got the same boundary so
   "Show as text" cannot throw on inert cards. Chart series guards added
   (string `series` no longer reaches `drawChart`, which threw asynchronously).
3. **Composer stuck busy forever after cancel** (`app.js`): Escape/Stop called
   `view.cancelActiveTurn()` only — `app.activeTurn` stayed set and the SSE
   stream stayed attached, so Send never returned and the refused-send
   announcement looped. New `App.cancelActiveTurn()`: abort the stream, cancel
   server-side, reset busy, resubscribe from the last sequence. Verified live:
   cancel POST recorded, Send returns, stream resubscribed.
4. **Event-gap reload permanently deleted the composer** (`app.js`):
   `reloadAfterEventGap` calls `view.renderInto(this.mount)`, which
   `replaceChildren()`s the mount — the composer was never re-appended, so
   after a 409 event-gap the user could not type at all. Fixed by re-appending
   `composer.wrap` after the re-render.
5. **Compare panel closed on refused sends, dropping the submission**
   (`composer.js`): `onCommit` closed the panel unconditionally even when the
   send was refused (busy) or material was empty. `dispatch` is now awaited and
   returns acceptance; the panel closes only on an accepted send (verified:
   busy submit keeps the panel open with edits, free submit closes it).
6. **Compare/Classify panel Close desynced the toolbar toggle**
   (`composer.js`): the panel's own Close button cleared `hidden` without
   updating `compareOpen`/`analyzeOpen`, so the toolbar button could never
   reopen the panel. Both panels now route Close through the same state.
7. **Pair chip hard-clipped long labels** (`css/app.css`): added
   `overflow: hidden; text-overflow: ellipsis` to `.topbar__pair`.
8. Anti-slop lint violations from the parent's lint run fixed without rule
   disables (`no-runtime-typeof` ×6, useless spread, unused variable,
   `new Array(n).fill`), reusing `util.js` `asString`/`isPlainObject` boundary
   parsers. `oxlint -c … serve/mlx_omarchy_assistant/static tests/js` exits 0.

Dead code removed: `ConversationView.activeAbort` was assigned nowhere; the
abort in `cancelActiveTurn` was a no-op. Field and line deleted.

## Flows verified in the browser (fixture input, screenshots attached)

- Composer typing while busy: textarea stays editable, Send↔Stop swap,
  typed-while-pending text survives acceptance and the done event
  (`01-typing-while-busy-1024.png`).
- Cancel while busy via Escape and via Stop button (`02-after-cancel-1024.png`).
- Draft compare confirmation preserving source: draft turn keeps composer text;
  `comparison_draft` status prefills the panel with options/criteria/source;
  submit records mode `compare` with the source text preserved verbatim and
  options/criteria riding separately (`03-compare-panel-open-1024.png`).
- Typed choice/score cards: bare (no probabilities), scored (72% hint),
  abstained note, decision card with 70% + "How this was decided", and
  "Show as text" fallback (`04-typed-cards-1024.png`).
- Malformed/injected components inert: hostile string `series` renders nothing
  executable (`window.__FIXTURE_PWNED__` never set, zero `<img>` injected),
  unknown `type` renders nothing, valid chart still renders
  (`05-injected-cards-1024.png`). Zero page errors across the whole session.
- Context selection: modal opens with focus inside, "All/Selected/None" modes,
  checkboxes disabled unless "Selected turns", saved payload captured as
  `{selected_turn_ids:["fx-turn-0"], pinned_constraints:"…"}`, drawer closes,
  Escape closes (`06-context-drawer-1024.png`, `08-context-drawer-375.png`).
- Keyboard: first Tab focuses the skip link (revealed at top 8px), Enter jumps
  to the composer; Shift+Enter inserts a newline without sending; Enter sends.
- Widths 375/768/1024/1440: no horizontal overflow anywhere; composer and
  topbar wrap cleanly (`07-width-{375,768,1024,1440}.png`).

## Regression tests retained

- `tests/js/assistant-ui-cards.test.mjs` — typed-card no-probability render,
  13 malformed payload cases inert, `plainText` fail-safe, panel-close toggle
  desync, analyze inline validation error.
- Existing suites still pass: `bun tests/js/*.test.mjs` → 4/4 green.

## Verification limits

- Hardware is inaccessible from this session; no model, tokenizer, TTS, or
  server process was exercised. Every payload above is fixture data.
- The fixture page booted the real `App` against a stubbed `fetch` transport
  and a scripted SSE `ReadableStream` — same-origin, CSP `script-src 'self'`
  respected; the transport stub is labeled in the fixture source and was
  deleted with the fixture.
