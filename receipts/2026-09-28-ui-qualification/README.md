# MLX Chat UI qualification — 2026-09-28

Real-app screenshots and accessibility pass of the `mlx_omarchy_assistant` static
UI, driven against the running `<project-m2>` host. All UI files edited in this
pass live under `serve/mlx_omarchy_assistant/static/`. JS tests live under
`tests/js/`.

## Captures

24 PNGs at the required widths (375, 768, 1024, 1440), full page, deviceScaleFactor 1,
plus one 200% zoom evidence frame. Every file shows the state named in its filename.

### Screenshots

| File | State | What is visible |
| --- | --- | --- |
| `setup-375.png`, `setup-768.png`, `setup-1024.png`, `setup-1440.png` | setup | First-run wizard with the two pairs (Everyday, Quality), Working preference radios, Voice replies, Context cap, Download approval checkbox, Start setup/Cancel. Radios sit cleanly on the right of each pair card and to the left of each preference option. |
| `empty-375.png`, `empty-768.png`, `empty-1024.png`, `empty-1440.png` | empty | New conversation with no messages. Header chip "Everyday" + Online badge. Chat log shows "Type a question, or try one of these:" and the two example chips. Composer present with all controls. |
| `chat-375.png`, `chat-768.png`, `chat-1024.png`, `chat-1440.png` | chat | One finished chat turn: user bubble "Reply with one short sentence." above an assistant bubble with the model's reply plus a "Read aloud" action. Composer returned to the ready state (Send visible, Stop hidden). |
| `decision-375.png`, `decision-768.png`, `decision-1024.png`, `decision-1440.png` | decision | One finished Compare options turn with options `cat` and `elephant`, criteria "shorter spelling". The decision card lists both options with the selected option highlighted, "Selected probability: 54%", the criteria line, and "How this was decided". |
| `voice-375.png`, `voice-768.png`, `voice-1024.png`, `voice-1440.png` | voice | The microphone control in its explanatory state: the Microphone button is disabled (greyed), with inline status text "Dictation: Voice unqualified" and "Speech: Voice unqualified" rendered in the warn colour. The button's `title` attribute carries the longer explanation "Voice is not qualified on this machine". |
| `error-375.png`, `error-768.png`, `error-1024.png`, `error-1440.png` | error | Compare-options submit with only one option named. Inline alert "A comparison needs at least two named options." rendered in the error colour under the Submit button. The composer draft "Which is shorter to type?" is preserved. The compare panel is still open and focus is on the empty option input. |
| `zoom200-512x900.png` | 200% zoom | Same chat view rendered at viewport 512×900 with deviceScaleFactor 2 (effective 1024×1800). Topbar wraps cleanly (brand left, Online chip top-right, action buttons row 2). Composer controls reflow with no clipping. No horizontal page scroll. |

## Accessibility checks

| Check | Result | Evidence |
| --- | --- | --- |
| Keyboard only — Tab order reaches every control, focus always visible, Escape stops a running response, no keyboard trap | Pass | Tab traversal across the running page (skip link → brand → topbar actions → chat example chips → composer textarea → Compare options / Classify or score / Microphone / Conversation mode / History & constraints / Output allowance / Send / Close / option inputs / Submit comparison) without trapping. Focused element reports `outline: solid 2px rgb(122, 162, 247)` from `--mlx-focus` and `matches(':focus-visible')` is true. Escape in a running turn flipped send hidden / stop visible to send visible / stop hidden. |
| 200% zoom at 1024 px wide — no clipped or overlapping controls, no horizontal page scroll | Pass | Viewport 512×900 with deviceScaleFactor 2. `document.documentElement.scrollWidth === window.innerWidth === 512`; `body.scrollWidth === 512`. Composer (`right: 469`), send button (`right: 469`), topbar (`right: 512`), and example chips (`right: 496`) all sit inside the viewport. Captured in `zoom200-512x900.png`. |
| Contrast — body text, muted text, buttons, "Ready offline" chip | Pass | WCAG ratios sampled against the live `/api/theme`-applied palette: body text on canvas 10.59, muted on canvas 8.20, button text on focus 8.34, Ready offline text on canvas 6.79, Ready offline border on canvas 6.79, focus-ink on focus 8.34, warn (Voice unqualified text) on canvas 8.55, error on canvas 6.46. All above 4.5. |
| Reduced motion — no animation runs under `prefers-reduced-motion: reduce` | Pass | `window.matchMedia('(prefers-reduced-motion: reduce)').matches === true`. `document.getAnimations()` returns no running animations. The only declared `@keyframes` rule is `pulse` (recording indicator), already gated by `@media (prefers-reduced-motion: reduce) { .recording-indicator__dot { animation: none; } }`. |
| Screen reader semantics — landmarks, labels on all controls, live region for streaming text, no unlabeled buttons | Pass | Landmarks present: `header[role=banner]`, `main#view[role=main]`, `footer[role=contentinfo]`, three `<dialog>` drawers (`history-drawer`, `details-drawer`, `transfer-drawer`), and the setup/chat regions. A live region exists with `id="live-region" aria-live="polite" aria-atomic="true"`. A second polite status lives in the topbar (`#topbar-error[role="status"]`). Every button has an accessible name (text content, `aria-label`, or `title`); every interactive input has either an `aria-label`, a wrapping `<label>`, or a `<label for=…>`. After this pass: the compare panel's option inputs each carry `aria-label="Option N"`, the criteria textarea carries `aria-label="Criteria"`, and the material textarea keeps `aria-label="Supplied material"`. The new error paragraph carries `role="alert"`. Skip link `.skip-link` points to `#composer-text`. |
| Console — no errors or warnings during all of the above | Pass | `tab.console({limit:200})` empty. `tab.errors({limit:100})` shows two `net::ERR_ABORTED` entries from the intentional `streamAbort.abort()` call inside `app.cancelActiveTurn()` — that abort is the documented behaviour for "user pressed Escape / Stop response" and is fired by the app itself, not by a third-party failure. No JS exceptions, no console warnings. |

## Defects found and fixed

1. **Silent compare-submit when fewer than two options are named** — `composer.js:391` returned without surfacing the cause and without keeping focus on the offending input. Real user impact: clicking Submit comparison with one (or zero) named options appeared to do nothing, and the typed draft was not visibly preserved. Now shows a `role="alert"` paragraph ("A comparison needs at least two named options."), focuses the first option input, and leaves the panel and composer draft untouched. Visible in `error-*.png`.
2. **Silent compare-submit when supplied material is empty** — `composer.js:259` focused the textarea and returned without naming the cause. Now shows the same alert ("Add the text the decision should be made on before submitting.") and focuses the material field.
3. **Compare panel inputs lacked accessible names** — `composer.js:417` and `composer.js:376` built the option text inputs and criteria textarea with placeholders only. Placeholders are not a substitute for labels per WCAG SC 1.3.1 / 4.1.2. Now each option input carries `aria-label="Option N"` (and the matching Remove button carries `aria-label="Remove option N"`), and the criteria textarea carries `aria-label="Criteria"`.
4. **`.message__error` had no style rule** — `chat.js` `setError()` appends `<div class="message__error">` and the analyze panel uses the same class for its row errors, but only `.message--error` (the bubble) had a rule. The inner error text inherited body styling. Added `.message__error { color: var(--mlx-error); font-size: .875rem; margin: .375rem 0 0; }`.
5. **Setup pair and preference radios were detached from their labels** — `setup__pair` (grid, input appended last) and `setup__preference label` (flex column, input on its own row) pushed each radio into the bottom-centre of its card, several pixels away from the visible text. Real user impact: at 375 px the radios floated into the gutter of the next card. Fixed with `setup__pair` as a 1fr / auto grid that places the radio at the right edge of each card, and `setup__preference label` with absolute-positioned radio at the left edge. Recaptured in `setup-*.png`.

## Files changed

- `serve/mlx_omarchy_assistant/static/css/app.css` (radio layout, `.message__error` style)
- `serve/mlx_omarchy_assistant/static/js/composer.js` (compare-submit error paths, aria-labels)

Both files were copied to the `<project-m2>` checkout so the running app reflected the edits. The assistant serves static files directly, so no restart was needed.

## JS tests

All four JS test files pass against the patched code:

```
$ bun tests/js/assistant-ui-cards.test.mjs   # assistant ui card regression tests passed
$ bun tests/js/assistant-ui.test.mjs         # assistant ui js tests passed
$ bun tests/js/transfer.test.mjs            # transfer js tests passed
$ bun tests/js/util.test.mjs                 # util js tests passed
```

## Servers and tunnels

The capture used one app instance with the Everyday pair loaded and a second instance with an empty home and no pair, for the setup screen. Each browser tab used its own one-use launch token. Both instances and their tunnels were stopped afterward, and a process check on `<project-m2>` found no assistant process.
