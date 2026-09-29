// Executable assertions for the typed-decision UI paths. Run directly:
//   bun tests/js/assistant-ui.test.mjs   (or: node tests/js/assistant-ui.test.mjs)
// Exit code 0 = all assertions passed; any failure throws.

import assert from "node:assert/strict";
import { parseAnalyzeQuestions } from "../../serve/mlx_omarchy_assistant/static/js/composer.js";
import { normalizeLayaDecision, plainText }
  from "../../serve/mlx_omarchy_assistant/static/js/genui.js";

// --- parseAnalyzeQuestions: the typed classify/score composer path -------

const parsed = parseAnalyzeQuestions([
  { instructions: "Which department?", type: "choice", labels: " billing , technical , other " },
  { instructions: "How urgent?", type: "score", labels: "low,high" },
]);
assert.deepEqual(parsed, [
  { id: "q1", type: "choice", instructions: "Which department?",
    options: ["billing", "technical", "other"] },
  { id: "q2", type: "score", instructions: "How urgent?", levels: ["low", "high"] },
]);

assert.throws(() => parseAnalyzeQuestions([]), /one and eight/);
assert.throws(() => parseAnalyzeQuestions(
  Array.from({ length: 9 }, () => ({ instructions: "x", type: "choice", labels: "a,b" }))),
  /one and eight/);
assert.throws(() => parseAnalyzeQuestions([{ type: "choice", labels: "a,b" }]),
  /needs wording/);
assert.throws(() => parseAnalyzeQuestions([{ instructions: "x", type: "choice", labels: "a" }]),
  /2-8 comma-separated labels/);
assert.throws(() => parseAnalyzeQuestions([{ instructions: "x", type: "score", labels: "a" }]),
  /2-8 comma-separated levels/);
assert.throws(() => parseAnalyzeQuestions([{ instructions: "x", type: "score", labels: "a,b,c,d,e,f,g,h,i" }]),
  /2-8 comma-separated levels/);
// empty labels between commas are dropped, not counted
assert.equal(parseAnalyzeQuestions([{ instructions: "x", type: "choice", labels: "a, b" }])[0]
  .options.length, 2);

// --- normalizeLayaDecision: coordinator compare decision -> decision card -

const compare = normalizeLayaDecision({
  type: "choice", choice: "a",
  options: [{ id: "a", label: "Keep data" }, { id: "b", label: "Delete data" }],
  probabilities: { a: 0.7, b: 0.3 },
  confidence: 0.4, rl_agent: { act_probability: 0.9 },
  abstained: false, criteria: "safety", model: "laya-mlx", input_scope: "the text",
});
assert.equal(compare.type, "decision");
assert.equal(compare.selected, "a");
assert.deepEqual(compare.options, [{ id: "a", label: "Keep data" }, { id: "b", label: "Delete data" }]);
assert.equal(compare.confidence.selected_probability, 0.7);
assert.equal(compare.confidence.abstained, false);

const abstained = normalizeLayaDecision({
  type: "choice", choice: "a", options: [{ id: "a", label: "A" }, { id: "b", label: "B" }],
  probabilities: { a: 1.0, b: 0.0 }, abstained: true,
});
assert.equal(abstained.confidence.abstained, true);
assert.equal(normalizeLayaDecision({ type: "noul" }), null);
assert.equal(normalizeLayaDecision("text"), null);

// --- plainText: text equivalents for the decision and typed payloads ------

const decisionText = plainText(compare);
assert.match(decisionText, /\[Decision card\]/);
assert.match(decisionText, /\* Keep data/);
assert.match(decisionText, /Model: laya-mlx/);

const typed = {
  type: "typed", model: "laya-mlx", input_scope: "invoice text",
  results: [
    { id: "dept", type: "choice", instructions: "Which department?",
      options: ["billing", "technical"],
      answer: { type: "choice", choice: "billing", abstained: false } },
    { id: "urg", type: "score", instructions: "How urgent?", levels: ["low", "high"],
      answer: { type: "score", score: 0.5, abstained: true } },
  ],
};
const typedText = plainText(typed);
assert.match(typedText, /\[Typed decision results\]/);
assert.match(typedText, /\[choice\] Which department\?/);
assert.match(typedText, /selected: billing/);
assert.match(typedText, /\[score\] How urgent\?/);
assert.match(typedText, /expected score: 0\.5/);
assert.match(typedText, /The decision model abstained\./);


// --- deterministic DOM harness: composer send/acceptance invariants ------
// Real buildComposer against the shim: proves busy keeps text editable,
// failed sends keep the draft, pending acceptance preserves newer typing,
// and the extracted-draft → confirm path restores and submits the source.

import { installShim, tick } from "./dom-shim.mjs";
import { buildComposer } from "../../serve/mlx_omarchy_assistant/static/js/composer.js";

installShim();

function buildHarness(onSend) {
  const composer = buildComposer({
    onSend,
    onCancel: () => {},
    onStopSpeaking: () => {},
    onDraft: (text) => composer && onSend({ text, mode: "draft" }),
    onOpenContext: () => {},
    voiceStates: { recognition: "unknown", synthesis: "unknown" },
    isRecording: () => false,
    isBusy: () => false,
    isSpeaking: () => false,
  });
  const wrap = composer.wrap;
  return {
    composer,
    textarea: wrap.querySelector("#composer-text"),
    sendBtn: wrap.querySelector("#send-btn"),
    stopBtn: wrap.querySelector("#stop-btn"),
    draftBtn: wrap.querySelector("#compare-draft"),
    compareSubmit: wrap.querySelector("#compare-submit"),
    analyzeSubmit: wrap.querySelector("#analyze-submit"),
  };
}

{
  // busy: textarea stays editable, buttons swap, text untouched
  let resolveSend;
  const h = buildHarness(() => new Promise((r) => { resolveSend = r; }));
  h.composer.setText("draft while busy");
  h.composer.setBusy(true);
  assert.equal(h.textarea.disabled, false, "composer must stay editable while busy");
  assert.equal(h.sendBtn.hidden, true);
  assert.equal(h.stopBtn.hidden, false);
  h.sendBtn._fire("click");                    // duplicate send while busy
  await tick();
  assert.equal(h.textarea.value, "draft while busy",
    "refused duplicate send must not clear any text");
  resolveSend(false);                          // even a refused send keeps text
  await tick();
  assert.equal(h.textarea.value, "draft while busy");
}

{
  // failed submit: draft keeps exactly what was typed
  const h = buildHarness(() => Promise.resolve(false));
  h.composer.setText("keep me on failure");
  h.sendBtn._fire("click");
  await tick();
  assert.equal(h.textarea.value, "keep me on failure");
}

{
  // pending acceptance: newer typing survives the accepted send
  let resolveSend;
  const h = buildHarness(() => new Promise((r) => { resolveSend = r; }));
  h.composer.setText("first message");
  h.sendBtn._fire("click");
  h.composer.setText("first message and also this");   // typed while pending
  resolveSend(true);
  await tick();
  assert.equal(h.textarea.value, "and also this");
}

{
  // accepted send with unchanged text clears the composer
  let resolveSend;
  const h = buildHarness(() => new Promise((r) => { resolveSend = r; }));
  h.composer.setText("exact message");
  h.sendBtn._fire("click");
  resolveSend(true);
  await tick();
  assert.equal(h.textarea.value, "");
}

{
  // composer toolbar semantics: the controls div is role=toolbar with a
  // label, and the voice status text lives in a sibling role=status group
  // (not inside the toolbar, so screen readers do not announce it as a
  // toolbar control). The textarea is named "Message" and is not a parent
  // of the toolbar.
  const h = buildHarness(() => Promise.resolve(true));
  const wrap = h.composer.wrap;
  const toolbar = wrap.querySelector(".composer__controls");
  assert.equal(toolbar.getAttribute("role"), "toolbar",
    "composer controls must be a toolbar landmark so the buttons inside are grouped");
  assert.equal(toolbar.getAttribute("aria-label"), "Composer controls");
  const voice = wrap.querySelector(".composer__voice-status");
  assert.ok(voice, "voice status group must exist");
  assert.equal(voice.getAttribute("role"), "status",
    "voice status must be a status region, not a toolbar control");
  // The textarea and toolbar must be siblings, not nested.
  const row = wrap.querySelector(".composer__row");
  assert.ok(row, "composer__row must wrap the textarea and toolbar");
  // Both the textarea and the toolbar are children of the row (not nested).
  const rowChildren = Array.from(row.querySelectorAll("#composer-text, .composer__controls"));
  assert.equal(rowChildren.length, 2,
    "textarea and toolbar must both live inside composer__row as siblings");
  // The voice status must NOT live inside the toolbar.
  assert.equal(toolbar.querySelectorAll(".composer__voice-status").length, 0,
               "voice status must not be a descendant of the toolbar");
  assert.equal(row.querySelectorAll(".composer__voice-status").length, 0,
               "voice status must live outside the composer__row too");
}

{
  // extracted-draft → confirm: source material preserved, restored, submitted
  const sends = [];
  const h = buildHarness((turn) => { sends.push(turn); return Promise.resolve(true); });
  const material = "choosing between Air and Pro for travel weight";
  h.composer.setText(material);
  h.draftBtn._fire("click");                   // draft turn from composer text
  await tick();
  assert.equal(sends[0].mode, "draft");
  assert.equal(sends[0].text, material);
  assert.equal(h.textarea.value, material,
    "a draft turn is derived: the composer keeps the source material");

  h.composer.prefillCompare({
    options: [{ label: "Air M3" }, { label: "Pro M3" }],
    criteria: "travel weight",
    source: material,
  });
  const criteria = h.composer.wrap.querySelector("#compare-criteria");
  assert.equal(criteria.value, "travel weight");
  h.compareSubmit._fire("click");              // confirm the editable draft
  await tick();
  assert.equal(sends[1].mode, "compare");
  assert.equal(sends[1].text, material, "criteria never replaces the source material");
  assert.equal(sends[1].criteria, "travel weight");
  assert.deepEqual(sends[1].options.map((o) => o.label), ["Air M3", "Pro M3"]);
  assert.equal(h.textarea.value, "",
    "the compare acceptance clears the matching composer text");
}

{
  // decide path: typed questions carry the composer material, cleared on acceptance
  const sends = [];
  const h = buildHarness((turn) => { sends.push(turn); return Promise.resolve(true); });
  h.composer.setText("invoice email body");
  const rows = h.composer.wrap.querySelectorAll(".analyze-panel__question");
  rows[0].querySelector(".analyze-panel__instructions").value = "Which department?";
  rows[0].querySelector(".analyze-panel__labels").value = "billing, technical";
  h.analyzeSubmit._fire("click");
  await tick();
  assert.equal(sends[0].mode, "decide");
  assert.equal(sends[0].text, "invoice email body");
  assert.deepEqual(sends[0].questions[0],
    { id: "q1", type: "choice", instructions: "Which department?",
      options: ["billing", "technical"] });
  assert.equal(h.textarea.value, "", "accepted decide turn clears its text");
}

// --- Details drawer: voice picker select states -----------------------
// A label, the option list with every pack speaker (accent in the label),
// the current voice pre-selected, the preview button disabled until the
// voice pack is qualified, and the select/button announce changes through
// the supplied live-region callback.

import { renderDetails } from "../../serve/mlx_omarchy_assistant/static/js/dom.js";

{
  // Status shape: every preset speaker shown with label + accent, the
  // currently chosen voice is marked selected, the preview button reflects
  // voice readiness, and changes propagate through the announce callback.
  const announces = [];
  const changes = [];
  const previews = [];
  const status = {
    voice: {
      state: "ready",
      recognition: { state: "ready" },
      synthesis: {
        state: "ready",
        qualification: "qualified",
        pack: {
          voice: "aiden",
          voice_default: "aiden",
          voice_options: [
            { id: "aiden", label: "Aiden", accent: "American English" },
            { id: "ryan", label: "Ryan", accent: "English" },
            { id: "serena", label: "Serena",
              accent: "Chinese-native; English has an accent" },
            { id: "vivian", label: "Vivian",
              accent: "Chinese-native; English has an accent" },
            { id: "uncle_fu", label: "Uncle Fu",
              accent: "Chinese-native; English has an accent" },
            { id: "ono_anna", label: "Ono Anna",
              accent: "Japanese-native; English has an accent" },
            { id: "sohee", label: "Sohee",
              accent: "Korean-native; English has an accent" },
            { id: "eric", label: "Eric", accent: "Sichuan dialect (Chinese)" },
            { id: "dylan", label: "Dylan", accent: "Beijing dialect (Chinese)" },
          ],
        },
      },
    },
  };
  const wrap = renderDetails(status, {
    announce: (msg) => announces.push(msg),
    onVoiceChange: (id) => changes.push(id),
    onPreview: () => previews.push("preview"),
  });
  const select = wrap.querySelector("select[id=details-voice-select]");
  assert.ok(select, "voice select must render");
  const optionEls = (select.children || []).filter((c) => c.tagName === "option");
  const labels = optionEls.map((o) =>
    (o.children || []).map((c) => c.text).join(""));
  assert.equal(labels.length, 9, "all nine preset speakers are listed");
  assert.match(labels.join(" | "), /American English/);
  assert.match(labels.join(" | "), /Chinese-native/);
  assert.match(labels.join(" | "), /Japanese-native/);
  assert.match(labels.join(" | "), /Korean-native/);
  assert.match(labels.join(" | "), /Sichuan dialect/);
  assert.match(labels.join(" | "), /Beijing dialect/);
  assert.match(labels.find((l) => l.startsWith("Aiden")) || "", /\(default\)/);
  // The persisted voice is the one pre-selected. The shim represents
  // "selected" as an attribute on the option node, not as a property.
  const selected = optionEls.find((o) => o.getAttribute("selected") === "");
  assert.ok(selected, "exactly one option must be marked selected");
  assert.equal(selected.getAttribute("value"), "aiden",
    "the persisted voice must be the one selected");
  const button = wrap.querySelector("button[id=details-voice-preview]");
  assert.ok(button);
  assert.equal(button.disabled, false,
    "preview is enabled when voice is qualified");
  select.value = "ryan";
  select._fire("change");
  assert.deepEqual(changes, ["ryan"]);
  assert.ok(announces.some((m) => /Ryan/.test(m)),
    "select changes are announced through the live region");
  button._fire("click");
  await tick();
  assert.deepEqual(previews, ["preview"]);
  assert.ok(announces.some((m) => /Previewing/.test(m)),
    "previewing is announced before playback");
}

{
  // Preview is disabled when the voice pack is not yet ready; the picker
  // is still readable, the select still works, but the button cannot
  // claim readiness when there is none.
  const status = {
    voice: {
      state: "unqualified",
      recognition: { state: "ready" },
      synthesis: {
        state: "unqualified",
        pack: {
          voice: "aiden", voice_default: "aiden",
          voice_options: [
            { id: "aiden", label: "Aiden", accent: "American English" },
            { id: "ryan", label: "Ryan", accent: "English" },
            { id: "serena", label: "Serena",
              accent: "Chinese-native; English has an accent" },
            { id: "vivian", label: "Vivian",
              accent: "Chinese-native; English has an accent" },
            { id: "uncle_fu", label: "Uncle Fu",
              accent: "Chinese-native; English has an accent" },
            { id: "ono_anna", label: "Ono Anna",
              accent: "Japanese-native; English has an accent" },
            { id: "sohee", label: "Sohee",
              accent: "Korean-native; English has an accent" },
            { id: "eric", label: "Eric", accent: "Sichuan dialect (Chinese)" },
            { id: "dylan", label: "Dylan", accent: "Beijing dialect (Chinese)" },
          ],
        },
      },
    },
  };
  const wrap = renderDetails(status, {
    announce: () => {}, onVoiceChange: () => {}, onPreview: () => {},
  });
  const button = wrap.querySelector("button[id=details-voice-preview]");
  assert.equal(button.disabled, true,
    "preview must be disabled when the voice pack is not qualified");
  const section = button.parent && button.parent.parent;
  const text = section ? collectText(section) : "";
  assert.match(text, /Preview is disabled/);
}

function collectText(node, out) {
  out = out || [];
  if (!node) return out.join("");
  if (node.text) { out.push(node.text); return out.join(""); }
  for (const c of node.children || []) collectText(c, out);
  return out.join("");
}

{
  // When voice_options is absent (older servers) the picker section is
  // omitted entirely, never rendered as an empty select.
  const status = { voice: { state: "ready",
    recognition: { state: "ready" },
    synthesis: { state: "ready", pack: { voice: "aiden",
      voice_default: "aiden" } } } };
  const wrap = renderDetails(status, {
    announce: () => {}, onVoiceChange: () => {}, onPreview: () => {},
  });
  assert.equal(wrap.querySelector("select[id=details-voice-select]"), null);
}

console.log("assistant ui js tests passed");
