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

console.log("assistant ui js tests passed");
