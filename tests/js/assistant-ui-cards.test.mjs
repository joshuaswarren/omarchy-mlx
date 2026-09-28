// Regression tests for discovered UI defects. Run directly:
//   bun tests/js/assistant-ui-cards.test.mjs   (or: node)
// Exit code 0 = all assertions passed; any failure throws.

import assert from "node:assert/strict";

import { installShim, tick } from "./dom-shim.mjs";
import { renderComponent, plainText }
  from "../../serve/mlx_omarchy_assistant/static/js/genui.js";
import { buildComposer }
  from "../../serve/mlx_omarchy_assistant/static/js/composer.js";

installShim();

function textOf(node) {
  if (!node) return "";
  if (node.text !== undefined) return node.text;
  return (node.children || []).map(textOf).join("");
}

// --- typed cards must render without answer probabilities ---------------
// Regression: typedConfidenceNote returned null and appendChild(null) threw,
// aborting the whole typed card after the first probability-less result.

{
  const choiceNoProb = {
    type: "typed", model: "fixture", results: [{
      id: "dept", type: "choice", instructions: "Which department?",
      options: ["billing", "technical"],
      answer: { type: "choice", choice: "billing", abstained: false },
    }],
  };
  const card = renderComponent(choiceNoProb, () => {});
  assert.ok(card, "choice result without probabilities must still render");
  const resultCard = card.querySelector(".card--typed-result");
  assert.ok(resultCard, "typed wrapper contains the choice result card");
  assert.match(textOf(resultCard.querySelector(".card__title")), /Which department\?/);
  assert.equal(textOf(resultCard).includes("billing — selected"), true);

  const scoreNoProb = {
    type: "typed", model: "fixture", results: [{
      id: "urg", type: "score", instructions: "How urgent?",
      levels: ["low", "high"],
      answer: { type: "score", score: 0.5, abstained: false },
    }],
  };
  const scoreCard = renderComponent(scoreNoProb, () => {});
  assert.ok(scoreCard, "score result without probabilities must still render");

  const scoreNoLevels = {
    type: "typed", model: "fixture", results: [{
      id: "x", type: "score", instructions: "Rate",
      answer: { type: "score", score: 1, abstained: true },
    }],
  };
  const bare = renderComponent(scoreNoLevels, () => {});
  assert.ok(bare, "score result without levels must still render");
  const bareResult = bare.querySelector(".card--typed-result");
  const hint = textOf(bareResult.querySelector(".card__hint"));
  assert.equal(hint.includes("0--1"), false, "must not print a negative scale");
  assert.match(hint, /Expected score: 1/);
  assert.match(textOf(bareResult), /abstained/i);
}

// --- malformed / injected components render inert, never throw ----------

const malformed = [
  ["non-object string", "delete all data"],
  ["non-object number", 42],
  ["unknown type", { type: "exfiltrate", options: [{ id: "a", label: "x" }] }],
  ["comparison without columns", { type: "comparison", rows: [{ id: "r", label: "R", values: [1] }] }],
  ["decision without options", { type: "decision", selected: "a" }],
  ["chart with numeric series", { type: "chart", series: 7 }],
  ["chart with string series", { type: "chart", series: "<img src=x onerror=window.__pwned=1>" }],
  ["checklist without items", { type: "checklist" }],
  ["timeline without entries", { type: "timeline" }],
  ["form without fields", { type: "form" }],
  ["facts without cards", { type: "facts" }],
  ["typed without results", { type: "typed" }],
  ["choice payload without options", { type: "choice", choice: "a" }],
];

for (const [name, payload] of malformed) {
  assert.doesNotThrow(() => renderComponent(payload, () => {}),
    `${name} must render inert, not throw`);
}

{
  const hostile = {
    type: "chart",
    series: [{ label: "<img src=x onerror=window.__pwned=1>", unit: "",
               values: [{ label: "a", value: 1 }] }],
  };
  const card = renderComponent(hostile, () => {});
  assert.ok(card, "hostile series label must render as inert text");
}

// --- plainText survives the same malformed payloads ---------------------

assert.equal(plainText("x"), "");
for (const [name, payload] of malformed) {
  assert.doesNotThrow(() => plainText(payload), `plainText(${name}) must not throw`);
}
assert.equal(plainText({ type: "comparison", rows: [] }), "");

// --- compare/analyze panel Close must keep the toolbar toggle working ---
// Regression: the panel's own Close button cleared hidden without syncing
// the toolbar state, so the Compare/Classify button could never reopen it.

function buildHarness(extra = {}) {
  return buildComposer({
    onSend: () => Promise.resolve(true),
    onCancel: () => {},
    onStopSpeaking: () => {},
    onDraft: () => {},
    onOpenContext: () => {},
    voiceStates: { recognition: "unknown", synthesis: "unknown" },
    isRecording: () => false,
    isBusy: () => false,
    isSpeaking: () => false,
    ...extra,
  });
}

{
  const composer = buildHarness();
  const wrap = composer.wrap;
  const compareBtn = wrap.querySelector("#compare-btn");
  const panel = wrap.querySelector("#compare-panel");
  compareBtn._fire("click");
  assert.equal(panel.hidden, false, "Compare button opens the panel");
  wrap.querySelector("#compare-panel-close")._fire("click");
  assert.equal(panel.hidden, true, "Panel Close closes it");
  compareBtn._fire("click");
  assert.equal(panel.hidden, false, "Compare button must reopen after panel Close");
  assert.equal(compareBtn.attributes["aria-pressed"], "true");
  composer.closeComparePanel();
  assert.equal(panel.hidden, true);
}

{
  const composer = buildHarness();
  const wrap = composer.wrap;
  const analyzeBtn = wrap.querySelector("#analyze-btn");
  const panel = wrap.querySelector("#analyze-panel");
  analyzeBtn._fire("click");
  assert.equal(panel.hidden, false, "Classify button opens the panel");
  panel.querySelectorAll(".compare-panel__remove")[0]._fire("click");
  assert.equal(panel.hidden, true, "Panel Close closes it");
  analyzeBtn._fire("click");
  assert.equal(panel.hidden, false, "Classify button must reopen after panel Close");
}

{
  // Analyze panel validation error stays dismissible and reportable.
  const composer = buildHarness();
  const wrap = composer.wrap;
  wrap.querySelector("#analyze-btn")._fire("click");
  const analyzePanelNode = wrap.querySelector("#analyze-panel");
  wrap.querySelector("#analyze-submit")._fire("click");   // no questions filled
  const error = analyzePanelNode.querySelector(".message__error");
  assert.ok(error, "invalid analyze rows surface an inline error");
  assert.match(textOf(error), /needs wording/);
}

await tick();
console.log("assistant ui card regression tests passed");
