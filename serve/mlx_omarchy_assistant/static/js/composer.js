import { el } from "./dom.js";
import { asString } from "./util.js";

const CONVERSATION_MODE_KEY = "mlx-chat.conversation-mode";

export function buildComposer({
  onSend, onCancel, onStopSpeaking, onMicStart, onMicStop, onMicCancel, onMicResult,
  voiceStates, onConversationModeChange, onCompareToggle, compareActive, isRecording,
  isBusy, isSpeaking, onDraft, onOpenContext,
}) {
  const wrap = el("footer", { class: "composer", role: "contentinfo" });
  const states = voiceStates || {};
  const recognitionState = states.recognition || "unknown";
  const synthesisState = states.synthesis || "unknown";
  const textarea = el("textarea", {
    id: "composer-text", class: "composer__textarea",
    placeholder: "Type a message. Enter sends, Shift+Enter adds a newline.",
    "aria-label": "Message",
    rows: "2",
  });
  let imeComposing = false;
  textarea.addEventListener("compositionstart", () => { imeComposing = true; });
  textarea.addEventListener("compositionend", () => { imeComposing = false; });
  textarea.addEventListener("keydown", (event) => {
    if (event.key === "Escape") {
      if (isRecording && isRecording()) { event.preventDefault(); if (onMicCancel) onMicCancel(); return; }
      if (isBusy && isBusy()) { event.preventDefault(); if (onCancel) onCancel(); return; }
    }
    if (event.key === "Enter" && !event.shiftKey && !imeComposing) {
      event.preventDefault();
      submit();
    }
  });
  const comparePanel = renderComparePanel({
    onCommit: async (data) => {
      if (await submitCompare(data)) closeComparePanel();
    },
    onClose: () => closeComparePanel(),
    onDraft: () => {
      const text = textarea.value.trim();
      if (onDraft && text) {
        void dispatch({ text, mode: "draft" }).then((ok) => {
          if (ok) closeComparePanel();
        });
      }
    },
  });
  const analyzePanel = renderAnalyzePanel({
    onClose: () => closeAnalyzePanel(),
    onSubmit: (questions) => {
      const text = textarea.value.trim();
      if (!text || !onSend) return;
      const max = allowance();
      onSend(max ? { text, mode: "decide", questions, max_tokens: max }
                 : { text, mode: "decide", questions });
      clear();
      closeAnalyzePanel();
    },
  });
  let compareOpen = !!compareActive;
  let analyzeOpen = false;

  const controls = el("div", { class: "composer__controls" });
  const compareBtn = el("button", { type: "button", class: "btn btn--ghost",
    "aria-pressed": compareOpen ? "true" : "false",
    id: "compare-btn",
  }, "Compare options");
  compareBtn.addEventListener("click", () => {
    if (compareOpen) closeComparePanel(); else openComparePanel();
    if (onCompareToggle) onCompareToggle(compareOpen);
  });

  const analyzeBtn = el("button", { type: "button", class: "btn btn--ghost",
    "aria-pressed": "false", id: "analyze-btn",
    "aria-controls": "analyze-panel",
  }, "Classify or score");
  analyzeBtn.addEventListener("click", () => {
    if (analyzeOpen) closeAnalyzePanel(); else openAnalyzePanel();
  });

  const contextBtn = el("button", { type: "button", class: "btn btn--ghost",
    id: "context-btn" }, "History & constraints");
  contextBtn.addEventListener("click", () => { if (onOpenContext) onOpenContext(); });

  const allowanceSelect = el("select", { id: "output-allowance", class: "composer__allowance",
    "aria-label": "Output allowance in tokens" },
    el("option", { value: "" }, "Answer length: automatic"),
    el("option", { value: "256" }, "Short (256 tokens)"),
    el("option", { value: "1024" }, "Standard (1024 tokens)"),
    el("option", { value: "4096" }, "Long (4096 tokens)"));

  const micBtn = el("button", { type: "button", class: "btn btn--ghost",
    id: "mic-btn",
  }, "Microphone");
  micBtn.disabled = recognitionState !== "ready";
  micBtn.title = voiceTitle(recognitionState);
  let latch = false;
  let pointerDownAt = 0;
  let pointerStartX = 0, pointerStartY = 0;
  micBtn.addEventListener("pointerdown", (event) => {
    if (micBtn.disabled) return;
    pointerDownAt = Date.now();
    pointerStartX = event.clientX; pointerStartY = event.clientY;
    micBtn.setPointerCapture(event.pointerId);
  });
  micBtn.addEventListener("pointerup", () => {
    if (micBtn.disabled) return;
    const dt = Date.now() - pointerDownAt;
    const dx = Math.abs(event.clientX - pointerStartX);
    const dy = Math.abs(event.clientY - pointerStartY);
    if (dt < 350 && dx < 8 && dy < 8) {
      latch = !latch;
      if (latch) startMic(); else stopMic();
    } else if (!latch) {
      stopMic();
    }
  });
  micBtn.addEventListener("pointercancel", () => stopMic());

  async function startMic() {
    if (micBtn.disabled) return;
    micBtn.textContent = "Listening…";
    micBtn.classList.add("recording-indicator");
    try {
      if (isSpeaking && isSpeaking()) {
        // Barge-in: stop playback first, then begin the explicit recording.
        if (onStopSpeaking) onStopSpeaking();
      }
      await onMicStart();
    }
    catch (err) {
      micBtn.textContent = "Microphone";
      micBtn.classList.remove("recording-indicator");
      latch = false;
      if (onMicResult) onMicResult({ error: err.message || "Microphone failed to start" });
    }
  }
  function stopMic() {
    micBtn.textContent = "Microphone";
    micBtn.classList.remove("recording-indicator");
    if (onMicStop) onMicStop();
  }

  const convToggle = el("label", { class: "setup__checkbox conv-toggle" },
    el("input", { type: "checkbox", id: "conv-mode" }),
    el("span", { class: "conv-toggle__label" }, "Conversation mode"),
  );
  const convCheckbox = convToggle.querySelector("input");
  if (localStorage.getItem(CONVERSATION_MODE_KEY) === "on") convCheckbox.checked = true;
  convCheckbox.addEventListener("change", () => {
    localStorage.setItem(CONVERSATION_MODE_KEY, convCheckbox.checked ? "on" : "off");
    if (onConversationModeChange) onConversationModeChange(convCheckbox.checked);
  });

  const sendBtn = el("button", { type: "button", class: "btn", id: "send-btn" }, "Send");
  sendBtn.addEventListener("click", submit);
  const stopBtn = el("button", { type: "button", class: "btn btn--ghost",
    id: "stop-btn", hidden: true }, "Stop response");
  stopBtn.addEventListener("click", () => { if (onCancel) onCancel(); });
  const stopSpeakBtn = el("button", { type: "button", class: "btn btn--ghost",
    id: "stop-speak-btn", hidden: true }, "Stop speaking");
  stopSpeakBtn.addEventListener("click", () => { if (onStopSpeaking) onStopSpeaking(); });

  controls.appendChild(compareBtn);
  controls.appendChild(analyzeBtn);
  controls.appendChild(micBtn);
  controls.appendChild(convToggle);
  controls.appendChild(contextBtn);
  controls.appendChild(allowanceSelect);
  controls.appendChild(el("div", { class: "spacer" }));
  controls.appendChild(sendBtn);
  controls.appendChild(stopBtn);
  controls.appendChild(stopSpeakBtn);

  const micStatus = el("span", { class: "voice-status",
    dataset: { state: recognitionState } }, `Dictation: ${voiceLabel(recognitionState)}`);
  const speakStatus = el("span", { class: "voice-status",
    dataset: { state: synthesisState } }, `Speech: ${voiceLabel(synthesisState)}`);
  controls.appendChild(micStatus);
  controls.appendChild(speakStatus);

  const row = el("div", { class: "composer__row" }, textarea, controls);
  wrap.appendChild(row);
  wrap.appendChild(comparePanel);
  comparePanel.hidden = !compareOpen;
  wrap.appendChild(analyzePanel);
  analyzePanel.hidden = !analyzeOpen;

  function openComparePanel() {
    compareOpen = true;
    comparePanel.hidden = false;
    compareBtn.setAttribute("aria-pressed", "true");
  }
  function closeComparePanel() {
    compareOpen = false;
    comparePanel.hidden = true;
    compareBtn.setAttribute("aria-pressed", "false");
  }
  function openAnalyzePanel() {
    analyzeOpen = true;
    analyzePanel.hidden = false;
    analyzeBtn.setAttribute("aria-pressed", "true");
  }
  function closeAnalyzePanel() {
    analyzeOpen = false;
    analyzePanel.hidden = true;
    analyzeBtn.setAttribute("aria-pressed", "false");
  }

  function currentText() { return textarea.value; }
  function setText(value) {
    textarea.value = value || "";
    textarea.focus();
  }
  function clear() { textarea.value = ""; }

  function allowance() {
    const parsed = Number.parseInt(allowanceSelect.value, 10);
    return Number.isFinite(parsed) && parsed > 0 ? parsed : undefined;
  }

  function acceptSent(text) {
    // Remove exactly the accepted message; anything typed meanwhile survives.
    const current = textarea.value;
    if (current.trim() === text) {
      textarea.value = "";
      return;
    }
    const index = current.indexOf(text);
    if (index >= 0) {
      textarea.value = (current.slice(0, index) + current.slice(index + text.length)).trim();
    }
  }

  async function dispatch(turn) {
    if (!onSend) return false;
    const accepted = await onSend(turn);
    // Acceptance-based clearing: the draft leaves the composer only after
    // the send was accepted; refusals and in-flight requests keep it. A
    // draft turn is derived from the material, so the composer keeps it.
    if (accepted && turn.mode !== "draft") acceptSent(turn.text);
    return accepted;
  }

  function submit() {
    const text = textarea.value.trim();
    if (!text) return;
    const max = allowance();
    void dispatch(max ? { text, max_tokens: max } : { text });
  }

  function submitCompare({ options, criteria }) {
    if (!options || options.length < 2) return Promise.resolve(false);
    // The supplied material decides the comparison: the panel's editable
    // source box when the draft restored it there, otherwise the composer
    // text. Criteria and options ride separately and never replace it.
    const panelMaterial = comparePanel.querySelector("#compare-material");
    const material = ((panelMaterial && panelMaterial.value) || textarea.value || "").trim();
    if (!material) {
      const submitError = comparePanel.querySelector("#compare-error");
      if (submitError) {
        submitError.hidden = false;
        submitError.textContent = "Add the text the decision should be made on before submitting.";
      }
      const materialField = comparePanel.querySelector("#compare-material");
      (materialField || textarea).focus();
      return Promise.resolve(false);
    }
    const submitError = comparePanel.querySelector("#compare-error");
    if (submitError) { submitError.hidden = true; submitError.textContent = ""; }
    return dispatch({
      text: material,
      mode: "compare",
      options: options.map((o, idx) => ({ id: o.id || `opt-${idx + 1}`, label: o.label })),
      criteria,
    });
  }

  function setBusy(busy) {
    // The composer stays editable at every stage; only the button swaps.
    sendBtn.hidden = !!busy;
    stopBtn.hidden = !busy;
  }

  function setSpeaking(active) {
    stopSpeakBtn.hidden = !active;
  }

  function setVoiceStates(next) {
    const states = next || {};
    const recognition = states.recognition || "unknown";
    const synthesis = states.synthesis || "unknown";
    micStatus.dataset.state = recognition;
    micStatus.textContent = `Dictation: ${voiceLabel(recognition)}`;
    speakStatus.dataset.state = synthesis;
    speakStatus.textContent = `Speech: ${voiceLabel(synthesis)}`;
    micBtn.disabled = recognition !== "ready";
    micBtn.title = voiceTitle(recognition);
  }

  function prefillCompare({ options, criteria: criteriaText, source }) {
    compareOpen = true;
    comparePanel.hidden = false;
    compareBtn.setAttribute("aria-pressed", "true");
    const list = comparePanel.querySelector("#compare-options");
    list.replaceChildren();
    const rows = Math.max(2, (options || []).length);
    for (let i = 0; i < rows; i++) {
      addOptionRow(list, (options && options[i] && options[i].label) || "");
    }
    const criteriaInput = comparePanel.querySelector("#compare-criteria");
    if (criteriaInput) criteriaInput.value = criteriaText || "";
    const material = comparePanel.querySelector("#compare-material");
    const sourceText = asString(source);
    if (material && sourceText !== undefined) material.value = sourceText;
    const first = list.querySelector("input[type=text]");
    if (first) first.focus();
  }

  return { wrap, setText, currentText, setBusy, setSpeaking, setVoiceStates,
           prefillCompare,
           focus: () => textarea.focus(), closeComparePanel };
}

function voiceTitle(state) {
  switch (state) {
    case "ready": return "Press and hold, or click to latch, to record";
    case "unqualified": return "Voice is not qualified on this machine";
    case "missing": return "Voice pack is missing";
    default: return "Voice status unknown";
  }
}

function voiceLabel(state) {
  switch (state) {
    case "ready": return "Voice ready";
    case "unqualified": return "Voice unqualified";
    case "missing": return "Voice pack missing";
    default: return "Voice unknown";
  }
}

function renderComparePanel({ onCommit, onDraft, onClose }) {
  const panel = el("section", { class: "compare-panel", "aria-label": "Compare options",
    id: "compare-panel" });
  panel.appendChild(el("header", { class: "compare-panel__header" },
    el("h4", {}, "Compare options"),
    el("button", { type: "button", class: "compare-panel__remove",
      id: "compare-panel-close",
      onClick: () => { if (onClose) onClose(); } },
      "Close")));

  const list = el("div", { id: "compare-options" });
  panel.appendChild(list);

  const addBtn = el("button", { type: "button", class: "btn btn--ghost",
    id: "add-option",
  }, "Add option");
  addBtn.addEventListener("click", () => addOptionRow(list));
  panel.appendChild(addBtn);

  const draftBtn = el("button", { type: "button", class: "btn btn--ghost",
    id: "compare-draft" }, "Draft options from my message");
  draftBtn.addEventListener("click", () => { if (onDraft) onDraft(); });
  panel.appendChild(draftBtn);

  panel.appendChild(el("label", { for: "compare-material",
    class: "compare-panel__criteria-label" }, "Your supplied material"));
  const materialInput = el("textarea", { id: "compare-material", rows: "2",
    placeholder: "The text the decision should be made on (restored from your draft)",
    "aria-label": "Supplied material" });
  panel.appendChild(materialInput);

  panel.appendChild(el("label", { for: "compare-criteria",
    class: "compare-panel__criteria-label" }, "Criteria"));
  const criteria = el("textarea", { id: "compare-criteria", rows: "2",
    placeholder: "What should the decision optimise for?",
    "aria-label": "Criteria" });
  panel.appendChild(criteria);

  const commit = el("div", { class: "card__actions" });
  commit.appendChild(el("button", { type: "button", class: "btn", id: "compare-submit" }, "Submit comparison"));
  panel.appendChild(commit);

  addOptionRow(list, "Option A");
  addOptionRow(list, "Option B");

  const submitError = el("p", { class: "message__error", id: "compare-error",
    role: "alert", hidden: true });
  panel.appendChild(submitError);

  panel.querySelector("#compare-submit").addEventListener("click", () => {
    const rows = list.querySelectorAll(".compare-panel__option");
    const options = [];
    rows.forEach((row) => {
      const input = row.querySelector("input[type=text]");
      const label = (input?.value || "").trim();
      if (label) options.push({ label });
    });
    if (options.length < 2) {
      submitError.hidden = false;
      submitError.textContent = "A comparison needs at least two named options.";
      const first = list.querySelector("input[type=text]");
      if (first) first.focus();
      return;
    }
    submitError.hidden = true;
    submitError.textContent = "";
    onCommit({ options, criteria: criteria.value });
  });

  return panel;
}

function addOptionRow(list, seedLabel = "") {
  if (list.querySelectorAll(".compare-panel__option").length >= 8) return;
  const row = el("div", { class: "compare-panel__option" });
  const idx = list.querySelectorAll(".compare-panel__option").length + 1;
  const input = el("input", { type: "text", placeholder: "Describe an option", value: seedLabel,
    "aria-label": `Option ${idx}` });
  const remove = el("button", { type: "button", class: "compare-panel__remove",
    "aria-label": `Remove option ${idx}` }, "Remove");
  remove.addEventListener("click", () => { row.remove(); });
  row.appendChild(input); row.appendChild(remove);
  list.appendChild(row);
}

export function parseAnalyzeQuestions(rows) {
  // Client-side shape check only; the coordinator re-validates everything
  // against the decision model's real tokenizer before dispatch.
  if (!Array.isArray(rows) || rows.length < 1 || rows.length > 8) {
    throw new Error("Send between one and eight questions");
  }
  return rows.map((row, index) => {
    const n = index + 1;
    const instructions = String((row && row.instructions) || "").trim();
    const kind = row && row.type === "score" ? "score" : "choice";
    const labels = String((row && row.labels) || "")
      .split(",").map((label) => label.trim()).filter(Boolean);
    if (!instructions) throw new Error("Question " + n + " needs wording");
    if (labels.length < 2 || labels.length > 8) {
      throw new Error("Question " + n + " needs 2-8 comma-separated " +
        (kind === "score" ? "levels" : "labels"));
    }
    const question = { id: "q" + n, type: kind, instructions };
    question[kind === "score" ? "levels" : "options"] = labels;
    return question;
  });
}

function renderAnalyzePanel({ onClose, onSubmit }) {
  const panel = el("section", { class: "compare-panel analyze-panel",
    "aria-label": "Classify or score supplied material", id: "analyze-panel" });
  panel.appendChild(el("header", { class: "compare-panel__header" },
    el("h4", {}, "Classify or score"),
    el("button", { type: "button", class: "compare-panel__remove",
      onClick: () => { if (onClose) onClose(); } }, "Close")));
  panel.appendChild(el("p", { class: "setup__hint" },
    "Runs the local decision model on your message. Write the message first, then add questions."));

  const list = el("div", { id: "analyze-questions" });
  panel.appendChild(list);
  const addRow = () => {
    if (list.children.length >= 8) return;
    const row = el("div", { class: "analyze-panel__question" });
    const wording = el("input", { type: "text", class: "analyze-panel__instructions",
      placeholder: "Question, e.g. Which department is this?", "aria-label": "Question" });
    const kind = el("select", { class: "analyze-panel__type", "aria-label": "Question type" },
      el("option", { value: "choice" }, "Classification"),
      el("option", { value: "score" }, "Score scale"));
    const labels = el("input", { type: "text", class: "analyze-panel__labels",
      placeholder: "Labels, comma-separated (2-8)", "aria-label": "Labels or score levels" });
    const remove = el("button", { type: "button", class: "compare-panel__remove",
      "aria-label": "Remove question" }, "Remove");
    remove.addEventListener("click", () => { if (list.children.length > 1) row.remove(); });
    row.append(wording, kind, labels, remove);
    list.appendChild(row);
  };
  addRow();
  const addBtn = el("button", { type: "button", class: "btn btn--ghost",
    id: "analyze-add" }, "Add question");
  addBtn.addEventListener("click", addRow);
  panel.appendChild(addBtn);

  const submit = el("button", { type: "button", class: "btn", id: "analyze-submit" },
    "Run typed questions");
  submit.addEventListener("click", () => {
    const rows = [];
    list.querySelectorAll(".analyze-panel__question").forEach((row) => {
      rows.push({
        instructions: row.querySelector(".analyze-panel__instructions").value,
        type: row.querySelector(".analyze-panel__type").value,
        labels: row.querySelector(".analyze-panel__labels").value,
      });
    });
    panel.querySelectorAll(".message__error").forEach((node) => node.remove());
    let questions;
    try { questions = parseAnalyzeQuestions(rows); }
    catch (err) {
      panel.appendChild(el("p", { class: "message__error", role: "alert" }, err.message));
      return;
    }
    onSubmit(questions);
  });
  panel.appendChild(el("div", { class: "card__actions" }, submit));
  return panel;
}
