import { isFunctionValue, asString } from "./util.js";

function isNonFalsy(v) { return v !== null && v !== undefined && v !== false; }

export function el(tag, attrs, ...children) {
  const node = document.createElement(tag);
  for (const key in attrs || {}) {
    const value = attrs[key];
    if (!isNonFalsy(value)) continue;
    if (key === "class") node.className = String(value);
    else if (key === "dataset") Object.assign(node.dataset, value);
    else if (key.startsWith("on") && isFunctionValue(value)) {
      node.addEventListener(key.slice(2).toLowerCase(), value);
    } else if (value === true) node.setAttribute(key, "");
    else node.setAttribute(key, String(value));
  }
  for (const child of children.flat()) {
    if (!isNonFalsy(child)) continue;
    const asText = asString(child);
    node.appendChild(asText !== undefined
      ? document.createTextNode(asText) : child);
  }
  return node;
}

export function announce(live, text) {
  if (!live) return;
  live.textContent = "";
  if (text) live.textContent = text;
}

export function dl(rows) {
  const list = el("dl");
  for (const pair of rows) {
    const k = pair[0], v = pair[1];
    if (!isNonFalsy(v)) continue;
    list.appendChild(el("dt", {}, k));
    list.appendChild(el("dd", {}, String(v)));
  }
  return list;
}

export function renderDetails(status, controls) {
  const wrap = el("div", { class: "drawer__body" });
  if (!status) {
    wrap.appendChild(el("p", { class: "drawer__empty" }, "No status available."));
    return wrap;
  }
  const pair = status.active_pair;
  if (pair) {
    wrap.appendChild(el("section", { class: "detail-section" },
      el("h3", {}, "Active pair"),
      dl([
        ["Pair", pair.id], ["Chat model", pair.chat_model],
        ["Context tokens", pair.context_tokens],
        ["Memory", pair.memory_note || ""],
        ["Ready offline", pair.ready_offline ? "yes" : "no"],
      ])));
  }
  if (status.context) {
    wrap.appendChild(el("section", { class: "detail-section" },
      el("h3", {}, "Context usage"),
      dl([
        ["Limit", status.context.limit_tokens ? `${status.context.limit_tokens.toLocaleString()} tokens` : ""],
        ["Used", status.context.used_tokens ? `${status.context.used_tokens.toLocaleString()} tokens` : ""],
        ["Reason", status.context.reason || ""],
      ])));
  }
  if (status.voice) {
    const voice = status.voice;
    const recognition = voice.recognition || {};
    const synthesis = voice.synthesis || {};
    wrap.appendChild(el("section", { class: "detail-section" },
      el("h3", {}, "Voice"),
      dl([
        ["Dictation", recognition.state || voice.state || ""],
        ["Dictation detail", recognition.detail || ""],
        ["Speech", synthesis.state || ""],
        ["Speech qualification", synthesis.qualification || synthesis.detail || ""],
      ])));
    const pack = synthesis.pack || {};
    const options = Array.isArray(pack.voice_options) ? pack.voice_options : [];
    if (options.length > 0) {
      const section = el("section", { class: "detail-section",
                                       "aria-labelledby": "details-voice-title" },
        el("h3", { id: "details-voice-title" }, "Voice picker"),
        el("p", { class: "setup__hint" },
          "Pick which engine's preset speaker reads replies aloud. "
          + "Qwen3-TTS speakers without an American English accent read "
          + "English with their native accent; Kokoro voices are American "
          + "English."));
      const fieldId = "details-voice-select";
      const select = el("select", {
        id: fieldId, name: "voice", class: "field-input",
        "aria-label": "Reply voice",
      });
      const current = pack.voice || pack.voice_default || options[0].id;
      const optionEls = [];
      const usableEngines = new Map(((synthesis.engines) || [])
        .map((engine) => [engine.id, engine.usable !== false]));
      let group = null;
      let groupKey = null;
      for (const opt of options) {
        const key = opt.engine_label || opt.engine || "";
        if (key !== groupKey) {
          groupKey = key;
          group = el("optgroup", { label: key });
          if (opt.engine && usableEngines.get(opt.engine) === false) {
            group.setAttribute("disabled", "");
            group.setAttribute("label", `${key} (not downloaded)`);
          }
          select.appendChild(group);
        }
        const isDefault = opt.id === pack.voice_default;
        const tag = isDefault ? " (default)" : "";
        const option = el("option", { value: opt.id },
          `${opt.label} - ${opt.accent}${tag}`);
        if (opt.id === current) option.setAttribute("selected", "");
        group.appendChild(option);
        optionEls.push(option);
      }
      const label = el("label", { for: fieldId },
        "Reply voice", select);
      const voiceReady = (voice.state === "ready");
      const previewButton = el("button", {
        type: "button", class: "message__action", id: "details-voice-preview",
        "aria-describedby": "details-voice-hint",
      }, "Preview");
      const hint = el("p", { class: "setup__hint", id: "details-voice-hint" },
        voiceReady
          ? "Preview speaks one fixed sentence in the selected voice."
          : "Preview is disabled until the voice pack is qualified.");
      previewButton.disabled = !voiceReady;
      let previewInFlight = false;
      const announce = (controls && controls.announce) || (() => {});
      const previewNow = (controls && controls.onPreview) || (() => {});
      const changeVoice = (controls && controls.onVoiceChange) || (() => {});
      select.addEventListener("change", () => {
        const id = select.value;
        const opt = options.find((o) => o.id === id);
        changeVoice(id, opt);
        announce(`Voice set to ${opt ? opt.label : id}.`);
      });
      previewButton.addEventListener("click", () => {
        if (previewButton.disabled || previewInFlight) return;
        previewInFlight = true;
        previewButton.disabled = true;
        const selected = optionEls.find(
            (o) => o.getAttribute("value") === select.value)
            || optionEls.find((o) => o.getAttribute("selected") === "");
        const label = (selected
          && ((selected.children || []).map((c) => c.text).join("")
              || selected.textContent)) || select.value;
        announce(`Previewing ${label}.`);
        Promise.resolve(previewNow(select.value))
          .catch((err) => announce(`Preview failed: ${err && err.message
              ? err.message : "unknown error"}`))
          .finally(() => {
            previewInFlight = false;
            previewButton.disabled = !voiceReady;
          });
      });
      section.appendChild(el("div", { class: "detail-section__row" },
        label, previewButton));
      section.appendChild(hint);
      wrap.appendChild(section);
    }
  }
  if (status.theme) {
    wrap.appendChild(el("section", { class: "detail-section" },
      el("h3", {}, "Theme"),
      dl([["Name", status.theme.name], ["Source", status.theme.source],
          ["Status", status.theme.status || ""]])));
  }
  return wrap;
}
