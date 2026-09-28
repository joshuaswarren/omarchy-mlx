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

export function renderDetails(status) {
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
  }
  if (status.theme) {
    wrap.appendChild(el("section", { class: "detail-section" },
      el("h3", {}, "Theme"),
      dl([["Name", status.theme.name], ["Source", status.theme.source],
          ["Status", status.theme.status || ""]])));
  }
  return wrap;
}
