import { el } from "./dom.js";
import { isPlainObject } from "./util.js";

const chartInstances = new WeakMap();

const ACTIONS_BY_TYPE = {
  comparison: ["select", "sort", "filter", "edit"],
  chart: ["expand"],
  checklist: ["edit", "submit"],
  timeline: ["expand"],
  form: ["submit"],
  facts: ["expand"],
};

function button(label, opts = {}) {
  const node = el("button", { type: "button" }, label);
  if (opts.onClick) node.addEventListener("click", opts.onClick);
  if (opts.disabled) node.disabled = true;
  return node;
}

function asNumber(value) {
  if (Number.isFinite(value)) return value;
  return undefined;
}

function asNumberOrZero(value) {
  const n = asNumber(value);
  return n === undefined ? 0 : n;
}

function formatNumber(value) {
  if (value === null || value === undefined) return "";
  const n = asNumber(value);
  return n !== undefined ? n.toLocaleString() : String(value);
}

function renderDecision(component, _onAction) {
  const confidence = component.confidence || {};
  const wrap = el("section", { class: "card card--decision", dataset: { type: "decision" } });
  if (component.title) wrap.appendChild(el("h3", { class: "card__title" }, component.title));
  const list = el("ul", { class: "card__options" });
  for (const option of component.options) {
    const li = el("li", { dataset: { id: option.id } },
      option.label + (option.id === component.selected ? " — selected" : ""));
    if (option.id === component.selected) li.classList.add("card__selected");
    list.appendChild(li);
  }
  wrap.appendChild(list);
  if (confidence.abstained) {
    wrap.appendChild(el("p", { class: "card__abstained" },
      "The decision model abstained; treat the result as uncertain."));
  } else {
    const prob = asNumber(confidence.selected_probability);
    if (prob !== undefined) {
      wrap.appendChild(el("p", { class: "card__hint" },
        `Selected probability: ${(prob * 100).toFixed(0)}%`));
    }
  }
  wrap.appendChild(el("p", { class: "card__hint" }, `Criteria: ${component.criteria || ""}`));
  if (component.model || isPlainObject(component.probabilities)) {
    wrap.appendChild(el("div", { class: "card__actions" },
      button("How this was decided", { onClick: () => toggleHowDecided(wrap, component) })));
  }
  return wrap;
}

function toggleHowDecided(wrap, component) {
  const existing = wrap.querySelector(".card__how");
  if (existing) { existing.remove(); return; }
  const probabilities = isPlainObject(component.probabilities) ? component.probabilities : {};
  const rows = [["Model", component.model || ""],
    ["Input scope", component.input_scope || "the supplied material"]];
  for (const [id, p] of Object.entries(probabilities)) {
    const label = (component.options || []).find((o) => o.id === id)?.label || id;
    rows.push([`P(${label})`, asNumber(p) !== undefined ? (p * 100).toFixed(1) + "%" : String(p)]);
  }
  rows.push(["Abstained", component.confidence?.abstained ? "yes" : "no"]);
  wrap.appendChild(el("div", { class: "card__how" },
    rows.map(([k, v]) => el("div", {}, `${k}: ${v}`))));
}

function renderComparison(component, onAction) {
  const wrap = el("section", { class: "card card--comparison", dataset: { type: "comparison" } });
  if (component.title) wrap.appendChild(el("h3", { class: "card__title" }, component.title));
  const table = el("table", { class: "card__table" });
  const thead = el("thead", {}, el("tr", {},
    el("th", {}, "Option"),
    ...component.columns.map((c) => el("th", { dataset: { col: c.id } }, c.label))));
  const tbody = el("tbody");
  for (const row of component.rows) {
    const tr = el("tr", { dataset: { id: row.id } });
    tr.appendChild(el("td", {}, row.label));
    row.values.forEach((cell, idx) => {
      tr.appendChild(el("td", { dataset: { kind: component.columns[idx].kind } },
        cell === null || cell === undefined ? "" : formatNumber(cell)));
    });
    tbody.appendChild(tr);
  }
  table.appendChild(thead); table.appendChild(tbody);
  wrap.appendChild(table);
  const actions = el("div", { class: "card__actions card__sort" });
  if (ACTIONS_BY_TYPE.comparison.includes("sort")) {
    for (const col of component.columns) {
      actions.appendChild(button(`Sort by ${col.label}`, {
        onClick: () => onAction?.({ action: "sort",
                                              values: { column_id: col.id, direction: "asc" } }),
      }));
    }
  }
  if (ACTIONS_BY_TYPE.comparison.includes("filter")) {
    actions.appendChild(button("Filter rows", {
      onClick: () => {
        const query = (prompt("Filter rows by text") || "").slice(0, 100);
        if (query) { onAction?.({ action: "filter", values: { query } }); }
      },
    }));
  }
  wrap.appendChild(actions);
  return wrap;
}

function drawChart(canvas, component) {
  if (!Array.isArray(component.series) || component.series.length === 0) return;
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  const css = getComputedStyle(document.documentElement);
  const ink = css.getPropertyValue("--mlx-text").trim() || "#c0caf5";
  const dim = css.getPropertyValue("--mlx-dim").trim() || "#8a91ad";
  const border = css.getPropertyValue("--mlx-border").trim() || "#2f3447";
  const focus = css.getPropertyValue("--mlx-focus").trim() || "#7aa2f7";
  const palette = [focus, ink, dim, "#bb9af7"];
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = Math.max(1, Math.round(rect.width * dpr));
  canvas.height = Math.max(1, Math.round(rect.height * dpr));
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, rect.width, rect.height);
  ctx.font = "12px " + (css.getPropertyValue("--mlx-font-mono").trim() || "monospace");
  ctx.fillStyle = ink;
  const padding = { l: 36, r: 12, t: 12, b: 28 };
  const inner = { w: rect.width - padding.l - padding.r,
                   h: rect.height - padding.t - padding.b };
  const allValues = component.series.flatMap((s) => s.values.map((p) => asNumberOrZero(p.value)));
  const min = Math.min(0, ...allValues);
  const max = Math.max(0.0001, ...allValues);
  const range = max - min || 1;
  ctx.strokeStyle = border;
  ctx.beginPath();
  ctx.moveTo(padding.l, padding.t + inner.h);
  ctx.lineTo(padding.l + inner.w, padding.t + inner.h);
  ctx.stroke();
  ctx.fillStyle = dim;
  ctx.fillText(min.toLocaleString(), 0, padding.t + inner.h);
  ctx.fillText(max.toLocaleString(), 0, padding.t + 10);
  if (component.kind === "bar") {
    const groups = component.series[0].values.length;
    const groupWidth = inner.w / Math.max(1, groups);
    const seriesCount = component.series.length;
    component.series.forEach((series, sIdx) => {
      ctx.fillStyle = palette[sIdx % palette.length];
      series.values.forEach((point, idx) => {
        const x = padding.l + idx * groupWidth + sIdx * (groupWidth / seriesCount) + 2;
        const w = Math.max(2, groupWidth / seriesCount - 4);
        const ratio = (asNumberOrZero(point.value) - min) / range;
        const y = padding.t + inner.h * (1 - ratio);
        ctx.fillRect(x, y, w, padding.t + inner.h - y);
      });
    });
    ctx.fillStyle = dim;
    component.series[0].values.forEach((point, idx) => {
      if (idx % Math.ceil(component.series[0].values.length / 8) !== 0) return;
      const x = padding.l + idx * groupWidth + groupWidth / 2;
      ctx.fillText(point.label.slice(0, 12), x - 16, padding.t + inner.h + 16);
    });
  } else {
    const groups = component.series[0].values.length;
    const step = groups > 1 ? inner.w / (groups - 1) : inner.w;
    component.series.forEach((series, sIdx) => {
      ctx.strokeStyle = palette[sIdx % palette.length];
      ctx.lineWidth = 2;
      ctx.beginPath();
      series.values.forEach((point, idx) => {
        const x = padding.l + idx * step;
        const ratio = (asNumberOrZero(point.value) - min) / range;
        const y = padding.t + inner.h * (1 - ratio);
        if (idx === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      });
      ctx.stroke();
    });
  }
}

function renderChart(component, _onAction) {
  const wrap = el("section", { class: "card card--chart", dataset: { type: "chart" } });
  if (component.title) wrap.appendChild(el("h3", { class: "card__title" }, component.title));
  for (const series of Array.isArray(component.series) ? component.series : []) {
    const label = el("div", { class: "card__series-label",
                              dataset: { estimate: String(!!series.estimate) } },
                    `${series.label}${series.unit ? " (" + series.unit + ")" : ""}`);
    wrap.appendChild(label);
    if (series.source) wrap.appendChild(el("div", { class: "card__source" }, `Source: ${series.source}`));
  }
  const canvas = el("canvas", { width: "480", height: "180" });
  wrap.appendChild(canvas);
  chartInstances.set(canvas, component);
  queueMicrotask(() => drawChart(canvas, component));
  if (ACTIONS_BY_TYPE.chart.includes("expand")) {
    wrap.appendChild(el("div", { class: "card__actions" },
      button("Show data table", { onClick: () => toggleDataTable(wrap, component) })));
  }
  return wrap;
}

function toggleDataTable(wrap, component) {
  let existing = wrap.querySelector("table");
  if (existing) { existing.remove(); return; }
  const table = el("table", { class: "card__table" });
  const thead = el("thead", {}, el("tr", {},
    el("th", {}, "Series"),
    el("th", {}, "Label"),
    el("th", {}, "Value")));
  const tbody = el("tbody");
  for (const s of component.series) {
    for (const p of s.values) {
      tbody.appendChild(el("tr", {},
        el("td", {}, s.label),
        el("td", {}, p.label),
        el("td", {}, formatNumber(p.value))));
    }
  }
  table.appendChild(thead); table.appendChild(tbody);
  wrap.appendChild(table);
}

function renderChecklist(component, onAction) {
  const wrap = el("section", { class: "card card--checklist", dataset: { type: "checklist" } });
  if (component.title) wrap.appendChild(el("h3", { class: "card__title" }, component.title));
  const list = el("ul");
  for (const item of component.items) {
    const cb = el("input", { type: "checkbox", dataset: { id: item.id } });
    cb.checked = !!item.done;
    cb.id = `cl-${item.id}`;
    cb.addEventListener("change", () => { item.done = cb.checked; });
    const label = el("label", { for: cb.id }, item.text);
    list.appendChild(el("li", {}, cb, label));
  }
  wrap.appendChild(list);
  const actions = el("div", { class: "card__actions" });
  actions.appendChild(button("Send updated list", {
    onClick: () => onAction?.({
      action: "submit",
      values: { items: component.items.map((i) => ({ id: i.id, text: i.text, done: !!i.done })) },
    }),
  }));
  wrap.appendChild(actions);
  return wrap;
}

function renderTimeline(component, _onAction) {
  const wrap = el("section", { class: "card card--timeline", dataset: { type: "timeline" } });
  if (component.title) wrap.appendChild(el("h3", { class: "card__title" }, component.title));
  const list = el("ol");
  for (const entry of component.entries) {
    list.appendChild(el("li", {},
      entry.when ? el("span", { class: "card__when" }, entry.when) : null,
      el("span", {}, entry.text)));
  }
  wrap.appendChild(list);
  return wrap;
}

function renderForm(component, onAction) {
  const wrap = el("section", { class: "card card--form", dataset: { type: "form" } });
  if (component.title) wrap.appendChild(el("h3", { class: "card__title" }, component.title));
  const form = el("form", { class: "card__form" });
  const inputs = new Map();
  for (const field of component.fields) {
    const required = field.required ? el("span", { class: "required-mark", "aria-hidden": "true" }, " *") : null;
    let inputNode;
    if (field.kind === "text") {
      inputNode = el("input", { type: "text", id: `f-${field.id}`, name: field.id,
                                placeholder: field.placeholder || "" });
    } else if (field.kind === "numeric") {
      inputNode = el("input", { type: "number", id: `f-${field.id}`, name: field.id });
      if (field.min != null) inputNode.min = field.min;
      if (field.max != null) inputNode.max = field.max;
      if (field.step != null) inputNode.step = field.step;
    } else {
      inputNode = el("select", { id: `f-${field.id}`, name: field.id });
      inputNode.appendChild(el("option", { value: "" }, "(choose)"));
      for (const choice of field.choices) {
        inputNode.appendChild(el("option", { value: choice.id }, choice.label));
      }
    }
    if (field.initial !== undefined && field.initial !== null && field.initial !== "") {
      inputNode.value = String(field.initial);
    }
    inputs.set(field.id, inputNode);
    const label = el("label", { for: `f-${field.id}` }, field.label);
    if (required) label.appendChild(required);
    form.appendChild(el("div", { class: "form-field" }, label, inputNode));
  }
  const submitBtn = button("Submit answers", {
    onClick: () => {
      submitBtn.disabled = true;
      const values = {};
      for (const [fid, node] of inputs.entries()) {
        values[fid] = node.value;
      }
      if (onAction) onAction({ action: "submit", values: { values } });
    },
  });
  form.appendChild(el("div", { class: "card__actions" }, submitBtn));
  wrap.appendChild(form);
  return wrap;
}

function renderFacts(component, _onAction) {
  const wrap = el("section", { class: "card card--facts", dataset: { type: "facts" } });
  if (component.title) wrap.appendChild(el("h3", { class: "card__title" }, component.title));
  for (const card of component.cards) {
    const detail = el("div", { class: "fact-card" });
    detail.appendChild(el("h4", {}, card.heading));
    detail.appendChild(el("p", {}, card.text));
    if (card.sources) {
      const toggle = button(`Show sources (${card.sources.length})`, {
        onClick: () => {
          let existing = detail.querySelector(".source");
          if (existing) { existing.remove(); return; }
          const src = el("div", { class: "source" });
          for (const source of card.sources) {
            src.appendChild(el("div", {}, `[${source.ref}] ${source.text}`));
          }
          detail.appendChild(src);
        },
      });
      detail.appendChild(toggle);
    }
    wrap.appendChild(detail);
  }
  return wrap;
}

const RENDERERS = {
  decision: renderDecision,
  typed: renderTyped,
  comparison: renderComparison,
  chart: renderChart,
  checklist: renderChecklist,
  timeline: renderTimeline,
  form: renderForm,
  facts: renderFacts,
};

export function renderComponent(component, onAction) {
  if (!isPlainObject(component)) return null;
  if (component.type === "choice") component = normalizeLayaDecision(component) || component;
  const renderer = RENDERERS[component.type];
  if (!renderer) return null;
  try {
    return renderer(component, (payload) => {
      if (onAction) onAction(payload);
    });
  } catch {
    // Model-supplied payloads are untrusted input: a malformed or hostile
    // component renders nothing instead of breaking the whole conversation.
    return null;
  }
}

export function plainText(component) {
  if (!isPlainObject(component)) return "";
  try {
    switch (component.type) {
    case "decision": {
      const lines = [];
      if (component.title) lines.push(`[${component.title}]`);
      lines.push("[Decision card]");
      for (const o of component.options) {
        lines.push(`  ${o.id === component.selected ? "*" : " "} ${o.label}`);
      }
      if (component.confidence && component.confidence.abstained)
        lines.push("The decision model abstained.");
      if (component.criteria) lines.push(`Criteria: ${component.criteria}`);
      if (component.model) lines.push(`Model: ${component.model}`);
      return lines.join("\n");
    }
    case "choice": {
      const lines = ["[Decision]"];
      for (const o of component.options || []) {
        lines.push(`  ${o.id === component.choice ? "*" : " "} ${o.label}`);
      }
      if (component.abstained) lines.push("The decision model abstained.");
      if (component.criteria) lines.push(`Criteria: ${component.criteria}`);
      lines.push(`Model: ${component.model || ""}`);
      return lines.join("\n");
    }
    case "typed": {
      const lines = ["[Typed decision results]"];
      for (const result of component.results || []) {
        const answer = result.answer || {};
        lines.push(`${result.type === "score" ? "[score]" : "[choice]"} ${result.instructions}`);
        for (const label of result.options || result.levels || []) {
          lines.push(`  ${label}`);
        }
        if (result.type === "score") {
          lines.push(`  expected score: ${answer.score}`);
        } else {
          lines.push(`  selected: ${answer.choice}`);
        }
        if (answer.abstained) lines.push("  The decision model abstained.");
      }
      lines.push(`Model: ${component.model || ""}`);
      return lines.join("\n");
    }
    case "comparison": {
      const lines = ["[Comparison]"];
      lines.push(["Option", ...component.columns.map((c) => c.label)].join(" | "));
      for (const row of component.rows) {
        lines.push([
          row.label,
          ...row.values.map((v) => v == null ? "" : formatNumber(v)),
        ].join(" | "));
      }
      return lines.join("\n");
    }
    case "chart": {
      const lines = ["[Chart]"];
      for (const s of component.series) {
        lines.push(`${s.label}${s.estimate ? " (estimate)" : ""}: ` +
          s.values.map((p) => `${p.label}=${p.value}`).join(", "));
      }
      return lines.join("\n");
    }
    case "checklist":
      return component.items.map((i) => `[${i.done ? "x" : " "}] ${i.text}`).join("\n");
    case "timeline":
      return component.entries
        .map((e) => `${e.when ? e.when + ": " : ""}${e.text}`).join("\n");
    case "form":
      return component.fields.map((f) => `${f.label}${f.required ? " (required)" : ""}`).join("\n")
        + "\n(Submit sends your answers as a new message.)";
    case "facts":
      return component.cards.map((c) => `${c.heading}: ${c.text}`).join("\n");
    default:
      return "";
    }
  } catch {
    return "";
  }
}

export function redrawCharts() {
  // chartInstances is a WeakMap; iterating isn't necessary — drawChart
  // reads colors from getComputedStyle each call, so theme changes that
  // trigger a re-render of the page will repaint every chart.
}

// ---------------------------------------------------------------------------
// Coordinator decision payloads (compare + typed classify/score). These are
// not generative-UI components: the coordinator validates them against the
// Laya contract before emitting, and they render read-only with provenance.
// ---------------------------------------------------------------------------

export function normalizeLayaDecision(payload) {
  if (!isPlainObject(payload) || payload.type !== "choice" || !Array.isArray(payload.options)) {
    return null;
  }
  const probabilities = isPlainObject(payload.probabilities) ? payload.probabilities : {};
  return {
    type: "decision",
    options: payload.options,
    selected: payload.choice,
    criteria: payload.criteria || "",
    confidence: { selected_probability: asNumber(probabilities[payload.choice]),
                  abstained: !!payload.abstained },
    model: payload.model,
    input_scope: payload.input_scope,
    probabilities,
  };
}

function typedConfidenceNote(answer, selectedProbability) {
  if (answer.abstained) {
    return el("p", { class: "card__abstained" },
      "The decision model abstained; treat this result as uncertain.");
  }
  const prob = asNumber(selectedProbability);
  return prob !== undefined
    ? el("p", { class: "card__hint" }, "Selected probability: " + (prob * 100).toFixed(0) + "%")
    : null;
}

function renderChoiceResult(result) {
  const answer = result.answer || {};
  const card = el("div", { class: "card card--decision card--typed-result" });
  card.appendChild(el("h4", { class: "card__title" }, result.instructions));
  const list = el("ul", { class: "card__options" });
  for (const label of result.options || []) {
    const li = el("li", { dataset: { id: label } },
      label + (label === answer.choice ? " — selected" : ""));
    if (label === answer.choice) li.classList.add("card__selected");
    list.appendChild(li);
  }
  card.appendChild(list);
  const probabilities = isPlainObject(answer.probabilities) ? answer.probabilities : {};
  const note = typedConfidenceNote(answer, probabilities[answer.choice]);
  if (note) card.appendChild(note);
  return card;
}

function renderScoreResult(result) {
  const answer = result.answer || {};
  const card = el("div", { class: "card card--decision card--typed-result" });
  card.appendChild(el("h4", { class: "card__title" }, result.instructions));
  const levelCount = (result.levels || []).length;
  const scale = levelCount > 0 ? " on the 0-" + (levelCount - 1) + " scale" : "";
  card.appendChild(el("p", { class: "card__hint" },
    "Expected score: " + formatNumber(answer.score) + scale));
  const list = el("ul", { class: "card__options" });
  const probabilities = isPlainObject(answer.probabilities) ? answer.probabilities : {};
  (result.levels || []).forEach((level, index) => {
    const p = probabilities[String(index)];
    list.appendChild(el("li", {},
      level + (asNumber(p) !== undefined ? " — " + (p * 100).toFixed(0) + "%" : "")));
  });
  card.appendChild(list);
  const note = typedConfidenceNote(answer);
  if (note) card.appendChild(note);
  return card;
}

function renderTyped(component) {
  const wrap = el("section", { class: "card card--typed", dataset: { type: "typed" } });
  wrap.appendChild(el("h3", { class: "card__title" }, "Typed decision results"));
  if (component.model) {
    wrap.appendChild(el("p", { class: "card__hint" }, "Decision model: " + component.model));
  }
  for (const result of component.results || []) {
    wrap.appendChild(result.type === "score"
      ? renderScoreResult(result) : renderChoiceResult(result));
  }
  return wrap;
}
