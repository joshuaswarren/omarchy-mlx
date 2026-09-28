import { el } from "./dom.js";
import { asNumber } from "./util.js";

const PREFERENCE_HINTS = {
  "fast": "Prefer lower response latency",
  "balanced": "Balance response latency with answer quality",
  "long-context": "Allow larger context at the cost of prefill time",
};

const VOICE_DETAIL_HINTS = {
  ready:        "Voice pack is installed and qualified on this machine.",
  unqualified:  "Voice is not qualified on this machine (no qualifying accelerator). You can still install the pack; replies will refuse to play until the runtime is ready.",
  missing:      "Voice pack is not installed locally. Approve the download below to add it.",
};

function setLiveStatus(statusEl, message) {
  if (statusEl) statusEl.textContent = message || "";
}

function voiceCheckboxAttrs(state) {
  const attrs = { type: "checkbox", name: "voice", id: "setup-voice" };
  if (state === "ready") attrs.checked = true;
  return attrs;
}

export function renderSetup(mount, status, { onSubmit, onCancel }) {
  mount.replaceChildren();
  const container = el("section", { class: "setup", role: "region",
                                    "aria-labelledby": "setup-title" });
  container.appendChild(el("div", { class: "setup__intro" },
    el("h2", { id: "setup-title" }, "Set up local chat"),
    el("p", {}, "Choose a model pair, a working preference, and confirm any downloads."),
  ));

  const liveStatus = el("div", { class: "visually-hidden", role: "status", "aria-live": "polite" });
  container.appendChild(liveStatus);

  const card = el("form", { class: "setup__card", id: "setup-form" });
  card.addEventListener("submit", (event) => {
    event.preventDefault();
    const data = new FormData(card);
    const preference = String(data.get("preference") || "balanced");
    const pairId = String(data.get("pair") || "");
    const voice = data.get("voice") === "on";
    const contextRaw = String(data.get("context") || "").trim();
    const parsed = contextRaw ? Number.parseInt(contextRaw, 10) : NaN;
    const contextTokens = Number.isFinite(parsed) && parsed > 0 ? parsed : undefined;
    const approveDownload = data.get("approve") === "on";
    const payload = { pair_id: pairId, approve_download: approveDownload,
                      preference, voice };
    if (contextTokens !== undefined)
      payload.context_tokens = contextTokens;
    if (onSubmit) onSubmit(payload);
  });

  const pairs = (status && Array.isArray(status.pairs)) ? status.pairs : [];
  const sectionPair = el("section", { class: "setup__section" },
    el("h3", {}, "Model pair"),
    el("p", { class: "setup__hint" },
      pairs.length === 0
        ? "No model pairs are present in the local catalog. Repair the installation before setup."
        : "Choose a pair. Untested pairs cannot start until hardware qualification passes."),
  );
    if (pairs.length > 0) {
    const list = el("div", { class: "setup__pair-list" });
    for (const pair of pairs) {
      const id = `pair-${pair.id}`;
      const checked = pair.recommended || pair.id === (status && status.recommended_pair);
      const inputAttrs = { type: "radio", name: "pair", value: pair.id, id };
      if (checked) inputAttrs.checked = true;
      const input = el("input", inputAttrs);
      const qualification = pair.qualification?.status || "unknown";
      const meta = [
        `Chat: ${pair.chat_model}`,
        `Decisions: ${pair.decision_model}`,
        `Qualification: ${qualification}`,
      ];
      list.appendChild(el("label", { class: "setup__pair", for: id },
        el("h4", {}, pair.label || pair.id),
        el("p", {}, meta.join(" · ")),
        input));
    }
    sectionPair.appendChild(list);
    if (status.recommendation_error) {
      sectionPair.appendChild(el("p", { class: "setup__hint" }, status.recommendation_error));
    }
  }
  card.appendChild(sectionPair);

  const sectionPref = el("section", { class: "setup__section" },
    el("h3", {}, "Working preference"));
  const prefWrap = el("div", { class: "setup__preference" });
  for (const pref of ["fast", "balanced", "long-context"]) {
    const id = `pref-${pref}`;
    const inputAttrs = { type: "radio", name: "preference", value: pref, id };
    if (pref === "balanced") inputAttrs.checked = true;
    const labelText = pref === "balanced" ? "Balanced" : (pref === "fast" ? "Fast" : "Long context");
    prefWrap.appendChild(el("label", { for: id },
      el("input", inputAttrs),
      labelText,
      el("small", {}, PREFERENCE_HINTS[pref]),
    ));
  }
  sectionPref.appendChild(prefWrap);
  card.appendChild(sectionPref);

  const voice = (status && status.voice) || {};
  const synthesis = voice.synthesis || {};
  const voiceState = synthesis.state || voice.state || "missing";
  const sectionVoice = el("section", { class: "setup__section" },
    el("h3", {}, "Voice replies"),
    el("p", { class: "setup__hint" },
      synthesis.detail || voice.detail || VOICE_DETAIL_HINTS[voiceState] || "Voice status is unknown."),
    el("label", { class: "setup__checkbox" },
      el("input", voiceCheckboxAttrs(voiceState)),
      el("div", {},
        el("strong", {}, "Download the voice pack with this setup"),
        el("p", { class: "setup__hint" },
          voiceState === "ready"
            ? "Voice pack is already installed; downloading is a no-op."
            : voiceState === "unqualified"
              ? "Install the pack anyway. Spoken replies will stay disabled until the accelerator is qualified."
              : "Required for spoken replies; the chat path is unaffected."))),
    voiceState !== "ready" && asNumber(voice.download_bytes) !== undefined
      ? el("p", { class: "setup__hint" },
          `Voice pack size: ${formatBytes(voice.download_bytes)}.`)
      : null);
  card.appendChild(sectionVoice);

  const contextSpec = (status && status.context) || {};
  const contextMax = asNumber(contextSpec.max_tokens);
  const contextStepRaw = asNumber(contextSpec.step_tokens);
  const contextStep = contextStepRaw !== undefined && contextStepRaw > 0 ? contextStepRaw : 1;
  const contextAttrs = { type: "number", name: "context", id: "setup-context", class: "field-input",
                          "aria-label": "Context cap in tokens",
                          inputmode: "numeric", step: String(contextStep),
                          placeholder: contextMax ? `${contextMax.toLocaleString()} (max)` : "Admitted maximum" };
  if (contextMax !== undefined) {
    contextAttrs.min = String(contextStep);
    contextAttrs.max = String(contextMax);
  }
  const contextInput = el("input", contextAttrs);
  const sectionContext = el("section", { class: "setup__section" },
    el("h3", {}, "Context cap (optional)"),
    el("p", { class: "setup__hint" },
      contextMax !== undefined
        ? `Admitted maximum: ${contextMax.toLocaleString()} tokens. Step is ${contextStep.toLocaleString()} to match the backend allocator. Leave blank to use the maximum.`
        : "Backend has not reported an admitted maximum yet; leave blank or type a positive integer."),
    contextInput);
  card.appendChild(sectionContext);

  const sectionDownload = el("section", { class: "setup__section" },
    el("h3", {}, "Download"));
  const componentRows = (status && Array.isArray(status.download_components))
    ? status.download_components : [];
  if (componentRows.length > 0) {
    const table = el("ul", { class: "transfer-components" });
    for (const comp of componentRows) {
      const bytes = asNumber(comp.bytes);
      const size = bytes !== undefined ? formatBytes(bytes)
                 : (comp.state === "cached" ? "cached locally" : "size unknown — coordinator did not report");
      table.appendChild(el("li", {},
        el("strong", {}, comp.label || comp.id),
        el("span", { class: "setup__hint" }, ` · ${size}`)));
    }
    sectionDownload.appendChild(table);
  }
  const totalKnown = asNumber(status && status.total_download_bytes);
  const hasUnknown = !!(status && status.download_has_unknown === true);
  const hasMeasuredTotal = totalKnown !== undefined && totalKnown > 0;
  if (hasMeasuredTotal || hasUnknown) {
    // A zero or absent total with unknown components means "not measured",
    // never "nothing to download"; the label must not show a made-up 0 B.
    const summaryParts = [];
    if (hasMeasuredTotal) summaryParts.push(`${formatBytes(totalKnown)} of measured components`);
    if (hasUnknown) summaryParts.push("components whose size has not been reported yet");
    sectionDownload.appendChild(el("label", { class: "setup__checkbox" },
      el("input", { type: "checkbox", name: "approve", id: "setup-approve",
                    required: true }),
      el("div", {},
        el("strong", {}, `Allow downloading ${summaryParts.join(" plus ")}`),
        el("p", { class: "setup__hint" },
          "Unknown sizes are not assumed cached or free; the coordinator reports them by component."))));
  } else {
    sectionDownload.appendChild(el("p", { class: "setup__hint" },
      "Coordinator did not report download requirements; you can submit and the server will tell you what it needs."));
  }
  card.appendChild(sectionDownload);

  const actions = el("div", { class: "setup__actions" });
  const submitBtn = el("button", { type: "submit", class: "btn" }, "Start setup");
  actions.appendChild(submitBtn);
  if (onCancel) actions.appendChild(el("button", { type: "button", class: "btn btn--ghost",
                                                   onClick: () => onCancel() }, "Cancel"));
  card.appendChild(actions);

  container.appendChild(card);
  mount.appendChild(container);

  return {
    setBusy(busy, message) {
      submitBtn.disabled = !!busy;
      setLiveStatus(liveStatus, message || "");
      const inputs = card.querySelectorAll("input");
      for (const node of inputs) node.disabled = !!busy;
    },
  };
}

export function renderSetupProgress(mount, status) {
  mount.replaceChildren();
  const container = el("section", { class: "setup", role: "region",
                                    "aria-labelledby": "setup-title" });
  container.appendChild(el("h2", { id: "setup-title" }, "Preparing local chat…"));
  container.appendChild(el("p", { class: "setup__hint" },
    "Real work in progress. Progress lines reflect actual coordinator state."));

  const progress = el("div", { class: "setup__progress" });
  const steps = (status && Array.isArray(status.progress)) ? status.progress : [];
  if (steps.length === 0) {
    progress.appendChild(el("div", { class: "setup__progress-line",
                                     dataset: { state: "running" } },
      el("div", {}, "Working…"),
      el("small", {}, "Connecting")));
  }
  for (const step of steps) {
    progress.appendChild(el("div", { class: "setup__progress-line",
                                     dataset: { state: step.state || "pending" } },
      el("div", {}, step.label || step.id || "Step"),
      el("small", {}, step.detail || "")));
  }
  container.appendChild(progress);
  if (status && status.error) {
    container.appendChild(el("div", { class: "setup__progress-line",
                                      dataset: { state: "error" } },
      el("div", {}, status.error.title || "Setup failed"),
      el("small", {}, status.error.detail || "")));
  }
  mount.appendChild(container);
}

function formatBytes(n) {
  if (!Number.isFinite(n) || n <= 0) return "0 B";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let i = 0;
  let value = n;
  while (value >= 1024 && i < units.length - 1) { value /= 1024; i += 1; }
  return `${value.toFixed(value < 10 && i > 0 ? 1 : 0)} ${units[i]}`;
}
