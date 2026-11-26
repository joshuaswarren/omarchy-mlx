// Offline-bundle transfer: "Prepare another computer" export and
// "Install from bundle" import, driven by the parent's /api/transfer job
// endpoint. Every server write is an explicit button; nothing runs on
// drawer open and no path is ever model-supplied — the user types the
// absolute local path themselves, because the browser cannot browse the
// server filesystem.

import { el } from "./dom.js";
import { fetchTransfer, postTransfer } from "./api.js";
import { asString, asNumber } from "./util.js";

const POLL_INTERVAL_MS = 1000;

export function normalizeLicenses(result) {
  const licenses = (result || {}).licenses;
  if (Array.isArray(licenses)) {
    return licenses.map((l) => asString(l)).filter((l) => l !== undefined);
  }
  if (licenses && licenses instanceof Object) return Object.keys(licenses);
  return [];
}

export function licenseDetails(result) {
  const details = (result || {}).license_details;
  return details && details instanceof Object ? details : {};
}

export function missingWheels(result) {
  const missing = (result || {}).missing_wheels;
  if (!Array.isArray(missing)) return [];
  return missing.map((m) => asString(m)).filter((m) => m !== undefined);
}

export function totalBytes(result) {
  return asNumber((result || {}).total_bytes);
}

export function buildTransferRequest({ action, bundle, output, pairIds,
                                       voice, approvedLicenses, wheelCaches }) {
  const payload = { action };
  if (bundle) payload.bundle = bundle;
  if (output) payload.output = output;
  if (Array.isArray(pairIds) && pairIds.length > 0) payload.pair_ids = pairIds;
  payload.voice = voice === true;
  if (Array.isArray(approvedLicenses)) payload.approved_licenses = approvedLicenses;
  if (Array.isArray(wheelCaches)) payload.wheel_caches = wheelCaches;
  return payload;
}

export function planRows(result) {
  const rows = [];
  const r = result || {};
  for (const pair of Array.isArray(r.pairs) ? r.pairs : []) {
    rows.push([`Pair ${pair.pair_id || "?"}`,
      `${formatBytes(asNumber(pair.bytes))} · ${pair.arch || "arch?"} · ${pair.file_count ?? "?"} files`]);
  }
  for (const pack of Array.isArray(r.voice) ? r.voice : []) {
    rows.push([`Voice pack ${pack.pack_id || "?"}`,
      `${formatBytes(asNumber(pack.bytes))} · license ${pack.license || "?"}`]);
  }
  const wheels = Array.isArray(r.wheels) ? r.wheels : [];
  const missing = missingWheels(r);
  if (missing.length > 0) {
    rows.push(["Missing wheels (need download)", missing.join(", ")]);
  } else if (wheels.length > 0) {
    rows.push([`Cached wheels (${wheels.length} present in wheelhouse)`]);
  }
  const bytes = totalBytes(r);
  if (bytes !== undefined) rows.push(["Total transfer size", formatBytes(bytes)]);
  const req = r.requirements;
  if (req && req instanceof Object) {
    const entries = Object.entries(req).map(([d, v]) => `${d}==${v}`);
    if (entries.length > 0) rows.push(["Requirements", entries.join(", ")]);
  }
  return rows;
}

function formatBytes(n) {
  if (!Number.isFinite(n) || n <= 0) return "0 B";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let i = 0;
  let value = n;
  while (value >= 1024 && i < units.length - 1) { value /= 1024; i += 1; }
  return `${value.toFixed(value < 10 && i > 0 ? 1 : 0)} ${units[i]}`;
}

export function renderTransferPanel(container, { status, onDone } = {}) {
  let pollTimer = null;
  let destroyed = false;
  container.replaceChildren();

  const intro = el("p", { class: "setup__hint" },
    "Export this installation (runtime wheels, pair manifest, approved model assets) for another computer, or install from an exported bundle. Paths are typed by you as absolute paths on this machine; nothing is written until you press an explicit button.");

  const modeWrap = el("div", { class: "setup__preference", role: "tablist" });
  const exportBtn = el("button", { type: "button", class: "btn btn--ghost",
    role: "tab", "aria-selected": "true" }, "Export bundle");
  const importBtn = el("button", { type: "button", class: "btn btn--ghost",
    role: "tab", "aria-selected": "false" }, "Install from bundle");
  modeWrap.appendChild(exportBtn);
  modeWrap.appendChild(importBtn);

  const exportPane = el("section", { class: "setup__section" });
  const importPane = el("section", { class: "setup__section", hidden: true });
  const statusLine = el("p", { class: "setup__hint", role: "status", "aria-live": "polite" });
  const resultBox = el("div", { class: "detail-section" });

  container.appendChild(intro);
  container.appendChild(modeWrap);
  container.appendChild(exportPane);
  container.appendChild(importPane);
  container.appendChild(statusLine);
  container.appendChild(resultBox);

  const activePairId = (status && status.active_pair && status.active_pair.id) || "";
  const pairs = (status && Array.isArray(status.pairs)) ? status.pairs : [];

  // --- export pane ---
  const pairSelect = el("select", { id: "transfer-pair", class: "field-input" });
  const pairIds = new Set(pairs.map((p) => p.id).filter(Boolean));
  if (activePairId && !pairIds.has(activePairId)) {
    pairSelect.appendChild(el("option", { value: activePairId },
      (status.active_pair.label || activePairId) + " (active)"));
  }
  for (const p of pairs) {
    pairSelect.appendChild(el("option", { value: p.id }, p.label || p.id));
  }
  if (activePairId) pairSelect.value = activePairId;
  const voiceCheck = el("input", { type: "checkbox", id: "transfer-voice" });
  const outputPath = el("input", { type: "text", id: "transfer-output", class: "field-input",
    placeholder: "exports/mlx-bundle.zip", spellcheck: "false" });
  const planBtn = el("button", { type: "button", class: "btn btn--ghost" }, "Inspect export plan");
  const prepareBtn = el("button", { type: "button", class: "btn", disabled: true }, "Prepare bundle");
  prepareBtn.disabled = true;
  const licenseBox = el("div", { class: "transfer-licenses", role: "group",
    "aria-label": "License approval" });
  const planNote = el("p", { class: "setup__hint" },
    "Plan is a cache-only inspection: the tool checks the wheel closure, lists every model and pack it would need, and reports any missing wheels. Nothing is downloaded or written until you press an explicit Prepare bundle button.");

  exportPane.appendChild(el("div", { class: "form-field" },
    el("label", { for: "transfer-pair" }, "Pair to export"), pairSelect));
  exportPane.appendChild(el("label", { class: "setup__checkbox" },
    voiceCheck,
    el("div", {},
      el("strong", {}, "Include voice pack"),
      el("p", { class: "setup__hint" }, "Adds the TTS pack, if installed and licensed."))));
  exportPane.appendChild(el("div", { class: "form-field" },
    el("label", { for: "transfer-output" }, "Output bundle file (absolute path)"), outputPath));
  exportPane.appendChild(planNote);
  exportPane.appendChild(licenseBox);
  exportPane.appendChild(el("div", { class: "card__actions" }, planBtn, prepareBtn));

  // --- import pane ---
  const bundlePath = el("input", { type: "text", id: "transfer-bundle", class: "field-input",
    placeholder: "/path/to/exported-bundle", spellcheck: "false" });
  const inspectBtn = el("button", { type: "button", class: "btn btn--ghost" }, "Inspect bundle");
  const installBtn = el("button", { type: "button", class: "btn" }, "Install bundle");
  installBtn.disabled = true;
  const importLicenseBox = el("div", { class: "transfer-licenses", role: "group",
    "aria-label": "License approval" });
  importPane.appendChild(el("div", { class: "form-field" },
    el("label", { for: "transfer-bundle" }, "Bundle file (absolute path)"), bundlePath));
  importPane.appendChild(importLicenseBox);
  importPane.appendChild(el("div", { class: "card__actions" }, inspectBtn, installBtn));

  exportBtn.addEventListener("click", () => setMode("export"));
  importBtn.addEventListener("click", () => setMode("import"));

  function setMode(mode) {
    const isExport = mode === "export";
    exportPane.hidden = !isExport;
    importPane.hidden = isExport;
    exportBtn.setAttribute("aria-selected", String(isExport));
    importBtn.setAttribute("aria-selected", String(!isExport));
  }

  function setBusy(button, busy) {
    button.disabled = busy;
  }

  function clearApproval(box) { box.replaceChildren(); }

  function renderLicenses(box, result, onAllApproved) {
    clearApproval(box);
    const licenses = normalizeLicenses(result);
    if (licenses.length === 0) {
      box.appendChild(el("p", { class: "setup__hint" }, "No licenses reported for this bundle."));
      onAllApproved(true);
      return;
    }
    const details = licenseDetails(result);
    const checks = [];
    for (const name of licenses) {
      const detail = details[name] || {};
      const models = Array.isArray(detail.models) ? detail.models.join(", ") : "";
      const bytes = asNumber(detail.bytes);
      const cb = el("input", { type: "checkbox", id: `lic-${box.id}-${name}` });
      cb.addEventListener("change", () => {
        onAllApproved(checks.every((c) => c.checked));
      });
      checks.push(cb);
      const label = el("label", { for: cb.id, class: "setup__checkbox transfer-license-row" },
        cb,
        el("div", {},
          el("strong", {}, `I accept license: ${name}`),
          el("p", { class: "setup__hint" },
            `${models || "covers this bundle's models"}${bytes !== undefined ? ` · ${formatBytes(bytes)}` : ""}`)));
      box.appendChild(label);
    }
    onAllApproved(false);
  }

  function showResult(result) {
    resultBox.replaceChildren();
    if (!result) return;
    const dl = el("dl");
    const rows = planRows(result);
    for (const [k, v] of rows) {
      dl.appendChild(el("dt", {}, k));
      dl.appendChild(el("dd", {}, String(v)));
    }
    if (dl.childNodes.length > 0) {
      resultBox.appendChild(el("h3", {}, "Result"));
      resultBox.appendChild(dl);
    }
  }

  async function pollJob(button, { onComplete }) {
    if (destroyed) return;
    try {
      const job = await fetchTransfer();
      const state = asString(job && job.state) || "idle";
      if (state === "running") {
        statusLine.textContent = "Working…";
        pollTimer = window.setTimeout(() => pollJob(button, { onComplete }), POLL_INTERVAL_MS);
        return;
      }
      setBusy(button, false);
      if (state === "error") {
        const errText = asString(job && job.error) || (job && job.error && job.error.message) || "Transfer failed";
        statusLine.textContent = `Transfer failed: ${errText}`;
        return;
      }
      if (state === "complete") {
        statusLine.textContent = "Done.";
        showResult(job.result);
        if (onComplete) onComplete(job.result);
      }
    } catch (err) {
      setBusy(button, false);
      statusLine.textContent = `Transfer status failed: ${err.message || "unknown error"}`;
    }
  }

  function startJob(button, payload, { onComplete, onAccepted } = {}) {
    statusLine.textContent = "Starting…";
    setBusy(button, true);
    postTransfer(payload).then(() => {
      if (onAccepted) onAccepted();
      pollTimer = window.setTimeout(() => pollJob(button, { onComplete }), POLL_INTERVAL_MS);
    }).catch((err) => {
      statusLine.textContent = `Transfer request failed: ${err.message || err.code || "unknown error"}`;
      setBusy(button, false);
    });
  }

  planBtn.addEventListener("click", () => {
    resultBox.replaceChildren();
    clearApproval(licenseBox);
    prepareBtn.disabled = true;
    const payload = buildTransferRequest({
      action: "plan",
      pairIds: [pairSelect.value].filter(Boolean),
      voice: voiceCheck.checked,
    });
    startJob(planBtn, payload, { onComplete: (result) => {
      planBtn.disabled = false;
      renderLicenses(licenseBox, result, (all) => {
        prepareBtn.disabled = !(all && outputPath.value.trim());
      });
      outputPath.addEventListener("input", () => {
        const all = licenseBox.querySelectorAll("input[type=checkbox]");
        const allChecked = all.length === 0 || Array.from(all).every((c) => c.checked);
        prepareBtn.disabled = !(allChecked && outputPath.value.trim());
      });
    } });
  });

  prepareBtn.addEventListener("click", () => {
    const out = outputPath.value.trim();
    if (!out.startsWith("/")) {
      statusLine.textContent = "Output must be an absolute path on this machine.";
      return;
    }
    const approved = Array.from(licenseBox.querySelectorAll("input[type=checkbox]:checked"))
      .map((c) => c.closest("label").querySelector("strong").textContent.replace("I accept license: ", ""));
    const payload = buildTransferRequest({
      action: "prepare", output: out,
      pairIds: [pairSelect.value].filter(Boolean),
      voice: voiceCheck.checked, approvedLicenses: approved,
    });
    startJob(prepareBtn, payload, { onComplete: () => {
      prepareBtn.disabled = false;
      if (onDone) onDone();
    } });
  });

  inspectBtn.addEventListener("click", () => {
    resultBox.replaceChildren();
    clearApproval(importLicenseBox);
    installBtn.disabled = true;
    const bundle = bundlePath.value.trim();
    if (!bundle.startsWith("/")) {
      statusLine.textContent = "Bundle path must be absolute on this machine.";
      return;
    }
    const payload = buildTransferRequest({ action: "inspect", bundle });
    startJob(inspectBtn, payload, { onComplete: (result) => {
      inspectBtn.disabled = false;
      renderLicenses(importLicenseBox, result, (all) => {
        installBtn.disabled = !all;
      });
    } });
  });

  installBtn.addEventListener("click", () => {
    const bundle = bundlePath.value.trim();
    if (!bundle.startsWith("/")) {
      statusLine.textContent = "Bundle path must be absolute on this machine.";
      return;
    }
    const approved = Array.from(importLicenseBox.querySelectorAll("input[type=checkbox]:checked"))
      .map((c) => c.closest("label").querySelector("strong").textContent.replace("I accept license: ", ""));
    const payload = buildTransferRequest({
      action: "install", bundle, approvedLicenses: approved,
    });
    startJob(installBtn, payload, { onComplete: () => {
      installBtn.disabled = false;
      if (onDone) onDone();
    } });
  });

  return {
    destroy() {
      destroyed = true;
      if (pollTimer) {
        window.clearTimeout(pollTimer);
        pollTimer = null;
      }
    },
  };
}

