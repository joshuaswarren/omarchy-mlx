// Theme polling and application: read /api/theme, apply CSS variables to
// <html> only — DOM focus and form state survive the update because no node
// is replaced.

import { fetchTheme } from "./api.js";
import { swallow } from "./util.js";

const CSS_VARS = [
  "--mlx-canvas", "--mlx-panel", "--mlx-border",
  "--mlx-text", "--mlx-dim", "--mlx-focus", "--mlx-focus-ink",
  "--mlx-ready", "--mlx-error", "--mlx-warn",
  "--mlx-font-sans", "--mlx-font-mono",
];

const THEME_VAR_KEYS = new Set(CSS_VARS);
let lastSignature = "";
let polling = false;

function signatureOf(css) {
  return CSS_VARS.map((k) => css[k] || "").join("|");
}

function apply(css) {
  if (!css) return;
  const root = document.documentElement;
  for (const key of CSS_VARS) {
    if (css[key]) root.style.setProperty(key, css[key]);
  }
  // --mlx-mode isn't in our variables list — it's a marker only.
  if (css["--mlx-mode"]) root.dataset.mlxMode = css["--mlx-mode"];
}

export function themeStatus(css) {
  return {
    name: (css && css.name) || "Built-in dark",
    slug: (css && css.slug) || null,
    source: (css && css.source) || "fallback",
    status: (css && css.status) || "theme not loaded",
  };
}

export async function refreshTheme() {
  const data = await fetchTheme();
  if (!data || !data.css) return data;
  const sig = signatureOf(data.css);
  if (sig !== lastSignature) {
    lastSignature = sig;
    apply(data.css);
    document.body.removeAttribute("data-theme-loading");
  }
  return data;
}

export function startThemePolling(intervalMs = 5000) {
  if (polling) return;
  polling = true;
  const tick = async () => {
    try { await refreshTheme(); }
    catch { swallow(); }
  };
  tick();
  const handle = setInterval(tick, intervalMs);
  document.addEventListener("visibilitychange", () => {
    if (!document.hidden) tick();
  });
  window.addEventListener("pagehide", () => clearInterval(handle), { once: true });
  return () => { clearInterval(handle); polling = false; };
}

export function themeVarsUsed() {
  return [...THEME_VAR_KEYS];
}
