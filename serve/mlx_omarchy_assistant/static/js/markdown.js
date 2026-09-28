// Tiny Markdown renderer, escape-first.
//
// The model can attempt HTML, JavaScript, SVG, javascript: URLs and image
// embeds; the renderer escapes every input character first, then applies a
// fixed set of safe substitutions to the escaped text. No model-supplied
// bytes survive into the DOM as markup except for the small whitelist of
// generated tags produced by these helpers.

const ESCAPES = {
  "&": "&amp;",
  "<": "&lt;",
  ">": "&gt;",
  '"': "&quot;",
  "'": "&#39;",
};

function escapeHtml(text) {
  return text.replace(/[&<>"']/g, (ch) => ESCAPES[ch]);
}

const ALLOWED_LINK = /^(https?:\/\/[^\s<>"']+|mailto:[^\s<>"']+)$/i;

function safeHref(url) {
  if (!url) return null;
  if (ALLOWED_LINK.test(url)) return url;
  return null;
}

// Split out fenced code blocks first so anything between ``` is left as a
// <pre> verbatim, with markdown-style emphasis turned off inside it.
function splitCodeFences(text) {
  const segments = [];
  let i = 0;
  while (i < text.length) {
    const start = text.indexOf("```", i);
    if (start === -1) { segments.push({ kind: "text", body: text.slice(i) }); break; }
    if (start > i) segments.push({ kind: "text", body: text.slice(i, start) });
    const end = text.indexOf("```", start + 3);
    if (end === -1) {
      segments.push({ kind: "code", body: text.slice(start + 3) });
      break;
    }
    segments.push({ kind: "code", body: text.slice(start + 3, end) });
    i = end + 3;
  }
  return segments;
}

function applyInline(escapedText) {
  // Code spans: `x` → <code>x</code>. Replace greedily on the escaped text.
  escapedText = escapedText.replace(/`([^`\n]+)`/g, (_, code) => `<code>${code}</code>`);
  // Links: [text](https:// or mailto:) → <a>; otherwise the label stays plain text.
  escapedText = escapedText.replace(/\[([^\]]+)\]\(([^)\s]+)\)/g, (_, label, url) => {
    const href = safeHref(url);
    if (!href) return escapeHtml(label);
    return `<a href="${escapeHtml(href)}" rel="noopener noreferrer" target="_blank">${escapeHtml(label)}</a>`;
  });
  escapedText = escapedText.replace(/\*\*([^*\n]+)\*\*/g, "<strong>$1</strong>");
  escapedText = escapedText.replace(/(^|[^*])\*([^*\n]+)\*/g, "$1<em>$2</em>");
  return escapedText;
}

function renderParagraph(text) {
  // Single-line break becomes <br>; double becomes paragraph split.
  const lines = text.split(/\n{2,}/);
  return lines.map((line) => {
    const flat = line.split("\n").map(applyInline).join("<br>");
    return `<p>${flat}</p>`;
  }).join("");
}

function renderList(text, ordered) {
  const items = text.split(/\n/).filter((l) => l.length)
    .map((l) => l.replace(/^\s*([-*]|\d+\.)\s+/, ""));
  const tag = ordered ? "ol" : "ul";
  const inner = items.map((it) => `<li>${applyInline(it)}</li>`).join("");
  return `<${tag}>${inner}</${tag}>`;
}

function renderHeading(text, level) {
  const safe = Math.min(Math.max(level, 3), 4);
  const m = text.match(/^#{1,6}\s+(.*)$/);
  const inner = m ? applyInline(m[1]) : applyInline(text);
  return `<h${safe}>${inner}</h${safe}>`;
}

function renderBlockquote(text) {
  const inner = text.split("\n").map((l) => l.replace(/^>\s?/, "")).join("<br>");
  return `<blockquote>${applyInline(inner)}</blockquote>`;
}

export function renderMarkdown(text) {
  if (!text) return document.createDocumentFragment();
  const segments = splitCodeFences(text);
  const frag = document.createDocumentFragment();
  for (const seg of segments) {
    if (seg.kind === "code") {
      const pre = document.createElement("pre");
      const code = document.createElement("code");
      // Code body was already escaped inside <pre> via textContent assignment.
      code.textContent = seg.body.replace(/^\w*\n/, "");
      pre.appendChild(code);
      frag.appendChild(pre);
      continue;
    }
    const lines = seg.body.split("\n");
    let i = 0;
    while (i < lines.length) {
      const line = lines[i];
      if (!line.trim()) { i += 1; continue; }
      if (/^#{1,6}\s/.test(line)) {
        const wrapped = document.createElement("div");
        wrapped.innerHTML = renderHeading(line, (line.match(/^#+/)?.[0] || "###").length);
        frag.appendChild(wrapped.firstElementChild);
        i += 1; continue;
      }
      if (/^>\s?/.test(line)) {
        const collected = [line];
        i += 1;
        while (i < lines.length && /^>\s?/.test(lines[i])) { collected.push(lines[i]); i += 1; }
        const wrapped = document.createElement("div");
        wrapped.innerHTML = renderBlockquote(collected.join("\n"));
        frag.appendChild(wrapped.firstElementChild);
        continue;
      }
      if (/^[-*]\s/.test(line)) {
        const collected = [line];
        i += 1;
        while (i < lines.length && /^[-*]\s/.test(lines[i])) { collected.push(lines[i]); i += 1; }
        const wrapped = document.createElement("div");
        wrapped.innerHTML = renderList(collected.join("\n"), false);
        frag.appendChild(wrapped.firstElementChild);
        continue;
      }
      if (/^\d+\.\s/.test(line)) {
        const collected = [line];
        i += 1;
        while (i < lines.length && /^\d+\.\s/.test(lines[i])) { collected.push(lines[i]); i += 1; }
        const wrapped = document.createElement("div");
        wrapped.innerHTML = renderList(collected.join("\n"), true);
        frag.appendChild(wrapped.firstElementChild);
        continue;
      }
      // Paragraph: collect until blank line or block marker.
      const para = [line];
      i += 1;
      while (i < lines.length && lines[i].trim() &&
             !/^(#{1,6}\s|>\s?|[-*]\s|\d+\.\s|```)/.test(lines[i])) {
        para.push(lines[i]); i += 1;
      }
      const wrapped = document.createElement("div");
      wrapped.innerHTML = renderParagraph(para.join("\n"));
      // The wrapper holds a sequence of <p> elements — promote them all.
      while (wrapped.firstChild) frag.appendChild(wrapped.firstChild);
    }
  }
  return frag;
}

// Plain-text fallback used by the "Show as text" button on generative cards.
export function stripMarkdown(text) {
  if (!text) return "";
  return text.replace(/```[\s\S]*?```/g, "")
             .replace(/`([^`]+)`/g, "$1")
             .replace(/\[([^\]]+)\]\([^)]+\)/g, "$1")
             .replace(/[*_>#-]/g, "")
             .replace(/\s+/g, " ")
             .trim();
}

// Split a flat assistant message into sentences suitable for TTS, skipping
// sentences inside fenced code blocks, sentences that consist of a bare URL,
// and the assistant's markdown decoration characters.
export function splitSentences(text) {
  if (!text) return [];
  const segments = splitCodeFences(text);
  const out = [];
  for (const seg of segments) {
    if (seg.kind === "code") continue;
    const stripped = seg.body.replace(/[#>*`_-]/g, " ");
    // Strip bare URLs and image references — they are skipped by design.
    const noUrls = stripped.replace(/https?:\/\/\S+/g, " ")
                           .replace(/mailto:\S+/g, " ")
                           .replace(/!\[[^\]]*\]\([^)]+\)/g, " ");
    const parts = noUrls.split(/(?<=[.!?])\s+(?=[A-Z0-9"'])|\n+/);
    for (const part of parts) {
      const trimmed = part.trim();
      if (trimmed) out.push(trimmed);
    }
  }
  return out;
}
