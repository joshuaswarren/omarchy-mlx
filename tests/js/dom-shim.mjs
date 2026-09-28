/* Minimal deterministic DOM shim covering exactly the element surface that
   composer.js (via dom.js el()) touches at build time and when tests fire
   handlers. Runs real composer behavior; renders nothing. */

function matches(node, part) {
  const m = part.match(/^(?:([a-z][a-z0-9]*))?(?:#([\w-]+))?(?:\.([\w-]+))?(?:\[([^=\]]+)=([^\]]+)\])?(:checked)?$/i);
  if (!m) return false;
  const [, tag, id, cls, attr, attrVal, checked] = m;
  if (tag && node.tagName !== tag.toLowerCase()) return false;
  if (id && node.id !== id) return false;
  if (cls && !(node.className || "").split(/\s+/).includes(cls)) return false;
  if (attr && String(node.getAttribute(attr)) !== attrVal) return false;
  if (checked && !node.checked) return false;
  return Boolean(tag || id || cls || attr || checked);
}

function makeElement(tag) {
  const node = {
    tagName: String(tag).toLowerCase(),
    className: "",
    dataset: {},
    children: [],
    handlers: {},
    attributes: {},
    value: "",
    hidden: false,
    disabled: false,
    checked: false,
    focusCount: 0,
    setAttribute(name, val) {
      if (name === "class") { node.className = String(val); return; }
      node.attributes[name] = val;
      node[name] = val === "" ? true : val;
    },
    getAttribute(name) { return node.attributes[name]; },
    appendChild(child) { node.children.push(child); child.parent = node; return child; },
    append(...kids) { for (const k of kids) { node.children.push(k); k.parent = node; } },
    remove() {
      const p = node.parent;
      if (p) p.children = p.children.filter((c) => c !== node);
    },
    replaceChildren(...kids) {
      node.children = kids;
      for (const k of kids) k.parent = node;
    },
    addEventListener(type, fn) { (node.handlers[type] ||= []).push(fn); },
    removeEventListener(type, fn) {
      node.handlers[type] = (node.handlers[type] || []).filter((f) => f !== fn);
    },
    _fire(type, event = { preventDefault() {}, key: "", clientX: 0, clientY: 0, pointerId: 1 }) {
      const fns = (node.handlers[type] || []).slice();
      for (const fn of fns) fn(event);
    },
    focus() { node.focusCount += 1; },
    setPointerCapture() {},
    getContext: () => null,
    classList: {
      add(...names) { for (const n of names) if (!node.className.includes(n)) node.className = (node.className + " " + n).trim(); },
      remove(...names) { for (const n of names) node.className = node.className.split(/\s+/).filter((c) => c !== n).join(" "); },
      contains(name) { return node.className.split(/\s+/).includes(name); },
    },
    querySelector(sel) {
      return collectDescendants(node).find((c) => selectorMatches(c, sel)) || null;
    },
    querySelectorAll(sel) {
      return collectDescendants(node).filter((c) => selectorMatches(c, sel));
    },
  };
  return node;
}

function collectDescendants(node, out = []) {
  for (const child of node.children || []) {
    if (child && child.tagName) {
      out.push(child);
      collectDescendants(child, out);
    }
  }
  return out;
}

function selectorMatches(node, selector) {
  return selector.split(",").some((part) => matches(node, part.trim()));
}

export function installShim() {
  globalThis.document = {
    createElement: makeElement,
    createTextNode: (text) => ({ text: String(text) }),
    getElementById: () => null,
    body: makeElement("body"),
  };
  const storage = {};
  globalThis.localStorage = {
    getItem: (k) => (k in storage ? storage[k] : null),
    setItem: (k, v) => { storage[k] = String(v); },
    removeItem: (k) => { delete storage[k]; },
  };
}

export const tick = () => new Promise((resolve) => setTimeout(resolve, 0));
