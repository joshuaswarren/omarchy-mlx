import { cancelTurn, heartbeat, openConversation, transcribe } from "./api.js";
import { renderComponent, plainText as plainTextComponent } from "./genui.js";
import { renderMarkdown } from "./markdown.js";
import { el, announce } from "./dom.js";
import { swallow } from "./util.js";

const HEARTBEAT_MS = 5_000;

export class ConversationView {
  constructor({ conversation, recorder, speaker, live, jumpChip }) {
    this.conversation = conversation;
    this.recorder = recorder;
    this.speaker = speaker;
    this.live = live;
    this.jumpChip = jumpChip;
    this.activeTurnId = null;
    this.lastTextNode = null;
    this.lastSequence = conversation.sequence || 0;
    this.scrollAtBottom = true;
    this.boundJump = () => {
      if (!this.logEl) return;
      this.logEl.scrollTop = this.logEl.scrollHeight;
      this.jumpChip.hidden = true;
      this.scrollAtBottom = true;
    };
    this.jumpChip.addEventListener("click", this.boundJump);
    this.jumpChip.hidden = true;
    this.sentences = [];
    this.sentenceCounter = 0;
    this.speakReplies = false;
    this.speakerReady = true;
  }

  setConversation(conversation) {
    this.conversation = conversation;
    this.lastSequence = conversation.sequence || 0;
  }

  renderInto(mount) {
    this.mount = mount;
    mount.replaceChildren();
    const log = el("div", { class: "chat", role: "log", "aria-live": "off" });
    mount.appendChild(log);
    if (this.conversation.messages && this.conversation.messages.length) {
      for (const msg of this.conversation.messages) this._appendMessage(log, msg);
    } else {
      this._renderExamples(log);
    }
    this.logEl = log;
  }

  destroy() {
    this.jumpChip.removeEventListener("click", this.boundJump);
    this.cancelActiveTurn({ silent: true });
    if (this.speaker) this.speaker.stop();
  }

  _renderExamples(log) {
    log.appendChild(el("p", { class: "setup__hint" },
      "Type a question, or try one of these:"));
    const wrap = el("div", { class: "chat__examples" });
    const examples = [
      { label: "Explain how to read a smoke recipe aloud",
        text: "Explain, in plain English, how to read a recipe like a short story." },
      { label: "Compare two laptop choices",
        compare: true,
        options: [{ label: '13" MacBook Air M3' }, { label: '14" MacBook Pro M3' }],
        criteria: "Video editing performance in a travel-friendly machine" },
    ];
    for (const ex of examples) {
      const btn = el("button", { type: "button", onClick: () => {
        if (ex.compare && this.onPrefillCompare) {
          this.onPrefillCompare({ options: ex.options, criteria: ex.criteria });
          return;
        }
        if (this.onSend) this.onSend({ text: ex.text });
      } }, ex.label);
      wrap.appendChild(btn);
    }
    log.appendChild(wrap);
  }

  _appendMessage(log, msg) {
    const role = msg.role || "user";
    const bubble = el("article", { class: `message message--${role}`,
                                   dataset: { turnId: msg.turn_id || "" } });
    bubble.appendChild(el("span", { class: "message__role" },
      role === "user" ? "You" : "Assistant"));
    if (role === "assistant") {
      const text = el("div", { class: "message__text", dataset: { role: "assistant" } });
      if (msg.content) text.appendChild(renderMarkdown(msg.content));
      bubble.appendChild(text);
      this.lastTextNode = text;
    } else {
      bubble.appendChild(el("div", { class: "message__text" },
        renderMarkdown(msg.content || "")));
    }
    if (msg.status === "stopped") bubble.classList.add("message--stopped");
    if (msg.error) {
      bubble.classList.add("message--error");
      bubble.appendChild(el("div", { class: "message__error" },
        msg.error.title || "Response failed",
        msg.error.detail ? " " + msg.error.detail : ""));
    }
    if (msg.components && msg.components.length) {
      const cards = el("div", { class: "message__cards" });
      for (const component of msg.components) cards.appendChild(this._renderComponentCard(component, msg));
      bubble.appendChild(cards);
    }
    if (msg.decision) {
      const cards = el("div", { class: "message__cards" });
      cards.appendChild(this._renderComponentCard(msg.decision, msg));
      bubble.appendChild(cards);
    }
    log.appendChild(bubble);
    return bubble;
  }

  _renderComponentCard(component, msg) {
    const card = renderComponent(component, (payload) =>
      this._submitCardAction(card, component, payload, msg));
    if (!card) return document.createTextNode("");
    const titleBar = el("div", { class: "card__actions" },
      el("button", { type: "button", class: "message__action",
        onClick: () => toggleTextVersion(card, component) }, "Show as text"));
    card.insertBefore(titleBar, card.firstChild);
    return card;
  }

  _submitCardAction(card, component, payload, msg) {
    if (!this.conversation || !msg) return;
    if (!msg.turn_id) return;
    card.dataset.busy = "true";
    import("./api.js").then(({ postAction }) =>
      postAction(this.conversation.id, {
        turn_id: msg.turn_id,
        component_id: component.id,
        revision: component.revision || 0,
        action_id: String(component.id || component._id || ""),
        action: payload.action,
        values: payload.values || {},
      }).then((resp) => {
        card.dataset.busy = "false";
        if (resp && resp.conversation) this.conversation = resp.conversation;
      }).catch((err) => {
        card.dataset.busy = "false";
        announce(this.live, `Action failed: ${err.message || err.code || "unknown error"}`);
      }));
  }

  appendUser(text) {
    if (!this.logEl) return null;
    const bubble = el("article", { class: "message message--user" },
      el("span", { class: "message__role" }, "You"),
      el("div", { class: "message__text" }, renderMarkdown(text)));
    this.logEl.appendChild(bubble);
    return bubble;
  }

  appendAssistant() {
    // One live bubble per turn: status events must not stack empty bubbles.
    if (this.lastAssistantBubble && this._liveTurnId === (this.activeTurnId || "")) {
      return this.lastAssistantBubble;
    }
    const bubble = el("article", { class: "message message--assistant" },
      el("span", { class: "message__role" }, "Assistant"),
      el("div", { class: "message__text", dataset: { role: "assistant" } }));
    bubble._mlxSentences = [];
    bubble._mlxTurnId = this.activeTurnId || "";
    this._liveTurnId = this.activeTurnId || "";
    const meta = el("div", { class: "message__meta" });
    const readBtn = el("button", { type: "button",
      onClick: () => this._readBubbleAloud(bubble, readBtn) }, "Read aloud");
    meta.appendChild(readBtn);
    bubble.appendChild(meta);
    this.logEl.appendChild(bubble);
    this.lastTextNode = bubble.querySelector(".message__text");
    this.lastAssistantBubble = bubble;
    this.sentenceCounter = 0;
    return bubble;
  }

  enableContinue(announcement = "") {
    const bubble = this.lastAssistantBubble;
    if (!bubble) return;
    const meta = bubble.querySelector(".message__meta");
    if (!meta || meta.querySelector(".message__action--continue")) return;
    if (announcement) {
      const note = el("span", { class: "message__action-note" }, announcement);
      meta.appendChild(note);
    }
    const btn = el("button", {
      type: "button", class: "message__action message__action--continue",
    }, "Continue");
    btn.addEventListener("click", () => {
      btn.disabled = true;
      const note = meta.querySelector(".message__action-note");
      if (note) note.textContent = "Continuing…";
      if (this.onSend) this.onSend({ text: "Continue." });
    });
    meta.appendChild(btn);
  }

  _readBubbleAloud(bubble, readBtn) {
    if (this.speakerReady === false) {
      announce(this.live, "Speech pack is not ready. Install the voice pack in setup.");
      return;
    }
    const sentences = bubble._mlxSentences || [];
    if (sentences.length === 0) {
      announce(this.live, "Nothing to read yet.");
      return;
    }
    const turnId = bubble._mlxTurnId || this.activeTurnId || "";
    for (const [seq, text] of sentences) {
      this.speaker.enqueue(turnId, seq, text);
    }
    readBtn.textContent = "Queued for reading";
    readBtn.disabled = true;
  }

  setSpeakerReady(ready) {
    this.speakerReady = !!ready;
  }

  appendText(text) {
    if (!this.lastTextNode) this.appendAssistant();
    this.lastTextNode.appendChild(document.createTextNode(text));
    this._splitForSpeech(text);
    this._scrollIfNearBottom();
  }

  appendCards(components, msg) {
    if (!this.lastAssistantBubble) this.appendAssistant();
    let cards = this.lastAssistantBubble.querySelector(".message__cards");
    if (!cards) {
      cards = el("div", { class: "message__cards" });
      this.lastAssistantBubble.appendChild(cards);
    }
    for (const component of components) cards.appendChild(this._renderComponentCard(component, msg || {}));
    this._scrollIfNearBottom();
  }

  setError(err) {
    if (!this.lastAssistantBubble) this.appendAssistant();
    this.lastAssistantBubble.classList.add("message--error");
    this.lastAssistantBubble.appendChild(el("div", { class: "message__error" },
      err.title || "Response failed", err.detail ? " " + err.detail : ""));
  }

  setStopped() {
    if (this.lastAssistantBubble) this.lastAssistantBubble.classList.add("message--stopped");
  }

  _splitForSpeech(text) {
    const bubble = this.lastAssistantBubble;
    if (!bubble) return;
    import("./markdown.js").then(({ splitSentences }) => {
      const segs = splitSentences(text);
      for (const s of segs) {
        this.sentenceCounter += 1;
        const entry = [this.sentenceCounter, s];
        bubble._mlxSentences.push(entry);
        if (this.speakReplies) {
          this.speaker.enqueue(this.activeTurnId || bubble._mlxTurnId || "",
                               entry[0], s);
        }
      }
    });
  }

  setSpeakReplies(enabled) {
    this.speakReplies = !!enabled;
  }

  setTurn(turnId) {
    this.activeTurnId = turnId;
  }

  clearTurn() {
    this.activeTurnId = null;
    if (this.heartbeatHandle) { clearInterval(this.heartbeatHandle); this.heartbeatHandle = null; }
  }

  startHeartbeat() {
    if (this.heartbeatHandle) clearInterval(this.heartbeatHandle);
    this.heartbeatHandle = setInterval(() => {
      if (!this.activeTurnId || !this.conversation) return;
      heartbeat(this.conversation.id, this.activeTurnId).catch(() => {});
    }, HEARTBEAT_MS);
  }

  async cancelActiveTurn({ silent = false } = {}) {
    if (this.heartbeatHandle) { clearInterval(this.heartbeatHandle); this.heartbeatHandle = null; }
    if (this.activeTurnId && this.conversation) {
      try { await cancelTurn(this.conversation.id, this.activeTurnId); }
      catch { swallow(); }
    }
    this.activeTurnId = null;
    if (this.speaker) this.speaker.reset();
    if (!silent) announce(this.live, "Response stopped");
  }

  _scrollIfNearBottom() {
    if (!this.logEl) return;
    const distance = this.logEl.scrollHeight - this.logEl.scrollTop - this.logEl.clientHeight;
    if (distance > 120) {
      this.jumpChip.hidden = false;
      this.scrollAtBottom = false;
    } else {
      this.logEl.scrollTop = this.logEl.scrollHeight;
      this.scrollAtBottom = true;
      this.jumpChip.hidden = true;
    }
  }

  attachScrollObserver() {
    if (!this.logEl) return;
    this.logEl.addEventListener("scroll", () => {
      const distance = this.logEl.scrollHeight - this.logEl.scrollTop - this.logEl.clientHeight;
      this.scrollAtBottom = distance <= 32;
      if (this.scrollAtBottom) this.jumpChip.hidden = true;
    });
  }
}

function toggleTextVersion(card, component) {
  const existing = card.querySelector(".card__text-fallback");
  if (existing) {
    existing.remove();
    const btn = card.querySelector(".message__action");
    if (btn) btn.textContent = "Show as text";
    return;
  }
  const text = plainTextComponent(component);
  const block = el("pre", { class: "card__text-fallback" }, text);
  card.appendChild(block);
  const btn = card.querySelector(".message__action");
  if (btn) btn.textContent = "Hide text version";
}

export async function openConversationState(id) {
  return openConversation(id);
}

export { transcribe };
