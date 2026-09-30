import { ConversationView } from "./chat.js";
import { Recorder, SpeakQueue, playPreview } from "./voice.js";
import { exchangeFragment, fetchSession, fetchStatus, postSetup,
         newConversation, openConversation, listConversations,
         deleteConversation, transcribe, postContext,
         setVoice, previewVoice } from "./api.js";
import { renderSetup, renderSetupProgress } from "./setup.js";
import { buildComposer } from "./composer.js";
import { refreshTheme, startThemePolling } from "./theme.js";
import { el, announce, renderDetails } from "./dom.js";
import { asString, asNumber, isPlainObject, swallow, tokenFromHash } from "./util.js";

const STATUS_POLL_IDLE = 4_000;
const STATUS_POLL_BUSY = 1_000;

export class App {
  constructor(root) {
    this.root = root;
    this.mount = root.querySelector("#view");
    this.live = root.querySelector("#live-region");
    this.jumpChip = el("button", { type: "button", class: "jump-chip",
      "aria-label": "Jump to latest message" }, "New response");
    this.jumpChip.hidden = true;
    document.body.appendChild(this.jumpChip);
    this.continueChip = el("button", { type: "button",
      class: "jump-chip jump-chip--continue",
      "aria-label": "Continue reading the response aloud" }, "Continue reading");
    this.continueChip.hidden = true;
    this.continueChip.addEventListener("click", () => this._continueReading());
    document.body.appendChild(this.continueChip);

    this.recorder = new Recorder({
      onMeter: () => {},
      onTick: (elapsed) => announce(this.live, `Recording ${elapsed.toFixed(1)} seconds`),
      onWarn: () => announce(this.live, "About to hit the 30 second limit."),
      onStop: ({ blob, peakRms, reason }) => this._handleRecordingStop(blob, peakRms, reason),
      onDeviceLost: () => this._handleDeviceLost(),
    });
    this.speaker = new SpeakQueue();
    this.speaker.attachHooks({
      onTruncate: () => {
        this._continueBubble = this.view ? this.view.lastAssistantBubble : null;
        this.continueChip.hidden = !this._continueBubble;
        announce(this.live, "Speech paused at the playback limit. Continue reading is available.");
      },
      onAudioDone: ({ truncated }) => {
        if (this.composer) this.composer.setSpeaking(false);
        this.continueChip.hidden = !truncated || !this._continueBubble;
        announce(this.live,
          truncated ? "Spoken reply paused. Continue reading is available."
                    : "Finished speaking.");
      },
      onError: (err) => announce(this.live, `Speech failed: ${err.message || "unknown error"}`),
      onSpeakingChange: (active) => {
        this.speakerSpeaking = active;
        if (this.composer) this.composer.setSpeaking(active);
      },
    });
    this.speakReplies = false;
    this.speakerSpeaking = false;
    this.voiceStates = { recognition: "unknown", synthesis: "unknown" };
    this.conversationModeEnabled = false;
    this._suppressComposerClear = false;

    window.addEventListener("pagehide", () => {
      this.recorder.cancel().catch(() => {});
      this.speaker.stop();
    });
  }

  _micMessage(text) {
    announce(this.live, text);
    if (this.composer) this.composer.setMicNotice(text);
  }

  async _handleRecordingStop(blob, peakRms, reason) {
    // The recorder can stop on its own (30 s cap); the button must follow it.
    if (this.composer) this.composer.resetMicUi();
    const prefix = reason === "limit" ? "Recording stopped at the 30 second limit. " : "";
    if (peakRms < 0.005) {
      this._micMessage(`${prefix}No speech detected.`);
      return;
    }
    this._micMessage(`${prefix}Transcribing…`);
    let text;
    try {
      ({ text } = await transcribe(blob));
    } catch (err) {
      this._micMessage(`Transcribe failed: ${err.message || "unknown error"}`);
      return;
    }
    const transcript = (text || "").trim();
    if (!transcript) {
      this._micMessage("No speech detected.");
      return;
    }
    if (this.composer) this.composer.setMicNotice(prefix.trim());
    this._mergeTranscriptIntoComposer(transcript);
  }

  _handleDeviceLost() {
    this._micMessage("Recording stopped: microphone disconnected.");
    if (this.composer) this.composer.resetMicUi();
  }

  _mergeTranscriptIntoComposer(transcript) {
    if (!this.composer) {
      this._pendingTranscript = transcript;
      announce(this.live, "Transcript ready. Type your reply, then send.");
      return;
    }
    const existing = this.composer.currentText().trim();
    const composed = existing ? `${existing} ${transcript}`.trim() : transcript;
    this._suppressComposerClear = true;
    try {
      this.composer.setText(composed);
    } finally {
      this._suppressComposerClear = false;
    }
    if (existing) {
      // Typed text survives recording; the combined draft needs review
      // before sending, even in conversation mode.
      announce(this.live, "Transcript added to your draft. Review, then press Send.");
      return;
    }
    if (this.conversationModeEnabled) {
      this._suppressComposerClear = true;
      try { this.onSend({ text: transcript }); }
      finally { this._suppressComposerClear = false; }
    } else {
      announce(this.live, "Edit the transcript, then press Send.");
    }
  }

  async boot() {
    const token = tokenFromHash(window.location.hash);
    if (window.location.hash) {
      history.replaceState(null, "", window.location.pathname + window.location.search);
    }
    if (token) {
      try { await exchangeFragment(token); }
      catch (err) {
        this.showLaunchError(err.message || "This launch link is no longer valid.");
        return;
      }
    }
    try { await fetchSession(); }
    catch { this.showLaunchError("Session lookup failed."); return; }
    startThemePolling();
    refreshTheme().catch(swallow);
    await this.pollStatus();
    this._wireButtons();
  }

  _wireButtons() {
    document.getElementById("new-chat-btn").addEventListener("click",
      () => this.newConversation({ save: false }));

    const historyDrawer = document.getElementById("history-drawer");
    const historyBody = document.getElementById("history-body");
    document.getElementById("history-btn").addEventListener("click", async () => {
      historyBody.replaceChildren(el("p", { class: "drawer__empty" }, "Loading…"));
      historyDrawer.showModal();
      try {
        const data = await listConversations();
        const items = (data && data.conversations) || [];
        historyBody.replaceChildren();
        if (items.length === 0) {
          historyBody.appendChild(el("p", { class: "drawer__empty" },
            "No saved conversations yet."));
          return;
        }
        const list = el("ul", { class: "drawer__list" });
        for (const conv of items) {
          const open = el("button", { type: "button",
            onClick: async () => {
              historyDrawer.close();
              await this.openConversation(conv.id);
              if (this.status && this.status.state === "ready") this.renderChat(this.status);
            } },
            el("strong", {}, conv.title || "Conversation"),
            el("small", {}, `${conv.message_count || 0} messages · ${new Date((conv.updated_at || conv.created_at) * 1000).toLocaleString()}`));
          const deleteBtn = el("button", { type: "button", onClick: async () => {
            if (!confirm("Delete this conversation? This cannot be undone.")) return;
            await deleteConversation(conv.id);
            this._refreshHistoryDrawer(historyBody);
          } }, "Delete");
          list.appendChild(el("li", {}, open, deleteBtn));
        }
        historyBody.appendChild(list);
      } catch (err) {
        historyBody.replaceChildren(el("p", { class: "drawer__empty" },
          `Could not load history: ${err.message || "unknown error"}`));
      }
    });

    const detailsDrawer = document.getElementById("details-drawer");
    const detailsBody = document.getElementById("details-body");
    document.getElementById("details-btn").addEventListener("click", async () => {
      detailsBody.replaceChildren(el("p", { class: "drawer__empty" }, "Loading…"));
      detailsDrawer.showModal();
      try {
        const status = await fetchStatus();
        const self = this;
        detailsBody.replaceChildren(renderDetails(status, {
          announce: (msg) => announce(self.live, msg),
          onVoiceChange: async (id) => {
            try {
              await setVoice(id);
            } catch (err) {
              announce(self.live,
                `Voice change failed: ${err.message || "unknown error"}`);
              throw err;
            }
          },
          onPreview: async () => {
            const payload = await previewVoice();
            await playPreview(payload);
          },
        }));
      } catch (err) {
        detailsBody.replaceChildren(el("p", { class: "drawer__empty" },
          `Could not load details: ${err.message || "unknown error"}`));
      }
    });

    const transferDrawer = document.getElementById("transfer-drawer");
    const transferBody = document.getElementById("transfer-body");
    let transferPanel = null;
    document.getElementById("transfer-btn").addEventListener("click", () => {
      transferBody.replaceChildren(el("p", { class: "drawer__empty" }, "Loading…"));
      transferDrawer.showModal();
      import("./transfer.js").then(({ renderTransferPanel }) => {
        if (transferPanel) transferPanel.destroy();
        transferPanel = renderTransferPanel(transferBody, {
          status: this.status,
          onDone: () => this.pollStatus(),
        });
      }).catch((err) => {
        transferBody.replaceChildren(el("p", { class: "drawer__empty" },
          `Could not load transfer panel: ${err.message || "unknown error"}`));
      });
    });
    transferDrawer.addEventListener("close", () => {
      if (transferPanel) { transferPanel.destroy(); transferPanel = null; }
    });
  }

  _refreshHistoryDrawer(body) {
    body.replaceChildren(el("p", { class: "drawer__empty" }, "Loading…"));
    listConversations().then((data) => {
      const items = (data && data.conversations) || [];
      body.replaceChildren();
      if (items.length === 0) {
        body.appendChild(el("p", { class: "drawer__empty" },
          "No saved conversations yet."));
        return;
      }
      const list = el("ul", { class: "drawer__list" });
      for (const conv of items) {
        const open = el("button", { type: "button",
          onClick: async () => {
            document.getElementById("history-drawer").close();
            await this.openConversation(conv.id);
            if (this.status && this.status.state === "ready") this.renderChat(this.status);
          } },
          el("strong", {}, conv.title || "Conversation"),
          el("small", {}, `${conv.message_count || 0} messages · ${new Date((conv.updated_at || conv.created_at) * 1000).toLocaleString()}`));
        list.appendChild(el("li", {}, open));
      }
      body.appendChild(list);
    });
  }

  async pollStatus() {
    this._setPollRate(this._desiredPollMs || STATUS_POLL_IDLE);
    let status;
    try { status = await fetchStatus(); }
    catch { return; }
    this.handleStatus(status);
    this.updateVoiceStates(status);
    this._setPollRate((status && status.state === "preparing") || this.activeTurn
      ? STATUS_POLL_BUSY : STATUS_POLL_IDLE);
  }

  _setPollRate(ms) {
    this._desiredPollMs = ms;
    if (this._statusInterval && this._activePollMs === ms) return;
    clearInterval(this._statusInterval);
    this._activePollMs = ms;
    this._statusInterval = setInterval(() => this.pollStatus(), ms);
  }

  showLaunchError(message) {
    this.mount.replaceChildren();
    this.mount.appendChild(el("div", { class: "setup" },
      el("h2", {}, "Launch required"),
      el("p", { class: "setup__hint" }, message),
      el("p", { class: "setup__hint" },
        "Open MLX Chat again from the desktop launcher; one-time launch links expire after first use.")));
  }

  async handleStatus(status) {
    this.status = status;
    if (!status) return;
    if (status.error && !this.activeTurn) {
      this.showError(status.error);
      return;
    }
    if (status.state === "preparing") {
      renderSetupProgress(this.mount, status);
      return;
    }
    if (status.state === "ready" && status.active_pair) {
      if (!this.conversation) {
        const conv = await newConversation(false);
        await this.openConversation(conv.id);
      }
      this.renderChat(status);
      return;
    }
    this.renderSetupWizard(status);
  }

  showError(err) {
    const message = asString(err);
    const title = message !== undefined ? message
      : (err && err.title) || "Local chat is not ready";
    const detail = message !== undefined ? ""
      : (err && err.detail) || "Open MLX Chat from the launcher to retry.";
    this.mount.replaceChildren();
    this.mount.appendChild(el("div", { class: "setup" },
      el("h2", {}, title),
      el("p", { class: "setup__hint" }, detail),
      el("div", { class: "setup__actions" },
        el("button", { type: "button", class: "btn",
          onClick: () => this.boot() }, "Retry"))));
  }

  renderSetupWizard(status) {
    // Re-mount only when setup-relevant status actually changed; identity
    // polls must never wipe the user's half-filled form.
    const signature = JSON.stringify({
      pairs: status.pairs || null,
      recommended: (status && status.recommended_pair) || null,
      voice: {
        state: (status.voice && status.voice.state) || null,
        synthesis: (status.voice && status.voice.synthesis && status.voice.synthesis.state) || null,
        detail: (status.voice && status.voice.detail) || null,
        synthDetail: (status.voice && status.voice.synthesis && status.voice.synthesis.detail) || null,
      },
      context: {
        max: (status.context && status.context.max_tokens) || null,
        step: (status.context && status.context.step_tokens) || null,
      },
      download: {
        total: (status && status.total_download_bytes) ?? null,
        unknown: (status && status.download_has_unknown) === true,
        components: status.download_components || null,
      },
      error: (status && status.error) || null,
    });
    if (this._setupSignature === signature && this.setupController) return;
    this._setupSignature = signature;
    this.view = renderSetup(this.mount, status, {
      onSubmit: async (payload) => {
        if (!this.setupController) return;
        this.setupController.setBusy(true, "Starting setup…");
        try { await postSetup(payload); }
        catch (err) {
          this.setupController.setBusy(false, "");
          announce(this.live, `Setup failed: ${err.message || "unknown error"}`);
        }
      },
      onCancel: () => this.pollStatus(),
    });
    this.setupController = this.view;
  }

  _continueReading() {
    const bubble = this._continueBubble;
    if (!bubble || !this.view) return;
    const done = this.speaker.completedSequences();
    const remaining = (bubble._mlxSentences || [])
      .filter(([seq]) => !done.has(seq));
    this.continueChip.hidden = true;
    if (remaining.length === 0) {
      announce(this.live, "Nothing left to read.");
      return;
    }
    // resumeFromTruncation enqueues under the speaker's current turn id,
    // which matches this bubble because its sentences were enqueued there.
    this.speaker.resumeFromTruncation(remaining);
    announce(this.live, "Continuing to read aloud.");
  }

  updateVoiceStates(status) {
    const voice = (status && status.voice) || {};
    const recognition = voice.recognition || {};
    const synthesis = voice.synthesis || {};
    const fallbackState = voice.state || "unknown";
    const synthesisUnusable = synthesis.usable === false
      || synthesis.qualification === false
      || synthesis.qualified === false;
    this.voiceStates = {
      recognition: recognition.state || fallbackState,
      synthesis: synthesisUnusable ? "unqualified"
        : (synthesis.state || fallbackState),
    };
    if (this.composer) this.composer.setVoiceStates(this.voiceStates);
    if (this.view instanceof ConversationView) {
      const synthReady = ["ready", "qualified"].includes(this.voiceStates.synthesis);
      this.view.setSpeakerReady(synthReady);
    }
  }

  stopSpeaking() {
    this.speaker.stop();
    import("./api.js").then(({ cancelVoice }) => cancelVoice("tts").catch(() => {}));
    announce(this.live, "Stopped speaking.");
  }

  async renderChat(status) {
    if (this.view instanceof ConversationView) {
      this.speaker.setConversationId(this.conversation.id);
      this.updateHeader(status);
      return;
    }
    this.mount.replaceChildren();
    this.view = new ConversationView({
      conversation: this.conversation,
      recorder: this.recorder,
      speaker: this.speaker,
      live: this.live,
      jumpChip: this.jumpChip,
    });
    this.view.renderInto(this.mount);
    this.view.attachScrollObserver();
    this.view.onSend = (turn) => this.onSend(turn);
    this.view.onPrefillCompare = (data) => {
      if (this.composer) this.composer.prefillCompare(data);
    };
    this.speaker.setConversationId(this.conversation.id);

    const composerApi = buildComposer({
      onSend: (turn) => this.onSend(turn),
      onCancel: () => this.cancelActiveTurn(),
      onStopSpeaking: () => this.stopSpeaking(),
      onDraft: (text) => this.onSend({ text, mode: "draft" }),
      onOpenContext: () => this._openContextDrawer(),
      onMicStart: async () => { await this.recorder.start(); },
      onMicStop: async () => { await this.recorder.stop(); },
      onMicCancel: async () => {
        await this.recorder.cancel();
        this._micMessage("Recording cancelled.");
      },
      onMicResult: (msg) => this._micMessage(msg.error || "Recording stopped"),
      onConversationModeChange: (enabled) => {
        this.conversationModeEnabled = !!enabled;
        this.speakReplies = !!enabled;
        if (this.view) this.view.setSpeakReplies(enabled);
      },
      onCompareToggle: () => {},
      voiceStates: this.voiceStates,
      isRecording: () => this.recorder.state === "recording",
      isBusy: () => !!this.activeTurn,
      isSpeaking: () => this.speakerSpeaking,
    });
    this.composer = composerApi;
    this.mount.appendChild(composerApi.wrap);
    composerApi.focus();
    if (this.conversationModeEnabled && this.view) {
      this.view.setSpeakReplies(true);
    }
    this.updateVoiceStates(status);
    if (this._pendingTranscript) {
      this._mergeTranscriptIntoComposer(this._pendingTranscript);
      this._pendingTranscript = null;
    }
    this.updateHeader(status);

    this.openStream();
  }

  updateHeader(status) {
    const chip = document.getElementById("pair-chip");
    const badge = document.getElementById("ready-badge");
    const label = document.getElementById("ready-badge-label");
    if (status && status.active_pair) {
      chip.hidden = false;
      chip.textContent = status.active_pair.label || status.active_pair.id;
    } else {
      chip.hidden = true;
    }
    if (status && status.active_pair && status.active_pair.ready_offline) {
      badge.hidden = false;
      badge.dataset.state = "ready";
      label.textContent = "Ready offline";
    } else if (status && status.active_pair) {
      badge.hidden = false;
      badge.dataset.state = "working";
      label.textContent = "Online";
    } else {
      badge.hidden = true;
    }
    const errorEl = document.getElementById("topbar-error");
    if (status && status.error) {
      errorEl.hidden = false;
      errorEl.textContent = status.error.title || status.error.detail || "Issue";
    } else if (this.activeTurn) {
      errorEl.hidden = false;
      errorEl.textContent = "Generating…";
    } else {
      errorEl.hidden = true;
      errorEl.textContent = "";
    }
  }

  async openConversation(id) {
    const conv = await openConversation(id);
    this.conversation = conv;
  }

  async openStream() {
    if (this.streamAbort) this.streamAbort.abort();
    const ctrl = new AbortController();
    this.streamAbort = ctrl;
    const { openEvents } = await import("./api.js");
    try {
      await openEvents(this.conversation.id, this.view.lastSequence, {
        signal: ctrl.signal,
        onEvent: (event) => this.handleEvent(event),
      });
    } catch (_err) {
      if (_err.name === "AbortError") return;
      // Parent closes the events stream after an internal EventGap; the
      // response carries 409 reload:true. Refetch the canonical conversation
      // and resubscribe from the new cursor. Never resubmit POST /turns.
      await this.reloadAfterEventGap();
    }
  }

  async reloadAfterEventGap() {
    try {
      const conv = await openConversation(this.conversation.id);
      this.conversation = conv;
      this.view.setConversation(conv);
      this.view.renderInto(this.mount);
      // renderInto replaces the mount's children, which removes the
      // composer; re-append it so the user can still type and resend.
      if (this.composer) {
        this.composer.wrap.remove();
        this.mount.appendChild(this.composer.wrap);
      }
      this.activeTurn = null;
      this.view.clearTurn();
      if (this.composer) this.composer.setBusy(false);
      announce(this.live, "Connection refreshed; please resend your last message.");
    } catch {
      announce(this.live, "Connection lost; could not refresh. Try again in a moment.");
    }
    setTimeout(() => this.openStream().catch(() => {}), 1500);
  }

  handleEvent({ event, data }) {
    if (!data) return;
    if (data.conversation_id && data.conversation_id !== this.conversation.id) return;
    const seq = asNumber(data.sequence);
    if (seq !== undefined) this.view.lastSequence = seq;
    // The SSE data line carries the whole store event; the payload rides in
    // its .data field. Guard for alternative shapes at the boundary.
    const payload = isPlainObject(data.data) ? data.data : {};
    if (event === "status") {
      this.view.appendAssistant();
      const message = asString(payload.message);
      if (message !== undefined) this.view.appendText(message);
      const state = asString(payload.state);
      if (state === "comparison_draft") {
        // Editable confirmation: the model proposed these; nothing is scored
        // until the user edits and submits them from the Compare panel. The
        // originating material is restored there, visibly editable.
        this.composer?.prefillCompare({ options: payload.options,
                                        criteria: payload.criteria,
                                        source: payload.source });
        announce(this.live, "Draft options ready. Review and edit them in the Compare panel.");
      }
      if (state === "output_truncated" && payload.continue === true) {
        this.view.enableContinue("The response reached the output allowance. Press Continue to keep going.");
      }
      return;
    }
    if (event === "text") {
      const text = asString(payload.text);
      if (text !== undefined) this.view.appendText(text);
      return;
    }
    if (event === "component") {
      if (payload.id != null) payload._id = payload.id;
      this.view.appendCards([payload], { turn_id: data.turn_id });
      return;
    }
    if (event === "decision") {
      this.view.appendCards([payload], { turn_id: data.turn_id });
      return;
    }
    if (event === "done") {
      this.activeTurn = null;
      this.view.clearTurn();
      announce(this.live, "Response complete.");
      this.composer.setBusy(false);
      this.updateHeader(this.status);
      return;
    }
    if (event === "error") {
      this.activeTurn = null;
      this.view.clearTurn();
      this.view.setError({ title: asString(payload.code) || "Response failed",
                           detail: asString(payload.message) || "" });
      announce(this.live, `Response failed: ${payload.code || "unknown error"}`);
      this.composer.setBusy(false);
      this.updateHeader(this.status);
      return;
    }
  }

  _openContextDrawer() {
    if (!this.conversation) return;
    let drawer = document.getElementById("context-drawer");
    if (!drawer) {
      drawer = el("dialog", { class: "drawer", id: "context-drawer",
        "aria-label": "History and constraints" });
      document.body.appendChild(drawer);
    }
    // Whole turn pairs (user + assistant share a turn ID), completed only;
    // the coordinator always includes the current turn itself.
    const messages = this.conversation.messages || [];
    const pairs = [];
    for (let i = 0; i + 1 < messages.length; i += 2) {
      const pair = [messages[i], messages[i + 1]];
      if (messages[i].role === "user" && messages[i + 1].role === "assistant"
          && messages[i].turn_id && messages[i].turn_id === messages[i + 1].turn_id
          && pair.every((m) => m.status === "complete")) {
        pairs.push(pair);
      }
    }
    const context = isPlainObject(this.conversation.context) ? this.conversation.context : {};
    const selection = Array.isArray(context.selected_turn_ids) ? context.selected_turn_ids : null;
    const pinned = asString(context.pinned_constraints) || "";
    const initialMode = selection === null ? "all" : (selection.length === 0 ? "none" : "selected");

    const body = el("div", { class: "drawer__body" });
    const note = el("p", { class: "setup__hint", role: "status" });
    const radios = el("div", { class: "setup__preference", role: "radiogroup",
      "aria-label": "History selection" });
    const radioDefs = [
      ["all", "All history", "Every completed turn is sent back to the model."],
      ["selected", "Selected turns", "Only the checked turn pairs below are sent."],
      ["none", "No history", "The model sees only the current message and pinned constraints."],
    ];
    for (const [value, label, hint] of radioDefs) {
      const id = `ctx-mode-${value}`;
      const input = el("input", { type: "radio", name: "ctx-mode", value, id });
      if (initialMode === value) input.checked = true;
      radios.appendChild(el("label", { for: id }, input, label, el("small", {}, hint)));
    }
    const list = el("ul", { class: "drawer__list", id: "context-turns" });
    for (const pair of pairs) {
      const id = `ctx-turn-${pair[0].turn_id}`;
      const input = el("input", { type: "checkbox", value: pair[0].turn_id, id });
      if (selection && selection.includes(pair[0].turn_id)) input.checked = true;
      input.disabled = initialMode !== "selected";
      const snippet = String(pair[0].content || "").slice(0, 80);
      list.appendChild(el("li", {}, el("label", { for: id }, input,
        snippet || "(empty message)")));
    }
    radios.addEventListener("change", () => {
      const mode = radios.querySelector("input[name=ctx-mode]:checked").value;
      list.querySelectorAll("input[type=checkbox]").forEach((box) => {
        box.disabled = mode !== "selected";
      });
    });
    const pinnedInput = el("textarea", { id: "context-pinned", rows: "3",
      "aria-label": "Pinned constraints",
      placeholder: "Constraints the model must keep visible, e.g. \u201calways answer in French\u201d" });
    pinnedInput.value = pinned;
    const save = el("button", { type: "button", class: "btn" }, "Save context");
    save.addEventListener("click", async () => {
      const mode = radios.querySelector("input[name=ctx-mode]:checked").value;
      const selected = [...list.querySelectorAll("input[type=checkbox]:checked")]
        .map((box) => box.value);
      const payload = {
        selected_turn_ids: mode === "all" ? null : mode === "none" ? [] : selected,
        pinned_constraints: pinnedInput.value,
      };
      save.disabled = true;
      try {
        const record = await postContext(this.conversation.id, payload);
        if (record && record.id) this.conversation = record;
        else this.conversation.context = payload;
        announce(this.live, "Context saved.");
        drawer.close();
      } catch (err) {
        note.textContent = `Could not save: ${err.message || err.code || "unknown error"}`;
      } finally {
        save.disabled = false;
      }
    });
    body.appendChild(note);
    body.appendChild(el("h3", {}, "Sent history"));
    body.appendChild(radios);
    body.appendChild(pairs.length ? list
      : el("p", { class: "drawer__empty" }, "No completed turns yet."));
    body.appendChild(el("h3", {}, "Pinned constraints"));
    body.appendChild(pinnedInput);
    body.appendChild(el("div", { class: "card__actions" }, save));
    drawer.replaceChildren(body);
    drawer.showModal();
  }

  async onSend(turn) {
    if (!this.conversation) return false;
    if (this.activeTurn) {
      // Duplicate send while a response runs: refuse without touching any
      // draft text anywhere in the composer.
      announce(this.live, "Wait for the current response to finish. Your draft is kept.");
      return false;
    }
    let turnId;
    try {
      const { postTurn } = await import("./api.js");
      const resp = await postTurn(this.conversation.id, turn);
      turnId = resp.turn_id;
    } catch (err) {
      if (err.code === 409) {
        announce(this.live, "Another response is already running in this conversation.");
      } else {
        announce(this.live, `Could not send: ${err.message || err.code || "unknown error"}`);
      }
      // Failed send: the composer keeps the draft; nothing was cleared.
      return false;
    }
    this.view.appendUser(turn.text);
    this.activeTurn = turnId;
    this.view.setTurn(turnId);
    this.view.startHeartbeat();
    this.composer.setBusy(true);
    this.updateHeader(this.status);
    announce(this.live, "Assistant is responding.");
    return true;
  }

  async cancelActiveTurn() {
    // Detach the live stream first so the cancelled turn's events stop
    // arriving, then cancel server-side and reset the busy state. Without
    // the reset the composer stayed on Stop forever after Escape.
    if (this.streamAbort) { this.streamAbort.abort(); this.streamAbort = null; }
    await this.view.cancelActiveTurn();
    this.activeTurn = null;
    this.composer.setBusy(false);
    this.updateHeader(this.status);
    this.openStream().catch(() => {});
  }

  async newConversation({ save = false } = {}) {
    if (this.streamAbort) { this.streamAbort.abort(); this.streamAbort = null; }
    this.activeTurn = null;
    if (this.view && this.view.destroy) this.view.destroy();
    const conv = await newConversation(save);
    await this.openConversation(conv.id);
    if (this.status && this.status.state === "ready" && this.status.active_pair) {
      this.renderChat(this.status);
    }
  }
}

const app = new App(document.body);
// The UI component fixture imports this module without a live backend;
// it drives an App instance directly and must not boot the real session.
if (!window.__MLX_UI_FIXTURE__) app.boot();
export { app };
