// API client: same-origin fetch with CSRF, SSE reader, heartbeat.
//
// Contract reminders (parent owns the contract):
//   GET  /api/session            -> {csrf, session_id}
//   POST /api/session            -> exchanges the launch fragment for a cookie
//   GET  /api/status             -> {state, pairs, active_pair, context, voice, theme, error?}
//   GET  /api/theme              -> same shape as status.theme
//   POST /api/setup              -> async; progress via /api/status
//   POST /api/conversations      -> {id, messages: []}
//   GET  /api/conversations      -> {conversations: []}
//   GET  /api/conversations/:id  -> state incl. messages, last_sequence
//   DELETE /api/conversations/:id
//   GET  /api/conversations/:id/export  -> JSON
//   PATCH /api/conversations/:id {save:bool}
//   POST /api/conversations/:id/turns  {text, mode, options?, criteria?, max_tokens?}
//        -> {turn_id}; 409 if a turn is already running.
//   GET  /api/conversations/:id/events?after=N
//        SSE named events: status, decision, text, component, audio, error, done
//        409 indicates the parent has closed the stream (EventGap recovery):
//        caller refetches GET /conversations/:id and resubscribes — never
//        resubmit POST /turns.
//   POST /api/conversations/:id/cancel     {turn_id}
//   POST /api/conversations/:id/heartbeat  {turn_id}  (every 5s while work runs)
//   POST /api/conversations/:id/actions    {turn_id, component_id, revision, action_id, action, values}
//   POST /api/conversations/:id/context    {selected_turn_ids: null|[]|[turnId], pinned_constraints: str}
//        -> the record; selection null means all completed history, [] none;
//           the current turn is always included server-side.
//   POST /api/transcribe binary audio/wav   -> {text}
//   POST /api/speak same path, SSE stream {conversation_id, turn_id, text, sentence_sequence}
//        events: audio{data:base64 pcm16le, sample_rate, encoding}, done, error

const csrfHeader = "X-Assistant-CSRF";
let csrfToken = "";
let sessionId = "";

function swallow() { return undefined; }

function tryParseJSON(text) {
  try {
    return JSON.parse(text);
  } catch {
    return null;
  }
}

export function setSession({ csrf = csrfToken, session_id = sessionId } = {}) {
  if (csrf) csrfToken = csrf;
  if (session_id) sessionId = session_id;
}

export function currentSession() {
  return { csrf: csrfToken, session_id: sessionId };
}

async function request(path, options = {}) {
  const headers = new Headers(options.headers || {});
  if (options.body && !headers.has("Content-Type") &&
      !(options.body instanceof FormData) && !(options.body instanceof Blob) &&
      !(options.body instanceof ArrayBuffer)) {
    headers.set("Content-Type", "application/json");
  }
  if (csrfToken && options.method && options.method !== "GET") {
    headers.set(csrfHeader, csrfToken);
  }
  const res = await fetch(path, { ...options, headers, credentials: "same-origin" });
  if (res.status === 401) {
    const err = new Error("session invalid");
    err.code = "unauthorised";
    throw err;
  }
  return res;
}

export async function exchangeFragment(fragment) {
  // POST /api/session exchanges the launch fragment. The fragment must never
  // be logged, written to localStorage, or echoed back in any URL.
  const res = await request("/api/session", {
    method: "POST",
    body: JSON.stringify({ token: fragment }),
  });
  if (!res.ok) {
    const err = new Error(`session exchange failed (${res.status})`);
    err.code = "exchange_failed";
    throw err;
  }
  return res.json().catch(() => ({}));
}

export async function fetchSession() {
  const res = await request("/api/session");
  if (!res.ok) {
    if (res.status === 401) return null;
    throw new Error(`session lookup failed (${res.status})`);
  }
  const data = await res.json();
  if (data && data.csrf) csrfToken = data.csrf;
  if (data && data.session_id) sessionId = data.session_id;
  return data;
}

export async function fetchStatus() {
  const res = await request("/api/status");
  if (!res.ok) throw new Error(`status failed (${res.status})`);
  return res.json();
}

export async function fetchTheme() {
  const res = await request("/api/theme");
  if (!res.ok) throw new Error(`theme failed (${res.status})`);
  return res.json();
}

export async function postSetup(payload) {
  const res = await request("/api/setup", {
    method: "POST",
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    const err = new Error(`setup failed (${res.status})`);
    err.code = res.status;
    await res.json().then((d) => { err.detail = d; }, swallow);
    throw err;
  }
  return res.json().then((d) => d, swallow);
}

export async function listConversations() {
  const res = await request("/api/conversations");
  if (!res.ok) throw new Error(`history list failed (${res.status})`);
  return res.json();
}

export async function openConversation(id) {
  const res = await request(`/api/conversations/${encodeURIComponent(id)}`);
  if (!res.ok) throw new Error(`conversation open failed (${res.status})`);
  return res.json();
}

export async function newConversation(save = false) {
  const res = await request("/api/conversations", {
    method: "POST",
    body: JSON.stringify({ save }),
  });
  if (!res.ok) throw new Error(`new conversation failed (${res.status})`);
  return res.json();
}

export async function patchConversation(id, body) {
  const res = await request(`/api/conversations/${encodeURIComponent(id)}`, {
    method: "PATCH",
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`conversation patch failed (${res.status})`);
  return res.json().catch(() => ({}));
}

export async function deleteConversation(id) {
  const res = await request(`/api/conversations/${encodeURIComponent(id)}`, {
    method: "DELETE",
  });
  if (!res.ok) throw new Error(`conversation delete failed (${res.status})`);
  return res;
}

export async function exportConversationUrl(id) {
  return `/api/conversations/${encodeURIComponent(id)}/export`;
}

export async function postTurn(conversationId, turn) {
  const res = await request(`/api/conversations/${encodeURIComponent(conversationId)}/turns`, {
    method: "POST",
    body: JSON.stringify(turn),
  });
  if (!res.ok) {
    const err = new Error(`turn failed (${res.status})`);
    err.code = res.status;
    await res.json().then((d) => { err.detail = d; }, swallow);
    throw err;
  }
  return res.json();
}

export async function postContext(conversationId, context) {
  const res = await request(`/api/conversations/${encodeURIComponent(conversationId)}/context`, {
    method: "POST",
    body: JSON.stringify(context),
  });
  if (!res.ok) {
    const err = new Error(`context update failed (${res.status})`);
    err.code = res.status;
    await res.json().then((d) => { err.detail = d && d.error; }, swallow);
    throw err;
  }
  return res.json().catch(() => ({}));
}

export async function cancelTurn(conversationId, turnId) {
  const res = await request(`/api/conversations/${encodeURIComponent(conversationId)}/cancel`, {
    method: "POST",
    body: JSON.stringify({ turn_id: turnId }),
  });
  if (!res.ok && res.status !== 409) {
    throw new Error(`cancel failed (${res.status})`);
  }
  return res;
}

export async function heartbeat(conversationId, turnId) {
  const res = await request(`/api/conversations/${encodeURIComponent(conversationId)}/heartbeat`, {
    method: "POST",
    body: JSON.stringify({ turn_id: turnId }),
  });
  return res;
}

export async function postAction(conversationId, action) {
  const res = await request(`/api/conversations/${encodeURIComponent(conversationId)}/actions`, {
    method: "POST",
    body: JSON.stringify(action),
  });
  if (!res.ok) {
    const err = new Error(`action failed (${res.status})`);
    err.code = res.status;
    await res.json().then((d) => { err.detail = d; }, swallow);
    throw err;
  }
  return res.json().then((d) => d, swallow);
}

export async function transcribe(wavBlob) {
  const res = await request("/api/transcribe", {
    method: "POST",
    headers: { "Content-Type": "audio/wav" },
    body: wavBlob,
  });
  if (!res.ok) throw new Error(`transcribe failed (${res.status})`);
  return res.json();
}

export async function streamSpeak({ conversation_id, turn_id, text, sentence_sequence }, { signal, onEvent } = {}) {
  // Streaming TTS on the SAME path: SSE events audio{data:base64 pcm16le,
  // sample_rate, encoding}, done{...}, error{...}. 409 before the stream
  // starts means the GPU is busy; the caller parks and retries.
  const headers = { "Content-Type": "application/json" };
  if (csrfToken) headers[csrfHeader] = csrfToken;
  const res = await fetch("/api/speak", {
    method: "POST",
    credentials: "same-origin",
    headers,
    body: JSON.stringify({ conversation_id, turn_id, text, sentence_sequence }),
    signal,
  });
  if (res.status === 409) {
    const err = new Error("speak busy");
    err.code = "busy";
    throw err;
  }
  if (!res.ok || !res.body) throw new Error(`speak failed (${res.status})`);
  await readSseStream(res.body, onEvent);
}

export async function cancelVoice(kind) {
  // Parent-owned endpoint; the server is the only party that can confirm a
  // stop succeeded. Surface every failure (no swallowed try/catch) so the
  // caller can tell the user when cancellation could not be confirmed.
  return request("/api/voice/cancel", {
    method: "POST",
    body: JSON.stringify({ kind }),
  });
}

export async function setVoice(voiceId) {
  // Same-origin JSON route: validated against the pinned pack; an unknown
  // voice comes back as a 400 with the offending id named, never a swap.
  return request("/api/voice", {
    method: "POST",
    body: JSON.stringify({ voice: voiceId }),
  });
}

export async function previewVoice() {
  // Renders one fixed sentence in the currently chosen voice and returns
  // {sample_rate, encoding: "pcm16le", data: base64}. 409 before the
  // response starts means the GPU is busy; the caller announces that and
  // waits for the next user gesture.
  const headers = { "Content-Type": "application/json" };
  if (csrfToken) headers[csrfHeader] = csrfToken;
  const res = await fetch("/api/voice/preview", {
    method: "POST", credentials: "same-origin", headers, body: "{}",
  });
  if (res.status === 409) {
    const err = new Error("preview busy");
    err.code = "busy";
    throw err;
  }
  if (!res.ok) {
    let message = `preview failed (${res.status})`;
    try {
      const body = await res.json();
      if (body && body.error) message = body.error;
    } catch { /* ignore non-JSON */ }
    throw new Error(message);
  }
  return res.json();
}

export async function fetchTransfer() {
  const res = await request("/api/transfer");
  if (!res.ok) throw new Error(`transfer status failed (${res.status})`);
  return res.json();
}

export async function postTransfer(payload) {
  const res = await request("/api/transfer", {
    method: "POST",
    body: JSON.stringify(payload),
  });
  if (!res.ok && res.status !== 202) {
    const err = new Error(`transfer request failed (${res.status})`);
    err.code = res.status;
    await res.json().then((d) => { err.detail = d; }, swallow);
    throw err;
  }
  return res.json().then((d) => d, swallow);
}

async function readSseStream(body, onEvent) {
  const reader = body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";
  let current = { event: "message", data: "" };
  const handleLine = (line) => {
    if (!line) {
      if (current.data) {
        const safe = tryParseJSON(current.data);
        if (onEvent) onEvent({ event: current.event, data: safe });
      }
      current = { event: "message", data: "" };
      return;
    }
    if (line.startsWith(":")) return;
    const colon = line.indexOf(":");
    const field = colon === -1 ? line : line.slice(0, colon);
    const value = colon === -1 ? "" : line.slice(colon + 1).replace(/^ /, "");
    if (field === "event") current.event = value;
    else if (field === "data") current.data = current.data ? current.data + "\n" + value : value;
  };
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() ?? "";
    for (const line of lines) handleLine(line);
  }
  if (buffer) handleLine(buffer);
}

// ---------------------------------------------------------------------------
// SSE — fetch-based so the same-origin CSRF header can be attached and the
// stream can be cancelled on demand. Reconnect uses `after=` (last sequence
// number seen) and never resubmits any turn.
// ---------------------------------------------------------------------------

export async function openEvents(conversationId, after, { signal, onEvent } = {}) {
  const params = new URLSearchParams({ after: String(after || 0) });
  const res = await fetch(`/api/conversations/${encodeURIComponent(conversationId)}/events?${params}`, {
    credentials: "same-origin",
    headers: csrfToken ? { [csrfHeader]: csrfToken } : {},
    signal,
  });
  if (!res.ok || !res.body) throw new Error(`events failed (${res.status})`);
  await readSseStream(res.body, onEvent);
}
