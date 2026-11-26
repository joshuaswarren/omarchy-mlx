// Browser end-to-end for speech input over the real app, Chromium fake microphone.
// node e2e.mjs <launch-url> <fake-mic.wav> <out-dir> [denied|latency]
// "latency": at 1440 px, 2 warm-ups then 30 turns of 5.0 s recording; each
// turn times, in the page, the stop click to the transcript in the composer.
// "denied" drops --use-fake-ui-for-media-stream (it auto-accepts every capture
// request), adds --deny-permission-prompts, and runs the permission-denied
// scenario, then the same with getUserMedia rejecting NotAllowedError in-page.
import { spawn } from "node:child_process";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";

const [url, micWav, out, mode] = process.argv.slice(2);
const denied = mode === "denied";
const latencyMode = mode === "latency";
mkdirSync(out, { recursive: true });
const origin = new URL(url).origin;
// Port 0: Chromium picks a free port and writes it to DevToolsActivePort, so
// this run can never attach to a browser another run left behind.
const chrome = spawn("/usr/bin/chromium", [
  "--headless=new", "--remote-debugging-port=0", `--user-data-dir=${out}/profile`,
  "--no-first-run", "--no-default-browser-check", "--disable-crash-reporter", "--disable-breakpad", "--autoplay-policy=no-user-gesture-required",
  "--use-fake-device-for-media-stream", ...(denied ? ["--deny-permission-prompts"] : ["--use-fake-ui-for-media-stream"]),
  `--use-file-for-fake-audio-capture=${micWav}`, "about:blank",
], { stdio: "ignore", detached: true });
const stopChrome = () => { try { process.kill(-chrome.pid, "SIGKILL"); } catch {} };
process.on("exit", stopChrome);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

let wsUrl;
const portFile = `${out}/profile/DevToolsActivePort`;
for (let i = 0; i < 100 && !wsUrl; i++) {
  try {
    const port = existsSync(portFile) ? readFileSync(portFile, "utf8").split("\n")[0] : null;
    if (port) wsUrl = (await (await fetch(`http://127.0.0.1:${port}/json/version`)).json()).webSocketDebuggerUrl;
    else await sleep(100);
  }
  catch { await sleep(100); }
}
const ws = new WebSocket(wsUrl);
await new Promise((r) => ws.addEventListener("open", r, { once: true }));
let nextId = 0;
const pending = new Map();
const listeners = [];
ws.addEventListener("message", (m) => {
  const msg = JSON.parse(m.data);
  if (msg.id && pending.has(msg.id)) {
    const { resolve, reject } = pending.get(msg.id); pending.delete(msg.id);
    msg.error ? reject(new Error(JSON.stringify(msg.error))) : resolve(msg.result);
  } else for (const l of listeners) l(msg);
});
const send = (method, params = {}, sessionId) => new Promise((resolve, reject) => {
  const id = ++nextId; pending.set(id, { resolve, reject });
  ws.send(JSON.stringify({ id, method, params, sessionId }));
});

const { targetId } = await send("Target.createTarget", { url: "about:blank" });
const { sessionId: s } = await send("Target.attachToTarget", { targetId, flatten: true });
const page = (method, params) => send(method, params, s);
let transcribeRequests = 0;
let lastTranscribeId = null;
listeners.push((msg) => {
  if (msg.sessionId === s && msg.method === "Network.requestWillBeSent"
      && msg.params.request.url.endsWith("/api/transcribe")) { transcribeRequests++; lastTranscribeId = msg.params.requestId; }
});
await page("Network.enable");
await page("Page.enable");
await page("Runtime.enable");
// Keep a handle on every capture stream so a scenario can end its track (device loss).
await page("Page.addScriptToEvaluateOnNewDocument", { source: `
  window.__mlxStreams = [];
  const gum = navigator.mediaDevices.getUserMedia.bind(navigator.mediaDevices);
  navigator.mediaDevices.getUserMedia = async (c) => {
    if (window.__mlxDeny) throw new DOMException("Permission denied", "NotAllowedError");
    const st = await gum(c); window.__mlxStreams.push(st); return st; };` });
const evaluate = async (expression) =>
  (await page("Runtime.evaluate", { expression, awaitPromise: true, returnByValue: true })).result.value;
const state = () => evaluate(`({ button: document.querySelector("#mic-btn")?.textContent,
  disabled: document.querySelector("#mic-btn")?.disabled,
  live: document.querySelector("#live-region")?.textContent,
  draft: document.querySelector("#composer-text")?.value })`);
async function waitFor(predicate, ms) {
  const end = Date.now() + ms;
  while (Date.now() < end) { const st = await state(); if (predicate(st)) return st; await sleep(100); }
  return state();
}
async function clickMic() {
  const box = await evaluate(`(() => { const r = document.querySelector("#mic-btn").getBoundingClientRect();
    return { x: r.x + r.width / 2, y: r.y + r.height / 2 }; })()`);
  for (const type of ["mouseMoved", "mousePressed", "mouseReleased"])
    await page("Input.dispatchMouseEvent", { type, x: box.x, y: box.y, button: "left", clickCount: 1 });
}
async function setDraft(text) {
  await evaluate(`(() => { const t = document.querySelector("#composer-text"); t.value = ""; t.focus(); })()`);
  await page("Input.insertText", { text });
}
async function shot(name) {
  const { data } = await page("Page.captureScreenshot", { format: "png" });
  writeFileSync(`${out}/${name}.png`, Buffer.from(data, "base64"));
}

const DRAFT = "Typed draft before speaking.";
const results = [];
if (latencyMode) {
  await page("Emulation.setDeviceMetricsOverride", { width: 1440, height: 900, deviceScaleFactor: 1, mobile: false });
  await send("Browser.setPermission", { origin, permission: { name: "microphone" }, setting: "granted" });
  await page("Page.navigate", { url });
  await waitFor((st) => st.button && st.disabled === false, 60_000);
  await evaluate(`(() => {
    const btn = document.querySelector("#mic-btn"), box = document.querySelector("#composer-text");
    btn.addEventListener("pointerup", () => {
      if (btn.textContent === "Listening…") { window.__tStop = performance.now(); window.__tDone = null; }
    }, true);
    setInterval(() => { if (window.__tStop && !window.__tDone && box.value.length > 0) window.__tDone = performance.now(); }, 5);
  })()`);
  const turns = [];
  for (let i = 0; i < 32; i++) {
    await evaluate(`(() => { document.querySelector("#composer-text").value = ""; window.__tStop = null; window.__tDone = null; })()`);
    const requestsBefore = transcribeRequests;
    await clickMic();
    const afterStart = (await state()).button;
    await sleep(5_000);
    await clickMic();
    const afterStop = (await state()).button;
    const end = Date.now() + 12_000;
    let t = null;
    while (Date.now() < end) {
      t = await evaluate("window.__tDone && window.__tStop ? window.__tDone - window.__tStop : null");
      if (t !== null) break;
      await sleep(50);
    }
    const st = await state();
    turns.push({ i, warmup: i < 2, stop_to_transcript_ms: t === null ? null : Math.round(t * 10) / 10,
                 words: (st.draft || "").split(/\s+/).filter(Boolean).length, live: st.live,
                 afterStart, afterStop, raw: await evaluate("({ stop: window.__tStop, done: window.__tDone })") });
    turns.at(-1).requests = transcribeRequests - requestsBefore;
    if (turns.at(-1).requests) {
      // The exact WAV the page uploaded, kept so an empty result can be checked offline.
      const body = await page("Network.getRequestPostData", { requestId: lastTranscribeId }).catch((e) => ({ error: String(e) }));
      if (body.postData !== undefined)
        writeFileSync(`${out}/upload-${String(i).padStart(2, "0")}.wav`,
                      Buffer.from(body.postData, body.base64Encoded ? "base64" : "latin1"));
      turns.at(-1).upload = body.error ? body.error : { base64Encoded: !!body.base64Encoded };
    }
    if (i === 0) await shot("1440-latency-turn0");
    writeFileSync(`${out}/browser_latency_turns.json`, JSON.stringify(turns, null, 1));
    await sleep(500);
  }
  const ms = turns.filter((x) => !x.warmup).map((x) => x.stop_to_transcript_ms);
  const ok = ms.filter((x) => x !== null).sort((a, b) => a - b);
  const summary = { n: ms.length, missing: ms.length - ok.length,
                    p50_ms: ok[Math.ceil(ok.length / 2) - 1], p95_ms: ok[Math.ceil(ok.length * 0.95) - 1], max_ms: ok.at(-1) };
  writeFileSync(`${out}/browser_latency.json`, JSON.stringify({ summary, turns }, null, 1));
  await shot("1440-latency-last");
  console.log(JSON.stringify(summary));
  ws.close(); stopChrome();
  process.exit(0);
}
for (const width of [1440, 375]) {
  await page("Emulation.setDeviceMetricsOverride", { width, height: width === 375 ? 812 : 900,
    deviceScaleFactor: 1, mobile: width === 375 });
  if (denied) await send("Browser.resetPermissions", {});
  else await send("Browser.setPermission", { origin, permission: { name: "microphone" }, setting: "granted" });
  await page("Page.navigate", { url: width === 1440 ? url : `${origin}/` });
  const ready = await waitFor((st) => st.button && st.disabled === false, 60_000);
  results.push({ width, scenario: "loaded", ...ready });
  await shot(`${width}-0-ready`);

  const run = async (scenario, body) => {
    await setDraft(DRAFT);
    await evaluate(`(() => { window.__mlxLive = [];
      if (!window.__mlxLiveObs) { const r = document.querySelector("#live-region");
        window.__mlxLiveObs = new MutationObserver(() => window.__mlxLive.push(r.textContent));
        window.__mlxLiveObs.observe(r, { childList: true, characterData: true, subtree: true }); } })()`);
    const before = transcribeRequests;
    const t0 = Date.now();
    const final = await body();
    final.announcements = (await evaluate("window.__mlxLive")).filter((a) => !/^Recording \d/.test(a));
    results.push({ width, scenario, ms: Date.now() - t0, transcribeRequests: transcribeRequests - before,
                   draftIntact: final.draft.startsWith(DRAFT), ...final });
    await shot(`${width}-${results.length}-${scenario}`);
  };
  if (!denied) {
  await run("stop-by-user-transcript", async () => {
    await clickMic(); await sleep(9_000);
    const stoppedAt = Date.now(); await clickMic();
    const st = await waitFor((x) => x.draft.length > DRAFT.length + 5, 30_000);
    return { ...st, stopToTranscriptMs: Date.now() - stoppedAt };
  });
  await run("thirty-second-limit", async () => {
    await clickMic();
    const st = await waitFor((x) => x.draft.length > DRAFT.length + 5 && !/Transcribing/.test(x.live || ""), 45_000);
    const log = await evaluate("window.__mlxLive");
    return { ...st, limitAnnounced: log.some((a) => /stopped at the 30 second limit/.test(a)) };
  });
  await run("cancel-escape", async () => {
    await clickMic(); await sleep(2_000);
    await evaluate(`document.querySelector("#composer-text").focus()`);
    for (const type of ["keyDown", "keyUp"])
      await page("Input.dispatchKeyEvent", { type, key: "Escape", code: "Escape", windowsVirtualKeyCode: 27 });
    await sleep(1_500);
    return state();
  });
  await run("device-loss", async () => {
    await clickMic(); await sleep(2_000);
    await evaluate(`window.__mlxStreams.at(-1).getAudioTracks().forEach((t) => { t.stop(); t.dispatchEvent(new Event("ended")); })`);
    return waitFor((x) => /disconnected/.test(x.live || ""), 5_000);
  });
  } else {
  await run("permission-denied", async () => {
    await send("Browser.setPermission", { origin, permission: { name: "microphone" }, setting: "denied" });
    await clickMic();
    return waitFor((x) => /permission/i.test(x.live || ""), 5_000);
  });
    // Headless Chromium granted capture above; cancel that recording first.
    await evaluate(`document.querySelector("#composer-text").focus()`);
    for (const type of ["keyDown", "keyUp"])
      await page("Input.dispatchKeyEvent", { type, key: "Escape", code: "Escape", windowsVirtualKeyCode: 27 });
    await waitFor((x) => x.button === "Microphone", 5_000);
    await run("permission-denied-injected", async () => {
      await evaluate("window.__mlxDeny = true");
      await clickMic();
      return waitFor((x) => /permission/i.test(x.live || ""), 5_000);
    });
  }
}
writeFileSync(`${out}/results.json`, JSON.stringify(results, null, 1));
console.log(JSON.stringify(results.map(({ width, scenario, button, live, draftIntact, transcribeRequests, ms, stopToTranscriptMs }) =>
  ({ width, scenario, button, live, draftIntact, transcribeRequests, ms, stopToTranscriptMs })), null, 1));
ws.close(); stopChrome();
process.exit(0);
