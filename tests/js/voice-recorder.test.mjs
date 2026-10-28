// Recorder upload bounds. Run directly: bun tests/js/voice-recorder.test.mjs
// The recognizer refuses any upload over 30.0 s, so the frame that crosses
// the cap and every frame that arrives while stop() awaits cleanup must stay
// out of the WAV.

import assert from "node:assert/strict";

const tracks = [];
let node = null;
globalThis.window = { AudioWorkletNode: true, setInterval: () => 1 };
globalThis.clearInterval = () => {};
Object.defineProperty(globalThis, "navigator", {
  configurable: true,
  value: {
    mediaDevices: {
      getUserMedia: async () => {
        const track = { addEventListener() {}, stop() {} };
        tracks.push(track);
        return { getAudioTracks: () => [track], getTracks: () => [track] };
      },
    },
  },
});
globalThis.AudioContext = class {
  constructor() { this.sampleRate = 48000; this.currentTime = 0; this.audioWorklet = { addModule: async () => {} }; }
  createMediaStreamSource() { return { connect() {}, disconnect() {} }; }
  close() { return new Promise((resolve) => setTimeout(resolve, 5)); }
};
globalThis.AudioWorkletNode = class {
  constructor() { this.port = {}; node = this; }
  disconnect() {}
};

const { Recorder } = await import("../../serve/mlx_omarchy_assistant/static/js/voice.js");

const stops = [];
const recorder = new Recorder({ onStop: (s) => stops.push(s) });
assert.equal(await recorder.start(), true);

// A 100-sample lead-in then 128-sample worklet frames at 48 kHz: the cap
// (1,440,000 samples) falls mid-frame on frame 11250. Then frames keep
// arriving during cleanup.
const frame = (n = 128) => new Float32Array(n).fill(0.25);
node.port.onmessage({ data: frame(100) });
for (let i = 0; i < 11250; i++) node.port.onmessage({ data: frame() });
for (let i = 0; i < 50; i++) node.port.onmessage({ data: frame() });
await new Promise((resolve) => setTimeout(resolve, 30));

assert.equal(stops.length, 1, "exactly one stop for one cap crossing");
assert.equal(stops[0].reason, "limit");
assert.equal(stops[0].duration, 30);
const wav = new DataView(await stops[0].blob.arrayBuffer());
assert.equal(wav.getUint32(24, true), 16000);
assert.equal(wav.getUint32(40, true), 30 * 16000 * 2, "WAV holds exactly 30.0 s at 16 kHz");
assert.equal(recorder.state, "idle");

// A user stop under the cap keeps every sample.
const recorder2 = new Recorder({ onStop: (s) => stops.push(s) });
await recorder2.start();
for (let i = 0; i < 375; i++) node.port.onmessage({ data: frame() });  // 1.0 s
await recorder2.stop();
assert.equal(stops[1].reason, "user");
assert.equal(stops[1].duration, 1);
assert.equal(new DataView(await stops[1].blob.arrayBuffer()).getUint32(40, true), 16000 * 2);

// Truncation announces once per turn even when both cap paths trip; a
// resume (reset) re-arms it.
const { SpeakQueue } = await import("../../serve/mlx_omarchy_assistant/static/js/voice.js");
{
  let truncations = 0;
  const queue = new SpeakQueue();
  queue.attachHooks({ onTruncate: () => { truncations += 1; } });
  queue._markTruncated();
  queue._markTruncated();
  assert.equal(truncations, 1, "duplicate cap paths must announce once");
  queue.reset();
  queue._markTruncated();
  assert.equal(truncations, 2, "resume must re-arm the truncation event");
}

// ---------------------------------------------------------------------------
// Hands-free (wake word) silence auto-stop.
// ---------------------------------------------------------------------------
const silent = () => new Float32Array(128);

// Speech then 160 ms of silence (60 frames at 48 kHz) stops with the
// named reason; the config is consumed so the next recording runs plain.
{
  const stopsHF = [];
  const hf = new Recorder({ onStop: (s) => stopsHF.push(s) });
  await hf.start();
  hf.setHandsFree({ silenceMs: 160, floorRms: 0.01 });
  for (let i = 0; i < 5; i++) node.port.onmessage({ data: frame() });   // speech
  for (let i = 0; i < 59; i++) node.port.onmessage({ data: silent() }); // not yet
  await new Promise((r) => setTimeout(r, 10));
  assert.equal(stopsHF.length, 0, "silence under the gap must not stop");
  node.port.onmessage({ data: silent() });                              // 160 ms
  await new Promise((r) => setTimeout(r, 10));
  assert.equal(stopsHF.length, 1, "speech-then-gap must stop hands-free");
  assert.equal(stopsHF[0].reason, "silence");
  assert.equal(hf.state, "idle");
  assert.equal(hf._hf, null, "hands-free config is consumed by the stop");

  // Without setHandsFree the same pattern keeps recording (push-to-talk
  // keeps its semantics).
  const plain = new Recorder({ onStop: (s) => stopsHF.push(s) });
  await plain.start();
  for (let i = 0; i < 5; i++) node.port.onmessage({ data: frame() });
  for (let i = 0; i < 80; i++) node.port.onmessage({ data: silent() });
  await new Promise((r) => setTimeout(r, 10));
  assert.equal(stopsHF.length, 1, "silence must not stop a plain recording");
  await plain.stop();
}

// Silence alone (never speech) never auto-stops.
{
  let stopped = false;
  const quiet = new Recorder({ onStop: () => { stopped = true; } });
  await quiet.start();
  quiet.setHandsFree({ silenceMs: 100, floorRms: 0.01 });
  for (let i = 0; i < 120; i++) node.port.onmessage({ data: silent() });
  await new Promise((r) => setTimeout(r, 10));
  assert.equal(stopped, false, "room silence alone must not stop");
  await quiet.stop();
}

// ---------------------------------------------------------------------------
// Startup failure and microphone release (issue #266).
// ---------------------------------------------------------------------------
// Fakes that count resource lifetimes: every track stop() and context close()
// is recorded, and the track keeps the real registered "ended" listener so a
// test can fire device loss the way the browser does. restore() hands the
// shared stubs above back so the earlier scenarios stay untouched.
function micWorld({ gum, contextError, workletError } = {}) {
  const counts = { gumCalls: 0, tracksStopped: 0, contextsClosed: 0 };
  let ended = null;
  const track = {
    addEventListener(type, fn) { if (type === "ended") ended = fn; },
    stop() { counts.tracksStopped++; },
  };
  const world = {
    counts,
    stream: { getAudioTracks: () => [track], getTracks: () => [track] },
    fireDeviceLost: () => ended(),
  };
  const savedGum = navigator.mediaDevices.getUserMedia;
  const SavedContext = globalThis.AudioContext;
  navigator.mediaDevices.getUserMedia = gum ?? (async () => {
    counts.gumCalls++;
    return world.stream;
  });
  globalThis.AudioContext = class {
    constructor() {
      if (contextError) throw contextError;
      this.sampleRate = 48000;
      this.currentTime = 0;
      this.audioWorklet = { addModule: async () => { if (workletError) throw workletError; } };
    }
    createMediaStreamSource() { return { connect() {}, disconnect() {} }; }
    close() { counts.contextsClosed++; return new Promise((r) => setTimeout(r, 5)); }
  };
  world.restore = () => {
    navigator.mediaDevices.getUserMedia = savedGum;
    globalThis.AudioContext = SavedContext;
  };
  return world;
}

// getUserMedia rejecting must surface the named browser error, leave the
// recorder idle, and never reach the upload callback (nothing was acquired).
{
  const upload = [];
  const world = micWorld({ gum: async () => {
    throw Object.assign(new Error("denied"), { name: "NotAllowedError" });
  } });
  try {
    const recorder = new Recorder({ onStop: (s) => upload.push(s) });
    await assert.rejects(recorder.start(), /Microphone permission denied/);
    assert.equal(recorder.state, "idle");
    assert.equal(upload.length, 0, "no upload callback after permission denial");
    navigator.mediaDevices.getUserMedia = async () => {
      throw Object.assign(new Error("gone"), { name: "NotFoundError" });
    };
    await assert.rejects(recorder.start(), /No microphone found/);
    assert.equal(recorder.state, "idle");
    assert.equal(upload.length, 0);
  } finally { world.restore(); }
}

// AudioContext construction failing after the mic was acquired must release
// the stream, and the recorder must start cleanly afterwards.
{
  const upload = [];
  const world = micWorld({ contextError: new Error("no audio host") });
  try {
    const recorder = new Recorder({ onStop: (s) => upload.push(s) });
    await assert.rejects(recorder.start(), /no audio host/);
    assert.equal(recorder.state, "idle");
    assert.equal(world.counts.tracksStopped, 1, "acquired stream stopped on context failure");
    assert.equal(upload.length, 0);
    world.restore();
    assert.equal(await recorder.start(), true, "recorder starts again after the failure");
    await recorder.stop();
    assert.equal(upload.length, 1);
  } finally { world.restore(); }
}

// audioWorklet.addModule failing after the mic was acquired must stop the
// track and close the context; a later start still works.
{
  const world = micWorld({ workletError: new Error("worklet module rejected") });
  try {
    const recorder = new Recorder({ onStop: () => {} });
    await assert.rejects(recorder.start(), /worklet module rejected/);
    assert.equal(recorder.state, "idle");
    assert.equal(world.counts.tracksStopped, 1, "acquired track stopped on worklet failure");
    assert.equal(world.counts.contextsClosed, 1, "created context closed on worklet failure");
    world.restore();
    assert.equal(await recorder.start(), true, "recorder starts again after the failure");
    await recorder.stop();
  } finally { world.restore(); }
}

// A second start while acquisition is pending is refused (one capture at
// most), and a cancel during the pending acquisition releases the mic when
// it finally arrives instead of going live.
{
  let release;
  const gate = new Promise((resolve) => { release = resolve; });
  const world = micWorld({ gum: () => {
    world.counts.gumCalls++;
    return gate.then(() => world.stream);
  } });
  try {
    const recorder = new Recorder({ onStop: () => {} });
    const pending = recorder.start();
    assert.equal(await recorder.start(), false, "second start refused while acquiring");
    assert.equal(world.counts.gumCalls, 1, "at most one capture is acquired");
    await recorder.cancel();
    release();
    assert.equal(await pending, false, "cancelled acquisition never records");
    assert.equal(recorder.state, "idle");
    assert.equal(world.counts.tracksStopped, 1, "late-arriving stream released after cancel");
    assert.equal(world.counts.contextsClosed, 1);
  } finally { world.restore(); }
}

// Device loss mid-recording: firing the track's registered "ended" notifies
// onDeviceLost exactly once, releases the resources, and skips the upload.
{
  const upload = [];
  const lost = [];
  const world = micWorld();
  try {
    const recorder = new Recorder({
      onStop: (s) => upload.push(s),
      onDeviceLost: (e) => lost.push(e),
    });
    assert.equal(await recorder.start(), true);
    world.fireDeviceLost();
    await new Promise((resolve) => setTimeout(resolve, 30));
    assert.equal(lost.length, 1, "one device-loss notification");
    assert.equal(lost[0].reason, "device");
    assert.equal(upload.length, 0, "no transcription upload on device loss");
    assert.equal(recorder.state, "idle");
    assert.equal(world.counts.tracksStopped, 1, "track stopped after device loss");
    assert.equal(world.counts.contextsClosed, 1, "context closed after device loss");
  } finally { world.restore(); }
}

console.log("voice-recorder hands-free checks passed");

console.log("voice recorder js tests passed");
