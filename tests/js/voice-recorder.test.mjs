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

console.log("voice-recorder hands-free checks passed");

console.log("voice recorder js tests passed");
