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

console.log("voice recorder js tests passed");
