// Read-aloud playout order and continuity. Run directly:
//   bun tests/js/speak-queue.test.mjs
// The server serialises /api/speak (409 while another speech stream holds
// the GPU), so the queue must request sentences one at a time, in order,
// start the next request as soon as the previous stream closes, and
// schedule every chunk on one playhead: no overlapping sentences, no
// reordering, and no gap while synthesis stays ahead of playback.

import assert from "node:assert/strict";

let now = 0;
let timers = [];
let starts = [];
let requests = [];
let busyUntilRequest = 0;
let serverBusy = false;
let plan = () => [];

function at(time, fn) { timers.push({ time, fn }); }

globalThis.window = {
  setTimeout: (fn, ms) => { const t = { time: now + ms / 1000, fn }; timers.push(t); return t; },
  clearTimeout: (t) => { timers = timers.filter((x) => x !== t); },
};
globalThis.AudioContext = class {
  get currentTime() { return now; }
  get destination() { return {}; }
  createBuffer(_channels, length, rate) {
    return { duration: length / rate, getChannelData: () => ({ set() {} }) };
  }
  createBufferSource() {
    const src = {
      connect() {},
      disconnect() {},
      stop() {},
      start(when) {
        starts.push({ seq: src.seq, at: when, end: when + src.buffer.duration });
        at(when + src.buffer.duration, () => src.onended && src.onended());
      },
    };
    src.seq = currentSeq;
    return src;
  }
};

const RATE = 24000;
let currentSeq = 0;
const pcm = (seconds) => Buffer.alloc(Math.round(seconds * RATE) * 2).toString("base64");

globalThis.fetch = async (_url, { body }) => {
  const { sentence_sequence: seq } = JSON.parse(body);
  requests.push({ seq, at: now });
  if (serverBusy || requests.length <= busyUntilRequest) return { status: 409, ok: false };
  serverBusy = true;
  const encoder = new TextEncoder();
  const chunks = plan(seq);
  const t0 = now;
  const stream = new ReadableStream({
    start(controller) {
      for (const [dt, seconds] of chunks) {
        at(t0 + dt, () => {
          currentSeq = seq;
          const data = { sample_rate: RATE, encoding: "pcm16le", data: pcm(seconds) };
          controller.enqueue(encoder.encode(`event: audio\ndata: ${JSON.stringify(data)}\n\n`));
        });
      }
      at(t0 + chunks[chunks.length - 1][0] + 0.01, () => {
        serverBusy = false;
        controller.enqueue(encoder.encode("event: done\ndata: {}\n\n"));
        controller.close();
      });
    },
  });
  return { status: 200, ok: true, body: stream };
};

const { SpeakQueue } = await import("../../serve/mlx_omarchy_assistant/static/js/voice.js");

async function run(sentences) {
  now = 0; timers = []; starts = []; requests = []; serverBusy = false;
  const done = [];
  const errors = [];
  const queue = new SpeakQueue();
  queue.setConversationId("c");
  queue.attachHooks({ onAudioDone: (d) => done.push({ ...d, at: now }), onError: (e) => errors.push(e.message) });
  for (const seq of sentences) queue.enqueue("t", seq, `Sentence ${seq}.`);
  for (let guard = 0; guard < 5000; guard++) {
    for (let i = 0; i < 4; i++) await new Promise((r) => setTimeout(r, 0));
    if (timers.length === 0) break;
    timers.sort((a, b) => a.time - b.time);
    const t = timers.shift();
    now = Math.max(now, t.time);
    t.fn();
  }
  return { done, errors };
}

function assertSerialPlayout(label) {
  for (let i = 1; i < starts.length; i++) {
    assert.ok(starts[i].seq >= starts[i - 1].seq, `${label}: sentence ${starts[i].seq} played before ${starts[i - 1].seq}`);
    assert.ok(starts[i].at >= starts[i - 1].end - 1e-9,
      `${label}: sentence ${starts[i].seq} starts at ${starts[i].at} while ${starts[i - 1].seq} plays until ${starts[i - 1].end}`);
  }
}

// Long sentences (two 3 s chunks, server done in 2.7 s): the second
// sentence's audio arrives while the first is still playing. It must queue
// behind it on the shared playhead instead of playing over it.
{
  busyUntilRequest = 0;
  plan = () => [[1.3, 3.0], [2.7, 3.0]];
  const { done, errors } = await run([1, 2]);
  assert.deepEqual(errors, []);
  assert.deepEqual(starts.map((s) => s.seq), [1, 1, 2], "the 10 s playback cap stops sentence 2's second chunk");
  assertSerialPlayout("long sentences");
  assert.deepEqual(done.map((d) => d.truncated), [true], "one audio-done for the whole read, marked truncated");
}

// Synthesis faster than playback: sentence 2 is requested when sentence 1's
// stream closes, not after its playout, so playback after the first chunk
// has no gap at all.
{
  busyUntilRequest = 0;
  plan = () => [[1.3, 2.5]];
  const { done, errors } = await run([1, 2, 3]);
  assert.deepEqual(errors, []);
  assert.deepEqual(starts.map((s) => s.seq), [1, 2, 3]);
  assertSerialPlayout("pipelined");
  assert.ok(requests[1].at < starts[0].end, "sentence 2 is requested while sentence 1 still plays");
  for (let i = 1; i < starts.length; i++) {
    assert.ok(Math.abs(starts[i].at - starts[i - 1].end) < 1e-9, `gap before sentence ${starts[i].seq}`);
  }
  assert.equal(done.length, 1);
  assert.ok(done[0].at >= starts[starts.length - 1].end - 1e-9, "audio-done waits for the last chunk to finish");
}

// A busy server (another model operation holds the GPU) retries the SAME
// sentence first; later sentences never jump ahead of it.
{
  busyUntilRequest = 2;
  plan = () => [[1.3, 2.0]];
  const { errors } = await run([1, 2]);
  assert.deepEqual(errors, []);
  assert.deepEqual(requests.map((r) => r.seq), [1, 1, 1, 2]);
  assert.deepEqual(starts.map((s) => s.seq), [1, 2]);
  assertSerialPlayout("busy retry");
}

console.log("speak queue js tests passed");
