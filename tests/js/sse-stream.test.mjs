// Byte framing of the assistant's SSE reader, through the public openEvents
// and streamSpeak APIs. Run directly:
//   bun tests/js/sse-stream.test.mjs   (or: node tests/js/sse-stream.test.mjs)
//
// The controlled fetch returns a real Response over a real ReadableStream of
// UTF-8 bytes, so every case goes through the private reader's decoding, line
// splitting and event accumulation. Network chunks can split a delimiter, a
// JSON value or a multibyte character; each event must still arrive once, in
// order, and a stream that ends or fails mid-event must not emit a partial or
// synthetic event. No model or GPU is needed.

import assert from "node:assert/strict";
import { openEvents, streamSpeak } from "../../serve/mlx_omarchy_assistant/static/js/api.js";

const encoder = new TextEncoder();
const bytes = (text) => encoder.encode(text);
const concat = (...parts) => {
  const out = new Uint8Array(parts.reduce((n, p) => n + p.length, 0));
  let at = 0;
  for (const p of parts) { out.set(p, at); at += p.length; }
  return out;
};
const indexOfBytes = (haystack, needle) => {
  outer: for (let i = 0; i + needle.length <= haystack.length; i++) {
    for (let j = 0; j < needle.length; j++) if (haystack[i + j] !== needle[j]) continue outer;
    return i;
  }
  return -1;
};

function deferred() {
  let resolve;
  const promise = new Promise((r) => { resolve = r; });
  return { promise, resolve };
}

// Expected events, written out by hand rather than derived through the
// parser under test.
const TEXT = "café \u{1F600}";
const EXPECTED = [
  { event: "delta", data: { seq: 1, text: "café \u{1F600}" } },
  { event: "audio", data: { data: "AAE=", sample_rate: 16000, encoding: "pcm16le" } },
  { event: "done", data: { seq: 2 } },
];

// The canonical stream: three blank-line-terminated events. U+00E9 and
// U+1F600 go on the wire as raw UTF-8 bytes, not JSON escapes.
const DELTA_LF = `event: delta\ndata: {"seq":1,"text":"${TEXT}"}\n\n`;
const CANONICAL_LF = bytes(
  DELTA_LF +
  'event: audio\ndata: {"data":"AAE=","sample_rate":16000,"encoding":"pcm16le"}\n\n' +
  'event: done\ndata: {"seq":2}\n\n',
);
assert.ok(indexOfBytes(CANONICAL_LF, [0xc3, 0xa9]) > 0, "U+00E9 is on the wire as UTF-8");
assert.ok(indexOfBytes(CANONICAL_LF, [0xf0, 0x9f, 0x98, 0x80]) > 0, "U+1F600 is on the wire as UTF-8");

// The same events with CRLF line endings, comment/heartbeat lines, and the
// delta's JSON split across two data: lines after a comma. The reader joins
// data lines with "\n", which is legal JSON whitespace there.
const CANONICAL_CRLF = bytes(
  ": heartbeat\r\n" +
  "\r\n" +
  'event: delta\r\ndata: {"seq":1,\r\n' +
  ": mid-event comment\r\n" +
  `data: "text":"${TEXT}"}\r\n\r\n` +
  'event: audio\r\ndata: {"data":"AAE=","sample_rate":16000,"encoding":"pcm16le"}\r\n\r\n' +
  ": keep-alive\r\n\r\n" +
  'event: done\r\ndata: {"seq":2}\r\n\r\n',
);

// A response whose body hands out one scripted step per read. With a
// high-water mark of 0 the stream pulls only while the reader waits, so a
// step runs after every earlier chunk has reached the reader.
const enqueue = (chunk) => (controller) => controller.enqueue(chunk);
const close = () => (controller) => controller.close();
const fail = (error) => (controller) => controller.error(error);

function scriptedResponse(steps) {
  let next = 0;
  const body = new ReadableStream({
    pull(controller) {
      const step = steps[next++];
      if (!step) throw new Error("fixture: read past the end of the script");
      return step(controller);
    },
  }, { highWaterMark: 0 });
  return new Response(body, { status: 200, headers: { "Content-Type": "text/event-stream" } });
}

function chunkedResponse(chunks) {
  return scriptedResponse([...chunks.map(enqueue), close()]);
}

// Replace fetch with one that returns the given response, and restore it
// after the case whether it passed or failed.
async function withFetch(makeResponse, run) {
  const original = globalThis.fetch;
  globalThis.fetch = async (_url, init = {}) => makeResponse(init);
  try {
    return await run();
  } finally {
    globalThis.fetch = original;
  }
}

const APIS = [
  ["openEvents", (options) => openEvents("c1", 0, options)],
  ["streamSpeak", (options) => streamSpeak(
    { conversation_id: "c1", turn_id: "t1", text: "Hello.", sentence_sequence: 1 }, options)],
];

async function collect(call, makeResponse, signal) {
  const events = [];
  await withFetch(makeResponse, () => call({ signal, onEvent: (e) => events.push(e) }));
  return events;
}

function partitions(stream) {
  const out = [["one chunk", [stream]]];
  out.push(["one-byte chunks", Array.from(stream, (b) => Uint8Array.of(b))]);
  for (let i = 1; i < stream.length; i++) {
    out.push([`split at byte ${i}`, [stream.subarray(0, i), stream.subarray(i)]]);
  }
  return out;
}

// sse_utf8_chunk_partitions_preserve_events
{
  for (const [api, call] of APIS) {
    for (const [label, chunks] of partitions(CANONICAL_LF)) {
      const events = await collect(call, () => chunkedResponse(chunks));
      assert.deepEqual(events, EXPECTED, `${api}, ${label}`);
      const text = events[0].data.text;
      assert.ok(text.includes("é") && text.includes("\u{1F600}"), `${api}, ${label}: code points kept`);
      assert.ok(!text.includes("�"), `${api}, ${label}: no replacement character`);
    }
  }
}

// sse_crlf_comments_and_multiline_data
{
  const crlf = indexOfBytes(CANONICAL_CRLF, bytes("}\r\n\r\nevent: audio"));
  const betweenCrAndLf = crlf + 4; // "}\r\n\r" | "\nevent: audio"
  assert.equal(CANONICAL_CRLF[betweenCrAndLf - 1], 0x0d);
  assert.equal(CANONICAL_CRLF[betweenCrAndLf], 0x0a);
  for (const [api, call] of APIS) {
    const events = await collect(call, () => chunkedResponse([
      CANONICAL_CRLF.subarray(0, betweenCrAndLf), CANONICAL_CRLF.subarray(betweenCrAndLf),
    ]));
    assert.deepEqual(events, EXPECTED, `${api}: chunk boundary between CR and LF`);
    for (const [label, chunks] of partitions(CANONICAL_CRLF)) {
      const got = await collect(call, () => chunkedResponse(chunks));
      assert.deepEqual(got, EXPECTED, `${api}, CRLF, ${label}`);
    }
  }
}

// sse_incomplete_eof_does_not_emit_terminal_success
{
  const delta = bytes(DELTA_LF);
  const cases = [
    ["done without its blank line", bytes('event: done\ndata: {"seq":2}\n')],
    ["done without any line end", bytes('event: done\ndata: {"seq":2}')],
    ["JSON cut mid-value", bytes('event: delta\ndata: {"seq":2,"text":"ca')],
    ["JSON cut mid-UTF-8 sequence", concat(bytes('event: delta\ndata: {"seq":2,"text":"caf'), Uint8Array.of(0xc3))],
  ];
  for (const [api, call] of APIS) {
    for (const [label, tail] of cases) {
      for (const chunks of [[delta, tail], [concat(delta, tail)]]) {
        const events = await collect(call, () => chunkedResponse(chunks));
        assert.deepEqual(events, [EXPECTED[0]], `${api}, ${label}: only the complete delta`);
        assert.ok(!events.some((e) => e.event === "done" || e.data === null), `${api}, ${label}`);
      }
    }
  }
}

// sse_read_failure_preserves_completed_prefix
{
  for (const [api, call] of APIS) {
    const sentinel = new Error("connection reset");
    const events = [];
    await assert.rejects(
      withFetch(
        () => scriptedResponse([
          enqueue(bytes(DELTA_LF)),
          enqueue(bytes('event: done\ndata: {"se')),
          fail(sentinel),
        ]),
        () => call({ onEvent: (e) => events.push(e) }),
      ),
      (err) => err === sentinel,
      `${api}: rejects with the stream's error`,
    );
    assert.deepEqual(events, [EXPECTED[0]], `${api}: only the complete delta`);
  }
}

// sse_abort_during_pending_read_emits_nothing_late
//
// This models fetch-body cancellation by erroring the real stream with the
// signal's AbortError. It tests consumption of an aborted response body, not
// native browser fetch cancellation.
{
  for (const [api, call] of APIS) {
    const ctrl = new AbortController();
    const delivered = deferred();
    const readPending = deferred();
    const events = [];
    let listening = false;
    let removeListener = () => {};
    const makeResponse = (init) => {
      assert.equal(init.signal, ctrl.signal, `${api}: fetch receives the caller's signal`);
      let streamController;
      const onAbort = () => streamController.error(init.signal.reason);
      removeListener = () => { init.signal.removeEventListener("abort", onAbort); listening = false; };
      init.signal.addEventListener("abort", onAbort);
      listening = true;
      return scriptedResponse([
        enqueue(bytes(DELTA_LF)),
        (controller) => { streamController = controller; readPending.resolve(); },
      ]);
    };
    const pending = withFetch(makeResponse, () => call({
      signal: ctrl.signal,
      onEvent: (e) => { events.push(e); delivered.resolve(); },
    }));
    try {
      await delivered.promise;
      await readPending.promise;
      ctrl.abort();
      await assert.rejects(pending, (err) => err.name === "AbortError", `${api}: rejects with AbortError`);
    } finally {
      removeListener();
    }
    assert.equal(listening, false, `${api}: fixture removed its abort listener`);
    assert.deepEqual(events, [EXPECTED[0]], `${api}: no later event and no repeated delta`);
  }
}

console.log("sse stream js tests passed");
