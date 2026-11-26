// Executable assertions for followEvents (the chat events-stream follower). Run directly:
//   bun tests/js/events-follow.test.mjs   (or: node tests/js/events-follow.test.mjs)
//
// The server ends the events stream when no turn runs and after 15 s while one
// runs. followEvents must reopen from the last sequence while a turn is active
// and stop when it is not.

import assert from "node:assert/strict";
import { followEvents } from "../../serve/mlx_omarchy_assistant/static/js/api.js";

const noSleep = async () => {};

// 1. A turn starts, the first stream ends with no event (server had not begun
//    the turn), the second carries text, the third carries done and the turn
//    ends: three opens, each from the last sequence seen.
{
  let seq = 0;
  let active = true;
  const opens = [];
  const seen = [];
  const script = [
    [],
    [{ sequence: 1, type: "text" }, { sequence: 2, type: "text" }],
    [{ sequence: 3, type: "done" }],
  ];
  await followEvents({
    open: async (after, { onEvent }) => {
      opens.push(after);
      for (const e of script[opens.length - 1]) {
        seq = e.sequence;
        seen.push(e.type);
        onEvent({ event: e.type, data: e });
        if (e.type === "done") active = false;
      }
    },
    getAfter: () => seq,
    isActive: () => active,
    sleep: noSleep,
    onEvent: () => {},
  });
  assert.deepEqual(opens, [0, 0, 2]);
  assert.deepEqual(seen, ["text", "text", "done"]);
}

// 2. Idle conversation: one open, no reopen.
{
  let opens = 0;
  await followEvents({
    open: async () => { opens += 1; },
    getAfter: () => 0,
    isActive: () => false,
    sleep: noSleep,
    onEvent: () => {},
  });
  assert.equal(opens, 1);
}

// 3. Back off while streams carry nothing, reset once events arrive.
{
  let seq = 0;
  let calls = 0;
  const delays = [];
  await followEvents({
    open: async () => {
      calls += 1;
      if (calls === 4) seq = 1;
    },
    getAfter: () => seq,
    isActive: () => calls < 6,
    sleep: async (ms) => { delays.push(ms); },
    onEvent: () => {},
  });
  assert.deepEqual(delays, [500, 1000, 2000, 250, 500]);
}

// 4. The delay is capped.
{
  let calls = 0;
  const delays = [];
  await followEvents({
    open: async () => { calls += 1; },
    getAfter: () => 0,
    isActive: () => calls < 8,
    sleep: async (ms) => { delays.push(ms); },
    onEvent: () => {},
  });
  assert.equal(Math.max(...delays), 4000);
}

// 5. An abort ends the loop without another open.
{
  const ctrl = new AbortController();
  let opens = 0;
  await followEvents({
    open: async () => { opens += 1; ctrl.abort(); },
    getAfter: () => 0,
    isActive: () => true,
    signal: ctrl.signal,
    sleep: noSleep,
    onEvent: () => {},
  });
  assert.equal(opens, 1);
}

// 6. An error from open propagates, so the caller can refetch the conversation.
{
  await assert.rejects(
    followEvents({
      open: async () => { throw new Error("events failed (409)"); },
      getAfter: () => 0,
      isActive: () => true,
      sleep: noSleep,
      onEvent: () => {},
    }),
    /events failed \(409\)/,
  );
}

console.log("events follow js tests passed");
