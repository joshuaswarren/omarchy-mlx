// Executable assertions for the UI boundary helpers. Run directly:
//   bun tests/js/util.test.mjs   (or: node tests/js/util.test.mjs)
// Exit code 0 = all assertions passed; any failure throws.

import assert from "node:assert/strict";
import { asNumber, asString, tokenFromHash } from "../../serve/mlx_omarchy_assistant/static/js/util.js";

// asNumber: primitive JSON numbers must pass through.
assert.equal(asNumber(5), 5);
assert.equal(asNumber(0), 0);
assert.equal(asNumber(-3), -3);
assert.equal(asNumber(5.5), 5.5);
assert.equal(asNumber(1e21), 1e21);
assert.ok(Object.is(asNumber(-0), -0));

// asNumber: everything else must be rejected without coercion.
assert.equal(asNumber(null), undefined);
assert.equal(asNumber(undefined), undefined);
assert.equal(asNumber("5"), undefined);
assert.equal(asNumber(""), undefined);
assert.equal(asNumber("abc"), undefined);
assert.equal(asNumber(true), undefined);
assert.equal(asNumber(false), undefined);
assert.equal(asNumber(NaN), undefined);
assert.equal(asNumber(Infinity), undefined);
assert.equal(asNumber(-Infinity), undefined);
assert.equal(asNumber([5]), undefined);
assert.equal(asNumber({ valueOf: () => 5 }), undefined);

// asString: strings pass, non-strings rejected.
assert.equal(asString("hello"), "hello");
assert.equal(asString(""), "");
assert.equal(asString(5), undefined);
assert.equal(asString(null), undefined);
assert.equal(asString(["a"]), undefined);

// tokenFromHash: the CLI launches /#token=VALUE.
assert.equal(tokenFromHash("#token=abc123"), "abc123");
assert.equal(tokenFromHash("#token=abc123&other=1"), "abc123");
assert.equal(tokenFromHash("#other=1&token=xyz"), "xyz");
assert.equal(tokenFromHash("#token="), "");
assert.equal(tokenFromHash("#"), null);
assert.equal(tokenFromHash(""), null);
assert.equal(tokenFromHash("#foo=bar"), null);
assert.equal(tokenFromHash(null), null);
// URLSearchParams decodes percent escapes in the token value.
assert.equal(tokenFromHash("#token=a%2Bb"), "a+b");

console.log("util js tests passed");
