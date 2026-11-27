"""Offline tests for scripts/mirror_community_data.py."""

import contextlib
import http.client
import io
import json
import os
import sys
import tempfile
import threading
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import mirror_community_data as mirror  # noqa: E402

BASE_URL = "https://example.invalid"

# Balanced, so a decoder with room to spare would return a dict.
DEEP_JSON = '{"a":' * 200_000 + "1" + "}" * 200_000


def run_with_small_stack(fn):
    """Return fn() run in a thread with a pinned 4 MiB stack.

    Python 3.14 bounds JSON nesting by the C stack, which follows
    ulimit -s; the pinned stack makes DEEP_JSON overflow on any host.
    """
    outcome = {}

    def target():
        try:
            outcome["value"] = fn()
        except BaseException as exc:  # re-raised in the caller
            outcome["error"] = exc

    previous = threading.stack_size(4 * 1024 * 1024)
    try:
        thread = threading.Thread(target=target)
        thread.start()
    finally:
        threading.stack_size(previous)
    thread.join()
    if "error" in outcome:
        raise outcome["error"]
    return outcome["value"]


class _Response(io.BytesIO):
    """A 200 response: status, headers and the unread byte count."""

    status = 200

    def __init__(self, body):
        super().__init__(body)
        self.headers = http.client.HTTPMessage()
        self.headers["Content-Length"] = str(len(body))
        self._size = len(body)

    @property
    def length(self):
        return self._size - self.tell()


def _fake_urlopen(bodies):
    def urlopen(request, timeout=None):
        return _Response(bodies[request.full_url])
    return urlopen


def _sha(digit):
    return str(digit) * 64


def _summary_rows(records):
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "SUMMARY.md"
        mirror.write_summary(
            out,
            BASE_URL,
            records,
            {"generated_at": "2026-09-03T18:09:05Z", "schema_version": 1,
             "count": len(records)},
        )
        text = out.read_text(encoding="utf-8")
    rows = {}
    for line in text.splitlines():
        if line.startswith("| ["):
            short = line.split("[", 1)[1].split("]", 1)[0]
            rows[short] = line
    return text, rows


class MirrorSummaryArchiveLinkTests(unittest.TestCase):
    def test_quick_row_keeps_result_link_without_archive_link(self):
        sha = _sha("1")
        quick = {
            "content_sha256": sha,
            "schema_version": 1,
            "kind": "quick",
            "chip": "apple,t8103",
            "kernel": "7.1.6-1-1-ARCH",
            "mesa_driver": "Honeykrisp",
            "mlx_version": "0.32.2",
            "benchmark": [],
            "files": [],
        }
        text, rows = _summary_rows([quick])
        expected = (
            f"| [{sha[:12]}]({BASE_URL}/v1/results/{sha}) "
            f"| quick | apple,t8103 | 7.1.6-1-1-ARCH | Honeykrisp | 0.32.2 | 0 |"
        )
        self.assertEqual(rows[sha[:12]], expected)
        self.assertNotIn(f"/v1/results/{sha}/archive", text)

    def test_deep_and_e2e_rows_keep_archive_links(self):
        deep_sha = _sha("2")
        deep_zero_sha = _sha("3")
        e2e_sha = _sha("4")
        records = [
            {
                "content_sha256": deep_sha,
                "kind": "deep",
                "benchmark": [{"n": 256, "tflops": 1.0, "median_ms": 1.0}],
            },
            {
                "content_sha256": deep_zero_sha,
                "kind": "deep",
                "benchmark": [],
            },
            {
                "content_sha256": e2e_sha,
                "kind": "omarchy-mac-e2e",
                "benchmark": [],
            },
        ]
        _, rows = _summary_rows(records)
        self.assertIn(
            f"| [2]({BASE_URL}/v1/results/{deep_sha}/archive) |",
            rows[deep_sha[:12]],
        )
        self.assertIn(
            f"| [0]({BASE_URL}/v1/results/{deep_zero_sha}/archive) |",
            rows[deep_zero_sha[:12]],
        )
        self.assertIn(
            f"| [0]({BASE_URL}/v1/results/{e2e_sha}/archive) |",
            rows[e2e_sha[:12]],
        )

    def test_explicit_archive_field_controls_archive_link(self):
        quick_null_sha = _sha("5")
        deep_none_sha = _sha("6")
        deep_false_sha = _sha("a")
        deep_with_sha = _sha("7")
        quick_with_sha = _sha("8")
        records = [
            {
                "content_sha256": quick_null_sha,
                "kind": "quick",
                "archive": None,
                "benchmark": [],
            },
            {
                "content_sha256": deep_none_sha,
                "kind": "deep",
                "archive": None,
                "benchmark": [{"n": 256, "tflops": 1.0, "median_ms": 1.0}],
            },
            {
                "content_sha256": deep_false_sha,
                "kind": "deep",
                "archive": False,
                "benchmark": [{"n": 256, "tflops": 1.0, "median_ms": 1.0}],
            },
            {
                "content_sha256": deep_with_sha,
                "kind": "deep",
                "archive": {
                    "total_bytes": 1024,
                    "chunk_bytes": 786432,
                    "chunk_count": 1,
                },
                "benchmark": [{"n": 256, "tflops": 1.0, "median_ms": 1.0}],
            },
            {
                "content_sha256": quick_with_sha,
                "kind": "quick",
                "archive": {
                    "total_bytes": 512,
                    "chunk_bytes": 786432,
                    "chunk_count": 1,
                },
                "benchmark": [],
            },
        ]
        text, rows = _summary_rows(records)
        self.assertTrue(rows[quick_null_sha[:12]].endswith("| 0 |"))
        self.assertNotIn(f"/v1/results/{quick_null_sha}/archive", text)
        self.assertTrue(rows[deep_none_sha[:12]].endswith("| 2 |"))
        self.assertNotIn(f"/v1/results/{deep_none_sha}/archive", text)
        self.assertTrue(rows[deep_false_sha[:12]].endswith("| 2 |"))
        self.assertNotIn(f"/v1/results/{deep_false_sha}/archive", text)
        self.assertIn(
            f"| [2]({BASE_URL}/v1/results/{deep_with_sha}/archive) |",
            rows[deep_with_sha[:12]],
        )
        self.assertIn(
            f"| [0]({BASE_URL}/v1/results/{quick_with_sha}/archive) |",
            rows[quick_with_sha[:12]],
        )

    def test_nested_quick_payload_omits_archive_link(self):
        sha = _sha("9")
        nested_quick = {
            "content_sha256": sha,
            "payload": {"report": "mlx-omarchy-quick"},
        }
        text, rows = _summary_rows([nested_quick])
        self.assertTrue(rows[sha[:12]].endswith("| 0 |"))
        self.assertNotIn(f"/v1/results/{sha}/archive", text)


class MirrorIndexDepthTests(unittest.TestCase):
    def setUp(self):
        with self.assertRaises(RecursionError):
            run_with_small_stack(lambda: json.loads(DEEP_JSON))
        record = {"content_sha256": _sha("1"), "kind": "deep"}
        self.bodies = {
            f"{BASE_URL}/v1/dataset/latest.jsonl":
                (json.dumps(record) + "\n").encode("utf-8"),
            f"{BASE_URL}/v1/results": DEEP_JSON.encode("ascii"),
        }
        patcher = mock.patch.object(
            mirror.q, "urlopen", _fake_urlopen(self.bodies))
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_too_deeply_nested_index_returns_none(self):
        self.assertIsNone(
            run_with_small_stack(lambda: mirror.fetch_index(BASE_URL)))

    def test_main_mirrors_without_index_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "snapshot"
            with mock.patch("sys.stdout", io.StringIO()):
                rc = run_with_small_stack(lambda: mirror.main(
                    ["--base-url", BASE_URL, "--out-dir", str(out)]))
            self.assertEqual(rc, 0)
            index = json.loads((out / "index.json").read_text("utf-8"))
            self.assertIsNone(index["generated_at"])
            self.assertIsNone(index["schema_version"])
            self.assertEqual(index["count"], 1)
            self.assertEqual(
                (out / "latest.jsonl").read_bytes(),
                self.bodies[f"{BASE_URL}/v1/dataset/latest.jsonl"])
            self.assertIn("dataset generated_at: -",
                          (out / "SUMMARY.md").read_text("utf-8"))


ZW = "\u200b"  # breaks a GitHub autolink without showing


def _deep_list(depth):
    value = []
    for _ in range(depth):
        value = [value]
    return value


def _table_rows(path):
    return [line for line in path.read_text("utf-8").splitlines()
            if line.startswith("| [")]


class MirrorSummaryEscapeTests(unittest.TestCase):
    def test_untrusted_fields_stay_inside_their_cells(self):
        record = {
            "content_sha256": "dead] [x](https://evil.invalid/) y",
            "kind": "deep|extra",
            "chip": "M1 @ 3.2 GHz, Max@ 3\n\n# Injected https://evil.invalid/",
            "kernel": "7.1.6\r\n| a | b | x&#64;b.invalid",
            "mesa_driver": "[click](https://evil.invalid/)",
            "mlx_version": "<img src=x> C:\\dir WWW.evil.invalid a@b.invalid",
            "benchmark": [],
        }
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "SUMMARY.md"
            mirror.write_summary(out, BASE_URL, [record],
                                 {"generated_at": "2026-11-23\n# Injected"})
            lines = out.read_text(encoding="utf-8").splitlines()
        url = (f"{BASE_URL}/v1/results/"
               "dead%5D%20%5Bx%5D%28https%3A%2F%2Fevil.invalid%2F%29%20y")
        self.assertIn(
            "Records: 1 | dataset generated_at: 2026-11-23 # Injected"
            f" | source: {BASE_URL}", lines)
        self.assertIn(
            f"| [dead\\] \\[x\\](ht]({url}) | deep\\|extra "
            f"| M1 @ 3.2 GHz, Max@ 3 # Injected https:{ZW}//evil.invalid/ "
            "| 7.1.6 \\| a \\| b \\| x\\&#64;b.invalid "
            f"| \\[click\\](https:{ZW}//evil.invalid/) "
            f"| \\<img src=x> C:\\\\dir WWW{ZW}.evil.invalid a{ZW}@b.invalid "
            f"| [0]({url}/archive) |", lines)
        self.assertFalse([line for line in lines
                          if line.startswith(("# Injected", "| a |"))])

    def test_dot_segment_or_blank_hash_links_stay_on_results(self):
        records = [{"content_sha256": sha, "kind": "deep"}
                   for sha in ("..", ".", " " * 12 + "abc")]
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "SUMMARY.md"
            mirror.write_summary(out, BASE_URL, records, None)
            rows = _table_rows(out)
        unknown = f"{BASE_URL}/v1/results/unknown"
        blank = f"{BASE_URL}/v1/results/{'%20' * 12}abc"
        self.assertEqual(rows, [
            f"| [unknown]({unknown}) | deep | - | - | - | - "
            f"| [0]({unknown}/archive) |",
            f"| [unknown]({unknown}) | deep | - | - | - | - "
            f"| [0]({unknown}/archive) |",
            f"| [unknown]({blank}) | deep | - | - | - | - "
            f"| [0]({blank}/archive) |",
        ])

    def test_unreadable_record_fields_render_dash(self):
        deep = _deep_list(200_000)
        nested = {
            "content_sha256": _sha("b"),
            "kind": "deep",
            "benchmark": [{"n": deep, "tflops": 1.0}],
            "payload": {"mesa": {"gpu": {"driverName": deep,
                                         "deviceName": deep}}},
        }
        huge = {
            "content_sha256": _sha("c"),
            "kind": "deep",
            "benchmark": [{"n": 256, "tflops": 10 ** 400}],
        }
        for accessor in (mirror.q.record_chip, mirror.q.record_mesa,
                         mirror.q.record_benchmarks):
            with self.assertRaises(RecursionError):
                run_with_small_stack(lambda: accessor(nested))
        with self.assertRaises(OverflowError):
            mirror.q.record_benchmarks(huge)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "SUMMARY.md"
            run_with_small_stack(lambda: mirror.write_summary(
                out, BASE_URL, [nested, huge], None))
            rows = _table_rows(out)
        self.assertEqual(rows, [
            f"| [{sha[:12]}]({BASE_URL}/v1/results/{sha}) "
            f"| deep | - | - | - | - "
            f"| [-]({BASE_URL}/v1/results/{sha}/archive) |"
            for sha in (_sha("b"), _sha("c"))
        ])

    def test_lone_surrogates_still_write_summary(self):
        record = {"content_sha256": "ab\ud800", "kind": "deep",
                  "chip": "M1\udc00"}
        index = {"generated_at": "2026\udc00", "schema_version": 1}
        bodies = {
            f"{BASE_URL}/v1/dataset/latest.jsonl":
                (json.dumps(record) + "\n").encode("ascii"),
            f"{BASE_URL}/v1/results": json.dumps(index).encode("ascii"),
        }
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(mirror.q, "urlopen",
                                  _fake_urlopen(bodies)), \
                mock.patch("sys.stdout", io.StringIO()), \
                mock.patch("sys.stderr", io.StringIO()) as stderr:
            out = Path(tmp) / "snapshot"
            self.assertEqual(
                mirror.main(["--base-url", BASE_URL, "--out-dir", str(out)]),
                0)
            self.assertEqual(stderr.getvalue(), "")
            self.assertEqual(
                sorted(p.name for p in out.iterdir()),
                ["SUMMARY.md", "index.json", "latest.jsonl"])
            summary = (out / "SUMMARY.md").read_text("utf-8")
            rows = _table_rows(out / "SUMMARY.md")
        self.assertIn("dataset generated_at: 2026?", summary)
        url = f"{BASE_URL}/v1/results/ab%3F"
        self.assertEqual(
            rows, [f"| [ab?]({url}) | deep | M1? | - | - | - "
                   f"| [0]({url}/archive) |"])


class MirrorIndexFieldTypeTests(unittest.TestCase):
    def _fetch(self, body):
        bodies = {f"{BASE_URL}/v1/results": body.encode("utf-8")}
        with mock.patch.object(mirror.q, "urlopen",
                               _fake_urlopen(bodies)):
            return mirror.fetch_index(BASE_URL)

    def test_scalar_fields_pass_through(self):
        index = {"generated_at": "2026-09-03T18:09:05Z",
                 "schema_version": 1, "count": 3}
        self.assertEqual(self._fetch(json.dumps(index)), index)
        self.assertEqual(
            self._fetch('{"schema_version": "1"}')["schema_version"], "1")

    def test_wrong_typed_fields_read_as_none(self):
        empty = {"generated_at": None, "schema_version": None, "count": None}
        for body in (
            '{"generated_at": ["2026"], "schema_version": true,'
            ' "count": {"n": 1}}',
            '{"generated_at": 7, "schema_version": 1.5, "count": "3"}',
            '{"generated_at": null, "schema_version": [1], "count": false}',
        ):
            with self.subTest(body=body):
                self.assertEqual(self._fetch(body), empty)

    def test_main_drops_nested_index_field_and_keeps_the_rest(self):
        nested = "[" * 450 + "]" * 450
        record = {"content_sha256": _sha("1"), "kind": "deep"}
        bodies = {
            f"{BASE_URL}/v1/dataset/latest.jsonl":
                (json.dumps(record) + "\n").encode("utf-8"),
            f"{BASE_URL}/v1/results": (
                f'{{"generated_at": {nested}, "schema_version": 2,'
                ' "count": 1}').encode("ascii"),
        }
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(mirror.q, "urlopen",
                                  _fake_urlopen(bodies)), \
                mock.patch("sys.stdout", io.StringIO()):
            out = Path(tmp) / "snapshot"
            self.assertEqual(
                mirror.main(["--base-url", BASE_URL, "--out-dir", str(out)]),
                0)
            text = (out / "index.json").read_text("utf-8")
            summary = (out / "SUMMARY.md").read_text("utf-8")
        index = json.loads(text)
        self.assertIsNone(index["generated_at"])
        self.assertEqual(index["schema_version"], 2)
        self.assertLess(len(text), 1024)
        self.assertIn("dataset generated_at: -", summary)


class _Reply(_Response):
    """_Response with a chosen status, Content-Length and read size.

    taken counts the body bytes read. length is None when http.client
    could not parse the Content-Length, as on a real response.
    """

    def __init__(self, body, status=200, length="auto", chunk=None):
        super().__init__(body)
        self.status, self.chunk, self.taken = status, chunk, 0
        if length != "auto":
            del self.headers["Content-Length"]
            self._size = None
            if length is not None:
                self.headers["Content-Length"] = length
                with contextlib.suppress(ValueError):  # as http.client
                    self._size = int(length)
                if self._size is not None and self._size < 0:
                    self._size = None

    @property
    def length(self):
        return None if self._size is None else self._size - self.taken

    def read(self, n=-1):
        if self.chunk is not None:
            n = self.chunk if n is None or n < 0 else min(n, self.chunk)
        data = super().read(n)
        self.taken += len(data)
        return data


def _serve(replies):
    """urlopen replacement answering each URL with replies[url]()."""
    def urlopen(request, timeout=None):
        reply = replies[request.full_url]
        if isinstance(reply, BaseException):
            raise reply
        return reply()
    return urlopen


JSONL_URL = f"{BASE_URL}/v1/dataset/latest.jsonl"
INDEX_URL = f"{BASE_URL}/v1/results"
INDEX = {"generated_at": "2026-09-03T18:09:05Z", "schema_version": 1,
         "count": 1}


class MirrorFetchBoundsTests(unittest.TestCase):
    def jsonl(self, reply, max_bytes):
        with mock.patch.object(mirror.q, "urlopen",
                               _serve({JSONL_URL: reply})):
            return mirror.fetch_jsonl_capped(JSONL_URL, max_bytes)

    def index(self, reply):
        with mock.patch.object(mirror.q, "urlopen",
                               _serve({INDEX_URL: reply})):
            return mirror.fetch_index(BASE_URL)

    def test_jsonl_keeps_whole_lines_within_the_cap(self):
        body = b"aaaa\nbbbb\ncccc\n"
        for max_bytes, expected in ((15, (body, False)),
                                    (14, (b"aaaa\nbbbb\n", True)),
                                    (10, (b"aaaa\nbbbb\n", True)),
                                    (9, (b"aaaa\n", True)),
                                    (3, (b"", True))):
            with self.subTest(max_bytes=max_bytes):
                self.assertEqual(self.jsonl(
                    lambda: _Reply(body, chunk=4), max_bytes), expected)

    def test_jsonl_over_the_cap_is_read_only_to_cap_plus_one(self):
        reply = _Reply(b"aaaa\n" * 100, length=None)
        self.assertEqual(self.jsonl(lambda: reply, 12),
                         (b"aaaa\naaaa\n", True))
        self.assertEqual(reply.taken, 13)

    def test_jsonl_bad_status_or_cut_body_raises(self):
        for reply, message in (
                (lambda: _Reply(b"a\n", status=206), "HTTP 206"),
                (lambda: _Reply(b"a\n", length="10"), "8 bytes short"),
                (lambda: _Reply(b"a\n", length="+2"),
                 "invalid Content-Length"),
                (lambda: _Reply(b"a\n", length="9" * 5000),
                 "invalid Content-Length")):
            with self.subTest(message=message):
                with self.assertRaises(mirror.q.DatasetError) as caught:
                    self.jsonl(reply, 1024)
                self.assertIn(message, str(caught.exception))

    def test_index_reads_whole_body_in_short_reads(self):
        body = json.dumps(dict(INDEX, generated_at="x" * 5000)).encode()
        index = self.index(lambda: _Reply(body, chunk=1000))
        self.assertEqual(index["generated_at"], "x" * 5000)

    def test_index_refuses_bad_length_status_cut_or_oversize(self):
        body = json.dumps(INDEX).encode()
        self.assertEqual(self.index(lambda: _Reply(body)), INDEX)
        over = mirror.MAX_INDEX_BYTES + 1
        for name, reply in (
                ("plus sign", lambda: _Reply(body, length="+%d" % len(body))),
                ("negative", lambda: _Reply(body, length="-1")),
                ("empty", lambda: _Reply(body, length="")),
                ("5000 digits", lambda: _Reply(body, length="9" * 5000)),
                ("over cap", lambda: _Reply(body, length=str(over))),
                ("cut short", lambda: _Reply(body, length=str(len(body) + 1))),
                ("status 203", lambda: _Reply(body, status=203)),
                ("long body", lambda: _Reply(b" " * over + body,
                                             length=None))):
            with self.subTest(name):
                self.assertIsNone(self.index(reply))

    def test_index_over_cap_length_is_refused_unread(self):
        reply = _Reply(json.dumps(INDEX).encode(),
                       length=str(mirror.MAX_INDEX_BYTES + 1))
        self.assertIsNone(self.index(lambda: reply))
        self.assertEqual(reply.taken, 0)

    def test_unparsed_length_reaches_the_reader(self):
        reply = _Reply(b"{}", length="9" * 5000)
        self.assertIsNone(reply.length)
        with mock.patch.object(mirror.q, "read_response",
                               wraps=mirror.q.read_response) as spy:
            self.assertIsNone(self.index(lambda: reply))
        spy.assert_called_once()

    def test_malformed_http_reply_is_skipped(self):
        bad = http.client.BadStatusLine("GARBAGE")
        self.assertIsNone(self.index(bad))
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(mirror.q, "urlopen",
                                  _serve({JSONL_URL: bad})), \
                mock.patch("sys.stderr", io.StringIO()) as stderr:
            out = Path(tmp) / "snapshot"
            self.assertEqual(
                mirror.main(["--base-url", BASE_URL, "--out-dir", str(out)]),
                0)
            self.assertEqual(list(out.iterdir()), [])
        self.assertIn("endpoint unavailable", stderr.getvalue())

    def test_http_errors_are_closed(self):
        index_fp, jsonl_fp = io.BytesIO(b"gone"), io.BytesIO(b"gone")
        self.assertIsNone(self.index(urllib.error.HTTPError(
            INDEX_URL, 404, "Not Found", {}, index_fp)))
        self.assertTrue(index_fp.closed)
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(mirror.q, "urlopen", _serve({
                    JSONL_URL: urllib.error.HTTPError(
                        JSONL_URL, 503, "Unavailable", {}, jsonl_fp)})), \
                mock.patch("sys.stderr", io.StringIO()) as stderr:
            out = Path(tmp) / "snapshot"
            self.assertEqual(
                mirror.main(["--base-url", BASE_URL, "--out-dir", str(out)]),
                0)
            self.assertEqual(list(out.iterdir()), [])
        self.assertTrue(jsonl_fp.closed)
        self.assertIn("endpoint unavailable", stderr.getvalue())

    def test_main_writes_nothing_on_a_cut_dataset(self):
        replies = {JSONL_URL: lambda: _Reply(b'{"kind": "deep"}\n',
                                             length="1000"),
                   INDEX_URL: lambda: _Reply(json.dumps(INDEX).encode())}
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(mirror.q, "urlopen", _serve(replies)), \
                mock.patch("sys.stderr", io.StringIO()) as stderr:
            out = Path(tmp) / "snapshot"
            self.assertEqual(
                mirror.main(["--base-url", BASE_URL, "--out-dir", str(out)]),
                0)
            self.assertEqual(list(out.iterdir()), [])
        self.assertIn("983 bytes short", stderr.getvalue())


class MirrorWriteTests(unittest.TestCase):
    RECORD = (json.dumps({"content_sha256": _sha("1"), "kind": "deep"})
              + "\n").encode("utf-8")

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.out = self.root / "snapshot"
        self.out.mkdir()
        patcher = mock.patch.object(mirror.q, "urlopen", _fake_urlopen({
            JSONL_URL: self.RECORD, INDEX_URL: json.dumps(INDEX).encode()}))
        patcher.start()
        self.addCleanup(patcher.stop)

    def mirror(self):
        with mock.patch("sys.stdout", io.StringIO()), \
                mock.patch("sys.stderr", io.StringIO()) as stderr:
            rc = mirror.main(["--base-url", BASE_URL,
                              "--out-dir", str(self.out)])
        self.assertEqual(rc, 0)
        return stderr.getvalue()

    def names(self):
        return sorted(p.name for p in self.out.iterdir())

    def test_planted_symlinks_are_replaced_not_followed(self):
        victim = self.root / "victim.txt"
        victim.write_bytes(b"keep me\n")
        for name in ("latest.jsonl", "index.json", "SUMMARY.md"):
            (self.out / name).symlink_to(victim)
        self.assertEqual(self.mirror(), "")
        self.assertEqual(victim.read_bytes(), b"keep me\n")
        self.assertEqual(self.names(),
                         ["SUMMARY.md", "index.json", "latest.jsonl"])
        for name in self.names():
            self.assertFalse((self.out / name).is_symlink(), name)
        self.assertEqual((self.out / "latest.jsonl").read_bytes(),
                         self.RECORD)

    def test_failed_write_keeps_the_old_snapshot_and_no_temp_files(self):
        old = {"latest.jsonl": b"old\n", "index.json": b"{}\n",
               "SUMMARY.md": b"# old\n"}
        for name, data in old.items():
            (self.out / name).write_bytes(data)
        real_open = os.open

        def full_disk(path, flags, *args, **kwargs):
            if "SUMMARY.md" in os.fspath(path):
                raise OSError(28, "No space left on device")
            return real_open(path, flags, *args, **kwargs)

        with mock.patch.object(mirror.os, "open", full_disk):
            self.assertIn("No space left on device", self.mirror())
        self.assertEqual(self.names(), sorted(old))
        for name, data in old.items():
            self.assertEqual((self.out / name).read_bytes(), data, name)

    def test_failed_rename_leaves_no_complete_looking_snapshot(self):
        real_replace = os.replace
        for failing, left in (("SUMMARY.md", ["index.json"]),
                              ("latest.jsonl", ["SUMMARY.md", "index.json"])):
            with self.subTest(failing=failing):
                out = self.root / failing
                self.out = out
                out.mkdir()

                def replace(src, dst, failing=failing):
                    if os.fspath(dst).endswith(failing):
                        raise OSError(30, "Read-only file system")
                    return real_replace(src, dst)

                with mock.patch.object(mirror.os, "replace", replace):
                    self.assertIn("Read-only file system", self.mirror())
                self.assertEqual(self.names(), left)

    def test_files_take_the_umask_mode(self):
        previous = os.umask(0o027)
        try:
            self.mirror()
        finally:
            os.umask(previous)
        for name in self.names():
            self.assertEqual((self.out / name).stat().st_mode & 0o777, 0o640)


if __name__ == "__main__":
    unittest.main()
