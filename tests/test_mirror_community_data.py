"""Offline tests for scripts/mirror_community_data.py."""

import io
import json
import sys
import tempfile
import threading
import unittest
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
    def __init__(self, body):
        super().__init__(body)
        self.headers = {"Content-Length": str(len(body))}


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
            mirror.urllib.request, "urlopen", _fake_urlopen(self.bodies))
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
                mock.patch.object(mirror.urllib.request, "urlopen",
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
        with mock.patch.object(mirror.urllib.request, "urlopen",
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
                mock.patch.object(mirror.urllib.request, "urlopen",
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


if __name__ == "__main__":
    unittest.main()
