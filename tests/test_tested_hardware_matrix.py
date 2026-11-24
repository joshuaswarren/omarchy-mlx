"""Every chip in a community dataset snapshot must appear in docs/TESTED-HARDWARE.md.

Regression for issue #147: apple,t6031 (16-inch M3 Max) landed in the
dataset while the hardware matrix had no row for it.
"""

import contextlib
import io
import json
import sys
import tempfile
import threading
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import check_tested_hardware as chk  # noqa: E402

DOCS = REPO_ROOT / "docs" / "TESTED-HARDWARE.md"


def write_jsonl(records):
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False,
                                     encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
        return Path(handle.name)


def on_pinned_stack(fn):
    """fn() on a 4 MiB thread stack, so deep-JSON recursion does not follow
    the runner's ulimit -s (Python 3.14 bounds recursion by C stack)."""
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


DEEP_JSON = b"[" * 200_000 + b"]" * 200_000
GOOD = b'{"chip": "apple,t8103"}\n'


def run_main(argv):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = chk.main(argv)
    return code, out.getvalue(), err.getvalue()


class MatrixParseTests(unittest.TestCase):
    def test_matrix_lists_linux_and_macos_chip_names(self):
        chips = chk.matrix_chips(DOCS)
        self.assertIn("apple,t6020", chips)
        self.assertIn("apple,t6022", chips)
        self.assertIn("apple,t6031", chips)
        self.assertIn("Apple M5 Max", chips)
        self.assertIn("linux,dummy-virt", chips)

    def test_aggregate_rows_do_not_count_as_chips(self):
        chips = chk.matrix_chips(DOCS)
        self.assertNotIn("chip not captured", chips)
        self.assertNotIn("GPU / SoC", chips)


class MissingChipTests(unittest.TestCase):
    def test_chip_missing_from_matrix_is_named(self):
        fixture = write_jsonl([
            {"chip": "apple,t8103", "kind": "quick"},
            {"chip": "apple,t6031", "kind": "deep"},
            {"chip": "apple,t9999", "kind": "deep"},
            {"kind": "quick"},  # chip not captured
        ])
        try:
            gone = chk.missing(chk.dataset_chips(fixture),
                               chk.matrix_chips(DOCS))
            self.assertEqual(gone, ["apple,t9999"])
        finally:
            fixture.unlink()

    def test_exit_code_names_chip_and_fails(self):
        fixture = write_jsonl([{"chip": "apple,t9999", "kind": "deep"}])
        try:
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                code = chk.main([str(fixture), "--docs", str(DOCS)])
            self.assertEqual(code, 1)
            self.assertIn("apple,t9999", err.getvalue())
        finally:
            fixture.unlink()

    def test_all_dataset_chips_present_passes(self):
        fixture = write_jsonl([
            {"chip": "apple,t8103", "kind": "quick"},
            {"chip": "apple,t6022", "kind": "deep"},
            {"chip": "apple,t6031", "kind": "deep"},
            {"chip": "apple,t6020", "kind": "omarchy-mac-e2e"},
            {"chip": "Apple M5 Max", "kind": "deep"},
            {"chip": "linux,dummy-virt", "kind": "quick"},
            {"kind": "deep"},  # chip not captured
        ])
        try:
            err = io.StringIO()
            out = io.StringIO()
            with contextlib.redirect_stderr(err), contextlib.redirect_stdout(out):
                code = chk.main([str(fixture), "--docs", str(DOCS)])
            self.assertEqual(code, 0)
            self.assertIn("ok:", out.getvalue())
        finally:
            fixture.unlink()


class MalformedInputTests(unittest.TestCase):
    """A snapshot or matrix the check cannot read fails it with one error
    line and exit 1, never a traceback and never "ok:"."""

    BAD_LINES = {
        "array": (b"[1, 2]", "record is list, not an object"),
        "string": (b'"foo"', "record is str, not an object"),
        "number": (b"123", "record is int, not an object"),
        "null": (b"null", "record is NoneType, not an object"),
        "bool": (b"true", "record is bool, not an object"),
        "list chip": (b'{"chip": ["x"]}', "chip is list, not a string"),
        "object chip": (b'{"chip": {"a": 1}}', "chip is dict, not a string"),
        "int chip": (b'{"chip": 123}', "chip is int, not a string"),
        "true chip": (b'{"chip": true}', "chip is bool, not a string"),
        "false chip": (b'{"chip": false}', "chip is bool, not a string"),
        "zero chip": (b'{"chip": 0}', "chip is int, not a string"),
        "bad json": (b"{bad", "not a JSON line"),
    }

    def setUp(self):
        scratch = tempfile.TemporaryDirectory()
        self.addCleanup(scratch.cleanup)
        self.dir = Path(scratch.name)

    def write(self, name, data):
        path = self.dir / name
        path.write_bytes(data)
        return path

    def check_failed(self, result, *fragments):
        code, out, err = result
        self.assertEqual(code, 1)
        self.assertNotIn("ok:", out)
        self.assertTrue(err.startswith("error: "), err)
        self.assertNotIn("Traceback", err)
        for fragment in fragments:
            self.assertIn(fragment, err)

    def test_bad_line_names_file_and_line(self):
        for name, (line, reason) in self.BAD_LINES.items():
            with self.subTest(name):
                fixture = self.write(f"{name}.jsonl", GOOD + b"\n" + line + b"\n")
                self.check_failed(run_main([str(fixture), "--docs", str(DOCS)]),
                                  f"{fixture}:3: ", reason)
                with self.assertRaises(chk.DatasetError):
                    chk.dataset_chips(fixture)

    def test_lone_bool_chip_is_not_reported_as_a_chip_name(self):
        fixture = self.write("bool.jsonl", b'{"chip": true}\n')
        result = run_main([str(fixture), "--docs", str(DOCS)])
        self.assertNotIn("missing from", result[2])
        self.check_failed(result, f"{fixture}:1: chip is bool")

    def test_byte_order_mark_is_rejected(self):
        fixture = self.write("bom.jsonl", b"\xef\xbb\xbf" + GOOD)
        self.check_failed(run_main([str(fixture), "--docs", str(DOCS)]),
                          f"{fixture}:1: not a JSON line")

    def test_deeply_nested_line_fails_cleanly(self):
        with self.assertRaises(RecursionError):
            on_pinned_stack(lambda: json.loads(DEEP_JSON))
        fixture = self.write("deep.jsonl", DEEP_JSON + b"\n")
        self.check_failed(
            on_pinned_stack(lambda: run_main([str(fixture), "--docs", str(DOCS)])),
            f"{fixture}:1: not a JSON line")

    def test_missing_files_fail_cleanly(self):
        fixture = self.write("good.jsonl", GOOD)
        absent = self.dir / "absent"
        for dataset, docs, named in ((absent, DOCS, absent),
                                     (fixture, absent, absent),
                                     (self.dir, DOCS, self.dir),
                                     (fixture, self.dir, self.dir)):
            with self.subTest(dataset=dataset.name, docs=docs.name):
                self.check_failed(run_main([str(dataset), "--docs", str(docs)]),
                                  str(named))

    def test_non_utf8_files_fail_cleanly(self):
        fixture = self.write("good.jsonl", GOOD)
        bad = self.write("bad.jsonl", GOOD + b'{"chip": "apple,t8103\xff"}\n')
        docs = self.write("bad.md", b"## Apple silicon\n\xff\n")
        self.check_failed(run_main([str(bad), "--docs", str(DOCS)]),
                          f"{bad}:2: not UTF-8 text")
        self.check_failed(run_main([str(fixture), "--docs", str(docs)]),
                          f"{docs}: not UTF-8 text")
        with self.assertRaises(chk.DatasetError):
            chk.dataset_chips(bad)
        with self.assertRaises(chk.DatasetError):
            chk.matrix_chips(docs)

    def test_line_endings_match_text_mode(self):
        fixture = self.write("cr.jsonl", b'{"chip": "apple,t8103"}\r'
                             b'{"chip": "apple,t6031"}\r\n{"chip": "apple,t9999"}')
        self.assertEqual(chk.dataset_chips(fixture),
                         ["apple,t6031", "apple,t8103", "apple,t9999"])

    def test_blank_chip_is_not_captured_and_padding_is_trimmed(self):
        fixture = self.write("blank.jsonl",
                             b'{"chip": "   "}\n{"chip": ""}\n{"chip": null}\n'
                             b'{"chip": " apple,t8103 "}\n\n  \n')
        self.assertEqual(chk.dataset_chips(fixture), ["apple,t8103"])
        code, out, _ = run_main([str(fixture), "--docs", str(DOCS)])
        self.assertEqual(code, 0)
        self.assertIn("ok:", out)


if __name__ == "__main__":
    unittest.main()
