"""Every chip in a community dataset snapshot must appear in docs/TESTED-HARDWARE.md.

Regression for issue #147: apple,t6031 (16-inch M3 Max) landed in the
dataset while the hardware matrix had no row for it.
"""

import contextlib
import io
import json
import sys
import tempfile
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


class MatrixParseTests(unittest.TestCase):
    def test_matrix_lists_linux_and_macos_chip_names(self):
        chips = chk.matrix_chips(DOCS)
        self.assertIn("apple,t6020", chips)
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


if __name__ == "__main__":
    unittest.main()
