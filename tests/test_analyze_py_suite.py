import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "tools" / "analyze-py-suite.py"


def run_analyzer(report_dir, *args):
    return subprocess.run(
        [sys.executable, str(SCRIPT), str(report_dir), *map(str, args)],
        capture_output=True,
        text=True,
        timeout=10,
    )


def junit_xml(*cases, tests=None, failures=0, errors=0, skipped=0):
    tests = len(cases) if tests is None else tests
    return (
        "<?xml version='1.0'?><testsuites name='pytest tests'>"
        f"<testsuite name='pytest' tests='{tests}' failures='{failures}' "
        f"errors='{errors}' skipped='{skipped}'>"
        + "".join(cases)
        + "</testsuite></testsuites>"
    )


PASS_CASE = "<testcase classname='test_ops' name='test_passes'/>"


class AnalyzePySuiteTests(unittest.TestCase):
    def test_valid_all_pass_report_succeeds(self):
        with tempfile.TemporaryDirectory() as td:
            Path(td, "test_ops.py.xml").write_text(
                junit_xml(PASS_CASE), encoding="utf-8"
            )
            result = run_analyzer(td)

        self.assertEqual(result.returncode, 0, result.stderr)

    def test_missing_directory_fails(self):
        result = run_analyzer("/nonexistent/mlx-python-reports")

        self.assertNotEqual(result.returncode, 0)

    def test_directory_without_xml_reports_fails(self):
        with tempfile.TemporaryDirectory() as td:
            result = run_analyzer(td)

        self.assertNotEqual(result.returncode, 0)

    def test_malformed_xml_fails(self):
        with tempfile.TemporaryDirectory() as td:
            Path(td, "test_ops.py.xml").write_text("<testsuites>", encoding="utf-8")
            result = run_analyzer(td)

        self.assertNotEqual(result.returncode, 0)

    def test_all_skipped_report_succeeds_with_warning(self):
        skipped = "<testcase classname='test_ops' name='test_skipped'><skipped/></testcase>"
        with tempfile.TemporaryDirectory() as td:
            Path(td, "test_conv_transpose.py.xml").write_text(
                junit_xml(skipped, skipped=1), encoding="utf-8"
            )
            result = run_analyzer(td)

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("all skipped", result.stderr)

    def test_mixed_skipped_and_executed_report_succeeds(self):
        skipped = "<testcase classname='test_ops' name='test_skipped'><skipped/></testcase>"
        with tempfile.TemporaryDirectory() as td:
            Path(td, "test_ops.py.xml").write_text(
                junit_xml(PASS_CASE + skipped, tests=2, skipped=1),
                encoding="utf-8",
            )
            result = run_analyzer(td)

        self.assertEqual(result.returncode, 0, result.stderr)

    def test_failure_is_classified_without_becoming_analyzer_failure(self):
        failed = (
            "<testcase classname='test_ops' name='test_gap'>"
            "<failure message='RuntimeError: [omarchy] FFT is not implemented for GPU'/>"
            "</testcase>"
        )
        with tempfile.TemporaryDirectory() as td:
            report_dir = Path(td)
            report_dir.joinpath("test_fft.py.xml").write_text(
                junit_xml(failed, failures=1), encoding="utf-8"
            )
            csv_path = report_dir / "classification.csv"
            result = run_analyzer(report_dir, "--csv", csv_path)
            with csv_path.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["kind"], "named")
        self.assertEqual(rows[0]["detail"], "FFT")

    def test_refused_gap_is_named(self):
        failed = (
            "<testcase classname='test_compile' name='test_inf_constant'>"
            "<failure message='RuntimeError: [omarchy] Compiled tape bfloat16 is refused: bf12 fragments'/>"
            "</testcase>"
        )
        with tempfile.TemporaryDirectory() as td:
            report_dir = Path(td)
            report_dir.joinpath("test_compile.py.xml").write_text(
                junit_xml(failed, failures=1), encoding="utf-8"
            )
            csv_path = report_dir / "classification.csv"
            result = run_analyzer(report_dir, "--csv", csv_path)
            with csv_path.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["kind"], "named")
        self.assertEqual(rows[0]["detail"], "Compiled tape bfloat16")


if __name__ == "__main__":
    unittest.main()
