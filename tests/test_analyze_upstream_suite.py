import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "tools" / "analyze-upstream-suite.py"


def run_analyzer(report_dir, *args):
    return subprocess.run(
        [sys.executable, str(SCRIPT), str(report_dir), *map(str, args)],
        capture_output=True,
        text=True,
        timeout=10,
    )


def doctest_xml(*cases):
    return (
        "<?xml version='1.0'?><doctest><TestSuite>"
        + "".join(cases)
        + "</TestSuite></doctest>"
    )


PASS_CASE = (
    "<TestCase name='passes' skipped='false'>"
    "<OverallResultsAsserts test_case_success='true'/>"
    "</TestCase>"
)


def failure_case(message):
    return (
        "<TestCase name='fails' skipped='false'>"
        "<Expression success='false'><Exception>"
        + message
        + "</Exception></Expression>"
        "<OverallResultsAsserts test_case_success='false'/>"
        "</TestCase>"
    )


class AnalyzeUpstreamSuiteTests(unittest.TestCase):
    def test_valid_all_pass_report_succeeds(self):
        with tempfile.TemporaryDirectory() as td:
            Path(td, "ops_tests.xml").write_text(
                doctest_xml(PASS_CASE), encoding="utf-8"
            )
            result = run_analyzer(td)

        self.assertEqual(result.returncode, 0, result.stderr)

    def test_missing_directory_fails(self):
        result = run_analyzer("/nonexistent/mlx-cpp-reports")

        self.assertNotEqual(result.returncode, 0)

    def test_directory_without_xml_reports_fails(self):
        with tempfile.TemporaryDirectory() as td:
            result = run_analyzer(td)

        self.assertNotEqual(result.returncode, 0)

    def test_malformed_xml_fails(self):
        with tempfile.TemporaryDirectory() as td:
            Path(td, "ops_tests.xml").write_text("<doctest>", encoding="utf-8")
            result = run_analyzer(td)

        self.assertNotEqual(result.returncode, 0)

    def test_report_without_executed_cases_fails(self):
        skipped = "<TestCase name='skipped' skipped='true'/>"
        with tempfile.TemporaryDirectory() as td:
            Path(td, "ops_tests.xml").write_text(
                doctest_xml(skipped), encoding="utf-8"
            )
            result = run_analyzer(td)

        self.assertNotEqual(result.returncode, 0)

    def test_incomplete_case_is_not_evidence_of_execution(self):
        with tempfile.TemporaryDirectory() as td:
            Path(td, "ops_tests.xml").write_text(
                doctest_xml("<TestCase name='unfinished'/>")
            )
            result = run_analyzer(td)
        self.assertNotEqual(result.returncode, 0)

    def test_full_scope_failures_are_classified_and_analyzer_succeeds(self):
        reports = {
            "fft_tests.xml": "RuntimeError: [omarchy] FFT is not implemented",
            "export_import_tests.xml": "RuntimeError: export failed",
            "linalg_tests.xml": "RuntimeError: inverse failed",
            "ops_tests.xml": "RuntimeError: FFT plan failed",
        }
        with tempfile.TemporaryDirectory() as td:
            report_dir = Path(td)
            for name, message in reports.items():
                report_dir.joinpath(name).write_text(
                    doctest_xml(failure_case(message)), encoding="utf-8"
                )
            csv_path = report_dir / "classification.csv"
            result = run_analyzer(report_dir, "--csv", csv_path)
            with csv_path.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual({row["file"] for row in rows}, {Path(n).stem for n in reports})
        self.assertEqual({row["category"] for row in rows}, {"a"})


if __name__ == "__main__":
    unittest.main()
