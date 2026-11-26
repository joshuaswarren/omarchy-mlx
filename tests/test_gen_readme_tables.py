"""scripts/gen_readme_tables.py: fail-closed markers, idempotent fills, drift detection, stale chip blocks."""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "gen_readme_tables.py"

HEADER = (
    "chip\tmodel\tmetric\tlinux\tmacos\tgovernor\tdriver_name\tdriver_commit\twheel_commit\truns\t"
    "start_load_max\tane_busy\tcontrol_spread_pct\tp_spread_pct\tnoisy\tinterleaved\tdate\tcaption\t"
    "pv_decode\tpv_prefill"
)


def row(governor, linux="17.5"):
    cells = ["M1 Max (G13C)", "gemma-4-26B-A4B", "decode", linux, "", governor, "package", "69bde15340",
             "18059c80", "2"] + [""] * 10
    return "\t".join(cells) + "\n"


OFF_ROW = row("performance (filler off)")
ON_ROW = row("performance (filler on)")

README = "# omarchy-mlx\n\n## How close to macOS\n\nHand-written table here.\n"
README_MARKED = (
    "# omarchy-mlx\n\n## How close to macOS\n\n<!-- BEGIN GENERATED: bench-main -->\nold\n"
    "<!-- END GENERATED: bench-main -->\n\nafter\n"
)
BENCHMARKS = "# How close to macOS\n"
NO_ROWS = "_no sourced rows for this chip._"


class Workdir:
    def __init__(self, tmp, rows, readme=README):
        self.tmp = Path(tmp)
        self.set_rows(rows)
        (self.tmp / "README.md").write_text(readme)
        (self.tmp / "BENCHMARKS.md").write_text(BENCHMARKS)

    def set_rows(self, rows):
        (self.tmp / "rows.tsv").write_text(HEADER + "\n" + rows)

    def run(self, *extra):
        return subprocess.run(
            [sys.executable, str(SCRIPT), str(self.tmp / "rows.tsv"), "--readme", str(self.tmp / "README.md"),
             "--benchmarks", str(self.tmp / "BENCHMARKS.md"), *extra],
            capture_output=True, text=True, check=False)

    def read(self, name):
        return (self.tmp / name).read_text()


class GenReadmeTablesTests(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.addCleanup(self._td.cleanup)
        self.tmp = self._td.name

    def test_filler_off_row_without_markers_exits_non_zero_and_changes_no_file(self):
        w = Workdir(self.tmp, OFF_ROW)
        r = w.run()
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("bench-main", r.stderr)
        self.assertIn("## How close to macOS", r.stderr)
        self.assertEqual(w.read("README.md"), README)
        self.assertEqual(w.read("BENCHMARKS.md"), BENCHMARKS)

    def test_header_only_rows_file_has_nothing_to_generate(self):
        w = Workdir(self.tmp, "")
        r = w.run("--check")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertEqual(w.read("README.md"), README)

    def test_fills_an_existing_marker_region_and_leaves_the_rest_alone(self):
        w = Workdir(self.tmp, OFF_ROW, README_MARKED)
        r = w.run()
        self.assertEqual(r.returncode, 0, r.stderr)
        readme = w.read("README.md")
        self.assertTrue(readme.startswith("# omarchy-mlx\n\n## How close to macOS\n\n<!-- BEGIN GENERATED: bench-main -->\n"))
        self.assertIn("| gemma-4-26B-A4B | 17.5 |", readme)
        self.assertNotIn("\nold\n", readme)
        self.assertTrue(readme.endswith("<!-- END GENERATED: bench-main -->\n\nafter\n"))
        self.assertIn("| gemma-4-26B-A4B | 17.5 |", w.read("BENCHMARKS.md"))

    def test_second_run_changes_nothing(self):
        w = Workdir(self.tmp, OFF_ROW, README_MARKED)
        self.assertEqual(w.run().returncode, 0)
        first = (w.read("README.md"), w.read("BENCHMARKS.md"))
        r = w.run()
        self.assertEqual(r.returncode, 0)
        self.assertNotIn("updated", r.stdout)
        self.assertEqual((w.read("README.md"), w.read("BENCHMARKS.md")), first)
        self.assertEqual(w.run("--check").returncode, 0)

    def test_check_exits_one_after_a_rows_value_changes_and_writes_nothing(self):
        w = Workdir(self.tmp, OFF_ROW, README_MARKED)
        self.assertEqual(w.run().returncode, 0)
        before = (w.read("README.md"), w.read("BENCHMARKS.md"))
        w.set_rows(row("performance (filler off)", linux="18.5"))
        r = w.run("--check")
        self.assertEqual(r.returncode, 1)
        self.assertIn("DRIFT", r.stdout)
        self.assertEqual((w.read("README.md"), w.read("BENCHMARKS.md")), before)

    def test_chip_whose_rows_are_all_removed_is_rewritten_to_the_placeholder(self):
        w = Workdir(self.tmp, ON_ROW)
        self.assertEqual(w.run().returncode, 0)
        self.assertIn("| gemma-4-26B-A4B | 17.5 |", w.read("BENCHMARKS.md"))
        w.set_rows("")
        before = w.read("BENCHMARKS.md")
        r = w.run("--check")
        self.assertEqual(r.returncode, 1)
        self.assertEqual(w.read("BENCHMARKS.md"), before)
        self.assertEqual(w.run().returncode, 0)
        after = w.read("BENCHMARKS.md")
        self.assertIn(NO_ROWS, after)
        self.assertNotIn("17.5", after)
        self.assertEqual(w.run("--check").returncode, 0)

    def test_non_numeric_cell_exits_non_zero_names_the_column_and_changes_no_file(self):
        w = Workdir(self.tmp, row("performance (filler off)", linux="fast"), README_MARKED)
        r = w.run()
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("column linux is not a number: fast", r.stderr)
        self.assertIn("rows.tsv:2", r.stderr)
        self.assertEqual(w.read("README.md"), README_MARKED)
        self.assertEqual(w.read("BENCHMARKS.md"), BENCHMARKS)

    def test_control_cell_takes_the_decode_spread_when_the_prefill_row_comes_first(self):
        def metric_row(metric, spread):
            cells = ["M1 Max (G13C)", "gemma-4-26B-A4B", metric, "17.5", "", "performance (filler on)",
                     "package", "69bde15340", "18059c80", "2", "", "", spread] + [""] * 7
            return "\t".join(cells) + "\n"

        w = Workdir(self.tmp, metric_row("prefill", "0.17") + metric_row("decode", "0.15"))
        self.assertEqual(w.run().returncode, 0)
        out = w.read("BENCHMARKS.md")
        self.assertIn("control spread 0.15%", out)
        self.assertNotIn("0.17", out)


if __name__ == "__main__":
    unittest.main()
