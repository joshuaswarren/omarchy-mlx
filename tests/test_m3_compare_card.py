"""tests/test_m3_compare_card.py: M3 comparator emits parity-card rows that
gen_readme_tables.py accepts, and refuses every snapshot/prompt/SHA/warmup
mismatch without emitting."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMPARE = ROOT / "scripts" / "m3_compare_card.py"
BASELINE = ROOT / "scripts" / "m3_macos_baseline.py"
GEN = ROOT / "scripts" / "gen_readme_tables.py"

HEADER = (
    "chip\tmodel\tmetric\tlinux\tmacos\tgovernor\tdriver_name\tdriver_commit\twheel_commit\truns\t"
    "start_load_max\tane_busy\tcontrol_spread_pct\tp_spread_pct\tnoisy\tinterleaved\tdate\tcaption\t"
    "pv_decode\tpv_prefill"
)

sys.path.insert(0, str(ROOT / "scripts"))
import m3_macos_baseline  # noqa: E402


def cell(mode="decode", ctx=512, repo="SiddhJagani/Qwen3.8-2B-mlx-4Bit",
         revision="0867d98b", prompt="The history of France begins in antiquity.",
         input_sha="in" + "0" * 62, out_shas=None, toks=None, warmups=1,
         decode_tokens=64):
    if out_shas is None:
        out_shas = ["ab" + "0" * 62] * 5
    if toks is None:
        toks = [50.1, 50.2, 50.3, 50.4, 50.5] if mode == "decode" \
            else [400.1, 400.2, 400.3, 400.4, 400.5]
    d = {
        "repo": repo, "revision": revision, "prompt": prompt, "mode": mode,
        "ctx": ctx, "input_sha256": input_sha, "output_sha256": out_shas,
        "timed_tok_s": toks, "median_tok_s": toks[2], "warmups_discarded": warmups,
        "decode_tokens": decode_tokens,
    }
    return d


def write_json(tmp, name, obj):
    p = Path(tmp) / name
    p.write_text(json.dumps(obj))
    return str(p)


def run_compare(*extra):
    return subprocess.run(
        [sys.executable, str(COMPARE), *extra],
        capture_output=True, text=True, check=False)


def fixture_pair(tmp):
    """A matching decode+prefill cell pair on both sides."""
    files = {"linux": [], "macos": []}
    for side, base in (("linux", 50.0), ("macos", 49.0)):
        files[side].append(write_json(tmp, f"{side}-dec.json",
                                      cell("decode", toks=[base + i / 10 for i in range(5)])))
        files[side].append(write_json(tmp, f"{side}-pf.json",
                                      cell("prefill", toks=[base * 8 + i for i in range(5)])))
    return files


class CompareCardTests(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.addCleanup(self._td.cleanup)
        self.tmp = self._td.name
        self.files = fixture_pair(self.tmp)
        self.base_args = ["--linux", *self.files["linux"],
                          "--macos", *self.files["macos"],
                          "--chip", "14-inch M2 Max (G14C)",
                          "--date", "2026-11-27"]

    def test_pass_emits_two_tsv_rows_with_both_sides(self):
        r = run_compare(*self.base_args)
        self.assertEqual(r.returncode, 0, r.stderr)
        lines = [l for l in r.stdout.splitlines() if l.strip()]
        self.assertEqual(len(lines), 2)
        fields = [l.split("\t") for l in lines]
        for f in fields:
            self.assertEqual(len(f), 20)
        self.assertEqual(fields[0][2], "decode")
        self.assertEqual(fields[1][2], "prefill")
        self.assertEqual(fields[0][3], "50.200")  # linux median
        self.assertEqual(fields[0][4], "49.200")  # macos median
        self.assertEqual(fields[0][9], "10")      # runs pooled across sides
        self.assertEqual(fields[0][16], "2026-11-27")

    def test_emitted_rows_validate_through_gen_readme_tables(self):
        r = run_compare(*self.base_args)
        self.assertEqual(r.returncode, 0, r.stderr)
        tmp = Path(self.tmp)
        (tmp / "rows.tsv").write_text(HEADER + "\n" + r.stdout)
        (tmp / "README.md").write_text(
            "# omarchy-mlx\n\n## How close to macOS\n\n"
            "<!-- BEGIN GENERATED: bench-main -->\nold\n"
            "<!-- END GENERATED: bench-main -->\n\nafter\n")
        (tmp / "BENCHMARKS.md").write_text("# How close to macOS\n")
        g = subprocess.run(
            [sys.executable, str(GEN), str(tmp / "rows.tsv"),
             "--readme", str(tmp / "README.md"),
             "--benchmarks", str(tmp / "BENCHMARKS.md")],
            capture_output=True, text=True, check=False)
        self.assertEqual(g.returncode, 0, g.stderr)
        out = (tmp / "BENCHMARKS.md").read_text()
        self.assertIn("50.200 / 49.200 = 102.0%", out)
        self.assertIn("Qwen3.8-2B", out)
        self.assertIn("14-inch M2 Max (G14C)", out)
        # filler-on G14C rows belong to BENCHMARKS.md, not the g13c README table
        self.assertIn("\nold\n", (tmp / "README.md").read_text())

    def test_refuses_output_sha_mismatch_across_sides(self):
        self.files["macos"][0] = write_json(
            self.tmp, "macos-dec.json", cell("decode", out_shas=["cd" + "0" * 62] * 5))
        r = run_compare(*self.base_args)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("output SHA parity broken", r.stderr)
        self.assertEqual(r.stdout.strip(), "")

    def test_refuses_nondeterministic_side(self):
        self.files["linux"][0] = write_json(
            self.tmp, "linux-dec.json",
            cell("decode", out_shas=["ab" + "0" * 62] * 4 + ["cd" + "0" * 62]))
        r = run_compare(*self.base_args)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("nondeterministic", r.stderr)

    def test_refuses_input_sha_mismatch(self):
        self.files["macos"][0] = write_json(
            self.tmp, "macos-dec.json", cell("decode", input_sha="zz" + "0" * 62))
        r = run_compare(*self.base_args)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("input_sha256 differs", r.stderr)

    def test_refuses_warmup_mismatch(self):
        self.files["macos"][0] = write_json(
            self.tmp, "macos-dec.json", cell("decode", warmups=2))
        r = run_compare(*self.base_args)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("warmups_discarded differs", r.stderr)

    def test_refuses_prompt_mismatch(self):
        self.files["macos"][0] = write_json(
            self.tmp, "macos-dec.json", cell("decode", prompt="a different prompt"))
        r = run_compare(*self.base_args)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("prompt differs", r.stderr)

    def test_refuses_revision_mismatch(self):
        self.files["macos"][0] = write_json(
            self.tmp, "macos-dec.json", cell("decode", revision="deadbeef"))
        r = run_compare(*self.base_args)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("revision differs", r.stderr)

    def test_missing_side_is_pending_not_fatal(self):
        # linux has decode+prefill, macos only decode -> prefill cell pending,
        # decode row still emitted.
        r = run_compare("--linux", *self.files["linux"],
                        "--macos", self.files["macos"][0],
                        "--chip", "14-inch M2 Max (G14C)")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("PENDING", r.stderr)
        self.assertIn("prefill", r.stderr)
        lines = [l for l in r.stdout.splitlines() if l.strip()]
        self.assertEqual(len(lines), 1)
        self.assertEqual(lines[0].split("\t")[2], "decode")

    def test_baseline_pure_functions_match_par45(self):
        self.assertEqual(
            m3_macos_baseline.digest([1, 2, 3]),
            "8a6ae15122001229edb8866f56e342af12ae8187203c3e3b33931743e7c0c48d")
        self.assertEqual(m3_macos_baseline.tile_ids([9, 8, 7], 7),
                         [9, 8, 7, 9, 8, 7, 9])
        self.assertEqual(m3_macos_baseline.PROMPT,
                         "The history of France begins in antiquity.")
        s = m3_macos_baseline.summarize([10.0, 11.0, 12.0, 13.0, 14.0])
        self.assertEqual(s["median_tok_s"], 12.0)

    def test_baseline_window_gate(self):
        nonce = "a" * 32
        with self.assertRaises(SystemExit):
            m3_macos_baseline.check_window(nonce)          # bad/no marker dir
        with self.assertRaises(SystemExit):
            m3_macos_baseline.check_window("not-hex")      # malformed nonce
        marker = Path.home() / f".m2-window-{nonce}"
        marker.mkdir(exist_ok=True)
        try:
            m3_macos_baseline.check_window(nonce)          # admitted
        finally:
            marker.rmdir()


if __name__ == "__main__":
    unittest.main()
