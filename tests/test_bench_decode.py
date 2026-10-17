"""Tests for bench_decode.py: exact generated-ID recording and error exits.

Stdlib unittest only, no network, no mlx import: the runner's identity
and validation paths run without a GPU.
"""

import json
import os
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch, io  # noqa: F401 - used via patcher below

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import bench_decode  # noqa: E402

SCRIPT = REPO_ROOT / "scripts" / "bench_decode.py"


class IdsDigestTests(unittest.TestCase):
    def test_digest_covers_exact_ids_in_order(self):
        ids = [5, 9214, 0, 151643, 7]
        self.assertEqual(
            bench_decode.ids_digest(ids),
            bench_decode.ids_digest(list(ids)))

    def test_digest_changes_when_any_id_changes(self):
        ids = [10, 20, 30]
        self.assertNotEqual(bench_decode.ids_digest(ids),
                            bench_decode.ids_digest([10, 21, 30]))
        self.assertNotEqual(bench_decode.ids_digest(ids),
                            bench_decode.ids_digest(ids[::-1]))

    def test_digest_is_short_and_ascii_stable(self):
        d = bench_decode.ids_digest([1])
        self.assertEqual(len(d), 16)
        int(d, 16)  # hex


class RunGenerationTests(unittest.TestCase):
    def test_collects_response_token_attribute(self):
        class Resp:
            def __init__(self, token):
                self.token = token

        ids = []
        times, n = bench_decode.run_generation(
            (Resp(i) for i in (3, 1, 4)), ids)
        self.assertEqual((n, len(times)), (3, 3))
        self.assertEqual(ids, [3, 1, 4])

    def test_no_ids_list_still_works(self):
        times, n = bench_decode.run_generation(iter((1, 2)))
        self.assertEqual(n, 2)

    def test_prompt_tokens_from_first_response_only(self):
        class Resp:
            def __init__(self, token, prompt_tokens):
                self.token = token
                self.prompt_tokens = prompt_tokens

        stats = {}
        ids = []
        times, n = bench_decode.run_generation(
            (Resp(i, 123) for i in (3, 1, 4)), ids, stats)
        self.assertEqual((n, ids), (3, [3, 1, 4]))
        self.assertEqual(stats["prompt_tokens"], 123)

    def test_missing_prompt_tokens_is_none_never_zero(self):
        stats = {}
        bench_decode.run_generation(iter((1, 2)), [], stats)
        self.assertIsNone(stats["prompt_tokens"])


class ReportTests(unittest.TestCase):
    def test_short_burst_exits_nonzero_without_rate(self):
        with self.assertRaises(SystemExit) as cm:
            bench_decode.report(prefill_ns=0, token_times=[1, 2, 3],
                                requested=32)
        self.assertEqual(cm.exception.code, 1)

    def test_pinned_run_prints_counted_rate_and_id_lines(self):
        import io

        ids = [7, 8, 9, 10]
        times = [1000, 2000, 3000, 4000]
        with patch("sys.stdout", new=io.StringIO()) as out:
            tps = bench_decode.report(500, times, 4, ids)
        text = out.getvalue()
        self.assertIn("over 3 tokens", text)
        self.assertIn(f"generated_ids sha256:{bench_decode.ids_digest(ids)}",
                      text)
        result = json.loads(text.strip().splitlines()[-1])
        self.assertEqual(result["engine"], "bench_decode")
        self.assertEqual(result["generated"], 4)
        self.assertEqual(result["ids_sha256_16"], bench_decode.ids_digest(ids))
        self.assertEqual(result["ids_first"], [7, 8, 9])
        self.assertEqual(result["ids_last"], [8, 9, 10])
        self.assertAlmostEqual(tps, 1_000_000.0)

    def test_prompt_tokens_line_matches_matrix_contract(self):
        import io

        class Resp:
            def __init__(self, token, prompt_tokens):
                self.token = token
                self.prompt_tokens = prompt_tokens

        ids, stats = [], {}
        times, n = bench_decode.run_generation(
            (Resp(i, 123) for i in (3, 1, 4)), ids, stats)
        self.assertEqual(stats["prompt_tokens"], 123)
        with patch("sys.stdout", new=io.StringIO()) as out:
            bench_decode.report(2_000_000, times, 3, ids,
                                prompt_tokens=stats["prompt_tokens"],
                                device="probe-gpu")
        text = out.getvalue()
        # bench_matrix PROMPT_TOKENS_RE parses exactly this line form.
        self.assertIn("prompt_tokens 123", text)
        self.assertTrue(any(ln == "prompt_tokens 123"
                            for ln in text.splitlines()))
        result = json.loads(text.strip().splitlines()[-1])
        self.assertEqual(result["prompt_tokens"], 123)
        self.assertEqual(result["prefill_tps"], 61500.0)  # 123 / 0.002s
        self.assertEqual(result["device"], "probe-gpu")

    def test_missing_prompt_tokens_emits_no_line_ever_zero(self):
        import io

        stats = {}
        bench_decode.run_generation(iter((1, 2)), [], stats)
        self.assertIsNone(stats["prompt_tokens"])
        with patch("sys.stdout", new=io.StringIO()) as out:
            bench_decode.report(1000, [1, 2], 2, [1, 2],
                                prompt_tokens=stats["prompt_tokens"])
        text = out.getvalue()
        # No plain-text line at all when unknown (consumers parse null
        # from its absence); the JSON key still records null.
        self.assertFalse(any(ln.startswith("prompt_tokens ")
                             for ln in text.splitlines()))
        self.assertNotIn("prompt_tokens unknown", text)
        result = json.loads(text.strip().splitlines()[-1])
        self.assertIsNone(result["prompt_tokens"])
        self.assertIsNone(result["prefill_tps"])


class CliExitTests(unittest.TestCase):
    def test_self_test_passes_without_mlx(self):
        r = subprocess.run([sys.executable, str(SCRIPT), "--self-test"],
                           capture_output=True, text=True, timeout=60)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("self-test: OK", r.stdout)

    def test_missing_model_exits_2(self):
        r = subprocess.run([sys.executable, str(SCRIPT), "--prompt", "p",
                            "--tokens", "8"],
                           capture_output=True, text=True, timeout=60)
        self.assertEqual(r.returncode, 2)

    def test_zero_tokens_exits_2(self):
        r = subprocess.run([sys.executable, str(SCRIPT), "--model", "m",
                            "--prompt", "p", "--tokens", "1"],
                           capture_output=True, text=True, timeout=60)
        self.assertEqual(r.returncode, 2)

    def test_missing_wheel_file_exits_2(self):
        r = subprocess.run([sys.executable, str(SCRIPT), "--model", "m",
                            "--prompt", "p", "--tokens", "8",
                            "--wheel", "/nonexistent/run.whl"],
                           capture_output=True, text=True, timeout=60)
        self.assertEqual(r.returncode, 2)
        self.assertIn("does not exist", r.stderr)


class ClampOrderTests(unittest.TestCase):
    def test_clamp_runs_before_mlx_is_imported(self):
        src = SCRIPT.read_text()
        clamp = src.index("perf_placement import apply_from_env")
        self.assertLess(clamp, src.index("installed_provenance("))
        self.assertLess(clamp, src.index("from mlx_lm.utils import load"))
        self.assertLess(clamp, src.index("import mlx.core as mx"))

    def test_clamp_line_prints_before_the_run_fails(self):
        """Behavior check: with the switch off the script reports it on stdout
        before it fails on the missing model (no syscall happens)."""
        env = dict(os.environ, MLX_OMARCHY_UCLAMP_MIN="0", HF_HUB_OFFLINE="1")
        r = subprocess.run([sys.executable, str(SCRIPT), "--model", "/nonexistent/model-dir",
                            "--prompt", "p", "--tokens", "8"],
                           capture_output=True, text=True, timeout=120, env=env, check=False)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("uclamp: MLX_OMARCHY_UCLAMP_MIN off or invalid", r.stdout)


if __name__ == "__main__":
    unittest.main()
