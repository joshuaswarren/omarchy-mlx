"""Tests for MLX_OMARCHY_DEFER_TOKENS benchmark timing guard.

Stdlib unittest only, no network, no mlx import. Verifies that standard
benchmark runners pin MLX_OMARCHY_DEFER_TOKENS=0, record the value in their
output, and refuse to print tok/s when it is unset while the installed
mlx_lm.generate.generate_step defaults to deferred yield.
"""

import io
import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import bench_decode  # noqa: E402
import bench_matrix  # noqa: E402
import defer_tokens_guard  # noqa: E402
import subcap_ab_bench  # noqa: E402


PATCHED_GENERATE_STEP_SRC = '''
def generate_step(prompt, model, *, max_tokens=-1):
    defer_tokens = int(os.environ.get("MLX_OMARCHY_DEFER_TOKENS", "8"))
    return defer_tokens
'''

STOCK_GENERATE_STEP_SRC = '''
def generate_step(prompt, model, *, max_tokens=-1):
    yield 0, None
'''

ZERO_DEFAULT_GENERATE_STEP_SRC = '''
def generate_step(prompt, model, *, max_tokens=-1):
    defer_tokens = int(os.environ.get("MLX_OMARCHY_DEFER_TOKENS", "0"))
    return defer_tokens
'''


class DeferTokensGuardTests(unittest.TestCase):
    def test_detects_patched_default_vs_stock(self):
        self.assertEqual(
            defer_tokens_guard.package_default_defers(PATCHED_GENERATE_STEP_SRC),
            (True, 8),
        )
        self.assertEqual(
            defer_tokens_guard.package_default_defers(STOCK_GENERATE_STEP_SRC),
            (False, 0),
        )
        self.assertEqual(
            defer_tokens_guard.package_default_defers(ZERO_DEFAULT_GENERATE_STEP_SRC),
            (False, 0),
        )

    def test_pin_sets_zero_when_unset_and_preserves_explicit(self):
        env = {}
        self.assertEqual(defer_tokens_guard.pin_defer_tokens(env), "0")
        self.assertEqual(env["MLX_OMARCHY_DEFER_TOKENS"], "0")

        env_explicit = {"MLX_OMARCHY_DEFER_TOKENS": "4"}
        self.assertEqual(defer_tokens_guard.pin_defer_tokens(env_explicit), "4")
        self.assertEqual(env_explicit["MLX_OMARCHY_DEFER_TOKENS"], "4")

    def test_refuses_when_unset_and_package_default_defers(self):
        err = io.StringIO()
        with patch("sys.stderr", err):
            with self.assertRaises(SystemExit) as cm:
                defer_tokens_guard.require_defer_tokens(
                    "bench_decode",
                    env={},
                    generate_step_fn=PATCHED_GENERATE_STEP_SRC,
                )
        self.assertEqual(cm.exception.code, 3)
        self.assertIn("REFUSING TO EMIT TOK/S", err.getvalue())
        self.assertIn("MLX_OMARCHY_DEFER_TOKENS", err.getvalue())

    def test_allows_unset_on_stock_non_deferring_package(self):
        got = defer_tokens_guard.require_defer_tokens(
            "bench_decode",
            env={},
            generate_step_fn=STOCK_GENERATE_STEP_SRC,
        )
        self.assertIsNone(got)

    def test_bench_decode_report_refuses_without_printing_tok_s(self):
        out = io.StringIO()
        err = io.StringIO()
        env = {k: v for k, v in os.environ.items()
               if k != "MLX_OMARCHY_DEFER_TOKENS"}
        with patch.dict(os.environ, env, clear=True), \
             patch.object(
                 defer_tokens_guard, "package_default_defers",
                 return_value=(True, 8)), \
             patch("sys.stdout", out), \
             patch("sys.stderr", err):
            with self.assertRaises(SystemExit) as cm:
                bench_decode.report(
                    prefill_ns=500,
                    token_times=[1000, 2000, 3000, 4000],
                    requested=4,
                    ids=[7, 8, 9, 10],
                )
        self.assertEqual(cm.exception.code, 3)
        self.assertNotIn("tok/s", out.getvalue())
        self.assertIn("REFUSING TO EMIT TOK/S", err.getvalue())

    def test_bench_decode_records_defer_tokens_in_stdout_json_and_matrix(self):
        out = io.StringIO()
        with patch.dict(os.environ, {"MLX_OMARCHY_DEFER_TOKENS": "0"}), \
             patch("sys.stdout", out):
            bench_decode.report(
                prefill_ns=500,
                token_times=[1000, 2000, 3000, 4000],
                requested=4,
                ids=[7, 8, 9, 10],
            )
        text = out.getvalue()
        self.assertIn("defer_tokens 0", text)
        parsed = json.loads(text.strip().splitlines()[-1])
        self.assertEqual(parsed["defer_tokens"], "0")
        metrics = bench_matrix.parse_bench_output(text)
        self.assertEqual(metrics["defer_tokens"], "0")

    def test_bench_matrix_manifest_pins_defer_tokens_zero(self):
        manifest = json.loads(
            (REPO_ROOT / "scripts" / "bench_matrix.json").read_text()
        )
        self.assertEqual(
            manifest["generation"]["env"].get("MLX_OMARCHY_DEFER_TOKENS"),
            "0",
        )

    def test_subcap_ab_driver_guards_and_records_defer_tokens(self):
        self.assertIn("require_defer_tokens", subcap_ab_bench.DECODE_DRIVER)
        self.assertIn('"defer_tokens": defer_tokens', subcap_ab_bench.DECODE_DRIVER)

    def test_qualification_kit_guard_matches_scripts_guard(self):
        a = (REPO_ROOT / "scripts" / "defer_tokens_guard.py").read_text()
        b = (REPO_ROOT / "qualification-kit" / "scripts" / "defer_tokens_guard.py").read_text()
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()
