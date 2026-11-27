"""scripts/wheel_card_row.py: card-row formatting and the deferral guard."""
import importlib.util
import io
import os
from contextlib import redirect_stdout
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "wheel_card_row", ROOT / "scripts" / "wheel_card_row.py")
wheel_card_row = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wheel_card_row)


def fake_result() -> dict:
    return {
        "decode": {"median_tok_s": 65.123, "output_sha256": ["efb521ae04af" + "0" * 52]},
        "prefill": {"median_tok_s": 551.72},
    }


def row_text(result, **kw) -> str:
    buf = io.StringIO()
    args = dict(machine="16-inch M1 Max MacBook Pro (32-core GPU)", model_label="4b",
                ctx=512, result=result, macos_decode=None, macos_prefill=None,
                mesa="1650fc1a33", pr="")
    args.update(kw)
    with redirect_stdout(buf):
        wheel_card_row.print_row(**args)
    return buf.getvalue()


class CardRowTest(unittest.TestCase):
    def test_row_has_every_card_column(self):
        out = row_text(fake_result())
        line = next(l for l in out.splitlines() if l.startswith("| 16-inch"))
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        self.assertEqual(len(cells), 12)
        self.assertEqual(cells[0], "16-inch M1 Max MacBook Pro (32-core GPU)")
        self.assertIn("Qwen3-4B-Instruct-2507-4bit", cells[1])
        self.assertIn("50d42775", cells[1])
        self.assertEqual(cells[2], "512")
        self.assertEqual(cells[3], "65.12 tok/s")
        self.assertEqual(cells[4], "551.72 tok/s")
        self.assertEqual(cells[5], "pending")
        self.assertEqual(cells[6], "pending")
        self.assertEqual(cells[7], "pending")
        self.assertEqual(cells[9], "`efb521ae04af`")
        self.assertEqual(cells[10], "pending")
        self.assertEqual(cells[11], "`1650fc1a33`")

    def test_share_from_macos_numbers(self):
        out = row_text(fake_result(), macos_decode=104.53, macos_prefill=660.26)
        self.assertIn("62.3% (−37.7%)", out)
        self.assertIn("104.53 tok/s", out)
        self.assertIn("660.26 tok/s", out)

    def test_share_sign_positive(self):
        out = row_text(fake_result(), macos_decode=50.0)
        self.assertIn("(+30.2%)", out)

    def test_guard_pins_unset_defer_to_zero(self):
        saved = os.environ.pop("MLX_OMARCHY_DEFER_TOKENS", None)
        try:
            wheel_card_row.check_guard()
            self.assertEqual(os.environ["MLX_OMARCHY_DEFER_TOKENS"], "0")
        finally:
            if saved is None:
                os.environ.pop("MLX_OMARCHY_DEFER_TOKENS", None)
            else:
                os.environ["MLX_OMARCHY_DEFER_TOKENS"] = saved

    def test_guard_refuses_nonzero_defer(self):
        saved = os.environ.get("MLX_OMARCHY_DEFER_TOKENS")
        os.environ["MLX_OMARCHY_DEFER_TOKENS"] = "8"
        try:
            with self.assertRaises(SystemExit):
                wheel_card_row.check_guard()
        finally:
            if saved is None:
                os.environ.pop("MLX_OMARCHY_DEFER_TOKENS", None)
            else:
                os.environ["MLX_OMARCHY_DEFER_TOKENS"] = saved

    def test_digest_is_stable_sha256(self):
        self.assertEqual(
            wheel_card_row.digest([1, 2, 3]),
            "8a6ae15122001229edb8866f56e342af12ae8187203c3e3b33931743e7c0c48d")


if __name__ == "__main__":
    unittest.main()
