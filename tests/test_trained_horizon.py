"""The trained-window horizon (issue 468, minefield 55/61): a request past the
trained window is a warning on the wire, never a silent quality drop, and the
bundled catalog must say unknown rather than inherit the advertised window."""
import contextlib
import io
import json
import unittest
from pathlib import Path

from mlx_omarchy_serve import _mlxlm_server as shim

EXPECTED = tuple(["self", "tokenizer", "request", "args"])


class FakeGenerator:
    _tokenize = staticmethod(lambda self, tokenizer, request, args: ([0] * 5, "x"))


class TrainedHorizonTests(unittest.TestCase):
    def test_a_request_past_the_trained_window_warns_on_stderr(self):
        shim.EXPECTED_TOKENIZE_SIGNATURE = EXPECTED
        module = type("M", (), {"ResponseGenerator": FakeGenerator})
        shim.install(module, limit=1000, trained=6)
        gen = module.ResponseGenerator()
        args = type("A", (), {"max_tokens": 10})()
        buffer = io.StringIO()
        with contextlib.redirect_stderr(buffer):
            gen._tokenize(None, None, args)  # 5 prompt + 10 output = 15, past 6, under 1000
        self.assertIn("trained window", buffer.getvalue())

    def test_a_request_inside_the_trained_window_is_silent(self):
        shim.EXPECTED_TOKENIZE_SIGNATURE = EXPECTED
        module = type("M", (), {"ResponseGenerator": FakeGenerator})
        shim.install(module, limit=1000, trained=100)
        gen = module.ResponseGenerator()
        args = type("A", (), {"max_tokens": 10})()
        buffer = io.StringIO()
        with contextlib.redirect_stderr(buffer):
            gen._tokenize(None, None, args)  # 15 total, inside 100
        self.assertEqual(buffer.getvalue(), "")

    def test_every_bundled_entry_states_its_trained_window_even_when_unknown(self):
        bundled = json.loads(Path(shim.__file__).with_name("catalog.json").read_text())
        for entry in (bundled["models"] if isinstance(bundled, dict) else bundled):
            self.assertIn("trained_tokens", entry["context"], entry["id"])


if __name__ == "__main__":
    unittest.main()
