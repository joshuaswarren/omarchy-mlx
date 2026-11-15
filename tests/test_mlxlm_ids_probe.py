"""Tests for the managed shim's env-gated ids probe (MLX_OMARCHY_SERVE_IDS_PROBE).

Failing-first contract (Main): exact-token comparison on the managed route
needs real generated id arrays from the managed child. The probe hook must:
  1. record every raw token id at the add_token boundary (pre-detok), and
  2. emit ONE PROBE: JSON line per request at reset() with ids + sha16,
  3. wrap ALL registered detokenizer classes,
  4. never be installed when the env is unset (main() gates it).

A fake detokenizer hierarchy + fake mlx_lm.tokenizer_utils module stands in
for the real classes; no MLX required.
"""

import contextlib
import io
import json
import sys
import unittest
import unittest.mock
import types
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "serve"))

from mlx_omarchy_serve import _mlxlm_server  # noqa: E402


class _FakeDetok:
    def __init__(self):
        self.text = []

    def add_token(self, token):
        self.text.append(str(token))

    def reset(self):
        self.text = []


def _fake_tokenizer_utils():
    """A stand-in mlx_lm.tokenizer_utils with a small class family.
    The classes carry real add_token/reset (the probe wraps EXISTING
    methods; a class without them is skipped by design)."""
    def add_token(self, token):
        self.seen.append(token)

    def reset(self):
        self.seen = []

    base = type("StreamingDetokenizer", (),
                {"add_token": add_token, "reset": reset,
                 "__init__": lambda self: setattr(self, "seen", [])})
    naive = type("NaiveStreamingDetokenizer", (base,), {})
    bpe = type("BPEStreamingDetokenizer", (base,), {})
    mod = types.ModuleType("mlx_lm.tokenizer_utils")
    mod.StreamingDetokenizer = base
    mod.NaiveStreamingDetokenizer = naive
    mod.BPEStreamingDetokenizer = bpe
    mod.SPMStreamingDetokenizer = None  # absent -> skipped by wrap()
    return mod


class IdsProbeTests(unittest.TestCase):
    def test_records_ids_and_emits_on_reset(self):
        fake = _fake_tokenizer_utils()
        events = []
        _mlxlm_server.install_ids_probe(
            None, emit=events.append) if False else None
        with unittest.mock.patch.dict(
                sys.modules, {"mlx_lm": types.ModuleType("mlx_lm"),
                              "mlx_lm.tokenizer_utils": fake}):
            _mlxlm_server.install_ids_probe(None, emit=events.append)
        for cls in (fake.StreamingDetokenizer, fake.NaiveStreamingDetokenizer,
                    fake.BPEStreamingDetokenizer):
            d = cls()
            d.add_token(1596)
            d.add_token(1144)
            d.reset()
        self.assertEqual(len(events), 3)
        for ev in events:
            self.assertEqual(ev["event"], "generation")
            self.assertEqual(ev["ids"], [1596, 1144])
            self.assertEqual(ev["n"], 2)
            self.assertEqual(len(ev["ids_sha16"]), 16)

    def test_stderr_emit_matches_probe_line_format(self):
        fake = _fake_tokenizer_utils()
        err = io.StringIO()
        with unittest.mock.patch.dict(
                sys.modules, {"mlx_lm": types.ModuleType("mlx_lm"),
                              "mlx_lm.tokenizer_utils": fake}):
            with contextlib.redirect_stderr(err):
                _mlxlm_server.install_ids_probe(None)
                d = fake.StreamingDetokenizer()
                d.add_token(42)
                d.reset()
        line = err.getvalue().splitlines()[0]
        self.assertTrue(line.startswith("PROBE: "), line)
        ev = json.loads(line.split("PROBE: ", 1)[1])
        self.assertEqual(ev["ids"], [42])

    def test_main_skips_probe_without_env(self):
        calls = []
        fake_mod = types.ModuleType("mlx_lm")
        fake_mod.server = types.SimpleNamespace(main=lambda: calls.append("serve"))
        with unittest.mock.patch.dict(
                os.environ, {_mlxlm_server.PROBE_ENV: "",
                             _mlxlm_server.LIMIT_ENV: "4096"}):
            with unittest.mock.patch.dict(
                    sys.modules, {"mlx_lm": fake_mod,
                                  "mlx_lm.server": fake_mod.server}), \
                 unittest.mock.patch.object(_mlxlm_server, "check_pinned",
                                            lambda m: "0.31.3"), \
                 unittest.mock.patch.object(_mlxlm_server, "install",
                                            lambda m, l, trained=None: calls.append("install")), \
                 unittest.mock.patch.object(_mlxlm_server, "install_ids_probe",
                                            lambda m: calls.append("ids_probe")):
                _mlxlm_server.main()
        self.assertEqual(calls, ["install", "serve"],
                         "ids probe must NOT be installed when env unset")

    def test_main_installs_probe_with_env(self):
        calls = []
        fake_mod = types.ModuleType("mlx_lm")
        fake_mod.server = types.SimpleNamespace(main=lambda: calls.append("serve"))
        with unittest.mock.patch.dict(
                os.environ, {_mlxlm_server.PROBE_ENV: "1",
                             _mlxlm_server.LIMIT_ENV: "4096"}):
            with unittest.mock.patch.dict(
                    sys.modules, {"mlx_lm": fake_mod,
                                  "mlx_lm.server": fake_mod.server}), \
                 unittest.mock.patch.object(_mlxlm_server, "check_pinned",
                                            lambda m: "0.31.3"), \
                 unittest.mock.patch.object(_mlxlm_server, "install",
                                            lambda m, l, trained=None: calls.append("install")), \
                 unittest.mock.patch.object(_mlxlm_server, "install_ids_probe",
                                            lambda m: calls.append("ids_probe")):
                _mlxlm_server.main()
        self.assertEqual(calls, ["install", "ids_probe", "serve"])


import os  # noqa: E402  (used in the env-gated tests above)

if __name__ == "__main__":
    unittest.main()
