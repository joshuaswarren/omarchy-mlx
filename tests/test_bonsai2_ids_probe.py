"""Failing-first tests for the bonsai2 ids-probe hook (default OFF, env-gated).

The hook contract: per-uid/add_token accumulates, reset/finalize flushes.
The bonsai shim uses the SAME mlx_lm.tokenizer_utils streaming classes,
so the class-level hooks work identically."""
import importlib
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _fake_tokenizer_utils():
    def add_token(self, token):
        self.seen.append(token)
    def reset(self):
        self.seen = []
    def finalize(self):
        self.seen = []
    base = type("StreamingDetokenizer", (),
                {"add_token": add_token, "reset": reset, "finalize": finalize,
                 "__init__": lambda self: setattr(self, "seen", [])})
    # Each subclass defines its OWN override of add_token/finalize in
    # __dict__, mirroring the real mlx_lm 0.31.3 classes (NaiveStreaming
    # Detokenizer etc. override the base methods in their own __dict__).
    naive = type("NaiveStreamingDetokenizer", (base,),
                 {"add_token": add_token, "reset": reset, "finalize": finalize})
    bpe = type("BPEStreamingDetokenizer", (base,),
               {"add_token": add_token, "reset": reset, "finalize": finalize})
    spm = type("SPMStreamingDetokenizer", (base,),
               {"add_token": add_token, "reset": reset, "finalize": finalize})
    mod = types.ModuleType("mlx_lm.tokenizer_utils")
    mod.StreamingDetokenizer = base
    mod.NaiveStreamingDetokenizer = naive
    mod.BPEStreamingDetokenizer = bpe
    mod.SPMStreamingDetokenizer = spm
    return mod


def _fresh_module_under_test():
    for mod in (
            "mlx_omarchy_bonsai2.ids_probe",
            "mlx_omarchy_bonsai2",
            "mlx_lm.generate",
            "mlx_lm.tokenizer_utils",
            "mlx_lm",
    ):
        sys.modules.pop(mod, None)
    sys.path.insert(0, str(REPO_ROOT / "serve"))
    fake_tok = _fake_tokenizer_utils()
    fake_generate = types.ModuleType("mlx_lm.generate")
    fake_generate.stream_generate = lambda *a, **k: iter(())
    sys.modules["mlx_lm"] = types.ModuleType("mlx_lm")
    sys.modules["mlx_lm.tokenizer_utils"] = fake_tok
    sys.modules["mlx_lm.generate"] = fake_generate
    # mlx_lm 0.31.3 flattens mlx_lm.generate to the stream_generate
    # function itself (re-exported at the package level); the hook's
    # fallback path needs the function on the package attr.
    sys.modules["mlx_lm"].generate = fake_generate.stream_generate
    mod = importlib.import_module("mlx_omarchy_bonsai2.ids_probe")
    return mod, fake_tok


class Bonsai2IdsProbeTests(unittest.TestCase):
    def setUp(self):
        self._saved_env = os.environ.copy()
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.mod, self.fake = _fresh_module_under_test()

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self._saved_env)
        for mod in (
                "mlx_omarchy_bonsai2.ids_probe",
                "mlx_omarchy_bonsai2",
                "mlx_lm.generate",
                "mlx_lm.tokenizer_utils",
                "mlx_lm",
        ):
            sys.modules.pop(mod, None)

    def test_noop_without_env(self):
        os.environ[self.mod.PROBE_ENV] = "0"
        state = self.mod.install_ids_probe(emit=lambda e: self.fail(e))
        self.assertIsNone(state)
        self.assertFalse(getattr(self.fake.StreamingDetokenizer,
                                "_ids_probe_installed", False))

    def test_records_ids_and_emits_on_reset_when_env_on(self):
        os.environ[self.mod.PROBE_ENV] = "1"
        events = []
        self.mod.install_ids_probe(emit=events.append)
        d = self.fake.StreamingDetokenizer()
        d.add_token(11)
        d.add_token(22)
        d.reset()
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["ids"], [11, 22])
        self.assertEqual(events[0]["event"], "generation")
        self.assertEqual(len(events[0]["ids_sha16"]), 16)

    def test_records_ids_and_emits_on_finalize_when_env_on(self):
        os.environ[self.mod.PROBE_ENV] = "1"
        events = []
        self.mod.install_ids_probe(emit=events.append)
        d = self.fake.StreamingDetokenizer()
        d.add_token(100)
        d.add_token(200)
        d.add_token(300)
        d.finalize()
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["ids"], [100, 200, 300])
        self.assertEqual(events[0]["event"], "generation")

    def test_subclass_overrides_add_token_and_finalize(self):
        # Subclasses (NaiveStreamingDetokenizer, BPEStreamingDetokenizer,
        # SPMStreamingDetokenizer) in mlx_lm 0.31.3 override add_token
        # and finalize in their own __dict__. The wrap must patch each
        # subclass's override; patching only the base class leaves the
        # override unwrapped and ids accumulate into a black hole.
        os.environ[self.mod.PROBE_ENV] = "1"
        events = []
        self.mod.install_ids_probe(emit=events.append)
        for cls in (self.fake.NaiveStreamingDetokenizer,
                    self.fake.BPEStreamingDetokenizer):
            d = cls()
            d.add_token(7)
            d.add_token(8)
            d.finalize()
        # One flush event per finalize() call across both subclasses.
        self.assertEqual(len(events), 2,
                         f"expected 2 flush events total, got {len(events)}")
        for e in events:
            self.assertEqual(e["ids"], [7, 8])


if __name__ == "__main__":
    unittest.main()
