"""Muse Spark adapter tests (CPU): registration, community config loading, and logits against the transformers reference.

The reference logits in fixtures/muse/muse_tiny_reference.json come from transformers MuseSparkTextModel (fp32, CPU) on the tiny weights
of fixtures/muse/weights.py. The test rebuilds those weights from numpy, loads them through the vendored class and compares every
position. The fixture is tiny (hidden size 64, four layers, four experts, sliding window 8, 24 tokens) and it is not a model-quality test.
"""

import importlib
import json
import sys
import unittest
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).resolve().parent / "fixtures" / "muse"
for path in (REPO / "serve", FIXTURES):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")

import mlx_omarchy_muse  # noqa: E402
import weights  # noqa: E402

# Measured with mlx-lm 0.32.0 and mlx 0.32.3 on CPU: 9.6e-06 one-shot and 9.3e-06 incremental, against logits up to 0.90.
# Each bound is about ten times the measured error.
ONE_SHOT_MAX_ABS = 1e-4
INCREMENTAL_MAX_ABS = 1e-4


def _model():
    mlx_omarchy_muse.register()
    module = importlib.import_module(mlx_omarchy_muse.MODEL_NAME)
    state, lm_head, ids = weights.make_weights()
    reference = json.loads((FIXTURES / "muse_tiny_reference.json").read_text())
    args = module.ModelArgs.from_dict({**reference["config"], "model_type": "muse_spark"})
    model = module.Model(args)
    loaded = {"model.language_model." + k: mx.array(v) for k, v in state.items()}
    loaded["lm_head.weight"] = mx.array(lm_head)
    model.load_weights(list(model.sanitize(loaded).items()), strict=True)
    mx.eval(model.parameters())
    return model, ids, np.array(reference["logits"], dtype=np.float32), reference


class RegistrationTests(unittest.TestCase):
    def test_vendored_file_is_unchanged(self):
        import hashlib

        digest = hashlib.sha256(mlx_omarchy_muse.MODEL_PATH.read_bytes()).hexdigest()
        self.assertEqual(digest, mlx_omarchy_muse.SOURCE_SHA256)

    def test_register_is_idempotent(self):
        mlx_omarchy_muse.register()
        self.assertIsNotNone(importlib.import_module(mlx_omarchy_muse.MODEL_NAME))
        self.assertFalse(mlx_omarchy_muse.register())

    def test_published_config_resolves(self):
        from mlx_lm import utils

        mlx_omarchy_muse.register()
        config = json.loads((FIXTURES / "community_config_3bit.json").read_text())
        self.assertEqual(config["model_type"], "muse_spark")
        model_class, args_class = utils._get_classes(dict(config))
        args = args_class.from_dict(config)
        self.assertEqual(model_class.__module__, mlx_omarchy_muse.MODEL_NAME)
        self.assertEqual((args.hidden_size, args.num_hidden_layers, args.num_local_experts, args.num_experts_per_tok), (8192, 62, 256, 8))
        self.assertEqual(args.layer_types.count("full_attention"), 16)
        self.assertEqual(args.layer_types.count("sliding_attention"), 46)


class ReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model, cls.ids, cls.ref, cls.reference = _model()

    def test_fixture_weights_match_the_reference_run(self):
        state, lm_head, _ = weights.make_weights()
        self.assertEqual(weights.weights_digest(state, lm_head), self.reference["weights_sha256"])

    def test_logits_match_reference_at_every_position(self):
        out = np.array(self.model(mx.array(self.ids)[None])[0])
        self.assertEqual(out.shape, self.ref.shape)
        diff = float(np.abs(out - self.ref).max())
        self.assertLess(diff, ONE_SHOT_MAX_ABS, f"max abs {diff} against reference magnitude {float(np.abs(self.ref).max())}")
        self.assertTrue(np.array_equal(out.argmax(-1), self.ref.argmax(-1)))

    def test_incremental_decode_crosses_the_sliding_window(self):
        from mlx_lm.models.cache import make_prompt_cache

        self.assertGreater(len(self.ids), self.reference["config"]["sliding_window"] * 2)
        cache = make_prompt_cache(self.model)
        start = 12
        rows = [np.array(self.model(mx.array(self.ids[:start])[None], cache=cache)[0, -1])]
        for token in self.ids[start:]:
            rows.append(np.array(self.model(mx.array([[int(token)]]), cache=cache)[0, -1]))
        got = np.stack(rows)
        want = self.ref[start - 1 :]
        diff = float(np.abs(got - want).max())
        self.assertLess(diff, INCREMENTAL_MAX_ABS, f"max abs {diff}")
        self.assertTrue(np.array_equal(got.argmax(-1), want.argmax(-1)))

    def test_layer_schedule_has_both_attention_kinds(self):
        self.assertEqual(sorted(set(self.reference["layer_types"])), ["full_attention", "sliding_attention"])


if __name__ == "__main__":
    unittest.main()
