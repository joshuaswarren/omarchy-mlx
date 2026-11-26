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

    def _with_upstream_module(self, env):
        """Plant a stand-in for mlx-lm's own muse_spark module, run register(), return (result, module now registered). Restores the real state."""
        import os
        import types
        from unittest import mock

        name = mlx_omarchy_muse.MODEL_NAME
        mlx_omarchy_muse.register()
        ours = sys.modules[name]
        upstream = types.ModuleType(name)
        upstream.__file__ = "/upstream/mlx_lm/models/muse_spark.py"
        sys.modules[name] = upstream
        try:
            with mock.patch.dict(os.environ, env):
                if "MLX_OMARCHY_MUSE_UPSTREAM" not in env:
                    os.environ.pop("MLX_OMARCHY_MUSE_UPSTREAM", None)
                result = mlx_omarchy_muse.register()
            return result, sys.modules[name], upstream
        finally:
            sys.modules[name] = ours
            importlib.import_module("mlx_lm.models").muse_spark = ours

    def test_register_replaces_an_mlx_lm_module_of_the_same_name(self):
        result, now, upstream = self._with_upstream_module({})
        self.assertTrue(result)
        self.assertIsNot(now, upstream)
        self.assertEqual(now.__file__, str(mlx_omarchy_muse.MODEL_PATH))

    def test_upstream_switch_keeps_the_mlx_lm_module(self):
        result, now, upstream = self._with_upstream_module({"MLX_OMARCHY_MUSE_UPSTREAM": "1"})
        self.assertFalse(result)
        self.assertIs(now, upstream)

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


_PRUNE_SCRIPT = """
import importlib, json, sys
import mlx.core as mx
import mlx_omarchy_muse
mlx_omarchy_muse.register()
module = importlib.import_module(mlx_omarchy_muse.MODEL_NAME)
config = json.loads(sys.argv[1])
class Group:
    def __init__(self, rank, size): self._r, self._s = rank, size
    def rank(self): return self._r
    def size(self): return self._s
out = {}
for rank in (0, 1):
    model = module.Model(module.ModelArgs.from_dict(config))
    real = model.model.embed(mx.array([[1, 2, 3]]))
    model.model.pipeline(Group(rank, 2))
    got = model.model.embed(mx.array([[1, 2, 3]]))
    out[rank] = {"embed": type(model.model.embed_tokens).__name__, "head": type(getattr(model, "lm_head", None)).__name__,
                 "shape": list(got.shape) == list(real.shape), "dtype": got.dtype == real.dtype,
                 "vocab": model.lm_head.weight.shape[0] if "lm_head" in model and "weight" in model.lm_head else None}
print(json.dumps(out))
"""


class PruneTest(unittest.TestCase):
    def _run(self, flag, tie=False):
        import os
        import subprocess

        reference = json.loads((FIXTURES / "muse_tiny_reference.json").read_text())
        config = {**reference["config"], "model_type": "muse_spark", "tie_word_embeddings": tie}
        env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(REPO / "serve")]), "MLX_OMARCHY_PIPELINE_PRUNE": flag}
        done = subprocess.run([sys.executable, "-c", _PRUNE_SCRIPT, json.dumps(config)], env=env, capture_output=True, text=True, check=True)
        return json.loads(done.stdout.strip().splitlines()[-1]), config

    def test_off_keeps_every_tensor(self):
        out, _ = self._run("0")
        for rank in ("0", "1"):
            self.assertEqual((out[rank]["embed"], out[rank]["head"]), ("Embedding", "Linear"))

    def test_on_keeps_everything_when_embeddings_are_tied(self):
        out, _ = self._run("1", tie=True)
        for rank in ("0", "1"):
            self.assertEqual(out[rank]["embed"], "Embedding")
            self.assertEqual(out[rank]["head"], "NoneType")

    def test_on_keeps_embed_on_the_first_layer_rank_and_the_head_on_rank_zero(self):
        out, config = self._run("1")
        self.assertEqual((out["0"]["embed"], out["0"]["head"]), ("_ZeroEmbed", "Linear"))
        self.assertEqual((out["1"]["embed"], out["1"]["head"]), ("Embedding", "_NoHead"))
        self.assertEqual(out["1"]["vocab"], config["vocab_size"])
        for rank in ("0", "1"):
            self.assertTrue(out[rank]["shape"] and out[rank]["dtype"])


if __name__ == "__main__":
    unittest.main()
