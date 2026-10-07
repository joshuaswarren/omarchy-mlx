"""Bonsai-2 pack loader tests: reload exactness, accounting, refusals (CPU).

The fixture is a real tiny prism_hadamard_qwen35 pack (schema 2) built
in-memory; the reference is the same in-memory TextModel the pack was
serialized from, so a correct loader must reproduce its logits bit-exactly
(the pack's own save_and_verify contract). Refusal cases mutate the pack
config and must fail with PackError, never approximate.
"""

import hashlib
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "serve") not in sys.path:
    sys.path.insert(0, str(REPO / "serve"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pytest

mx = pytest.importorskip("mlx.core")

import bonsai2_fixture
from bonsai2_fixture import TINY_TEXT_CONFIG, build_tiny_pack, packed_targets
from mlx_omarchy_bonsai2.loader import PackError, load_text_model


def _last_logits(model, ids):
    logits = model(mx.array([ids]))
    return np.array(logits[:, -1, :].astype(mx.float32))


class LoaderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        cls.pack_dir, cls.reference = build_tiny_pack(cls.tmp)
        cls.loaded, cls.info = load_text_model(cls.pack_dir)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_reload_logits_are_bit_exact(self):
        ids = [3, 4, 5, 6, 7, 8]
        np.testing.assert_array_equal(
            _last_logits(self.loaded, ids), _last_logits(self.reference, ids)
        )

    def test_cache_bearing_decode_matches_reference(self):
        from mlx_lm.models.cache import ArraysCache, KVCache

        def decode(model):
            cache = [ArraysCache(size=2) if l.is_linear else KVCache() for l in model.layers]
            out = []
            x = mx.array([[3, 4, 5]])
            for step in range(4):
                logits = model(x if step == 0 else x[:, -1:], cache=cache)
                out.append(int(mx.argmax(logits[0, -1])))
            return out

        self.assertEqual(decode(self.loaded), decode(self.reference))

    def test_load_info_is_complete(self):
        info = self.info
        self.assertEqual(info["model_type"], "prism_hadamard_qwen35")
        self.assertEqual(info["quantization"], {"bits": 2, "group_size": 128, "mode": "affine"})
        self.assertEqual(info["packed_modules"], len(packed_targets(TINY_TEXT_CONFIG)))
        self.assertEqual(info["hadamard_block"], [bonsai2_fixture.BLOCK])
        self.assertGreater(info["resident_bytes"], 0)
        self.assertTrue(all(v for v in info["license_files"].values()))
        self.assertEqual(
            info["config_sha256"],
            hashlib.sha256((self.pack_dir / "config.json").read_bytes()).hexdigest(),
        )
        self.assertEqual(info["max_position_embeddings"], TINY_TEXT_CONFIG["max_position_embeddings"])

    def test_vision_tensors_excluded_and_accounted(self):
        self.assertIn("visual", self.info["excluded_bytes"])
        self.assertGreater(self.info["excluded_bytes"]["visual"], 0)
        header_bytes = 2 * 2 * 2  # fake visual.tower.weight: (2,2) float16
        self.assertEqual(self.info["excluded_bytes"]["visual"], header_bytes)


class LoaderRefusalTests(unittest.TestCase):
    def _mutated_pack(self, mutate):
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        pack_dir, _ = build_tiny_pack(tmp)
        config = json.loads((pack_dir / "config.json").read_text())
        mutate(config)
        (pack_dir / "config.json").write_text(json.dumps(config))
        return pack_dir

    def test_rejects_wrong_schema_version(self):
        pack_dir = self._mutated_pack(lambda c: c.update(schema_version=1))
        with self.assertRaises(PackError):
            load_text_model(pack_dir)

    def test_rejects_wrong_model_type(self):
        pack_dir = self._mutated_pack(lambda c: c.update(model_type="qwen3_5"))
        with self.assertRaises(PackError):
            load_text_model(pack_dir)

    def test_rejects_wrong_quantization(self):
        pack_dir = self._mutated_pack(lambda c: c["quantization"].update(bits=4))
        with self.assertRaises(PackError):
            load_text_model(pack_dir)

    def test_rejects_unsupported_hadamard_block(self):
        def mutate(config):
            config["modules"][0]["block"] = 768

        pack_dir = self._mutated_pack(mutate)
        with self.assertRaises(PackError):
            load_text_model(pack_dir)

    def test_rejects_record_with_unexpected_sign_but_no_block(self):
        def mutate(config):
            config["modules"][0]["block"] = 0

        pack_dir = self._mutated_pack(mutate)
        with self.assertRaises(PackError):
            load_text_model(pack_dir)

    def test_rejects_non_float16_record(self):
        def mutate(config):
            config["modules"][0]["dtype"] = "bfloat16"

        pack_dir = self._mutated_pack(mutate)
        with self.assertRaises(PackError):
            load_text_model(pack_dir)

    def test_rejects_inconsistent_layer_types(self):
        def mutate(config):
            tc = dict(config["text_config"])
            tc["layer_types"] = ["full_attention"] * tc["num_hidden_layers"]
            config["text_config"] = tc

        pack_dir = self._mutated_pack(mutate)
        with self.assertRaises(PackError):
            load_text_model(pack_dir)


if __name__ == "__main__":
    print("provenance: mlx %s on %s (CPU reference)" % (mx.__version__, mx.default_device()))
    unittest.main()
