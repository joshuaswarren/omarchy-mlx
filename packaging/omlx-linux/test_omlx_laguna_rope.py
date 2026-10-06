#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Contract tests for the Laguna rope_parameters guard (patch 0007).

Laguna checkpoints ship ``rope_parameters`` as a per-layer-type map
(``full_attention`` / ``sliding_attention`` dicts) plus non-dict top-level
scalars (``original_max_position_embeddings``). Transformers'
``RotaryEmbeddingConfigMixin.validate_rope`` calls ``.get`` on every value,
so the scalar aborts tokenizer config validation with ``'int' object has no
attribute 'get'``. The patch installs a guard that coerces the map to the
nested-only shape the upstream remote-code ``LagunaConfig.__post_init__``
produces, without enabling ``trust_remote_code``.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

OMLX_SRC = Path(os.environ.get("OMLX_SRC", "/tmp/omlx-applied"))
sys.path.insert(0, str(OMLX_SRC))

from omlx.utils.laguna_rope import (
    coerce_nested_rope_parameters,
    install_rope_parameters_guard,
)

FIXTURE = Path(__file__).resolve().parent / "fixtures/laguna-hf-config.json"


class CoerceNestedRopeParameters(unittest.TestCase):
    def test_drops_scalars_from_layer_type_map(self):
        rope = {
            "full_attention": {"rope_type": "yarn", "factor": 32.0},
            "sliding_attention": {"rope_type": "default"},
            "original_max_position_embeddings": 4096,
        }

        coerced = coerce_nested_rope_parameters(rope)

        self.assertEqual(
            coerced,
            {
                "full_attention": {"rope_type": "yarn", "factor": 32.0},
                "sliding_attention": {"rope_type": "default"},
            },
        )

    def test_captured_laguna_config_coerces_to_layer_dicts(self):
        config = json.loads(FIXTURE.read_text())

        coerced = coerce_nested_rope_parameters(config["rope_parameters"])

        self.assertEqual(set(coerced), {"full_attention", "sliding_attention"})
        self.assertTrue(all(isinstance(v, dict) for v in coerced.values()))
        self.assertEqual(
            coerced["full_attention"]["original_max_position_embeddings"], 4096
        )

    def test_flat_rope_map_is_returned_unchanged(self):
        rope = {"rope_type": "yarn", "factor": 32.0, "beta_fast": 64.0}

        self.assertIs(coerce_nested_rope_parameters(rope), rope)

    def test_non_dict_is_returned_unchanged(self):
        self.assertIs(coerce_nested_rope_parameters(42), 42)
        self.assertIs(coerce_nested_rope_parameters(None), None)


try:
    from transformers.modeling_rope_utils import RotaryEmbeddingConfigMixin
except ImportError:  # pragma: no cover - dev boxes without transformers
    RotaryEmbeddingConfigMixin = None


@unittest.skipIf(RotaryEmbeddingConfigMixin is None, "transformers unavailable")
class RopeParametersGuard(unittest.TestCase):
    def test_guard_coerces_before_delegate_and_is_idempotent(self):
        recorded = {}

        def recorder(self):
            recorded["rope"] = self.rope_parameters

        original = RotaryEmbeddingConfigMixin.validate_rope
        RotaryEmbeddingConfigMixin.validate_rope = recorder
        try:
            self.assertTrue(install_rope_parameters_guard())
            self.assertFalse(install_rope_parameters_guard())

            stub = SimpleNamespace(
                rope_parameters={
                    "full_attention": {"rope_type": "yarn"},
                    "original_max_position_embeddings": 4096,
                }
            )
            RotaryEmbeddingConfigMixin.validate_rope(stub)

            self.assertEqual(
                recorded["rope"], {"full_attention": {"rope_type": "yarn"}}
            )
        finally:
            RotaryEmbeddingConfigMixin.validate_rope = original

    def test_guard_leaves_flat_maps_untouched(self):
        recorded = {}

        def recorder(self):
            recorded["rope"] = self.rope_parameters

        original = RotaryEmbeddingConfigMixin.validate_rope
        RotaryEmbeddingConfigMixin.validate_rope = recorder
        try:
            install_rope_parameters_guard()

            flat = {"rope_type": "yarn", "factor": 32.0}
            stub = SimpleNamespace(rope_parameters=flat)
            RotaryEmbeddingConfigMixin.validate_rope(stub)

            self.assertIs(recorded["rope"], flat)
        finally:
            RotaryEmbeddingConfigMixin.validate_rope = original


if __name__ == "__main__":
    unittest.main()
