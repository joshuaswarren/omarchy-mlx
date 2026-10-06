#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Contract tests for STS audio discovery without model_type (patch 0008).

mlx-community/DeepFilterNet-mlx and mlx-community/sam-audio-small ship
complete checkpoints whose config.json has neither model_type nor
architectures, so discovery classified them as llm and the LLM loader
aborted with KeyError: 'model_type'. The STS engine already resolves
these families from directory-name hints and the checkpoints carry family-
specific config shapes; discovery must use the same signals.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

OMLX_SRC = Path(os.environ.get("OMLX_SRC", "/tmp/omlx-applied"))
sys.path.insert(0, str(OMLX_SRC))

from omlx.model_discovery import detect_model_type

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def _model_dir(root, name, config):
    path = root / name
    path.mkdir(parents=True)
    (path / "config.json").write_text(json.dumps(config))
    return path


class StsDiscoveryWithoutModelType(unittest.TestCase):
    def test_deepfilternet3_config_shape_is_audio_sts(self):
        config = json.loads((FIXTURES / "deepfilternet3-config.json").read_text())

        with tempfile.TemporaryDirectory() as tmp:
            path = _model_dir(Path(tmp), "DeepFilterNet-mlx", config)
            self.assertEqual(detect_model_type(path), "audio_sts")

    def test_sam_audio_config_shape_is_audio_sts(self):
        config = json.loads((FIXTURES / "sam-audio-config.json").read_text())

        with tempfile.TemporaryDirectory() as tmp:
            path = _model_dir(Path(tmp), "sam-audio-small", config)
            self.assertEqual(detect_model_type(path), "audio_sts")

    def test_directory_name_hints_classify_audio_sts(self):
        with tempfile.TemporaryDirectory() as tmp:
            deepfilter = _model_dir(Path(tmp), "DeepFilterNet-mlx", {"sample_rate": 48000})
            sam = _model_dir(Path(tmp), "sam-audio-base", {"in_channels": 768})
            self.assertEqual(detect_model_type(deepfilter), "audio_sts")
            self.assertEqual(detect_model_type(sam), "audio_sts")

    def test_plain_llm_config_is_not_audio_sts(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _model_dir(
                Path(tmp),
                "Qwen3-4B",
                {"model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"]},
            )
            self.assertEqual(detect_model_type(path), "llm")

    def test_kokoro_config_stays_audio_tts(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _model_dir(
                Path(tmp),
                "Kokoro-82M",
                {"istftnet": {}, "plbert": {}, "vocab": {}},
            )
            self.assertEqual(detect_model_type(path), "audio_tts")


if __name__ == "__main__":
    unittest.main()
