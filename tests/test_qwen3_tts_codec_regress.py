"""Qwen3-TTS codec decoder regression: pinned codes must reproduce the
reference (Metal) waveform on the Omarchy Vulkan backend.

The 2026-09-28 TTS hum defect scrambled the codec composition — the RVQ
embedding sum, the k=1 projections, and the final two-operand add — while
every isolated op passed its own battery, so the assistant spoke noise.
The fixtures here are the reference decoder output for fixed codes on the
same pinned pack (shapes and provenance in meta.json). Any backend
regression that corrupts this composition fails here with a numeric gap
instead of audible noise.

Honest-skip contract (same as test_assistant_synthesis.py): real decoder
execution needs the project mlx wheel, mlx-audio, an available
accelerator, and a local voice pack; without one of those the test skips
and names the missing prerequisite. Execution is never faked.
"""

import json
import os
import sys
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURES = Path(__file__).resolve().parent / "fixtures" / "qwen3_tts_codec"
PACK_ENV = "MLX_OMARCHY_TTS_TEST_PACK"

REL_RMS_TOLERANCE = 1e-4
MAX_ABS_TOLERANCE = 1e-3


def _load_fixtures():
    codes = np.load(FIXTURES / "codes.npy")
    expected_quantized = np.load(FIXTURES / "expected_quantized.npy")
    expected_audio = np.load(FIXTURES / "expected_audio.npy")
    meta = json.loads((FIXTURES / "decoder_meta.json").read_text())
    return codes, expected_quantized, expected_audio, meta


def _gap(got, want):
    got64 = got.astype(np.float64)
    want64 = want.astype(np.float64)
    if got64.shape != want64.shape:
        return None
    diff = np.abs(got64 - want64)
    root = float(np.sqrt((want64 ** 2).mean())) or 1e-30
    return {
        "rel_rms": float(np.sqrt((diff ** 2).mean()) / root),
        "max_abs": float(diff.max()),
    }


@unittest.skipUnless(os.environ.get(PACK_ENV),
                     f"set {PACK_ENV} to the local voice pack directory "
                     "to run the decoder regression on hardware")
class Qwen3TtsCodecRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pack = Path(os.environ[PACK_ENV])
        try:
            import mlx.core as mx  # noqa: F401
        except ImportError as exc:
            raise unittest.SkipTest(f"mlx wheel unavailable: {exc}")
        try:
            from mlx_audio.tts.utils import load_model
        except ImportError as exc:
            raise unittest.SkipTest(f"mlx-audio unavailable: {exc}")
        sys.path.insert(0, str(REPO_ROOT / "serve"))
        from mlx_omarchy_assistant import synthesis
        probe = synthesis.probe_accelerator()
        if not probe.get("available"):
            raise unittest.SkipTest(
                f"accelerator unavailable: {probe.get('detail')}")
        cls.model = load_model(cls.pack)
        cls.raw = cls._load_tokenizer_raw(cls.pack)

    @staticmethod
    def _load_tokenizer_raw(pack):
        """The speech tokenizer without mx.compile, for per-stage checks.

        Mirrors the post-load path in mlx-audio's qwen3_tts post_load_hook
        (config -> sanitize -> load_weights -> eval) minus the final
        `mx.compile(tokenizer.decoder)` wrap, so the quantizer composition
        is observable directly. The compiled production path is asserted
        separately through model.speech_tokenizer.decode.
        """
        import mlx.core as mx
        from mlx_audio.tts.models.qwen3_tts.config import (
            filter_dict_for_dataclass)
        from mlx_audio.tts.models.qwen3_tts.speech_tokenizer import (
            Qwen3TTSSpeechTokenizer,
            Qwen3TTSTokenizerDecoderConfig,
            Qwen3TTSTokenizerEncoderConfig,
            Qwen3TTSTokenizerConfig,
        )
        st_path = pack / "speech_tokenizer"
        cfg = json.loads((st_path / "config.json").read_text())
        dec_cfg = Qwen3TTSTokenizerDecoderConfig(**filter_dict_for_dataclass(
            Qwen3TTSTokenizerDecoderConfig, cfg["decoder_config"]))
        enc_cfg = Qwen3TTSTokenizerEncoderConfig(**filter_dict_for_dataclass(
            Qwen3TTSTokenizerEncoderConfig, cfg["encoder_config"]))
        tok_cfg = Qwen3TTSTokenizerConfig(
            encoder_config=enc_cfg, decoder_config=dec_cfg)
        for key, value in cfg.items():
            if key not in ("decoder_config", "encoder_config") \
                    and hasattr(tok_cfg, key):
                setattr(tok_cfg, key, value)
        tokenizer = Qwen3TTSSpeechTokenizer(tok_cfg)
        weights = {}
        for weights_file in st_path.glob("*.safetensors"):
            weights.update(mx.load(str(weights_file)))
        weights = Qwen3TTSSpeechTokenizer.sanitize(weights)
        tokenizer.load_weights(list(weights.items()), strict=False)
        mx.eval(tokenizer.parameters())
        tokenizer.eval()
        return tokenizer

    def test_decoder_reproduces_reference_from_pinned_codes(self):
        import mlx.core as mx

        codes_np, expected_quantized, expected_audio, meta = _load_fixtures()
        self.assertEqual(codes_np.ndim, 3, "fixture codes must be [1, T, G]")
        quantizer = self.raw.decoder["quantizer"]

        # Stage 1: quantizer composition (gather sum + k=1 projections).
        codes_nqt = mx.transpose(mx.array(codes_np), (0, 2, 1))
        quantized = quantizer.decode(codes_nqt)
        got_quantized = np.asarray(quantized.astype(mx.float32))
        gap = _gap(got_quantized, expected_quantized)
        self.assertIsNotNone(
            gap,
            f"quantized shape {got_quantized.shape} != fixture "
            f"{expected_quantized.shape}")
        self.assertLess(
            gap["rel_rms"], REL_RMS_TOLERANCE,
            f"quantizer composition diverged from the reference: "
            f"{json.dumps(gap)}")

        # Stage 2: the compiled production decode path end to end.
        audio, lengths = self.model.speech_tokenizer.decode(
            mx.array(codes_np))
        wav = audio[0]
        valid = int(lengths[0])
        if 0 < valid < wav.shape[0]:
            wav = wav[:valid]
        got_audio = np.asarray(wav.astype(mx.float32)).reshape(1, 1, -1)
        gap = _gap(got_audio, expected_audio)
        self.assertIsNotNone(
            gap,
            f"audio shape {got_audio.shape} != fixture "
            f"{expected_audio.shape}")
        self.assertLess(
            gap["rel_rms"], REL_RMS_TOLERANCE,
            f"decoder waveform diverged from the reference: "
            f"{json.dumps(gap)}")
        self.assertLessEqual(gap["max_abs"], MAX_ABS_TOLERANCE)

    def test_fixture_shapes_are_recorded(self):
        codes_np, expected_quantized, expected_audio, meta = _load_fixtures()
        self.assertEqual(meta["codes_shape"], list(codes_np.shape))
        upsample = int(meta["total_upsample"])
        frames = int((codes_np[0, :, 0] > 0).sum())
        self.assertEqual(
            list(expected_audio.shape), [1, 1, frames * upsample],
            "fixture audio length must equal valid frames * upsample")
        self.assertEqual(
            list(expected_quantized.shape),
            [1, meta.get("codebook_dim", expected_quantized.shape[1]),
             codes_np.shape[1]],
            "fixture quantized tensor must be [1, codebook_dim, T]")


if __name__ == "__main__":
    unittest.main()
