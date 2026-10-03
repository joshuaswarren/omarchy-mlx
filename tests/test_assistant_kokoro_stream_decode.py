"""Streamed decoding reproduces the whole decoder (needs mlx + mlx_audio 0.5.6).

Builds mlx_audio's real Kokoro Decoder (Kokoro's istftnet config, random
weights) and checks the window arithmetic the product relies on: with the
last stage's statistics taken from the whole call, the windowed schedule
returns the whole call's samples, window seams included; wrapping the norms
changes nothing while no stream runs; a one-window utterance takes the
unchanged decoder. Skipped where mlx or mlx_audio is not installed.
"""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "serve"))

try:
    import mlx.core as mx
    import mlx.nn as nn
    import numpy as np
    from mlx.utils import tree_map
    from mlx_audio.tts.models.kokoro.istftnet import Decoder
except Exception:  # pragma: no cover - runtime not installed on this host
    Decoder = None

from mlx_omarchy_assistant import kokoro_stream  # noqa: E402

KOKORO_ISTFTNET = dict(
    resblock_kernel_sizes=[3, 7, 11], upsample_rates=[10, 6],
    upsample_initial_channel=512, resblock_dilation_sizes=[[1, 3, 5]] * 3,
    upsample_kernel_sizes=[20, 12], gen_istft_n_fft=20, gen_istft_hop_size=5)


@unittest.skipIf(Decoder is None, "mlx / mlx_audio not installed")
class StreamedDecodeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        mx.random.seed(7)
        cls.dec = Decoder(dim_in=512, style_dim=128, dim_out=80, **KOKORO_ISTFTNET)
        # Random init overflows the exp() in the inverse STFT; scaled weights
        # keep every sample finite.
        cls.dec.update(tree_map(lambda p: p * 0.03, cls.dec.parameters()))
        frames = kokoro_stream.WINDOW_FRAMES + 5
        cls.inputs = (mx.random.normal((1, 512, frames)) * 0.1,
                      120.0 + 30.0 * mx.random.uniform(shape=(1, 2 * frames)),
                      mx.random.normal((1, 2 * frames)) * 0.1,
                      mx.random.normal((1, 128)) * 0.1)

    def _whole(self, inputs, seed=11):
        mx.random.seed(seed)
        audio = self.dec(*inputs)[0]
        mx.eval(audio)
        return np.asarray(audio).reshape(-1)

    def _oracle_stats(self, inputs):
        layers = kokoro_stream.last_stage_adains(self.dec)
        seen = {}

        class Recorder(nn.Module):
            def __init__(self, name, inner):
                super().__init__()
                self._name, self._inner = name, inner

            def __call__(self, x):
                seen[self._name] = (np.asarray(mx.mean(x, axis=2).reshape(-1)),
                                    np.asarray(mx.var(x, axis=2).reshape(-1)))
                return self._inner(x)

        originals = {n: a.norm for n, a in layers.items()}
        for n, a in layers.items():
            a.norm = Recorder(n, originals[n])
        try:
            reference = self._whole(inputs)
        finally:
            for n, a in layers.items():
                a.norm = originals[n]
        return reference, seen

    def test_windows_reproduce_the_whole_decoder(self):
        reference, oracle = self._oracle_stats(self.inputs)
        layers = kokoro_stream.last_stage_adains(self.dec)
        originals = {n: a.norm for n, a in layers.items()}
        streamer = kokoro_stream.KokoroStreamer(
            SimpleNamespace(model=SimpleNamespace(decoder=self.dec)), {"voice": oracle})
        try:
            self.assertTrue(np.array_equal(self._whole(self.inputs), reference),
                            "wrapped norms change nothing while no stream runs")
            streamer._table["current"] = streamer._stats["voice"]
            mx.random.seed(11)
            pieces = list(streamer._decode(*self.inputs))
            streamer._table["current"] = None
            got = np.concatenate(pieces)
            self.assertTrue(np.isfinite(reference).all())
            self.assertEqual(len(pieces), 2, "one seam inside the utterance")
            self.assertEqual(got.shape, reference.shape)
            scale = float(np.max(np.abs(reference)))
            self.assertLess(float(np.max(np.abs(got - reference))) / scale, 1e-5)
            short = tuple(x[:, :, :10] if x.ndim == 3 else x[:, :20] if i in (1, 2) else x
                          for i, x in enumerate(self.inputs))
            mx.random.seed(11)
            one = streamer._whole(*short)
            self.assertTrue(np.array_equal(one, self._whole(short)),
                            "a one-window utterance takes the unchanged decoder")
        finally:
            for n, a in layers.items():
                a.norm = originals[n]


@unittest.skipIf(Decoder is None, "mlx / mlx_audio not installed")
class SeamJoinTests(unittest.TestCase):
    """_stream_ranges composes context-overlapped utterance renders into
    exactly the whole-call audio when the renderer itself is exact, and
    never emits more or fewer samples than the shared timeline."""

    @classmethod
    def setUpClass(cls):
        mx.random.seed(7)
        cls.dec = Decoder(dim_in=512, style_dim=128, dim_out=80,
                          **KOKORO_ISTFTNET)
        cls.dec.update(tree_map(lambda p: p * 0.03, cls.dec.parameters()))

    def _streamer(self):
        return kokoro_stream.KokoroStreamer(
            SimpleNamespace(model=SimpleNamespace(decoder=self.dec)), None)

    def test_exact_renders_join_bit_exact(self):
        import numpy as np
        streamer = self._streamer()
        frames = 40
        rng = np.random.default_rng(5)
        whole = rng.standard_normal(frames * kokoro_stream.FRAME_SAMPLES
                                    ).astype(np.float32) * 0.1
        ranges = [(0, 15), (15, 27), (27, frames)]
        expected_slices = [(0, 23), (7, 35), (19, frames)]
        calls = iter(expected_slices)

        def exact_render(asr, F0, N, s, voice):
            a, b = next(calls)
            return whole[a * kokoro_stream.FRAME_SAMPLES:
                         b * kokoro_stream.FRAME_SAMPLES]

        streamer._render = exact_render
        got = np.concatenate(list(streamer._stream_ranges(
            None, np.zeros((1, frames), np.float32),
            np.zeros((1, frames), np.float32), None, "af_heart", ranges)))
        self.assertEqual(got.shape, whole.shape)
        self.assertTrue(np.array_equal(got, whole),
                        "exact context renders reproduce the whole call")

    def test_seam_step_stays_inside_the_render_error(self):
        import numpy as np
        streamer = self._streamer()
        frames = 40
        rng = np.random.default_rng(6)
        whole = rng.standard_normal(frames * kokoro_stream.FRAME_SAMPLES
                                    ).astype(np.float32) * 0.1
        ranges = [(0, 15), (15, frames)]
        err = 1e-3
        calls = iter([(0, 23), (7, frames)])

        def noisy_render(asr, F0, N, s, voice):
            a, b = next(calls)
            piece = whole[a * kokoro_stream.FRAME_SAMPLES:
                          b * kokoro_stream.FRAME_SAMPLES]
            return piece + rng.standard_normal(piece.shape).astype(np.float32) * err

        streamer._render = noisy_render
        got = np.concatenate(list(streamer._stream_ranges(
            None, np.zeros((1, frames), np.float32),
            np.zeros((1, frames), np.float32), None, "af_heart", ranges)))
        seam = 15 * kokoro_stream.FRAME_SAMPLES
        band = np.abs(np.diff(got[seam - 300:seam + 300]))
        self.assertLess(float(band.max()), 10 * err,
                        "the crossfade keeps the seam step within render noise")


if __name__ == "__main__":
    unittest.main()
