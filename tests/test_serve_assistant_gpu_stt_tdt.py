# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Fast TDT greedy decode: exact-contract equality against the reference loop.

The reference greedy TDT loop syncs the GPU once per frame. The fast decoder
evaluates the joint network over the whole remaining tail per emitted token
and walks the argmax results on the host, so the token and timestamp output
must be identical. Tests build a real tiny ParakeetTDT with fixed-seed random
weights and compare both decoders exactly.
"""

import importlib.util
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "serve"))

HAS_MLX = importlib.util.find_spec("mlx") is not None
HAS_MLX_AUDIO = importlib.util.find_spec("mlx_audio") is not None


def _load_tdt_module():
    from mlx_omarchy_assistant import gpu_stt_tdt

    return gpu_stt_tdt


@unittest.skipUnless(HAS_MLX and HAS_MLX_AUDIO, "mlx and mlx_audio required")
class FastTdtDecodeEqualityTest(unittest.TestCase):
    """fast_tdt_decode(model, mel) must equal ParakeetTDT.decode(mel)[0]."""

    @classmethod
    def setUpClass(cls):
        cls.tdt = _load_tdt_module()
        from mlx_audio.stt.models.parakeet.parakeet import (
            ParakeetTDT,
            ParakeetTDTArgs,
            TDTDecodingArgs,
        )
        from mlx_audio.stt.models.parakeet.conformer import ConformerArgs
        from mlx_audio.stt.models.parakeet.rnnt import (
            JointArgs,
            JointNetworkArgs,
            PredictArgs,
            PredictNetworkArgs,
        )
        from mlx_audio.stt.models.parakeet.audio import PreprocessArgs

        cls.ParakeetTDT = ParakeetTDT
        cls.tdt_args = ParakeetTDTArgs(
            preprocessor=PreprocessArgs(
                sample_rate=16_000,
                normalize="per_feature",
                window_size=0.025,
                window_stride=0.01,
                window="hann",
                features=80,
                n_fft=400,
                dither=0.0,
            ),
            encoder=ConformerArgs(
                feat_in=80,
                n_layers=2,
                d_model=32,
                n_heads=4,
                ff_expansion_factor=4,
                subsampling_factor=8,
                self_attention_model="rel_pos",
                subsampling="dw_striding",
                conv_kernel_size=9,
                subsampling_conv_channels=16,
                pos_emb_max_len=5000,
            ),
            decoder=PredictArgs(
                blank_as_pad=True,
                vocab_size=16,
                prednet=PredictNetworkArgs(
                    pred_hidden=32, pred_rnn_layers=2, rnn_hidden_size=None
                ),
            ),
            joint=JointArgs(
                num_classes=16,
                vocabulary=[chr(ord("a") + i) for i in range(16)],
                jointnet=JointNetworkArgs(
                    joint_hidden=32,
                    activation="relu",
                    encoder_hidden=32,
                    pred_hidden=32,
                ),
                num_extra_outputs=4,
            ),
            decoding=TDTDecodingArgs(
                model_type="tdt", durations=[0, 1, 2, 4], greedy=None
            ),
        )

    def _model(self, max_symbols=None, favor_duration_index=1):
        import mlx.core as mx
        import mlx.nn as nn

        model = self.ParakeetTDT(self.tdt_args)
        mx.random.seed(7)
        # Deterministic in-place init: every weight gets a fixed value.
        flat = nn.utils.tree_flatten(model.parameters())
        new_flat = []
        for path, tensor in flat:
            mx.random.seed(sum(ord(c) for c in path) + tensor.size)
            if "running_var" in path:
                # BatchNorm variance must stay positive or eval mode NaNs.
                new_flat.append((path, mx.abs(mx.random.uniform(
                    low=-0.08, high=0.08, shape=tensor.shape,
                    dtype=tensor.dtype)) + 0.5))
            else:
                new_flat.append((path, mx.random.uniform(
                    low=-0.08, high=0.08, shape=tensor.shape,
                    dtype=tensor.dtype)))
        model.update(nn.utils.tree_unflatten(new_flat))
        # The joint's final linear drives both heads. Build its bias explicitly
        # (mlx slice assignment does not write through): tokens beat blank by a
        # small margin, and one duration index dominates so the walk always
        # advances - or, with index 0, the max_symbols cap advances it.
        final_linear = model.joint.joint_net[2]
        n_classes = final_linear.bias.shape[0]
        n_durations = n_classes - model.blank_id - 1
        values = [0.05] * model.blank_id + [-1.0]
        values += [-1.0 if i != favor_duration_index else 50.0
                   for i in range(n_durations)]
        final_linear.bias = mx.array(values, dtype=final_linear.bias.dtype)
        model.max_symbols = max_symbols
        model.eval()
        return model

    def _mel(self, seconds, seed):
        import mlx.core as mx

        mx.random.seed(seed)
        return mx.random.uniform(
            low=-1.0, high=1.0, shape=(1, seconds * 100, 80), dtype=mx.float32
        )

    def _reference_tokens(self, model, mel):
        """Token sequence from the reference per-frame loop, verbatim."""
        import mlx.core as mx

        batch_features, lengths = model.encoder(mel)
        mx.eval(batch_features, lengths)
        features = batch_features[0:1]
        max_length = int(lengths[0])
        vocabulary = model.vocabulary
        blank_id = model.blank_id
        durations = model.durations
        from mlx_audio.stt.models.parakeet import tokenizer

        hypothesis = []
        last_token = blank_id
        decoder_hidden = model._make_initial_decoder_state(1, features.dtype)
        time = 0
        new_symbols = 0
        while time < max_length:
            feature = features[:, time : time + 1]
            current_token = mx.array([[last_token]], dtype=mx.int32)
            pred_token, decision, hidden, cell = model._compiled_tdt_step(
                feature, current_token, decoder_hidden[0], decoder_hidden[1]
            )
            mx.eval(pred_token, decision, hidden, cell)
            pred_token_i = int(pred_token)
            decision_i = int(decision)
            duration_i = durations[decision_i]
            if pred_token_i != blank_id:
                last_token = pred_token_i
                decoder_hidden = (hidden, cell)
                if not tokenizer.is_special_token(last_token, vocabulary):
                    hypothesis.append(pred_token_i)
            time += duration_i
            new_symbols += 1
            if duration_i != 0:
                new_symbols = 0
            elif model.max_symbols is not None and model.max_symbols <= new_symbols:
                time += 1
                new_symbols = 0
        return hypothesis

    def _check_equal(self, model, mel):
        reference = model.decode(mel)[0]
        fast = self.tdt.fast_tdt_decode(model, mel)
        ref_tokens = self._reference_tokens(model, mel)
        fast_tokens = [t.id for s in fast.sentences for t in s.tokens]
        self.assertEqual(fast_tokens, ref_tokens)
        self.assertEqual(fast.text, reference.text)

    def test_equals_reference_loop(self):
        for seed in (11, 12, 13):
            model = self._model()
            mel = self._mel(seconds=2, seed=seed)
            self._check_equal(model, mel)

    def test_equals_reference_loop_with_max_symbols(self):
        model = self._model(max_symbols=3, favor_duration_index=0)
        mel = self._mel(seconds=2, seed=14)
        self._check_equal(model, mel)

    def test_supports_rejects_non_tdt_models(self):
        class NotTDT:
            pass

        self.assertFalse(self.tdt.supports(NotTDT()))


if __name__ == "__main__":
    unittest.main()
