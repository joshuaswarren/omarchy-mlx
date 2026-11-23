# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Batched greedy TDT decoding for the GPU speech-recognition worker.

mlx-audio's ``ParakeetTDT.decode`` runs one GPU round-trip per frame: a
compiled LSTM-plus-joint step followed by ``mx.eval`` and two ``int()`` host
syncs, for every frame the decoder visits. On the pinned mlx-audio 0.5.6 and
current 0.6.3 the loop is identical. That loop is host-bound: hundreds of
small dependent calls each paying a full device round-trip.

The decoder state (token embedding and LSTM) only changes when a token is
emitted, so the joint network over a whole tail of frames can be evaluated
in one batched call against the fixed decoder state, argmaxed once, and
walked on the host. Per emitted token that is one decoder step, one batched
joint, and one sync, instead of one sync per frame.

The contract is exact: the same argmax over the same logits in the same
order, the same duration walk including the ``max_symbols`` cap, and the
same AlignedToken construction. ``tests/test_serve_assistant_gpu_stt_tdt.py``
asserts token-for-token equality against the reference loop.
"""

from __future__ import annotations

import numpy as np


def supports(model: object) -> bool:
    """True only for ParakeetTDT-shaped models with a TDT decoding config."""
    return (
        hasattr(model, "encoder")
        and hasattr(model, "joint")
        and hasattr(model, "durations")
        and hasattr(model, "blank_id")
        and hasattr(model, "vocabulary")
        and hasattr(model, "encoder_config")
        and hasattr(model, "preprocessor_config")
        and hasattr(model, "_make_initial_decoder_state")
        and hasattr(getattr(model, "decoder", None), "prediction")
        and getattr(model, "durations", None) is not None
    )


def fast_tdt_decode(model, mel) -> object:
    """Greedy TDT decode of one mel, exact-equivalent to ParakeetTDT.decode.

    ``mel`` is ``[T, mel_dim]`` or ``[1, T, mel_dim]``; the result is the
    single ``AlignedResult`` for batch entry 0.
    """
    import mlx.core as mx

    from mlx_audio.stt.models.nemo.alignment import (
        AlignedToken,
        sentences_to_result,
        tokens_to_sentences,
    )
    from mlx_audio.stt.models.parakeet import tokenizer

    batch_size = mel.shape[0]
    if len(mel.shape) == 2:
        batch_size = 1
        mel = mx.expand_dims(mel, 0)
    if batch_size != 1:
        raise ValueError("fast_tdt_decode decodes one batch entry; got "
                         f"batch_size={batch_size}")

    batch_features, lengths = model.encoder(mel)
    mx.eval(batch_features, lengths)
    features = batch_features[0:1]
    max_length = int(lengths[0])

    vocabulary = model.vocabulary
    blank_id = model.blank_id
    durations = model.durations
    max_symbols = model.max_symbols
    subsampling = model.encoder_config.subsampling_factor
    sample_rate = model.preprocessor_config.sample_rate
    hop_length = model.preprocessor_config.hop_length

    hypothesis: list[AlignedToken] = []
    last_token = blank_id
    dec_state = model._make_initial_decoder_state(1, features.dtype)
    time = 0
    new_symbols = 0

    while time < max_length:
        # One decoder step per segment: the step's output depends only on
        # the current token and state, never on the frame.
        current_token = mx.array([[last_token]], dtype=mx.int32)
        embedded = model.decoder.prediction["embed"](current_token)
        blank_mask = mx.expand_dims(current_token == blank_id, axis=-1)
        embedded = mx.where(blank_mask, mx.zeros_like(embedded), embedded)
        decoder_output, (hidden_out, cell_out) = model.decoder.prediction[
            "dec_rnn"](embedded, dec_state)
        decoder_output = decoder_output.astype(features.dtype)
        hidden_out = hidden_out.astype(features.dtype)
        cell_out = cell_out.astype(features.dtype)
        mx.eval(decoder_output, hidden_out, cell_out)

        segment_start = time
        logits = model.joint(features[:, segment_start:max_length, :], decoder_output)
        pred_ids = mx.argmax(logits[0, :, 0, : blank_id + 1], axis=-1)
        decisions = mx.argmax(logits[0, :, 0, blank_id + 1 :], axis=-1)
        mx.eval(pred_ids, decisions)
        pred_host = np.asarray(pred_ids).tolist()
        decision_host = np.asarray(decisions).tolist()

        adopted = False
        # The reference loop consults exactly the frames the duration walk
        # lands on: after a blank frame with duration d it re-enters at
        # time + d, never at the skipped frames in between. Index the
        # precomputed tail by absolute frame, not by walk order, so the
        # visited frames and timestamps match the reference exactly.
        while time < max_length:
            local = time - segment_start
            duration_i = durations[decision_host[local]]
            if pred_host[local] != blank_id:
                last_token = pred_host[local]
                dec_state = (hidden_out, cell_out)
                if not tokenizer.is_special_token(last_token, vocabulary):
                    hypothesis.append(AlignedToken(
                        last_token,
                        start=time * subsampling / sample_rate * hop_length,
                        duration=duration_i * subsampling / sample_rate
                        * hop_length,
                        text=tokenizer.decode([last_token], vocabulary),
                    ))
                adopted = True
            time += duration_i
            new_symbols += 1
            if duration_i != 0:
                new_symbols = 0
            elif max_symbols is not None and max_symbols <= new_symbols:
                time += 1
                new_symbols = 0
            if adopted:
                break

    return sentences_to_result(tokens_to_sentences(hypothesis))
