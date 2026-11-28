# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Fragmentary short-clip decodes get one silence-tailed re-encode.

The whole-encoder .anec runs fp16; at knife-edge attention margins a
recording cut mid-word resolves differently than the fp32 reference and
emits an empty string or a filler fragment instead of the spoken word.
Appending the tail silence the cut removed restores the natural ending,
so a speech-bearing clip whose decode is a fragment retries once on the
padded waveform.

The stubs below stand in for the accelerator stages the same way
test_parakeet_prefetch_stream.py does, so the test runs without mlx.
"""

import contextlib
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

TOOLS = Path(__file__).resolve().parents[2] / "overlay" / "tools"
sys.path.insert(0, str(TOOLS))

import coreml.parakeet_dictation as pd
from coreml.parakeet_dictation import DictationTranscriber

CHUNK = pd._CHUNK_SAMPLES
WINDOW = 8
WIDTH = 4

WORDS = {1: "Why.", 2: "One.", 3: "", 4: "Nine."}


class _LazyArray:
    def __init__(self, data):
        self._data = data

    @property
    def shape(self):
        return self._data.shape

    def astype(self, dtype):
        return _LazyArray(self._data.astype(dtype))

    def __getitem__(self, key):
        return _LazyArray(self._data[key])

    def __array__(self, dtype=None, copy=None):
        return self._data if dtype is None else self._data.astype(dtype)


class _ThreadStreams:
    float32 = np.float32
    int32 = np.int32
    gpu = "gpu"

    @staticmethod
    def stream(_device):
        return contextlib.nullcontext()

    @staticmethod
    def array(data):
        return _LazyArray(np.asarray(data))

    @staticmethod
    def zeros(shape, dtype):
        return _LazyArray(np.zeros(shape, dtype))

    @staticmethod
    def eval(*_arrays):
        return None


def _transcriber(encode_lengths, decode_scripts):
    """A DictationTranscriber whose stages are recorded and scripted.

    ``encode_lengths`` receives each chunk's sample count as the feature
    stage sees it.  ``decode_scripts`` is consumed one entry per
    ``_tdt_decode`` call; each entry is a list of token ids that the fake
    tokenizer maps to words through ``WORDS``.
    """
    mx = _ThreadStreams
    calls = {"decode": 0}

    def extract(waveform, duration_aware):
        encode_lengths.append(int(waveform.shape[0]))
        marker = mx.array(np.full((1, 1), waveform.shape[0], np.float32))
        return SimpleNamespace(mel=marker, mask=marker,
                               encoder_features=marker, encoder_mask=marker)

    def run(inputs, wanted, stop_after):
        hidden = np.zeros(WINDOW * WIDTH, dtype=np.float32)
        mask = np.ones(WINDOW, dtype=np.int32)
        return {"encoder_hidden": mx.array(hidden.reshape(1, WINDOW, WIDTH)),
                "encoder_mask": mx.array(mask.reshape(1, WINDOW))}

    def tdt_decode(encoder, valid_frames, initial_hidden, initial_cell,
                   **_callbacks):
        tokens = decode_scripts[min(calls["decode"], len(decode_scripts) - 1)]
        calls["decode"] += 1
        return SimpleNamespace(token_ids=list(tokens),
                               hidden=initial_hidden, cell=initial_cell)

    transcriber = DictationTranscriber()
    transcriber._started = True
    transcriber._mx = mx
    transcriber._np = np
    transcriber._extract_chunk_features = extract
    transcriber._runner = SimpleNamespace(run=run)
    transcriber._tdt_decode = tdt_decode
    transcriber._reference = SimpleNamespace(tdt=None)
    transcriber._tokenizer = SimpleNamespace(
        decode=lambda tokens: " ".join(WORDS[t] for t in tokens))
    return transcriber


def _speech(size):
    rng = np.random.RandomState(0)
    return (rng.rand(size).astype(np.float32) - 0.5) * 0.4


def test_fragmentary_decode_retries_with_a_silence_tail():
    encode_lengths = []
    transcriber = _transcriber(encode_lengths, [[1], [2]])
    waveform = _speech(8_000)

    text = transcriber.transcribe_waveform(waveform, 16_000)

    assert text == "One."
    assert encode_lengths == [8_000, 8_000 + 16_000]


def test_a_clean_decode_is_not_retried():
    encode_lengths = []
    transcriber = _transcriber(encode_lengths, [[4]])
    waveform = _speech(8_000)

    text = transcriber.transcribe_waveform(waveform, 16_000)

    assert text == "Nine."
    assert encode_lengths == [8_000]


def test_a_fragment_on_silence_is_not_retried():
    encode_lengths = []
    transcriber = _transcriber(encode_lengths, [[3], [2]])
    waveform = np.zeros(8_000, dtype=np.float32)

    text = transcriber.transcribe_waveform(waveform, 16_000)

    assert text == ""
    assert encode_lengths == [8_000]


def test_an_empty_retry_keeps_the_fragment():
    encode_lengths = []
    transcriber = _transcriber(encode_lengths, [[1], [3]])
    waveform = _speech(8_000)

    text = transcriber.transcribe_waveform(waveform, 16_000)

    assert text == "Why."
    assert len(encode_lengths) == 2


def test_a_multi_chunk_fragment_is_not_retried():
    encode_lengths = []
    transcriber = _transcriber(encode_lengths, [[1], [1]])
    waveform = _speech(CHUNK + 1_600)

    transcriber.transcribe_waveform(waveform, 16_000)

    assert encode_lengths == [CHUNK, 1_600]


@pytest.mark.parametrize(("text", "expected"), [
    ("", True),
    ("Si.", True),
    ("Oh.", True),
    ("Mm.", True),
    ("Why.", True),
    ("Yeah..", True),
    ("I've been", True),
    ("por", True),
    ("One.", False),
    ("Two.", False),
    ("Six", False),
    ("Nine.", False),
    ("Yes.", False),
])
def test_fragmentary_classification(text, expected):
    assert pd._fragmentary(text) is expected
