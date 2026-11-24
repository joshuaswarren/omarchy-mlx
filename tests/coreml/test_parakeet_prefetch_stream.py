# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Longform prefetch hands the decoder evaluated encoder output (#794).

With MLX_OMARCHY_LONGFORM_PREFETCH=1, _encode_chunk runs on an executor
thread, and MLX GPU streams are thread-local. A lazy array built on that
thread cannot be evaluated on the main thread. The trailing partial chunk
of a recording that is not a 30 s multiple crossed as an unevaluated slice
and failed with "There is no Stream(gpu, N) in current thread."

The thread-streams backend models that contract (a lazy array evaluates
only on the thread that built it), so the test runs without mlx. The
mlx-gpu backend runs the same path on mx.gpu where one is available.
"""

import contextlib
import math
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

TOOLS = Path(__file__).resolve().parents[2] / "overlay" / "tools"
sys.path.insert(0, str(TOOLS))

from coreml.parakeet_dictation import DictationTranscriber

CHUNK = 480_000  # 30 s at 16 kHz
WINDOW = 8  # encoder frames per full chunk
WIDTH = 4  # encoder hidden width


class _LazyArray:
    def __init__(self, data):
        self._data = data
        self._pending_on = threading.get_ident()

    @property
    def shape(self):
        return self._data.shape

    def _evaluate(self):
        if self._pending_on not in (None, threading.get_ident()):
            raise RuntimeError("There is no Stream(gpu, 1) in current thread.")
        self._pending_on = None

    def astype(self, dtype):
        return _LazyArray(self._data.astype(dtype))

    def __getitem__(self, key):
        return _LazyArray(self._data[key])

    def __array__(self, dtype=None, copy=None):
        self._evaluate()
        return self._data if dtype is None else self._data.astype(dtype)


class _ThreadStreams:
    """The slice of mlx.core the transcriber uses, with per-thread streams."""

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
    def eval(*arrays):
        for array in arrays:
            array._evaluate()


def _mlx_gpu():
    mx = pytest.importorskip("mlx.core")
    if not mx.is_available(mx.gpu):
        pytest.skip("mlx has no GPU device on this machine")
    return mx


@pytest.fixture(params=["thread-streams", "mlx-gpu"])
def mx(request):
    return _ThreadStreams() if request.param == "thread-streams" else _mlx_gpu()


def _transcriber(mx, encode_threads, decoded_frames):
    def extract(waveform, duration_aware):
        samples = mx.array(np.full((1, 1), waveform.shape[0], np.float32))
        return SimpleNamespace(mel=samples, mask=samples,
                               encoder_features=samples, encoder_mask=samples)

    def run(inputs, wanted, stop_after):
        encode_threads.add(threading.get_ident())
        samples = int(np.asarray(inputs["input_features"])[0, 0])
        valid = math.ceil(samples * WINDOW / CHUNK)
        hidden = samples + np.arange(WINDOW * WIDTH, dtype=np.float32)
        mask = (np.arange(WINDOW) < valid).astype(np.int32)
        return {"encoder_hidden": mx.array(hidden.reshape(1, WINDOW, WIDTH)),
                "encoder_mask": mx.array(mask.reshape(1, WINDOW))}

    def tdt_decode(encoder, valid_frames, initial_hidden, initial_cell,
                   **_callbacks):
        mx.eval(encoder)  # the decoder's first step evaluates the encoder
        assert tuple(encoder.shape) == (1, valid_frames, WIDTH)
        decoded_frames.append(valid_frames)
        checksum = int(np.asarray(encoder).astype(np.float64).sum())
        return SimpleNamespace(token_ids=[valid_frames, checksum],
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
        decode=lambda tokens: " ".join(map(str, tokens)))
    return transcriber


def test_prefetch_decodes_a_trailing_partial_chunk(mx, monkeypatch):
    waveform = np.zeros(2 * CHUNK + 24_000, dtype=np.float32)  # 61.5 s

    monkeypatch.delenv("MLX_OMARCHY_LONGFORM_PREFETCH", raising=False)
    serial_frames = []
    serial = _transcriber(mx, set(), serial_frames).transcribe_waveform(
        waveform, 16_000)

    monkeypatch.setenv("MLX_OMARCHY_LONGFORM_PREFETCH", "1")
    encode_threads, prefetch_frames = set(), []
    prefetch = _transcriber(mx, encode_threads, prefetch_frames)
    transcript = prefetch.transcribe_waveform(waveform, 16_000)

    assert threading.get_ident() not in encode_threads
    assert serial_frames == prefetch_frames == [WINDOW, WINDOW, 1]
    assert transcript == serial
