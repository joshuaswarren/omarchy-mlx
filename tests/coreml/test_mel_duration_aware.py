# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Duration-aware mel contract: statistics over the audio's frames, padding zeroed, mask on the audio only.

The pinned fixture keeps the macOS reference contract (3001-frame statistics, all-valid mask). Short clips
normalized that way put their speech at the wrong scale, which made the Parakeet CLI hallucinate over the padding
or return nothing on LibriSpeech test-clean clips of 1.3 to 2.4 s.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

TOOLS = Path(__file__).resolve().parents[2] / "overlay" / "tools" / "coreml"
sys.path.insert(0, str(TOOLS))

import mel_reference
from reference import ReferenceLock
from vulkan_mel import CHUNK_SAMPLES, N_FRAMES, valid_mel_frames

CFG = ReferenceLock.load(TOOLS / "parakeet-reference.lock").mel


def _speech_like(seconds: float) -> np.ndarray:
    rng = np.random.default_rng(7)
    t = np.arange(int(seconds * 16000), dtype=np.float32) / 16000
    tone = 0.2 * np.sin(2 * np.pi * 440 * t) + 0.1 * np.sin(2 * np.pi * 1210 * t)
    return (tone + 0.05 * rng.standard_normal(t.size)).astype(np.float32)


@pytest.mark.parametrize(
    "samples,frames",
    [
        (1, 2),  # at least 2: the standard deviation divides by n - 1
        (160, 2),
        (319, 2),
        (320, 3),
        (30_640, 192),  # LibriSpeech 1188-133604-0030, 1.915 s
        (CHUNK_SAMPLES - 1, 3000),
        (CHUNK_SAMPLES, N_FRAMES),
        (CHUNK_SAMPLES + 5_000, N_FRAMES),
    ],
)
def test_valid_mel_frames_follows_the_audio_length(samples, frames):
    assert valid_mel_frames(samples) == frames


def test_statistics_over_valid_frames_normalize_the_audio_and_zero_the_padding():
    rng = np.random.default_rng(3)
    valid = 120
    logmel = np.full((N_FRAMES, 128), -23.0, np.float32)  # constant padding, as log(guard) is
    logmel[:valid] = rng.normal(-8.0, 3.0, (valid, 128)).astype(np.float32)

    out = mel_reference.normalize(logmel, CFG, valid)

    assert out.shape == (N_FRAMES, 128)
    assert not out[valid:].any()
    assert np.allclose(out[:valid].mean(axis=0), 0.0, atol=1e-4)
    assert np.allclose(out[:valid].std(axis=0, ddof=1), 1.0, atol=1e-3)


def test_pinned_statistics_put_short_speech_at_the_wrong_scale():
    # The defect this contract fixes: 3001-frame statistics on a short clip.
    rng = np.random.default_rng(3)
    valid = 120
    logmel = np.full((N_FRAMES, 128), -23.0, np.float32)
    logmel[:valid] = rng.normal(-8.0, 3.0, (valid, 128)).astype(np.float32)

    legacy = mel_reference.normalize(logmel, CFG)

    assert legacy[:valid].mean() > 2.0


def test_default_normalize_is_unchanged_by_the_valid_argument():
    rng = np.random.default_rng(5)
    logmel = rng.normal(-8.0, 3.0, (N_FRAMES, 128)).astype(np.float32)

    assert np.array_equal(
        mel_reference.normalize(logmel, CFG), mel_reference.normalize(logmel, CFG, N_FRAMES)
    )


def test_duration_aware_chunk_marks_exactly_the_audio_frames():
    wave = _speech_like(1.5)
    valid = valid_mel_frames(wave.size)

    feats, mask = mel_reference.extract_chunk_features(wave, CFG, duration_aware=True)
    pinned_feats, pinned_mask = mel_reference.extract_chunk_features(wave, CFG)

    assert mask.dtype == np.int32
    assert int(mask.sum()) == valid == wave.size // 160 + 1
    assert mask[:valid].all() and not mask[valid:].any()
    assert not feats[valid:].any()
    assert np.allclose(feats[:valid].mean(axis=0), 0.0, atol=1e-3)
    assert pinned_mask.all()
    assert pinned_feats[:valid].mean() > 1.0  # the pinned contract's scale error on the same audio


def test_a_full_window_is_identical_in_both_contracts():
    wave = _speech_like(30.0)[:CHUNK_SAMPLES]

    aware, aware_mask = mel_reference.extract_chunk_features(wave, CFG, duration_aware=True)
    pinned, pinned_mask = mel_reference.extract_chunk_features(wave, CFG)

    assert np.array_equal(aware, pinned)
    assert np.array_equal(aware_mask, pinned_mask)
