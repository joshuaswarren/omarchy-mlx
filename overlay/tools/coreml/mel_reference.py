# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Reference-derived host mel preprocessing for the pinned Parakeet capture.

Derived from ``mweinbach/parakeet-coreml-swift`` @
``75aec2a1c991319657ff4dec5f602c12da6c5012`` (Apache-2.0),
``Sources/ParakeetTDT/MelFeatureExtractor.swift`` and
``MelFilterBank.swift``, reproducing the pinned algorithm op-for-op with
installed NumPy only.

Status: exact portable CPU reference frontend for the pinned Parakeet fixture.

This module is a fixture oracle, not a qualified installed runtime. It uses
NumPy tensor operations on CPU and therefore does not satisfy plan §3.3 or the
backend-trace gate. Run the exact fixture comparison with:

    python3 overlay/tools/coreml/mel_reference.py <capture-dir>

Framing semantics note (verified against the golden tail rows): the
Swift source windows ``padded[t*hop .. t*hop+win-1]`` and places the
result at FFT bins ``[56 .. 455]``. That is 56 samples earlier than
``torch.stft(center=True)`` would window the same frame; the golden
capture's silence-tail structure (identical rows from frame 1046, not
1044) confirms the Swift offset is what was captured.
"""

from __future__ import annotations

import base64
import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from dft_geometry_probe import candidate_spectrum
from dotpr import mel_projection
from reference import MelConfig, ReferenceLock
from vulkan_mel import valid_mel_frames

CHUNK_SAMPLES = 3000 * 160  # 30 s chunk, as chunked by ParakeetTranscriber
ENCODER_MAX_TIME = 3000     # encoder input frames per chunk (traced shape)

# Exact 400-sample nonperiodic Hann coefficients captured from the pinned
# public Swift reference. Keeping their float32 bits avoids platform libm drift.
_HANN_400 = np.frombuffer(base64.b64decode(
    "AAAAAAAAgjgAAII5ADwSOgD4gToACss6gCcSOwDhRjsA14E7gEWkO4C5yjvAMPU7QNQRPOAOKzygRkY84HljPGBTgTyg5ZE8"
    "oHKjPFD5tTxgeMk8wO7dPABb8zzo3QQ96IcQPcCqHD2gRSk9yFc2PWDgQz2Q3lE9aFFgPQg4bz14kX49ZC6HPXRMjz1wopc9"
    "zC+gPfzzqD107rE9pB67PfSDxD3MHc49jOvXPZjs4T1QIOw9DIb2PZKOAD508gU+WG4LPuABET62rBY+fG4cPtJGIj5gNSg+"
    "vDkuPoxTND5qgjo+8MVAPrwdRz5kiU0+fghUPqaaWj5uP2E+avZnPiq/bj5EmXU+SIR8PuK/gT6mRYU+MtOIPk5ojD7CBJA+"
    "UqiTPsNSlz7YA5s+VruePgJ5oj6cPKY+7AWqPq/UrT6pqLE+nIG1PkxfuT50Qb0+2yfBPkASxT5gAMk+AfLMPt7m0D663tQ+"
    "T9nYPmXW3D601eA+/dbkPgDa6D593uw+LuTwPtbq9D4x8vg+Afr8Pv+AAD/4hAI/xogEP0uMBj9ljwg/9pEKP9mTDD/wlA4/"
    "GpUQPziUEj8mkhQ/x44WP/mJGD+bgxo/kHscP7ZxHj/tZSA/FFgiPw9IJD+7NSY/+iAoP64JKj+27ys/9NItP0qzLz+YkDE/"
    "wWozP6ZBNT8qFTc/LuU4P5axOj9Eejw/Gz8+PwAAQD/SvEE/e3VDP9opRT/T2UY/UIVIPzEsSj9czks/uGtNPyoETz+Yl1A/"
    "5iVSPwCvUz/KMlU/KLFWPwgqWD9PnVk/5gpbP7RyXD+k1F0/oDBfP5CGYD9g1mE/+h9jP0hjZD85oGU/tNZmP6oGaD8EMGk/"
    "sFJqP51uaz+5g2w/8JFtPzKZbj9wmW8/mJJwP5uEcT9ob3I/8lJzPykvdD8CBHU/bNF1P1uXdj/DVXc/mAx4P867eD9aY3k/"
    "MAN6P0mbej+ZK3s/F7R7P7o0fD98rXw/Ux59PziHfT8m6H0/FkF+PwKSfj/l2n4/uht/P35Ufz8shX8/wq1/Pz7Ofz+c5n8/"
    "3PZ/P/z+fz/8/n8/3PZ/P5zmfz8+zn8/w61/PyyFfz9+VH8/uht/P+bafj8Ckn4/FkF+PyfofT85h30/VB59P3ytfD+7NHw/"
    "GLR7P5orez9Km3o/MgN6P1pjeT/Pu3g/mAx4P8RVdz9cl3Y/bNF1PwQEdT8rL3Q/9FJzP2pvcj+bhHE/mJJwP3GZbz80mW4/"
    "8pFtP7qDbD+ebms/slJqPwUwaT+qBmg/ttZmPzqgZT9LY2Q/+x9jP2HWYT+ShmA/ojBfP6fUXT+2clw/5wpbP1GdWT8JKlg/"
    "KrFWP8wyVT8Cr1M/6iVSP5mXUD8qBE8/umtNP13OSz8zLEo/VIVIP9fZRj/aKUU/fXVDP9S8QT///z8/HT8+P0h6PD+YsTo/"
    "L+U4PywVNz+nQTU/xGozP52QMT9Nsy8/9tItP7rvKz+wCSo/+yAoP701Jj8TSCQ/F1giP+1lID+5cR4/knscP5yDGj/6iRg/"
    "yo4WPyiSFD88lBI/HZUQP/GUDj/ckww/+pEKP2mPCD9MjAY/yogEP/mEAj8AgQA/BPr8Pjry+D7b6vQ+MOTwPoLe7D4D2ug+"
    "BNfkPr/V4D5s1tw+VNnYPsHe1D7j5tA+AvLMPmcAyT5JEsU+4ifBPnhBvT5SX7k+oIG1PqiosT6y1K0+8gWqPqI8pj4KeaI+"
    "XLuePtoDmz7IUpc+VKiTPsEEkD5SaIw+MdOIPq9FhT7qv4E+UIR8PlSZdT4yv24+ePZnPng/YT6qmlo+ighUPmiJTT66HUc+"
    "9sVAPmqCOj6gUzQ+yjkuPmY1KD7iRiI+hG4cPrqsFj7qARE+XG4LPoDyBT6YjgA+EIb2PWAg7D2c7OE9hOvXPeAdzj0AhMQ9"
    "vB67PYTusT0E9Kg93C+gPXiilz10TI89bC6HPYCRfj0gOG89cFFgPYjeUT2I4EM94Fc2PchFKT3Yqhw9+IcQPQjeBD0gW/M8"
    "wO7dPIB4yTxQ+bU8oHKjPLDlkTxgU4E8AHpjPOBGRjwgDys8YNQRPAAx9TvAuco7wEWkO0DXgTuA4UY7ACgSOwAKyzoA+IE6"
    "ADwSOgAAgjkAAII4AAAAAA=="
), dtype="<f4")


def hann_window(win_length: int) -> np.ndarray:
    """Return the exact pinned 400-sample nonperiodic Hann window."""
    if win_length != 400:
        raise ValueError("only the pinned 400-sample Hann window is qualified")
    return _HANN_400.copy()

def mel_filterbank(cfg: MelConfig) -> np.ndarray:
    """Slaney-normalised mel filterbank, float64 math cast to float32.

    Same construction as ``librosa.filters.mel(htk=False, norm="slaney")``
    and ``MelFilterBank.swift``: rows [n_mels, n_fft/2 + 1].
    """
    f_sp = 200.0 / 3.0
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = np.log(6.4) / 27.0
    def hz_to_mel(hz):
        with np.errstate(divide="ignore"):
            return np.where(hz >= min_log_hz,
                            min_log_mel + np.log(hz / min_log_hz) / logstep,
                            hz / f_sp)
    def mel_to_hz(mel):
        return np.where(mel >= min_log_mel,
                        min_log_hz * np.exp(logstep * (mel - min_log_mel)),
                        f_sp * mel)

    f_min, f_max = 0.0, cfg.sample_rate / 2.0
    mels = hz_to_mel(f_min) + (hz_to_mel(f_max) - hz_to_mel(f_min)) * (
        np.arange(cfg.n_mels + 2) / (cfg.n_mels + 1))
    hz_pts = mel_to_hz(mels)
    fft_freqs = np.arange(cfg.n_fft // 2 + 1) * cfg.sample_rate / cfg.n_fft
    lower, center, upper = hz_pts[:-2], hz_pts[1:-1], hz_pts[2:]
    enorm = 2.0 / (upper - lower)
    tri = np.maximum(
        0.0,
        np.minimum((fft_freqs[None, :] - lower[:, None]) / (center - lower)[:, None],
                   (upper[:, None] - fft_freqs[None, :]) / (upper - center)[:, None]))
    return (tri * enorm[:, None]).astype(np.float32)


def preemphasize(waveform: np.ndarray, cfg: MelConfig) -> np.ndarray:
    """``y[0]=x[0]``; ``y[n]=x[n]-p*x[n-1]``, plain float32 mul-sub.

    The pinned binary compiles without FMA contraction here (FMA-emulated
    arithmetic matches the golden capture strictly worse; see receipt).
    """
    x = np.ascontiguousarray(waveform, dtype=np.float32)
    if x.size == 0:
        raise ValueError("waveform must be non-empty")
    y = np.empty_like(x)
    y[0] = x[0]
    y[1:] = (x[1:] - np.float32(cfg.preemphasis) * x[:-1]).astype(np.float32)
    return y


def stft_frames(waveform: np.ndarray, cfg: MelConfig) -> np.ndarray:
    """Windowed frames exactly as ``MelFeatureExtractor.extract`` lays them out.

    Returns ``(num_frames, n_fft)`` float32. Zero-pads ``n_fft/2`` both
    sides, then per frame t places ``padded[t*hop .. t*hop+win-1] * hann``
    at bins ``[(n_fft-win)/2 .. ]`` (Swift offset; see module docstring).
    """
    pre = preemphasize(waveform, cfg)
    padded = np.zeros(pre.size + cfg.n_fft, np.float32)
    padded[cfg.n_fft // 2:cfg.n_fft // 2 + pre.size] = pre
    num_frames = (padded.size - cfg.n_fft) // cfg.hop_length + 1
    if num_frames <= 0:
        return np.zeros((0, cfg.n_fft), np.float32)
    win_offset = (cfg.n_fft - cfg.win_length) // 2
    idx = np.arange(cfg.win_length)[None, :] + cfg.hop_length * np.arange(num_frames)[:, None]
    frames = np.zeros((num_frames, cfg.n_fft), np.float32)
    frames[:, win_offset:win_offset + cfg.win_length] = (
        padded[idx] * hann_window(cfg.win_length)[None, :]).astype(np.float32)
    return frames


def logmel(frames: np.ndarray, cfg: MelConfig,
           filterbank: np.ndarray | None = None) -> np.ndarray:
    """Apply the pinned vDSP DFT, power, mel projection, and log stages."""
    fb = mel_filterbank(cfg) if filterbank is None else filterbank
    re, im = candidate_spectrum(frames, "vdsp-radix4-dif")
    mag = np.sqrt(re * re + im * im).astype(np.float32)
    power = (mag * mag).astype(np.float32)
    mel = mel_projection(fb, power)
    guarded = (mel + np.float32(cfg.log_guard)).astype(np.float32)
    return np.log(guarded.astype(np.float64)).astype(np.float32)


def normalize(logmel_frames: np.ndarray, cfg: MelConfig, valid: int | None = None) -> np.ndarray:
    """Per-mel-bin mean/std over the frame axis, float32 sequential sums.

    Bessel-corrected variance (n-1), ``max(n-1, 1)`` denominator, output
    ``(x - mean) / (std + epsilon)``. Sequential accumulation is emulated
    exactly with float32 ``cumsum`` (the reference loops t in order).
    ``valid`` limits the statistics to the first ``valid`` frames and zeroes the rest.
    """
    lm = np.ascontiguousarray(logmel_frames, dtype=np.float32)
    head = lm if valid is None else lm[:valid]
    n_t = np.float32(head.shape[0])
    mean = np.cumsum(head, axis=0, dtype=np.float32)[-1] / n_t
    dev = (head - mean[None, :]).astype(np.float32)
    denom = np.float32(max(head.shape[0] - 1, 1))
    var = np.cumsum((dev * dev).astype(np.float32), axis=0, dtype=np.float32)[-1] / denom
    std = np.sqrt(var).astype(np.float32)
    out = ((lm - mean[None, :]) / (std[None, :] + np.float32(cfg.epsilon))).astype(np.float32)
    if valid is not None:
        out[valid:] = 0
    return out


def chunk_for_mel(waveform: np.ndarray) -> np.ndarray:
    """First-reference chunk contract: one 30 s chunk, zero-padded tail."""
    x = np.ascontiguousarray(waveform, dtype=np.float32)
    if x.size == 0:
        raise ValueError("waveform must be non-empty")
    if x.size > CHUNK_SAMPLES:
        raise ValueError("diagnostic covers the single-chunk reference path")
    if x.size < CHUNK_SAMPLES:
        x = np.concatenate([x, np.zeros(CHUNK_SAMPLES - x.size, np.float32)])
    return x


def extract_chunk_features(waveform_30s: np.ndarray, cfg: MelConfig, *,
                           duration_aware: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """(mel [3001,128] float32, attention mask [3001] int32) for one chunk.

    Default mirrors ``MelFeatureExtractor.extract``: every mel frame is valid
    (mask all ones) because the caller pre-pads the chunk to 30 s.
    ``duration_aware`` is NeMo's contract for the general path: statistics over
    the frames that carry audio, padded frames zero, mask one only on those frames.
    """
    valid = valid_mel_frames(int(np.asarray(waveform_30s).size)) if duration_aware else None
    frames = stft_frames(chunk_for_mel(waveform_30s), cfg)
    feats = normalize(logmel(frames, cfg), cfg, valid)
    mask = np.ones(frames.shape[0], np.int32)
    if valid is not None:
        mask[valid:] = 0
    return feats, mask


def encoder_slice(feats: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Encoder input contract: first 3000 frames of each chunk.

    ``ModelRunner.runEncoder`` copies ``min(frames, 3000)`` rows into a
    zero-filled [1,3000,128] buffer; the 3001st mel frame is computed
    but never fed. This slice reproduces that.
    """
    return feats[:ENCODER_MAX_TIME], mask[:ENCODER_MAX_TIME]


@dataclass
class DivergenceReport:
    exact: bool
    shape_expected: tuple
    shape_actual: tuple
    bit_exact_count: int
    total: int
    max_abs: float
    mean_abs: float
    first_divergence: tuple | None  # (frame, bin, golden, ours, ulps)
    structural: dict


def compare_capture(capture_dir: Path, lock_path: Path | None = None) -> DivergenceReport:
    """Recompute mel for a golden capture and locate the first divergence.

    Numeric bits, signed zeros, shapes, masks, and encoder slicing are checked
    exactly. Any mismatch remains a hard failure.
    """
    lock = ReferenceLock.load(lock_path or Path(__file__).parent / "parakeet-reference.lock")
    cfg = lock.mel
    for name in ("waveform.npy", "mel.npy", "mel_mask.npy",
                 "encoder_input_features.npy", "encoder_input_mask.npy"):
        with (capture_dir / name).open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != lock.macos_reference_paths.get(name):
            raise ValueError(f"{name}: capture SHA-256 does not match the reference lock")
    wave = np.load(capture_dir / "waveform.npy")
    golden = np.load(capture_dir / "mel.npy")
    ours, mask = extract_chunk_features(wave, cfg)

    if ours.shape != golden.shape or ours.dtype != np.float32 or golden.dtype != np.float32:
        raise ValueError("mel comparison requires matching shapes and float32 arrays")
    if not np.isfinite(ours).all() or not np.isfinite(golden).all():
        raise ValueError("mel comparison requires finite arrays")
    same = ours.view(np.uint32) == golden.view(np.uint32)
    diff = np.abs(ours.astype(np.float64) - golden.astype(np.float64))
    bad = np.argwhere(~same)
    first = None
    if bad.size:
        f, b = bad[0]
        golden_v = np.float32(golden[f, b])
        ulps = abs(np.float32(ours[f, b]) - golden_v) / abs(np.spacing(golden_v))
        first = (int(f), int(b), float(golden_v), float(ours[f, b]), float(ulps))

    g_mask = np.load(capture_dir / "mel_mask.npy")
    g_eif = np.load(capture_dir / "encoder_input_features.npy")
    g_eim = np.load(capture_dir / "encoder_input_mask.npy")
    tail_identical = (golden[1046:].view(np.uint32) == golden[1046].view(np.uint32)).all() if golden.shape[0] > 1046 else False
    es, em = encoder_slice(ours, mask)
    structural = {
        "shape_matches": ours.shape == golden.shape,
        "mask_all_ones": bool((mask == 1).all()) and mask.shape == g_mask.shape,
        "mask_equals_golden": mask.dtype == g_mask.dtype == np.int32 and bool(np.array_equal(mask, g_mask)),
        "encoder_slice_shape": g_eif.shape == (1, *es.shape),
        "encoder_slice_mask_equals_golden": g_eim.shape == (1, *em.shape) and em.dtype == g_eim.dtype == np.int32 and bool(np.array_equal(em, g_eim[0])),
        "encoder_slice_features_exact": g_eif.shape == (1, *es.shape) and es.dtype == g_eif.dtype == np.float32 and es.tobytes() == g_eif[0].tobytes(),
        "silence_tail_identical_rows_from_1046": bool(tail_identical),
    }

    return DivergenceReport(
        exact=bool(same.all()),
        shape_expected=golden.shape,
        shape_actual=ours.shape,
        bit_exact_count=int(same.sum()),
        total=int(diff.size),
        max_abs=float(diff.max()),
        mean_abs=float(diff.mean()),
        first_divergence=first,
        structural=structural,
    )


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__.strip())
        return 2
    rep = compare_capture(Path(argv[1]))
    print(f"shape: expected {rep.shape_expected}, computed {rep.shape_actual}")
    print(f"exact golden equality: {rep.exact}")
    print(f"bit-identical values: {rep.bit_exact_count}/{rep.total}"
          f" ({100.0 * rep.bit_exact_count / rep.total:.2f}%)")
    print(f"max |diff|: {rep.max_abs:.6e}   mean |diff|: {rep.mean_abs:.6e}")
    if rep.first_divergence:
        f, b, g, o, u = rep.first_divergence
        print(f"first divergence: frame {f} bin {b}: golden {g!r} vs computed {o!r}"
              f" ({u:.0f} ulps)")
    for k, v in rep.structural.items():
        print(f"structural {k}: {v}")
    if not rep.exact or not all(rep.structural.values()):
        print("VERDICT: exact pinned mel gate failed. CPU fixture oracle only; no installed-runtime qualification.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
