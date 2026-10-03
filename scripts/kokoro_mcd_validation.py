#!/usr/bin/env python3
"""MCD instrument validation (Main directive 2026-10-03): controlled pairs
decide whether the metric is valid before any bar is read from it.

Checks, in order:
  V1 identical signal        -> 0 by construction
  V2 +40 dB SNR white noise  -> small
  V3 +20 dB SNR white noise  -> larger than V2, still small
  V4 1-sample delay          -> near 0 (DTW absorbs it)
  V5 different sentence      -> large (the metric can go high)
  V6 same-text re-render pair (the floor) -> where does it sit?
  V7 per-coefficient diff stats on V6 (is one coefficient dominating?)
  V8 variants: CMVN per utterance, log10 scaling, no band
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kokoro_objective_ab import SR, aligned_frames_cost, load_wav, f0_pyin  # noqa: E402

RNG = np.random.default_rng(11)


def add_noise(x: np.ndarray, snr_db: float) -> np.ndarray:
    sig = float(np.sqrt(np.mean(x * x)))
    n = RNG.standard_normal(x.size).astype(np.float32)
    n *= sig / (10 ** (snr_db / 20)) / (float(np.sqrt(np.mean(n * n))) or 1.0)
    return x + n


def mfcc(x: np.ndarray, cmvn: bool = False, log10: bool = False):
    import librosa
    m = librosa.feature.mfcc(y=x, sr=SR, n_fft=1024, hop_length=256,
                             n_mels=80, n_mfcc=25)[1:]
    if log10:
        m = m / np.log(10)
    if cmvn:
        m = (m - m.mean(axis=1, keepdims=True)) / (m.std(axis=1, keepdims=True) + 1e-8)
    return m


def mcd_variants(a: np.ndarray, b: np.ndarray) -> dict:
    from scipy.spatial.distance import cdist
    import librosa.sequence
    out = {}
    for name, kw in (("plain", {}), ("cmvn", {"cmvn": True}),
                     ("log10", {"log10": True}),
                     ("cmvn_log10", {"cmvn": True, "log10": True})):
        ma, mb = mfcc(a, **kw), mfcc(b, **kw)
        cost = cdist(ma.T, mb.T, metric="euclidean")
        _, path = librosa.sequence.dtw(C=cost, band_rad=0.15)
        aligned = cost[path[:, 0], path[:, 1]]
        out[name] = round(float(10 * np.sqrt(2) / np.log(10) * aligned.mean()), 2)
        if name == "plain":
            out["path_len"] = int(path.shape[0])
            out["path_ratio"] = round(path.shape[0] / max(1, ma.shape[1]), 3)
    return out


def coef_stats(a: np.ndarray, b: np.ndarray) -> list:
    import librosa
    from scipy.spatial.distance import cdist
    import librosa.sequence
    ma = librosa.feature.mfcc(y=a, sr=SR, n_fft=1024, hop_length=256,
                              n_mels=80, n_mfcc=25)[1:]
    mb = librosa.feature.mfcc(y=b, sr=SR, n_fft=1024, hop_length=256,
                              n_mels=80, n_mfcc=25)[1:]
    cost = cdist(ma.T, mb.T, metric="euclidean")
    _, path = librosa.sequence.dtw(C=cost, band_rad=0.15)
    d = (ma[:, path[:, 0]] - mb[:, path[:, 1]])
    rms = np.sqrt((d ** 2).mean(axis=1))
    order = np.argsort(rms)[::-1][:6]
    return [(int(i), round(float(rms[i]), 2)) for i in order]


def main() -> None:
    suite = Path(sys.argv[1])
    a = load_wav(suite / "main" / "sent00.wav")
    f = load_wav(suite / "floor" / "sent00.wav")
    other = load_wav(suite / "main" / "sent07.wav")

    print("V1 self        ", mcd_variants(a, a))
    print("V2 +40dB noise ", mcd_variants(a, add_noise(a, 40.0)))
    print("V3 +20dB noise ", mcd_variants(a, add_noise(a, 20.0)))
    print("V4 +1 sample   ", mcd_variants(a, np.concatenate([a[:1], a])))
    print("V5 other text  ", mcd_variants(a, other))
    print("V6 floor pair  ", mcd_variants(a, f))
    print("V7 coef rms    ", coef_stats(a, f))
    print("V8 done")


if __name__ == "__main__":
    main()
