#!/usr/bin/env python3
"""Objective A/B: Kokoro streamed vs whole-call decoding on recorded pairs.

Implements the bars pre-registered in the lab notebook
(entries/KokoroQualify/20261003T181100Z-omp-studio-local-kokoro-objective-qualify.md):
per-pair duration delta, voiced-F0 contour agreement (pyin, numpy
autocorrelation fallback), mel-cepstral distortion after DTW alignment,
click/discontinuity statistics at the probe-logged chunk boundaries, and
pad-silence runs at utterance joins. The same statistics between two
whole-call renders of the same text give the same-wheel run-to-run floor.

Analysis only: reads wav files and the serve probe log, writes one JSON.
No synthesis, no hardware.

Usage:
    kokoro_objective_ab.py MAIN_DIR STREAM_DIR PROBE_JSONL OUT_JSON \
        [--floor-a DIR --floor-b DIR]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

SR = 24000
F0_MIN, F0_MAX = 60.0, 500.0
WINDOW_CHUNK_S = 0.6          # one 24-frame generator window
STEP_HALF_MS = 2.0            # click step search half-window
FLUX_HOP_S = 0.005            # spectral flux frame hop
FLUX_WIN_S = 0.025            # spectral flux window
SILENCE_THRESH = 3e-4         # ~ -70 dBFS
SILENCE_SEARCH_S = 0.09
DURATION_BAR = 0.02
F0_CORR_BAR = 0.95
F0_CENTS_BAR = 20.0
MCD_BAR_DB = 3.0
MCD_FLOOR_MARGIN_DB = 1.0
CLICK_MARGIN = 1.10
SILENCE_BAR_S = 0.100


# ------------------------------------------------------------------ audio io

def load_wav(path: Path) -> np.ndarray:
    import scipy.io.wavfile
    sr, x = scipy.io.wavfile.read(path)
    if sr != SR or x.ndim != 1 or x.dtype != np.int16:
        raise ValueError(f"{path}: expected mono 16-bit {SR} Hz, got "
                         f"{sr} Hz ndim={x.ndim} dtype={x.dtype}")
    return x.astype(np.float32) / 32768.0


# ------------------------------------------------------------------ boundaries

def boundaries_from_probe(probe_path: Path) -> dict[str, dict[str, list[int]]]:
    """Utterance/window joins and sentence joins, in samples, per item.

    Chunk edges come from the probe log's per-chunk durations (exact
    multiples of one 600-sample frame). An edge after chunk i is an
    utterance join when the next chunk was not ready at playout time
    (arrival gap > one window) or chunk i is an utterance tail (not a full
    window); it is a plain window join otherwise. A seq change is a
    sentence join (the browser issues one request per sentence).
    """
    out: dict[str, dict[str, list[int]]] = {}
    for line in probe_path.read_text().splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if event.get("event") != "item" or not event.get("chunks"):
            continue
        utt, win, sent, cum = [], [], [], 0
        seq = event["chunks"][0]["seq"]
        for i, chunk in enumerate(event["chunks"]):
            cum += round(chunk["dur"] * SR)
            last = i + 1 >= len(event["chunks"])
            if last:
                continue  # the file end is not a join
            nxt = event["chunks"][i + 1]
            edge = {"pos": cum, "arrival_gap": nxt["t"] - chunk["t"]}
            if nxt["seq"] != seq:
                sent.append(edge)
                seq = nxt["seq"]
            elif chunk["dur"] != WINDOW_CHUNK_S or edge["arrival_gap"] > WINDOW_CHUNK_S:
                utt.append(edge)
            else:
                win.append(edge)
        out[event["item"]] = {"utterance": utt, "window": win, "sentence": sent}
    return out


# ------------------------------------------------------------------ click statistics

def step_at(x: np.ndarray, idx: int, half_ms: float = STEP_HALF_MS) -> float:
    half = int(sr_half(half_ms))
    d = np.abs(np.diff(x))
    lo, hi = max(1, idx - half), min(d.size, idx + half)
    return float(d[lo:hi].max()) if hi > lo else 0.0


def sr_half(ms: float) -> float:
    return SR * ms / 1000.0


def flux_series(x: np.ndarray) -> np.ndarray:
    import librosa
    S = np.abs(librosa.stft(x, n_fft=1024, hop_length=int(FLUX_HOP_S * SR),
                            win_length=int(FLUX_WIN_S * SR), window="hann"))
    return np.linalg.norm(np.diff(S, axis=1), axis=0)


def flux_at(flux: np.ndarray, idx: int) -> float:
    t = min(idx // int(FLUX_HOP_S * SR), flux.size - 1)
    return float(flux[t])


def click_report(x_stream: np.ndarray, x_whole: np.ndarray,
                 edges: list[dict]) -> dict:
    """Boundary step/flux vs the whole-call file's own distribution."""
    d_whole = np.abs(np.diff(x_whole))
    flux_whole = flux_series(x_whole)
    ref = {"p999_step": float(np.quantile(d_whole, 0.999)),
           "max_step": float(d_whole.max()),
           "p999_flux": float(np.quantile(flux_whole, 0.999)),
           "max_flux": float(flux_whole.max())}
    flux_stream = flux_series(x_stream)
    rows, worst = [], {"step": 0.0, "flux": 0.0}
    for edge in edges:
        step = step_at(x_stream, edge["pos"])
        flux = flux_at(flux_stream, edge["pos"])
        rows.append({"pos": edge["pos"], "class": edge.get("class", "utterance"),
                     "step": step, "flux": flux,
                     "whole_step": step_at(x_whole, edge["pos"]),
                     "whole_flux": flux_at(flux_whole, edge["pos"])})
        worst["step"] = max(worst["step"], step)
        worst["flux"] = max(worst["flux"], flux)
    for key, ref_key in (("step", "step"), ("flux", "flux")):
        limit = max(CLICK_MARGIN * ref[f"p999_{ref_key}"], ref[f"max_{ref_key}"])
        worst[f"{key}_limit"] = limit
        worst[f"{key}_pass"] = bool(worst[key] <= limit)
    return {"reference": ref, "boundaries": rows, "worst": worst}


def longest_silence(x: np.ndarray, idx: int) -> float:
    search = int(SILENCE_SEARCH_S * SR)
    lo, hi = max(0, idx - search), min(x.size, idx + search)
    run = best = 0
    for quiet in np.abs(x[lo:hi]) < SILENCE_THRESH:
        run = run + 1 if quiet else 0
        best = max(best, run)
    return best / SR


# ------------------------------------------------------------------ F0

def f0_pyin(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    import librosa
    f0, voiced, _ = librosa.pyin(x, fmin=F0_MIN, fmax=F0_MAX, sr=SR,
                                 frame_length=2048, hop_length=256)
    return f0, np.asarray(voiced, dtype=bool)


def f0_autocorr(x: np.ndarray, win_s: float = 0.04, hop_s: float = 0.01
                ) -> tuple[np.ndarray, np.ndarray]:
    """Pure-numpy normalized autocorrelation F0; 0 where unvoiced."""
    win, hop = int(win_s * SR), int(hop_s * SR)
    lag_min, lag_max = int(SR / F0_MAX), int(SR / F0_MIN)
    rms = np.sqrt(np.convolve(x * x, np.ones(win) / win, mode="same"))
    gate = 10 ** (-45 / 20) * float(np.abs(x).max())
    f0 = np.zeros((x.size - win) // hop + 1)
    for i in range(f0.size):
        seg = x[i * hop:i * hop + win]
        seg = seg - seg.mean()
        energy = float(seg @ seg)
        if energy <= 0 or rms[i * hop + win // 2] < gate:
            continue
        ac = np.correlate(seg, seg, mode="full")[win - 1:]
        ac /= energy
        band = ac[lag_min:lag_max + 1]
        peak = int(band.argmax())
        if band[peak] < 0.5:
            continue
        j = lag_min + peak
        y0, y1, y2 = ac[j - 1], ac[j], ac[j + 1]
        denom = y0 - 2 * y1 + y2
        shift = 0.5 * (y0 - y2) / denom if denom else 0.0
        f0[i] = SR / (j + shift)
    return f0, f0 > 0


def f0_metrics(fa: np.ndarray, va: np.ndarray, fb: np.ndarray, vb: np.ndarray
               ) -> dict:
    n = min(fa.size, fb.size)
    va, vb = va[:n], vb[:n]
    fa, fb = np.nan_to_num(fa[:n]), np.nan_to_num(fb[:n])
    both = va & vb & (fa > 0) & (fb > 0)
    if both.sum() < 10:
        return {"voiced_frames": int(both.sum()), "corr": None,
                "median_cents": None, "pass": False}
    a, b = fa[both], fb[both]
    corr = float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 0 and np.std(b) > 0 else None
    cents = float(np.median(np.abs(1200 * np.log2(b / a))))
    return {"voiced_frames": int(both.sum()), "corr": corr,
            "median_cents": cents,
            "pass": bool(corr is not None and corr >= F0_CORR_BAR
                         and cents <= F0_CENTS_BAR)}


def aligned_frames_cost(a: np.ndarray, b: np.ndarray, band_rad: float = 0.15
                        ) -> tuple[np.ndarray, np.ndarray]:
    """(per-aligned-frame cost along the path, path) from a band-constrained
    (Sakoe-Chiba) DTW over CMVN-normalized 24-coefficient MFCC frames, c0
    excluded.

    Instrument correction (Main directive 2026-10-03, validated on
    controlled pairs, scripts/kokoro_mcd_validation.py): without per-utterance
    mean+variance normalization the coefficient L2 explodes on any gain or
    noise difference (a +40 dB-SNR copy of the SAME wav scored 84 dB). With
    CMVN: self 0.0, +40 dB 8.1, +20 dB 18.5, 1-sample delay 0.06, different
    text 29.6, same-text re-render 3.6 dB — the textbook ranges."""
    import librosa
    from scipy.spatial.distance import cdist
    ma = librosa.feature.mfcc(y=a, sr=SR, n_fft=1024, hop_length=256,
                              n_mels=80, n_mfcc=25)[1:]
    mb = librosa.feature.mfcc(y=b, sr=SR, n_fft=1024, hop_length=256,
                              n_mels=80, n_mfcc=25)[1:]
    ma = (ma - ma.mean(axis=1, keepdims=True)) / (ma.std(axis=1, keepdims=True) + 1e-8)
    mb = (mb - mb.mean(axis=1, keepdims=True)) / (mb.std(axis=1, keepdims=True) + 1e-8)
    cost = cdist(ma.T, mb.T, metric="euclidean")
    import librosa.sequence
    _, path = librosa.sequence.dtw(C=cost, band_rad=band_rad)
    return cost[path[:, 0], path[:, 1]], path


# ------------------------------------------------------------------ evaluation

def pair_metrics(x_whole: np.ndarray, x_stream: np.ndarray) -> dict:
    out = {"dur_whole_s": x_whole.size / SR, "dur_stream_s": x_stream.size / SR,
           "duration_delta": abs(x_stream.size - x_whole.size) / x_whole.size}
    fa, va = f0_pyin(x_whole)
    fb, vb = f0_pyin(x_stream)
    if not va.any() or not vb.any():
        fa, va = f0_autocorr(x_whole)
        fb, vb = f0_autocorr(x_stream)
        out["f0_estimator"] = "autocorr"
    else:
        out["f0_estimator"] = "pyin"
    aligned, path = aligned_frames_cost(x_whole, x_stream)
    out["mcd_db"] = float(10 * np.sqrt(2) / np.log(10) * aligned.mean())
    keep = ((path[:, 0] < fa.size) & (path[:, 1] < fb.size))
    out["f0"] = f0_metrics(fa[path[keep, 0]], va[path[keep, 0]],
                           fb[path[keep, 1]], vb[path[keep, 1]])
    out["duration_pass"] = bool(out["duration_delta"] <= DURATION_BAR)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("main_dir", type=Path)
    ap.add_argument("stream_dir", type=Path)
    ap.add_argument("probe", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--floor-a", type=Path, default=None,
                    help="whole-call render 1 for the noise floor (e.g. main r3)")
    ap.add_argument("--floor-b", type=Path, default=None,
                    help="whole-call render 2, same text (e.g. main r4)")
    args = ap.parse_args()

    bounds = boundaries_from_probe(args.probe)
    items = sorted(p.stem for p in args.main_dir.glob("*.wav"))
    rows = []
    for item in items:
        x_whole = load_wav(args.main_dir / f"{item}.wav")
        x_stream = load_wav(args.stream_dir / f"{item}.wav")
        row = {"item": item, **pair_metrics(x_whole, x_stream)}
        edges = bounds.get(item, {})
        for kind in ("utterance", "window"):
            for edge in edges.get(kind, []):
                edge["class"] = kind
        join_edges = edges.get("utterance", []) + edges.get("window", [])
        if join_edges:
            report = click_report(x_stream, x_whole, join_edges)
            row["clicks"] = report["reference"] | {"worst": report["worst"]}
            row["clicks_pass"] = all(report["worst"][k]
                                     for k in ("step_pass", "flux_pass"))
        row["utterance_joins"] = [
            {"pos": e["pos"], "silence_s": longest_silence(x_stream, e["pos"]),
             "step": step_at(x_stream, e["pos"])}
            for e in edges.get("utterance", [])]
        row["silence_pass"] = all(j["silence_s"] <= SILENCE_BAR_S
                                  for j in row["utterance_joins"])
        sentence_edges = edges.get("sentence", [])
        if sentence_edges:
            for edge in sentence_edges:
                edge["class"] = "sentence"
            row["sentence_joins"] = click_report(x_stream, x_whole,
                                                 sentence_edges)["worst"]
        rows.append(row)
        print(json.dumps({"item": item, "duration_delta": row["duration_delta"],
                          "f0": row["f0"], "mcd_db": row["mcd_db"],
                          "clicks_pass": row.get("clicks_pass"),
                          "silence_pass": row["silence_pass"]}), flush=True)

    result: dict = {"pairs": rows}
    if args.floor_a and args.floor_b:
        items_a = sorted(p.stem for p in args.floor_a.glob("*.wav"))
        floor_rows = []
        for item in items_a:
            a = load_wav(args.floor_a / f"{item}.wav")
            b = load_wav(args.floor_b / f"{item}.wav")
            row = {"item": item, **pair_metrics(a, b)}
            floor_rows.append(row)
            print(json.dumps({"floor_item": item,
                              "duration_delta": row["duration_delta"],
                              "f0": row["f0"], "mcd_db": row["mcd_db"]}),
                  flush=True)
        result["floor"] = floor_rows
        result["floor_summary"] = {
            "mcd_db_median": float(np.median([r["mcd_db"] for r in floor_rows])),
            "mcd_db_max": float(np.max([r["mcd_db"] for r in floor_rows])),
            "duration_delta_median": float(np.median(
                [r["duration_delta"] for r in floor_rows])),
            "f0_corr_median": _median_or_none(
                [r["f0"]["corr"] for r in floor_rows])}
        mcd_floor = result["floor_summary"]["mcd_db_median"]
        for row in rows:
            row["mcd_bar_db"] = max(MCD_BAR_DB, mcd_floor + MCD_FLOOR_MARGIN_DB)
            row["mcd_pass"] = bool(row["mcd_db"] <= MCD_BAR_DB
                                   or row["mcd_db"] <= mcd_floor + MCD_FLOOR_MARGIN_DB)
    else:
        for row in rows:
            row["mcd_bar_db"] = MCD_BAR_DB
            row["mcd_pass"] = bool(row["mcd_db"] <= MCD_BAR_DB)

    sentences = [r for r in rows if r["item"] != "para"]
    result["summary"] = {
        "duration_all_pass": all(r["duration_pass"] for r in rows),
        "duration_delta_median": float(np.median(
            [r["duration_delta"] for r in rows])),
        "f0_all_pass": all(r["f0"]["pass"] for r in rows),
        "f0_corr_median": _median_or_none([r["f0"]["corr"] for r in rows]),
        "f0_cents_median": _median_or_none(
            [r["f0"]["median_cents"] for r in rows]),
        "mcd_sentence_median_db": float(np.median(
            [r["mcd_db"] for r in sentences])),
        "mcd_all_pass": all(r["mcd_pass"] for r in rows),
        "clicks_all_pass": all(r.get("clicks_pass", True) for r in rows),
        "silence_all_pass": all(r["silence_pass"] for r in rows),
        "silence_max_s": max((j["silence_s"] for r in rows
                              for j in r["utterance_joins"]), default=0.0),
        "estimators": sorted({r["f0_estimator"] for r in rows}),
        "all_objective_bars_pass": bool(
            all(r["duration_pass"] for r in rows)
            and all(r["f0"]["pass"] for r in rows)
            and all(r["mcd_pass"] for r in rows)
            and all(r.get("clicks_pass", True) for r in rows)
            and all(r["silence_pass"] for r in rows)),
    }
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result["summary"], indent=2))
    return 0


def _median_or_none(values: list) -> float | None:
    vals = [v for v in values if v is not None]
    return float(np.median(vals)) if vals else None


if __name__ == "__main__":
    raise SystemExit(main())
