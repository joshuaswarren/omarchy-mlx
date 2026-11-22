"""Score the silero VAD stage on FluidAudio's mini50 set (the V25 protocol).

Same labelled set and per-clip rule as the V25 energy-gate receipt
(the earlier energy-gate receipt's protocol): clip is
speech iff any frame clears the threshold. Frames are silero's 512 samples
(32 ms). The decision code under test is the shipped SileroVad (see
tests/test_assistant_vad.py for the known-answer self-test of the framing,
state and threshold logic).

Run:  python3 score_mini50.py --mini50 <dir> --model <silero_vad.onnx> \
        --thresholds 0.3,0.5,0.7 --out summary.json
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from mlx_omarchy_assistant import vad


def prf(labels, predictions):
    tp = sum(1 for l, p in zip(labels, predictions) if l == 1 and p)
    fn = sum(1 for l, p in zip(labels, predictions) if l == 1 and not p)
    fp = sum(1 for l, p in zip(labels, predictions) if l == 0 and p)
    tn = sum(1 for l, p in zip(labels, predictions) if l == 0 and not p)

    def ratio(num, den):
        return round(num / den, 4) if den else None

    predicted, actual = tp + fp, tp + fn
    return {
        "accuracy": ratio(tp + tn, tp + fn + fp + tn),
        "precision": ratio(tp, predicted),
        "recall": ratio(tp, actual),
        "f1": ratio(2 * tp, predicted + actual) if predicted + actual else None,
        "specificity": ratio(tn, fp + tn),
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
    }


def main(argv=None) -> int:
    import soundfile as sf

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mini50", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--thresholds", default="0.5")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    thresholds = [float(t) for t in args.thresholds.split(",")]

    clips = []
    for label, group in ((1, "speech"), (0, "noise")):
        for path in sorted((args.mini50 / group).glob("*.wav")):
            data, rate = sf.read(path, dtype="int16", always_2d=False)
            if data.ndim > 1:
                data = data[:, 0]
            assert rate == 16000, f"{path}: {rate} Hz"
            clips.append((label, path.name, data.tobytes()))
    audio_s = sum(len(pcm) / 2 / 16000 for _, _, pcm in clips)

    started = time.perf_counter()
    rows = []
    for label, name, pcm in clips:
        decisions = {}
        for threshold in thresholds:
            model = vad.SileroVad(args.model, threshold=threshold)
            decisions[str(threshold)] = bool(model.process(pcm))
        rows.append({"file": name, "label": label, "decisions": decisions})
    elapsed = time.perf_counter() - started

    table = {}
    for threshold in thresholds:
        predictions = [int(r["decisions"][str(threshold)]) for r in rows]
        table[str(threshold)] = prf([r["label"] for r in rows], predictions)
    # silence frames skip the model when a pre-filter is set; measure how
    # many frames an rms pre-filter at 300 would have rejected
    silent = 0
    total_frames = 0
    for _, _, pcm in clips:
        for off in range(0, len(pcm), 1280):
            x = np.frombuffer(pcm[off:off + 1280], dtype=np.int16).astype(np.float64)
            total_frames += 1
            if float(np.sqrt(np.mean(x ** 2))) < 300.0:
                silent += 1
    summary = {
        "dataset": "alexwengg/musan_mini50 (FluidAudio vad-benchmark mini50)",
        "model": "silero-vad v5 ONNX (MIT), CPU onnxruntime, 512-sample frames",
        "clips": len(rows),
        "audio_seconds": round(audio_s, 2),
        "cpu_processing_seconds": round(elapsed, 3),
        "cpu_rtfx_all_thresholds": round(audio_s / elapsed, 1),
        "frames_total": total_frames,
        "frames_silent_under_rms300": silent,
        "prefilter_skip_share": round(silent / total_frames, 4) if total_frames else None,
        "table": table,
        "rows": rows,
    }
    args.out.write_text(json.dumps(summary, indent=1))
    for threshold, row in table.items():
        print(f"thr={threshold} acc={row['accuracy']} P={row['precision']} "
              f"R={row['recall']} F1={row['f1']} spec={row['specificity']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
