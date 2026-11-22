# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Per-stage RTFx breakdown on the GPU speech path (card V2).

Runs the shipped GPU STT worker (the same route the FLEURS row used) over a
wav corpus under the GPU lock and reports where the time goes per request:
load, resample, mel, encoder, decoder (the TDT step loop), joint and post,
in milliseconds, ranked. Audio decode via ffmpeg is outside the timed
window, matching FluidAudio's harness convention and matched_asr.py.

Usage (jwm2, from the branch checkout so the instrumented worker runs):

    flock -x -w 900 /tmp/m2-gpu.lock timeout -k 60 3600 \
        ~/.local/share/mlx-omarchy/venv/bin/python \
        harness/stage_timers_fleurs.py \
        --serve serve --model ~/.local/share/mlx-omarchy \
        --fleurs ~/scratch/wB4-m2/fleurs-fluidaudio-en-us-350 \
        --limit 350 --out ~/scratch/w71-v2-stagetimers
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


def _pcm_16k(path: Path) -> np.ndarray:
    pcm = subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-i", str(path),
         "-ar", "16000", "-ac", "1", "-f", "f32le", "pipe:1"],
        capture_output=True, check=True).stdout
    return np.frombuffer(pcm, dtype="<f4")


def _median_rows(rows: list[dict]) -> list[tuple[str, float, float]]:
    """(stage, median ms, share of median request total) ranked by median."""
    stages = sorted(rows[0]["stage_ms"])
    medians = {s: statistics.median(r["stage_ms"][s] for r in rows) for s in stages}
    wall = statistics.median(r["wall_ms"] for r in rows)
    accounted = sum(medians.values())
    ranked = sorted(medians.items(), key=lambda kv: -kv[1])
    return [(s, ms, 100.0 * ms / wall) for s, ms in ranked] + \
        [("unaccounted (IPC, payload copy, loop)", wall - accounted,
          100.0 * (wall - accounted) / wall)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="per-stage STT timers")
    parser.add_argument("--serve", required=True, type=Path,
                        help="checkout dir that holds mlx_omarchy_assistant/")
    parser.add_argument("--model", required=True, type=Path,
                        help="model home with the pinned parakeet files")
    parser.add_argument("--fleurs", required=True, type=Path,
                        help="corpus directory of wav files")
    parser.add_argument("--limit", type=int, default=350)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)

    sys.path.insert(0, str(args.serve.resolve()))
    from mlx_omarchy_assistant import gpu_stt

    model_dir = gpu_stt.model_dir(args.model)
    worker = gpu_stt._WorkerHandle(model_dir)
    wavs = sorted(args.fleurs.rglob("*.wav"))[: args.limit]
    if not wavs:
        print(f"no wav files under {args.fleurs}", file=sys.stderr)
        return 2
    print(f"stage-timer run: {len(wavs)} clips from {args.fleurs}", flush=True)

    rows: list[dict] = []
    try:
        for index, path in enumerate(wavs, 1):
            waveform = _pcm_16k(path)
            payload = waveform.astype("<f4", copy=False).tobytes()
            t0 = time.perf_counter()
            response = worker.request(
                {"op": "transcribe", "sample_rate": 16_000}, payload,
                timeout=300.0)
            wall_ms = (time.perf_counter() - t0) * 1000.0
            if not response.get("ok"):
                print(f"clip {index} failed: {response}", file=sys.stderr)
                continue
            rows.append({"file": path.name,
                         "audio_s": len(waveform) / 16_000,
                         "wall_ms": wall_ms,
                         "load_ms": response.get("load_ms"),
                         "stage_ms": response["stage_ms"]})
            if index % 50 == 0:
                print(f"  {index}/{len(wavs)} clips", flush=True)
    finally:
        worker.shutdown()
    if not rows:
        return 1

    audio_total = sum(r["audio_s"] for r in rows)
    proc_total = sum(r["wall_ms"] for r in rows) / 1000.0
    args.out.mkdir(parents=True, exist_ok=True)
    receipt = {
        "clips": len(rows),
        "audio_seconds": round(audio_total, 2),
        "processing_seconds": round(proc_total, 3),
        "overall_rtfx": round(audio_total / proc_total, 2),
        "model_dir": str(model_dir),
        "rows": rows,
    }
    (args.out / "stage-timers.json").write_text(json.dumps(receipt, indent=1))

    load = statistics.median(r["load_ms"] for r in rows if r["load_ms"])
    print(f"\noverall RTFx {receipt['overall_rtfx']}x "
          f"({audio_total:.1f} s audio / {proc_total:.1f} s processing, "
          f"{len(rows)} clips); median model load {load:.0f} ms")
    print("\n| Stage | median ms | share of request |")
    print("|---|---:|---:|")
    for stage, ms, share in _median_rows(rows):
        print(f"| {stage} | {ms:.2f} | {share:.1f}% |")
    print(f"\nreceipt: {args.out / 'stage-timers.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
