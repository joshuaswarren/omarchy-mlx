# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Compare the portable mel pipeline with the pinned macOS stage capture.

The reference lock authenticates the input waveform, golden mel tensor,
stage-manifest bytes, and every required stage tensor. The comparison then
re-derives each stage with certified upstream inputs and reports the first
bitwise divergence.

Usage:
    python3 mel_stage_compare.py <capture-dir> <stage-dump-dir>

Exits 0 only if the evidence gate passes and every stage is byte-identical.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from dft_geometry_probe import candidate_spectrum
from dotpr import mel_projection
from mel_reference import chunk_for_mel, hann_window, mel_filterbank, preemphasize
from reference import ReferenceLock

TOOLS = Path(__file__).resolve().parent
STAGE_NAMES = (
    "dft_imag",
    "dft_real",
    "frames",
    "hann",
    "logmel",
    "mean",
    "mel_fb",
    "mel_mask",
    "mel_pinned",
    "mel_stepwise",
    "melproj",
    "power",
    "preemph",
    "std",
)


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_stage_evidence(
    capture_dir: Path, dumps_dir: Path, lock: ReferenceLock
) -> dict[str, str]:
    paths = lock.macos_reference_paths
    for name in (
        "waveform.npy",
        "mel.npy",
        "mel_mask.npy",
        "encoder_input_features.npy",
        "encoder_input_mask.npy",
    ):
        expected = paths.get(name)
        if expected is None:
            raise ValueError(f"{name}: reference lock does not pin this capture file")
        if sha256_file(capture_dir / name) != expected:
            raise ValueError(f"{name}: capture SHA-256 does not match the reference lock")

    manifest_path = dumps_dir / "manifest.json"
    expected_manifest = paths.get("mel_stage_manifest.json")
    if expected_manifest is None or sha256_file(manifest_path) != expected_manifest:
        raise ValueError("stage manifest SHA-256 does not match the reference lock")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_stages = {
        name: paths.get(f"mel_stage/{name}.npy") for name in STAGE_NAMES
    }
    if any(digest is None for digest in expected_stages.values()):
        raise ValueError("reference lock does not pin every required mel stage")
    if manifest != expected_stages:
        raise ValueError("stage manifest content does not match the reference lock")
    for name, expected in expected_stages.items():
        actual = sha256_file(dumps_dir / f"{name}.npy")
        if actual != expected:
            raise ValueError(
                f"{name}.npy: stage SHA-256 does not match the reference lock"
            )
    return manifest


def stage_diff(
    name: str, computed: np.ndarray, golden: np.ndarray
) -> tuple[bool, str]:
    if computed.shape != golden.shape:
        return False, (
            f"{name:10s}: SHAPE MISMATCH computed={computed.shape} "
            f"golden={golden.shape}"
        )
    if computed.dtype != golden.dtype:
        return False, (
            f"{name:10s}: DTYPE MISMATCH computed={computed.dtype} "
            f"golden={golden.dtype}"
        )

    computed_bytes = computed.tobytes(order="C")
    golden_bytes = golden.tobytes(order="C")
    if computed_bytes == golden_bytes:
        return True, f"{name:10s}: EXACT ({golden.size} values)"

    computed_u8 = np.frombuffer(computed_bytes, dtype=np.uint8)
    golden_u8 = np.frombuffer(golden_bytes, dtype=np.uint8)
    differing_bytes = computed_u8 != golden_u8
    differing_elements = differing_bytes.reshape(golden.size, golden.dtype.itemsize).any(
        axis=1
    )
    flat_index = int(np.flatnonzero(differing_elements)[0])
    first = tuple(int(v) for v in np.unravel_index(flat_index, golden.shape))
    diff = np.abs(computed.astype(np.float64) - golden.astype(np.float64))
    detail = (
        f"{name:10s}: DIVERGE n_diff={int(differing_elements.sum())}/{golden.size} "
        f"max|d|={float(np.max(diff)):.3e} first bitwise divergence={first} "
        f"golden={golden[first]!r} computed={computed[first]!r}"
    )
    return False, detail


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__.strip())
        return 2
    capture_dir, dumps_dir = Path(argv[1]), Path(argv[2])
    lock = ReferenceLock.load(TOOLS / "parakeet-reference.lock")
    cfg = lock.mel

    try:
        verify_stage_evidence(capture_dir, dumps_dir, lock)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"evidence gate failed: {error}")
        return 3

    golden = np.load(capture_dir / "mel.npy", allow_pickle=False)
    mel_pinned = np.load(dumps_dir / "mel_pinned.npy", allow_pickle=False)
    pinned_ok, pinned_line = stage_diff("mel_pinned", mel_pinned, golden)
    if not pinned_ok:
        print(pinned_line)
        return 3
    print("gate: pinned capture and stage evidence verified; mel_pinned == golden mel.npy")

    wave = np.load(capture_dir / "waveform.npy", allow_pickle=False)
    chunk = chunk_for_mel(wave)
    names = (
        "preemph",
        "hann",
        "mel_fb",
        "frames",
        "dft_real",
        "dft_imag",
        "power",
        "melproj",
        "logmel",
        "mean",
        "std",
    )
    mac = {
        name: np.load(dumps_dir / f"{name}.npy", allow_pickle=False)
        for name in names
    }
    ok = True

    ref_pre = preemphasize(chunk, cfg)
    good, line = stage_diff("preemph", ref_pre, mac["preemph"])
    print(line)
    ok &= good

    ref_hann = hann_window(cfg.win_length)
    good, line = stage_diff("hann", ref_hann, mac["hann"])
    print(line)
    ok &= good

    ref_fb = mel_filterbank(cfg)
    good, line = stage_diff("mel_fb", ref_fb, mac["mel_fb"])
    print(line)
    ok &= good

    padded = np.zeros(ref_pre.size + cfg.n_fft, np.float32)
    padded[cfg.n_fft // 2 : cfg.n_fft // 2 + ref_pre.size] = ref_pre
    idx = np.arange(cfg.win_length)[None, :] + cfg.hop_length * np.arange(
        mac["frames"].shape[0]
    )[:, None]
    ref_frames = np.zeros_like(mac["frames"])
    offset = (cfg.n_fft - cfg.win_length) // 2
    ref_frames[:, offset : offset + cfg.win_length] = (
        padded[idx] * mac["hann"][None, :]
    ).astype(np.float32)
    good, line = stage_diff("frames", ref_frames, mac["frames"])
    print(line)
    ok &= good

    dft_real, dft_imag = candidate_spectrum(mac["frames"], "vdsp-radix4-dif")
    good, line = stage_diff("dft_real", dft_real, mac["dft_real"])
    print(line)
    ok &= good
    good, line = stage_diff("dft_imag", dft_imag, mac["dft_imag"])
    print(line)
    ok &= good

    re32, im32 = mac["dft_real"], mac["dft_imag"]
    magnitude = np.sqrt(re32 * re32 + im32 * im32).astype(np.float32)
    good, line = stage_diff(
        "power", (magnitude * magnitude).astype(np.float32), mac["power"]
    )
    print(line)
    ok &= good

    ref_projection = mel_projection(mac["mel_fb"], mac["power"])
    good, line = stage_diff("melproj", ref_projection, mac["melproj"])
    print(line)
    ok &= good

    guarded = (mac["melproj"] + np.float32(cfg.log_guard)).astype(np.float32)
    ref_logmel = np.log(guarded.astype(np.float64)).astype(np.float32)
    good, line = stage_diff("logmel", ref_logmel, mac["logmel"])
    print(line)
    ok &= good

    logmel = mac["logmel"]
    n_frames = np.float32(logmel.shape[0])
    mean = np.cumsum(logmel, axis=0, dtype=np.float32)[-1] / n_frames
    deviation = (logmel - mean[None, :]).astype(np.float32)
    denominator = np.float32(max(logmel.shape[0] - 1, 1))
    variance = np.cumsum(
        (deviation * deviation).astype(np.float32), axis=0, dtype=np.float32
    )[-1] / denominator
    std = np.sqrt(variance).astype(np.float32)
    good, line = stage_diff("mean", mean, mac["mean"])
    print(line)
    ok &= good
    good, line = stage_diff("std", std, mac["std"])
    print(line)
    ok &= good

    print(
        "VERDICT:",
        "all stages byte-identical"
        if ok
        else (
            "divergent stages listed above (first divergent preprocessing stage "
            "is the earliest DIVERGE line whose upstream inputs were certified)"
        ),
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
