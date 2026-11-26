# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Generate and compare valid 512-point real-DFT impulse probes.

The impulse basis alternates a unit sample at ``2*c`` and ``2*c+1`` for
``c = 0..255``. Candidate 256-point complex cores feed the established
real-DFT untangle, then exact float32 and ULP signatures are compared with a
``mel-stage-capture --mode probe-dft`` dump.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from dotpr import _fma32

N_REAL = 512
N_COMPLEX = N_REAL // 2
N_BINS = N_REAL // 2 + 1


def f32(value: np.ndarray) -> np.ndarray:
    return np.asarray(value, dtype=np.float32)


def c64(value: np.ndarray) -> np.ndarray:
    return np.asarray(value, dtype=np.complex64)


def twiddle(length: int, indices: np.ndarray) -> np.ndarray:
    return np.exp(-2j * np.pi * indices / length).astype(np.complex64)


def digit_reverse_base4(length: int) -> np.ndarray:
    stages = length.bit_length() // 2
    if 4**stages != length:
        raise ValueError(f"radix-4 length must be a power of four, got {length}")
    result = np.zeros(length, dtype=np.int64)
    for source in range(length):
        value = source
        target = 0
        for _ in range(stages):
            target = 4 * target + value % 4
            value //= 4
        result[source] = target
    return result


def radix4_dit(values: np.ndarray) -> np.ndarray:
    z = np.take(c64(values), digit_reverse_base4(values.shape[1]), axis=1)
    length = z.shape[1]
    stage = 4
    while stage <= length:
        quarter = stage // 4
        groups = z.reshape(z.shape[0], length // stage, stage)
        indices = np.arange(quarter)
        a = groups[:, :, :quarter]
        b = c64(groups[:, :, quarter:2 * quarter] * twiddle(stage, indices))
        c = c64(groups[:, :, 2 * quarter:3 * quarter] * twiddle(stage, 2 * indices))
        d = c64(groups[:, :, 3 * quarter:] * twiddle(stage, 3 * indices))
        ac_sum = c64(a + c)
        ac_diff = c64(a - c)
        bd_sum = c64(b + d)
        bd_diff = c64(b - d)
        minus_i_bd = c64(bd_diff.imag - 1j * bd_diff.real)
        out = np.empty_like(groups)
        out[:, :, :quarter] = c64(ac_sum + bd_sum)
        out[:, :, quarter:2 * quarter] = c64(ac_diff + minus_i_bd)
        out[:, :, 2 * quarter:3 * quarter] = c64(ac_sum - bd_sum)
        out[:, :, 3 * quarter:] = c64(ac_diff - minus_i_bd)
        z = out.reshape(z.shape)
        stage *= 4
    return z


def radix4_dif(values: np.ndarray) -> np.ndarray:
    z = c64(values).copy()
    length = z.shape[1]
    stage = length
    while stage >= 4:
        quarter = stage // 4
        groups = z.reshape(z.shape[0], length // stage, stage)
        a = groups[:, :, :quarter]
        b = groups[:, :, quarter:2 * quarter]
        c = groups[:, :, 2 * quarter:3 * quarter]
        d = groups[:, :, 3 * quarter:]
        ac_sum = c64(a + c)
        ac_diff = c64(a - c)
        bd_sum = c64(b + d)
        bd_diff = c64(b - d)
        minus_i_bd = c64(bd_diff.imag - 1j * bd_diff.real)
        indices = np.arange(quarter)
        out = np.empty_like(groups)
        out[:, :, :quarter] = c64(ac_sum + bd_sum)
        out[:, :, quarter:2 * quarter] = c64(
            (ac_diff + minus_i_bd) * twiddle(stage, indices)
        )
        out[:, :, 2 * quarter:3 * quarter] = c64(
            (ac_sum - bd_sum) * twiddle(stage, 2 * indices)
        )
        out[:, :, 3 * quarter:] = c64(
            (ac_diff - minus_i_bd) * twiddle(stage, 3 * indices)
        )
        z = out.reshape(z.shape)
        stage //= 4
    return np.take(z, digit_reverse_base4(length), axis=1)




def vdsp_radix4_dif(values: np.ndarray) -> np.ndarray:
    z = c64(values)
    real = np.array(z.real, dtype=np.float32, copy=True)
    imag = np.array(z.imag, dtype=np.float32, copy=True)
    batch, length = real.shape
    prefix_count = 1
    remaining = length
    while remaining > 1:
        next_remaining = remaining // 4
        shape = (batch, 4, next_remaining, prefix_count)
        grouped_real = real.reshape(shape)
        grouped_imag = imag.reshape(shape)
        xr = [grouped_real[:, branch] for branch in range(4)]
        xi = [grouped_imag[:, branch] for branch in range(4)]
        if prefix_count == 1:
            ac_real = f32(xr[0] + xr[2])
            ac_imag = f32(xi[0] + xi[2])
            ad_real = f32(xr[0] - xr[2])
            ad_imag = f32(xi[0] - xi[2])
            bd_real = f32(xr[1] + xr[3])
            bd_imag = f32(xi[1] + xi[3])
            bm_real = f32(xr[1] - xr[3])
            bm_imag = f32(xi[1] - xi[3])
            yr = (
                f32(ac_real + bd_real),
                f32(ad_real + bm_imag),
                f32(ac_real - bd_real),
                f32(ad_real - bm_imag),
            )
            yi = (
                f32(ac_imag + bd_imag),
                f32(ad_imag - bm_real),
                f32(ac_imag - bd_imag),
                f32(ad_imag + bm_real),
            )
        else:
            angles = [
                2.0 * math.pi * index / (4 * prefix_count)
                for index in range(prefix_count)
            ]
            cos1 = np.asarray([math.cos(angle) for angle in angles], dtype=np.float32)[
                None, None, :
            ]
            tan1 = np.asarray([math.tan(angle) for angle in angles], dtype=np.float32)[
                None, None, :
            ]
            cos2 = np.asarray(
                [math.cos(2.0 * angle) for angle in angles], dtype=np.float32
            )[None, None, :]
            tan2 = np.asarray(
                [math.tan(2.0 * angle) for angle in angles], dtype=np.float32
            )[None, None, :]
            ratio3 = np.asarray(
                [math.cos(3.0 * angle) / math.cos(angle) for angle in angles],
                dtype=np.float32,
            )[None, None, :]
            tan3 = np.asarray(
                [math.tan(3.0 * angle) for angle in angles], dtype=np.float32
            )[None, None, :]
            ti1 = _fma32(-xr[1], tan1, xi[1])
            tr1 = _fma32(xi[1], tan1, xr[1])
            ti2 = _fma32(-xr[2], tan2, xi[2])
            tr2 = _fma32(xi[2], tan2, xr[2])
            if remaining == 16:
                special = prefix_count // 2
                ti2[:, :, special] = -xr[2][:, :, special]
                tr2[:, :, special] = xi[2][:, :, special]
            ti3 = _fma32(-xr[3], tan3, xi[3])
            tr3 = _fma32(xi[3], tan3, xr[3])
            eip = _fma32(ti2, cos2, xi[0])
            eim = _fma32(-ti2, cos2, xi[0])
            erp = _fma32(tr2, cos2, xr[0])
            erm = _fma32(-tr2, cos2, xr[0])
            if remaining == 16:
                special = prefix_count // 2
                eip[:, :, special] = f32(
                    xi[0][:, :, special] + ti2[:, :, special]
                )
                eim[:, :, special] = f32(
                    xi[0][:, :, special] - ti2[:, :, special]
                )
                erp[:, :, special] = f32(
                    xr[0][:, :, special] + tr2[:, :, special]
                )
                erm[:, :, special] = f32(
                    xr[0][:, :, special] - tr2[:, :, special]
                )
            oip = _fma32(ti3, ratio3, ti1)
            oim = _fma32(-ti3, ratio3, ti1)
            orp = _fma32(tr3, ratio3, tr1)
            orm = _fma32(-tr3, ratio3, tr1)
            yr = (
                _fma32(orp, cos1, erp),
                _fma32(oim, cos1, erm),
                _fma32(-orp, cos1, erp),
                _fma32(-oim, cos1, erm),
            )
            yi = (
                _fma32(oip, cos1, eip),
                _fma32(-orm, cos1, eim),
                _fma32(-oip, cos1, eip),
                _fma32(orm, cos1, eim),
            )
        real = np.concatenate(yr, axis=2).reshape(batch, length)
        imag = np.concatenate(yi, axis=2).reshape(batch, length)
        prefix_count *= 4
        remaining //= 4
    result = np.empty((batch, length), dtype=np.complex64)
    result.real = real
    result.imag = imag
    return result


def split_radix_dit(values: np.ndarray) -> np.ndarray:
    z = c64(values)
    length = z.shape[1]
    if length == 1:
        return z.copy()
    if length == 2:
        return np.stack((c64(z[:, 0] + z[:, 1]), c64(z[:, 0] - z[:, 1])), axis=1)
    quarter = length // 4
    even = split_radix_dit(z[:, ::2])
    odd_one = split_radix_dit(z[:, 1::4])
    odd_three = split_radix_dit(z[:, 3::4])
    indices = np.arange(quarter)
    first = c64(odd_one * twiddle(length, indices))
    third = c64(odd_three * twiddle(length, 3 * indices))
    odd_sum = c64(first + third)
    odd_diff = c64(first - third)
    minus_i_diff = c64(odd_diff.imag - 1j * odd_diff.real)
    out = np.empty_like(z)
    out[:, :quarter] = c64(even[:, :quarter] + odd_sum)
    out[:, quarter:2 * quarter] = c64(even[:, quarter:] + minus_i_diff)
    out[:, 2 * quarter:3 * quarter] = c64(even[:, :quarter] - odd_sum)
    out[:, 3 * quarter:] = c64(even[:, quarter:] - minus_i_diff)
    return out


def vdsp_real_coefficients() -> tuple[np.ndarray, np.ndarray]:
    """Reproduce the float32 coefficient table used by the 512-point zrop setup.

    The setup evaluates float32 angles. Five cosine entries follow Darwin's
    scalar float rounding instead of the correctly rounded double result.
    """
    indices = np.arange(1, N_COMPLEX // 2 + 1, dtype=np.float32)
    angles = f32(np.float32(math.pi) * indices / np.float32(N_COMPLEX))
    cosine = f32([math.cos(float(angle)) for angle in angles])
    sine = f32([math.sin(float(angle)) for angle in angles])
    for index, direction in ((49, 1), (86, -1), (105, 1), (110, 1), (119, 1)):
        cosine[index - 1] = np.nextafter(
            cosine[index - 1], np.float32(direction * math.inf), dtype=np.float32
        )
    return cosine, sine


def untangle(core: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    core_real = f32(core.real)
    core_imag = f32(core.imag)
    low = np.arange(1, N_COMPLEX // 2 + 1)
    mirrored = N_COMPLEX - low
    direct_real = core_real[:, low]
    direct_imag = core_imag[:, low]
    mirrored_real = core_real[:, mirrored]
    mirrored_imag = core_imag[:, mirrored]
    imag_sum = f32(direct_imag + mirrored_imag)
    real_diff = f32(mirrored_real - direct_real)
    real_sum = f32(direct_real + mirrored_real)
    imag_diff = f32(direct_imag - mirrored_imag)
    cosine, sine = vdsp_real_coefficients()
    weighted_real = f32(f32(cosine * imag_sum) + f32(sine * real_diff))
    weighted_imag = f32(f32(cosine * real_diff) - f32(sine * imag_sum))
    half = np.float32(0.5)
    real = np.zeros((core.shape[0], N_BINS), dtype=np.float32)
    imag = np.zeros_like(real)
    twice_real_zero = f32(core_real[:, 0] + core_real[:, 0])
    twice_imag_zero = f32(core_imag[:, 0] + core_imag[:, 0])
    real[:, 0] = f32(half * f32(twice_real_zero + twice_imag_zero))
    real[:, -1] = f32(half * f32(twice_real_zero - twice_imag_zero))
    real[:, low] = f32(half * f32(real_sum + weighted_real))
    imag[:, low] = f32(half * f32(weighted_imag + imag_diff))
    real[:, mirrored] = f32(half * f32(real_sum - weighted_real))
    imag[:, mirrored] = f32(half * f32(weighted_imag - imag_diff))
    return real, imag


def candidate_core(frames: np.ndarray, candidate: str) -> np.ndarray:
    packed = np.empty((frames.shape[0], N_COMPLEX), dtype=np.complex64)
    packed.real = frames[:, 0::2]
    packed.imag = frames[:, 1::2]
    functions = {
        "radix4-dit": radix4_dit,
        "radix4-dif": radix4_dif,
        "split-radix-dit": split_radix_dit,
        "vdsp-radix4-dif": vdsp_radix4_dif,
    }
    return functions[candidate](packed)


def candidate_spectrum(frames: np.ndarray, candidate: str) -> tuple[np.ndarray, np.ndarray]:
    frames = np.asarray(frames)
    if frames.dtype != np.float32:
        raise ValueError("DFT frames must have float32 dtype")
    if frames.ndim != 2:
        raise ValueError("DFT frames must be two-dimensional")
    if frames.shape[1] != N_REAL:
        raise ValueError(f"DFT frames must have exactly {N_REAL} samples")
    return untangle(candidate_core(frames, candidate))


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_probe(directory: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    manifest = json.loads((directory / "manifest.json").read_text())
    for name, expected in manifest.items():
        path = directory / f"{name}.npy"
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"manifest mismatch for {name}: expected {expected}, got {actual}")
    frames = np.load(directory / "probe_in.npy").reshape(-1, N_REAL)
    real = np.load(directory / "probe_dft_real.npy")
    imag = np.load(directory / "probe_dft_imag.npy")
    if frames.dtype != np.float32 or real.dtype != np.float32 or imag.dtype != np.float32:
        raise ValueError("probe arrays must be float32")
    if real.shape != (frames.shape[0], N_BINS) or imag.shape != real.shape:
        raise ValueError(
            f"invalid probe shapes: input={frames.shape}, real={real.shape}, imag={imag.shape}"
        )
    if not np.isfinite(frames).all() or not np.isfinite(real).all() or not np.isfinite(imag).all():
        raise ValueError("probe arrays contain non-finite values")
    return frames, real, imag


def load_core_probe(directory: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    manifest = json.loads((directory / "manifest.json").read_text())
    for name, expected in manifest.items():
        path = directory / f"{name}.npy"
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"manifest mismatch for {name}: expected {expected}, got {actual}")
    frames = np.load(directory / "probe_in.npy").reshape(-1, N_REAL)
    real = np.load(directory / "probe_core_real.npy")
    imag = np.load(directory / "probe_core_imag.npy")
    if frames.dtype != np.float32 or real.dtype != np.float32 or imag.dtype != np.float32:
        raise ValueError("core probe arrays must be float32")
    if real.shape != (frames.shape[0], N_COMPLEX) or imag.shape != real.shape:
        raise ValueError(
            f"invalid core probe shapes: input={frames.shape}, real={real.shape}, imag={imag.shape}"
        )
    if not np.isfinite(frames).all() or not np.isfinite(real).all() or not np.isfinite(imag).all():
        raise ValueError("core probe arrays contain non-finite values")
    return frames, real, imag


def load_stage(directory: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    manifest = json.loads((directory / "manifest.json").read_text())
    for name in ("frames", "dft_real", "dft_imag"):
        expected = manifest[name]
        actual = sha256_file(directory / f"{name}.npy")
        if actual != expected:
            raise ValueError(
                f"manifest mismatch for {name}: expected {expected}, got {actual}"
            )
    frames = np.load(directory / "frames.npy")
    real = np.load(directory / "dft_real.npy")
    imag = np.load(directory / "dft_imag.npy")
    if frames.dtype != np.float32 or real.dtype != np.float32 or imag.dtype != np.float32:
        raise ValueError("stage arrays must be float32")
    if frames.ndim != 2 or frames.shape[1] != N_REAL:
        raise ValueError(f"invalid stage frame shape: {frames.shape}")
    if real.shape != (frames.shape[0], N_BINS) or imag.shape != real.shape:
        raise ValueError(
            f"invalid stage shapes: input={frames.shape}, real={real.shape}, imag={imag.shape}"
        )
    if not np.isfinite(frames).all() or not np.isfinite(real).all() or not np.isfinite(imag).all():
        raise ValueError("stage arrays contain non-finite values")
    return frames, real, imag


def ordered_float_bits(values: np.ndarray) -> np.ndarray:
    bits = values.view(np.uint32)
    positive = bits ^ np.uint32(0x80000000)
    return np.where(bits & np.uint32(0x80000000), ~bits, positive).astype(np.int64)


def score(candidate_real: np.ndarray, candidate_imag: np.ndarray,
          reference_real: np.ndarray, reference_imag: np.ndarray) -> dict[str, object]:
    candidate = np.stack((candidate_real, candidate_imag), axis=-1)
    reference = np.stack((reference_real, reference_imag), axis=-1)
    bit_exact = candidate.view(np.uint32) == reference.view(np.uint32)
    numeric_exact = candidate == reference
    signed_ulp = ordered_float_bits(candidate) - ordered_float_bits(reference)
    mismatch_ulp = signed_ulp[~numeric_exact]
    histogram = {
        str(delta): int((mismatch_ulp == delta).sum())
        for delta in range(-8, 9)
        if (mismatch_ulp == delta).any()
    }
    histogram["<-8"] = int((mismatch_ulp < -8).sum())
    histogram[">8"] = int((mismatch_ulp > 8).sum())
    row_mismatches = (~numeric_exact).sum(axis=(1, 2)).astype(np.uint16)
    return {
        "bit_mismatches": int((~bit_exact).sum()),
        "numeric_mismatches": int((~numeric_exact).sum()),
        "signed_zero_mismatches": int((numeric_exact & ~bit_exact).sum()),
        "values": int(bit_exact.size),
        "exact_rows": int((row_mismatches == 0).sum()),
        "max_abs": float(np.max(np.abs(
            candidate.astype(np.float64) - reference.astype(np.float64)
        ))),
        "max_ulp_on_numeric_mismatch": int(np.max(np.abs(mismatch_ulp), initial=0)),
        "signed_ulp_histogram": histogram,
        "row_mismatch_signature_sha256": hashlib.sha256(row_mismatches.tobytes()).hexdigest(),
    }


def validate_impulse_basis(frames: np.ndarray) -> None:
    expected = np.eye(N_REAL, dtype=np.float32)
    if frames.shape != expected.shape or frames.tobytes() != expected.tobytes():
        raise ValueError("impulse probe input must be the exact 512x512 float32 identity basis")


def make_input(path: Path) -> None:
    basis = np.eye(N_REAL, dtype="<f4")
    path.write_bytes(basis.tobytes())
    print(json.dumps({
        "path": str(path),
        "shape": list(basis.shape),
        "dtype": str(basis.dtype),
        "sha256": sha256_file(path),
    }, sort_keys=True))


def compare(impulse_dump: Path, core_dump: Path | None, general_dump: Path | None,
            stage_dump: Path | None, json_out: Path | None) -> None:
    impulse_frames, impulse_real, impulse_imag = load_probe(impulse_dump)
    validate_impulse_basis(impulse_frames)
    data: dict[str, object] = {
        "impulse_manifest_sha256": sha256_file(impulse_dump / "manifest.json"),
        "candidates": {},
    }
    core_reference = None
    if core_dump is not None:
        core_frames, core_real, core_imag = load_core_probe(core_dump)
        validate_impulse_basis(core_frames)
        if core_frames.tobytes() != impulse_frames.tobytes():
            raise ValueError("real and complex impulse probes used different inputs")
        core_reference = core_real, core_imag
        data["core_manifest_sha256"] = sha256_file(core_dump / "manifest.json")
    datasets = {"impulse": (impulse_frames, impulse_real, impulse_imag)}
    if general_dump is not None:
        datasets["general"] = load_probe(general_dump)
    if stage_dump is not None:
        datasets["stage-3001"] = load_stage(stage_dump)
        data["stage_manifest_sha256"] = sha256_file(stage_dump / "manifest.json")
    for candidate in (
        "vdsp-radix4-dif",
        "radix4-dit",
        "radix4-dif",
        "split-radix-dit",
    ):
        candidate_results = {}
        if core_reference is not None:
            core = candidate_core(impulse_frames, candidate)
            candidate_results["complex-core"] = score(
                core.real, core.imag, core_reference[0], core_reference[1]
            )
        for dataset, (frames, reference_real, reference_imag) in datasets.items():
            candidate_real, candidate_imag = candidate_spectrum(frames, candidate)
            candidate_results[dataset] = score(
                candidate_real, candidate_imag, reference_real, reference_imag
            )
        data["candidates"][candidate] = candidate_results
    rendered = json.dumps(data, indent=2, sort_keys=True)
    if json_out is not None:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(rendered + "\n")
    print(rendered)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    make_parser = subparsers.add_parser("make-input")
    make_parser.add_argument("path", type=Path)
    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument("--impulse-dump", type=Path, required=True)
    compare_parser.add_argument("--core-dump", type=Path)
    compare_parser.add_argument("--general-dump", type=Path)
    compare_parser.add_argument("--stage-dump", type=Path)
    compare_parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    if args.command == "make-input":
        make_input(args.path)
    else:
        compare(args.impulse_dump, args.core_dump, args.general_dump,
                args.stage_dump, args.json_out)


if __name__ == "__main__":
    main()
