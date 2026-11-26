#!/usr/bin/env python3
# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Compare every captured native Core ML joint tensor with MLX Vulkan."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

from coreml.reference import ReferenceLock
from coreml.vulkan_joint import run_joint

_CAPTURE_SCHEMA = "parakeet-tdt-tensors/1"
_KEYS = {
    "joint_encoder_frame": ((1, 640), np.dtype("<f4")),
    "joint_decoder_state": ((1, 640), np.dtype("<f4")),
    "joint_token_logits": ((1, 8193), np.dtype("<f4")),
    "joint_duration_logits": ((1, 5), np.dtype("<f4")),
}
_EXPECTED_CAPTURE_COUNTS = {
    "manifest_entries": 170,
    "decisions": 146,
    "traces": 16,
    "npy": 157,
}


class ComparisonError(ValueError):
    """The native capture does not satisfy the pinned comparison contract."""


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
def _load_manifest(
    capture_dir: Path,
) -> tuple[dict[str, str], dict[str, bytes], str]:
    manifest_bytes = (capture_dir / "manifest.sha256").read_bytes()
    entries = {}
    for line in manifest_bytes.decode("utf-8").splitlines():
        digest, separator, name = line.partition("  ")
        relative = PurePosixPath(name)
        if (
            not separator
            or len(digest) != 64
            or relative.is_absolute()
            or not relative.parts
            or ".." in relative.parts
            or name in entries
        ):
            raise ComparisonError("capture manifest has an invalid entry")
        entries[name] = digest
    if len(entries) != _EXPECTED_CAPTURE_COUNTS["manifest_entries"]:
        raise ComparisonError(
            f"capture manifest has {len(entries)} entries, "
            f"expected {_EXPECTED_CAPTURE_COUNTS['manifest_entries']}"
        )
    snapshots = {}
    for name, expected in entries.items():
        path = capture_dir.joinpath(*PurePosixPath(name).parts)
        if path.is_symlink() or not path.is_file():
            raise ComparisonError(f"capture manifest entry is not a regular file: {name}")
        data = path.read_bytes()
        if _sha256(data) != expected:
            raise ComparisonError(f"capture manifest hash differs: {name}")
        snapshots[name] = data
    return entries, snapshots, _sha256(manifest_bytes)


def _load_array(
    snapshots: dict[str, bytes], manifest: dict[str, str], name: str, key: str
) -> np.ndarray:
    if name not in manifest:
        raise ComparisonError(f"capture tensor is not authenticated by the manifest: {name}")
    expected_shape, expected_dtype = _KEYS[key]
    value = np.load(io.BytesIO(snapshots[name]), allow_pickle=False)
    if value.shape != expected_shape or value.dtype != expected_dtype:
        raise ComparisonError(
            f"{key} must be {expected_dtype}{expected_shape}, got {value.dtype}{value.shape}"
        )
    return value


def _tensor_identity(value: np.ndarray, npy_sha256: str) -> dict[str, Any]:
    return {
        "shape": list(value.shape),
        "dtype": value.dtype.str,
        "npy_sha256": npy_sha256,
        "raw_bitpattern_sha256": _sha256(value.tobytes()),
    }


def _metrics(actual: np.ndarray, golden: np.ndarray) -> dict[str, Any]:
    actual64 = actual.astype(np.float64)
    golden64 = golden.astype(np.float64)
    difference = actual64 - golden64
    absolute = np.abs(difference)
    denominator = float(np.linalg.norm(golden64.ravel()))
    relative_l2 = float(np.linalg.norm(difference.ravel()) / denominator)
    actual_bits = actual.view(np.uint32)
    golden_bits = golden.view(np.uint32)
    return {
        "actual_shape": list(actual.shape),
        "golden_shape": list(golden.shape),
        "actual_dtype": actual.dtype.str,
        "golden_dtype": golden.dtype.str,
        "max_absolute_error": float(absolute.max()),
        "mean_absolute_error": float(absolute.mean()),
        "relative_l2_error": relative_l2,
        "actual_nan_count": int(np.isnan(actual).sum()),
        "golden_nan_count": int(np.isnan(golden).sum()),
        "actual_inf_count": int(np.isinf(actual).sum()),
        "golden_inf_count": int(np.isinf(golden).sum()),
        "bit_exact": bool(actual.tobytes() == golden.tobytes()),
        "different_bitpatterns": int(np.count_nonzero(actual_bits != golden_bits)),
        "element_count": int(actual.size),
        "actual_raw_bitpattern_sha256": _sha256(actual.tobytes()),
        "golden_raw_bitpattern_sha256": _sha256(golden.tobytes()),
    }


def compare(capture_dir: Path, package_path: Path) -> dict[str, Any]:
    import mlx.core as mx

    capture_dir = capture_dir.resolve()
    package_path = package_path.resolve()
    manifest, snapshots, manifest_digest = _load_manifest(capture_dir)
    parent_receipt = json.loads((capture_dir.parent / "receipt.json").read_text())
    if parent_receipt.get("capture_sha256", {}).get("manifest.sha256") != manifest_digest:
        raise ComparisonError("parent receipt does not authenticate the capture manifest")
    lock = ReferenceLock.load()
    if parent_receipt.get("model_revision") != lock.model_revision:
        raise ComparisonError("capture model revision differs from the pinned reference")
    if (
        parent_receipt.get("capture") != "ane"
        or parent_receipt.get("compute_units_requested") != "ane"
    ):
        raise ComparisonError("capture did not request Core ML ANE compute units")
    validation_bytes = (capture_dir / "validation.json").read_bytes()
    if parent_receipt.get("capture_sha256", {}).get("validation.json") != _sha256(validation_bytes):
        raise ComparisonError("parent receipt does not authenticate capture validation")
    validation = json.loads(validation_bytes)
    if not validation.get("manifest_hashes_valid") or any(
        validation.get(key) != expected
        for key, expected in _EXPECTED_CAPTURE_COUNTS.items()
    ):
        raise ComparisonError("capture validation counts differ")
    if any(parent_receipt.get("counts", {}).get(key) != expected for key, expected in _EXPECTED_CAPTURE_COUNTS.items()):
        raise ComparisonError("parent receipt capture counts differ")
    tensor_index = json.loads(snapshots["tdt_tensors.json"])
    traces = tensor_index.get("traces")
    if tensor_index.get("schema") != _CAPTURE_SCHEMA or tensor_index.get("limit") != 16:
        raise ComparisonError("capture tensor index contract differs")
    if not isinstance(traces, list) or [trace.get("index") for trace in traces] != list(range(16)):
        raise ComparisonError("capture must contain exactly traces 0 through 15")

    results = []
    token_actual_all = []
    token_golden_all = []
    duration_actual_all = []
    duration_golden_all = []
    token_decisions = 0
    duration_decisions = 0
    for trace in traces:
        paths = trace.get("tensor_paths")
        if not isinstance(paths, dict) or not _KEYS.keys() <= paths.keys():
            raise ComparisonError(f"trace {trace['index']} lacks a complete joint tensor set")
        arrays = {
            key: _load_array(snapshots, manifest, paths[key], key)
            for key in _KEYS
        }
        output = run_joint(
            mx.array(arrays["joint_encoder_frame"]),
            mx.array(arrays["joint_decoder_state"]),
            package_path=package_path,
        )
        mx.eval(output.token_logits, output.duration_logits)
        actual_token = np.asarray(output.token_logits)
        actual_duration = np.asarray(output.duration_logits)
        for key, value in (
            ("joint_token_logits", actual_token),
            ("joint_duration_logits", actual_duration),
        ):
            expected_shape, expected_dtype = _KEYS[key]
            if value.shape != expected_shape or value.dtype != expected_dtype:
                raise ComparisonError(
                    f"runtime {key} must be {expected_dtype}{expected_shape}, "
                    f"got {value.dtype}{value.shape}"
                )
        golden_token = arrays["joint_token_logits"]
        golden_duration = arrays["joint_duration_logits"]
        actual_token_id = int(actual_token.argmax())
        actual_duration_index = int(actual_duration.argmax())
        token_match = actual_token_id == trace.get("token_id")
        duration_match = actual_duration_index == trace.get("duration_index")
        token_decisions += int(token_match)
        duration_decisions += int(duration_match)
        results.append(
            {
                "index": trace["index"],
                "inputs": {
                    key: _tensor_identity(arrays[key], manifest[paths[key]])
                    for key in ("joint_encoder_frame", "joint_decoder_state")
                },
                "token_logits": _metrics(actual_token, golden_token),
                "duration_logits": _metrics(actual_duration, golden_duration),
                "decision": {
                    "actual_token_id": actual_token_id,
                    "golden_token_id": trace["token_id"],
                    "token_match": token_match,
                    "actual_duration_index": actual_duration_index,
                    "golden_duration_index": trace["duration_index"],
                    "duration_match": duration_match,
                },
            }
        )
        token_actual_all.append(actual_token)
        token_golden_all.append(golden_token)
        duration_actual_all.append(actual_duration)
        duration_golden_all.append(golden_duration)

    token_aggregate = _metrics(
        np.concatenate(token_actual_all), np.concatenate(token_golden_all)
    )
    duration_aggregate = _metrics(
        np.concatenate(duration_actual_all), np.concatenate(duration_golden_all)
    )
    bit_exact = token_aggregate["bit_exact"] and duration_aggregate["bit_exact"]
    decision_exact = token_decisions == len(traces) and duration_decisions == len(traces)
    return {
        "schema": "mlx-omarchy.parakeet-vulkan-joint-native-comparison/1",
        "capture": {
            "path": str(capture_dir),
            "manifest_sha256": manifest_digest,
            "manifest_entries_verified": len(manifest),
            "validation_counts": _EXPECTED_CAPTURE_COUNTS,
            "model_revision": lock.model_revision,
            "compute_units_requested": "ane",
            "backend_claim": parent_receipt["backend_claim"],
        },
        "package_path": str(package_path),
        "runtime": {
            "version": mx.__version__,
            "device": mx.device_info().get("device_name"),
            "architecture": mx.device_info().get("architecture"),
        },
        "implementation": {
            "callable": "coreml.vulkan_joint.run_joint",
            "joint_source": "overlay/tools/coreml/vulkan_joint.py",
            "joint_source_sha256": _sha256(Path(run_joint.__code__.co_filename).read_bytes()),
            "comparison_source": "overlay/tools/coreml/compare_vulkan_joint_golden.py",
            "comparison_source_sha256": _sha256(Path(__file__).read_bytes()),
        },
        "traces_compared": len(traces),
        "aggregate": {
            "token_logits": token_aggregate,
            "duration_logits": duration_aggregate,
            "token_decisions_matching": token_decisions,
            "duration_decisions_matching": duration_decisions,
        },
        "traces": results,
        "parity_verdict": {
            "full_tensor_bit_exact": bit_exact,
            "captured_joint_decisions_exact": decision_exact,
            "statement": (
                "All captured joint token and duration decisions match; full output tensors are not bit-exact."
                if decision_exact and not bit_exact
                else "Captured joint outputs and decisions are bit-exact."
                if decision_exact
                else "One or more captured joint decisions differ."
            ),
            "tolerance_policy": "No numerical acceptance tolerance was introduced or applied; complete tensor errors and bit patterns are reported.",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = compare(args.capture_dir, args.package)
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(text, end="")
    else:
        args.output.write_text(text, encoding="utf-8")
    return 0 if result["parity_verdict"]["captured_joint_decisions_exact"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
