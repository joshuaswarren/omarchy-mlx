# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Build and register the whole-program Parakeet encoder bundle.

Reproduces the single-program encoder container from the archived hwx
(private evidence store) with the hwxv2 converter at a pinned commit,
verifies the result byte-for-byte against the hardware-proven container
sha, emits the bundle manifest (tile_shift 9, raw selector-addressed
bindings), and registers the bundle in the content-addressed compiled
cache so every later hit re-verifies the stored digests.

Provenance: receipts/2026-09-22-encoder-direct-exec — 13701 TDs, ONE
submit, rc=0 on jwm1 (T8103), hidden bit-exact vs the macOS gold capture.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path

_TOOLS = Path(__file__).resolve().parent.parent
for entry in (str(_TOOLS), str(_TOOLS / "coreml"), str(_TOOLS / "ane-export")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

CONVERTER_REPO_COMMIT = "6a963f975ae191b1b303ddf3cb5234c9e49ad837"
CONVERTER_REPO_PATH = "tools/hwxv2-to-anec.py"
HWX_SHA256 = "020428fca545648f5991155e9790238a5a2043a5100e4086afde2f10f51a18ef"
# The hardware-proven container.
ANEC_SHA256 = "13c744231524d440b0a774155343df9ade0bbcbc37edc4b1ccf9698e580d5453"

TILE_SHIFT = 9
TILE_UNIT = 1 << TILE_SHIFT
IN_CHANNELS, OUT_CHANNELS = 128, 640
IN_SHAPE = (1, 128, 3000, 1)   # dense input_features [3000, 128] fp16
OUT_SHAPE = (1, 640, 375, 1)   # encoder_hidden [375, 640] fp16

MANIFEST_VERSION = 4
BUNDLE_NAME = "parakeet-encoder-whole"


class BuildError(RuntimeError):
    """The bundle build refused; the reason is named."""


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ane_experiments_repo(explicit: str | None) -> Path:
    candidates = [Path(explicit)] if explicit else []
    candidates.append(Path.home() / "src" / "ane-linux-experiments")
    for candidate in candidates:
        if (candidate / ".git").exists():
            return candidate
    raise BuildError(
        "the ane-linux-experiments checkout is required to load the pinned "
        f"converter ({CONVERTER_REPO_COMMIT[:12]}); pass --ane-experiments"
    )


def _load_pinned_converter(repo: Path) -> str:
    """Return the converter source at the pinned commit, via git only.

    The working tree is never trusted: `git show` pins the exact bytes the
    proven container was built with.
    """
    result = subprocess.run(
        ["git", "-C", str(repo), "show",
         f"{CONVERTER_REPO_COMMIT}:{CONVERTER_REPO_PATH}"],
        capture_output=True, text=True, check=False,
    )
    if result.returncode != 0:
        raise BuildError(
            f"cannot read converter at {CONVERTER_REPO_COMMIT[:12]} from "
            f"{repo}: {result.stderr.strip()[:300]}"
        )
    return result.stdout


def _convert(repo: Path, hwx: bytes) -> bytes:
    """Run the pinned converter and apply the surface-tile coverage fix.

    The pinned converter sizes surface tile counts from the tile-DMA plane
    stride (N*C*plane bytes); this container's task stream addresses its
    surfaces through selector registers rather than NCHW placement, so the
    engine never touches more than the element bytes. The proven container
    sizes every surface at its element bytes rounded up to the 512-B tile
    unit; the same sizing here reproduces it byte-for-byte.
    """
    import importlib.util

    with tempfile.NamedTemporaryFile(
        suffix="_hwxv2_to_anec.py", mode="w", delete=False
    ) as handle:
        handle.write(_load_pinned_converter(repo))
        converter_path = handle.name
    spec = importlib.util.spec_from_file_location("hwxv2_to_anec", converter_path)
    converter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(converter)

    container = bytearray(converter.convert_hwx(
        hwx, IN_CHANNELS, OUT_CHANNELS,
        in_shape=IN_SHAPE, out_shape=OUT_SHAPE,
    ))

    def tile_count(data_bytes: int) -> int:
        return max(1, -(-data_bytes // TILE_UNIT))

    # Both channels of a direction carry the same engine window: the
    # hidden/features element bytes, rounded to the 512-B tile unit
    # (the proven container's tiles[4..7] = 938, 938, 1500, 1500).
    fixes = {
        4: tile_count(480000),   # dst0/dst1 window (encoder_hidden)
        5: tile_count(480000),   # dst1 output mask
        6: tile_count(768000),   # src0 (staging window, features-sized)
        7: tile_count(768000),   # src1 input_features
    }
    tiles = struct.unpack_from("<32I", container, 40)
    for channel, count in fixes.items():
        if tiles[channel] != count:
            struct.pack_into("<I", container, 40 + channel * 4, count)
    return bytes(container)


def manifest_payloads(anec_sha: str, byte_size: int) -> list[dict]:
    return [{
        "role": "anec",
        "path": "program-0.anec",
        "sha256": anec_sha,
        "byte_size": byte_size,
    }]


def _build_manifest(container: bytes, hwx_sha: str, source_commit: str) -> dict:
    (_, _, td_count, _, _,
     source_count, destination_count) = struct.unpack_from("<QIIQQII", container, 0)
    tiles = struct.unpack_from("<32I", container, 40)

    def alloc(channel: int) -> int:
        return tiles[channel] * TILE_UNIT

    if source_count != 2 or destination_count != 2:
        raise BuildError(
            f"container declares {source_count} sources / "
            f"{destination_count} destinations, want 2/2"
        )
    hidden_bytes = 375 * 640 * 2
    mask_bytes = 375 * 2
    staged_mask_bytes = 3000 * 2
    features_bytes = 768000
    anec_sha = _sha256_bytes(container)
    # release_asset.model_sha256 is the canonical payload-collection
    # identity: sha256 over the compact, key-sorted JSON of the sorted
    # payload records (the same domain bundle.cpp checks).
    payload_records = sorted(
        manifest_payloads(anec_sha, len(container)),
        key=lambda record: record["path"],
    )
    collection_identity = _sha256_bytes(
        json.dumps(payload_records, sort_keys=True,
                   separators=(",", ":")).encode()
    )

    def raw(tensor: str, channel: int, logical: int) -> dict:
        return {
            "tensor": tensor,
            "channel": channel,
            "dtype": "float16",
            "raw": True,
            "logical_bytes": logical,
            "allocation_bytes": alloc(channel),
        }

    return {
        "manifest_version": MANIFEST_VERSION,
        "tile_shift": TILE_SHIFT,
        "name": BUNDLE_NAME,
        "graph_hash": hwx_sha,
        "task_descriptors": td_count,
        "inputs": [
            {"name": "attention_mask", "index": 0, "dtype": "float16",
             "shape": [3000, 1, 1, 1], "byte_size": staged_mask_bytes,
             "stride": alloc(6)},
            {"name": "input_features", "index": 1, "dtype": "float16",
             "shape": [3000, 128, 1, 1], "byte_size": features_bytes,
             "stride": alloc(7)},
        ],
        "outputs": [
            {"name": "encoder_hidden", "index": 0, "dtype": "float16",
             "shape": [375, 640, 1, 1], "byte_size": hidden_bytes,
             "stride": alloc(4)},
            {"name": "output_mask", "index": 1, "dtype": "float16",
             "shape": [375, 1, 1, 1], "byte_size": mask_bytes,
             "stride": alloc(5)},
        ],
        "logical_results": [
            {"name": "encoder_hidden", "dtype": "float16",
             "shape": [375, 640, 1, 1], "tensor": "encoder_hidden",
             "element_offset": 0, "element_count": 375 * 640,
             "conversion": "identity"},
            {"name": "output_mask", "dtype": "float16",
             "shape": [375, 1, 1, 1], "tensor": "output_mask",
             "element_offset": 0, "element_count": 375,
             "conversion": "identity"},
        ],
        "state": [],
        "intermediates": [],
        "programs": [{
            "payload": "program-0.anec",
            "operation": "whole-encoder",
            "encoder": "apple-whole-encoder-hwxv2",
            "task_descriptors": td_count,
            "scratch_bytes": tiles[3] * TILE_UNIT,
            "inputs": [
                raw("attention_mask", 6, staged_mask_bytes),
                raw("input_features", 7, features_bytes),
            ],
            "outputs": [
                raw("encoder_hidden", 4, hidden_bytes),
                raw("output_mask", 5, mask_bytes),
            ],
        }],
        "dispatch_plan": [0],
        "payloads": manifest_payloads(anec_sha, len(container)),
        "compiler": {
            "host_build": f"converter pinned at {CONVERTER_REPO_COMMIT[:12]}",
            "toolchain": f"hwxv2-to-anec {CONVERTER_REPO_COMMIT}",
            "target": "h13",
        },
        "driver_abi_major": 1,
        "provenance": {
            "source_repo": "joshuaswarren/mlx-omarchy",
            "source_commit": source_commit,
            "exported_at": time.strftime("%Y-%m-%d"),
        },
        "release_asset": {
            "model": BUNDLE_NAME,
            "model_sha256": collection_identity,
        },
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hwx", required=True, type=Path,
                        help="archived encoder hwx (private evidence store)")
    parser.add_argument("--ane-experiments", default=None, type=Path,
                        help="ane-linux-experiments checkout (default: ~/src)")
    parser.add_argument("--cache-root", type=Path, default=None,
                        help="compiled-cache root (default: the model cache)")
    parser.add_argument("--out", type=Path, default=None,
                        help="also copy the bundle files to this directory")
    args = parser.parse_args(argv[1:])

    hwx_sha = _sha256_file(args.hwx)
    if hwx_sha != HWX_SHA256:
        raise BuildError(
            f"{args.hwx} sha256 {hwx_sha} is not the archived encoder hwx "
            f"{HWX_SHA256}; refusing to convert an unknown container"
        )
    container = _convert(_ane_experiments_repo(
        str(args.ane_experiments) if args.ane_experiments else None
    ), args.hwx.read_bytes())
    actual = _sha256_bytes(container)
    if actual != ANEC_SHA256:
        raise BuildError(
            f"converted container sha256 {actual} does not match the "
            f"hardware-proven container {ANEC_SHA256}; the converter "
            f"pipeline drifted"
        )
    td_count = struct.unpack_from("<I", container, 12)[0]
    print(f"container verified: {actual} ({len(container)} bytes, {td_count} TDs)")

    repo_commit = subprocess.run(
        ["git", "-C", str(Path(__file__).resolve().parents[3]),
         "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    manifest = _build_manifest(container, hwx_sha, repo_commit)

    from coreml.compiled_cache import (
        ArtifactDigest,
        CacheKey,
        CompiledCache,
        CompilerIdentity,
        default_cache_root,
    )

    key = CacheKey.create(
        model_artifacts=(ArtifactDigest(
            path="encoder.hwx", sha256=hwx_sha,
            size=args.hwx.stat().st_size,
        ),),
        selected_function="encoder",
        static_input_shapes={
            "attention_mask": (3000, 1, 1, 1),
            "input_features": (3000, 128, 1, 1),
        },
        compiler=CompilerIdentity(
            name="hwxv2-to-anec",
            version=CONVERTER_REPO_COMMIT[:12],
            commit=CONVERTER_REPO_COMMIT,
            target="h13",
            package_schema="hwxv2",
        ),
        operations=("whole-encoder",),
        bundle_schema=MANIFEST_VERSION,
        driver_abi_major=1,
        firmware_identity="T8103-H13",
        frontend_version="mlx-omarchy-whole-encoder-4",
    )
    cache = CompiledCache(args.cache_root) if args.cache_root else CompiledCache(
        default_cache_root()
    )

    def produce(bundle: Path) -> None:
        bundle.mkdir(parents=True)
        (bundle / "program-0.anec").write_bytes(container)
        (bundle / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )

    result = cache.get_or_create(key, produce)
    print(f"bundle {'cache-hit' if result.hit else 'published'}: {result.bundle}")
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        for name in ("program-0.anec", "manifest.json"):
            (args.out / name).write_bytes((result.bundle / name).read_bytes())
        print(f"copied to {args.out}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv))
    except BuildError as error:
        print(f"build-whole-encoder-bundle: {error}", file=sys.stderr)
        raise SystemExit(1)
