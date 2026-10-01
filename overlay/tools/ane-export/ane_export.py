#!/usr/bin/env python3
"""Export a small elementwise/matmul region as an ANE bundle for mlx-omarchy.

Runs on macOS (Apple silicon) with Xcode and ANECompiler installed. The tool:

1. Emits a hand-authored MIL ``program(1.3)`` text in the coremlc dialect
   (runtime input tensor + fp16 const riding ``weights.bin`` via BLOBFILE)
   from a JSON descriptor - no MLX or coremltools dependency.
2. Emits ``weights.bin``: 0x40-byte file header, 0x40-byte 0xDEADBEEF blob
   record, raw fp16 payload at offset 0x80 (format per
   ``receipts/2026-08-31-mil-oneop-proof.md``).
3. Compiles through the private ``ANECCompile`` entry point via the
   companion ``ane-compile-hwx`` tool, producing ``model.hwx``.
4. Converts the HWX to the Linux ``.anec`` container with the canonical
   ``hwxv2-to-anec.py`` from ``joshuaswarren/ane-linux-experiments``, passing
   ``--weights`` so the weight payload offset comes out of the blob record.
5. Packages a ``bundle/`` directory (manifest.json + payloads) that passes
   ``mlx-omarchy-info --check-bundle`` on Linux.

Usage:
    python3 ane_export.py DESCRIPTOR.json --out-dir EXPORT_DIR

Descriptor fields:
    op           "add" | "mul" | "matmul"          (required)
    input_shape  list of positive ints             (required)
    weight_shape list of positive ints, matmul only
    dtype        "fp16"                            (default; only fp16 today)
    const_value  float, fp16 const fill value      (default 0.25)
    name         bundle name                       (default derived)

The scratch layout this tool expects (see the mlx-omarchy exporter README):
    ane-compile-hwx              built from ane-compile-hwx.mm (Xcode clang++)
    hwxv2-to-anec.py             HWX -> ANEC converter, copied unmodified from
                                 ane-linux-experiments (52a3211 or later)
"""

import argparse
import hashlib
import json
import math
import re
import struct
import subprocess
import sys
import datetime
from pathlib import Path

TILE_ALIGNMENT = 0x4000
FP16_SIZES = {"fp16": 2}
WEIGHTS_HEADER_VERSION = 1
WEIGHTS_ENTRY_COUNT = 2
BLOB_MAGIC = 0xDEADBEEF
BLOB_VERSION = 1
BLOB_DATA_OFFSET = 128  # 0x40 header + 0x40 blob record
CONVERTER = "hwxv2-to-anec.py"
# ane-linux-experiments 52a3211 made the converter read the weight payload
# offset out of the blob record and derive nchw geometry from the task's
# tile-DMA counts. An earlier copy takes no --weights, emits a header-prefixed
# weight stream and convention geometry, and the export is wrong but silent -
# so refuse it by the two names that fix introduced.
CONVERTER_MARKERS = ("--weights", "derive_strides")


def die(message: str) -> "NoReturn":  # type: ignore[valid-type]
    print(f"ane_export: error: {message}", file=sys.stderr)
    sys.exit(1)


def log(message: str) -> None:
    print(f"ane_export: {message}")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def product(shape: list) -> int:
    result = 1
    for dim in shape:
        result *= int(dim)
    return result


def tile_stride(byte_size: int) -> int:
    if byte_size <= TILE_ALIGNMENT:
        return TILE_ALIGNMENT
    return int(math.ceil(byte_size / TILE_ALIGNMENT)) * TILE_ALIGNMENT


def fp16_bytes(values: list) -> bytes:
    return struct.pack(f"<{len(values)}e", *values)


def build_weights(payload: bytes) -> bytes:
    """weights.bin: 0x40 file header + 0x40 blob record + raw payload."""
    header = struct.pack("<II56x", WEIGHTS_HEADER_VERSION, WEIGHTS_ENTRY_COUNT)
    record = struct.pack(
        "<IIQQQ32x",
        BLOB_MAGIC,
        BLOB_VERSION,
        len(payload),
        0,
        BLOB_DATA_OFFSET,
    )
    return header + record + payload


def build_mil(descriptor: dict, in_shape: str, out_shape: str,
              weight_shape: str) -> str:
    """MIL program(1.3) text in the coremlc 3520.5.1 dialect.

    buildInfo is replicated from the retained coremlc capture; the program is
    authored here. Consts ride weights.bin via BLOBFILE - the compiler rejects
    inline fp16 const tensors.
    """
    blob = (
        f"BLOBFILE(path = string(\"@model_path/weights.bin\"),"
        f" offset = uint64(64))"
    )
    body = {
        "add": "add(x = t1, y = t0)",
        "mul": "mul(x = t1, y = t0)",
        "matmul": "matmul(x = t1, y = t0, transpose_x = false,"
                  " transpose_y = false)",
    }[descriptor["op"]]
    weight_type = weight_shape if weight_shape else in_shape
    return (
        "program(1.3)\n"
        "[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3520.4.1\"},"
        " {\"coremlc-version\", \"3520.5.1\"}})]\n"
        "{\n"
        f"    func main<ios18>(tensor<fp16, {in_shape}> t1) {{\n"
        f"        tensor<fp16, {weight_type}> t0 = const()[name = string(\"t0\"),"
        f" val = tensor<fp16, {weight_type}>({blob})];\n"
        f"        tensor<fp16, {out_shape}> t2 = {body}[name = string(\"t2\")];\n"
        "    } -> (t2);\n"
        "}\n"
    )


def run_tool(argv: list, cwd: Path, stage: str) -> str:
    result = subprocess.run(
        argv, cwd=cwd, text=True, capture_output=True, timeout=180
    )
    transcript = (
        f"$ {' '.join(argv)}\n{result.stdout}{result.stderr}"
        f"EXIT={result.returncode}\n"
    )
    if result.returncode != 0:
        (Path.cwd() / "export.log").open("a").write(transcript)
        die(f"{stage} failed (exit {result.returncode}); see export.log")
    return transcript


def probe(command: list, fallback: str = "") -> str:
    try:
        result = subprocess.run(
            command, text=True, capture_output=True, timeout=30
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        pass
    if not fallback:
        die(f"could not probe {' '.join(command)}; pass it explicitly")
    return fallback


def check_converter(path: Path) -> None:
    """Refuse a converter that predates the export fix."""
    if not path.exists():
        die(f"missing {path}; copy tools/{CONVERTER} from "
            "ane-linux-experiments (52a3211 or later)")
    source = path.read_text(errors="replace")
    missing = [name for name in CONVERTER_MARKERS if name not in source]
    if missing:
        die(f"{path} predates the export fix (no {', '.join(missing)}); "
            "copy it again from ane-linux-experiments 52a3211 or later")


def compile_region(tools_dir: Path, capture: Path, hwx_output: Path,
                   anec_path: Path, weights_path: Path, in_elems: int,
                   out_elems: int) -> dict:
    """Compile + convert. Returns converter facts (td-count, workspace)."""
    transcript = run_tool(
        ["./ane-compile-hwx", str(capture), str(hwx_output), "h13"],
        cwd=tools_dir, stage="ANECCompile",
    )
    convert = run_tool(
        [sys.executable, CONVERTER,
         str(hwx_output / "model.hwx"), str(anec_path),
         str(in_elems), str(out_elems), "--weights", str(weights_path)],
        cwd=tools_dir, stage="hwxv2-to-anec",
    )
    (Path.cwd() / "export.log").open("a").write(transcript + convert)
    td = re.search(r"td-count=(\d+)", convert)
    workspace = re.search(r"workspace=(0x[0-9a-fA-F]+)", convert)
    if not td or not workspace:
        die(f"cannot parse converter output: {convert.strip()}")
    return {
        "task_descriptors": int(td.group(1)),
        "workspace_bytes": int(workspace.group(1), 16),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export a small region as an ANE bundle (run on macOS)."
    )
    parser.add_argument("descriptor", type=Path, help="JSON descriptor file")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--tools-dir", type=Path, default=None,
                        help="dir with ane-compile-hwx + converter "
                             "(default: this script's directory)")
    parser.add_argument("--source-repo", default="joshuaswarren/omarchy-mlx")
    parser.add_argument("--source-commit", default="",
                        help="40-hex commit of the source repo")
    parser.add_argument("--firmware-min", default="26.0")
    parser.add_argument("--firmware-max", default="26.6")
    parser.add_argument("--macos-build", default="",
                        help="override probed sw_vers -buildVersion")
    parser.add_argument("--anecompiler", default="",
                        help="override probed compiler identity string")
    args = parser.parse_args()

    descriptor = json.loads(args.descriptor.read_text())
    op = descriptor.get("op")
    if op not in ("add", "mul", "matmul"):
        die("descriptor 'op' must be add, mul, or matmul")
    in_shape = [int(d) for d in descriptor["input_shape"]]
    if not in_shape or any(d <= 0 for d in in_shape):
        die("descriptor 'input_shape' must list positive integers")
    weight_shape = [int(d) for d in descriptor.get("weight_shape", [])]
    if op == "matmul" and len(weight_shape) != 2:
        die("matmul descriptor needs 'weight_shape' = [m, n]")
    dtype = descriptor.get("dtype", "fp16")
    if dtype != "fp16":
        die(f"unsupported dtype '{dtype}' (only fp16 today)")
    const_value = float(descriptor.get("const_value", 0.25))

    if op == "matmul":
        out_shape = [in_shape[0], weight_shape[1]]
        weight_elems = product(weight_shape)
    else:
        out_shape = list(in_shape)
        weight_elems = product(in_shape)
    in_elems = product(in_shape)
    out_elems = product(out_shape)

    name = descriptor.get("name") or (
        f"ane-{op}-fp16-{'x'.join(map(str, in_shape))}"
        + (f"-{'x'.join(map(str, weight_shape))}" if op == "matmul" else "")
    )

    tools_dir = (args.tools_dir or Path(__file__).resolve().parent).resolve()
    if not (tools_dir / "ane-compile-hwx").exists():
        die(f"missing {tools_dir}/ane-compile-hwx (build ane-compile-hwx.mm)")
    check_converter(tools_dir / CONVERTER)

    out_dir = args.out_dir.resolve()
    bundle_dir = out_dir / "bundle"
    capture_dir = out_dir / "capture"
    hwx_dir = out_dir / "hwx-output"
    for directory in (bundle_dir, capture_dir, hwx_dir):
        directory.mkdir(parents=True, exist_ok=True)

    # 1-2: MIL text + weights.bin.
    in_text = "[" + ", ".join(map(str, in_shape)) + "]"
    out_text = "[" + ", ".join(map(str, out_shape)) + "]"
    weight_text = ""
    if op == "matmul":
        weight_text = "[" + ", ".join(map(str, weight_shape)) + "]"
    mil = build_mil(descriptor, in_text, out_text, weight_text)
    mil_path = capture_dir / "model.mil"
    mil_path.write_text(mil)

    weights_payload = fp16_bytes([const_value] * weight_elems)
    weights_bin = build_weights(weights_payload)
    capture_weights = capture_dir / "weights.bin"
    capture_weights.write_bytes(weights_bin)
    log(f"model.mil={len(mil)}B weights.bin={len(weights_bin)}B")

    # 3-4: compile + convert.
    anec_path = bundle_dir / "model.anec"
    facts = compile_region(tools_dir, capture_dir, hwx_dir, anec_path,
                           capture_weights, in_elems, out_elems)
    bundle_weights = bundle_dir / "weights.bin"
    bundle_weights.write_bytes(weights_bin)

    # 5: manifest.
    elem_size = FP16_SIZES[dtype]
    ws_bytes = max(TILE_ALIGNMENT, facts["workspace_bytes"])
    commit = args.source_commit
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        die("--source-commit must be a 40-hex commit hash")

    macos_build = args.macos_build or probe(["sw_vers", "-buildVersion"])
    if args.anecompiler:
        anecompiler = args.anecompiler
    else:
        product_version = probe(["sw_vers", "-productVersion"])
        ane_version = probe(
            ["plutil", "-extract", "CFBundleShortVersionString", "raw",
             "/System/Library/PrivateFrameworks/ANECompiler.framework/"
             "Resources/Info.plist"]
        )
        coreml_version = probe(["xcrun", "coremlcompiler", "version"])
        anecompiler = (
            f"ANECompiler {ane_version} (macOS {product_version},"
            f" coremlcompiler {coreml_version})"
        )

    anec_sha = sha256_file(anec_path)
    manifest = {
        "manifest_version": 1,
        "name": name,
        "graph_hash": sha256_file(mil_path),
        "task_descriptors": facts["task_descriptors"],
        "inputs": [{
            "name": "t1", "index": 0, "dtype": "float16",
            "shape": in_shape,
            "byte_size": in_elems * elem_size,
            "stride": tile_stride(in_elems * elem_size),
        }],
        "outputs": [{
            "name": "t2", "index": 1, "dtype": "float16",
            "shape": out_shape,
            "byte_size": out_elems * elem_size,
            "stride": tile_stride(out_elems * elem_size),
        }],
        "state": [],
        "workspace": [{
            "name": "workspace", "index": 0, "dtype": "uint8",
            "shape": [ws_bytes], "byte_size": ws_bytes, "stride": ws_bytes,
        }],
        "payloads": [
            {"role": "anec", "path": "model.anec", "sha256": anec_sha,
             "byte_size": anec_path.stat().st_size},
            {"role": "weights", "path": "weights.bin",
             "sha256": sha256_file(bundle_weights),
             "byte_size": bundle_weights.stat().st_size},
        ],
        "compiler": {"macos_build": macos_build, "anecompiler": anecompiler},
        "firmware": {"min": args.firmware_min, "max": args.firmware_max},
        "provenance": {
            "source_repo": args.source_repo,
            "source_commit": commit,
            "exported_at": datetime.datetime.now(datetime.timezone.utc)
            .strftime("%Y-%m-%d"),
        },
        "release_asset": {"model": name, "model_sha256": anec_sha},
    }
    (bundle_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    log(f"graph_hash={manifest['graph_hash']}")
    log(f"task_descriptors={manifest['task_descriptors']} "
        f"workspace={ws_bytes:#x}")
    log(f"bundle ready: {bundle_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
