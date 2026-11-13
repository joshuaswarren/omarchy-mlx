# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Build and register the lane-3 (H16, driver ABI 3) whole Parakeet encoder.

The H16 counterpart of build_whole_encoder_bundle.py. Its input is the output
of the one-time macOS harvest (ane-harvest.sh):

  hwx-compiled/encoder-ane/<target>/model.hwx
      Apple ANECCompile for the M4 generation of the CoreML encoder MIL
      rewritten by h16_encoder_mil.py: the int32/bool mask arithmetic
      ANECCompile refuses becomes fp16 arithmetic on an fp16 attention_mask
      input, and encoder_hidden is the only output (the ABI 3 driver
      accepts one output per program).
  hwx-compiled/encoder-ane/target                 the target that compiled
  capture/ane/encoder_input_mask.npy              golden mask input
  capture/ane/encoder_input_features.npy          golden features input
  capture/ane/encoder_hidden.npy                  golden output
  meta/parakeet-reference.lock                    lock the capture used

Lane 3 runs the HWX itself (no ANEC conversion). The HWX is validated with
the rules of the kernel tree's reference section builder h16-sections.py,
loaded from an aurora checkout and pinned by SHA-256 (the same pattern as the
pinned hwxv2 converter); its sha256 is measured here and pinned in the
schema-5 manifest. The bundle is registered in the content-addressed
compiled cache, so every later hit re-verifies the stored digests.

--check runs the bundle once on this machine's ABI 3 node through
mlx-omarchy-ane-worker --device h16 and compares encoder_hidden with the
golden under the lock's numerical contract.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path

_TOOLS = Path(__file__).resolve().parent.parent
for entry in (str(_TOOLS), str(_TOOLS / "coreml")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

# Reference section builder: aurora-silicon/linux (Apple-compiled HWX:
# split __KERN_n kernel data and __TEXT,__const constants)
# tools/testing/selftests/drivers/ane/h16-sections.py (GPL-2.0-only; loaded
# at build time, never copied into this repository).
H16_SECTIONS_PATH = "tools/testing/selftests/drivers/ane/h16-sections.py"
H16_SECTIONS_SHA256 = "bd45fc0fc5ced98dac177d51d805c9ef01343e9703fc37c3a2cc20e60d6e5836"
# ane-linux-experiments commit of tools/ane-compile-hwx.mm, the ANECCompile
# wrapper the harvest builds (ane-harvest.sh ALE_COMMIT).
COMPILER_WRAPPER_COMMIT = "0a769e44f282ccb9101efc01be7edf51cf849687"

HWX_SUBTYPE_H16G = 7
MANIFEST_VERSION = 5
DRIVER_ABI_MAJOR = 3
BUNDLE_NAME = "parakeet-encoder-whole-h16"
PAYLOAD = "program-0.hwx"
# The encoder MIL rewritten by h16_encoder_mil.py (fp16 mask arithmetic,
# encoder_hidden only), compiled by ANECCompile for h16g.
HARVEST_PROGRAM = "encoder-ane"
# The inputs of the M1/M2 whole-encoder container: fp16 attention_mask [3000]
# (1 valid frame, 0 padding) and dense fp16 input_features [3000, 128] ->
# encoder_hidden [375, 640].
MASK_SHAPE = (3000,)
MASK_BYTES = MASK_SHAPE[0] * 2
FEATURES_SHAPE = (3000, 128)
HIDDEN_SHAPE = (375, 640)
FEATURES_BYTES = FEATURES_SHAPE[0] * FEATURES_SHAPE[1] * 2
HIDDEN_BYTES = HIDDEN_SHAPE[0] * HIDDEN_SHAPE[1] * 2


class BuildError(RuntimeError):
    """The bundle build or check refused; the reason is named."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_reference_sections(path: Path):
    """Import h16-sections.py after checking its pinned digest."""
    path = Path(path)
    if path.is_dir():
        path = path / H16_SECTIONS_PATH
    if not path.is_file():
        raise BuildError(
            f"reference section builder not found at {path}; pass "
            "--kernel-tree with an aurora checkout"
        )
    actual = _sha256_file(path)
    if actual != H16_SECTIONS_SHA256:
        raise BuildError(
            f"{path} sha256 {actual} is not the pinned h16-sections.py "
            f"{H16_SECTIONS_SHA256}"
        )
    spec = importlib.util.spec_from_file_location("h16_sections_reference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def harvest_layout(harvest: Path) -> dict[str, Path | str]:
    """Resolve the harvest files; every one must exist."""
    program = harvest / "hwx-compiled" / HARVEST_PROGRAM
    target_file = program / "target"
    if target_file.is_file():
        target = target_file.read_text().strip()
    else:
        found = sorted(p.parent.name for p in program.glob("*/model.hwx"))
        if len(found) != 1:
            raise BuildError(
                f"{program} has no target file and {len(found)} compiled "
                "targets; the harvest did not compile the rewritten encoder"
            )
        target = found[0]
    layout = {
        "target": target,
        "hwx": program / target / "model.hwx",
        "features": harvest / "capture" / "ane" / "encoder_input_features.npy",
        "mask": harvest / "capture" / "ane" / "encoder_input_mask.npy",
        "hidden": harvest / "capture" / "ane" / "encoder_hidden.npy",
        "lock": harvest / "meta" / "parakeet-reference.lock",
        "run_log": harvest / "meta" / "run.log",
    }
    for name in ("hwx", "features", "mask", "hidden", "lock"):
        if not Path(layout[name]).is_file():
            raise BuildError(f"harvest file missing: {layout[name]}")
    return layout


def validate_hwx(reference, hwx: bytes) -> dict:
    """The section builder's rules plus the whole-encoder surface contract."""
    try:
        sections, ids, refs, tasks = reference.build(hwx)
        surfaces, _text, _constants, _kernels, scratch, _descriptor = \
            reference.parse(hwx)
    except reference.HwxError as error:
        raise BuildError(f"HWX refused by the section builder: {error}") from error
    subtype = struct.unpack_from("<I", hwx, 8)[0]
    if subtype != HWX_SUBTYPE_H16G:
        raise BuildError(
            f"HWX CPU subtype {subtype} is not H16G ({HWX_SUBTYPE_H16G})"
        )
    named = {(s.get("name"), s["direction"]): (i, s) for i, s in enumerate(surfaces)}
    wanted = (("attention_mask", 1), ("input_features", 1), ("encoder_hidden", 2))
    if len(surfaces) != len(wanted) or any(key not in named for key in wanted):
        raise BuildError(
            "HWX surfaces are " + ", ".join(
                f"{s.get('name')} ({'in' if s['direction'] == 1 else 'out'})"
                for s in surfaces) + "; the lane-3 whole encoder is the inputs "
            "attention_mask and input_features and the output encoder_hidden"
        )
    (mask_index, mask), (in_index, feature), (out_index, hidden) = (
        named[key] for key in wanted)
    # Raw staging copies dense row-major bytes verbatim, so the surface's
    # tensor span must be exactly the dense tensor; a single row (the mask)
    # may end in the 64-byte row padding, which stays zero.
    if mask["span"] not in (MASK_BYTES, (MASK_BYTES + 63) // 64 * 64):
        raise BuildError(
            f"HWX input attention_mask spans {mask['span']} bytes; fp16 "
            f"{MASK_SHAPE[0]} is {MASK_BYTES} (one 64-byte aligned row)"
        )
    if feature["span"] != FEATURES_BYTES:
        raise BuildError(
            f"HWX input {feature.get('name')} spans {feature['span']} bytes; "
            f"fp16 {FEATURES_SHAPE[0]}x{FEATURES_SHAPE[1]} is {FEATURES_BYTES}"
        )
    if hidden["span"] != HIDDEN_BYTES:
        raise BuildError(
            f"HWX output {hidden.get('name')} spans {hidden['span']} bytes; "
            f"fp16 {HIDDEN_SHAPE[0]}x{HIDDEN_SHAPE[1]} is {HIDDEN_BYTES}"
        )
    if len(sections[0]) > 0x10000 or any(len(s) % 4 for s in sections):
        raise BuildError("a built section is outside the H16_LOAD size rules")
    return {
        "tasks": tasks,
        "scratch_bytes": scratch[1] if scratch else 0,
        "mask": {"surface": mask_index, "allocation": mask["allocation"],
                 "name": mask.get("name")},
        "input": {"surface": in_index, "allocation": feature["allocation"],
                  "name": feature.get("name")},
        "output": {"surface": out_index, "allocation": hidden["allocation"],
                   "name": hidden.get("name")},
        "ids": ids,
        "refs": refs,
    }


def _host_build(run_log: Path) -> str:
    fields = {}
    if run_log.is_file():
        for line in run_log.read_text(errors="replace").splitlines():
            key, _, value = line.partition(":")
            if key.strip() in ("ProductVersion", "BuildVersion") and value.strip():
                fields.setdefault(key.strip(), value.strip())
    if fields:
        return "macOS {} ({}) ANECCompile".format(
            fields.get("ProductVersion", "?"), fields.get("BuildVersion", "?"))
    return "macOS ANECCompile (ane-harvest)"


def build_manifest(info: dict, hwx_sha: str, hwx_size: int, target: str,
                   host_build: str, source_commit: str) -> dict:
    payloads = [{"role": "hwx", "path": PAYLOAD, "sha256": hwx_sha,
                 "byte_size": hwx_size}]
    collection = hashlib.sha256(json.dumps(
        sorted(payloads, key=lambda record: record["path"]),
        sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    def raw(tensor: str, surface: dict, logical: int) -> dict:
        return {"tensor": tensor, "surface": surface["surface"],
                "dtype": "float16", "raw": True, "logical_bytes": logical,
                "allocation_bytes": surface["allocation"]}

    return {
        "manifest_version": MANIFEST_VERSION,
        "name": BUNDLE_NAME,
        "graph_hash": hwx_sha,
        "task_descriptors": info["tasks"],
        "inputs": [{"name": "attention_mask", "index": 0, "dtype": "float16",
                    "shape": [*MASK_SHAPE], "byte_size": MASK_BYTES,
                    "stride": info["mask"]["allocation"]},
                   {"name": "input_features", "index": 1, "dtype": "float16",
                    "shape": [*FEATURES_SHAPE], "byte_size": FEATURES_BYTES,
                    "stride": info["input"]["allocation"]}],
        "outputs": [{"name": "encoder_hidden", "index": 0, "dtype": "float16",
                     "shape": [*HIDDEN_SHAPE], "byte_size": HIDDEN_BYTES,
                     "stride": info["output"]["allocation"]}],
        "logical_results": [{"name": "encoder_hidden", "dtype": "float16",
                             "shape": [*HIDDEN_SHAPE], "tensor": "encoder_hidden",
                             "element_offset": 0,
                             "element_count": HIDDEN_SHAPE[0] * HIDDEN_SHAPE[1],
                             "conversion": "identity"}],
        "state": [],
        "intermediates": [],
        "programs": [{
            "payload": PAYLOAD,
            "operation": "whole-encoder",
            "encoder": "apple-whole-encoder-h16g",
            "task_descriptors": info["tasks"],
            "scratch_bytes": info["scratch_bytes"],
            "inputs": [raw("attention_mask", info["mask"], MASK_BYTES),
                       raw("input_features", info["input"], FEATURES_BYTES)],
            "outputs": [raw("encoder_hidden", info["output"], HIDDEN_BYTES)],
        }],
        "dispatch_plan": [0],
        "payloads": payloads,
        "compiler": {
            "host_build": host_build,
            "toolchain": f"ANECCompile TargetArchitecture {target} via "
                         f"ane-compile-hwx {COMPILER_WRAPPER_COMMIT[:12]}",
            "target": "h16g",
        },
        "driver_abi_major": DRIVER_ABI_MAJOR,
        "provenance": {
            "source_repo": "joshuaswarren/omarchy-mlx",
            "source_commit": source_commit,
            "exported_at": time.strftime("%Y-%m-%d"),
        },
        "release_asset": {"model": BUNDLE_NAME, "model_sha256": collection},
    }


def cache_key(hwx_sha: str, hwx_size: int, target: str):
    from coreml.compiled_cache import ArtifactDigest, CacheKey, CompilerIdentity

    return CacheKey.create(
        model_artifacts=(ArtifactDigest(path="encoder.hwx", sha256=hwx_sha,
                                        size=hwx_size),),
        selected_function="encoder",
        static_input_shapes={"attention_mask": MASK_SHAPE,
                             "input_features": FEATURES_SHAPE},
        compiler=CompilerIdentity(
            name="ANECCompile",
            version=target,
            commit=COMPILER_WRAPPER_COMMIT,
            target="h16g",
            package_schema="hwx",
        ),
        operations=("whole-encoder",),
        bundle_schema=MANIFEST_VERSION,
        driver_abi_major=DRIVER_ABI_MAJOR,
        firmware_identity="T6041-H16",
        frontend_version="mlx-omarchy-whole-encoder-h16-2",
    )


def build(harvest: Path, reference, cache, source_commit: str,
          repo_lock: Path) -> tuple[Path, bool, dict]:
    layout = harvest_layout(harvest)
    lock_sha = _sha256_file(Path(layout["lock"]))
    if lock_sha != _sha256_file(repo_lock):
        raise BuildError(
            f"harvest lock {layout['lock']} differs from {repo_lock}; the "
            "goldens were captured for another reference"
        )
    hwx_path = Path(layout["hwx"])
    hwx = hwx_path.read_bytes()
    hwx_sha = hashlib.sha256(hwx).hexdigest()
    info = validate_hwx(reference, hwx)
    manifest = build_manifest(info, hwx_sha, len(hwx), str(layout["target"]),
                              _host_build(Path(layout["run_log"])), source_commit)

    def produce(bundle: Path) -> None:
        bundle.mkdir(parents=True)
        (bundle / PAYLOAD).write_bytes(hwx)
        (bundle / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    result = cache.get_or_create(
        cache_key(hwx_sha, len(hwx), str(layout["target"])), produce)
    record = {
        "hwx": str(hwx_path), "hwx_sha256": hwx_sha, "hwx_bytes": len(hwx),
        "target": layout["target"], "tasks": info["tasks"],
        "scratch_bytes": info["scratch_bytes"],
        "mask_surface": info["mask"], "input_surface": info["input"],
        "output_surface": info["output"],
        "golden_sha256": {
            "encoder_input_features.npy": _sha256_file(Path(layout["features"])),
            "encoder_input_mask.npy": _sha256_file(Path(layout["mask"])),
            "encoder_hidden.npy": _sha256_file(Path(layout["hidden"])),
        },
        "lock_sha256": lock_sha,
    }
    return result.bundle, result.hit, record


def dense_features(array) -> "np.ndarray":
    import numpy as np

    values = np.asarray(array)
    if values.size != FEATURES_BYTES // 2:
        raise BuildError(f"input features have {values.size} elements, want "
                         f"{FEATURES_BYTES // 2}")
    values = values.reshape(values.shape[-2:]) if values.ndim > 2 else values
    if values.shape == (FEATURES_SHAPE[1], FEATURES_SHAPE[0]):
        values = values.T
    return np.ascontiguousarray(values.reshape(FEATURES_SHAPE)).astype("<f2")


def dense_mask(array) -> "np.ndarray":
    import numpy as np

    values = np.asarray(array).reshape(-1)
    if values.size != MASK_SHAPE[0]:
        raise BuildError(f"input mask has {values.size} elements, want "
                         f"{MASK_SHAPE[0]}")
    if not np.isin(values, (0, 1)).all():
        raise BuildError("input mask holds values other than 0 and 1")
    return np.ascontiguousarray(values).astype("<f2")


def compare(hidden, golden, contract: dict) -> dict:
    import numpy as np

    got = np.asarray(hidden, dtype=np.float32).reshape(HIDDEN_SHAPE)
    want = np.asarray(golden, dtype=np.float32).reshape(HIDDEN_SHAPE)
    diff = np.abs(got - want)
    measured = {
        "encoder_max_abs_err": float(diff.max()),
        "encoder_mean_abs_err": float(diff.mean()),
        "encoder_rel_l2_err": float(np.linalg.norm(got - want) / np.linalg.norm(want)),
        "nan_count": int(np.isnan(got).sum()),
        "inf_count": int(np.isinf(got).sum()),
    }
    failures = [
        name for name in ("encoder_max_abs_err", "encoder_mean_abs_err",
                          "encoder_rel_l2_err")
        if not measured[name] <= contract[name]
    ]
    if measured["nan_count"] > contract.get("nan_count_allowed", 0):
        failures.append("nan_count")
    if measured["inf_count"] > contract.get("inf_count_allowed", 0):
        failures.append("inf_count")
    return {"measured": measured, "failures": failures,
            "bounds": {k: contract[k] for k in ("encoder_max_abs_err",
                                                 "encoder_mean_abs_err",
                                                 "encoder_rel_l2_err")}}


def check(bundle: Path, harvest: Path, worker: Path, device: str,
          deadline_ms: int, lock: Path) -> dict:
    """One run on the ABI 3 node, compared with the golden hidden."""
    import numpy as np

    if device == "h16" and not any(Path("/dev/accel").glob("accel*")):
        raise BuildError("--check needs this machine's /dev/accel ABI 3 node")
    layout = harvest_layout(harvest)
    contract = json.loads(lock.read_text())["numerical_contract"]
    features = dense_features(np.load(layout["features"]))
    mask = dense_mask(np.load(layout["mask"]))
    golden = np.load(layout["hidden"])
    with tempfile.TemporaryDirectory(prefix="h16-check-") as scratch:
        mask_file = Path(scratch) / "attention_mask.bin"
        mask_file.write_bytes(mask.tobytes())
        source = Path(scratch) / "input_features.bin"
        target = Path(scratch) / "encoder_hidden.bin"
        source.write_bytes(features.tobytes())
        argv = [str(worker), "--bundle", str(bundle), "--device", device,
                "--deadline-ms", str(deadline_ms), "--iterations", "1",
                "--input", f"attention_mask={mask_file}",
                "--input", f"input_features={source}",
                "--save", f"encoder_hidden={target}"]
        run = subprocess.run(argv, capture_output=True, text=True, check=False)
        if run.returncode != 0:
            raise BuildError(
                f"worker exited {run.returncode}: "
                f"{(run.stderr or run.stdout).strip()[-600:]}")
        raw = target.read_bytes()
    if len(raw) != HIDDEN_BYTES:
        raise BuildError(f"worker saved {len(raw)} bytes, want {HIDDEN_BYTES}")
    hidden = np.frombuffer(raw, dtype="<f2")
    report = compare(hidden, golden, contract)
    report["worker_stdout"] = [line for line in run.stdout.splitlines()
                               if line.startswith(("device ", "worker "))]
    return report


def _default_worker() -> Path:
    found = os.environ.get("MLX_OMARCHY_ANE_WORKER", "")
    if found:
        return Path(found)
    from shutil import which

    located = which("mlx-omarchy-ane-worker")
    if not located:
        raise BuildError("mlx-omarchy-ane-worker not found; pass --worker")
    return Path(located)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--harvest", required=True, type=Path,
                        help="unpacked ane-harvest-<date> directory")
    parser.add_argument("--kernel-tree", type=Path, default=None,
                        help="aurora checkout holding the pinned h16-sections.py "
                             "(or the script itself)")
    parser.add_argument("--cache-root", type=Path, default=None,
                        help="compiled-cache root (default: the model cache)")
    parser.add_argument("--out", type=Path, default=None,
                        help="also copy the bundle files to this directory")
    parser.add_argument("--check", action="store_true",
                        help="run the bundle once on the ABI 3 node and compare "
                             "with the golden encoder_hidden")
    parser.add_argument("--worker", type=Path, default=None)
    parser.add_argument("--device", default="h16",
                        help="worker --device value (default h16)")
    parser.add_argument("--deadline-ms", type=int, default=20000)
    args = parser.parse_args(argv[1:])

    from coreml.compiled_cache import CompiledCache, default_cache_root

    if args.kernel_tree is None:
        raise BuildError("--kernel-tree is required (pinned h16-sections.py)")
    reference = load_reference_sections(args.kernel_tree)
    repo_lock = Path(__file__).resolve().parent / "parakeet-reference.lock"
    commit = subprocess.run(
        ["git", "-C", str(Path(__file__).resolve().parents[3]), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True).stdout.strip()
    cache = CompiledCache(args.cache_root or default_cache_root())
    bundle, hit, record = build(args.harvest, reference, cache, commit, repo_lock)
    print(json.dumps(record, indent=2, sort_keys=True))
    print(f"bundle {'cache-hit' if hit else 'published'}: {bundle}")
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        for name in (PAYLOAD, "manifest.json"):
            (args.out / name).write_bytes((bundle / name).read_bytes())
        print(f"copied to {args.out}")
    if args.check:
        report = check(bundle, args.harvest, args.worker or _default_worker(),
                       args.device, args.deadline_ms, repo_lock)
        print(json.dumps(report, indent=2, sort_keys=True))
        if report["failures"]:
            print("check FAILED: " + ", ".join(report["failures"]), file=sys.stderr)
            return 1
        print("check passed: encoder_hidden within the lock's numerical contract")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv))
    except BuildError as error:
        print(f"build-whole-encoder-bundle-h16: {error}", file=sys.stderr)
        raise SystemExit(1)
