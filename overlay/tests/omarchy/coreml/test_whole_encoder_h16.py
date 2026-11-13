# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Host tests for the lane-3 (H16, ABI 3) whole-encoder bundle and its
runtime selection.

The real CoreML-compiled encoder HWX comes from a one-time macOS capture run;
these tests use a synthetic stand-in with the same surface contract (one
fp16 input of 3000x128, one output of 375x640). It is a container for the
section builder's rules, not a runnable program. The builder cases need the
kernel tree's pinned h16-sections.py: point MLX_OMARCHY_H16_SECTIONS at it
(an aurora checkout or the script); without it they are skipped.
"""

import hashlib
import json
import os
import shutil
import struct
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

try:
    from _bootstrap import _TOOLS  # plain-script execution
except ImportError:  # package discovery (omarchy.coreml.*)
    from ._bootstrap import _TOOLS  # noqa: F401

import numpy as np

from coreml import ane_lane
from coreml import build_whole_encoder_bundle_h16 as h16
from coreml.compiled_cache import CompiledCache

LOCK = Path(_TOOLS) / "coreml" / "parakeet-reference.lock"
FEATURES_ALLOC = 0xbc000  # 768000 rounded up to 16 KiB
HIDDEN_ALLOC = 0x78000    # 480000 rounded up to 16 KiB
MASK_ALLOC = 0x4000       # 6000 in one 64-byte aligned row
MASK = ("attention_mask", 1, 0x32000000, MASK_ALLOC, 6016)
FEATURES = ("input_features", 1, 0x30000000, FEATURES_ALLOC, 768000)
HIDDEN = ("encoder_hidden", 2, 0x31000000, HIDDEN_ALLOC, 480000)


def synthetic_hwx(surfaces=None, subtype=7, scratch=None, tasks=2):
    """Python twin of tests/omarchy/ane/h16_test_hwx.h."""
    if surfaces is None:
        surfaces = [MASK, FEATURES, HIDDEN]
    slots = [(6 + i, vm) for i, (_, _, vm, _, _) in enumerate(surfaces)]
    if scratch:
        slots.append((3, scratch[0]))
    b = bytearray(32)

    def put32(o, v):
        if len(b) < o + 4:
            b.extend(bytes(o + 4 - len(b)))
        struct.pack_into("<I", b, o, v & 0xffffffff)

    def put64(o, v):
        put32(o, v)
        put32(o + 4, v >> 32)

    def name(o, s):
        raw = s.encode()
        if len(b) < o + len(raw):
            b.extend(bytes(o + len(raw) - len(b)))
        b[o:o + len(raw)] = raw

    commands = 0

    def segment(seg, vm, vmsize, prot, sect, size):
        nonlocal commands
        pos = len(b)
        b.extend(bytes(72 + 80))
        put32(pos, 0x19); put32(pos + 4, 152); name(pos + 8, seg)
        put64(pos + 24, vm); put64(pos + 32, vmsize)
        put32(pos + 56, prot); put32(pos + 60, prot); put32(pos + 64, 1)
        name(pos + 72, sect); name(pos + 88, seg)
        put64(pos + 104, vm); put64(pos + 112, size)
        commands += 1
        return pos

    for _, direction, vm, alloc, _ in surfaces:
        segment("__FVMLIB", vm, alloc, direction, "__fvmlib", alloc)
    stream = bytearray(16)
    struct.pack_into("<I", stream, 0, 0 if subtype == 9 else 1)
    for _ in range(tasks):
        words = 9 + 2 * len(slots)
        pos = len(stream)
        stream.extend(bytes(4 * words))
        struct.pack_into("<I", stream, pos, words << 16)
        rec = pos + 36
        for slot, _ in slots:
            struct.pack_into("<II", stream, rec, (1 << 29) | (slot << 23) | 1, 0x1234)
            rec += 8
        stream.extend(bytes((-len(stream)) % 16))
    text = segment("__TEXT", 0x10000000, 0x4000, 5, "__text", len(stream))
    if scratch:
        segment("__DATA", scratch[0], scratch[1], 3, "__bss", scratch[1])
    pos = len(b)
    b.extend(bytes(0x10 * 66))
    put32(pos, 4); put32(pos + 4, 0x10 * 66); put32(pos + 8, 4)
    for slot, base in slots:
        put64(pos + 0x10 * (slot + 1), base)
    commands += 1
    for label, direction, vm, _, span in surfaces:
        pos = len(b)
        b.extend(bytes(0x90))
        put32(pos, 4); put32(pos + 4, 0x90); put32(pos + 8, 3)
        put32(pos + 0x14, direction); put32(pos + 0x28, 1); put64(pos + 0x50, span)
        put32(pos + 0x7c, 0x80); name(pos + 0x80, label)
        commands += 1
    for label, _, vm, _, _ in surfaces:
        pos = len(b)
        b.extend(bytes(40))
        put32(pos, 0x40); put32(pos + 4, 40); put64(pos + 16, vm); name(pos + 24, label)
        commands += 1
    sizeofcmds = len(b) - 32
    offset = -(-len(b) // 0x4000) * 0x4000
    b.extend(bytes(offset - len(b)))
    b.extend(stream)
    put32(text + 72 + 48, offset)
    put32(0, 0xbeefface); put32(4, 0x80); put32(8, subtype)
    put32(16, commands); put32(20, sizeofcmds)
    return bytes(b)


def make_capture(root: Path, hwx: bytes, target="h16g", lock=LOCK):
    program = root / "hwx-compiled" / "encoder-ane"
    (program / target).mkdir(parents=True)
    (program / target / "model.hwx").write_bytes(hwx)
    (program / "target").write_text(target + "\n")
    capture = root / "capture" / "ane"
    capture.mkdir(parents=True)
    rng = np.random.default_rng(7)
    mask = np.zeros((1, 3000), np.int32)
    mask[0, :1801] = 1
    np.save(capture / "encoder_input_mask.npy", mask)
    np.save(capture / "encoder_input_features.npy",
            rng.standard_normal((1, 3000, 128)).astype(np.float32))
    np.save(capture / "encoder_hidden.npy",
            rng.standard_normal((1, 375, 640)).astype(np.float32))
    (root / "meta").mkdir()
    shutil.copy(lock, root / "meta" / "parakeet-reference.lock")
    (root / "meta" / "run.log").write_text(
        "ProductName:\tmacOS\nProductVersion:\t26.7.1\nBuildVersion:\t25G241\n")


def reference_or_skip():
    path = os.environ.get("MLX_OMARCHY_H16_SECTIONS", "")
    if not path:
        raise unittest.SkipTest("MLX_OMARCHY_H16_SECTIONS not set")
    return h16.load_reference_sections(Path(path))


class LaneTest(unittest.TestCase):
    def test_bundle_lane_follows_schema_and_abi(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.json"
            for document, lane in (
                ({"manifest_version": 5, "driver_abi_major": 3}, 3),
                ({"manifest_version": 5, "driver_abi_major": 1}, None),
                ({"manifest_version": 4, "driver_abi_major": 1}, 1),
                ({"manifest_version": 4, "driver_abi_major": 2}, 2),
                ({"manifest_version": 4, "driver_abi_major": 3}, None),
            ):
                path.write_text(json.dumps(document))
                self.assertEqual(ane_lane.bundle_lane(path), lane, document)
            path.write_text("{")
            self.assertIsNone(ane_lane.bundle_lane(path))

    def test_h16_machine_reads_the_device_tree(self):
        with tempfile.TemporaryDirectory() as tmp:
            node = Path(tmp) / "soc" / "ane@26b000000"
            node.mkdir(parents=True)
            (node / "compatible").write_bytes(b"apple,t6041-ane\0apple,ane\0")
            self.assertTrue(ane_lane.h16_machine(Path(tmp)))
            (node / "compatible").write_bytes(b"apple,t6000-ane\0")
            self.assertFalse(ane_lane.h16_machine(Path(tmp)))
        self.assertFalse(ane_lane.h16_machine(Path("/nonexistent")))

    def test_no_accel_directory_is_no_lane_3(self):
        self.assertIsNone(ane_lane.h16_node(Path("/nonexistent/accel")))

    def test_get_caps_request_number(self):
        # DRM_IOWR(0x40 + 0x00, struct drm_ane_get_caps) as h16-run.py builds it.
        self.assertEqual(ane_lane.IOCTL_GET_CAPS, 0xc0206440)


class BuilderTest(unittest.TestCase):
    def setUp(self):
        self.reference = reference_or_skip()
        self.tmp = Path(tempfile.mkdtemp(prefix="h16-bundle-test-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def build(self, hwx, **capture):
        root = self.tmp / f"capture-{len(list(self.tmp.iterdir()))}"
        make_capture(root, hwx, **capture)
        cache = CompiledCache(self.tmp / "cache")
        return h16.build(root, self.reference, cache, "c" * 40, LOCK), root

    def test_stand_in_builds_a_schema_5_bundle(self):
        hwx = synthetic_hwx(scratch=(0x40000000, 0x200000))
        (bundle, hit, record), _ = self.build(hwx)
        self.assertFalse(hit)
        manifest = json.loads((bundle / "manifest.json").read_text())
        self.assertEqual(manifest["manifest_version"], 5)
        self.assertEqual(manifest["driver_abi_major"], 3)
        self.assertEqual(manifest["compiler"]["target"], "h16g")
        self.assertEqual(manifest["compiler"]["host_build"],
                         "macOS 26.7.1 (25G241) ANECCompile")
        self.assertEqual(manifest["name"], "parakeet-encoder-whole-h16")
        sha = hashlib.sha256(hwx).hexdigest()
        self.assertEqual(manifest["graph_hash"], sha)
        self.assertEqual(manifest["payloads"][0],
                         {"role": "hwx", "path": "program-0.hwx",
                          "sha256": sha, "byte_size": len(hwx)})
        program = manifest["programs"][0]
        self.assertEqual(program["operation"], "whole-encoder")
        self.assertEqual(program["task_descriptors"], 2)
        self.assertEqual(program["scratch_bytes"], 0x200000)
        self.assertEqual([b["tensor"] for b in program["inputs"]],
                         ["attention_mask", "input_features"])
        self.assertEqual(program["inputs"][0]["surface"], 0)
        self.assertEqual(program["inputs"][0]["allocation_bytes"], MASK_ALLOC)
        self.assertEqual(program["inputs"][0]["logical_bytes"], 6000)
        self.assertEqual(program["inputs"][1]["surface"], 1)
        self.assertEqual(program["inputs"][1]["allocation_bytes"], FEATURES_ALLOC)
        self.assertEqual(program["outputs"][0]["surface"], 2)
        self.assertEqual(program["outputs"][0]["logical_bytes"], 480000)
        self.assertEqual((bundle / "program-0.hwx").read_bytes(), hwx)
        self.assertEqual(record["hwx_sha256"], sha)
        self.assertEqual(ane_lane.bundle_lane(bundle / "manifest.json"), 3)
        collection = hashlib.sha256(json.dumps(
            manifest["payloads"], sort_keys=True, separators=(",", ":")
        ).encode()).hexdigest()
        self.assertEqual(manifest["release_asset"]["model_sha256"], collection)
        # A second build is a verified cache hit.
        cache = CompiledCache(self.tmp / "cache")
        again = h16.build(self.tmp / "capture-0", self.reference, cache, "c" * 40, LOCK)
        self.assertTrue(again[1])

    def test_surface_order_follows_the_hwx(self):
        (bundle, _, _), _ = self.build(synthetic_hwx([HIDDEN, FEATURES, MASK]))
        program = json.loads((bundle / "manifest.json").read_text())["programs"][0]
        self.assertEqual(program["inputs"][0]["surface"], 2)   # attention_mask
        self.assertEqual(program["inputs"][1]["surface"], 1)   # input_features
        self.assertEqual(program["outputs"][0]["surface"], 0)

    def test_refusals(self):
        cases = {
            # The section builder's own rule fires first.
            "section builder: an H16 program has exactly one output": synthetic_hwx([
                MASK, FEATURES, HIDDEN,
                ("encoder_mask", 2, 0x33000000, 0x4000, 750)]),
            "the lane-3 whole encoder is the inputs attention_mask and "
            "input_features": synthetic_hwx([FEATURES, HIDDEN]),
            "input_features (in), encoder_hidden (out), features (in)": synthetic_hwx([
                FEATURES, HIDDEN, ("features", 1, 0x32000000, MASK_ALLOC, 6016)]),
            "attention_mask spans 6400 bytes": synthetic_hwx([
                ("attention_mask", 1, 0x32000000, MASK_ALLOC, 6400), FEATURES, HIDDEN]),
            "spans 760000 bytes": synthetic_hwx([
                MASK, ("input_features", 1, 0x30000000, FEATURES_ALLOC, 760000), HIDDEN]),
            "CPU subtype 9": synthetic_hwx(subtype=9),
            "section builder: not an H16G or H17 HWX": b"\0" * 64,
            "section builder: surface input_features has no matching tensor": synthetic_hwx([
                MASK, ("input_features", 1, 0x30000000, 0x4000, 768000), HIDDEN]),
        }
        for expected, hwx in cases.items():
            with self.subTest(expected):
                with self.assertRaises(h16.BuildError) as raised:
                    self.build(hwx)
                self.assertIn(expected, str(raised.exception))

    def test_lock_mismatch_is_refused(self):
        other = self.tmp / "other.lock"
        other.write_text("{}\n")
        with self.assertRaises(h16.BuildError) as raised:
            self.build(synthetic_hwx(), lock=other)
        self.assertIn("differs from", str(raised.exception))

    def test_reference_pin(self):
        fake = self.tmp / "h16-sections.py"
        fake.write_text("raise SystemExit('never imported')\n")
        with self.assertRaises(h16.BuildError) as raised:
            h16.load_reference_sections(fake)
        self.assertIn("is not the pinned h16-sections.py", str(raised.exception))

    def test_missing_capture_file(self):
        root = self.tmp / "empty"
        make_capture(root, synthetic_hwx())
        (root / "capture" / "ane" / "encoder_hidden.npy").unlink()
        with self.assertRaises(h16.BuildError) as raised:
            h16.capture_layout(root)
        self.assertIn("capture file missing", str(raised.exception))


class CheckTest(unittest.TestCase):
    CONTRACT = json.loads(LOCK.read_text())["numerical_contract"]

    def test_contract_bounds(self):
        golden = np.random.default_rng(3).standard_normal((1, 375, 640)).astype(np.float32)
        close = (golden + 0.001).astype(np.float16)
        report = h16.compare(close, golden, self.CONTRACT)
        self.assertEqual(report["failures"], [])
        self.assertEqual(report["bounds"]["encoder_max_abs_err"], 0.3)
        far = golden.copy()
        far[0, 0, 0] += 1.0
        self.assertIn("encoder_max_abs_err",
                      h16.compare(far, golden, self.CONTRACT)["failures"])
        nan = golden.copy()
        nan[0, 1, 1] = np.nan
        self.assertIn("nan_count", h16.compare(nan, golden, self.CONTRACT)["failures"])

    def test_mask_is_dense_3000_fp16_of_zeros_and_ones(self):
        mask = np.zeros((1, 3000), np.int32)
        mask[0, :504] = 1
        dense = h16.dense_mask(mask)
        self.assertEqual(dense.shape, (3000,))
        self.assertEqual(dense.dtype, np.dtype("<f2"))
        self.assertEqual(float(dense.sum()), 504.0)
        with self.assertRaises(h16.BuildError):
            h16.dense_mask(mask * 2)
        with self.assertRaises(h16.BuildError):
            h16.dense_mask(np.ones(2999, np.int32))

    def test_features_are_dense_3000x128_fp16(self):
        values = np.arange(3000 * 128, dtype=np.float32).reshape(1, 3000, 128) % 100
        dense = h16.dense_features(values)
        self.assertEqual(dense.shape, (3000, 128))
        self.assertEqual(dense.dtype, np.dtype("<f2"))
        flipped = h16.dense_features(values.reshape(3000, 128).T[None])
        self.assertTrue(np.array_equal(dense, flipped))

    def test_check_runs_the_worker_once_and_compares(self):
        tmp = Path(tempfile.mkdtemp(prefix="h16-check-test-"))
        self.addCleanup(shutil.rmtree, tmp, True)
        capture = tmp / "capture"
        make_capture(capture, synthetic_hwx())
        golden = np.load(capture / "capture" / "ane" / "encoder_hidden.npy")
        answer = tmp / "answer.bin"
        answer.write_bytes(golden.astype("<f2").tobytes())
        worker = tmp / "fake-worker"
        worker.write_text(
            "#!" + sys.executable + "\n"
            "import sys, shutil\n"
            "args = sys.argv[1:]\n"
            "assert args[args.index('--device') + 1] == 'fake-h16-for-test'\n"
            "inputs = [a.split('=', 1) for a, f in zip(args[1:], args) if f == '--input']\n"
            "assert [n for n, _ in inputs] == ['attention_mask', 'input_features']\n"
            "mask = open(inputs[0][1], 'rb').read()\n"
            "assert len(mask) == 6000 and mask[:2] == b'\\x00\\x3c' and mask[-2:] == bytes(2)\n"
            "assert len(open(inputs[1][1], 'rb').read()) == 768000\n"
            "save = args[args.index('--save') + 1].split('=', 1)[1]\n"
            f"shutil.copy({str(answer)!r}, save)\n"
            "print('device lane=3 node=/dev/accel/accel0 fake')\n")
        worker.chmod(0o755)
        bundle = tmp / "bundle"
        bundle.mkdir()
        report = h16.check(bundle, capture, worker, "fake-h16-for-test", 1000, LOCK)
        self.assertEqual(report["failures"], [])
        self.assertLess(report["measured"]["encoder_max_abs_err"], 0.01)

    def test_check_refuses_without_accel(self):
        if any(Path("/dev/accel").glob("accel*")):
            self.skipTest("this host has an accel node")
        with self.assertRaises(h16.BuildError) as raised:
            h16.check(Path("/nonexistent"), Path("/nonexistent"), Path("/bin/false"),
                      "h16", 1000, LOCK)
        self.assertIn("/dev/accel", str(raised.exception))


def import_encoder_without_mlx():
    """vulkan_encoder with mlx.core stubbed: discovery and lane selection
    are plain Python; nothing here evaluates a tensor."""
    stub = types.ModuleType("mlx")
    stub.core = mock.MagicMock()
    with mock.patch.dict(sys.modules, {"mlx": stub, "mlx.core": stub.core}):
        sys.modules.pop("coreml.vulkan_encoder", None)
        from coreml import vulkan_encoder
        return vulkan_encoder


class RuntimeSelectionTest(unittest.TestCase):
    def setUp(self):
        self.encoder = import_encoder_without_mlx()
        self.tmp = Path(tempfile.mkdtemp(prefix="h16-select-test-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        for name, version, abi in (("parakeet-encoder-whole", 4, 1),
                                   ("parakeet-encoder-whole-h16", 5, 3)):
            bundle = self.tmp / "cache" / name / "bundle"
            bundle.mkdir(parents=True)
            (bundle / "manifest.json").write_text(json.dumps({
                "manifest_version": version, "driver_abi_major": abi, "name": name,
                "programs": [{"operation": "whole-encoder"}]}))
        (self.tmp / "cache" / "parakeet-encoder-whole-h16" / "cache.json").write_text(
            json.dumps({"bundle_artifacts": [
                {"path": "manifest.json", "sha256": "a" * 64, "size": 1},
                {"path": "program-0.hwx", "sha256": "b" * 64, "size": 1}]}))
        patcher = mock.patch("coreml.compiled_cache.default_cache_root",
                             return_value=self.tmp / "cache")
        patcher.start()
        self.addCleanup(patcher.stop)
        env = mock.patch.dict(os.environ, {"MLX_OMARCHY_WHOLE_ENCODER_BUNDLE": ""})
        env.start()
        self.addCleanup(env.stop)

    def island(self, node, machine):
        # vulkan_encoder imports ane_lane as a top-level module.
        lane = self.encoder.ane_lane
        with mock.patch.object(lane, "h16_node", return_value=node), \
             mock.patch.object(lane, "h16_machine", return_value=machine):
            return self.encoder.AneIsland(
                Path("/bin/true"), Path("/nonexistent/libane.so"),
                self.tmp / "bundles", self.tmp / "scratch",
                seal_assets={"bundles": {}, "libane": {}})

    def test_lane_3_selects_the_hwx_bundle(self):
        island = self.island(Path("/dev/accel/accel0"), True)
        self.assertEqual(island.whole_bundle.parent.name, "parakeet-encoder-whole-h16")
        self.assertEqual(island.device, "h16")
        self.assertEqual(island.resident_bundles, set())
        self.assertIs(island.encoder_island(), island)
        self.assertEqual(island.encoder_path, "h16-whole-encoder")
        pins = island._seal_kwargs(island._session_bundles())
        self.assertEqual(pins, {"seal_expects": {"parakeet-encoder-whole-h16": {
            "manifest.json": "a" * 64, "program-0.hwx": "b" * 64}}})

    def test_lanes_1_and_2_never_take_the_hwx_bundle(self):
        island = self.island(None, False)
        self.assertEqual(island.whole_bundle.parent.name, "parakeet-encoder-whole")
        self.assertIsNone(island.device)
        self.assertEqual(island.encoder_path, "whole-encoder")

    def test_h16_without_the_bundle_runs_the_encoder_on_the_gpu(self):
        shutil.rmtree(self.tmp / "cache" / "parakeet-encoder-whole-h16")
        island = self.island(Path("/dev/accel/accel0"), True)
        self.assertIsNone(island.whole_bundle)
        self.assertIsNone(island.encoder_island())
        self.assertEqual(island.encoder_path, "gpu")

    def test_h16_without_the_abi_3_node_runs_on_the_gpu(self):
        island = self.island(None, True)
        self.assertIsNone(island.whole_bundle)  # the M1/M2 bundle is not taken
        self.assertIsNone(island.encoder_island())

    def test_explicit_bundle_must_match_the_lane(self):
        m1 = self.tmp / "cache" / "parakeet-encoder-whole" / "bundle"
        with mock.patch.dict(os.environ, {"MLX_OMARCHY_WHOLE_ENCODER_BUNDLE": str(m1)}):
            with self.assertRaises(self.encoder.EncoderRunError) as raised:
                self.encoder._discover_whole_bundle(lane3=True)
            self.assertIn("lane 3 (schema 5, ABI 3)", str(raised.exception))
            self.assertEqual(self.encoder._discover_whole_bundle(lane3=False), m1)

    def test_launch_argv_uses_the_device(self):
        island = self.island(Path("/dev/accel/accel0"), True)
        island._mode = "launch"
        seen = {}

        def fake_run(argv, **_):
            seen["argv"] = argv
            raise RuntimeError("stop after argv")

        with mock.patch.object(self.encoder.subprocess, "run", side_effect=fake_run):
            with self.assertRaises(RuntimeError):
                island._submit_launch(island.whole_bundle, "t", {}, {})
        argv = seen["argv"]
        self.assertEqual(argv[argv.index("--device") + 1], "h16")
        self.assertNotIn("--libane", argv)


if __name__ == "__main__":
    unittest.main()
