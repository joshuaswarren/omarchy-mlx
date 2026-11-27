"""Bounded-read and metadata-validation tests for mlx_omarchy_bonsai2.loader.

Loads serve/mlx_omarchy_bonsai2/loader.py with minimal stub modules under a
private module name so the suite runs in mlx-free CI without polluting
sys.modules. Every test input is finite and bounded.
"""

from __future__ import annotations

import errno
import hashlib
import importlib.util
import json
import os
import struct
import sys
import tempfile
import threading
import tracemalloc
import types
import unittest
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parents[1]
LOADER_PATH = REPO / "serve" / "mlx_omarchy_bonsai2" / "loader.py"
_PRIVATE_MODULE_NAME = "mlx_omarchy_bonsai2._bonsai2_loader_bounds_under_test"
_PINNED_STACK = 4 * 1024 * 1024
_DEEP_JSON = b"[" * 200_000 + b"]" * 200_000


def _load_loader_module():
    mlx_mod = types.ModuleType("mlx")
    mlx_core = types.ModuleType("mlx.core")
    mlx_utils = types.ModuleType("mlx.utils")
    mlx_utils.tree_flatten = lambda tree: []
    mlx_nn = types.ModuleType("mlx.nn")
    mlx_nn.Embedding = type("Embedding", (), {})
    mlx_nn.Linear = type("Linear", (), {})
    mlx_mod.core, mlx_mod.utils, mlx_mod.nn = mlx_core, mlx_utils, mlx_nn

    pkg_mod = types.ModuleType("mlx_omarchy_bonsai2")
    pkg_mod.__path__ = []
    packed_mod = types.ModuleType("mlx_omarchy_bonsai2.packed")
    packed_mod.ALLOWED_BLOCKS = (512, 1024, 2048, 4096)
    packed_mod.BITS = 2
    packed_mod.GROUP_SIZE = 128
    packed_mod.Packed = type("Packed", (), {})
    packed_mod.PackedError = type("PackedError", (ValueError,), {})
    pkg_mod.packed = packed_mod

    stubs = {
        "mlx": mlx_mod, "mlx.core": mlx_core, "mlx.utils": mlx_utils, "mlx.nn": mlx_nn,
        "mlx_omarchy_bonsai2": pkg_mod, "mlx_omarchy_bonsai2.packed": packed_mod,
    }
    spec = importlib.util.spec_from_file_location(_PRIVATE_MODULE_NAME, LOADER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    module.__package__ = "mlx_omarchy_bonsai2"
    with mock.patch.dict(sys.modules, stubs):
        spec.loader.exec_module(module)
    return module, stubs


loader, _STUB_MODULES = _load_loader_module()
PackError = loader.PackError


def _run_bounded(fn, timeout: float = 5.0):
    """Run fn on a 4 MiB stack daemon thread with a hard timeout."""
    outcome = {}

    def target():
        try:
            outcome["value"] = fn()
        except BaseException as exc:
            outcome["error"] = exc

    prev = threading.stack_size(_PINNED_STACK)
    try:
        t = threading.Thread(target=target, daemon=True)
        t.start()
    finally:
        threading.stack_size(prev)
    t.join(timeout)
    if t.is_alive():
        raise TimeoutError(f"did not finish within {timeout} s")
    if "error" in outcome:
        raise outcome["error"]
    return outcome["value"]


def _unblock_fifo(path: Path) -> None:
    try:
        fd = os.open(path, os.O_WRONLY | os.O_NONBLOCK)
    except OSError:
        return
    os.close(fd)


def _valid_config() -> dict:
    return {
        "schema_version": 2,
        "model_type": "prism_hadamard_qwen35",
        "base_model_type": "qwen3_5",
        "tensor_namespace": "mlx-vlm-qwen3_5",
        "quantization": {"bits": 2, "group_size": 128, "mode": "affine"},
        "text_config": {
            "num_hidden_layers": 4, "full_attention_interval": 2,
            "num_key_value_heads": 4, "head_dim": 64, "max_position_embeddings": 512,
        },
        "modules": [
            {"path": "model.embed_tokens", "block": 512, "embedding": True, "dtype": "float16"}
        ],
    }


def _valid_header(include_metadata: bool = False) -> dict:
    header = {
        "language_model.model.embed_tokens.weight": {
            "dtype": "U32", "shape": [2, 2], "data_offsets": [0, 16],
        },
        "vision_tower.weight": {
            "dtype": "F16", "shape": [3, 4], "data_offsets": [16, 40],
        },
    }
    if include_metadata:
        header["__metadata__"] = {"format": "pt", "creator": "test"}
    return header


def _pack_safetensors(header, payload_size: int | None = None) -> bytes:
    raw = header if isinstance(header, bytes) else json.dumps(header).encode("utf-8")
    if payload_size is None:
        payload_size = 0
        if isinstance(header, dict):
            for key, info in header.items():
                if key == "__metadata__" or not isinstance(info, dict):
                    continue
                offsets = info.get("data_offsets")
                if (
                    isinstance(offsets, list) and len(offsets) == 2
                    and all(isinstance(v, int) and not isinstance(v, bool) for v in offsets)
                    and offsets[1] > payload_size
                ):
                    payload_size = offsets[1]
    return struct.pack("<Q", len(raw)) + raw + (b"\x00" * payload_size)


class Bonsai2LoaderBoundsTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(sys.set_int_max_str_digits, sys.get_int_max_str_digits())
        sys.set_int_max_str_digits(4300)

    def _tmp(self) -> Path:
        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        return Path(td.name)

    def _make_pack(self, config=None, safetensors=None) -> Path:
        pack_dir = self._tmp() / "pack"
        pack_dir.mkdir()
        raw_cfg = (
            config if isinstance(config, bytes)
            else json.dumps(_valid_config() if config is None else config).encode("utf-8")
        )
        (pack_dir / "config.json").write_bytes(raw_cfg)
        raw_st = (
            safetensors if isinstance(safetensors, bytes)
            else _pack_safetensors(_valid_header() if safetensors is None else safetensors)
        )
        (pack_dir / "model.safetensors").write_bytes(raw_st)
        return pack_dir

    def _assert_pack_error(
        self, fn, *expected_substrings: str, bounded: bool = False, timeout: float = 5.0
    ) -> str:
        with self.assertRaises(PackError) as ctx:
            _run_bounded(fn, timeout=timeout) if bounded else fn()
        msg = str(ctx.exception)
        for sub in expected_substrings:
            self.assertIn(sub, msg)
        return msg

    def _assert_low_mem_pack_error(self, fn, *expected_substrings: str) -> str:
        if not tracemalloc.is_tracing():
            tracemalloc.start()
            self.addCleanup(tracemalloc.stop)
        tracemalloc.reset_peak()
        before = tracemalloc.get_traced_memory()[0]
        msg = self._assert_pack_error(fn, *expected_substrings)
        self.assertLess(tracemalloc.get_traced_memory()[1] - before, 4 << 20)
        return msg

    def _replace_with_non_regular(self, target: Path, kind: str) -> None:
        target.unlink()
        if kind == "devnull-symlink":
            os.symlink(os.devnull, target)
        elif kind == "directory":
            target.mkdir()
        elif kind == "fifo":
            os.mkfifo(target)
            self.addCleanup(_unblock_fifo, target)
        elif kind == "fifo-with-writer":
            os.mkfifo(target)
            rw_fd = os.open(target, os.O_RDWR | os.O_NONBLOCK)
            self.addCleanup(os.close, rw_fd)
            os.write(rw_fd, json.dumps(_valid_config()).encode("utf-8"))

    def test_stub_loader_does_not_pollute_sys_modules(self):
        self.assertNotIn(_PRIVATE_MODULE_NAME, sys.modules)
        for name, stub in _STUB_MODULES.items():
            self.assertIsNot(sys.modules.get(name), stub)

    def test_valid_tiny_pack_footprint_and_sha256(self):
        for label, set_metadata, metadata in (
            ("no-metadata", False, None),
            ("null-metadata", True, None),
            ("pt-metadata", True, {"format": "pt", "creator": "test"}),
            ("mlx-metadata", True, {"format": "mlx"}),
        ):
            with self.subTest(label):
                cfg_bytes = json.dumps(_valid_config(), indent=2).encode("utf-8")
                hdr = _valid_header()
                if set_metadata:
                    hdr["__metadata__"] = metadata
                pack_dir = self._make_pack(
                    config=cfg_bytes,
                    safetensors=_pack_safetensors(hdr),
                )
                facts = loader.pack_footprint(pack_dir)
                self.assertEqual(facts["total_header_bytes"], 40)
                self.assertEqual(facts["live_weights_bytes"], 16)
                self.assertEqual(facts["kv_bytes_per_token"], (4 // 2) * 2 * 4 * 64 * 2)
                self.assertEqual(facts["max_position_embeddings"], 512)
                self.assertEqual(facts["packed_modules"], 1)
                self.assertEqual(loader.config_sha256(pack_dir), hashlib.sha256(cfg_bytes).hexdigest())

    def test_hf_cache_relative_symlinks_and_dangling_symlinks(self):
        root = self._tmp()
        blobs, regular_dir, snap_dir = root / "blobs", root / "regular", root / "snapshots" / "rev1"
        for d in (blobs, regular_dir, snap_dir):
            d.mkdir(parents=True)
        cfg_bytes = json.dumps(_valid_config()).encode("utf-8")
        st_bytes = _pack_safetensors(_valid_header(include_metadata=True))
        cfg_blob = blobs / hashlib.sha256(cfg_bytes).hexdigest()
        st_blob = blobs / hashlib.sha256(st_bytes).hexdigest()
        cfg_blob.write_bytes(cfg_bytes)
        st_blob.write_bytes(st_bytes)
        (regular_dir / "config.json").write_bytes(cfg_bytes)
        (regular_dir / "model.safetensors").write_bytes(st_bytes)
        os.symlink(os.path.relpath(cfg_blob, snap_dir), snap_dir / "config.json")
        os.symlink(os.path.relpath(st_blob, snap_dir), snap_dir / "model.safetensors")

        self.assertEqual(loader.pack_footprint(snap_dir), loader.pack_footprint(regular_dir))
        self.assertEqual(loader.config_sha256(snap_dir), loader.config_sha256(regular_dir))

        enoent = os.strerror(errno.ENOENT)
        for label, bad_name, good_name, good_blob, call_sha in (
            ("dangling-config-symlink", "config.json", "model.safetensors", st_blob, True),
            ("dangling-safetensors-symlink", "model.safetensors", "config.json", cfg_blob, False),
        ):
            with self.subTest(label):
                dangling = root / label
                dangling.mkdir()
                bad_link = dangling / bad_name
                os.symlink(f"../../blobs/missing-{bad_name}", bad_link)
                os.symlink(os.path.relpath(good_blob, dangling), dangling / good_name)
                self._assert_pack_error(
                    lambda d=dangling: loader.pack_footprint(d), "cannot read", str(bad_link), enoent
                )
                if call_sha:
                    self._assert_pack_error(
                        lambda d=dangling: loader.config_sha256(d), "cannot read", str(bad_link), enoent
                    )

    def test_safetensors_short_truncated_and_over_cap_headers(self):
        valid_body = json.dumps(_valid_header()).encode("utf-8")
        for label, raw, reason in (
            ("empty", b"", "too short"),
            ("3-bytes", b"\x01\x02\x03", "too short"),
            ("7-bytes", b"\x00" * 7, "too short"),
            ("truncated-body", struct.pack("<Q", len(valid_body) + 16) + valid_body, "truncated"),
        ):
            with self.subTest(label):
                pack_dir = self._make_pack(safetensors=raw)
                self._assert_pack_error(lambda p=pack_dir: loader.pack_footprint(p), reason)

        with self.subTest("truncated-after-fstat"):
            pack_dir = self._make_pack(
                safetensors=struct.pack("<Q", len(valid_body) + 16) + valid_body
            )
            st_ino, real_fstat = (pack_dir / "model.safetensors").stat().st_ino, os.fstat

            def _stale_fstat(fd):
                st = real_fstat(fd)
                if st.st_ino == st_ino:
                    return types.SimpleNamespace(st_mode=st.st_mode, st_size=st.st_size + 64)
                return st

            with mock.patch("os.fstat", side_effect=_stale_fstat):
                self._assert_pack_error(
                    lambda: loader.pack_footprint(pack_dir), "truncated"
                )

        cap = 16 << 20
        with self.subTest("over-cap-2^64-1"):
            pack_dir = self._make_pack(safetensors=struct.pack("<Q", (1 << 64) - 1))
            self._assert_low_mem_pack_error(lambda: loader.pack_footprint(pack_dir), "exceeds")

        for label, claimed, file_size, expected in (
            ("over-cap-by-1-sparse", cap + 1, 8 + cap + 1, ("model.safetensors", "exceeds")),
            ("sparse-truncated-precheck", cap, 8 << 20, ("truncated",)),
        ):
            with self.subTest(label):
                pack_dir = self._make_pack()
                with open(pack_dir / "model.safetensors", "wb") as f:
                    f.write(struct.pack("<Q", claimed))
                    f.truncate(file_size)
                self._assert_low_mem_pack_error(lambda p=pack_dir: loader.pack_footprint(p), *expected)

    def test_safetensors_malformed_json_and_tensor_entries(self):
        def _with_entry(key, value):
            hdr = _valid_header()
            hdr[key] = value
            return _pack_safetensors(hdr, payload_size=40)

        def _with_extra_raw(raw_value: bytes) -> bytes:
            hdr = _valid_header()
            hdr["vision_tower.weight"]["calibration"] = "__EXTRA__"
            raw = json.dumps(hdr).encode("utf-8").replace(b'"__EXTRA__"', raw_value)
            return _pack_safetensors(raw, payload_size=40)

        cases = (
            ("zero-length-header", struct.pack("<Q", 0), "not valid JSON", False),
            ("corrupt-json", _pack_safetensors(b"{not json", 0), "not valid JSON", False),
            ("invalid-utf8", _pack_safetensors(b'{"a": "\xff"}', 0), "not valid JSON", False),
            ("huge-int", _pack_safetensors(b'{"a": ' + b"1" * 5000 + b"}", 0), "not valid JSON", False),
            ("deep-json", _pack_safetensors(_DEEP_JSON, 0), "not valid JSON", True),
            ("nonfinite-nan", _with_extra_raw(b"NaN"), "not valid JSON", False),
            ("nonfinite-inf", _with_extra_raw(b"Infinity"), "not valid JSON", False),
            ("nonfinite-neg-inf", _with_extra_raw(b"-Infinity"), "not valid JSON", False),
            ("header-list", _pack_safetensors(b"[]", 0), "not a JSON object", False),
            ("header-str", _pack_safetensors(b'"x"', 0), "not a JSON object", False),
            ("header-null", _pack_safetensors(b"null", 0), "not a JSON object", False),
            ("metadata-non-str-value", _with_entry("__metadata__", {"revision": 1}), "__metadata__ must map strings to strings", False),
            ("metadata-list", _with_entry("__metadata__", []), "__metadata__ must map strings to strings", False),
            ("entry-int", _with_entry("vision.a", 1), "vision.a", False),
            ("entry-list", _with_entry("vision.a", []), "vision.a", False),
            ("entry-null", _with_entry("vision.a", None), "vision.a", False),
            ("dtype-missing", _with_entry("vision.a", {"shape": [2], "data_offsets": [0, 8]}), "vision.a", False),
            ("dtype-int", _with_entry("vision.a", {"dtype": 16, "shape": [2], "data_offsets": [0, 8]}), "vision.a", False),
            ("dtype-null", _with_entry("vision.a", {"dtype": None, "shape": [2], "data_offsets": [0, 8]}), "vision.a", False),
            ("dtype-unhashable", _with_entry("vision.a", {"dtype": [], "shape": [2], "data_offsets": [0, 8]}), "malformed metadata", False),
            ("shape-missing", _with_entry("vision.a", {"dtype": "F16", "data_offsets": [0, 8]}), "vision.a", False),
            ("shape-non-list", _with_entry("vision.a", {"dtype": "F16", "shape": "2", "data_offsets": [0, 8]}), "vision.a", False),
            ("shape-negative-matching-span", _with_entry("vision.a", {"dtype": "U8", "shape": [-2, -2], "data_offsets": [0, 4]}), "malformed metadata", False),
            ("shape-bool-matching-span", _with_entry("vision.a", {"dtype": "U8", "shape": [True], "data_offsets": [0, 1]}), "malformed metadata", False),
            ("shape-float", _with_entry("vision.a", {"dtype": "F16", "shape": [2.0], "data_offsets": [0, 8]}), "vision.a", False),
            ("long-tensor-name", _with_entry("k" * 200, 1), "k" * 80, False),
        )
        for label, raw, reason, bounded in cases:
            with self.subTest(label):
                pack_dir = self._make_pack(safetensors=raw)
                msg = self._assert_pack_error(
                    lambda p=pack_dir: loader.pack_footprint(p), reason, bounded=bounded
                )
                if label == "long-tensor-name":
                    self.assertNotIn("k" * 81, msg)

        with self.subTest("finite-extra-entry-field-accepted"):
            pack_dir = self._make_pack(safetensors=_with_extra_raw(b"0.5"))
            facts = loader.pack_footprint(pack_dir)
            self.assertEqual((facts["total_header_bytes"], facts["live_weights_bytes"]), (40, 16))

    def test_safetensors_dtype_and_byte_span_contract(self):
        accepted = (
            ("BOOL", 1, []), ("U8", 1, [2]), ("I8", 1, [2]), ("F8_E4M3", 1, [2]),
            ("F8_E8M0", 1, [2]), ("U16", 2, [2]), ("I16", 2, [2]), ("F16", 2, [2]),
            ("BF16", 2, [2]), ("U32", 4, [2]), ("I32", 4, [2]), ("F32", 4, []),
            ("U64", 8, [1]), ("I64", 8, [1]), ("C64", 8, [1]),
        )
        with self.subTest("all-accepted-dtypes-scalar-and-zero-dim"):
            hdr, offset = {}, 0
            for idx, (dtype, itemsize, shape) in enumerate(accepted):
                nbytes = itemsize * (shape[0] if shape else 1)
                hdr[f"language_model.t{idx}"] = {
                    "dtype": dtype, "shape": shape, "data_offsets": [offset, offset + nbytes],
                }
                offset += nbytes
            hdr["vision.zero"] = {"dtype": "F32", "shape": [0, 4], "data_offsets": [offset, offset]}
            pack_dir = self._make_pack(safetensors=_pack_safetensors(hdr, payload_size=offset))
            facts = loader.pack_footprint(pack_dir)
            self.assertEqual((facts["live_weights_bytes"], facts["total_header_bytes"]), (offset, offset))

        with self.subTest("trailing-zero-dim-accepted"):
            hdr = _valid_header()
            hdr["language_model.zero_tail"] = {"dtype": "F16", "shape": [4, 0], "data_offsets": [16, 16]}
            pack_dir = self._make_pack(safetensors=_pack_safetensors(hdr, payload_size=40))
            facts = loader.pack_footprint(pack_dir)
            self.assertEqual((facts["total_header_bytes"], facts["live_weights_bytes"]), (40, 16))

        for bad_dtype in ("F64", "F8_E5M2", "F4", "UNKNOWN"):
            with self.subTest(f"unsupported-dtype-{bad_dtype}"):
                hdr = _valid_header()
                hdr["vision.bad"] = {"dtype": bad_dtype, "shape": [1], "data_offsets": [0, 8]}
                pack_dir = self._make_pack(safetensors=_pack_safetensors(hdr, payload_size=40))
                self._assert_pack_error(
                    lambda p=pack_dir: loader.pack_footprint(p), "vision.bad", "unsupported dtype"
                )

        with self.subTest("unsupported-dtype-clipped"):
            hdr = _valid_header()
            hdr["vision.bad"] = {"dtype": "D" * 1000, "shape": [1], "data_offsets": [0, 8]}
            pack_dir = self._make_pack(safetensors=_pack_safetensors(hdr, payload_size=40))
            msg = self._assert_pack_error(
                lambda p=pack_dir: loader.pack_footprint(p),
                "vision.bad", "unsupported dtype", "D" * 80,
            )
            self.assertNotIn("D" * 81, msg)
            self.assertLess(len(msg), 300)

        for label, entry in (
            ("shape-larger-than-span", {"dtype": "F32", "shape": [1048576], "data_offsets": [0, 4]}),
            ("span-larger-than-shape", {"dtype": "F16", "shape": [2], "data_offsets": [0, 8]}),
            ("scalar-zero-span", {"dtype": "F32", "shape": [], "data_offsets": [0, 0]}),
            ("zero-dim-nonzero-span", {"dtype": "F16", "shape": [0, 4], "data_offsets": [0, 2]}),
        ):
            with self.subTest(label):
                hdr = _valid_header()
                hdr["vision.span"] = entry
                pack_dir = self._make_pack(safetensors=_pack_safetensors(hdr, payload_size=40))
                self._assert_pack_error(
                    lambda p=pack_dir: loader.pack_footprint(p), "vision.span", "byte span"
                )

    def test_safetensors_invalid_data_offsets_refused(self):
        for label, offsets, payload_size in (
            ("inverted", [1000, 0], 1024),
            ("bool-begin", [False, 8], 64),
            ("empty-list", [], 64),
            ("len-1", [0], 64),
            ("len-3", [0, 8, 16], 64),
            ("float-end", [0, 8.0], 64),
            ("missing", None, 64),
            ("string", "0,8", 64),
            ("end-past-eof", [33, 41], 40),
            ("end-way-past-eof", [9992, 10_000], 40),
        ):
            with self.subTest(label):
                hdr = _valid_header()
                entry = {"dtype": "F16", "shape": [2, 2]}
                if offsets is not None:
                    entry["data_offsets"] = offsets
                hdr["vision.b"] = entry
                pack_dir = self._make_pack(safetensors=_pack_safetensors(hdr, payload_size=payload_size))
                expected = (
                    ("vision.b", "invalid data_offsets") if label == "inverted"
                    else ("vision.b", "malformed metadata") if label == "bool-begin"
                    else ("vision.b",)
                )
                self._assert_pack_error(lambda p=pack_dir: loader.pack_footprint(p), *expected)

        for label, entry in (
            ("bool-end-matching-span", {"dtype": "U8", "shape": [1], "data_offsets": [0, True]}),
            ("negative-begin-matching-span", {"dtype": "U8", "shape": [4], "data_offsets": [-4, 0]}),
        ):
            with self.subTest(label):
                hdr = _valid_header()
                hdr["vision.b"] = entry
                pack_dir = self._make_pack(safetensors=_pack_safetensors(hdr, payload_size=40))
                self._assert_pack_error(
                    lambda p=pack_dir: loader.pack_footprint(p), "vision.b", "malformed metadata"
                )

        with self.subTest("zero-byte-and-exact-eof-accepted"):
            hdr = _valid_header()
            hdr["vision.empty"] = {"dtype": "F16", "shape": [0], "data_offsets": [40, 40]}
            pack_dir = self._make_pack(safetensors=_pack_safetensors(hdr, payload_size=40))
            facts = loader.pack_footprint(pack_dir)
            self.assertEqual((facts["total_header_bytes"], facts["live_weights_bytes"]), (40, 16))

    def test_non_regular_and_io_errors_refused_without_blocking(self):
        for filename, kinds, with_sha in (
            ("model.safetensors", ("devnull-symlink", "directory", "fifo"), False),
            ("config.json", ("devnull-symlink", "directory", "fifo", "fifo-with-writer"), True),
        ):
            for kind in kinds:
                with self.subTest(f"{filename}:{kind}"):
                    pack_dir = self._make_pack()
                    target = pack_dir / filename
                    self._replace_with_non_regular(target, kind)
                    is_fifo = kind.startswith("fifo")
                    fns = [lambda p=pack_dir: loader.pack_footprint(p)]
                    if with_sha:
                        fns.append(lambda p=pack_dir: loader.config_sha256(p))
                    for fn in fns:
                        self._assert_pack_error(
                            fn, str(target), "not a regular file", bounded=is_fifo, timeout=2.0
                        )

            with self.subTest(f"{filename}:read-eio"):
                pack_dir = self._make_pack()
                target = pack_dir / filename
                target_ino, real_read = target.stat().st_ino, os.read

                def _fail_target(fd, n, ino=target_ino):
                    if os.fstat(fd).st_ino == ino:
                        raise OSError(errno.EIO, "Input/output error")
                    return real_read(fd, n)

                with mock.patch("os.read", side_effect=_fail_target):
                    self._assert_pack_error(
                        lambda p=pack_dir: loader.pack_footprint(p), str(target), "Input/output error"
                    )
                    if with_sha:
                        self._assert_pack_error(
                            lambda p=pack_dir: loader.config_sha256(p), str(target), "Input/output error"
                        )

            if os.geteuid() != 0:
                with self.subTest(f"{filename}:mode-000"):
                    pack_dir = self._make_pack()
                    target = pack_dir / filename
                    target.chmod(0)
                    self.addCleanup(target.chmod, 0o600)
                    self._assert_pack_error(
                        lambda p=pack_dir: loader.pack_footprint(p), "cannot read", str(target)
                    )
                    if with_sha:
                        self._assert_pack_error(
                            lambda p=pack_dir: loader.config_sha256(p), "cannot read", str(target)
                        )

    def test_config_malformed_json_utf8_and_non_object(self):
        empty_pack = self._make_pack(config=b"")
        self._assert_pack_error(
            lambda: loader.pack_footprint(empty_pack), "not valid JSON", bounded=True, timeout=2.0
        )
        for label, payload, reason, bounded in (
            ("corrupt-json", b"{not json", "not valid JSON", False),
            ("invalid-utf8", b'{"model_type": "\xff"}', "not valid JSON", False),
            ("utf8-bom", b"\xef\xbb\xbf" + json.dumps(_valid_config()).encode("utf-8"), "not valid JSON", False),
            ("huge-int", b'{"schema_version": ' + b"1" * 5000 + b"}", "not valid JSON", False),
            ("deep-json", _DEEP_JSON, "not valid JSON", True),
            ("list-empty", b"[]", "not a JSON object", False),
            ("string", b'"hello"', "not a JSON object", False),
            ("null", b"null", "not a JSON object", False),
            ("number", b"42", "not a JSON object", False),
        ):
            with self.subTest(label):
                pack_dir = self._make_pack(config=payload)
                self._assert_pack_error(
                    lambda p=pack_dir: loader.pack_footprint(p), reason, bounded=bounded
                )

    def test_config_size_cap_sparse_and_boundary(self):
        with self.subTest("sparse-over-default-cap"):
            pack_dir = self._make_pack()
            cfg_path = pack_dir / "config.json"
            with open(cfg_path, "wb") as f:
                f.truncate((16 << 20) + 1)
            for label, fn in (
                ("pack_footprint", lambda: loader.pack_footprint(pack_dir)),
                ("config_sha256", lambda: loader.config_sha256(pack_dir)),
            ):
                with self.subTest(label):
                    self._assert_low_mem_pack_error(fn, str(cfg_path), "exceeds")

        with self.subTest("at-default-cap"):
            cap = 16 << 20
            base_raw = json.dumps(_valid_config()).encode("utf-8")
            pack_dir = self._make_pack()
            cfg_path = pack_dir / "config.json"
            digest, pad_chunk = hashlib.sha256(), b" " * (1 << 20)
            with open(cfg_path, "wb") as f:
                f.write(base_raw)
                digest.update(base_raw)
                remaining = cap - len(base_raw)
                while remaining > 0:
                    piece = pad_chunk if remaining >= len(pad_chunk) else b" " * remaining
                    f.write(piece)
                    digest.update(piece)
                    remaining -= len(piece)
            self.assertEqual(cfg_path.stat().st_size, cap)
            self.assertEqual(loader.pack_footprint(pack_dir)["live_weights_bytes"], 16)
            self.assertEqual(loader.config_sha256(pack_dir), digest.hexdigest())

    def test_config_read_stops_at_cap_plus_one_when_fstat_underreports(self):
        small_cap = 4096
        base_raw = json.dumps(_valid_config()).encode("utf-8")
        padded = base_raw + b" " * (32768 - len(base_raw))
        pack_dir = self._make_pack(config=padded)
        real_fstat, real_read = os.fstat, os.read

        def _underreporting_fstat(fd):
            st = real_fstat(fd)
            return types.SimpleNamespace(st_mode=st.st_mode, st_size=16)

        for label, fn in (
            ("config_sha256", lambda: loader.config_sha256(pack_dir)),
            ("pack_footprint", lambda: loader.pack_footprint(pack_dir)),
        ):
            with self.subTest(label):
                bytes_read = 0

                def _counting_read(fd, n):
                    nonlocal bytes_read
                    chunk = real_read(fd, min(n, 1024))
                    bytes_read += len(chunk)
                    return chunk

                with (
                    mock.patch.object(loader, "_CONFIG_MAX_BYTES", small_cap),
                    mock.patch("os.fstat", side_effect=_underreporting_fstat),
                    mock.patch("os.read", side_effect=_counting_read),
                ):
                    self._assert_pack_error(fn, "exceeds")
                self.assertGreater(bytes_read, small_cap)
                self.assertLessEqual(bytes_read, small_cap + 1)
                self.assertLess(bytes_read, len(padded))

    def test_load_text_model_refuses_bad_config(self):
        mlx_lm = types.ModuleType("mlx_lm")
        mlx_lm_models = types.ModuleType("mlx_lm.models")
        qwen = types.ModuleType("mlx_lm.models.qwen3_5")
        qwen.TextModel, qwen.TextModelArgs = object, object
        mlx_lm.models, mlx_lm_models.qwen3_5 = mlx_lm_models, qwen
        stubs = {
            "mlx_lm": mlx_lm, "mlx_lm.models": mlx_lm_models, "mlx_lm.models.qwen3_5": qwen,
        }
        with mock.patch.dict(sys.modules, stubs):
            with self.subTest("non-object-json"):
                pack_dir = self._make_pack(config=b"[]")
                self._assert_pack_error(
                    lambda: loader.load_text_model(pack_dir), "not a JSON object"
                )
            with self.subTest("fifo"):
                pack_dir = self._make_pack()
                cfg_path = pack_dir / "config.json"
                self._replace_with_non_regular(cfg_path, "fifo")
                self._assert_pack_error(
                    lambda: loader.load_text_model(pack_dir),
                    str(cfg_path), "not a regular file", bounded=True, timeout=2.0,
                )


if __name__ == "__main__":
    unittest.main()
