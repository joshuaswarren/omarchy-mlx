"""Fail-closed manifest and source-checkpoint tests for mlx_omarchy_laya.convert."""

import builtins
import contextlib
import errno
import fcntl
import hashlib
import importlib.util
import io
import json
import math
import os
import stat
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
sys.path.insert(0, str(REPO / "serve"))

from mlx_omarchy_laya import convert  # noqa: E402

_PINNED_STACK = 4 * 1024 * 1024
# Longer than Linux PATH_MAX, so the kernel refuses it before any filesystem lookup.
_OVER_PATH_MAX = "x" * 4096
DEEP_JSON = b"[" * 200_000 + b"]" * 200_000
DEEP_CATALOG_ID_MANIFEST = (
    b'{"catalog_id": ' + b"[" * 20_000 + b"]" * 20_000
    + b', "source_revision": "rev1", "estimated_bytes": 1024,'
    b' "weights_header": {"weight_bytes": 512}}'
)


def _run_pinned(fn):
    """Run fn on a 4 MiB daemon thread, so JSON nesting limits do not follow
    ulimit -s (Python 3.14 bounds recursion by C stack) and a blocking open
    fails the test after 10 s instead of hanging it."""
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
    t.join(10.0)
    if t.is_alive():
        raise TimeoutError("did not finish within 10 s")
    if "error" in outcome:
        raise outcome["error"]
    return outcome["value"]


def _valid_manifest() -> dict:
    return {"catalog_id": "laya-mlx", "source_revision": "55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851",
            "estimated_bytes": 1024 * 1024, "weights_header": {"weight_bytes": 512 * 1024}}


def _make_checkpoint_dir(root: Path, manifest=None) -> Path:
    for rel in convert.FILES:
        dst = root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(b"{}")
    if manifest is not None:
        raw = manifest if isinstance(manifest, bytes) else json.dumps(manifest).encode("utf-8")
        (root / "manifest.json").write_bytes(raw)
    return root


class CheckpointStateTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(sys.set_int_max_str_digits, sys.get_int_max_str_digits())
        sys.set_int_max_str_digits(4300)

    def _tmp(self) -> Path:
        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        return Path(td.name)

    def _new_dir(self, manifest=None) -> Path:
        return _make_checkpoint_dir(self._tmp(), manifest=manifest)

    def _state(self, d):
        return _run_pinned(lambda: convert.checkpoint_state(d))

    def _assert_manifest_error(self, d, reason):
        state, details = self._state(d)
        self.assertEqual(state, "invalid")
        self.assertIn(reason, details.get("manifest_error", ""), details)

    def test_no_manifest_is_raw(self):
        state, details = self._state(self._new_dir())
        self.assertEqual(state, "raw")
        self.assertIn("reason", details)

    def test_valid_manifest_is_converted(self):
        manifest = _valid_manifest()
        expected = ("converted", {"catalog_id": manifest["catalog_id"],
                                  "estimated_bytes": manifest["estimated_bytes"]})
        with self.subTest("regular"):
            self.assertEqual(self._state(self._new_dir(manifest)), expected)
        with self.subTest("symlink"):
            target = self._tmp() / "real-manifest.json"
            target.write_text(json.dumps(manifest), encoding="utf-8")
            d = self._new_dir()
            os.symlink(target, d / "manifest.json")
            self.assertEqual(self._state(d), expected)

    def test_damaged_content_is_invalid(self):
        not_json, not_object = "not valid JSON", "not a JSON object"
        for name, payload, reason in [
            ("corrupt-json", b"{not json", not_json),
            ("invalid-utf8", b'{"catalog_id": "\xff"}', not_json),
            ("huge-int", b'{"estimated_bytes": ' + b"1" * 5000 + b"}", not_json),
            ("deep-nesting", DEEP_JSON, not_json),
            ("list-nonempty", b"[1]", not_object), ("string-nonempty", b'"x"', not_object),
            ("int-nonzero", b"42", not_object), ("list-empty", b"[]", not_object),
            ("null", b"null", not_object), ("false", b"false", not_object),
            ("zero", b"0", not_object), ("string-empty", b'""', not_object),
        ]:
            with self.subTest(name):
                self._assert_manifest_error(self._new_dir(payload), reason)

    def test_empty_object_lists_all_missing_fields(self):
        expected = ["catalog_id", "source_revision", "estimated_bytes",
                    "weights_header.weight_bytes"]
        self.assertEqual(self._state(self._new_dir(b"{}")),
                         ("invalid", {"missing_manifest_fields": expected}))

    def test_non_regular_or_oversized_manifest_is_invalid(self):
        oversized = json.dumps({**_valid_manifest(), "padding": "a" * (1 << 20)}).encode("utf-8")
        self.assertGreater(len(oversized), 1 << 20)
        for name, build, reason in [
            ("directory", lambda p: p.mkdir(), "unreadable"),
            ("dangling-symlink", lambda p: os.symlink(p.parent / "does-not-exist.json", p),
             "unreadable"),
            ("oversized", lambda p: p.write_bytes(oversized), "exceeds"),
        ]:
            with self.subTest(name):
                d = self._new_dir()
                build(d / "manifest.json")
                self._assert_manifest_error(d, reason)

    def test_large_manifest_is_rejected_without_reading_it_all(self):
        d = self._new_dir()
        with open(d / "manifest.json", "wb") as f:
            f.truncate(64 << 20)  # sparse: no disk use
        if not tracemalloc.is_tracing():
            tracemalloc.start()
            self.addCleanup(tracemalloc.stop)
        tracemalloc.reset_peak()
        before = tracemalloc.get_traced_memory()[0]
        self._assert_manifest_error(d, "exceeds")
        self.assertLess(tracemalloc.get_traced_memory()[1] - before, 16 << 20)

    def test_dangling_symlink_with_failing_lstat_is_invalid(self):
        d = self._new_dir()
        os.symlink(d / "does-not-exist.json", d / "manifest.json")
        denied = PermissionError(13, "lstat denied")
        with mock.patch("os.lstat", side_effect=denied):
            self._assert_manifest_error(d, "lstat denied")

    def test_fifo_manifest_is_invalid_without_blocking(self):
        d = self._new_dir()
        os.mkfifo(d / "manifest.json")
        self._assert_manifest_error(d, "not a regular file")

    @unittest.skipIf(os.geteuid() == 0, "root ignores mode 000 permissions")
    def test_unreadable_mode_000_manifest_is_invalid(self):
        d = self._new_dir(_valid_manifest())
        mpath = d / "manifest.json"
        mpath.chmod(0)
        self.addCleanup(mpath.chmod, 0o600)
        self._assert_manifest_error(d, "unreadable")

    def test_mistyped_fields_are_invalid(self):
        for name, override, field in [
            ("catalog_id-int", {"catalog_id": 1}, "catalog_id"),
            ("source_revision-int", {"source_revision": 1}, "source_revision"),
            ("estimated_bytes-str", {"estimated_bytes": "1"}, "estimated_bytes"),
            ("estimated_bytes-bool", {"estimated_bytes": True}, "estimated_bytes"),
            ("weight_bytes-str", {"weights_header": {"weight_bytes": "512"}},
             "weights_header.weight_bytes"),
            ("weight_bytes-bool", {"weights_header": {"weight_bytes": True}},
             "weights_header.weight_bytes"),
        ]:
            with self.subTest(name):
                d = self._new_dir({**_valid_manifest(), **override})
                self.assertEqual(self._state(d), ("invalid", {"missing_manifest_fields": [field]}))

    def test_cli_check_prints_invalid_json(self):
        for name, payload in [("corrupt-json", b"{not json"), ("empty-object", b"{}"),
                              ("deep-catalog-id", DEEP_CATALOG_ID_MANIFEST)]:
            with self.subTest(name):
                d, buf = self._new_dir(payload), io.StringIO()

                def run_cli():
                    with contextlib.redirect_stdout(buf):
                        convert.main(["--check", str(d)])

                _run_pinned(run_cli)
                out = json.loads(buf.getvalue())
                self.assertEqual(out["state"], "invalid")
                if name == "deep-catalog-id":  # 3.11 stops at decode, 3.14 at the type check
                    self.assertTrue(out.get("missing_manifest_fields") == ["catalog_id"]
                                    or "manifest_error" in out, out)

    @unittest.skipIf(os.geteuid() == 0, "root ignores mode 000 permissions")
    def test_untraversable_dir_is_invalid(self):
        cases = [
            ("dir-000", None, 0, "model.safetensors"),
            ("dir-0400", None, 0o400, "model.safetensors"),
            ("parent-000", "parent", 0, "model.safetensors"),
            ("encoder-000", "encoder", 0, "encoder/config.json"),
            ("tokenizer-000", "tokenizer", 0, "tokenizer/tokenizer.json"),
        ]
        for label, target, mode, expected_name in cases:
            with self.subTest(label):
                if target == "parent":
                    parent = self._tmp()
                    d = _make_checkpoint_dir(parent / "model")
                    lock_path = parent
                else:
                    d = self._new_dir()
                    lock_path = d if target is None else d / target
                lock_path.chmod(mode)
                self.addCleanup(lock_path.chmod, 0o700)
                self.assertEqual(
                    self._state(d),
                    ("invalid", {"unreadable_file": "%s is unreadable: %s"
                                 % (expected_name, os.strerror(errno.EACCES))}),
                )

    def test_unstatable_path_is_invalid(self):
        with self.subTest("name-too-long"):
            d = self._tmp() / _OVER_PATH_MAX
            self.assertEqual(
                self._state(d),
                ("invalid", {"unreadable_file": "model.safetensors is unreadable: "
                             + os.strerror(errno.ENAMETOOLONG)}),
            )
        with self.subTest("symlink-loop"):
            d = self._new_dir()
            weights = d / "model.safetensors"
            weights.unlink()
            os.symlink("model.safetensors", weights)
            self.assertEqual(
                self._state(d),
                ("invalid", {"unreadable_file": "model.safetensors is unreadable: "
                             + os.strerror(errno.ELOOP)}),
            )

    @unittest.skipIf(os.geteuid() == 0, "root ignores mode 000 permissions")
    def test_cli_check_prints_unreadable_json(self):
        d = self._new_dir()
        d.chmod(0)
        self.addCleanup(d.chmod, 0o700)
        buf = io.StringIO()

        def run_cli():
            with contextlib.redirect_stdout(buf):
                convert.main(["--check", str(d)])

        _run_pinned(run_cli)
        self.assertEqual(
            json.loads(buf.getvalue()),
            {"state": "invalid",
             "unreadable_file": "model.safetensors is unreadable: "
             + os.strerror(errno.EACCES)},
        )

    def test_missing_file_reports_missing_files(self):
        with self.subTest("absent-file"):
            d = self._new_dir()
            (d / "tokenizer/tokenizer.json").unlink()
            self.assertEqual(
                self._state(d),
                ("invalid", {"missing_files": ["tokenizer/tokenizer.json"]}),
            )
        with self.subTest("file-at-encoder-dir"):
            d = self._new_dir()
            (d / "encoder/config.json").unlink()
            (d / "encoder").rmdir()
            (d / "encoder").write_bytes(b"{}")
            self.assertEqual(
                self._state(d),
                ("invalid", {"missing_files": ["encoder/config.json"]}),
            )
        with self.subTest("embedded-nul"):
            d = Path(str(self._tmp()) + "/a\0b")
            self.assertEqual(self._state(d), ("invalid", {"missing_files": convert.FILES}))


def _tiny_enc_cfg() -> dict:
    return {"model_type": "modernbert", "architectures": ["ModernBertModel"],
            "hidden_size": 4, "intermediate_size": 8,
            "num_hidden_layers": 1, "vocab_size": 10}


def _tiny_agent_cfg() -> dict:
    return {"head_layers": 1, "max_len": 128, "head_max_len": 64}


def _tiny_header() -> dict:
    h, inter, vocab = 4, 8, 10
    shapes = {
        "encoder.embeddings.tok_embeddings.weight": (vocab, h),
        "encoder.embeddings.norm.weight": (h,), "encoder.final_norm.weight": (h,),
        "type_emb.weight": (3, h),
        "scorer.0.weight": (h,), "scorer.0.bias": (h,),
        "scorer.1.weight": (h, h), "scorer.1.bias": (h,),
        "scorer.3.weight": (1, h), "scorer.3.bias": (1,),
        "act_head.0.weight": (256, h + 4), "act_head.0.bias": (256,),
        "act_head.2.weight": (2, 256), "act_head.2.bias": (2,),
        "temperature": (3,),
        **{"encoder.layers.0." + k: s for k, s in (
            ("attn.Wqkv.weight", (3 * h, h)), ("attn.Wo.weight", (h, h)),
            ("mlp.Wi.weight", (2 * inter, h)), ("mlp.Wo.weight", (h, inter)),
            ("mlp_norm.weight", (h,)),
        )},
        "head.layers.0.self_attn.in_proj_weight": (3 * h, h),
        "head.layers.0.self_attn.in_proj_bias": (3 * h,),
    }
    for k, s in (("self_attn.out_proj", (h, h)), ("norm1", (h,)), ("norm2", (h,)),
                 ("linear1", (4 * h, h)), ("linear2", (h, 4 * h))):
        shapes["head.layers.0.%s.weight" % k] = s
        shapes["head.layers.0.%s.bias" % k] = (s[0],)
    header, offset = {}, 0
    for name, shape in shapes.items():
        nbytes = math.prod(shape) * 4
        header[name] = {"dtype": "F32", "shape": list(shape), "data_offsets": [offset, offset + nbytes]}
        offset += nbytes
    return header


def _as_bytes(val) -> bytes:
    return val if isinstance(val, bytes) else json.dumps(val).encode("utf-8")


def _pack_header(payload) -> bytes:
    raw = _as_bytes(payload)
    return struct.pack("<Q", len(raw)) + raw


def _make_source_snapshot(root: Path, enc_cfg=None, agent_cfg=None, weights=None, skip=()) -> Path:
    entries = {
        "encoder/config.json": _as_bytes(_tiny_enc_cfg() if enc_cfg is None else enc_cfg),
        "rl_agent_config.json": _as_bytes(_tiny_agent_cfg() if agent_cfg is None else agent_cfg),
        "tokenizer/tokenizer.json": b"{}",
        "tokenizer/tokenizer_config.json": b"{}",
    }
    for rel, content in entries.items():
        if rel in skip:
            continue
        dst = root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(content)
    if "model.safetensors" not in skip:
        weights_path = root / "model.safetensors"
        if callable(weights):
            weights(weights_path)
        else:
            weights_path.write_bytes(_pack_header(_tiny_header()) if weights is None else weights)
    return root


class _ConvertBase(unittest.TestCase):
    def setUp(self):
        self.addCleanup(sys.set_int_max_str_digits, sys.get_int_max_str_digits())
        sys.set_int_max_str_digits(4300)
        self.enterContext(mock.patch.object(convert, "_apply_aliases", side_effect=lambda h: h))

    def _tmp(self) -> Path:
        return Path(self.enterContext(tempfile.TemporaryDirectory()))

    def _assert_exit(self, fn, reason):
        with self.assertRaises(SystemExit) as ctx:
            _run_pinned(fn)
        msg = str(ctx.exception)
        self.assertTrue(msg.startswith("convert: "), msg)
        self.assertIn(reason, msg)
        return msg


class ValidateWeightsTests(_ConvertBase):
    def _weights(self, raw: bytes) -> Path:
        path = self._tmp() / "model.safetensors"
        path.write_bytes(raw)
        return path

    def test_valid_header_and_existing_errors(self):
        cfg = _tiny_enc_cfg()
        with self.subTest("valid"):
            path = self._weights(_pack_header(_tiny_header()))
            tensors = _run_pinned(lambda: convert._validate_weights(path, cfg, 1))
            self.assertEqual(len(tensors), len(_tiny_header()))
            self.assertEqual(tensors["temperature"], {"dtype": "F32", "shape": [3]})
        with self.subTest("missing-tensor"):
            hdr = _tiny_header()
            del hdr["temperature"]
            path = self._weights(_pack_header(hdr))
            self._assert_exit(lambda: convert._validate_weights(path, cfg, 1), "missing expected tensors")
        with self.subTest("shape-mismatch"):
            hdr = _tiny_header()
            hdr["temperature"]["shape"] = [4]
            path = self._weights(_pack_header(hdr))
            self._assert_exit(lambda: convert._validate_weights(path, cfg, 1), "shape mismatch")

    def test_malformed_header_is_rejected(self):
        cfg = _tiny_enc_cfg()
        not_json, not_obj, bad_meta, trunc, exceeds = (
            "not valid JSON", "not a JSON object", "malformed metadata",
            "is truncated", "exceeds",
        )

        def _mutated(key, value):
            return _pack_header({**_tiny_header(), key: value})

        rows = [
            ("empty-file", b"", "too short"),
            ("3-byte-file", b"\x01\x02\x03", "too short"),
            ("length-2^64-1", struct.pack("<Q", (1 << 64) - 1), exceeds),
            ("length-2^40", struct.pack("<Q", 1 << 40), exceeds),
            ("length-past-eof", struct.pack("<Q", 3) + b"{}", trunc),
            ("length-0", struct.pack("<Q", 0), not_json),
            ("corrupt-json", _pack_header(b"{not json"), not_json),
            ("invalid-utf8", _pack_header(b'{"a": "\xff"}'), not_json),
            ("huge-int", _pack_header(b'{"a": ' + b"1" * 5000 + b"}"), not_json),
            ("deep-json", _pack_header(DEEP_JSON), not_json),
            ("header-list", _pack_header(b"[]"), not_obj),
            ("header-str", _pack_header(b'"x"'), not_obj),
            ("header-null", _pack_header(b"null"), not_obj),
            ("entry-int", _mutated("temperature", 1), bad_meta),
            ("entry-no-shape", _mutated("temperature", {"dtype": "F32"}), bad_meta),
            ("shape-str", _mutated("temperature", {"dtype": "F32", "shape": "3"}), bad_meta),
            ("shape-bool", _mutated("scorer.3.bias", {"dtype": "F32", "shape": [True]}), bad_meta),
            ("shape-neg", _mutated("temperature", {"dtype": "F32", "shape": [-1]}), bad_meta),
            ("entry-no-dtype", _mutated("temperature", {"shape": [3]}), bad_meta),
            ("dtype-int", _mutated("temperature", {"dtype": 1, "shape": [3]}), bad_meta),
            ("long-key-truncated", _mutated("k" * 200, 1), bad_meta),
        ]
        for name, raw, reason in rows:
            with self.subTest(name):
                path = self._weights(raw)
                msg = self._assert_exit(lambda p=path: convert._validate_weights(p, cfg, 1), reason)
                if name == "long-key-truncated":
                    self.assertIn("k" * 80, msg)
                    self.assertNotIn("k" * 81, msg)
        with self.subTest("over-cap"):
            n = 100_000_000 + 1
            path = self._weights(struct.pack("<Q", n))
            real_fstat = os.fstat
            with mock.patch(
                "os.fstat",
                side_effect=lambda fd: types.SimpleNamespace(
                    st_mode=real_fstat(fd).st_mode, st_size=8 + n,
                ),
            ):
                self._assert_exit(
                    lambda: convert._validate_weights(path, cfg, 1),
                    "exceeds the 100000000-byte safetensors limit",
                )
        with self.subTest("short-read"):
            body = _as_bytes(_tiny_header())
            path = self._weights(struct.pack("<Q", len(body) + 10) + body)
            real_fstat = os.fstat

            def _inflated_fstat(fd):
                st = real_fstat(fd)
                return types.SimpleNamespace(st_mode=st.st_mode, st_size=st.st_size + 10)

            with mock.patch("os.fstat", side_effect=_inflated_fstat):
                self._assert_exit(lambda: convert._validate_weights(path, cfg, 1), trunc)
        with self.subTest("fifo"):
            path = self._tmp() / "model.safetensors"
            os.mkfifo(path)
            self._assert_exit(
                lambda: convert._validate_weights(path, cfg, 1),
                "model.safetensors in source snapshot is not a regular file",
            )

    def test_huge_header_length_does_not_allocate(self):
        cfg = _tiny_enc_cfg()
        if not tracemalloc.is_tracing():
            tracemalloc.start()
            self.addCleanup(tracemalloc.stop)
        for name, claimed, reason in [
            ("at-cap", 100_000_000, "is truncated"),
            ("2^40", 1 << 40, "exceeds"),
            ("2^64-1", (1 << 64) - 1, "exceeds"),
        ]:
            with self.subTest(name):
                path = self._weights(struct.pack("<Q", claimed))
                tracemalloc.reset_peak()
                before = tracemalloc.get_traced_memory()[0]
                self._assert_exit(lambda p=path: convert._validate_weights(p, cfg, 1), reason)
                self.assertLess(tracemalloc.get_traced_memory()[1] - before, 4 << 20)

    def test_excess_layer_counts_are_rejected(self):
        path = self._weights(_pack_header(_tiny_header()))
        with self.subTest("num_hidden_layers"):
            cfg = {**_tiny_enc_cfg(), "num_hidden_layers": 7}
            self._assert_exit(
                lambda: convert._validate_weights(path, cfg, 1),
                "need more tensors than the header",
            )
        with self.subTest("head_layers"):
            cfg = _tiny_enc_cfg()
            self._assert_exit(
                lambda: convert._validate_weights(path, cfg, 3),
                "need more tensors than the header",
            )

    def test_bad_encoder_config_values_are_rejected(self):
        path = self._weights(_pack_header(_tiny_header()))
        missing = object()
        bad_vals = (("missing", missing), ("str", "4"), ("bool", True),
                    ("zero", 0), ("neg", -1), ("float", 4.0))
        huge_vals = (("int64-max", 1 << 63), ("huge-4300", 10**4300 - 1))
        for key in ("hidden_size", "intermediate_size", "num_hidden_layers", "vocab_size"):
            rows = bad_vals if key == "num_hidden_layers" else bad_vals + huge_vals
            for label, val in rows:
                with self.subTest("%s-%s" % (key, label)):
                    cfg = _tiny_enc_cfg()
                    if val is missing:
                        del cfg[key]
                    else:
                        cfg[key] = val
                    self._assert_exit(lambda c=cfg: convert._validate_weights(path, c, 1),
                                      "encoder/config.json %s must be a positive integer" % key)


class ConvertTests(_ConvertBase):
    def setUp(self):
        super().setUp()
        self.enterContext(mock.patch.dict(
            sys.modules, {"mlx": types.ModuleType("mlx"), "mlx.core": types.ModuleType("mlx.core")}
        ))
        self.emit_mock = self.enterContext(
            mock.patch.object(convert, "_emit_weights", return_value=(False, "0" * 64))
        )
        self.sha_mock = self.enterContext(
            mock.patch.object(convert, "_sha256", wraps=convert._sha256)
        )

    def _call_convert(self, src: Path, out: Path):
        return convert.convert(src, "convaiinnovations/laya", "rev0123456789", out, "laya-mlx")

    def test_valid_snapshot_returns_manifest(self):
        for label, archs, expected_arch in (
            ("default", ["ModernBertModel"], "ModernBertModel"),
            ("empty-architectures", [], None),
            ("int-architectures", 5, None),
            ("null-architectures", None, None),
            ("int-first-architecture", [5], None),
            ("nested-list-architecture", [[["ModernBertModel"]]], None),
            ("max-len-architecture", ["A" * 256], "A" * 256),
            ("over-len-architecture", ["A" * 257], None),
            ("long-non-ascii-architecture", ["\u00e9" * 300_000], None),
        ):
            with self.subTest(label):
                self.emit_mock.reset_mock()
                self.sha_mock.reset_mock()
                base = self._tmp()
                enc_raw = json.dumps(
                    {**_tiny_enc_cfg(), "architectures": archs},
                    ensure_ascii=False,
                ).encode("utf-8")
                self.assertLessEqual(len(enc_raw), 1 << 20)
                src = _make_source_snapshot(base / "src", enc_cfg=enc_raw)
                with contextlib.redirect_stdout(io.StringIO()):
                    manifest = _run_pinned(lambda s=src, o=base / "out": self._call_convert(s, o))
                self.assertEqual(manifest["context_max_tokens"], 128)
                self.assertEqual(manifest["head_max_tokens"], 64)
                self.assertEqual(manifest["encoder_architecture"], expected_arch)
                self.assertEqual(list(manifest["source_files_sha256"]), convert.FILES)
                self.assertEqual(self.sha_mock.call_count, len(convert.FILES) - 2)
                self.emit_mock.assert_called_once()
                manifest_raw = (base / "out" / "manifest.json").read_bytes()
                self.assertLessEqual(len(manifest_raw), convert._MANIFEST_MAX_BYTES)
                self.assertEqual(json.loads(manifest_raw.decode("utf-8")), manifest)

    def test_unsupported_model_type_message_is_bounded(self):
        missing = object()
        rows = (
            ("dict", {"a": 1}, ("dict",), ("{'a': 1}",)),
            ("long-str", "x" * 1000, ("x" * 80,), ("x" * 81,)),
            ("missing", missing, ("model_type None",), ()),
        )
        for label, model_type, expected_parts, forbidden_parts in rows:
            with self.subTest(label):
                self.emit_mock.reset_mock()
                self.sha_mock.reset_mock()
                base = self._tmp()
                enc_cfg = _tiny_enc_cfg()
                if model_type is missing:
                    del enc_cfg["model_type"]
                else:
                    enc_cfg["model_type"] = model_type
                src = _make_source_snapshot(base / "src", enc_cfg=enc_cfg)
                with contextlib.redirect_stdout(io.StringIO()):
                    msg = self._assert_exit(
                        lambda s=src, o=base / "out": self._call_convert(s, o),
                        "is not modernbert",
                    )
                for part in expected_parts:
                    self.assertIn(part, msg)
                for part in forbidden_parts:
                    self.assertNotIn(part, msg)
                self.emit_mock.assert_not_called()
                self.sha_mock.assert_not_called()

    def test_malformed_snapshot_is_rejected(self):
        def _agent_with(**overrides):
            return {k: v for k, v in {**_tiny_agent_cfg(), **overrides}.items() if v is not None}

        enc_bytes = _as_bytes(_tiny_enc_cfg())
        oversized_enc = b" " * ((1 << 20) + 1 - len(enc_bytes)) + enc_bytes
        pos = "rl_agent_config.json %s must be a positive integer"
        rows = [
            ("encoder-corrupt", {"enc_cfg": b"{not json"}, "encoder/config.json is not valid JSON"),
            ("encoder-list", {"enc_cfg": b"[]"}, "encoder/config.json is not a JSON object"),
            ("encoder-oversized", {"enc_cfg": oversized_enc}, "exceeds"),
            ("agent-corrupt", {"agent_cfg": b"{not json"}, "rl_agent_config.json is not valid JSON"),
            ("agent-null", {"agent_cfg": b"null"}, "rl_agent_config.json is not a JSON object"),
            ("head_layers-missing", {"agent_cfg": _agent_with(head_layers=None)}, pos % "head_layers"),
            ("head_layers-str", {"agent_cfg": _agent_with(head_layers="x")}, pos % "head_layers"),
            ("head_layers-zero", {"agent_cfg": _agent_with(head_layers=0)}, pos % "head_layers"),
            ("max_len-missing", {"agent_cfg": _agent_with(max_len=None)}, pos % "max_len"),
            ("head_max_len-bool", {"agent_cfg": _agent_with(head_max_len=True)}, pos % "head_max_len"),
            ("missing-tokenizer", {"skip": ("tokenizer/tokenizer.json",)},
             "cannot read tokenizer/tokenizer.json in source snapshot"),
            ("weights-dir", {"weights": lambda p: p.mkdir()},
             "cannot read model.safetensors in source snapshot"),
            ("weights-fifo", {"weights": lambda p: os.mkfifo(p)},
             "model.safetensors in source snapshot is not a regular file"),
            ("weights-1-byte", {"weights": b"x"}, "too short"),
        ]
        for name, kwargs, reason in rows:
            with self.subTest(name):
                self.emit_mock.reset_mock()
                self.sha_mock.reset_mock()
                base = self._tmp()
                src = _make_source_snapshot(base / "src", **kwargs)
                with contextlib.redirect_stdout(io.StringIO()):
                    self._assert_exit(lambda s=src, o=base / "out": self._call_convert(s, o), reason)
                self.emit_mock.assert_not_called()
                self.assertFalse((base / "out").exists())
                if name != "missing-tokenizer":
                    self.sha_mock.assert_not_called()

    @unittest.skipIf(os.geteuid() == 0, "root ignores mode 000 permissions")
    def test_unreadable_mode_000_weights_is_rejected(self):
        base = self._tmp()
        src = _make_source_snapshot(base / "src")
        wpath = src / "model.safetensors"
        wpath.chmod(0)
        self.addCleanup(wpath.chmod, 0o600)
        with contextlib.redirect_stdout(io.StringIO()):
            self._assert_exit(
                lambda: self._call_convert(src, base / "out"),
                "cannot read model.safetensors in source snapshot",
            )
        self.emit_mock.assert_not_called()
        self.sha_mock.assert_not_called()

    def test_read_error_after_open_is_rejected(self):
        path = self._tmp() / "source.bin"
        path.write_bytes(b"{}")
        real_open = builtins.open

        class _FailingRead:
            def __init__(self, f):
                self._f = f

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return self._f.__exit__(*exc)

            def fileno(self):
                return self._f.fileno()

            def close(self):
                return self._f.close()

            def read(self, *args, **kwargs):
                raise OSError(errno.EIO, "Input/output error")

        def _failing_open(file, *args, **kwargs):
            f = real_open(file, *args, **kwargs)
            return _FailingRead(f) if file == path else f

        for label, call, name in (
            ("sha256", lambda: convert._sha256(path, "encoder/config.json"),
             "encoder/config.json"),
            ("read_config", lambda: convert._read_config(path, "encoder/config.json"),
             "encoder/config.json"),
            ("validate_weights", lambda: convert._validate_weights(path, _tiny_enc_cfg(), 1),
             "model.safetensors"),
            ("copy_source", lambda: convert._copy_source(
                path, path.parent / "copy.json", "encoder/config.json", "0" * 64),
             "encoder/config.json"),
        ):
            with self.subTest(label), mock.patch("builtins.open", side_effect=_failing_open):
                self._assert_exit(call, "cannot read %s in source snapshot" % name)
        self.assertEqual(sorted(p.name for p in path.parent.iterdir()), ["source.bin"])

    def test_large_config_is_rejected_without_reading_it_all(self):
        path = self._tmp() / "config.json"
        chunk = b" " * (1 << 20)
        with open(path, "wb") as f:
            f.write(b"{}")
            for _ in range(8):
                f.write(chunk)
        if not tracemalloc.is_tracing():
            tracemalloc.start()
            self.addCleanup(tracemalloc.stop)
        tracemalloc.reset_peak()
        before = tracemalloc.get_traced_memory()[0]
        self._assert_exit(
            lambda: convert._read_config(path, "encoder/config.json"),
            "exceeds",
        )
        self.assertLess(tracemalloc.get_traced_memory()[1] - before, 4 << 20)

    def test_extra_entries_in_snapshot_are_not_copied(self):
        def _outside_link(root: Path, dst: Path):
            secret = root / "outside.txt"
            secret.write_bytes(b"outside-secret")
            os.symlink(secret, dst)

        def _mode_000_file(root: Path, dst: Path):
            dst.write_bytes(b"secret")
            dst.chmod(0)
            self.addCleanup(dst.chmod, 0o600)

        def _mode_000_dir(root: Path, dst: Path):
            dst.mkdir()
            dst.chmod(0)
            self.addCleanup(dst.chmod, 0o700)

        cases = [
            ("regular-file", "encoder/extra.txt", lambda r, p: p.write_bytes(b"extra"), False),
            ("fifo", "tokenizer/extra.fifo", lambda r, p: os.mkfifo(p), False),
            ("dev-null-symlink", "encoder/null.bin", lambda r, p: os.symlink("/dev/null", p), False),
            ("dir-loop-symlink", "tokenizer/loop", lambda r, p: os.symlink(".", p), False),
            ("outside-symlink", "tokenizer/outside.txt", _outside_link, False),
            ("mode-000-file", "encoder/unreadable.txt", _mode_000_file, True),
            ("mode-000-dir", "tokenizer/unreadable_dir", _mode_000_dir, True),
        ]
        for label, rel, setup, skip_root in cases:
            with self.subTest(label):
                if skip_root and os.geteuid() == 0:
                    self.skipTest("root ignores mode 000 permissions")
                base = self._tmp()
                src = _make_source_snapshot(base / "src")
                out = base / "out"
                setup(base, src / rel)
                with contextlib.redirect_stdout(io.StringIO()):
                    _run_pinned(lambda s=src, o=out: self._call_convert(s, o))
                self.assertTrue((out / "manifest.json").is_file())
                self.assertEqual(sorted(p.name for p in (out / "encoder").iterdir()), ["config.json"])
                self.assertEqual(
                    sorted(p.name for p in (out / "tokenizer").iterdir()),
                    ["tokenizer.json", "tokenizer_config.json"],
                )

    def test_hf_cache_blob_symlinks_copy_as_regular_files(self):
        base = self._tmp()
        src = _make_source_snapshot(base / "src")
        blobs = base / "blobs"
        blobs.mkdir()
        expected = {}
        for name in convert.FILES:
            p = src / name
            data = p.read_bytes()
            digest = hashlib.sha256(data).hexdigest()
            blob = blobs / digest
            blob.write_bytes(data)
            p.unlink()
            os.symlink(os.path.relpath(blob, p.parent), p)
            expected[name] = (data, digest)
        out = base / "out"
        with contextlib.redirect_stdout(io.StringIO()):
            manifest = _run_pinned(lambda: self._call_convert(src, out))
        for name, (data, digest) in expected.items():
            self.assertEqual(manifest["source_files_sha256"][name], digest)
            if name == "model.safetensors":
                continue
            dst = out / name
            self.assertTrue(dst.is_file())
            self.assertFalse(dst.is_symlink())
            self.assertEqual(dst.read_bytes(), data)

    def test_copied_bytes_match_hashed_and_validated_bytes(self):
        with self.subTest("unchanged"):
            base = self._tmp()
            src = _make_source_snapshot(base / "src")
            out = base / "out"
            with contextlib.redirect_stdout(io.StringIO()):
                manifest = _run_pinned(lambda: self._call_convert(src, out))
            for name in convert.FILES:
                if name == "model.safetensors":
                    continue
                actual = hashlib.sha256((out / name).read_bytes()).hexdigest()
                self.assertEqual(actual, manifest["source_files_sha256"][name])

        with self.subTest("rerun-replaces-outputs"):
            (src / "tokenizer/tokenizer.json").write_bytes(b'{"v": 2}')
            with contextlib.redirect_stdout(io.StringIO()):
                _run_pinned(lambda: self._call_convert(src, out))
            self.assertEqual((out / "tokenizer/tokenizer.json").read_bytes(), b'{"v": 2}')

        for target_name in ("rl_agent_config.json", "tokenizer/tokenizer.json"):
            with self.subTest("changed-during-emit-%s" % target_name):
                base = self._tmp()
                src = _make_source_snapshot(base / "src")
                out = base / "out"
                target = src / target_name

                def _emit_and_rewrite(s, d, p=target):
                    p.write_bytes(p.read_bytes().replace(b"}", b', "mutated": 1}', 1))
                    return False, "0" * 64

                with mock.patch.object(convert, "_emit_weights", side_effect=_emit_and_rewrite):
                    with contextlib.redirect_stdout(io.StringIO()):
                        self._assert_exit(lambda s=src, o=out: self._call_convert(s, o),
                                          "%s changed in the source snapshot during conversion" % target_name)
                self.assertFalse((out / "manifest.json").exists())
                self.assertFalse((out / target_name).exists())

        with self.subTest("changed-after-read-config-encoder"):
            base = self._tmp()
            src = _make_source_snapshot(base / "src")
            out = base / "out"
            real_validate = convert._validate_weights

            def _validate_and_rewrite(*args, **kwargs):
                res = real_validate(*args, **kwargs)
                enc = src / "encoder/config.json"
                enc.write_bytes(enc.read_bytes().replace(b"}", b', "mutated": 1}', 1))
                return res

            with mock.patch.object(convert, "_validate_weights", side_effect=_validate_and_rewrite):
                with contextlib.redirect_stdout(io.StringIO()):
                    self._assert_exit(
                        lambda s=src, o=out: self._call_convert(s, o),
                        "%s changed in the source snapshot during conversion" % "encoder/config.json",
                    )
            self.assertFalse((out / "manifest.json").exists())
            self.assertFalse((out / "encoder/config.json").exists())

    def test_failed_rerun_removes_stale_manifest(self):
        base = self._tmp()
        src = _make_source_snapshot(base / "src")
        out = base / "out"
        with contextlib.redirect_stdout(io.StringIO()):
            _run_pinned(lambda: self._call_convert(src, out))
        self.assertTrue((out / "manifest.json").is_file())
        err = OSError(errno.ENOSPC, os.strerror(errno.ENOSPC))
        with mock.patch.object(convert, "_emit_weights", side_effect=err):
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(OSError) as ctx:
                _run_pinned(lambda: self._call_convert(src, out))
        self.assertEqual(ctx.exception.errno, errno.ENOSPC)
        self.assertFalse((out / "manifest.json").exists())

    def test_validation_failure_on_rerun_preserves_manifest(self):
        base = self._tmp()
        src = _make_source_snapshot(base / "src")
        out = base / "out"
        with contextlib.redirect_stdout(io.StringIO()):
            _run_pinned(lambda: self._call_convert(src, out))
        old_manifest = (out / "manifest.json").read_bytes()
        (src / "rl_agent_config.json").write_bytes(b"{not json")
        with contextlib.redirect_stdout(io.StringIO()):
            self._assert_exit(
                lambda: self._call_convert(src, out),
                "rl_agent_config.json is not valid JSON",
            )
        self.assertTrue((out / "manifest.json").is_file())
        self.assertEqual((out / "manifest.json").read_bytes(), old_manifest)

    def test_out_that_is_the_source_dir_is_refused(self):
        for label, link in (("same-path", False), ("symlink-to-source", True)):
            with self.subTest(label):
                base = self._tmp()
                src = _make_source_snapshot(base / "src")
                before = {n: (src / n).read_bytes() for n in convert.FILES}
                out = src
                if link:
                    out = base / "out"
                    os.symlink(src, out)
                with contextlib.redirect_stdout(io.StringIO()):
                    self._assert_exit(lambda o=out: self._call_convert(src, o),
                                      "is the source snapshot directory")
                self.assertEqual({n: (src / n).read_bytes() for n in convert.FILES}, before)
                self.emit_mock.assert_not_called()
                self.sha_mock.assert_not_called()

    def test_existing_outputs_are_replaced_not_written_through(self):
        base = self._tmp()
        src = _make_source_snapshot(base / "src")
        out = base / "out"
        with contextlib.redirect_stdout(io.StringIO()):
            _run_pinned(lambda: self._call_convert(src, out))
        outside = base / "outside.json"
        outside.write_bytes(b"outside")
        (out / "rl_agent_config.json").unlink()
        os.mkfifo(out / "rl_agent_config.json")
        (out / "tokenizer/tokenizer.json").unlink()
        os.symlink(outside, out / "tokenizer/tokenizer.json")
        with contextlib.redirect_stdout(io.StringIO()):
            _run_pinned(lambda: self._call_convert(src, out))
        self.assertEqual(outside.read_bytes(), b"outside")
        for name in ("rl_agent_config.json", "tokenizer/tokenizer.json"):
            dst = out / name
            self.assertFalse(dst.is_symlink())
            self.assertTrue(dst.is_file())
            self.assertEqual(dst.read_bytes(), (src / name).read_bytes())

    def test_failed_output_write_leaves_previous_file_and_no_part(self):
        real_replace = os.replace
        for target in ("tokenizer/tokenizer.json", "manifest.json"):
            with self.subTest(target):
                base = self._tmp()
                src = _make_source_snapshot(base / "src")
                out = base / "out"
                with contextlib.redirect_stdout(io.StringIO()):
                    _run_pinned(lambda: self._call_convert(src, out))
                old = (out / target).read_bytes()
                (src / "tokenizer/tokenizer.json").write_bytes(b'{"v": 2}')
                err = OSError(errno.ENOSPC, os.strerror(errno.ENOSPC))

                def _replace(a, b, t=out / target):
                    if Path(b) == t:
                        raise err
                    return real_replace(a, b)

                with mock.patch("os.replace", side_effect=_replace), \
                        contextlib.redirect_stdout(io.StringIO()), \
                        self.assertRaises(OSError) as ctx:
                    _run_pinned(lambda: self._call_convert(src, out))
                self.assertEqual(ctx.exception.errno, errno.ENOSPC)
                self.assertFalse((out / "manifest.json").exists())
                if target != "manifest.json":
                    self.assertEqual((out / target).read_bytes(), old)
                parts = [p for p in out.rglob("*") if p.name.endswith(".part")]
                self.assertEqual(parts, [])

    def test_failed_copy_write_stays_an_output_error(self):
        base = self._tmp()
        src = _make_source_snapshot(base / "src")
        out = base / "out"
        with contextlib.redirect_stdout(io.StringIO()):
            _run_pinned(lambda: self._call_convert(src, out))
        old = (out / "tokenizer/tokenizer.json").read_bytes()
        (src / "tokenizer/tokenizer.json").write_bytes(b'{"v": 2}')
        real_open = builtins.open
        part = out / "tokenizer/.tokenizer.json.part"

        class _FullDisk:
            def __init__(self, f):
                self._f = f

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return self._f.__exit__(*exc)

            def write(self, data):
                raise OSError(errno.ENOSPC, os.strerror(errno.ENOSPC))

        def _open(file, *args, **kwargs):
            f = real_open(file, *args, **kwargs)
            if isinstance(file, (str, os.PathLike)) and Path(file) == part:
                return _FullDisk(f)
            return f

        with mock.patch("builtins.open", side_effect=_open), \
                contextlib.redirect_stdout(io.StringIO()), \
                self.assertRaises(OSError) as ctx:
            _run_pinned(lambda: self._call_convert(src, out))
        self.assertEqual(ctx.exception.errno, errno.ENOSPC)
        self.assertEqual((out / "tokenizer/tokenizer.json").read_bytes(), old)
        self.assertEqual([p for p in out.rglob("*") if p.name.endswith(".part")], [])
        self.assertFalse((out / "manifest.json").exists())

    def test_concurrent_convert_into_same_out_is_refused(self):
        base = self._tmp()
        src = _make_source_snapshot(base / "src")
        out = base / "out"
        with contextlib.redirect_stdout(io.StringIO()):
            _run_pinned(lambda: self._call_convert(src, out))
        before_entries = sorted(p.relative_to(out) for p in out.rglob("*"))
        before_files = {
            p.relative_to(out): p.read_bytes()
            for p in sorted(out.rglob("*")) if p.is_file()
        }
        self.assertIn(Path("manifest.json"), before_files)
        (src / "tokenizer/tokenizer.json").write_bytes(b'{"v": 2}')
        fd = os.open(out, os.O_RDONLY)
        self.addCleanup(os.close, fd)
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.emit_mock.reset_mock()
        with contextlib.redirect_stdout(io.StringIO()):
            self._assert_exit(
                lambda: self._call_convert(src, out),
                "another convert is writing to %s" % out,
            )
        self.emit_mock.assert_not_called()
        self.assertEqual(sorted(p.relative_to(out) for p in out.rglob("*")), before_entries)
        self.assertEqual(
            {p.relative_to(out): p.read_bytes() for p in sorted(out.rglob("*")) if p.is_file()},
            before_files,
        )


class FromLocalCliTests(_ConvertBase):
    def _run_from_local(self, src: Path, out: Path, expected: str):
        with mock.patch.object(convert, "convert") as convert_mock:
            msg = self._assert_exit(
                lambda: convert.main(["--from-local", str(src), "--out", str(out)]),
                expected,
            )
        self.assertEqual(msg, expected)
        convert_mock.assert_not_called()

    @unittest.skipIf(os.geteuid() == 0, "root ignores mode 000 permissions")
    def test_from_local_untraversable_source_exits_with_reason(self):
        for label, lock_parent in (("source-000", False), ("parent-000", True)):
            with self.subTest(label):
                base = self._tmp()
                parent = base / "parent"
                src = _make_source_snapshot(parent / "src" if lock_parent else base / "src")
                lock_path = parent if lock_parent else src
                lock_path.chmod(0)
                self.addCleanup(lock_path.chmod, 0o700)
                self._run_from_local(
                    src,
                    base / "out",
                    "convert: cannot read model.safetensors in %s (%s)"
                    % (src, os.strerror(errno.EACCES)),
                )

    def test_from_local_unstatable_source_exits_with_reason(self):
        with self.subTest("name-too-long"):
            base = self._tmp()
            src = base / _OVER_PATH_MAX
            self._run_from_local(
                src,
                base / "out",
                "convert: cannot read model.safetensors in %s (%s)"
                % (src, os.strerror(errno.ENAMETOOLONG)),
            )
        with self.subTest("symlink-loop"):
            base = self._tmp()
            src = _make_source_snapshot(base / "src")
            weights = src / "model.safetensors"
            weights.unlink()
            os.symlink("model.safetensors", weights)
            self._run_from_local(
                src,
                base / "out",
                "convert: cannot read model.safetensors in %s (%s)"
                % (src, os.strerror(errno.ELOOP)),
            )

    def test_from_local_missing_weights_keeps_does_not_hold_message(self):
        base = self._tmp()
        src = base / "empty"
        src.mkdir()
        self._run_from_local(
            src,
            base / "out",
            "convert: %s does not hold model.safetensors" % src,
        )


class EmitWeightsTests(_ConvertBase):
    def setUp(self):
        super().setUp()
        self.addCleanup(os.umask, os.umask(0o022))
        pkg = sys.modules["mlx_omarchy_laya"]
        had_model_mod = "mlx_omarchy_laya.model" in sys.modules
        prev_model_mod = sys.modules.get("mlx_omarchy_laya.model")
        had_model_attr = hasattr(pkg, "model")
        prev_model_attr = getattr(pkg, "model", None)

        def _restore_model():
            if had_model_mod:
                sys.modules["mlx_omarchy_laya.model"] = prev_model_mod
            else:
                sys.modules.pop("mlx_omarchy_laya.model", None)
            if had_model_attr:
                pkg.model = prev_model_attr
            elif hasattr(pkg, "model"):
                delattr(pkg, "model")

        self.addCleanup(_restore_model)
        try:
            has_mlx = importlib.util.find_spec("mlx") is not None
        except ValueError:
            has_mlx = False
        if not has_mlx:
            self.enterContext(mock.patch.dict(
                sys.modules,
                {"mlx": types.ModuleType("mlx"), "mlx.core": types.ModuleType("mlx.core")},
            ))

    @staticmethod
    @contextlib.contextmanager
    def _umask(mask: int):
        prev = os.umask(mask)
        try:
            yield
        finally:
            os.umask(prev)

    @staticmethod
    def _serialize(tensors: dict) -> bytes:
        return "".join("%s=%s\n" % (k, tensors[k]) for k in sorted(tensors)).encode("utf-8")

    def _default_save(self, tensors: dict, path: str):
        with open(path, "wb") as f:
            f.write(self._serialize(tensors))

    def _fake_safetensors(self, save_file=None, tensors=None):
        data = (
            {"scorer.layers.0.weight": "w0", "temperature": "t0"}
            if tensors is None else tensors
        )
        writer = self._default_save if save_file is None else save_file
        pkg = types.ModuleType("safetensors")
        pkg.__path__ = []
        st_np = types.ModuleType("safetensors.numpy")
        st_np.load_file = lambda path: dict(data)
        st_np.save_file = writer
        pkg.numpy = st_np
        return mock.patch.dict(sys.modules, {"safetensors": pkg, "safetensors.numpy": st_np})

    def _make_src_out(self) -> tuple[Path, Path, Path]:
        base = self._tmp()
        src = base / "src.safetensors"
        src.write_bytes(b"src")
        out = base / "out"
        out.mkdir()
        return base, src, out

    @staticmethod
    def _replace_file_mode(out: Path) -> int:
        ref = out / "mode_ref.json"
        with convert._replace_file(ref) as f:
            f.write(b"{}")
        return stat.S_IMODE(os.lstat(ref).st_mode)

    @staticmethod
    def _parts(root: Path) -> list[Path]:
        return [p for p in root.rglob("*") if p.name.endswith(".part")]

    def test_emit_weights_writes_canonical_keys_and_hash(self):
        # Pin: passes on base.
        _, src, out = self._make_src_out()
        dst = out / "model.safetensors"
        seen = []

        def _save(tensors, path):
            seen.append(sorted(tensors))
            self._default_save(tensors, path)

        with self._fake_safetensors(save_file=_save):
            aliases, digest = _run_pinned(lambda: convert._emit_weights(src, dst))
        expected = self._serialize({"scorer.0.weight": "w0", "temperature": "t0"})
        self.assertTrue(stat.S_ISREG(os.lstat(dst).st_mode))
        self.assertTrue(aliases)
        self.assertEqual(seen, [["scorer.0.weight", "temperature"]])
        self.assertEqual(dst.read_bytes(), expected)
        self.assertEqual(digest, hashlib.sha256(expected).hexdigest())
        self.assertEqual(self._parts(out), [])

    def test_emit_weights_mode_matches_replace_file_outputs(self):
        def _atomic_080_save(tensors, path):
            p = Path(path)
            fd, tmp = tempfile.mkstemp(dir=p.parent)
            try:
                with os.fdopen(fd, "wb") as f:
                    f.write(self._serialize(tensors))
                os.replace(tmp, p)
            except BaseException:
                Path(tmp).unlink(missing_ok=True)
                raise

        for u in (0o022, 0o027, 0o077, 0o222):
            for label, writer in (
                ("in-place", self._default_save),
                ("atomic-0600", _atomic_080_save),
            ):
                with self.subTest(writer=label, umask=oct(u)):
                    _, src, out = self._make_src_out()
                    dst = out / "model.safetensors"
                    with self._umask(u):
                        expected_mode = self._replace_file_mode(out)
                        with self._fake_safetensors(save_file=writer):
                            aliases, digest = _run_pinned(lambda: convert._emit_weights(src, dst))
                    st = os.lstat(dst)
                    self.assertTrue(stat.S_ISREG(st.st_mode))
                    self.assertEqual(stat.S_IMODE(st.st_mode), expected_mode)
                    self.assertTrue(aliases)
                    self.assertEqual(digest, hashlib.sha256(dst.read_bytes()).hexdigest())
                    self.assertEqual(self._parts(out), [])

    def test_emit_weights_hash_read_failure_preserves_previous_file(self):
        real_open = builtins.open

        class _FailingRead:
            def __init__(self, f, err):
                self._f = f
                self._err = err

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return self._f.__exit__(*exc)

            def fileno(self):
                return self._f.fileno()

            def close(self):
                return self._f.close()

            def read(self, *args, **kwargs):
                raise self._err

        for label, exc in (
            ("eio", OSError(errno.EIO, os.strerror(errno.EIO))),
            ("keyboard-interrupt", KeyboardInterrupt("interrupted")),
        ):
            with self.subTest(label):
                _, src, out = self._make_src_out()
                dst = out / "model.safetensors"
                dst.write_bytes(b"previous-good-weights")
                saved = []

                def _save(tensors, path):
                    self._default_save(tensors, path)
                    saved.append(path)

                def _failing_open(file, *args, err=exc, **kwargs):
                    f = real_open(file, *args, **kwargs)
                    mode = args[0] if args else kwargs.get("mode", "r")
                    if mode == "rb" and str(file).endswith(".part"):
                        return _FailingRead(f, err)
                    return f

                with self._fake_safetensors(save_file=_save), \
                        mock.patch.object(convert, "open", create=True, side_effect=_failing_open), \
                        self.assertRaises(type(exc)) as ctx:
                    _run_pinned(lambda: convert._emit_weights(src, dst))
                self.assertIs(ctx.exception, exc)
                self.assertEqual(len(saved), 1)
                self.assertEqual(dst.read_bytes(), b"previous-good-weights")
                self.assertEqual(self._parts(out), [])

    def test_emit_weights_failed_write_preserves_previous_file(self):
        for label, exc in (
            ("enospc", OSError(errno.ENOSPC, os.strerror(errno.ENOSPC))),
            ("keyboard-interrupt", KeyboardInterrupt("interrupted")),
        ):
            with self.subTest(label):
                _, src, out = self._make_src_out()
                dst = out / "model.safetensors"
                dst.write_bytes(b"previous-good-weights")

                def _partial_fail(tensors, path, err=exc):
                    with open(path, "wb") as f:
                        f.write(b"partial-bytes")
                    raise err

                with self._fake_safetensors(save_file=_partial_fail):
                    with self.assertRaises(type(exc)) as ctx:
                        _run_pinned(lambda: convert._emit_weights(src, dst))
                self.assertIs(ctx.exception, exc)
                self.assertEqual(dst.read_bytes(), b"previous-good-weights")
                self.assertEqual(self._parts(out), [])

    def test_emit_weights_replaces_symlink_to_outside_file(self):
        base, src, out = self._make_src_out()
        outside = base / "outside.bin"
        outside.write_bytes(b"outside-secret")
        dst = out / "model.safetensors"
        os.symlink(outside, dst)
        with self._fake_safetensors():
            _, digest = _run_pinned(lambda: convert._emit_weights(src, dst))
        self.assertEqual(outside.read_bytes(), b"outside-secret")
        self.assertTrue(stat.S_ISREG(os.lstat(dst).st_mode))
        self.assertEqual(digest, hashlib.sha256(dst.read_bytes()).hexdigest())
        self.assertEqual(self._parts(out), [])

    def test_emit_weights_replaces_fifo_without_blocking(self):
        _, src, out = self._make_src_out()
        dst = out / "model.safetensors"
        os.mkfifo(dst)
        with self._fake_safetensors():
            _, digest = _run_pinned(lambda: convert._emit_weights(src, dst))
        self.assertTrue(stat.S_ISREG(os.lstat(dst).st_mode))
        self.assertEqual(digest, hashlib.sha256(dst.read_bytes()).hexdigest())
        self.assertEqual(self._parts(out), [])

    def test_emit_weights_replaces_dangling_symlink(self):
        base, src, out = self._make_src_out()
        target = base / "nowhere" / "target.bin"
        dst = out / "model.safetensors"
        os.symlink(target, dst)
        with self._fake_safetensors():
            _, digest = _run_pinned(lambda: convert._emit_weights(src, dst))
        self.assertFalse(target.exists())
        self.assertTrue(stat.S_ISREG(os.lstat(dst).st_mode))
        self.assertEqual(digest, hashlib.sha256(dst.read_bytes()).hexdigest())
        self.assertEqual(self._parts(out), [])

    def test_emit_weights_directory_at_dst_raises_oserror(self):
        # Pin: passes on base.
        _, src, out = self._make_src_out()
        dst = out / "model.safetensors"
        dst.mkdir()
        keep = dst / "keep"
        keep.write_bytes(b"keep")
        with self._fake_safetensors(), self.assertRaises(IsADirectoryError):
            _run_pinned(lambda: convert._emit_weights(src, dst))
        self.assertTrue(stat.S_ISDIR(os.lstat(dst).st_mode))
        self.assertEqual(keep.read_bytes(), b"keep")
        self.assertEqual(self._parts(out), [])

    def test_emit_weights_cleans_stale_part_file(self):
        _, src, out = self._make_src_out()
        stale = out / ".model.safetensors.part"
        stale.write_bytes(b"stale-from-killed-run")
        dst = out / "model.safetensors"
        with self._fake_safetensors():
            _, digest = _run_pinned(lambda: convert._emit_weights(src, dst))
        self.assertTrue(stat.S_ISREG(os.lstat(dst).st_mode))
        self.assertEqual(digest, hashlib.sha256(dst.read_bytes()).hexdigest())
        self.assertFalse(stale.exists())
        self.assertEqual(self._parts(out), [])

    def test_emit_weights_rejects_non_regular_part(self):
        with self.subTest("fifo"):
            _, src, out = self._make_src_out()
            dst = out / "model.safetensors"
            dst.write_bytes(b"previous-good-weights")

            def _put_fifo(tensors, path):
                os.unlink(path)
                os.mkfifo(path)

            with self._fake_safetensors(save_file=_put_fifo):
                self._assert_exit(
                    lambda: convert._emit_weights(src, dst),
                    "staged weights file .model.safetensors.part is not a regular file",
                )
            self.assertEqual(dst.read_bytes(), b"previous-good-weights")
            self.assertEqual(self._parts(out), [])

        with self.subTest("symlink"):
            base, src, out = self._make_src_out()
            outside = base / "outside.bin"
            outside.write_bytes(b"outside-secret")
            dst = out / "model.safetensors"
            dst.write_bytes(b"previous-good-weights")

            def _put_symlink(tensors, path):
                os.unlink(path)
                os.symlink(outside, path)

            with self._fake_safetensors(save_file=_put_symlink), self.assertRaises(OSError) as ctx:
                _run_pinned(lambda: convert._emit_weights(src, dst))
            self.assertEqual(ctx.exception.errno, errno.ELOOP)
            self.assertEqual(outside.read_bytes(), b"outside-secret")
            self.assertEqual(dst.read_bytes(), b"previous-good-weights")
            self.assertEqual(self._parts(out), [])

    def test_convert_failed_weights_emit_preserves_previous_weights(self):
        base = self._tmp()
        src = _make_source_snapshot(base / "src")
        out = base / "out"
        with self._fake_safetensors(), contextlib.redirect_stdout(io.StringIO()):
            manifest = _run_pinned(
                lambda: convert.convert(src, "convaiinnovations/laya", "rev0123456789", out, "laya-mlx")
            )
        first_weights = (out / "model.safetensors").read_bytes()
        self.assertEqual(manifest["weights_sha256"], hashlib.sha256(first_weights).hexdigest())
        self.assertTrue(manifest["key_aliases_applied"])
        self.assertTrue((out / "manifest.json").is_file())

        err = OSError(errno.ENOSPC, os.strerror(errno.ENOSPC))

        def _partial_enospc(tensors, path):
            with open(path, "wb") as f:
                f.write(b"partial-enospc")
            raise err

        with self._fake_safetensors(save_file=_partial_enospc), \
                contextlib.redirect_stdout(io.StringIO()), \
                self.assertRaises(OSError) as ctx:
            _run_pinned(
                lambda: convert.convert(src, "convaiinnovations/laya", "rev0123456789", out, "laya-mlx")
            )
        self.assertIs(ctx.exception, err)
        self.assertEqual((out / "model.safetensors").read_bytes(), first_weights)
        self.assertFalse((out / "manifest.json").exists())
        self.assertEqual(self._parts(out), [])

    @unittest.skipUnless(
        importlib.util.find_spec("safetensors") and importlib.util.find_spec("numpy"),
        "requires safetensors and numpy",
    )
    def test_emit_weights_real_safetensors_replaces_symlink_and_matches_mode(self):
        import numpy as np
        from safetensors.numpy import load_file, save_file

        base = self._tmp()
        src = base / "src.safetensors"
        orig = {
            "scorer.layers.0.weight": np.array([1.0, 2.0, 3.0], dtype=np.float32),
            "temperature": np.array([0.5, 1.5, 2.5], dtype=np.float32),
        }
        save_file(orig, str(src))
        outside = base / "outside.bin"
        outside.write_bytes(b"outside-weights")

        for u in (0o022, 0o027, 0o077, 0o222):
            with self.subTest(umask=oct(u)):
                out = base / ("out-%03o" % u)
                out.mkdir()
                dst = out / "model.safetensors"
                os.symlink(outside, dst)
                with self._umask(u):
                    expected_mode = self._replace_file_mode(out)
                    aliases, digest = _run_pinned(lambda d=dst: convert._emit_weights(src, d))
                self.assertTrue(aliases)
                self.assertEqual(outside.read_bytes(), b"outside-weights")
                st = os.lstat(dst)
                self.assertTrue(stat.S_ISREG(st.st_mode))
                self.assertEqual(stat.S_IMODE(st.st_mode), expected_mode)
                loaded = load_file(str(dst))
                self.assertEqual(sorted(loaded), ["scorer.0.weight", "temperature"])
                np.testing.assert_array_equal(
                    loaded["scorer.0.weight"], orig["scorer.layers.0.weight"]
                )
                np.testing.assert_array_equal(loaded["temperature"], orig["temperature"])
                self.assertEqual(digest, hashlib.sha256(dst.read_bytes()).hexdigest())
                self.assertEqual(self._parts(out), [])


if __name__ == "__main__":
    unittest.main()
