"""Fail-closed manifest and source-checkpoint tests for mlx_omarchy_laya.convert."""

import builtins
import contextlib
import errno
import io
import json
import math
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
                self.assertEqual(self.sha_mock.call_count, len(convert.FILES))
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
        ):
            with self.subTest(label), mock.patch("builtins.open", side_effect=_failing_open):
                self._assert_exit(call, "cannot read %s in source snapshot" % name)

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


if __name__ == "__main__":
    unittest.main()
