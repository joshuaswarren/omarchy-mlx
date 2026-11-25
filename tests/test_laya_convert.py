"""Fail-closed manifest tests for mlx_omarchy_laya.convert."""

import contextlib
import io
import json
import os
import sys
import tempfile
import threading
import tracemalloc
import unittest
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "serve"))

from mlx_omarchy_laya import convert  # noqa: E402

_PINNED_STACK = 4 * 1024 * 1024
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


if __name__ == "__main__":
    unittest.main()
