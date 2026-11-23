"""Serve catalog: schema v1 validation, load precedence, GitHub refresh.

Stdlib unittest only. The refresh tests run a local HTTP server so no test
touches the network.
"""

import contextlib
import unittest.mock
import http.server
import io
import json
import os
import sys
import tempfile
import threading
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "serve"))

from mlx_omarchy_serve import catalog  # noqa: E402


def bundled_catalog():
    return catalog.validate_file(catalog.bundled_path())


def entry(**overrides):
    base = {
        "id": "test-model",
        "repo": "mlx-community/Test-Model",
        "revision": "a" * 40,
        "kind": "chat",
        "license": "Apache-2.0",
        "family": "test-family",
        "priority": 1,
        "quant": {"bits": 4, "group_size": 64, "mode": "affine"},
        "memory": {"weights_bytes": 8 * 1024**3, "kv_bytes_per_token": 1024,
                   "peak_estimate_bytes": None},
        "context": {"max_tokens": 8192, "trained_tokens": 32768},
        "capability": {"arch": ["t8103"], "min_mem_gib": None},
        "qualification": {
            "generation": {"status": "qualified", "receipt": "receipts/x.md",
                           "date": "2026-09-20"},
            "http": {"status": "untested", "receipt": None, "date": None},
            "managed": {"status": "untested", "receipt": None, "date": None},
        },
        "recommended": False,
        "serve": {"backend": "mlx-lm", "module": None},
        "availability": {"size_bytes": 8 * 1024**3, "refreshed_at": None},
    }
    for key, value in overrides.items():
        if key == "memory" and isinstance(value, dict):
            base["memory"].update(value)
        else:
            base[key] = value
    return base


def catalog_of(*models, **top):
    obj = {
        "version": 1,
        "generated_at": "2026-09-20T00:00:00Z",
        "source": "https://example.invalid/catalog.json",
        "models": list(models),
    }
    obj.update(top)
    return obj


VALID = catalog_of(entry())
DEEPLY_NESTED = b"[" * 100_000 + b"]" * 100_000  # RecursionError in json.loads


class ValidateTests(unittest.TestCase):
    def test_bundled_seed_is_valid(self):
        obj = bundled_catalog()
        self.assertEqual(obj["version"], 1)
        self.assertTrue(obj["models"])

    def test_valid_fixture_passes(self):
        catalog.validate_catalog(VALID)

    def test_bundled_rejects(self):
        cases = {
            "unknown top key": {**VALID, "extra": 1},
            "bad version": {**VALID, "version": 2},
            "http source": {**VALID, "source": "http://example.invalid/c.json"},
            "bad generated_at": {**VALID, "generated_at": "yesterday"},
            "no models": {**VALID, "models": []},
            "too many models": {**VALID, "models": [entry(id=f"m{i:02d}", priority=i + 1)
                                                    for i in range(catalog.MAX_MODELS + 1)]},
        }
        for name, obj in cases.items():
            with self.subTest(name):
                with self.assertRaises(catalog.CatalogError):
                    catalog.validate_catalog(obj)

    def test_entry_rejects(self):
        cases = {
            "unknown entry key": entry(wat=1),
            "missing key": {k: v for k, v in entry().items() if k != "license"},
            "bad id": entry(id="Qwen Model"),
            "long id": entry(id="x" * 65),
            "duplicate id": catalog_of(entry(), entry()),
            "bad repo": entry(repo="no-slash"),
            "short revision": entry(revision="abc"),
            "upper revision": entry(revision="A" * 40),
            "bad kind": entry(kind="vision"),
            "license too long": entry(license="x" * 33),
            "priority zero": entry(priority=0),
            "duplicate priority in kind": catalog_of(entry(), entry(id="other", priority=1)),
            "bad quant bits": entry(quant={"bits": 3, "group_size": 32, "mode": "mxfp4"}),
            "bad fp shape": entry(quant={"bits": 4, "group_size": 64, "mode": "nvfp4"}),
            "bad affine group": entry(quant={"bits": 4, "group_size": 16, "mode": "affine"}),
            "zero weights": entry(memory={"weights_bytes": 0}),
            "negative kv": entry(memory={"kv_bytes_per_token": -1}),
            "trained not an int": entry(context={"max_tokens": 8192, "trained_tokens": "32k"}),
            "context too large": entry(context={"max_tokens": catalog.MAX_CONTEXT_TOKENS + 1, "trained_tokens": None}),
            "unknown arch": entry(capability={"arch": ["m3"], "min_mem_gib": None}),
            "gen qualified without receipt": entry(qualification={
                "generation": {"status": "qualified", "receipt": None, "date": "2026-09-20"},
                "http": {"status": "untested", "receipt": None, "date": None},
                "managed": {"status": "untested", "receipt": None, "date": None}}),
            "bad qual date": entry(qualification={
                "generation": {"status": "qualified", "receipt": "r", "date": "09/20"},
                "http": {"status": "untested", "receipt": None, "date": None},
                "managed": {"status": "untested", "receipt": None, "date": None}}),
            "http untested with receipt": entry(qualification={
                "generation": {"status": "qualified", "receipt": "r", "date": "2026-09-20"},
                "http": {"status": "untested", "receipt": "r", "date": "2026-09-20"},
                "managed": {"status": "untested", "receipt": None, "date": None}}),
            "recommended without gen qualification": entry(qualification={
                "generation": {"status": "untested", "receipt": None, "date": None},
                "http": {"status": "untested", "receipt": None, "date": None},
                "managed": {"status": "untested", "receipt": None, "date": None}},
                recommended=True),
            "recommended non-bool": entry(recommended="yes"),
            "unknown serve backend": entry(serve={"backend": "sglang", "module": None}),
            "module backend without module": entry(serve={"backend": "module", "module": None}),
            "availability bad size": entry(availability={"size_bytes": -5, "refreshed_at": None}),
        }
        for name, obj in cases.items():
            with self.subTest(name):
                with self.assertRaises(catalog.CatalogError):
                    catalog.validate_catalog(obj)

    def test_version_bool_rejected(self):
        obj = {**VALID, "version": True}
        with self.assertRaises(catalog.CatalogError):
            catalog.validate_catalog(obj)

    def test_priority_duplicate_nonadjacent_rejected(self):
        obj = catalog_of(entry(id="a", priority=1), entry(id="b", priority=2),
                         entry(id="c", priority=1))
        with self.assertRaises(catalog.CatalogError):
            catalog.validate_catalog(obj)

    def test_null_quant_is_honest_full_precision(self):
        catalog.validate_catalog(catalog_of(entry(quant=None)))

    def test_extension_optional_object_or_null(self):
        catalog.validate_catalog(catalog_of(entry()))  # absent
        catalog.validate_catalog(catalog_of(entry(extension=None)))
        catalog.validate_catalog(catalog_of(entry(
            extension={"kv_derivation": "2x10x2x256x2", "variant": "bf16"})))
        with self.assertRaises(catalog.CatalogError):
            catalog.validate_catalog(catalog_of(entry(extension="bf16 notes")))

    def test_managed_qualification_field_validation(self):
        cases = {
            "managed qualified with receipt/date": entry(qualification={
                "generation": {"status": "qualified", "receipt": "r", "date": "2026-09-20"},
                "http": {"status": "qualified", "receipt": "r", "date": "2026-09-20"},
                "managed": {"status": "qualified", "receipt": "mr", "date": "2026-09-20"}}),
            "managed untested without receipt": entry(qualification={
                "generation": {"status": "qualified", "receipt": "r", "date": "2026-09-20"},
                "http": {"status": "qualified", "receipt": "r", "date": "2026-09-20"},
                "managed": {"status": "untested", "receipt": None, "date": None}}),
            "managed qualified without receipt": entry(qualification={
                "generation": {"status": "qualified", "receipt": "r", "date": "2026-09-20"},
                "http": {"status": "qualified", "receipt": "r", "date": "2026-09-20"},
                "managed": {"status": "qualified", "receipt": None, "date": "2026-09-20"}}),
            "managed qualified without date": entry(qualification={
                "generation": {"status": "qualified", "receipt": "r", "date": "2026-09-20"},
                "http": {"status": "qualified", "receipt": "r", "date": "2026-09-20"},
                "managed": {"status": "qualified", "receipt": "r", "date": None}}),
            "managed unknown status": entry(qualification={
                "generation": {"status": "qualified", "receipt": "r", "date": "2026-09-20"},
                "http": {"status": "qualified", "receipt": "r", "date": "2026-09-20"},
                "managed": {"status": "n/a", "receipt": None, "date": None}}),
            "managed untested with receipt": entry(qualification={
                "generation": {"status": "qualified", "receipt": "r", "date": "2026-09-20"},
                "http": {"status": "qualified", "receipt": "r", "date": "2026-09-20"},
                "managed": {"status": "untested", "receipt": "r", "date": "2026-09-20"}}),
            "managed extra key": entry(qualification={
                "generation": {"status": "qualified", "receipt": "r", "date": "2026-09-20"},
                "http": {"status": "qualified", "receipt": "r", "date": "2026-09-20"},
                "managed": {"status": "qualified", "receipt": "r", "date": "2026-09-20",
                            "extra": 1}}),
            "managed missing entirely": entry(qualification={
                "generation": {"status": "qualified", "receipt": "r", "date": "2026-09-20"},
                "http": {"status": "qualified", "receipt": "r", "date": "2026-09-20"}}),
        }
        accept = {"managed qualified with receipt/date",
                  "managed untested without receipt"}
        for name, obj in cases.items():
            with self.subTest(name):
                if name in accept:
                    catalog.validate_catalog(catalog_of(obj))
                else:
                    with self.assertRaises(catalog.CatalogError):
                        catalog.validate_catalog(catalog_of(obj))

    def test_file_size_bound(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "big.json"
            path.write_bytes(b" " + b'"x":' * (catalog.CATALOG_MAX_BYTES // 4))
            with self.assertRaises(catalog.CatalogError):
                catalog.validate_file(path)
            bad = Path(tmp) / "bad.json"
            bad.write_bytes(b"{not json")
            with self.assertRaises(catalog.CatalogError):
                catalog.validate_file(bad)


class LoadTests(unittest.TestCase):
    def test_fallback_to_bundled_on_damaged_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            (home / "cache").mkdir()
            (home / "cache" / "recommended-catalog.json").write_text("{broken")
            obj = catalog.load_catalog(home)
            self.assertEqual(obj["version"], 1)

    def test_valid_cache_wins(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            cache_dir = home / "cache"
            cache_dir.mkdir()
            cached = catalog_of(entry(id="from-cache"))
            (cache_dir / "recommended-catalog.json").write_text(json.dumps(cached))
            obj = catalog.load_catalog(home)
            self.assertEqual(obj["models"][0]["id"], "from-cache")

    def test_no_usable_catalog_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            with unittest.mock.patch.object(catalog, "bundled_path",
                                            return_value=Path(tmp) / "missing.json"):
                with self.assertRaises(catalog.CatalogError):
                    catalog.load_catalog(Path(tmp))


class _Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        store = self.server.catalog_store
        if store.get("fail"):
            self.send_error(500)
            return
        match = self.headers.get("If-None-Match")
        store["if_none_match"].append(match)
        if store.get("before_reply"):
            store.pop("before_reply")()
        if match and store.get("etag") and match == store["etag"] and not store.get("changed"):
            self.send_response(304)
            self.end_headers()
            return
        body = store["body"]
        self.send_response(200)
        if store.get("etag"):
            self.send_header("ETag", store["etag"])
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
        store["requests"] += 1

    def log_message(self, *_args):
        pass


class RefreshTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.home = Path(self.tmp.name)
        # The local test server is not the approved GitHub URL; tests opt
        # into the explicit dev override. Enforcement tests below unset it.
        env_patch = unittest.mock.patch.dict(os.environ,
                                             {catalog.DEV_URL_OVERRIDE_ENV: "1"})
        env_patch.start()
        self.addCleanup(env_patch.stop)
        self.store = {"body": json.dumps(VALID).encode(), "etag": '"v1"',
                      "requests": 0, "fail": False, "changed": False,
                      "if_none_match": []}
        self.server = http.server.HTTPServer(("127.0.0.1", 0), _Handler)
        self.server.catalog_store = self.store
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        self.url = f"http://127.0.0.1:{self.server.server_port}/catalog.json"

    def refresh(self, **kw):
        return catalog.refresh(self.home, url=self.url, timeout=5, **kw)

    def test_fetch_validates_and_writes_atomically(self):
        result = self.refresh(ttl_hours=0)
        self.assertEqual(result["status"], "updated")
        obj = catalog.validate_file(catalog.cache_path(self.home))
        self.assertEqual(obj["version"], 1)
        leftovers = [p.name for p in (self.home / "cache").iterdir()]
        self.assertEqual(sorted(leftovers), ["recommended-catalog.json",
                                             "recommended-catalog.json.etag"])

    def test_ttl_fresh_skips_network(self):
        first = self.refresh(ttl_hours=0)
        self.assertEqual(first["status"], "updated")
        self.store["requests"] = 0
        second = self.refresh(ttl_hours=24)
        self.assertEqual(second["status"], "fresh")
        self.assertEqual(self.store["requests"], 0)

    def test_etag_304_keeps_cache(self):
        self.refresh(ttl_hours=0)
        cache = catalog.cache_path(self.home)
        body = cache.read_bytes()
        expired = cache.stat().st_mtime - 48 * 3600
        os.utime(cache, (expired, expired))
        result = self.refresh(ttl_hours=24)
        self.assertEqual(result["status"], "not-modified")
        self.assertEqual(self.store["if_none_match"][-1], '"v1"')
        self.assertEqual(cache.read_bytes(), body)
        self.assertEqual(self.refresh(ttl_hours=24)["status"], "fresh")  # the 304 renewed the TTL

    def _assert_damaged_cache_refetches(self, damage):
        self.assertEqual(self.refresh(ttl_hours=0)["status"], "updated")
        cache = catalog.cache_path(self.home)
        damage(cache)  # the stored .etag stays behind
        result = self.refresh(ttl_hours=24)  # a damaged cache is never fresh
        self.assertEqual(result["status"], "updated")
        self.assertIsNone(self.store["if_none_match"][-1])
        self.assertEqual(catalog.validate_file(cache)["version"], 1)

    def test_deleted_cache_refetches_despite_stored_etag(self):
        self._assert_damaged_cache_refetches(Path.unlink)

    def test_empty_cache_refetches_despite_stored_etag(self):
        self._assert_damaged_cache_refetches(lambda path: path.write_bytes(b""))

    def test_malformed_pair_reference_refetches_despite_stored_etag(self):
        def damage(path):
            obj = bundled_catalog()
            obj["pairs"][0]["chat_model"] = ["not", "an", "id"]  # unhashable
            path.write_text(json.dumps(obj))
        self._assert_damaged_cache_refetches(damage)

    def test_huge_integer_cache_refetches_despite_stored_etag(self):
        self._assert_damaged_cache_refetches(  # ValueError: int digit limit
            lambda path: path.write_bytes(b'{"version": ' + b"1" * 5000 + b"}"))

    def test_deeply_nested_cache_refetches_despite_stored_etag(self):
        self._assert_damaged_cache_refetches(lambda path: path.write_bytes(DEEPLY_NESTED))

    def test_deeply_nested_field_cache_refetches_despite_stored_etag(self):
        def damage(path):  # parses on 3.14; the validator's error message recurses
            deep = "[" * 50_000 + "]" * 50_000
            path.write_text(json.dumps(bundled_catalog()).replace(
                '"version": 1', '"version": ' + deep, 1))
        self._assert_damaged_cache_refetches(damage)

    def test_304_does_not_recreate_a_cache_deleted_in_flight(self):
        self.refresh(ttl_hours=0)
        cache = catalog.cache_path(self.home)
        self.store["before_reply"] = cache.unlink
        result = self.refresh(ttl_hours=0)
        self.assertEqual(self.store["if_none_match"][-1], '"v1"')
        self.assertEqual(result["status"], "fallback")
        self.assertFalse(cache.exists())

    def test_invalid_body_keeps_last_known_good(self):
        self.refresh(ttl_hours=0)
        self.store["body"] = b"{broken"
        self.store["changed"] = True
        result = self.refresh(ttl_hours=0)
        self.assertEqual(result["status"], "kept")
        self.assertEqual(catalog.load_catalog(self.home)["models"][0]["id"], "test-model")

    def test_fetch_failure_falls_back_to_bundled(self):
        self.store["fail"] = True
        result = self.refresh(ttl_hours=0)
        self.assertEqual(result["status"], "fallback")
        self.assertEqual(catalog.load_catalog(self.home)["models"][0]["id"],
                         bundled_catalog()["models"][0]["id"])

    def test_offline_skips(self):
        result = self.refresh(ttl_hours=0, offline=True)
        self.assertEqual(result["status"], "skipped")
        self.assertEqual(self.store["requests"], 0)

    def _refresh_with_ttl_env(self, value, **kw):
        stderr = io.StringIO()
        env = {"MLX_OMARCHY_CATALOG_TTL_HOURS": value}
        with unittest.mock.patch.dict(os.environ, env), contextlib.redirect_stderr(stderr):
            return self.refresh(**kw), stderr.getvalue()

    def test_bad_ttl_env_does_not_break_offline_refresh(self):
        for value in ("1h", "", "nan", "inf", "-inf"):
            with self.subTest(value=value):
                result, err = self._refresh_with_ttl_env(value, offline=True)
                self.assertEqual(result["status"], "skipped")
                self.assertNotIn("MLX_OMARCHY_CATALOG_TTL_HOURS", err)  # never parsed
        self.assertEqual(self.store["requests"], 0)

    def test_bad_ttl_env_warns_and_keeps_default_ttl(self):
        self.assertEqual(self.refresh(ttl_hours=0)["status"], "updated")
        cache = catalog.cache_path(self.home)
        for value in ("1h", "", "nan", "inf", "-inf"):
            with self.subTest(value=value):
                os.utime(cache)
                result, err = self._refresh_with_ttl_env(value)
                self.assertEqual(result["status"], "fresh")  # younger than 24h
                expired = cache.stat().st_mtime - 48 * 3600
                os.utime(cache, (expired, expired))
                result, _ = self._refresh_with_ttl_env(value)
                self.assertEqual(result["status"], "not-modified")  # older than 24h
                self.assertIn(f"ignoring MLX_OMARCHY_CATALOG_TTL_HOURS={value!r}", err)
                os.utime(cache)
                result, _ = self._refresh_with_ttl_env(value, ttl_hours=0)
                self.assertEqual(result["status"], "not-modified")  # the caller's TTL

    def test_finite_ttl_env_still_overrides(self):
        self.assertEqual(self.refresh(ttl_hours=0)["status"], "updated")
        result, err = self._refresh_with_ttl_env("0", ttl_hours=24)
        self.assertEqual(result["status"], "not-modified")  # 0 forces a refetch
        self.assertNotIn("MLX_OMARCHY_CATALOG_TTL_HOURS", err)
        cache = catalog.cache_path(self.home)
        aged = cache.stat().st_mtime - 30 * 3600
        os.utime(cache, (aged, aged))
        result, err = self._refresh_with_ttl_env("48")
        self.assertEqual(result["status"], "fresh")
        self.assertNotIn("MLX_OMARCHY_CATALOG_TTL_HOURS", err)

    def test_deeply_nested_body_rejected(self):
        self.store["body"] = DEEPLY_NESTED
        self.assertEqual(self.refresh(ttl_hours=0)["status"], "fallback")

    def test_oversized_body_rejected(self):
        self.store["body"] = (json.dumps(VALID) + " " * (catalog.CATALOG_MAX_BYTES + 1)).encode()
        result = self.refresh(ttl_hours=0)
        self.assertEqual(result["status"], "fallback")

    def test_refresh_rejects_non_approved_urls(self):
        with unittest.mock.patch.dict(os.environ, {catalog.DEV_URL_OVERRIDE_ENV: ""}):
            with self.assertRaises(catalog.CatalogError):
                catalog.refresh(self.home, url="https://evil.example/catalog.json",
                                ttl_hours=0)
            with self.assertRaises(catalog.CatalogError):
                catalog.refresh(
                    self.home,
                    url="http://raw.githubusercontent.com/joshuaswarren/omarchy-mlx/main/x.json",
                    ttl_hours=0)
            with self.assertRaises(catalog.CatalogError):
                catalog.refresh(
                    self.home,
                    url="https://raw.githubusercontent.com/other/repo/main/catalog.json",
                    ttl_hours=0)

    def test_dev_url_override_is_explicit_and_nondefault(self):
        with unittest.mock.patch.dict(os.environ,
                                      {catalog.DEV_URL_OVERRIDE_ENV: "1"}):
            result = catalog.refresh(self.home, url="https://127.0.0.1:9/x.json",
                                     ttl_hours=0, timeout=1)
        self.assertEqual(result["status"], "fallback")  # attempted, then kept

    def test_redirect_is_refused_not_followed(self):
        class Redirect(http.server.BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802
                self.send_response(302)
                self.send_header("Location", "https://evil.example/x")
                self.end_headers()

            def log_message(self, *_a):
                pass

        srv = http.server.HTTPServer(("127.0.0.1", 0), Redirect)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        self.addCleanup(srv.server_close)
        self.addCleanup(srv.shutdown)
        result = catalog.refresh(
            self.home, ttl_hours=0, timeout=5,
            url=f"http://127.0.0.1:{srv.server_port}/redirecting")
        self.assertEqual(result["status"], "fallback")  # 3xx is a fetch failure

    def test_changed_body_bypasses_304(self):
        self.refresh(ttl_hours=0)
        self.store["changed"] = True
        result = self.refresh(ttl_hours=0)
        self.assertEqual(result["status"], "updated")





class TrainedKeyOptional(unittest.TestCase):
    def test_an_old_catalog_without_trained_tokens_validates_and_reads_unknown(self):
        old = entry()
        del old["context"]["trained_tokens"]  # a pre-469 cache or custom catalog
        catalog.validate_catalog(catalog_of(old))
        self.assertIsNone(old["context"]["trained_tokens"])  # unknown, normalized to null in place

if __name__ == "__main__":
    unittest.main()
