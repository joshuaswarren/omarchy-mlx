"""Serve catalog: schema v1 validation, load precedence, GitHub refresh.

Stdlib unittest only. The refresh tests run a local HTTP server so no test
touches the network.
"""

import contextlib
import errno
import gc
import hashlib
import unittest.mock
import http.server
import io
import json
import os
import stat
import sys
import tempfile
import threading
import tracemalloc
import unittest
import warnings
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


def pair_catalog(pair_id="everyday"):
    pair = {"id": pair_id, "label": "Everyday", "chat_model": "test-model",
            "decision_model": "test-decisions", "priority": 1, "max_questions": 4,
            "routing_policy": "v1",
            "qualification": {"status": "untested", "receipt": None, "date": None}}
    return catalog_of(entry(), entry(id="test-decisions", kind="decisions"),
                      pairs=[pair])


def selection_evidence(**overrides):
    evidence = {
        "arch": ["t8103"], "backend": "mlx-lm", "chat_revision": "a" * 40,
        "decision_revision": "a" * 40, "date": "2026-09-20",
        "quality": {"task": "t", "score": 0.5, "receipt": "r"},
        "latency": {"decode_tokens_per_sec": 1, "target_tokens_per_sec": 1,
                    "first_visible_p95_ms": 1, "receipt": "r"},
        "runtime": {"mlx_version": "0", "extension_sha256": "b" * 64,
                    "libmlx_sha256": "b" * 64},
    }
    evidence.update(overrides)
    return evidence


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

    def test_patterns_reject_trailing_newline(self):
        # `$` also matches before a final "\n". Revision, date and SHA-256
        # fields hit their exact length caps first, so pin each pattern here.
        cases = (
            (catalog.ID_PATTERN, "test-model"),
            (catalog.REPO_PATTERN, "mlx-community/Test-Model"),
            (catalog.REVISION_PATTERN, "a" * 40),
            (catalog.DATE_PATTERN, "2026-09-20"),
            (catalog.SHA256_PATTERN, "b" * 64),
        )
        for pattern, good in cases:
            with self.subTest(pattern.pattern):
                self.assertIsNotNone(pattern.match(good))
                self.assertIsNone(pattern.match(good + "\n"))

    def test_dates_reject_non_ascii_digits(self):
        # str `\d` matches any Unicode decimal digit; a catalog date is ISO ASCII.
        qualified = {"status": "qualified", "receipt": "r", "date": "2026-09-20"}
        untested = {"status": "untested", "receipt": None, "date": None}

        def model_date(date):
            return catalog_of(entry(qualification={
                "generation": {**qualified, "date": date},
                "http": untested, "managed": untested}))

        def pair_date(date):
            obj = pair_catalog()
            obj["pairs"][0]["qualification"] = {**qualified, "date": date}
            return obj

        def evidence_date(date):
            obj = pair_catalog()
            obj["pairs"][0]["extension"] = {
                "selection_evidence": selection_evidence(date=date)}
            return obj

        for build in (model_date, pair_date, evidence_date):
            catalog.validate_catalog(build("2026-09-20"))
            for date in ("\u0662\u0660\u0662\u0666-\u0660\u0669-\u0662\u0660",  # Arabic-Indic
                         "\uff12\uff10\uff12\uff16-\uff10\uff19-\uff12\uff10",  # full-width
                         "\u0968\u0966\u0968\u096c-\u0966\u096f-\u0968\u0966",  # Devanagari
                         "2026-09-2\u0660"):  # one stray digit
                with self.subTest(build.__name__, date=date):
                    with self.assertRaisesRegex(catalog.CatalogError,
                                                "does not match required format"):
                        catalog.validate_catalog(build(date))

    def test_pair_numbers_reject_nan_and_infinity(self):
        # json.loads reads NaN/Infinity tokens; NaN <= 0 and inf <= 0 are both False.
        def with_value(field, value):
            obj = pair_catalog()
            pair = obj["pairs"][0]
            pair["first_text_budget_ms"] = 2000
            pair["extension"] = {"selection_evidence": selection_evidence()}
            if field == "first_text_budget_ms":
                pair[field] = value
            else:
                pair["extension"]["selection_evidence"]["latency"][field] = value
            return obj

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "catalog.json"
            for field in ("first_text_budget_ms", "decode_tokens_per_sec",
                          "target_tokens_per_sec", "first_visible_p95_ms"):
                for value in (1, 10**400):  # a huge int is finite, not an OverflowError
                    path.write_text(json.dumps(with_value(field, value)))
                    catalog.validate_file(path)
                for value in (float("nan"), float("inf"), float("-inf")):
                    with self.subTest(field, value=value):
                        path.write_text(json.dumps(with_value(field, value)))
                        with self.assertRaisesRegex(catalog.CatalogError,
                                                    "expected finite number > 0"):
                            catalog.validate_file(path)

    def test_ids_and_repo_reject_trailing_newline(self):
        catalog.validate_catalog(pair_catalog())
        cases = {
            "model id": catalog_of(entry(id="test-model\n")),
            "model repo": catalog_of(entry(repo="mlx-community/Test-Model\n")),
            "pair id": pair_catalog(pair_id="everyday\n"),
        }
        for name, obj in cases.items():
            with self.subTest(name):
                with self.assertRaisesRegex(catalog.CatalogError,
                                            "does not match required format"):
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
            exact = Path(tmp) / "exact.json"
            raw = catalog.bundled_path().read_bytes()
            exact.write_bytes(raw + b" " * (catalog.CATALOG_MAX_BYTES - len(raw)))
            catalog.validate_file(exact)  # the cap itself is accepted
            bad = Path(tmp) / "bad.json"
            bad.write_bytes(b"{not json")
            with self.assertRaises(catalog.CatalogError):
                catalog.validate_file(bad)

    def test_oversized_sparse_file_rejected_without_reading(self):
        if tracemalloc.is_tracing():  # the peak belongs to whoever started tracing
            self.skipTest("tracemalloc is already tracing")
        size = 64 * 1024 * 1024
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sparse.json"
            with open(path, "wb") as fh:
                fh.truncate(size)
            tracemalloc.start()
            try:
                with self.assertRaisesRegex(
                    catalog.CatalogError,
                    f"{size} bytes exceeds {catalog.CATALOG_MAX_BYTES}",
                ):
                    catalog.validate_file(path)
                _, peak = tracemalloc.get_traced_memory()
            finally:
                tracemalloc.stop()
            self.assertLess(peak, 4 * 1024 * 1024)

    def test_short_reads_are_joined(self):
        expected = bundled_catalog()
        real_open = open

        def short_open(*args, **kwargs):  # a short read is not end of file
            fh = real_open(*args, **kwargs)
            read = fh.read
            fh.read = lambda n=-1: read(min(n, 4096))
            return fh

        with unittest.mock.patch.object(catalog, "open", short_open, create=True):
            self.assertEqual(catalog.validate_file(catalog.bundled_path()), expected)

    def test_file_growth_after_fstat_rejected_with_bounded_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "grew.json"
            path.write_bytes(b" " * (catalog.CATALOG_MAX_BYTES + 4096))
            real_fstat = os.fstat
            dups = []  # a dup shares the file offset, so it shows how far the read went

            def small_fstat(fd):
                dups.append(os.dup(fd))
                self.addCleanup(os.close, dups[-1])
                vals = list(real_fstat(fd))
                vals[stat.ST_SIZE] = 16
                return os.stat_result(vals)

            with unittest.mock.patch.object(catalog.os, "fstat", side_effect=small_fstat):
                with self.assertRaisesRegex(
                    catalog.CatalogError,
                    f"{catalog.CATALOG_MAX_BYTES + 1} bytes exceeds {catalog.CATALOG_MAX_BYTES}",
                ):
                    catalog.validate_file(path)
            self.assertEqual(os.lseek(dups[0], 0, os.SEEK_CUR), catalog.CATALOG_MAX_BYTES + 1)


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

    def test_cache_with_non_ascii_date_falls_back_to_bundled(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            cache_dir = home / "cache"
            cache_dir.mkdir()
            untested = {"status": "untested", "receipt": None, "date": None}
            cached = catalog_of(entry(id="from-cache", qualification={
                "generation": {"status": "qualified", "receipt": "r",
                               "date": "٢٠٢٦-٠٩-٢٠"},
                "http": untested, "managed": untested}))
            (cache_dir / "recommended-catalog.json").write_text(
                json.dumps(cached, ensure_ascii=False), encoding="utf-8")
            obj = catalog.load_catalog(home)
            self.assertNotEqual(obj["models"][0]["id"], "from-cache")
            self.assertEqual(obj, bundled_catalog())

    def test_no_usable_catalog_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            with unittest.mock.patch.object(catalog, "bundled_path",
                                            return_value=Path(tmp) / "missing.json"):
                with self.assertRaises(catalog.CatalogError):
                    catalog.load_catalog(Path(tmp))

    @unittest.skipUnless(hasattr(os, "mkfifo"), "requires os.mkfifo")
    def test_fifo_cache_is_rejected_without_blocking(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            cache = catalog.cache_path(home)
            cache.parent.mkdir()
            os.mkfifo(cache)
            outcome: dict[str, object] = {}

            def run():
                try:
                    with self.assertRaisesRegex(catalog.CatalogError, "not a regular file"):
                        catalog.validate_file(cache)
                    outcome["loaded"] = catalog.load_catalog(home)
                except BaseException as exc:
                    outcome["error"] = exc

            worker = threading.Thread(target=run, daemon=True)
            worker.start()
            worker.join(timeout=10.0)
            self.assertFalse(worker.is_alive(), "FIFO catalog read blocked")
            if "error" in outcome:
                raise outcome["error"]
            self.assertEqual(outcome["loaded"], bundled_catalog())

    def test_symlink_to_valid_catalog_loads(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            target = home / "target-catalog.json"
            cached = catalog_of(entry(id="via-symlink"))
            target.write_text(json.dumps(cached), encoding="utf-8")
            cache = catalog.cache_path(home)
            cache.parent.mkdir()
            cache.symlink_to(target)
            self.assertEqual(catalog.validate_file(cache)["models"][0]["id"], "via-symlink")
            self.assertEqual(catalog.load_catalog(home)["models"][0]["id"], "via-symlink")


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
        if store.get("force_304") or (
            match and store.get("etag") and match == store["etag"] and not store.get("changed")
        ):
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

    def test_latin1_etag_is_revalidated(self):
        # obs-text (0x80-0xFF) is valid in an entity-tag, including non-printing 0xA0.
        for tag in ('"caf\u00e9"', '"a\u00a0b"'):
            with self.subTest(tag=tag):
                self.store["etag"] = tag
                self.assertEqual(self.refresh(ttl_hours=0)["status"], "updated")
                self.assertEqual(self.refresh(ttl_hours=0)["status"], "not-modified")
                self.assertEqual(self.store["if_none_match"][-1], tag)

    def _etag_path(self):
        return catalog.cache_path(self.home).with_name("recommended-catalog.json.etag")

    def _cache_digest(self):
        return hashlib.sha256(catalog.cache_path(self.home).read_bytes()).hexdigest()

    def _bound_etag_bytes(self, tag, digest=None):
        if digest is None:
            digest = self._cache_digest()
        raw = tag if isinstance(tag, bytes) else tag.encode("utf-8")
        return raw + b"\n" + digest.encode("ascii") + b"\n"

    def _write_bound_etag(self, tag, digest=None):
        self._etag_path().write_bytes(self._bound_etag_bytes(tag, digest))

    def test_damaged_etag_is_ignored(self):
        self.assertEqual(self.refresh(ttl_hours=0)["status"], "updated")
        etag = self._etag_path()
        digest = self._cache_digest()
        # Undecodable, header line breaks, text a header cannot encode, control
        # characters, surrounding whitespace (no strip on read), and anything
        # that is not one quoted entity-tag — all paired with the current cache
        # digest so only the tag check can reject them.
        for raw in (b"\xff\xfe", b"a\nb", b"a\rb", "\u20ac".encode(), b'"v\x001"',
                    b'"v\t1"', b'"v 1"', b'"v\x7f1"', b'"v\x1b1"', b"v1", b'"v1',
                    b'"v1", "v2"', b'"v"1"', b"", b"W/v1",
                    b' "v1"', b'"v1" ', b'"v1"\x0b', b'"v1"\xc2\xa0', b'"v1"\r'):
            with self.subTest(bound=raw):
                self._write_bound_etag(raw, digest)
                self.store["if_none_match"].clear()
                self.assertEqual(self.refresh(ttl_hours=0)["status"], "updated")
                self.assertEqual(self.store["if_none_match"], [None])
        # Malformed file shapes: missing/extra newlines, CRLF, uppercase digest.
        for file_bytes in (
            b'"v1"\n',
            b'"v1"\n' + digest.encode("ascii"),
            b'"v1"\n' + digest.encode("ascii") + b"\n\n",
            b'"v1"\r\n' + digest.encode("ascii") + b"\r\n",
            b'"v1"\n' + digest.upper().encode("ascii") + b"\n",
        ):
            with self.subTest(raw_file=file_bytes):
                etag.write_bytes(file_bytes)
                self.store["if_none_match"].clear()
                self.assertEqual(self.refresh(ttl_hours=0)["status"], "updated")
                self.assertEqual(self.store["if_none_match"], [None])

    def test_wellformed_etag_shapes_are_sent(self):
        self.assertEqual(self.refresh(ttl_hours=0)["status"], "updated")
        for raw, sent in ((b'W/"v1"', 'W/"v1"'), (b'"v1"', '"v1"'), (b'""', '""')):
            with self.subTest(raw=raw):
                self._write_bound_etag(raw)
                self.refresh(ttl_hours=0)
                self.assertEqual(self.store["if_none_match"][-1], sent)

    def test_mismatched_etag_digest_is_not_sent(self):
        # Interleaved concurrent refreshes: cache holds body X while .etag holds
        # the ETag bound to body Y. Refresh must fetch Y unconditionally instead
        # of sending Y's ETag and keeping stale X on 304.
        body_y = json.dumps(catalog_of(entry(id="new-y"))).encode()
        body_x = json.dumps(catalog_of(entry(id="stale-x"))).encode()
        self.store["body"] = body_y
        self.store["etag"] = '"v2"'
        self.assertEqual(self.refresh(ttl_hours=0)["status"], "updated")
        cache = catalog.cache_path(self.home)
        cache.write_bytes(body_x)  # interleaved writer leaves Y's .etag behind
        self.store["if_none_match"].clear()
        result = self.refresh(ttl_hours=0)
        self.assertEqual(result["status"], "updated")
        self.assertEqual(self.store["if_none_match"], [None])
        self.assertEqual(catalog.validate_file(cache)["models"][0]["id"], "new-y")
        self.assertEqual(self._etag_path().read_text(), f'"v2"\n{self._cache_digest()}\n')
        self.assertEqual(self.refresh(ttl_hours=0)["status"], "not-modified")
        self.assertEqual(self.store["if_none_match"][-1], '"v2"')

    def test_legacy_tag_only_etag_is_not_sent(self):
        self.assertEqual(self.refresh(ttl_hours=0)["status"], "updated")
        etag = self._etag_path()
        for raw in (b'"v1"', b'"v1"\n'):
            with self.subTest(raw=raw):
                etag.write_bytes(raw)
                self.store["if_none_match"].clear()
                result = self.refresh(ttl_hours=0)
                self.assertEqual(result["status"], "updated")
                self.assertEqual(self.store["if_none_match"], [None])
                self.assertEqual(etag.read_text(), f'"v1"\n{self._cache_digest()}\n')

    def test_symlinked_etag_is_never_read_or_sent(self):
        self.assertEqual(self.refresh(ttl_hours=0)["status"], "updated")
        secret = self.home / "secret.txt"
        secret_bytes = self._bound_etag_bytes('"private-token"')  # valid bound shape on purpose
        secret.write_bytes(secret_bytes)
        etag = self._etag_path()
        etag.unlink()
        etag.symlink_to(secret)
        self.store["if_none_match"].clear()
        self.assertEqual(self.refresh(ttl_hours=0)["status"], "updated")
        self.assertEqual(self.store["if_none_match"], [None])
        self.assertFalse(etag.is_symlink())  # the update replaced the link itself
        self.assertEqual(etag.read_bytes(), self._bound_etag_bytes('"v1"'))
        self.assertEqual(secret.read_bytes(), secret_bytes)

    @unittest.skipUnless(hasattr(os, "mkfifo"), "requires os.mkfifo")
    def test_fifo_etag_is_ignored_without_blocking(self):
        self.assertEqual(self.refresh(ttl_hours=0)["status"], "updated")
        etag = self._etag_path()
        for with_writer in (False, True):
            with self.subTest(with_writer=with_writer):
                etag.unlink(missing_ok=True)
                os.mkfifo(etag)
                fds = []
                if with_writer:  # a writer with a valid bound ETag already queued in the pipe
                    fds.append(os.open(etag, os.O_RDONLY | os.O_NONBLOCK))
                    fds.append(os.open(etag, os.O_WRONLY | os.O_NONBLOCK))
                    os.write(fds[-1], self._bound_etag_bytes('"v1"'))
                for fd in fds:
                    self.addCleanup(os.close, fd)
                self.store["if_none_match"].clear()
                outcome = {}
                worker = threading.Thread(
                    target=lambda: outcome.update(result=self.refresh(ttl_hours=0)),
                    daemon=True)
                worker.start()
                worker.join(timeout=10.0)
                self.assertFalse(worker.is_alive(), "FIFO .etag read blocked")
                self.assertEqual(outcome["result"]["status"], "updated")
                self.assertEqual(self.store["if_none_match"], [None])
                self.assertTrue(stat.S_ISREG(etag.lstat().st_mode))

    def test_etag_size_cap(self):
        self.assertEqual(self.refresh(ttl_hours=0)["status"], "updated")
        suffix_len = len(self._cache_digest()) + 2  # "\n<sha256>\n"
        at_cap = '"' + "a" * (catalog.ETAG_MAX_BYTES - suffix_len - 2) + '"'
        over_cap = '"' + "a" * (catalog.ETAG_MAX_BYTES - suffix_len - 1) + '"'
        for tag, sent in ((at_cap, at_cap), (over_cap, None)):
            with self.subTest(size=len(tag) + suffix_len):
                self._write_bound_etag(tag)
                self.store["if_none_match"].clear()
                self.refresh(ttl_hours=0)
                self.assertEqual(self.store["if_none_match"], [sent])

    def test_oversized_sparse_etag_is_ignored_without_reading(self):
        if tracemalloc.is_tracing():  # the peak belongs to whoever started tracing
            self.skipTest("tracemalloc is already tracing")
        self.assertEqual(self.refresh(ttl_hours=0)["status"], "updated")
        etag = self._etag_path()
        with open(etag, "wb") as fh:  # finite and sparse: no disk, no endless source
            fh.truncate(64 * 1024 * 1024)
        self.store["if_none_match"].clear()
        tracemalloc.start()
        try:
            result = self.refresh(ttl_hours=0)
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        self.assertEqual(result["status"], "updated")
        self.assertEqual(self.store["if_none_match"], [None])
        self.assertLess(peak, 4 * 1024 * 1024)

    def test_etag_growth_after_fstat_reads_at_most_cap_plus_one(self):
        self.assertEqual(self.refresh(ttl_hours=0)["status"], "updated")
        etag = self._etag_path()
        self._write_bound_etag('"' + "a" * (catalog.ETAG_MAX_BYTES + 4096) + '"')
        real_fstat = os.fstat
        etag_inode = (etag.stat().st_dev, etag.stat().st_ino)
        offsets = {}

        def small_fstat(fd):
            st = real_fstat(fd)
            if (st.st_dev, st.st_ino) == etag_inode:
                dup = os.dup(fd)  # shares the offset, so it shows how far the read went
                self.addCleanup(os.close, dup)
                offsets["dup"] = dup
                vals = list(st)
                vals[stat.ST_SIZE] = 4
                return os.stat_result(vals)
            return st

        self.store["if_none_match"].clear()
        with unittest.mock.patch.object(catalog.os, "fstat", side_effect=small_fstat):
            self.refresh(ttl_hours=0)
        self.assertEqual(self.store["if_none_match"], [None])
        self.assertEqual(os.lseek(offsets["dup"], 0, os.SEEK_CUR),
                         catalog.ETAG_MAX_BYTES + 1)

    def test_200_without_etag_removes_a_dangling_etag_symlink(self):
        self.store["etag"] = None
        self.assertEqual(self.refresh(ttl_hours=0)["status"], "updated")
        etag = self._etag_path()
        etag.symlink_to(self.home / "nowhere")
        self.assertEqual(self.refresh(ttl_hours=0)["status"], "updated")
        self.assertFalse(etag.is_symlink())
        self.assertFalse(os.path.lexists(etag))

    def test_malformed_server_etag_is_not_stored(self):
        self.assertEqual(self.refresh(ttl_hours=0)["status"], "updated")
        etag = self._etag_path()
        suffix_len = len(self._cache_digest()) + 2
        over_bound_cap = '"' + "a" * (catalog.ETAG_MAX_BYTES - suffix_len - 1) + '"'
        for tag in ("unquoted", over_bound_cap, '"' + "a" * catalog.ETAG_MAX_BYTES + '"'):
            with self.subTest(size=len(tag)):
                self.store["etag"] = '"v1"'
                self.refresh(ttl_hours=0)  # stores a good one first
                self.assertTrue(etag.exists())
                self.store["etag"] = tag
                self.store["changed"] = True
                self.assertEqual(self.refresh(ttl_hours=0)["status"], "updated")
                self.assertFalse(os.path.lexists(etag))  # the stale "v1" went too
                self.store["changed"] = False

    def test_server_etag_strips_only_http_whitespace(self):
        etag = self._etag_path()
        for raw in (' "v1" ', '\t"v1"\t', ' \t"v1"\t '):
            with self.subTest(accepted=raw):
                self.store["etag"] = raw
                self.store["changed"] = True
                self.assertEqual(self.refresh(ttl_hours=0)["status"], "updated")
                self.assertEqual(etag.read_bytes(), self._bound_etag_bytes('"v1"'))
        for raw in ('"v1"\x0b', '"v1"\u00a0', '\x0b"v1"', '\u00a0"v1"'):
            with self.subTest(rejected=raw):
                self.store["etag"] = '"v1"'
                self.store["changed"] = True
                self.assertEqual(self.refresh(ttl_hours=0)["status"], "updated")
                self.assertTrue(etag.exists())
                self.store["etag"] = raw
                self.assertEqual(self.refresh(ttl_hours=0)["status"], "updated")
                self.assertFalse(os.path.lexists(etag))
        self.store["changed"] = False

    def test_directory_at_etag_path_does_not_break_the_update(self):
        cache = catalog.cache_path(self.home)
        cache.parent.mkdir()
        self._etag_path().mkdir()
        for tag in ('"v1"', None):
            with self.subTest(etag=tag):
                self.store["etag"] = tag
                self.assertEqual(self.refresh(ttl_hours=0)["status"], "updated")
                self.assertEqual(catalog.validate_file(cache)["version"], 1)
                self.assertEqual(self.store["if_none_match"][-1], None)
        self.assertEqual(sorted(p.name for p in cache.parent.iterdir()),
                         ["recommended-catalog.json", "recommended-catalog.json.etag"])

    def test_directory_at_cache_path_returns_fallback_and_leaves_etag_untouched(self):
        cache = catalog.cache_path(self.home)
        cache.mkdir(parents=True)
        etag = self._etag_path()
        etag.write_text("keep-me")
        result = self.refresh(ttl_hours=0)
        self.assertEqual(result["status"], "fallback")
        self.assertIn("could not write cache:", result["detail"])
        self.assertTrue(cache.is_dir())
        self.assertEqual(etag.read_text(), "keep-me")
        self.assertEqual(sorted(p.name for p in cache.parent.iterdir()),
                         ["recommended-catalog.json", "recommended-catalog.json.etag"])

    @unittest.skipIf(hasattr(os, "geteuid") and os.geteuid() == 0,
                     "root ignores directory write permissions")
    def test_readonly_cache_dir_returns_fallback_or_kept_without_raising(self):
        cache = catalog.cache_path(self.home)
        cache.parent.mkdir(parents=True, exist_ok=True)
        with self.subTest("fallback"):
            cache.parent.chmod(0o500)
            try:
                result = self.refresh(ttl_hours=0)
            finally:
                cache.parent.chmod(0o700)
            self.assertEqual(result["status"], "fallback")
            self.assertIn("could not write cache:", result["detail"])
            self.assertFalse(self._etag_path().exists())

        with self.subTest("kept"):
            self.assertEqual(self.refresh(ttl_hours=0)["status"], "updated")
            old_body = cache.read_bytes()
            old_etag = self._etag_path().read_bytes()
            self.store["body"] = json.dumps(catalog_of(entry(id="newer"))).encode()
            self.store["etag"] = '"v2"'
            self.store["changed"] = True
            cache.parent.chmod(0o500)
            try:
                result = self.refresh(ttl_hours=0)
            finally:
                cache.parent.chmod(0o700)
            self.assertEqual(result["status"], "kept")
            self.assertIn("could not write cache:", result["detail"])
            self.assertEqual(cache.read_bytes(), old_body)
            self.assertEqual(self._etag_path().read_bytes(), old_etag)

    def _symlinked_cache(self):
        """A valid cache moved aside to target.json, with an old mtime, and a
        symlink to it at the cache path."""
        cache = catalog.cache_path(self.home)
        target = self.home / "target.json"
        target.write_bytes(cache.read_bytes())
        os.utime(target, (1_000_000, 1_000_000))
        cache.unlink()
        cache.symlink_to(target)
        return cache, target

    def test_symlinked_cache_is_refetched_and_its_target_untouched(self):
        self.assertEqual(self.refresh(ttl_hours=0)["status"], "updated")
        cache, target = self._symlinked_cache()
        before = target.read_bytes()
        self.store["if_none_match"].clear()
        self.assertEqual(self.refresh(ttl_hours=24)["status"], "updated")  # never fresh
        self.assertEqual(self.store["if_none_match"], [None])  # never revalidated
        self.assertFalse(cache.is_symlink())  # the fetch replaced the link itself
        self.assertEqual(target.stat().st_mtime, 1_000_000)
        self.assertEqual(target.read_bytes(), before)

    def test_unsolicited_304_without_if_none_match_never_renews(self):
        self.assertEqual(self.refresh(ttl_hours=0)["status"], "updated")
        cache = catalog.cache_path(self.home)
        etag = self._etag_path()
        self.store["force_304"] = True

        with self.subTest("no_etag"):
            etag.unlink()
            os.utime(cache, (1_000_000, 1_000_000))
            self.store["if_none_match"].clear()
            result = self.refresh(ttl_hours=24)
            self.assertEqual(self.store["if_none_match"], [None])
            self.assertEqual(result["status"], "kept")
            self.assertEqual(cache.stat().st_mtime, 1_000_000)

        with self.subTest("mismatched_etag"):
            self._write_bound_etag('"v1"', "0" * 64)
            os.utime(cache, (1_000_000, 1_000_000))
            self.store["if_none_match"].clear()
            result = self.refresh(ttl_hours=24)
            self.assertEqual(self.store["if_none_match"], [None])
            self.assertEqual(result["status"], "kept")
            self.assertEqual(cache.stat().st_mtime, 1_000_000)

        with self.subTest("no_cache"):
            cache.unlink()
            self.store["if_none_match"].clear()
            result = self.refresh(ttl_hours=0)
            self.assertEqual(self.store["if_none_match"], [None])
            self.assertEqual(result["status"], "fallback")
            self.assertFalse(cache.exists())

    def test_304_never_renews_a_cache_swapped_for_a_symlink_in_flight(self):
        self.assertEqual(self.refresh(ttl_hours=0)["status"], "updated")
        self.store["before_reply"] = self._symlinked_cache
        result = self.refresh(ttl_hours=0)
        self.assertEqual(self.store["if_none_match"][-1], '"v1"')
        self.assertEqual(result["status"], "kept")  # the symlink still serves
        self.assertEqual((self.home / "target.json").stat().st_mtime, 1_000_000)

    def test_304_does_not_renew_a_cache_replaced_in_flight(self):
        self.assertEqual(self.refresh(ttl_hours=0)["status"], "updated")
        cache = catalog.cache_path(self.home)
        other = json.dumps(catalog_of(entry(id="other"))).encode()

        def replace_cache():
            cache.write_bytes(other)
            os.utime(cache, (1_000_000, 1_000_000))

        self.store["before_reply"] = replace_cache
        result = self.refresh(ttl_hours=0)
        self.assertEqual(self.store["if_none_match"][-1], '"v1"')
        self.assertEqual(result["status"], "kept")
        self.assertEqual(cache.stat().st_mtime, 1_000_000)

    def test_304_does_not_renew_a_cache_replaced_after_recheck(self):
        # A replacement between the 304 recheck and utime(): the validated
        # descriptor is renewed, never the replacement placed at dest.
        self.assertEqual(self.refresh(ttl_hours=0)["status"], "updated")
        cache = catalog.cache_path(self.home)
        replacement = self.home / "replacement.json"
        replacement.write_bytes(json.dumps(catalog_of(entry(id="other"))).encode())
        os.utime(replacement, (1_000_000, 1_000_000))
        real_utime = os.utime
        swapped = []

        def swap_before_utime(*args, **kwargs):
            if not swapped:
                swapped.append(True)
                os.replace(replacement, cache)
            return real_utime(*args, **kwargs)

        with unittest.mock.patch.object(catalog.os, "utime",
                                        side_effect=swap_before_utime):
            result = self.refresh(ttl_hours=0)
        self.assertEqual(result["status"], "not-modified")
        self.assertEqual(len(swapped), 1)
        self.assertEqual(cache.stat().st_mtime, 1_000_000)
        self.assertEqual(catalog.validate_file(cache)["models"][0]["id"], "other")

    def test_304_does_not_renew_a_symlink_swapped_after_recheck(self):
        # A swap between the 304 recheck and utime(): the validated descriptor
        # is renewed, never the symlink or its target.
        self.assertEqual(self.refresh(ttl_hours=0)["status"], "updated")
        real_utime = os.utime
        swapped = []

        def swap_before_utime(*args, **kwargs):
            if not swapped:
                swapped.append(True)
                self._symlinked_cache()
            return real_utime(*args, **kwargs)

        with unittest.mock.patch.object(catalog.os, "utime",
                                        side_effect=swap_before_utime):
            result = self.refresh(ttl_hours=0)
        self.assertEqual(result["status"], "not-modified")
        self.assertEqual(len(swapped), 1)
        self.assertEqual((self.home / "target.json").stat().st_mtime, 1_000_000)

    def test_http_errors_close_response_without_resource_warning(self):
        self.assertEqual(self.refresh(ttl_hours=0)["status"], "updated")

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
        redirect_url = f"http://127.0.0.1:{srv.server_port}/redirecting"

        cases = (
            ("304", lambda: self.refresh(ttl_hours=0), "not-modified"),
            ("302", lambda: catalog.refresh(
                self.home, ttl_hours=0, timeout=5, url=redirect_url), "kept"),
        )
        for name, call, expected_status in cases:
            with self.subTest(name):
                gc.collect()
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    result = call()
                    gc.collect()
                self.assertEqual(result["status"], expected_status)
                res_warnings = [
                    str(w.message) for w in caught
                    if issubclass(w.category, ResourceWarning)
                ]
                self.assertEqual(res_warnings, [])

    def test_redirect_body_is_never_read(self):
        # A 302 that promises a 1 GiB body and sends 4 KiB, then waits. Reading
        # the body would block until the socket timeout; closing it ends here.
        released = threading.Event()

        class HugeRedirect(http.server.BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802
                self.send_response(302)
                self.send_header("Location", "https://evil.example/x")
                self.send_header("Content-Length", str(1 << 30))
                self.end_headers()
                self.wfile.write(b"x" * 4096)
                self.wfile.flush()
                self.connection.settimeout(20)
                try:
                    while self.connection.recv(65536):
                        pass
                except ConnectionResetError:  # closed with our 4 KiB unread: an RST
                    pass
                except OSError:  # timed out: the client kept the connection open
                    return
                released.set()  # the client closed its end

            def log_message(self, *_a):
                pass

        srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), HugeRedirect)
        srv.daemon_threads = True
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        self.addCleanup(srv.server_close)
        self.addCleanup(srv.shutdown)
        outcome = {}
        worker = threading.Thread(target=lambda: outcome.update(result=catalog.refresh(
            self.home, ttl_hours=0, timeout=30,
            url=f"http://127.0.0.1:{srv.server_port}/redirecting")), daemon=True)
        worker.start()
        worker.join(timeout=10.0)
        self.assertFalse(worker.is_alive(), "refresh read the redirect body")
        self.assertEqual(outcome["result"]["status"], "fallback")
        self.assertTrue(released.wait(10.0), "redirect connection left open")

    def test_atomic_write_closes_fd_and_unlinks_when_fdopen_fails(self):
        path = self.home / "out.bin"
        real_mkstemp = catalog.tempfile.mkstemp
        made = []

        def recording_mkstemp(*args, **kwargs):
            made.append(real_mkstemp(*args, **kwargs))
            return made[-1]

        with unittest.mock.patch.object(catalog.tempfile, "mkstemp",
                                        side_effect=recording_mkstemp), \
                unittest.mock.patch.object(catalog.os, "fdopen",
                                           side_effect=MemoryError("fdopen")):
            with self.assertRaises(MemoryError):
                catalog._atomic_write(path, b"data")
        fd, tmp = made[0]
        with self.assertRaises(OSError) as ctx:
            os.fstat(fd)
        self.assertEqual(ctx.exception.errno, errno.EBADF)
        self.assertFalse(os.path.lexists(tmp))
        self.assertFalse(os.path.lexists(path))

    def test_atomic_write_preserves_original_error_when_late_fdopen_fails(self):
        path = self.home / "out.bin"
        real_mkstemp = catalog.tempfile.mkstemp
        made = []

        def recording_mkstemp(*args, **kwargs):
            made.append(real_mkstemp(*args, **kwargs))
            return made[-1]

        def late_fdopen(fd, *args, **kwargs):
            os.close(fd)  # fdopen already closed fd before failing
            raise MemoryError("late fdopen")

        with unittest.mock.patch.object(catalog.tempfile, "mkstemp",
                                        side_effect=recording_mkstemp), \
                unittest.mock.patch.object(catalog.os, "fdopen",
                                           side_effect=late_fdopen):
            with self.assertRaisesRegex(MemoryError, "late fdopen"):
                catalog._atomic_write(path, b"data")
        fd, tmp = made[0]
        with self.assertRaises(OSError) as ctx:
            os.fstat(fd)
        self.assertEqual(ctx.exception.errno, errno.EBADF)
        self.assertFalse(os.path.lexists(tmp))
        self.assertFalse(os.path.lexists(path))

    def test_untraversable_cache_dir_returns_fallback_on_fetch_failure(self):
        # An untraversable cache dir: stat() of the cache path fails with EACCES,
        # which Python 3.11's Path.is_file() raises instead of returning False.
        dest = str(catalog.cache_path(self.home))
        real_stat = os.stat

        def denied_stat(path, *args, **kwargs):
            if os.fspath(path) == dest:
                raise PermissionError(errno.EACCES, "Permission denied", dest)
            return real_stat(path, *args, **kwargs)

        stat_patch = unittest.mock.patch("os.stat", denied_stat)
        stat_patch.start()
        self.addCleanup(stat_patch.stop)

        with self.subTest("http_error"):
            self.store["fail"] = True
            try:
                self.assertEqual(self.refresh(ttl_hours=0)["status"], "fallback")
            finally:
                self.store["fail"] = False

        with self.subTest("url_error"):
            refused = catalog.urllib.error.URLError("connection refused")
            with unittest.mock.patch.object(catalog.urllib.request.OpenerDirector, "open",
                                            side_effect=refused):
                self.assertEqual(self.refresh(ttl_hours=0)["status"], "fallback")

        with self.subTest("oversized_body"):
            self.store["fail"] = False
            self.store["body"] = b" " * (catalog.CATALOG_MAX_BYTES + 1)
            self.assertEqual(self.refresh(ttl_hours=0)["status"], "fallback")

        with self.subTest("invalid_body"):
            self.store["fail"] = False
            self.store["body"] = b"{broken"
            self.assertEqual(self.refresh(ttl_hours=0)["status"], "fallback")




class TrainedKeyOptional(unittest.TestCase):
    def test_an_old_catalog_without_trained_tokens_validates_and_reads_unknown(self):
        old = entry()
        del old["context"]["trained_tokens"]  # a pre-469 cache or custom catalog
        catalog.validate_catalog(catalog_of(old))
        self.assertIsNone(old["context"]["trained_tokens"])  # unknown, normalized to null in place

if __name__ == "__main__":
    unittest.main()
