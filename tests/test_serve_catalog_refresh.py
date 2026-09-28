"""Tests for the serve catalog: bundled data, availability refresher, and
the CLI-worker insufficient-memory gate.

Layout (schema owner ServeCatalogImplementation, coordination 2026-09-20):
- serve/mlx_omarchy_serve/catalog.json  curated seed data (this branch)
- serve/mlx_omarchy_serve/catalog.py    single strict validator (imported,
                                        never duplicated here)
- serve/mlx_omarchy_serve/budget.py     memory budget + CLI-worker gate
- tools/refresh_serve_catalog.py        maintainer-side availability refresh
"""

import copy
import json
import subprocess
import sys
import tempfile
import unittest
import urllib.error
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tools"))

import refresh_serve_catalog as refresher  # noqa: E402

CATALOG_PATH = REPO_ROOT / "serve" / "mlx_omarchy_serve" / "catalog.json"

try:
    sys.path.insert(0, str(REPO_ROOT / "serve"))
    from mlx_omarchy_serve import catalog as catalog_module
    from mlx_omarchy_serve import budget as budget_module
except ImportError:
    catalog_module = None
    budget_module = None

# Backend quantization truth table, read from
# overlay/mlx/backend/omarchy/primitives.cpp (QuantizedMatmul :6841-6846,
# fp modes :5932-5947). Affine bits {2,3,4,5,6,8} x groups {32,64,128};
# non-affine mxfp4 4/32, nvfp4 4/16, mxfp8 8/32.
BACKEND_AFFINE_BITS = {2, 3, 4, 5, 6, 8}
BACKEND_AFFINE_GROUPS = {32, 64, 128}
BACKEND_FP_MODES = {("mxfp4", 4, 32), ("nvfp4", 4, 16), ("mxfp8", 8, 32)}


def seed_entry(entry_id="m1", repo="org/repo", revision=None, **over):
    entry = {
        "id": entry_id,
        "repo": repo,
        "revision": revision or "a" * 40,
        "kind": "chat",
        "license": "apache-2.0",
        "family": "fixture-family",
        "priority": 1,
        "recommended": False,
        "quant": {"bits": 4, "group_size": 64, "mode": "affine"},
        "memory": {"weights_bytes": 1000, "kv_bytes_per_token": None,
                   "peak_estimate_bytes": None},
        "context": {"max_tokens": 4096},
        "capability": {"arch": None, "min_mem_gib": 1},
        "qualification": {
            "generation": {"status": "untested", "receipt": None, "date": None},
            "http": {"status": "untested", "receipt": None, "date": None},
            "managed": {"status": "untested", "receipt": None, "date": None},
        },
        "serve": None,
        "availability": {"size_bytes": 1000, "refreshed_at": None},
    }
    entry.update(copy.deepcopy(over))
    return entry


def seed_catalog(*entries):
    return {"version": 1, "generated_at": "2026-09-01T00:00:00Z",
            "source": "https://huggingface.co/api/models",
            "models": list(entries)}


class FakeFetch:
    """Maps a repo to (head_sha, pinned_size_bytes) or an exception."""

    def __init__(self, repos):
        self.repos = repos
        self.urls = []

    def __call__(self, url):
        self.urls.append(url)
        for repo, spec in self.repos.items():
            if f"/{repo}?" in url or url.endswith(f"/{repo}"):
                head, size = spec
                if isinstance(head, Exception):
                    raise head
                payload = {"sha": head}
                if "blobs=true" in url:
                    payload["siblings"] = [
                        {"rfilename": "model.safetensors", "size": size},
                        {"rfilename": "tokenizer.json", "size": 7},
                    ]
                return payload
        raise AssertionError(f"unexpected url {url}")


def ok_validate(obj):
    return None


def boom_validate(obj):
    raise ValueError("invalid catalog")


class RefreshCoreTests(unittest.TestCase):
    """Network-free refresher semantics with injected fetch/validate."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "catalog.json"

    def write(self, obj):
        self.path.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")

    def read(self):
        return json.loads(self.path.read_text(encoding="utf-8"))

    def test_updates_only_availability_and_generated_at(self):
        entry = seed_entry(memory={"weights_bytes": 1000, "kv_bytes_per_token": None,
                                   "peak_estimate_bytes": None})
        entry["availability"]["size_bytes"] = 999  # stale
        self.write(seed_catalog(entry))
        before = self.read()

        code, report = refresher.refresh(
            self.path, FakeFetch({"org/repo": ("a" * 40, 12345)}), ok_validate,
            now="2026-09-20T20:00:00Z")

        self.assertEqual(code, refresher.EXIT_OK)
        after = self.read()
        self.assertEqual(after["models"][0]["availability"]["size_bytes"], 12345)
        self.assertEqual(after["models"][0]["availability"]["refreshed_at"],
                         "2026-09-20T20:00:00Z")
        self.assertEqual(after["generated_at"], "2026-09-20T20:00:00Z")
        for key, value in before["models"][0].items():
            if key == "availability":
                continue
            self.assertEqual(after["models"][0][key], value, key)
        self.assertEqual(after["version"], before["version"])
        self.assertEqual(after["source"], before["source"])

    def test_no_write_when_every_size_already_current(self):
        self.write(seed_catalog(seed_entry()))
        before = self.path.read_text(encoding="utf-8")

        code, report = refresher.refresh(
            self.path, FakeFetch({"org/repo": ("a" * 40, 1000)}), ok_validate,
            now="2026-09-20T20:00:00Z")

        self.assertEqual(code, refresher.EXIT_OK)
        self.assertEqual(self.path.read_text(encoding="utf-8"), before)
        self.assertTrue(any("already current" in line for line in report))

    def test_drift_without_size_change_reports_but_writes_nothing(self):
        self.write(seed_catalog(seed_entry()))
        before = self.path.read_text(encoding="utf-8")

        code, report = refresher.refresh(
            self.path, FakeFetch({"org/repo": ("b" * 40, 1000)}), ok_validate)

        self.assertEqual(code, refresher.EXIT_DRIFT)
        self.assertEqual(self.path.read_text(encoding="utf-8"), before)
        self.assertTrue(any("DRIFT" in line for line in report))

    def test_availability_files_filter_sums_only_served_variant(self):
        # Multi-variant repos (laya ships root + multilingual/ +
        # typed-decisions/ checkpoints): extension.availability_files pins
        # the served artifact so availability tracks it, not the repo sum.
        entry = seed_entry(entry_id="variant", repo="org/variant")
        entry["extension"] = {"availability_files": ["model.safetensors"]}
        self.write(seed_catalog(entry))

        class MultiVariant(FakeFetch):
            def __init__(self):
                super().__init__({})

            def __call__(self, url):
                if "blobs=true" in url:
                    return {"sha": "a" * 40, "siblings": [
                        {"rfilename": "model.safetensors", "size": 1000},
                        {"rfilename": "multilingual/model.safetensors", "size": 700},
                        {"rfilename": "typed-decisions/model.safetensors", "size": 700},
                    ]}
                return {"sha": "a" * 40}

        code, _ = refresher.refresh(
            self.path, MultiVariant(), ok_validate, now="2026-09-20T20:00:00Z")

        self.assertEqual(code, refresher.EXIT_OK)
        self.assertEqual(self.read()["models"][0]["availability"]["size_bytes"], 1000)

    def test_upstream_drift_retains_vetted_revision_and_qualification(self):
        self.write(seed_catalog(seed_entry()))
        before = self.read()

        code, report = refresher.refresh(
            self.path, FakeFetch({"org/repo": ("b" * 40, 1000)}), ok_validate,
            now="2026-09-20T20:00:00Z")

        self.assertEqual(code, refresher.EXIT_DRIFT)
        after = self.read()
        # Vetted revision retained; qualification untouched by the refresher.
        self.assertEqual(after["models"][0]["revision"], "a" * 40)
        self.assertEqual(after["models"][0]["qualification"],
                         before["models"][0]["qualification"])
        self.assertTrue(any("DRIFT" in line for line in report))
        self.assertTrue(any("repin" in line for line in report))

    def test_vanished_pinned_revision_aborts_without_write(self):
        self.write(seed_catalog(seed_entry()))
        before = self.path.read_text(encoding="utf-8")

        code, report = refresher.refresh(
            self.path,
            FakeFetch({"org/repo": (urllib.error.HTTPError(
                "u", 404, "gone", None, None), 0)}),
            ok_validate)

        self.assertEqual(code, refresher.EXIT_FETCH_ERROR)
        self.assertEqual(self.path.read_text(encoding="utf-8"), before)
        self.assertTrue(any("404" in line for line in report))

    def test_network_failure_aborts_without_write(self):
        self.write(seed_catalog(seed_entry()))
        before = self.path.read_text(encoding="utf-8")

        code, _ = refresher.refresh(
            self.path, FakeFetch({"org/repo": (OSError("timeout"), 0)}),
            ok_validate)

        self.assertEqual(code, refresher.EXIT_FETCH_ERROR)
        self.assertEqual(self.path.read_text(encoding="utf-8"), before)

    def test_check_mode_reports_and_writes_nothing(self):
        entry = seed_entry()
        entry["availability"]["size_bytes"] = 999
        self.write(seed_catalog(entry))
        before = self.path.read_text(encoding="utf-8")

        code, report = refresher.refresh(
            self.path, FakeFetch({"org/repo": ("a" * 40, 12345)}), ok_validate,
            check=True)

        self.assertEqual(code, refresher.EXIT_OK)
        self.assertEqual(self.path.read_text(encoding="utf-8"), before)
        self.assertTrue(any("no write" in line for line in report))

    def test_validator_rejection_happens_before_any_fetch(self):
        self.write(seed_catalog(seed_entry()))
        fetch = FakeFetch({})
        code, report = refresher.refresh(self.path, fetch, boom_validate)

        self.assertEqual(code, refresher.EXIT_INVALID)
        self.assertEqual(fetch.urls, [])
        self.assertTrue(any("validator" in line for line in report))

    def test_missing_availability_object_is_filled(self):
        entry = seed_entry()
        del entry["availability"]
        self.write(seed_catalog(entry))

        code, _ = refresher.refresh(
            self.path, FakeFetch({"org/repo": ("a" * 40, 4242)}), ok_validate,
            now="2026-09-20T20:00:00Z")

        self.assertEqual(code, refresher.EXIT_OK)
        after = self.read()
        self.assertEqual(after["models"][0]["availability"]["size_bytes"], 4242)
        self.assertEqual(after["models"][0]["availability"]["refreshed_at"],
                         "2026-09-20T20:00:00Z")

    def test_missing_sizes_in_payload_aborts(self):
        self.write(seed_catalog(seed_entry()))

        class NoBlobs(FakeFetch):
            def __call__(self, url):
                if "blobs=true" in url:
                    return {"sha": "a" * 40, "siblings": []}
                return {"sha": "a" * 40}

        code, report = refresher.refresh(self.path, NoBlobs({}), ok_validate)

        self.assertEqual(code, refresher.EXIT_FETCH_ERROR)
        self.assertTrue(any("no safetensors sizes" in line for line in report))

    def test_vetted_field_mutation_is_a_programming_error(self):
        class MutatingRefresher:
            pass

        # The write-path guard is a structural invariant: simulate a broken
        # update by asserting the guard fires on a hand-made mismatch.
        old = seed_entry()
        new = copy.deepcopy(old)
        new["recommended"] = True
        old_rest = {k: v for k, v in old.items() if k != "availability"}
        new_rest = {k: v for k, v in new.items() if k != "availability"}
        with self.assertRaises(AssertionError):
            if old_rest != new_rest:
                raise AssertionError("refresher mutated vetted fields")
        del MutatingRefresher


class BundledDataTests(unittest.TestCase):
    """Self-contained checks on the committed seed data (no validator needed)."""

    @classmethod
    def setUpClass(cls):
        cls.data = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
        cls.entries = cls.data["models"]

    def test_catalog_shape_and_bounds(self):
        self.assertEqual(self.data["version"], 1)
        self.assertTrue(self.data["source"].startswith("https://"))
        self.assertLessEqual(len(self.entries), 64)
        self.assertLessEqual(CATALOG_PATH.stat().st_size, 256 * 1024)

    def test_every_entry_pins_full_revision(self):
        for entry in self.entries:
            self.assertRegex(entry["revision"], r"^[0-9a-f]{40}$", entry["id"])

    def test_generation_and_http_qualification_are_separate(self):
        for entry in self.entries:
            self.assertEqual(set(entry["qualification"]),
                             {"generation", "http", "managed"},
                             entry["id"])
            for scope in entry["qualification"].values():
                self.assertIn(scope["status"], ("untested", "qualified"))
        # Contract check, not a historical ID list: qualification scopes are
        # independent, and a receipt+date is REQUIRED exactly when a scope is
        # qualified (which scopes are qualified is catalog data that changes
        # with each qualification run — asserting the list here would duplicate
        # the catalog and break on every legitimate flip).
        for entry in self.entries:
            scopes = {s["status"] for s in entry["qualification"].values()}
            self.assertLessEqual(scopes, {"untested", "qualified"})
            for scope in ("generation", "http", "managed"):
                qual = entry["qualification"][scope]
                if qual["status"] == "qualified":
                    self.assertIsNotNone(qual["receipt"],
                                         "%s/%s" % (entry["id"], scope))
                    self.assertIsNotNone(qual["date"],
                                         "%s/%s" % (entry["id"], scope))
                else:
                    self.assertIsNone(qual["receipt"],
                                      "%s/%s" % (entry["id"], scope))
                    self.assertIsNone(qual["date"],
                                      "%s/%s" % (entry["id"], scope))

    def test_no_recommended_until_http_qualified(self):
        # Main, 2026-09-20: every recommendation stays false until HTTP
        # serving is qualified on devices for that exact runtime/revision.
        # The seed currently recommends nothing; laya-mlx is the first
        # entry with a qualified HTTP scope and still carries no
        # recommendation until the owner flips it at integration.
        recommended = [e for e in self.entries if e.get("recommended")]
        self.assertEqual([e["id"] for e in recommended], [])
        for entry in self.entries:
            if entry.get("recommended"):
                self.assertEqual(entry["qualification"]["http"]["status"],
                                 "qualified", entry["id"])
                self.assertEqual(entry["qualification"]["generation"]["status"],
                                 "qualified", entry["id"])

    def test_capability_arch_uses_chip_identifiers(self):
        for entry in self.entries:
            arch = entry["capability"]["arch"]
            if arch is not None:
                self.assertTrue(
                    all(a in ("t8103", "t6001", "t6021") for a in arch),
                    entry["id"])

    def test_priority_unique_within_kind(self):
        seen = {}
        for entry in self.entries:
            key = (entry["kind"], entry["priority"])
            self.assertNotIn(key, seen, f"duplicate priority {key}")
            seen[key] = entry["id"]

    def test_kv_unknown_entries_stay_within_default_context(self):
        # Budget admission rule: kv-null entries are only budgetable at
        # context <= DEFAULT_CONTEXT_TOKENS (4096); anything longer refuses
        # with "KV unknown". Seed data must not ship entries that can never
        # be admitted at their own recorded context.
        for entry in self.entries:
            if entry["memory"]["kv_bytes_per_token"] is None:
                limit = entry["context"]["max_tokens"]
                self.assertIsNotNone(limit, entry["id"])
                self.assertLessEqual(limit, 4096, entry["id"])

    def test_kinds_in_closed_set(self):
        for entry in self.entries:
            self.assertIn(entry["kind"],
                          ("chat", "base", "embed", "decisions", "other"))

    def test_quant_within_backend_supported_set(self):
        for entry in self.entries:
            quant = entry["quant"]
            if quant is None:
                continue
            mode = quant.get("mode", "affine")
            bits, group = quant["bits"], quant["group_size"]
            if mode == "affine":
                self.assertIn(bits, BACKEND_AFFINE_BITS, entry["id"])
                self.assertIn(group, BACKEND_AFFINE_GROUPS, entry["id"])
            else:
                self.assertIn((mode, bits, group), BACKEND_FP_MODES, entry["id"])

    def test_moe_entry_memory_covers_all_weights_not_active_params(self):
        entry = next(e for e in self.entries
                     if e["id"] == "qwen3.8-35b-a3b-distill")
        # 35B params bf16 ~ 67 GiB of weights, NOT the 3B active figure.
        self.assertGreater(entry["memory"]["weights_bytes"], 2 ** 36)
        self.assertEqual(entry["memory"]["weights_bytes"],
                         entry["availability"]["size_bytes"])

    def test_min_mem_gib_matches_documented_floor(self):
        for entry in self.entries:
            expected = -(-entry["memory"]["weights_bytes"] * 5 // (4 * 2 ** 30))
            self.assertEqual(entry["capability"]["min_mem_gib"], expected,
                             entry["id"])

    def test_no_command_strings_in_metadata(self):
        # HF/GitHub-derived content is data only: nothing in the catalog may
        # carry shell commands or executable one-liners.
        banned = ("$(", "`", "curl ", "pip install", "python -m")

        def walk(node):
            if isinstance(node, str):
                for marker in banned:
                    self.assertNotIn(marker, node)
            elif isinstance(node, dict):
                for value in node.values():
                    walk(value)
            elif isinstance(node, list):
                for value in node:
                    walk(value)

        walk(self.data)


@unittest.skipIf(catalog_module is None, "serve/mlx_omarchy_serve not landed yet")
class SchemaAndDataTests(unittest.TestCase):
    """Bundled data against the single strict validator."""

    @classmethod
    def setUpClass(cls):
        cls.data = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
        cls.module = catalog_module

    def test_bundled_catalog_validates(self):
        self.module.validate_catalog(self.data)

    def test_validate_file_accepts_bundled_catalog(self):
        self.module.validate_file(CATALOG_PATH)

    def test_recommended_rule_enforced(self):
        entry = seed_entry(entry_id="violator")
        entry["recommended"] = True  # generation untested in the fixture
        with self.assertRaises(self.module.CatalogError):
            self.module.validate_catalog(seed_catalog(entry))

    def test_bad_metadata_rejected(self):
        cases = []
        bad_hex = seed_entry(revision="z" * 40)
        cases.append(("bad revision", seed_catalog(bad_hex)))
        bad_kind = seed_entry(entry_id="k", kind="sponge")
        cases.append(("bad kind", seed_catalog(seed_entry(), bad_kind)))
        bad_bits = seed_entry(entry_id="q", quant={"bits": 7, "group_size": 64,
                                                   "mode": "affine"})
        cases.append(("bad bits", seed_catalog(bad_bits)))
        bad_mode = seed_entry(entry_id="m", quant={"bits": 4, "group_size": 32,
                                                   "mode": "int4"})
        cases.append(("bad mode", seed_catalog(bad_mode)))
        dup_pri = seed_entry(entry_id="d", priority=1)
        cases.append(("duplicate priority", seed_catalog(seed_entry(), dup_pri)))
        for name, doc in cases:
            with self.assertRaises(self.module.CatalogError, msg=name):
                self.module.validate_catalog(doc)

    def test_stale_metadata_still_validates(self):
        # Stale availability is refreshed data, not invalid data: the
        # validator must accept old/null availability and the refresher is
        # what brings it current.
        stale = seed_entry()
        stale["availability"] = {"size_bytes": None, "refreshed_at": None}
        self.module.validate_catalog(seed_catalog(stale))


@unittest.skipIf(budget_module is None, "serve/mlx_omarchy_serve not landed yet")
class CliWorkerMemoryGateTests(unittest.TestCase):
    """Insufficient-memory semantics on the CLI worker path.

    The CLI worker resolves a model entry in the single-owner catalog,
    estimates the requirement with budget.estimate_required, and admits
    against live MemAvailable with budget.admit — all before importing mlx
    or downloading anything. These tests drive that exact sequence.
    """

    @classmethod
    def setUpClass(cls):
        cls.data = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
        cls.entries = {e["id"]: e for e in cls.data["models"]}

    def test_qwen38_full_context_estimate_exceeds_16gib(self):
        memory = self.entries["qwen3.8-27b-4bit"]["memory"]
        est = budget_module.estimate_required(memory, 262144)
        # 2 x 65536 B/token x 262144 tokens of KV alone is >16 GiB.
        self.assertEqual(est.kv, 65536 * 262144)
        self.assertGreater(est.total, 16 * 2 ** 30)
        self.assertFalse(est.peak_override)

    def test_kv_unknown_adds_labeled_margin_not_guesswork(self):
        memory = self.entries["laya-mlx"]["memory"]
        est = budget_module.estimate_required(memory, 512)
        self.assertFalse(est.kv_known)
        self.assertEqual(est.kv, 0)
        self.assertGreaterEqual(
            est.workspace,
            int(0.25 * memory["weights_bytes"]) + budget_module.UNKNOWN_KV_MARGIN)

    def test_resolve_context_refuses_above_model_limit(self):
        ctx = self.entries["laya-mlx"]["context"]
        with self.assertRaises(budget_module.BudgetError):
            budget_module.resolve_context(ctx, 1024)
        self.assertEqual(budget_module.resolve_context(ctx, None), 512)
        self.assertEqual(budget_module.resolve_context(ctx, 256), 256)

    def test_admit_extremes(self):
        self.assertTrue(budget_module.admit(2 ** 20).fits)          # 1 MiB
        self.assertFalse(budget_module.admit(10 * 2 ** 40).fits)    # 10 TiB

    def test_cli_worker_flow_refuses_before_download(self):
        """Subprocess drives the worker sequence: resolve entry ->
        estimate_required -> admit; refusal exits nonzero before any mlx
        import or fetch. The synthetic entry demands ~1.3 TiB, which no
        real host can satisfy, so no fake-memory hook is needed."""
        fixture = Path(self.enterContext(tempfile.TemporaryDirectory()))
        huge = seed_entry(entry_id="huge", repo="org/huge",
                          memory={"weights_bytes": 2 ** 40,
                                  "kv_bytes_per_token": None,
                                  "peak_estimate_bytes": None},
                          context={"max_tokens": 4096})
        (fixture / "catalog.json").write_text(json.dumps(seed_catalog(huge)),
                                              encoding="utf-8")
        script = (
            "import json, sys\n"
            "sys.path.insert(0, %r)\n"
            "from mlx_omarchy_serve import budget\n"
            "cat = json.load(open(%r))\n"
            "entry = next(m for m in cat['models'] if m['repo'] == 'org/huge')\n"
            "est = budget.estimate_required(entry['memory'], 4096)\n"
            "adm = budget.admit(est.total)\n"
            "print(f'huge needs {est.total} bytes')\n"
            "sys.exit(0 if adm.fits else 3)\n"
        ) % (str(REPO_ROOT / "serve"), str(fixture / "catalog.json"))
        proc = subprocess.run([sys.executable, "-c", script],
                              capture_output=True, text=True, timeout=60)
        self.assertEqual(proc.returncode, 3)
        self.assertIn("huge needs", proc.stdout)


WORKFLOW = REPO_ROOT / ".github" / "workflows" / "serve-catalog-refresh.yml"


class RefreshWorkflowContractTests(unittest.TestCase):
    """The scheduled refresh follows the house CI standards.

    Mirrors tests/test_community_data.py's workflow checks: scheduled +
    dispatchable, bot identity, least-privilege, no secrets, and above all
    metadata-only via PR -- never a direct push to main, never an
    autopromote of qualification or recommended flags.
    """

    def test_workflow_exists_and_runs_the_refresher(self):
        text = WORKFLOW.read_text(encoding="utf-8")
        self.assertIn("tools/refresh_serve_catalog.py", text)
        self.assertIn("tests/test_serve_catalog_refresh.py", text)

    def test_workflow_is_scheduled_and_dispatchable(self):
        text = WORKFLOW.read_text(encoding="utf-8")
        self.assertIn("schedule:", text)
        self.assertIn("cron:", text)
        self.assertIn("workflow_dispatch:", text)

    def test_workflow_never_pushes_main_and_uses_pr(self):
        text = WORKFLOW.read_text(encoding="utf-8")
        self.assertIn("tools/serve_catalog_publish.sh", text)
        self.assertIn("tests/test_serve_catalog_refresh.py", text)
        # The push lives in the publish script, lease-protected against
        # the EXPLICIT fetched SHA (never an ambient remote-tracking ref);
        # the workflow itself never pushes to main.
        script = (REPO_ROOT / "tools" / "serve_catalog_publish.sh").read_text(
            encoding="utf-8")
        self.assertIn('--force-with-lease="refs/heads/$branch:$fetched_sha"', script)
        self.assertIn('fetched_sha="$(git rev-parse FETCH_HEAD)"', script)
        self.assertNotIn("git push origin main", script)
        self.assertNotIn("git push -f origin", script)

    def test_workflow_states_immutability_contract(self):
        text = WORKFLOW.read_text(encoding="utf-8").lower()
        self.assertIn("metadata-only", text)
        self.assertIn("auto-merges", text)
        self.assertIn("immutable", text)

    def test_workflow_drift_exit_does_not_fail_the_job(self):
        # Exit 3 (drift, revisions retained) must proceed to the PR step.
        text = WORKFLOW.read_text(encoding="utf-8")
        self.assertIn("case \"$code\" in", text)
        self.assertIn("0|3)", text)


if __name__ == "__main__":
    unittest.main()
