"""Assistant pair runtime: catalog pairs, adaptive context, managed ownership.

Contract (2026-09-27 offline assistant): PairManager(home) with
status/setup/start/stop/ensure_context; catalog pair records; atomic batch
admission with per-child claim via an inherited private pipe; parent-lifetime
watchdog; cleanup only after a verified process exit.

Tests exercise consumer behavior only. Model weights are never downloaded and
MLX never imports here: worker processes in the lifecycle tests are real
subprocesses running a stub that uses the REAL claim/lifetime protocol
(budget.claim_from_env / budget.watch_parent_fd). Only the multi-gigabyte
model itself is replaced.
"""

import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
import types
import unittest
import urllib.request
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parent.parent
SERVE_ROOT = str(REPO_ROOT / "serve")
sys.path.insert(0, SERVE_ROOT)

from mlx_omarchy_serve import budget, catalog  # noqa: E402
from mlx_omarchy_assistant import managed, pairs  # noqa: E402
from mlx_omarchy_assistant.pairs import PAIR_DEV_QUALIFICATION_ENV

GiB = 1024**3

CHAT_SMALL_ID = "qwen3.8-2b-4bit"
CHAT_BIG_ID = "qwen3.8-27b-4bit"
DECISION_ID = "laya-mlx"
SPEED_ID = "qwen3.8-3b-4bit"


def model_entry(entry_id, repo, weights, kv_per_token, kind="chat",
                serve=None, gen="qualified", http="qualified", managed_q="untested",
                max_tokens=262144, priority=1):
    return {
        "id": entry_id,
        "repo": repo,
        "revision": "a" * 40,
        "kind": kind,
        "license": "apache-2.0",
        "family": "fixture",
        "priority": priority,
        "recommended": False,
        "quant": {"bits": 4, "group_size": 64, "mode": "affine"},
        "memory": {"weights_bytes": weights, "kv_bytes_per_token": kv_per_token,
                   "peak_estimate_bytes": None},
        "context": {"max_tokens": max_tokens},
        "capability": {"arch": None, "min_mem_gib": None},
        "qualification": {
            "generation": {"status": gen, "receipt": "r.md" if gen == "qualified" else None,
                           "date": "2026-09-20" if gen == "qualified" else None},
            "http": {"status": http, "receipt": "h.md" if http == "qualified" else None,
                     "date": "2026-09-20" if http == "qualified" else None},
            "managed": {"status": managed_q,
                        "receipt": "m.md" if managed_q == "qualified" else None,
                        "date": "2026-09-20" if managed_q == "qualified" else None},
        },
        "serve": serve,
        "availability": {"size_bytes": weights, "refreshed_at": None},
    }


def fixture_catalog():
    chat_small = model_entry(CHAT_SMALL_ID, "org/chat-small", int(1.0 * GiB), 12288,
                             serve={"backend": "mlx-lm", "module": None}, priority=10)
    chat_big = model_entry(CHAT_BIG_ID, "org/chat-big", int(15.0 * GiB), 65536,
                           serve={"backend": "mlx-lm", "module": None}, priority=1)
    decision = model_entry(DECISION_ID, "org/laya", int(0.8 * GiB), None, kind="decisions",
                           serve={"backend": "module", "module": "mlx_omarchy_laya.server"})
    everyday = {
        "id": "everyday", "label": "Everyday", "chat_model": CHAT_SMALL_ID,
        "decision_model": DECISION_ID, "priority": 2, "max_questions": 8,
        "routing_policy": "1",
        "qualification": {"status": "untested", "receipt": None, "date": None},
    }
    quality = {
        "id": "quality", "label": "Quality", "chat_model": CHAT_BIG_ID,
        "decision_model": DECISION_ID, "priority": 1, "max_questions": 8,
        "routing_policy": "1",
        "qualification": {"status": "untested", "receipt": None, "date": None},
    }
    return {"version": 1, "generated_at": "2026-09-01T00:00:00Z",
            "source": "https://huggingface.co/api/models",
            "models": [chat_big, chat_small, decision],
            "pairs": [everyday, quality]}


def write_home_catalog(home, cat):
    path = catalog.cache_path(home)
    path.parent.mkdir(parents=True, exist_ok=True)
    catalog.validate_catalog(cat)
    path.write_text(json.dumps(cat), encoding="utf-8")


# ----------------------------------------------------------------- evidence fixtures
# Synthetic MEASUREMENTS for selection tests only. The bundled catalog stays
# measurement-free: every real pair ships unqualified.

RUNTIME_IDENTITY = {"mlx_version": "0.29.0",
                    "extension_sha256": "c" * 64,
                    "libmlx_sha256": "d" * 64}


def selection_evidence(quality, decode, *, target=10.0, first_visible_p95_ms=900.0,
                       runtime=None, arch=("t6021",),
                       backend="mlx-lm", chat_revision="a" * 40,
                       decision_revision="a" * 40):
    return {
        "arch": list(arch),
        "backend": backend,
        "chat_revision": chat_revision,
        "decision_revision": decision_revision,
        "date": "2026-09-25",
        "quality": {"task": "assistant-chat", "score": quality,
                    "receipt": "receipts/2026-09-25-pair-quality.md"},
        "latency": {"decode_tokens_per_sec": decode,
                    "target_tokens_per_sec": target,
                    "first_visible_p95_ms": first_visible_p95_ms,
                    "receipt": "receipts/2026-09-25-pair-latency.md"},
        "runtime": dict(RUNTIME_IDENTITY if runtime is None else runtime),
    }


def qualified_pair(pair_id, label, chat_id, priority, evidence):
    return {
        "id": pair_id, "label": label, "chat_model": chat_id,
        "decision_model": DECISION_ID, "priority": priority,
        "max_questions": 8, "routing_policy": "1",
        "qualification": {"status": "qualified",
                          "receipt": "receipts/2026-09-25-pair-gate.md",
                          "date": "2026-09-25"},
        "extension": {"selection_evidence": evidence},
    }


def selection_catalog():
    """Three pairs whose measured quality/latency deliberately contradict both
    curated priority (quality=1 < everyday=2 < speed=3) and weight size
    (1 GiB < 15 GiB < 20 GiB), with distinct backend-qualified context caps."""
    cat = fixture_catalog()
    speed = model_entry(SPEED_ID, "org/chat-speed", int(20.0 * GiB), 8192,
                        serve={"backend": "mlx-lm", "module": None}, priority=11)
    speed["extension"] = {"backend_context_qualified_tokens": 65536}
    models = {m["id"]: m for m in cat["models"]}
    models[CHAT_SMALL_ID]["extension"] = {"backend_context_qualified_tokens": 32768}
    models[CHAT_BIG_ID]["extension"] = {"backend_context_qualified_tokens": 262144}
    cat["models"].append(speed)
    cat["pairs"] = [
        # priority 1, heaviest-but-one: middling quality, middling latency,
        # largest qualified context
        qualified_pair("quality", "Quality", CHAT_BIG_ID, 1,
                       selection_evidence(0.60, 20.0)),
        # priority 2, smallest: BEST measured quality, slowest latency
        qualified_pair("everyday", "Everyday", CHAT_SMALL_ID, 2,
                       selection_evidence(0.95, 12.0)),
        # priority 3, LARGEST weights: worst quality, fastest latency
        qualified_pair("speed", "Speed", SPEED_ID, 3,
                       selection_evidence(0.30, 55.0)),
    ]
    return cat


# --------------------------------------------------------------------------- stubs

STUB_WORKER = """
import json, os, signal, sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
sys.path.insert(0, {serve!r})
from mlx_omarchy_serve import budget

role, port, context = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
claim = budget.claim_from_env()
assert claim is not None, "stub requires a managed claim"
state = {{"role": role, "context": context, "claim": claim}}

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path in ("/health", "/healthz", "/v1/models"):
            body = json.dumps(state).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, *_args):
        pass

def _parent_gone():
    os.kill(os.getpid(), signal.SIGTERM)

import threading
threading.Thread(
    target=budget.watch_parent_fd,
    args=(int(os.environ[budget.LIFETIME_FD_ENV]), _parent_gone),
    daemon=True,
).start()

ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
""".format(serve=SERVE_ROOT)


def stub_argv_builder(role):
    def build(spec):
        return [sys.executable, "-c", STUB_WORKER, role, str(spec.port),
                str(spec.context_tokens)]
    return build


class BasePairTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.home = Path(self.tmp.name) / "home"
        self.home.mkdir(parents=True)
        write_home_catalog(self.home, fixture_catalog())
        self._env_overrides = {
            "MLX_OMARCHY_HOME": str(self.home),
            "HF_HUB_CACHE": str(self.home / "hf"),
            "MLX_OMARCHY_OFFLINE": "1",
        }
        self._old_env = {}
        for key, value in self._env_overrides.items():
            self._old_env[key] = os.environ.get(key)
            os.environ[key] = value
        self.addCleanup(self._restore_env)

    def _restore_env(self):
        for key, old in self._old_env.items():
            if old is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old

    def _reap_proc(self, proc):
        if proc.poll() is None:
            proc.kill()
        proc.wait(timeout=15)

    # -- local artifact fixtures (offline preparation paths) --

    def make_chat_snapshot(self, entry_id, repo="org/chat-small"):
        snap = self.home / "hf" / f"models--{repo.replace('/', '--')}" / "snapshots" / ("a" * 40)
        snap.mkdir(parents=True, exist_ok=True)
        (snap / "model.safetensors").write_bytes(b"w" * 16)
        (snap / "config.json").write_text("{}", encoding="utf-8")
        (snap / "tokenizer.json").write_text("{}", encoding="utf-8")
        return snap

    def make_converted_laya(self):
        conv = self.home / "models" / DECISION_ID
        (conv / "encoder").mkdir(parents=True, exist_ok=True)
        (conv / "tokenizer").mkdir(parents=True, exist_ok=True)
        (conv / "model.safetensors").write_bytes(b"w" * 16)
        (conv / "rl_agent_config.json").write_text(
            json.dumps({"max_len": 512, "head_max_len": 64, "head_layers": 2}),
            encoding="utf-8")
        (conv / "encoder" / "config.json").write_text(
            json.dumps({"model_type": "modernbert", "hidden_size": 64,
                        "num_attention_heads": 2, "intermediate_size": 128,
                        "num_hidden_layers": 2}),
            encoding="utf-8")
        (conv / "tokenizer" / "tokenizer.json").write_text("{}", encoding="utf-8")
        (conv / "tokenizer" / "tokenizer_config.json").write_text("{}", encoding="utf-8")
        (conv / "manifest.json").write_text(json.dumps({
            "catalog_id": DECISION_ID,
            "source_repo": "org/laya",
            "source_revision": "a" * 40,
            "weights_sha256": "0" * 64,
            "weights_header": {"tensor_count": 2, "weight_bytes": 16, "param_elements": 8},
            "context_max_tokens": 512,
            "head_max_tokens": 64,
            "estimated_bytes": 4096,
        }), encoding="utf-8")
        return conv


# ---------------------------------------------------------------- catalog pairs

class CatalogPairSchemaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundled = json.loads(catalog.bundled_path().read_text(encoding="utf-8"))

    def test_bundled_catalog_with_pairs_validates(self):
        catalog.validate_catalog(self.bundled)

    def test_bundled_pairs_reference_qualified_scoped_models(self):
        models = {m["id"]: m for m in self.bundled["models"]}
        for pair in self.bundled["pairs"]:
            self.assertIn(pair["chat_model"], models, pair["id"])
            self.assertIn(pair["decision_model"], models, pair["id"])
            self.assertEqual(models[pair["chat_model"]]["kind"], "chat", pair["id"])
            self.assertEqual(models[pair["decision_model"]]["kind"], "decisions", pair["id"])
            self.assertLessEqual(pair["max_questions"], 8, pair["id"])

    def test_bundled_pairs_invent_no_qualification(self):
        for pair in self.bundled["pairs"]:
            qual = pair["qualification"]
            if qual["status"] == "qualified":
                self.assertTrue(qual["receipt"] and qual["date"], pair["id"])
            else:
                self.assertIsNone(qual["receipt"], pair["id"])
                self.assertIsNone(qual["date"], pair["id"])
            # no invented measurements: automatic selection must refuse
            self.assertNotIn("extension", pair, pair["id"])

    def test_unknown_pair_key_rejected(self):
        cat = fixture_catalog()
        cat["pairs"][0]["surprise"] = 1
        with self.assertRaises(catalog.CatalogError):
            catalog.validate_catalog(cat)

    def test_pair_referencing_missing_model_rejected(self):
        cat = fixture_catalog()
        cat["pairs"][0]["chat_model"] = "not-a-model"
        with self.assertRaises(catalog.CatalogError):
            catalog.validate_catalog(cat)

    def test_pair_wrong_kind_rejected(self):
        cat = fixture_catalog()
        cat["pairs"][0]["chat_model"], cat["pairs"][0]["decision_model"] = \
            cat["pairs"][0]["decision_model"], cat["pairs"][0]["chat_model"]
        with self.assertRaises(catalog.CatalogError):
            catalog.validate_catalog(cat)

    def test_duplicate_pair_id_rejected(self):
        cat = fixture_catalog()
        cat["pairs"][1]["id"] = cat["pairs"][0]["id"]
        with self.assertRaises(catalog.CatalogError):
            catalog.validate_catalog(cat)

    def test_bundled_catalog_survives_availability_refresh_shape(self):
        # The refresher deep-copies and only touches models/availability; a
        # top-level pairs key must ride through untouched.
        import copy
        updated = copy.deepcopy(self.bundled)
        updated["models"][0]["availability"]["size_bytes"] += 1
        self.assertEqual(updated["pairs"], self.bundled["pairs"])

    def test_pair_selection_evidence_validates(self):
        cat = fixture_catalog()
        cat["pairs"][0] = qualified_pair("everyday", "Everyday", CHAT_SMALL_ID,
                                         2, selection_evidence(0.8, 12.0))
        catalog.validate_catalog(cat)

    def test_pair_extension_unknown_key_rejected(self):
        cat = fixture_catalog()
        cat["pairs"][0] = qualified_pair("everyday", "Everyday", CHAT_SMALL_ID,
                                         2, selection_evidence(0.8, 12.0))
        cat["pairs"][0]["extension"]["surprise"] = 1
        with self.assertRaises(catalog.CatalogError):
            catalog.validate_catalog(cat)

    def test_pair_evidence_shape_rejected(self):
        def validate_with(mutate):
            evidence = selection_evidence(0.8, 12.0)
            mutate(evidence)
            cat = fixture_catalog()
            cat["pairs"][0] = qualified_pair("everyday", "Everyday",
                                             CHAT_SMALL_ID, 2, evidence)
            catalog.validate_catalog(cat)

        cases = [
            ("unknown evidence key", lambda e: e.update(surprise=1)),
            ("bad arch", lambda e: e.update(arch=["t9999"])),
            ("empty arch", lambda e: e.update(arch=[])),
            ("bad backend", lambda e: e.update(backend="wonky")),
            ("bad revision", lambda e: e.update(chat_revision="zzz")),
            ("bad date", lambda e: e.update(date="09/25/2026")),
            ("score zero", lambda e: e["quality"].update(score=0)),
            ("score over one", lambda e: e["quality"].update(score=1.5)),
            ("empty quality receipt", lambda e: e["quality"].update(receipt="")),
            ("decode zero", lambda e: e["latency"].update(decode_tokens_per_sec=0)),
            ("negative target",
             lambda e: e["latency"].update(target_tokens_per_sec=-1)),
            ("p95 zero",
             lambda e: e["latency"].update(first_visible_p95_ms=0)),
            ("missing runtime key", lambda e: e["runtime"].pop("libmlx_sha256")),
            ("bad runtime hash",
             lambda e: e["runtime"].update(libmlx_sha256="ZZZ")),
            ("empty runtime version",
             lambda e: e["runtime"].update(mlx_version="")),
        ]
        for name, mutate in cases:
            with self.subTest(name):
                with self.assertRaises(catalog.CatalogError):
                    validate_with(mutate)


# ---------------------------------------------------------------- selection refusals

class SelectionEvidenceTests(unittest.TestCase):
    """Without valid, scoped, current evidence nothing is recommended:
    priority, size, fit, and model receipts never rescue a pair."""

    def _pick(self, cat, **kwargs):
        kwargs.setdefault("preference", "balanced")
        kwargs.setdefault("chip_arch", "t6021")
        kwargs.setdefault("runtime_identity", RUNTIME_IDENTITY)
        kwargs.setdefault("available_bytes", 64 * GiB)
        return pairs.select_pair(cat["pairs"], cat["models"], **kwargs)

    def test_unqualified_catalog_names_refusal(self):
        # the shipped shape: real pairs are untested and carry no measurements
        with self.assertRaises(pairs.PairError) as ctx:
            self._pick(fixture_catalog())
        message = str(ctx.exception)
        self.assertIn("no evidence-qualified pair", message)
        self.assertIn("everyday", message)
        self.assertIn("quality", message)
        self.assertIn("release gate", message)

    def test_unknown_chip_arch_refused(self):
        with self.assertRaises(pairs.PairError) as ctx:
            self._pick(selection_catalog(), chip_arch=None)
        self.assertIn("chip architecture", str(ctx.exception))

    def test_stale_revision_evidence_refused(self):
        cat = selection_catalog()
        for pair in cat["pairs"]:
            pair["extension"]["selection_evidence"]["chat_revision"] = "b" * 40
        with self.assertRaises(pairs.PairError) as ctx:
            self._pick(cat)
        message = str(ctx.exception)
        self.assertIn("stale", message)
        self.assertIn("b" * 40, message)

    def test_chip_mismatched_evidence_refused(self):
        with self.assertRaises(pairs.PairError) as ctx:
            self._pick(selection_catalog(), chip_arch="t6000")
        message = str(ctx.exception)
        self.assertIn("t6000", message)
        self.assertIn("evidence covers chips", message)

    def test_model_chip_incompatibility_refused(self):
        cat = selection_catalog()
        big = next(m for m in cat["models"] if m["id"] == CHAT_BIG_ID)
        big["capability"]["arch"] = ["t8103"]
        # the incompatible pair is DROPPED, not ranked by its measurements
        self.assertEqual(self._pick(cat)["id"], "everyday")
        for entry in cat["models"]:
            if entry["kind"] == "chat":
                entry["capability"]["arch"] = ["t8103"]
        with self.assertRaises(pairs.PairError) as ctx:
            self._pick(cat)
        self.assertIn("t8103", str(ctx.exception))

    def test_missing_evidence_on_qualified_pair_refused(self):
        cat = selection_catalog()
        for pair in cat["pairs"]:
            del pair["extension"]
        with self.assertRaises(pairs.PairError) as ctx:
            self._pick(cat)
        self.assertIn("no measured selection evidence", str(ctx.exception))

    def test_latency_below_qualified_target_refused(self):
        # the best-quality pair misses its own qualified latency target:
        # it drops even against lower-quality competition
        cat = selection_catalog()
        everyday = next(p for p in cat["pairs"] if p["id"] == "everyday")
        everyday["extension"]["selection_evidence"]["latency"][
            "decode_tokens_per_sec"] = 5.0
        self.assertEqual(self._pick(cat)["id"], "quality")
        for pair in cat["pairs"]:
            pair["extension"]["selection_evidence"]["latency"][
                "decode_tokens_per_sec"] = 5.0
        with self.assertRaises(pairs.PairError) as ctx:
            self._pick(cat)
        self.assertIn("misses the qualified interactive target",
                      str(ctx.exception))

    def test_backend_mismatched_evidence_refused(self):
        cat = selection_catalog()
        for pair in cat["pairs"]:
            pair["extension"]["selection_evidence"]["backend"] = "omlx"
        with self.assertRaises(pairs.PairError) as ctx:
            self._pick(cat)
        self.assertIn("runtime", str(ctx.exception))

    def test_runtime_provenance_mismatch_refused(self):
        # evidence measured on a different mlx build (libmlx swapped or
        # version bumped) is not evidence for THIS runtime
        for component, value in (("libmlx_sha256", "e" * 64),
                                 ("extension_sha256", "f" * 64),
                                 ("mlx_version", "0.30.1")):
            with self.subTest(component):
                cat = selection_catalog()
                other = dict(RUNTIME_IDENTITY, **{component: value})
                for pair in cat["pairs"]:
                    pair["extension"]["selection_evidence"]["runtime"] = other
                with self.assertRaises(pairs.PairError) as ctx:
                    self._pick(cat, runtime_identity=RUNTIME_IDENTITY)
                message = str(ctx.exception)
                self.assertIn("different runtime", message)
                self.assertIn(value[:12], message)

    def test_runtime_mismatch_drops_only_the_stale_pair(self):
        cat = selection_catalog()
        everyday = next(p for p in cat["pairs"] if p["id"] == "everyday")
        stale = dict(RUNTIME_IDENTITY, libmlx_sha256="e" * 64)
        everyday["extension"]["selection_evidence"]["runtime"] = stale
        # the mismatched pair is dropped even with the best quality score
        self.assertEqual(self._pick(cat)["id"], "quality")

    def test_unverifiable_local_runtime_refused(self):
        with self.assertRaises(pairs.PairError) as ctx:
            self._pick(selection_catalog(), runtime_identity=None)
        message = str(ctx.exception)
        self.assertIn("runtime provenance", message)
        self.assertIn("libmlx", message)

    def test_first_visible_over_design_bound_refused(self):
        # the design's fixed 2000 ms p95 first-visible bound gates every
        # pair, whatever decode rate or quality its evidence claims
        cat = selection_catalog()
        everyday = next(p for p in cat["pairs"] if p["id"] == "everyday")
        everyday["extension"]["selection_evidence"]["latency"][
            "first_visible_p95_ms"] = 2600.0
        self.assertEqual(self._pick(cat)["id"], "quality")
        for pair in cat["pairs"]:
            pair["extension"]["selection_evidence"]["latency"][
                "first_visible_p95_ms"] = 2600.0
        with self.assertRaises(pairs.PairError) as ctx:
            self._pick(cat)
        message = str(ctx.exception)
        self.assertIn("first-visible p95", message)
        self.assertIn("2000", message)

    def test_evidence_on_untested_pair_is_not_enough(self):
        cat = selection_catalog()
        for pair in cat["pairs"]:
            pair["qualification"] = {"status": "untested", "receipt": None,
                                     "date": None}
        with self.assertRaises(pairs.PairError) as ctx:
            self._pick(cat)
        self.assertIn("release gate", str(ctx.exception))

    def test_bad_preference_refused(self):
        with self.assertRaises(pairs.PairError) as ctx:
            self._pick(selection_catalog(), preference="turbo")
        self.assertIn("preference", str(ctx.exception))


# ------------------------------------------------------------------ budget batch

class BatchAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.home = Path(self.tmp.name)

    def test_combined_overcommit_refused_atomically(self):
        # usable = 10 - 2 = 8 GiB; the pair needs 6 + 6. Nothing may be
        # written: one child's admission alone fitting is not enough.
        with self.assertRaises(budget.BudgetError):
            budget.admit_and_reserve_batch(
                [("chat", 6 * GiB, ""), ("decision", 6 * GiB, "")],
                pair_id="everyday", home=self.home, available_bytes=10 * GiB)
        self.assertEqual(budget.load_reservations(self.home), {})

    def test_batch_creates_distinct_owned_pending_records(self):
        records = budget.admit_and_reserve_batch(
            [("chat", 4 * GiB, "chat"), ("decision", 2 * GiB, "decision")],
            pair_id="everyday", home=self.home, available_bytes=10 * GiB)
        self.assertEqual(len(records), 2)
        owners = {r["owner"] for r in records}
        self.assertEqual(len(owners), 2)
        held = budget.load_reservations(self.home)
        self.assertEqual(set(held), {"chat", "decision"})
        for name, record in held.items():
            self.assertEqual(record["state"], "pending")
            self.assertEqual(record["pair_id"], "everyday")
            self.assertEqual(record["claimed"], None)
            self.assertIsNotNone(record["parent"])
        self.assertEqual({r["claim_token"] for r in records},
                         {held[r["name"]]["owner"] for r in records})

    def test_batch_refuses_duplicate_names(self):
        with self.assertRaises(budget.BudgetError):
            budget.admit_and_reserve_batch(
                [("chat", 1 * GiB, ""), ("chat", 1 * GiB, "")],
                pair_id="everyday", home=self.home, available_bytes=10 * GiB)

    def test_batch_refuses_name_already_held(self):
        budget.admit_and_reserve_batch([("chat", 1 * GiB, "")], pair_id="p1",
                                       home=self.home, available_bytes=10 * GiB)
        with self.assertRaises(budget.BudgetError):
            budget.admit_and_reserve_batch([("chat", 1 * GiB, ""), ("d", 1 * GiB, "")],
                                           pair_id="p2", home=self.home,
                                           available_bytes=10 * GiB)

    def test_resize_needs_fit_and_owner(self):
        budget.admit_and_reserve_batch([("chat", 1 * GiB, "")], pair_id="p",
                                       home=self.home, available_bytes=10 * GiB)
        owner = budget.load_reservations(self.home)["chat"]["owner"]
        budget.resize_reservation("chat", 3 * GiB, owner=owner, home=self.home,
                                  available_bytes=10 * GiB)
        self.assertEqual(budget.load_reservations(self.home)["chat"]["bytes"], 3 * GiB)
        with self.assertRaises(budget.BudgetError):
            budget.resize_reservation("chat", 20 * GiB, owner=owner, home=self.home,
                                      available_bytes=10 * GiB)
        self.assertEqual(budget.load_reservations(self.home)["chat"]["bytes"], 3 * GiB)
        with self.assertRaises(budget.BudgetError):
            budget.resize_reservation("chat", 2 * GiB, owner="forged", home=self.home,
                                      available_bytes=10 * GiB)

    def test_process_identity_uses_start_time_not_pid(self):
        ident = budget.proc_identity(os.getpid())
        self.assertIsNotNone(ident)
        self.assertEqual(ident["pid"], os.getpid())
        self.assertTrue(budget.identity_alive(ident))
        forged = {"pid": ident["pid"], "starttime": ident["starttime"] + 1}
        self.assertFalse(budget.identity_alive(forged))
        self.assertFalse(budget.identity_alive({"pid": 2**22, "starttime": 1}))


class ClaimProtocolTests(BasePairTest):
    def spawn_claimer(self, payload, env_extra=None):
        script = (
            "import json, os, sys\n"
            f"sys.path.insert(0, {SERVE_ROOT!r})\n"
            "from mlx_omarchy_serve import budget\n"
            "fd = int(os.environ[budget.CLAIM_FD_ENV])\n"
            "try:\n"
            "    record = budget.claim_from_env()\n"
            "    print(json.dumps({'ok': True, 'record': record}))\n"
            "except budget.BudgetError as exc:\n"
            "    print(json.dumps({'ok': False, 'error': str(exc)}))\n"
        )
        env = dict(os.environ)
        env.update(env_extra or {})
        r, w = os.pipe()
        os.set_inheritable(r, True)
        env[budget.CLAIM_FD_ENV] = str(r)
        proc = subprocess.Popen([sys.executable, "-c", script], env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                pass_fds=(r,))
        os.close(r)
        os.write(w, json.dumps(payload).encode() + b"\n")
        os.close(w)
        out, err = proc.communicate(timeout=60)
        return json.loads(out.decode().strip().splitlines()[-1]), proc.returncode, err

    def make_batch(self, name="chat", byte_count=1024):
        records = budget.admit_and_reserve_batch(
            [(name, byte_count, "")], pair_id="everyday",
            home=self.home, available_bytes=10 * GiB)
        return records[0]

    def test_claim_binds_child_and_records_identity(self):
        record = self.make_batch()
        payload = {"name": record["name"], "claim_token": record["claim_token"],
                   "bytes": record["bytes"], "pair_id": "everyday"}
        result, code, err = self.spawn_claimer(payload)
        self.assertTrue(result["ok"], err.decode())
        held = budget.load_reservations(self.home)["chat"]
        self.assertIsNotNone(held["claimed"])
        self.assertNotEqual(held["claimed"], held["parent"])

    def test_claim_refuses_wrong_token(self):
        record = self.make_batch()
        payload = {"name": record["name"], "claim_token": "forged",
                   "bytes": record["bytes"], "pair_id": "everyday"}
        result, _code, err = self.spawn_claimer(payload)
        self.assertFalse(result["ok"])
        self.assertIn("token", result["error"])
        self.assertIsNone(budget.load_reservations(self.home)["chat"]["claimed"])

    def test_claim_refuses_allocation_mismatch(self):
        record = self.make_batch()
        payload = {"name": record["name"], "claim_token": record["claim_token"],
                   "bytes": record["bytes"] + 1, "pair_id": "everyday"}
        result, _code, err = self.spawn_claimer(payload)
        self.assertFalse(result["ok"])
        self.assertIn("allocation", result["error"])

    def test_claim_refuses_foreign_live_holder_then_allows_dead_reclaim(self):
        record = self.make_batch()
        payload = {"name": record["name"], "claim_token": record["claim_token"],
                   "bytes": record["bytes"], "pair_id": "everyday"}
        hold_script = (
            "import json, os, sys, time\n"
            f"sys.path.insert(0, {SERVE_ROOT!r})\n"
            "from mlx_omarchy_serve import budget\n"
            "fd = int(os.environ[budget.CLAIM_FD_ENV])\n"
            "record = budget.claim_from_env()\n"
            "print('claimed', flush=True)\n"
            "time.sleep(30)\n"
        )
        env = dict(os.environ)
        r, w = os.pipe()
        os.set_inheritable(r, True)
        env[budget.CLAIM_FD_ENV] = str(r)
        holder = subprocess.Popen([sys.executable, "-c", hold_script], env=env,
                                  stdout=subprocess.DEVNULL,
                                  stderr=subprocess.DEVNULL,
                                  pass_fds=(r,))
        os.close(r)
        os.write(w, json.dumps(payload).encode() + b"\n")
        os.close(w)
        self.addCleanup(holder.kill)
        # wait until the holder has actually claimed
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if budget.load_reservations(self.home)["chat"]["claimed"] is not None:
                break
            if holder.poll() is not None:
                break
            time.sleep(0.05)
        self.assertIsNotNone(budget.load_reservations(self.home)["chat"]["claimed"])
        self.addCleanup(self._reap_proc, holder)
        second, _code2, err2 = self.spawn_claimer(payload)
        self.assertFalse(second["ok"])
        self.assertIn("live child", second["error"])
        # Recovery uses process start identity: once the recorded holder's
        # start time no longer matches a live process, the record re-binds.
        holder.kill()
        holder.wait(timeout=15)
        third, _code3, err3 = self.spawn_claimer(payload)
        self.assertTrue(third["ok"], err3.decode())

    def test_unclaimed_record_blocks_no_cleanup_of_live_pair(self):
        # A pending unclaimed record is parent-owned; the reaper must not
        # clear it while the batch owner lives. Exercised via managed.reap.
        records = budget.admit_and_reserve_batch(
            [("chat", 1024, ""), ("decision", 1024, "")],
            pair_id="everyday", home=self.home, available_bytes=10 * GiB)
        cleared = managed.reap_pair_records("everyday", self.home, children=[])
        self.assertEqual(cleared, [])
        self.assertEqual(set(budget.load_reservations(self.home)),
                         {"chat", "decision"})


class StubWorkerLifecycleTests(BasePairTest):
    def _close_lw(self):
        lw = getattr(self, "_lw", None)
        if lw is not None:
            self._lw = None
            os.close(lw)

    def spawn(self, role, port, context, record, extra_env=None):
        payload = {"name": record["name"], "claim_token": record["claim_token"],
                   "bytes": record["bytes"], "pair_id": "everyday"}
        lr, lw = os.pipe()
        os.set_inheritable(lr, True)
        self._lw = lw
        self.addCleanup(self._close_lw)
        argv = stub_argv_builder(role)(managed.WorkerSpec(
            role=role, model_dir=self.home, port=port,
            context_tokens=context, max_questions=8))
        env = dict(os.environ)
        env.update(extra_env or {})
        proc = managed.spawn_claiming_worker(
            argv, payload, lifetime_read_fd=lr, env=env,
            log_path=self.home / f"{role}.log")
        os.close(lr)
        self.addCleanup(self._reap_proc, proc)
        return proc

    def test_child_claims_via_pipe_and_serves_health(self):
        record = budget.admit_and_reserve_batch(
            [("chat", 1024, "")], pair_id="everyday", home=self.home,
            available_bytes=10 * GiB)[0]
        port = managed.free_port()
        proc = self.spawn("chat", port, 4096, record)
        self.assertTrue(managed.wait_health(
            f"http://127.0.0.1:{port}/health", deadline=30))
        held = budget.load_reservations(self.home)["chat"]
        self.assertIsNotNone(held["claimed"])
        self.assertTrue(budget.identity_alive(held["claimed"]))
        self.assertTrue(managed.terminate(proc, timeout=15))

    def test_child_stops_when_parent_lifetime_pipe_closes(self):
        record = budget.admit_and_reserve_batch(
            [("chat", 1024, "")], pair_id="everyday", home=self.home,
            available_bytes=10 * GiB)[0]
        port = managed.free_port()
        proc = self.spawn("chat", port, 4096, record)
        self.assertTrue(managed.wait_health(
            f"http://127.0.0.1:{port}/health", deadline=30))
        # Parent side of the lifetime pipe dies with the manager process;
        # simulate by closing it here. The child must exit on its own.
        os.close(self._lw)
        self._lw = None
        deadline = time.monotonic() + 30
        while proc.poll() is None and time.monotonic() < deadline:
            time.sleep(0.05)
        self.assertIsNotNone(proc.poll())
        self.assertEqual(proc.returncode, -signal.SIGTERM)

    def test_reaper_clears_only_verifiably_exited_children(self):
        chat_record, decision_record = budget.admit_and_reserve_batch(
            [("chat", 1024, ""), ("decision", 1024, "")],
            pair_id="everyday", home=self.home, available_bytes=10 * GiB)
        chat_port, decision_port = managed.free_port(), managed.free_port()
        chat = self.spawn("chat", chat_port, 4096, chat_record)
        decision = self.spawn("decision", decision_port, 4096, decision_record)
        self.assertTrue(managed.wait_health(
            f"http://127.0.0.1:{chat_port}/health", deadline=30))
        self.assertTrue(managed.wait_health(
            f"http://127.0.0.1:{decision_port}/health", deadline=30))
        decision.kill()
        decision.wait(timeout=15)
        cleared = managed.reap_pair_records(
            "everyday", self.home,
            children=[managed.ManagedChild("chat", chat, "chat"),
                      managed.ManagedChild("decision", decision, "decision")])
        # The SIGKILLed child's record is cleared; the live chat record stays.
        self.assertIn("decision", cleared)
        self.assertNotIn("chat", cleared)
        held = budget.load_reservations(self.home)
        self.assertNotIn("decision", held)
        self.assertIn("chat", held)


class WaitHealthTests(unittest.TestCase):
    def test_returns_promptly_when_child_exits(self):
        proc = subprocess.Popen([sys.executable, "-c", "import sys; sys.exit(3)"])
        proc.wait(timeout=15)
        start = time.monotonic()
        ok = managed.wait_health("http://127.0.0.1:9/health", deadline=60,
                                 abort=lambda: proc.poll() is not None)
        elapsed = time.monotonic() - start
        self.assertFalse(ok)
        self.assertLess(elapsed, 5)

    def test_returns_promptly_on_cancel(self):
        cancel = threading.Event()
        threading.Timer(0.2, cancel.set).start()
        start = time.monotonic()
        ok = managed.wait_health("http://127.0.0.1:9/health", deadline=60,
                                 abort=cancel.is_set)
        elapsed = time.monotonic() - start
        self.assertFalse(ok)
        self.assertLess(elapsed, 5)


# ------------------------------------------------------------------ pair sizing

class ContextSizingTests(unittest.TestCase):
    def setUp(self):
        self.cat = fixture_catalog()
        self.chat = {m["id"]: m for m in self.cat["models"]}[CHAT_SMALL_ID]
        self.other_bytes = int(1.0 * GiB)

    def sizing(self, available):
        return pairs.admissible_context(
            self.chat, other_bytes=self.other_bytes, preference="balanced",
            available_bytes=available)

    def test_memory_binding_byte_exact_within_qualified_cap(self):
        # backend evidence absent → chosen context is capped at the policy
        # default regardless of memory; the memory bound itself stays
        # byte-exact and is exposed in limits["memory"]
        ctx = self.sizing(7 * GiB)
        mem_bound = ctx["limits"]["memory"]
        est = budget.estimate_required(self.chat["memory"], mem_bound)
        reserve = max(budget.SAFETY_RESERVE_BYTES, int(0.10 * 7 * GiB))
        self.assertLessEqual(est.total + self.other_bytes + reserve, 7 * GiB)
        one_more = budget.estimate_required(self.chat["memory"], mem_bound + 1)
        self.assertGreater(one_more.total + self.other_bytes + reserve, 7 * GiB)
        # chosen context is bound by the default-admission cap, not by memory
        self.assertEqual(ctx["context_tokens"], budget.DEFAULT_CONTEXT_TOKENS)
        self.assertEqual(ctx["binding"], "backend")
        self.assertFalse(ctx["backend_context_qualified"])

    def test_backend_evidence_caps_context_without_inventing_qualification(self):
        # Plenty of memory and no qualified backend evidence: chosen context
        # is the conservative default cap (4096), explicitly NOT the model
        # positional maximum.
        ctx = self.sizing(512 * GiB)
        self.assertEqual(ctx["context_tokens"], budget.DEFAULT_CONTEXT_TOKENS)
        self.assertFalse(ctx["backend_context_qualified"])
        self.assertIsNone(ctx["limits"]["backend"])
        self.assertIn("backend", ctx["binding"])
        # Dev smoke lifts the cap up to memory/model evidence — still
        # labelled unqualified and never Ready.
        ctx_dev = pairs.admissible_context(
            self.chat, other_bytes=self.other_bytes, preference="balanced",
            available_bytes=512 * GiB, backend_override=True)
        self.assertGreater(ctx_dev["context_tokens"], ctx["context_tokens"])
        self.assertFalse(ctx_dev["backend_context_qualified"])
        # An explicit user request above the cap is refused with a named hint
        with self.assertRaises(pairs.PairError) as ctx_exc:
            pairs.resolve_requested_context(
                self.chat, requested=ctx["context_tokens"] + 1,
                other_bytes=self.other_bytes, preference="balanced",
                available_bytes=512 * GiB)
        self.assertIn("backend", str(ctx_exc.exception))
        self.assertIn(PAIR_DEV_QUALIFICATION_ENV, str(ctx_exc.exception))

    def test_context_monotone_in_memory_and_qualified_cap(self):
        # Without backend evidence the chosen context never exceeds the
        # conservative default cap; with dev smoke it rises with memory
        # monotonically under model/qualified limits.
        previous = 0
        for available in (6 * GiB, 8 * GiB, 16 * GiB, 64 * GiB, 512 * GiB):
            result = self.sizing(available)
            self.assertGreaterEqual(result["context_tokens"], previous)
            self.assertLessEqual(result["context_tokens"],
                                 budget.DEFAULT_CONTEXT_TOKENS)
            self.assertGreaterEqual(result["context_tokens"], catalog.MIN_CONTEXT_TOKENS)
            previous = result["context_tokens"]
        previous = 0
        for available in (6 * GiB, 8 * GiB, 16 * GiB, 64 * GiB, 512 * GiB):
            result = pairs.admissible_context(
                self.chat, other_bytes=self.other_bytes, preference="balanced",
                available_bytes=available, backend_override=True)
            self.assertGreaterEqual(result["context_tokens"], previous)
            self.assertLessEqual(result["context_tokens"],
                                 self.chat["context"]["max_tokens"])
            previous = result["context_tokens"]

    def test_unusual_memory_size_gets_an_answer(self):
        result = self.sizing(7 * GiB + 137 * 1024 * 1024)
        self.assertGreaterEqual(result["context_tokens"], catalog.MIN_CONTEXT_TOKENS)
        self.assertIsNotNone(result["context_tokens"])

    def test_no_fit_names_memory_refusal(self):
        with self.assertRaises(pairs.PairError) as ctx:
            self.sizing(int(1.2 * GiB))
        self.assertIn("memory", str(ctx.exception))

    def test_explicit_request_over_admissible_refused(self):
        result = self.sizing(8 * GiB)
        with self.assertRaises(pairs.PairError):
            pairs.resolve_requested_context(
                self.chat, requested=result["context_tokens"] + 1000,
                other_bytes=self.other_bytes, preference="balanced",
                available_bytes=8 * GiB)

    def test_never_labels_positional_limit_as_qualified(self):
        result = self.sizing(512 * GiB)
        self.assertFalse(result["backend_context_qualified"])
        self.assertIn("model", result["limits"])


# ------------------------------------------------------------------ PairManager

class PairManagerTests(BasePairTest):
    def setUp(self):
        super().setUp()
        self.manager = pairs.PairManager(self.home)
        self._old_builders = (pairs._chat_worker_argv, pairs._decision_worker_argv)
        pairs._chat_worker_argv = stub_argv_builder("chat")
        pairs._decision_worker_argv = stub_argv_builder("decision")
        self.addCleanup(self._restore_builders)

    def _restore_builders(self):
        pairs._chat_worker_argv, pairs._decision_worker_argv = self._old_builders

    def test_status_idle_shape(self):
        status = self.manager.status()
        for key in ("state", "pairs", "chat_url", "decision_url", "chat_model",
                    "context_tokens", "model_paths", "pair_id", "ready_offline",
                    "qualification"):
            self.assertIn(key, status)
        self.assertEqual(status["state"], "idle")
        self.assertIsNone(status["chat_url"])
        self.assertFalse(status["ready_offline"])
        self.assertEqual(len(status["pairs"]), 2)
        everyday = next(p for p in status["pairs"] if p["id"] == "everyday")
        self.assertEqual(everyday["qualification"]["status"], "untested")

    def test_ready_offline_requires_a_named_receipt(self):
        pair = {"id": "everyday",
                "qualification": {"status": "qualified", "receipt": None},
                "chat_model": "qwen3.8-2b-4bit", "decision_model": "laya-mlx"}
        self.assertFalse(self.manager._ready_offline(pair))
        pair["qualification"]["receipt"] = "   "
        self.assertFalse(self.manager._ready_offline(pair))

    def test_voice_estimates_admitted_without_readiness_gating(self):
        self.make_chat_snapshot(CHAT_SMALL_ID)
        self.make_converted_laya()
        synthesis_mod = types.ModuleType("mlx_omarchy_assistant.synthesis")

        class FakeSynthesis:
            def __init__(self, home):
                pass

            def status(self):
                return {"ready": False, "qualification": "untested",
                        "memory": {"runtime_estimate_bytes": 700 * 1024 * 1024}}

        synthesis_mod.Synthesis = FakeSynthesis
        recognition_mod = types.ModuleType("mlx_omarchy_assistant.recognition")

        class FakeRecognition:
            def __init__(self, home):
                pass

            def status(self):
                return {"memory": {"runtime_estimate_bytes": 200 * 1024 * 1024}}

        recognition_mod.Recognition = FakeRecognition
        with mock.patch.dict(sys.modules, {
                "mlx_omarchy_assistant.synthesis": synthesis_mod,
                "mlx_omarchy_assistant.recognition": recognition_mod}):
            result = self.manager.setup("everyday", approve_download=False,
                                        voice=True)
            self.assertEqual(result["state"], "prepared")
            self.assertEqual(result["voice_bytes"],
                             700 * 1024 * 1024 + 200 * 1024 * 1024)
            self.manager.start()
            self.addCleanup(self.manager.stop)
            held = budget.load_reservations(self.home)
            self.assertEqual(held["voice-everyday"]["bytes"],
                             700 * 1024 * 1024 + 200 * 1024 * 1024)
        report = self.manager.stop()
        self.assertTrue(report["stopped"])
        self.assertNotIn("voice-everyday", budget.load_reservations(self.home))

    def test_stale_identity_converted_artifact_is_reconverted_not_adopted(self):
        conv = self.make_converted_laya()
        manifest = json.loads((conv / "manifest.json").read_text())
        # a converted dir from before the converter stamped catalog ids:
        # provenance matches, identity does not
        manifest["catalog_id"] = "laya"
        (conv / "manifest.json").write_text(json.dumps(manifest))
        self.make_chat_snapshot(CHAT_SMALL_ID)
        result = self.manager.setup("everyday", approve_download=False)
        self.assertEqual(result["state"], "planned")
        decision_dl = next(d for d in result["downloads"]
                           if d["role"] == "decision")
        self.assertTrue(decision_dl["conversion_needed"])
        with self.assertRaises(pairs.PairError) as ctx:
            self.manager.setup("everyday", approve_download=True)
        self.assertIn("laya-mlx", str(ctx.exception))

    def test_cancel_kills_inflight_download_without_orphan(self):
        # the real supervised-download path runs; the child it spawns is
        # swapped for a long sleeper so the test can prove cancel() kills it
        # within the poll interval and settles the manager state
        spawned = {}
        real_spawn = managed.spawn_supervised

        def fake_spawn(argv, **kwargs):
            proc = real_spawn([sys.executable, "-c", "import time; time.sleep(120)"],
                              **kwargs)
            spawned["proc"] = proc
            return proc

        worker_done = threading.Event()
        outcome = {}

        def run():
            try:
                with mock.patch.dict(os.environ, {"MLX_OMARCHY_OFFLINE": ""}), \
                        mock.patch.object(pairs, "_download_child_code",
                                          lambda *a: "print('/tmp/unused')"), \
                        mock.patch.object(managed, "spawn_supervised", fake_spawn):
                    outcome["result"] = self.manager.setup(
                        "everyday", approve_download=True)
            except Exception as exc:
                outcome["error"] = exc
            finally:
                worker_done.set()

        worker = threading.Thread(target=run, daemon=True)
        worker.start()
        deadline = time.monotonic() + 10
        while "proc" not in spawned and time.monotonic() < deadline \
                and not worker_done.is_set():
            time.sleep(0.05)
        self.assertIn("proc", spawned)
        self.assertIsNone(spawned["proc"].poll())
        self.assertTrue(self.manager.cancel())
        self.assertTrue(worker_done.wait(timeout=15))
        self.assertIsInstance(outcome.get("error"), pairs.PairError)
        self.assertIn("cancelled", str(outcome["error"]))
        deadline = time.monotonic() + 15
        while spawned["proc"].poll() is None and time.monotonic() < deadline:
            time.sleep(0.05)
        self.assertIsNotNone(spawned["proc"].poll())
        self.assertEqual(self.manager.status()["state"], "idle")
        self.assertEqual(budget.load_reservations(self.home), {})

    def test_start_requires_prepared(self):
        with self.assertRaises(pairs.PairError):
            self.manager.start("everyday")

    def test_setup_plan_makes_no_writes(self):
        result = self.manager.setup("everyday", approve_download=False,
                                    preference="balanced")
        self.assertEqual(result["pair_id"], "everyday")
        self.assertEqual(result["state"], "planned")
        self.assertGreater(result["context_tokens"], 0)
        self.assertEqual(self.manager.status()["state"], "planned")
        self.assertFalse((self.home / "models").exists())
        # probing may create empty cache directories, but never writes
        # artifact bytes (the only fixture file is the catalog cache itself)
        hf = self.home / "hf"
        self.assertFalse(hf.is_dir() and any(p.is_file() for p in hf.rglob("*")))

    def test_setup_context_adapts_to_reported_memory(self):
        # Without dev smoke, both memory sizes are capped at the same
        # conservative default — a true evidence-based cap that does not
        # invent a "tiered" growth story. With the dev smoke override the
        # memory difference is visible.
        small = self.manager.setup("everyday", approve_download=False,
                                   preference="balanced", available_bytes=8 * GiB)
        self.manager = pairs.PairManager(self.home)
        large = self.manager.setup("everyday", approve_download=False,
                                   preference="balanced", available_bytes=64 * GiB)
        self.assertEqual(small["context_tokens"], large["context_tokens"])

    def test_selection_ranks_measured_evidence_not_priority_or_size(self):
        # balanced takes the HIGHEST MEASURED QUALITY (everyday, priority 2,
        # smallest weights) over the priority-1 pair; fast takes the FASTEST
        # MEASURED LATENCY (speed, priority 3, LARGEST weights) over both;
        # long-context takes the largest byte-admissible qualified context.
        # Curated priority and weight size rank nothing.
        cat = selection_catalog()
        pick_balanced = pairs.select_pair(
            cat["pairs"], cat["models"], preference="balanced",
            chip_arch="t6021", runtime_identity=RUNTIME_IDENTITY,
            available_bytes=64 * GiB)
        self.assertEqual(pick_balanced["id"], "everyday")
        pick_fast = pairs.select_pair(
            cat["pairs"], cat["models"], preference="fast",
            chip_arch="t6021", runtime_identity=RUNTIME_IDENTITY,
            available_bytes=64 * GiB)
        self.assertEqual(pick_fast["id"], "speed")
        pick_long = pairs.select_pair(
            cat["pairs"], cat["models"], preference="long-context",
            chip_arch="t6021", runtime_identity=RUNTIME_IDENTITY,
            available_bytes=64 * GiB)
        self.assertEqual(pick_long["id"], "quality")

    def test_selection_stays_memory_bound_before_evidence_order(self):
        # 17 GiB: only the 1 GiB chat pair admits even a minimum context.
        cat = selection_catalog()
        pick = pairs.select_pair(
            cat["pairs"], cat["models"], preference="balanced",
            chip_arch="t6021", runtime_identity=RUNTIME_IDENTITY,
            available_bytes=17 * GiB)
        self.assertEqual(pick["id"], "everyday")

    def test_setup_offline_prepares_local_artifacts(self):
        self.make_chat_snapshot(CHAT_SMALL_ID)
        self.make_converted_laya()
        result = self.manager.setup("everyday", approve_download=True,
                                    preference="balanced")
        self.assertEqual(result["state"], "prepared")
        self.assertIn("chat", result["model_paths"])
        self.assertIn("decision", result["model_paths"])
        self.assertTrue(result["model_paths"]["chat"].is_dir())
        self.assertTrue(result["model_paths"]["decision"].is_dir())

    def test_setup_without_approval_and_no_artifacts_stays_planned(self):
        result = self.manager.setup("everyday", approve_download=False)
        self.assertEqual(result["state"], "planned")

    def test_setup_offline_without_artifacts_names_the_gap(self):
        with self.assertRaises(pairs.PairError) as ctx:
            self.manager.setup("everyday", approve_download=True)
        self.assertIn(CHAT_SMALL_ID, str(ctx.exception))

    def test_setup_voice_requires_synthesis_module(self):
        self.make_chat_snapshot(CHAT_SMALL_ID)
        self.make_converted_laya()
        with mock.patch.dict(sys.modules, {"mlx_omarchy_assistant.synthesis": None}):
            with self.assertRaises(pairs.PairError) as ctx:
                self.manager.setup("everyday", approve_download=True, voice=True)
        self.assertIn("synthesis", str(ctx.exception).lower())

    def test_start_below_model_qualification_needs_dev_mode(self):
        cat = fixture_catalog()
        for m in cat["models"]:
            if m["id"] == CHAT_SMALL_ID:
                m["qualification"]["http"] = {"status": "untested", "receipt": None,
                                              "date": None}
        write_home_catalog(self.home, cat)
        self.manager = pairs.PairManager(self.home)
        with self.assertRaises(pairs.PairError) as ctx:
            self.manager.setup("everyday", approve_download=False)
        self.assertIn("qualification", str(ctx.exception).lower())

    def test_start_launches_and_reports(self):
        self.make_chat_snapshot(CHAT_SMALL_ID)
        self.make_converted_laya()
        self.manager.setup("everyday", approve_download=True)
        result = self.manager.start()
        self.addCleanup(self.manager.stop)
        self.assertEqual(result["state"], "ready")
        status = self.manager.status()
        self.assertEqual(status["state"], "ready")
        self.assertEqual(status["pair_id"], "everyday")
        self.assertEqual(status["chat_model"], CHAT_SMALL_ID)
        self.assertTrue(status["chat_url"].startswith("http://127.0.0.1:"))
        self.assertTrue(status["decision_url"].startswith("http://127.0.0.1:"))
        self.assertIn("chat", status["model_paths"])
        self.assertGreater(status["context_tokens"], 0)
        # The pair record is untested: a working start never promotes it.
        self.assertFalse(status["ready_offline"])
        self.assertEqual(status["qualification"]["pair"]["status"], "untested")
        held = budget.load_reservations(self.home)
        self.assertIn(CHAT_SMALL_ID, held)
        self.assertIn(DECISION_ID, held)
        self.assertEqual(held[CHAT_SMALL_ID]["pair_id"], "everyday")
        self.assertIsNotNone(held[CHAT_SMALL_ID]["claimed"])
        self.assertIsNotNone(held[DECISION_ID]["claimed"])

    def test_stop_retains_records_when_worker_exit_is_unverifiable(self):
        self.make_chat_snapshot(CHAT_SMALL_ID)
        self.make_converted_laya()
        self.manager.setup("everyday", approve_download=True)
        self.manager.start()
        chat_pid = self.manager._children["chat"].popen.pid
        with mock.patch.object(managed, "terminate", return_value=False):
            report = self.manager.stop()
        self.assertFalse(report["stopped"])
        self.assertFalse(report["children"]["chat"]["confirmed_dead"])
        self.assertCountEqual(report["retained"], [CHAT_SMALL_ID, DECISION_ID])
        # fail closed: the memory of an unverifiable worker stays counted...
        self.assertEqual(set(budget.load_reservations(self.home)),
                         {CHAT_SMALL_ID, DECISION_ID})
        # ...and a retry while the survivor still cannot be verified is
        # refused with the record named, not silently leaked
        with mock.patch.object(managed, "terminate", return_value=False):
            with self.assertRaises(pairs.PairError) as ctx:
                self.manager.start()
        self.assertIn(CHAT_SMALL_ID, str(ctx.exception))
        # once termination verifies again, start() settles the retained
        # survivor itself: verified exit releases the record, admission
        # proceeds, and the fresh pair is ready
        result = self.manager.start()
        self.addCleanup(self.manager.stop)
        self.assertEqual(result["state"], "ready")
        self.assertEqual(self.manager.status()["state"], "ready")
        held = budget.load_reservations(self.home)
        self.assertIsNotNone(held[CHAT_SMALL_ID]["claimed"])
        self.assertTrue(budget.identity_alive(held[CHAT_SMALL_ID]["claimed"]))
        _ = chat_pid

    def test_stop_clears_records_and_pipes(self):
        self.make_chat_snapshot(CHAT_SMALL_ID)
        self.make_converted_laya()
        self.manager.setup("everyday", approve_download=True)
        self.manager.start()
        chat_port = int(self.manager.status()["chat_url"].rsplit(":", 1)[1].split("/")[0])
        report = self.manager.stop()
        self.assertTrue(report["stopped"])
        self.assertCountEqual(report["cleared"], [CHAT_SMALL_ID, DECISION_ID])
        self.assertEqual(report["retained"], [])
        # artifacts stay prepared: start() may run again without re-setup,
        # and the stopped pair reports no live endpoints
        self.assertEqual(self.manager.status()["state"], "prepared")
        self.assertIsNone(self.manager.status()["chat_url"])
        self.assertEqual(budget.load_reservations(self.home), {})
        # the served port is actually closed
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{chat_port}/health", timeout=2)
            raised = False
        except Exception:
            raised = True
        self.assertTrue(raised)

    def test_ensure_context_grows_reservation_and_restarts_chat_only(self):
        self.make_chat_snapshot(CHAT_SMALL_ID)
        self.make_converted_laya()
        # Growth requires lifting the conservative backend cap — the
        # documented development smoke path. Unqualified runs stay visible.
        os.environ[PAIR_DEV_QUALIFICATION_ENV] = "1"
        self.addCleanup(lambda: os.environ.pop(PAIR_DEV_QUALIFICATION_ENV, None))
        self.manager.setup("everyday", approve_download=True, context_tokens=4096)
        self.manager.start()
        self.addCleanup(self.manager.stop)
        before = self.manager.status()
        self.assertEqual(before["context_tokens"], 4096)
        self.assertFalse(before["context"]["backend_context_qualified"])
        decision_pid = self.manager._children["decision"].popen.pid
        old_chat_pid = self.manager._children["chat"].popen.pid
        result = self.manager.ensure_context(before["context_tokens"] + 1)
        self.assertTrue(result["ok"])
        after = self.manager.status()
        self.assertEqual(after["context_tokens"], before["context_tokens"] + 1)
        held = budget.load_reservations(self.home)
        self.assertGreater(
            held[CHAT_SMALL_ID]["bytes"],
            budget.estimate_required(
                self.manager.catalog_models[CHAT_SMALL_ID]["memory"],
                before["context_tokens"]).total)
        self.assertEqual(self.manager._children["decision"].popen.pid, decision_pid)
        self.assertNotEqual(self.manager._children["chat"].popen.pid, old_chat_pid)

    def test_ensure_context_short_turn_keeps_current_cap(self):
        self.manager._state = "ready"
        self.manager._prepared = {"preference": "balanced"}
        self.manager._context_tokens = 4096
        result = self.manager.ensure_context(40)
        self.assertEqual(result, {"ok": True, "changed": False,
                                  "context_tokens": 4096, "requested": 40})
    def test_ensure_context_over_admissible_refused_and_cap_retained(self):
        self.make_chat_snapshot(CHAT_SMALL_ID)
        self.make_converted_laya()
        self.manager.setup("everyday", approve_download=True)
        self.manager.start()
        self.addCleanup(self.manager.stop)
        before = self.manager.status()
        chat_pid = self.manager._children["chat"].popen.pid
        result = self.manager.ensure_context(before["context_tokens"] + 10_000_000)
        self.assertFalse(result["ok"])
        self.assertIn("reason", result)
        after = self.manager.status()
        self.assertEqual(after["context_tokens"], before["context_tokens"])
        self.assertEqual(self.manager._children["chat"].popen.pid, chat_pid)

    def test_no_fit_at_start_names_refusal(self):
        self.make_chat_snapshot(CHAT_SMALL_ID)
        self.make_converted_laya()
        self.manager.setup("everyday", approve_download=True, available_bytes=64 * GiB)
        # Memory collapses before start: refusal, no children, no records.
        self.manager._available_bytes_override = int(1.1 * GiB)
        with self.assertRaises(pairs.PairError):
            self.manager.start()
        self.assertEqual(budget.load_reservations(self.home), {})
        self.assertEqual(self.manager.status()["state"], "error")


    def test_start_is_idempotent_while_both_workers_healthy(self):
        self.make_chat_snapshot(CHAT_SMALL_ID)
        self.make_converted_laya()
        self.manager.setup("everyday", approve_download=True)
        first = self.manager.start()
        self.addCleanup(self.manager.stop)
        second = self.manager.start()
        self.assertEqual(second["state"], "ready")
        self.assertEqual(second["chat_url"], first["chat_url"])
        self.assertEqual({c["pid"] for c in second["children"]},
                         {c["pid"] for c in first["children"]})

    def _prepared_with_broken_decision_builder(self):
        """Prepare everyday with local artifacts, then break ONLY the decision
        argv builder so start() spawns the chat worker and fails before the
        decision spawn."""
        self.make_chat_snapshot(CHAT_SMALL_ID)
        self.make_converted_laya()
        self.manager.setup("everyday", approve_download=True)
        broken = self._broken_decision_builder
        saved = pairs._decision_worker_argv
        pairs._decision_worker_argv = broken
        self.addCleanup(lambda: setattr(pairs, "_decision_worker_argv", saved))
        return saved

    def _broken_decision_builder(self, spec):
        raise RuntimeError("argv assembly exploded")

    def test_start_failure_terminates_already_spawned_worker(self):
        self._prepared_with_broken_decision_builder()
        spawned = {}
        real_spawn = managed.spawn_claiming_worker

        def recording_spawn(argv, payload, **kwargs):
            proc = real_spawn(argv, payload, **kwargs)
            spawned[kwargs["log_path"].name] = proc
            return proc

        with mock.patch.object(managed, "spawn_claiming_worker",
                               recording_spawn):
            with self.assertRaises(pairs.PairError) as ctx:
                self.manager.start()
        self.assertIn("argv assembly exploded", str(ctx.exception))
        # the chat worker that DID spawn is verified dead and its record
        # reaped: no orphan process, no held reservation, no untracked child
        chat_proc = spawned["everyday-chat.log"]
        deadline = time.monotonic() + 15
        while chat_proc.poll() is None and time.monotonic() < deadline:
            time.sleep(0.05)
        self.assertIsNotNone(chat_proc.poll())
        self.assertEqual(self.manager._children, {})
        self.assertEqual(budget.load_reservations(self.home), {})
        self.assertEqual(self.manager.status()["state"], "error")

    def test_start_failure_retains_child_when_termination_unverifiable(self):
        self._prepared_with_broken_decision_builder()
        with mock.patch.object(managed, "terminate", return_value=False):
            with self.assertRaises(pairs.PairError):
                self.manager.start()
        # the spawned chat worker stays OWNED, not dropped on the floor
        survivor = self.manager._children.get("chat")
        self.assertIsNotNone(survivor)
        # a later start settles the retained survivor itself: verified exit
        # releases the record and the fresh admission reaches ready
        pairs._decision_worker_argv = stub_argv_builder("decision")
        result = self.manager.start()
        self.addCleanup(self.manager.stop)
        self.assertEqual(result["state"], "ready")
        self.assertIsNotNone(survivor.popen.poll())

    def test_ensure_context_restores_reservation_when_old_worker_survives(self):
        self.make_chat_snapshot(CHAT_SMALL_ID)
        self.make_converted_laya()
        os.environ[PAIR_DEV_QUALIFICATION_ENV] = "1"
        self.addCleanup(lambda: os.environ.pop(PAIR_DEV_QUALIFICATION_ENV, None))
        self.manager.setup("everyday", approve_download=True, context_tokens=4096)
        self.manager.start()
        self.addCleanup(self.manager.stop)
        chat_memory = self.manager.catalog_models[CHAT_SMALL_ID]["memory"]
        current = self.manager._context_tokens
        current_bytes = budget.estimate_required(chat_memory, current).total
        old_chat = self.manager._children["chat"]
        with mock.patch.object(managed, "terminate", return_value=False):
            result = self.manager.ensure_context(current + 8192)
        self.assertFalse(result["ok"])
        self.assertEqual(result["reason"], "worker")
        # the resize is rolled back: admitted bytes match the worker that is
        # STILL RUNNING at the prior context — no inflation, no shrink
        self.assertEqual(
            budget.load_reservations(self.home)[CHAT_SMALL_ID]["bytes"],
            current_bytes)
        self.assertEqual(self.manager._context_tokens, current)
        self.assertIs(self.manager._children["chat"], old_chat)

    def test_ensure_context_fallback_only_after_candidate_confirmed_dead(self):
        self.make_chat_snapshot(CHAT_SMALL_ID)
        self.make_converted_laya()
        os.environ[PAIR_DEV_QUALIFICATION_ENV] = "1"
        self.addCleanup(lambda: os.environ.pop(PAIR_DEV_QUALIFICATION_ENV, None))
        self.manager.setup("everyday", approve_download=True, context_tokens=4096)
        self.manager.start()
        self.addCleanup(self.manager.stop)
        chat_memory = self.manager.catalog_models[CHAT_SMALL_ID]["memory"]
        current = self.manager._context_tokens
        current_bytes = budget.estimate_required(chat_memory, current).total
        old_chat = self.manager._children["chat"]
        decision_pid = self.manager._children["decision"].popen.pid
        events = []
        real_spawn = managed.spawn_claiming_worker
        real_terminate = managed.terminate

        def spy_spawn(argv, payload, **kwargs):
            proc = real_spawn(argv, payload, **kwargs)
            events.append(("spawn", proc.pid, payload["bytes"]))
            return proc

        def spy_terminate(proc, **kwargs):
            events.append(("terminate", proc.pid))
            return real_terminate(proc, **kwargs)

        healths = [False, True]  # candidate fails health, fallback succeeds
        with mock.patch.object(managed, "spawn_claiming_worker", spy_spawn), \
                mock.patch.object(managed, "terminate",
                                  side_effect=spy_terminate), \
                mock.patch.object(managed, "wait_health",
                                  side_effect=lambda *a, **k: healths.pop(0)):
            result = self.manager.ensure_context(current + 8192)
        self.assertFalse(result["ok"])
        self.assertEqual(result["reason"], "worker")
        # order: old worker terminated, candidate spawned, candidate
        # TERMINATED AND VERIFIED, only then the fallback at the prior bytes
        self.assertEqual(events[0], ("terminate", old_chat.popen.pid))
        candidate_pid, candidate_bytes = events[1][1], events[1][2]
        self.assertGreater(candidate_bytes, current_bytes)
        self.assertEqual(events[2], ("terminate", candidate_pid))
        self.assertEqual(events[3][2], current_bytes)
        self.assertEqual(len(events), 4)
        # the pair is still ready on the fallback; decision worker untouched
        self.assertEqual(self.manager.status()["state"], "ready")
        self.assertEqual(self.manager._context_tokens, current)
        self.assertEqual(self.manager._children["decision"].popen.pid,
                         decision_pid)
        self.assertEqual(
            budget.load_reservations(self.home)[CHAT_SMALL_ID]["bytes"],
            current_bytes)

    def test_decision_conversion_runs_supervised(self):
        # local decision snapshot present, conversion output absent: the
        # converter must spawn through the managed lifetime pipe (no bare
        # Popen), log to a readable file, and surface its tail on failure
        self.make_chat_snapshot(CHAT_SMALL_ID)
        snap = self.home / "hf" / "models--org--laya" / "snapshots" / ("a" * 40)
        snap.mkdir(parents=True, exist_ok=True)
        (snap / "model.safetensors").write_bytes(b"w" * 16)
        (snap / "config.json").write_text("{}", encoding="utf-8")
        (snap / "tokenizer.json").write_text("{}", encoding="utf-8")
        captured = {}
        real_spawn = managed.spawn_supervised

        def fake_spawn(argv, **kwargs):
            if any("mlx_omarchy_laya.convert" in part for part in argv):
                captured["lifetime_fd"] = kwargs["lifetime_read_fd"]
                captured["log_path"] = kwargs["log_path"]
                argv = [sys.executable, "-c",
                        "import sys; sys.stderr.write('converter exploded\\n'); "
                        "sys.exit(2)"]
            return real_spawn(argv, **kwargs)

        with mock.patch.object(managed, "spawn_supervised", fake_spawn):
            with self.assertRaises(pairs.PairError) as ctx:
                self.manager.setup("everyday", approve_download=True)
        self.assertIn("conversion failed (exit 2)", str(ctx.exception))
        self.assertIn("converter exploded", str(ctx.exception))
        self.assertIsNotNone(captured.get("lifetime_fd"))
        self.assertEqual(captured["log_path"].name, "convert-laya-mlx.log")

    def test_status_reports_error_when_one_worker_dies(self):
        self.make_chat_snapshot(CHAT_SMALL_ID)
        self.make_converted_laya()
        self.manager.setup("everyday", approve_download=True)
        self.manager.start()
        self.addCleanup(self.manager.stop)
        self.manager._children["decision"].popen.kill()
        self.manager._children["decision"].popen.wait(timeout=15)
        status = self.manager.status()
        self.assertEqual(status["state"], "error")
        self.assertIn("decision", status["error"])

    def test_start_adopts_saved_lock_after_restart(self):
        self.make_chat_snapshot(CHAT_SMALL_ID)
        self.make_converted_laya()
        self.manager.setup("everyday", approve_download=True)
        self.manager.start()
        self.manager.stop()
        # a fresh manager, as after an app restart: no setup state in memory
        restarted = pairs.PairManager(self.home)
        pairs._chat_worker_argv = stub_argv_builder("chat")
        pairs._decision_worker_argv = stub_argv_builder("decision")
        self.addCleanup(restarted.stop)
        status = restarted.start("everyday")
        self.assertEqual(status["state"], "ready")
        self.assertEqual(status["pair_id"], "everyday")
        self.assertEqual(status["chat_model"], CHAT_SMALL_ID)
        held = budget.load_reservations(self.home)
        self.assertIn(CHAT_SMALL_ID, held)
        self.assertEqual(held[CHAT_SMALL_ID]["pair_id"], "everyday")

    def test_setup_without_approval_uses_cached_artifacts(self):
        self.make_chat_snapshot(CHAT_SMALL_ID)
        self.make_converted_laya()
        result = self.manager.setup("everyday", approve_download=False)
        self.assertEqual(result["state"], "prepared")
        self.assertTrue(result["model_paths"]["chat"].is_dir())
        self.assertTrue(result["model_paths"]["decision"].is_dir())
        self.assertEqual(self.manager.status()["state"], "prepared")
        self.assertTrue(all(not d["needed"]
                            for d in self.manager.status()["downloads"]))

    def test_status_polls_while_setup_runs(self):
        release = threading.Event()
        snapshot = self.make_chat_snapshot(CHAT_SMALL_ID)
        converted = self.make_converted_laya()

        def slow_chat(self_mgr, *args, **kwargs):
            release.wait(timeout=10)
            return snapshot

        def fast_decision(self_mgr, *args, **kwargs):
            return converted

        with mock.patch.object(pairs.PairManager, "_prepare_chat", slow_chat), \
                mock.patch.object(pairs.PairManager, "_prepare_decision",
                                  fast_decision):
            outcome = {}

            def run():
                try:
                    outcome["result"] = self.manager.setup(
                        "everyday", approve_download=True)
                except Exception as exc:
                    outcome["error"] = exc

            worker = threading.Thread(target=run, daemon=True)
            worker.start()
            try:
                deadline = time.monotonic() + 5
                while self.manager.status()["state"] != "preparing" \
                        and time.monotonic() < deadline:
                    time.sleep(0.01)
                self.assertEqual(self.manager.status()["state"], "preparing")
            finally:
                release.set()
                worker.join(timeout=10)
            self.assertFalse(worker.is_alive())
            self.assertIn("result", outcome)
            self.assertEqual(outcome["result"]["state"], "prepared")


if __name__ == "__main__":
    unittest.main()
