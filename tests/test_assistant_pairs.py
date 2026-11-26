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

import contextlib
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import types
import unittest
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parent.parent
SERVE_ROOT = str(REPO_ROOT / "serve")
sys.path.insert(0, SERVE_ROOT)

from mlx_omarchy_serve import budget, catalog  # noqa: E402
from mlx_omarchy_assistant import managed, pairs  # noqa: E402
from mlx_omarchy_assistant.pairs import PAIR_DEV_QUALIFICATION_ENV

GiB = 1024**3

CHAT_SMALL_ID = "qwen3-4b-instruct-2507-4bit"
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
        "context": {"max_tokens": max_tokens, "trained_tokens": None},  # unknown, said out loud
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
        "first_text_budget_ms": 6500,
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

STUB_WORKER_MEM_AVAILABLE_ENV = "MLX_OMARCHY_TEST_STUB_MEM_AVAILABLE_BYTES"
PAIR_MANAGER_MEM_AVAILABLE_BYTES = 64 * GiB

STUB_WORKER = """
import json, os, signal, sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
sys.path.insert(0, {serve!r})
from mlx_omarchy_serve import budget

_stub_mem = os.environ.get({mem_env!r})
if _stub_mem:
    _stub_mem_bytes = int(_stub_mem)
    budget.mem_available = lambda: _stub_mem_bytes

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
""".format(serve=SERVE_ROOT, mem_env=STUB_WORKER_MEM_AVAILABLE_ENV)


def stub_argv_builder(role):
    def build(spec):
        return [sys.executable, "-c", STUB_WORKER, role, str(spec.port),
                str(spec.context_tokens)]
    return build


_real_bind = __import__("socket").socket.bind


def _bind_repeating(port):
    """A bind that hands out `port` for the first probe of port 0, the way the
    kernel may after a probe socket closes; later probes bind normally."""
    used = []

    def bind(self, address):
        if address[1] == 0 and not used:
            used.append(port)
            return _real_bind(self, (address[0], port))
        return _real_bind(self, address)
    return bind


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

def chat_total(memory, tokens):
    """What a pair admits for its chat worker: the worker is launched with the prefill chunk
    serve_cli.server_argv pins (--prefill-step-size), so the estimate carries that chunk's scratch."""
    step = budget.effective_prefill_step(budget.DEFAULT_PREFILL_STEP_TOKENS, tokens)
    return budget.estimate_required(memory, tokens, step).total


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


    def test_a_backend_context_limit_names_a_receipt_that_exists(self):
        # "reported, never inferred": a limit with no run behind it is refused
        for entry in self.bundled["models"]:
            extension = entry.get("extension") or {}
            if extension.get("backend_context_qualified_tokens") is None:
                continue
            with self.subTest(model=entry["id"]):
                receipt = extension.get("backend_context_receipt")
                self.assertTrue(receipt)
                self.assertTrue((Path(__file__).resolve().parents[1] / receipt).is_file())
            by_tokens = extension.get("backend_context_peak_bytes_by_tokens") or {}
            receipts_by_tokens = extension.get("backend_context_receipts_by_tokens") or {}
            for tok in by_tokens:
                with self.subTest(model=entry["id"], tokens=tok):
                    tier_receipt = receipts_by_tokens.get(tok)
                    self.assertTrue(tier_receipt, f"missing receipt for tier {tok}")
                    self.assertTrue((Path(__file__).resolve().parents[1] / tier_receipt).is_file())

    def test_tiered_backend_context_peaks_select_smallest_covering_tier(self):
        import copy
        chat = copy.deepcopy({m["id"]: m for m in self.bundled["models"]}["qwen3-4b-instruct-2507-4bit"])
        chat["extension"]["backend_context_peak_bytes_by_tokens"] = {
            "8192": 7 * GiB,
            "16384": 18123325440,
        }
        # Default context stays on the plain estimate (below 4096).
        self.assertLess(pairs.chat_estimate(chat, 4096).total, 6 * GiB)
        # 4097..8192 uses the 8192 measured peak; 8193..16384 keeps the 16384 peak.
        self.assertEqual(pairs.chat_estimate(chat, 8192).total, 7 * GiB)
        self.assertEqual(pairs.chat_estimate(chat, 8193).total, 18123325440)
        self.assertEqual(pairs.chat_estimate(chat, 16384).total, 18123325440)
        # A 16 GiB host (12 GiB available) admits 8192 but refuses 8193+.
        ctx_16 = pairs.admissible_context(chat, other_bytes=2 * GiB, preference="balanced",
                                          available_bytes=12 * GiB)
        self.assertEqual(ctx_16["context_tokens"], 8192)
        self.assertEqual(ctx_16["binding"], "memory")
        # A 24 GiB host still admits 16384.
        ctx_24 = pairs.admissible_context(chat, other_bytes=3 * GiB // 2, preference="balanced",
                                          available_bytes=24 * GiB)
        self.assertEqual(ctx_24["context_tokens"], 16384)
        # Malformed peak tiers fail closed.
        bad = copy.deepcopy(chat)
        bad["extension"]["backend_context_peak_bytes_by_tokens"] = {"notanint": 7 * GiB}
        with self.assertRaises(pairs.PairError):
            pairs.chat_estimate(bad, 8192)

    def test_compact_limit_is_not_admitted_where_the_measured_peak_does_not_fit(self):
        # The qualification run measured the chat worker's peak (VmHWM 17,698,560 kB = 16.9 GiB)
        # at 16384 tokens; the estimate alone says 7.7 GiB. A 16 GiB machine must stay at the
        # default, a 24 GiB one may take 16384.
        chat = {m["id"]: m for m in self.bundled["models"]}["qwen3-4b-instruct-2507-4bit"]
        decision = 3 * GiB // 2  # decision worker, assumed for the sizing
        small = pairs.admissible_context(chat, other_bytes=decision, preference="balanced",
                                         available_bytes=16 * GiB)
        self.assertLessEqual(small["context_tokens"], budget.DEFAULT_CONTEXT_TOKENS)
        self.assertEqual(small["binding"], "memory")
        large = pairs.admissible_context(chat, other_bytes=decision, preference="balanced",
                                         available_bytes=24 * GiB)
        self.assertEqual(large["context_tokens"], 16384)
        # the claim a worker makes at 16384 is the measured peak, not the 7.7 GiB estimate
        self.assertGreaterEqual(pairs.chat_estimate(chat, 16384).total, 17_698_560 * 1024)
        # and the default context keeps the plain estimate
        self.assertLess(pairs.chat_estimate(chat, 4096).total, 6 * GiB)

    def test_compact_chat_model_admits_an_omp_sized_prompt(self):
        # omp sends about 5.7k prompt tokens; the 4096 default refused it with HTTP 404
        chat = {m["id"]: m for m in self.bundled["models"]}["qwen3-4b-instruct-2507-4bit"]
        ctx = pairs.admissible_context(chat, other_bytes=0, preference="balanced",
                                       available_bytes=64 * GiB)
        self.assertGreaterEqual(ctx["context_tokens"], 16384)
        self.assertTrue(ctx["backend_context_qualified"])

    def test_compact_qwen35_4b_admits_8192_on_16gb_host(self):
        # Hybrid Qwen3.5-4B-MLX-4bit measured 6,729,600 kB (6.42 GiB) VmHWM at
        # 8192 on a 16 GB host, so 8192 fits alongside the 2.41 GiB decision
        # worker and 2.00 GiB desktop reserve (~10.83 GiB total).
        chat = {m["id"]: m for m in self.bundled["models"]}["qwen3.5-4b-mlx-4bit"]
        decision = 2589732096
        ctx_16 = pairs.admissible_context(chat, other_bytes=decision, preference="balanced",
                                          available_bytes=12 * GiB)
        self.assertEqual(ctx_16["context_tokens"], 8192)
        self.assertEqual(ctx_16["binding"], "backend")
        self.assertTrue(ctx_16["backend_context_qualified"])
        self.assertEqual(pairs.chat_estimate(chat, 8192).total, 6891110400)
        self.assertLess(pairs.chat_estimate(chat, 4096).total, 5 * GiB)
        tight = pairs.admissible_context(chat, other_bytes=decision, preference="balanced",
                                         available_bytes=9 * GiB)
        self.assertLessEqual(tight["context_tokens"], budget.DEFAULT_CONTEXT_TOKENS)
        self.assertEqual(tight["binding"], "memory")

    def test_bundled_chat_models_pin_revision_hashes_and_license(self):
        models = {entry["id"]: entry for entry in self.bundled["models"]}
        expected = {
            "qwen3.5-9b-mlx-4bit": ("938d8919941c6e7efd3c7150eff7fe9d12afa631",
                                     "mlx-community/Qwen3.5-9B-MLX-4bit", 5950221072),
            "qwen3-4b-instruct-2507-4bit": ("50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b",
                                             "mlx-community/Qwen3-4B-Instruct-2507-4bit", 2263022417),
            "qwen3.5-4b-mlx-4bit": ("32f3e8ecf65426fc3306969496342d504bfa13f3",
                                     "mlx-community/Qwen3.5-4B-MLX-4bit", 3034300695),
        }
        for model_id, (revision, repo, weights) in expected.items():
            with self.subTest(model=model_id):
                entry = models[model_id]
                self.assertEqual((entry["revision"], entry["repo"], entry["license"]),
                                 (revision, repo, "apache-2.0"))
                self.assertFalse(entry["recommended"])
                for group in ("generation", "http", "managed"):
                    qual = entry["qualification"][group]
                    self.assertEqual(qual["status"], "qualified")
                    self.assertTrue((Path(__file__).resolve().parents[1] / qual["receipt"]).is_file())
                self.assertEqual(entry["memory"]["weights_bytes"], weights)
                files = entry["extension"]["files"]
                self.assertTrue(files)
                self.assertEqual(sum(file["size_bytes"] for file in files
                                     if file["path"].endswith(".safetensors")), weights)
                for file in files:
                    self.assertEqual(len(file["sha256"]), 64, file["path"])
                    self.assertGreater(file["size_bytes"], 0, file["path"])
                self.assertNotIn("card_format", entry["extension"])

    def test_default_pair_selection_respects_memory_and_priority(self):
        large = pairs.select_default_pair(self.bundled["pairs"], self.bundled["models"],
                                          preference="balanced", available_bytes=96 * GiB)
        self.assertEqual(large["id"], "everyday")
        self.assertEqual(large["chat_model"], "qwen3.5-9b-mlx-4bit")
        # A 16 GiB-class host with 10 GiB available after OS/desktop reserve.
        compact = pairs.select_default_pair(self.bundled["pairs"], self.bundled["models"],
                                            preference="balanced", available_bytes=10 * GiB)
        self.assertEqual(compact["id"], "compact")
        self.assertEqual(compact["chat_model"], "qwen3.5-4b-mlx-4bit")
        with self.assertRaisesRegex(pairs.PairError, "no default pair fits"):
            pairs.select_default_pair(self.bundled["pairs"], self.bundled["models"],
                                      preference="balanced", available_bytes=2 * GiB)

    def test_unknown_pair_key_rejected(self):
        cat = fixture_catalog()
        cat["pairs"][0]["surprise"] = 1
        with self.assertRaises(catalog.CatalogError):
            catalog.validate_catalog(cat)

    def test_pair_first_text_budget_field(self):
        # Optional per-pair first-text budget in ms; only positive numbers.
        cat = fixture_catalog()
        cat["pairs"][0]["first_text_budget_ms"] = 2000
        catalog.validate_catalog(cat)
        for bad in (0, -1, "2000", True, None):
            broken = fixture_catalog()
            broken["pairs"][0]["first_text_budget_ms"] = bad
            with self.assertRaises(catalog.CatalogError):
                catalog.validate_catalog(broken)

    def test_bundled_quality_pair_carries_the_relaxed_budget(self):
        # Owner decision 2026-10-02: Quality's first-text budget is the
        # measured 6.5 s figure; every other pair keeps the 2.0 s default
        # by not carrying the field at all (one definition per value).
        quality = next(p for p in self.bundled["pairs"] if p["id"] == "quality")
        self.assertEqual(quality["first_text_budget_ms"], 6500)
        for pair in self.bundled["pairs"]:
            if pair["id"] != "quality":
                self.assertNotIn("first_text_budget_ms", pair, pair["id"])

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
        # chip_arch=None falls back to detection; pin detection to None
        # (undetected) so an Apple Silicon host's own chip id does not answer it.
        with mock.patch.object(pairs, "local_chip_arch", return_value=None), \
                self.assertRaises(pairs.PairError) as ctx:
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

    def test_first_text_budget_pair_field_replaces_default(self):
        # The per-pair catalog budget replaces the 2000 ms default for that
        # pair only (owner decision 2026-10-02: Quality ships 6500 ms, the
        # measured figure). Each pair is judged against its own bound.
        cat = selection_catalog()
        quality = next(p for p in cat["pairs"] if p["id"] == "quality")
        quality["first_text_budget_ms"] = 6500
        evidence = quality["extension"]["selection_evidence"]
        evidence["quality"]["score"] = 0.99  # best measured quality
        evidence["latency"]["first_visible_p95_ms"] = 2600.0
        # 2600 ms exceeds the old flat bound but sits inside Quality's own,
        # so the best-quality pair stays selectable.
        self.assertEqual(self._pick(cat)["id"], "quality")
        # Past ITS OWN 6500 ms bound the same evidence refuses, and the pair
        # falls back to the next-best valid candidate.
        evidence["latency"]["first_visible_p95_ms"] = 6600.0
        self.assertEqual(self._pick(cat)["id"], "everyday")
        # With every pair over its own bound, the joined refusal names each
        # pair's budget (6500 for Quality, 2000 for the rest).
        for pair in cat["pairs"]:
            bound = 6500 if pair["id"] == "quality" else 2000
            pair["extension"]["selection_evidence"]["latency"][
                "first_visible_p95_ms"] = bound + 100
        with self.assertRaises(pairs.PairError) as ctx:
            self._pick(cat)
        message = str(ctx.exception)
        self.assertIn("first-visible p95", message)
        self.assertIn("6500", message)
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

    def test_unclaimed_record_of_a_dead_creator_is_reaped_after_reboot(self):
        budget.admit_and_reserve_batch(
            [("chat", 1024, ""), ("decision", 1024, "")],
            pair_id="everyday", home=self.home, available_bytes=10 * GiB)
        path = budget.reservations_path(self.home)
        stale = json.loads(path.read_text())
        for record in stale.values():
            record["parent"]["starttime"] += 1
        path.write_text(json.dumps(stale))
        cleared = managed.reap_pair_records("everyday", self.home, children=[])
        self.assertEqual(sorted(cleared), ["chat", "decision"])
        self.assertEqual(budget.load_reservations(self.home), {})


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
        chat_port, decision_port = managed.free_ports(2)
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

    def test_ignores_environment_proxy(self):
        # A fresh interpreter: urllib's global opener keeps the proxies it
        # saw on first use, so an in-process run would depend on test order.
        class Health(BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.send_header("Content-Length", "0")
                self.end_headers()

            def log_message(self, *args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Health)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        # Bound but never listening: the proxy refuses every connection.
        dead_proxy = socket.socket()
        self.addCleanup(dead_proxy.close)
        dead_proxy.bind(("127.0.0.1", 0))
        env = {k: v for k, v in os.environ.items()
               if k.lower() not in ("http_proxy", "no_proxy")}
        env.update(http_proxy=f"http://127.0.0.1:{dead_proxy.getsockname()[1]}",
                   PYTHONPATH=SERVE_ROOT)
        probe = ("import sys; from mlx_omarchy_assistant import managed; "
                 "print(managed.wait_health(sys.argv[1], deadline=5))")
        out = subprocess.run(
            [sys.executable, "-c", probe,
             f"http://127.0.0.1:{server.server_address[1]}/health"],
            env=env, capture_output=True, text=True, timeout=30)
        self.assertEqual((out.returncode, out.stdout.strip()), (0, "True"),
                         out.stderr)


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
        est = chat_total(self.chat["memory"], mem_bound)
        reserve = max(budget.SAFETY_RESERVE_BYTES, int(0.10 * 7 * GiB))
        self.assertLessEqual(est + self.other_bytes + reserve, 7 * GiB)
        one_more = chat_total(self.chat["memory"], mem_bound + 1)
        self.assertGreater(one_more + self.other_bytes + reserve, 7 * GiB)
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
        # Pin host memory in both parent and spawned stub workers so
        # setup/start/ensure_context/status tests do not depend on the
        # machine's live /proc/meminfo; explicit available_bytes arguments and
        # _available_bytes_override still take precedence.
        mem_patcher = mock.patch.object(
            budget, "mem_available", return_value=PAIR_MANAGER_MEM_AVAILABLE_BYTES)
        mem_patcher.start()
        self.addCleanup(mem_patcher.stop)
        env_patcher = mock.patch.dict(
            os.environ,
            {STUB_WORKER_MEM_AVAILABLE_ENV: str(PAIR_MANAGER_MEM_AVAILABLE_BYTES)},
        )
        env_patcher.start()
        self.addCleanup(env_patcher.stop)
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
        # The per-pair first-text budget rides to the UI: Quality carries the
        # shipped 6500 ms override, pairs on the 2 s default carry none.
        quality = next(p for p in status["pairs"] if p["id"] == "quality")
        self.assertIsNone(everyday["first_text_budget_ms"])
        self.assertEqual(quality["first_text_budget_ms"], 6500)

    def test_ready_offline_requires_a_named_receipt(self):
        pair = {"id": "everyday",
                "qualification": {"status": "qualified", "receipt": None},
                "chat_model": CHAT_SMALL_ID, "decision_model": "laya-mlx"}
        self.assertFalse(self.manager._ready_offline(pair))
        pair["qualification"]["receipt"] = "   "
        self.assertFalse(self.manager._ready_offline(pair))

    def test_start_gives_the_two_workers_distinct_ports_when_the_kernel_repeats_one(self):
        # bind(0) after close can hand back the port it just gave out (about
        # 1 in 25,000 pairs on Linux). Two separate probes then gave both
        # workers one port: the second bind failed, "chat exited early (rc=1)".
        import socket
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            repeated = probe.getsockname()[1]
        self.make_chat_snapshot(CHAT_SMALL_ID)
        self.make_converted_laya()
        self.manager.setup("everyday", approve_download=True)
        with mock.patch.object(managed, "free_port", return_value=repeated), \
                mock.patch.object(managed.socket.socket, "bind", _bind_repeating(repeated)):
            result = self.manager.start()
        self.addCleanup(self.manager.stop)
        self.assertEqual(result["state"], "ready")
        status = self.manager.status()
        chat = int(status["chat_url"].split(":")[2].split("/")[0])
        decision = int(status["decision_url"].split(":")[2].split("/")[0])
        self.assertNotEqual(chat, decision)

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
                                   preference="balanced", available_bytes=10 * GiB)  # 8 GiB no longer holds 4096 once prefill scratch counts
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

    def make_raw_laya_snapshot(self, with_weights=True):
        # Laya's source layout has no top-level config.json: encoder/config.json,
        # tokenizer/, rl_agent_config.json, model.safetensors.
        snap = self.home / "hf" / "models--org--laya" / "snapshots" / ("a" * 40)
        (snap / "encoder").mkdir(parents=True)
        (snap / "tokenizer").mkdir()
        (snap / "encoder" / "config.json").write_text("{}", encoding="utf-8")
        (snap / "rl_agent_config.json").write_text("{}", encoding="utf-8")
        (snap / "tokenizer" / "tokenizer.json").write_text("{}", encoding="utf-8")
        if with_weights:
            (snap / "model.safetensors").write_bytes(b"w" * 16)
        return snap

    def test_setup_offline_converts_a_fully_cached_raw_laya_snapshot(self):
        # v0.7.5 refused this offline ("no raw snapshot to convert") because the
        # generic completeness gate demanded a top-level config.json that Laya's
        # layout does not have. A complete cached snapshot must reach conversion.
        self.make_chat_snapshot(CHAT_SMALL_ID)
        snap = self.make_raw_laya_snapshot()
        launched = []

        def fake_spawn(argv, **_kwargs):
            launched.append(argv)
            return types.SimpleNamespace(poll=lambda: 0, returncode=0)

        with mock.patch.object(pairs.managed, "spawn_supervised", fake_spawn):
            with self.assertRaises(pairs.PairError) as ctx:
                self.manager.setup("everyday", approve_download=True)
        self.assertNotIn("no raw snapshot", str(ctx.exception))
        self.assertEqual(len(launched), 1)
        argv = launched[0]
        self.assertIn("mlx_omarchy_laya.convert", argv)
        self.assertEqual(argv[argv.index("--from-local") + 1], str(snap))

    def test_setup_offline_names_the_missing_weights_in_a_partial_laya_snapshot(self):
        self.make_chat_snapshot(CHAT_SMALL_ID)
        self.make_raw_laya_snapshot(with_weights=False)
        with self.assertRaises(pairs.PairError) as ctx:
            self.manager.setup("everyday", approve_download=True)
        message = str(ctx.exception)
        self.assertIn(DECISION_ID, message)
        self.assertIn("model.safetensors", message)
        self.assertIn("org/laya", message)

    def test_setup_offline_says_when_the_cache_has_no_snapshot_for_the_pin(self):
        self.make_chat_snapshot(CHAT_SMALL_ID)
        with self.assertRaises(pairs.PairError) as ctx:
            self.manager.setup("everyday", approve_download=True)
        self.assertIn("no snapshot for revision aaaaaaaaaaaa", str(ctx.exception))

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
        # Keep lifetime-pipe EOF from killing workers while exit is deliberately unverifiable.
        with mock.patch.object(self.manager, "_close_lifetime"):
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
            chat_total(self.manager.catalog_models[CHAT_SMALL_ID]["memory"],
                       before["context_tokens"]))
        self.assertEqual(self.manager._children["decision"].popen.pid, decision_pid)
        self.assertNotEqual(self.manager._children["chat"].popen.pid, old_chat_pid)

    def test_ensure_context_uses_resident_decision_contribution(self):
        self.make_chat_snapshot(CHAT_SMALL_ID)
        self.make_converted_laya()
        os.environ[PAIR_DEV_QUALIFICATION_ENV] = "1"
        self.addCleanup(lambda: os.environ.pop(PAIR_DEV_QUALIFICATION_ENV, None))
        self.manager.setup("everyday", approve_download=True, context_tokens=4096)
        self.manager.start()
        self.addCleanup(self.manager.stop)
        decision_id = self.manager._prepared["decision_id"]
        decision_rec = budget.load_reservations(self.home)[decision_id]
        decision_full = decision_rec["bytes"]
        decision_floor = (decision_full * 3) // 4
        budget.set_reservation_state(
            decision_id, "resident", owner=decision_rec["owner"],
            resident_floor_bytes=decision_floor, home=self.home)
        target_tokens = 8192
        chat = self.manager.catalog_models[CHAT_SMALL_ID]
        chat_need = pairs.chat_estimate(chat, target_tokens, prompt_cache_size=0,
                                        ignore_peak_override=True).total
        # Post-start MemAvailable covers the remaining unmaterialized decision
        # headroom + chat_need + desktop reserve, but would falsely refuse if
        # ensure_context double-counted the full decision reservation.
        decision_remaining = decision_full - decision_floor
        # Solve tight_available = chat_need + decision_remaining + reserve + margin
        # with margin < decision_floor (so full decision_full would not fit).
        margin = max(1, decision_floor // 4)
        base_need = chat_need + decision_remaining + margin
        tight_available = max(base_need + budget.SAFETY_RESERVE_BYTES,
                              int(base_need / (1.0 - pairs.DESKTOP_RESERVE_FRACTION)) + 1024 * 1024)
        self.assertLess(tight_available,
                        chat_need + decision_full + pairs.desktop_reserve(tight_available))
        self.manager._available_bytes_override = tight_available
        result = self.manager.ensure_context(target_tokens)
        self.assertTrue(result["ok"], result)
        self.assertEqual(self.manager.status()["context_tokens"], target_tokens)

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
        chat_entry = self.manager.catalog_models[CHAT_SMALL_ID]
        current = self.manager._context_tokens
        current_bytes = pairs.chat_estimate(
            chat_entry, current,
            prompt_cache_size=self.manager._chat_prompt_cache_size).total
        self.assertEqual(
            budget.load_reservations(self.home)[CHAT_SMALL_ID]["bytes"],
            current_bytes)
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
        chat_entry = self.manager.catalog_models[CHAT_SMALL_ID]
        current = self.manager._context_tokens
        current_bytes = pairs.chat_estimate(
            chat_entry, current,
            prompt_cache_size=self.manager._chat_prompt_cache_size).total
        self.assertEqual(
            budget.load_reservations(self.home)[CHAT_SMALL_ID]["bytes"],
            current_bytes)
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

    def test_start_adopts_newest_lock_past_entries_that_are_not_lock_files(self):
        self.make_chat_snapshot(CHAT_SMALL_ID)
        self.make_converted_laya()
        self.manager.setup("everyday", approve_download=True)
        self.manager.start()
        self.manager.stop()
        locks = pairs.pair_lock_path(self.home, "unused").parent
        newer = time.time() + 60
        real_scandir = os.scandir

        def scandir_then_unlink(path):
            with real_scandir(path) as entries:
                listed = list(entries)
            (locks / "vanished.json").unlink()
            return contextlib.nullcontext(iter(listed))

        for name, make, scandir in (
                ("directory.json", Path.mkdir, real_scandir),
                ("fifo.json", os.mkfifo, real_scandir),
                ("vanished.json", Path.touch, scandir_then_unlink),
                ("dangling.json", lambda path: path.symlink_to(locks / "gone"), real_scandir),
                ("looping.json", lambda path: path.symlink_to(path), real_scandir),
                # the temp file a killed write_pair_lock() leaves behind
                ("everyday.json1a2b.tmp", Path.touch, real_scandir)):
            with self.subTest(name):
                path = locks / name
                make(path)
                try:
                    if not path.is_symlink():
                        os.utime(path, (newer, newer))
                    restarted = pairs.PairManager(self.home)
                    self.addCleanup(restarted.stop)
                    with mock.patch("os.scandir", scandir):
                        self.assertEqual(restarted._newest_lock_pair_id(), "everyday")
                    # only once the pick is right: a picked FIFO hangs start()
                    status = restarted.start()
                    self.assertEqual((status["state"], status["pair_id"]),
                                     ("ready", "everyday"))
                    restarted.stop()
                finally:
                    if path.is_dir() and not path.is_symlink():
                        path.rmdir()
                    else:
                        path.unlink(missing_ok=True)

    def test_newest_lock_orders_by_nanosecond_mtime_then_name(self):
        lock = {"version": 1, "models": {}}
        for pair_id in ("everyday", "quality"):
            pairs.write_pair_lock(self.home, pair_id, dict(lock, pair_id=pair_id))
        stamp = time.time_ns()
        os.utime(pairs.pair_lock_path(self.home, "quality"), ns=(stamp, stamp))
        os.utime(pairs.pair_lock_path(self.home, "everyday"), ns=(stamp + 1, stamp + 1))
        self.assertEqual(self.manager._newest_lock_pair_id(), "everyday")
        os.utime(pairs.pair_lock_path(self.home, "everyday"), ns=(stamp, stamp))
        self.assertEqual(self.manager._newest_lock_pair_id(), "quality")  # tie: last name

    @unittest.skipIf(os.geteuid() == 0, "root ignores directory permissions")
    def test_unreadable_lock_directory_is_pair_error_not_absent(self):
        # server.resume() reports a "no saved pair lock" message as absent
        pairs.write_pair_lock(self.home, "everyday",
                              {"version": 1, "pair_id": "everyday", "models": {}})
        locks = pairs.pair_lock_path(self.home, "unused").parent
        for path, mode in ((locks, 0o000), (locks, 0o300), (locks, 0o400),
                           (locks.parent, 0o000)):
            with self.subTest(path=path.name, mode=oct(mode)):
                path.chmod(mode)
                try:
                    with self.assertRaisesRegex(
                            pairs.PairError, "pair lock directory .* is unreadable"):
                        self.manager.adopt_saved()
                    with self.assertRaisesRegex(pairs.PairError, "no prepared pair"):
                        self.manager.start()
                finally:
                    path.chmod(0o755)
        real_scandir = os.scandir

        def scandir_then_lock(path):
            with real_scandir(path) as entries:
                listed = list(entries)
            locks.chmod(0o400)
            return contextlib.nullcontext(iter(listed))

        with self.subTest("locked during the scan"):
            try:
                with mock.patch("os.scandir", scandir_then_lock), self.assertRaisesRegex(
                        pairs.PairError, "pair lock directory .* is unreadable"):
                    self.manager.adopt_saved()
            finally:
                locks.chmod(0o755)

    def test_missing_or_file_lock_directory_is_no_saved_lock(self):
        locks = pairs.pair_lock_path(self.home, "unused").parent
        with self.assertRaisesRegex(pairs.PairError, "no saved pair lock to adopt"):
            self.manager.adopt_saved()
        locks.parent.mkdir(parents=True)
        locks.write_text("")
        with self.assertRaisesRegex(pairs.PairError, "no saved pair lock to adopt"):
            self.manager.adopt_saved()
        self.manager.home = self.home / "nul\0home"  # base's is_dir() was False
        with self.assertRaisesRegex(pairs.PairError, "no saved pair lock to adopt"):
            self.manager.adopt_saved()

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

    def test_fixture_isolates_pair_manager_from_host_meminfo(self):
        child_trap = (
            f"import sys; sys.path.insert(0, {SERVE_ROOT!r})\n"
            "from mlx_omarchy_serve import budget\n"
            "def _fail_meminfo(*_a, **_kw):\n"
            "    raise AssertionError('child read host /proc/meminfo')\n"
            "budget.parse_meminfo = _fail_meminfo\n"
        )

        def trap_child_meminfo(builder):
            def build(spec):
                argv = list(builder(spec))
                argv[2] = child_trap + argv[2]
                return argv
            return build

        with mock.patch.object(
                budget, "parse_meminfo",
                side_effect=AssertionError("read host /proc/meminfo")), \
                mock.patch.object(
                    pairs, "_chat_worker_argv",
                    trap_child_meminfo(pairs._chat_worker_argv)), \
                mock.patch.object(
                    pairs, "_decision_worker_argv",
                    trap_child_meminfo(pairs._decision_worker_argv)):
            plan = self.manager.setup("everyday", approve_download=False)
            status = pairs.PairManager(self.home).status()
            self.make_chat_snapshot(CHAT_SMALL_ID)
            self.make_converted_laya()
            self.manager.setup("everyday", approve_download=True)
            started = self.manager.start()
            self.addCleanup(self.manager.stop)
            held = budget.load_reservations(self.home)
            stopped = self.manager.stop()
        self.assertEqual(plan["state"], "planned")
        self.assertEqual(status["recommendation"]["pair_id"], "quality")
        self.assertEqual(started["state"], "ready")
        self.assertIsNotNone(held[CHAT_SMALL_ID]["claimed"])
        self.assertIsNotNone(held[DECISION_ID]["claimed"])
        self.assertTrue(stopped["stopped"])

    def test_chat_worker_enables_prompt_cache_size_1_when_budget_allows(self):
        self.make_chat_snapshot(CHAT_SMALL_ID)
        self.make_converted_laya()
        seen_specs = []
        base_builder = stub_argv_builder("chat")

        def recording_chat_argv(spec):
            seen_specs.append(spec)
            return base_builder(spec)

        with mock.patch.object(pairs, "_chat_worker_argv", recording_chat_argv):
            self.manager.setup("everyday", approve_download=True)
            status = self.manager.start()
            self.addCleanup(self.manager.stop)
        self.assertEqual(status["state"], "ready")
        self.assertEqual(status["chat_prompt_cache_size"], 1)
        self.assertEqual(len(seen_specs), 1)
        self.assertEqual(seen_specs[0].prompt_cache_size, 1)
        chat = self.manager.catalog_models[CHAT_SMALL_ID]
        expected_cached = pairs.chat_estimate(
            chat, status["context_tokens"], prompt_cache_size=1).total
        expected_uncached = pairs.chat_estimate(
            chat, status["context_tokens"], prompt_cache_size=0).total
        self.assertGreater(expected_cached, expected_uncached)
        held = budget.load_reservations(self.home)
        self.assertEqual(held[CHAT_SMALL_ID]["bytes"], expected_cached)
        # Real _chat_worker_argv forwards --prompt-cache-size 1
        real_argv = self._old_builders[0](seen_specs[0])
        idx = real_argv.index("--prompt-cache-size")
        self.assertEqual(real_argv[idx + 1], "1")

    def test_chat_worker_falls_back_to_prompt_cache_size_0_when_memory_is_tight(self):
        self.make_chat_snapshot(CHAT_SMALL_ID)
        self.make_converted_laya()
        chat = self.manager.catalog_models[CHAT_SMALL_ID]
        decision_bytes = pairs.decision_requirement_bytes(
            self.home / "models" / DECISION_ID, 8)
        uncached = pairs.chat_estimate(chat, 4096, prompt_cache_size=0).total
        cached = pairs.chat_estimate(chat, 4096, prompt_cache_size=1).total
        self.assertGreater(cached, uncached)
        # Pick an available_bytes where uncached fits with desktop_reserve
        # (10% or 2 GiB), while cached (+1 KV slot) does not.
        tight = uncached + decision_bytes + budget.SAFETY_RESERVE_BYTES + (cached - uncached) // 2
        self.assertEqual(
            pairs.chat_prompt_cache_size(chat, 4096, other_bytes=decision_bytes,
                                         available_bytes=tight),
            0,
        )
        self.manager._available_bytes_override = tight
        seen_specs = []
        base_builder = stub_argv_builder("chat")

        def recording_chat_argv(spec):
            seen_specs.append(spec)
            return base_builder(spec)

        with mock.patch.object(pairs, "_chat_worker_argv", recording_chat_argv):
            self.manager.setup("everyday", approve_download=True, context_tokens=4096,
                               available_bytes=tight)
            status = self.manager.start()
            self.addCleanup(self.manager.stop)
        self.assertEqual(status["state"], "ready")
        self.assertEqual(status["chat_prompt_cache_size"], 0)
        self.assertEqual(seen_specs[0].prompt_cache_size, 0)
        held = budget.load_reservations(self.home)
        self.assertEqual(held[CHAT_SMALL_ID]["bytes"], uncached)

    def test_chat_estimate_rejects_prompt_cache_when_kv_is_unknown(self):
        chat = dict(self.manager.catalog_models[CHAT_SMALL_ID])
        chat["memory"] = dict(chat["memory"], kv_bytes_per_token=None)
        self.assertEqual(
            pairs.chat_prompt_cache_size(chat, 4096, other_bytes=0,
                                         available_bytes=64 * GiB),
            0,
        )
        with self.assertRaises(pairs.PairError):
            pairs.chat_estimate(chat, 4096, prompt_cache_size=1)


def _on_pinned_stack(fn):
    """Run fn on a 4 MiB thread stack, so the JSON nesting limit does not
    follow the runner's ulimit -s (Python 3.14 bounds recursion by C stack)."""
    outcome = {}

    def target():
        try:
            outcome["value"] = fn()
        except BaseException as exc:  # re-raised in the caller
            outcome["error"] = exc

    previous = threading.stack_size(4 * 1024 * 1024)
    try:
        thread = threading.Thread(target=target)
        thread.start()
    finally:
        threading.stack_size(previous)
    thread.join()
    if "error" in outcome:
        raise outcome["error"]
    return outcome["value"]


DEEP_JSON = b"[" * 200_000 + b"]" * 200_000


class UndecodablePairFileTests(BasePairTest):
    """A pair lock or decision manifest that does not decode is the
    documented PairError (or {}), never a raw traceback."""

    PAYLOADS = {
        "invalid-utf8": b'{"version": 1, "pair_id": "everyday\xff"}',
        "huge-int": b'{"version": ' + b"1" * 5000 + b"}",
        "deep-nesting": DEEP_JSON,
    }

    def setUp(self):
        super().setUp()
        self.addCleanup(sys.set_int_max_str_digits, sys.get_int_max_str_digits())
        sys.set_int_max_str_digits(4300)
        with self.assertRaises(RecursionError):
            _on_pinned_stack(lambda: json.loads(DEEP_JSON))

    def _write_lock_bytes(self, data: bytes) -> None:
        path = pairs.pair_lock_path(self.home, "everyday")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    def test_undecodable_lock_is_pair_error(self):
        for name, data in self.PAYLOADS.items():
            with self.subTest(name):
                self._write_lock_bytes(data)
                with self.assertRaisesRegex(pairs.PairError, "is unreadable"):
                    _on_pinned_stack(lambda: pairs.load_pair_lock(self.home, "everyday"))

    def test_start_falls_back_past_undecodable_lock(self):
        manager = pairs.PairManager(self.home)
        for name, data in self.PAYLOADS.items():
            with self.subTest(name):
                self._write_lock_bytes(data)
                with self.assertRaisesRegex(pairs.PairError, "no prepared pair"):
                    _on_pinned_stack(lambda: manager.start("everyday"))

    def test_lock_model_needs_string_revision(self):
        def lock(models):
            return {"version": 1, "pair_id": "everyday", "models": models}

        pairs.write_pair_lock(self.home, "everyday",
                              lock({"chat": {"revision": "a" * 40}}))
        self.assertIsNotNone(pairs.load_pair_lock(self.home, "everyday"))
        for name, models in {"list-revision": {"chat": {"revision": [["a"]]}},
                             "no-revision": {"chat": {"id": "x"}},
                             "model-not-object": {"chat": "x"}}.items():
            with self.subTest(name):
                pairs.write_pair_lock(self.home, "everyday", lock(models))
                with self.assertRaisesRegex(pairs.PairError, "not a valid v1 lock"):
                    pairs.load_pair_lock(self.home, "everyday")

    def test_undecodable_manifest_reads_as_empty(self):
        model_dir = self.home / "models" / DECISION_ID
        model_dir.mkdir(parents=True)
        for name, data in self.PAYLOADS.items():
            with self.subTest(name):
                (model_dir / "manifest.json").write_bytes(data)
                self.assertEqual(_on_pinned_stack(lambda: pairs._read_manifest(model_dir)), {})


class DamagedDecisionArtifactTests(BasePairTest):
    """A converted decision artifact whose manifest or encoder config is
    damaged is a PairError from decision_requirement_bytes, never a raw
    traceback or a silently wrong byte count."""

    def setUp(self):
        super().setUp()
        self.addCleanup(sys.set_int_max_str_digits, sys.get_int_max_str_digits())
        sys.set_int_max_str_digits(4300)
        self.conv = self.make_converted_laya()
        self.manifest = json.loads((self.conv / "manifest.json").read_text())

    def _requirement(self):
        return _on_pinned_stack(lambda: pairs.decision_requirement_bytes(self.conv, 4))

    def _refused(self, pattern):
        with self.assertRaisesRegex(pairs.PairError, pattern):
            self._requirement()

    def test_non_object_manifest_is_no_manifest(self):
        for data in ("[1]", '"bad"', "123", "null"):
            with self.subTest(data):
                (self.conv / "manifest.json").write_text(data)
                self.assertEqual(pairs._read_manifest(self.conv), {})
                self._refused("manifest.json is missing, unreadable, empty or not a JSON object")

    def test_manifest_fields_size_like_the_worker_or_refuse(self):
        from mlx_omarchy_laya.server import total_estimate_bytes

        cfg = types.SimpleNamespace(hidden_size=64, num_attention_heads=2,
                                    intermediate_size=128)
        header = self.manifest["weights_header"]
        valid = {
            "as-converted": {},
            "weight-bytes-fallback": {"weights_header": {"weight_bytes": 16}},
            "default-context": {"context_max_tokens": None},
        }
        for name, change in valid.items():
            with self.subTest(name):
                manifest = dict(self.manifest, **change)
                (self.conv / "manifest.json").write_text(json.dumps(manifest))
                self.assertEqual(self._requirement(),
                                 total_estimate_bytes(manifest, cfg, "float16", 4))
        elements = r"weights_header\.param_elements must be an int >= 1"
        weight_bytes = r"weights_header\.weight_bytes must be an int >= 2"
        context = "context_max_tokens must be an int >= 1"
        cases = {
            "no-header": ("weights_header", None, "weights_header must be an object"),
            "list-header": ("weights_header", [1], "weights_header must be an object"),
            "string-elements": ("weights_header", {"param_elements": "8"}, elements),
            "bool-elements": ("weights_header", dict(header, param_elements=True), elements),
            "zero-elements": ("weights_header", dict(header, param_elements=0), elements),
            "no-element-count": ("weights_header", {"tensor_count": 2}, weight_bytes),
            "float-bytes": ("weights_header", {"weight_bytes": 16.5}, weight_bytes),
            "one-weight-byte": ("weights_header", {"weight_bytes": 1}, weight_bytes),
            "string-context": ("context_max_tokens", "512", context),
            "bool-context": ("context_max_tokens", True, context),
            "negative-context": ("context_max_tokens", -1, context),
            "list-context": ("context_max_tokens", [512], context),
        }
        for name, (key, value, pattern) in cases.items():
            with self.subTest(name):
                manifest = dict(self.manifest)
                if value is None:
                    del manifest[key]
                else:
                    manifest[key] = value
                (self.conv / "manifest.json").write_text(json.dumps(manifest))
                self._refused(pattern)

    def test_refusal_names_a_container_by_type(self):
        # a decoded container can be deeper than repr can recurse on 3.14
        deep = [1]
        for _ in range(1_000_000):
            deep = [deep]
        with self.assertRaisesRegex(pairs.PairError, "field must be an int >= 1, got list"):
            pairs._artifact_int(deep, "field")

    def test_estimate_overflow_is_pair_error(self):
        for name, (key, value) in {
                "float-workspace": ("context_max_tokens", 10 ** 160),
                "int-parameters": ("weights_header", {"param_elements": 10 ** 4299}),
        }.items():
            with self.subTest(name):
                manifest = dict(self.manifest, **{key: value})
                (self.conv / "manifest.json").write_text(json.dumps(manifest))
                self._refused("overflow the memory estimate")

    def test_damaged_encoder_config(self):
        path = self.conv / "encoder" / "config.json"
        good = json.loads(path.read_text())
        payloads = dict(UndecodablePairFileTests.PAYLOADS, **{
            "not-json": b"{",
            "non-object": b"[64, 2, 128]",
        })
        for name, data in payloads.items():
            with self.subTest(name):
                path.write_bytes(data)
                self._refused("encoder config .* is (unreadable|not a JSON object)")
        for name, value in {"missing": None, "bool": True, "zero": 0, "negative": -64,
                            "float": 64.0, "string": "64"}.items():
            with self.subTest(name):
                cfg = dict(good)
                if value is None:
                    del cfg["hidden_size"]
                else:
                    cfg["hidden_size"] = value
                path.write_text(json.dumps(cfg))
                self._refused("config.json hidden_size must be an int >= 1")
        path.unlink()
        self._refused("encoder config .* is unreadable")


if __name__ == "__main__":
    unittest.main()
