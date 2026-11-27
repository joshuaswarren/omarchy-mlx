"""Unit tests for the Laya typed-decision adapter — no model download needed.

Structural and calibration-semantics checks that always run: option
rendering, temperature bucketing, confidence, sequence-building invariants
against the upstream formulas, converter validation on a synthetic
checkpoint, and the exact CPU-refusal contract of the server. The full
numerical comparison against pinned upstream torch lives in
tests/test_laya_reference.py (env-gated).
"""

import contextlib
import http.client
import importlib.util
import json
import os
import signal
import struct
import subprocess
import sys
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from subprocess import TimeoutExpired  # noqa: F401  (used by server refusal test)

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "serve"))

from mlx_omarchy_laya.sequence import (  # noqa: E402
    QTYPES,
    LayaTokenizer,
    build_sequence,
    collate,
    confidence_from_probs,
    encode_questions,
    render_options,
    serialize_state,
    temp_bucket,
    to_internal,
)

CONVERTED = os.environ.get("LAYA_MLX_CKPT")  # converted checkpoint dir, if available
# The server and engine import mlx; hosts without it skip those tests.
HAS_MLX = importlib.util.find_spec("mlx") is not None

# The stubbed server tests patch signal.signal; keep the real one for deadlines.
_SIGNAL = signal.signal


class _Blocked(BaseException):
    """A read outlived its deadline. BaseException, so admission's error
    handling cannot turn a hang into a passing refusal."""


@contextlib.contextmanager
def _deadline(seconds=5):
    def fire(signum, frame):
        raise _Blocked("read blocked for %ss" % seconds)

    old = _SIGNAL(signal.SIGALRM, fire)
    started = time.monotonic()
    prev_delay, prev_interval = signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        _SIGNAL(signal.SIGALRM, old)
        if prev_delay:
            remaining = max(1e-3, prev_delay - (time.monotonic() - started))
            signal.setitimer(signal.ITIMER_REAL, remaining, prev_interval)


class CalibrationSemanticsTests(unittest.TestCase):
    def test_temp_bucket_matches_upstream_cardinality_edges(self):
        self.assertEqual(temp_bucket(QTYPES["choice"], 2), "choice:2")
        self.assertEqual(temp_bucket(QTYPES["choice"], 3), "choice:3-5")
        self.assertEqual(temp_bucket(QTYPES["choice"], 5), "choice:3-5")
        self.assertEqual(temp_bucket(QTYPES["choice"], 6), "choice:6-10")
        self.assertEqual(temp_bucket(QTYPES["choice"], 10), "choice:6-10")
        self.assertEqual(temp_bucket(QTYPES["choice"], 11), "choice:11+")
        self.assertEqual(temp_bucket(QTYPES["score"], 5), "score:3-5")
        self.assertEqual(temp_bucket(QTYPES["noul"], 2), "noul:2")

    def test_confidence_is_one_minus_normalized_entropy(self):
        p = np.array([0.7, 0.2, 0.1])
        ent = -(p * np.log(np.clip(p, 1e-12, 1))).sum()
        self.assertAlmostEqual(confidence_from_probs(p, 3), 1 - ent / np.log(3), places=12)
        self.assertEqual(confidence_from_probs(np.array([1.0]), 1), 1.0)

    def test_to_internal_matches_upstream_shape_rules(self):
        q = {"type": "choice", "instructions": {"k": "v"}, "criteria": ["a", "b"]}
        internal = to_internal(q)
        self.assertEqual(internal["ins"], json.dumps({"k": "v"}))
        self.assertEqual(internal["crit"], {"a": None, "b": None})


class RenderingTests(unittest.TestCase):
    def test_render_options_choice_uses_insertion_order_and_optional_text(self):
        q = {"t": "choice", "crit": {"billing": "invoices", "other": None}}
        self.assertEqual(render_options(q), ["billing: invoices", "other"])

    def test_render_options_score_is_level_indexed(self):
        q = {"t": "score", "crit": ["low", "high"]}
        self.assertEqual(render_options(q), ["level 0: low", "level 1: high"])

    def test_render_options_noul_defaults_match_upstream(self):
        self.assertEqual(
            render_options({"t": "noul", "crit": None}),
            ["false: no, the statement does not hold", "true: yes, the statement holds"],
        )
        self.assertEqual(
            render_options({"t": "noul", "crit": {"false": "nope", "true": "yep"}}),
            ["false: nope", "true: yep"],
        )

    def test_serialize_state_dumps_objects_without_ascii_escaping(self):
        self.assertEqual(serialize_state({"a": "é"}), '{"a": "é"}')
        self.assertEqual(serialize_state("raw"), "raw")


@unittest.skipUnless(importlib.util.find_spec("mlx") is not None, "mlx not installed")
class RopeTableTests(unittest.TestCase):
    """The load-time rotary tables must equal the per-sequence HF formula
    bit for bit, or the head's answers change."""

    def test_table_rows_equal_per_length_formula(self):
        import mlx.core as mx
        from mlx_omarchy_laya import model as laya_model

        cfg = laya_model.EncoderConfig(
            hidden_size=1024, num_attention_heads=16, num_hidden_layers=1,
            intermediate_size=2624, norm_eps=1e-5, local_attention=128,
            global_attn_every_n_layers=3, layer_types=["full_attention"],
            global_rope_theta=160000.0, local_rope_theta=10000.0)
        tables = laya_model.rope_tables(cfg, 512)
        hd = cfg.head_dim
        for theta in (10000.0, 160000.0):
            cos, sin = tables[theta]
            for T in (1, 96, 163, 512):
                inv_freq = mx.power(theta, -(mx.arange(0, hd, 2, dtype=mx.float32) / hd))
                freqs = mx.outer(mx.arange(T, dtype=mx.float32), inv_freq)
                emb = mx.concatenate([freqs, freqs], axis=-1)
                self.assertTrue(np.array_equal(np.array(cos[:T]), np.array(emb.cos())))
                self.assertTrue(np.array_equal(np.array(sin[:T]), np.array(emb.sin())))



@unittest.skipUnless(CONVERTED and Path(CONVERTED).exists(), "LAYA_MLX_CKPT not set; download/convert first")
class SequenceTests(unittest.TestCase):
    """Invariants of build_sequence verified with the real tokenizer."""

    @classmethod
    def setUpClass(cls):
        cls.tok = LayaTokenizer(Path(CONVERTED) / "tokenizer")
        cls.cfg = json.loads((Path(CONVERTED) / "rl_agent_config.json").read_text())

    def test_sequence_shape_and_marker_positions(self):
        q = to_internal({"type": "choice", "instructions": "pick", "criteria": {"a": None, "b": None}})
        ids, markers = build_sequence(self.tok, "some state text", q, self.cfg["max_len"], self.cfg["head_max_len"])
        self.assertEqual(ids[0], self.tok.cls_token_id)
        self.assertEqual(ids[-1], self.tok.sep_token_id)
        self.assertEqual(len(markers), 2)
        for m in markers:
            self.assertEqual(ids[m], self.tok.mask_token_id)
        self.assertLessEqual(len(ids), self.cfg["max_len"])

    def test_mask_token_in_state_is_replaced(self):
        q = to_internal({"type": "noul", "instructions": "i", "criteria": None})
        ids, _ = build_sequence(self.tok, "state with [MASK] inside", q, self.cfg["max_len"], self.cfg["head_max_len"])
        # the state segment trails the second [SEP]; markers live before it
        sep_positions = [i for i, t in enumerate(ids) if t == self.tok.sep_token_id]
        state_segment = ids[sep_positions[1] + 1:-1]
        self.assertNotIn(self.tok.mask_token_id, state_segment)
        self.assertTrue(state_segment)  # the state text actually made it in

    def test_overlong_options_trigger_even_shrink(self):
        long_text = " ".join(["optionword"] * 200)
        q = to_internal({"type": "choice", "instructions": "pick",
                         "criteria": {str(i): long_text for i in range(12)}})
        ids, markers = build_sequence(self.tok, "state", q, self.cfg["max_len"], self.cfg["head_max_len"])
        self.assertEqual(len(markers), 12)

    def test_state_truncation_respects_max_len(self):
        q = to_internal({"type": "noul", "instructions": "i", "criteria": None})
        ids, _ = build_sequence(self.tok, " ".join(["pad"] * 4000), q, self.cfg["max_len"], self.cfg["head_max_len"])
        self.assertEqual(len(ids), self.cfg["max_len"])

    def test_collate_pads_and_masks(self):
        items = [
            {"ids": [1, 2, 3], "markers": [1], "qtype": 0, "t": "choice", "crit": {}},
            {"ids": [4, 5], "markers": [1, 2], "qtype": 2, "t": "noul", "crit": None},
        ]
        b = collate(items, pad_id=0)
        self.assertEqual(b["input_ids"].tolist(), [[1, 2, 3], [4, 5, 0]])
        self.assertEqual(b["attention_mask"].tolist(), [[1, 1, 1], [1, 1, 0]])
        self.assertEqual(b["marker_mask"].tolist(), [[True, False], [True, True]])
        self.assertEqual(b["n_tokens"], 5)


class DecisionsValidationTests(unittest.TestCase):
    """POST /v1/decisions rejects malformed client input with 400 before the
    engine runs. The fake engine renders through the real encode_questions, so
    every request the validator admits must also render."""

    class _Tok:
        mask_token = "[MASK]"
        cls_token_id, sep_token_id, mask_token_id = 1, 2, 3

        def encode(self, text):
            return [10 + ord(c) % 50 for c in text]

    @classmethod
    def setUpClass(cls):
        import threading
        from http.server import ThreadingHTTPServer
        from types import SimpleNamespace

        from mlx_omarchy_laya import server as srv

        tok = cls._Tok()

        def system_one(state, questions):
            ids, _ = encode_questions(tok, state, questions, max_len=256, head_max_len=128)
            return {"answers": dict.fromkeys(ids)}

        state = SimpleNamespace(engine=SimpleNamespace(system_one=system_one), max_questions=64)
        handler = srv._make_handler(state)
        handler.log_message = lambda *args: None
        cls.httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        threading.Thread(target=cls.httpd.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def _post(self, body):
        conn = http.client.HTTPConnection(*self.httpd.server_address, timeout=10)
        try:
            conn.request("POST", "/v1/decisions", body if isinstance(body, bytes) else json.dumps(body))
            r = conn.getresponse()
            return r.status, json.loads(r.read())
        finally:
            conn.close()

    @staticmethod
    def _question(qtype, **fields):
        return {"state": "s", "questions": {"q": {"type": qtype, "instructions": "i", **fields}}}

    def test_non_object_body_is_400(self):
        for body in ([], "hi", 123, None):
            with self.subTest(body=body):
                self.assertEqual(self._post(body), (400, {"error": "request body must be a JSON object"}))

    def test_undecodable_body_is_400(self):
        cases = [
            ("invalid UTF-8", b'{"state": "\xff"}'),
            ("deep nesting", b"[" * 200000),
            ("over-long integer", b'{"state": ' + b"1" * 5000 + b"}"),
        ]
        for name, raw in cases:
            with self.subTest(name):
                code, out = self._post(raw)
                self.assertEqual(code, 400)
                self.assertTrue(out["error"].startswith("invalid JSON: "), out)

    def test_unrenderable_criteria_is_400(self):
        cases = [
            ("choice", {}),
            ("choice", {"criteria": "abc"}),
            ("choice", {"criteria": ["a", 1]}),
            ("score", {}),
            ("score", {"criteria": 5}),
            ("score", {"criteria": "abc"}),
            ("noul", {"criteria": "x"}),
            ("noul", {"criteria": {"true": 1}}),
        ]
        for qtype, fields in cases:
            with self.subTest(qtype=qtype, fields=fields):
                code, out = self._post(self._question(qtype, **fields))
                self.assertEqual(code, 400)
                self.assertTrue(out["error"].startswith("question 'q': 'criteria' must be"), out)

    def test_renderable_criteria_reach_the_engine(self):
        cases = [
            ("choice", {"criteria": {"a": None, "b": "bee"}}),
            ("choice", {"criteria": ["a", "b"], "instructions": {"k": "v"}}),
            ("score", {"criteria": ["low", "high"]}),
            ("noul", {}),
            ("noul", {"criteria": None}),
            ("noul", {"criteria": {"false": "nope", "true": "yep"}}),
        ]
        for qtype, fields in cases:
            with self.subTest(qtype=qtype, fields=fields):
                self.assertEqual(self._post(self._question(qtype, **fields)), (200, {"answers": {"q": None}}))


@unittest.skipUnless(HAS_MLX, "needs mlx (the server imports it)")
class _FakeHandlerTests(unittest.TestCase):
    """Server mechanics with a tiny synthetic checkpoint (no download)."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path("/tmp/laya-work/synthetic-ckpt")
        if not cls.tmp.exists():
            _build_synthetic_checkpoint(cls.tmp)
        cls.port = _free_port()

    def test_server_refuses_cpu_without_explicit_flag(self):
        proc = _spawn_server(self.tmp, self.port, allow_cpu=False)
        try:
            out, err = proc.communicate(timeout=120)
        except TimeoutExpired:
            proc.kill()
            out, err = proc.communicate()
        self.assertIn("requires the accelerated default device", out + err)


@unittest.skipUnless(CONVERTED and Path(CONVERTED).exists(),
                     "needs a converted checkpoint with the real tokenizer for the happy path")
class ServerHappyPathTests(unittest.TestCase):
    """Real subprocess server on CPU (explicit --allow-cpu): full HTTP contract."""

    SERVER_STATE = {"note": "synthetic load state for decisions-endpoint contract tests"}

    @classmethod
    def setUpClass(cls):
        cls.ckpt = Path(CONVERTED)
        # float32 on CPU: the fp16 default is the GPU serving dtype; CPU fp16
        # matmul is not a reference path. Hardware qualification uses the default.
        proc = _spawn_server(cls.ckpt, _free_port(), allow_cpu=True, dtype="float32")
        cls.proc = proc
        import time

        deadline = time.time() + 300
        while time.time() < deadline:
            try:
                with urllib.request.urlopen("http://127.0.0.1:%d/health" % proc._port, timeout=5) as r:
                    cls.health = json.load(r)
                return
            except Exception:
                time.sleep(1.0)
        raise AssertionError("server did not become healthy: %s" % _proc_err(proc))

    @classmethod
    def tearDownClass(cls):
        if cls.proc.poll() is None:
            cls.proc.kill()

    def _post(self, path, payload):
        req = urllib.request.Request(
            "http://127.0.0.1:%d%s" % (self.proc._port, path),
            data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                return r.status, json.load(r)
        except urllib.error.HTTPError as e:
            return e.code, json.load(e)

    def test_health_and_models_report_decisions_identity(self):
        self.assertEqual(self.health["status"], "ok")
        self.assertIn("cpu", self.health["device"])  # explicit --allow-cpu run
        with urllib.request.urlopen("http://127.0.0.1:%d/v1/models" % self.proc._port, timeout=10) as r:
            models = json.load(r)
        self.assertEqual(models["data"][0]["kind"], "decisions")
        self.assertEqual(models["data"][0]["id"], "laya")

    def test_decisions_envelope_contract(self):
        code, out = self._post("/v1/decisions", {"state": self.SERVER_STATE, "questions": {
            "dept": {"type": "choice", "instructions": "Which queue?", "criteria": {"a": None, "b": None}},
            "urg": {"type": "score", "instructions": "How bad?", "criteria": ["low", "mid", "high"]}}})
        self.assertEqual(code, 200)
        self.assertEqual(out["model"], "rl-agent")
        self.assertEqual(out["usage"]["output_tokens"], 0)
        for field in ("prompt_n", "cached_n", "prompt_ms", "prompt_per_second",
                      "predicted_n", "predicted_ms", "predicted_per_second"):
            self.assertIn(field, out["timings"], "timings missing %s" % field)
        self.assertEqual(out["timings"]["cached_n"], 0)
        self.assertEqual(out["timings"]["predicted_n"], 0)
        self.assertIn("choice", out["answers"]["dept"])
        self.assertIn("score", out["answers"]["urg"])

    def test_validation_errors(self):
        code, out = self._post("/v1/decisions", {"state": "s", "questions": {
            "x": {"type": "chat", "instructions": "nope"}}})
        self.assertEqual(code, 400)
        self.assertIn("type", out["error"])
        code, _ = self._post("/v1/decisions", {"questions": {}})
        self.assertEqual(code, 400)

    def test_question_cap_enforced_before_forward(self):
        big = {"q%d" % i: {"type": "noul", "instructions": "i", "criteria": None} for i in range(65)}
        code, out = self._post("/v1/decisions", {"state": self.SERVER_STATE, "questions": big})
        self.assertEqual(code, 400)
        self.assertIn("at most 64", out["error"])

    def test_port_conflict_does_not_disturb_first_server(self):
        # F2: a second server binding the SAME port must die on EADDRINUSE
        # while the first keeps serving (no crash, no disruption)
        proc_b = _spawn_server(self.ckpt, self.proc._port, allow_cpu=True, dtype="float32")
        try:
            out_b, err_b = proc_b.communicate(timeout=300)
        except TimeoutExpired:
            proc_b.kill()
            out_b, err_b = proc_b.communicate()
        self.assertNotEqual(proc_b.returncode, 0)
        self.assertIn("Address already in use", out_b + err_b)
        with urllib.request.urlopen("http://127.0.0.1:%d/health" % self.proc._port, timeout=5) as r:
            self.assertEqual(json.load(r)["status"], "ok")


class ManagedRelabelFailureTests(unittest.TestCase):
    """F2: a relabel failure after successful admission must clear the held
    reservation (owner-scoped) and preserve the original error — no residue."""

    def test_relabel_failure_clears_and_preserves_error(self):
        import sys
        import unittest.mock as mock

        sys.path.insert(0, str(REPO / "serve"))
        from mlx_omarchy_laya import server as srv

        ckpt = Path("/tmp/laya-work/converted/laya")
        if not ckpt.exists():
            self.skipTest("converted checkpoint not staged on this box")
        args = mock.Mock(model=str(ckpt), managed=True, dtype="float16",
                         max_questions=64, allow_cpu=True)
        clear_mock = mock.patch.object(srv, "_clear_memory").start()
        # registration succeeds (small admitted total), relabel then raises
        mock.patch.object(srv, "_register_pending",
                          return_value=({"required": 1}, "owner-token", 1)).start()
        mock.patch.object(srv, "_relabel_resident",
                          side_effect=RuntimeError("resident floor 842587660 exceeds the "
                                                   "admitted total 1; refusing to relabel")).start()
        try:
            with self.assertRaises(RuntimeError) as ctx:
                srv.LayaState(args)
            self.assertIn("refusing to relabel", str(ctx.exception))
            clear_mock.assert_called_once()
            _, kwargs = clear_mock.call_args
            self.assertEqual(kwargs.get("owner"), "owner-token")
            # MCQ-reviewed option (a): the phase-scoped guard labels the relabel
            # refusal "failed-relabel" (distinct from failed-startup) for forensics
            self.assertEqual(kwargs.get("why"), "failed-relabel")
        finally:
            mock.patch.stopall()


@unittest.skipUnless(HAS_MLX, "needs mlx (the server imports it)")
class StartupSignalCleanupTests(unittest.TestCase):
    """F3: SIGTERM/KeyboardInterrupt arriving DURING construction (mid-load)
    must clear the held reservation exactly once, then propagate."""

    def _runConstructorWithEngineRaise(self, engine_exc, expected_why="failed-startup"):
        import sys
        import unittest.mock as mock

        sys.path.insert(0, str(REPO / "serve"))
        from mlx_omarchy_laya import api, server as srv

        ckpt = Path("/tmp/laya-work/converted/laya")
        if not ckpt.exists():
            self.skipTest("converted checkpoint not staged on this box")
        args = mock.Mock(model=str(ckpt), managed=True, dtype="float16",
                         max_questions=64, allow_cpu=True)
        clear_mock = mock.patch.object(srv, "_clear_memory").start()
        register_mock = mock.patch.object(
            srv, "_register_pending",
            return_value=({"required": 1}, "owner-token", 1)).start()
        engine_mock = mock.patch.object(api, "LayaEngine",
                                        side_effect=engine_exc).start()
        try:
            with self.assertRaises(type(engine_exc)):
                srv.LayaState(args)
        finally:
            mock.patch.stopall()
        self.assertEqual(clear_mock.call_count, 1, "clear must run exactly once")
        _, kwargs = clear_mock.call_args
        self.assertEqual(kwargs.get("owner"), "owner-token")
        self.assertEqual(kwargs.get("why"), expected_why)

    def test_sigterm_mid_load_clears_once(self):
        self._runConstructorWithEngineRaise(SystemExit(0))

    def test_keyboard_interrupt_mid_load_clears_once(self):
        self._runConstructorWithEngineRaise(KeyboardInterrupt())

    def test_engine_failure_clears_once(self):
        self._runConstructorWithEngineRaise(RuntimeError("backend boom"))


@unittest.skipUnless(HAS_MLX, "needs mlx (the server imports it)")
class ManagedCoServingTests(unittest.TestCase):
    """Fail-closed reservation contract: --managed refuses startup when the
    budget registry is unavailable; standalone runs warn and serve."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path("/tmp/laya-work/synthetic-ckpt")
        if not cls.tmp.exists():
            _build_synthetic_checkpoint(cls.tmp)

    def _spawn(self, *extra):
        import subprocess

        argv = [sys.executable, "-m", "mlx_omarchy_laya.server", "--model", str(self.tmp),
                "--host", "127.0.0.1", "--port", str(_free_port()), "--allow-cpu", *extra]
        env = dict(os.environ)
        env["PYTHONPATH"] = str(REPO / "serve")
        return subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)

    def test_managed_startup_fails_without_budget_module(self):
        # no mlx_omarchy_serve on PYTHONPATH -> managed must refuse; the synthetic
        # ckpt also has no manifest footprint, which is itself a managed refusal
        proc = self._spawn("--managed")
        try:
            out, err = proc.communicate(timeout=120)
        except TimeoutExpired:
            proc.kill()
            out, err = proc.communicate()
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("managed co-serving", (out + err).lower())

    def test_unmanaged_startup_warns_instead_of_refusing(self):
        # Same environment, no --managed: the missing registry must only warn.
        # The synthetic tokenizer is a stub, so the process still exits nonzero
        # later at engine load; the point is the reservation path: warning text
        # present, fail-closed refusal language absent.
        proc = self._spawn()
        try:
            out, err = proc.communicate(timeout=120)
        except TimeoutExpired:
            proc.kill()
            out, err = proc.communicate()
        self.assertIn("memory reservation skipped", out + err)
        self.assertNotIn("refusing to serve", out + err)
        self.assertNotIn("managed co-serving", (out + err).lower())


class _StubbedServerTestCase(unittest.TestCase):
    """Shared stubbed-mlx setup for serve_main fail-closed tests."""

    def setUp(self):
        import contextlib
        import io
        import tempfile
        import types
        import unittest.mock as mock

        from mlx_omarchy_laya import server as srv
        from mlx_omarchy_serve import budget

        self.srv, self.budget, self.mock = srv, budget, mock
        self.contextlib, self.io, self.tempfile = contextlib, io, tempfile
        self.tmp = tmp = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.home = tmp / "home"
        self.ckpt = tmp / "ckpt"
        (self.ckpt / "encoder").mkdir(parents=True)
        self.manifest = {"catalog_id": "laya-test", "context_max_tokens": 16,
                         "weights_header": {"param_elements": 1000}}
        (self.ckpt / "manifest.json").write_text(json.dumps(self.manifest))
        self.cfg = cfg = types.SimpleNamespace(
            hidden_size=8, num_attention_heads=2, intermediate_size=16)
        self.total = srv.total_estimate_bytes(self.manifest, cfg, "float16", 64)

        core = types.ModuleType("mlx.core")
        core.float16, core.bfloat16, core.float32 = "float16", "bfloat16", "float32"
        mlx_pkg = types.ModuleType("mlx")
        mlx_pkg.core = core
        api = types.ModuleType("mlx_omarchy_laya.api")
        api.LayaEngine = self.engine = mock.Mock(side_effect=AssertionError("weights loaded"))
        model = types.ModuleType("mlx_omarchy_laya.model")
        self.load_encoder_config = mock.Mock(return_value=cfg)
        model.load_encoder_config = self.load_encoder_config
        self.enterContext(mock.patch.dict(sys.modules, {
            "mlx": mlx_pkg, "mlx.core": core,
            "mlx_omarchy_laya.api": api, "mlx_omarchy_laya.model": model}))
        self.enterContext(mock.patch.dict(os.environ, {"MLX_OMARCHY_HOME": str(self.home)}))
        os.environ.pop(budget.LIFETIME_FD_ENV, None)
        os.environ.pop(budget.CLAIM_FD_ENV, None)
        self.enterContext(mock.patch.object(srv.signal, "signal"))


class ClaimModeFailClosedTests(_StubbedServerTestCase):
    """A managed pair child whose claim cannot bind (malformed fd or payload,
    no --managed, a record for another model or allocation, a refusal under
    the budget lock) exits 3 with one `laya:` refusal line before any weight
    loads. Runs without a parent-lifetime fd or mlx: the mlx, engine and
    encoder-config imports are stubs."""

    def setUp(self):
        super().setUp()
        self.claim = {"name": "laya-test", "claim_token": "tok", "bytes": self.total,
                      "pair_id": "pair"}
        self.claim_reservation = self.enterContext(
            self.mock.patch.object(
                self.budget, "claim_reservation", wraps=self.budget.claim_reservation))

    def serve(self, payload=None, raw_fd=None, managed=True, expect=SystemExit):
        """Run serve_main; returns (raised exception, stderr)."""
        self.claim_reservation.reset_mock()
        self.engine.reset_mock()
        if payload is not None:
            r, w = os.pipe()
            self.addCleanup(os.close, r)
            os.write(w, payload if isinstance(payload, bytes) else json.dumps(payload).encode())
            os.close(w)
            raw_fd = str(r)
        if raw_fd is None:
            os.environ.pop(self.budget.CLAIM_FD_ENV, None)
        else:
            os.environ[self.budget.CLAIM_FD_ENV] = raw_fd
        argv = ["--model", str(self.ckpt), "--allow-cpu"] + (["--managed"] if managed else [])
        err = self.io.StringIO()
        with self.contextlib.redirect_stderr(err), self.assertRaises(expect) as ctx:
            self.srv.serve_main(argv)
        return ctx.exception, err.getvalue()

    def assertRefused(self, outcome, expected, *, reached_lock=False):
        exc, err = outcome
        self.assertEqual(exc.code, 3, err)
        lines = err.splitlines()
        self.assertEqual(len(lines), 1, err)
        self.assertTrue(lines[0].startswith("laya: "), err)
        self.assertIn(expected, lines[0])
        if reached_lock:
            self.claim_reservation.assert_called_once()
        else:
            self.claim_reservation.assert_not_called()
        self.engine.assert_not_called()

    def test_malformed_claim_fd_exits_3(self):
        self.addCleanup(sys.set_int_max_str_digits, sys.get_int_max_str_digits())
        sys.set_int_max_str_digits(4300)
        for raw in ("", "abc", " ", "x" * 500, "9" * 5000):
            with self.subTest(raw=raw[:12]):
                outcome = self.serve(raw_fd=raw)
                expected = ("laya: claim refused: MLX_OMARCHY_CLAIM_FD=%.40r "
                            "is not an fd number" % raw)
                self.assertRefused(outcome, expected)
                self.assertEqual(outcome[1].rstrip("\n"), expected)
        for raw in ("-1", str(2 ** 40)):
            with self.subTest(raw=raw[:12]):
                self.assertRefused(self.serve(raw_fd=raw),
                                   "claim refused: cannot read the claim from "
                                   "MLX_OMARCHY_CLAIM_FD=")

    def test_unreadable_claim_payload_exits_3(self):
        for payload in (b"not json", b"\xff", b"[]", {"name": "laya-test"}):
            with self.subTest(payload=payload):
                self.assertRefused(self.serve(payload=payload),
                                   "claim refused: cannot read the claim from")

    def test_unwrapped_read_errors_exit_3(self):
        # A budget module that lets OverflowError or RecursionError escape
        # read_claim_fd (huge fd, deeply nested payload) still fails closed.
        for exc in (OverflowError("fd too large"), RecursionError("deep")):
            with self.subTest(exc=type(exc).__name__), \
                    self.mock.patch.object(self.budget, "read_claim_fd", side_effect=exc):
                self.assertRefused(self.serve(raw_fd="3"), str(exc))

    def test_claim_without_managed_exits_3(self):
        self.assertRefused(self.serve(payload=self.claim, managed=False),
                           "claim mode (managed pair child) requires --managed")

    def test_claim_without_budget_module_exits_3(self):
        cases = (
            (True, "laya: claim mode requires mlx_omarchy_serve.budget on PYTHONPATH"),
            (False, "claim mode (managed pair child) requires --managed"),
        )
        with self.mock.patch.object(self.srv, "_budget_or_none", return_value=None):
            for managed, expected in cases:
                with self.subTest(managed=managed):
                    self.assertRefused(self.serve(raw_fd="3", managed=managed), expected)

    def test_claim_fd_env_matches_budget_module(self):
        self.assertEqual(self.srv._CLAIM_FD_ENV, self.budget.CLAIM_FD_ENV)

    def test_record_for_another_model_exits_3(self):
        self.assertRefused(self.serve(payload=dict(self.claim, name="other")),
                           "does not match this checkpoint's catalog_id 'laya-test'")

    def test_non_string_record_fields_exit_3(self):
        # Shared contract through the real pipe: either read_claim_fd (hardened
        # budget) or _claim_reserved (main budget) refuses before the lock.
        cases = (
            dict(self.claim, name=["laya-test"]),
            dict(self.claim, claim_token={"k": "v"}),
            dict(self.claim, pair_id=["pair"]),
        )
        for payload in cases:
            with self.subTest(payload=payload):
                self.assertRefused(self.serve(payload=payload), "claim refused")

    def test_record_allocation_mismatch_exits_3(self):
        # An int passes both budget readers, so _claim_reserved's exact wording holds.
        self.assertRefused(
            self.serve(payload=dict(self.claim, bytes=self.total + 1)),
            "claim refused: batch record 'laya-test' reserves %d bytes" % (self.total + 1))
        # Mistyped bytes may be refused by read_claim_fd or _claim_reserved.
        cases = (
            dict(self.claim, bytes=self.total + 0.5),
            dict(self.claim, bytes=float(self.total)),
            dict(self.claim, bytes="abc"),
            dict(self.claim, bytes=[self.total]),
            dict(self.claim, bytes=True),
            b'{"name": "laya-test", "claim_token": "tok", "bytes": Infinity}',
        )
        for payload in cases:
            with self.subTest(payload=payload):
                self.assertRefused(self.serve(payload=payload), "claim refused")

    def test_claim_reserved_refusal_wording_exits_3(self):
        # Patch read_claim_fd so _claim_reserved's own checks and wording are
        # exercised regardless of whether budget.read_claim_fd validates fields.
        cases = (
            ("name list", dict(self.claim, name=["laya-test"]),
             "claim refused: batch record name and claim_token must be strings, got list and str"),
            ("claim_token dict", dict(self.claim, claim_token={"k": "v"}),
             "claim refused: batch record name and claim_token must be strings, got str and dict"),
            ("bytes mismatch", dict(self.claim, bytes=self.total + 1),
             "claim refused: batch record 'laya-test' reserves %d bytes" % (self.total + 1)),
            ("bytes float", dict(self.claim, bytes=self.total + 0.5),
             "claim refused: batch record 'laya-test' bytes must be an int, got float"),
            ("bytes exact float", dict(self.claim, bytes=float(self.total)),
             "claim refused: batch record 'laya-test' bytes must be an int, got float"),
            ("bytes str", dict(self.claim, bytes="abc"),
             "claim refused: batch record 'laya-test' bytes must be an int, got str"),
            ("bytes list", dict(self.claim, bytes=[self.total]),
             "claim refused: batch record 'laya-test' bytes must be an int, got list"),
            ("bytes bool", dict(self.claim, bytes=True),
             "claim refused: batch record 'laya-test' bytes must be an int, got bool"),
            ("bytes inf", dict(self.claim, bytes=float("inf")),
             "claim refused: batch record 'laya-test' bytes must be an int, got float"),
            ("pair_id list", dict(self.claim, pair_id=["pair"]),
             "claim refused: batch record 'laya-test' pair_id must be a string or None, got list"),
        )
        for label, payload, expected in cases:
            with self.subTest(case=label), \
                    self.mock.patch.object(self.budget, "read_claim_fd", return_value=payload):
                self.assertRefused(self.serve(raw_fd="3"), expected)

    def test_unprintable_claim_fields_do_not_format_values(self):
        # Stack-independent check: refusal paths must never call str() or
        # repr() on untrusted claim values.
        class Unprintable:
            def __str__(self):
                raise RecursionError("deep str")

            def __repr__(self):
                raise RecursionError("deep repr")

        cases = (
            ("bytes", dict(self.claim, bytes=Unprintable()),
             "claim refused: batch record 'laya-test' bytes must be an int, got Unprintable"),
            ("name", dict(self.claim, name=Unprintable()),
             "claim refused: batch record name and claim_token must be strings, got Unprintable and str"),
            ("claim_token", dict(self.claim, claim_token=Unprintable()),
             "claim refused: batch record name and claim_token must be strings, got str and Unprintable"),
            ("pair_id", dict(self.claim, pair_id=Unprintable()),
             "claim refused: batch record 'laya-test' pair_id must be a string or None, got Unprintable"),
        )
        for field, payload, expected in cases:
            with self.subTest(field=field), \
                    self.mock.patch.object(self.budget, "read_claim_fd", return_value=payload):
                self.assertRefused(self.serve(raw_fd="3"), expected)

    def test_refusal_under_budget_lock_exits_3(self):
        # the registry under MLX_OMARCHY_HOME is empty: no record to claim
        self.assertRefused(self.serve(payload=self.claim),
                           "claim refused: no reservation named 'laya-test'",
                           reached_lock=True)

    def test_registry_oserror_under_budget_lock_exits_3(self):
        not_a_dir = self.tmp / "not-a-dir"
        not_a_dir.write_text("file")
        with self.mock.patch.dict(os.environ, {"MLX_OMARCHY_HOME": str(not_a_dir)}):
            self.assertRefused(self.serve(payload=self.claim), "laya: claim refused",
                               reached_lock=True)

    def test_valid_claim_binds_before_weights_load(self):
        # control: a matching claim binds, then the engine loads
        class Loaded(Exception):
            pass

        self.engine.side_effect = Loaded
        self.enterContext(self.mock.patch.object(self.srv, "_clear_memory"))
        claim = self.enterContext(self.mock.patch.object(
            self.budget, "claim_reservation",
            return_value={"claimed": [1, 2], "owner": "tok"}))
        no_pair = {k: v for k, v in self.claim.items() if k != "pair_id"}
        cases = (
            ("pair", self.claim, "pair"),
            ("omitted", no_pair, None),
            ("null", dict(self.claim, pair_id=None), None),
        )
        for label, payload, expected_pair in cases:
            with self.subTest(pair_id=label):
                claim.reset_mock()
                self.serve(payload=payload, expect=Loaded)
                claim.assert_called_once_with(
                    "laya-test", "tok", expected_bytes=self.total,
                    home=self.budget.default_home(), expected_pair=expected_pair)
                self.engine.assert_called_once()

    def test_unset_claim_fd_uses_ordinary_admission(self):
        class Admitted(Exception):
            pass

        reg = self.enterContext(self.mock.patch.object(
            self.srv, "_register_pending", side_effect=Admitted))
        read_claim = self.enterContext(self.mock.patch.object(
            self.srv, "_read_claim_payload"))
        for managed in (False, True):
            with self.subTest(managed=managed):
                reg.reset_mock()
                self.serve(managed=managed, expect=Admitted)
                reg.assert_called_once()
                args, kwargs = reg.call_args
                self.assertIs(kwargs["managed"] if "managed" in kwargs else args[2], managed)
                read_claim.assert_not_called()
                self.claim_reservation.assert_not_called()


class ManagedAdmissionFailClosedTests(_StubbedServerTestCase):
    """Under --managed, every pre-load admission or post-load relabel refusal
    exits 3 with a single `laya:` refusal line and no leaked reservation."""

    def _run_case(self, *, manifest=None, manifest_present=True, home_setup=None,
                  managed=True, expect=SystemExit, engine_ok=False, engine_exc=None,
                  err=None, no_home=False, make_manifest=None):
        case_tmp = Path(self.enterContext(self.tempfile.TemporaryDirectory()))
        ckpt = case_tmp / "ckpt"
        (ckpt / "encoder").mkdir(parents=True)
        if make_manifest is not None:
            make_manifest(ckpt / "manifest.json")
        elif manifest_present:
            raw = self.manifest if manifest is None else manifest
            data = raw if isinstance(raw, bytes) else json.dumps(raw).encode()
            (ckpt / "manifest.json").write_bytes(data)
        home = case_tmp / "home"
        if home_setup is not None:
            home_setup(home)
        self.engine.reset_mock()
        self.engine.side_effect = engine_exc or (
            None if engine_ok else AssertionError("weights loaded"))
        self.engine.return_value = self.mock.Mock(device=lambda: "Device(cpu, 0)", max_len=16)
        argv = ["--model", str(ckpt), "--allow-cpu"] + (["--managed"] if managed else [])
        err = err or self.io.StringIO()
        # every case must stop before serving: fail fast instead of binding a port
        with self.mock.patch.dict(os.environ, {"MLX_OMARCHY_HOME": str(home)}), \
                self._unresolvable_home() if no_home else self.contextlib.nullcontext(), \
                self.mock.patch.object(self.srv, "ThreadingHTTPServer",
                                       side_effect=AssertionError("server started")), \
                self.contextlib.redirect_stderr(err), self.assertRaises(expect) as ctx:
            self.srv.serve_main(argv)
        return ctx.exception, err.getvalue(), home

    def _unresolvable_home(self):
        """MLX_OMARCHY_HOME and HOME unset, no passwd entry for the uid:
        Path.home() raises RuntimeError inside budget.default_home()."""
        stack = self.contextlib.ExitStack()
        stack.enter_context(self.mock.patch.dict(os.environ))
        os.environ.pop("MLX_OMARCHY_HOME", None)
        os.environ.pop("HOME", None)
        stack.enter_context(self.mock.patch("pwd.getpwuid", side_effect=KeyError("uid")))
        return stack

    def _assert_preload_refused(self, outcome, *expected_substrings, check_empty_registry=True):
        exc, err, home = outcome
        self.assertEqual(exc.code, 3, err)
        lines = err.splitlines()
        self.assertEqual(len(lines), 1, err)
        self.assertTrue(lines[0].startswith("laya: "), err)
        for text in expected_substrings:
            self.assertIn(text, lines[0], err)
        self.engine.assert_not_called()
        if check_empty_registry:
            regs = self.budget.load_reservations(home) if home.is_dir() else {}
            self.assertEqual(regs, {})

    def test_preload_manifest_and_config_errors_exit_3(self):
        manifest_cases = (
            ("no manifest", dict(manifest_present=False),
             ("managed co-serving requires a converted manifest",)),
            ("bad json", dict(manifest=b"{not json"),
             ("managed admission failed", "JSONDecodeError")),
            ("non-utf8", dict(manifest=b"\xff\xfe"),
             ("managed admission failed", "UnicodeDecodeError")),
            ("non-object manifest", dict(manifest=[1]),
             ("managed admission failed", "ValueError", "not a JSON object")),
            ("empty list manifest", dict(manifest=[]),
             ("managed admission failed", "ValueError", "not a JSON object")),
            ("zero manifest", dict(manifest=0),
             ("managed admission failed", "ValueError", "not a JSON object")),
            ("manifest is a directory", dict(make_manifest=lambda p: p.mkdir()),
             ("managed admission failed", "IsADirectoryError")),
            ("dangling manifest symlink", dict(make_manifest=lambda p: p.symlink_to("missing-blob")),
             ("managed admission failed", "FileNotFoundError")),
            ("manifest is a FIFO", dict(make_manifest=os.mkfifo),
             ("managed admission failed", "ValueError", "not a regular file")),
            ("manifest over the cap", dict(manifest=b" " * ((1 << 20) + 1)),
             ("managed admission failed", "ValueError", "exceeds 1048576 bytes")),
            ("weights_header not object", dict(manifest=dict(self.manifest, weights_header=[1])),
             ("managed admission failed", "laya-test", "AttributeError")),
            ("param_elements Infinity",
             dict(manifest=b'{"catalog_id":"laya-test","weights_header":{"param_elements":Infinity}}'),
             ("managed admission failed", "laya-test", "OverflowError")),
            ("param_elements str",
             dict(manifest=dict(self.manifest, weights_header={"param_elements": "x"})),
             ("managed admission failed", "laya-test", "ValueError")),
            ("context_max_tokens list",
             dict(manifest=dict(self.manifest, context_max_tokens=[1])),
             ("managed admission failed", "laya-test", "TypeError")),
        )
        for label, kwargs, expected in manifest_cases:
            with self.subTest(case=label), _deadline():
                self._assert_preload_refused(self._run_case(**kwargs), *expected)

        # stack-independent: raised, not produced by nesting depth
        for exc in (IsADirectoryError(21, "Is a directory"), RecursionError("deep manifest")):
            with self.subTest(case=type(exc).__name__), \
                    self.mock.patch.object(self.srv, "_read_manifest", side_effect=exc):
                self._assert_preload_refused(
                    self._run_case(), "managed admission failed", type(exc).__name__)

        enc_cases = (
            ("encoder OSError", FileNotFoundError(2, "missing encoder config"), "FileNotFoundError"),
            ("encoder ValueError", ValueError("invalid encoder config"), "ValueError"),
            ("encoder KeyError", KeyError("num_attention_heads"), "KeyError"),
        )
        for label, exc, exc_name in enc_cases:
            with self.subTest(case=label):
                self.load_encoder_config.side_effect = exc
                try:
                    self._assert_preload_refused(
                        self._run_case(), "managed admission failed", "laya-test", exc_name)
                finally:
                    self.load_encoder_config.side_effect = None

    def test_preload_budget_errors_exit_3(self):
        with self.subTest(case="does not fit"), \
                self.mock.patch.object(self.budget, "mem_available", return_value=1):
            self._assert_preload_refused(
                self._run_case(), "managed admission failed", "laya-test", "BudgetError", "does not fit")

        with self.subTest(case="corrupt registry"):
            self._assert_preload_refused(
                self._run_case(home_setup=lambda h: (h.mkdir(), (h / "reservations.json").write_text("{bad"))),
                "managed admission failed", "laya-test", "BudgetError", "unreadable",
                check_empty_registry=False)

        with self.subTest(case="registry home is a file"):
            self._assert_preload_refused(
                self._run_case(home_setup=lambda h: h.write_text("not a dir")),
                "managed admission failed", "laya-test", "FileExistsError")

        with self.subTest(case="held by another owner"), \
                self.mock.patch.object(self.budget, "mem_available", return_value=1 << 40):
            outcome = self._run_case(
                home_setup=lambda h: self.budget.set_reservation("laya-test", 10, home=h, owner="other"))
            self._assert_preload_refused(
                outcome, "managed admission failed", "laya-test", "BudgetError", "already held",
                check_empty_registry=False)
            regs = self.budget.load_reservations(outcome[2])
            self.assertEqual(list(regs), ["laya-test"])
            self.assertEqual(regs["laya-test"]["owner"], "other")

    def test_relabel_failures_clear_reservation_and_exit_3(self):
        cases = (
            ("relabel BudgetError",
             lambda: self.mock.patch.object(
                 self.budget, "set_reservation_state",
                 side_effect=self.budget.BudgetError("no reservation named 'laya-test'")),
             ("managed relabel failed", "laya-test", "BudgetError")),
            ("relabel OSError",
             lambda: self.mock.patch.object(
                 self.budget, "set_reservation_state",
                 side_effect=OSError(28, "No space left on device")),
             ("managed relabel failed", "laya-test", "OSError", "No space left on device")),
            ("relabel floor mismatch",
             lambda: self.mock.patch.object(self.srv, "relabel_admissible", return_value=False),
             ("refusing to relabel", "laya-test")),
        )
        for label, patch_fn, expected in cases:
            with self.subTest(case=label), \
                    self.mock.patch.object(self.budget, "mem_available", return_value=1 << 40), \
                    self.mock.patch.object(self.budget, "clear_reservation",
                                           wraps=self.budget.clear_reservation) as clear_spy, \
                    self.mock.patch.object(self.srv, "_clear_memory",
                                           wraps=self.srv._clear_memory) as why_spy, \
                    patch_fn():
                exc, err, home = self._run_case(engine_ok=True)
                self.assertEqual(exc.code, 3, err)
                self.assertNotIn("Traceback", err)
                lines = err.splitlines()
                refusal_lines = [ln for ln in lines if not ln.startswith("laya: admit | ")]
                self.assertEqual(len(refusal_lines), 1, err)
                self.assertEqual(refusal_lines[0], lines[-1], err)
                self.assertTrue(refusal_lines[0].startswith("laya: "), err)
                for text in expected:
                    self.assertIn(text, refusal_lines[0], err)
                self.engine.assert_called_once()
                clear_spy.assert_called_once()
                self.assertEqual(why_spy.call_args.kwargs["why"], "failed-relabel")
                self.assertEqual(self.budget.load_reservations(home), {})

    def _stderr_failing_on(self, prefix, on_match=None):
        io = self.io

        class Stream(io.StringIO):
            def write(self, text):
                if text.startswith(prefix):
                    if on_match is not None:
                        on_match()
                    raise OSError(28, "No space left on device")
                return super().write(text)

        return Stream()

    def test_failure_after_admission_clears_reservation(self):
        # the reservation is committed before the admit lines print
        with self.mock.patch.object(self.budget, "mem_available", return_value=1 << 40):
            exc, err, home = self._run_case(err=self._stderr_failing_on("laya: admit |"))
        self.assertEqual(exc.code, 3, err)
        self.assertEqual(err.splitlines(), [
            "laya: managed admission failed for laya-test: OSError: "
            "[Errno 28] No space left on device"])
        self.engine.assert_not_called()
        self.assertEqual(self.budget.load_reservations(home), {})

    def test_failure_after_claim_clears_reservation(self):
        payload = {"name": "laya-test", "claim_token": "tok", "bytes": self.total,
                   "pair_id": "pair"}
        with self.mock.patch.dict(os.environ, {self.budget.CLAIM_FD_ENV: "3"}), \
                self.mock.patch.object(self.srv, "_read_claim_payload", return_value=payload), \
                self.mock.patch.object(self.budget, "claim_reservation",
                                       return_value={"claimed": [1, 2], "owner": "tok"}), \
                self.mock.patch.object(self.srv, "_clear_memory") as clear:
            exc, err, home = self._run_case(err=self._stderr_failing_on("laya: claimed"))
        self.assertEqual(exc.code, 3, err)
        self.assertIn("managed admission failed for laya-test: OSError", err)
        self.engine.assert_not_called()
        clear.assert_called_once_with("laya-test", True, why="failed-startup", owner="tok",
                                      home=home)

    def test_multiline_refusal_is_collapsed_to_one_line(self):
        manifest = dict(self.manifest, catalog_id="laya\ntest")
        with self.mock.patch.object(self.budget, "mem_available", return_value=1):
            self._assert_preload_refused(
                self._run_case(manifest=manifest),
                "managed admission failed", "laya test", "BudgetError", "does not fit")

    def test_managed_engine_load_error_keeps_original_exception(self):
        # weight loading is not admission: its errors are not converted to exit 3
        with self.mock.patch.object(self.budget, "mem_available", return_value=1 << 40):
            exc, err, home = self._run_case(engine_exc=OSError("weights unreadable"),
                                            expect=OSError)
        self.assertEqual(str(exc), "weights unreadable")
        self.assertEqual(self.budget.load_reservations(home), {})

    def test_standalone_relabel_error_keeps_original_exception(self):
        with self.mock.patch.object(self.budget, "mem_available", return_value=1 << 40), \
                self.mock.patch.object(self.budget, "set_reservation_state",
                                       side_effect=self.budget.BudgetError("relabel boom")):
            exc, err, home = self._run_case(managed=False, engine_ok=True,
                                            expect=self.budget.BudgetError)
        self.assertEqual(str(exc), "relabel boom")
        self.assertEqual(self.budget.load_reservations(home), {})

    def test_standalone_corrupt_manifest_keeps_original_exception(self):
        exc, err, home = self._run_case(
            manifest=b"{not json", managed=False, expect=json.JSONDecodeError)
        self.assertIsInstance(exc, json.JSONDecodeError)
        self.assertEqual(err, "")
        self.engine.assert_not_called()
        self.assertEqual(self.budget.load_reservations(home) if home.is_dir() else {}, {})

    def test_standalone_fifo_manifest_raises_before_admission(self):
        with _deadline():
            exc, err, home = self._run_case(
                make_manifest=os.mkfifo, managed=False, expect=ValueError)
        self.assertIn("not a regular file", str(exc))
        self.assertEqual(err, "")
        self.engine.assert_not_called()
        self.assertEqual(self.budget.load_reservations(home) if home.is_dir() else {}, {})

    def test_unresolvable_home_premise(self):
        # control: the harness premise, true on base and head
        with self._unresolvable_home():
            with self.assertRaises(RuntimeError):
                self.budget.default_home()

    def test_managed_unresolvable_home_exits_3_before_admission(self):
        with self.mock.patch.object(self.budget, "admit_and_reserve",
                                    wraps=self.budget.admit_and_reserve) as admit:
            self._assert_preload_refused(
                self._run_case(no_home=True),
                "managed co-serving cannot locate the budget registry",
                "Could not determine home directory", "MLX_OMARCHY_HOME")
        admit.assert_not_called()

    def test_claim_unresolvable_home_exits_3_before_claim(self):
        payload = {"name": "laya-test", "claim_token": "tok", "bytes": self.total,
                   "pair_id": "pair"}
        with self.mock.patch.dict(os.environ, {self.budget.CLAIM_FD_ENV: "3"}), \
                self.mock.patch.object(self.srv, "_read_claim_payload", return_value=payload), \
                self.mock.patch.object(self.budget, "claim_reservation",
                                       wraps=self.budget.claim_reservation) as claim:
            self._assert_preload_refused(
                self._run_case(no_home=True),
                "managed co-serving cannot locate the budget registry",
                "Could not determine home directory")
        claim.assert_not_called()

    def _lose_home_after_admission(self, *args, **kwargs):
        # the registry home vanishes after admission
        admitted_home = Path(os.environ.pop("MLX_OMARCHY_HOME"))
        self.assertIn("laya-test", self.budget.load_reservations(admitted_home))
        os.environ.pop("HOME", None)
        return self.mock.Mock(device=lambda: "Device(cpu, 0)", max_len=16)

    def test_managed_pinned_home_relabels_and_clears_at_shutdown(self):
        with self.mock.patch.object(self.budget, "mem_available", return_value=1 << 40), \
                self.mock.patch("pwd.getpwuid", side_effect=KeyError("uid")), \
                self.mock.patch.object(self.budget, "set_reservation_state",
                                       wraps=self.budget.set_reservation_state) as relabel, \
                self.mock.patch.object(self.srv, "_clear_memory",
                                       wraps=self.srv._clear_memory) as clear:
            exc, err, home = self._run_case(
                engine_exc=self._lose_home_after_admission, expect=AssertionError)
        self.assertEqual(str(exc), "server started")
        self.assertNotIn("could not clear", err)
        self.engine.assert_called_once()
        relabel.assert_called_once()
        self.assertEqual(relabel.call_args.kwargs["home"], home)
        clear.assert_called_once()
        self.assertEqual(clear.call_args.kwargs["why"], "shutdown")
        self.assertEqual(clear.call_args.kwargs["home"], home)
        self.assertEqual(self.budget.load_reservations(home), {})

    def test_managed_pinned_home_clears_on_relabel_failure(self):
        with self.mock.patch.object(self.budget, "mem_available", return_value=1 << 40), \
                self.mock.patch("pwd.getpwuid", side_effect=KeyError("uid")), \
                self.mock.patch.object(self.budget, "set_reservation_state",
                                       side_effect=self.budget.BudgetError("relabel boom")), \
                self.mock.patch.object(self.srv, "_clear_memory",
                                       wraps=self.srv._clear_memory) as clear:
            exc, err, home = self._run_case(engine_exc=self._lose_home_after_admission)
        self.assertEqual(exc.code, 3, err)
        self.assertNotIn("Traceback", err)
        self.assertNotIn("could not clear", err)
        self.engine.assert_called_once()
        clear.assert_called_once()
        self.assertEqual(clear.call_args.kwargs["why"], "failed-relabel")
        self.assertEqual(clear.call_args.kwargs["home"], home)
        lines = [ln for ln in err.splitlines() if not ln.startswith("laya: admit | ")]
        self.assertEqual(len(lines), 1, err)
        self.assertIn("managed relabel failed", lines[0])
        self.assertIn("relabel boom", lines[0])
        self.assertEqual(self.budget.load_reservations(home), {})

    def test_managed_pinned_home_survives_env_loss_before_relabel(self):
        def drop_home():
            os.environ.pop("MLX_OMARCHY_HOME", None)
            os.environ.pop("HOME", None)

        with self.mock.patch.object(self.budget, "mem_available", return_value=1 << 40), \
                self.mock.patch("pwd.getpwuid", side_effect=KeyError("uid")):
            with self.subTest("during encoder config load"), \
                    self.mock.patch.object(self.budget, "admit_and_reserve",
                                           wraps=self.budget.admit_and_reserve) as admit:
                def drop_then_cfg(*_a, **_kw):
                    drop_home()
                    return self.cfg

                self.load_encoder_config.side_effect = drop_then_cfg
                try:
                    exc, err, home = self._run_case(expect=AssertionError)
                finally:
                    self.load_encoder_config.side_effect = None
                self.assertEqual(str(exc), "weights loaded")
                admit.assert_called_once()
                self.assertEqual(admit.call_args.kwargs["home"], home)
                self.assertNotIn("could not clear", err)
                self.assertTrue((home / "reservations.json").is_file())
                self.assertEqual(self.budget.load_reservations(home), {})

            with self.subTest("during admit print"):
                exc, err, home = self._run_case(
                    err=self._stderr_failing_on("laya: admit |", on_match=drop_home))
                self.assertEqual(exc.code, 3, err)
                self.assertIn("managed admission failed", err)
                self.assertNotIn("could not clear", err)
                self.engine.assert_not_called()
                self.assertTrue((home / "reservations.json").is_file())
                self.assertEqual(self.budget.load_reservations(home), {})

            with self.subTest("during engine load"):
                def drop_then_fail(*_a, **_kw):
                    drop_home()
                    raise OSError("weights unreadable")

                exc, err, home = self._run_case(engine_exc=drop_then_fail, expect=OSError)
                self.assertEqual(str(exc), "weights unreadable")
                self.assertNotIn("could not clear", err)
                self.engine.assert_called_once()
                self.assertTrue((home / "reservations.json").is_file())
                self.assertEqual(self.budget.load_reservations(home), {})

    def test_standalone_unresolvable_home_keeps_original_exception(self):
        # control: standalone keeps base behavior
        exc, err, _ = self._run_case(no_home=True, managed=False, expect=RuntimeError)
        self.assertIs(type(exc), RuntimeError)
        self.assertEqual(str(exc), "Could not determine home directory.")
        self.assertEqual(err, "")
        self.engine.assert_not_called()

    def test_standalone_invalid_reservation_precedes_home_resolution(self):
        # control: standalone validation order matches base (BudgetError before home lookup)
        bad_manifest = dict(self.manifest, weights_header={"param_elements": -100000000})
        exc, err, _ = self._run_case(
            manifest=bad_manifest, no_home=True, managed=False, expect=self.budget.BudgetError)
        self.assertIn("positive integer bytes", str(exc))
        self.assertEqual(err, "")
        self.engine.assert_not_called()


class BoundedJsonReadTests(unittest.TestCase):
    """The manifest and encoder config reads stop on a FIFO, a device or an
    oversized file instead of hanging or exhausting memory, refuse JSON that
    is not an object, and still follow Hugging Face snapshot blob symlinks."""

    CAP = 1 << 20
    MANIFEST = {"catalog_id": "laya-test", "context_max_tokens": 16,
                "weights_header": {"param_elements": 1000}}
    CONFIG = {"hidden_size": 8, "num_attention_heads": 2, "num_hidden_layers": 1,
              "intermediate_size": 16, "layer_types": ["full_attention"]}

    def setUp(self):
        import tempfile
        import types
        import unittest.mock as mock

        import mlx_omarchy_laya
        from mlx_omarchy_laya import server as srv

        self.srv = srv
        self.tmp = Path(self.enterContext(tempfile.TemporaryDirectory()))
        # model.py imports mlx.core at the top; load_encoder_config does not use it
        core = types.ModuleType("mlx.core")
        pkg = types.ModuleType("mlx")
        pkg.core = core
        self.enterContext(mock.patch.dict(sys.modules, {"mlx": pkg, "mlx.core": core}))
        sys.modules.pop("mlx_omarchy_laya.model", None)
        had = hasattr(mlx_omarchy_laya, "model")
        old = getattr(mlx_omarchy_laya, "model", None)

        def restore():
            if had:
                mlx_omarchy_laya.model = old
            elif hasattr(mlx_omarchy_laya, "model"):
                del mlx_omarchy_laya.model
        self.addCleanup(restore)
        self.model = importlib.import_module("mlx_omarchy_laya.model")

    def _dir(self, name):
        d = self.tmp / name
        d.mkdir()
        return d

    def _manifest(self, d):
        with _deadline():
            return self.srv._read_manifest(d)

    def _config(self, d):
        with _deadline():
            return self.model.load_encoder_config(d)

    def _both(self, build):
        """Yield (label, reader, path) for manifest.json and encoder config.json
        each built by build(path)."""
        for label, name, reader in (("manifest", "manifest.json", self._manifest),
                                    ("config", "config.json", self._config)):
            d = self._dir("%s-%d" % (label, len(list(self.tmp.iterdir()))))
            build(d / name)
            yield label, reader, d

    def test_regular_and_hf_blob_symlink_load_the_same(self):
        for label, name, reader, obj in (
                ("manifest", "manifest.json", self._manifest, self.MANIFEST),
                ("config", "config.json", self._config, self.CONFIG)):
            plain, linked = self._dir(label + "-plain"), self._dir(label + "-hf")
            (plain / name).write_text(json.dumps(obj))
            (linked / "blobs").mkdir()
            (linked / "blobs" / "0123abcd").write_text(json.dumps(obj))
            (linked / name).symlink_to("blobs/0123abcd")
            with self.subTest(file=label):
                self.assertEqual(reader(linked), reader(plain))
        self.assertEqual(self._manifest(self.tmp / "manifest-plain"), self.MANIFEST)
        self.assertEqual(self._config(self.tmp / "config-plain").layer_types,
                         ["full_attention"])

    def test_fifo_is_refused_without_blocking(self):
        for label, reader, d in self._both(os.mkfifo):
            with self.subTest(file=label, writer=False), \
                    self.assertRaisesRegex(ValueError, "is not a regular file"):
                reader(d)
            # a writer that never closes: reading to EOF would block forever
            fifo = next(p for p in d.iterdir())
            fd = os.open(fifo, os.O_RDWR)
            try:
                os.write(fd, json.dumps(self.MANIFEST).encode())
                with self.subTest(file=label, writer=True), \
                        self.assertRaisesRegex(ValueError, "is not a regular file"):
                    reader(d)
            finally:
                os.close(fd)

    def test_device_and_directory_are_refused(self):
        for label, reader, d in self._both(lambda p: p.symlink_to(os.devnull)):
            with self.subTest(file=label, kind="devnull"), \
                    self.assertRaisesRegex(ValueError, "is not a regular file"):
                reader(d)
        for label, reader, d in self._both(lambda p: p.mkdir()):
            with self.subTest(file=label, kind="directory"), \
                    self.assertRaises(IsADirectoryError):
                reader(d)

    def test_size_cap_boundary(self):
        for size, ok in ((self.CAP, True), (self.CAP + 1, False)):
            def build(p, size=size):
                obj = self.MANIFEST if p.name == "manifest.json" else self.CONFIG
                body = json.dumps(obj).encode()
                p.write_bytes(body + b" " * (size - len(body)))
            for label, reader, d in self._both(build):
                with self.subTest(file=label, size=size):
                    if ok:
                        self.assertTrue(reader(d))
                    else:
                        with self.assertRaisesRegex(ValueError, "exceeds 1048576 bytes"):
                            reader(d)

    def test_non_object_json_is_refused(self):
        for text in ("[]", "[1]", "0", "123", '"x"', "null", "true"):
            for label, reader, d in self._both(lambda p, t=text: p.write_text(t)):
                with self.subTest(file=label, json=text), \
                        self.assertRaisesRegex(ValueError, "is not a JSON object"):
                    reader(d)

    def test_missing_manifest_is_empty_while_dangling_symlink_and_missing_config_raise(self):
        d = self._dir("empty")
        self.assertEqual(self._manifest(d), {})
        with self.assertRaises(FileNotFoundError):
            self._config(d)
        (d / "manifest.json").symlink_to("blobs/missing")
        with self.assertRaises(FileNotFoundError):
            self._manifest(d)

    def test_manifest_lstat_error_is_not_treated_as_missing(self):
        import unittest.mock

        d = self._dir("lstat-err")
        (d / "manifest.json").symlink_to("blobs/missing")
        with unittest.mock.patch.object(
                self.srv.os, "lstat", side_effect=PermissionError(13, "Permission denied")), \
                self.assertRaises(PermissionError):
            self._manifest(d)

    @unittest.skipUnless(sys.platform.startswith("linux"), "RLIMIT_AS child is Linux-only")
    def test_reads_are_bounded_in_a_memory_capped_child(self):
        # an unbounded read of these exhausts memory; run it where that is safe
        sparse = self._dir("sparse")
        with open(sparse / "manifest.json", "wb") as f:
            f.truncate(4 << 30)
        zero = self._dir("zero")
        (zero / "manifest.json").symlink_to("/dev/zero")
        child = (
            "import resource, sys\n"
            "resource.setrlimit(resource.RLIMIT_AS, (1 << 30, 1 << 30))\n"
            "sys.path.insert(0, sys.argv[1])\n"
            "from pathlib import Path\n"
            "from mlx_omarchy_laya import server\n"
            "for d in sys.argv[2:]:\n"
            "    try:\n"
            "        print('OK', server._read_manifest(Path(d)))\n"
            "    except BaseException as e:\n"
            "        print(type(e).__name__, e)\n"
        )
        out = subprocess.run(
            [sys.executable, "-I", "-c", child, str(REPO / "serve"), str(sparse), str(zero)],
            capture_output=True, text=True, timeout=60, stdin=subprocess.DEVNULL)
        self.assertEqual(out.stdout.splitlines(), [
            "ValueError %s exceeds 1048576 bytes" % (sparse / "manifest.json"),
            "ValueError %s is not a regular file" % (zero / "manifest.json"),
        ], out.stderr)


class _RecordFile(str):
    """A RECORD entry: importlib.metadata's PackagePath carries .hash."""


def _fake_omarchy_dist(root, recorded, version):
    """An installed mlx-omarchy under root whose mlx/lib/libmlx.so holds
    b"libmlx" and whose RECORD pins sha256(recorded); recorded=None leaves
    the binary without a RECORD entry (a source build)."""
    import base64
    import hashlib
    import types

    lib = Path(root) / "mlx" / "lib" / "libmlx.so"
    lib.parent.mkdir(parents=True, exist_ok=True)
    lib.write_bytes(b"libmlx")
    files = []
    if recorded is not None:
        entry = _RecordFile("mlx/lib/libmlx.so")
        entry.hash = types.SimpleNamespace(mode="sha256", value=base64.urlsafe_b64encode(
            hashlib.sha256(recorded).digest()).decode().rstrip("="))
        files.append(entry)
    return types.SimpleNamespace(version=version, files=files, locate_file=lambda rel: Path(root) / rel)


class QualifyDeviceGuardTests(unittest.TestCase):
    """qualify refuses a non-GPU default device unless --allow-cpu, and a
    verified=mismatch provenance even with --allow-cpu, in both direct and
    --url modes. mlx.core is stubbed so the guards run without mlx;
    str(mx.default_device()) is "Device(cpu, 0)", never "cpu"."""

    @classmethod
    def setUpClass(cls):
        import threading
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

        fixture = json.loads((REPO / "tests/fixtures/laya/laya_reference.json").read_text())
        by_questions = {json.dumps(c["questions"], sort_keys=True): c["expected_mlx_fp32_cpu"]
                        for c in fixture["cases"]}
        cls.posts = posts = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                posts.append(self.path)
                out = json.dumps(by_questions[json.dumps(body["questions"], sort_keys=True)]).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(out)))
                self.end_headers()
                self.wfile.write(out)

            def log_message(self, *a):
                pass

        cls.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=cls.httpd.serve_forever, daemon=True).start()
        cls.url = "http://127.0.0.1:%d" % cls.httpd.server_port

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def setUp(self):
        del self.posts[:]

    def _qualify(self, default, *argv, dist=None):
        """Run qualify.main against stubbed mlx. dist is the installed
        mlx-omarchy distribution; None means no metadata (an editable build).
        stdout and stderr stay on self.out and self.err when main exits."""
        import contextlib
        import importlib.metadata
        import io
        import types
        import unittest.mock as mock

        class Device:
            def __init__(self, kind):
                self.kind = kind

            def __eq__(self, other):
                return isinstance(other, Device) and other.kind == self.kind

            def __str__(self):
                return "Device(%s, 0)" % self.kind

        core = types.ModuleType("mlx.core")
        core.__version__ = "stub"
        # Real mx.gpu is a DeviceType that compares equal to Device(gpu, 0);
        # a fresh Device per call keeps an identity check from passing.
        core.gpu = Device("gpu")
        core.default_device = lambda: Device(default)
        pkg = types.ModuleType("mlx")
        pkg.core = core
        from mlx_omarchy_laya import qualify

        if dist is None:
            installed = mock.patch("importlib.metadata.distribution",
                                   side_effect=importlib.metadata.PackageNotFoundError("mlx-omarchy"))
        else:
            installed = mock.patch("importlib.metadata.distribution", return_value=dist)
        self.out, self.err = io.StringIO(), io.StringIO()
        with mock.patch.dict(sys.modules, {"mlx": pkg, "mlx.core": core}), installed, \
                contextlib.redirect_stdout(self.out), contextlib.redirect_stderr(self.err):
            rc = qualify.main(["--model", str(REPO / "no-such-checkpoint"), *argv])
        return rc, self.out.getvalue()

    def test_cpu_default_refuses_url_mode_without_allow_cpu(self):
        with self.assertRaises(SystemExit) as cm:
            self._qualify("cpu", "--url", self.url)
        self.assertIn("default device is Device(cpu, 0), not mx.gpu", str(cm.exception.code))
        self.assertEqual(self.posts, [], "no case may be sent to the server before the refusal")

    def test_cpu_default_refuses_direct_mode_without_allow_cpu(self):
        with self.assertRaises(SystemExit) as cm:
            self._qualify("cpu")
        self.assertIn("default device is Device(cpu, 0), not mx.gpu", str(cm.exception.code))

    def test_allow_cpu_runs_url_mode_and_labels_device(self):
        rc, out = self._qualify("cpu", "--url", self.url, "--allow-cpu")
        self.assertEqual(rc, 0)
        self.assertEqual(len(self.posts), 6)
        self.assertIn("device: Device(cpu, 0)", out)
        self.assertIn("VERDICT: PASS", out)

    def test_gpu_default_runs_url_mode_without_allow_cpu(self):
        rc, out = self._qualify("gpu", "--url", self.url)
        self.assertEqual(rc, 0)
        self.assertEqual(len(self.posts), 6)
        self.assertIn("device: Device(gpu, 0)", out)
        self.assertIn("VERDICT: PASS", out)

    def test_receipt_prints_the_loaded_mlx_provenance_line(self):
        # mlx_provenance.py prints indented JSON; its last line is "}".
        rc, out = self._qualify("gpu", "--url", self.url)
        self.assertEqual(rc, 0)
        line, = [ln for ln in out.splitlines() if ln.startswith("provenance")]
        self.assertTrue(line.startswith("provenance: mlx-omarchy "), line)
        self.assertIn(" mx=stub ", line)  # the mlx this interpreter loaded
        self.assertIn(" verified=no-metadata ", line)
        self.assertIn(" harness=", line)

    def _assert_refused(self, *argv, dist, default="gpu"):
        with self.assertRaises(SystemExit) as cm:
            self._qualify(default, *argv, dist=dist)
        self.assertEqual(cm.exception.code, 3)
        self.assertEqual(self.posts, [], "no case may be sent to the server before the refusal")
        err = self.err.getvalue()
        self.assertEqual(self.out.getvalue(), "", "a refused run prints no receipt")
        self.assertIn(" verified=mismatch ", err)
        self.assertIn("REFUSING TO EMIT NUMBERS: ", err)
        return err

    def test_record_mismatch_refuses_before_any_case(self):
        import tempfile

        for mode in (("--url", self.url), ()):  # () = direct engine, never loaded
            with self.subTest(mode=mode or "direct"), tempfile.TemporaryDirectory() as tmp:
                err = self._assert_refused(*mode, dist=_fake_omarchy_dist(tmp, b"stale", "stub"))
                self.assertIn("RECORD says", err)

    def test_version_mismatch_refuses(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            err = self._assert_refused("--url", self.url, dist=_fake_omarchy_dist(tmp, b"libmlx", "1.0"))
        self.assertIn("compiled mx.__version__ 'stub' != installed distribution version '1.0'", err)

    def test_allow_cpu_does_not_waive_the_refusal(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            self._assert_refused("--url", self.url, "--allow-cpu", default="cpu",
                                 dist=_fake_omarchy_dist(tmp, b"stale", "stub"))

    def test_only_a_mismatch_refuses(self):
        import tempfile

        for recorded, verdict in ((b"libmlx", "match"), (None, "unverified"), (b"stale", "mismatch")):
            with self.subTest(verdict=verdict), tempfile.TemporaryDirectory() as tmp:
                del self.posts[:]
                dist = _fake_omarchy_dist(tmp, recorded, "stub")
                if verdict == "mismatch":
                    self._assert_refused("--url", self.url, dist=dist)
                    continue
                rc, out = self._qualify("gpu", "--url", self.url, dist=dist)
                self.assertEqual(rc, 0)
                self.assertEqual(len(self.posts), 6)
                self.assertIn(" verified=%s " % verdict, out)
                self.assertIn("VERDICT: PASS", out)


class QualifyProvenanceTests(unittest.TestCase):
    def test_provenance_reports_record_match_and_refuses_mismatch(self):
        # A match returns the receipt line naming the verdict and the loaded
        # hash; a RECORD mismatch prints that line to stderr and exits 3, the
        # scripts/mlx_provenance.py refusal contract.
        import contextlib
        import hashlib
        import io
        import tempfile
        import types
        import unittest.mock as mock

        from mlx_omarchy_laya import qualify

        core = types.ModuleType("mlx.core")
        core.__version__ = "1.0"
        pkg = types.ModuleType("mlx")
        pkg.core = core
        loaded = " libmlx.so=sha256:" + hashlib.sha256(b"libmlx").hexdigest()[:16]
        for recorded, verdict in ((b"libmlx", "match"), (b"stale", "mismatch")):
            err = io.StringIO()
            with self.subTest(verdict=verdict), tempfile.TemporaryDirectory() as tmp, \
                    mock.patch.dict(sys.modules, {"mlx": pkg, "mlx.core": core}), \
                    mock.patch("importlib.metadata.distribution",
                               return_value=_fake_omarchy_dist(tmp, recorded, "1.0")), \
                    contextlib.redirect_stderr(err):
                prefix = "provenance: mlx-omarchy 1.0 mx=1.0 verified=%s harness=" % verdict
                if verdict == "match":
                    line = qualify._provenance(REPO)
                    self.assertTrue(line.startswith(prefix), line)
                    self.assertTrue(line.endswith(loaded), line)
                    self.assertEqual(err.getvalue(), "")
                else:
                    with self.assertRaises(SystemExit) as cm:
                        qualify._provenance(REPO)
                    self.assertEqual(cm.exception.code, 3)
                    line, refusal = err.getvalue().splitlines()
                    self.assertTrue(line.startswith(prefix), line)
                    self.assertTrue(line.endswith(loaded), line)
                    self.assertTrue(refusal.startswith("REFUSING TO EMIT NUMBERS: mlx/lib/libmlx.so at "), refusal)

    def test_mismatch_refuses_when_the_line_cannot_be_formatted(self):
        # provenance_line() runs git with a 5 s timeout; a slow tree must not
        # turn a detected mismatch into "provenance: unavailable" and a PASS.
        import contextlib
        import io
        import subprocess
        import tempfile
        import types
        import unittest.mock as mock

        from mlx_omarchy_laya import qualify

        def slow_git(cmd, *a, **kw):  # rev-parse answers; the status call times out
            if "status" in cmd:
                raise subprocess.TimeoutExpired(cmd, 5)
            return subprocess.CompletedProcess(cmd, 0, stdout="abc1234\n", stderr="")

        core = types.ModuleType("mlx.core")
        core.__version__ = "1.0"
        pkg = types.ModuleType("mlx")
        pkg.core = core
        err = io.StringIO()
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.dict(sys.modules, {"mlx": pkg, "mlx.core": core}), \
                mock.patch("importlib.metadata.distribution", return_value=_fake_omarchy_dist(tmp, b"stale", "1.0")), \
                mock.patch("subprocess.run", side_effect=slow_git), \
                contextlib.redirect_stderr(err):
            with self.assertRaises(SystemExit) as cm:
                qualify._provenance(REPO)
        self.assertEqual(cm.exception.code, 3)
        line, refusal = err.getvalue().splitlines()
        self.assertTrue(line.startswith("provenance: unavailable: TimeoutExpired: "), line)
        self.assertTrue(refusal.startswith("REFUSING TO EMIT NUMBERS: mlx/lib/libmlx.so at "), refusal)

    def test_provenance_reports_no_mlx(self):
        import unittest.mock as mock

        from mlx_omarchy_laya import qualify

        with mock.patch.dict(sys.modules, {"mlx": None, "mlx.core": None}):  # import mlx.core raises
            line = qualify._provenance(REPO)
        self.assertTrue(line.startswith("provenance: mlx-omarchy None mx=None verified=no-mlx harness="), line)
        self.assertTrue(line.endswith(" files=unresolved"), line)

    def test_missing_script_still_prints_a_provenance_line(self):
        from mlx_omarchy_laya import qualify

        self.assertEqual(qualify._provenance(REPO / "no-such-repo"), "provenance: script not present")


# ---------------------------------------------------------------- helpers


def _free_port():
    import socket

    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _proc_err(proc):
    return proc.stderr.read() if proc.stderr else ""


def _spawn_server(ckpt, port, allow_cpu, dtype=None):
    argv = [sys.executable, "-m", "mlx_omarchy_laya.server", "--model", str(ckpt),
            "--host", "127.0.0.1", "--port", str(port)]
    if allow_cpu:
        argv.append("--allow-cpu")
    if dtype:
        argv += ["--dtype", dtype]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO / "serve")
    proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
    proc._port = port
    return proc


def _build_synthetic_checkpoint(out: Path):
    """Tiny random-weight Laya checkpoint with the real key set and shapes.

    Random weights make endpoint/loader/shape tests runnable without the
    843 MB download; they prove plumbing, not model semantics.
    """
    import mlx.core as mx

    rng = np.random.default_rng(7)
    hidden, inter, heads_hd, layers = 64, 128, 32, 2
    weights = {}

    def w(name, shape):
        weights[name] = (rng.standard_normal(shape) * 0.02).astype(np.float16)

    w("encoder.embeddings.tok_embeddings.weight", (128, hidden))
    w("encoder.embeddings.norm.weight", (hidden,))
    w("encoder.final_norm.weight", (hidden,))
    for i in range(layers):
        p = "encoder.layers.%d." % i
        w(p + "attn.Wqkv.weight", (3 * hidden, hidden))
        w(p + "attn.Wo.weight", (hidden, hidden))
        w(p + "mlp.Wi.weight", (2 * inter, hidden))
        w(p + "mlp.Wo.weight", (hidden, inter))
        w(p + "mlp_norm.weight", (hidden,))
        if i > 0:
            w(p + "attn_norm.weight", (hidden,))
    w("type_emb.weight", (3, hidden))
    w("scorer.0.weight", (hidden,))
    w("scorer.0.bias", (hidden,))
    w("scorer.1.weight", (hidden, hidden))
    w("scorer.1.bias", (hidden,))
    w("scorer.3.weight", (1, hidden))
    w("scorer.3.bias", (1,))
    w("act_head.0.weight", (16, hidden + 4))
    w("act_head.0.bias", (16,))
    w("act_head.2.weight", (2, 16))
    w("act_head.2.bias", (2,))
    w("temperature", (3,))

    out.mkdir(parents=True, exist_ok=True)
    (out / "model.safetensors").write_bytes(_safetensors_bytes(weights))
    (out / "rl_agent_config.json").write_text(json.dumps({
        "encoder": "answerdotai/ModernBERT-large", "head_layers": 2, "max_len": 128,
        "head_max_len": 48, "max_prefixes": 6, "act_costs": {"escalate": 0.5},
        "cost_wrong_act": 3.0, "amp_dtype": "bf16", "model_name": "laya-synthetic",
        "temperature": [1.0, 1.0, 1.0], "temperature_by_options": {},
    }))
    (out / "encoder").mkdir(exist_ok=True)
    (out / "encoder/config.json").write_text(json.dumps({
        "model_type": "modernbert", "hidden_size": hidden, "num_attention_heads": 2,
        "num_hidden_layers": layers, "intermediate_size": inter, "norm_eps": 1e-5,
        "local_attention": 16, "global_attn_every_n_layers": 3,
        "layer_types": ["full_attention", "sliding_attention"],
        "rope_parameters": {"full_attention": {"rope_theta": 160000.0},
                            "sliding_attention": {"rope_theta": 10000.0}},
        "vocab_size": 128,
    }))
    if CONVERTED and Path(CONVERTED).exists():
        import shutil

        shutil.copytree(Path(CONVERTED) / "tokenizer", out / "tokenizer", dirs_exist_ok=True)
    else:
        (out / "tokenizer").mkdir(exist_ok=True)
        (out / "tokenizer/tokenizer.json").write_text(json.dumps({"version": "1.0", "model": {}}))
        (out / "tokenizer/tokenizer_config.json").write_text(json.dumps({}))


def _safetensors_bytes(weights: dict) -> bytes:
    header = {}
    offset = 0
    for name in sorted(weights):
        arr = weights[name]
        nbytes = arr.nbytes
        header[name] = {"dtype": "F16", "shape": list(arr.shape), "data_offsets": [offset, offset + nbytes]}
        offset += nbytes
    hj = json.dumps(header).encode()
    pad = (8 - len(hj) % 8) % 8
    hj += b" " * pad
    blob = b"".join(weights[n].tobytes() for n in sorted(weights))
    return struct.pack("<Q", len(hj)) + hj + blob


if __name__ == "__main__":
    unittest.main()
