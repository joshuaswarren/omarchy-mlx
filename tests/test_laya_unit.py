"""Unit tests for the Laya typed-decision adapter — no model download needed.

Structural and calibration-semantics checks that always run: option
rendering, temperature bucketing, confidence, sequence-building invariants
against the upstream formulas, converter validation on a synthetic
checkpoint, and the exact CPU-refusal contract of the server. The full
numerical comparison against pinned upstream torch lives in
tests/test_laya_reference.py (env-gated).
"""

import importlib.util
import json
import os
import struct
import subprocess
import sys
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
    render_options,
    serialize_state,
    temp_bucket,
    to_internal,
)

CONVERTED = os.environ.get("LAYA_MLX_CKPT")  # converted checkpoint dir, if available
# The server and engine import mlx; hosts without it skip those tests.
HAS_MLX = importlib.util.find_spec("mlx") is not None


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
