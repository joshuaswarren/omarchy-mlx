"""system_one inference: typed questions -> calibrated typed answers.

Port of the pinned upstream RLAgent.system_one
(convaiinnovations/laya revision 1c5edc17a7acd8701df6fc341c0d179f1c62c982,
rl_agent_api.py) on top of the mlx.core model port. Calibration semantics
are preserved exactly: per-(qtype, cardinality) temperatures from
rl_agent_config.json, 4-decimal rounding, act_probability = softmax of the
escalate/answer action head, output_tokens always 0 (single forward pass).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from . import model as laya_model
from .sequence import LayaTokenizer, collate, encode_questions, temp_bucket


class LayaEngine:
    def __init__(self, model_dir: str | Path, dtype=None, require_gpu: bool = True):
        model_dir = Path(model_dir)
        self.model_dir = model_dir
        if require_gpu and mx.default_device() != mx.gpu:
            raise RuntimeError(
                "mlx_omarchy_laya requires the accelerated default device (mx.gpu, the "
                "Omarchy Apple-GPU backend); default device is %s. Start the server on "
                "Apple-silicon hardware with the mlx-omarchy wheel, or pass "
                "--allow-cpu explicitly for reference/testing runs." % mx.default_device()
            )
        with open(model_dir / "rl_agent_config.json") as f:
            self.cfg = json.load(f)
        self.max_len = int(self.cfg["max_len"])
        self.head_max_len = int(self.cfg["head_max_len"])
        self.head_layers = int(self.cfg["head_layers"])
        self.temperature = self.cfg.get("temperature", [1.0, 1.0, 1.0])
        self.temperature_by_options = self.cfg.get("temperature_by_options", {})
        self.tok = LayaTokenizer(model_dir / "tokenizer")
        self.enc_cfg = laya_model.load_encoder_config(model_dir / "encoder")
        if dtype is None:
            dtype = mx.float16  # checkpoint weights are stored fp16
        if isinstance(dtype, str):
            dtype = getattr(mx, dtype)
        self.dtype = dtype
        self.weights = laya_model.load_weights(model_dir, dtype)
        # materialize the casts now: weights are truly resident before the server
        # relabels its reservation "resident" (admission-honest co-serving)
        for w in self.weights.values():
            mx.eval(w)
        self.rope = laya_model.rope_tables(self.enc_cfg, self.max_len)
        mx.eval(self.rope)

    @property
    def model_id(self) -> str:
        name = self.cfg.get("model_name")
        if not name or name == "rl-agent":  # upstream hard-codes rl-agent in cfg; not a catalog identity
            return self.model_dir.name
        return name

    def device(self) -> str:
        return str(mx.default_device())

    def system_one(self, state, questions: dict) -> dict:
        """questions: {id: {"type": "choice"|"score"|"noul", "instructions", "criteria"}}.

        Returns the upstream envelope:
        {"model": "rl-agent", "answers": {id: ...}, "usage": {"input_tokens": N, "output_tokens": 0}}
        """
        from .sequence import confidence_from_probs

        t0 = time.perf_counter()
        ids_in_order, items = encode_questions(self.tok, state, questions, self.max_len, self.head_max_len)
        b = collate(items, self.tok.pad_token_id)

        t_fwd0 = time.perf_counter()
        logits, act = laya_model.forward(
            self.weights,
            self.enc_cfg,
            self.rope,
            mx.array(b["input_ids"]),
            mx.array(b["attention_mask"]),
            mx.array(b["marker_pos"]),
            mx.array(b["marker_mask"]),
            mx.array(b["qtype"]),
            self.head_layers,
        )
        mx.eval(logits, act)
        t_fwd1 = time.perf_counter()
        prompt_ms = max((t_fwd0 - t0) * 1000.0, 0.001)
        predicted_ms = max((t_fwd1 - t_fwd0) * 1000.0, 0.0)
        # mlx-serve cross-server timings contract (ServePerformanceAudit): identical
        # field names for every backend. Laya has no KV cache and no autoregressive
        # decode: cached_n and predicted_n are honestly 0; predicted_ms is the
        # single forward pass wall time.
        timings = {
            "prompt_n": b["n_tokens"],
            "cached_n": 0,
            "prompt_ms": round(prompt_ms, 3),
            "prompt_per_second": round(b["n_tokens"] / prompt_ms * 1000.0, 3) if prompt_ms > 0 else 0.0,
            "predicted_n": 0,
            "predicted_ms": round(predicted_ms, 3),
            "predicted_per_second": 0.0,
        }
        logits = np.array(logits.astype(mx.float32))
        act_p = np.array(mx.softmax(act.astype(mx.float32), axis=-1))
        answers = {}
        for r, qid in enumerate(ids_in_order):
            q = items[r]
            k = len(q["markers"])
            qt = q["qtype"]
            z = logits[r, :k] / self.temperature_by_options.get(temp_bucket(qt, k), self.temperature[qt])
            p = np.exp(z - z.max())
            p = p / p.sum()
            ext = {"act_probability": float(act_p[r, 0])}
            if q["t"] == "choice":
                keys = list(q["crit"].keys())
                answers[qid] = {
                    "type": "choice",
                    "choice": keys[int(p.argmax())],
                    "probabilities": {kk: round(float(v), 4) for kk, v in zip(keys, p)},
                    "confidence": round(confidence_from_probs(p, k), 4),
                    "rl_agent": ext,
                }
            elif q["t"] == "score":
                answers[qid] = {
                    "type": "score",
                    "score": round(float((np.arange(k) * p).sum()), 4),
                    "legend": {str(i): c for i, c in enumerate(q["crit"])},
                    "probabilities": {str(i): round(float(v), 4) for i, v in enumerate(p)},
                    "confidence": round(confidence_from_probs(p, k), 4),
                    "rl_agent": ext,
                }
            else:
                answers[qid] = {"type": "noul", "noul": round(float(p[1]), 4), "rl_agent": ext}
        return {
            "model": "rl-agent",
            "answers": answers,
            "usage": {"input_tokens": b["n_tokens"], "output_tokens": 0},
            "timings": timings,
        }
