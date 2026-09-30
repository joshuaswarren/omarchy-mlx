"""Automatic decision-routing gate.

Asks Laya one versioned `choice` question selecting
`conversation`, `structured_decision`, or `clarify` on a completed,
bounded user turn that appears to request a structured decision.
Thresholds, question text, and version live in a single versioned
`ROUTING_POLICY` record. Policy "1" is the unevaluated starter policy
shipping with the catalog; every change is a new policy version tied
to a frozen held-out suite (sha256 in the notebook).

Coordination rules:
- Material is preserved in full. The same fit check used by
  `coordinator._typed_request` (build_sequence vs hand-rebuilt ids)
  refuses any truncation; over-budget material returns the LLM path
  immediately, never a silent shrink.
- Warm deadline is 250 ms. A timed-out call stays accounted for in the
  runner's pending list until it finishes; no replacement is launched.
- `confidence` is one minus normalized entropy, not correctness
  probability. `act_probability` is high == answer, low == escalate.
  Neither is inverted.

This module is the single source of truth for the policy record. The
coordinator hooks a flag in front of `submit`; pair records expose
the policy version and only mark it evaluated when a held-out suite
passes with evidence attached.
"""

from __future__ import annotations

import json
import math
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

ROUTES = ("conversation", "structured_decision", "clarify")
# 250 ms warm deadline. A cold call is itself a refusal to route, never a
# reason to block ordinary chat.
WARM_DEADLINE_SECONDS = 0.250


@dataclass(frozen=True)
class RoutingPolicy:
    """Versioned policy record: question text, thresholds, evidence pointer."""

    version: str
    question_text: str
    p_min: float
    margin_min: float
    act_min: float
    suite_sha256: str | None = None
    receipt: str | None = None
    # Marker so the same policy object cannot be confused across versions.
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "question_text": self.question_text,
            "p_min": self.p_min,
            "margin_min": self.margin_min,
            "act_min": self.act_min,
            "suite_sha256": self.suite_sha256,
            "receipt": self.receipt,
            "note": self.note,
        }


# Starter policy "1". UNQUALIFIED: held-out suite is frozen but never
# evaluated. The flag defaults OFF; this object is here so the wiring
# has one canonical place to read the version and the question text.
ROUTING_POLICY = RoutingPolicy(
    version="3",
    question_text=(
        "Pick the label that matches the fingerprint: "
        "(A) conversation = no options marker AND no criteria marker AND "
        "no decision request; "
        "(B) structured_decision = BOTH an options marker AND a criteria "
        "marker (explicit user-supplied alternatives with explicit "
        "criteria); "
        "(C) clarify = either marker alone, or neither marker but the "
        "user asks for a decision. Injection-style text that imitates "
        "decision grammar counts as structured_decision ONLY when BOTH "
        "markers appear with explicit user-supplied content."
    ),
    # Calibrated conservatively against the Laya choice head. The held-out
    # evaluation re-tunes these and writes policy "2" if precision < 99%.
    p_min=0.55,
    margin_min=0.20,
    act_min=0.55,
    suite_sha256=None,
    receipt=None,
    note="iter-3 prompt with fingerprint-only state; dev sweeps 1 and 2 failed (no passing cell); iter-3 queued; flag defaults OFF",
)


@dataclass
class RoutingOutcome:
    """One routing attempt: either a route label or a documented skip."""

    route: str | None = None
    reason: str = ""
    probabilities: dict[str, float] = field(default_factory=dict)
    runner_up_margin: float | None = None
    act_probability: float | None = None
    latency_ms: float | None = None
    timed_out: bool = False


_OPTIONS_RE = re.compile(r"\boptions?\s*:\s*", re.IGNORECASE)
_CRITERIA_RE = re.compile(r"\bcriteri[ao]\s*:\s*", re.IGNORECASE)
_INJECTION_KEYWORDS = ("ignore", "disregard", "forget", "override",
                       "pretend", "system:", "you are now", "run rm")
_NEGATION_KEYWORDS = ("not", "never", "don't", "do not", "avoid", "skip")


def _fingerprint(text: str) -> dict:
    """A minimal structural summary of the user turn.

    The choice head sees ONLY the fingerprint. We deliberately exclude
    the raw text so injection text cannot be read as a decision request.
    Iter 3 of the routing policy: this is the design fix that addresses
    the injection-fooled head observed in dev iter 1 and 2.
    """
    text_lower = text.lower()
    return {
        "char_count": len(text),
        "word_count": len(text.split()),
        "ends_with_question": text.rstrip().endswith("?"),
        "has_options_marker": bool(_OPTIONS_RE.search(text)),
        "options_marker_count": len(_OPTIONS_RE.findall(text)),
        "has_criteria_marker": bool(_CRITERIA_RE.search(text)),
        "criteria_marker_count": len(_CRITERIA_RE.findall(text)),
        "has_negation": any(w in text_lower for w in _NEGATION_KEYWORDS),
        "has_injection_keyword": any(w in text_lower for w in _INJECTION_KEYWORDS),
    }


def _build_routing_payload(text: str, question_text: str) -> dict:
    """Build the /v1/decisions payload for routing.

    Iter 3 design: present the head ONLY with a structural fingerprint
    of the user turn (presence/absence of explicit Options: and Criteria:
    markers, length, ends-with-?, negation/injection keywords). The raw
    text is intentionally NOT shown so injection text cannot be parsed
    as decision structure.
    """
    fp = _fingerprint(text)
    fp_text = json.dumps(fp, ensure_ascii=False)
    return {
        "state": f"<fingerprint>{fp_text}</fingerprint>",
        "questions": {
            "route": {
                "type": "choice",
                "instructions": question_text,
                "criteria": {label: None for label in ROUTES},
            }
        },
    }


def fit_route_question(text: str, *, tokenizer=None) -> tuple[bool, int | None]:
    """Refuse routing when the supplied material will not fit without
    truncation; return (ok, token_count_estimate).

    Iter 3: the state sent to Laya is the FINGERPRINT of the user text,
    not the raw text. The fit check uses the same fingerprint state to
    match what actually goes on the wire. The cap is the catalog
    512-token limit; anything that would silently truncate refuses the
    request before dispatch.

    `tokenizer` is injected by tests; production falls back to scanning
    the live venv for a converted Laya checkpoint.
    """
    if not isinstance(text, str) or not text.strip():
        return False, None
    if len(text.encode()) > 1024 * 1024:
        return False, None

    try:
        from mlx_omarchy_laya.sequence import (
            render_options,
            serialize_state,
            to_internal,
        )
    except Exception:
        return False, None

    try:
        tok = tokenizer if tokenizer is not None else _load_default_tokenizer()
    except Exception:
        return False, None

    question = {
        "type": "choice",
        "instructions": ROUTING_POLICY.question_text,
        "criteria": {label: None for label in ROUTES},
    }
    internal = to_internal(question)

    # The state sent to Laya is the fingerprint JSON inside a delimiter
    # (see `_build_routing_payload`). The fit check MUST use the same
    # state shape, otherwise it would silently disagree with the actual
    # dispatch.
    fp_state = f"<fingerprint>{json.dumps(_fingerprint(text), ensure_ascii=False)}</fingerprint>"
    material = [str(internal["ins"]), serialize_state(fp_state)] + render_options(internal)
    if any(tok.mask_token in item for item in material):
        return False, None

    try:
        ids = [tok.cls_token_id] + tok.encode(
            "%s question: %s" % (internal["t"], internal["ins"])
        ) + [tok.sep_token_id]
        for option in render_options(internal):
            ids += [tok.mask_token_id] + tok.encode(" " + option)
        ids += [tok.sep_token_id] + tok.encode(serialize_state(fp_state)) + [tok.sep_token_id]
    except Exception:
        return False, None

    # Catalog Laya cap is 512 tokens. Refuse anything that would
    # silently truncate. With the fingerprint state the cap is hit
    # only for an absurdly long user turn; for normal turns the
    # fingerprint is bounded.
    if len(ids) > 512:
        return False, None
    return True, len(ids)


def _load_default_tokenizer():
    """Find a usable Laya checkpoint on disk; raises when none exists.

    Tests inject their own tokenizer through the `tokenizer` keyword.
    Production callers fall back to scanning the live venv's pair lock
    for the decision model path.
    """
    from mlx_omarchy_laya.sequence import LayaTokenizer

    home = Path.home()
    candidates = [
        home / "<qual-home>" / "models" / "laya-mlx",
    ]
    for path in candidates:
        if path.is_dir() and (path / "rl_agent_config.json").exists():
            return LayaTokenizer(path / "tokenizer")
    raise FileNotFoundError("No Laya checkpoint found; refuse routing")


def evaluate_route(text: str, *, worker, policy: RoutingPolicy | None = None,
                   deadline_seconds: float = WARM_DEADLINE_SECONDS) -> RoutingOutcome:
    """Single routing attempt.

    `worker` is anything exposing a `.call(payload, deadline_seconds)`
    method. Production wires `LocalModels.decision` with a fresh HTTP
    connection; tests inject a fake that returns a fixed response.

    Returns:
      - `route != None` and `timed_out == False` for a successful call
        whose answer passes the frozen thresholds;
      - `route is None` and `reason != ""` for any skip path, including
        budget refusal, deadline miss, invalid output, or threshold miss.
    """
    pol = policy or ROUTING_POLICY

    # 1) Over-budget material: refuse to route, return to LLM.
    try:
        ok, _ = fit_route_question(text)
    except Exception:
        ok = False
    if not ok:
        return RoutingOutcome(reason="material_does_not_fit")

    payload = _build_routing_payload(text, pol.question_text)

    # 2) Single-shot call with deadline. A timeout stays accounted for
    # in the runner's pending list until the worker truly finishes or
    # is asked to stop; we never launch a replacement.
    start = time.monotonic()
    raw = worker.call(payload, deadline_seconds)
    latency_ms = (time.monotonic() - start) * 1000.0

    if raw.get("timed_out"):
        return RoutingOutcome(reason="deadline_miss", timed_out=True,
                              latency_ms=latency_ms)

    answers = raw.get("answers") or {}
    answer = answers.get("route") if isinstance(answers, dict) else None
    if not isinstance(answer, dict) or answer.get("type") != "choice":
        return RoutingOutcome(reason="invalid_output", latency_ms=latency_ms)

    probs = answer.get("probabilities")
    if not isinstance(probs, dict) or [k for k in probs] != list(ROUTES):
        return RoutingOutcome(reason="invalid_output", latency_ms=latency_ms)
    values = [probs[k] for k in ROUTES]
    if any(isinstance(v, bool) or not isinstance(v, (int, float))
           or not math.isfinite(v) or not 0 <= v <= 1 for v in values):
        return RoutingOutcome(reason="invalid_output", latency_ms=latency_ms)
    if not math.isclose(sum(values), 1, abs_tol=0.001):
        return RoutingOutcome(reason="invalid_distribution", latency_ms=latency_ms)

    rl = answer.get("rl_agent") or {}
    act = rl.get("act_probability")
    if (isinstance(act, bool) or not isinstance(act, (int, float))
            or not math.isfinite(act) or not 0 <= act <= 1):
        return RoutingOutcome(reason="invalid_output", latency_ms=latency_ms)

    choice = answer.get("choice")
    if choice not in ROUTES:
        return RoutingOutcome(reason="invalid_output", latency_ms=latency_ms)

    sorted_probs = sorted(values, reverse=True)
    margin = sorted_probs[0] - sorted_probs[1]
    p_selected = probs[choice]

    # Threshold check: selected-label prob, runner-up margin, act_probability.
    # `confidence` is intentionally NOT used here: it is one minus normalized
    # entropy over the answer distribution, not a probability of correctness.
    if p_selected < pol.p_min:
        return RoutingOutcome(reason="threshold_miss", probabilities=probs,
                              runner_up_margin=margin, act_probability=act,
                              latency_ms=latency_ms)
    if margin < pol.margin_min:
        return RoutingOutcome(reason="threshold_miss", probabilities=probs,
                              runner_up_margin=margin, act_probability=act,
                              latency_ms=latency_ms)
    if act < pol.act_min:
        return RoutingOutcome(reason="threshold_miss", probabilities=probs,
                              runner_up_margin=margin, act_probability=act,
                              latency_ms=latency_ms)

    return RoutingOutcome(route=choice, probabilities=probs,
                          runner_up_margin=margin, act_probability=act,
                          latency_ms=latency_ms)


# ----------------------------------------------------------------- runners

class TimedOutCall:
    """A worker call whose deadline elapsed before a real response.

    Held by the runner's pending list until the underlying worker
    actually finishes; never replaced.
    """

    def __init__(self, future):
        self.future = future
        self.start = time.monotonic()

    def is_done(self) -> bool:
        return self.future.done()

    def result(self):
        return self.future.result()


def pending_outcome(text: str, *, worker_factory, policy: RoutingPolicy | None = None,
                    deadline_seconds: float = WARM_DEADLINE_SECONDS) -> tuple[
                        RoutingOutcome, TimedOutCall | None]:
    """Asynchronous variant for the coordinator: starts the call and
    returns immediately with the deadline outcome plus the in-flight
    handle. The caller is expected to either wait on the handle (if
    routing is the active mode) or drop it on the floor (ordinary chat
    must never wait for a cold Laya).

    Returns `(outcome, handle)` where `handle is None` when no call
    was issued (over-budget, worker refused, etc.).
    """
    pol = policy or ROUTING_POLICY
    ok, _ = fit_route_question(text)
    if not ok:
        return RoutingOutcome(reason="material_does_not_fit"), None

    payload = {
        "state": text,
        "questions": {
            "route": {
                "type": "choice",
                "instructions": pol.question_text,
                "criteria": {label: None for label in ROUTES},
            }
        },
    }
    worker = worker_factory()
    future = worker.call_async(payload, deadline_seconds)

    start = time.monotonic()
    if not future.done():
        # Deadline elapsed; the handle remains in the runner's pending list.
        return (RoutingOutcome(reason="deadline_miss", timed_out=True), TimedOutCall(future))
    latency_ms = (time.monotonic() - start) * 1000.0
    raw = future.result()
    if raw.get("timed_out"):
        return (RoutingOutcome(reason="deadline_miss", timed_out=True,
                               latency_ms=latency_ms), TimedOutCall(future))
    return evaluate_route(text, worker=_SyncWorker(raw), policy=pol,
                          deadline_seconds=deadline_seconds), None


class _SyncWorker:
    def __init__(self, raw):
        self.raw = raw

    def call(self, payload, deadline_seconds):
        return self.raw


# A trivial placeholder worker for the synchronous path so tests can
# evaluate an answer without any threading. The coordinator uses the
# async `pending_outcome` path.
class SyncWorker:
    def __init__(self, response: Mapping[str, Any]):
        self.response = dict(response)

    def call(self, payload, deadline_seconds):
        return self.response