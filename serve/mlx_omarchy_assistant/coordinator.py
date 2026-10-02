"""One conversation across owned local chat and decision workers."""

from __future__ import annotations

import concurrent.futures
import http.client
import json
import math
import re
import threading
import time
import uuid
from pathlib import Path
from urllib.parse import urlsplit

from .history import BusyError, ConversationStore
from .routing import ROUTING_POLICY, RoutingOutcome, evaluate_route
from .speech_yield import SpeechYieldScheduler, YieldClient
from mlx_omarchy_serve._mlxlm_server import YIELD_SECRET_HEADER


class DecisionInputError(ValueError):
    pass


MAX_QUESTIONS = 8
# Task-derived output ceilings; the user-selected max_tokens always wins.
# Auto allowances are backed by the admitted context in _admit_output.
TASK_OUTPUT_ALLOWANCE = {"chat": 2048, "compare": 1024, "decide": 1024, "draft": 1024}
MIN_AUTO_ALLOWANCE = 256
# Ordinary chat sends no card schema: cards come from card_promotion.  A message
# that names a chart, graph, form, decision, options, facts, or sources, every
# Laya turn, and every model whose catalog entry declares fenced-json cards get
# the full schema.
FULL_CARD_CUES = re.compile(r"\b(charts?|graphs?|forms?|decisions?|options?|facts?|sources?)\b", re.IGNORECASE)
REPETITION_PENALTY = 1.1

# Routing gate hook. The flag defaults OFF; only flipping it on after a
# held-out suite passes (recorded in the notebook and pinned on the pair)
# lets `submit(mode="auto")` succeed.
ROUTING_GATE_OFF = "off"
ROUTING_GATE_ON = "on"


def _routing_gate_enabled(manager) -> bool:
    """Whether the live pair record has the routing gate turned ON.

    Reads the manager's currently selected pair record, never a static
    catalog entry: routing approval is per-pair evidence (the suite
    sha256 and the receipt path), not a global flag.
    """
    try:
        status = manager.status()
    except Exception:
        return False
    extension = (status or {}).get("extension") or {}
    evidence = extension.get("selection_evidence") or {}
    routing = evidence.get("routing")
    if not isinstance(routing, dict):
        return False
    return (routing.get("gate") == ROUTING_GATE_ON
            and isinstance(routing.get("suite_sha256"), str)
            and bool(routing.get("suite_sha256").strip())
            and isinstance(routing.get("receipt"), str)
            and bool(routing.get("receipt").strip())
            and routing.get("policy_version") == ROUTING_POLICY.version)

DRAFT_PROMPT = (
    "Extract a comparison draft from the user's request. Reply with ONLY one JSON "
    "object, no prose and no code fence, shaped "
    '{"criteria": "<what should decide the comparison>", "options": '
    '[{"label": "<option>"}]} with 2 to 8 options. Use only options and criteria '
    "the user supplied or clearly implied; invent nothing. If the request names "
    'no usable options, reply {"options": [], "criteria": ""}.'
)


def validate_decision(answer, options):
    keys = [option["id"] for option in options]
    if not isinstance(answer, dict) or answer.get("type") != "choice" or answer.get("choice") not in keys:
        raise DecisionInputError("Decision worker returned an invalid choice")
    probabilities = answer.get("probabilities")
    if not isinstance(probabilities, dict) or list(probabilities) != keys:
        raise DecisionInputError("Decision worker changed the option identifiers or order")
    values = list(probabilities.values())
    act = answer.get("rl_agent", {}).get("act_probability")
    confidence = answer.get("confidence")
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 <= v <= 1
           for v in values + [act, confidence]):
        raise DecisionInputError("Decision worker returned invalid probabilities")
    if not math.isclose(sum(values), 1, abs_tol=0.001):
        raise DecisionInputError("Decision probabilities do not sum to one")
    return dict(answer, abstained=act < 0.5, options=options)



_CLAIM_WORDS = ("choose", "chose", "chosen", "select", "selected", "pick", "picked")



def format_decision_event(payload):
    """Terminal text for a coordinator decision event. Empty when unusable."""
    if not isinstance(payload, dict):
        return ""
    if payload.get("type") == "typed":
        lines = ["Decisions:"]
        for result in payload.get("results") or []:
            if not isinstance(result, dict):
                continue
            answer = result.get("answer") or {}
            instructions = result.get("instructions") or result.get("id") or "Question"
            if result.get("type") == "choice":
                lines.append("%s: %s" % (instructions, answer.get("choice", "")))
                if answer.get("abstained"):
                    lines.append("The decision model abstained.")
            elif result.get("type") == "score":
                lines.append("%s: %s" % (instructions, answer.get("score", "")))
        return "\n".join(lines)
    if payload.get("type") != "choice" and "choice" not in payload:
        return ""
    lines = ["Decision:"]
    selected = payload.get("choice")
    for option in payload.get("options") or []:
        if not isinstance(option, dict):
            continue
        mark = "*" if option.get("id") == selected else " "
        lines.append("%s %s" % (mark, option.get("label", "")))
    if payload.get("criteria"):
        lines.append("Criteria: " + str(payload["criteria"]))
    if payload.get("abstained"):
        lines.append("The decision model abstained.")
    if payload.get("model"):
        lines.append("Decided by: " + str(payload["model"]))
    return "\n".join(lines)


def explanation_disagrees(decision, text):
    """True when an explanation claims one other option and omits the Laya choice.

    Conservative on purpose: mentioning the selected label, listing several
    options, or abstaining does not count. The Laya result is never rewritten.
    """
    if not isinstance(decision, dict) or decision.get("abstained"):
        return False
    options = decision.get("options")
    choice = decision.get("choice")
    if not isinstance(options, list) or not isinstance(choice, str) or not isinstance(text, str):
        return False
    labels = {}
    for option in options:
        if not isinstance(option, dict):
            return False
        ident, label = option.get("id"), option.get("label")
        if not isinstance(ident, str) or not isinstance(label, str) or not label.strip():
            return False
        labels[ident] = label
    selected = labels.get(choice)
    if not selected:
        return False
    lowered = text.lower()
    if selected.lower() in lowered:
        return False
    others = [label for ident, label in labels.items()
              if ident != choice and label.lower() in lowered]
    if len(others) != 1:
        return False
    words = set(lowered.replace(".", " ").replace(",", " ").split())
    return any(word in words for word in _CLAIM_WORDS)


def _check_labels(value, low, high, cap, what):
    if not isinstance(value, list) or not low <= len(value) <= high:
        raise DecisionInputError("%s: supply %d to %d entries" % (what, low, high))
    labels = []
    for entry in value:
        if not isinstance(entry, str) or not entry.strip() or len(entry) > cap:
            raise DecisionInputError("%s: entries must be non-empty text of at most %d characters"
                                     % (what, cap))
        if entry in labels:
            raise DecisionInputError("%s: entries must be distinct" % what)
        labels.append(entry)
    return labels


# ---------------------------------------------------------------------------
# Card-format capability lookup (lazy, cached, never raises)
# ---------------------------------------------------------------------------

_FENCED_JSON_CACHE: dict[str, bool] = {}


def _model_emits_fenced_cards(chat_model_id: str) -> bool:
    """True when the catalog entry declares ``extension.card_format ==
    "fenced-json"`` (a measured fence capability).  Cached; a missing or
    broken catalog means no capability."""
    if chat_model_id not in _FENCED_JSON_CACHE:
        fenced = False
        try:
            from mlx_omarchy_serve import catalog as _catalog
            for entry in _catalog.load_catalog().get("models") or ():
                if entry.get("id") == chat_model_id:
                    fenced = (entry.get("extension") or {}).get("card_format") == "fenced-json"
                    break
        except Exception:
            fenced = False
        _FENCED_JSON_CACHE[chat_model_id] = fenced
    return _FENCED_JSON_CACHE[chat_model_id]


def _typed_request(model_path, text, questions):
    """Build one batched /v1/decisions payload for Laya-shaped questions.

    Every question is fit-checked against the real tokenizer and renderer:
    the complete token input is reconstructed by hand and compared with the
    build_sequence output, so any renderer shortening (option labels,
    instructions, state) or material overflow refuses the whole request
    before dispatch. Nothing is ever truncated.
    """
    from mlx_omarchy_laya.sequence import LayaTokenizer, build_sequence, render_options, serialize_state, to_internal

    if not isinstance(text, str) or not text.strip():
        raise DecisionInputError("Supply the material to decide on")
    if len(text.encode()) > 1024 * 1024:
        raise DecisionInputError("The decision material exceeds the 1 MiB limit")
    if not isinstance(questions, dict) or not questions or len(questions) > MAX_QUESTIONS:
        raise DecisionInputError("Batch at most %d typed questions" % MAX_QUESTIONS)
    path = Path(model_path)
    config = json.loads((path / "rl_agent_config.json").read_text())
    tok = LayaTokenizer(path / "tokenizer")
    for qid, question in questions.items():
        if not isinstance(qid, str) or not qid.strip() or len(qid) > 64:
            raise DecisionInputError("Question IDs must be non-empty and at most 64 characters")
        internal = to_internal(question)
        material = [str(internal["ins"]), serialize_state(text)] + render_options(internal)
        if any(tok.mask_token in item for item in material):
            raise DecisionInputError("The decision input contains a reserved mask token; use the chat model")
        full = [tok.cls_token_id] + tok.encode("%s question: %s" % (internal["t"], internal["ins"])) + [tok.sep_token_id]
        for option in render_options(internal):
            full += [tok.mask_token_id] + tok.encode(" " + option)
        full += [tok.sep_token_id] + tok.encode(serialize_state(text)) + [tok.sep_token_id]
        actual, _ = build_sequence(tok, text, internal, config["max_len"], config["head_max_len"])
        if full != actual or len(full) > config["max_len"]:
            raise DecisionInputError("The complete decision input does not fit without truncation. Use the chat model")
    return {"state": text, "questions": questions}


def decision_request(model_path, text, options, criteria):
    if not isinstance(options, list) or not 2 <= len(options) <= 8:
        raise DecisionInputError("Compare between two and eight explicit options")
    if not isinstance(criteria, str) or not criteria.strip():
        raise DecisionInputError("Supply the criteria for this comparison")
    identifiers = set()
    for option in options:
        if not isinstance(option, dict) or set(option) != {"id", "label"}:
            raise DecisionInputError("Each option needs an ID and label")
        if not all(isinstance(option[key], str) and option[key].strip() for key in ("id", "label")):
            raise DecisionInputError("Option IDs and labels cannot be empty")
        if len(option["id"]) > 64 or option["id"] in identifiers:
            raise DecisionInputError("Option IDs must be distinct and at most 64 characters")
        identifiers.add(option["id"])
    question = {"type": "choice", "instructions": criteria,
                "criteria": {option["id"]: option["label"] for option in options}}
    return _typed_request(model_path, text, {"comparison": question})


def typed_questions_request(model_path, text, spec):
    """Validate the typed question list (classification and score) and build
    one batched decision request; all questions share a single forward pass."""
    if not isinstance(spec, list) or not 1 <= len(spec) <= MAX_QUESTIONS:
        raise DecisionInputError("Send between one and %d typed questions" % MAX_QUESTIONS)
    questions = {}
    for item in spec:
        if not isinstance(item, dict) or set(item) - {"id", "type", "instructions", "options", "levels"}:
            raise DecisionInputError("Each question needs id, type, instructions and options or levels")
        qid = item.get("id")
        if not isinstance(qid, str) or not qid.strip() or len(qid) > 64 or qid.strip() in questions:
            raise DecisionInputError("Question IDs must be distinct, non-empty, at most 64 characters")
        qid = qid.strip()
        kind = item.get("type")
        instructions = item.get("instructions")
        if kind not in ("choice", "score"):
            raise DecisionInputError("Question types are choice (classification) and score")
        if not isinstance(instructions, str) or not instructions.strip() or len(instructions) > 512:
            raise DecisionInputError("Each question needs instructions of at most 512 characters")
        if kind == "choice":
            labels = _check_labels(item.get("options"), 2, 8, 200, "Classification options")
            question = {"type": "choice", "instructions": instructions,
                        "criteria": {label: None for label in labels}}
        else:
            levels = _check_labels(item.get("levels"), 2, 8, 64, "Score levels")
            question = {"type": "score", "instructions": instructions, "criteria": levels}
        questions[qid] = question
    return _typed_request(model_path, text, questions)


def validate_score_answer(answer, levels):
    if not isinstance(answer, dict) or answer.get("type") != "score":
        raise DecisionInputError("Decision worker returned an invalid score")
    score = answer.get("score")
    if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
        raise DecisionInputError("Decision worker returned an invalid score")
    keys = [str(index) for index in range(len(levels))]
    legend = answer.get("legend")
    if not isinstance(legend, dict) or [legend.get(key) for key in keys] != list(levels):
        raise DecisionInputError("Decision worker changed the score levels")
    probabilities = answer.get("probabilities")
    if not isinstance(probabilities, dict) or list(probabilities) != keys:
        raise DecisionInputError("Decision worker changed the score distribution")
    values = list(probabilities.values())
    act = answer.get("rl_agent", {}).get("act_probability")
    confidence = answer.get("confidence")
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 <= v <= 1
           for v in values + [act, confidence]):
        raise DecisionInputError("Decision worker returned invalid probabilities")
    if not math.isclose(sum(values), 1, abs_tol=0.001):
        raise DecisionInputError("Score probabilities do not sum to one")
    return dict(answer, abstained=act < 0.5, levels=list(levels))


def parse_draft(text):
    """Parse and validate an LLM comparison draft; raises when unusable.

    This is a schema check on real model output, never extraction: a draft
    that does not validate is refused and the user keeps the manual panel.
    """
    raw = text.strip() if isinstance(text, str) else ""
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("The draft contained no JSON object")
    draft = json.loads(raw[start:end + 1])
    if not isinstance(draft, dict) or set(draft) - {"criteria", "options"}:
        raise ValueError("The draft must contain exactly criteria and options")
    criteria, options = draft.get("criteria"), draft.get("options")
    if not isinstance(criteria, str) or not criteria.strip() or len(criteria) > 500:
        raise ValueError("The draft criteria must be text of at most 500 characters")
    if not isinstance(options, list) or not 2 <= len(options) <= 8:
        raise ValueError("The draft must hold between two and eight options")
    labels = []
    for option in options:
        if not isinstance(option, dict) or set(option) - {"label"} \
                or not isinstance(option.get("label"), str) or not option["label"].strip() \
                or len(option["label"]) > 200:
            raise ValueError("Draft option labels must be text of 1-200 characters")
        label = option["label"].strip()
        if any(item["label"].casefold() == label.casefold() for item in labels):
            raise ValueError("Draft option labels must be distinct")
        labels.append({"label": label})
    return {"criteria": criteria.strip(), "options": labels}


def typed_results(spec, answers, text):
    """Validate every typed answer and bundle the decision-card payload.

    spec was already schema-validated by typed_questions_request; answers
    come from the decision worker and are re-validated here against the
    requested labels and levels. Nothing unvalidated reaches the UI.
    """
    results = []
    for item in spec:
        qid = item["id"].strip()
        answer = answers.get(qid)
        if item["type"] == "choice":
            labels = _check_labels(item.get("options"), 2, 8, 200, "Classification options")
            validated = validate_decision(answer, [{"id": label, "label": label} for label in labels])
            results.append({"id": qid, "type": "choice",
                            "instructions": item["instructions"].strip(),
                            "options": labels, "answer": validated})
        else:
            levels = _check_labels(item.get("levels"), 2, 8, 64, "Score levels")
            validated = validate_score_answer(answer, levels)
            results.append({"id": qid, "type": "score",
                            "instructions": item["instructions"].strip(),
                            "levels": levels, "answer": validated})
    return {"type": "typed", "model": "laya-mlx", "input_scope": text, "results": results}


class LocalModels:
    def __init__(self, manager):
        self.manager = manager
        self.connection = None
        self.lock = threading.Lock()
        self.tokenizer = None
        self.tokenizer_path = None

    def _connect(self, base):
        parsed = urlsplit(base)
        if parsed.scheme != "http" or parsed.hostname not in ("127.0.0.1", "::1", "localhost"):
            raise ValueError("Model workers must use local loopback HTTP")
        connection = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=90)
        with self.lock:
            self.connection = connection
        return connection

    def close_connection(self):
        with self.lock:
            if self.connection:
                if self.connection.sock:
                    import socket
                    try:
                        self.connection.sock.shutdown(socket.SHUT_RDWR)
                    except OSError:
                        pass
                self.connection.close()
                self.connection = None

    def count(self, path, messages):
        if self.tokenizer_path != path:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
            self.tokenizer_path = path
        encoded = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                     enable_thinking=False)
        # transformers >= 5 returns a BatchEncoding here (measured: len()==2
        # — input_ids + attention_mask), older ones a flat id list; a batched
        # shape wraps ids in one more list. len() of the container silently
        # admitted every turn at 2 prompt tokens.
        if not isinstance(encoded, list):
            encoded = encoded["input_ids"]
        if encoded and isinstance(encoded[0], list):
            encoded = encoded[0]
        return len(encoded)

    def decision(self, pair, payload):
        connection = self._connect(pair["decision_url"])
        try:
            connection.request("POST", "/v1/decisions", json.dumps(payload), {"Content-Type": "application/json"})
            response = connection.getresponse()
            raw = response.read(1024 * 1024 + 1)
            if len(raw) > 1024 * 1024:
                raise ValueError("Decision response is too large")
            if response.status != 200:
                raise RuntimeError("Decision worker refused the request (HTTP %d): %s"
                                   % (response.status, raw[:500].decode("utf-8", "replace")))
            return json.loads(raw)
        finally:
            self.close_connection()

    def chat(self, pair, messages, max_tokens, cancel, phases=None, yield_headers=None):
        connection = self._connect(pair["chat_url"])
        model = str((pair.get("model_paths") or {}).get("chat") or pair["chat_model"])
        # Greedy decoding loops on the 2B model ("Extra socks" until the token cap);
        # a mild penalty removed the loops in the 2026-09-28 card experiment.
        payload = {"model": model, "messages": messages, "max_tokens": max_tokens,
                   "stream": True, "temperature": 0, "repetition_penalty": REPETITION_PENALTY,
                   "chat_template_kwargs": {"enable_thinking": False}}
        finish = None
        try:
            headers = {"Content-Type": "application/json"}
            if yield_headers:
                headers.update(yield_headers)
            connection.request("POST", "/v1/chat/completions", json.dumps(payload),
                               headers)
            if phases is not None:
                phases["chat_sent"] = time.monotonic()
            response = connection.getresponse()
            if phases is not None:
                phases["chat_headers"] = time.monotonic()
            if response.status != 200:
                detail = response.read(8192).decode("utf-8", "replace")
                raise RuntimeError("Chat worker refused the request (HTTP %d): %s"
                                   % (response.status, detail[:500]))
            while not cancel.is_set():
                line = response.readline(1024 * 1024 + 1)
                if not line:
                    raise RuntimeError("Chat worker disconnected before completing the response")
                if len(line) > 1024 * 1024:
                    raise ValueError("Chat worker event is too large")
                if not line.startswith(b"data:"):
                    continue
                data = line[5:].strip()
                if data == b"[DONE]":
                    break
                event = json.loads(data)
                if event.get("error"):
                    raise RuntimeError("Chat worker returned an inference error")
                for choice in event.get("choices", []):
                    reason = choice.get("finish_reason")
                    if reason:
                        finish = reason
                    text = choice.get("delta", {}).get("content", "")
                    if text:
                        if not isinstance(text, str):
                            raise ValueError("Chat worker returned invalid text")
                        yield ("delta", text)
            if not cancel.is_set():
                yield ("finish", finish)
        finally:
            self.close_connection()


class _RoutingWorker:
    """Deadline-bounded routing calls with at most one call in flight.

    Uses its own LocalModels (its own connection), so a late routing call
    can never close a chat stream. A call that misses the deadline keeps
    running on the single pool thread and stays recorded in `inflight`;
    until it finishes, new routing attempts are refused (`busy`) instead
    of queueing a replacement.
    """

    def __init__(self, manager):
        self.models = LocalModels(manager)
        self.pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self.lock = threading.Lock()
        self.inflight = None

    def bind(self, pair):
        worker = self

        class Bound:
            def call(self, payload, deadline_seconds):
                return worker.call(pair, payload, deadline_seconds)

        return Bound()

    def call(self, pair, payload, deadline_seconds):
        with self.lock:
            if self.inflight is not None and not self.inflight.done():
                return {"busy": True}
            self.inflight = self.pool.submit(self.models.decision, pair, payload)
            future = self.inflight
        try:
            return future.result(timeout=deadline_seconds)
        except concurrent.futures.TimeoutError:
            return {"timed_out": True}
        except Exception as error:
            return {"error": str(error)[:200]}

    def close(self):
        self.models.close_connection()
        self.pool.shutdown(wait=True, cancel_futures=True)


class Coordinator:
    def __init__(self, home, manager, store=None):
        self.home = Path(home)
        self.store = store if store is not None else ConversationStore(home)
        self.manager = manager
        self.models = LocalModels(manager)
        self.router = _RoutingWorker(manager)
        self.gpu = threading.Lock()
        self.speech = SpeechYieldScheduler(self.gpu)
        self.lock = threading.RLock()
        self.turns = {}
        self.actions = set()
        self.closed = threading.Event()
        self.failure = None
        self.watch = threading.Thread(target=self._watch, daemon=True)
        self.watch.start()

    def submit(self, cid, payload):
        if self.failure:
            raise RuntimeError(self.failure)
        if self.closed.is_set():
            raise RuntimeError("The assistant is shutting down")
        if set(payload) - {"text", "mode", "options", "criteria", "questions",
                           "max_tokens", "routing_evidence"}:
            raise ValueError("Unknown turn field")
        mode = payload.get("mode", "chat")
        if mode not in ("chat", "compare", "decide", "draft", "auto"):
            raise ValueError("Unknown turn mode")
        if mode == "auto" and not _routing_gate_enabled(self.manager):
            raise ValueError(
                "Automatic routing stays disabled until the held-out routing suite passes. "
                "Use explicit Compare options or change the mode to chat/decide."
            )
        maximum = payload.get("max_tokens")
        if maximum is not None and (isinstance(maximum, bool) or not isinstance(maximum, int)
                                    or not 1 <= maximum <= 262144):
            raise ValueError("Output allowance must be a positive token count within the model limit")
        if mode == "decide" and (not isinstance(payload.get("questions"), list)
                                 or not 1 <= len(payload["questions"]) <= MAX_QUESTIONS):
            raise ValueError("Send between one and %d typed questions" % MAX_QUESTIONS)
        # Automatic routing runs BEFORE the GPU is acquired: a warm 250 ms
        # call is the only thing that can run synchronously; cold Laya or
        # an over-budget material falls back to chat immediately, without
        # blocking ordinary chat or holding the GPU.
        if mode == "auto":
            payload = self._resolve_auto_route(cid, payload)
        if not self.gpu.acquire(blocking=False):
            raise BusyError("Another local response is active. Wait or stop it before sending")
        try:
            turn = self.store.begin(cid, payload.get("text"))
            job = {"cancel": threading.Event(), "heartbeat": time.monotonic(), "conversation_id": cid,
                   "ttft": {"phases": {"submit": time.monotonic()}}}
            with self.lock:
                self.turns[turn] = job
            thread = threading.Thread(target=self._run, args=(cid, turn, payload, maximum, job), daemon=True)
            job["thread"] = thread
            thread.start()
            return turn
        except BaseException:
            self.gpu.release()
            raise

    def _resolve_auto_route(self, cid, payload):
        """Run the routing call (warm 250 ms) BEFORE acquiring the GPU.

        Records the routing decision in the conversation so the UI shows
        the same 'How this was decided' details, even when the answer is
        `clarify` or the router skipped entirely. Always returns a
        payload whose `mode` is one of `chat`, `compare`, or `decide` so
        the GPU-bound worker only sees the worker modes it knows.

        A `structured_decision` route compares only the options the user
        wrote (from the explicit payload or the structure extractor); it
        needs criteria too, otherwise the turn becomes `clarify`.
        """
        try:
            pair = self.manager.start()
        except Exception as error:
            # Pair start refused; ordinary chat still has to work.
            routing = {"policy_version": ROUTING_POLICY.version,
                       "route": None, "reason": "pair_start_failed",
                       "model": "laya-mlx", "use_chat_model_available": True,
                       "probabilities": {}, "runner_up_margin": None,
                       "act_probability": None, "latency_ms": None,
                       "timed_out": False, "error": str(error)[:200]}
            new_payload = dict(payload)
            new_payload["mode"] = "chat"
            new_payload["routing"] = routing
            return new_payload
        routing = self._auto_route(pair, payload["text"])
        new_payload = dict(payload)
        new_payload["routing"] = routing
        new_payload["mode"] = "chat"
        if routing.get("route") == "structured_decision":
            options = payload.get("options")
            criteria = payload.get("criteria")
            if not (isinstance(options, list) and 2 <= len(options) <= 8):
                labels = routing.get("options") or []
                options = [{"id": "option-%d" % (i + 1), "label": label}
                           for i, label in enumerate(labels)]
                criteria = routing.get("criteria")
            if not (2 <= len(options) <= 8 and isinstance(criteria, str) and criteria.strip()):
                routing["route"] = "clarify"
                routing["reason"] = "missing_options_or_criteria"
                return new_payload
            try:
                decision_request(pair["model_paths"]["decision"], payload["text"], options, criteria)
            except DecisionInputError:
                routing["route"] = None
                routing["reason"] = "material_does_not_fit"
                return new_payload
            new_payload.update(mode="compare", options=options, criteria=criteria)
        return new_payload

    def _ttft_dump(self, cid, turn, mode, phases):
        """Append one line of per-phase TTFT timestamps to the assistant logs.

        Phases hold time.monotonic() seconds (comparable across the assistant
        and worker processes on one host); "prompt_tokens" is a token count.
        Logging must never break a turn."""
        try:
            logs = self.home / "assistant" / "logs"
            logs.mkdir(parents=True, exist_ok=True)
            record = {"turn": turn, "conversation": cid, "mode": mode, "phases": phases}
            with open(logs / "ttft-phases.jsonl", "a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, allow_nan=False) + "\n")
        except (OSError, ValueError):
            pass

    def _run(self, cid, turn, payload, maximum, job):
        cancel = job["cancel"]
        ttft = job.get("ttft")
        phases = ttft["phases"] if ttft else None

        def phase(name):
            if phases is not None:
                phases[name] = time.monotonic()
        failed = False
        speech_client = None
        speech_secret = None
        speech_capable = False
        try:
            from .components import SCHEMA_PROMPT, validate_components
            pair = self.manager.start()
            phase("pair_start")
            if cancel.is_set():
                return
            record = self.store.get(cid)
            refused = record.get("context_error")
            if refused:
                raise ValueError("Saved history selection is invalid; inference is blocked until it is repaired: %s"
                                 % refused)
            mode = payload.get("mode", "chat")
            result = None
            if mode == "draft":
                self._run_draft(cid, turn, pair, payload, maximum, cancel)
                return
            # Emit the recorded routing decision (if any) so the UI can
            # show how the auto-route was chosen.
            if "routing" in payload:
                self.store.emit(cid, turn, "routing", payload["routing"])
            user_text = payload.get("text") or ""
            full_schema = (mode in ("compare", "decide")
                           or bool(FULL_CARD_CUES.search(user_text))
                           or _model_emits_fenced_cards(pair.get("chat_model") or ""))
            messages = [{"role": "system", "content": "Answer the user using their supplied facts. "
                         + (SCHEMA_PROMPT if full_schema else "")}]
            messages.extend(self._selected_history(record, turn))
            phase("prompt_built")
            if mode in ("compare", "decide"):
                path = pair["model_paths"]["decision"]
                if mode == "compare":
                    request = decision_request(path, payload["text"], payload.get("options"), payload.get("criteria"))
                    raw = self.models.decision(pair, request)
                    result = validate_decision(raw.get("answers", {}).get("comparison"), payload["options"])
                    result.update({"criteria": payload["criteria"], "model": "laya-mlx",
                                   "input_scope": payload["text"]})
                else:
                    request = typed_questions_request(path, payload["text"], payload.get("questions"))
                    raw = self.models.decision(pair, request)
                    result = typed_results(payload.get("questions"), raw.get("answers", {}), payload["text"])
                self.store.emit(cid, turn, "decision", result)
                phase("decision_done")
                note = ("Explain this supplied Laya result without changing its choice. "
                        if mode == "compare" else
                        "Explain these supplied Laya results without changing any of them. ")
                messages[0]["content"] += ("\n" + note
                                           + "If abstained, say that it abstained. Its confidence is not factual accuracy. "
                                           + json.dumps(result, ensure_ascii=False))
            allowance, required = self._admit_output(pair, messages, maximum, mode,
                                                     phases=phases)
            phase("admit_done")
            pair = self.manager.status()
            self.store.emit(cid, turn, "status", {"state": "generating", "context_tokens": required,
                                                 "context_limit": pair["context_tokens"],
                                                 "output_tokens": allowance})
            phase("generating_status_emitted")
            buffer = ""
            component_buffer = None
            component_count = 0
            repaired = False
            finish_reason = None
            marker = "```assistant-ui\n"
            # Mirror every chunk the user sees so card_promotion can derive
            # a card from the exact text the UI rendered.  We never
            # promote when the model already emitted a valid fenced block.
            reply_text_parts: list[str] = []
            raw_text_parts: list[str] = []  # every chunk the model emitted,
            #    kept even after a fence starts so an invalid/truncated fence
            #    can still surface the model's own output for the user
            #    (PairGates card-defect-compact4b, 2026-10-01).
            saw_visible_text = False      # did any text event reach the UI?
            saw_visible_component = False  # did any validated component reach the UI?
            # Cooperative TTS scheduling: probe the chat worker's yield gate
            # and, when present, park generation at chunk boundaries so
            # queued speak requests can synthesize between chunks. Without
            # the gate (standalone or foreign worker) speaks stay busy —
            # never a fake pause.
            try:
                speech_client = YieldClient(pair["chat_url"])
                speech_secret = uuid.uuid4().hex
                speech_capable = speech_client.probe(speech_secret)
            except (OSError, ValueError):
                speech_client = None
                speech_capable = False
            phase("yield_probe_done")
            stream = self.models.chat(
                pair, messages, allowance, cancel, phases=phases,
                yield_headers=({YIELD_SECRET_HEADER: speech_secret}
                               if speech_capable else None))
            first_chunk_seen = False
            dumped_ttft = phases is None
            try:
                while True:
                    if speech_capable:
                        self.speech.yield_point(speech_client, speech_secret,
                                                cancel)
                    try:
                        event = next(stream)
                    except StopIteration:
                        break
                    if not first_chunk_seen:
                        first_chunk_seen = True
                        phase("first_chunk")
                    if event[0] == "finish":
                        finish_reason = event[1]
                        continue
                    text = event[1]
                    raw_text_parts.append(text)
                    if cancel.is_set():
                        break
                    buffer += text
                    while buffer:
                        if component_buffer is not None:
                            end = buffer.find("```")
                            if end < 0:
                                if len(buffer.encode()) > 65536:
                                    raise ValueError("Generated interface exceeds the 64 KiB limit")
                                break
                            envelope, buffer = buffer[:end], buffer[end + 3:]
                            component_buffer = None
                            components = self._validated(envelope, component_count, turn)
                            if components is None and not repaired and not cancel.is_set():
                                repaired = True
                                components = self._repair_components(pair, envelope, allowance,
                                                                     cancel, component_count)
                            if components:
                                for component in components:
                                    component_count += 1
                                    trusted = dict(component, id=uuid.uuid4().hex, turn_id=turn, revision=1)
                                    self.store.emit(cid, turn, "component", trusted)
                                    saw_visible_component = True
                            elif not cancel.is_set():
                                self.store.emit(cid, turn, "status", {"state": "invalid_component",
                                    "message": "The generated interface was invalid."})
                                self._emit_raw_fallback(cid, turn, raw_text_parts, marker)
                            continue
                        start = buffer.find(marker)
                        if start >= 0:
                            if start:
                                self.store.emit(cid, turn, "text", {"text": buffer[:start]})
                                reply_text_parts.append(buffer[:start])
                                saw_visible_text = True
                            buffer = buffer[start + len(marker):]
                            component_buffer = ""
                            continue
                        keep = 0
                        for length in range(1, min(len(buffer), len(marker) - 1) + 1):
                            if buffer.endswith(marker[:length]):
                                keep = length
                        ready = buffer[:-keep] if keep else buffer
                        if ready:
                            self.store.emit(cid, turn, "text", {"text": ready})
                            reply_text_parts.append(ready)
                            saw_visible_text = True
                            if not dumped_ttft:
                                dumped_ttft = True
                                phase("first_text_emitted")
                                self._ttft_dump(cid, turn, mode, phases)
                        buffer = buffer[-keep:] if keep else ""
                        break
            finally:
                stream.close()
            # Finish rules (card-defect fix, 2026-10-01): anything still buffered
            # when the stream ends must reach the user as text.
            if buffer and component_buffer is None:
                self.store.emit(cid, turn, "text", {"text": buffer})
                reply_text_parts.append(buffer)
                saw_visible_text = True
            if component_buffer is not None:
                # Unterminated fence: repair once or surface the raw text.
                if not repaired and not cancel.is_set():
                    repaired = True
                    components = self._repair_components(pair, buffer, allowance,
                                                         cancel, component_count)
                    if components:
                        for component in components:
                            component_count += 1
                            trusted = dict(component, id=uuid.uuid4().hex, turn_id=turn, revision=1)
                            self.store.emit(cid, turn, "component", trusted)
                            saw_visible_component = True
                        component_buffer = None
                if component_buffer is not None:
                    self.store.emit(cid, turn, "status", {"state": "invalid_component",
                        "message": "The interface was incomplete; the model output is shown below."})
                    self._emit_raw_fallback(cid, turn, raw_text_parts, marker)
            if finish_reason == "length" and not cancel.is_set():
                # The model ran out of tokens mid-reply.  A truncated fence is
                # as bad as an invalid one from the user's point of view; show
                # the raw text and offer to continue.
                if component_buffer is not None:
                    self._emit_raw_fallback(cid, turn, raw_text_parts, marker)
                self.store.emit(cid, turn, "status", {"state": "output_truncated",
                    "message": "The response reached the output allowance. Send Continue to keep going.",
                    "continue": True})
            # Card promotion: when the model emitted no valid fenced block,
            # try to derive one card from the rendered markdown.  Only chat
            # turns are eligible (compare/decide/draft produce structured
            # payloads of their own).  Hostile, oversize, or empty replies
            # return None from extract_text and emit nothing.
            if (not cancel.is_set() and mode == "chat" and component_count == 0):
                from .card_promotion import extract_text as _promote_card
                reply_text = "".join(reply_text_parts)
                promoted = _promote_card(reply_text, user_text)
                if promoted is not None:
                    try:
                        validated = validate_components(
                            {"version": 1, "components": [promoted]})
                    except Exception:
                        validated = None
                    if validated:
                        for component in validated:
                            component_count += 1
                            trusted = dict(component, id=uuid.uuid4().hex,
                                           turn_id=turn, revision=1)
                            self.store.emit(cid, turn, "component", trusted)
                            saw_visible_component = True
            # Card-defect catch-all: if the turn ended with no visible text
            # and no validated component, emit a short explanation.  Without
            # this the user sees an empty reply (12d8d9c0b).
            if (not saw_visible_text and not saw_visible_component
                    and not cancel.is_set() and not failed):
                msg = ("The model produced no readable reply this turn "
                       "(no text and no rendered interface).")
                self.store.emit(cid, turn, "text", {"text": msg})
                self.store.emit(cid, turn, "status",
                                 {"state": "empty_reply",
                                    "message": "Turn completed without a visible reply."})

            if (mode in ("compare", "decide") and not cancel.is_set()
                    and explanation_disagrees(result,
                                              self.store.get(cid)["messages"][-1].get("content", ""))):
                notice = "The explanation disagrees with the Laya result. The Laya result is unchanged."
                self.store.emit(cid, turn, "text", {"text": "\n" + notice})
                self.store.emit(cid, turn, "status", {"state": "disagreement"})
        except Exception as error:
            failed = True
            if not cancel.is_set():
                self.store.emit(cid, turn, "error", {"code": type(error).__name__, "message": str(error)[:1000]})
        finally:
            # Serve queued speak requests one last time while the workers
            # still live; when the gate is gone (or the worker already died)
            # refuse them promptly instead of leaving the queue to time out.
            try:
                if speech_capable:
                    try:
                        self.speech.yield_point(speech_client, speech_secret,
                                                threading.Event())
                    except Exception:
                        self.speech.refuse_pending()
                else:
                    self.speech.refuse_pending()
            except Exception:
                pass
            try:
                if cancel.is_set() or failed:
                    try:
                        report = self.manager.stop()
                        if not report.get("stopped"):
                            raise RuntimeError("Unverified worker exits: " + str(report.get("retained", [])))
                    except Exception as error:
                        self.failure = "Worker shutdown failed; restart the assistant: " + str(error)
                        self.store.emit(cid, turn, "error", {"code": "WorkerShutdown", "message": self.failure})
                self.store.finish(cid, turn, stopped=cancel.is_set() or failed)
            finally:
                with self.lock:
                    self.turns.pop(turn, None)
                self.gpu.release()

    def _selected_history(self, record, turn):
        """Chat messages for the admitted history, per the context contract.

        record["context"]["selected_turn_ids"] selects whole completed turn
        pairs: null means all, [] means none, a list means only those turn
        IDs (unknown IDs are ignored). The current turn is always included.
        pinned_constraints ride in their own system message; nothing here
        summarizes or drops user-pinned text.
        """
        context = record.get("context")
        context = context if isinstance(context, dict) else {}
        selection = context.get("selected_turn_ids")
        pinned = context.get("pinned_constraints")
        messages = []
        if isinstance(pinned, str) and pinned.strip():
            messages.append({"role": "system", "content":
                             "The user pinned these constraints; keep them satisfied and visible: "
                             + pinned.strip()})
        groups, order = {}, []
        for message in record["messages"]:
            turn_id = message.get("turn_id")
            if not turn_id:
                continue
            if turn_id not in groups:
                groups[turn_id] = []
                order.append(turn_id)
            groups[turn_id].append(message)
        for turn_id in order:
            group = groups[turn_id]
            if any(message.get("status") != "complete" for message in group):
                continue
            if isinstance(selection, list) and turn_id not in selection:
                continue
            messages.extend({"role": message["role"], "content": message["content"]}
                            for message in group
                            if message.get("role") in ("user", "assistant") and message.get("content"))
        if record["messages"]:
            latest = groups.get(turn) or []
            for message in latest:
                if message.get("role") == "user" and message.get("content"):
                    messages.append({"role": "user", "content": message["content"]})
        return messages

    def _admit_output(self, pair, messages, maximum, mode, phases=None):
        """Token-count the complete prompt and admit prompt + allowance.

        An explicit max_tokens is a promise: honored exactly, or the turn
        refuses by name. Without one the allowance comes from the task and
        is backed by the admitted context — when admission refuses growth,
        the allowance shrinks to what the already-admitted context holds,
        never below MIN_AUTO_ALLOWANCE, and the status event reports it.
        Returns (allowance, required prompt+allowance token count).
        """
        prompt = self.models.count(pair["model_paths"]["chat"], messages)
        if phases is not None:
            phases["count_done"] = time.monotonic()
            phases["prompt_tokens"] = prompt
        ceiling = maximum if maximum is not None else TASK_OUTPUT_ALLOWANCE.get(mode, TASK_OUTPUT_ALLOWANCE["chat"])
        context = self.manager.ensure_context(prompt + ceiling)
        if context.get("ok"):
            return ceiling, prompt + ceiling
        if maximum is not None:
            raise ValueError(
                "The requested output allowance of %d tokens does not fit: %s. "
                "Lower the allowance or shorten the selected history; it was not reduced silently."
                % (maximum, context.get("reason", "the admitted context limit")))
        admitted = context.get("context_tokens")
        if isinstance(admitted, int) and admitted - prompt >= MIN_AUTO_ALLOWANCE:
            return admitted - prompt, admitted
        raise ValueError("The conversation does not fit the admitted context (%s); "
                         "free memory or reduce the selected history"
                         % (context.get("reason", "limit reached"),))

    def _run_draft(self, cid, turn, pair, payload, maximum, cancel):
        """Natural-language comparison draft: one real chat call plus at most
        one bounded repair, then an editable confirmation in the UI. The draft
        is validated model output; a refusal is reported, never regex-extracted
        and never presented as user-supplied facts."""
        from .components import ENVELOPE_MAX_BYTES
        blob = payload["text"].encode("utf-8", "replace")[:ENVELOPE_MAX_BYTES].decode("utf-8", "replace")
        messages = [{"role": "system", "content": DRAFT_PROMPT},
                    {"role": "user", "content": blob}]
        allowance, _required = self._admit_output(pair, messages, maximum, "draft")
        text = ""
        for kind, chunk in self.models.chat(self.manager.status(), messages, allowance, cancel):
            if kind == "finish":
                break
            text += chunk
            if len(text.encode()) > ENVELOPE_MAX_BYTES + 4096:
                break
        draft = None
        try:
            draft = parse_draft(text)
        except (ValueError, TypeError):
            if not cancel.is_set():
                draft = self._repair_draft(pair, text, allowance, cancel)
        if cancel.is_set():
            return
        if draft is None:
            self.store.emit(cid, turn, "status", {
                "state": "draft_unavailable",
                "message": "The model could not produce a valid options draft. "
                           "Enter the options yourself in the Compare options panel."})
            return
        listing = "\n".join(["Draft ready. Review it in the Compare options panel before scoring:"]
                            + ["- " + option["label"] for option in draft["options"]]
                            + ["Criteria: " + draft["criteria"]])
        self.store.emit(cid, turn, "text", {"text": listing})
        self.store.emit(cid, turn, "status", {"state": "comparison_draft",
                                             "options": draft["options"],
                                             "criteria": draft["criteria"],
                                             "source": payload["text"]})

    def _repair_draft(self, pair, rejected, allowance, cancel):
        """One bounded repair attempt for an invalid draft; None on failure.
        Admission is re-checked first so a refusal keeps the error honest."""
        from .components import ENVELOPE_MAX_BYTES
        blob = rejected.encode("utf-8", "replace")[:ENVELOPE_MAX_BYTES].decode("utf-8", "replace")
        messages = [{"role": "system", "content": DRAFT_PROMPT
                     + "\nThe previous reply was not a valid draft JSON object. "
                       "Return ONLY the JSON object."},
                    {"role": "user", "content": blob}]
        budget = min(allowance, 2048)
        context = self.manager.ensure_context(
            self.models.count(pair["model_paths"]["chat"], messages) + budget)
        if not context.get("ok"):
            return None
        text = ""
        for kind, chunk in self.models.chat(self.manager.status(), messages, budget, cancel):
            if kind == "finish":
                break
            text += chunk
            if len(text.encode()) > ENVELOPE_MAX_BYTES + 4096:
                break
        try:
            return parse_draft(text)
        except (ValueError, TypeError):
            return None

    def _auto_route(self, pair, text):
        """One automatic routing call with a 250 ms warm deadline.

        Returns the routing event the UI shows, including skips: a skip
        is a real answer (use the chat model), not a hidden failure.
        Deadline misses and in-flight accounting live in `_RoutingWorker`.
        """
        outcome = evaluate_route(
            text,
            worker=self.router.bind(pair),
            model_path=pair["model_paths"]["decision"],
            policy=ROUTING_POLICY,
            deadline_seconds=0.250,
        )
        return {
            "policy_version": ROUTING_POLICY.version,
            "route": outcome.route,
            "reason": outcome.reason,
            "probabilities": outcome.probabilities,
            "runner_up_margin": outcome.runner_up_margin,
            "act_probability": outcome.act_probability,
            "latency_ms": outcome.latency_ms,
            "timed_out": outcome.timed_out,
            "options": list(outcome.options),
            "criteria": outcome.criteria,
            "use_chat_model_available": True,
            "model": "laya-mlx",
        }

    def _validated(self, envelope, component_count, turn):
        """Validate one assistant-ui envelope; returns components or None.

        Never executes model output: JSON parse plus schema validation only.
        A rejected envelope is never rendered; the caller reports it.
        """
        from .components import validate_components
        try:
            components = validate_components(json.loads(envelope))
            if component_count + len(components) > 32:
                raise ValueError("Too many generated components")
            return components
        except (ValueError, TypeError, KeyError):
            return None

    def _emit_raw_fallback(self, cid, turn, raw_text_parts, marker):
        """Surface the model's own output as text when its interface failed.

        Append the raw chunks to the message, fenced in a code block so the
        UI cannot interpret stray ``` as a real card.  Skipped if the stream
        is empty.  Updates ``saw_visible_text`` so the catch-all guard knows
        something already reached the user.
        """
        raw = "".join(raw_text_parts)
        if not raw:
            return False
        block = (
            "The model's reply could not be rendered as a card. "
            "Here is its raw output:\n\n```\n" + raw + "\n```")
        self.store.emit(cid, turn, "text", {"text": block})
        return True


    def _repair_components(self, pair, envelope, allowance, cancel,
                           component_count=0):
        """One bounded repair chat call for an invalid assistant-ui block.

        Budget: the caller allows at most one repair per turn, tokens are
        capped at 2048, and the input carries the schema plus only the
        rejected envelope (64 KiB cap), never the conversation.  The repair
        is token-counted with the actual tokenizer and admitted through
        ensure_context first; a refusal returns None so the good prose and
        the invalid_component status survive untouched.  The output is never
        rendered without passing validate_components again.
        """
        from .components import (validate_components, ENVELOPE_MAX_BYTES,
                                 SCHEMA_PROMPT)
        blob = envelope.encode("utf-8", "replace")[:ENVELOPE_MAX_BYTES]
        # Chat templates such as Qwen 3.5's refuse a conversation without a
        # user turn, so the rejected block travels as the user message.
        messages = [{"role": "system", "content":
                     SCHEMA_PROMPT
                     + "\nThe previous ```assistant-ui block failed "
                       "validation. Return ONLY one corrected "
                       "```assistant-ui fenced block with the same intent "
                       "and no prose."},
                    {"role": "user", "content": "Rejected block:\n"
                     + blob.decode("utf-8", "replace")}]
        required = (self.models.count(pair["model_paths"]["chat"], messages)
                    + min(allowance, 2048))
        context = self.manager.ensure_context(required)
        if not context.get("ok"):
            return None
        pair = self.manager.status()
        text = ""
        for kind, payload in self.models.chat(pair, messages,
                                              min(allowance, 2048), cancel):
            if kind == "finish":
                break
            text += payload
            if len(text.encode()) > ENVELOPE_MAX_BYTES + 4096:
                break
        start = text.find("```assistant-ui\n")
        if start < 0:
            return None
        body = text[start + len("```assistant-ui\n"):]
        end = body.find("```")
        if end < 0:
            return None
        try:
            components = validate_components(json.loads(body[:end]))
        except (ValueError, TypeError, KeyError):
            return None
        if component_count + len(components) > 32:
            return None
        return components

    def heartbeat(self, cid, turn):
        with self.lock:
            job = self.turns.get(turn)
            if job and job["conversation_id"] == cid:
                job["heartbeat"] = time.monotonic()

    def cancel(self, cid, turn):
        with self.lock:
            job = self.turns.get(turn)
            if not job or job["conversation_id"] != cid:
                return
            self.store.cancel(cid, turn)
            job["cancel"].set()
        self.models.close_connection()

    def action(self, cid, payload):
        from .components import validate_action
        record = self.store.get(cid)
        if record["active_turn"]:
            raise BusyError("Wait for the current response before submitting this control")
        action_id = payload.get("action_id")
        if not isinstance(action_id, str) or not 1 <= len(action_id) <= 64:
            raise ValueError("Invalid action ID")
        with self.lock:
            key = (cid, action_id)
            if key in self.actions:
                raise ValueError("This action was already submitted")
            component = next((c for m in record["messages"] for c in m.get("components", [])
                              if c["id"] == payload.get("component_id")), None)
            if component is None or component["turn_id"] != payload.get("turn_id") or component["revision"] != payload.get("revision"):
                raise ValueError("This control is stale. Refresh the conversation")
            if record["messages"][-1]["turn_id"] != component["turn_id"]:
                raise ValueError("This control belongs to an earlier turn. Refresh before submitting")
            values = validate_action(component, payload.get("action"), payload.get("values"))
            text = "Submitted interface values: " + json.dumps(values, ensure_ascii=False)
            turn = self.submit(cid, {"text": text})
            self.actions.add(key)
            return turn

    def _watch(self):
        while not self.closed.wait(2):
            with self.lock:
                expired = [(job["conversation_id"], turn) for turn, job in self.turns.items()
                           if time.monotonic() - job["heartbeat"] > 20]
            for cid, turn in expired:
                self.cancel(cid, turn)

    def close(self):
        """Shut down; raises unless every worker stopped and no thread survived."""
        self.closed.set()
        with self.lock:
            jobs = list(self.turns.items())
        for turn, job in jobs:
            self.cancel(job["conversation_id"], turn)
        for turn, job in jobs:
            job["thread"].join(timeout=10)
        survivors = [turn for turn, job in jobs if job["thread"].is_alive()]
        report = self.manager.stop()
        if not report.get("stopped"):
            raise RuntimeError("Worker shutdown unverified: "
                               + str(report.get("retained", [])))
        if survivors:
            raise RuntimeError("Turn threads survived shutdown: "
                               + ", ".join(survivors))
        self.models.close_connection()
        self.router.close()
        self.watch.join(timeout=3)
