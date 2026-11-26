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

import functools
import math
import re
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
    """Versioned policy record: question text, thresholds, evidence pointer.

    Stage order: injection guard -> structure extractor -> deterministic
    decision -> Laya head (grey turns only). A grey turn (>= 2 options
    without an explicit label plus criteria) becomes `structured_decision`
    only when the head's act_probability >= act_min, p(structured_decision)
    >= sd_min, and the head does not say conversation with
    p(conversation) >= conversation_veto. Fewer than 2 options never
    produce `structured_decision`.
    """

    version: str
    question_text: str
    act_min: float
    sd_min: float
    conversation_veto: float
    negation_removes: bool
    suite_sha256: str | None = None
    receipt: str | None = None
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "question_text": self.question_text,
            "act_min": self.act_min,
            "sd_min": self.sd_min,
            "conversation_veto": self.conversation_veto,
            "negation_removes": self.negation_removes,
            "suite_sha256": self.suite_sha256,
            "receipt": self.receipt,
            "note": self.note,
        }


ROUTING_POLICY = RoutingPolicy(
    version="3",
    question_text=(
        "Classify this user turn as exactly one of: conversation (the "
        "user wants an explanation or chat, no bounded choice), "
        "structured_decision (the user supplies explicit alternatives to "
        "pick from, even when phrased as a question), or clarify (the "
        "user's request is missing options, criteria, or scope and the "
        "assistant needs to ask before deciding)."
    ),
    # Set from the dev sweep before any held-out contact (see receipt).
    # sd_min: most conservative value on the max-recall plateau. act_min:
    # act_probability was 1.0 on every dev record, so dev cannot tune it;
    # 0.5 is the codebase's abstention line.
    act_min=0.50,
    sd_min=0.30,
    conversation_veto=0.60,
    negation_removes=False,
    suite_sha256=None,
    receipt=None,
    note="policy '3': injection guard + structure extractor + deterministic stage + Laya head",
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
    head_called: bool = False
    options: tuple = ()
    criteria: str = ""


# ---------------------------------------------------------------- injection guard
#
# A deterministic guard runs in front of the Laya choice head. When the
# guard matches, the turn is diverted to `conversation` (route = conversation,
# reason = "injection_guard") BEFORE the head sees it. The head only
# classifies benign turns.
#
# Patterns are drawn from the public prompt-injection taxonomy
# (Greshake 2023, Perez & Ribeiro 2022, Branch 2024, OWASP LLM01:2025):
#   - verb x target: ignore/disregard/forget/drop/override/bypass + previous/
#     prior/above/all/your + instructions/rules/prompt/context/policy
#   - role-override prefix: "system:" / "system override" / "system prompt"
#   - persona-override: "you are now ..." / "pretend ..."
#   - output-shape override: "output only ..." / "respond in JSON only"
#   - tool-call / shell injection: "run <command>" / "execute ..."
# We use word-boundary regex, case-insensitive, with synonyms.
# DO NOT COPY the dev-set injection templates: the held-out suite uses
# different wording. Patterns are general.


_STRICT_INJECTION_VERBS = (
    "ignore", "disregard", "drop", "override", "bypass",
    "clear", "erase", "wipe", "delete",
)
# "forget" / "skip" / "overwrite" require an explicit meta-noun to fire —
# they appear in benign contexts (forget the budget, skip the warm-up).
_FORGET_LIKE_VERBS = ("forget", "skip", "overwrite")
# Targets that signal an override attempt when combined with a strict
# verb: "everything", "all", "this" are abstract pronouns (forget
# everything = override attempt; forget the budget = benign).
_TARGETS = (
    "previous", "prior", "above", "all", "your",
    "earlier", "above-mentioned", "system",
)
# Abstract override-target pronouns — pairing these with a strict verb
# is an injection attempt on its own (no meta-noun needed). "Forget
# everything above" -> injection. "Forget the earlier budget" -> no
# match because "earlier" is in _TARGETS but "budget" is not abstract.
_ABSTRACT_OVERRIDE_TARGETS = (
    "everything", "all", "this", "that", "every", "any",
)
# Meta-nouns: these are the words that signal "the user is talking about
# the prompt / policy / instructions themselves".
_META_NOUNS = (
    "instructions", "instruction", "rules", "rule", "prompt", "prompts",
    "context", "directives", "directive",
    "guidelines", "guideline", "constraints", "constraint",
    "system",
)

_STRICT_VERB_RE = re.compile(r"\b(?:" + "|".join(_STRICT_INJECTION_VERBS) + r")\b",
                              re.IGNORECASE)
_FORGET_RE = re.compile(r"\b(?:" + "|".join(_FORGET_LIKE_VERBS) + r")\b",
                        re.IGNORECASE)
_META_NOUN_RE = re.compile(r"\b(?:" + "|".join(_META_NOUNS) + r")\b",
                            re.IGNORECASE)
_TARGET_RE = re.compile(r"\b(?:" + "|".join(_TARGETS) + r")\b", re.IGNORECASE)
_ABSTRACT_TARGET_RE = re.compile(
    r"\b(?:" + "|".join(_ABSTRACT_OVERRIDE_TARGETS) + r")\b", re.IGNORECASE)

# Role-override prefix
_SYSTEM_PREFIX_RE = re.compile(
    r"(?:^|\n)\s*(?:system\s*(?:override|prompt|message|role|note)?"
    r"|assistant\s*:|user\s*:\s*system)\s*[:\-]",
    re.IGNORECASE,
)
_PERSONA_OVERRIDE_RE = re.compile(
    r"\byou\s+are\s+now\s+(?:an?\s+)?(?:unrestricted|new|different|evil|"
    r"jailbroken|developer|root|admin|hacker|free)\b",
    re.IGNORECASE,
)
_PRETEND_RE = re.compile(
    r"\bpretend\s+(?:to\s+be|you\s+are|that|it's)\b", re.IGNORECASE,
)
_OUTPUT_SHAPE_RE = re.compile(
    r"\b(?:output|respond|reply|answer)\s+(?:only|with|in|using)\s+"
    r"(?:json|the\s+string|the\s+word|one\s+word|a\s+single)",
    re.IGNORECASE,
)
_SHELL_INJECTION_RE = re.compile(
    r"\b(?:run|execute|invoke|call)\s+(?:rm|wget|curl|chmod|chown|"
    r"sudo|bash|sh|python|powershell)\b",
    re.IGNORECASE,
)
# "forget the earlier budget" is a benign phrasal — see tests for near-miss set


def _is_injection(text: str) -> bool:
    """True iff the user turn carries prompt-injection phrasing.

    Patterns are general (drawn from the public prompt-injection
    taxonomy: Greshake 2023, Perez & Ribeiro 2022, Branch 2024,
    OWASP LLM01:2025) and intentionally not copied from any test
    suite. Returns False on benign text; the head then classifies as
    normal.

    Three families of pattern, sentence-scoped:
      1. Strong verb + meta-noun in the same sentence (ignore the
         instructions, override the policy, clear the guidelines).
         The strong verbs are ignore/disregard/drop/override/bypass/
         clear/erase/wipe/delete.
      2. Forget-like verb (forget/skip/overwrite) + meta-noun in the
         same sentence. Without a meta-noun these appear in benign
         contexts (forget the budget, skip the warm-up).
      3. Role-override prefix / persona-override / output-shape /
         shell-injection (each is sentence-independent).
    """
    if not isinstance(text, str) or not text:
        return False
    if _SYSTEM_PREFIX_RE.search(text):
        return True
    if _PERSONA_OVERRIDE_RE.search(text):
        return True
    if _PRETEND_RE.search(text):
        return True
    if _OUTPUT_SHAPE_RE.search(text):
        return True
    if _SHELL_INJECTION_RE.search(text):
        return True
    # Sentence-scoped verb + meta-noun check.
    for sentence in re.split(r"[.\n!?]+", text):
        if not sentence.strip():
            continue
        has_strict_verb = bool(_STRICT_VERB_RE.search(sentence))
        has_forget_verb = bool(_FORGET_RE.search(sentence))
        has_meta_noun = bool(_META_NOUN_RE.search(sentence))
        if has_meta_noun and (has_strict_verb or has_forget_verb):
            return True
        # Abstract override-target pronoun + strict verb: injection even
        # without a meta-noun ("Forget everything above", "Ignore all").
        # forget-like verbs are included here too because "forget
        # everything" is a clear override attempt.
        if (has_strict_verb or has_forget_verb) and _ABSTRACT_TARGET_RE.search(sentence):
            return True
    return False


# ---------------------------------------------------------------- structure extractor
#
# Deterministic extractor that runs BETWEEN the injection guard and
# the Laya choice head. It extracts candidate options from explicit
# labels ("Options:", "Choices:", "Alternatives:") and from generic
# A/B / X vs Y / "either ... or ..." patterns, then handles negation
# ("do not pick X" removes X). Used by `evaluate_route` to override
# the head's choice when the extractor is more reliable than the
# head — particularly the "looks like decision but no options
# supplied" case the head gets wrong.

_OPTION_LABEL_RE = re.compile(
    r"\b(?:options?|choices?|alternatives?)\s*:\s*",
    re.IGNORECASE,
)
_CRITERIA_LABEL_RE = re.compile(
    r"\b(?:criteri[ao]|based\s+on|prioritis\w*|ranking)\s*[:\s]+",
    re.IGNORECASE,
)
# Generic two-item comparison patterns: "A or B", "A vs B",
# "either A or B", "whether A or B".
_OR_RE = re.compile(
    r"\b(?:either|whether|choose\s+between|pick\s+between|select\s+between)?\s*"
    r"([A-Za-z][\w\s\-]{0,40}?)\s+(?:or|vs\.?|versus)\s+([A-Za-z][\w\s\-]{0,40}?)\b",
    re.IGNORECASE,
)
# "between A and B" is a choice only after a choice verb ("choose between",
# "deciding between", "torn between"); "the difference between A and B"
# asks for an explanation.
_BETWEEN_RE = re.compile(
    r"\b(?:choos\w*|pick\w*|decid\w*|select\w*|compar\w*|torn)\s+"
    r"(?:[\w\-']+\s+){0,3}?between\s+([\w\-' ]{1,40}?)\s+and\s+([\w\-' ]{1,40}?)"
    r"(?=\s*(?:[.,;:?!]|$)|\s+(?:for|to|on|in|at|this|when)\b)",
    re.IGNORECASE,
)
# Numbered/lettered list: "1. foo  2. bar" or "(a) foo (b) bar" or
# "- foo\n- bar".
_NUMBERED_RE = re.compile(
    r"(?:^|\n)\s*(?:\d+[.)]|\(\d+\)|[a-z][.)]|\(\w\)|[-*])\s+\S+",
    re.IGNORECASE | re.MULTILINE,
)
_NEGATION_RE = re.compile(
    r"\b(?:do\s+not|don'?t|never|avoid|skip|not)\s+(?:pick|choose|select|"
    r"the)?\s*(?:the\s+)?(.+?)(?:\.|,|;|$)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class _Extracted:
    options: tuple
    has_explicit_label: bool
    criteria: str
    negated: tuple


def _split_options(chunk: str) -> list:
    parts = []
    for piece in re.split(r"\s*(?:,|;|/|\bor\b|\n)\s*", chunk):
        p = piece.strip().strip("\"'`").strip()
        if p and p not in ("and", "or"):
            parts.append(p)
    return parts


def _extract_structure(text: str) -> "_Extracted":
    """Deterministic structure extractor (grammar, not templates).

    Sources of candidate options, in order:
      1. An explicit label ("Options:", "Choices:", "Alternatives:"): the
         items up to the end of that sentence, split on , ; / or newline.
      2. Unlabelled "A or B" / "A vs B" / "either A or B" with each side a
         short noun phrase (<= 4 words). A bare single character on either
         side is a placeholder variable ("A or B?"), not a supplied option.
      3. A numbered, lettered or bulleted list of >= 2 lines.
    Negation ("do not pick X", "never choose X", "avoid X") is recorded;
    the matching option moves to `negated`.
    """
    if not isinstance(text, str) or not text:
        return _Extracted(options=(), has_explicit_label=False,
                          criteria="", negated=())

    options = []
    has_label = bool(_OPTION_LABEL_RE.search(text))
    for m in _OPTION_LABEL_RE.finditer(text):
        tail = text[m.end():]
        stop = re.search(r"\n|\.(?:\s|$)", tail)
        chunk = tail[:stop.start()] if stop else tail
        options.extend(_split_options(chunk.strip()))

    if not has_label:
        for m in list(_OR_RE.finditer(text)) + list(_BETWEEN_RE.finditer(text)):
            a = " ".join(m.group(1).split()[-4:])
            b = " ".join(m.group(2).split()[:4])
            if len(a) > 1 and len(b) > 1:
                options.extend([a, b])
        bullets = [re.sub(r"^\s*(?:\d+[.)]|\(\d+\)|[a-z][.)]|\(\w\)|[-*])\s+",
                           "", m.group(0)).strip()
                   for m in _NUMBERED_RE.finditer(text)]
        if len(bullets) >= 2:
            options.extend(bullets)

    seen = set()
    deduped = []
    for o in options:
        if o.lower() not in seen:
            seen.add(o.lower())
            deduped.append(o)

    negated = []
    for m in _NEGATION_RE.finditer(text):
        rest = m.group(1).strip().lower()
        if not rest:
            continue
        for o in deduped:
            low = o.lower()
            if rest.startswith(low) or low.startswith(rest.split()[0]):
                negated.append(o)
                break
    remaining = tuple(o for o in deduped if o not in negated)

    criteria = ""
    m = _CRITERIA_LABEL_RE.search(text)
    if m:
        tail = text[m.end():]
        stop = re.search(r"\n|\.(?:\s|$)", tail)
        criteria = (tail[:stop.start()] if stop else tail).strip()

    return _Extracted(options=remaining, has_explicit_label=has_label,
                      criteria=criteria, negated=tuple(negated))


# Words that ask for a choice. Present without >= 2 options -> clarify.
_DECISION_WORDING_RE = re.compile(
    r"\b(?:which|choose|choosing|pick|picking|decide|deciding|select|recommend"
    r"|compare|prefer|versus|vs|should\s+(?:i|we)|better|best)\b",
    re.IGNORECASE,
)


def _candidate_options(extracted: "_Extracted", policy: "RoutingPolicy") -> tuple:
    if policy.negation_removes:
        return extracted.options
    return extracted.options + extracted.negated


def deterministic_route(text: str, extracted: "_Extracted", policy: "RoutingPolicy"):
    """Decide without the head when the structure is unambiguous.

    Returns (route, reason), or None when the head must be consulted:
      - explicit option label, >= 2 options, and criteria -> structured_decision
      - < 2 options and decision wording -> clarify (never a decision)
      - < 2 options and no decision wording -> chat model (route None)
      - otherwise (grammatical options without a label, or a label
        without criteria) -> None, ask the head.
    """
    options = _candidate_options(extracted, policy)
    if len(options) < 2:
        if _DECISION_WORDING_RE.search(text):
            return "clarify", "missing_options"
        return None, "no_decision_structure"
    if extracted.has_explicit_label and extracted.criteria:
        return "structured_decision", "explicit_structure"
    return None


def decide_route(answer: Mapping[str, Any], policy: "RoutingPolicy") -> tuple:
    """Route a grey turn (>= 2 candidate options) from a validated head answer.

    Returns (route, reason); route None means "use the chat model". Pure,
    so recorded head answers replay without a GPU.
    """
    probs = answer["probabilities"]
    if answer["choice"] == "conversation" and probs["conversation"] >= policy.conversation_veto:
        return "conversation", "head_veto"
    if (answer["rl_agent"]["act_probability"] < policy.act_min
            or probs["structured_decision"] < policy.sd_min):
        return None, "threshold_miss"
    return "structured_decision", "extractor_and_head"


def _build_routing_payload(text: str, question_text: str) -> dict:
    """Build the /v1/decisions payload for routing (iter-1 path).

    The user turn is presented to the head as raw state. The injection
    guard runs in front of the head and diverts injection text before
    this point; the head never sees injection phrasing here.
    """
    return {
        "state": text,
        "questions": {
            "route": {
                "type": "choice",
                "instructions": question_text,
                "criteria": {label: None for label in ROUTES},
            }
        },
    }


def fit_route_question(text: str, *, model_path=None, tokenizer=None) -> tuple[bool, int | None]:
    """Refuse routing when the supplied material will not fit without
    truncation; return (ok, token_count_estimate).

    The state sent to Laya is the raw user text (iter-1 path). The
    fit check uses the same state to match what actually goes on the
    wire. The cap is the catalog 512-token limit; anything that would
    silently truncate refuses the request before dispatch.

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
        tok = tokenizer if tokenizer is not None else _tokenizer_for(str(model_path))
    except Exception:
        return False, None

    question = {
        "type": "choice",
        "instructions": ROUTING_POLICY.question_text,
        "criteria": {label: None for label in ROUTES},
    }
    internal = to_internal(question)

    material = [str(internal["ins"]), serialize_state(text)] + render_options(internal)
    if any(tok.mask_token in item for item in material):
        return False, None

    try:
        ids = [tok.cls_token_id] + tok.encode(
            "%s question: %s" % (internal["t"], internal["ins"])
        ) + [tok.sep_token_id]
        for option in render_options(internal):
            ids += [tok.mask_token_id] + tok.encode(" " + option)
        ids += [tok.sep_token_id] + tok.encode(serialize_state(text)) + [tok.sep_token_id]
    except Exception:
        return False, None

    # Catalog Laya cap is 512 tokens. Refuse anything that would
    # silently truncate.
    if len(ids) > 512:
        return False, None
    return True, len(ids)


@functools.lru_cache(maxsize=4)
def _tokenizer_for(model_path: str):
    """Laya tokenizer of the converted decision model, loaded once per path."""
    from mlx_omarchy_laya.sequence import LayaTokenizer

    return LayaTokenizer(Path(model_path) / "tokenizer")


def evaluate_route(text: str, *, worker, model_path=None, policy: RoutingPolicy | None = None,
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

    # 0) Injection guard: deterministic regex set. When matched, divert
    #    to `conversation` BEFORE the head sees the text. The head only
    #    classifies benign turns. The guard is conservative on near-miss
    #    phrases (see the unit-test near-miss set).
    if _is_injection(text):
        return RoutingOutcome(route="conversation", reason="injection_guard")

    # 1) Deterministic structure: no head call when unambiguous.
    extracted = _extract_structure(text)
    options = _candidate_options(extracted, pol)
    decided = deterministic_route(text, extracted, pol)
    if decided is not None:
        return RoutingOutcome(route=decided[0], reason=decided[1],
                              options=options, criteria=extracted.criteria)

    # 2) Grey case: the head decides. Over-budget material refuses routing.
    try:
        ok, _ = fit_route_question(text, model_path=model_path)
    except Exception:
        ok = False
    if not ok:
        return RoutingOutcome(reason="material_does_not_fit")

    payload = _build_routing_payload(text, pol.question_text)

    # 2) Single-shot call with deadline. A still-running earlier call
    # refuses this one ("busy") rather than queueing a replacement.
    start = time.monotonic()
    raw = worker.call(payload, deadline_seconds)
    latency_ms = (time.monotonic() - start) * 1000.0

    if raw.get("busy"):
        return RoutingOutcome(reason="previous_call_pending", latency_ms=latency_ms)
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

    # `confidence` is intentionally NOT used: it is one minus normalized
    # entropy over the answer distribution, not a probability of correctness.
    route, reason = decide_route(answer, pol)
    return RoutingOutcome(route=route, reason=reason, probabilities=probs,
                          runner_up_margin=margin, act_probability=act,
                          latency_ms=latency_ms, head_called=True,
                          options=options, criteria=extracted.criteria)

