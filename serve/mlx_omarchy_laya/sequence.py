"""Jev-style question rendering and sequence building.

Verbatim port of the pinned upstream rl_common.py rendering path
(render_options / build_sequence / serialize_state / collate_items) from
convaiinnovations/laya revision 1c5edc17a7acd8701df6fc341c0d179f1c62c982,
using the `tokenizers` library directly on the checkpoint's tokenizer.json
(PreTrainedTokenizerFast is a wrapper around the same engine). No
trust_remote_code, no pickle: tokenizer.json is plain JSON data.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

QTYPES = {"choice": 0, "score": 1, "noul": 2}
QTYPE_NAMES = {v: k for k, v in QTYPES.items()}


class LayaTokenizer:
    """Minimal fast-tokenizer wrapper exposing the pieces build_sequence needs."""

    def __init__(self, tokenizer_dir: str | Path):
        from tokenizers import Tokenizer

        tokenizer_dir = Path(tokenizer_dir)
        self._tok = Tokenizer.from_file(str(tokenizer_dir / "tokenizer.json"))
        with open(tokenizer_dir / "tokenizer_config.json") as f:
            cfg = json.load(f)
        self.cls_token = cfg["cls_token"]
        self.sep_token = cfg["sep_token"]
        self.mask_token = cfg["mask_token"]
        self.pad_token = cfg["pad_token"]
        self.cls_token_id = self._tok.token_to_id(self.cls_token)
        self.sep_token_id = self._tok.token_to_id(self.sep_token)
        self.mask_token_id = self._tok.token_to_id(self.mask_token)
        self.pad_token_id = self._tok.token_to_id(self.pad_token)

    def encode(self, text: str) -> list:
        return self._tok.encode(text, add_special_tokens=False).ids


def serialize_state(state) -> str:
    if isinstance(state, str):
        return state
    return json.dumps(state, ensure_ascii=False)


def to_internal(qdef: dict) -> dict:
    """Upstream RLAgent._to_internal."""
    t = qdef["type"]
    crit = qdef.get("criteria")
    if t == "choice" and isinstance(crit, list):
        crit = {c: None for c in crit}
    ins = qdef["instructions"]
    if not isinstance(ins, str):
        ins = json.dumps(ins)
    return {"t": t, "ins": ins, "crit": crit}


def render_options(q: dict) -> list:
    """Option texts in label-index order. Noul is always [false, true] so p[1] == noul."""
    t, crit = q["t"], q.get("crit")
    if t == "choice":
        return [k if not v else "%s: %s" % (k, v) for k, v in crit.items()]
    if t == "score":
        return ["level %d: %s" % (i, c) for i, c in enumerate(crit)]
    crit = crit or {}
    return [
        "false: " + (crit.get("false") or "no, the statement does not hold"),
        "true: " + (crit.get("true") or "yes, the statement holds"),
    ]


def build_sequence(tok: LayaTokenizer, state, q: dict, max_len: int, head_max_len: int,
                   option_order=None, truncate_left: bool = False):
    """[CLS] <type> instructions [SEP] [MASK] opt0 [MASK] opt1 ... [SEP] state [SEP].

    Returns (input_ids, marker_positions). Verbatim upstream logic.
    """
    mask_tok = tok.mask_token
    opts = render_options(q)
    order = option_order if option_order is not None else list(range(len(opts)))
    ins = str(q["ins"]).replace(mask_tok, " ")
    head_ids = tok.encode("%s question: %s" % (q["t"], ins))
    opt_ids = []
    for i in order:
        opt_ids.append([tok.mask_token_id] + tok.encode(" " + opts[i].replace(mask_tok, " "))[:48])
    opt_budget = head_max_len - sum(len(o) for o in opt_ids)
    if opt_budget < 16:  # too many / too long options: shrink every option text evenly
        per = max(4, (head_max_len - 16) // max(1, len(opt_ids)))
        opt_ids = [o[:per] for o in opt_ids]
        opt_budget = head_max_len - sum(len(o) for o in opt_ids)
    head_ids = head_ids[: max(8, opt_budget)]
    ids = [tok.cls_token_id] + head_ids + [tok.sep_token_id]
    markers = []
    for o in opt_ids:
        markers.append(len(ids))
        ids.extend(o)
    ids.append(tok.sep_token_id)
    room = max(0, max_len - len(ids) - 1)
    st = tok.encode(serialize_state(state).replace(mask_tok, " "))
    st = st[-room:] if truncate_left else st[:room]
    ids = ids + st + [tok.sep_token_id]
    return ids[:max_len], [m for m in markers if m < max_len]


def encode_questions(tok, state, questions: dict, max_len: int, head_max_len: int):
    """questions: {id: {type, instructions, criteria}} -> (ids_in_order, items).

    Mirrors upstream system_one: raises ValueError when a question's options do
    not fit in head_max_len.
    """
    ids_in_order = list(questions.keys())
    items = []
    for qid in ids_in_order:
        q = to_internal(questions[qid])
        seq, markers = build_sequence(tok, state, q, max_len, head_max_len)
        if len(markers) != len(render_options(q)):
            raise ValueError(
                "question %r: options do not fit in head_max_len=%d tokens" % (qid, head_max_len)
            )
        items.append(
            {
                "ids": seq,
                "markers": markers,
                "qtype": QTYPES[q["t"]],
                "t": q["t"],
                "crit": q["crit"],
            }
        )
    return ids_in_order, items


def collate(items, pad_id: int):
    """Upstream collate_items restricted to the fields inference needs -> numpy."""
    n, L = len(items), max(len(it["ids"]) for it in items)
    kmax = max(len(it["markers"]) for it in items)
    ids = np.full((n, L), pad_id, dtype=np.int64)
    att = np.zeros((n, L), dtype=np.int64)
    mpos = np.zeros((n, kmax), dtype=np.int64)
    mmask = np.zeros((n, kmax), dtype=bool)
    for i, it in enumerate(items):
        ids[i, : len(it["ids"])] = it["ids"]
        att[i, : len(it["ids"])] = 1
        k = len(it["markers"])
        mpos[i, :k] = it["markers"]
        mmask[i, :k] = True
    return {
        "input_ids": ids,
        "attention_mask": att,
        "marker_pos": mpos,
        "marker_mask": mmask,
        "qtype": np.array([it["qtype"] for it in items], dtype=np.int64),
        "n_tokens": int(att.sum()),
    }


def temp_bucket(qtype: int, k: int) -> str:
    """Per-cardinality temperature key: a 2-option noul and a 20-option choice need different scaling."""
    size = "2" if k <= 2 else "3-5" if k <= 5 else "6-10" if k <= 10 else "11+"
    return "%s:%s" % (QTYPE_NAMES[int(qtype)], size)


def confidence_from_probs(p: np.ndarray, k: int) -> float:
    """Jev-style confidence: 1 - normalized entropy of the answer distribution."""
    import math

    if k < 2:
        return 1.0
    p = p[:k]
    ent = -(p * np.log(np.clip(p, 1e-12, 1))).sum()
    return float(1 - ent / math.log(k))
