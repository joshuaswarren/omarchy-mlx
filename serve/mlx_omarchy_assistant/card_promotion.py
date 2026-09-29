"""Deterministic card promotion: turn a finished chat reply into zero or one card.

Why this exists
---------------
The 2B Everyday model produces rich markdown (checklists, tables, lists) but
ignores the ``assistant-ui`` fenced JSON the schema prompt asks for.  This
module is the post-generation fallback: when the model emits no valid
fenced envelope, the coordinator parses the reply's markdown into one card
if and only if the reply is unambiguous enough to be safely promoted.

Design contract (binding, 2026-09-29)
------------------------------------
1. **Conservative promotion.**  Only an unambiguous reply becomes a card.
   Plain bullet lists, prose, code blocks, and anything inside a fenced
   block (including the ``assistant-ui`` envelope itself) are inert.
2. **Single source of validation.**  Every promoted component passes
   :func:`components.validate_components`.  When the validator would
   reject a candidate (too many rows, missing columns, bad ids, ...) the
   parser either truncates to fit or returns ``None``; it never emits an
   invalid card.
3. **Honest label.**  Cards built here are labelled ``assistant-built`` so
   the UI can mark them as derived from the reply, not model-asserted JSON.
   The card carries the kind in its ``source_kind`` field and a short title
   suffix that names the source (the reply).
4. **No HTML/JS/SVG/CSS**, no remote image URLs, no shell — all data is the
   reply's own markdown text after stripping fenced blocks.  This is
   identical to the model envelope's trust model.
5. **Bounded work.**  A single parse is ``O(n)`` over the reply (linear
   scan, no nested quantifiers, no backtracking-prone regexes).  The parser
   refuses any input over 1 MiB (about 16x the envelope cap) so a hostile
   reply cannot tie the coordinator up.
6. **User intent is honored.**  When the user message names a card kind
   (checklist, table, timeline, chart, facts) the parser allows promoting
   a plain list of >= 3 items into that kind even if the markdown is
   shaped as a bullet list; the intent cue is checked first, structure
   second.

Rule summary (the spec is the code; see ``derive_component``)
-------------------------------------------------------------
- Task-list syntax (``- [ ]`` / ``- [x]``), >= 3 items
  -> ``checklist`` (items carry ``done`` from the checkbox state).
- Markdown pipe table (header row, ``|---|---|`` separator, >= 2 data
  rows, 2..8 columns, all rows have the same column count, cells bounded)
  -> ``comparison`` (column ids auto-generated from header text).
- Ordered or bulleted list where >= 3 items start with a time, weekday,
  date or ``Day N`` marker
  -> ``timeline`` (entries carry the leading marker as ``when``).
- Plain bullet/ordered list (>= 3 items) where the user message names
  ``checklist`` / ``table`` / ``timeline`` / ``chart`` / ``facts``
  -> ``checklist`` / ``comparison`` / ``timeline`` / ``(refused)`` / ``facts``
  respectively.  Comparison requires at least 2 columns (we wrap the
  single-column list as a 1-column comparison is invalid; we instead
  fall through to checklist).

If none of the above match, ``derive_component`` returns ``None`` and the
caller renders the prose only.

What this module does NOT do
----------------------------
- Charts and forms cannot be derived from markdown alone and are not
  promoted here.  When the user asks for them the coordinator still
  sends the full schema so the model can emit the JSON.
- Decisions (a typed Laya result) are not derived from markdown; the
  compare/decide coordinator path remains the source.
- The parser does not invent data.  If a row's value cannot be parsed as
  text or a finite number, it is left as text.

Stdlib only.
"""

from __future__ import annotations

import re
from typing import Iterable

# Bound: refuse anything larger than the model envelope cap (64 KiB) by a
# generous margin so a hostile reply cannot tie the coordinator up.
MAX_INPUT_BYTES = 1 * 1024 * 1024  # 1 MiB
# Hard caps that mirror components.validate_components (kept in sync here
# so we truncate before calling the validator, never after).
MAX_CHECKLIST_ITEMS = 50
MIN_CHECKLIST_ITEMS = 3
MAX_COMPARISON_COLUMNS = 8
MIN_COMPARISON_COLUMNS = 2
MAX_COMPARISON_ROWS = 32
MIN_COMPARISON_ROWS = 2
MAX_TIMELINE_ENTRIES = 50
MIN_TIMELINE_ENTRIES = 3
MAX_FACTS = 24
MIN_FACTS = 1
# Component title cap (components.validate_components caps at 120).
_TITLE_CAP = 100
# Per-cell cap to keep rows bounded.
_CELL_CAP = 300
# Source tag surfaced to the UI as a title suffix.
_SOURCE_SUFFIX = " (from reply)"

# ---------------------------------------------------------------------------
# Intent cues (compiled once)
# ---------------------------------------------------------------------------

# Words/phrases in the user message that explicitly ask for a card kind.
# Used to relax the structural rules (a plain bullet list can become a
# checklist/table/timeline when the user asked for it).  Charts and
# forms cannot be derived from markdown and are not produced by this
# parser; the coordinator still sends the full schema for those.
_INTENT_CHECKLIST = re.compile(
    r"\b(check[\s-]?list|to[\s-]?do(?:s)?|todo(?:s)?|steps?|action items?|ticks?)\b",
    re.IGNORECASE,
)
_INTENT_COMPARISON = re.compile(
    r"\b(comparison|compare|comparing|table|tabular|side[\s-]by[\s-]side|versus|vs\.?)\b",
    re.IGNORECASE,
)
_INTENT_TIMELINE = re.compile(
    r"\b(timeline|schedule|agenda|itinerary|plan my (?:day|week|monday|tuesday|wednesday|thursday|friday|saturday|sunday)|milestones?)\b",
    re.IGNORECASE,
)
_INTENT_FACTS = re.compile(
    r"\b(facts?|key points?|highlights?|summary of facts|three facts|two facts|four facts)\b",
    re.IGNORECASE,
)
# Tokens that always warrant the full schema (charts/forms/decisions come
# from the model, not from markdown promotion).  Mirrors the coordinator's
# FULL_CARD_CUES set.
_FULL_SCHEMA_INTENT = re.compile(
    r"\b(charts?|graphs?|forms?|decisions?|options?|facts?|sources?)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _id_from_text(text: str, prefix: str, used: set, fallback: str) -> str:
    """Return a slug-style id that satisfies the components._ID_RE.

    Falls back to ``fallback`` when the text cannot produce a unique id.
    """
    raw = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    if not raw:
        raw = fallback
    raw = raw[: 64 - len(prefix)]
    candidate = f"{prefix}{raw}"
    suffix = 1
    while candidate in used or not _valid_id(candidate):
        suffix += 1
        candidate = f"{prefix}{raw}{suffix}"
        if suffix > 50:  # bounded; degenerate input cannot loop forever
            candidate = f"{prefix}{fallback}{len(used)}"
            used.add(candidate)
            return candidate
    used.add(candidate)
    return candidate


def _valid_id(value: str) -> bool:
    return bool(re.match(r"^[a-z0-9][a-z0-9_-]{0,63}$", value))


def _truncate(text: str, cap: int) -> str:
    text = text.strip()
    if len(text) <= cap:
        return text
    return text[: cap - 1] + "\u2026"


def _title(text: str, kind: str) -> str:
    return _truncate(text, _TITLE_CAP) + _SOURCE_SUFFIX


# ---------------------------------------------------------------------------
# Block segmentation (fenced code blocks are inert)
# ---------------------------------------------------------------------------


def _strip_fenced_blocks(text: str) -> tuple[str, list[str]]:
    """Return ``(text without fenced blocks, list of raw fenced blocks)``.

    Both `` ``` ... ``` `` and `` ~~~ ... ~~~ `` fences are honored, with
    an optional leading ``info string`` (e.g. ``assistant-ui``).  Anything
    inside a fence is preserved verbatim in the returned list and removed
    from the scan text so the parser cannot promote it.
    """
    fenced: list[str] = []
    out: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        # Find a fence opener: ``` or ~~~, run of 3+ of the same char.
        m = re.search(r"(^|\n)(`{3,}|~{3,})[^\n]*\n", text[i:])
        if not m:
            out.append(text[i:])
            break
        start = i + m.end()  # position after opener line's newline
        # Determine fence char and run length to find matching closer.
        fence_char = m.group(2)[0]
        fence_len = len(m.group(2))
        # Scan for a closing line that is exactly the same fence char run.
        # The closing fence must be on its own line and at least as long.
        close_pat = re.compile(
            r"(^|\n)" + re.escape(fence_char) + "{" + str(fence_len) + r",}\s*(?=\n|$)"
        )
        c = close_pat.search(text[start:])
        if not c:
            # Unterminated fence: keep the rest as inert (do not promote).
            fenced.append(text[i + m.start(2):])
            break
        body_end = start + c.start()
        # Closing fence starts at start + c.start(1) + 1; capture the
        # block from the opener to (and including) the closer line.
        closer_start = start + c.start() + (1 if c.group(1) else 0)
        closer_line_end = text.find("\n", closer_start)
        if closer_line_end < 0:
            closer_line_end = len(text)
        else:
            closer_line_end += 1
        fenced.append(text[i + m.start(2):closer_line_end])
        out.append(text[i: i + m.start(2)])
        out.append("\n")
        i = closer_line_end
    return "".join(out), fenced


# ---------------------------------------------------------------------------
# Markdown patterns
# ---------------------------------------------------------------------------

# Task list: ``- [ ] foo`` or ``- [x] foo`` (case-insensitive x).
_TASK_RE = re.compile(
    r"^[ \t]*(?:[-*+]|\d+\.)\s+\[(?P<mark>[ xX])\]\s*(?P<text>.+?)\s*$"
)

# Ordered/unordered list item (non-task).
_LIST_RE = re.compile(r"^[ \t]*(?:[-*+]|\d+\.)\s+(?P<text>.+?)\s*$")

# Heading line (``#``, ``##``, ...).
_HEADING_RE = re.compile(r"^[ \t]{0,3}#{1,6}\s+(?P<text>.+?)\s*$")

# Markdown pipe table row.
_TABLE_ROW_RE = re.compile(r"^[ \t]*\|?(?P<row>.+?)\|?[ \t]*$")

# Separator line: ``|---|---|`` (or with alignment ``:---:`` etc.).
_TABLE_SEP_RE = re.compile(
    r"^[ \t]*\|?\s*:?-{2,}:?\s*(?:\|\s*:?-{2,}:?\s*)+\|?\s*$"
)

# Time/weekday/date/Day N marker for timeline detection.  These are the
# common shapes small models actually emit; the parser is conservative
# and only fires when >= 3 items carry such a marker.
_TIME_HHMM = re.compile(r"\b\d{1,2}:\d{2}\b")
_TIME_HH = re.compile(r"\b\d{1,2}\s*(?:am|pm)\b", re.IGNORECASE)
_WEEKDAYS = (
    "monday", "tuesday", "wednesday", "thursday",
    "friday", "saturday", "sunday",
    "mon", "tue", "wed", "thu", "fri", "sat", "sun",
)
_DATE = re.compile(r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\s+\d{1,2}\b", re.IGNORECASE)
_DAY_N = re.compile(r"\bday\s+\d+\b", re.IGNORECASE)


def _has_timeline_marker(text: str) -> bool:
    if _TIME_HHMM.search(text) or _TIME_HH.search(text):
        return True
    if _DATE.search(text):
        return True
    if _DAY_N.search(text):
        return True
    low = text.lower().strip()
    # Weekday must appear at the start (e.g. "Mon: standup") so a sentence
    # like "we met on Monday" doesn't masquerade as a timeline entry.
    for wd in _WEEKDAYS:
        if low.startswith(wd + ":") or low.startswith(wd + ","):
            return True
    return False


def _first_marker(text: str) -> str:
    """Return the leading timeline marker in ``text`` if present, else ""."""
    m = _TIME_HHMM.search(text)
    if m:
        return m.group(0)
    m = _TIME_HH.search(text)
    if m:
        return m.group(0)
    m = _DATE.search(text)
    if m:
        return m.group(0)
    m = _DAY_N.search(text)
    if m:
        return m.group(0)
    low = text.lower().strip()
    for wd in _WEEKDAYS:
        if low.startswith(wd + ":") or low.startswith(wd + ","):
            return text[: len(wd) + 1].rstrip(":,").strip()
    return ""


# ---------------------------------------------------------------------------
# Section splitter (used to bound which blocks we consider)
# ---------------------------------------------------------------------------


def _iter_blocks(text: str) -> Iterable[tuple[str, str]]:
    """Yield ``(kind, body)`` for each markdown block in ``text``.

    ``kind`` is one of ``"heading"``, ``"task_list"``, ``"list"``,
    ``"table"``, ``"paragraph"``.  A paragraph whose every non-empty line
    starts with a time/date/Day-N marker is re-emitted as a ``list`` block
    so the timeline pass can promote it without writing a second parser.
    """
    lines = text.splitlines()
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        stripped = line.strip()
        if not stripped:
            i += 1
            continue
        m = _HEADING_RE.match(line)
        if m:
            yield "heading", m.group("text")
            i += 1
            continue
        if _TASK_RE.match(line):
            # collect consecutive task lines
            items: list[str] = []
            while i < n and _TASK_RE.match(lines[i]):
                items.append(lines[i])
                i += 1
            yield "task_list", "\n".join(items)
            continue
        if _LIST_RE.match(line):
            items = []
            while i < n and _LIST_RE.match(lines[i]):
                items.append(lines[i])
                i += 1
            yield "list", "\n".join(items)
            continue
        if _TABLE_ROW_RE.match(line) and i + 1 < n and _TABLE_SEP_RE.match(lines[i + 1]):
            rows = []
            while i < n and _TABLE_ROW_RE.match(lines[i]):
                rows.append(lines[i])
                i += 1
            yield "table", "\n".join(rows)
            continue
        # Plain paragraph: collect until blank or non-paragraph line.
        para = [line]
        i += 1
        while i < n:
            nxt = lines[i]
            ns = nxt.strip()
            if not ns:
                break
            if (_HEADING_RE.match(nxt) or _TASK_RE.match(nxt) or _LIST_RE.match(nxt)
                    or (_TABLE_ROW_RE.match(nxt) and i + 1 < n and _TABLE_SEP_RE.match(lines[i + 1]))):
                break
            para.append(nxt)
            i += 1
        joined = "\n".join(para)
        para_lines = [ln for ln in para if ln.strip()]
        if (len(para_lines) >= MIN_TIMELINE_ENTRIES
                and all(_has_timeline_marker(ln) for ln in para_lines)):
            # Re-emit as a list so the timeline pass picks it up.
            yield "list", "\n".join(f"- {ln}" for ln in para_lines)
        else:
            yield "paragraph", joined


# ---------------------------------------------------------------------------
# Per-kind parsers
# ---------------------------------------------------------------------------


def _parse_task_list(body: str, used_ids: set) -> list[dict]:
    out: list[dict] = []
    for line in body.splitlines():
        m = _TASK_RE.match(line)
        if not m:
            continue
        text = _truncate(m.group("text"), 200)
        if not text:
            continue
        done = m.group("mark").lower() == "x"
        iid = _id_from_text(text, "t", used_ids, "item")
        out.append({"id": iid, "text": text, "done": done})
    return out


def _split_table_row(line: str) -> list[str]:
    line = line.strip()
    if line.startswith("|"):
        line = line[1:]
    if line.endswith("|"):
        line = line[:-1]
    return [c.strip() for c in line.split("|")]


def _parse_table(body: str) -> tuple[list[str], list[list[str]]] | None:
    lines = body.splitlines()
    if len(lines) < 3:
        return None
    header = _split_table_row(lines[0])
    if not _TABLE_SEP_RE.match(lines[1]):
        return None
    sep_cols = _split_table_row(lines[1])
    if len(header) != len(sep_cols) or len(header) < MIN_COMPARISON_COLUMNS \
            or len(header) > MAX_COMPARISON_COLUMNS:
        return None
    data_rows: list[list[str]] = []
    for line in lines[2:]:
        if not line.strip():
            continue
        cells = _split_table_row(line)
        if len(cells) != len(header):
            return None  # inconsistent column count -> not a real table
        data_rows.append(cells)
    if len(data_rows) < MIN_COMPARISON_ROWS:
        return None
    if len(data_rows) > MAX_COMPARISON_ROWS:
        data_rows = data_rows[:MAX_COMPARISON_ROWS]
    return header, data_rows


def _parse_list_items(body: str) -> list[str]:
    out: list[str] = []
    for line in body.splitlines():
        m = _LIST_RE.match(line)
        if not m:
            continue
        text = m.group("text").strip()
        if text:
            out.append(text)
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_text(reply: str, user_text: str = "") -> dict | None:
    """Parse a chat reply into one assistant-built component, or None.

    Returns a dict ``{"type": "...", "title": "...", ...}`` ready for
    :func:`components.validate_components` to confirm.  Returns ``None``
    when the reply has nothing safe to promote.
    """
    if not isinstance(reply, str):
        return None
    if len(reply.encode("utf-8")) > MAX_INPUT_BYTES:
        return None  # bounded: refuse hostile input

    scan, _fenced = _strip_fenced_blocks(reply)
    if not scan.strip():
        return None

    user_intent = _classify_intent(user_text or "")

    component = _derive_component(scan, user_intent)
    if component is None:
        return None

    # Honest label: every derived card carries an obvious "from reply" tag.
    existing = component.get("title", "")
    if not existing.endswith(_SOURCE_SUFFIX):
        component["title"] = (_truncate(existing, _TITLE_CAP - len(_SOURCE_SUFFIX))
                              if existing else "") + _SOURCE_SUFFIX
    return component


def _classify_intent(user_text: str) -> dict:
    """Return which kinds the user explicitly asked for, in priority order."""
    text = user_text or ""
    return {
        "checklist": bool(_INTENT_CHECKLIST.search(text)),
        "comparison": bool(_INTENT_COMPARISON.search(text)),
        "timeline": bool(_INTENT_TIMELINE.search(text)),
        "facts": bool(_INTENT_FACTS.search(text)),
    }


def user_requested_full_schema(user_text: str) -> bool:
    """True when the user named a kind this parser cannot promote (charts,
    forms, decisions, options, sources).  Mirrors ``FULL_CARD_CUES``."""
    if not user_text:
        return False
    return bool(_FULL_SCHEMA_INTENT.search(user_text))


def _derive_component(scan: str, user_intent: dict) -> dict | None:
    used_ids: set[str] = set()

    # Pass 1: task-list block of >= 3 items wins immediately.
    for kind, body in _iter_blocks(scan):
        if kind == "task_list":
            items = _parse_task_list(body, used_ids)
            if len(items) >= MIN_CHECKLIST_ITEMS:
                items = items[:MAX_CHECKLIST_ITEMS]
                return {
                    "type": "checklist",
                    "items": items,
                }

    # Pass 2: pipe table wins immediately (a markdown table is unambiguous).
    for kind, body in _iter_blocks(scan):
        if kind == "table":
            parsed = _parse_table(body)
            if parsed is not None:
                header, rows = parsed
                columns = []
                for idx, label in enumerate(header):
                    label = _truncate(label or f"col-{idx + 1}", 120) or f"col-{idx + 1}"
                    cid = _id_from_text(label, "c", used_ids, f"col{idx + 1}")
                    columns.append({"id": cid, "label": label, "kind": "text"})
                out_rows = []
                for r_idx, cells in enumerate(rows):
                    cells = [_truncate(c, _CELL_CAP) for c in cells]
                    row_label = _truncate(cells[0] or f"row-{r_idx + 1}", 120) or f"row-{r_idx + 1}"
                    rid = _id_from_text(row_label, "r", used_ids, f"row{r_idx + 1}")
                    out_rows.append({
                        "id": rid,
                        "label": row_label,
                        "values": cells,
                    })
                return {
                    "type": "comparison",
                    "columns": columns,
                    "rows": out_rows,
                }

    # Pass 3: lists.  Two ways to promote a list:
    #   - timeline: >= 3 items with a leading time/weekday/date/Day-N marker
    #   - checklist (plain list): >= 3 items where the user asked for a list
    #   - comparison (plain list): >= 3 items where the user asked for a
    #     comparison and we can split label/values heuristically (we need
    #     at least 2 columns; if the list is one-token-per-item we fall
    #     back to a 1-column comparison is INVALID so we use checklist).
    list_blocks = [(k, b) for k, b in _iter_blocks(scan) if k == "list"]
    if not list_blocks:
        return None

    # Aggregate all list items in document order so a reply that splits
    # one logical list across multiple bullet sections (a common 2B
    # pattern: 4 bullets, blank line, 4 more bullets) is still recognised.
    all_items: list[str] = []
    for _kind, body in list_blocks:
        all_items.extend(_parse_list_items(body))
    if len(all_items) < MIN_CHECKLIST_ITEMS:
        return None

    # Timeline detection: need >= 3 items with a marker.
    timeline_markers = [(_first_marker(it), it) for it in all_items]
    marked = [(m, it) for m, it in timeline_markers if m]
    if len(marked) >= MIN_TIMELINE_ENTRIES:
        entries = []
        for m, text in marked[:MAX_TIMELINE_ENTRIES]:
            # Strip the marker prefix from the entry text so the body reads clean.
            stripped = text
            for prefix in (m + ":", m + ",", m):
                if stripped.lower().startswith(prefix.lower()):
                    stripped = stripped[len(prefix):].lstrip(" ,:")
                    break
            iid = _id_from_text(text, "e", used_ids, "entry")
            entries.append({
                "id": iid,
                "when": _truncate(m, 40),
                "text": _truncate(stripped or text, 300),
            })
        return {"type": "timeline", "entries": entries}

    # Comparison intent with a single token-per-item list: refuse (a
    # one-column comparison is invalid; the validator rejects it).
    # Fall through to checklist promotion.

    # User explicitly asked for a checklist -> promote plain list.
    if user_intent.get("checklist"):
        items = []
        for text in all_items[:MAX_CHECKLIST_ITEMS]:
            iid = _id_from_text(text, "t", used_ids, "item")
            items.append({"id": iid, "text": _truncate(text, 200), "done": False})
        return {"type": "checklist", "items": items}

    # User asked for a comparison and items look comparable (the items
    # might be ``Name | col1 | col2`` style, or "Item: a, b").  We only
    # promote a comparison if at least 2 of 3 items contain a separator.
    if user_intent.get("comparison"):
        cols = _try_split_comparison_rows(all_items)
        if cols is not None:
            header, rows = cols
            columns = []
            for idx, label in enumerate(header):
                label = _truncate(label or f"col-{idx + 1}", 120) or f"col-{idx + 1}"
                cid = _id_from_text(label, "c", used_ids, f"col{idx + 1}")
                columns.append({"id": cid, "label": label, "kind": "text"})
            out_rows = []
            for r_idx, cells in enumerate(rows):
                cells = [_truncate(c, _CELL_CAP) for c in cells]
                row_label = _truncate(cells[0] or f"row-{r_idx + 1}", 120) or f"row-{r_idx + 1}"
                rid = _id_from_text(row_label, "r", used_ids, f"row{r_idx + 1}")
                out_rows.append({"id": rid, "label": row_label, "values": cells})
            return {"type": "comparison", "columns": columns, "rows": out_rows}
        # Fallback: emit a checklist so the user still gets a card; the
        # title notes the source.  Better than nothing for "compare X and Y".
        items = []
        for text in all_items[:MAX_CHECKLIST_ITEMS]:
            iid = _id_from_text(text, "t", used_ids, "item")
            items.append({"id": iid, "text": _truncate(text, 200), "done": False})
        return {"type": "checklist", "items": items}

    # User asked for a timeline: defer to the marker check above (would
    # have matched).  Without enough markers, refuse.

    # User asked for facts: each list item becomes a "facts" card entry.
    if user_intent.get("facts"):
        cards = []
        for text in all_items[:MAX_FACTS]:
            heading, _, body = text.partition(":")
            heading = heading.strip() or text[:40].strip()
            body = body.strip() or text
            cid = _id_from_text(heading + "-" + body, "k", used_ids, "card")
            cards.append({
                "id": cid,
                "heading": _truncate(heading, 120),
                "text": _truncate(body, 600),
            })
        if len(cards) >= MIN_FACTS:
            return {"type": "facts", "cards": cards}

    # Default: do not promote.  A plain bullet list without intent cues
    # is the most common 2B output and frequently is just an explanation;
    # promoting it would be overreach.
    return None


def _try_split_comparison_rows(items: list[str]) -> tuple[list[str], list[list[str]]] | None:
    """If the items look like ``label | a | b`` or ``label: a, b``, build a
    2..8 column comparison.  Returns ``None`` when the items don't share a
    consistent split shape."""
    # Try pipe split first.
    pipe_splits = []
    for it in items:
        if "|" not in it:
            pipe_splits = []
            break
        parts = [p.strip() for p in it.split("|")]
        pipe_splits.append(parts)
    if pipe_splits:
        ncols = len(pipe_splits[0])
        if not (MIN_COMPARISON_COLUMNS <= ncols <= MAX_COMPARISON_COLUMNS):
            return None
        if not all(len(p) == ncols for p in pipe_splits):
            return None
        if len(pipe_splits) < MIN_COMPARISON_ROWS:
            return None
        header = [f"col-{i + 1}" for i in range(ncols)]
        header[0] = "Option"
        return header, pipe_splits[:MAX_COMPARISON_ROWS]
    # Try colon-then-comma: "Item: a, b"
    colon_splits = []
    for it in items:
        if ":" not in it:
            colon_splits = []
            break
        head, _, tail = it.partition(":")
        parts = [head.strip()] + [p.strip() for p in tail.split(",") if p.strip()]
        colon_splits.append(parts)
    if colon_splits:
        ncols = len(colon_splits[0])
        if not (MIN_COMPARISON_COLUMNS <= ncols <= MAX_COMPARISON_COLUMNS):
            return None
        if not all(len(p) == ncols for p in colon_splits):
            return None
        if len(colon_splits) < MIN_COMPARISON_ROWS:
            return None
        header = ["Option"] + [f"col-{i + 2}" for i in range(ncols - 1)]
        return header, colon_splits[:MAX_COMPARISON_ROWS]
    return None
