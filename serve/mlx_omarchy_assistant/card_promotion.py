"""Deterministic card promotion: turn a finished chat reply into zero or one card.

The chat models write rich markdown (task lists, pipe tables, numbered
plans) but do not reliably emit the ``assistant-ui`` fenced JSON.  When a
chat turn ends without a valid fenced envelope, the coordinator calls
:func:`extract_text` once on the rendered reply.  The result is either
``None`` (prose only) or one component that the coordinator validates with
:func:`components.validate_components` before emitting it.

Rules
-----
A card is built only when BOTH hold:

1. **The user asked for that artifact.**  The user message must name the
   kind: checklist / to-do / action items / a ``<noun>ing list``
   (checklist); compare / comparison / contrast / tabular / side by side /
   versus (comparison); timeline / schedule / agenda / itinerary / roadmap /
   "plan my <day>" (timeline); facts / key points / key takeaways (facts).
   Weaker cues count unless the message leads with an explanation request
   ("explain", "describe", "why", "how does", "what is", "tell me about"):
   "steps" or "tasks" ask for a checklist; "in a table", "table of ...",
   "2x2 table" ask for a comparison; "milestones", "phases" ask for a
   timeline; "highlights" asks for facts.  "What is a hash table" and
   "describe the phases of the moon" stay prose.  A request for prose ("in a
   paragraph", "in one sentence", "no bullets", "without using a list")
   turns promotion off.  A table or list inside an answer to a question
   that did not ask for an artifact stays prose.
2. **The reply has the matching structure** (outside fenced code blocks):

   - checklist: a task list (``- [ ]`` / ``- [x]``) or a plain bullet or
     numbered list, at least 3 items.
   - comparison: a pipe table (header, ``|---|`` separator, 2-8 columns,
     at least 2 data rows, every row the same width), or 2 or more headed
     sections of bullets (``### Cost`` or a whole-line ``**Cost**``): one row
     per section, a column per bullet label (``**Price:** ...``) that two or
     more sections share, or one Details column when none is shared.
   - timeline: at least 3 list items or headings that carry a time,
     weekday, date, ``Day/Week/Month/Phase/Step/Stage N`` or ``QN``
     marker, or a leading label (``**Discovery** - ...`` / ``Pilot: ...``);
     or a pipe table whose first column becomes ``when``.
   - facts: at least 2 list items.

When the message names several kinds, the first one mentioned that the
reply can satisfy wins.  Only one card is produced per reply.

Limits
------
- Charts, forms, decisions and sourced facts cannot be derived from
  markdown; the coordinator sends the full schema when the user names them.
- The parser never invents data: cells and items are the reply's own text,
  truncated to the validator's bounds.
- Every card's title ends with ``(from reply)`` so the UI shows that the
  application built it from the reply, not that the model emitted JSON.
- Input over 1 MiB is refused.  Every pattern is linear (no nested or
  overlapping quantifiers) and each line is scanned a bounded number of
  times, so hostile input costs O(n).

Stdlib only.
"""

from __future__ import annotations

import re

MAX_INPUT_BYTES = 1024 * 1024
MIN_LIST_ITEMS = 3
MIN_FACTS = 2
MAX_ITEMS = 50  # checklist items and timeline entries (validator bound)
MAX_FACTS = 24
MIN_COLUMNS = 2
MAX_COLUMNS = 8
MIN_ROWS = 2
MAX_ROWS = 32
_ITEM_CAP = 200
_CELL_CAP = 300
_LABEL_CAP = 120
_WHEN_CAP = 40
_FACT_CAP = 600
_SOURCE_SUFFIX = " (from reply)"
_KIND_TITLES = {"checklist": "Checklist", "comparison": "Comparison",
                "timeline": "Timeline", "facts": "Facts"}

# ---------------------------------------------------------------------------
# User intent
# ---------------------------------------------------------------------------

_STRONG_INTENT = {
    "checklist": re.compile(
        r"\b(?:check[\s-]?lists?|to[\s-]?dos?|todos?|action items?|task lists?"
        r"|[a-z]{2,20}ing lists?|grocery lists?)\b", re.IGNORECASE),
    "comparison": re.compile(
        r"\b(?:compare|compares|compared|comparing|comparison|contrast|contrasting|tabular"
        r"|side[\s-]by[\s-]side|versus|vs)\b", re.IGNORECASE),
    "timeline": re.compile(
        r"\b(?:timelines?|schedules?|scheduled|agenda|itinerary|roadmap"
        r"|plan my (?:day|week|weekend|morning|afternoon|evening|monday|tuesday"
        r"|wednesday|thursday|friday|saturday|sunday))\b", re.IGNORECASE),
    "facts": re.compile(
        r"\b(?:facts?|key points?|key takeaways?|takeaways)\b", re.IGNORECASE),
}
# Weak cues are topic words as often as artifact requests; they count only
# when the message does not lead with an explanation request ("what is a
# hash table", "describe the phases of the moon" ask for prose).
_WEAK_INTENT = {
    "checklist": re.compile(r"\b(?:steps?|tasks?)\b", re.IGNORECASE),
    "timeline": re.compile(
        r"\b(?:milestones?|phases?|phased|week[\s-]by[\s-]week|day[\s-]by[\s-]day)\b",
        re.IGNORECASE),
    "facts": re.compile(r"\bhighlights\b", re.IGNORECASE),
    "comparison": re.compile(
        r"\b(?:(?:in|as|into) (?:a |the )?(?:[\w-]+ ){0,2}tables?"
        r"|tables? (?:of|comparing|showing|with)|\d+\s*x\s*\d+ tables?)\b", re.IGNORECASE),
}
_EXPLAIN_LEAD = re.compile(
    r"^\W*(?:(?:briefly|quickly|please|can you|could you)\s+)*"
    r"(?:explain|describe|why|how (?:does|do|did|is|are|can|would)"
    r"|what is|what's|what are the differences?|tell me about)\b",
    re.IGNORECASE)
_PROSE_REQUEST = re.compile(
    r"\b(?:in|as) (?:a |one |two |a single |a short |plain )?"
    r"(?:paragraph|sentence|prose|plain words|plain text)s?\b"
    r"|\b(?:no|not as an?|not in an?) (?:bullets?|bullet points|lists?|tables?)\b"
    r"|\bwithout (?:using |any |making )?(?:an? )?"
    r"(?:bullets?|bullet points|lists?|tables?|formatting)\b",
    re.IGNORECASE)


def requested_kinds(user_text: str) -> list[str]:
    """Card kinds the user asked for, in the order they are mentioned."""
    text = user_text or ""
    if _PROSE_REQUEST.search(text):
        return []
    found: dict[str, int] = {}
    for kind, pattern in _STRONG_INTENT.items():
        m = pattern.search(text)
        if m:
            found[kind] = m.start()
    if not _EXPLAIN_LEAD.search(text):
        for kind, pattern in _WEAK_INTENT.items():
            m = None if kind in found else pattern.search(text)
            if m:
                found[kind] = m.start()
    return sorted(found, key=found.__getitem__)


# ---------------------------------------------------------------------------
# Markdown scanning (linear patterns only)
# ---------------------------------------------------------------------------

_FENCE = re.compile(r"[ \t]*(`{3,}|~{3,})")
_TASK = re.compile(r"[ \t]*(?:[-*+]|\d{1,9}[.)])[ \t]+\[([ xX])\][ \t]*(\S.*)")
_ITEM = re.compile(r"[ \t]*(?:[-*+]|\d{1,9}[.)])[ \t]+(\S.*)")
_HEADING = re.compile(r"[ \t]{0,3}#{1,6}[ \t]+(\S.*)")
_SEP_CELL = re.compile(r":?-+:?")
_EMPHASIS = re.compile(r"\*\*|`")

_TIME = re.compile(r"\b\d{1,2}(?::\d{2}(?:\s?[ap]\.?m\.?)?|\s?[ap]\.?m\.?)(?!\w)",
                   re.IGNORECASE)
_DATE = re.compile(
    r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?\s+\d{1,4}\b"
    r"|\b\d{4}-\d{2}(?:-\d{2})?\b|\b(?:1[5-9]|20)\d{2}\b", re.IGNORECASE)
_PERIOD = re.compile(
    r"\b(?:day|week|month|year|phase|step|stage|sprint|quarter)s?\s+\d{1,3}"
    r"(?:\s*[-\u2013]\s*\d{1,3})?\b|\bq[1-4]\b", re.IGNORECASE)
_WEEKDAY = re.compile(
    r"(?:mon|tue|tues|wed|thu|thur|thurs|fri|sat|sun)(?:day|nesday|sday|rsday|urday)?\b",
    re.IGNORECASE)
_BOLD_LINE = re.compile(r"[ \t]*\*\*([^*]{1,80})\*\*:?[ \t]*")
_NUMBERING = re.compile(r"\d{1,3}[.)][ \t]*")
_LEAD_LABEL = re.compile(r"([^:\u2013\u2014|]{1,40}?)\s*(?::|\s[-\u2013\u2014]\s)\s*(\S.*)")


def _strip_fences(text: str) -> list[str]:
    """Lines outside fenced code blocks.  An unterminated fence hides the rest."""
    out: list[str] = []
    fence: tuple[str, int] | None = None
    for line in text.splitlines():
        m = _FENCE.match(line)
        if fence is None:
            if m:
                fence = (m.group(1)[0], len(m.group(1)))
            else:
                out.append(line)
        elif (m and m.group(1)[0] == fence[0] and len(m.group(1)) >= fence[1]
              and not line[m.end():].strip()):
            fence = None
    return out


def _clean(text: str, cap: int) -> str:
    text = _EMPHASIS.sub("", text).strip()
    return text if len(text) <= cap else text[: cap - 1] + "\u2026"


def _row_cells(line: str) -> list[str]:
    line = line.strip()
    if line.startswith("|"):
        line = line[1:]
    if line.endswith("|"):
        line = line[:-1]
    return [cell.strip() for cell in line.split("|")]


def _is_separator(line: str) -> bool:
    if "-" not in line or "|" not in line:
        return False
    cells = _row_cells(line)
    return len(cells) >= MIN_COLUMNS and all(_SEP_CELL.fullmatch(c) for c in cells)


class _Blocks:
    """One pass over the reply: task items, list items, headings, other
    non-blank lines, pipe tables, and headed sections of bullets.

    A section starts at a ``#`` heading or a line that is entirely bold.  Its
    top-level bullets are ``[label, parts]`` (``**Cost:** low`` gives label
    ``Cost``); deeper bullets are appended to the preceding one's parts."""

    def __init__(self, lines: list[str]):
        self.tasks: list[tuple[bool, str]] = []
        self.items: list[str] = []
        self.headings: list[str] = []
        self.lines: list[str] = []
        self.tables: list[tuple[list[str], list[list[str]]]] = []
        self.sections: list[tuple[str, list[list]]] = []
        section: list[list] | None = None
        top_indent = 0
        i, n = 0, len(lines)
        while i < n:
            line = lines[i]
            if "|" in line and i + 1 < n and _is_separator(lines[i + 1]):
                header = _row_cells(line)
                rows: list[list[str]] = []
                i += 2
                consistent = True
                while i < n and "|" in lines[i] and lines[i].strip():
                    cells = _row_cells(lines[i])
                    consistent = consistent and len(cells) == len(header)
                    rows.append(cells)
                    i += 1
                if (consistent and MIN_COLUMNS <= len(header) <= MAX_COLUMNS
                        and len(rows) >= MIN_ROWS):
                    self.tables.append((header, rows[:MAX_ROWS]))
                continue
            m = _TASK.match(line)
            if m:
                self.tasks.append((m.group(1) in "xX", m.group(2)))
                i += 1
                continue
            m = _ITEM.match(line)
            if m:
                self.items.append(m.group(1))
                if section is not None:
                    indent = len(line) - len(line.lstrip())
                    if not section or indent <= top_indent:
                        top_indent = indent
                        section.append(_labelled(m.group(1)))
                    else:
                        section[-1][1].append(_clean(m.group(1), _CELL_CAP))
                i += 1
                continue
            m = _HEADING.match(line) or _BOLD_LINE.fullmatch(line)
            if m:
                self.headings.append(m.group(1))
                section = []
                title = _clean(_NUMBERING.sub("", m.group(1), count=1), _LABEL_CAP)
                self.sections.append((title, section))
            elif line.strip():
                self.lines.append(line.strip())
            i += 1


def _labelled(text: str) -> list:
    """``[label, [body]]`` for a bullet; label is None when it has none."""
    plain = _clean(text, _CELL_CAP)
    if plain.endswith(":") and len(plain) <= 41:
        return [plain[:-1].strip(), []]
    m = _LEAD_LABEL.match(plain)
    return [m.group(1).strip(), [m.group(2)]] if m else [None, [plain]]


# ---------------------------------------------------------------------------
# Per-kind builders.  Each returns a component dict or None.
# ---------------------------------------------------------------------------

class _Ids:
    def __init__(self):
        self.used: set[str] = set()

    def make(self, prefix: str, text: str) -> str:
        slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:40]
        base = f"{prefix}-{slug}" if slug else prefix
        candidate, n = base, 1
        while candidate in self.used:
            n += 1
            candidate = f"{base}-{n}"
        self.used.add(candidate)
        return candidate


def _checklist(blocks: _Blocks) -> dict | None:
    entries = blocks.tasks if len(blocks.tasks) >= MIN_LIST_ITEMS else [
        (False, text) for text in blocks.items]
    if len(entries) < MIN_LIST_ITEMS:
        return None
    ids = _Ids()
    items = []
    for done, text in entries[:MAX_ITEMS]:
        text = _clean(text, _ITEM_CAP)
        if text:
            items.append({"id": ids.make("t", text), "text": text, "done": done})
    return {"type": "checklist", "items": items} if len(items) >= MIN_LIST_ITEMS else None


def _section_table(blocks: _Blocks) -> tuple[list[str], list[list[str]]] | None:
    """A table from two or more headed sections of bullets: one row per
    section; a column per bullet label shared by two or more sections, or a
    single Details column when no label is shared."""
    sections = [(title, entries) for title, entries in blocks.sections
                if title and entries][:MAX_ROWS]
    if len(sections) < MIN_ROWS:
        return None
    counts: dict[str, list] = {}  # lower-case label -> [first spelling, sections]
    for _, entries in sections:
        seen = set()
        for label, _ in entries:
            if label and label.lower() not in seen:
                seen.add(label.lower())
                counts.setdefault(label.lower(), [label, 0])[1] += 1
    shared = [shown for shown, count in counts.values() if count >= 2][:MAX_COLUMNS - 1]
    rows = []
    for title, entries in sections:
        if shared:
            by_label = {}
            for label, parts in entries:
                if label:
                    by_label.setdefault(label.lower(), " ".join(parts))
            if any(label.lower() in by_label for label in shared):
                rows.append([title] + [by_label.get(label.lower(), "") for label in shared])
        else:
            rows.append([title, "; ".join(f"{label}: {' '.join(parts)}" if label and parts
                                          else label or " ".join(parts)
                                          for label, parts in entries)])
    header = ["Item"] + shared if shared else ["Aspect", "Details"]
    return (header, rows) if len(rows) >= MIN_ROWS else None


def _comparison(blocks: _Blocks) -> dict | None:
    table = blocks.tables[0] if blocks.tables else _section_table(blocks)
    if table is None:
        return None
    header, rows = table
    ids = _Ids()
    columns = []
    for index, label in enumerate(header):
        label = _clean(label, _LABEL_CAP) or f"Column {index + 1}"
        columns.append({"id": ids.make("c", label), "label": label, "kind": "text"})
    out_rows = []
    for index, cells in enumerate(rows):
        cells = [_clean(cell, _CELL_CAP) for cell in cells]
        label = _clean(cells[0], _LABEL_CAP) or f"Row {index + 1}"
        out_rows.append({"id": ids.make("r", label), "label": label, "values": cells})
    return {"type": "comparison", "columns": columns, "rows": out_rows}


def _marker(text: str) -> str:
    """The time/date/period marker in the first 60 characters of ``text``."""
    head = text[:60]
    for pattern in (_PERIOD, _TIME, _DATE):
        m = pattern.search(head)
        if m:
            return m.group(0)
    m = _WEEKDAY.match(head)
    return m.group(0) if m else ""


def _timeline_entry(text: str, labels: bool = True) -> tuple[str, str] | None:
    """Split ``text`` into ``(when, what)``.  A marker at the start wins over
    a leading label so ``9:30 standup`` keeps its time.  With ``labels``
    off (unbulleted prose lines) only a leading marker counts."""
    plain = _clean(text, 10_000)
    when = _marker(plain)
    starts = bool(when) and plain.lower().startswith(when.lower())
    label = _LEAD_LABEL.match(plain) if labels else None
    if when and (starts or (labels and not label)):
        rest = plain.replace(when, "", 1).strip(" :,-\u2013\u2014")
        return when, rest or plain
    if label:
        return label.group(1).strip(), label.group(2)
    return None


def _timeline(blocks: _Blocks) -> dict | None:
    candidates: list[tuple[str, str]] = []
    if blocks.tables:
        header, rows = blocks.tables[0]
        for cells in rows:
            what = " \u2014 ".join(c for c in cells[1:] if c)
            if cells[0] and what:
                candidates.append((cells[0], what))
    sources = ((blocks.items, True), ([t for _, t in blocks.tasks], True),
               (blocks.headings, True), (blocks.lines, False))
    for source, labels in sources:
        if len(candidates) >= MIN_LIST_ITEMS:
            break
        entries = [e for e in (_timeline_entry(t, labels) for t in source) if e]
        if len(entries) >= MIN_LIST_ITEMS:
            candidates = entries
    if len(candidates) < MIN_LIST_ITEMS:
        return None
    ids = _Ids()
    out = []
    for when, what in candidates[:MAX_ITEMS]:
        when = _clean(when, _WHEN_CAP)
        what = _clean(what, _CELL_CAP)
        if when and what:
            out.append({"id": ids.make("e", when), "when": when, "text": what})
    return {"type": "timeline", "entries": out} if len(out) >= MIN_LIST_ITEMS else None


def _facts(blocks: _Blocks) -> dict | None:
    texts = blocks.items or [t for _, t in blocks.tasks]
    if len(texts) < MIN_FACTS:
        return None
    ids = _Ids()
    cards = []
    for text in texts[:MAX_FACTS]:
        plain = _clean(text, 10_000)
        m = _LEAD_LABEL.match(plain)
        heading, body = (m.group(1), m.group(2)) if m else (f"Fact {len(cards) + 1}", plain)
        heading, body = _clean(heading, _LABEL_CAP), _clean(body, _FACT_CAP)
        if heading and body:
            cards.append({"id": ids.make("k", heading), "heading": heading, "text": body})
    return {"type": "facts", "cards": cards} if len(cards) >= MIN_FACTS else None


_BUILDERS = {"checklist": _checklist, "comparison": _comparison,
             "timeline": _timeline, "facts": _facts}


def extract_text(reply: str, user_text: str = "") -> dict | None:
    """Return one component built from ``reply``'s markdown, or None."""
    if not isinstance(reply, str) or len(reply.encode("utf-8", "replace")) > MAX_INPUT_BYTES:
        return None
    kinds = requested_kinds(user_text)
    if not kinds:
        return None
    blocks = _Blocks(_strip_fences(reply))
    for kind in kinds:
        component = _BUILDERS[kind](blocks)
        if component is not None:
            component["title"] = _KIND_TITLES[kind] + _SOURCE_SUFFIX
            return component
    return None
