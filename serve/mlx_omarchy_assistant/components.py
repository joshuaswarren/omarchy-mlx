"""Generative-UI component schema: validation, action authorization, prompt.

The chat model may propose interactive cards instead of only prose. It emits a
versioned declarative envelope inside one fenced ``assistant-ui`` block; this
module is the single authority for what that envelope may contain and which
actions a rendered card may submit. Validation never executes HTML, CSS, JS or
SVG, and the renderer draws every pixel itself (charts are application-owned
canvas code), so a hostile envelope can only fail validation or render as
bounded data.

Trust model
-----------
- The model supplies *content* and *item ids* (option ids, row ids, field ids)
  so ids stay stable through request, response, display and explanation.
- The coordinator supplies *trusted identity*: it stamps ``id`` (component id),
  ``revision`` and ``turn_id`` after validation. A model envelope containing
  any of those keys is rejected, never honoured.
- Actions are allowlisted per component type and re-validated here
  (``validate_action``) before the coordinator turns one into a new user turn.

Bounds (design 2026-09-27): 64 KiB per envelope, 32 components per turn,
object depth 4 (the component object is depth 1; arrays are length-bounded and
element-validated but do not add depth), 20 form fields per form, 1000 total
table/chart data values per envelope.

Stdlib only.
"""

from __future__ import annotations

import copy
import json
import re

ENVELOPE_VERSION = 1
ENVELOPE_MAX_BYTES = 64 * 1024
MAX_COMPONENTS = 32
MAX_OBJECT_DEPTH = 4
MAX_FORM_FIELDS = 20
MAX_DATA_VALUES = 1000

ACTIONS = ("edit", "sort", "filter", "select", "expand", "submit")

_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")


class ComponentError(ValueError):
    """An envelope or action failed validation; the plain answer stays usable."""


# --------------------------------------------------------------------------
# primitive field validators (value -> normalized value, or ComponentError)
# --------------------------------------------------------------------------


def _fail(component_type: str, field: str, why: str):
    raise ComponentError(f"{component_type}.{field}: {why}")


def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _str_value(component_type: str, field: str, value, cap: int, allow_empty: bool = False):
    if not isinstance(value, str):
        _fail(component_type, field, "must be a string")
    if len(value) > cap:
        _fail(component_type, field, f"longer than {cap} characters")
    if not allow_empty and not value.strip():
        _fail(component_type, field, "must not be empty")
    return value


def _id_value(component_type: str, field: str, value):
    value = _str_value(component_type, field, value, 64)
    if not _ID_RE.match(value):
        _fail(component_type, field, "must match [a-z0-9][a-z0-9_-]{0,63}")
    return value


def _bool_value(component_type: str, field: str, value):
    if not isinstance(value, bool):
        _fail(component_type, field, "must be a boolean")
    return value


def _finite_number(component_type: str, field: str, value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(component_type, field, "must be a number")
    value = float(value)
    if value != value or value in (float("inf"), float("-inf")):
        _fail(component_type, field, "must be finite")
    return value


def _probability(component_type: str, field: str, value):
    value = _finite_number(component_type, field, value)
    if not 0.0 <= value <= 1.0:
        _fail(component_type, field, "must be between 0 and 1")
    return value


def _list_value(component_type: str, field: str, value, lo: int, hi: int):
    if not isinstance(value, list):
        _fail(component_type, field, "must be an array")
    if not lo <= len(value) <= hi:
        _fail(component_type, field, f"must contain {lo}-{hi} items")
    return value


def _obj(component_type: str, value, where: str = ""):
    if not isinstance(value, dict):
        _fail(component_type, where or "value", "must be an object")
    return value


def _only_keys(component_type: str, obj: dict, allowed: set, where: str = ""):
    extra = set(obj) - allowed
    if extra:
        _fail(component_type, where, f"unknown key(s): {', '.join(sorted(extra))}")


# --------------------------------------------------------------------------
# per-type specs: required/optional fields and permitted actions
# --------------------------------------------------------------------------


def _v_options(component_type, field, value):
    value = _list_value(component_type, field, value, 2, 8)
    return [
        {"id": _id_value(component_type, field, _obj(component_type, o, field)["id"]),
         "label": _str_value(component_type, field, _obj(component_type, o, field).get("label"), 120)}
        for o in value
    ]


def _v_columns(component_type, field, value):
    value = _list_value(component_type, field, value, 1, 8)
    out = []
    for col in value:
        col = _obj(component_type, col, field)
        _only_keys(component_type, col, {"id", "label", "kind"}, field)
        kind = col.get("kind", "text")
        if kind not in ("text", "number"):
            _fail(component_type, field, "column kind must be 'text' or 'number'")
        out.append({"id": _id_value(component_type, field, col["id"]),
                    "label": _str_value(component_type, field, col.get("label"), 120),
                    "kind": kind})
    ids = [c["id"] for c in out]
    if len(set(ids)) != len(ids):
        _fail(component_type, field, "column ids must be unique")
    return out


def _v_rows(component_type, field, value, columns):
    value = _list_value(component_type, field, value, 2, 32)
    col_ids = [c["id"] for c in columns]
    out = []
    for row in value:
        row = _obj(component_type, row, field)
        _only_keys(component_type, row, {"id", "label", "values"}, field)
        cells = row.get("values")
        if not isinstance(cells, list) or len(cells) != len(columns):
            _fail(component_type, field, "each row needs one value per column")
        clean = []
        for cell in cells:
            if cell is None:
                clean.append(None)
            elif isinstance(cell, (str)):
                clean.append(_str_value(component_type, field, cell, 300, allow_empty=True))
            else:
                clean.append(_finite_number(component_type, field, cell))
        out.append({"id": _id_value(component_type, field, row["id"]),
                    "label": _str_value(component_type, field, row.get("label"), 120),
                    "values": clean})
    ids = [r["id"] for r in out]
    if len(set(ids)) != len(ids):
        _fail(component_type, field, "row ids must be unique")
    return out


def _v_series(component_type, field, value):
    value = _list_value(component_type, field, value, 1, 4)
    out = []
    for s in value:
        s = _obj(component_type, s, field)
        _only_keys(component_type, s,
                   {"id", "label", "unit", "source", "estimate", "values"}, field)
        points = _list_value(component_type, "series.values", s.get("values"), 2, 200)
        clean_points = []
        for p in points:
            p = _obj(component_type, p, "series.values[]")
            _only_keys(component_type, p, {"label", "value"}, "series.values[]")
            clean_points.append({
                "label": _str_value(component_type, "series.values[]", p["label"], 40),
                "value": _finite_number(component_type, "series.values[]", p["value"]),
            })
        out.append({
            "id": _id_value(component_type, field, s["id"]),
            "label": _str_value(component_type, field, s.get("label"), 120),
            "unit": _str_value(component_type, field, s.get("unit", ""), 16, allow_empty=True),
            # Where the numbers came from. Estimates must say so; the renderer
            # shows the tag and the source beside every plot.
            "source": _str_value(component_type, field, s.get("source"), 120),
            "estimate": _bool_value(component_type, field, s.get("estimate", False)),
            "values": clean_points,
        })
    ids = [s["id"] for s in out]
    if len(set(ids)) != len(ids):
        _fail(component_type, field, "series ids must be unique")
    return out


def _v_checklist(component_type, field, value, require_text=True, text_cap=200):
    value = _list_value(component_type, field, value, 1, 50)
    out = []
    for item in value:
        item = _obj(component_type, item, field)
        _only_keys(component_type, item, {"id", "text", "done"}, field)
        entry = {"id": _id_value(component_type, field, item["id"]),
                 "done": _bool_value(component_type, field, item.get("done", False))}
        if require_text:
            entry["text"] = _str_value(component_type, field, item.get("text"), text_cap)
        out.append(entry)
    ids = [i["id"] for i in out]
    if len(set(ids)) != len(ids):
        _fail(component_type, field, "item ids must be unique")
    return out


def _v_form_fields(component_type, field, value):
    value = _list_value(component_type, field, value, 1, MAX_FORM_FIELDS)
    out = []
    for f in value:
        f = _obj(component_type, f, field)
        kind = f.get("kind")
        if kind not in ("text", "choice", "numeric"):
            _fail(component_type, field, "field kind must be text, choice or numeric")
        allowed = {"id", "label", "kind", "required", "placeholder",
                   "choices", "min", "max", "step", "initial"}
        _only_keys(component_type, f, allowed, field)
        entry = {
            "id": _id_value(component_type, field, f["id"]),
            "label": _str_value(component_type, field, f.get("label"), 120),
            "kind": kind,
            "required": _bool_value(component_type, field, f.get("required", False)),
            "placeholder": _str_value(component_type, field, f.get("placeholder", ""), 120,
                                      allow_empty=True),
        }
        if kind == "choice":
            choices = _list_value(component_type, "choices", f.get("choices"), 2, 12)
            entry["choices"] = [
                {"id": _id_value(component_type, "choices", _obj(component_type, c, "choices")["id"]),
                 "label": _str_value(component_type, "choices", _obj(component_type, c, "choices").get("label"), 120)}
                for c in choices
            ]
            choice_ids = [c["id"] for c in entry["choices"]]
            if len(set(choice_ids)) != len(choice_ids):
                _fail(component_type, "choices", "choice ids must be unique")
            initial = f.get("initial")
            if initial is not None:
                if initial not in choice_ids:
                    _fail(component_type, "initial", "must be one of the choice ids")
                entry["initial"] = initial
        elif kind == "numeric":
            for key in ("min", "max"):
                if f.get(key) is not None:
                    entry[key] = _finite_number(component_type, key, f[key])
            if entry.get("min") is not None and entry.get("max") is not None \
                    and entry["min"] > entry["max"]:
                _fail(component_type, field, "min exceeds max")
            entry["step"] = _finite_number(component_type, "step", f.get("step", 1))
            if entry["step"] <= 0:
                _fail(component_type, field, "step must be positive")
            if f.get("initial") is not None:
                initial = _finite_number(component_type, "initial", f["initial"])
                _check_numeric_bounds(component_type, field, initial, entry)
                entry["initial"] = initial
        else:  # text
            if f.get("initial") is not None:
                entry["initial"] = _str_value(component_type, "initial", f["initial"], 500,
                                              allow_empty=True)
        out.append(entry)
    ids = [f["id"] for f in out]
    if len(set(ids)) != len(ids):
        _fail(component_type, field, "field ids must be unique")
    return out


def _check_numeric_bounds(component_type, field, value, spec):
    if spec.get("min") is not None and value < spec["min"]:
        _fail(component_type, field, "below the field minimum")
    if spec.get("max") is not None and value > spec["max"]:
        _fail(component_type, field, "above the field maximum")


# component type -> (allowed keys, required keys, builder, actions)
SPECS = {
    "decision": (
        {"options", "selected", "criteria", "confidence", "model", "notes"},
        {"options", "selected", "criteria"},
        lambda ctype, comp: {
            "options": _v_options(ctype, "options", comp["options"]),
            "selected": _id_value(ctype, "selected", comp["selected"]),
            "criteria": _str_value(ctype, "criteria", comp["criteria"], 500),
            **({"confidence": _v_confidence(ctype, comp["confidence"])}
               if comp.get("confidence") is not None else {}),
            **({"model": _str_value(ctype, "model", comp["model"], 80)} if comp.get("model") else {}),
            **({"notes": _str_value(ctype, "notes", comp["notes"], 300)} if comp.get("notes") else {}),
        },
        ("select", "edit", "submit"),
    ),
    "comparison": (
        {"columns", "rows"},
        {"columns", "rows"},
        lambda ctype, comp: {
            "columns": _v_columns(ctype, "columns", comp["columns"]),
            "rows": None,  # filled below (needs columns)
        },
        ("select", "sort", "filter", "edit"),
    ),
    "chart": (
        {"kind", "series"},
        {"kind", "series"},
        lambda ctype, comp: {
            "kind": comp["kind"] if comp.get("kind") in ("bar", "line") else _fail(
                ctype, "kind", "must be 'bar' or 'line'"),
            "series": _v_series(ctype, "series", comp["series"]),
        },
        ("expand",),
    ),
    "checklist": (
        {"items"},
        {"items"},
        lambda ctype, comp: {"items": _v_checklist(ctype, "items", comp["items"])},
        ("edit", "submit"),
    ),
    "timeline": (
        {"entries"},
        {"entries"},
        lambda ctype, comp: {
            "entries": [
                {"id": _id_value(ctype, "entries", _obj(ctype, e, "entries")["id"]),
                 "when": _str_value(ctype, "entries", _obj(ctype, e, "entries").get("when", ""), 40,
                                    allow_empty=True),
                 "text": _str_value(ctype, "entries", _obj(ctype, e, "entries").get("text"), 300)}
                for e in _list_value(ctype, "entries", comp["entries"], 1, 50)
            ],
        },
        ("expand",),
    ),
    "form": (
        {"fields"},
        {"fields"},
        lambda ctype, comp: {"fields": _v_form_fields(ctype, "fields", comp["fields"])},
        ("submit",),
    ),
    "facts": (
        {"cards"},
        {"cards"},
        lambda ctype, comp: {"cards": _v_cards(ctype, comp["cards"])},
        ("expand",),
    ),
}

FORBIDDEN_IDENTITY_KEYS = {"id", "revision", "turn_id", "conversation_id"}


def _v_confidence(ctype, value):
    out = {"selected_probability": _probability(
        ctype, "confidence.selected_probability", value.get("selected_probability", 0.0))}
    if value.get("runner_up") is not None:
        ru = _obj(ctype, value["runner_up"], "confidence.runner_up")
        _only_keys(ctype, ru, {"id", "probability"}, "confidence.runner_up")
        out["runner_up"] = {"id": _id_value(ctype, "confidence.runner_up", ru["id"]),
                            "probability": _probability(
                                ctype, "confidence.runner_up.probability", ru["probability"])}
    out["abstained"] = _bool_value(ctype, "confidence.abstained", value.get("abstained", False))
    if value.get("entropy") is not None:
        out["entropy"] = _probability(ctype, "confidence.entropy", value["entropy"])
    return out


def _v_cards(ctype, value):
    value = _list_value(ctype, "cards", value, 1, 24)
    out = []
    for card in value:
        card = _obj(ctype, card, "cards")
        _only_keys(ctype, card, {"id", "heading", "text", "sources"}, "cards")
        entry = {"id": _id_value(ctype, "cards", card["id"]),
                 "heading": _str_value(ctype, "cards", card.get("heading"), 120),
                 "text": _str_value(ctype, "cards", card.get("text"), 600)}
        if card.get("sources") is not None:
            sources = _list_value(ctype, "sources", card["sources"], 1, 4)
            entry["sources"] = [
                {"ref": _str_value(ctype, "sources", _obj(ctype, s, "sources").get("ref"), 40),
                 "text": _str_value(ctype, "sources", _obj(ctype, s, "sources").get("text"), 400)}
                for s in sources
            ]
        out.append(entry)
    ids = [c["id"] for c in out]
    if len(set(ids)) != len(ids):
        _fail(ctype, "cards", "card ids must be unique")
    return out


def _check_depth(obj, limit: int = MAX_OBJECT_DEPTH):
    if isinstance(obj, dict):
        if limit == 0:
            raise ComponentError("envelope nests objects deeper than the allowed depth")
        for value in obj.values():
            _check_depth(value, limit - 1)
    elif isinstance(obj, list):
        for value in obj:
            _check_depth(value, limit)


def parse_envelope(payload) -> dict:
    """Parse raw fenced-block text/bytes or an already-decoded dict."""
    if isinstance(payload, (bytes, bytearray)):
        payload = bytes(payload).decode("utf-8")
    if isinstance(payload, str):
        if len(payload.encode("utf-8")) > ENVELOPE_MAX_BYTES:
            raise ComponentError(
                f"envelope exceeds the {ENVELOPE_MAX_BYTES} byte bound")
        try:
            payload = json.loads(payload, parse_constant=_reject_constant)
        except (json.JSONDecodeError, ValueError) as exc:
            raise ComponentError(f"envelope is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ComponentError("envelope must be a JSON object")
    if set(payload) != {"version", "components"}:
        raise ComponentError("envelope must contain exactly 'version' and 'components'")
    if payload["version"] != ENVELOPE_VERSION or not _is_int(payload["version"]):
        raise ComponentError(f"envelope version must be {ENVELOPE_VERSION}")
    if not isinstance(payload["components"], list):
        raise ComponentError("components must be an array")
    if len(payload["components"]) > MAX_COMPONENTS:
        raise ComponentError(f"more than {MAX_COMPONENTS} components per envelope")
    _check_depth(payload["components"])
    return payload


def _reject_constant(name):  # json NaN/Infinity hook
    raise ValueError(f"{name} is not valid JSON data")


def validate_components(payload) -> list[dict]:
    """Validate one model envelope; return normalized component dicts.

    Raises ComponentError with a short reason on any violation. The caller
    (coordinator) stamps trusted identity fields afterwards.
    """
    envelope = parse_envelope(payload)
    components = []
    data_values = 0
    for raw in envelope["components"]:
        if not isinstance(raw, dict):
            raise ComponentError("each component must be an object")
        ctype = raw.get("type")
        if ctype not in SPECS:
            raise ComponentError(f"unknown component type: {ctype!r}")
        present = set(raw)
        if present & FORBIDDEN_IDENTITY_KEYS:
            raise ComponentError(
                f"component type {ctype} carries coordinator-assigned key(s): "
                f"{', '.join(sorted(present & FORBIDDEN_IDENTITY_KEYS))}")
        allowed_keys, required_keys, build, _actions = SPECS[ctype]
        _only_keys(ctype, raw, allowed_keys | {"type", "title"}, "")
        missing = required_keys - present
        if missing:
            raise ComponentError(f"component type {ctype} missing: {', '.join(sorted(missing))}")
        component = {"type": ctype}
        if raw.get("title") is not None:
            component["title"] = _str_value(ctype, "title", raw["title"], 120)
        component.update(build(ctype, raw))
        if ctype == "comparison":
            component["rows"] = _v_rows(ctype, "rows", raw["rows"], component["columns"])
        selected = component.get("selected")
        if ctype == "decision" and selected not in {o["id"] for o in component["options"]}:
            raise ComponentError("decision.selected must be one of the option ids")
        if ctype == "chart":
            data_values += sum(len(s["values"]) for s in component["series"])
        if ctype == "comparison":
            data_values += sum(len(r["values"]) for r in component["rows"])
        if data_values > MAX_DATA_VALUES:
            raise ComponentError(f"more than {MAX_DATA_VALUES} table/chart data values")
        components.append(component)
    return components


# --------------------------------------------------------------------------
# action validation (coordinator calls before turning an action into a turn)
# --------------------------------------------------------------------------


def component_actions(component: dict) -> tuple:
    spec = SPECS.get(component.get("type"))
    return spec[3] if spec else ()


def _form_field_map(component):
    return {f["id"]: f for f in component.get("fields", [])}


def _validated_form_values(ctype, values, component):
    fields = _form_field_map(component)
    if not isinstance(values, dict):
        _fail(ctype, "values", "must be an object")
    unknown = set(values) - set(fields)
    if unknown:
        _fail(ctype, "values", f"unknown field id(s): {', '.join(sorted(unknown))}")
    clean = {}
    for fid, spec in fields.items():
        if fid not in values or values[fid] is None or values[fid] == "":
            if spec.get("required"):
                _fail(ctype, "values", f"field {fid} is required")
            continue
        if spec["kind"] == "text":
            clean[fid] = _str_value(ctype, f"values.{fid}", values[fid], 500, allow_empty=True)
        elif spec["kind"] == "choice":
            if values[fid] not in {c["id"] for c in spec["choices"]}:
                _fail(ctype, f"values.{fid}", "must be one of the field choices")
            clean[fid] = values[fid]
        else:
            number = _finite_number(ctype, f"values.{fid}", values[fid])
            _check_numeric_bounds(ctype, f"values.{fid}", number, spec)
            clean[fid] = number
    return clean


def validate_action(component: dict, action: str, values=None) -> dict:
    """Validate a user-triggered card action; return a normalized action dict.

    ``component`` must be a server-validated component (coordinator-stamped
    identity fields are ignored here). The coordinator still owns the stale
    revision check before applying the result.
    """
    if not isinstance(component, dict) or component.get("type") not in SPECS:
        raise ComponentError("action against an unknown component type")
    ctype = component["type"]
    if action not in component_actions(component):
        raise ComponentError(f"action {action!r} is not allowed on a {ctype} card")
    if values is None:
        values = {}
    if not isinstance(values, dict):
        raise ComponentError("action values must be an object")
    if len(json.dumps(values)) > 16 * 1024:
        raise ComponentError("action values exceed the 16 KiB bound")

    clean: dict = {}
    if action == "select":
        key = "option_id" if ctype == "decision" else "row_id"
        wanted = (values.get(key),)
        pool = ([o["id"] for o in component.get("options", [])]
                if ctype == "decision" else [r["id"] for r in component.get("rows", [])])
        if len(wanted) != 1 or wanted[0] not in pool:
            _fail(ctype, key, "must select an existing id")
        clean[key] = wanted[0]
    elif action == "sort":
        column_id = values.get("column_id")
        if column_id not in [c["id"] for c in component.get("columns", [])]:
            _fail(ctype, "column_id", "must be an existing column")
        direction = values.get("direction", "asc")
        if direction not in ("asc", "desc"):
            _fail(ctype, "direction", "must be 'asc' or 'desc'")
        clean = {"column_id": column_id, "direction": direction}
    elif action == "filter":
        clean = {"query": _str_value(ctype, "query", values.get("query", ""), 100,
                                     allow_empty=True)}
    elif action == "expand":
        if ctype == "facts" and values.get("card_id") is not None:
            if values["card_id"] not in [c["id"] for c in component.get("cards", [])]:
                _fail(ctype, "card_id", "must be an existing card")
            clean = {"card_id": values["card_id"]}
    elif action == "edit":
        if ctype == "checklist":
            edits = values.get("items")
            if not isinstance(edits, list) or not edits:
                _fail(ctype, "items", "must be a non-empty array")
            known = {i["id"] for i in component.get("items", [])}
            seen = set()
            for edit in edits:
                edit = _obj(ctype, edit, "items")
                _only_keys(ctype, edit, {"id", "done"}, "items")
                iid = _id_value(ctype, "items", edit.get("id"))
                if iid not in known or iid in seen:
                    _fail(ctype, "items", f"unknown or duplicate item id {iid!r}")
                seen.add(iid)
            clean = {"items": [{"id": e["id"], "done": _bool_value(ctype, "items.done", e["done"])}
                               for e in edits]}
        # decision/comparison edit carries no values; the composer prefills
        # from the card data server-side.
    elif action == "submit":
        if ctype == "form":
            clean = {"values": _validated_form_values(ctype, values.get("values"), component)}
        elif ctype == "checklist":
            items = values.get("items")
            if not isinstance(items, list) or not items:
                _fail(ctype, "items", "must be a non-empty array")
            known = {i["id"] for i in component.get("items", [])}
            clean = {"items": _v_checklist(ctype, "items", items)}
            submitted = {i["id"] for i in clean["items"]}
            if not submitted <= known:
                _fail(ctype, "items", "submitted items must belong to this checklist")
        elif ctype == "decision":
            # explain-style submit: the card asks the chat model about itself.
            clean = {}
    return {"action": action, "type": ctype, "values": clean}


# --------------------------------------------------------------------------
# plain-text equivalent (history/export fallback; renderer mirrors this)
# --------------------------------------------------------------------------


def plain_text(component: dict) -> str:
    """Deterministic text rendering of a validated component."""
    ctype = component.get("type", "unknown")
    title = component.get("title")
    lines = [f"[{title}]" if title else f"[{ctype} card]"]
    if ctype == "decision":
        for option in component["options"]:
            mark = "*" if option["id"] == component.get("selected") else " "
            lines.append(f"{mark} {option['label']}")
        lines.append(f"Criteria: {component.get('criteria', '')}")
        conf = component.get("confidence") or {}
        if conf.get("abstained"):
            lines.append("The decision model abstained.")
        elif conf.get("selected_probability") is not None:
            lines.append(f"Selected probability: {conf['selected_probability']:.2f}")
        if component.get("model"):
            lines.append(f"Decided by: {component['model']}")
        if component.get("notes"):
            lines.append(component["notes"])
    elif ctype == "comparison":
        cols = component["columns"]
        lines.append(" | ".join(["Option"] + [c["label"] for c in cols]))
        for row in component["rows"]:
            cells = ["" if v is None else (f"{v:g}" if isinstance(v, float) else str(v))
                     for v in row["values"]]
            lines.append(" | ".join([row["label"]] + cells))
    elif ctype == "chart":
        for series in component["series"]:
            tag = " (estimate)" if series.get("estimate") else ""
            points = ", ".join(f"{p['label']}={p['value']:g}" for p in series["values"])
            lines.append(f"{series['label']}{tag} in {series['unit'] or 'units'} "
                         f"({series.get('source', 'source unspecified')}): {points}")
    elif ctype == "checklist":
        lines.extend(f"[{'x' if i['done'] else ' '}] {i['text']}" for i in component["items"])
    elif ctype == "timeline":
        lines.extend(f"{e['when'] + ': ' if e.get('when') else ''}{e['text']}"
                     for e in component["entries"])
    elif ctype == "form":
        for field in component["fields"]:
            kind = field["kind"]
            extra = ""
            if kind == "choice":
                extra = " (" + ", ".join(c["label"] for c in field["choices"]) + ")"
            elif kind == "numeric":
                lo, hi = field.get("min"), field.get("max")
                extra = f" ({lo if lo is not None else '-inf'} to {hi if hi is not None else 'inf'})"
            required = " required" if field.get("required") else ""
            lines.append(f"{field['label']}{extra}{required}")
        lines.append("(Submit sends your answers as a new message.)")
    elif ctype == "facts":
        for card in component["cards"]:
            lines.append(f"{card['heading']}: {card['text']}")
            for source in card.get("sources", []):
                lines.append(f"  Source {source['ref']}: {source['text']}")
    return "\n".join(lines)


SCHEMA_PROMPT = f"""\
Interactive cards: when a card genuinely helps, end your reply with EXACTLY ONE \
fenced code block tagged assistant-ui containing one JSON object, and keep the \
readable answer as plain text outside the block:

```assistant-ui
{{"version": {ENVELOPE_VERSION}, "components": [ ... ]}}
```

The block must be complete, valid JSON. Bounds: at most {MAX_COMPONENTS} components, \
{MAX_FORM_FIELDS} form fields per form, {MAX_DATA_VALUES} total table/chart values. \
If the card would not fit or would be incomplete, omit it entirely and answer in text.

Never put HTML, CSS, JavaScript, SVG, image URLs or shell commands anywhere in the \
block. Data must come from the conversation or be clearly labelled estimates \
("estimate": true, with the series "source" naming where numbers came from).

Component types ("type" is required; "title" up to 120 chars is optional):

- decision: {{"type": "decision", "options": [{{"id": "opt-1", "label": "..."}}, ...] \
(2-8 options), "selected": "<an option id>", "criteria": "why this was chosen" (<=500 chars), \
"confidence": {{"selected_probability": 0.0-1.0, "runner_up": {{"id": "...", \
"probability": 0.0-1.0}}, "abstained": false}}, "model": "...", "notes": "..."}}
- comparison: {{"type": "comparison", "columns": [{{"id": "c1", "label": "...", \
"kind": "text"|"number"}}] (1-8), "rows": [{{"id": "r1", "label": "...", \
"values": [one cell per column; string, number or null]}}] (2-32 rows)}}
- chart: {{"type": "chart", "kind": "bar"|"line", "series": [{{"id": "s1", "label": "...", \
"unit": "kg" (<=16 chars), "source": "where these numbers came from", "estimate": true|false, \
"values": [{{"label": "...", "value": 1.0}}] (2-200 points, finite numbers only)}}] (1-4 series)}}
- checklist: {{"type": "checklist", "items": [{{"id": "t1", "text": "...", "done": false}}] \
(1-50 items)}}
- timeline: {{"type": "timeline", "entries": [{{"id": "e1", "when": "Mon" (<=40 chars), \
"text": "..."}}] (1-50 entries)}}
- form: {{"type": "form", "fields": [{{"id": "f1", "label": "...", "kind": "text"|"choice"|"numeric", \
"required": true, "placeholder": "...", "choices": [{{"id": "a", "label": "..."}}] (choice only, \
2-12), "min": 0, "max": 10, "step": 1, "initial": ...}}] (1-{MAX_FORM_FIELDS} fields)}}
- facts: {{"type": "facts", "cards": [{{"id": "k1", "heading": "...", "text": "..." \
(<=600 chars), "sources": [{{"ref": "user msg", "text": "the quoted material"}}] (1-4)}}] \
(1-24 cards)}}

ids are lowercase letters, digits, "-" or "_" (<=64 chars) and must be unique within \
a component. Do not invent measurements; do not present estimates as measurements. \
The application renders these cards itself; you cannot add new component types, \
fields, actions, URLs, code or styling.
"""
