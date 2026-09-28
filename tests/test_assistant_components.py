"""Generative-UI component schema tests.

The component layer is the trust boundary between the chat model and the
renderer; these tests pin down every structural, identity, action and
content-bound guard without depending on the network or on the renderer.
"""

import json
import math
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "serve"))

from mlx_omarchy_assistant.components import (  # noqa: E402
    ComponentError, MAX_COMPONENTS, MAX_DATA_VALUES, MAX_FORM_FIELDS,
    SCHEMA_PROMPT, ENVELOPE_MAX_BYTES, ENVELOPE_VERSION, FORBIDDEN_IDENTITY_KEYS,
    plain_text, validate_action, validate_components,
)


def envelope(*components, version=ENVELOPE_VERSION):
    return {"version": version, "components": list(components)}


def decision_envelope(selected="opt-2", with_confidence=True):
    options = [
        {"id": "opt-1", "label": "A"},
        {"id": "opt-2", "label": "B"},
        {"id": "opt-3", "label": "C"},
    ]
    comp = {
        "type": "decision",
        "options": options,
        "selected": selected,
        "criteria": "lower memory footprint",
    }
    if with_confidence:
        comp["confidence"] = {"selected_probability": 0.62, "abstained": False}
    return envelope(comp)


class SchemaPromptTests(unittest.TestCase):
    def test_prompt_contains_fence_and_version(self):
        self.assertIn("```assistant-ui", SCHEMA_PROMPT)
        self.assertIn(f'"version": {ENVELOPE_VERSION}', SCHEMA_PROMPT)

    def test_prompt_names_every_type(self):
        for ctype in ("decision", "comparison", "chart", "checklist",
                      "timeline", "form", "facts"):
            self.assertIn(ctype, SCHEMA_PROMPT, f"type {ctype} missing")

    def test_prompt_warns_against_html_and_code(self):
        for forbidden in ("HTML", "CSS", "JavaScript", "image URLs", "shell"):
            self.assertIn(forbidden, SCHEMA_PROMPT)


class EnvelopeBoundTests(unittest.TestCase):
    def test_decision_round_trips(self):
        components = validate_components(decision_envelope())
        self.assertEqual(len(components), 1)
        self.assertEqual(components[0]["type"], "decision")
        self.assertEqual(components[0]["selected"], "opt-2")

    def test_forbidden_identity_keys_rejected(self):
        for key in ("id", "revision", "turn_id", "conversation_id"):
            comp = {"type": "decision", "id": "comp-1", "options": [
                {"id": "a", "label": "A"}, {"id": "b", "label": "B"}],
                "selected": "a", "criteria": "ok"}
            with self.assertRaises(ComponentError):
                validate_components(envelope({**comp, key: "x"}))

    def test_unknown_type_rejected(self):
        with self.assertRaises(ComponentError):
            validate_components(envelope({"type": "evil"}))

    def test_unknown_top_level_key_rejected(self):
        payload = envelope({"type": "decision", "options": [
            {"id": "a", "label": "A"}, {"id": "b", "label": "B"}],
            "selected": "a", "criteria": "ok"})
        payload["hack"] = {"script": "alert(1)"}
        with self.assertRaises(ComponentError):
            validate_components(payload)

    def test_unknown_field_rejected(self):
        comp = {"type": "decision", "options": [
                {"id": "a", "label": "A"}, {"id": "b", "label": "B"}],
                "selected": "a", "criteria": "ok", "title": "Pick"}
        comp["javascript"] = "alert(1)"
        with self.assertRaises(ComponentError):
            validate_components(envelope(comp))

    def test_version_must_match(self):
        payload = decision_envelope()
        payload["version"] = 2
        with self.assertRaises(ComponentError):
            validate_components(payload)

    def test_max_components_enforced(self):
        comps = []
        for i in range(MAX_COMPONENTS + 1):
            comps.append({"type": "decision", "options": [
                {"id": f"a-{i}", "label": "A"}, {"id": f"b-{i}", "label": "B"}],
                "selected": f"a-{i}", "criteria": "ok"})
        with self.assertRaises(ComponentError):
            validate_components(envelope(*comps))

    def test_max_form_fields_enforced(self):
        fields = [{"id": f"f{i}", "label": f"F{i}", "kind": "text"} for i in range(MAX_FORM_FIELDS + 1)]
        with self.assertRaises(ComponentError):
            validate_components(envelope({"type": "form", "fields": fields}))

    def test_max_data_values_enforced(self):
        # Two chart series that combined exceed 1000 values.
        series = [{"id": "a", "label": "A", "unit": "", "source": "x", "estimate": False,
                   "values": [{"label": str(j), "value": float(j)} for j in range(800)]},
                  {"id": "b", "label": "B", "unit": "", "source": "x", "estimate": False,
                   "values": [{"label": str(j), "value": float(j)} for j in range(300)]}]
        with self.assertRaises(ComponentError):
            validate_components(envelope({"type": "chart", "kind": "bar", "series": series}))

    def test_byte_size_enforced(self):
        # 64 KiB envelope + a single oversized string field that pushes it over.
        payload = decision_envelope()
        payload["components"][0]["criteria"] = "x" * (ENVELOPE_MAX_BYTES)
        raw = json.dumps(payload)
        self.assertGreater(len(raw), ENVELOPE_MAX_BYTES)
        with self.assertRaises(ComponentError):
            validate_components(raw)

    def test_nan_rejected(self):
        payload = envelope({"type": "chart", "kind": "bar", "series": [{
            "id": "a", "label": "A", "unit": "", "source": "x", "estimate": False,
            "values": [{"label": "p", "value": float("nan")}]}]})
        with self.assertRaises(ComponentError):
            validate_components(payload)

    def test_infinity_rejected(self):
        payload = envelope({"type": "chart", "kind": "bar", "series": [{
            "id": "a", "label": "A", "unit": "", "source": "x", "estimate": False,
            "values": [{"label": "p", "value": float("inf")}]}]})
        with self.assertRaises(ComponentError):
            validate_components(payload)

    def test_string_caps_enforced(self):
        payload = envelope({"type": "decision", "title": "x" * 200,
                            "options": [{"id": "a", "label": "A"}, {"id": "b", "label": "B"}],
                            "selected": "a", "criteria": "ok"})
        with self.assertRaises(ComponentError):
            validate_components(payload)

    def test_decision_must_select_known_option(self):
        with self.assertRaises(ComponentError):
            validate_components(decision_envelope(selected="missing"))

    def test_object_depth_enforced(self):
        payload = envelope({"type": "decision", "options": [
            {"id": "a", "label": "A"}, {"id": "b", "label": "B"}],
            "selected": "a", "criteria": "ok",
            "confidence": {"selected_probability": 0.6,
                           "runner_up": {"id": "b", "probability": 0.3,
                                         "deep": {"nested": {"too": {"deep": 1}}}}}})
        with self.assertRaises(ComponentError):
            validate_components(payload)

    def test_ids_must_match_pattern(self):
        bad = decision_envelope()
        bad["components"][0]["options"].append({"id": "Bad-ID!", "label": "X"})
        with self.assertRaises(ComponentError):
            validate_components(bad)


class ComparisonValidationTests(unittest.TestCase):
    def _comparison(self):
        return {"type": "comparison",
                "columns": [{"id": "c1", "label": "Cost", "kind": "number"}],
                "rows": [{"id": "r1", "label": "Option A", "values": [10]},
                         {"id": "r2", "label": "Option B", "values": [12]}]}

    def test_basic(self):
        comps = validate_components(envelope(self._comparison()))
        self.assertEqual(comps[0]["columns"][0]["kind"], "number")

    def test_cell_count_must_match_columns(self):
        bad = self._comparison()
        bad["rows"][0]["values"] = []
        with self.assertRaises(ComponentError):
            validate_components(envelope(bad))

    def test_unknown_column_id_in_sort(self):
        comp = validate_components(envelope(self._comparison()))[0]
        with self.assertRaises(ComponentError):
            validate_action(comp, "sort", {"column_id": "missing", "direction": "asc"})


class ChartValidationTests(unittest.TestCase):
    def test_kind_enum(self):
        bad = {"type": "chart", "kind": "pie", "series": [{
            "id": "s", "label": "S", "unit": "", "source": "user", "estimate": False,
            "values": [{"label": "a", "value": 1.0}, {"label": "b", "value": 2.0}]}]}
        with self.assertRaises(ComponentError):
            validate_components(envelope(bad))

    def test_series_with_estimates(self):
        ok = {"type": "chart", "kind": "bar", "series": [{
            "id": "s1", "label": "Estimated savings", "unit": "USD",
            "source": "model estimate, based on user's stated budget",
            "estimate": True,
            "values": [{"label": "month 1", "value": 100.0},
                       {"label": "month 2", "value": 120.0}]}]}
        comp = validate_components(envelope(ok))[0]
        self.assertTrue(comp["series"][0]["estimate"])

    def test_values_must_be_finite(self):
        bad = {"type": "chart", "kind": "bar", "series": [{
            "id": "s", "label": "S", "unit": "", "source": "x", "estimate": False,
            "values": [{"label": "a", "value": float("nan")},
                       {"label": "b", "value": 1.0}]}]}
        with self.assertRaises(ComponentError):
            validate_components(envelope(bad))


class FormValidationTests(unittest.TestCase):
    def _text(self):
        return {"id": "f1", "label": "Name", "kind": "text", "required": True}

    def test_basic_text_form(self):
        comps = validate_components(envelope({"type": "form", "fields": [self._text()]}))
        self.assertTrue(comps[0]["fields"][0]["required"])

    def test_choice_membership_required(self):
        bad = {"id": "f1", "label": "X", "kind": "choice", "choices": [
            {"id": "a", "label": "A"}, {"id": "b", "label": "B"}],
            "initial": "z"}
        with self.assertRaises(ComponentError):
            validate_components(envelope({"type": "form", "fields": [bad]}))

    def test_numeric_min_max(self):
        bad = {"id": "f1", "label": "X", "kind": "numeric", "min": 5, "max": 3}
        with self.assertRaises(ComponentError):
            validate_components(envelope({"type": "form", "fields": [bad]}))

    def test_action_submit_validates_form_values(self):
        comp = validate_components(envelope({"type": "form", "fields": [
            {"id": "n", "label": "Name", "kind": "text", "required": True}]}))[0]
        ok = validate_action(comp, "submit", {"values": {"n": "Alice"}})
        self.assertEqual(ok["values"]["values"]["n"], "Alice")
        with self.assertRaises(ComponentError):
            validate_action(comp, "submit", {"values": {"n": ""}})
        with self.assertRaises(ComponentError):
            validate_action(comp, "submit", {"values": {"x": "y"}})

    def test_numeric_submit_validates_bounds(self):
        comp = validate_components(envelope({"type": "form", "fields": [
            {"id": "n", "label": "N", "kind": "numeric", "min": 1, "max": 10}]}))[0]
        with self.assertRaises(ComponentError):
            validate_action(comp, "submit", {"values": {"n": 99}})


class ActionAuthorizationTests(unittest.TestCase):
    def test_unknown_action(self):
        comp = validate_components(decision_envelope())[0]
        with self.assertRaises(ComponentError):
            validate_action(comp, "execute_shell")

    def test_select_action_membership(self):
        comp = validate_components(decision_envelope(selected="opt-1"))[0]
        ok = validate_action(comp, "select", {"option_id": "opt-1"})
        self.assertEqual(ok["values"]["option_id"], "opt-1")
        with self.assertRaises(ComponentError):
            validate_action(comp, "select", {"option_id": "missing"})

    def test_filter_action_query_cap(self):
        comp = validate_components(envelope({"type": "comparison",
            "columns": [{"id": "c1", "label": "X", "kind": "text"}],
            "rows": [{"id": "r1", "label": "Y", "values": ["x"]},
                     {"id": "r2", "label": "Z", "values": ["y"]}]}))[0]
        ok = validate_action(comp, "filter", {"query": "abc"})
        self.assertEqual(ok["values"]["query"], "abc")
        with self.assertRaises(ComponentError):
            validate_action(comp, "filter", {"query": "x" * 200})

    def test_checklist_edit_subset_and_unique(self):
        comp = validate_components(envelope({"type": "checklist", "items": [
            {"id": "t1", "text": "a", "done": False},
            {"id": "t2", "text": "b", "done": True}]}))[0]
        ok = validate_action(comp, "edit", {"items": [{"id": "t1", "done": True}]})
        self.assertEqual(ok["values"]["items"], [{"id": "t1", "done": True}])
        with self.assertRaises(ComponentError):
            validate_action(comp, "edit", {"items": [{"id": "missing", "done": True}]})
        with self.assertRaises(ComponentError):
            validate_action(comp, "edit", {"items": [{"id": "t1", "done": True},
                                                    {"id": "t1", "done": False}]})

    def test_facts_expand_card_id_must_exist(self):
        comp = validate_components(envelope({"type": "facts", "cards": [
            {"id": "k1", "heading": "H", "text": "T"}]}))[0]
        validate_action(comp, "expand", {"card_id": "k1"})
        with self.assertRaises(ComponentError):
            validate_action(comp, "expand", {"card_id": "missing"})

    def test_action_values_size_cap(self):
        comp = validate_components(decision_envelope())[0]
        big = {"description": "x" * 20_000}
        with self.assertRaises(ComponentError):
            validate_action(comp, "filter", big)


class PlainTextTests(unittest.TestCase):
    def test_decision_text_includes_selection(self):
        comp = validate_components(decision_envelope())[0]
        text = plain_text(comp)
        self.assertIn("B", text)
        self.assertIn("lower memory footprint", text)

    def test_chart_text_labels_estimates(self):
        comp = validate_components(envelope({"type": "chart", "kind": "bar", "series": [{
            "id": "s1", "label": "Cost", "unit": "USD", "source": "invoice",
            "estimate": True,
            "values": [{"label": "q1", "value": 12.0}, {"label": "q2", "value": 18.0}]}]}))[0]
        text = plain_text(comp)
        self.assertIn("estimate", text.lower())
        self.assertIn("invoice", text)


if __name__ == "__main__":
    unittest.main()
