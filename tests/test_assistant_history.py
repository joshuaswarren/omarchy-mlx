"""Context selection and durable checkpoint behavior for the history store.

UNIT ONLY. No inference: turns are driven directly through
begin/emit/finish/cancel, which is exactly what the coordinator calls.
Covers the shared context-selection contract (set_context, record.context),
restart persistence of that context, and the bounded-checkpoint save policy
that replaced per-token fsync. Hardware and generation behavior live elsewhere.
"""

import json
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "serve"))

from mlx_omarchy_assistant.history import (  # noqa: E402
    BusyError, CHECKPOINT_BYTES, CHECKPOINT_SECONDS, ConversationStore, EventGap)

DEFAULT_CONTEXT = {"selected_turn_ids": None, "pinned_constraints": ""}
HISTORY_DIR = "assistant/history"


class HistoryTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.home = Path(self.tmp.name)
        self.store = ConversationStore(self.home)

    def new_conversation(self, save=True):
        return self.store.create(save=save)["id"]

    def complete_turn(self, cid, text="question", reply="answer"):
        turn = self.store.begin(cid, text)
        self.store.emit(cid, turn, "text", {"text": reply})
        self.store.finish(cid, turn)
        return turn

    def frozen_clock(self):
        """Real wall time for updated_at; monotonic pinned so time-bound saves never fire."""
        clock = mock.patch("mlx_omarchy_assistant.history.time").start()
        self.addCleanup(mock.patch.stopall)
        clock.time.side_effect = time.time
        clock.monotonic.return_value = 500.0
        return clock


class ContextSelectionTest(HistoryTest):
    def test_create_defaults_to_all_turns(self):
        record = self.store.get(self.new_conversation())
        self.assertEqual(record["context"], DEFAULT_CONTEXT)

    def test_select_completed_turns_and_pin_constraints(self):
        cid = self.new_conversation()
        first = self.complete_turn(cid, text="q1", reply="a1")
        second = self.complete_turn(cid, text="q2", reply="a2")
        record = self.store.set_context(cid, [second, first], "Reply only from the pinned manual excerpt.")
        self.assertEqual(record["context"], {
            "selected_turn_ids": [second, first],
            "pinned_constraints": "Reply only from the pinned manual excerpt."})
        self.assertEqual(self.store.get(cid)["context"]["selected_turn_ids"], [second, first])

    def test_null_and_empty_selection(self):
        cid = self.new_conversation()
        self.complete_turn(cid)
        self.assertEqual(self.store.set_context(cid, [], "")["context"]["selected_turn_ids"], [])
        self.assertIsNone(self.store.set_context(cid, None, "")["context"]["selected_turn_ids"])

    def test_rejects_unknown_duplicate_nonstring_ids(self):
        cid = self.new_conversation()
        turn = self.complete_turn(cid)
        for bad in (["0" * 32], [turn, turn], [42], [None], turn, 5):
            with self.assertRaises(ValueError, msg=repr(bad)):
                self.store.set_context(cid, bad, "")
        self.assertEqual(self.store.get(cid)["context"], DEFAULT_CONTEXT)

    def test_rejects_active_turn_id_with_busy_error(self):
        cid = self.new_conversation()
        self.complete_turn(cid)
        turn = self.store.begin(cid, "working question")
        with self.assertRaises(BusyError):
            self.store.set_context(cid, [turn], "")
        self.store.finish(cid, turn)
        self.assertEqual(self.store.set_context(cid, [turn], "")["context"]["selected_turn_ids"], [turn])

    def test_rejects_turn_from_other_conversation(self):
        foreign = self.complete_turn(self.new_conversation())
        cid = self.new_conversation()
        with self.assertRaises(ValueError):
            self.store.set_context(cid, [foreign], "")

    def test_pinned_constraints_bounded_in_bytes(self):
        cid = self.new_conversation()
        self.complete_turn(cid)
        with self.assertRaises(ValueError):
            self.store.set_context(cid, None, "x" * (64 * 1024 + 1))
        with self.assertRaises(ValueError):
            self.store.set_context(cid, None, 7)
        bounded = self.store.set_context(cid, None, "x" * (64 * 1024))
        self.assertEqual(len(bounded["context"]["pinned_constraints"].encode()), 64 * 1024)

    def test_set_context_persists_and_survives_restart(self):
        cid = self.new_conversation()
        turn = self.complete_turn(cid)
        self.store.set_context(cid, [turn], "Never invent order numbers.")
        restored = ConversationStore(self.home).get(cid)
        self.assertEqual(restored["context"],
                         {"selected_turn_ids": [turn], "pinned_constraints": "Never invent order numbers."})

    def test_older_saved_record_defaults_context(self):
        cid = self.new_conversation()
        self.complete_turn(cid)
        path = self.home / HISTORY_DIR / (cid + ".json")
        legacy = json.loads(path.read_text())
        legacy.pop("context", None)
        path.write_text(json.dumps(legacy))
        record = ConversationStore(self.home).get(cid)
        self.assertEqual(record["context"], DEFAULT_CONTEXT)
        self.assertNotIn("context_error", record)
        self.assertEqual(len(record["messages"]), 2)

    def test_malformed_saved_context_refuses_by_name_and_preserves_raw(self):
        cid = self.new_conversation()
        turn = self.complete_turn(cid)
        self.store.set_context(cid, [turn], "Keep this pinned rule.")
        path = self.home / HISTORY_DIR / (cid + ".json")
        damaged_values = ({"selected_turn_ids": 5, "pinned_constraints": "x"},
                          {"selected_turn_ids": [turn], "pinned_constraints": None},
                          {"selected_turn_ids": [turn], "pinned_constraints": "x" * (64 * 1024 + 1)},
                          {"selected_turn_ids": [5], "pinned_constraints": "x"},
                          "pinned", None)
        for damaged in damaged_values:
            record = json.loads(path.read_text())
            record["context"] = damaged
            path.write_text(json.dumps(record))
            loaded = ConversationStore(self.home).get(cid)
            self.assertEqual(loaded["context"], DEFAULT_CONTEXT)
            self.assertIsInstance(loaded.get("context_error"), str)
            self.assertTrue(loaded["context_error"])
            self.assertEqual(loaded["context_raw"], damaged)
            self.assertEqual(loaded["messages"][-1]["content"], "answer")

    def test_dangling_saved_selection_refuses_by_name(self):
        cid = self.new_conversation()
        self.complete_turn(cid)
        path = self.home / HISTORY_DIR / (cid + ".json")
        record = json.loads(path.read_text())
        record["context"] = {"selected_turn_ids": ["f" * 32], "pinned_constraints": "pin"}
        path.write_text(json.dumps(record))
        loaded = ConversationStore(self.home).get(cid)
        self.assertEqual(loaded["context"], DEFAULT_CONTEXT)
        self.assertIn("withheld", loaded["context_error"])
        self.assertEqual(loaded["context_raw"]["pinned_constraints"], "pin")

    def test_explicit_repair_clears_error_and_survives_restart(self):
        cid = self.new_conversation()
        turn = self.complete_turn(cid)
        path = self.home / HISTORY_DIR / (cid + ".json")
        record = json.loads(path.read_text())
        record["context"] = {"selected_turn_ids": 5, "pinned_constraints": "x"}
        path.write_text(json.dumps(record))
        loaded = ConversationStore(self.home)
        self.assertIn("context_error", loaded.get(cid))
        with self.assertRaises(ValueError):
            loaded.set_context(cid, [turn], "x" * (64 * 1024 + 1))  # failed repair keeps the refusal
        self.assertIn("context_error", loaded.get(cid))
        repaired = loaded.set_context(cid, [turn], "Repaired pinned rule.")
        self.assertNotIn("context_error", repaired)
        self.assertNotIn("context_raw", repaired)
        restored = ConversationStore(self.home).get(cid)
        self.assertEqual(restored["context"],
                         {"selected_turn_ids": [turn], "pinned_constraints": "Repaired pinned rule."})
        self.assertNotIn("context_error", restored)

    def test_valid_saved_context_loads_without_error(self):
        cid = self.new_conversation()
        turn = self.complete_turn(cid)
        self.store.set_context(cid, [turn], "Cite the pinned spec only.")
        restored = ConversationStore(self.home).get(cid)
        self.assertEqual(restored["context"]["pinned_constraints"], "Cite the pinned spec only.")
        self.assertNotIn("context_error", restored)
        self.assertNotIn("context_raw", restored)

    def test_export_includes_context_without_dropping_constraints(self):
        cid = self.new_conversation()
        turn = self.complete_turn(cid)
        self.store.set_context(cid, [turn], "Cite the pinned spec only.")
        exported = self.store.get(cid)
        self.assertEqual(exported["context"]["pinned_constraints"], "Cite the pinned spec only.")
        self.assertEqual(exported["context"]["selected_turn_ids"], [turn])
        exported["context"]["pinned_constraints"] = "tampered"
        self.assertEqual(self.store.get(cid)["context"]["pinned_constraints"], "Cite the pinned spec only.")

    def test_unsaved_conversation_stays_off_disk(self):
        cid = self.new_conversation(save=False)
        turn = self.complete_turn(cid)
        self.store.set_context(cid, [turn], "memory only")
        self.assertFalse((self.home / HISTORY_DIR / (cid + ".json")).exists())


class CheckpointTest(HistoryTest):
    def test_streamed_tokens_defer_disk_writes_within_bounds(self):
        self.frozen_clock()
        cid = self.new_conversation()
        turn = self.store.begin(cid, "count")
        with mock.patch.object(self.store, "_save", wraps=self.store._save) as saves:
            for _ in range(50):
                self.store.emit(cid, turn, "text", {"text": "word "})
            self.assertEqual(saves.call_count, 0)
        self.assertEqual(self.store.get(cid)["messages"][-1]["content"], "word " * 50)

    def test_size_bound_triggers_atomic_checkpoint(self):
        cid = self.new_conversation()
        turn = self.store.begin(cid, "big")
        with mock.patch.object(self.store, "_save", wraps=self.store._save) as saves:
            self.store.emit(cid, turn, "text", {"text": "x" * (CHECKPOINT_BYTES + 1)})
            self.assertGreaterEqual(saves.call_count, 1)
        on_disk = json.loads((self.home / HISTORY_DIR / (cid + ".json")).read_text())
        self.assertEqual(on_disk["messages"][-1]["content"], "x" * (CHECKPOINT_BYTES + 1))
        self.assertEqual(on_disk["active_turn"], turn)
        restored = ConversationStore(self.home).get(cid)
        self.assertEqual(restored["messages"][-1]["status"], "stopped")
        self.assertEqual(restored["messages"][-1]["content"], "x" * (CHECKPOINT_BYTES + 1))

    def test_time_bound_triggers_checkpoint(self):
        clock = self.frozen_clock()
        cid = self.new_conversation()
        turn = self.store.begin(cid, "slow")
        with mock.patch.object(self.store, "_save", wraps=self.store._save) as saves:
            self.store.emit(cid, turn, "text", {"text": "a"})
            self.assertEqual(saves.call_count, 0)
            clock.monotonic.return_value = 500.0 + CHECKPOINT_SECONDS + 1
            self.store.emit(cid, turn, "text", {"text": "b"})
            self.assertEqual(saves.call_count, 1)
        self.assertEqual(self.store.get(cid)["messages"][-1]["content"], "ab")

    def test_finish_flushes_everything(self):
        cid = self.new_conversation()
        turn = self.store.begin(cid, "q")
        self.store.emit(cid, turn, "text", {"text": "partial answer"})
        self.store.emit(cid, turn, "decision", {"kind": "plan", "steps": ["one"]})
        self.store.finish(cid, turn)
        on_disk = json.loads((self.home / HISTORY_DIR / (cid + ".json")).read_text())
        self.assertEqual(on_disk["messages"][-1]["content"], "partial answer")
        self.assertEqual(on_disk["messages"][-1]["decision"], {"kind": "plan", "steps": ["one"]})
        self.assertEqual(on_disk["messages"][-1]["status"], "complete")
        self.assertIsNone(on_disk["active_turn"])
        sequence = on_disk["sequence"]
        restored = ConversationStore(self.home)
        self.assertEqual(restored.get(cid)["sequence"], sequence)
        self.assertEqual(restored.events(cid, sequence), [])

    def test_cancel_flushes_partial_content(self):
        cid = self.new_conversation()
        turn = self.store.begin(cid, "q")
        self.store.emit(cid, turn, "text", {"text": "spoken so far"})
        self.assertTrue(self.store.cancel(cid, turn))
        on_disk = json.loads((self.home / HISTORY_DIR / (cid + ".json")).read_text())
        self.assertEqual(on_disk["messages"][-1]["content"], "spoken so far")
        self.store.finish(cid, turn, stopped=True)
        restored = ConversationStore(self.home).get(cid)
        self.assertEqual(restored["messages"][-1]["status"], "stopped")
        self.assertEqual(restored["messages"][-1]["content"], "spoken so far")

    def test_crash_mid_stream_resumes_as_stopped_with_bounded_loss(self):
        cid = self.new_conversation()
        turn = self.store.begin(cid, "q")
        self.store.emit(cid, turn, "text", {"text": "kept"})  # under both bounds: not yet on disk
        restored = ConversationStore(self.home)
        record = restored.get(cid)
        self.assertIsNone(record["active_turn"])
        self.assertEqual(record["messages"][-1]["status"], "stopped")
        self.assertEqual(record["messages"][-1]["content"], "")
        follow = restored.begin(cid, "next question")
        restored.finish(cid, follow)
        self.assertEqual([m["status"] for m in restored.get(cid)["messages"]],
                         ["complete", "stopped", "complete", "complete"])

    def test_restart_preserves_sequence_and_gap_semantics(self):
        cid = self.new_conversation()
        self.complete_turn(cid, text="q", reply="kept")
        sequence = self.store.get(cid)["sequence"]
        restored = ConversationStore(self.home)
        record = restored.get(cid)
        self.assertIsNone(record["active_turn"])
        self.assertEqual(record["messages"][-1]["status"], "complete")
        self.assertEqual(record["sequence"], sequence)
        self.assertEqual(restored.events(cid, sequence), [])
        with self.assertRaises(EventGap):
            restored.events(cid, sequence - 1)


if __name__ == "__main__":
    unittest.main()
