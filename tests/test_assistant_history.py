"""Context selection and durable checkpoint behavior for the history store.

UNIT ONLY. No inference: turns are driven directly through
begin/emit/finish/cancel, which is exactly what the coordinator calls.
Covers the shared context-selection contract (set_context, record.context),
restart persistence of that context, and the bounded-checkpoint save policy
that replaced per-token fsync. Hardware and generation behavior live elsewhere.
"""

import codecs
import json
import os
import stat
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "serve"))

from mlx_omarchy_assistant import history as history_module  # noqa: E402
from mlx_omarchy_assistant.history import (  # noqa: E402
    BusyError, CHECKPOINT_BYTES, CHECKPOINT_SECONDS, ConversationStore, EventGap, MAX_HISTORY)

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
                          {"pinned_constraints": "x"},
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


class CorruptLoadTest(HistoryTest):
    def saved_template(self, good_cid, bad_cid):
        path = self.home / HISTORY_DIR / (good_cid + ".json")
        record = json.loads(path.read_text(encoding="utf-8"))
        record["id"] = bad_cid
        return record

    def pad_to_canonical_size(self, record, size):
        base = json.dumps(record, ensure_ascii=False, allow_nan=False).encode("utf-8")
        record["messages"][0]["content"] += "x" * (size - len(base))
        canonical = json.dumps(record, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.assertEqual(len(canonical), size)
        return canonical

    def test_deeply_nested_json_is_skipped(self):
        good = self.new_conversation()
        self.complete_turn(good)
        bad = "b" * 32
        deep = "[" * 200000 + "]" * 200000
        (self.home / HISTORY_DIR / (bad + ".json")).write_bytes(deep.encode("utf-8"))
        failure = []

        def run():
            try:
                with self.assertRaises(RecursionError):
                    json.loads(deep)
                restored = ConversationStore(self.home)
                self.assertIn(good, restored.records)
                self.assertNotIn(bad, restored.records)
            except BaseException as exc:
                failure.append(exc)

        old = threading.stack_size(4 * 1024 * 1024)
        try:
            worker = threading.Thread(target=run, daemon=True)
            worker.start()
        finally:
            threading.stack_size(old)
        worker.join(timeout=60)
        self.assertFalse(worker.is_alive(), "ConversationStore hung on deeply nested JSON")
        if failure:
            raise failure[0]

    def test_json_loads_recursion_error_is_skipped(self):
        good = self.new_conversation()
        self.complete_turn(good)
        bad = "b" * 32
        payload = '{"id": "' + bad + '"}'
        (self.home / HISTORY_DIR / (bad + ".json")).write_bytes(payload.encode("utf-8"))
        real_loads = json.loads

        def fake_loads(text, *args, **kwargs):
            if text == payload:
                raise RecursionError("synthetic depth limit")
            return real_loads(text, *args, **kwargs)

        with mock.patch("mlx_omarchy_assistant.history.json.loads", side_effect=fake_loads):
            restored = ConversationStore(self.home)
        self.assertIn(good, restored.records)
        self.assertNotIn(bad, restored.records)

    def test_vanished_file_between_glob_and_open_is_skipped(self):
        good = self.new_conversation()
        self.complete_turn(good)
        missing = "c" * 32
        missing_path = self.home / HISTORY_DIR / (missing + ".json")
        real_glob = Path.glob

        def glob_with_missing(directory, pattern, *args, **kwargs):
            if directory == self.home / HISTORY_DIR and pattern == "*.json":
                yield missing_path
            yield from real_glob(directory, pattern, *args, **kwargs)

        with mock.patch.object(Path, "glob", glob_with_missing):
            restored = ConversationStore(self.home)
        self.assertIn(good, restored.records)
        self.assertNotIn(missing, restored.records)

    def test_fifo_history_entry_does_not_block_startup(self):
        if not hasattr(os, "mkfifo"):
            self.skipTest("os.mkfifo unavailable")
        good = self.new_conversation()
        self.complete_turn(good)
        fifo_id = "d" * 32
        os.mkfifo(self.home / HISTORY_DIR / (fifo_id + ".json"))
        result = {}

        def run():
            try:
                result["store"] = ConversationStore(self.home)
            except BaseException as exc:
                result["error"] = exc

        worker = threading.Thread(target=run, daemon=True)
        worker.start()
        worker.join(timeout=10)
        self.assertFalse(worker.is_alive(), "ConversationStore blocked on FIFO")
        if "error" in result:
            raise result["error"]
        restored = result["store"]
        self.assertIn(good, restored.records)
        self.assertNotIn(fifo_id, restored.records)

    def test_non_ascii_history_loads_under_c_locale(self):
        good = self.new_conversation()
        user_text = "h\u00e9llo \u2713"
        reply_text = "\u00fcn\u00efcode reply"
        self.complete_turn(good, text=user_text, reply=reply_text)
        code = (
            "import json, locale, sys\n"
            "from pathlib import Path\n"
            "sys.path.insert(0, sys.argv[1])\n"
            "from mlx_omarchy_assistant.history import ConversationStore\n"
            "store = ConversationStore(Path(sys.argv[2]))\n"
            "cid = sys.argv[3]\n"
            "record = store.records.get(cid)\n"
            "print(json.dumps({"
            "'preferred': locale.getpreferredencoding(False), "
            "'loaded': cid in store.records, "
            "'messages': [m['content'] for m in record['messages']] if record else []"
            "}, ensure_ascii=True))\n"
        )
        env = dict(os.environ, LC_ALL="C", LANG="C", PYTHONUTF8="0")
        env.pop("PYTHONIOENCODING", None)
        proc = subprocess.run(
            [sys.executable, "-c", code, str(REPO_ROOT / "serve"), str(self.home), good],
            env=env, capture_output=True, text=True, timeout=60,
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        data = json.loads(proc.stdout)
        if codecs.lookup(data["preferred"]).name == "utf-8":
            self.skipTest("preferred encoding is UTF-8 under LC_ALL=C (%s)" % data["preferred"])
        self.assertTrue(data["loaded"])
        self.assertEqual(data["messages"], [user_text, reply_text])

    def test_non_finite_numbers_are_skipped(self):
        good = self.new_conversation()
        self.complete_turn(good)
        bad = "e" * 32
        bad_path = self.home / HISTORY_DIR / (bad + ".json")
        cases = (
            ("created_at_nan", lambda r: r.update(created_at="__NUM__"), "NaN"),
            ("updated_at_inf", lambda r: r.update(updated_at="__NUM__"), "Infinity"),
            ("updated_at_overflow", lambda r: r.update(updated_at="__NUM__"), "1e400"),
            ("nested_component_nan", lambda r: r["messages"][-1]["components"].append({"score": "__NUM__"}), "NaN"),
        )
        for label, mutate, token in cases:
            with self.subTest(case=label):
                record = self.saved_template(good, bad)
                mutate(record)
                raw = json.dumps(record, ensure_ascii=False).replace('"__NUM__"', token)
                bad_path.write_bytes(raw.encode("utf-8"))
                restored = ConversationStore(self.home)
                self.assertIn(good, restored.records)
                self.assertNotIn(bad, restored.records)
                json.dumps(restored.list(), ensure_ascii=False, allow_nan=False).encode()

    def test_lone_surrogate_escapes_are_skipped(self):
        good = self.new_conversation()
        self.complete_turn(good)
        bad = "a" * 32
        bad_path = self.home / HISTORY_DIR / (bad + ".json")
        cases = (
            ("title_high_surrogate", lambda r: r.update(title="__SUR__"), r"\ud800"),
            ("component_low_surrogate", lambda r: r["messages"][-1]["components"].append({"label": "__SUR__"}), r"\udc80"),
        )
        for label, mutate, escape in cases:
            with self.subTest(case=label):
                record = self.saved_template(good, bad)
                mutate(record)
                raw = json.dumps(record, ensure_ascii=False).replace("__SUR__", escape)
                bad_path.write_bytes(raw.encode("ascii"))
                restored = ConversationStore(self.home)
                self.assertIn(good, restored.records)
                self.assertNotIn(bad, restored.records)
                json.dumps(restored.list(), ensure_ascii=False, allow_nan=False).encode()

    def test_record_depth_cap(self):
        good = self.new_conversation()
        self.complete_turn(good)
        candidate = "1" * 32
        candidate_path = self.home / HISTORY_DIR / (candidate + ".json")
        cap = history_module.MAX_DEPTH
        # Root record dict is depth 1, so cap - 1 nested containers inside
        # context reaches exact container depth cap.
        cases = (
            ("at_cap_list", "[" * (cap - 1) + "]" * (cap - 1), True),
            ("at_cap_dict", '{"a":' * (cap - 1) + "0" + "}" * (cap - 1), True),
            ("over_cap_list", "[" * cap + "]" * cap, False),
            ("over_cap_dict", '{"a":' * cap + "0" + "}" * cap, False),
            ("deep_495_list", "[" * 494 + "]" * 494, False),
        )
        template = self.saved_template(good, candidate)
        template["context"] = "__DEEP__"
        template_raw = json.dumps(template, ensure_ascii=False)
        for label, nested, should_load in cases:
            with self.subTest(case=label):
                raw = template_raw.replace('"__DEEP__"', nested)
                candidate_path.write_bytes(raw.encode("utf-8"))
                restored = ConversationStore(self.home)
                self.assertIn(good, restored.records)
                if should_load:
                    self.assertIn(candidate, restored.records)
                    self.assertEqual(restored.get(candidate)["id"], candidate)
                else:
                    self.assertNotIn(candidate, restored.records)

    def test_symlink_history_entry_is_skipped(self):
        if not hasattr(os, "symlink"):
            self.skipTest("os.symlink unavailable")
        good = self.new_conversation()
        self.complete_turn(good)
        link_id = "3" * 32
        target = self.home / "outside.json"
        target.write_bytes(json.dumps(self.saved_template(good, link_id), ensure_ascii=False).encode("utf-8"))
        os.symlink(target, self.home / HISTORY_DIR / (link_id + ".json"))
        restored = ConversationStore(self.home)
        self.assertIn(good, restored.records)
        self.assertNotIn(link_id, restored.records)

    def test_max_history_size_boundary_and_oversized_entry(self):
        good = self.new_conversation()
        self.complete_turn(good)
        exact_id = "4" * 32
        over_id = "5" * 32
        exact_payload = self.pad_to_canonical_size(self.saved_template(good, exact_id), MAX_HISTORY)
        (self.home / HISTORY_DIR / (exact_id + ".json")).write_bytes(exact_payload)

        over_payload = json.dumps(self.saved_template(good, over_id), ensure_ascii=False, allow_nan=False).encode("utf-8")
        over_padded = over_payload + b" " * (MAX_HISTORY + 1 - len(over_payload))
        self.assertEqual(len(over_padded), MAX_HISTORY + 1)
        (self.home / HISTORY_DIR / (over_id + ".json")).write_bytes(over_padded)

        restored = ConversationStore(self.home)
        self.assertIn(good, restored.records)
        self.assertIn(exact_id, restored.records)
        self.assertNotIn(over_id, restored.records)
        restored.set_saved(exact_id, True)

    def test_post_read_size_guard_skips_oversized_entry(self):
        good = self.new_conversation()
        self.complete_turn(good)
        over_id = "8" * 32
        payload = json.dumps(self.saved_template(good, over_id), ensure_ascii=False, allow_nan=False).encode("utf-8")
        padded = payload + b" " * (MAX_HISTORY + 1 - len(payload))
        self.assertEqual(len(padded), MAX_HISTORY + 1)
        (self.home / HISTORY_DIR / (over_id + ".json")).write_bytes(padded)

        real_fstat = os.fstat

        def fake_fstat(fd):
            values = list(real_fstat(fd))
            values[stat.ST_SIZE] = MAX_HISTORY
            return os.stat_result(values)

        with mock.patch("mlx_omarchy_assistant.history.os.fstat", side_effect=fake_fstat):
            restored = ConversationStore(self.home)
        self.assertIn(good, restored.records)
        self.assertNotIn(over_id, restored.records)

    def test_normalization_growth_past_max_history_is_skipped(self):
        good = self.new_conversation()
        self.complete_turn(good)
        bad = "9" * 32
        bad_path = self.home / HISTORY_DIR / (bad + ".json")
        cases = (
            ("malformed_context", lambda r: r.update(context=1)),
            ("missing_context", lambda r: r.pop("context")),
        )
        for label, mutate in cases:
            with self.subTest(case=label):
                record = self.saved_template(good, bad)
                mutate(record)
                bad_path.write_bytes(self.pad_to_canonical_size(record, MAX_HISTORY))
                restored = ConversationStore(self.home)
                self.assertIn(good, restored.records)
                self.assertNotIn(bad, restored.records)

    def test_compact_file_exceeding_canonical_max_history_is_skipped(self):
        good = self.new_conversation()
        self.complete_turn(good)
        bad = "7" * 32
        record = self.saved_template(good, bad)
        base_compact = json.dumps(record, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        record["messages"][0]["content"] += "x" * (MAX_HISTORY - len(base_compact))
        compact = json.dumps(record, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        canonical = json.dumps(record, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.assertLessEqual(len(compact), MAX_HISTORY)
        self.assertGreater(len(canonical), MAX_HISTORY)
        (self.home / HISTORY_DIR / (bad + ".json")).write_bytes(compact)
        restored = ConversationStore(self.home)
        self.assertIn(good, restored.records)
        self.assertNotIn(bad, restored.records)

    def test_fifo_with_valid_record_is_skipped(self):
        if not hasattr(os, "mkfifo") or not hasattr(os, "O_NONBLOCK"):
            self.skipTest("non-blocking FIFO unavailable")
        good = self.new_conversation()
        self.complete_turn(good)
        fifo_id = "6" * 32
        fifo_path = self.home / HISTORY_DIR / (fifo_id + ".json")
        os.mkfifo(fifo_path)
        payload = json.dumps(self.saved_template(good, fifo_id), ensure_ascii=False).encode("utf-8")
        result = {}

        def run():
            try:
                result["store"] = ConversationStore(self.home)
            except BaseException as exc:
                result["error"] = exc

        wfd = os.open(fifo_path, os.O_RDWR | os.O_NONBLOCK)
        try:
            self.assertEqual(os.write(wfd, payload), len(payload))
            worker = threading.Thread(target=run, daemon=True)
            worker.start()
            worker.join(timeout=5)
            blocked = worker.is_alive()
        finally:
            os.close(wfd)
        self.assertFalse(blocked, "ConversationStore blocked on FIFO")
        if "error" in result:
            raise result["error"]
        restored = result["store"]
        self.assertIn(good, restored.records)
        self.assertNotIn(fifo_id, restored.records)

    def test_invalid_record_shapes_are_skipped(self):
        good = self.new_conversation()
        self.complete_turn(good)
        bad = "f" * 32
        bad_path = self.home / HISTORY_DIR / (bad + ".json")
        cases = (
            ("missing_title", lambda r: r.pop("title")),
            ("missing_created_at", lambda r: r.pop("created_at")),
            ("bool_created_at", lambda r: r.update(created_at=True)),
            ("str_updated_at", lambda r: r.update(updated_at="2026")),
            ("bool_updated_at", lambda r: r.update(updated_at=True)),
            ("null_updated_at", lambda r: r.update(updated_at=None)),
            ("str_sequence", lambda r: r.update(sequence="7")),
            ("bool_sequence", lambda r: r.update(sequence=True)),
            ("negative_sequence", lambda r: r.update(sequence=-1)),
            ("non_dict_message", lambda r: r["messages"].append("x")),
            ("int_message_content", lambda r: r["messages"][0].update(content=5)),
            ("missing_message_role", lambda r: r["messages"][0].pop("role")),
            ("missing_message_turn_id", lambda r: r["messages"][0].pop("turn_id")),
            ("missing_message_status", lambda r: r["messages"][0].pop("status")),
        )
        for label, mutate in cases:
            with self.subTest(case=label):
                record = self.saved_template(good, bad)
                mutate(record)
                bad_path.write_bytes(json.dumps(record, ensure_ascii=False).encode("utf-8"))
                restored = ConversationStore(self.home)
                self.assertIn(good, restored.records)
                self.assertNotIn(bad, restored.records)

    def _assert_emit_depth_limit(self, kind, start_depth):
        cap = history_module.MAX_DEPTH

        def nested(levels, wrap):
            value = 0
            for _ in range(levels):
                value = wrap(value)
            return value

        over_dict = nested(cap - start_depth + 2, lambda v: {"a": v})
        over_tuple = {"details": nested(cap - start_depth + 1, lambda v: (v,))}
        at_limit = nested(cap - start_depth + 1, lambda v: {"a": v})

        clock = self.frozen_clock()
        cid = self.new_conversation()
        turn = self.store.begin(cid, "q")
        path = self.home / HISTORY_DIR / (cid + ".json")
        before_record = self.store.get(cid)
        before_bytes = path.read_bytes()
        before_pending = list(self.store.pending[cid])
        clock.monotonic.return_value = 500.0 + CHECKPOINT_SECONDS + 1

        for shape, over_deep in (("dict", over_dict), ("tuple", over_tuple)):
            with self.subTest(shape=shape):
                with self.assertRaises(ValueError):
                    self.store.emit(cid, turn, kind, over_deep)
                after_record = self.store.get(cid)
                self.assertEqual(after_record["messages"][-1], before_record["messages"][-1])
                self.assertEqual(after_record["sequence"], before_record["sequence"])
                self.assertEqual(after_record.get("updated_at"), before_record.get("updated_at"))
                self.assertEqual(self.store.pending[cid], before_pending)
                self.assertEqual(path.read_bytes(), before_bytes)

        self.assertTrue(self.store.emit(cid, turn, kind, at_limit))
        self.store.finish(cid, turn)
        restored = ConversationStore(self.home)
        self.assertIn(cid, restored.records)
        restored_message = restored.get(cid)["messages"][-1]
        if kind == "decision":
            self.assertEqual(restored_message["decision"], at_limit)
        else:
            self.assertEqual(restored_message["components"], [at_limit])

    def test_emit_decision_depth_limit(self):
        self._assert_emit_depth_limit("decision", 4)

    def test_emit_component_depth_limit(self):
        self._assert_emit_depth_limit("component", 5)


if __name__ == "__main__":
    unittest.main()
