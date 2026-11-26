import threading
import tempfile
import unittest
from pathlib import Path

from mlx_omarchy_assistant.coordinator import Coordinator, ConversationStore, BusyError, DecisionInputError, validate_decision, explanation_disagrees, format_decision_event


class ConversationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.home = Path(self.directory.name)
        self.store = ConversationStore(self.home)

    def test_history_requires_opt_in_and_delete_removes_saved_data(self):
        private = self.store.create()
        self.store.begin(private['id'], 'private message')
        self.assertEqual(ConversationStore(self.home).list(), [])
        saved = self.store.create(save=True)
        turn = self.store.begin(saved['id'], 'keep this')
        self.store.emit(saved['id'], turn, 'text', {'text': 'saved answer'})
        self.store.finish(saved['id'], turn)
        restored = ConversationStore(self.home)
        messages = restored.get(saved['id'])['messages']
        self.assertEqual([m['content'] for m in messages], ['keep this', 'saved answer'])
        restored.delete(saved['id'])
        self.assertEqual(ConversationStore(self.home).list(), [])
    def test_history_summary_counts_messages_and_tracks_updates(self):
        cid = self.store.create(save=True)['id']
        before = self.store.list()[0]
        self.assertEqual(before['message_count'], 0)
        turn = self.store.begin(cid, 'hello')
        self.store.emit(cid, turn, 'text', {'text': 'answer'})
        self.store.finish(cid, turn)
        after = ConversationStore(self.home).list()[0]
        self.assertEqual(after['message_count'], 2)
        self.assertGreaterEqual(after['updated_at'], before['updated_at'])

    def test_busy_cancel_and_late_result_do_not_overwrite_new_turn(self):
        cid = self.store.create()['id']
        first = self.store.begin(cid, 'first')
        with self.assertRaises(BusyError):
            self.store.begin(cid, 'overlap')
        self.store.emit(cid, first, 'text', {'text': 'partial'})
        self.store.cancel(cid, first)
        self.assertFalse(self.store.emit(cid, first, 'text', {'text': 'late'}))
        self.store.finish(cid, first, stopped=True)
        second = self.store.begin(cid, 'second')
        self.assertFalse(self.store.emit(cid, first, 'done', {}))
        self.store.emit(cid, second, 'text', {'text': 'new'})
        self.store.finish(cid, second)
        self.assertEqual([m['content'] for m in self.store.get(cid)['messages']], ['first', 'partial', 'second', 'new'])

    def test_resume_replays_events_without_starting_work(self):
        cid = self.store.create()['id']
        turn = self.store.begin(cid, 'hello')
        self.store.emit(cid, turn, 'text', {'text': 'one'})
        cursor = self.store.events(cid, 0)[-1]['sequence']
        self.store.emit(cid, turn, 'text', {'text': ' two'})
        resumed = self.store.events(cid, cursor)
        self.assertEqual([e['data']['text'] for e in resumed], [' two'])
        self.assertEqual(self.store.get(cid)['active_turn'], turn)

    def test_restart_marks_interrupted_turn_stopped(self):
        cid = self.store.create(save=True)['id']
        turn = self.store.begin(cid, 'question')
        self.store.emit(cid, turn, 'text', {'text': 'partial'})
        restored = ConversationStore(self.home).get(cid)
        self.assertIsNone(restored['active_turn'])
        self.assertEqual(restored['messages'][-1]['status'], 'stopped')

    def test_path_like_id_cannot_read_or_delete_other_files(self):
        for cid in ('../private', '/tmp/private', '', 'a' * 100):
            with self.assertRaises(ValueError):
                self.store.get(cid)


class DecisionTests(unittest.TestCase):
    def test_preserves_options_and_rejects_invalid_distribution(self):
        options = [{'id': 'a', 'label': 'Keep data'}, {'id': 'b', 'label': 'Delete data'}]
        result = {'type': 'choice', 'choice': 'a', 'probabilities': {'a': 0.7, 'b': 0.3},
                  'confidence': 0.1, 'rl_agent': {'act_probability': 0.9}}
        self.assertEqual(validate_decision(result, options)['choice'], 'a')
        for invalid in ({'a': 0.7, 'c': 0.3}, {'a': float('nan'), 'b': 0.3}, {'a': 0.9, 'b': 0.9}):
            with self.assertRaises(DecisionInputError):
                validate_decision(dict(result, probabilities=invalid), options)

    def test_abstention_signal_is_not_entropy_confidence(self):
        options = [{'id': 'a', 'label': 'A'}, {'id': 'b', 'label': 'B'}]
        result = {'type': 'choice', 'choice': 'a', 'probabilities': {'a': 1.0, 'b': 0.0},
                  'confidence': 1.0, 'rl_agent': {'act_probability': 0.01}}
        self.assertTrue(validate_decision(result, options)['abstained'])


class WorkerLifetimeTests(unittest.TestCase):
    def test_failed_shutdown_finishes_turn_but_blocks_new_work(self):
        class FailedManager:
            def start(self):
                raise RuntimeError('worker connection lost')

            def stop(self):
                raise RuntimeError('worker still alive')

        with tempfile.TemporaryDirectory() as directory:
            app = Coordinator(Path(directory), FailedManager())
            cid = app.store.create()['id']
            turn = app.store.begin(cid, 'hello')
            app.gpu.acquire()
            app._run(cid, turn, {}, 1, {'cancel': threading.Event()})
            self.assertIsNone(app.store.get(cid)['active_turn'])
            self.assertTrue(app.gpu.acquire(blocking=False))
            app.gpu.release()
            with self.assertRaisesRegex(RuntimeError, 'worker still alive'):
                app.submit(cid, {'text': 'must not overlap'})
            app.closed.set()
            app.watch.join(3)

if __name__ == '__main__':
    unittest.main()


class DisagreementTests(unittest.TestCase):
    def decision(self, choice="short"):
        return {"type": "choice", "choice": choice, "abstained": False,
                "options": [{"id": "short", "label": "short answer"},
                            {"id": "long", "label": "long answer"}]}

    def test_explanation_disagreement_is_conservative(self):
        decision = self.decision()
        self.assertTrue(explanation_disagrees(decision, "I choose the long answer."))
        self.assertFalse(explanation_disagrees(decision, "I choose the short answer."))
        self.assertFalse(explanation_disagrees(decision, "long answer and short answer both exist."))
        self.assertFalse(explanation_disagrees(decision, "The long answer is mentioned."))
        abstained = dict(decision, abstained=True)
        self.assertFalse(explanation_disagrees(abstained, "I choose the long answer."))
        self.assertFalse(explanation_disagrees({"type": "typed"}, "I choose the long answer."))


    def test_terminal_decision_text_keeps_the_laya_choice(self):
        rendered = format_decision_event({
            "type": "choice", "choice": "short", "criteria": "lower latency",
            "options": [{"id": "short", "label": "short answer"},
                        {"id": "long", "label": "long answer"}],
            "model": "laya-mlx", "abstained": False})
        self.assertIn("* short answer", rendered)
        self.assertIn("long answer", rendered)
        self.assertNotIn("* long answer", rendered)
        self.assertIn("Decided by: laya-mlx", rendered)
        typed = format_decision_event({
            "type": "typed",
            "results": [{"type": "choice", "instructions": "Which?",
                         "answer": {"choice": "billing", "abstained": True}}]})
        self.assertIn("Which?: billing", typed)
        self.assertIn("abstained", typed)
        self.assertEqual(format_decision_event({"type": "other"}), "")
