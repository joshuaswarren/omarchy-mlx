"""The coordinator's decision-config read and per-turn timing log append.

Every input is finite: sparse files stay about 1 MiB, and each FIFO open runs
under a SIGALRM watchdog so a regression fails instead of hanging the run."""

import contextlib
import json
import os
import signal
import stat
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "serve"))

from mlx_omarchy_assistant.coordinator import Coordinator, DecisionInputError, _typed_request
from mlx_omarchy_laya import jsonfile, sequence


class Hung(BaseException):
    """Raised by the watchdog; BaseException so no except Exception swallows it."""


@contextlib.contextmanager
def watchdog(seconds=5):
    def fire(signum, frame):
        raise Hung("blocked for %s s" % seconds)
    previous = signal.signal(signal.SIGALRM, fire)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


class ConfigRead(Exception):
    """Stands in for the tokenizer: reaching it means the config loaded."""


QUESTIONS = {"q": {"type": "choice", "instructions": "Pick one", "criteria": {"a": "A", "b": "B"}}}


class DecisionConfigTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.model = Path(temp.name) / "model"
        self.model.mkdir()
        self.config = self.model / "rl_agent_config.json"
        tokenizer = mock.patch.object(sequence, "LayaTokenizer", side_effect=ConfigRead)
        tokenizer.start()
        self.addCleanup(tokenizer.stop)

    def build(self):
        return _typed_request(self.model, "Some material", QUESTIONS)

    def assertRefused(self, pattern):
        with self.assertRaisesRegex(ValueError, pattern) as caught:
            self.build()
        # Not an input error: automatic routing would report a broken model
        # file as material that does not fit.
        self.assertNotIsInstance(caught.exception, DecisionInputError)

    def test_regular_and_snapshot_symlinked_configs_load(self):
        self.config.write_text(json.dumps({"max_len": 512, "head_max_len": 512}))
        with self.assertRaises(ConfigRead):
            self.build()
        blob = self.model.parent / "blob"
        self.config.rename(blob)
        self.config.symlink_to(blob)
        with self.assertRaises(ConfigRead):
            self.build()

    def test_fifo_is_refused_without_blocking(self):
        os.mkfifo(self.config)
        with watchdog():
            self.assertRefused("not a regular file")

    def test_oversized_config_is_refused(self):
        with open(self.config, "wb") as stream:
            stream.truncate((1 << 20) + 1)
        self.assertRefused("exceeds")

    def test_malformed_configs_are_refused(self):
        for raw, pattern in ((b"\xff{}", "utf-8"), (b"{", "Expecting"), (b"[1]", "not a JSON object")):
            self.config.write_bytes(raw)
            self.assertRefused(pattern)

    def test_nesting_overflow_is_a_value_error(self):
        self.config.write_text("{}")

        def overflow(path):
            raise RecursionError("depth probe")
        with mock.patch.object(jsonfile, "read_json_object", overflow):
            self.assertRefused("nested too deeply")


class TimingLogTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.home = Path(temp.name)
        self.logs = self.home / "assistant" / "logs"
        self.log = self.logs / "ttft-phases.jsonl"
        self.owner = types.SimpleNamespace(home=self.home)

    def dump(self, turn):
        Coordinator._ttft_dump(self.owner, "c1", turn, "chat", {"submit": 1.5})

    def test_lines_append_to_a_private_regular_file(self):
        self.dump("t1")
        self.dump("t2")
        lines = [json.loads(line) for line in self.log.read_text().splitlines()]
        self.assertEqual([line["turn"] for line in lines], ["t1", "t2"])
        self.assertEqual(lines[0], {"turn": "t1", "conversation": "c1", "mode": "chat",
                                    "phases": {"submit": 1.5}})
        self.assertEqual(stat.S_IMODE(self.log.stat().st_mode) & 0o077, 0)

    def test_symlink_is_not_followed(self):
        self.logs.mkdir(parents=True)
        target = self.home / "elsewhere.txt"
        target.write_text("keep\n")
        self.log.symlink_to(target)
        self.dump("t1")
        self.assertEqual(target.read_text(), "keep\n")
        self.assertTrue(self.log.is_symlink())

    def test_fifo_without_a_reader_does_not_block(self):
        self.logs.mkdir(parents=True)
        os.mkfifo(self.log)
        with watchdog():
            self.dump("t1")

    def test_fifo_with_a_reader_gets_nothing(self):
        self.logs.mkdir(parents=True)
        os.mkfifo(self.log)
        reader = os.open(self.log, os.O_RDONLY | os.O_NONBLOCK)
        self.addCleanup(os.close, reader)
        with watchdog():
            self.dump("t1")
            # Every writer has closed, so an empty FIFO reads as EOF.
            self.assertEqual(os.read(reader, 4096), b"")


if __name__ == "__main__":
    unittest.main()
