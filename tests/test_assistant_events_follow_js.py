"""Executable JS test for the chat events-stream follower.

Runs tests/js/events-follow.test.mjs under bun (or node) and fails on a
nonzero exit. The .mjs file asserts that followEvents reopens the stream from
the last sequence while a turn is active, backs off, and stops when idle.
"""

import subprocess
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
JS_TEST = REPO_ROOT / "tests" / "js" / "events-follow.test.mjs"


def _runner():
    for name in ("bun", "node"):
        try:
            resolved = subprocess.run(["which", name], check=True,
                                      capture_output=True, text=True).stdout.strip()
            if resolved:
                return resolved
        except (FileNotFoundError, subprocess.CalledProcessError):
            continue
    return None


class EventsFollowJsTests(unittest.TestCase):
    def test_events_follow_assertions_pass(self):
        runner = _runner()
        if runner is None:
            self.skipTest("no JS runtime available")
        result = subprocess.run([runner, str(JS_TEST)],
                                capture_output=True, text=True, timeout=60)
        self.assertEqual(
            result.returncode, 0,
            f"JS assertions failed:\nstdout: {result.stdout}\nstderr: {result.stderr}")
        self.assertIn("events follow js tests passed", result.stdout)


if __name__ == "__main__":
    unittest.main()
