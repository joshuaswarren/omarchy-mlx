"""Executable JS behavior tests for the typed-decision UI paths.

Runs tests/js/assistant-ui.test.mjs under bun (or node) and fails on a
nonzero exit. The .mjs file holds real assertions against composer.js's
typed-question parser and genui.js's decision normalization/plain-text
rendering — not source-text checks.
"""

import subprocess
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
JS_TEST = REPO_ROOT / "tests" / "js" / "assistant-ui.test.mjs"


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


class AssistantUiJsTests(unittest.TestCase):
    def test_assistant_ui_assertions_pass(self):
        runner = _runner()
        if runner is None:
            self.skipTest("no JS runtime available")
        result = subprocess.run([runner, str(JS_TEST)],
                                capture_output=True, text=True, timeout=60)
        self.assertEqual(
            result.returncode, 0,
            f"JS assertions failed:\nstdout: {result.stdout}\nstderr: {result.stderr}")
        self.assertIn("assistant ui js tests passed", result.stdout)


if __name__ == "__main__":
    unittest.main()
