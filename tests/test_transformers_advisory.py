"""The launchers must not print transformers' missing-torch advisory.

Issue #33: transformers (5.x pinned; same advisory path in 4.x) prints
``[transformers] PyTorch was not found. Models won't be available and only
tokenizers, configuration and file/data utilities can be used.`` on stderr
when torch is absent, right onto the streamed ``Assistant:`` line. That is
an advisory, not a failure: mlx-omarchy uses transformers for tokenizers
only and intentionally ships no torch. ``serve/mlx_omarchy_paths.py`` — the
module every launcher imports before transformers — sets
``TRANSFORMERS_NO_ADVISORY_WARNINGS=1`` (honored by ``Logger.warning_advice``
in both 4.x and 5.x) unless the user set it explicitly.

Skipped where transformers is missing or torch is present: the advisory
cannot fire there, so the assertions would prove nothing.
"""

import os
import subprocess
import sys
import unittest
from pathlib import Path

SERVE = Path(__file__).resolve().parents[1] / "serve"

# What every launcher runs, in order: the shared bootstrap, then the chat
# path's transformers surface. Import-level only: no GPU, no model download.
CHAT_IMPORT = (
    "import sys; sys.path.insert(0, {serve!r})\n"
    "import mlx_omarchy_paths\n"
    "from transformers import AutoTokenizer\n"
).format(serve=str(SERVE))

ADVISORY_ENV = "TRANSFORMERS_NO_ADVISORY_WARNINGS"


class TransformersAdvisoryTests(unittest.TestCase):
    def setUp(self):
        try:
            import transformers  # noqa: F401
        except ImportError:
            self.skipTest("transformers not installed")
        try:
            import torch  # noqa: F401
        except ImportError:
            return
        self.skipTest("torch installed; the advisory cannot fire")

    def test_launcher_bootstrap_sets_advisory_env(self):
        sys.path.insert(0, str(SERVE))
        try:
            import mlx_omarchy_paths  # noqa: F401
        finally:
            sys.path.remove(str(SERVE))
        self.assertEqual(os.environ.get(ADVISORY_ENV), "1")

    def test_tokenizer_import_is_quiet_without_torch(self):
        # Strip the var so the verdict rests on the bootstrap, not the
        # parent's environment.
        env = {k: v for k, v in os.environ.items() if k != ADVISORY_ENV}
        result = subprocess.run(
            [sys.executable, "-c", CHAT_IMPORT],
            env=env, capture_output=True, text=True, timeout=120, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertNotIn("PyTorch was not found", result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
