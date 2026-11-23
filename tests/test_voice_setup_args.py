"""The omarchy-voice-setup arg list must parse through the prefetch parser.

The launcher execs `-m mlx_omarchy_assistant.prefetch` (V19P-2: the
download step is the prefetch command, not the assistant serve CLI). The
test reads the argument list straight out of the shipped script and feeds
it through the real build_parser, so a drift between the script and the
parser fails here first.
"""

import shlex
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "serve"))

from mlx_omarchy_assistant import prefetch  # noqa: E402

SETUP = REPO_ROOT / "packaging" / "voice" / "omarchy-voice-setup"
PAIRS = ("everyday", "quality", "compact")


def setup_argv(pair):
    """Argv after `-m mlx_omarchy_assistant.prefetch` in the exec line, PAIR bound."""
    lines = [ln for ln in SETUP.read_text().splitlines()
             if ln.strip() and not ln.lstrip().startswith("#")]
    script = "\n".join(lines)
    exec_at = script.index("mlx_omarchy_assistant.prefetch")
    tail = script[exec_at + len("mlx_omarchy_assistant.prefetch"):]
    tail = tail.replace("\\\n", " ")
    argv = shlex.split(tail)
    return [pair if a == "$PAIR" else a for a in argv]


class VoiceSetupArgsTest(unittest.TestCase):
    def test_setup_args_validate_for_every_pair(self):
        parser = prefetch.build_parser()
        for pair in PAIRS:
            with self.subTest(pair=pair):
                argv = setup_argv(pair)
                args = parser.parse_args(argv)
                self.assertEqual(args.pair, pair)
                self.assertEqual(args.wake_word, "hey_jarvis")
                self.assertFalse(args.manifest_only)

    def test_setup_args_reject_unknown_pair(self):
        # A pair outside the pinned set must be refused by the parser itself.
        parser = prefetch.build_parser()
        argv = setup_argv("bogus")
        with self.assertRaises(SystemExit):
            parser.parse_args(argv)

    def test_setup_args_have_no_serve_flags(self):
        # The prefetch exec line must not carry serve-path flags; the
        # download step never starts the assistant server.
        for pair in PAIRS:
            argv = setup_argv(pair)
            for flag in ("--once", "--prompt", "--serve", "--port"):
                self.assertNotIn(flag, argv)


if __name__ == "__main__":
    unittest.main()
