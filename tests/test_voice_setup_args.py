"""The omarchy-voice-setup arg list must parse through the real CLI parser.

Regression for w7Q's CHANGES on the VXP-3 skeleton: the script exec'd
`--pair --wake-word --yes --once` with no --prompt, which validate_args
rejects before any download. The test reads the argument list straight out
of the shipped script and feeds it through build_parser + validate_args,
so a drift between the script and the parser fails here first.
"""

import shlex
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "serve"))

from mlx_omarchy_assistant import __main__ as assistant_cli  # noqa: E402

SETUP = REPO_ROOT / "packaging" / "voice" / "omarchy-voice-setup"
PAIRS = ("everyday", "quality", "compact")


def setup_argv(pair):
    """Argv after `-m mlx_omarchy_assistant` in the exec line, PAIR bound."""
    lines = [ln for ln in SETUP.read_text().splitlines()
             if ln.strip() and not ln.lstrip().startswith("#")]
    script = "\n".join(lines)
    exec_at = script.index("mlx_omarchy_assistant")
    tail = script[exec_at + len("mlx_omarchy_assistant"):]
    tail = tail.replace("\\\n", " ")
    argv = shlex.split(tail)
    return [pair if a == "$PAIR" else a for a in argv]


class VoiceSetupArgsTest(unittest.TestCase):
    def test_setup_args_validate_for_every_pair(self):
        parser = assistant_cli.build_parser()
        for pair in PAIRS:
            with self.subTest(pair=pair):
                argv = setup_argv(pair)
                args = parser.parse_args(argv)
                assistant_cli.validate_args(parser, args)
                self.assertEqual(args.pair, pair)
                self.assertTrue(args.once)
                self.assertTrue(args.prompt)

    def test_setup_args_fail_closed_without_prompt(self):
        # The pre-fix arg list: --once with no --prompt must still be refused.
        parser = assistant_cli.build_parser()
        argv = [a for a in setup_argv("everyday")
                if a not in ("--prompt", "Say ready.")]
        args = parser.parse_args(argv)
        with self.assertRaises(SystemExit):
            assistant_cli.validate_args(parser, args)


if __name__ == "__main__":
    unittest.main()
