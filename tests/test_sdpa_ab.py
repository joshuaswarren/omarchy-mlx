"""Tests for scripts/sdpa-ab.sh gate-env wiring (regression).

Regression for the false-pass found in review: the same-wheel equivalence
subshell assigned a literal variable named GATE_ENV=1 instead of setting
the gate variable whose NAME is $GATE_ENV, so the gate stayed OFF and the
"equivalence" measured the ungated path. These checks pin the corrected
pattern and observe the actual env a child bash would see.
"""

import re
import subprocess
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "sdpa-ab.sh"


class GateEnvWiringTests(unittest.TestCase):
    def test_script_uses_constructed_env_name_for_equivalence(self):
        text = SCRIPT.read_text(encoding="utf-8")
        # The corrected pattern: env assigns the variable NAMED by
        # $GATE_ENV. The broken literal GATE_ENV=1 must never return.
        self.assertIn('env "$GATE_ENV=1"', text)
        self.assertNotRegex(text, r"(?<!\")GATE_ENV=1 bash")
        # The timing legs export the constructed name too.
        self.assertIn('export "$GATE_ENV=1"', text)
        self.assertIn('export "$GATE_ENV=0"', text)

    def test_observed_env_inside_eq_cmd_child(self):
        # Observe what an EQ_CMD child actually sees, using the same
        # subshell shape as the runner. GATE_ENV is set first, exactly
        # as the runner has it before line "env \"$GATE_ENV=1\"" runs
        # (a same-line assignment prefix would NOT be visible to the
        # expansion, which is a shell scoping rule, not the runner's
        # behavior).
        cmd = ('GATE_ENV=MLX_OMARCHY_PROBE; '
               'cd /tmp && '
               'env "$GATE_ENV=1" bash -c \'echo observed=$MLX_OMARCHY_PROBE\'')
        r = subprocess.run(["bash", "-c", cmd],
                           capture_output=True, text=True, timeout=30)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("observed=1", r.stdout)
        cmd = 'GATE_ENV=MLX_OMARCHY_PROBE GATE_ENV=1 bash -c ' \
              '\'echo observed=${MLX_OMARCHY_PROBE:-UNSET}\''
        r = subprocess.run(["bash", "-c", cmd],
                           capture_output=True, text=True, timeout=30)
        self.assertIn("observed=UNSET", r.stdout)


class SameWheelRefusalTests(unittest.TestCase):
    """Same-SHA invocation must refuse without a named gate and without
    an equivalence case (bounded: both refusals fire before any build)."""

    SCRIPT_RUN = ["bash", str(SCRIPT), "1111111", "1111111"]

    def test_no_gate_env_refuses(self):
        r = subprocess.run(self.SCRIPT_RUN, capture_output=True,
                           text=True, timeout=30,
                           env={"PATH": "/usr/bin:/bin", "HOME": "/tmp"})
        self.assertEqual(r.returncode, 2)
        self.assertIn("Refusing to run without a named gate", r.stderr)

    def test_no_eq_cmd_refuses(self):
        import os

        env = dict(os.environ)
        env.update({"GATE_ENV": "MLX_OMARCHY_PROBE", "TEMPLATE": "/tmp",
                    "SRC": "/tmp"})
        r = subprocess.run(self.SCRIPT_RUN, capture_output=True,
                           text=True, timeout=30, env=env)
        self.assertEqual(r.returncode, 2)
        self.assertIn("Refusing to run without an equivalence case",
                      r.stderr)

    def test_bad_gate_name_refuses(self):
        import os

        env = dict(os.environ)
        env.update({"GATE_ENV": "NOT_OMARCHY", "EQ_CMD": "true",
                    "TEMPLATE": "/tmp", "SRC": "/tmp"})
        r = subprocess.run(self.SCRIPT_RUN, capture_output=True,
                           text=True, timeout=30, env=env)
        self.assertEqual(r.returncode, 2)
        self.assertIn("MLX_OMARCHY_* gate name", r.stderr)


if __name__ == "__main__":
    unittest.main()
