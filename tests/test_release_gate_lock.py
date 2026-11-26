#!/usr/bin/env python3
"""gate_lock contract tests (CPU-only, CT-runnable, no mlx needed).

Covers the three gate_lock modes against a mock lock file:

  A. GPU_TURN_TICKET=1 while the lock is HELD -> the wrapped command runs
     (no flock, no block). This is the gpu-turn ticket contract: gpu-turn
     itself never exports anything, so the DRIVER must set GPU_TURN_TICKET=1.
  B. no ticket while the lock is HELD -> gate_lock returns 99 and prints
     the named GPU_LOCK_BUSY error instead of hanging.
  C. no ticket while the lock is FREE -> the command runs under flock.
"""
import os
import pathlib
import subprocess
import tempfile
import unittest

REPO = pathlib.Path(__file__).resolve().parents[1]
ENV_SH = REPO / "scripts" / "release-gates" / "env.sh"


def run_gate_lock(lock: str, ticket: str, setup: str, body: str) -> subprocess.CompletedProcess:
    """Run gate_lock in a bash subshell with the repo env.sh sourced.

    `setup` runs before gate_lock (e.g. start a lock holder); `body` is the
    command handed to gate_lock.
    """
    script = f"""
set -uo pipefail
export TAG=test-gate-lock
export ASSETS_DIR={lock}.assets
export GATE_ROOT={lock}.root
export LOG_DIR={lock}.root/logs
export GPU_LOCK={lock}
export GPU_TURN_TICKET={ticket}
. "{ENV_SH}"
{setup}
gate_lock -x -w 2 {body}
"""
    return subprocess.run(["bash", "-c", script], capture_output=True, text=True,
                          timeout=60)


class GateLockTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.lock = os.path.join(self.tmp.name, "mock-gpu.lock")
        pathlib.Path(self.lock).write_text("")

    def tearDown(self):
        self.tmp.cleanup()

    def test_a_ticket_mode_runs_despite_held_lock(self):
        marker = pathlib.Path(self.tmp.name) / "ran"
        r = run_gate_lock(
            self.lock, "1",
            setup=f'flock "{self.lock}" -c "sleep 30" & HPID=$!; sleep 0.3; '
                  f'trap "kill -9 $HPID 2>/dev/null" EXIT',
            body=f'echo ran > "{marker}"')
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertTrue(marker.exists(),
                        "command must run in ticket mode even with the lock held")

    def test_b_no_ticket_and_held_lock_named_error(self):
        r = run_gate_lock(
            self.lock, "",
            setup=f'flock "{self.lock}" -c "sleep 30" & HPID=$!; sleep 0.3; '
                  f'trap "kill -9 $HPID 2>/dev/null" EXIT',
            body="echo should-not-run")
        self.assertEqual(r.returncode, 99, f"expected 99, got {r.returncode}: {r.stderr}")
        self.assertIn("GPU_LOCK_BUSY", r.stderr)
        self.assertIn("GPU_TURN_TICKET", r.stderr)
        self.assertNotIn("should-not-run", r.stdout, "command must not run")

    def test_c_no_ticket_free_lock_runs_under_flock(self):
        marker = pathlib.Path(self.tmp.name) / "ran"
        r = run_gate_lock(self.lock, "", setup="", body=f'echo ran > "{marker}"')
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertTrue(marker.exists())


if __name__ == "__main__":
    unittest.main()
