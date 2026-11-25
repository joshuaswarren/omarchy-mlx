"""Submit-path failure injection: rejection, drops, and host-signal order.

Observable contracts of the submit worker and the recovery ladder, each
exercised by driving real submissions with the deterministic
fault-injection env hooks in CommandEncoder::submit():

- MLX_OMARCHY_TEST_REJECT_SUBMIT=<ordinals>: the Nth QueueSubmit returns
  VK_ERROR_DEVICE_LOST, the way the kernel's -EINVAL admission rejection
  maps through vk_device_set_lost (issue omarchy-mlx#526). The caller
  must receive an invalid-submission error naming the rejected
  parameters, and a job already queued behind the rejection must observe
  the poisoned encoder before it enters the driver.

- MLX_OMARCHY_TEST_DROP_SUBMIT=<ordinals>: the Nth QueueSubmit never
  reaches the driver. Later submissions wait in-order on its completion
  value. If the worker resolves that wait inside vkQueueSubmit while
  holding queue_mutex_, the watchdog cannot take the mutex to resubmit
  and shutdown cannot join the worker: this test hangs on that shape.
  The worker must wait outside queue_mutex_ and run queued recovery
  jobs between wait ticks.

- MLX_OMARCHY_TEST_DROP_SIGNAL=<ordinals>: the Nth batch executes but
  its signals are stripped, so dependent submissions park on a counter
  that never advances. Recovery must publish the lost value without
  overtaking work still queued in the submit FIFO: the completions
  counter may only cross values whose submissions were already issued
  (pending_ placeholders keep drained_value_ truthful even when a host
  signal lands early). This test hangs on a worker that waits inside
  the driver call, and reports wrong output if a wait returns before
  its batch really ran.

Each case runs in a subprocess so the process-global submit ordinal and
the fault env are deterministic. A subprocess that exits non-zero, hangs
past the timeout, or returns wrong output fails.
"""

import os
import subprocess
import sys
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Short watchdog interval so the recovery ladder fires quickly; the
# resubmit round-trip itself is sub-second.
FAST_WATCHDOG = str(3 * 1000 * 1000 * 1000)

# Back-to-back submissions with a single host read at the end: every
# eval is its own submit, each later submit waits in-order on the
# previous completion, and cv 3-6 all park behind cv 2 when cv 2's
# signal never lands.
WORKLOAD = r"""
import mlx.core as mx
import numpy as np

a_np = np.arange(64, dtype=np.float32).reshape(8, 8)
a = mx.array(a_np)
mx.eval(a)
outs = []
for i in range(4):
    b = (a + i) @ a
    mx.eval(b)
    outs.append(b)
t = outs[0] + outs[1] + outs[2] + outs[3]
got = float(mx.sum(t).item())
want = float(sum(((a_np + i) @ a_np).sum() for i in range(4)))
print("RESULT", got)
print("REF", want)
"""


def _outputs(res):
    vals = {}
    for line in res.stdout.splitlines():
        for key in ("RESULT", "REF"):
            if line.startswith(key + " "):
                vals[key] = float(line.split(None, 1)[1])
    return vals["RESULT"], vals["REF"]


def _run(prog, env_extra, timeout=60):
    env = dict(os.environ)
    env["MLX_OMARCHY_HANG_NO_PROGRESS_NS"] = FAST_WATCHDOG
    env.update(env_extra)
    try:
        return subprocess.run(
            [sys.executable, "-c", prog],
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=REPO_ROOT,
        )
    except subprocess.TimeoutExpired as e:
        return subprocess.CompletedProcess(
            e.args, -9, e.stdout or "",
            (e.stderr or "") + f"\nTIMED OUT after {timeout}s")


class TestSubmitFaultInjection(unittest.TestCase):
    def test_rejected_submit_surfaces_invalid_submission(self):
        """The Nth QueueSubmit's synchronous failure reaches the caller
        as an invalid submission naming the parameters the kernel
        rejected (ordinal, completion value, encoder, node count, work
        estimate, command-buffer presence, wait/signal values), and the
        next queued job observes the poisoned encoder before it enters
        the driver."""
        prog = r"""
import mlx.core as mx

try:
    a = mx.arange(64, dtype=mx.float32).reshape(8, 8)
    mx.eval(a)          # whichever submit is ordinal 1 is rejected
    b = a + 1
    mx.eval(b)          # encoder poison rethrows it here if not earlier
    c = (b @ a) + 1
    mx.eval(c)
    print("NOERROR")
except RuntimeError as e:
    print("ERROR", str(e))
"""
        res = _run(prog, {"MLX_OMARCHY_TEST_REJECT_SUBMIT": "1"})
        self.assertEqual(
            res.returncode, 0, f"stderr: {res.stderr[-2000:]}")
        self.assertIn(
            "ERROR", res.stdout, res.stdout + res.stderr[-2000:])
        line = res.stdout.split("ERROR", 1)[1]
        self.assertIn("invalid submission", line)
        for param in (
            "ordinal=",
            "encoder=",
            "cv=",
            "nodes=",
            "work=",
            "command_buffer=",
            "wait_values=",
            "signal_values=",
        ):
            self.assertIn(param, line, f"missing {param} in: {line}")

    def test_dropped_submit_does_not_block_recovery(self):
        """A dropped submission must not park the recovery ladder or the
        worker behind a dependent wait. The watchdog resubmits through
        the recovery lane while the dependent job waits outside
        queue_mutex_; the result must still be correct."""
        res = _run(
            WORKLOAD,
            {"MLX_OMARCHY_TEST_DROP_SUBMIT": "2"},
            timeout=90,
        )
        self.assertEqual(
            res.returncode,
            0,
            f"exit={res.returncode} stderr: {res.stderr[-3000:]}")
        result, ref = _outputs(res)
        self.assertAlmostEqual(
            result, ref, delta=max(1.0, abs(ref) * 1e-3))

    def test_stripped_signal_recovers_in_order(self):
        """An executed-but-unsignaled batch is recovered without
        overtaking submissions still queued behind it: every queued
        job's entry blocks the drain until its own signal issues, so
        drained_value_ never marks unsubmitted work complete."""
        res = _run(
            WORKLOAD,
            {"MLX_OMARCHY_TEST_DROP_SIGNAL": "2"},
            timeout=90,
        )
        self.assertEqual(
            res.returncode,
            0,
            f"exit={res.returncode} stderr: {res.stderr[-3000:]}")
        self.assertIn(
            "SUBMIT-RECOVER",
            res.stderr,
            f"watchdog recovery must run; stderr: {res.stderr[-2000:]}")
        result, ref = _outputs(res)
        self.assertAlmostEqual(
            result, ref, delta=max(1.0, abs(ref) * 1e-3))


if __name__ == "__main__":
    unittest.main()
