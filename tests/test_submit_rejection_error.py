"""Submit-path failure injection: rejection, drops, and host-signal order.

Three observable contracts of the submit worker and the recovery ladder,
each exercised by driving real submissions with the deterministic
fault-injection env hooks in CommandEncoder::submit():

- MLX_OMARCHY_TEST_REJECT_SUBMIT=<ordinals>: the Nth QueueSubmit returns
  VK_ERROR_DEVICE_LOST, the way the kernel's -EINVAL admission rejection
  maps through vk_device_set_lost (issue omarchy-mlx#526). The caller
  must receive an invalid-submission error naming the rejected
  parameters, and a job already queued behind the rejection must observe
  the poisoned encoder before it enters the driver.

- MLX_OMARCHY_TEST_DROP_SUBMIT=<ordinals>: the Nth QueueSubmit never
  reaches the driver. The dependent submission's in-order wait must
  resolve through the watchdog's resubmission, which cannot progress if
  the worker holds queue_mutex_ inside a blocked submit: this test
  hangs on a worker that waits inside the driver call.

- MLX_OMARCHY_TEST_DROP_SIGNAL=<ordinals>: the Nth batch executes but
  its signals are stripped. Recovery must replay the batch's ORIGINAL
  completion value - host-signaling a freshly reserved value would
  advance the shared counter past submissions still queued in the FIFO
  and mark them complete before they run.

Each case runs in a subprocess so the process-global ordinal counter and
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

WORKLOAD = r"""
import mlx.core as mx

a = mx.arange(64, dtype=mx.float32).reshape(8, 8)
b = a + 1
for _ in range(4):
    b = (b @ a) + 1
mx.eval(b)
expected = None
a2 = mx.arange(64, dtype=mx.float32).reshape(8, 8)
b2 = a2 + 1
for _ in range(4):
    b2 = (b2 @ a2) + 1
mx.eval(b2)
import numpy as np
ref = np.arange(64, dtype=np.float32).reshape(8, 8) + 1
for _ in range(4):
    ref = ref @ (np.arange(64, dtype=np.float32).reshape(8, 8)) + 1
print("RESULT", float(mx.sum(b).item()))
print("REF", float(ref.sum()))
"""


def _run(prog, env_extra, timeout=60):
    env = dict(os.environ)
    env["MLX_OMARCHY_HANG_NO_PROGRESS_NS"] = FAST_WATCHDOG
    env.update(env_extra)
    return subprocess.run(
        [sys.executable, "-c", prog],
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=REPO_ROOT,
    )


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

a = mx.arange(64, dtype=mx.float32).reshape(8, 8)
b = a + 1
mx.eval(b)  # submit 1: rejected on the worker; drains empty
try:
    c = (b @ a) + 1
    mx.eval(c)  # submit 2: submit() rethrows the recorded rejection
except RuntimeError as e:
    print("ERROR", str(e))
"""
        res = _run(prog, {"MLX_OMARCHY_TEST_REJECT_SUBMIT": "1"})
        self.assertEqual(
            res.returncode, 0, f"stderr: {res.stderr[-2000:]}")
        self.assertIn("ERROR", res.stdout, res.stdout + res.stderr[-2000:])
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
        worker behind a dependent wait: the watchdog resubmits through
        the recovery lane while the dependent job waits outside
        queue_mutex_. On a worker that waits inside vkQueueSubmit this
        test hangs until the timeout."""
        res = _run(
            WORKLOAD,
            {"MLX_OMARCHY_TEST_DROP_SUBMIT": "1"},
            timeout=90,
        )
        self.assertEqual(
            res.returncode,
            0,
            f"exit={res.returncode} stderr: {res.stderr[-3000:]}")
        self.assertIn("RESULT", res.stdout, res.stdout + res.stderr[-1000:])
        self.assertIn("REF", res.stdout)
        result = float(res.stdout.split("RESULT", 1)[1].split()[0])
        ref = float(res.stdout.split("REF", 1)[1].split()[0])
        self.assertAlmostEqual(result, ref, places=2)

    def test_stripped_signal_replays_original_value(self):
        """An executed-but-unsignaled batch is recovered by replaying its
        original completion value, not by jumping the counter to a fresh
        value that would mark still-queued submissions complete."""
        res = _run(
            WORKLOAD,
            {"MLX_OMARCHY_TEST_DROP_SIGNAL": "1"},
            timeout=90,
        )
        self.assertEqual(
            res.returncode,
            0,
            f"exit={res.returncode} stderr: {res.stderr[-3000:]}")
        self.assertIn("RESULT", res.stdout)
        self.assertIn("REF", res.stdout)
        result = float(res.stdout.split("RESULT", 1)[1].split()[0])
        ref = float(res.stdout.split("REF", 1)[1].split()[0])
        self.assertAlmostEqual(result, ref, places=2)
        self.assertIn(
            "replaying executed-but-unsignaled",
            res.stderr,
            "recovery must replay the batch's original completion value; "
            f"stderr: {res.stderr[-2000:]}",
        )


if __name__ == "__main__":
    unittest.main()
