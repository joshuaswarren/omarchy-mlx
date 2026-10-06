"""Live check of the CPU PD hold (omarchy backend, cpu_pd_hold.h).

The hold is an open fd on /dev/cpu_dma_latency (PM-QoS request, int32 0)
while GPU work is in flight, dropped within about half a second of idle.
Needs an Apple GPU with the omarchy backend and /dev/cpu_dma_latency
writable by the user (packaging/udev/70-omarchy-mlx-cpu-dma-latency.rules);
skipped otherwise. Each case runs in a child process because the hold is
decided once per process at the first submission.

Run: python3 -m unittest tests.test_cpu_pd_hold -q
"""

import os
import subprocess
import sys
import textwrap
import unittest

PMQOS = "/dev/cpu_dma_latency"

CHILD = textwrap.dedent(
    """
    import os, sys, threading, time
    import mlx.core as mx

    def holds():
        n = 0
        for fd in os.listdir("/proc/self/fd"):
            try:
                if os.readlink("/proc/self/fd/" + fd) == "/dev/cpu_dma_latency":
                    n += 1
            except OSError:
                pass
        return n

    a = mx.random.normal((4096, 4096)).astype(mx.float16)
    b = mx.random.normal((4096, 4096)).astype(mx.float16)
    mx.eval(a, b)
    seen = []
    stop = False
    def sampler():
        while not stop:
            seen.append(holds())
            time.sleep(0.02)
    th = threading.Thread(target=sampler)
    th.start()
    t0 = time.time()
    while time.time() - t0 < 1.5:
        mx.eval(a @ b)
    during = max(seen) if seen else -1
    time.sleep(1.0)
    stop = True
    th.join()
    after = holds()
    print("RESULT during=%d after=%d" % (during, after))
    """
)


def run_child(env_value):
    env = dict(os.environ)
    if env_value is None:
        env.pop("MLX_OMARCHY_CPU_PD_HOLD", None)
    else:
        env["MLX_OMARCHY_CPU_PD_HOLD"] = env_value
    out = subprocess.run(
        [sys.executable, "-c", CHILD], env=env, capture_output=True, text=True, timeout=300
    )
    for line in out.stdout.splitlines():
        if line.startswith("RESULT "):
            kv = dict(x.split("=") for x in line.split()[1:])
            return int(kv["during"]), int(kv["after"])
    raise AssertionError("no RESULT line: " + out.stdout[-400:] + out.stderr[-400:])


@unittest.skipUnless(os.access(PMQOS, os.W_OK), PMQOS + " not writable by this user")
class CpuPdHoldLiveTest(unittest.TestCase):
    def test_forced_hold_is_held_during_work_and_released_after_idle(self):
        during, after = run_child("1")
        self.assertGreaterEqual(during, 1)
        self.assertEqual(after, 0)

    def test_off_switch_never_opens_the_request(self):
        during, after = run_child("0")
        self.assertEqual(during, 0)
        self.assertEqual(after, 0)


if __name__ == "__main__":
    unittest.main()
