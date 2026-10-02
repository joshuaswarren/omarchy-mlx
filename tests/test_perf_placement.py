"""Tests for the unprivileged uclamp_min serving placement (perf_hold ticket).

Run: PYTHONPATH=serve python3 -m unittest tests.test_perf_placement -q
"""

import ctypes
import sys
import unittest
from unittest import mock

sys.path.insert(0, "serve")

from mlx_omarchy_serve import perf_placement as pp


class ClampFromEnvTest(unittest.TestCase):
    def test_default_is_1024(self):
        self.assertEqual(pp._clamp_from_env({}), 1024)

    def test_zero_disables(self):
        self.assertIsNone(pp._clamp_from_env({pp.UCLAMP_ENV: "0"}))

    def test_explicit_value(self):
        self.assertEqual(pp._clamp_from_env({pp.UCLAMP_ENV: "512"}), 512)

    def test_garbage_disables(self):
        self.assertIsNone(pp._clamp_from_env({pp.UCLAMP_ENV: "banana"}))


class ApplyUclampTest(unittest.TestCase):
    def test_success_passes_flags_and_value(self):
        seen = {}

        def fake_syscall(nr, pid, buf, flags):
            seen["nr"], seen["pid"], seen["buf"], seen["flags"] = nr, pid, buf, flags
            return 0

        ok, detail = pp.apply_uclamp_min(1024, syscall=fake_syscall)
        self.assertTrue(ok)
        self.assertEqual(seen["nr"], pp._SCHED_SETATTR[__import__("os").uname().machine])
        self.assertEqual(seen["pid"], 0)  # calling thread
        self.assertEqual(seen["flags"], 0)  # syscall's own flags arg, not sched_flags
        size, policy, flags, nice, prio, rt, dl, period, umin, umax = \
            pp._SCHED_ATTR.unpack(seen["buf"])
        self.assertEqual(size, pp._SCHED_ATTR.size)
        self.assertEqual(flags, pp.SCHED_FLAG_KEEP_POLICY | pp.SCHED_FLAG_UTIL_CLAMP_MIN)
        self.assertEqual(umin, 1024)
        self.assertEqual(umax, 1024)

    def test_eperm_is_silent_failure(self):
        def fake_syscall(nr, pid, buf, flags):
            ctypes.set_errno(1)  # EPERM
            return -1

        ok, detail = pp.apply_uclamp_min(1024, syscall=fake_syscall)
        self.assertFalse(ok)
        self.assertIn("errno=1", detail)

    def test_unsupported_arch_refuses(self):
        fake_uname = lambda: type("U", (), {"machine": "riscv64"})()
        with mock.patch.object(pp.os, "uname", fake_uname):
            ok, detail = pp.apply_uclamp_min(1024, syscall=lambda *a: 0)
        self.assertFalse(ok)
        self.assertIn("unsupported", detail)

    def test_apply_from_env_off(self):
        seen = []
        ok, detail = pp.apply_from_env({pp.UCLAMP_ENV: "0"}, log=seen.append)
        self.assertFalse(ok)
        self.assertIn("off", detail)

    def test_apply_from_env_default_uses_default(self):
        calls = []

        def fake_syscall(nr, pid, buf, flags):
            calls.append(pp._SCHED_ATTR.unpack(buf)[8])
            return 0

        ok, _ = pp.apply_from_env({}, syscall=fake_syscall)
        self.assertTrue(ok)
        self.assertEqual(calls, [1024])


if __name__ == "__main__":
    unittest.main()
