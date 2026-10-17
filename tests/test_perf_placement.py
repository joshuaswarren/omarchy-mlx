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


class AllThreadsTest(unittest.TestCase):
    def test_every_tid_gets_the_clamp(self):
        calls = []

        def fake_syscall(nr, pid, buf, flags):
            calls.append((pid, pp._SCHED_ATTR.unpack(buf)[8]))
            return 0

        ok, fail = pp.apply_uclamp_min_all_threads(1024, syscall=fake_syscall, tids=[11, 12, 13])
        self.assertEqual((ok, fail), (3, 0))
        self.assertEqual(calls, [(11, 1024), (12, 1024), (13, 1024)])

    def test_refused_tids_are_counted_not_raised(self):
        def fake_syscall(nr, pid, buf, flags):
            ctypes.set_errno(1)
            return -1 if pid == 12 else 0

        ok, fail = pp.apply_uclamp_min_all_threads(1024, syscall=fake_syscall, tids=[11, 12, 13])
        self.assertEqual((ok, fail), (2, 1))

    def test_apply_from_env_all_threads_off_switch(self):
        ok, detail = pp.apply_from_env({pp.UCLAMP_ENV: "0"}, all_threads=True)
        self.assertFalse(ok)
        self.assertIn("off", detail)

    def test_apply_from_env_all_threads_reports_the_count(self):
        def fake_syscall(nr, pid, buf, flags):
            return 0

        with mock.patch.object(pp.os, "listdir", return_value=["5", "6"]):
            ok, detail = pp.apply_from_env({}, syscall=fake_syscall, all_threads=True)
        self.assertTrue(ok)
        self.assertIn("on 2 threads", detail)

    def test_serve_entry_clamps_before_and_after_the_mlx_import(self):
        import pathlib

        src = (pathlib.Path(pp.__file__).parent / "_mlxlm_server.py").read_text()
        early = src.index("apply_from_env()")
        imp = src.index("import mlx_lm.server as server_module")
        late = src.index("apply_from_env(all_threads=True)")
        self.assertLess(early, imp)
        self.assertLess(imp, late)

    @unittest.skipUnless(sys.platform == "linux", "Linux only")
    def test_real_syscall_reaches_a_second_thread(self):
        import os
        import threading

        getattr_nr = {"x86_64": 315, "aarch64": 275}.get(os.uname().machine)
        if getattr_nr is None:
            self.skipTest("unsupported architecture")
        libc = ctypes.CDLL(None, use_errno=True)

        def util_min(tid):
            buf = ctypes.create_string_buffer(pp._SCHED_ATTR.size)
            rc = libc.syscall(ctypes.c_long(getattr_nr), ctypes.c_uint(tid), buf,
                              ctypes.c_uint(pp._SCHED_ATTR.size), ctypes.c_uint(0))
            return pp._SCHED_ATTR.unpack(buf.raw)[8] if rc == 0 else None

        started = threading.Event()
        stop = threading.Event()
        box = {}

        def worker():
            box["tid"] = threading.get_native_id()
            started.set()
            stop.wait(10)

        th = threading.Thread(target=worker)
        th.start()
        started.wait(10)
        try:
            before = util_min(box["tid"])
            ok, fail = pp.apply_uclamp_min_all_threads(777)
            if ok == 0:
                self.skipTest("kernel refuses sched_setattr here")
            self.assertEqual(util_min(box["tid"]), 777)
            self.assertEqual(fail, 0)
            self.assertNotEqual(before, 777)
        finally:
            stop.set()
            th.join()


if __name__ == "__main__":
    unittest.main()
