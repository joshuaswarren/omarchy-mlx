#!/usr/bin/env python3
"""Issue #58 part 3: unit tests for scripts/pair_memory_common.py.

Stdlib unittest only, no network: the pure helpers run on dicts and temp
files; the /proc walkers run against the live test process and a reaped
child. The HTTP/session plumbing (call, wait_for_setup, backend_peak) is
out of scope here; it needs a live server.
"""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import pair_memory_common as pmc  # noqa: E402


def reaped_child_pid():
    """Spawn `true`, wait for it, return its (now gone) pid."""
    p = subprocess.Popen(["true"])
    p.wait()
    return p.pid


class MeminfoTests(unittest.TestCase):
    def test_meminfo_bytes_kb_suffix(self):
        self.assertEqual(pmc.meminfo_bytes({"MemTotal": "16384 kB"},
                                           "MemTotal"), 16384 * 1024)

    def test_meminfo_bytes_unitless(self):
        self.assertEqual(pmc.meminfo_bytes({"HugePages_Total": "42"},
                                           "HugePages_Total"), 42)

    def test_meminfo_bytes_missing_key_is_zero(self):
        self.assertEqual(pmc.meminfo_bytes({}, "MemFree"), 0)

    def test_meminfo_bytes_empty_value_is_zero(self):
        self.assertEqual(pmc.meminfo_bytes({"MemFree": ""}, "MemFree"), 0)

    def test_parse_meminfo_reads_live_proc(self):
        mi = pmc.parse_meminfo()
        self.assertIn("MemTotal", mi)
        self.assertGreater(pmc.meminfo_bytes(mi, "MemTotal"), 0)


class ProcWalkerTests(unittest.TestCase):
    def test_children_of_self_contains_self(self):
        pids = pmc.children_of(os.getpid())
        self.assertIn(os.getpid(), pids)
        self.assertTrue(all(isinstance(p, int) for p in pids))

    def test_rss_kb_live_process_is_positive_bytes(self):
        rss = pmc.rss_kb(os.getpid())
        self.assertIsInstance(rss, int)
        self.assertGreater(rss, 0)

    def test_rss_kb_reaped_pid_is_none(self):
        self.assertIsNone(pmc.rss_kb(reaped_child_pid()))

    def test_pss_kb_reaped_pid_is_none(self):
        self.assertIsNone(pmc.pss_kb(reaped_child_pid()))

    def test_drm_bytes_shape_no_render_fds(self):
        d = pmc.drm_bytes(reaped_child_pid())
        self.assertEqual(d, {"drm_renderD_fds": 0, "drm_allocated_bytes": 0})

    def test_drm_bytes_live_pid_structure(self):
        d = pmc.drm_bytes(os.getpid())
        self.assertEqual(set(d), {"drm_renderD_fds", "drm_allocated_bytes"})
        self.assertIsInstance(d["drm_renderD_fds"], int)
        self.assertIsInstance(d["drm_allocated_bytes"], int)
        self.assertGreaterEqual(d["drm_allocated_bytes"], 0)


class LogAndLockReadersTests(unittest.TestCase):
    def test_find_chat_port_returns_last_line_and_strips_period(self):
        with tempfile.TemporaryDirectory() as td:
            logdir = Path(td) / "assistant" / "logs"
            logdir.mkdir(parents=True)
            (logdir / "everyday-chat.log").write_text(
                "Starting httpd at 127.0.0.1:8901.\n"
                "worker restarted\n"
                "Starting httpd at 127.0.0.1:8917.\n")
            port = pmc.find_chat_port(td, "everyday", timeout_s=5)
        self.assertEqual(port, "127.0.0.1:8917")

    def test_find_chat_port_missing_log_times_out_to_none(self):
        with tempfile.TemporaryDirectory() as td:
            self.assertIsNone(pmc.find_chat_port(td, "nope", timeout_s=0))

    def test_read_context_tokens_from_pair_lock(self):
        with tempfile.TemporaryDirectory() as td:
            lockdir = Path(td) / "assistant" / "pair-locks"
            lockdir.mkdir(parents=True)
            (lockdir / "everyday.json").write_text(
                json.dumps({"context_tokens": 4096}))
            self.assertEqual(pmc.read_context_tokens(td, "everyday"), 4096)

    def test_read_context_tokens_missing_lock_is_none(self):
        with tempfile.TemporaryDirectory() as td:
            self.assertIsNone(pmc.read_context_tokens(td, "absent"))


if __name__ == "__main__":
    unittest.main()
