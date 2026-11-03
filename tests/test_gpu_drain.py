"""A SIGTERM must wait for queued GPU work before the process dies (issue 339).

Killing a process that has a second compute job queued behind a running one
makes the kernel driver log a GPU page fault and time out. The same loop with
a handler that runs mx.synchronize() first did not fault (13-inch M1, 0 of 6
against 6 of 6). These tests pin the behaviour a supervisor sees: the drain
runs before the process ends, the process still ends by SIGTERM, and a second
SIGTERM during a stuck drain ends it at once.
"""

import multiprocessing
import os
import signal
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SERVE = REPO / "serve"
if str(SERVE) not in sys.path:
    sys.path.insert(0, str(SERVE))

from mlx_omarchy_serve import gpu_drain  # noqa: E402

SIGTERM_BIT = 1 << (signal.SIGTERM - 1)


def _caught(pid):
    """True once the process has a SIGTERM handler installed (Linux /proc)."""
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("SigCgt:"):
                return bool(int(line.split()[1], 16) & SIGTERM_BIT)
    except OSError:
        pass
    return False


def _wait_caught(proc, timeout=15.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if _caught(proc.pid):
            return
        time.sleep(0.02)
    raise AssertionError("child never installed a SIGTERM handler")


def _slow_drain_child(marker, seconds):
    def drain():
        Path(marker + ".started").write_text("1")
        time.sleep(seconds)
        Path(marker).write_text("drained")

    gpu_drain.install(drain)
    time.sleep(60)


def _failing_drain_child(marker):
    def drain():
        Path(marker).write_text("tried")
        raise RuntimeError("device lost")

    gpu_drain.install(drain)
    time.sleep(60)


def _worker_child(marker):
    """The real voice worker loop with a stub mlx.core that records the drain."""
    core = types.ModuleType("mlx.core")

    def synchronize(*_args):
        Path(marker).write_text("drained")

    core.synchronize = synchronize
    package = types.ModuleType("mlx")
    package.core = core
    sys.modules["mlx"] = package
    sys.modules["mlx.core"] = core
    from mlx_omarchy_assistant import synthesis

    parent_end, child_end = multiprocessing.get_context("fork").Pipe()
    synthesis._worker_main(child_end, "/nonexistent")


def _stt_child(marker):
    """The real GPU STT worker main(), stalled in load_model, with a stub mlx.core."""
    core = types.ModuleType("mlx.core")
    core.synchronize = lambda *_a: Path(marker).write_text("drained")
    package = types.ModuleType("mlx")
    package.core = core
    sys.modules["mlx"] = package
    sys.modules["mlx.core"] = core
    utils = types.ModuleType("mlx_audio.stt.utils")
    utils.load_model = lambda _dir: time.sleep(60)
    for name in ("mlx_audio", "mlx_audio.stt", "mlx_audio.stt.models"):
        sys.modules[name] = types.ModuleType(name)
    sys.modules["mlx_audio.stt.utils"] = utils
    from mlx_omarchy_assistant import gpu_stt_worker

    gpu_stt_worker.main(["--model-dir", "/nonexistent"])


class GpuDrainTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.marker = os.path.join(self.tmp.name, "marker")
        self.ctx = multiprocessing.get_context("fork")

    def _run(self, target, *args):
        proc = self.ctx.Process(target=target, args=args)
        proc.start()
        self.addCleanup(lambda: proc.is_alive() and proc.kill())
        _wait_caught(proc)
        return proc

    def test_sigterm_drains_then_dies_by_sigterm(self):
        proc = self._run(_slow_drain_child, self.marker, 0.3)
        proc.terminate()
        proc.join(10)
        self.assertEqual(Path(self.marker).read_text(), "drained")
        self.assertEqual(proc.exitcode, -signal.SIGTERM)

    def test_second_sigterm_ends_a_stuck_drain_at_once(self):
        proc = self._run(_slow_drain_child, self.marker, 30)
        proc.terminate()
        end = time.monotonic() + 10
        while not Path(self.marker + ".started").exists():
            self.assertLess(time.monotonic(), end, "drain never started")
            time.sleep(0.02)
        proc.terminate()
        proc.join(5)
        self.assertEqual(proc.exitcode, -signal.SIGTERM)
        self.assertFalse(Path(self.marker).exists())

    def test_a_failing_drain_still_ends_the_process_by_sigterm(self):
        proc = self._run(_failing_drain_child, self.marker)
        proc.terminate()
        proc.join(10)
        self.assertEqual(Path(self.marker).read_text(), "tried")
        self.assertEqual(proc.exitcode, -signal.SIGTERM)

    def test_install_off_the_main_thread_is_refused(self):
        import threading

        result = []
        thread = threading.Thread(target=lambda: result.append(gpu_drain.install(lambda: None)))
        thread.start()
        thread.join()
        self.assertEqual(result, [False])

    def test_the_voice_worker_drains_on_sigterm(self):
        proc = self._run(_worker_child, self.marker)
        proc.terminate()
        proc.join(10)
        self.assertEqual(Path(self.marker).read_text(), "drained")
        self.assertEqual(proc.exitcode, -signal.SIGTERM)

    def test_the_gpu_stt_worker_drains_on_sigterm(self):
        proc = self._run(_stt_child, self.marker)
        proc.terminate()
        proc.join(10)
        self.assertEqual(Path(self.marker).read_text(), "drained")
        self.assertEqual(proc.exitcode, -signal.SIGTERM)


if __name__ == "__main__":
    unittest.main()
