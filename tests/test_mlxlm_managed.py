"""The mlx-lm shim as a managed pair worker.

Under the assistant manager the shim must claim its reservation before any
weight loads and must exit when the manager's lifetime pipe closes. Without
that, a manager killed hard leaves the chat worker running (it holds the
weights and the GPU) while the reaper frees its unclaimed reservation.

The REAL shim runs in a subprocess against a minimal fake mlx_lm.server (the
version pin, the _tokenize hook and a /v1/models endpoint), spawned through
the REAL manager-side plumbing.
"""

import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parent.parent
SERVE_ROOT = str(REPO_ROOT / "serve")
sys.path.insert(0, SERVE_ROOT)

from mlx_omarchy_assistant import managed, pairs  # noqa: E402
from mlx_omarchy_serve import budget  # noqa: E402
from tests.test_assistant_pairs import (  # noqa: E402
    CHAT_SMALL_ID, DECISION_ID, BasePairTest, stub_argv_builder)

GiB = 1 << 30
SHIM = [sys.executable, "-m", "mlx_omarchy_serve._mlxlm_server"]

# Importing mlx_lm.server stands in for loading weights: it writes a marker.
FAKE_SERVER = """
import os, sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
open(os.environ["FAKE_IMPORTED"], "w").close()
__version__ = "0.31.3"
class ResponseGenerator:
    def _tokenize(self, tokenizer, request, args):
        return ([], [], [], None)
class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Length", "2")
        self.end_headers()
        self.wfile.write(b"{}")
    def log_message(self, *args):
        pass
def main():
    port = int(sys.argv[sys.argv.index("--port") + 1])
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
"""


def make_fake_mlx_lm(root: Path) -> Path:
    """A directory to put on PYTHONPATH that provides mlx_lm.server."""
    package = root / "mlx_lm"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (package / "server.py").write_text(FAKE_SERVER)
    return root


def wait_until(test, predicate, what, seconds=60):
    deadline = time.monotonic() + seconds
    while not predicate():
        test.assertLess(time.monotonic(), deadline, what)
        time.sleep(0.05)


class ManagedShimTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        self.home = root / "home"
        self.home.mkdir()
        self.imported = root / "imported"
        self.env = dict(
            os.environ,
            PYTHONPATH=os.pathsep.join([str(make_fake_mlx_lm(root / "fake")), SERVE_ROOT]),
            MLX_OMARCHY_HOME=str(self.home),
            MLX_OMARCHY_SERVE_CONTEXT_LIMIT="4096",
            FAKE_IMPORTED=str(self.imported),
        )
        self.argv = SHIM + ["--port", str(managed.free_port())]
        self.manager_end = None

    def _reap(self, proc):
        if proc.poll() is None:
            proc.kill()
        proc.wait(timeout=15)

    def _close_manager_end(self):
        if self.manager_end is not None:
            os.close(self.manager_end)
            self.manager_end = None

    def _spawn_managed(self, claim_token=None):
        record = budget.admit_and_reserve_batch(
            [("chat", 1024, "")], pair_id="everyday", home=self.home,
            available_bytes=10 * GiB)[0]
        payload = {"name": "chat", "bytes": record["bytes"],
                   "claim_token": claim_token or record["claim_token"],
                   "pair_id": "everyday"}
        lifetime_r, self.manager_end = os.pipe()
        os.set_inheritable(lifetime_r, True)
        self.addCleanup(self._close_manager_end)
        proc = managed.spawn_claiming_worker(
            self.argv, payload, lifetime_read_fd=lifetime_r, env=self.env,
            log_path=self.home / "chat.log")
        os.close(lifetime_r)
        self.addCleanup(self._reap, proc)
        return proc

    def test_claims_before_loading_and_exits_when_the_manager_dies(self):
        proc = self._spawn_managed()
        wait_until(self, self.imported.exists, "worker never reached the server import")
        held = budget.load_reservations(self.home)["chat"]["claimed"]
        self.assertEqual(held["pid"], proc.pid)
        self._close_manager_end()
        wait_until(self, lambda: proc.poll() is not None, "worker outlived its manager")
        self.assertEqual(proc.returncode, -signal.SIGTERM)

    def test_refused_claim_exits_before_anything_loads(self):
        proc = self._spawn_managed(claim_token="not-the-token")
        self.assertEqual(proc.wait(timeout=60), 3)
        self.assertFalse(self.imported.exists())
        self.assertIsNone(budget.load_reservations(self.home)["chat"]["claimed"])

    def test_standalone_launch_serves_without_a_manager(self):
        proc = subprocess.Popen(self.argv, env=self.env, stdin=subprocess.DEVNULL,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.addCleanup(self._reap, proc)
        port = self.argv[-1]
        self.assertTrue(managed.wait_health(f"http://127.0.0.1:{port}/v1/models", 60))


class RealShimUnderPairManagerTests(BasePairTest):
    """The real chat worker beside the stub decision worker, started by the
    real PairManager: the claim must pass the manager's identity checks."""

    def test_start_claims_the_chat_worker_and_it_follows_the_manager(self):
        fake = make_fake_mlx_lm(self.home / "fake")
        imported = self.home / "imported"
        self.make_chat_snapshot(CHAT_SMALL_ID)
        self.make_converted_laya()
        manager = pairs.PairManager(self.home)
        with mock.patch.object(pairs, "_decision_worker_argv", stub_argv_builder("decision")), \
                mock.patch.dict(os.environ, {"PYTHONPATH": str(fake),
                                             "FAKE_IMPORTED": str(imported)}):
            manager.setup("everyday", approve_download=True)
            self.assertEqual(manager.start()["state"], "ready")
        self.addCleanup(manager.stop)
        chat = manager._children["chat"].popen
        claimed = budget.load_reservations(self.home)[CHAT_SMALL_ID]["claimed"]
        self.assertEqual(claimed["pid"], chat.pid)
        # the manager dies: its end of the lifetime pipe closes
        manager._close_lifetime()
        wait_until(self, lambda: chat.poll() is not None, "chat worker outlived its manager")
        self.assertEqual(chat.returncode, -signal.SIGTERM)
        decision = manager._children["decision"].popen
        wait_until(self, lambda: decision.poll() is not None, "decision worker outlived its manager")
        managed.reap_pair_records("everyday", self.home, children=[])
        self.assertNotIn(CHAT_SMALL_ID, budget.load_reservations(self.home))
        self.assertNotIn(DECISION_ID, budget.load_reservations(self.home))


if __name__ == "__main__":
    unittest.main()
