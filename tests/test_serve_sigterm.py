"""SIGTERM-safe launch lifecycle: the real sleeper-child signal path.

Main-mandated proof for the CLI's own launch reservation:
- a REAL child process (a stub mlx_lm whose server main sleeps) is spawned
  through the actual CLI, which holds a pending launch reservation;
- SIGTERM to the CLI must terminate the child (bounded wait, escalating to
  kill), release the reservation, and exit — never orphan a live child
  while its reservation was already cleared.

Requires no network and no mlx: the stub mlx_lm package is written into a
temp dir and put on the child's PYTHONPATH.
"""

import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SERVE_DIR = REPO_ROOT / "serve"

STUB_MLXLM_INIT = 'from ._version import __version__\n'
STUB_MLXLM_VERSION = '__version__ = "0.31.3"\n'
STUB_MLXLM_SERVER = '''
import os
import time

__version__ = "0.31.3"


class ResponseGenerator:
    def _tokenize(self, tokenizer, request, args):
        return ([], [], [], None)


def main():
    ready = os.environ["SLEEPER_READY_FILE"]
    with open(ready, "w") as fh:
        fh.write("1")
    deadline = time.time() + 120
    while time.time() < deadline:
        time.sleep(0.2)
'''

CATALOG = {
    "version": 1,
    "generated_at": "2026-09-20T00:00:00Z",
    "source": "https://example.invalid/catalog.json",
    "models": [{
        "id": "sleeper",
        "repo": "mlx-community/Sleeper",
        "revision": "c" * 40,
        "kind": "chat",
        "license": "Apache-2.0",
        "family": None,
        "priority": 1,
        "quant": {"bits": 4, "group_size": 64, "mode": "affine"},
        "memory": {"weights_bytes": 1 * 1024**3, "kv_bytes_per_token": 1024,
                   "peak_estimate_bytes": None},
        "context": {"max_tokens": 1024, "trained_tokens": None},
        "capability": {"arch": None, "min_mem_gib": None},
        "qualification": {
            "generation": {"status": "qualified", "receipt": "r", "date": "2026-09-20"},
            "http": {"status": "qualified", "receipt": "h", "date": "2026-09-20"},
        },
        "recommended": True,
        "serve": {"backend": "mlx-lm", "module": None},
        "availability": {"size_bytes": 1 * 1024**3, "refreshed_at": None},
        "extension": {"download_patterns": ["model.safetensors",
                                            "config.json", "tokenizer.json"]},
    }],
}


class SigtermLaunchTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.home = Path(self.tmp.name) / "home"
        self.pkg = Path(self.tmp.name) / "stubs"
        stub_mlxlm = self.pkg / "mlx_lm"
        stub_mlxlm.mkdir(parents=True)
        (self.pkg / "mlx_omarchy_serve").touch()  # marker only; real pkg via path
        (stub_mlxlm / "__init__.py").write_text(STUB_MLXLM_INIT)
        (stub_mlxlm / "_version.py").write_text(STUB_MLXLM_VERSION)
        (stub_mlxlm / "server.py").write_text(STUB_MLXLM_SERVER)
        self.ready_file = self.pkg / "sleeper-ready"
        # a complete local artifact: the launch-lifecycle path (reserve ->
        # spawn -> SIGTERM -> terminate -> release) needs no hub at all
        self.model_dir = self.home / "sleeper-model"
        self.model_dir.mkdir(parents=True)
        for name in ("model.safetensors", "config.json", "tokenizer.json"):
            (self.model_dir / name).write_bytes(b"x" * 64)

    def launch_env(self):
        return {**os.environ,
                "MLX_OMARCHY_HOME": str(self.home),
                "MLX_OMARCHY_OFFLINE": "1",
                "SLEEPER_READY_FILE": str(self.ready_file),
                "PYTHONPATH": f"{SERVE_DIR}:{self.pkg}"}

    def read_reservations(self):
        path = self.home / "reservations.json"
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {}

    def child_pids(self, parent_pid):
        out = subprocess.run(["pgrep", "-P", str(parent_pid)],
                             capture_output=True, text=True)
        return [int(line) for line in out.stdout.split()]

    def wait_until(self, predicate, timeout=60, interval=0.2):
        deadline = time.time() + timeout
        while time.time() < deadline:
            result = predicate()
            if result:
                return result
            time.sleep(interval)
        return None

    def test_sigterm_releases_reservation_and_kills_child(self):
        cli = subprocess.Popen(
            [sys.executable, "-m", "mlx_omarchy_serve", "serve",
             str(self.model_dir), "--weights-gib", "1", "--yes"],
            env=self.launch_env(),
            cwd=str(REPO_ROOT),
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        # the launch flow admits + reserves BEFORE spawning the child
        reserved = self.wait_until(lambda: self.read_reservations())
        self.assertTrue(reserved, "launch reservation never appeared")
        self.assertTrue(
            self.wait_until(lambda: self.ready_file.is_file()),
            "stub server child never became ready",
        )
        children = self.child_pids(cli.pid)
        self.assertTrue(children, "no live child while reservation is pending")
        child_pid = children[0]

        os.kill(cli.pid, signal.SIGTERM)
        cli.wait(timeout=60)

        # reservation released only after the child is confirmed dead
        deadline = time.time() + 30
        while time.time() < deadline:
            if not self.read_reservations():
                break
            time.sleep(0.2)
        self.assertEqual(self.read_reservations(), {},
                         "SIGTERM must still release the reservation once the "
                         "child is confirmed dead")
        with self.assertRaises(ProcessLookupError):
            for _ in range(50):
                os.kill(child_pid, 0)
                time.sleep(0.1)
            os.kill(child_pid, 0)


if __name__ == "__main__":
    unittest.main()
