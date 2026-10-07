"""Regression test: managed-mode bonsai-2 server releases its reservation
on SIGTERM; SIGKILL-cleanable via the existing owner-token API.

Bug history (Main review):
  _release_reservation(name) called budget.clear_reservation(name) WITHOUT
  the owner token, so the budget API raised "reservation owned by another
  holder" and the exception was swallowed. Result: SIGTERM/kill left a
  dead owner entry on disk; future MLX admission read the orphan and
  under-budgeted MemAvailable.

This test exercises the REAL serve_main subprocess against the CHECKED-IN
REPO/serve source. All imports resolve through REPO/serve and tests/; no
/tmp shadow, no hardcoded staged paths. The spawn child gets an explicit
PYTHONPATH=str(SERVE) so it cannot inherit a shadow path.

OUR packages in REPO/serve are mandatory. A missing/import-broken
mlx_omarchy_bonsai2 or mlx_omarchy_serve in REPO/serve is a
REGRESSION and the test must FAIL, not skip. The mlx_omarchy_serve
runtime budget API is OUR checked-in package too -- the unified
integration checkout carries it alongside mlx_omarchy_bonsai2.
There is no skipUnless guard on these packages.

Registry read semantics: missing file -> empty registry (clean state
before the first launch). JSONDecodeError -> RAISE (a corrupt registry
is not evidence of a clean cleanup; the test must observe actual JSON
validity).
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SERVE = REPO / "serve"
TESTS = Path(__file__).resolve().parent

# OUR package in REPO/serve is mandatory. Assert the source files exist
# before any import -- a missing file is a regression, not a skip case.
_BONSAI_SERVER_PY = SERVE / "mlx_omarchy_bonsai2" / "server.py"
assert _BONSAI_SERVER_PY.is_file(), (
    "mlx_omarchy_bonsai2.server missing at %s -- checkout is broken; "
    "this test must FAIL, not skip." % _BONSAI_SERVER_PY
)

for p in (str(SERVE), str(TESTS)):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# OUR package imports. No skipUnless -- a missing/import-broken
# mlx_omarchy_serve in REPO/serve is a REGRESSION and the test
# collection must FAIL loudly.
from mlx_omarchy_serve import budget  # noqa: E402


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _reservation_path(home: Path) -> Path:
    return home / ".local" / "share" / "mlx-omarchy" / "reservations.json"


def _read_registry(home: Path) -> dict:
    """Read the registry, never silently treating malformed JSON as empty.

    Missing file -> {} (legitimate "no reservations yet" state).
    JSONDecodeError -> raises: a corrupt registry is not evidence of
    a clean cleanup. Tests that depend on this must observe actual
    JSON validity; the helper refuses to launder corrupt input into
    "the entry is gone".
    """
    path = _reservation_path(home)
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _wait_for_resident(home: Path, deadline_s: float = 60.0) -> dict | None:
    """Wait for a resident managed entry. The post-load relabel takes
    time after admission -- polling for state == "resident" avoids the
    stale-pending race that masks the relabel."""
    path = _reservation_path(home)
    end = time.time() + deadline_s
    while time.time() < end:
        if path.exists():
            try:
                data = _read_registry(home)
            except json.JSONDecodeError:
                # Treat corrupt registry as "not yet resident"; keep polling.
                data = {}
            for _name, payload in data.items():
                if payload.get("state") == "resident" and payload.get("owner"):
                    return data
        time.sleep(0.2)
    return None


def _wait_for_clear(home: Path, reservation_name: str, deadline_s: float = 30.0) -> bool:
    """Wait until the named reservation is absent from a valid registry.

    A registry that fails to parse mid-wait is NOT treated as cleared:
    the test will surface that as a leftover entry when reading again,
    rather than silently passing through a corrupt state.
    """
    end = time.time() + deadline_s
    while time.time() < end:
        try:
            data = _read_registry(home)
        except json.JSONDecodeError:
            # Mid-wait corruption: keep polling; do not declare clear.
            time.sleep(0.2)
            continue
        if reservation_name not in data:
            return True
        time.sleep(0.2)
    return False


def _spawn(python: str, pack_dir: Path, home: Path, port: int, *, allow_cpu: bool = True):
    """Spawn serve_main as a subprocess. Imports resolve through REPO/serve
    only (no /tmp/bonsai2-window or any other staged shadow) so CI runs
    against the checked-in source."""
    argv = [
        python,
        "-c",
        (
            "import sys; sys.path.insert(0, %r); "
            "from mlx_omarchy_bonsai2 import serve_main; "
            "serve_main(sys.argv[1:])" % str(SERVE)
        ),
        "--model", str(pack_dir),
        "--host", "127.0.0.1",
        "--port", str(port),
        "--model-id", "bonsai2-release-test",
        "--max-context", "256",
        "--managed",
    ]
    if allow_cpu:
        argv.append("--allow-cpu")
    env = os.environ.copy()
    env["HOME"] = str(home)
    env["MLX_OMARCHY_HOME"] = str(home / ".local" / "share" / "mlx-omarchy")
    env["PYTHONPATH"] = str(SERVE)
    return subprocess.Popen(
        argv,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )


@unittest.skipUnless(importlib.util.find_spec("mlx"), "needs mlx (serve_main loads the pack with it)")
class ReservationReleaseTests(unittest.TestCase):
    """Real serve_main subprocess + real budget API + real reservations.json."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.home = self.tmp / "home"
        self.home.mkdir()
        import bonsai2_fixture  # noqa: F401

        self.pack_dir, _ = bonsai2_fixture.build_tiny_pack(self.tmp / "pack")
        self.port = _free_port()
        self._diag = []
        self._saved_env = {
            "HOME": os.environ.get("HOME"),
            "MLX_OMARCHY_HOME": os.environ.get("MLX_OMARCHY_HOME"),
        }
        os.environ["HOME"] = str(self.home)
        os.environ["MLX_OMARCHY_HOME"] = str(self.home / ".local" / "share" / "mlx-omarchy")

    def tearDown(self):
        for child in getattr(self, "_children", []):
            if child.poll() is None:
                child.kill()
                child.wait(timeout=5)
        for line in self._diag:
            sys.stderr.write(line)
        shutil.rmtree(self.tmp, ignore_errors=True)
        for k, v in self._saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def _capture_diagnostics(self, child: subprocess.Popen):
        try:
            out, err = child.communicate(timeout=2)
        except subprocess.TimeoutExpired:
            out, err = b"", b""
        self._diag.append("\n--- child stdout ---\n%s\n" % out.decode("utf-8", "replace"))
        self._diag.append("\n--- child stderr ---\n%s\n" % err.decode("utf-8", "replace"))

    def test_sigterm_releases_owner_entry(self):
        """SIGTERM after healthy serve must clear the owner entry."""
        child = _spawn(sys.executable, self.pack_dir, self.home, self.port)
        self._children = [child]
        entry = _wait_for_resident(self.home, deadline_s=60)
        if entry is None:
            self._capture_diagnostics(child)
            self.fail("managed server never registered a resident reservation")
        self.assertEqual(
            len(entry), 1, "expected exactly one owner entry, got %r" % entry
        )
        [(reservation_name, payload)] = list(entry.items())
        self.assertTrue(payload.get("owner"), "reservation missing owner token")
        self.assertEqual(payload["state"], "resident")
        self.assertGreater(payload["resident_floor_bytes"], 0)

        import urllib.request
        with urllib.request.urlopen(
            "http://127.0.0.1:%d/health" % self.port, timeout=10
        ) as resp:
            self.assertEqual(resp.status, 200)
            body = json.loads(resp.read())
            self.assertEqual(body["model"], "bonsai2-release-test")
            self.assertEqual(body["reservation_name"], reservation_name)
            self.assertGreater(body["reservation_bytes"], 0)

        child.send_signal(signal.SIGTERM)
        try:
            child.wait(timeout=30)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait(timeout=5)
            self._capture_diagnostics(child)
            self.fail("server did not exit after SIGTERM within 30s")

        self.assertTrue(
            _wait_for_clear(self.home, reservation_name, deadline_s=10),
            "reservation %r still in registry after SIGTERM: %r"
            % (reservation_name, _read_registry(self.home)),
        )

    def test_sigkill_owner_scoped_clear_succeeds(self):
        """SIGKILL is not handled by the server; the operator cleanup
        path uses the existing budget.clear_reservation API with the
        captured owner token after dead-PID is verified. NEVER force-
        clear an unknown or live entry."""
        child = _spawn(sys.executable, self.pack_dir, self.home, self.port)
        self._children = [child]
        entry = _wait_for_resident(self.home, deadline_s=60)
        if entry is None:
            self._capture_diagnostics(child)
            self.fail("managed server never registered a resident reservation")
        [(reservation_name, payload)] = list(entry.items())
        owner_token = payload["owner"]
        child.kill()
        child.wait(timeout=10)

        try:
            os.kill(int(owner_token.split("-")[0][3:]), 0)
            self.fail("server pid still alive after kill")
        except ProcessLookupError:
            pass

        cleared = budget.clear_reservation(reservation_name, owner=owner_token)
        self.assertTrue(
            cleared,
            "budget.clear_reservation returned False for known owner",
        )
        self.assertNotIn(
            reservation_name,
            _read_registry(self.home),
            "owner-scoped clear did not remove the entry",
        )


if __name__ == "__main__":
    unittest.main()
