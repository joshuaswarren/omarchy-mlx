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

import contextlib
import faulthandler
import importlib
import importlib.util
import io
import json
import os
import queue
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path
from unittest import mock

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
            "from mlx_omarchy_bonsai2.server import serve_main; "
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


class ManagedHomePinAndReleaseSafetyTests(unittest.TestCase):
    """Mlx-free in-process tests for --managed home pinning and release gating.

    Drives the real serve_main() and real mlx_omarchy_serve.budget against
    temp registries so CI (which has no mlx installed) runs every case.
    """

    def setUp(self):
        self.tmp = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.home_a = self.tmp / "registry-a"
        self.home_b = self.tmp / "registry-b"
        self.user_a = self.tmp / "user-a"
        self.user_b = self.tmp / "user-b"
        self.pack_dir = self.tmp / "pack"
        self.pack_dir.mkdir()
        self.server = importlib.import_module("mlx_omarchy_bonsai2.server")
        self.facts = {
            "total_header_bytes": 1024 * 1024,
            "kv_bytes_per_token": 1024,
            "max_position_embeddings": 256,
        }
        self.resident_bytes = 512 * 1024

    def _make_state(self, args, *, jobs=()):
        q = queue.Queue()
        for job in jobs:
            q.put((job, queue.Queue()))
        q.put((None, None))
        info = {
            "resident_bytes": self.resident_bytes,
            "kv_bytes_per_token": self.facts["kv_bytes_per_token"],
            "max_position_embeddings": self.facts["max_position_embeddings"],
            "packed_modules": 1,
            "excluded_bytes": {},
            "attribution": "stub",
        }
        return types.SimpleNamespace(
            info=info,
            catalog_id=args.model_id,
            device="cpu",
            max_context=args.max_context or info["max_position_embeddings"],
            reservation_name=args.reservation_name,
            owner=args.owner,
            reservation_bytes=None,
            job_queue=q,
        )

    def _run_serve(
        self,
        *,
        managed=True,
        extra_args=(),
        state_factory=None,
        pack_footprint_side_effect=None,
        fake_budget=None,
        no_home=False,
    ):
        argv = [
            "--model", str(self.pack_dir),
            "--host", "127.0.0.1",
            "--port", "0",
            "--model-id", "bonsai2-pin-test",
            "--allow-cpu",
        ]
        if managed:
            argv.append("--managed")
        argv.extend(extra_args)

        mlx_core = types.ModuleType("mlx.core")
        mlx_core.gpu = "gpu"
        mlx_core.default_device = lambda: "cpu"
        mlx_pkg = types.ModuleType("mlx")
        mlx_pkg.core = mlx_core

        pack_fp = mock.MagicMock(
            return_value=dict(self.facts),
            side_effect=pack_footprint_side_effect,
        )
        loader_mod = types.ModuleType("mlx_omarchy_bonsai2.loader")
        loader_mod.pack_footprint = pack_fp

        state_ctor = mock.MagicMock(
            side_effect=state_factory or (lambda args: self._make_state(args))
        )

        modules_patch = {
            "mlx": mlx_pkg,
            "mlx.core": mlx_core,
            "mlx_omarchy_bonsai2.loader": loader_mod,
        }
        if fake_budget is not None:
            modules_patch["mlx_omarchy_serve"] = types.SimpleNamespace(budget=fake_budget)

        err = io.StringIO()
        out = io.StringIO()
        with contextlib.ExitStack() as stack:
            stack.enter_context(mock.patch.dict(sys.modules, modules_patch))
            stack.enter_context(mock.patch.dict(os.environ))
            os.environ.pop("MLX_OMARCHY_SERVE_IDS_PROBE", None)
            if no_home:
                os.environ.pop("MLX_OMARCHY_HOME", None)
                os.environ.pop("HOME", None)
                stack.enter_context(
                    mock.patch.object(
                        Path,
                        "home",
                        side_effect=RuntimeError("Could not determine home directory."),
                    )
                )
            else:
                os.environ["MLX_OMARCHY_HOME"] = str(self.home_a)
                os.environ["HOME"] = str(self.user_a)
            stack.enter_context(mock.patch.object(self.server, "Bonsai2State", state_ctor))
            stack.enter_context(mock.patch.object(signal, "signal"))
            stack.enter_context(mock.patch.object(faulthandler, "enable"))
            stack.enter_context(mock.patch.object(faulthandler, "dump_traceback_later"))
            stack.enter_context(mock.patch.object(budget, "mem_available", return_value=1 << 40))
            stack.enter_context(mock.patch.object(budget, "gpu_client_cap", return_value=None))
            stack.enter_context(contextlib.redirect_stderr(err))
            stack.enter_context(contextlib.redirect_stdout(out))
            exc = None
            try:
                self.server.serve_main(argv)
            except BaseException as caught:
                exc = caught
        return exc, err.getvalue(), state_ctor, pack_fp

    def _drop_env_and_break_home(self, home_broken):
        os.environ.pop("MLX_OMARCHY_HOME", None)
        os.environ.pop("HOME", None)
        home_broken[0] = True

    def _patch_breakable_home(self):
        home_broken = [False]
        real_home = Path.home

        def maybe_home():
            if home_broken[0]:
                raise RuntimeError("Could not determine home directory.")
            return real_home()

        return home_broken, mock.patch.object(Path, "home", side_effect=maybe_home)

    def test_managed_unresolvable_home_exits_3_before_pack_and_without_release(self):
        exc, err, state_ctor, pack_fp = self._run_serve(managed=True, no_home=True)
        self.assertIsInstance(exc, SystemExit, err)
        self.assertEqual(exc.code, 3, err)
        lines = err.splitlines()
        self.assertEqual(len(lines), 1, err)
        self.assertTrue(lines[0].startswith("bonsai2: --managed: "), lines[0])
        self.assertIn("budget registry", lines[0])
        self.assertIn("MLX_OMARCHY_HOME", lines[0])
        self.assertNotIn("reservation release", err)
        state_ctor.assert_not_called()
        pack_fp.assert_not_called()

    def test_managed_pinned_home_survives_env_change_during_load(self):
        for mode in ("repoint", "drop"):
            with self.subTest(mode=mode):
                if self.home_a.exists():
                    shutil.rmtree(self.home_a)
                if self.home_b.exists():
                    shutil.rmtree(self.home_b)
                seen = {}
                home_broken, patch_home = self._patch_breakable_home()

                def state_factory(args, _mode=mode, _seen=seen, _broken=home_broken):
                    if _mode == "repoint":
                        os.environ["MLX_OMARCHY_HOME"] = str(self.home_b)
                        os.environ["HOME"] = str(self.user_b)
                    else:
                        self._drop_env_and_break_home(_broken)

                    def inspect_registry(_emit):
                        _seen["name"] = args.reservation_name
                        _seen["owner"] = args.owner
                        _seen["entries"] = budget.load_reservations(self.home_a)

                    return self._make_state(args, jobs=(inspect_registry,))

                with patch_home:
                    exc, err, _, _ = self._run_serve(managed=True, state_factory=state_factory)
                self.assertIsNone(exc, err)
                self.assertIn("name", seen)
                self.assertIn(seen["name"], seen["entries"])
                entry = seen["entries"][seen["name"]]
                self.assertEqual(entry["state"], "resident")
                self.assertEqual(entry["owner"], seen["owner"])
                self.assertEqual(entry["resident_floor_bytes"], self.resident_bytes)
                self.assertEqual(budget.load_reservations(self.home_a), {})
                self.assertFalse((self.home_b / "reservations.json").exists())
                self.assertNotIn("reservation release", err)

    def test_pre_admission_refusals_release_nothing(self):
        with self.subTest("managed: atomic API missing"):
            clear_calls = []
            fake_budget = types.SimpleNamespace(
                admit=lambda required, home=None: None,
                clear_reservation=lambda *a, **k: clear_calls.append((a, k)),
            )
            exc, err, state_ctor, pack_fp = self._run_serve(managed=True, fake_budget=fake_budget)
            self.assertIsInstance(exc, SystemExit, err)
            self.assertEqual(exc.code, 3, err)
            self.assertEqual(clear_calls, [])
            self.assertNotIn("reservation release", err)
            state_ctor.assert_not_called()
            pack_fp.assert_not_called()

        with self.subTest("managed: budget without default_home"):
            clear_calls = []
            atomic_spy = mock.MagicMock()
            fake_budget = types.SimpleNamespace(
                admit_and_reserve=atomic_spy,
                clear_reservation=lambda *a, **k: clear_calls.append((a, k)),
            )
            exc, err, state_ctor, pack_fp = self._run_serve(managed=True, fake_budget=fake_budget)
            self.assertIsInstance(exc, SystemExit, err)
            self.assertEqual(exc.code, 3, err)
            self.assertEqual(clear_calls, [])
            atomic_spy.assert_not_called()
            state_ctor.assert_not_called()
            pack_fp.assert_not_called()

        with self.subTest("managed: pack_footprint error"):
            clear_calls = []
            atomic_spy = mock.MagicMock()
            fake_budget = types.SimpleNamespace(
                default_home=lambda: self.home_a,
                estimate_required=budget.estimate_required,
                admit_and_reserve=atomic_spy,
                clear_reservation=lambda *a, **k: clear_calls.append((a, k)),
            )
            exc, err, state_ctor, _ = self._run_serve(
                managed=True,
                fake_budget=fake_budget,
                pack_footprint_side_effect=ValueError("corrupt pack header"),
            )
            self.assertIsInstance(exc, SystemExit, err)
            self.assertEqual(exc.code, 3, err)
            self.assertEqual(clear_calls, [])
            atomic_spy.assert_not_called()
            self.assertNotIn("reservation release", err)
            state_ctor.assert_not_called()

        for exc_cls in (budget.BudgetError, ValueError):
            with self.subTest("managed: admission refusal", exc=exc_cls.__name__):
                clear_calls = []

                def refuse(name, byte_count, *, note="", owner=None, home=None, state="pending",
                           _cls=exc_cls):
                    raise _cls("does not fit")

                fake_budget = types.SimpleNamespace(
                    default_home=lambda: self.home_a,
                    estimate_required=budget.estimate_required,
                    admit_and_reserve=refuse,
                    clear_reservation=lambda *a, **k: clear_calls.append((a, k)),
                )
                exc, err, state_ctor, _ = self._run_serve(managed=True, fake_budget=fake_budget)
                self.assertIsInstance(exc, SystemExit, err)
                self.assertEqual(exc.code, 3, err)
                self.assertEqual(clear_calls, [])
                self.assertNotIn("reservation release", err)
                state_ctor.assert_not_called()

        with self.subTest("standalone: weight-load failure before set_reservation"):
            if self.home_a.exists():
                shutil.rmtree(self.home_a)

            def boom(_args):
                raise RuntimeError("weights unreadable")

            with mock.patch.object(
                budget, "clear_reservation", wraps=budget.clear_reservation
            ) as clear_spy:
                exc, err, _, _ = self._run_serve(managed=False, state_factory=boom)
            self.assertIsInstance(exc, RuntimeError)
            self.assertEqual(str(exc), "weights unreadable")
            clear_spy.assert_not_called()
            self.assertFalse((self.home_a / "reservations.json").exists())
            self.assertNotIn("reservation release", err)

        with self.subTest("standalone: set_reservation fails, serving continues"):
            with mock.patch.object(
                budget, "clear_reservation", wraps=budget.clear_reservation
            ) as clear_spy:
                exc, err, _, _ = self._run_serve(managed=False, no_home=True)
            self.assertIsNone(exc, err)
            self.assertIn("bonsai2: memory reservation skipped", err)
            clear_spy.assert_not_called()
            self.assertNotIn("reservation release", err)

    def test_post_admission_failures_release_with_pinned_home(self):
        cases = (
            ("weight-load RuntimeError", (), "load_err"),
            ("KeyboardInterrupt during load", (), "kbd"),
            ("max-context overflow", ("--max-context", "512"), "max_ctx"),
            ("relabel failure", (), "relabel_err"),
            ("SIGTERM after admit commit", (), "sigterm_admit"),
        )
        for label, extra_args, kind in cases:
            with self.subTest(label):
                if self.home_a.exists():
                    shutil.rmtree(self.home_a)
                admitted = {}
                home_broken, patch_home = self._patch_breakable_home()
                real_admit = budget.admit_and_reserve

                def wrapped_admit(*a, **k):
                    res = real_admit(*a, **k)
                    admitted["name"] = a[0]
                    admitted["owner"] = k["owner"]
                    admitted["home"] = k.get("home")
                    if kind == "sigterm_admit":
                        self._drop_env_and_break_home(home_broken)
                        raise SystemExit(0)
                    return res

                def state_factory(args, _kind=kind, _broken=home_broken):
                    self._drop_env_and_break_home(_broken)
                    if _kind == "load_err":
                        raise RuntimeError("weight load failed")
                    if _kind == "kbd":
                        raise KeyboardInterrupt
                    return self._make_state(args)

                with contextlib.ExitStack() as stack:
                    stack.enter_context(patch_home)
                    stack.enter_context(
                        mock.patch.object(budget, "admit_and_reserve", side_effect=wrapped_admit)
                    )
                    clear_spy = stack.enter_context(
                        mock.patch.object(
                            budget, "clear_reservation", wraps=budget.clear_reservation
                        )
                    )
                    relabel_spy = None
                    if kind == "relabel_err":
                        relabel_spy = stack.enter_context(
                            mock.patch.object(
                                budget,
                                "set_reservation_state",
                                side_effect=budget.BudgetError("relabel boom"),
                            )
                        )
                    exc, err, _, _ = self._run_serve(
                        managed=True, extra_args=extra_args, state_factory=state_factory
                    )

                if kind == "load_err":
                    self.assertIsInstance(exc, RuntimeError)
                    self.assertEqual(str(exc), "weight load failed")
                elif kind == "kbd":
                    self.assertIsNone(exc, err)
                elif kind == "max_ctx":
                    self.assertIsInstance(exc, SystemExit, err)
                    self.assertEqual(exc.code, 2, err)
                elif kind == "relabel_err":
                    self.assertIsInstance(exc, SystemExit, err)
                    self.assertEqual(exc.code, 3, err)
                    self.assertEqual(relabel_spy.call_args.kwargs.get("home"), self.home_a)
                    self.assertIn("resident transition failed", err)
                elif kind == "sigterm_admit":
                    self.assertIsInstance(exc, SystemExit, err)
                    self.assertEqual(exc.code, 0, err)

                self.assertEqual(budget.load_reservations(self.home_a), {})
                self.assertNotIn("reservation release", err)
                clear_spy.assert_called_once()
                self.assertEqual(clear_spy.call_args.args[0], admitted["name"])
                self.assertEqual(admitted["home"], self.home_a)
                self.assertEqual(clear_spy.call_args.kwargs["owner"], admitted["owner"])
                self.assertEqual(clear_spy.call_args.kwargs.get("home"), self.home_a)

    def test_standalone_relabel_failure_after_set_reservation_still_clears(self):
        """Control: standalone set_reservation succeeded, relabel failed -> still clears."""
        observed = {}

        def fail_relabel(name, state, **kwargs):
            observed["pending"] = budget.load_reservations(self.home_a)
            observed["name"] = name
            observed["owner"] = kwargs.get("owner")
            raise budget.BudgetError("relabel boom")

        with mock.patch.object(budget, "set_reservation_state", side_effect=fail_relabel), \
                mock.patch.object(
                    budget, "clear_reservation", wraps=budget.clear_reservation
                ) as clear_spy:
            exc, err, _, _ = self._run_serve(managed=False)

        self.assertIsNone(exc, err)
        self.assertIn(observed["name"], observed["pending"])
        self.assertEqual(observed["pending"][observed["name"]]["state"], "pending")
        clear_spy.assert_called_once()
        self.assertEqual(clear_spy.call_args.args[0], observed["name"])
        self.assertEqual(clear_spy.call_args.kwargs["owner"], observed["owner"])
        self.assertIsNone(clear_spy.call_args.kwargs.get("home"))
        self.assertEqual(budget.load_reservations(self.home_a), {})
        self.assertIn("reservation resident transition skipped", err)
        self.assertNotIn("reservation release", err)


if __name__ == "__main__":
    unittest.main()
