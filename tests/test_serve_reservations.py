"""Reservation API contract: validation, admission invariants, atomicity.

Stdlib unittest only. Tests use a temporary MLX_OMARCHY_HOME so the real
~/.local/share/mlx-omarchy is untouched. Each test defends a behavior a
plausible bug would break.
"""

import json
import multiprocessing
import os
import subprocess
import tempfile
import threading
import unittest
import unittest.mock
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(REPO_ROOT / "serve"))

from mlx_omarchy_serve import budget  # noqa: E402

GiB = 1024**3


def _make_home():
    """Return (home_dir, cleanup_fn). The home has a writable reservations dir."""
    tmp = tempfile.TemporaryDirectory()
    home = Path(tmp.name)
    (home / "reservations.json.lock").touch()  # ensure parent exists for set_reservation
    return home, tmp.cleanup


class ReservationInputValidationTests(unittest.TestCase):
    def setUp(self):
        self.home, self.cleanup = _make_home()
        self.addCleanup(self.cleanup)
        self._patcher = unittest.mock.patch.object(budget, "default_home",
                                                   lambda: self.home)
        self._patcher.start()
        self.addCleanup(self._patcher.stop)

    def test_reserve_rejects_zero_bytes(self):
        with self.assertRaises(budget.BudgetError):
            budget.set_reservation("svc", 0)

    def test_reserve_rejects_negative_bytes(self):
        with self.assertRaises(budget.BudgetError):
            budget.set_reservation("svc", -1)

    def test_reserve_rejects_empty_name(self):
        with self.assertRaises(budget.BudgetError):
            budget.set_reservation("", 1024)


class ReservationLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.home, self.cleanup = _make_home()
        self.addCleanup(self.cleanup)
        self._patcher = unittest.mock.patch.object(budget, "default_home",
                                                   lambda: self.home)
        self._patcher.start()
        self.addCleanup(self._patcher.stop)

    def test_set_reservation_state_rejects_unknown_name(self):
        with self.assertRaises(budget.BudgetError) as cm:
            budget.set_reservation_state("never-set", "resident")
        self.assertIn("never-set", str(cm.exception))

    def test_clear_reservation_returns_false_for_missing(self):
        self.assertFalse(budget.clear_reservation("never-set"))


class AdmissionInvariantTests(unittest.TestCase):
    def setUp(self):
        self.home, self.cleanup = _make_home()
        self.addCleanup(self.cleanup)
        self._patcher = unittest.mock.patch.object(budget, "default_home",
                                                   lambda: self.home)
        self._patcher.start()
        self.addCleanup(self._patcher.stop)

    def test_admit_safety_reserve_subtracted(self):
        with unittest.mock.patch.object(budget, "mem_available", lambda: int(16 * GiB)):
            adm = budget.admit(8 * GiB)
        self.assertTrue(adm.fits)
        # 16 - 2 (safety) - 0 (pending reservations) - 8 (model) = 6
        self.assertEqual(adm.headroom, int(6 * GiB))

    def test_admit_refuses_when_over_floor(self):
        with unittest.mock.patch.object(budget, "mem_available", lambda: int(16 * GiB)):
            adm = budget.admit(20 * GiB)
        self.assertFalse(adm.fits)
        self.assertLess(adm.headroom, 0)

    def test_pending_reservations_count_against_admit_resident_does_not(self):
        """A pending reservation subtracts from headroom; a fully materialized
        resident one with resident_floor_bytes==bytes subtracts NOTHING.
        This is the load-bearing invariant for co-serving — a regression
        would OOM."""
        budget.set_reservation("pending-svc", int(4 * GiB),
                               note="p", home=self.home, state="pending")
        # Resident entry: a fully materialized load — floor matches bytes,
        # so the budget treats the entire reservation as already-counted-in-
        # MemAvailable (its unmaterialized peak headroom is zero).
        budget.set_reservation("resident-svc", int(4 * GiB),
                               note="r", home=self.home, state="resident")
        # Patch the file to set resident_floor_bytes on the resident entry.
        import json as _json
        regs = budget.load_reservations(self.home)
        regs["resident-svc"]["resident_floor_bytes"] = int(4 * GiB)
        budget.reservations_path(self.home).write_text(
            _json.dumps(regs, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        with unittest.mock.patch.object(budget, "mem_available", lambda: int(16 * GiB)):
            adm = budget.admit(8 * GiB)
        # 16 - 2 (safety) - 4 (only the pending one counts) - 8 (model) = 2
        self.assertEqual(adm.headroom, int(2 * GiB),
                         f"pending vs resident invariant violated: headroom={adm.headroom}")
        self.assertTrue(adm.fits)


def _child_set_with_home(task):
    """Sets a reservation named by the task index; returns True if it succeeded.
    The name must not come from the pid: a pool worker can take two tasks."""
    home_path, index = task
    try:
        budget.set_reservation(f"child-{index}", int(1 * GiB), note="atomic",
                               home=Path(home_path))
        return True
    except Exception:  # pragma: no cover (only fires on a real bug)
        return False


class ConcurrentSetReservationTests(unittest.TestCase):
    def test_concurrent_set_reservation_atomic(self):
        """N=8 child processes call set_reservation with unique names.
        Parent verifies all N reservations landed (no lost updates)."""
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            # Patch default_home in this parent so the children (which import
            # the budget module) see the same home.
            with unittest.mock.patch.object(budget, "default_home", lambda: home):
                N = 8
                with multiprocessing.Pool(processes=N,
                                          initializer=_install_patch,
                                          initargs=(str(home),)) as pool:
                    results = pool.map(_child_set_with_home,
                                       [(str(home), i) for i in range(N)])
                self.assertTrue(all(results),
                                f"some child failed: {results}")
                # Parent reads the file directly (bypassing the patch is fine;
                # it just queries the actual file).
                regs = budget.load_reservations(home)
                self.assertEqual(len(regs), N,
                                 f"lost update: expected {N} entries, got {len(regs)}")
                # Each entry should be exactly 1 GiB.
                for name, val in regs.items():
                    self.assertEqual(val["bytes"], int(1 * GiB),
                                     f"{name} has wrong bytes")


def _deep_list(depth):
    """A depth-deep nested list, built without recursion."""
    deep = []
    for _ in range(depth):
        deep = [deep]
    return deep


def _raised_on_small_stack(fn):
    """Run fn in a thread with a 4 MiB stack, so the nesting limit does not
    depend on the host's ulimit -s. Returns the exception raised, or None."""
    caught = []

    def run():
        try:
            fn()
        except BaseException as exc:  # returned for the caller to assert on
            caught.append(exc)

    old = threading.stack_size(4 * 1024 * 1024)
    try:
        worker = threading.Thread(target=run)
        worker.start()
    finally:
        threading.stack_size(old)
    worker.join()
    return caught[0] if caught else None


class UnreadableReservationsFileTests(unittest.TestCase):
    """A reservations file that is not valid UTF-8 JSON, or nests too
    deeply, is a BudgetError (CLI exit 2), never a raw traceback."""

    CASES = {
        "invalid_utf8": b'{"x": {"bytes": 1, "note": "\xff"}}',
        "deep_nesting": b"[" * 200000 + b"]" * 200000,
        "huge_int": b'{"x": {"bytes": ' + b"1" * 5000 + b"}}",
    }

    def _home_with(self, raw):
        home, cleanup = _make_home()
        self.addCleanup(cleanup)
        (home / budget.RESERVATIONS_FILE).write_bytes(raw)
        return home

    def _load_on_small_stack(self, home):
        return _raised_on_small_stack(lambda: budget.load_reservations(home))

    def test_load_raises_budget_error(self):
        self.addCleanup(sys.set_int_max_str_digits, sys.get_int_max_str_digits())
        sys.set_int_max_str_digits(4300)  # the CLI child drops PYTHONINTMAXSTRDIGITS too
        for label, raw in self.CASES.items():
            with self.subTest(label):
                exc = self._load_on_small_stack(self._home_with(raw))
                self.assertIsInstance(exc, budget.BudgetError)
                self.assertIn("refusing to guess", str(exc))

    def test_deeply_nested_field_raises_budget_error(self):
        # Decodes on 3.14, then the validator's repr() recurses.
        for field in ("note", "state"):
            with self.subTest(field):
                home = self._home_with(b"{}")
                parsed = {"x": {"bytes": 1, field: _deep_list(200000)}}
                with unittest.mock.patch.object(budget.json, "loads",
                                                return_value=parsed):
                    exc = self._load_on_small_stack(home)
                self.assertIsInstance(exc, budget.BudgetError)
                self.assertIn("refusing to guess", str(exc))

    def test_cli_unreserve_exits_2(self):
        for label, raw in self.CASES.items():
            with self.subTest(label):
                home = self._home_with(raw)
                env = dict(os.environ, MLX_OMARCHY_HOME=str(home),
                           PYTHONPATH=str(REPO_ROOT / "serve"))
                env.pop("PYTHONINTMAXSTRDIGITS", None)
                # Pin the child's stack at 8 MiB before exec: Python 3.14
                # bounds JSON nesting by the C stack, so a bigger ulimit -s
                # would let the deep payload decode.
                proc = subprocess.run(
                    ["/bin/sh", "-c", 'ulimit -S -s 8192 2>/dev/null; exec "$@"', "sh",
                     sys.executable, "-m", "mlx_omarchy_serve", "unreserve", "x"],
                    env=env, capture_output=True, text=True, timeout=60)
                self.assertEqual(proc.returncode, 2, proc.stderr)
                self.assertIn("refusing to guess", proc.stderr)
                self.assertNotIn("Traceback", proc.stderr)


GOOD_CLAIM = {"name": "chat", "claim_token": "pid1-0123456789ab",
              "bytes": 1024, "pair_id": "everyday"}

MISTYPED_CLAIM_FIELDS = (
    ("name", 1), ("name", ""), ("name", ["chat"]), ("name", True),
    ("claim_token", 5), ("claim_token", ""), ("claim_token", {"t": 1}),
    ("bytes", True), ("bytes", False), ("bytes", 1.5), ("bytes", 1024.0),
    ("bytes", 0), ("bytes", -1), ("bytes", "1024"), ("bytes", [1024]),
    ("pair_id", 7), ("pair_id", ["everyday"]), ("pair_id", True),
)


def _claim_raw(key, value):
    payload = dict(GOOD_CLAIM)
    payload[key] = value
    return json.dumps(payload).encode()


def _claim_literal(literal):
    """A payload whose bytes field is a bare JSON literal (NaN, Infinity)."""
    return (b'{"name": "chat", "claim_token": "pid1-0123456789ab", "bytes": '
            + literal + b'}')


class ClaimPipeValidationTests(unittest.TestCase):
    """The managed child's claim pipe fails closed: a malformed fd or a
    malformed, deep or mistyped payload is a BudgetError (the worker's
    exit-3 path), never a raw traceback and never a coerced claim."""

    def setUp(self):
        self.home, self.cleanup = _make_home()
        self.addCleanup(self.cleanup)

    def pipe_fd(self, raw):
        """A real OS pipe carrying raw; the writer closes its end like the
        parent manager does. Returns the read end, closed at cleanup."""
        r, w = os.pipe()

        def write_all():
            try:
                view = memoryview(raw)
                while view:
                    view = view[os.write(w, view):]
            except OSError:
                pass  # reader closed early (test failure path)
            finally:
                os.close(w)

        writer = threading.Thread(target=write_all)
        writer.start()
        self.addCleanup(writer.join)
        self.addCleanup(os.close, r)
        return r

    def claim_from_pipe(self, raw):
        """claim_from_env over a real pipe, with claim_reservation stubbed
        so a payload that slips past validation is caught, not claimed."""
        fd = self.pipe_fd(raw)
        claim = unittest.mock.Mock(return_value={"claimed": True})
        with unittest.mock.patch.dict(os.environ, {budget.CLAIM_FD_ENV: str(fd)}), \
                unittest.mock.patch.object(budget, "claim_reservation", claim):
            try:
                return budget.claim_from_env(home=self.home), claim
            except budget.BudgetError as exc:
                return exc, claim

    def test_valid_payloads_pass_through_unchanged(self):
        no_pair = {k: v for k, v in GOOD_CLAIM.items() if k != "pair_id"}
        for payload in (GOOD_CLAIM, no_pair, dict(GOOD_CLAIM, pair_id=None),
                        dict(GOOD_CLAIM, pair_id=""), dict(GOOD_CLAIM, extra="kept")):
            with self.subTest(payload=payload):
                fd = self.pipe_fd(json.dumps(payload).encode() + b"\n")
                self.assertEqual(budget.read_claim_fd(fd), payload)
        # "" is forwarded as-is so claim_reservation refuses it as a pair
        # mismatch; it must not collapse to None, which skips the pair check.
        for payload, pair in ((GOOD_CLAIM, "everyday"), (no_pair, None),
                              (dict(GOOD_CLAIM, pair_id=""), "")):
            with self.subTest(pair_id=pair):
                result, claim = self.claim_from_pipe(json.dumps(payload).encode())
                self.assertEqual(result, {"claimed": True})
                claim.assert_called_once_with(
                    "chat", "pid1-0123456789ab", expected_bytes=1024,
                    home=self.home, expected_pair=pair)

    def test_mistyped_fields_raise_budget_error(self):
        for key, value in MISTYPED_CLAIM_FIELDS:
            with self.subTest(key=key, value=value):
                with self.assertRaises(budget.BudgetError) as cm:
                    budget.read_claim_fd(self.pipe_fd(_claim_raw(key, value)))
                self.assertIn(f"claim payload field {key!r}", str(cm.exception))

    def test_non_finite_bytes_raise_budget_error(self):
        for literal in (b"NaN", b"Infinity", b"-Infinity"):
            with self.subTest(literal=literal):
                with self.assertRaises(budget.BudgetError) as cm:
                    budget.read_claim_fd(self.pipe_fd(_claim_literal(literal)))
                self.assertIn("claim payload field 'bytes'", str(cm.exception))

    def test_deeply_nested_payload_raises_budget_error(self):
        # 64 KiB of "[" is the largest payload the pipe accepts. 3.11 and
        # most 3.14 builds overflow the decoder on the 4 MiB stack, and that
        # RecursionError must become a BudgetError; a 3.14 build with smaller
        # frames decodes it all and refuses the unclosed array instead. The
        # stack stays 4 MiB because 3.14 on musl and macOS assumes a 4 MB
        # stack rather than reading it. The twin below injects the
        # RecursionError on any build.
        fd = self.pipe_fd(b"[" * (64 * 1024))
        exc = _raised_on_small_stack(lambda: budget.read_claim_fd(fd))
        self.assertIsInstance(exc, budget.BudgetError)
        self.assertIn("claim payload is not valid JSON", str(exc))

    def test_decoder_recursion_error_raises_budget_error(self):
        # Stack-independent twin of the deep-payload test.
        fd = self.pipe_fd(json.dumps(GOOD_CLAIM).encode())
        with unittest.mock.patch.object(budget.json, "loads",
                                        side_effect=RecursionError("deep")):
            with self.assertRaises(budget.BudgetError) as cm:
                budget.read_claim_fd(fd)
        self.assertIn("claim payload is not valid JSON", str(cm.exception))

    def test_non_int_fd_raises_budget_error_without_reading(self):
        # bool is an int: True would read the process's stdout (fd 1).
        read = unittest.mock.Mock(side_effect=AssertionError("os.read called"))
        with unittest.mock.patch.object(budget.os, "read", read):
            for fd in (True, False, "3", b"3", 3.0, None):
                with self.subTest(fd=fd):
                    with self.assertRaises(budget.BudgetError) as cm:
                        budget.read_claim_fd(fd)
                    self.assertIn("claim fd must be an int", str(cm.exception))
        read.assert_not_called()

    def test_out_of_range_fd_raises_budget_error(self):
        for fd in (2**31, 2**70, -(2**70)):
            with self.subTest(fd=fd):
                with self.assertRaises(budget.BudgetError) as cm:
                    budget.read_claim_fd(fd)
                self.assertIn("claim pipe unreadable", str(cm.exception))

    def test_claim_from_env_rejects_mistyped_payload_before_claiming(self):
        cases = [(key, _claim_raw(key, value)) for key, value in MISTYPED_CLAIM_FIELDS]
        cases += [("bytes", _claim_raw("bytes", "abc")),
                  ("bytes", _claim_raw("bytes", {})),
                  ("bytes", _claim_literal(b"NaN")),
                  ("bytes", _claim_literal(b"Infinity"))]
        for key, raw in cases:
            with self.subTest(raw=raw):
                result, claim = self.claim_from_pipe(raw)
                self.assertIsInstance(result, budget.BudgetError)
                self.assertIn(f"claim payload field {key!r}", str(result))
                claim.assert_not_called()

    def test_claim_from_env_out_of_range_fd_raises_budget_error(self):
        with unittest.mock.patch.dict(os.environ,
                                      {budget.CLAIM_FD_ENV: str(2**70)}):
            with self.assertRaises(budget.BudgetError) as cm:
                budget.claim_from_env(home=self.home)
        self.assertIn("claim pipe unreadable", str(cm.exception))


def _install_patch(home_str):  # pragma: no cover (multiprocessing init)
    """Child initializer: rebind budget.default_home before the target runs."""
    import os
    home = Path(home_str)
    # Rebind budget module default_home so child calls land in our temp dir.
    import sys as _sys
    _sys.path.insert(0, str(Path(home_str).parent.parent / "serve"))
    # Re-import after path injection to pick up the same module
    from mlx_omarchy_serve import budget as _budget
    _budget.default_home = lambda: home


if __name__ == "__main__":
    unittest.main()
