"""Reservation API contract: validation, admission invariants, atomicity.

Stdlib unittest only. Tests use a temporary MLX_OMARCHY_HOME so the real
~/.local/share/mlx-omarchy is untouched. Each test defends a behavior a
plausible bug would break.
"""

import multiprocessing
import tempfile
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
