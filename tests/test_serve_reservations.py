"""Reservation API contract: validation, admission invariants, atomicity.

Stdlib unittest only. Tests use a temporary MLX_OMARCHY_HOME so the real
~/.local/share/mlx-omarchy is untouched. Each test defends a behavior a
plausible bug would break.
"""

import contextlib
import io
import json
import multiprocessing
import os
import socket
import stat
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


class TransactionLockOpenTests(unittest.TestCase):
    """The transaction lock opens without truncating, without following a
    symlink and without blocking; a damaged lock path is a BudgetError and
    is left as it was. A regular lock keeps its inode and still excludes."""

    UNUSABLE = r"^reservations lock unusable \(.*reservations\.json\.lock'\); refusing to lock$"

    def setUp(self):
        self.home, cleanup = _make_home()
        self.addCleanup(cleanup)
        self.path = budget.reservations_path(self.home)
        self.lock = self.path.with_name("reservations.json.lock")
        self.lock.unlink()

    def _keep_reservation(self):
        budget.set_reservation("keep", 4096, note="existing", home=self.home)
        self.lock.unlink()
        return self.path.read_bytes()

    def test_symlink_lock_refused_and_targets_untouched(self):
        before = self._keep_reservation()
        victim = self.home / "notes.txt"
        victim.write_bytes(b"user data\n" * 100)
        for target in (self.path.name, str(victim), "/dev/null"):
            with self.subTest(target=target):
                try:
                    self.lock.symlink_to(target)
                    with self.assertRaisesRegex(budget.BudgetError, self.UNUSABLE):
                        budget.set_reservation("svc", 1024, home=self.home)
                    self.assertEqual(os.readlink(self.lock), target)
                finally:
                    self.lock.unlink(missing_ok=True)
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(victim.read_bytes(), b"user data\n" * 100)
        self.assertEqual(set(budget.load_reservations(self.home)), {"keep"})

    def test_dangling_symlink_lock_refused_without_creating_target(self):
        target = self.home / "elsewhere"
        self.lock.symlink_to(target)
        with self.assertRaisesRegex(budget.BudgetError, self.UNUSABLE):
            budget.set_reservation("svc", 1024, home=self.home)
        self.assertFalse(os.path.lexists(target))
        self.assertFalse(self.path.exists())

    def test_every_transaction_refuses_a_symlink_lock(self):
        before = self._keep_reservation()
        self.lock.symlink_to(self.path.name)
        calls = {
            "set_reservation": lambda: budget.set_reservation(
                "svc", 1024, home=self.home),
            "set_reservation_state": lambda: budget.set_reservation_state(
                "keep", "resident", home=self.home),
            "clear_reservation": lambda: budget.clear_reservation(
                "keep", home=self.home),
            "admit_and_reserve": lambda: budget.admit_and_reserve(
                "svc", 1024, owner="o", home=self.home, available_bytes=GiB),
            "admit_and_reserve_batch": lambda: budget.admit_and_reserve_batch(
                [("svc", 1024, "")], pair_id="p", home=self.home,
                available_bytes=GiB),
            "claim_reservation": lambda: budget.claim_reservation(
                "keep", "token", expected_bytes=4096, home=self.home),
            "resize_reservation": lambda: budget.resize_reservation(
                "keep", 8192, owner="o", home=self.home, available_bytes=GiB),
        }
        for name, call in calls.items():
            with self.subTest(call=name):
                with self.assertRaisesRegex(budget.BudgetError, self.UNUSABLE):
                    call()
        self.assertEqual(self.path.read_bytes(), before)

    @unittest.skipUnless(hasattr(os, "mkfifo"), "requires os.mkfifo")
    def test_fifo_lock_refused_without_blocking(self):
        os.mkfifo(self.lock)
        for reader in (False, True):
            with self.subTest(reader=reader):
                if reader:  # a reader lets the open succeed; the type check refuses
                    fd = os.open(self.lock, os.O_RDONLY | os.O_NONBLOCK)
                    self.addCleanup(os.close, fd)
                    expected = "^reservations lock is not a regular file; refusing to lock$"
                else:
                    expected = self.UNUSABLE
                caught = []

                def run():
                    try:
                        budget.set_reservation("svc", 1024, home=self.home)
                    except BaseException as exc:
                        caught.append(exc)

                worker = threading.Thread(target=run, daemon=True)
                worker.start()
                worker.join(timeout=10.0)
                self.assertFalse(worker.is_alive(), "FIFO lock open blocked")
                self.assertEqual(len(caught), 1)
                self.assertIsInstance(caught[0], budget.BudgetError)
                self.assertRegex(str(caught[0]), expected)
        self.assertTrue(stat.S_ISFIFO(os.lstat(self.lock).st_mode))
        self.assertFalse(self.path.exists())

    def test_directory_lock_refused(self):
        self.lock.mkdir()
        with self.assertRaisesRegex(budget.BudgetError, self.UNUSABLE):
            budget.set_reservation("svc", 1024, home=self.home)
        self.assertTrue(self.lock.is_dir())

    @unittest.skipIf(hasattr(os, "geteuid") and os.geteuid() == 0, "root ignores file modes")
    def test_unwritable_lock_refused(self):
        self.lock.touch()
        self.lock.chmod(0o400)
        with self.assertRaisesRegex(budget.BudgetError, self.UNUSABLE):
            budget.set_reservation("svc", 1024, home=self.home)
        self.assertFalse(self.path.exists())

    def test_lock_created_private_and_reused_without_truncation(self):
        budget.set_reservation("svc", 1024, home=self.home)
        st = os.lstat(self.lock)
        self.assertTrue(stat.S_ISREG(st.st_mode))
        self.assertEqual(stat.S_IMODE(st.st_mode) & 0o077, 0)
        self.lock.write_bytes(b"held")
        budget.set_reservation("svc2", 1024, home=self.home)
        self.assertEqual(self.lock.read_bytes(), b"held")
        self.assertEqual(os.lstat(self.lock).st_ino, st.st_ino)

    def test_held_lock_still_excludes_a_transaction(self):
        self.lock.touch()
        holder = os.open(self.lock, os.O_RDONLY)
        self.addCleanup(os.close, holder)
        budget.fcntl.flock(holder, budget.fcntl.LOCK_EX)
        done = threading.Event()
        worker = threading.Thread(
            target=lambda: (budget.set_reservation("svc", 1024, home=self.home),
                            done.set()),
            daemon=True)
        worker.start()
        self.assertFalse(done.wait(0.5), "transaction ran while the lock was held")
        budget.fcntl.flock(holder, budget.fcntl.LOCK_UN)
        self.assertTrue(done.wait(10.0), "transaction never took the released lock")
        worker.join(10.0)
        self.assertEqual(set(budget.load_reservations(self.home)), {"svc"})


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
        "deep_nesting": b"[" * 200000,
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
                self.assertIn("reservations file unreadable", str(exc))
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
                self.assertIn("reservations file unreadable", proc.stderr)
                self.assertIn("refusing to guess", proc.stderr)
                self.assertNotIn("Traceback", proc.stderr)


class ReservationsFileReadBoundsTests(unittest.TestCase):
    """Reservations file reads are non-blocking, reject non-regular files,
    and bound reads and writes to RESERVATIONS_MAX_BYTES."""

    def _home_with(self, raw):
        home, cleanup = _make_home()
        self.addCleanup(cleanup)
        budget.reservations_path(home).write_bytes(raw)
        return home

    @unittest.skipUnless(hasattr(os, "mkfifo"), "requires os.mkfifo")
    def test_fifo_reservations_rejected_without_blocking(self):
        home, cleanup = _make_home()
        self.addCleanup(cleanup)
        path = budget.reservations_path(home)
        os.mkfifo(path)
        outcome: dict[str, BaseException] = {}

        def run():
            try:
                expected = "reservations file is not a regular file; refusing to guess"
                with self.assertRaises(budget.BudgetError) as cm:
                    budget.load_reservations(home)
                self.assertEqual(str(cm.exception), expected)
                with self.assertRaises(budget.BudgetError) as cm:
                    budget.set_reservation("svc", 1024, home=home)
                self.assertEqual(str(cm.exception), expected)
            except BaseException as exc:
                outcome["error"] = exc

        worker = threading.Thread(target=run, daemon=True)
        worker.start()
        worker.join(timeout=10.0)
        self.assertFalse(worker.is_alive(), "FIFO reservations read blocked")
        if "error" in outcome:
            raise outcome["error"]
        self.assertTrue(stat.S_ISFIFO(os.lstat(path).st_mode))

    def test_symlink_to_dev_null_rejected_and_preserved(self):
        dev_null = Path("/dev/null")
        try:
            is_chr = stat.S_ISCHR(os.stat(dev_null).st_mode)
        except OSError:
            is_chr = False
        if not is_chr:
            self.skipTest("/dev/null is not a character device")
        home, cleanup = _make_home()
        self.addCleanup(cleanup)
        path = budget.reservations_path(home)
        path.symlink_to(dev_null)
        expected = "reservations file is not a regular file; refusing to guess"
        with self.assertRaises(budget.BudgetError) as cm:
            budget.load_reservations(home)
        self.assertEqual(str(cm.exception), expected)
        with self.assertRaises(budget.BudgetError) as cm:
            budget.set_reservation("svc", 1024, home=home)
        self.assertEqual(str(cm.exception), expected)
        self.assertTrue(path.is_symlink())
        self.assertEqual(os.readlink(path), str(dev_null))

    def test_oversized_sparse_file_rejected_without_reading(self):
        home, cleanup = _make_home()
        self.addCleanup(cleanup)
        size = 64 * 1024 * 1024
        path = budget.reservations_path(home)
        with open(path, "wb") as fh:
            fh.truncate(size)
        open_ctx, reads = self._short_read_open()
        with open_ctx:
            with self.assertRaises(budget.BudgetError) as cm:
                budget.load_reservations(home)
        self.assertEqual(
            str(cm.exception),
            f"reservations file is {size} bytes, over the "
            f"{budget.RESERVATIONS_MAX_BYTES}-byte limit; refusing to guess",
        )
        self.assertEqual(reads[0], 0)

    def _short_read_open(self, limit=4096):
        real_open = open
        reads = [0]

        def short_open(*args, **kwargs):
            fh = real_open(*args, **kwargs)
            read = fh.read

            def limited_read(n=-1):
                reads[0] += 1
                return read(limit if n < 0 else min(n, limit))

            fh.read = limited_read
            return fh

        return unittest.mock.patch.object(budget, "open", short_open, create=True), reads

    def test_file_growth_after_fstat_rejected_with_bounded_read(self):
        cap = budget.RESERVATIONS_MAX_BYTES
        home = self._home_with(b" " * (cap + 4096))
        real_fstat = os.fstat
        expected = (
            f"reservations file is {cap + 1} bytes, over the "
            f"{cap}-byte limit; refusing to guess"
        )

        for label, short_reads in (("full_read", False), ("short_reads", True)):
            with self.subTest(label):
                dups = []  # a dup shares the file offset, so it shows how far the read went

                def small_fstat(fd):
                    dup_fd = os.dup(fd)
                    self.addCleanup(os.close, dup_fd)
                    dups.append(dup_fd)
                    vals = list(real_fstat(fd))
                    vals[stat.ST_SIZE] = 16
                    return os.stat_result(vals)

                open_ctx, reads = (
                    self._short_read_open() if short_reads
                    else (contextlib.nullcontext(), None)
                )
                with unittest.mock.patch.object(
                    budget.os, "fstat", side_effect=small_fstat
                ), open_ctx:
                    with self.assertRaises(budget.BudgetError) as cm:
                        budget.load_reservations(home)
                self.assertEqual(str(cm.exception), expected)
                self.assertEqual(len(dups), 1)
                self.assertEqual(os.lseek(dups[0], 0, os.SEEK_CUR), cap + 1)
                if reads is not None:
                    self.assertGreater(reads[0], 1)

    def test_short_reads_are_joined(self):
        note = "x" * 12000
        payload = {"svc": {"bytes": 1024, "note": note, "state": "pending"}}
        raw = json.dumps(payload).encode("utf-8")
        home = self._home_with(raw)
        open_ctx, reads = self._short_read_open()
        with open_ctx:
            loaded = budget.load_reservations(home)
        self.assertGreater(reads[0], 2)
        self.assertEqual(loaded["svc"]["bytes"], 1024)
        self.assertEqual(loaded["svc"]["note"], note)

    def test_exact_size_limit_loads_and_cap_plus_one_refused(self):
        cap = 256 * 1024
        body = json.dumps({"svc": {"bytes": 2048, "note": "at-cap"}}).encode("utf-8")
        exact_home = self._home_with(body + b" " * (cap - len(body)))
        loaded = budget.load_reservations(exact_home)
        self.assertEqual(loaded["svc"]["bytes"], 2048)

        over_home = self._home_with(body + b" " * (cap + 1 - len(body)))
        with self.assertRaises(budget.BudgetError) as cm:
            budget.load_reservations(over_home)
        self.assertEqual(
            str(cm.exception),
            f"reservations file is {cap + 1} bytes, over the "
            f"{cap}-byte limit; refusing to guess",
        )

    def test_write_over_size_limit_refused_without_clobbering_file(self):
        cap = budget.RESERVATIONS_MAX_BYTES
        home, cleanup = _make_home()
        self.addCleanup(cleanup)
        path = budget.reservations_path(home)

        budget.set_reservation("svc-1", 1024, note="", home=home)
        base_len = len(path.read_bytes())
        max_note = "x" * (cap - base_len)
        budget.set_reservation("svc-1", 1024, note=max_note, home=home)
        before = path.read_bytes()
        self.assertEqual(len(before), cap)

        with unittest.mock.patch.object(
            budget.tempfile, "mkstemp", wraps=tempfile.mkstemp
        ) as mkstemp:
            with self.assertRaises(budget.BudgetError) as cm:
                budget.set_reservation("svc-1", 1024, note=max_note + "x", home=home)
            self.assertEqual(
                str(cm.exception),
                f"reservations file would be {cap + 1} bytes, over the "
                f"{cap}-byte limit; refusing to write",
            )
            with self.assertRaises(budget.BudgetError) as cm:
                budget.set_reservation("svc-2", 1024, note="extra", home=home)
            self.assertIn(f"over the {cap}-byte limit; refusing to write", str(cm.exception))
        mkstemp.assert_not_called()

        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(list(home.glob("*.tmp")), [])
        loaded = budget.load_reservations(home)
        self.assertEqual(set(loaded), {"svc-1"})
        self.assertEqual(loaded["svc-1"]["note"], max_note)
        self.assertTrue(budget.clear_reservation("svc-1", home=home))
        self.assertEqual(budget.load_reservations(home), {})

    def test_kept_path_behaviors_pin(self):
        """Pin kept behavior: valid symlink loads; missing file and dangling
        symlink return {}; directory raises BudgetError ('refusing to guess')."""
        with self.subTest("missing_file"):
            home, cleanup = _make_home()
            self.addCleanup(cleanup)
            self.assertEqual(budget.load_reservations(home), {})

        with self.subTest("dangling_symlink"):
            home, cleanup = _make_home()
            self.addCleanup(cleanup)
            budget.reservations_path(home).symlink_to(home / "does-not-exist.json")
            self.assertEqual(budget.load_reservations(home), {})

        with self.subTest("symlink_to_valid_file"):
            home, cleanup = _make_home()
            self.addCleanup(cleanup)
            target = home / "real-reservations.json"
            target.write_text(
                json.dumps({"svc": {"bytes": 4096, "note": "via-link"}}),
                encoding="utf-8",
            )
            budget.reservations_path(home).symlink_to(target)
            loaded = budget.load_reservations(home)
            self.assertEqual(loaded["svc"]["bytes"], 4096)
            self.assertEqual(loaded["svc"]["note"], "via-link")

        with self.subTest("directory"):
            home, cleanup = _make_home()
            self.addCleanup(cleanup)
            budget.reservations_path(home).mkdir()
            with self.assertRaisesRegex(
                budget.BudgetError,
                r"^reservations file unreadable \(.*\); refusing to guess$",
            ):
                budget.load_reservations(home)

    def test_nonblocking_would_block_read_rejected_without_clobbering_file(self):
        home, cleanup = _make_home()
        self.addCleanup(cleanup)
        budget.set_reservation("existing", 1024, note="keep", home=home)
        path = budget.reservations_path(home)
        before = path.read_bytes()
        real_open = open

        def would_block_open(file, *args, **kwargs):
            fh = real_open(file, *args, **kwargs)
            if Path(file) == path:
                chunks = iter((b"{}", None, b" "))
                fh.read = lambda _n=-1: next(chunks, b"")
            return fh

        expected = "reservations file read would block; refusing to guess"
        with unittest.mock.patch.object(budget, "open", would_block_open, create=True):
            with self.assertRaises(budget.BudgetError) as cm:
                budget.load_reservations(home)
            self.assertEqual(str(cm.exception), expected)
            with self.assertRaises(budget.BudgetError) as cm:
                budget.set_reservation("svc", 2048, home=home)
            self.assertEqual(str(cm.exception), expected)
        self.assertEqual(path.read_bytes(), before)

    def test_file_not_found_after_open_rejected_without_clobbering_file(self):
        home, cleanup = _make_home()
        self.addCleanup(cleanup)
        budget.set_reservation("existing", 1024, note="keep", home=home)
        path = budget.reservations_path(home)
        before = path.read_bytes()
        pattern = r"^reservations file unreadable \(.*\); refusing to guess$"
        real_fstat = os.fstat
        reservations_ino = os.stat(path).st_ino

        def vanished_fstat(fd):  # only the reservations file vanishes, not its lock
            st = real_fstat(fd)
            if st.st_ino == reservations_ino:
                raise FileNotFoundError("vanished")
            return st

        with unittest.mock.patch.object(budget.os, "fstat", vanished_fstat):
            with self.assertRaisesRegex(budget.BudgetError, pattern):
                budget.load_reservations(home)
            with self.assertRaisesRegex(budget.BudgetError, pattern):
                budget.set_reservation("svc", 2048, home=home)
        self.assertEqual(path.read_bytes(), before)


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


def _non_read_fds(test):
    """Real open fds that are not a readable pipe end: socket, regular file
    holding a valid claim, directory, pipe write end, pipe reopened O_RDWR,
    and pipe reopened O_PATH."""
    sock_a, sock_b = socket.socketpair()
    test.addCleanup(sock_a.close)
    test.addCleanup(sock_b.close)
    reg = tempfile.TemporaryFile()
    test.addCleanup(reg.close)
    reg.write(json.dumps(GOOD_CLAIM).encode())
    reg.seek(0)
    tmp_dir = tempfile.TemporaryDirectory()
    test.addCleanup(tmp_dir.cleanup)
    dir_fd = os.open(tmp_dir.name, os.O_RDONLY)
    test.addCleanup(os.close, dir_fd)
    pipe_r, pipe_w = os.pipe()
    test.addCleanup(os.close, pipe_r)
    test.addCleanup(os.close, pipe_w)
    pipe_rw = os.open(f"/proc/self/fd/{pipe_r}", os.O_RDWR)
    test.addCleanup(os.close, pipe_rw)
    pipe_path = os.open(f"/proc/self/fd/{pipe_r}", os.O_PATH)
    test.addCleanup(os.close, pipe_path)
    return sock_a.fileno(), reg.fileno(), dir_fd, pipe_w, pipe_rw, pipe_path


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

    def test_claim_from_env_rejects_malformed_values_before_reading(self):
        self.addCleanup(sys.set_int_max_str_digits, sys.get_int_max_str_digits())
        sys.set_int_max_str_digits(4300)
        read = unittest.mock.Mock(side_effect=AssertionError("os.read called"))
        claim = unittest.mock.Mock(side_effect=AssertionError("claim attempted"))
        values = ("", "\u0665", "\u00b2", " 3 ", "3 ", "+3", "-1",
                  "1_0", "3.0", "abc", "9" * 5000)
        with unittest.mock.patch.object(budget.os, "read", read), \
                unittest.mock.patch.object(budget, "claim_reservation", claim):
            for raw in values:
                with self.subTest(raw=raw[:24]):
                    with unittest.mock.patch.object(
                            budget.os, "environ", {budget.CLAIM_FD_ENV: raw}):
                        with self.assertRaises(budget.BudgetError) as cm:
                            budget.claim_from_env(home=self.home)
                    msg = str(cm.exception)
                    self.assertIn(budget.CLAIM_FD_ENV, msg)
                    self.assertLess(len(msg), 200, msg)
            with unittest.mock.patch.object(budget.os, "environ", {}):
                self.assertIsNone(budget.claim_from_env(home=self.home))
        read.assert_not_called()
        claim.assert_not_called()

    def test_read_claim_fd_refuses_non_pipe_fds_without_reading_or_closing(self):
        self.addCleanup(sys.set_int_max_str_digits, sys.get_int_max_str_digits())
        sys.set_int_max_str_digits(4300)
        real_fds = _non_read_fds(self)
        cases = [(str(fd), fd) for fd in (0, 1, 2, -1, 2**31 - 1, *real_fds)]
        cases.append(("huge", 10**5000))
        read = unittest.mock.Mock(side_effect=AssertionError("os.read called"))
        with unittest.mock.patch.object(budget.os, "read", read):
            for label, fd in cases:
                with self.subTest(fd=label):
                    with self.assertRaises(budget.BudgetError) as cm:
                        budget.read_claim_fd(fd)
                    self.assertIn("claim pipe unreadable", str(cm.exception))
        read.assert_not_called()
        for fd in real_fds:
            os.fstat(fd)

    def test_mlxlm_worker_exits_3_on_a_malformed_claim_fd(self):
        from mlx_omarchy_serve import _mlxlm_server
        reg = tempfile.TemporaryFile()
        self.addCleanup(reg.close)
        reg.write(json.dumps(GOOD_CLAIM).encode())
        reg.seek(0)
        claim = unittest.mock.Mock(side_effect=AssertionError("claim attempted"))
        read = unittest.mock.Mock(side_effect=AssertionError("os.read called"))
        for raw in ("", str(reg.fileno()), "\u0665"):
            with self.subTest(raw=raw):
                err = io.StringIO()
                with unittest.mock.patch.object(
                            budget.os, "environ", {budget.CLAIM_FD_ENV: raw}), \
                        unittest.mock.patch.object(sys, "stderr", err), \
                        unittest.mock.patch.object(budget, "claim_reservation", claim), \
                        unittest.mock.patch.object(budget.os, "read", read):
                    with self.assertRaises(SystemExit) as cm:
                        _mlxlm_server.start_managed()
                self.assertEqual(cm.exception.code, 3)
                lines = err.getvalue().splitlines()
                self.assertEqual(len(lines), 1, err.getvalue())
                self.assertTrue(lines[0].startswith("error: "), err.getvalue())
        claim.assert_not_called()
        read.assert_not_called()


NON_INT_BYTES = (True, False, 1.5, 1024.0, "1024", None, [1024])


class ReservationByteCountTests(unittest.TestCase):
    """Reservation writers and the claim take an exact positive int: a bool,
    float or string is a BudgetError, never coerced with int()."""

    def setUp(self):
        self.home, self.cleanup = _make_home()
        self.addCleanup(self.cleanup)

    def assertNothingWritten(self):
        self.assertFalse(budget.reservations_path(self.home).exists())

    def test_set_reservation_rejects_non_int_bytes(self):
        for value in NON_INT_BYTES:
            with self.subTest(value=value):
                with self.assertRaises(budget.BudgetError) as cm:
                    budget.set_reservation("svc", value, home=self.home)
                self.assertIn("positive integer bytes", str(cm.exception))
                self.assertNothingWritten()
        budget.set_reservation("svc", 1024, home=self.home)
        self.assertEqual(budget.load_reservations(self.home)["svc"]["bytes"], 1024)

    def test_admit_and_reserve_rejects_non_int_bytes(self):
        for value in NON_INT_BYTES:
            with self.subTest(value=value):
                with self.assertRaises(budget.BudgetError) as cm:
                    budget.admit_and_reserve(
                        "svc", value, owner="o", home=self.home,
                        available_bytes=64 * GiB, gpu_client_cap_bytes=64 * GiB)
                self.assertIn("positive integer bytes", str(cm.exception))
                self.assertNothingWritten()
        budget.admit_and_reserve(
            "adm", 2048, owner="o", home=self.home,
            available_bytes=64 * GiB, gpu_client_cap_bytes=64 * GiB)
        self.assertEqual(budget.load_reservations(self.home)["adm"]["bytes"], 2048)

    def test_claim_rejects_bad_expected_bytes_before_the_lock(self):
        self.addCleanup(sys.set_int_max_str_digits, sys.get_int_max_str_digits())
        sys.set_int_max_str_digits(4300)
        transaction = unittest.mock.Mock(side_effect=AssertionError("lock taken"))
        cases = [(repr(v), v) for v in NON_INT_BYTES + (0, -1)]
        cases.append(("huge negative", -(10**5000)))
        with unittest.mock.patch.object(budget, "_transaction", transaction):
            for label, value in cases:
                with self.subTest(value=label):
                    with self.assertRaises(budget.BudgetError) as cm:
                        budget.claim_reservation("chat", "tok", expected_bytes=value,
                                                 home=self.home)
                    self.assertIn("expected_bytes must be a positive integer",
                                  str(cm.exception))
        transaction.assert_not_called()


class ParentLifetimeWatchTests(unittest.TestCase):
    """The lifetime watchdog never closes stdio, never dies on a raw
    exception, and fails closed: a malformed MLX_OMARCHY_PARENT_LIFETIME_FD
    is a BudgetError before any thread starts, and an fd that cannot be
    watched still runs on_eof (the child stops)."""

    def test_from_env_accepts_only_an_open_pipe_above_stdio(self):
        # int() of 5000 digits raises only under the default limit; pin it.
        self.addCleanup(sys.set_int_max_str_digits, sys.get_int_max_str_digits())
        sys.set_int_max_str_digits(4300)
        thread = unittest.mock.Mock(side_effect=AssertionError("thread started"))
        values = ("", "\u00b2", "\u0665", "1" * 5000, "999999999999999999999",
                  str(2**31), "-1", "+5", " 5", "5 ", "1_0", "abc", "3.0",
                  "0", "1", "2", str(2**31 - 1),
                  *(str(fd) for fd in _non_read_fds(self)))
        with unittest.mock.patch.object(budget.threading, "Thread", thread):
            for raw in values:
                with self.subTest(raw=raw[:24]):
                    with unittest.mock.patch.object(
                            budget.os, "environ", {budget.LIFETIME_FD_ENV: raw}):
                        with self.assertRaises(budget.BudgetError) as cm:
                            budget.watch_parent_from_env()
                    self.assertIn(budget.LIFETIME_FD_ENV, str(cm.exception))
            with unittest.mock.patch.object(budget.os, "environ", {}):
                self.assertFalse(budget.watch_parent_from_env())
        thread.assert_not_called()

        child = (
            "import os, signal, sys, time\n"
            "from mlx_omarchy_serve import budget\n"
            "def on_term(*_):\n"
            "    os.write(1, b'fired\\n')\n"
            "    os._exit(0)\n"
            "signal.signal(signal.SIGTERM, on_term)\n"
            "r, w = os.pipe()\n"
            "os.environ[budget.LIFETIME_FD_ENV] = str(r)\n"
            "print(f'started={budget.watch_parent_from_env()}', flush=True)\n"
            "os.close(w)\n"
            "deadline = time.monotonic() + 30.0\n"
            "while time.monotonic() < deadline:\n"
            "    time.sleep(max(0.0, deadline - time.monotonic()))\n"
            "sys.exit(1)\n"
        )
        env = {k: v for k, v in os.environ.items()
               if not k.startswith("PYTHON")}
        env["PYTHONPATH"] = str(REPO_ROOT / "serve")
        proc = subprocess.run(
            [sys.executable, "-B", "-c", child],
            env=env, stdin=subprocess.DEVNULL, capture_output=True,
            text=True, timeout=60)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(proc.stdout, "started=True\nfired\n")

    def test_watch_parent_fd_runs_on_eof_once_for_every_fd(self):
        # True is fd 1: opening it with closefd=True would close stdout.
        fdopen = unittest.mock.Mock(side_effect=AssertionError("fd opened"))
        with unittest.mock.patch.object(budget.os, "fdopen", fdopen):
            for fd in (True, False, 0, 1, 2, -1, -(2**70), 2**31 - 1, 2**31,
                       2**70, "5", 5.0, None, *_non_read_fds(self)):
                with self.subTest(fd=fd):
                    on_eof = unittest.mock.Mock()
                    budget.watch_parent_fd(fd, on_eof)
                    on_eof.assert_called_once_with()
        fdopen.assert_not_called()

        r, w = os.pipe()  # the watchdog closes r
        os.close(w)
        on_eof = unittest.mock.Mock()
        budget.watch_parent_fd(r, on_eof)
        on_eof.assert_called_once_with()

    def test_mlxlm_worker_exits_3_on_a_malformed_lifetime_fd(self):
        from mlx_omarchy_serve import _mlxlm_server
        # Mocked so a regression fails here instead of watching (and
        # closing) the runner's own stdout and SIGTERMing it.
        claim = unittest.mock.Mock(side_effect=AssertionError("claim attempted"))
        thread = unittest.mock.Mock(side_effect=AssertionError("thread started"))
        kill = unittest.mock.Mock(side_effect=AssertionError("kill sent"))
        for raw in ("", "\u00b2", "1", str(2**70)):
            with self.subTest(raw=raw[:24]):
                err = io.StringIO()
                with unittest.mock.patch.object(
                            budget.os, "environ", {budget.LIFETIME_FD_ENV: raw}), \
                        unittest.mock.patch.object(sys, "stderr", err), \
                        unittest.mock.patch.object(budget, "claim_from_env", claim), \
                        unittest.mock.patch.object(budget.threading, "Thread", thread), \
                        unittest.mock.patch.object(budget.os, "kill", kill):
                    with self.assertRaises(SystemExit) as cm:
                        _mlxlm_server.start_managed()
                self.assertEqual(cm.exception.code, 3)
                lines = err.getvalue().splitlines()
                self.assertEqual(len(lines), 1, err.getvalue())
                self.assertTrue(lines[0].startswith(
                    f"error: {budget.LIFETIME_FD_ENV}"), err.getvalue())
        for mock in (claim, thread, kill):
            mock.assert_not_called()

    def test_laya_worker_exits_3_on_a_malformed_lifetime_fd(self):
        # Refused before LayaState: one stderr line, nothing loaded.
        with tempfile.TemporaryDirectory() as tmp:
            env = {k: v for k, v in os.environ.items()
                   if not k.startswith("PYTHON")}
            env.update(MLX_OMARCHY_HOME=tmp,
                       PYTHONPATH=str(REPO_ROOT / "serve"),
                       PYTHONIOENCODING="utf-8")
            env.pop(budget.CLAIM_FD_ENV, None)
            for raw in ("", "abc", "1", str(2**70)):
                with self.subTest(raw=raw[:24]):
                    env[budget.LIFETIME_FD_ENV] = raw
                    proc = subprocess.run(
                        [sys.executable, "-B", "-m", "mlx_omarchy_laya.server",
                         "--model", str(Path(tmp) / "absent")],
                        env=env, stdin=subprocess.DEVNULL, capture_output=True,
                        timeout=60)
                    self.assertEqual(proc.returncode, 3, proc.stderr)
                    self.assertEqual(proc.stderr.count(b"\n"), 1, proc.stderr)
                    self.assertTrue(proc.stderr.startswith(
                        b"laya: " + budget.LIFETIME_FD_ENV.encode()), proc.stderr)


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
