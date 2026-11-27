"""Bounded local file reads and the fallback lock write in the offline transfer.

Every input is finite: oversized files are sparse or run against a lowered
cap, and each FIFO open runs under a SIGALRM watchdog so a regression fails
instead of hanging the run."""
from __future__ import annotations

import base64
import contextlib
import hashlib
import json
import os
import shutil
import signal
import stat
import sys
import tempfile
import unittest
from importlib import metadata as im
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "serve"))

from mlx_omarchy_assistant import transfer  # noqa: E402

DEEP = b"[" * 100_000 + b"]" * 100_000  # past the parser's recursion limit
TOO_DEEP = b"{\"a\": " + b"[" * 40 + b"]" * 40 + b"}"  # parses, but past the depth cap


class Hung(BaseException):
    """Raised by the watchdog; BaseException so no except Exception swallows it."""


@contextlib.contextmanager
def watchdog(seconds=5):
    def fire(signum, frame):
        raise Hung(f"blocked for {seconds} s")
    previous = signal.signal(signal.SIGALRM, fire)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def sparse(path: Path, size: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as stream:
        stream.truncate(size)
    return path


def fifo(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    os.mkfifo(path)
    return path


def b64sha(data: bytes) -> str:
    return base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()


# A FIFO swapped in after the is_file() check: is_file() says yes, the read must refuse.
def swapped_after_check():
    return mock.patch.object(Path, "is_file", return_value=True)


def no_pairs_module():
    return mock.patch.object(transfer, "_pairs_module", return_value=None)


@contextlib.contextmanager
def forbid_path_reads():
    """Fail on any Path.read_bytes/read_text: the code under test must use the bounded read."""
    def fail(self, *args, **kwargs):
        raise AssertionError(f"unbounded Path read of {self}")
    with mock.patch.object(Path, "read_bytes", fail), \
            mock.patch.object(Path, "read_text", fail):
        yield


class TempDir(unittest.TestCase):
    def setUp(self) -> None:
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)


class ReadRegular(TempDir):
    def test_regular_file_up_to_the_cap_is_read_whole(self) -> None:
        path = self.root / "f"
        path.write_bytes(b"x" * 64)
        self.assertEqual(transfer._read_regular(path, 64), b"x" * 64)
        self.assertEqual(transfer._read_regular(self.root, 64, "f"), b"x" * 64)

    def test_fifo_refused_without_blocking(self) -> None:
        pipe = fifo(self.root / "pipe")
        with watchdog(), self.assertRaisesRegex(transfer.TransferError, "non-regular"):
            transfer._read_regular(pipe, 64)

    def test_oversized_file_refused_before_any_read(self) -> None:
        big = sparse(self.root / "big", (1 << 20) + 1)
        with mock.patch.object(transfer, "_read_chunks",
                               side_effect=AssertionError("read before size check")), \
                self.assertRaisesRegex(transfer.TransferError, "exceeds 1048576 bytes"):
            transfer._read_regular(big, 1 << 20)

    @unittest.skipUnless(Path("/proc/self/status").exists(), "needs procfs")
    def test_understated_size_is_capped_by_the_read(self) -> None:
        # procfs reports st_size 0 for a file that holds far more than 8 bytes.
        with self.assertRaisesRegex(transfer.TransferError, "exceeds 8 bytes"):
            transfer._read_regular(Path("/proc/self/status"), 8)

    def test_text_must_be_utf8(self) -> None:
        path = self.root / "t"
        path.write_bytes(b"ok \xff")
        with self.assertRaisesRegex(transfer.TransferError, "not valid UTF-8"):
            transfer._read_text(path, 64)


class LoadLockFallback(TempDir):
    """Without the pairs module the lock is read here, under its 1 MiB cap."""

    def lock_path(self) -> Path:
        path = self.root / "assistant" / "pair-locks" / "p.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def load(self):
        with no_pairs_module():
            return transfer.load_lock(self.root, "p")

    def test_missing_lock_is_none_and_valid_lock_loads(self) -> None:
        self.assertIsNone(self.load())
        self.lock_path().write_text('{"pair_id": "p"}')
        with forbid_path_reads():
            self.assertEqual(self.load(), {"pair_id": "p"})

    def test_bad_locks_are_refused(self) -> None:
        cases = (
            ("list", b"[1]", "is not a JSON object"),
            ("utf8", b'{"a": "\xff"}', "unreadable"),
            ("recursion", DEEP, "unreadable"),
            ("depth", TOO_DEEP, "nests deeper than 32 levels"),
            ("syntax", b"{", "unreadable"),
        )
        for label, body, pattern in cases:
            with self.subTest(label):
                self.lock_path().write_bytes(body)
                with self.assertRaisesRegex(transfer.TransferError, pattern) as caught:
                    self.load()
                # a local file is not a damaged archive
                self.assertNotIsInstance(caught.exception, transfer.UnsafeArchive)

    def test_oversized_lock_refused(self) -> None:
        sparse(self.lock_path(), (1 << 20) + 1)
        with self.assertRaisesRegex(transfer.TransferError, "exceeds 1048576 bytes"):
            self.load()

    def test_fifo_swapped_in_after_the_check_is_refused(self) -> None:
        fifo(self.lock_path())
        with watchdog(), swapped_after_check(), \
                self.assertRaisesRegex(transfer.TransferError, "non-regular"):
            self.load()


class VoiceReceipt(TempDir):
    def valid_receipt(self) -> dict:
        return {
            "pack_id": "vp", "license": "a",
            "files": {"w.bin": {"bytes": 1, "sha256": hashlib.sha256(b"w").hexdigest()}},
        }

    def pack(self, receipt: bytes | None = None) -> Path:
        pack = self.root / "voice" / "vp"
        pack.mkdir(parents=True)
        (pack / "w.bin").write_bytes(b"w")
        if receipt is None:
            receipt = json.dumps(self.valid_receipt()).encode()
        (pack / "manifest.json").write_bytes(receipt)
        return pack

    def add(self, pack: Path):
        return transfer._BundleBuilder().add_voice_pack(pack, {"a"}, set())

    @contextlib.contextmanager
    def count_receipt_reads(self):
        real = transfer._read_regular
        reads: list[str] = []

        def spy(path: Path, max_bytes: int, rel: str | None = None) -> bytes:
            target = str(transfer._display_path(path, rel))
            if Path(target).name == "manifest.json":
                reads.append(target)
            return real(path, max_bytes, rel)

        with mock.patch.object(transfer, "_read_regular", side_effect=spy):
            yield reads

    def test_receipt_is_read_once_and_shipped_verbatim(self) -> None:
        pack = self.pack()
        builder = transfer._BundleBuilder()
        with forbid_path_reads(), self.count_receipt_reads() as reads:
            summary = builder.add_voice_pack(pack, {"a"}, set())
        self.assertEqual(len(reads), 1)
        self.assertEqual(summary["pack_id"], "vp")
        self.assertEqual(builder.blobs["voice/vp/manifest.json"],
                         (pack / "manifest.json").read_bytes())

    def test_bad_receipts_are_refused(self) -> None:
        base = json.dumps(self.valid_receipt()).encode()
        utf8_body = b'{"extra": "\xff", ' + base[1:]
        depth_body = b'{"extra": ' + b"[" * 40 + b"]" * 40 + b", " + base[1:]
        cases = (
            ("list", b"[1]", "is not a JSON object"),
            ("utf8", utf8_body, "unreadable"),
            ("recursion", DEEP, "unreadable"),
            ("depth", depth_body, "nests deeper than 32 levels"),
        )
        for label, body, pattern in cases:
            with self.subTest(label):
                pack = self.pack(body)
                try:
                    with self.assertRaisesRegex(transfer.TransferError, pattern):
                        self.add(pack)
                finally:
                    shutil.rmtree(pack)

    def test_oversized_receipt_refused(self) -> None:
        pack = self.pack()
        sparse(pack / "manifest.json", (1 << 20) + 1)
        with self.assertRaisesRegex(transfer.TransferError, "exceeds 1048576 bytes"):
            self.add(pack)

    def test_symlinked_or_fifo_receipt_refused(self) -> None:
        pack = self.pack()
        real = self.root / "real.json"
        (pack / "manifest.json").rename(real)
        (pack / "manifest.json").symlink_to(real)
        with self.assertRaisesRegex(transfer.TransferError, "refusing symlinked file"):
            self.add(pack)
        (pack / "manifest.json").unlink()
        fifo(pack / "manifest.json")
        with watchdog(), swapped_after_check(), \
                self.assertRaisesRegex(transfer.TransferError, "non-regular"):
            self.add(pack)


class PlanReceipt(VoiceReceipt):
    def add(self, pack: Path):
        app = {"wheel": "a.whl", "version": "1", "size": 1, "sha256": "0"}
        with mock.patch.object(transfer, "prepare_wheelhouse",
                               return_value={"wheels": {}, "missing": []}), \
                mock.patch.object(transfer, "build_app_wheel", return_value=app), \
                mock.patch.object(transfer, "load_lock", return_value={"models": {}}):
            return transfer._plan_bundle(self.root, pair_ids=["p"], roots=[],
                                         site_path=None, voice=True)

    def test_receipt_is_read_once_and_shipped_verbatim(self) -> None:
        self.pack()
        with forbid_path_reads(), self.count_receipt_reads() as reads:
            plan = self.add(self.root / "voice" / "vp")
        self.assertEqual(len(reads), 1)
        self.assertEqual(plan["voice"][0]["license"], "a")


class StagedLocks(TempDir):
    def materialize_args(self, staging: Path) -> tuple:
        return (staging, self.root / "imports", {"pair_id": "p", "licenses": []},
                {"bundle_id": "b", "platform": {}})

    def staged_import(self) -> tuple[Path, Path, dict]:
        home = self.root / "home"
        staging = home / "transfer" / "staging" / "b"
        staged = {
            "home": home, "staging": staging,
            "backup_root": home / "transfer" / "backup" / "b", "tops": ["p"],
            "manifest": {"pairs": [{"pair_id": "p", "licenses": []}], "voice": [],
                         "bundle_id": "b", "wheels": [], "licenses": {},
                         "total_bytes": 0},
        }
        return home, staging, staged

    def test_staged_pair_lock_must_be_a_regular_bounded_object(self) -> None:
        staging = self.root / "staging"
        lock = fifo(staging / "p" / "lock.json")
        args = self.materialize_args(staging)
        with watchdog(), self.assertRaisesRegex(transfer.TransferError, "non-regular"):
            transfer._materialize_pair(*args)
        lock.unlink()
        for body in (b"[1]", DEEP, TOO_DEEP):
            lock.write_bytes(body)
            with self.assertRaises(transfer.TransferError):
                transfer._materialize_pair(*args)
        lock.write_bytes(json.dumps({"models": {"chat": {"id": "m"}}}).encode())
        with forbid_path_reads():
            transfer._materialize_pair(*args)
        written = json.loads((staging / "p" / "transfer-lock.json").read_text())
        self.assertEqual(written["models"]["chat"]["path"],
                         str(self.root / "imports" / "p" / "chat"))

    def test_staged_pair_lock_symlink_refused(self) -> None:
        staging = self.root / "staging"
        outside = self.root / "real-lock.json"
        outside.write_text(json.dumps({"models": {"chat": {"id": "m"}}}))
        (staging / "p").mkdir(parents=True)
        (staging / "p" / "lock.json").symlink_to(outside)
        with self.assertRaisesRegex(transfer.TransferError, "refusing symlinked file"):
            transfer._materialize_pair(*self.materialize_args(staging))
        self.assertFalse((staging / "p" / "transfer-lock.json").exists())

    def test_staged_pair_lock_oversized_refused(self) -> None:
        staging = self.root / "staging"
        sparse(staging / "p" / "lock.json", (1 << 20) + 1)
        with self.assertRaisesRegex(transfer.TransferError, "exceeds 1048576 bytes"):
            transfer._materialize_pair(*self.materialize_args(staging))
        self.assertFalse((staging / "p" / "transfer-lock.json").exists())

    def test_activated_transfer_lock_fifo_refused_and_swap_rolled_back(self) -> None:
        home, staging, staged = self.staged_import()
        fifo(staging / "p" / "transfer-lock.json")
        with watchdog(), self.assertRaisesRegex(transfer.TransferError, "non-regular"):
            transfer.commit_import(home, staged)
        self.assertTrue(stat.S_ISFIFO(
            (staging / "p" / "transfer-lock.json").lstat().st_mode))
        self.assertFalse((home / "assistant" / "imports" / "p").exists())

    def test_activated_transfer_lock_oversized_refused_and_swap_rolled_back(self) -> None:
        home, staging, staged = self.staged_import()
        sparse(staging / "p" / "transfer-lock.json", (1 << 20) + 1)
        with self.assertRaisesRegex(transfer.TransferError, "exceeds 1048576 bytes"):
            transfer.commit_import(home, staged)
        self.assertEqual((staging / "p" / "transfer-lock.json").lstat().st_size,
                         (1 << 20) + 1)
        self.assertFalse((home / "assistant" / "imports" / "p").exists())

    def test_activated_transfer_lock_symlink_refused_and_swap_rolled_back(self) -> None:
        home, staging, staged = self.staged_import()
        outside = self.root / "real-transfer-lock.json"
        outside.write_text('{"pair_id": "p"}\n')
        (staging / "p").mkdir(parents=True)
        (staging / "p" / "transfer-lock.json").symlink_to(outside)
        with self.assertRaisesRegex(transfer.TransferError, "refusing symlinked file"):
            transfer.commit_import(home, staged)
        self.assertTrue((staging / "p" / "transfer-lock.json").is_symlink())
        self.assertFalse((home / "assistant" / "imports" / "p").exists())

    def test_activated_transfer_lock_non_object_refused_and_swap_rolled_back(self) -> None:
        home, staging, staged = self.staged_import()
        (staging / "p").mkdir(parents=True)
        (staging / "p" / "transfer-lock.json").write_bytes(b"[1]")
        with self.assertRaisesRegex(transfer.TransferError, "is not a JSON object"):
            transfer.commit_import(home, staged)
        self.assertEqual((staging / "p" / "transfer-lock.json").read_bytes(), b"[1]")
        self.assertFalse((home / "assistant" / "imports" / "p").exists())


class VenvRecords(TempDir):
    def venv(self) -> Path:
        site = self.root / "venv" / "lib" / "python3.11" / "site-packages"
        info = site / "pkgc-1.0.0.dist-info"
        info.mkdir(parents=True)
        module = b"V = 1\n"
        (site / "pkgc.py").write_bytes(module)
        (info / "METADATA").write_text("Metadata-Version: 2.1\nName: pkgc\nVersion: 1.0.0\n")
        (info / "RECORD").write_text(f"pkgc.py,sha256={b64sha(module)},{len(module)}\n"
                                     "pkgc-1.0.0.dist-info/RECORD,,\n")
        return self.root / "venv"

    def verify(self) -> None:
        transfer.verify_venv_records(self.root / "venv", {"pkgc": "1.0.0"})

    def info(self) -> Path:
        return self.root / "venv/lib/python3.11/site-packages/pkgc-1.0.0.dist-info"

    def test_clean_install_verifies_without_path_reads(self) -> None:
        self.venv()
        with forbid_path_reads():
            self.verify()

    def test_fifo_metadata_refused_without_blocking(self) -> None:
        self.venv()
        (self.info() / "METADATA").unlink()
        fifo(self.info() / "METADATA")
        with watchdog(), self.assertRaisesRegex(transfer.TransferError, "non-regular"):
            self.verify()

    def test_fifo_record_or_record_target_refused_without_blocking(self) -> None:
        self.venv()
        site = self.info().parent
        for target in (self.info() / "RECORD", site / "pkgc.py"):
            with self.subTest(target.name):
                saved = target.read_bytes()
                try:
                    target.unlink()
                    fifo(target)
                    with watchdog(), swapped_after_check(), \
                            self.assertRaisesRegex(transfer.TransferError, "non-regular"):
                        self.verify()
                finally:
                    target.unlink(missing_ok=True)
                    target.write_bytes(saved)

    def test_oversized_metadata_or_record_refused(self) -> None:
        self.venv()
        for name in ("METADATA", "RECORD"):
            with self.subTest(name):
                path = self.info() / name
                saved = path.read_bytes()
                try:
                    sparse(path, 65)
                    with mock.patch.object(transfer, "_LOCAL_FILE_MAX_BYTES", 64), \
                            self.assertRaisesRegex(transfer.TransferError, "exceeds 64 bytes"):
                        self.verify()
                finally:
                    path.write_bytes(saved)

    def test_non_utf8_record_refused(self) -> None:
        self.venv()
        (self.info() / "RECORD").write_bytes(b"pkgc\xff.py,sha256=AAAA,1\n")
        with self.assertRaisesRegex(transfer.TransferError, "not valid UTF-8"):
            self.verify()

    def test_non_utf8_metadata_refused(self) -> None:
        self.venv()
        (self.info() / "METADATA").write_bytes(
            b"Metadata-Version: 2.1\nName: pkgc\nVersion: 1.0.0\nSummary: bad \xff\n"
        )
        with self.assertRaisesRegex(transfer.TransferError, "not valid UTF-8"):
            self.verify()


class AppPackaging(TempDir):
    """Every file the app wheel is built from is read under the local cap."""

    CAP = 256

    def repo(self) -> Path:
        repo = self.root / "repo"
        files = {
            "install.sh": 'FAKE_PKG="$PREFIX/mlx_omarchy_fake"\n'
                          "for fake_file in a.py; do\n  :\ndone\n",
            "serve/mlx_omarchy_fake/a.py": "A = 1\n",
            "serve/mlx_omarchy_assistant/b.py": "B = 1\n",
            "overlay/tools/coreml/c.py": "C = 1\n",
            "demo/chat.py": "D = 1\n",
            "serve/mlx_omarchy_paths.py": "P = 1\n",
        }
        for rel, body in files.items():
            (repo / rel).parent.mkdir(parents=True, exist_ok=True)
            (repo / rel).write_text(body)
        return repo

    def build(self, repo: Path):
        with mock.patch.object(transfer, "_LOCAL_FILE_MAX_BYTES", self.CAP):
            return transfer.build_app_wheel(self.root / "wh", repo=repo)

    def test_small_repo_builds_without_path_reads(self) -> None:
        repo = self.repo()
        with forbid_path_reads():
            entry = self.build(repo)
        self.assertTrue((self.root / "wh" / entry["wheel"]).is_file())

    def test_each_oversized_repo_file_refused(self) -> None:
        repo = self.repo()
        for rel in ("install.sh", "serve/mlx_omarchy_fake/a.py",
                    "serve/mlx_omarchy_assistant/b.py", "overlay/tools/coreml/c.py",
                    "demo/chat.py", "serve/mlx_omarchy_paths.py"):
            with self.subTest(rel):
                path = repo / rel
                saved = path.read_bytes()
                try:
                    sparse(path, self.CAP + 1)
                    with self.assertRaisesRegex(transfer.TransferError,
                                                f"exceeds {self.CAP} bytes"):
                        self.build(repo)
                finally:
                    path.write_bytes(saved)

    def test_fifo_install_sh_and_package_file_refused_without_blocking(self) -> None:
        repo = self.repo()
        for rel in ("install.sh", "serve/mlx_omarchy_assistant/b.py"):
            with self.subTest(rel):
                path = repo / rel
                saved = path.read_bytes()
                try:
                    path.unlink()
                    fifo(path)
                    with watchdog(), swapped_after_check(), \
                            self.assertRaisesRegex(transfer.TransferError, "non-regular"):
                        self.build(repo)
                finally:
                    path.unlink(missing_ok=True)
                    path.write_bytes(saved)

    def test_non_utf8_install_sh_refused(self) -> None:
        repo = self.repo()
        (repo / "install.sh").write_bytes(b"# \xff\n")
        with self.assertRaisesRegex(transfer.TransferError, "not valid UTF-8"):
            self.build(repo)

    def test_symlinked_package_file_swapped_after_check_is_refused(self) -> None:
        repo = self.repo()
        outside = self.root / "outside.txt"
        outside.write_bytes(b"SECRET = 1\n")
        target = repo / "serve" / "mlx_omarchy_assistant" / "b.py"
        target.unlink()
        target.symlink_to(outside)
        with mock.patch.object(Path, "is_symlink", return_value=False), \
                self.assertRaisesRegex(transfer.TransferError, "refusing symlinked file"):
            self.build(repo)
        self.assertFalse((self.root / "wh").exists())

    def installed(self, name: str):
        site = self.root / "site"
        site.mkdir(exist_ok=True)
        file = im.PackagePath(f"mlx_omarchy_assistant/{name}")
        file.hash = None
        dist = mock.Mock(files=[file], version="1.0.0")
        dist.locate_file.return_value = site
        return site / file, mock.patch.object(im, "distribution", return_value=dist)

    def test_installed_app_files_are_bounded_regular_reads(self) -> None:
        path, dist = self.installed("x.py")
        path.parent.mkdir(parents=True)
        path.write_bytes(b"X = 1\n")
        with dist, forbid_path_reads(), \
                mock.patch.object(transfer, "_LOCAL_FILE_MAX_BYTES", self.CAP):
            entry = transfer._app_wheel_from_installed(self.root / "wh")
        self.assertEqual(entry["version"], "1.0.0")
        sparse(path, self.CAP + 1)
        with dist, mock.patch.object(transfer, "_LOCAL_FILE_MAX_BYTES", self.CAP), \
                self.assertRaisesRegex(transfer.TransferError, f"exceeds {self.CAP} bytes"):
            transfer._app_wheel_from_installed(self.root / "wh")
        path.unlink()
        fifo(path)
        with dist, watchdog(), self.assertRaisesRegex(transfer.TransferError, "non-regular"):
            transfer._app_wheel_from_installed(self.root / "wh")


class ActivateLockFallback(TempDir):
    """Without the pairs module the lock is written through a unique temp file."""

    def setUp(self) -> None:
        super().setUp()
        self.locks = self.root / "assistant" / "pair-locks"
        self.locks.mkdir(parents=True)

    def activate(self, lock: dict):
        with no_pairs_module():
            return transfer._activate_lock(self.root, "p", lock, self.root / "backup")

    def test_link_at_the_old_temp_name_is_not_followed(self) -> None:
        victim = self.root / "victim.txt"
        victim.write_text("keep\n")
        (self.locks / "p.json.tmp").symlink_to(victim)
        final, backup = self.activate({"pair_id": "p"})
        self.assertEqual(victim.read_text(), "keep\n")
        self.assertIsNone(backup)
        self.assertEqual(json.loads(final.read_text()), {"pair_id": "p"})
        self.assertEqual(stat.S_IMODE(final.stat().st_mode), 0o600)
        self.assertEqual(sorted(p.name for p in self.locks.iterdir()),
                         ["p.json", "p.json.tmp"])

    def test_failed_write_removes_only_its_own_temp_and_restores_the_old_lock(self) -> None:
        (self.locks / "other.json.tmp").write_text("unrelated\n")
        for error in (OSError("disk full"), KeyboardInterrupt()):
            with self.subTest(type(error).__name__):
                (self.locks / "p.json").write_text('{"old": true}\n')
                with mock.patch.object(transfer.os, "replace", side_effect=error), \
                        self.assertRaisesRegex(transfer.TransferError,
                                               "lock activation failed"):
                    self.activate({"pair_id": "p"})
                self.assertEqual(sorted(p.name for p in self.locks.iterdir()),
                                 ["other.json.tmp", "p.json"])
                self.assertEqual((self.locks / "p.json").read_text(), '{"old": true}\n')
                self.assertEqual((self.locks / "other.json.tmp").read_text(), "unrelated\n")


if __name__ == "__main__":
    unittest.main()
