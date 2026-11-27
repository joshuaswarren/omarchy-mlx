"""The serve copies must match the installed mlx-omarchy wheel.

The venv carries serve/ as unpackaged copies (install.sh SERVE_PACKAGES),
so a wheel-only reinstall leaves an old shim running against a new wheel
without a word. The shim must refuse to start (loud, override-able) when
its build stamp disagrees with the wheel's version, and install.sh must
write that stamp.
"""

import base64
import contextlib
import hashlib
import importlib.metadata
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import unittest
import unittest.mock
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "serve"))

from mlx_omarchy_serve import _mlxlm_server as shim  # noqa: E402


_B64URL_ALPHABET = (
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"
)


class _WatchdogTimeout(BaseException):
    pass


@contextlib.contextmanager
def _watchdog(seconds: int = 5):
    """Fail with AssertionError instead of hanging if a blocking open regresses.

    Raises a BaseException subclass so `except OSError` / `except Exception`
    inside the code under test cannot swallow a timeout, and checks `fired`
    after restoring the signal handler as a second guard.
    """
    if not hasattr(signal, "SIGALRM"):
        yield
        return
    fired = False

    def _on_alarm(signum, frame):
        nonlocal fired
        fired = True
        raise _WatchdogTimeout(f"operation blocked for more than {seconds}s")

    prev = signal.signal(signal.SIGALRM, _on_alarm)
    signal.alarm(seconds)
    try:
        try:
            yield
        except _WatchdogTimeout as exc:
            raise AssertionError(str(exc)) from None
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, prev)
    if fired:
        raise AssertionError(f"operation blocked for more than {seconds}s")


def _truncate_sparse(testcase: unittest.TestCase, path: Path, size: int) -> None:
    try:
        with open(path, "wb") as fh:
            fh.truncate(size)
    except OSError as exc:
        testcase.skipTest(f"filesystem does not support sparse truncate: {exc}")


def _zero_size_stat(st: os.stat_result) -> os.stat_result:
    return os.stat_result((
        st.st_mode, st.st_ino, st.st_dev, st.st_nlink,
        st.st_uid, st.st_gid, 0, st.st_atime, st.st_mtime,
        st.st_ctime,
    ))


class WheelCommitTests(unittest.TestCase):
    def test_local_version_segment_wins(self):
        self.assertEqual(
            shim._wheel_commit("0.32.4.dev202611170306+ef7923c374"),
            "ef7923c374")

    def test_release_tag(self):
        self.assertEqual(shim._wheel_commit("0.7.38"), "0.7.38")

    def test_missing_version(self):
        self.assertIsNone(shim._wheel_commit(None))
        self.assertIsNone(shim._wheel_commit(""))


class ResolveStampMismatchTests(unittest.TestCase):
    def test_commit_stamp_matches_wheel_local_version(self):
        refusal, warning = shim.resolve_stamp_mismatch(
            "ef7923c374", "0.32.4.dev202611170306+ef7923c374")
        self.assertIsNone(refusal)
        self.assertIsNone(warning)

    def test_release_tag_matches_release_wheel(self):
        refusal, warning = shim.resolve_stamp_mismatch("v0.7.38", "0.7.38")
        self.assertIsNone(refusal)
        self.assertIsNone(warning)

    def test_missing_stamp_refuses(self):
        refusal, warning = shim.resolve_stamp_mismatch(None, "0.32.4.dev+ef7923c374")
        self.assertIsNotNone(refusal)
        self.assertIn("build stamp", refusal)
        self.assertIsNone(warning)

    def test_missing_wheel_with_stamp_refuses(self):
        refusal, warning = shim.resolve_stamp_mismatch("ef7923c374", None)
        self.assertIsNotNone(refusal)
        self.assertIn("wheel version", refusal)

    def test_no_stamp_and_no_wheel_warns_but_runs(self):
        # Dev from a checkout: nothing on either side to compare.
        refusal, warning = shim.resolve_stamp_mismatch(None, None)
        self.assertIsNone(refusal)
        self.assertIsNotNone(warning)
        self.assertIn("cannot run", warning)

    def test_override_covers_a_missing_wheel(self):
        refusal, warning = shim.resolve_stamp_mismatch(
            "ef7923c374", None, drift_allowed=True)
        self.assertIsNone(refusal)
        self.assertIsNotNone(warning)
        self.assertIn(shim.STAMP_ENV, warning)

    def test_override_covers_a_missing_stamp(self):
        refusal, warning = shim.resolve_stamp_mismatch(
            None, "0.32.4.dev+ef7923c374", drift_allowed=True)
        self.assertIsNone(refusal)
        self.assertIsNotNone(warning)
        self.assertIn(shim.STAMP_ENV, warning)

    def test_checkout_import_only_warns_on_mismatch(self):
        refusal, warning = shim.resolve_stamp_mismatch(
            "7ec5347", "0.32.4.dev202611170306+ef7923c374", installed=False)
        self.assertIsNone(refusal)
        self.assertIsNotNone(warning)
        self.assertIn("installed copies", warning)

    def test_checkout_import_only_warns_on_missing_wheel(self):
        refusal, warning = shim.resolve_stamp_mismatch(
            "ef7923c374", None, installed=False)
        self.assertIsNone(refusal)
        self.assertIsNotNone(warning)

    def test_mismatch_refuses_and_names_both_sides(self):
        refusal, warning = shim.resolve_stamp_mismatch(
            "7ec5347", "0.32.4.dev202611170306+ef7923c374")
        self.assertIsNotNone(refusal)
        self.assertIn("7ec5347", refusal)
        self.assertIn("ef7923c374", refusal)
        self.assertIsNone(warning)

    def test_drift_override_warns_instead_of_refusing(self):
        refusal, warning = shim.resolve_stamp_mismatch(
            "7ec5347", "0.32.4.dev202611170306+ef7923c374", drift_allowed=True)
        self.assertIsNone(refusal)
        self.assertIsNotNone(warning)
        self.assertIn(shim.STAMP_ENV, warning)


class ReadStampTests(unittest.TestCase):
    def test_reads_the_stamp_file_next_to_the_module(self):
        with tempfile.TemporaryDirectory() as d:
            stamp = Path(d) / shim.STAMP_FILE
            stamp.write_text("ef7923c374\n")
            self.assertEqual(
                shim.read_build_stamp(directory=Path(d)), "ef7923c374")

    def test_missing_file_reads_as_none(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertIsNone(shim.read_build_stamp(directory=Path(d)))

    def test_whitespace_only_stamp_reads_as_none(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / shim.STAMP_FILE).write_text("  \t\r\n")
            self.assertIsNone(shim.read_build_stamp(directory=Path(d)))

    def test_symlink_to_regular_stamp_is_followed(self):
        with tempfile.TemporaryDirectory() as d:
            target = Path(d) / "real-stamp"
            target.write_text("ef7923c374\n")
            (Path(d) / shim.STAMP_FILE).symlink_to(target)
            self.assertEqual(
                shim.read_build_stamp(directory=Path(d)), "ef7923c374")

    def test_dangling_symlink_reads_as_none(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / shim.STAMP_FILE).symlink_to(Path(d) / "missing")
            self.assertIsNone(shim.read_build_stamp(directory=Path(d)))

    def test_fifo_without_writer_returns_none_without_blocking(self):
        with tempfile.TemporaryDirectory() as d:
            os.mkfifo(Path(d) / shim.STAMP_FILE)
            with _watchdog(5):
                self.assertIsNone(shim.read_build_stamp(directory=Path(d)))

    def test_fifo_with_writer_returns_none_without_consuming(self):
        with tempfile.TemporaryDirectory() as d:
            fifo = Path(d) / shim.STAMP_FILE
            os.mkfifo(fifo)
            peer = os.open(fifo, os.O_RDWR | os.O_NONBLOCK)
            try:
                os.write(peer, b"ef7923c374\n")
                with _watchdog(5):
                    self.assertIsNone(shim.read_build_stamp(directory=Path(d)))
                self.assertEqual(os.read(peer, 64), b"ef7923c374\n")
            finally:
                os.close(peer)

    def test_symlink_loop_returns_none(self):
        with tempfile.TemporaryDirectory() as d:
            a = Path(d) / shim.STAMP_FILE
            b = Path(d) / "other"
            a.symlink_to(b)
            b.symlink_to(a)
            self.assertIsNone(shim.read_build_stamp(directory=Path(d)))

    def test_unix_socket_returns_none(self):
        # AF_UNIX sun_path is capped at 108 bytes; use a short /tmp dir.
        with tempfile.TemporaryDirectory(dir="/tmp") as d:
            sock_path = Path(d) / shim.STAMP_FILE
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                sock.bind(str(sock_path))
                self.assertIsNone(shim.read_build_stamp(directory=Path(d)))
            finally:
                sock.close()

    def test_symlink_to_dev_null_returns_none_without_reading(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / shim.STAMP_FILE).symlink_to("/dev/null")
            with unittest.mock.patch.object(
                    shim.os, "read", side_effect=AssertionError("must not read")):
                self.assertIsNone(shim.read_build_stamp(directory=Path(d)))

    def test_directory_named_like_stamp_returns_none(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / shim.STAMP_FILE).mkdir()
            self.assertIsNone(shim.read_build_stamp(directory=Path(d)))

    def test_invalid_utf8_stamp_returns_none(self):
        with tempfile.TemporaryDirectory() as d:
            stamp = Path(d) / shim.STAMP_FILE
            for raw in (b"\xff\xfe\n", b"ef7923c374\xc3"):
                stamp.write_bytes(raw)
                self.assertIsNone(shim.read_build_stamp(directory=Path(d)))

    def test_stamp_at_cap_succeeds_and_cap_plus_one_returns_none(self):
        with tempfile.TemporaryDirectory() as d:
            stamp = Path(d) / shim.STAMP_FILE
            cap = shim.BUILD_STAMP_MAX_BYTES
            token = "ef7923c374"
            stamp.write_bytes(token.encode() + b" " * (cap - len(token)))
            self.assertEqual(shim.read_build_stamp(directory=Path(d)), token)
            stamp.write_bytes(token.encode() + b" " * (cap + 1 - len(token)))
            self.assertIsNone(shim.read_build_stamp(directory=Path(d)))

    def test_sparse_oversized_stamp_returns_none_without_reading(self):
        with tempfile.TemporaryDirectory() as d:
            stamp = Path(d) / shim.STAMP_FILE
            _truncate_sparse(self, stamp, 4 << 30)
            with unittest.mock.patch.object(
                    shim.os, "read", side_effect=AssertionError("must not read")):
                self.assertIsNone(shim.read_build_stamp(directory=Path(d)))

    def test_stamp_reads_zero_st_size_regular_file_within_cap(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / shim.STAMP_FILE).write_text("ef7923c374\n")
            real_fstat = os.fstat
            with unittest.mock.patch.object(
                    shim.os, "fstat", side_effect=lambda fd: _zero_size_stat(real_fstat(fd))):
                self.assertEqual(
                    shim.read_build_stamp(directory=Path(d)), "ef7923c374")

    def test_stamp_when_fstat_size_lies_stops_at_cap_plus_one(self):
        with tempfile.TemporaryDirectory() as d:
            stamp = Path(d) / shim.STAMP_FILE
            cap = shim.BUILD_STAMP_MAX_BYTES
            stamp.write_bytes(b"a" * (cap + 1024))
            real_fstat = os.fstat
            real_read = os.read
            seen = []

            def counting_read(fd, n):
                chunk = real_read(fd, n)
                seen.append(len(chunk))
                return chunk

            with unittest.mock.patch.object(
                    shim.os, "fstat", side_effect=lambda fd: _zero_size_stat(real_fstat(fd))), \
                    unittest.mock.patch.object(shim.os, "read", side_effect=counting_read):
                self.assertIsNone(shim.read_build_stamp(directory=Path(d)))
            self.assertEqual(sum(seen), cap + 1)

    def test_stamp_reassembles_short_reads(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / shim.STAMP_FILE).write_text("ef7923c374\n")
            real_read = os.read
            calls = []

            def short_read(fd, n):
                chunk = real_read(fd, min(n, 3))
                calls.append(len(chunk))
                return chunk

            with unittest.mock.patch.object(shim.os, "read", side_effect=short_read):
                self.assertEqual(
                    shim.read_build_stamp(directory=Path(d)), "ef7923c374")
            self.assertGreater(len(calls), 3)

    def test_embedded_nul_directory_returns_none(self):
        self.assertIsNone(shim.read_build_stamp(directory="/tmp/bad\x00dir"))

    def test_open_flags_and_fd_cleanup_on_refusal_and_read_error(self):
        with tempfile.TemporaryDirectory() as d:
            fifo = Path(d) / shim.STAMP_FILE
            os.mkfifo(fifo)
            opened = []
            closed = []
            real_open = os.open
            real_close = os.close

            def track_open(path, flags, *args, **kwargs):
                fd = real_open(path, flags, *args, **kwargs)
                opened.append((fd, flags))
                return fd

            def track_close(fd):
                closed.append(fd)
                return real_close(fd)

            with _watchdog(5), \
                    unittest.mock.patch.object(shim.os, "open", side_effect=track_open), \
                    unittest.mock.patch.object(shim.os, "close", side_effect=track_close):
                self.assertIsNone(shim.read_build_stamp(directory=Path(d)))
            self.assertEqual(len(opened), 1)
            fd, flags = opened[0]
            self.assertTrue(flags & os.O_NONBLOCK)
            self.assertTrue(flags & os.O_CLOEXEC)
            self.assertTrue(flags & os.O_NOCTTY)
            self.assertEqual(closed, [fd])

            # Post-fstat read error must also close the opened fd cleanly.
            fifo.unlink()
            (Path(d) / shim.STAMP_FILE).write_text("ef7923c374\n")
            opened.clear()
            closed.clear()
            with unittest.mock.patch.object(shim.os, "open", side_effect=track_open), \
                    unittest.mock.patch.object(shim.os, "close", side_effect=track_close), \
                    unittest.mock.patch.object(shim.os, "read", side_effect=OSError("io fail")):
                self.assertIsNone(shim.read_build_stamp(directory=Path(d)))
            self.assertEqual(len(opened), 1)
            self.assertEqual(closed, [opened[0][0]])


class RecordVerifiedMissingStampTests(unittest.TestCase):
    def test_record_verified_missing_stamp_warns_but_runs(self):
        refusal, warning = shim.resolve_stamp_mismatch(
            None, "0.32.4.dev+ef7923c374", record_verified=True)
        self.assertIsNone(refusal)
        self.assertIsNotNone(warning)
        self.assertIn("RECORD", warning)

    def test_record_verified_does_not_rescue_a_stale_stamp(self):
        # A present-but-different stamp still refuses: the wheel may have
        # moved on from what the copies were stamped with.
        refusal, warning = shim.resolve_stamp_mismatch(
            "7ec5347", "0.32.4.dev+ef7923c374", record_verified=True)
        self.assertIsNotNone(refusal)
        self.assertIsNone(warning)

    def test_override_beats_record_verification_in_wording(self):
        refusal, warning = shim.resolve_stamp_mismatch(
            None, "0.32.4.dev+ef7923c374", drift_allowed=True,
            record_verified=True)
        self.assertIsNone(refusal)
        self.assertIsNotNone(warning)
        self.assertIn(shim.STAMP_ENV, warning)


class VerifyAgainstRecordTests(unittest.TestCase):
    """verify_serve_copies_against_record against a real distribution.

    A temp venv-style wheel install exercises importlib.metadata for
    real: a fake hash object once hid an AttributeError the reviewer
    found, because importlib.metadata exposes hash.mode and hash.value,
    not the hashlib methods a fake can carry.
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        self.root = root / "site-packages" / "mlx_omarchy_serve"
        self.root.mkdir(parents=True)
        self.content = {
            "__init__.py": "x = 1\n",
            "_mlxlm_server.py": "print('serve')\n",
        }
        for name, text in self.content.items():
            (self.root / name).write_text(text)
        distinfo = root / "site-packages" / (
            "mlx_omarchy-0.32.4.dev202611210000+ef7923c374.dist-info")
        distinfo.mkdir()
        self.distinfo = distinfo
        self._write_record({k: v.encode() for k, v in self.content.items()})

    def tearDown(self):
        self._tmp.cleanup()

    def test_all_copies_match_a_real_record(self):
        # Point metadata's site dir scan at the temp tree by patching
        # distribution discovery with a real PathDistribution.
        dist = importlib.metadata.PathDistribution(self.distinfo)
        with unittest.mock.patch.object(
                importlib.metadata, "files",
                return_value=dist.files):
            self.assertTrue(
                shim.verify_serve_copies_against_record(self.root))

    def test_one_divergent_copy_fails_verification(self):
        (self.root / "_mlxlm_server.py").write_text("print('tampered')\n")
        self._run_expect(False)

    def test_copy_absent_from_the_record_fails_verification(self):
        (self.root / "extra.py").write_text("print('extra')\n")
        self._run_expect(False)

    def test_non_sha256_record_value_fails_closed(self):
        # Rewrite RECORD with an md5 value; the verifier must refuse
        # rather than compare, and refuse without crashing.
        record = self.distinfo / "RECORD"
        text = record.read_text(encoding="utf-8")
        record.write_text(
            text.replace("sha256=", "md5="), encoding="utf-8")
        self._run_expect(False)

    def test_malformed_record_value_fails_closed(self):
        record = self.distinfo / "RECORD"
        record.write_text(
            "mlx_omarchy_serve/__init__.py,sha256=!!,3\n"
            "mlx_omarchy_serve/_mlxlm_server.py,sha256=!!,15\n",
            encoding="utf-8")
        self._run_expect(False)

    def test_non_canonical_or_junk_wrapped_record_value_fails_closed(self):
        init_b64 = base64.urlsafe_b64encode(
            hashlib.sha256(self.content["__init__.py"].encode()).digest()
        ).rstrip(b"=").decode()
        srv_b64 = base64.urlsafe_b64encode(
            hashlib.sha256(self.content["_mlxlm_server.py"].encode()).digest()
        ).rstrip(b"=").decode()
        # Flip bit 0 of the 43rd char: guaranteed non-canonical trailing bit
        # decoding to the exact same 32-byte digest.
        non_canonical = (
            srv_b64[:-1]
            + _B64URL_ALPHABET[_B64URL_ALPHABET.index(srv_b64[-1]) ^ 1]
        )
        bad_variants = [
            f"!!{srv_b64}!!",
            f"{srv_b64}=",
            f"{srv_b64}==",
            srv_b64.replace("-", "+").replace("_", "/") if ("-" in srv_b64 or "_" in srv_b64) else f"{srv_b64[:-1]}+",
            non_canonical,
        ]
        for variant in bad_variants:
            (self.distinfo / "RECORD").write_text(
                f"mlx_omarchy_serve/__init__.py,sha256={init_b64},6\n"
                f"mlx_omarchy_serve/_mlxlm_server.py,sha256={variant},15\n",
                encoding="utf-8")
            with self.subTest(variant=variant):
                self._run_expect(False)
                self._run_expect_via_distribution(False)

    def _write_record(self, entries):
        rows = [["RECORD"]]
        for name, data in entries.items():
            path = f"mlx_omarchy_serve/{name}"
            digest = hashlib.sha256(data).digest()
            encoded = base64.urlsafe_b64encode(digest).rstrip(b"=").decode()
            rows.append([path, f"sha256={encoded}", str(len(data))])
        (self.distinfo / "RECORD").write_text(
            "".join(",".join(row) + "\n" for row in rows), encoding="utf-8")

    def test_symlink_to_regular_copy_matches_record(self):
        real_copy = Path(self._tmp.name) / "real_server.py"
        real_copy.write_text(self.content["_mlxlm_server.py"])
        target = self.root / "_mlxlm_server.py"
        target.unlink()
        target.symlink_to(real_copy)
        self._run_expect(True)

    def test_symlinked_package_root_matches_record(self):
        linked_parent = Path(self._tmp.name) / "alt-site"
        linked_parent.mkdir()
        linked_pkg = linked_parent / "mlx_omarchy_serve"
        linked_pkg.symlink_to(self.root)
        dist = importlib.metadata.PathDistribution(self.distinfo)
        with unittest.mock.patch.object(
                importlib.metadata, "distribution", return_value=dist):
            self.assertTrue(
                shim.verify_serve_copies_against_record(linked_pkg))

    def test_recorded_fifo_without_writer_returns_false(self):
        target = self.root / "_mlxlm_server.py"
        target.unlink()
        os.mkfifo(target)
        with _watchdog(5):
            self._run_expect(False)

    def test_recorded_fifo_with_writer_returns_false_without_consuming(self):
        target = self.root / "_mlxlm_server.py"
        target.unlink()
        os.mkfifo(target)
        peer = os.open(target, os.O_RDWR | os.O_NONBLOCK)
        try:
            payload = self.content["_mlxlm_server.py"].encode()
            os.write(peer, payload)
            with _watchdog(5):
                self._run_expect(False)
            self.assertEqual(os.read(peer, len(payload) + 16), payload)
        finally:
            os.close(peer)

    def test_recorded_symlink_loop_and_unix_socket_return_false(self):
        target = self.root / "_mlxlm_server.py"
        target.unlink()
        other = self.root / "loop_target"
        target.symlink_to(other)
        other.symlink_to(target)
        self._run_expect(False)
        target.unlink()
        other.unlink()

        with tempfile.TemporaryDirectory(dir="/tmp") as sd:
            sock_path = Path(sd) / "s.sock"
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                sock.bind(str(sock_path))
                target.symlink_to(sock_path)
                self._run_expect(False)
            finally:
                sock.close()

    def test_unrecorded_fifo_and_sparse_copy_return_false_before_open(self):
        real_open = os.open

        def guard_open(path, *args, **kwargs):
            if Path(path).name in ("extra_fifo.py", "extra_big.py"):
                raise AssertionError(f"must not open unrecorded copy {path}")
            return real_open(path, *args, **kwargs)

        fifo = self.root / "extra_fifo.py"
        os.mkfifo(fifo)
        with _watchdog(5), \
                unittest.mock.patch.object(shim.os, "open", side_effect=guard_open):
            self._run_expect(False)
        fifo.unlink()

        big = self.root / "extra_big.py"
        _truncate_sparse(self, big, 4 << 30)
        with unittest.mock.patch.object(shim.os, "open", side_effect=guard_open):
            self._run_expect(False)

    def test_recorded_symlink_to_dev_null_fails_even_for_empty_hash(self):
        self._write_record({
            "__init__.py": self.content["__init__.py"].encode(),
            "_mlxlm_server.py": b"",
        })
        target = self.root / "_mlxlm_server.py"
        target.unlink()
        target.symlink_to("/dev/null")
        self._run_expect(False)

    def test_recorded_directory_named_dot_py_fails(self):
        target = self.root / "_mlxlm_server.py"
        target.unlink()
        target.mkdir()
        self._run_expect(False)

    def test_unreadable_subdirectory_fails_closed(self):
        if os.geteuid() == 0:
            self.skipTest("root bypasses chmod 000")
        sub = self.root / "hidden"
        sub.mkdir()
        (sub / "evil.py").write_text("x = 2\n")
        sub.chmod(0)
        try:
            self._run_expect(False)
            self._run_expect_via_distribution(False)
        finally:
            sub.chmod(0o700)

    def test_symlinked_subdirectory_is_verified_and_cycles_fail_closed(self):
        ext = Path(self._tmp.name) / "ext_sub"
        ext.mkdir()
        (ext / "helper.py").write_text("y = 2\n")
        sublink = self.root / "sub"
        sublink.symlink_to(ext)
        # Unrecorded file behind a symlinked subpackage must fail verification.
        self._run_expect(False)
        # Once recorded with its matching hash, the symlinked subpackage passes.
        self._write_record({
            "__init__.py": self.content["__init__.py"].encode(),
            "_mlxlm_server.py": self.content["_mlxlm_server.py"].encode(),
            "sub/helper.py": b"y = 2\n",
        })
        self._run_expect(True)
        self._run_expect_via_distribution(True)

        # Two sibling directory symlinks pointing to the same directory (no
        # ancestor cycle): fails until both aliased paths are recorded, then passes.
        sublink2 = self.root / "sub2"
        sublink2.symlink_to(ext)
        self._run_expect(False)
        self._write_record({
            "__init__.py": self.content["__init__.py"].encode(),
            "_mlxlm_server.py": self.content["_mlxlm_server.py"].encode(),
            "sub/helper.py": b"y = 2\n",
            "sub2/helper.py": b"y = 2\n",
        })
        self._run_expect(True)
        self._run_expect_via_distribution(True)
        sublink2.unlink()

        # A directory symlink cycle must fail closed immediately on detection
        # (not after spinning until ENAMETOOLONG).
        (ext / "back").symlink_to(self.root)
        real_scandir = os.scandir
        scandir_calls = []

        def track_scandir(path):
            scandir_calls.append(path)
            return real_scandir(path)

        with unittest.mock.patch.object(shim.os, "scandir", side_effect=track_scandir):
            self._run_expect(False)
        self.assertLessEqual(len(scandir_calls), 3)

    def test_recorded_copy_at_cap_succeeds_and_cap_plus_one_fails(self):
        cap = shim.SERVE_COPY_MAX_BYTES
        at_cap = b"#" * cap
        self._write_record({
            "__init__.py": self.content["__init__.py"].encode(),
            "_mlxlm_server.py": at_cap,
        })
        (self.root / "_mlxlm_server.py").write_bytes(at_cap)
        self._run_expect(True)

        over_cap = b"#" * (cap + 1)
        self._write_record({
            "__init__.py": self.content["__init__.py"].encode(),
            "_mlxlm_server.py": over_cap,
        })
        (self.root / "_mlxlm_server.py").write_bytes(over_cap)
        self._run_expect(False)

    def test_recorded_sparse_oversized_copy_fails_without_reading(self):
        target = self.root / "_mlxlm_server.py"
        _truncate_sparse(self, target, 4 << 30)
        real_read = os.read
        opened_target = []

        def guard_read(fd, n):
            if os.fstat(fd).st_size == (4 << 30):
                opened_target.append(n)
            return real_read(fd, n)

        with unittest.mock.patch.object(shim.os, "read", side_effect=guard_read):
            self._run_expect(False)
        self.assertEqual(opened_target, [])

    def test_recorded_copy_zero_st_size_within_cap_succeeds(self):
        target_ino = (self.root / "_mlxlm_server.py").stat().st_ino
        real_fstat = os.fstat

        def fake_fstat(fd):
            st = real_fstat(fd)
            return _zero_size_stat(st) if st.st_ino == target_ino else st

        with unittest.mock.patch.object(shim.os, "fstat", side_effect=fake_fstat):
            self._run_expect(True)

    def test_recorded_copy_when_fstat_size_lies_stops_at_cap_plus_one(self):
        cap = shim.SERVE_COPY_MAX_BYTES
        target = self.root / "_mlxlm_server.py"
        data = b"#" * (cap + 4096)
        self._write_record({
            "__init__.py": self.content["__init__.py"].encode(),
            "_mlxlm_server.py": data,
        })
        target.write_bytes(data)
        target_ino = target.stat().st_ino
        real_fstat = os.fstat
        real_read = os.read
        seen = []

        def fake_fstat(fd):
            st = real_fstat(fd)
            return _zero_size_stat(st) if st.st_ino == target_ino else st

        def counting_read(fd, n):
            chunk = real_read(fd, n)
            if real_fstat(fd).st_ino == target_ino:
                seen.append(len(chunk))
            return chunk

        with unittest.mock.patch.object(shim.os, "fstat", side_effect=fake_fstat), \
                unittest.mock.patch.object(shim.os, "read", side_effect=counting_read):
            self._run_expect(False)
        self.assertEqual(sum(seen), cap + 1)

    def test_recorded_copy_reassembles_short_reads(self):
        real_read = os.read
        calls = []

        def short_read(fd, n):
            chunk = real_read(fd, min(n, 4))
            calls.append(len(chunk))
            return chunk

        with unittest.mock.patch.object(shim.os, "read", side_effect=short_read):
            self._run_expect(True)
        self.assertGreater(len(calls), 4)

    def test_malformed_record_fails_before_opening_copies(self):
        (self.distinfo / "RECORD").write_text(
            "mlx_omarchy_serve/__init__.py,sha256=!!,3\n"
            "mlx_omarchy_serve/_mlxlm_server.py,sha256=!!,15\n",
            encoding="utf-8")
        with unittest.mock.patch.object(
                shim.os, "open", side_effect=AssertionError("must not open")):
            self._run_expect(False)

    def test_embedded_nul_directory_returns_false(self):
        self.assertFalse(
            shim.verify_serve_copies_against_record(directory="/tmp/bad\x00dir"))

    def test_real_distribution_record_fifo_sparse_duplicates_and_quoted_csv(self):
        record = self.distinfo / "RECORD"
        record.unlink()
        os.mkfifo(record)
        with _watchdog(5):
            self._run_expect_via_distribution(False)

        record.unlink()
        _truncate_sparse(self, record, 4 << 30)
        with unittest.mock.patch.object(shim.os, "read") as mock_read:
            self._run_expect_via_distribution(False)
        mock_read.assert_not_called()

        # Duplicate conflicting entries (bad/empty/md5/malformed first, valid
        # sha256 second) must fail closed on BOTH distribution and patched-files paths.
        record.unlink()
        init_b64 = base64.urlsafe_b64encode(
            hashlib.sha256(self.content["__init__.py"].encode()).digest()
        ).rstrip(b"=").decode()
        srv_b64 = base64.urlsafe_b64encode(
            hashlib.sha256(self.content["_mlxlm_server.py"].encode()).digest()
        ).rstrip(b"=").decode()
        for first_spec in (f"sha256={init_b64}", "", "md5=deadbeef", "sha256"):
            record.write_text(
                f"mlx_omarchy_serve/__init__.py,sha256={init_b64},6\n"
                f"mlx_omarchy_serve/_mlxlm_server.py,{first_spec},15\n"
                f"mlx_omarchy_serve/_mlxlm_server.py,sha256={srv_b64},15\n",
                encoding="utf-8")
            with self.subTest(first_spec=first_spec):
                self._run_expect_via_distribution(False)
                self._run_expect(False)

        # Quoted CSV filename containing an embedded newline stays intact.
        nl_name = "line1\nline2.py"
        nl_data = b"z = 9\n"
        (self.root / nl_name).write_bytes(nl_data)
        nl_b64 = base64.urlsafe_b64encode(
            hashlib.sha256(nl_data).digest()
        ).rstrip(b"=").decode()
        record.write_text(
            f"mlx_omarchy_serve/__init__.py,sha256={init_b64},6\n"
            f"mlx_omarchy_serve/_mlxlm_server.py,sha256={srv_b64},15\n"
            f'"mlx_omarchy_serve/{nl_name}",sha256={nl_b64},{len(nl_data)}\n',
            encoding="utf-8")
        self._run_expect_via_distribution(True)

    def _run_expect(self, expected):
        dist = importlib.metadata.PathDistribution(self.distinfo)
        with unittest.mock.patch.object(
                importlib.metadata, "files",
                return_value=dist.files):
            self.assertEqual(
                shim.verify_serve_copies_against_record(self.root),
                expected)

    def _run_expect_via_distribution(self, expected):
        dist = importlib.metadata.PathDistribution(self.distinfo)
        with unittest.mock.patch.object(
                importlib.metadata, "distribution",
                return_value=dist):
            self.assertEqual(
                shim.verify_serve_copies_against_record(self.root),
                expected)


class BoundedMetadataAndInstalledRootTests(unittest.TestCase):
    def test_read_wheel_version_bounds_metadata_fifo_sparse_and_valid(self):
        with tempfile.TemporaryDirectory() as d:
            distinfo = Path(d) / "mlx_omarchy-0.32.4.dist-info"
            distinfo.mkdir()
            dist = importlib.metadata.PathDistribution(distinfo)

            meta = distinfo / "METADATA"
            meta.write_text(
                "Metadata-Version: 2.1\n"
                "Name: mlx-omarchy\n"
                "Version: 0.32.4.dev202611210000+ef7923c374\n",
                encoding="utf-8")
            with unittest.mock.patch.object(
                    importlib.metadata, "distribution", return_value=dist):
                self.assertEqual(
                    shim._read_wheel_version(),
                    ("0.32.4.dev202611210000+ef7923c374", True))

            # FIFO at METADATA must not block and must not fall back to PKG-INFO,
            # while still reporting wheel_present=True so startup fails closed.
            meta.unlink()
            os.mkfifo(meta)
            (distinfo / "PKG-INFO").write_text(
                "Metadata-Version: 2.1\nName: mlx-omarchy\nVersion: 9.9.9\n",
                encoding="utf-8")
            with _watchdog(5), unittest.mock.patch.object(
                    importlib.metadata, "distribution", return_value=dist):
                self.assertEqual(shim._read_wheel_version(), (None, True))

            # 4 GiB sparse METADATA must fail closed without calling os.read.
            meta.unlink()
            _truncate_sparse(self, meta, 4 << 30)
            with unittest.mock.patch.object(
                    importlib.metadata, "distribution", return_value=dist), \
                    unittest.mock.patch.object(shim.os, "read") as mock_read:
                self.assertEqual(shim._read_wheel_version(), (None, True))
            mock_read.assert_not_called()

            # Invalid UTF-8 METADATA reports (None, True).
            meta.unlink()
            meta.write_bytes(b"\xff\xfe\n")
            with unittest.mock.patch.object(
                    importlib.metadata, "distribution", return_value=dist):
                self.assertEqual(shim._read_wheel_version(), (None, True))

            # Genuine PKG-INFO fallback when METADATA is absent.
            meta.unlink()
            with unittest.mock.patch.object(
                    importlib.metadata, "distribution", return_value=dist):
                self.assertEqual(shim._read_wheel_version(), ("9.9.9", True))

            # Single-file .egg-info distribution.
            egg = Path(d) / "mlx_omarchy-0.7.38.egg-info"
            egg.write_text(
                "Metadata-Version: 2.1\nName: mlx-omarchy\nVersion: 0.7.38\n",
                encoding="utf-8")
            egg_dist = importlib.metadata.PathDistribution(egg)
            with unittest.mock.patch.object(
                    importlib.metadata, "distribution", return_value=egg_dist):
                self.assertEqual(shim._read_wheel_version(), ("0.7.38", True))

            # Absent distribution reports (None, False).
            with unittest.mock.patch.object(
                    importlib.metadata, "distribution",
                    side_effect=importlib.metadata.PackageNotFoundError("mlx-omarchy")):
                self.assertEqual(shim._read_wheel_version(), (None, False))

            # Patched importlib.metadata.version() is still honored.
            with unittest.mock.patch.object(
                    importlib.metadata, "version", return_value="0.7.39"):
                self.assertEqual(shim._read_wheel_version(), ("0.7.39", True))

    def test_installed_server_refuses_exit_4_when_metadata_and_stamp_unreadable(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            sp = root / "site-packages"
            pkg = sp / "mlx_omarchy_serve"
            pkg.mkdir(parents=True)
            shutil.copy2(
                REPO_ROOT / "serve" / "mlx_omarchy_serve" / "_mlxlm_server.py",
                pkg / "_mlxlm_server.py")
            (pkg / "__init__.py").write_text("")
            # No .build-commit stamp, and both METADATA and RECORD are FIFOs.
            distinfo = sp / "mlx_omarchy-0.32.4.dist-info"
            distinfo.mkdir()
            os.mkfifo(distinfo / "METADATA")
            os.mkfifo(distinfo / "RECORD")

            script = (
                "import sys; "
                f"sys.path.insert(0, {str(sp)!r}); "
                "from mlx_omarchy_serve import _mlxlm_server as s; "
                "s.check_build_stamp()"
            )
            env = {k: v for k, v in os.environ.items()
                   if k not in ("MLX_OMARCHY_ALLOW_SERVE_DRIFT", "PYTHONPATH")}
            proc = subprocess.run(
                [sys.executable, "-S", "-c", script],
                env=env, stdin=subprocess.DEVNULL,
                capture_output=True, text=True, timeout=10,
            )
            self.assertEqual(proc.returncode, 4, proc.stderr)
            self.assertIn("wheel version", proc.stderr)

    def test_is_installed_copy_and_symlinked_site_packages_exit_4(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            store = root / "pkgstore"
            pkg = store / "mlx_omarchy_serve"
            pkg.mkdir(parents=True)
            mod_path = pkg / "_mlxlm_server.py"
            shutil.copy2(
                REPO_ROOT / "serve" / "mlx_omarchy_serve" / "_mlxlm_server.py",
                mod_path)
            (pkg / "__init__.py").write_text("")
            (pkg / shim.STAMP_FILE).write_text("deadbeef00\n")
            distinfo = store / "mlx_omarchy-0.32.4.dev202611210000+ef7923c374.dist-info"
            distinfo.mkdir()
            (distinfo / "METADATA").write_text(
                "Metadata-Version: 2.1\n"
                "Name: mlx-omarchy\n"
                "Version: 0.32.4.dev202611210000+ef7923c374\n",
                encoding="utf-8")

            # Symlink a site-packages root to `store` (which has no
            # "site-packages" segment in its resolved path).
            sp = root / "venv" / "lib" / "site-packages"
            sp.parent.mkdir(parents=True)
            sp.symlink_to(store)

            # In-process check: checkout path is False; symlinked site root is True.
            self.assertFalse(shim._is_installed_copy(mod_path))
            with unittest.mock.patch("site.getsitepackages", return_value=[str(sp)]):
                self.assertTrue(shim._is_installed_copy(mod_path))

            # Even when imported via the resolved `store` path, a site root
            # symlink pointing at `store` classifies the copy as installed.
            script = (
                "import site, sys; "
                f"sys.path.insert(0, {str(store)!r}); "
                f"site.getsitepackages = lambda: [{str(sp)!r}]; "
                "from mlx_omarchy_serve import _mlxlm_server as s; "
                "s.check_build_stamp()"
            )
            env = {k: v for k, v in os.environ.items()
                   if k not in ("MLX_OMARCHY_ALLOW_SERVE_DRIFT", "PYTHONPATH")}
            proc = subprocess.run(
                [sys.executable, "-S", "-c", script],
                env=env, stdin=subprocess.DEVNULL,
                capture_output=True, text=True, timeout=10,
            )
            self.assertEqual(proc.returncode, 4, proc.stderr)
            self.assertIn("deadbeef00", proc.stderr)
            self.assertIn("ef7923c374", proc.stderr)


if __name__ == "__main__":
    unittest.main()
