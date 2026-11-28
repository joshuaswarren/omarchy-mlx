"""Bounded regular-file manifest reads and atomic output writes for
scripts/bench_matrix.py.

Every input is finite: oversized manifests are sparse or run against a lowered
cap, and each FIFO test runs under a SIGALRM watchdog so a regression fails
instead of hanging the test run.
"""
from __future__ import annotations

import contextlib
import errno
import io
import json
import os
import resource
import signal
import stat
import subprocess
import sys
import tempfile
import tracemalloc
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import bench_matrix  # noqa: E402

DEEP_JSON = "[" * 200_000 + "]" * 200_000


def _minimal_manifest() -> dict:
    return {
        "schema": bench_matrix.SCHEMA,
        "generation": {
            "temp": 0,
            "seed": 0,
            "warmup_tokens": 4,
            "engine": "bench_decode",
            "engine_script": "bench_decode.py",
            "env": {},
        },
        "models": [],
        "prompts": {},
        "workloads": [],
    }


class Hung(BaseException):
    """Raised by the watchdog; BaseException so no except Exception swallows it."""


@contextlib.contextmanager
def watchdog(seconds: float = 5.0):
    def fire(signum, frame):
        raise Hung(f"blocked for {seconds} s")

    previous = signal.signal(signal.SIGALRM, fire)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def sparse_file(path: Path, size: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as stream:
        stream.truncate(size)
    return path


def make_fifo(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    os.mkfifo(path)
    return path


@contextlib.contextmanager
def forbid_path_io():
    """Fail on any unbounded Path.read_*/write_* call."""
    def fail(self, *args, **kwargs):
        raise AssertionError(f"unbounded Path I/O on {self}")

    with (
        mock.patch.object(Path, "read_bytes", fail),
        mock.patch.object(Path, "read_text", fail),
        mock.patch.object(Path, "write_bytes", fail),
        mock.patch.object(Path, "write_text", fail),
    ):
        yield


def run_cli(argv: list[str]) -> tuple[int, str, str]:
    out = io.StringIO()
    err = io.StringIO()
    rc = 0
    with (
        mock.patch.object(sys, "argv", ["bench_matrix.py", *argv]),
        redirect_stdout(out),
        redirect_stderr(err),
    ):
        try:
            bench_matrix.main()
        except SystemExit as exc:
            rc = exc.code if isinstance(exc.code, int) else 1
    return rc, out.getvalue(), err.getvalue()


class TempDirCase(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)


class BoundedRegularReaderTests(TempDirCase):
    def test_reads_regular_file_up_to_exact_cap_and_empty_at_zero_cap(self) -> None:
        p = self.root / "ok.txt"
        p.write_bytes(b"a" * 64)
        self.assertEqual(bench_matrix._read_regular_bytes(p, 64), b"a" * 64)
        self.assertEqual(bench_matrix._read_regular_text(p, 64), "a" * 64)

        # Multi-chunk assembly when file spans several _READ_CHUNK reads.
        with mock.patch.object(bench_matrix, "_READ_CHUNK", 17):
            self.assertEqual(bench_matrix._read_regular_bytes(p, 64), b"a" * 64)

        empty = self.root / "empty.txt"
        empty.write_bytes(b"")
        self.assertEqual(bench_matrix._read_regular_bytes(empty, 0), b"")

    def test_open_uses_nonblock_and_cloexec(self) -> None:
        p = self.root / "flags.txt"
        p.write_bytes(b"ok")
        seen_flags: list[int] = []
        real_open = os.open

        def spy_open(path, flags, *args, **kwargs):
            seen_flags.append(flags)
            return real_open(path, flags, *args, **kwargs)

        with mock.patch.object(bench_matrix.os, "open", side_effect=spy_open):
            self.assertEqual(bench_matrix._read_regular_bytes(p, 64), b"ok")
        self.assertEqual(len(seen_flags), 1)
        expected = os.O_RDONLY | bench_matrix._O_NONBLOCK | bench_matrix._O_CLOEXEC
        self.assertEqual(seen_flags[0] & expected, expected)

    def test_follows_symlink_to_regular_file(self) -> None:
        real = self.root / "real.txt"
        real.write_text("hello\n", encoding="utf-8")
        link = self.root / "link.txt"
        link.symlink_to(real)
        self.assertEqual(bench_matrix._read_regular_text(link, 64), "hello\n")

    def test_refuses_fifo_and_symlink_to_fifo_without_blocking(self) -> None:
        pipe = make_fifo(self.root / "pipe")
        with watchdog(), self.assertRaisesRegex(ValueError, "non-regular"):
            bench_matrix._read_regular_bytes(pipe, 64)

        link = self.root / "link"
        link.symlink_to(pipe)
        with watchdog(), self.assertRaisesRegex(ValueError, "non-regular"):
            bench_matrix._read_regular_bytes(link, 64)

    def test_refuses_directory_device_and_symlinks_to_them(self) -> None:
        dir_link = self.root / "dir_link"
        dir_link.symlink_to(self.root)
        for target in (self.root, dir_link):
            with self.assertRaisesRegex(ValueError, "non-regular"):
                bench_matrix._read_regular_bytes(target, 64)

        devnull = Path("/dev/null")
        if devnull.exists():
            dev_link = self.root / "dev_link"
            dev_link.symlink_to(devnull)
            for target in (devnull, dev_link):
                with self.assertRaisesRegex(ValueError, "non-regular"):
                    bench_matrix._read_regular_bytes(target, 64)

    def test_refuses_oversized_and_3gib_sparse_file_before_any_read(self) -> None:
        over = sparse_file(self.root / "over.bin", 1025)
        with (
            mock.patch.object(
                bench_matrix.os,
                "read",
                side_effect=AssertionError("os.read called past st_size cap"),
            ),
            self.assertRaisesRegex(ValueError, "exceeds 1024 bytes"),
        ):
            bench_matrix._read_regular_bytes(over, 1024)

        huge = sparse_file(self.root / "huge.json", 3 << 30)
        with (
            mock.patch.object(
                bench_matrix.os,
                "read",
                side_effect=AssertionError("os.read called on 3 GiB sparse file"),
            ),
            self.assertRaisesRegex(ValueError, "exceeds"),
        ):
            bench_matrix._read_regular_bytes(huge, bench_matrix.MAX_MANIFEST_BYTES)

    def test_understated_st_size_is_capped_by_chunked_read(self) -> None:
        # Synthetic understated st_size (0) with > cap bytes on disk.
        f = self.root / "growing.bin"
        f.write_bytes(b"x" * 100)
        real_fstat = os.fstat

        def zero_size_fstat(fd: int):
            st = real_fstat(fd)
            return os.stat_result((
                st.st_mode, st.st_ino, st.st_dev, st.st_nlink, st.st_uid,
                st.st_gid, 0, st.st_atime, st.st_mtime, st.st_ctime,
            ))

        with (
            mock.patch.object(bench_matrix, "_READ_CHUNK", 16),
            mock.patch.object(bench_matrix.os, "fstat", side_effect=zero_size_fstat),
            self.assertRaisesRegex(ValueError, "exceeds 32 bytes"),
        ):
            bench_matrix._read_regular_bytes(f, 32)

        # Also test real procfs file when present on Linux.
        proc_status = Path("/proc/self/status")
        if proc_status.exists():
            with self.assertRaisesRegex(ValueError, "exceeds 8 bytes"):
                bench_matrix._read_regular_bytes(proc_status, 8)

    def test_refuses_invalid_utf8_nul_path_missing_and_negative_cap(self) -> None:
        bad = self.root / "bad.txt"
        bad.write_bytes(b"ok\xff")
        with self.assertRaisesRegex(ValueError, "not valid UTF-8"):
            bench_matrix._read_regular_text(bad, 64)

        with self.assertRaisesRegex(OSError, "cannot read"):
            bench_matrix._read_regular_bytes(self.root / "missing.json", 64)

        with self.assertRaisesRegex(ValueError, "invalid input path"):
            bench_matrix._read_regular_bytes(str(self.root / "bad\x00.json"), 64)

        with self.assertRaisesRegex(ValueError, "invalid input path"):
            bench_matrix._read_regular_bytes("", 64)

        with self.assertRaisesRegex(ValueError, "non-negative"):
            bench_matrix._read_regular_bytes(bad, -1)

    def test_closes_fd_on_every_error_and_success_path(self) -> None:
        p = self.root / "f.bin"
        p.write_bytes(b"0123456789")
        closed: list[int] = []
        real_close = os.close

        def track_close(fd: int) -> None:
            closed.append(fd)
            real_close(fd)

        # 1. normal read
        with mock.patch.object(bench_matrix.os, "close", side_effect=track_close):
            self.assertEqual(bench_matrix._read_regular_bytes(p, 64), b"0123456789")
        self.assertEqual(len(closed), 1)

        # 2. fstat failure
        with (
            mock.patch.object(bench_matrix.os, "close", side_effect=track_close),
            mock.patch.object(
                bench_matrix.os, "fstat", side_effect=OSError(errno.EIO, "io")
            ),
            self.assertRaisesRegex(OSError, "cannot read"),
        ):
            bench_matrix._read_regular_bytes(p, 64)
        self.assertEqual(len(closed), 2)

        # 3. non-regular file (FIFO)
        pipe = make_fifo(self.root / "p.fifo")
        with (
            watchdog(),
            mock.patch.object(bench_matrix.os, "close", side_effect=track_close),
            self.assertRaisesRegex(ValueError, "non-regular"),
        ):
            bench_matrix._read_regular_bytes(pipe, 64)
        self.assertEqual(len(closed), 3)

        # 4. oversized st_size
        with (
            mock.patch.object(bench_matrix.os, "close", side_effect=track_close),
            self.assertRaisesRegex(ValueError, "exceeds 4 bytes"),
        ):
            bench_matrix._read_regular_bytes(p, 4)
        self.assertEqual(len(closed), 4)

        # 5. os.read failure
        with (
            mock.patch.object(bench_matrix.os, "close", side_effect=track_close),
            mock.patch.object(
                bench_matrix.os, "read", side_effect=OSError(errno.EIO, "read err")
            ),
            self.assertRaisesRegex(OSError, "cannot read"),
        ):
            bench_matrix._read_regular_bytes(p, 64)
        self.assertEqual(len(closed), 5)

        # 6. chunked-read overflow with understated st_size
        real_fstat = os.fstat

        def zero_size(fd: int):
            st = real_fstat(fd)
            return os.stat_result((
                st.st_mode, st.st_ino, st.st_dev, st.st_nlink, st.st_uid,
                st.st_gid, 0, st.st_atime, st.st_mtime, st.st_ctime,
            ))

        with (
            mock.patch.object(bench_matrix.os, "close", side_effect=track_close),
            mock.patch.object(bench_matrix.os, "fstat", side_effect=zero_size),
            self.assertRaisesRegex(ValueError, "exceeds 4 bytes"),
        ):
            bench_matrix._read_regular_bytes(p, 4)
        self.assertEqual(len(closed), 6)


class ManifestLoadingTests(TempDirCase):
    def test_loads_repo_default_manifest_and_symlinked_manifest(self) -> None:
        default_path = SCRIPTS_DIR / "bench_matrix.json"
        data = bench_matrix.load_manifest(default_path)
        self.assertEqual(data["schema"], bench_matrix.SCHEMA)

        link = self.root / "manifest-link.json"
        link.symlink_to(default_path)
        self.assertEqual(bench_matrix.load_manifest(link)["schema"], bench_matrix.SCHEMA)

    def test_refuses_non_object_json_values(self) -> None:
        for idx, payload in enumerate(("[1, 2, 3]", '"hello"', "42", "true", "null")):
            with self.subTest(payload=payload):
                p = self.root / f"non_dict_{idx}.json"
                p.write_text(payload, encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "must be a JSON object"):
                    bench_matrix.load_manifest(p)

    def test_enforces_exact_32_level_depth_cap_and_allows_wide_siblings(self) -> None:
        # Depth 32 nested dicts: outer dict (1) + 31 inner dicts = 32 containers.
        ok_nested: dict = {"leaf": 1}
        for _ in range(bench_matrix.MAX_JSON_DEPTH - 1):
            ok_nested = {"a": ok_nested}
        p_ok = self.root / "depth32.json"
        p_ok.write_text(json.dumps(ok_nested), encoding="utf-8")
        self.assertIsInstance(bench_matrix.load_manifest(p_ok), dict)

        # Depth 33 nested dicts: rejected.
        too_deep_dict = {"a": ok_nested}
        p_bad_dict = self.root / "depth33_dict.json"
        p_bad_dict.write_text(json.dumps(too_deep_dict), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "nests deeper than 32 levels"):
            bench_matrix.load_manifest(p_bad_dict)

        # Depth 33 via nested list inside dict: rejected.
        nested_list = json.dumps({"schema": bench_matrix.SCHEMA, "x": []}).replace(
            "[]", "[" * 32 + "]" * 32
        )
        p_bad_list = self.root / "depth33_list.json"
        p_bad_list.write_text(nested_list, encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "nests deeper than 32 levels"):
            bench_matrix.load_manifest(p_bad_list)

        # Repeated-key over-depth value followed by a shallow duplicate key
        # must still be rejected even though json.loads drops the first value.
        dup_over = (
            '{"schema": "bench-matrix/1", "x": '
            + "[" * 32 + "]" * 32
            + ', "x": 1}'
        )
        p_dup_over = self.root / "dup_over.json"
        p_dup_over.write_text(dup_over, encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "nests deeper than 32 levels"):
            bench_matrix.load_manifest(p_dup_over)

        # Wide sibling structure (5,000 items at depth 3) passes in O(1) extra space.
        wide = {"schema": bench_matrix.SCHEMA, "items": [{"i": i} for i in range(5000)]}
        p_wide = self.root / "wide.json"
        p_wide.write_text(json.dumps(wide), encoding="utf-8")
        loaded = bench_matrix.load_manifest(p_wide)
        self.assertEqual(len(loaded["items"]), 5000)

        # max_depth=0 accepts scalar text and rejects container text.
        bench_matrix._check_json_depth("123", "scalar", max_depth=0)
        with self.assertRaisesRegex(ValueError, "nests deeper than 0 levels"):
            bench_matrix._check_json_depth("{}", "empty", max_depth=0)

    def test_depth_scanner_escapes_and_constant_space(self) -> None:
        # Brackets and escaped quotes/backslashes inside strings do not count
        # toward nesting depth, while brackets after an escaped backslash do.
        quoted_brackets = json.dumps({
            "schema": bench_matrix.SCHEMA,
            "s1": "[" * 100 + "]" * 100,
            "s2": 'escaped \\" and trailing \\',
        })
        p_quoted = self.root / "quoted.json"
        p_quoted.write_text(quoted_brackets, encoding="utf-8")
        self.assertEqual(
            bench_matrix.load_manifest(p_quoted)["schema"], bench_matrix.SCHEMA
        )

        after_escaped_backslash = (
            '{"schema": "bench-matrix/1", "s": "trailing \\\\", "x": '
            + "[" * 32 + "]" * 32
            + "}"
        )
        p_after_bs = self.root / "after_escaped_backslash.json"
        p_after_bs.write_text(after_escaped_backslash, encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "nests deeper than 32 levels"):
            bench_matrix.load_manifest(p_after_bs)

        # Unterminated string ending in a lone backslash scans cleanly.
        bench_matrix._check_json_depth('{"a": "unterminated\\', "unterminated")

        # Bare backslash outside strings does not decrement nesting depth.
        with self.assertRaisesRegex(ValueError, "nests deeper than 32 levels"):
            bench_matrix._check_json_depth(
                "[" * 16 + "\\" * 20 + "[" * 17, "bare-backslash"
            )

        # Peak extra allocation stays O(1) across many escape sequences (unlike
        # a quantified regex escape loop, which allocates O(escapes) mark state).
        many_escapes = json.dumps({
            "schema": bench_matrix.SCHEMA,
            "notes": "\n".join(f'line {i} "[[[" \\' for i in range(4096)),
        })
        p_escapes = self.root / "many_escapes.json"
        p_escapes.write_text(many_escapes, encoding="utf-8")
        self.assertEqual(
            bench_matrix.load_manifest(p_escapes)["schema"], bench_matrix.SCHEMA
        )

        was_tracing = tracemalloc.is_tracing()
        if not was_tracing:
            tracemalloc.start()
        try:
            baseline, _ = tracemalloc.get_traced_memory()
            tracemalloc.reset_peak()
            bench_matrix._check_json_depth(many_escapes, "constant-space")
            _cur, peak = tracemalloc.get_traced_memory()
        finally:
            if not was_tracing:
                tracemalloc.stop()
        self.assertLess(peak - baseline, 64 << 10)

    def test_catches_recursion_error_deterministically_and_refuses_200k_deep_json(self) -> None:
        p = self.root / "small.json"
        p.write_text("{}", encoding="utf-8")
        with (
            mock.patch.object(
                bench_matrix.json,
                "loads",
                side_effect=RecursionError("simulated recursion limit"),
            ),
            self.assertRaisesRegex(ValueError, "is not valid JSON: simulated recursion limit"),
        ):
            bench_matrix.load_manifest(p)

        deep = self.root / "deep.json"
        deep.write_text(DEEP_JSON, encoding="utf-8")
        # _check_json_depth runs before json.loads, so deep input is rejected
        # without invoking the JSON decoder on either Python 3.11 or 3.14.
        with (
            mock.patch.object(
                bench_matrix.json,
                "loads",
                side_effect=AssertionError("json.loads called past depth cap"),
            ),
            self.assertRaisesRegex(ValueError, "nests deeper than 32 levels"),
        ):
            bench_matrix.load_manifest(deep)


class AtomicOutputWriteTests(TempDirCase):
    def test_planted_symlink_at_out_is_replaced_not_followed(self) -> None:
        victim = self.root / "victim.txt"
        victim.write_text("keep\n", encoding="utf-8")
        out = self.root / "results.json"
        out.symlink_to(victim)

        bench_matrix._write_text_atomic(out, '{"ok": true}\n')

        self.assertEqual(victim.read_text(encoding="utf-8"), "keep\n")
        self.assertFalse(out.is_symlink())
        self.assertTrue(out.is_file())
        self.assertEqual(out.read_text(encoding="utf-8"), '{"ok": true}\n')

    def test_dangling_and_directory_symlinks_are_replaced_not_followed(self) -> None:
        target = self.root / "never-create.txt"
        out = self.root / "dangling.json"
        out.symlink_to(target)
        bench_matrix._write_text_atomic(out, '{"ok": 1}\n')
        self.assertFalse(target.exists())
        self.assertFalse(out.is_symlink())
        self.assertEqual(out.read_text(encoding="utf-8"), '{"ok": 1}\n')

        subdir = self.root / "target_dir"
        subdir.mkdir()
        dir_link = self.root / "dir_symlink.json"
        dir_link.symlink_to(subdir)
        bench_matrix._write_text_atomic(dir_link, '{"ok": 2}\n')
        self.assertFalse(dir_link.is_symlink())
        self.assertEqual(dir_link.read_text(encoding="utf-8"), '{"ok": 2}\n')
        self.assertEqual(list(subdir.iterdir()), [])

    def test_relative_filename_in_cwd_writes_atomically(self) -> None:
        old_cwd = os.getcwd()
        os.chdir(self.root)
        try:
            bench_matrix._write_text_atomic("rel.json", '{"rel": true}\n')
        finally:
            os.chdir(old_cwd)
        out = self.root / "rel.json"
        self.assertEqual(out.read_text(encoding="utf-8"), '{"rel": true}\n')
        self.assertEqual([p.name for p in self.root.iterdir()], ["rel.json"])

    def test_temp_open_uses_exclusive_nofollow_0600_and_dir_fd(self) -> None:
        out = self.root / "results.json"
        seen_opens: list[tuple[object, int, int, int | None]] = []
        seen_replaces: list[dict[str, object]] = []
        real_open = os.open
        real_replace = os.replace

        def spy_open(path, flags, mode=0o777, *, dir_fd=None):
            seen_opens.append((path, flags, mode, dir_fd))
            return real_open(path, flags, mode, dir_fd=dir_fd)

        def spy_replace(src, dst, *, src_dir_fd=None, dst_dir_fd=None):
            seen_replaces.append({
                "src": src,
                "dst": dst,
                "src_dir_fd": src_dir_fd,
                "dst_dir_fd": dst_dir_fd,
            })
            return real_replace(src, dst, src_dir_fd=src_dir_fd, dst_dir_fd=dst_dir_fd)

        with (
            mock.patch.object(bench_matrix.os, "open", side_effect=spy_open),
            mock.patch.object(bench_matrix.os, "replace", side_effect=spy_replace),
        ):
            bench_matrix._write_text_atomic(out, "ok\n")

        self.assertEqual(len(seen_opens), 2)
        _dir_path, dir_flags, _dir_mode, dir_parent = seen_opens[0]
        self.assertIsNone(dir_parent)
        self.assertEqual(
            dir_flags & (os.O_RDONLY | bench_matrix._O_DIRECTORY | bench_matrix._O_CLOEXEC),
            os.O_RDONLY | bench_matrix._O_DIRECTORY | bench_matrix._O_CLOEXEC,
        )
        _tmp_path, tmp_flags, tmp_mode, tmp_dir_fd = seen_opens[1]
        expected = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | bench_matrix._O_NOFOLLOW
            | bench_matrix._O_CLOEXEC
        )
        self.assertEqual(tmp_flags & expected, expected)
        self.assertEqual(tmp_mode, 0o600)
        self.assertIsNotNone(tmp_dir_fd)
        self.assertEqual(len(seen_replaces), 1)
        self.assertEqual(seen_replaces[0]["src_dir_fd"], tmp_dir_fd)
        self.assertEqual(seen_replaces[0]["dst_dir_fd"], tmp_dir_fd)
        self.assertEqual(os.stat(out).st_mode & 0o777, 0o600)
        self.assertEqual([p.name for p in self.root.iterdir()], [out.name])

    def test_replacing_private_0600_report_does_not_broaden_permissions(self) -> None:
        out = self.root / "private.json"
        out.write_text("old\n", encoding="utf-8")
        os.chmod(out, 0o600)
        bench_matrix._write_text_atomic(out, '{"new": true}\n')
        self.assertEqual(os.stat(out).st_mode & 0o777, 0o600)

    def test_long_multibyte_output_name_writes_cleanly(self) -> None:
        out = self.root / ("\u4e2d" * 80)
        bench_matrix._write_text_atomic(out, '{"ok": true}\n')
        self.assertEqual(out.read_text(encoding="utf-8"), '{"ok": true}\n')

    def test_failed_write_leaves_existing_out_and_sibling_tmp_untouched(self) -> None:
        out = self.root / "results.json"
        out.write_text("previous\n", encoding="utf-8")
        other_tmp = self.root / ".bench_matrix.keepme.tmp"
        other_tmp.write_text("unrelated\n", encoding="utf-8")

        class FailingWriter:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def write(self, data):
                raise OSError(errno.ENOSPC, "No space left on device")

            def flush(self):
                pass

        def fake_fdopen(fd, *args, **kwargs):
            os.close(fd)
            return FailingWriter()

        with (
            mock.patch.object(bench_matrix.os, "fdopen", side_effect=fake_fdopen),
            self.assertRaisesRegex(OSError, "cannot write"),
        ):
            bench_matrix._write_text_atomic(out, "new\n")

        self.assertEqual(out.read_text(encoding="utf-8"), "previous\n")
        self.assertEqual(other_tmp.read_text(encoding="utf-8"), "unrelated\n")
        self.assertEqual(sorted(p.name for p in self.root.iterdir()), [other_tmp.name, out.name])

    def test_close_error_aborts_before_publish_and_cleans_up_temp_file(self) -> None:
        out = self.root / "results.json"
        out.write_text("original\n", encoding="utf-8")
        real_fdopen = os.fdopen

        class FailingCloseWriter:
            def __init__(self, inner):
                self._inner = inner

            def __enter__(self):
                self._inner.__enter__()
                return self

            def __exit__(self, exc_type, exc, tb):
                self._inner.__exit__(exc_type, exc, tb)
                raise OSError(errno.EIO, "delayed close I/O error")

            def write(self, data):
                return self._inner.write(data)

            def flush(self):
                return self._inner.flush()

        with (
            mock.patch.object(
                bench_matrix.os,
                "fdopen",
                side_effect=lambda fd, *a, **kw: FailingCloseWriter(real_fdopen(fd, *a, **kw)),
            ),
            mock.patch.object(
                bench_matrix.os,
                "replace",
                side_effect=AssertionError("os.replace called despite close error"),
            ),
            self.assertRaisesRegex(OSError, "cannot write"),
        ):
            bench_matrix._write_text_atomic(out, "new\n")
        self.assertEqual(out.read_text(encoding="utf-8"), "original\n")
        self.assertEqual([p.name for p in self.root.iterdir()], [out.name])

    def test_non_regular_output_destinations_refused_and_preserved(self) -> None:
        out_dir = self.root / "out_dir"
        out_dir.mkdir()
        with self.assertRaisesRegex(ValueError, "non-regular output"):
            bench_matrix._write_text_atomic(out_dir, "data\n")
        self.assertTrue(out_dir.is_dir())
        self.assertEqual([p.name for p in self.root.iterdir()], ["out_dir"])

        out_fifo = make_fifo(self.root / "out.fifo")
        with watchdog(), self.assertRaisesRegex(ValueError, "non-regular output"):
            bench_matrix._write_text_atomic(out_fifo, "data\n")
        self.assertTrue(stat.S_ISFIFO(os.lstat(out_fifo).st_mode))
        self.assertEqual(
            sorted(p.name for p in self.root.iterdir()),
            ["out.fifo", "out_dir"],
        )

        # Mocked character device / socket destination is also refused cleanly.
        for label, mode in (("device", stat.S_IFCHR | 0o600), ("socket", stat.S_IFSOCK | 0o600)):
            with self.subTest(label):
                fake_st = os.stat_result((mode, 1, 1, 1, os.geteuid(), 0, 0, 0, 0, 0))
                with (
                    mock.patch.object(bench_matrix.os, "stat", return_value=fake_st),
                    self.assertRaisesRegex(ValueError, "non-regular output"),
                ):
                    bench_matrix._write_text_atomic(self.root / f"dst_{label}", "data\n")

        # Also verify os.replace failure cleans up the staged temp file.
        with (
            mock.patch.object(
                bench_matrix.os,
                "replace",
                side_effect=IsADirectoryError(errno.EISDIR, "Is a directory"),
            ),
            self.assertRaisesRegex(OSError, "cannot write"),
        ):
            bench_matrix._write_text_atomic(self.root / "race_dir", "data\n")
        self.assertEqual(
            sorted(p.name for p in self.root.iterdir()),
            ["out.fifo", "out_dir"],
        )

    def test_colliding_temp_name_retries_and_exhaustion_fails_cleanly(self) -> None:
        out = self.root / "results.json"
        planted = self.root / ".bench_matrix.deadbeefdeadbeef.tmp"
        planted.write_text("do-not-overwrite\n", encoding="utf-8")
        tokens = iter(["deadbeefdeadbeef", "0123456789abcdef"])
        with mock.patch.object(
            bench_matrix.secrets, "token_hex", side_effect=lambda n: next(tokens)
        ):
            bench_matrix._write_text_atomic(out, "fresh\n")
        self.assertEqual(planted.read_text(encoding="utf-8"), "do-not-overwrite\n")
        self.assertEqual(out.read_text(encoding="utf-8"), "fresh\n")

        out2 = self.root / "results2.json"
        planted2 = self.root / ".bench_matrix.aaaaaaaaaaaaaaaa.tmp"
        planted2.write_text("existing\n", encoding="utf-8")
        with (
            mock.patch.object(bench_matrix.secrets, "token_hex", return_value="a" * 16),
            self.assertRaisesRegex(OSError, "cannot create unique temp output"),
        ):
            bench_matrix._write_text_atomic(out2, "data\n")
        self.assertEqual(planted2.read_text(encoding="utf-8"), "existing\n")
        self.assertFalse(out2.exists())

    def test_invalid_output_paths_and_missing_parent_refused(self) -> None:
        for bad in (
            "",
            ".",
            "..",
            "/",
            "dir/",
            "dir/.",
            "dir/..",
            "bad\x00.json",
            b"bytes.json",
        ):
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, "invalid output path"):
                bench_matrix._write_text_atomic(bad, "x\n")

        with self.assertRaisesRegex(OSError, "cannot open output directory"):
            bench_matrix._write_text_atomic(self.root / "no-such-dir" / "out.json", "x\n")

        not_a_dir = self.root / "plain.txt"
        not_a_dir.write_text("hi", encoding="utf-8")
        with self.assertRaisesRegex(OSError, "cannot open output directory"):
            bench_matrix._write_text_atomic(not_a_dir / "out.json", "x\n")

    def test_vfat_style_0644_and_group_writable_parent_succeed(self) -> None:
        out = self.root / "results.json"
        real_fstat = os.fstat

        def fake_fmask_0022(fd: int):
            st = real_fstat(fd)
            return os.stat_result((
                (st.st_mode & ~0o777) | 0o644,
                st.st_ino, st.st_dev, st.st_nlink, st.st_uid, st.st_gid,
                st.st_size, st.st_atime, st.st_mtime, st.st_ctime,
            ))

        with mock.patch.object(bench_matrix.os, "fstat", side_effect=fake_fmask_0022):
            bench_matrix._write_text_atomic(out, '{"ok": true}\n')
        self.assertEqual(out.read_text(encoding="utf-8"), '{"ok": true}\n')
        self.assertEqual([p.name for p in self.root.iterdir()], [out.name])

        sub = self.root / "umask002"
        sub.mkdir()
        os.chmod(sub, 0o775)
        out2 = sub / "results.json"
        bench_matrix._write_text_atomic(out2, '{"ok": true}\n')
        self.assertEqual(out2.read_text(encoding="utf-8"), '{"ok": true}\n')
        self.assertEqual([p.name for p in sub.iterdir()], [out2.name])

    def test_bad_staged_ownership_mode_or_type_refused_before_fdopen(self) -> None:
        real_fstat = os.fstat
        cases = (
            ("group-writable", 0, stat.S_IFREG | 0o664),
            ("world-writable", 0, stat.S_IFREG | 0o602),
            ("foreign-owner", 1, stat.S_IFREG | 0o600),
            ("non-regular", 0, stat.S_IFIFO | 0o600),
        )
        for label, uid_delta, full_mode in cases:
            with self.subTest(label):
                sub = self.root / label
                sub.mkdir()
                out = sub / "results.json"
                out.write_text("original\n", encoding="utf-8")

                def fake_bad_file(fd: int, _delta=uid_delta, _mode=full_mode):
                    st = real_fstat(fd)
                    return os.stat_result((
                        _mode, st.st_ino, st.st_dev, st.st_nlink,
                        st.st_uid + _delta, st.st_gid, st.st_size,
                        st.st_atime, st.st_mtime, st.st_ctime,
                    ))

                with (
                    mock.patch.object(bench_matrix.os, "fstat", side_effect=fake_bad_file),
                    mock.patch.object(
                        bench_matrix.os,
                        "fdopen",
                        side_effect=AssertionError("fdopen called on refused temp file"),
                    ),
                    self.assertRaisesRegex(OSError, "not a private regular file"),
                ):
                    bench_matrix._write_text_atomic(out, "secret\n")
                self.assertEqual(out.read_text(encoding="utf-8"), "original\n")
                self.assertEqual([p.name for p in sub.iterdir()], [out.name])

    def test_fdopen_and_fstat_failures_close_fds_and_unlink_temp_file(self) -> None:
        out = self.root / "results.json"
        closed: list[int] = []
        real_close = os.close

        def track_close(fd: int) -> None:
            closed.append(fd)
            real_close(fd)

        with (
            mock.patch.object(bench_matrix.os, "close", side_effect=track_close),
            mock.patch.object(bench_matrix.os, "fdopen", side_effect=RuntimeError("fdopen boom")),
            self.assertRaisesRegex(RuntimeError, "fdopen boom"),
        ):
            bench_matrix._write_text_atomic(out, "data\n")
        self.assertEqual(len(closed), 2)
        self.assertEqual(list(self.root.iterdir()), [])

        closed.clear()
        with (
            mock.patch.object(bench_matrix.os, "close", side_effect=track_close),
            mock.patch.object(
                bench_matrix.os, "fstat", side_effect=OSError(errno.EIO, "fstat boom")
            ),
            self.assertRaisesRegex(OSError, "cannot write"),
        ):
            bench_matrix._write_text_atomic(out, "data\n")
        self.assertEqual(len(closed), 2)
        self.assertEqual(list(self.root.iterdir()), [])


class CLIBoundsIntegrationTests(TempDirCase):
    def _write_valid_manifest(self, name: str = "valid.json") -> Path:
        p = self.root / name
        p.write_text(json.dumps(_minimal_manifest()), encoding="utf-8")
        return p

    def test_plan_mode_uses_no_unbounded_path_io(self) -> None:
        manifest = self._write_valid_manifest()
        out = self.root / "plan_out.json"
        with forbid_path_io():
            rc, stdout, stderr = run_cli([
                "--mode", "plan",
                "--manifest", str(manifest),
                "--out", str(out),
            ])
        self.assertEqual(rc, 0, stderr)
        self.assertEqual(stderr, "")
        parsed_stdout = json.loads(stdout)
        parsed_file = json.loads(out.read_text(encoding="utf-8"))
        self.assertEqual(parsed_stdout, parsed_file)
        self.assertEqual(parsed_file["schema"], bench_matrix.SCHEMA)

    def test_cli_refuses_unsafe_manifests_cleanly_with_exit_1(self) -> None:
        pipe = make_fifo(self.root / "m.fifo")
        with watchdog():
            rc, _out, err = run_cli(["--mode", "plan", "--manifest", str(pipe)])
        self.assertEqual(rc, 1)
        self.assertIn("ERROR: cannot read manifest:", err)
        self.assertIn("non-regular", err)
        self.assertNotIn("Traceback", err)

        sparse = sparse_file(self.root / "sparse.json", 3 << 30)
        rc, _out, err = run_cli(["--mode", "plan", "--manifest", str(sparse)])
        self.assertEqual(rc, 1)
        self.assertIn("ERROR: cannot read manifest:", err)
        self.assertIn("exceeds", err)
        self.assertNotIn("Traceback", err)

        deep = self.root / "deep.json"
        deep.write_text(DEEP_JSON, encoding="utf-8")
        rc, _out, err = run_cli(["--mode", "plan", "--manifest", str(deep)])
        self.assertEqual(rc, 1)
        self.assertIn("ERROR: cannot read manifest:", err)
        self.assertIn("nests deeper than 32 levels", err)
        self.assertNotIn("Traceback", err)

        shallow_list = self.root / "list.json"
        shallow_list.write_text("[1, 2, 3]", encoding="utf-8")
        rc, _out, err = run_cli(["--mode", "plan", "--manifest", str(shallow_list)])
        self.assertEqual(rc, 1)
        self.assertIn("ERROR: cannot read manifest:", err)
        self.assertIn("must be a JSON object", err)
        self.assertNotIn("Traceback", err)

        too_deep_dict = self.root / "too_deep_dict.json"
        too_deep_dict.write_text(
            '{"schema":"bench-matrix/1","x":' + "[" * 40 + "]" * 40 + "}",
            encoding="utf-8",
        )
        rc, _out, err = run_cli(["--mode", "plan", "--manifest", str(too_deep_dict)])
        self.assertEqual(rc, 1)
        self.assertIn("ERROR: cannot read manifest:", err)
        self.assertIn("nests deeper than 32 levels", err)
        self.assertNotIn("Traceback", err)

        wrong_schema = self.root / "wrong_schema.json"
        wrong_schema.write_text('{"schema": "other/99"}', encoding="utf-8")
        rc, _out, err = run_cli(["--mode", "plan", "--manifest", str(wrong_schema)])
        self.assertEqual(rc, 1)
        self.assertIn("ERROR: manifest schema 'other/99'", err)
        self.assertNotIn("Traceback", err)

    def test_cli_out_symlink_replaced_and_out_errors_report_cleanly(self) -> None:
        manifest = self._write_valid_manifest()
        victim = self.root / "victim.txt"
        victim.write_text("KEEP-ME\n", encoding="utf-8")
        out_link = self.root / "out-link.json"
        out_link.symlink_to(victim)

        rc, _out, err = run_cli([
            "--mode", "plan",
            "--manifest", str(manifest),
            "--out", str(out_link),
        ])
        self.assertEqual(rc, 0, err)
        self.assertEqual(victim.read_text(encoding="utf-8"), "KEEP-ME\n")
        self.assertFalse(out_link.is_symlink())
        self.assertEqual(json.loads(out_link.read_text(encoding="utf-8"))["mode"], "plan")

        out_dir = self.root / "out_dir"
        out_dir.mkdir()
        rc, _out, err = run_cli([
            "--mode", "plan",
            "--manifest", str(manifest),
            "--out", str(out_dir),
        ])
        self.assertEqual(rc, 1)
        self.assertIn("ERROR: cannot write output:", err)
        self.assertIn("non-regular output", err)
        self.assertNotIn("Traceback", err)

        out_fifo = make_fifo(self.root / "out.fifo")
        with watchdog():
            rc, _out, err = run_cli([
                "--mode", "plan",
                "--manifest", str(manifest),
                "--out", str(out_fifo),
            ])
        self.assertEqual(rc, 1)
        self.assertIn("ERROR: cannot write output:", err)
        self.assertIn("non-regular output", err)
        self.assertNotIn("Traceback", err)
        self.assertTrue(stat.S_ISFIFO(os.lstat(out_fifo).st_mode))

        rc, _out, err = run_cli([
            "--mode", "plan",
            "--manifest", str(manifest),
            "--out", "",
        ])
        self.assertEqual(rc, 1)
        self.assertIn("ERROR: cannot write output:", err)
        self.assertIn("invalid output path", err)
        self.assertNotIn("Traceback", err)

        ro_dir = self.root / "readonly"
        ro_dir.mkdir()
        os.chmod(ro_dir, 0o555)
        self.addCleanup(lambda: os.chmod(ro_dir, 0o755))
        if os.geteuid() != 0:
            rc, _out, err = run_cli([
                "--mode", "plan",
                "--manifest", str(manifest),
                "--out", str(ro_dir / "out.json"),
            ])
            self.assertEqual(rc, 1)
            self.assertIn("ERROR: cannot write output:", err)
            self.assertNotIn("Traceback", err)

    def test_cli_rlimit_fsize_preserves_existing_out_and_leaves_no_temp_files(self) -> None:
        manifest = self._write_valid_manifest()
        existing = self.root / "existing.json"
        existing.write_text('{"previous": "must-survive"}\n', encoding="utf-8")

        def limit_fsize() -> None:
            signal.signal(signal.SIGXFSZ, signal.SIG_IGN)
            resource.setrlimit(resource.RLIMIT_FSIZE, (64, 64))

        proc = subprocess.run(
            [
                sys.executable,
                "-B",
                str(SCRIPTS_DIR / "bench_matrix.py"),
                "--mode", "plan",
                "--manifest", str(manifest),
                "--out", str(existing),
            ],
            preexec_fn=limit_fsize,
            capture_output=True,
            text=True,
            timeout=10,
        )
        self.assertEqual(proc.returncode, 1)
        self.assertIn("ERROR: cannot write output:", proc.stderr)
        self.assertNotIn("Traceback", proc.stderr)
        self.assertEqual(
            existing.read_text(encoding="utf-8"),
            '{"previous": "must-survive"}\n',
        )
        leftovers = [p.name for p in self.root.iterdir() if p.name.startswith(".bench_matrix.")]
        self.assertEqual(leftovers, [])

    def test_expect_pins_refusal_still_exits_4_and_writes_out_atomically(self) -> None:
        manifest = self._write_valid_manifest()
        out = self.root / "refusal.json"
        rc, _stdout, stderr = run_cli([
            "--mode", "run",
            "--manifest", str(manifest),
            "--expect-pins", "missing-model=deadbeef",
            "--out", str(out),
        ])
        self.assertEqual(rc, 4, stderr)
        emitted = json.loads(out.read_text(encoding="utf-8"))
        self.assertIn("refusal", emitted)
        self.assertEqual(os.stat(out).st_mode & 0o777, 0o600)


if __name__ == "__main__":
    unittest.main()
