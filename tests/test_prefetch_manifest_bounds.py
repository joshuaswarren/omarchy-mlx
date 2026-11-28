"""Atomic manifest writes, CLI OSError handling, and artifact-root edge cases.

UNIT only, no network and no model artifacts. Exercises:
- write_manifest() and --manifest-out replacing planted/dangling/dir symlinks
  without following them, with mode 0600 and pinned O_PATH dir_fd staging
  (including searchable, writable 0300 output directories);
- refusal of directory-only output spellings ('file/', 'file/.', 'file/..',
  '', '.', '/') without overwriting an existing file;
- refusal to replace existing special files (FIFOs) or directories before
  os.replace(), preserving the original entry intact with zero temp files;
- mid-write failure via RLIMIT_FSIZE and delayed os.close(fd) error preserving
  any existing manifest intact with single-owner fd cleanup (closefd=False);
- temp-name collisions (regular file, symlink, and 16-retry exhaustion),
  fstat ownership/permission/regular-file guards (including vfat fmask=0022),
  invalid paths (empty, dot, dotdot, NUL, lone surrogate) on both
  _write_text_atomic() and write_manifest(), and fd hygiene;
- clean `prefetch: ...` stderr + exit 1 on CLI OSError paths;
- _artifact_roots() keeping valid symlinked roots and voice/ regular files
  working while refusing dangling/looping/non-dir --home, unreadable voice
  and voice/<pack> directories at discovery time, broken/looping symlinks,
  and non-directory top-level roots instead of tracing back or silently skipping.
"""

from __future__ import annotations

import builtins
import contextlib
import errno
import io
import json
import os
import signal
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "serve"))

from mlx_omarchy_assistant import pairs, prefetch  # noqa: E402


class _TestAlarmTimeout(BaseException):
    """Non-OSError alarm exception so caught OSErrors cannot mask a FIFO hang."""


def _stage_home(root: Path) -> Path:
    home = root / "home"
    (home / "voice" / "kokoro-82m-bf16").mkdir(parents=True)
    (home / "wake").mkdir()
    (home / "vad").mkdir()
    (home / "voice" / "kokoro-82m-bf16" / "a.bin").write_bytes(b"weights" * 32)
    (home / "wake" / "hey_jarvis.onnx").write_bytes(b"wake" * 16)
    (home / "vad" / "silero_vad.onnx").write_bytes(b"vad" * 16)
    return home


def _open_fd_count() -> int | None:
    for fd_dir in ("/proc/self/fd", "/dev/fd"):
        if os.path.isdir(fd_dir):
            return len(os.listdir(fd_dir))
    return None


def _assert_fd_count(case: unittest.TestCase, before: int | None) -> None:
    if before is not None:
        case.assertEqual(_open_fd_count(), before)


def _require_posix_perm_enforcement(case: unittest.TestCase, root: Path) -> None:
    """Skip chmod-based tests when running as root or with CAP_DAC_OVERRIDE."""
    probe = root / ".perm-probe"
    probe.mkdir(exist_ok=True)
    probe.chmod(0o000)
    try:
        try:
            fd = os.open(probe, os.O_RDONLY | os.O_DIRECTORY)
        except PermissionError:
            return
        os.close(fd)
        case.skipTest("process bypasses POSIX permission bits (root or CAP_DAC_OVERRIDE)")
    finally:
        probe.chmod(0o700)
        probe.rmdir()


@contextlib.contextmanager
def _alarm_deadline(seconds: int):
    def _raise_alarm(_signum, _frame):
        raise _TestAlarmTimeout(f"operation blocked past {seconds}s")

    prev = signal.signal(signal.SIGALRM, _raise_alarm)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, prev)


class AtomicManifestWriteTest(unittest.TestCase):
    def test_write_manifest_replaces_planted_symlink(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            home = _stage_home(root)
            victim = root / "victim.txt"
            victim.write_text("keep\n", encoding="utf-8")
            assistant = home / "assistant"
            assistant.mkdir()
            target = assistant / prefetch.MANIFEST_NAME
            target.symlink_to(victim)

            manifest = prefetch.build_manifest(home, None)
            written = prefetch.write_manifest(home, manifest)

            self.assertEqual(written, target)
            self.assertEqual(victim.read_text(encoding="utf-8"), "keep\n")
            self.assertFalse(target.is_symlink())
            self.assertEqual(stat.S_IMODE(target.lstat().st_mode), 0o600)
            self.assertEqual(json.loads(target.read_text(encoding="utf-8")), manifest)
            self.assertEqual([p.name for p in assistant.iterdir()], [prefetch.MANIFEST_NAME])

    def test_write_manifest_replaces_dangling_and_dir_symlinks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            home = _stage_home(root)
            manifest = prefetch.build_manifest(home, None)
            assistant = home / "assistant"
            assistant.mkdir()
            target = assistant / prefetch.MANIFEST_NAME

            missing_target = root / "does-not-exist"
            target.symlink_to(missing_target)
            prefetch.write_manifest(home, manifest)
            self.assertFalse(missing_target.exists())
            self.assertFalse(target.is_symlink())
            self.assertEqual(stat.S_IMODE(target.lstat().st_mode), 0o600)

            target.unlink()
            other_dir = root / "other-dir"
            other_dir.mkdir()
            (other_dir / "keep.txt").write_text("stay\n", encoding="utf-8")
            target.symlink_to(other_dir)
            prefetch.write_manifest(home, manifest)
            self.assertFalse(target.is_symlink())
            self.assertTrue(other_dir.is_dir())
            self.assertEqual((other_dir / "keep.txt").read_text(encoding="utf-8"), "stay\n")

    def test_cli_manifest_out_replaces_planted_and_dangling_symlinks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            home = _stage_home(root)
            victim = root / "victim.txt"
            victim.write_text("keep\n", encoding="utf-8")
            out = root / "out.json"
            out.symlink_to(victim)

            stdout, stderr = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                code = prefetch.main([
                    "--home", str(home), "--manifest-only",
                    "--manifest-out", str(out),
                ])

            self.assertEqual(code, 0, stderr.getvalue())
            self.assertEqual(stderr.getvalue(), "")
            self.assertEqual(victim.read_text(encoding="utf-8"), "keep\n")
            self.assertFalse(out.is_symlink())
            self.assertEqual(stat.S_IMODE(out.lstat().st_mode), 0o600)
            parsed = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(parsed, prefetch.build_manifest(home, None))

            # Dangling symlink at --manifest-out
            out2 = root / "dangling.json"
            missing = root / "missing-target"
            out2.symlink_to(missing)
            stdout2, stderr2 = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(stdout2), contextlib.redirect_stderr(stderr2):
                code2 = prefetch.main([
                    "--home", str(home), "--manifest-only",
                    "--manifest-out", str(out2),
                ])
            self.assertEqual(code2, 0, stderr2.getvalue())
            self.assertFalse(missing.exists())
            self.assertFalse(out2.is_symlink())
            self.assertEqual(stat.S_IMODE(out2.lstat().st_mode), 0o600)

    def test_cli_refuses_directory_only_spellings_without_overwriting_existing_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            home = _stage_home(root)
            report = root / "report.json"
            report.write_text('{"keep": true}\n', encoding="utf-8")

            for bad_arg in (f"{report}/", f"{report}/.", f"{report}/..", "", ".", "..", "/"):
                with self.subTest(bad_arg=bad_arg):
                    stdout, stderr = io.StringIO(), io.StringIO()
                    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                        code = prefetch.main([
                            "--home", str(home), "--manifest-only",
                            "--manifest-out", bad_arg,
                        ])
                    self.assertEqual(code, 1)
                    self.assertEqual(stdout.getvalue(), "")
                    self.assertTrue(stderr.getvalue().startswith("prefetch: "), stderr.getvalue())
                    self.assertNotIn("Traceback", stderr.getvalue())
                    self.assertEqual(report.read_text(encoding="utf-8"), '{"keep": true}\n')

            # No leftover temp files in root
            self.assertEqual(sorted(p.name for p in root.iterdir()), ["home", "report.json"])

    def test_refuses_to_replace_existing_fifo_and_preserves_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            home = _stage_home(root)
            out_dir = root / "out"
            out_dir.mkdir()
            fifo = out_dir / "manifest.fifo"
            os.mkfifo(fifo)

            fds_before = _open_fd_count()
            with self.assertRaises(PermissionError) as ctx:
                prefetch._write_text_atomic(fifo, '{"ok": true}\n')
            self.assertEqual(ctx.exception.errno, errno.EPERM)
            self.assertTrue(stat.S_ISFIFO(fifo.lstat().st_mode))
            self.assertEqual([p.name for p in out_dir.iterdir()], ["manifest.fifo"])
            _assert_fd_count(self, fds_before)

            stdout, stderr = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                code = prefetch.main([
                    "--home", str(home), "--manifest-only",
                    "--manifest-out", str(fifo),
                ])
            self.assertEqual(code, 1)
            self.assertEqual(stdout.getvalue(), "")
            self.assertTrue(stderr.getvalue().startswith("prefetch: "), stderr.getvalue())
            self.assertNotIn("Traceback", stderr.getvalue())
            self.assertTrue(stat.S_ISFIFO(fifo.lstat().st_mode))
            self.assertEqual([p.name for p in out_dir.iterdir()], ["manifest.fifo"])

    @unittest.skipUnless(
        prefetch._O_SEARCH,
        "search-only directory open flag (O_PATH/O_SEARCH) unavailable",
    )
    def test_write_text_atomic_works_in_mode_0300_output_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _require_posix_perm_enforcement(self, root)
            home = _stage_home(root)
            wx_dir = root / "wx_only"
            wx_dir.mkdir()
            wx_dir.chmod(0o300)
            target = wx_dir / "manifest.json"
            try:
                stdout, stderr = io.StringIO(), io.StringIO()
                with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                    code = prefetch.main([
                        "--home", str(home), "--manifest-only",
                        "--manifest-out", str(target),
                    ])
                self.assertEqual(code, 0, stderr.getvalue())
                self.assertEqual(stderr.getvalue(), "")
                self.assertEqual(
                    json.loads(target.read_text(encoding="utf-8")),
                    prefetch.build_manifest(home, None),
                )
                self.assertEqual(stat.S_IMODE(target.lstat().st_mode), 0o600)
            finally:
                wx_dir.chmod(0o700)
            self.assertEqual([p.name for p in wx_dir.iterdir()], ["manifest.json"])

    def test_write_text_atomic_relative_filename_in_cwd(self):
        with tempfile.TemporaryDirectory() as tmp:
            old_cwd = os.getcwd()
            os.chdir(tmp)
            try:
                prefetch._write_text_atomic("manifest.json", '{"ok": true}\n')
                target = Path(tmp) / "manifest.json"
                self.assertEqual(target.read_text(encoding="utf-8"), '{"ok": true}\n')
                self.assertEqual(stat.S_IMODE(target.lstat().st_mode), 0o600)
                self.assertEqual([p.name for p in Path(tmp).iterdir()], ["manifest.json"])
            finally:
                os.chdir(old_cwd)

    def test_write_text_atomic_temp_collision_retries_and_exhaustion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "manifest.json"
            victim = root / "victim.txt"
            victim.write_text("untouched\n", encoding="utf-8")

            # Pre-existing regular file and symlink at colliding candidate names
            colliding_file = root / ".prefetch.aaaaaaaaaaaaaaaa.tmp"
            colliding_file.write_text("preexisting\n", encoding="utf-8")
            colliding_link = root / ".prefetch.bbbbbbbbbbbbbbbb.tmp"
            colliding_link.symlink_to(victim)

            tokens = iter(["aaaaaaaaaaaaaaaa", "bbbbbbbbbbbbbbbb", "cccccccccccccccc"])
            with mock.patch.object(prefetch.secrets, "token_hex", side_effect=lambda _n: next(tokens)):
                prefetch._write_text_atomic(target, '{"ok": 1}\n')

            self.assertEqual(target.read_text(encoding="utf-8"), '{"ok": 1}\n')
            self.assertEqual(colliding_file.read_text(encoding="utf-8"), "preexisting\n")
            self.assertEqual(victim.read_text(encoding="utf-8"), "untouched\n")
            self.assertTrue(colliding_link.is_symlink())

            # 16 collisions in a row -> EEXIST without touching colliding file or leaking fds
            fds_before = _open_fd_count()
            with mock.patch.object(prefetch.secrets, "token_hex", return_value="aaaaaaaaaaaaaaaa"):
                with self.assertRaises(FileExistsError) as ctx:
                    prefetch._write_text_atomic(target, '{"ok": 2}\n')
            self.assertEqual(ctx.exception.errno, errno.EEXIST)
            self.assertEqual(target.read_text(encoding="utf-8"), '{"ok": 1}\n')
            self.assertEqual(colliding_file.read_text(encoding="utf-8"), "preexisting\n")
            _assert_fd_count(self, fds_before)

    def test_write_text_atomic_uses_dir_fd_and_no_follow_flags(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "manifest.json"
            real_open = os.open
            real_lstat = os.lstat
            real_replace = os.replace
            real_unlink = os.unlink
            open_calls = []
            lstat_calls = []
            replace_calls = []
            unlink_calls = []

            def spy_open(path, flags, *args, **kwargs):
                fd = real_open(path, flags, *args, **kwargs)
                open_calls.append((path, flags, args, kwargs, fd))
                return fd

            def spy_lstat(path, **kwargs):
                lstat_calls.append((path, kwargs))
                return real_lstat(path, **kwargs)

            def spy_replace(src, dst, **kwargs):
                replace_calls.append((src, dst, kwargs))
                return real_replace(src, dst, **kwargs)

            def spy_unlink(path, **kwargs):
                unlink_calls.append((path, kwargs))
                return real_unlink(path, **kwargs)

            with (
                mock.patch.object(prefetch.os, "open", side_effect=spy_open),
                mock.patch.object(prefetch.os, "lstat", side_effect=spy_lstat),
                mock.patch.object(prefetch.os, "replace", side_effect=spy_replace),
            ):
                prefetch._write_text_atomic(target, '{"ok": true}\n')

            self.assertEqual(len(open_calls), 2)
            parent_path, parent_flags, _pargs, pkwargs, dir_fd = open_calls[0]
            self.assertEqual(parent_path, root)
            self.assertTrue(parent_flags & os.O_DIRECTORY)
            self.assertTrue(parent_flags & os.O_CLOEXEC)
            if prefetch._O_SEARCH:
                self.assertTrue(parent_flags & prefetch._O_SEARCH)
            self.assertEqual(pkwargs, {})

            tmp_name, tmp_flags, tmp_args, tmp_kwargs, _tmp_fd = open_calls[1]
            self.assertTrue(str(tmp_name).startswith(".prefetch."))
            self.assertTrue(tmp_flags & os.O_WRONLY)
            self.assertTrue(tmp_flags & os.O_CREAT)
            self.assertTrue(tmp_flags & os.O_EXCL)
            self.assertTrue(tmp_flags & os.O_NOFOLLOW)
            self.assertTrue(tmp_flags & os.O_CLOEXEC)
            self.assertEqual(tmp_args, (0o600,))
            self.assertEqual(tmp_kwargs, {"dir_fd": dir_fd})

            self.assertEqual(lstat_calls, [("manifest.json", {"dir_fd": dir_fd})])

            self.assertEqual(len(replace_calls), 1)
            r_src, r_dst, r_kwargs = replace_calls[0]
            self.assertEqual(r_src, tmp_name)
            self.assertEqual(r_dst, "manifest.json")
            self.assertEqual(r_kwargs, {"src_dir_fd": dir_fd, "dst_dir_fd": dir_fd})

            # On failure, unlink also uses dir_fd
            with (
                mock.patch.object(prefetch.os, "replace", side_effect=OSError(errno.EIO, "boom")),
                mock.patch.object(prefetch.os, "unlink", side_effect=spy_unlink),
            ):
                with self.assertRaises(OSError):
                    prefetch._write_text_atomic(target, '{"ok": false}\n')
            self.assertEqual(len(unlink_calls), 1)
            self.assertIn("dir_fd", unlink_calls[0][1])

    def test_write_text_atomic_fstat_guards_and_vfat_mode(self):
        real_fstat = os.fstat
        euid = os.geteuid()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "manifest.json"
            target.write_text('{"old": true}\n', encoding="utf-8")

            cases = [
                ("foreign-uid", euid + 1, stat.S_IFREG | 0o600),
                ("group-writable", euid, stat.S_IFREG | 0o620),
                ("other-writable", euid, stat.S_IFREG | 0o602),
                ("non-regular", euid, stat.S_IFIFO | 0o600),
            ]
            for label, fake_uid, fake_mode in cases:
                with self.subTest(label=label):
                    fds_before = _open_fd_count()

                    def fake_fstat(fd, _uid=fake_uid, _mode=fake_mode):
                        st = real_fstat(fd)
                        return os.stat_result((
                            _mode, st.st_ino, st.st_dev, st.st_nlink,
                            _uid, st.st_gid, st.st_size,
                            st.st_atime, st.st_mtime, st.st_ctime,
                        ))

                    with mock.patch.object(prefetch.os, "fstat", side_effect=fake_fstat):
                        with self.assertRaises(PermissionError) as ctx:
                            prefetch._write_text_atomic(target, '{"new": true}\n')
                    self.assertEqual(ctx.exception.errno, errno.EPERM)
                    self.assertEqual(target.read_text(encoding="utf-8"), '{"old": true}\n')
                    self.assertEqual([p.name for p in root.iterdir()], ["manifest.json"])
                    _assert_fd_count(self, fds_before)

            # vfat-style fmask=0022 (mode 0644, owned by current user) is accepted
            def vfat_fstat(fd):
                st = real_fstat(fd)
                return os.stat_result((
                    stat.S_IFREG | 0o644, st.st_ino, st.st_dev, st.st_nlink,
                    euid, st.st_gid, st.st_size,
                    st.st_atime, st.st_mtime, st.st_ctime,
                ))

            with mock.patch.object(prefetch.os, "fstat", side_effect=vfat_fstat):
                prefetch._write_text_atomic(target, '{"vfat": true}\n')
            self.assertEqual(target.read_text(encoding="utf-8"), '{"vfat": true}\n')

    def test_write_text_atomic_stream_open_failure_single_owner_close(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "manifest.json"
            target.write_text('{"old": true}\n', encoding="utf-8")
            fds_before = _open_fd_count()
            real_open = builtins.open
            real_os_close = os.close
            seen_closefd = []
            closed_fds = []

            def spy_builtin_open(file, *args, **kwargs):
                if isinstance(file, int):
                    seen_closefd.append(kwargs.get("closefd"))
                    raise OSError(errno.ENOMEM, "stream wrapper failed")
                return real_open(file, *args, **kwargs)

            def spy_os_close(fd):
                closed_fds.append(fd)
                return real_os_close(fd)

            with (
                mock.patch.object(builtins, "open", side_effect=spy_builtin_open),
                mock.patch.object(prefetch.os, "close", side_effect=spy_os_close),
            ):
                with self.assertRaises(OSError) as ctx:
                    prefetch._write_text_atomic(target, '{"new": true}\n')

            self.assertEqual(seen_closefd, [False])
            # Both temp fd and dir_fd closed exactly once each (no EBADF/double-close)
            self.assertEqual(len(closed_fds), 2)
            self.assertEqual(len(set(closed_fds)), 2)
            self.assertEqual(ctx.exception.errno, errno.ENOMEM)
            self.assertEqual(target.read_text(encoding="utf-8"), '{"old": true}\n')
            self.assertEqual([p.name for p in root.iterdir()], ["manifest.json"])
            _assert_fd_count(self, fds_before)

    def test_write_text_atomic_close_error_aborts_before_replace(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "manifest.json"
            target.write_text('{"old": true}\n', encoding="utf-8")
            fds_before = _open_fd_count()
            real_os_close = os.close
            real_os_open = os.open
            staged_fds = []

            def spy_os_open(path, flags, *args, **kwargs):
                fd = real_os_open(path, flags, *args, **kwargs)
                if "dir_fd" in kwargs:
                    staged_fds.append(fd)
                return fd

            def fail_staged_close(fd):
                real_os_close(fd)
                if fd in staged_fds:
                    raise OSError(errno.EIO, "delayed close I/O error")

            with (
                mock.patch.object(prefetch.os, "open", side_effect=spy_os_open),
                mock.patch.object(prefetch.os, "close", side_effect=fail_staged_close),
                mock.patch.object(prefetch.os, "replace") as mock_replace,
            ):
                with self.assertRaises(OSError) as ctx:
                    prefetch._write_text_atomic(target, '{"new": true}\n')

            mock_replace.assert_not_called()
            self.assertEqual(ctx.exception.errno, errno.EIO)
            self.assertIn(str(target), str(ctx.exception))
            self.assertEqual(target.read_text(encoding="utf-8"), '{"old": true}\n')
            self.assertEqual([p.name for p in root.iterdir()], ["manifest.json"])
            _assert_fd_count(self, fds_before)

    def test_write_text_atomic_fsize_preserves_existing_manifest_and_cleans_temp(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            home = _stage_home(root)
            assistant = home / "assistant"
            assistant.mkdir()
            existing = assistant / prefetch.MANIFEST_NAME
            old_text = '{"old": "intact"}\n'
            existing.write_text(old_text, encoding="utf-8")
            old_ino = existing.stat().st_ino

            # Child process for write_manifest() under RLIMIT_FSIZE=64
            proc1 = subprocess.run(
                [
                    sys.executable, "-c",
                    f"import errno, resource, sys\n"
                    f"from pathlib import Path\n"
                    f"from mlx_omarchy_assistant import prefetch\n"
                    f"m = prefetch.build_manifest(Path({str(home)!r}), None)\n"
                    f"resource.setrlimit(resource.RLIMIT_FSIZE, (64, 64))\n"
                    f"try:\n"
                    f"    prefetch.write_manifest(Path({str(home)!r}), m)\n"
                    f"except OSError as exc:\n"
                    f"    print(f'ERRNO={{exc.errno}} MSG={{exc}}', file=sys.stderr)\n"
                    f"    sys.exit(42 if exc.errno == errno.EFBIG else 99)",
                ],
                env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "serve")},
                stdin=subprocess.DEVNULL,
                capture_output=True, text=True, timeout=15, check=False,
            )
            self.assertEqual(proc1.returncode, 42, proc1.stderr)
            self.assertIn(f"ERRNO={errno.EFBIG}", proc1.stderr)
            self.assertIn(str(existing), proc1.stderr)
            self.assertNotIn(".prefetch.", proc1.stderr)
            self.assertEqual(existing.read_text(encoding="utf-8"), old_text)
            self.assertEqual(existing.stat().st_ino, old_ino)
            self.assertEqual([p.name for p in assistant.iterdir()], [prefetch.MANIFEST_NAME])

            # Child process for CLI --manifest-out under RLIMIT_FSIZE=64
            out_dir = root / "out"
            out_dir.mkdir()
            out = out_dir / "manifest.json"
            out.write_text(old_text, encoding="utf-8")
            out_ino = out.stat().st_ino
            proc2 = subprocess.run(
                [
                    sys.executable, "-c",
                    f"import resource, sys; from mlx_omarchy_assistant import prefetch; "
                    f"resource.setrlimit(resource.RLIMIT_FSIZE, (64, 64)); "
                    f"sys.exit(prefetch.main(['--home', {str(home)!r}, '--manifest-only', "
                    f"'--manifest-out', {str(out)!r}]))",
                ],
                env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "serve")},
                stdin=subprocess.DEVNULL,
                capture_output=True, text=True, timeout=15, check=False,
            )
            self.assertEqual(proc2.returncode, 1)
            self.assertEqual(proc2.stdout, "")
            self.assertTrue(proc2.stderr.startswith("prefetch: "), proc2.stderr)
            self.assertNotIn("Traceback", proc2.stderr)
            self.assertIn(os.strerror(errno.EFBIG), proc2.stderr)
            self.assertIn(str(out), proc2.stderr)
            self.assertNotIn(".prefetch.", proc2.stderr)
            self.assertEqual(out.read_text(encoding="utf-8"), old_text)
            self.assertEqual(out.stat().st_ino, out_ino)
            self.assertEqual([p.name for p in out_dir.iterdir()], ["manifest.json"])

    def test_write_text_atomic_and_write_manifest_invalid_paths_raise_einval(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bad_paths = [
                "",
                ".",
                "..",
                "/",
                f"{root}/out.json/",
                f"{root}/out.json/.",
                f"{root}/out.json/..",
                Path(""),
                Path("."),
                Path("/"),
                root / "..",
                Path("bad\x00name.json"),
                Path(f"{root}\x00x/out.json"),
                root / "\ud800.json",
            ]
            for bad in bad_paths:
                with self.subTest(path=repr(bad)):
                    with self.assertRaises(OSError) as ctx:
                        prefetch._write_text_atomic(bad, "{}\n")
                    self.assertEqual(ctx.exception.errno, errno.EINVAL)

            # Direct write_manifest() validates home before mkdir
            for bad_home in (Path(f"{root}/bad\x00home"), Path(f"{root}/\ud800home")):
                with self.subTest(bad_home=repr(bad_home)):
                    with self.assertRaises(OSError) as ctx:
                        prefetch.write_manifest(bad_home, {"version": 1})
                    self.assertEqual(ctx.exception.errno, errno.EINVAL)

    def test_write_text_atomic_synthetic_oserror_without_errno(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "manifest.json"
            with mock.patch.object(prefetch.os, "replace", side_effect=OSError("disk full")):
                with self.assertRaises(OSError) as ctx:
                    prefetch._write_text_atomic(target, "{}\n")
            self.assertIn(str(target), str(ctx.exception))
            self.assertIn("disk full", str(ctx.exception))
            self.assertEqual(list(Path(tmp).iterdir()), [])


class CliOsErrorHandlingTest(unittest.TestCase):
    def test_cli_manifest_out_unwritable_parent_exits_1_cleanly(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _require_posix_perm_enforcement(self, root)
            home = _stage_home(root)
            ro = root / "ro"
            ro.mkdir()
            ro.chmod(0o500)
            target = ro / "manifest.json"
            try:
                stdout, stderr = io.StringIO(), io.StringIO()
                with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                    code = prefetch.main([
                        "--home", str(home), "--manifest-only",
                        "--manifest-out", str(target),
                    ])
            finally:
                ro.chmod(0o700)

            self.assertEqual(code, 1)
            self.assertEqual(stdout.getvalue(), "")
            self.assertTrue(stderr.getvalue().startswith("prefetch: "), stderr.getvalue())
            self.assertNotIn("Traceback", stderr.getvalue())
            self.assertIn(str(target), stderr.getvalue())
            self.assertNotIn(".prefetch.", stderr.getvalue())

    def test_cli_manifest_out_missing_parent_and_dir_target_exit_1_cleanly(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            home = _stage_home(root)
            dir_parent = root / "dir_case"
            existing_dir = dir_parent / "adir"
            existing_dir.mkdir(parents=True)

            for bad_out in (str(root / "missing" / "manifest.json"), str(existing_dir), ".", ""):
                with self.subTest(bad_out=repr(bad_out)):
                    stdout, stderr = io.StringIO(), io.StringIO()
                    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                        code = prefetch.main([
                            "--home", str(home), "--manifest-only",
                            "--manifest-out", bad_out,
                        ])
                    self.assertEqual(code, 1)
                    self.assertEqual(stdout.getvalue(), "")
                    self.assertTrue(stderr.getvalue().startswith("prefetch: "), stderr.getvalue())
                    self.assertNotIn("Traceback", stderr.getvalue())
            # Sibling temp files would have landed in existing_dir.parent
            self.assertEqual([p.name for p in dir_parent.iterdir()], ["adir"])

    def test_cli_write_manifest_oserror_in_download_path_exits_1_cleanly(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            home = _stage_home(root)
            # Block home/assistant with a regular file so write_manifest() raises FileExistsError
            (home / "assistant").write_text("not a dir\n", encoding="utf-8")

            def fake_prefetch(h, _pair_id, *, wake=None):
                manifest = prefetch.build_manifest(h, None)
                prefetch.write_manifest(h, manifest)
                return manifest

            stdout, stderr = io.StringIO(), io.StringIO()
            with (
                mock.patch.object(prefetch, "prefetch", side_effect=fake_prefetch),
                contextlib.redirect_stdout(stdout),
                contextlib.redirect_stderr(stderr),
            ):
                code = prefetch.main(["--home", str(home), "--pair", "everyday"])

            self.assertEqual(code, 1)
            self.assertEqual(stdout.getvalue(), "")
            self.assertTrue(stderr.getvalue().startswith("prefetch: "), stderr.getvalue())
            self.assertNotIn("Traceback", stderr.getvalue())


class ArtifactRootsTest(unittest.TestCase):
    def test_valid_symlinked_roots_and_voice_regular_files_still_work(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ext = root / "external"
            (ext / "pack-a").mkdir(parents=True)
            (ext / "pack-a" / "model.bin").write_bytes(b"packa")
            (ext / "wake").mkdir(parents=True)
            (ext / "wake" / "w.onnx").write_bytes(b"wake")
            (ext / "vad").mkdir(parents=True)
            (ext / "vad" / "v.onnx").write_bytes(b"vad")
            (ext / "note.txt").write_text("note\n", encoding="utf-8")

            home = root / "home"
            (home / "voice").mkdir(parents=True)
            (home / "voice" / "voice.json").write_text('{"voice": "af_heart"}\n', encoding="utf-8")
            (home / "voice" / "note-link").symlink_to(ext / "note.txt")
            os.mkfifo(home / "voice" / "ignored.fifo")
            (home / "voice" / "pack-a").symlink_to(ext / "pack-a")
            (home / "wake").symlink_to(ext / "wake")
            (home / "vad").symlink_to(ext / "vad")

            fds_before = _open_fd_count()
            with _alarm_deadline(3):
                manifest = prefetch.build_manifest(home, None)
            _assert_fd_count(self, fds_before)
            self.assertEqual(
                [a["root"] for a in manifest["artifacts"]],
                ["vad", "voice/pack-a", "wake"],
            )
            self.assertEqual(manifest["file_count"], 3)

            # Symlinked top-level voice/ directory and symlinked home both work
            home2 = root / "home2"
            home2.mkdir()
            ext_voice = root / "ext-voice"
            (ext_voice / "kokoro").mkdir(parents=True)
            (ext_voice / "kokoro" / "w.bin").write_bytes(b"abc")
            (home2 / "voice").symlink_to(ext_voice)
            home2_link = root / "home2-link"
            home2_link.symlink_to(home2)
            manifest2 = prefetch.build_manifest(home2_link, None)
            self.assertEqual([a["root"] for a in manifest2["artifacts"]], ["voice/kokoro"])
            self.assertEqual(manifest2["file_count"], 1)

    def test_missing_home_returns_empty_while_dangling_or_looping_home_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            empty_home = root / "empty"
            empty_home.mkdir()
            self.assertEqual(prefetch._artifact_roots(empty_home), [])

            missing_home = root / "missing"
            self.assertEqual(prefetch._artifact_roots(missing_home), [])

            # Dangling --home symlink is refused (not a successful empty manifest)
            dangling_home = root / "dangling-home"
            dangling_home.symlink_to(root / "no-such-home")
            with self.assertRaises(pairs.PairError):
                prefetch._artifact_roots(dangling_home)
            stdout, stderr = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                code = prefetch.main(["--home", str(dangling_home), "--manifest-only"])
            self.assertEqual(code, 1)
            self.assertEqual(stdout.getvalue(), "")
            self.assertTrue(stderr.getvalue().startswith("prefetch: "), stderr.getvalue())

            # Looping --home symlink is also refused
            loop_home = root / "loop-home"
            loop_home.symlink_to(loop_home)
            with self.assertRaises(pairs.PairError):
                prefetch._artifact_roots(loop_home)

    def test_unreadable_voice_pack_and_top_level_roots_raise_pair_error_at_discovery(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _require_posix_perm_enforcement(self, root)

            # 1. voice/ chmod 000
            h1 = root / "h1"
            (h1 / "voice").mkdir(parents=True)
            (h1 / "voice").chmod(0o000)
            try:
                fds_before = _open_fd_count()
                with self.assertRaises(pairs.PairError):
                    prefetch._artifact_roots(h1)
                _assert_fd_count(self, fds_before)
                stdout, stderr = io.StringIO(), io.StringIO()
                with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                    code = prefetch.main(["--home", str(h1), "--manifest-only"])
                self.assertEqual(code, 1)
                self.assertEqual(stdout.getvalue(), "")
                self.assertTrue(stderr.getvalue().startswith("prefetch: "))
            finally:
                (h1 / "voice").chmod(0o700)

            # 2. voice/ chmod 400 (readable, not searchable) containing a pack
            h2 = root / "h2"
            (h2 / "voice" / "pack").mkdir(parents=True)
            (h2 / "voice").chmod(0o400)
            try:
                fds_before = _open_fd_count()
                with self.assertRaises(pairs.PairError):
                    prefetch._artifact_roots(h2)
                _assert_fd_count(self, fds_before)
            finally:
                (h2 / "voice").chmod(0o700)

            # 3. voice/<pack> chmod 000 rejected directly by _artifact_roots()
            h2b = root / "h2b"
            (h2b / "voice" / "locked-pack").mkdir(parents=True)
            (h2b / "voice" / "locked-pack").chmod(0o000)
            try:
                fds_before = _open_fd_count()
                with self.assertRaises(pairs.PairError):
                    prefetch._artifact_roots(h2b)
                _assert_fd_count(self, fds_before)
            finally:
                (h2b / "voice" / "locked-pack").chmod(0o700)

            # 4. wake/ chmod 000
            h3 = root / "h3"
            (h3 / "wake").mkdir(parents=True)
            (h3 / "wake").chmod(0o000)
            try:
                with self.assertRaises(pairs.PairError):
                    prefetch._artifact_roots(h3)
            finally:
                (h3 / "wake").chmod(0o700)

            # 5. home itself chmod 000
            h4 = root / "h4"
            h4.mkdir()
            h4.chmod(0o000)
            try:
                with self.assertRaises(pairs.PairError):
                    prefetch._artifact_roots(h4)
            finally:
                h4.chmod(0o700)

    def test_broken_symlinks_loops_and_non_dir_roots_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            regular = root / "reg.txt"
            regular.write_text("file\n", encoding="utf-8")
            fifo_target = root / "target.fifo"
            os.mkfifo(fifo_target)

            # Top-level voice / wake / vad: dangling symlink, symlink loop, regular file,
            # symlink to regular file, FIFO, symlink to FIFO
            for slot in ("voice", "wake", "vad"):
                for kind in ("dangling", "loop", "file", "file-link", "fifo", "fifo-link"):
                    with self.subTest(slot=slot, kind=kind):
                        home = root / f"{slot}-{kind}"
                        home.mkdir()
                        entry = home / slot
                        if kind == "dangling":
                            entry.symlink_to(root / "no-such-target")
                        elif kind == "loop":
                            entry.symlink_to(entry)
                        elif kind == "file":
                            entry.write_text("not a dir\n", encoding="utf-8")
                        elif kind == "file-link":
                            entry.symlink_to(regular)
                        elif kind == "fifo":
                            os.mkfifo(entry)
                        elif kind == "fifo-link":
                            entry.symlink_to(fifo_target)

                        with _alarm_deadline(3):
                            with self.assertRaises(pairs.PairError):
                                prefetch._artifact_roots(home)
                        with _alarm_deadline(3):
                            with self.assertRaises(pairs.PairError):
                                prefetch.build_manifest(home, None)

            # _open_optional_dir opens with O_RDONLY | O_NONBLOCK | O_DIRECTORY | O_CLOEXEC
            probe_dir = root / "probe-dir"
            probe_dir.mkdir()
            real_open = os.open
            seen_flags = []

            def spy_open(p, flags, *args, **kwargs):
                seen_flags.append(flags)
                return real_open(p, flags, *args, **kwargs)

            with mock.patch.object(prefetch.os, "open", side_effect=spy_open):
                fd = prefetch._open_optional_dir(probe_dir)
                self.assertIsNotNone(fd)
                os.close(fd)
            self.assertEqual(len(seen_flags), 1)
            self.assertTrue(seen_flags[0] & os.O_DIRECTORY)
            self.assertTrue(seen_flags[0] & os.O_NONBLOCK)
            self.assertTrue(seen_flags[0] & os.O_CLOEXEC)

            # voice/<pack> dangling symlink, symlink loop, and symlink through a regular file
            for kind in ("dangling", "loop", "through-file"):
                with self.subTest(voice_pack=kind):
                    home = root / f"voice-pack-{kind}"
                    (home / "voice").mkdir(parents=True)
                    entry = home / "voice" / "broken-pack"
                    if kind == "dangling":
                        entry.symlink_to(root / "no-such-pack")
                    elif kind == "loop":
                        entry.symlink_to(entry)
                    else:
                        voice_file = home / "voice" / "voice.json"
                        voice_file.write_text('{"voice": "af_heart"}\n', encoding="utf-8")
                        entry.symlink_to(voice_file / "weights")
                    with self.assertRaises(pairs.PairError):
                        prefetch._artifact_roots(home)
                    with self.assertRaises(pairs.PairError):
                        prefetch.build_manifest(home, None)

            # home itself is a regular file or FIFO -> refused without hanging
            with self.assertRaises(pairs.PairError) as reg_ctx:
                prefetch._check_home_dir(regular)
            self.assertIn(f"artifact path {regular} is not a directory", str(reg_ctx.exception))
            with self.assertRaises(pairs.PairError):
                prefetch._artifact_roots(regular)
            with _alarm_deadline(3):
                with self.assertRaises(pairs.PairError) as fifo_ctx:
                    prefetch._check_home_dir(fifo_target)
                self.assertIn(f"artifact path {fifo_target} is not a directory", str(fifo_ctx.exception))
                with self.assertRaises(pairs.PairError):
                    prefetch._artifact_roots(fifo_target)

            # NUL byte in home -> refused as PairError
            with self.assertRaises(pairs.PairError):
                prefetch._artifact_roots(Path(f"{root}\x00bad"))

    def test_vanished_voice_entry_mid_scan_is_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            home = root / "home"
            (home / "voice" / "real-pack").mkdir(parents=True)
            (home / "voice" / "real-pack" / "a.bin").write_bytes(b"ok")
            (home / "voice" / "ephemeral").mkdir()

            real_open = os.open

            def vanish_on_open(path, flags, *args, **kwargs):
                if path == "ephemeral" and "dir_fd" in kwargs:
                    (home / "voice" / "ephemeral").rmdir()
                return real_open(path, flags, *args, **kwargs)

            with mock.patch.object(prefetch.os, "open", side_effect=vanish_on_open):
                roots = prefetch._artifact_roots(home)
            self.assertEqual(roots, [("voice/real-pack", home / "voice" / "real-pack")])


if __name__ == "__main__":
    unittest.main()
